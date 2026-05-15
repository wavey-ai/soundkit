//! Floating-point CELT MDCT, ported from the official Opus `celt/mdct.c`.

use crate::celt::kiss_fft::{opus_fft_impl, KissFftCpx, KissFftState};

const PI: f64 = core::f64::consts::PI;

#[derive(Clone, Debug)]
pub struct MdctLookup {
    n: usize,
    maxshift: usize,
    kfft: Vec<KissFftState>,
    trig: Vec<f32>,
    trig_offset: Vec<usize>,
}

impl MdctLookup {
    pub fn new(n: usize, maxshift: usize) -> Option<Self> {
        if n < 4 || n % 4 != 0 {
            return None;
        }

        let base = KissFftState::new(n >> 2)?;
        let mut kfft = vec![base];
        for shift in 1..=maxshift {
            let nfft = n >> 2 >> shift;
            if nfft == 0 {
                return None;
            }
            let shifted = KissFftState::with_twiddles(nfft, Some(&kfft[0]))?;
            kfft.push(shifted);
        }

        let mut trig = Vec::new();
        let mut trig_offset = Vec::with_capacity(maxshift + 1);
        let mut cur_n = n;
        let mut n2 = n >> 1;
        for _ in 0..=maxshift {
            trig_offset.push(trig.len());
            for i in 0..n2 {
                trig.push(round_static_float_phase(
                    2.0 * PI * (i as f64 + 0.125) / cur_n as f64,
                ));
            }
            n2 >>= 1;
            cur_n >>= 1;
        }

        Some(Self {
            n,
            maxshift,
            kfft,
            trig,
            trig_offset,
        })
    }

    pub fn n(&self) -> usize {
        self.n
    }

    fn n_for_shift(&self, shift: usize) -> usize {
        assert!(shift <= self.maxshift);
        self.n >> shift
    }

    fn trig_for_shift(&self, shift: usize) -> &[f32] {
        let offset = self.trig_offset[shift];
        let len = self.n_for_shift(shift) >> 1;
        &self.trig[offset..offset + len]
    }
}

fn round_static_float_phase(phase: f64) -> f32 {
    let value = phase.cos();
    if value == 0.0 {
        return value as f32;
    }
    let scale = 10f64.powi(7 - value.abs().log10().floor() as i32);
    (value * scale).round() as f32 / scale as f32
}

/// Compute a forward MDCT and scale by `4/N`.
pub fn clt_mdct_forward(
    lookup: &MdctLookup,
    input: &[f32],
    output: &mut [f32],
    window: &[f32],
    overlap: usize,
    shift: usize,
    stride: usize,
) {
    let st = &lookup.kfft[shift];
    let trig = lookup.trig_for_shift(shift);
    let n = lookup.n_for_shift(shift);
    let n2 = n >> 1;
    let n4 = n >> 2;

    assert!(input.len() >= n);
    assert!(window.len() >= overlap);
    assert!(output.len() > stride * (n2 - 1));

    let mut f = vec![0.0f32; n2];
    let mut f2 = vec![KissFftCpx::default(); n4];

    let mut xp1 = overlap >> 1;
    let mut xp2 = n2 - 1 + (overlap >> 1);
    let mut yp = 0usize;
    let mut wp1 = overlap >> 1;
    let mut wp2 = ((overlap >> 1) - 1) as isize;
    let first = (overlap + 3) >> 2;
    let mut i = 0usize;
    while i < first {
        f[yp] = input[xp1 + n2] * window[wp2 as usize] + input[xp2] * window[wp1];
        f[yp + 1] = input[xp1] * window[wp1] - input[xp2 - n2] * window[wp2 as usize];
        yp += 2;
        xp1 += 2;
        xp2 -= 2;
        wp1 += 2;
        wp2 -= 2;
        i += 1;
    }

    wp1 = 0;
    wp2 = (overlap - 1) as isize;
    while i < n4 - first {
        f[yp] = input[xp2];
        f[yp + 1] = input[xp1];
        yp += 2;
        xp1 += 2;
        xp2 -= 2;
        i += 1;
    }

    while i < n4 {
        f[yp] = -input[xp1 - n2] * window[wp1] + input[xp2] * window[wp2 as usize];
        f[yp + 1] = input[xp1] * window[wp2 as usize] + input[xp2 + n2] * window[wp1];
        yp += 2;
        xp1 += 2;
        xp2 -= 2;
        wp1 += 2;
        wp2 -= 2;
        i += 1;
    }

    let scale = st.scale();
    for i in 0..n4 {
        let t0 = trig[i];
        let t1 = trig[n4 + i];
        let re = f[2 * i];
        let im = f[2 * i + 1];
        let yr = re * t0 - im * t1;
        let yi = im * t0 + re * t1;
        f2[st.bitrev()[i]] = KissFftCpx::new(yr * scale, yi * scale);
    }

    opus_fft_impl(st, &mut f2);

    let mut yp1 = 0usize;
    let mut yp2 = stride * (n2 - 1);
    for i in 0..n4 {
        let t0 = trig[i];
        let t1 = trig[n4 + i];
        let yr = f2[i].i * t1 - f2[i].r * t0;
        let yi = f2[i].r * t1 + f2[i].i * t0;
        output[yp1] = yr;
        output[yp2] = yi;
        yp1 += 2 * stride;
        if i + 1 < n4 {
            yp2 -= 2 * stride;
        }
    }
}

/// Compute a backward MDCT and perform the weighted overlap-add step.
pub fn clt_mdct_backward(
    lookup: &MdctLookup,
    input: &[f32],
    output: &mut [f32],
    window: &[f32],
    overlap: usize,
    shift: usize,
    stride: usize,
) {
    let st = &lookup.kfft[shift];
    let trig = lookup.trig_for_shift(shift);
    let n = lookup.n_for_shift(shift);
    let n2 = n >> 1;
    let n4 = n >> 2;

    assert!(input.len() > stride * (n2 - 1));
    assert!(output.len() >= (overlap >> 1) + n2);
    assert!(window.len() >= overlap);

    let mut f2 = vec![KissFftCpx::default(); n4];
    let mut xp1 = 0usize;
    let mut xp2 = stride * (n2 - 1);
    for i in 0..n4 {
        let rev = st.bitrev()[i];
        let x1 = input[xp1];
        let x2 = input[xp2];
        let yr = x2 * trig[i] + x1 * trig[n4 + i];
        let yi = x1 * trig[i] - x2 * trig[n4 + i];
        f2[rev] = KissFftCpx::new(yi, yr);
        xp1 += 2 * stride;
        if i + 1 < n4 {
            xp2 -= 2 * stride;
        }
    }

    opus_fft_impl(st, &mut f2);

    let mut buf = vec![0.0f32; n2];
    for i in 0..n4 {
        buf[2 * i] = f2[i].r;
        buf[2 * i + 1] = f2[i].i;
    }

    let mut yp0 = 0usize;
    let mut yp1 = n2 - 2;
    for i in 0..((n4 + 1) >> 1) {
        let re = buf[yp0 + 1];
        let im = buf[yp0];
        let t0 = trig[i];
        let t1 = trig[n4 + i];
        let yr = re * t0 + im * t1;
        let yi = re * t1 - im * t0;

        let re2 = buf[yp1 + 1];
        let im2 = buf[yp1];
        buf[yp0] = yr;
        buf[yp1 + 1] = yi;

        let t0 = trig[n4 - i - 1];
        let t1 = trig[n2 - i - 1];
        let yr = re2 * t0 + im2 * t1;
        let yi = re2 * t1 - im2 * t0;
        buf[yp1] = yr;
        buf[yp0 + 1] = yi;

        yp0 += 2;
        yp1 = yp1.saturating_sub(2);
    }

    let base = overlap >> 1;
    output[base..base + n2].copy_from_slice(&buf);

    for i in 0..overlap / 2 {
        let lo = i;
        let hi = overlap - 1 - i;
        let x1 = output[hi];
        let x2 = output[lo];
        let wp1 = window[i];
        let wp2 = window[overlap - 1 - i];
        output[lo] = x2 * wp2 - x1 * wp1;
        output[hi] = x2 * wp1 + x1 * wp2;
    }
}
