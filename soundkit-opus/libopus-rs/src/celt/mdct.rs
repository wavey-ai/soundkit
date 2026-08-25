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

#[derive(Clone, Debug, Default)]
pub struct MdctScratch {
    f2: Vec<KissFftCpx>,
    buf: Vec<f32>,
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
    let mut scratch = MdctScratch::default();
    clt_mdct_forward_with_scratch(
        lookup,
        input,
        output,
        window,
        overlap,
        shift,
        stride,
        &mut scratch,
    );
}

/// Compute a forward MDCT and scale by `4/N` using caller-owned scratch buffers.
pub fn clt_mdct_forward_with_scratch(
    lookup: &MdctLookup,
    input: &[f32],
    output: &mut [f32],
    window: &[f32],
    overlap: usize,
    shift: usize,
    stride: usize,
    scratch: &mut MdctScratch,
) {
    let st = &lookup.kfft[shift];
    let trig = lookup.trig_for_shift(shift);
    let n = lookup.n_for_shift(shift);
    let n2 = n >> 1;
    let n4 = n >> 2;

    assert!(input.len() >= n);
    assert!(window.len() >= overlap);
    assert!(output.len() > stride * (n2 - 1));

    if scratch.buf.len() < n2 {
        scratch.buf.resize(n2, 0.0);
    }
    let f = &mut scratch.buf[..n2];
    if scratch.f2.len() < n4 {
        scratch.f2.resize(n4, KissFftCpx::default());
    }
    let f2 = &mut scratch.f2[..n4];

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

    libopus_rs_kernels::mdct_forward_pre_rotate(f, trig, st.checked_bitrev(), st.scale(), f2);

    opus_fft_impl(st, f2);

    libopus_rs_kernels::mdct_forward_post_rotate(f2, trig, stride, output);
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
    let mut scratch = MdctScratch::default();
    clt_mdct_backward_with_scratch(
        lookup,
        input,
        output,
        window,
        overlap,
        shift,
        stride,
        &mut scratch,
    );
}

/// Compute a backward MDCT using caller-owned scratch buffers.
pub fn clt_mdct_backward_with_scratch(
    lookup: &MdctLookup,
    input: &[f32],
    output: &mut [f32],
    window: &[f32],
    overlap: usize,
    shift: usize,
    stride: usize,
    scratch: &mut MdctScratch,
) {
    let st = &lookup.kfft[shift];
    let trig = lookup.trig_for_shift(shift);
    let n = lookup.n_for_shift(shift);
    let n2 = n >> 1;
    let n4 = n >> 2;

    if scratch.f2.len() < n4 {
        scratch.f2.resize(n4, KissFftCpx::default());
    }
    let f2 = &mut scratch.f2[..n4];
    libopus_rs_kernels::mdct_backward_pre_rotate(input, trig, st.checked_bitrev(), stride, f2);

    opus_fft_impl(st, f2);

    libopus_rs_kernels::mdct_backward_post_rotate(f2, trig);

    let base = overlap >> 1;
    for (pair, value) in output[base..base + n2].chunks_exact_mut(2).zip(f2) {
        pair[0] = value.r;
        pair[1] = value.i;
    }

    libopus_rs_kernels::mdct_backward_mirror(output, window, overlap);
}
