//! Floating-point CELT KISS-FFT path, ported from the official Opus
//! `celt/kiss_fft.c` CUSTOM_MODES/non-FIXED_POINT implementation.

const MAX_FACTORS: usize = 8;
const PI: f64 = core::f64::consts::PI;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct KissFftCpx {
    pub r: f32,
    pub i: f32,
}

impl KissFftCpx {
    #[inline]
    pub const fn new(r: f32, i: f32) -> Self {
        Self { r, i }
    }
}

#[derive(Clone, Debug)]
pub struct KissFftState {
    nfft: usize,
    scale: f32,
    shift: i32,
    factors: [i16; 2 * MAX_FACTORS],
    bitrev: Vec<usize>,
    twiddles: Vec<KissFftCpx>,
}

impl KissFftState {
    pub fn new(nfft: usize) -> Option<Self> {
        Self::with_twiddles(nfft, None)
    }

    pub fn with_twiddles(nfft: usize, base: Option<&KissFftState>) -> Option<Self> {
        if nfft == 0 {
            return None;
        }

        let mut factors = [0i16; 2 * MAX_FACTORS];
        kf_factor(nfft, &mut factors)?;

        let (twiddles, shift) = if let Some(base) = base {
            let mut shift = 0i32;
            while shift < 32 && (nfft << shift) != base.nfft {
                shift += 1;
            }
            if shift >= 32 {
                return None;
            }
            (base.twiddles.clone(), shift)
        } else {
            (compute_twiddles(nfft), -1)
        };

        let mut bitrev = vec![0usize; nfft];
        compute_bitrev_table(0, &mut bitrev, 0, 1, 1, &factors);

        Some(Self {
            nfft,
            scale: 1.0 / nfft as f32,
            shift,
            factors,
            bitrev,
            twiddles,
        })
    }

    pub fn nfft(&self) -> usize {
        self.nfft
    }

    pub(crate) fn scale(&self) -> f32 {
        self.scale
    }

    pub(crate) fn bitrev(&self) -> &[usize] {
        &self.bitrev
    }
}

#[inline]
fn c_add(a: KissFftCpx, b: KissFftCpx) -> KissFftCpx {
    KissFftCpx::new(a.r + b.r, a.i + b.i)
}

#[inline]
fn c_sub(a: KissFftCpx, b: KissFftCpx) -> KissFftCpx {
    KissFftCpx::new(a.r - b.r, a.i - b.i)
}

#[inline]
fn c_mul(a: KissFftCpx, b: KissFftCpx) -> KissFftCpx {
    KissFftCpx::new(a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r)
}

#[inline]
fn c_mul_scalar(a: KissFftCpx, s: f32) -> KissFftCpx {
    KissFftCpx::new(a.r * s, a.i * s)
}

fn compute_bitrev_table(
    fout: usize,
    bitrev: &mut [usize],
    out_pos: usize,
    fstride: usize,
    in_stride: usize,
    factors: &[i16],
) {
    let p = factors[0] as usize;
    let m = factors[1] as usize;

    if m == 1 {
        for j in 0..p {
            bitrev[out_pos + j * fstride * in_stride] = fout + j;
        }
    } else {
        let mut fout = fout;
        let mut pos = out_pos;
        for _ in 0..p {
            compute_bitrev_table(fout, bitrev, pos, fstride * p, in_stride, &factors[2..]);
            pos += fstride * in_stride;
            fout += m;
        }
    }
}

fn kf_factor(nfft: usize, facbuf: &mut [i16; 2 * MAX_FACTORS]) -> Option<()> {
    let mut p = 4usize;
    let mut n = nfft;
    let nbak = n;
    let mut stages = 0usize;

    loop {
        while n % p != 0 {
            p = match p {
                4 => 2,
                2 => 3,
                _ => p + 2,
            };
            if p > 32_000 || p * p > n {
                p = n;
            }
        }
        n /= p;
        if p > 5 || stages >= MAX_FACTORS {
            return None;
        }
        facbuf[2 * stages] = p as i16;
        if p == 2 && stages > 1 {
            facbuf[2 * stages] = 4;
            facbuf[2] = 2;
        }
        stages += 1;
        if n <= 1 {
            break;
        }
    }

    for i in 0..stages / 2 {
        let j = stages - i - 1;
        facbuf.swap(2 * i, 2 * j);
    }

    n = nbak;
    for i in 0..stages {
        n /= facbuf[2 * i] as usize;
        facbuf[2 * i + 1] = n as i16;
    }
    Some(())
}

fn compute_twiddles(nfft: usize) -> Vec<KissFftCpx> {
    (0..nfft)
        .map(|i| {
            let phase = (-2.0 * PI / nfft as f64) * i as f64;
            KissFftCpx::new(
                round_static_float(phase.cos()),
                round_static_float(phase.sin()),
            )
        })
        .collect()
}

fn round_static_float(value: f64) -> f32 {
    if value == 0.0 {
        return value as f32;
    }
    let scale = 10f64.powi(7 - value.abs().log10().floor() as i32);
    (value * scale).round() as f32 / scale as f32
}

fn kf_bfly2(fout: &mut [KissFftCpx], m: usize, n: usize) {
    if m == 1 {
        for i in 0..n {
            let base = 2 * i;
            let t = fout[base + 1];
            fout[base + 1] = c_sub(fout[base], t);
            fout[base] = c_add(fout[base], t);
        }
    } else {
        debug_assert_eq!(m, 4);
        let tw = 0.707_106_77f32;
        for i in 0..n {
            let base = 8 * i;
            let fout2 = base + 4;

            let mut t = fout[fout2];
            fout[fout2] = c_sub(fout[base], t);
            fout[base] = c_add(fout[base], t);

            t = KissFftCpx::new(
                (fout[fout2 + 1].r + fout[fout2 + 1].i) * tw,
                (fout[fout2 + 1].i - fout[fout2 + 1].r) * tw,
            );
            fout[fout2 + 1] = c_sub(fout[base + 1], t);
            fout[base + 1] = c_add(fout[base + 1], t);

            t = KissFftCpx::new(fout[fout2 + 2].i, -fout[fout2 + 2].r);
            fout[fout2 + 2] = c_sub(fout[base + 2], t);
            fout[base + 2] = c_add(fout[base + 2], t);

            t = KissFftCpx::new(
                (fout[fout2 + 3].i - fout[fout2 + 3].r) * tw,
                -(fout[fout2 + 3].i + fout[fout2 + 3].r) * tw,
            );
            fout[fout2 + 3] = c_sub(fout[base + 3], t);
            fout[base + 3] = c_add(fout[base + 3], t);
        }
    }
}

fn kf_bfly4(
    fout: &mut [KissFftCpx],
    fstride: usize,
    st: &KissFftState,
    m: usize,
    n: usize,
    mm: usize,
) {
    if m == 1 {
        for i in 0..n {
            let base = 4 * i;
            let scratch0 = c_sub(fout[base], fout[base + 2]);
            fout[base] = c_add(fout[base], fout[base + 2]);
            let mut scratch1 = c_add(fout[base + 1], fout[base + 3]);
            fout[base + 2] = c_sub(fout[base], scratch1);
            fout[base] = c_add(fout[base], scratch1);
            scratch1 = c_sub(fout[base + 1], fout[base + 3]);
            fout[base + 1] = KissFftCpx::new(scratch0.r + scratch1.i, scratch0.i - scratch1.r);
            fout[base + 3] = KissFftCpx::new(scratch0.r - scratch1.i, scratch0.i + scratch1.r);
        }
    } else {
        let m2 = 2 * m;
        let m3 = 3 * m;
        for i in 0..n {
            let base = i * mm;
            for j in 0..m {
                let idx = base + j;
                let scratch0 = c_mul(fout[idx + m], st.twiddles[j * fstride]);
                let scratch1 = c_mul(fout[idx + m2], st.twiddles[j * fstride * 2]);
                let scratch2 = c_mul(fout[idx + m3], st.twiddles[j * fstride * 3]);

                let scratch5 = c_sub(fout[idx], scratch1);
                fout[idx] = c_add(fout[idx], scratch1);
                let scratch3 = c_add(scratch0, scratch2);
                let scratch4 = c_sub(scratch0, scratch2);
                fout[idx + m2] = c_sub(fout[idx], scratch3);
                fout[idx] = c_add(fout[idx], scratch3);

                fout[idx + m] = KissFftCpx::new(scratch5.r + scratch4.i, scratch5.i - scratch4.r);
                fout[idx + m3] = KissFftCpx::new(scratch5.r - scratch4.i, scratch5.i + scratch4.r);
            }
        }
    }
}

fn kf_bfly3(
    fout: &mut [KissFftCpx],
    fstride: usize,
    st: &KissFftState,
    m: usize,
    n: usize,
    mm: usize,
) {
    let m2 = 2 * m;
    let epi3 = st.twiddles[fstride * m];
    for i in 0..n {
        let base = i * mm;
        for k in 0..m {
            let idx = base + k;
            let scratch1 = c_mul(fout[idx + m], st.twiddles[k * fstride]);
            let scratch2 = c_mul(fout[idx + m2], st.twiddles[2 * k * fstride]);
            let scratch3 = c_add(scratch1, scratch2);
            let mut scratch0 = c_sub(scratch1, scratch2);

            fout[idx + m] = KissFftCpx::new(
                fout[idx].r - scratch3.r * 0.5,
                fout[idx].i - scratch3.i * 0.5,
            );
            scratch0 = c_mul_scalar(scratch0, epi3.i);
            fout[idx] = c_add(fout[idx], scratch3);

            fout[idx + m2] =
                KissFftCpx::new(fout[idx + m].r + scratch0.i, fout[idx + m].i - scratch0.r);
            fout[idx + m] =
                KissFftCpx::new(fout[idx + m].r - scratch0.i, fout[idx + m].i + scratch0.r);
        }
    }
}

fn kf_bfly5(
    fout: &mut [KissFftCpx],
    fstride: usize,
    st: &KissFftState,
    m: usize,
    n: usize,
    mm: usize,
) {
    let ya = st.twiddles[fstride * m];
    let yb = st.twiddles[fstride * 2 * m];

    for i in 0..n {
        let base = i * mm;
        for u in 0..m {
            let fout0 = base + u;
            let fout1 = fout0 + m;
            let fout2 = fout0 + 2 * m;
            let fout3 = fout0 + 3 * m;
            let fout4 = fout0 + 4 * m;

            let scratch0 = fout[fout0];
            let scratch1 = c_mul(fout[fout1], st.twiddles[u * fstride]);
            let scratch2 = c_mul(fout[fout2], st.twiddles[2 * u * fstride]);
            let scratch3 = c_mul(fout[fout3], st.twiddles[3 * u * fstride]);
            let scratch4 = c_mul(fout[fout4], st.twiddles[4 * u * fstride]);

            let scratch7 = c_add(scratch1, scratch4);
            let scratch10 = c_sub(scratch1, scratch4);
            let scratch8 = c_add(scratch2, scratch3);
            let scratch9 = c_sub(scratch2, scratch3);

            fout[fout0] = KissFftCpx::new(
                fout[fout0].r + scratch7.r + scratch8.r,
                fout[fout0].i + scratch7.i + scratch8.i,
            );

            let scratch5 = KissFftCpx::new(
                scratch0.r + scratch7.r * ya.r + scratch8.r * yb.r,
                scratch0.i + scratch7.i * ya.r + scratch8.i * yb.r,
            );
            let scratch6 = KissFftCpx::new(
                scratch10.i * ya.i + scratch9.i * yb.i,
                -(scratch10.r * ya.i + scratch9.r * yb.i),
            );

            fout[fout1] = c_sub(scratch5, scratch6);
            fout[fout4] = c_add(scratch5, scratch6);

            let scratch11 = KissFftCpx::new(
                scratch0.r + scratch7.r * yb.r + scratch8.r * ya.r,
                scratch0.i + scratch7.i * yb.r + scratch8.i * ya.r,
            );
            let scratch12 = KissFftCpx::new(
                scratch9.i * ya.i - scratch10.i * yb.i,
                scratch10.r * yb.i - scratch9.r * ya.i,
            );

            fout[fout2] = c_add(scratch11, scratch12);
            fout[fout3] = c_sub(scratch11, scratch12);
        }
    }
}

pub fn opus_fft_impl(st: &KissFftState, fout: &mut [KissFftCpx]) {
    assert!(fout.len() >= st.nfft);
    let shift = st.shift.max(0) as usize;
    let mut fstride = [0usize; MAX_FACTORS + 1];
    fstride[0] = 1;

    let mut l = 0usize;
    loop {
        let p = st.factors[2 * l] as usize;
        let m = st.factors[2 * l + 1] as usize;
        fstride[l + 1] = fstride[l] * p;
        l += 1;
        if m == 1 {
            break;
        }
    }

    let mut m = st.factors[2 * l - 1] as usize;
    for i in (0..l).rev() {
        let m2 = if i != 0 {
            st.factors[2 * i - 1] as usize
        } else {
            1
        };
        match st.factors[2 * i] {
            2 => kf_bfly2(fout, m, fstride[i]),
            3 => kf_bfly3(fout, fstride[i] << shift, st, m, fstride[i], m2),
            4 => kf_bfly4(fout, fstride[i] << shift, st, m, fstride[i], m2),
            5 => kf_bfly5(fout, fstride[i] << shift, st, m, fstride[i], m2),
            radix => unreachable!("unsupported FFT radix {radix}"),
        }
        m = m2;
    }
}

pub fn opus_fft(st: &KissFftState, fin: &[KissFftCpx], fout: &mut [KissFftCpx]) {
    assert!(fin.len() >= st.nfft);
    assert!(fout.len() >= st.nfft);
    for i in 0..st.nfft {
        let x = fin[i];
        fout[st.bitrev[i]] = KissFftCpx::new(x.r * st.scale, x.i * st.scale);
    }
    opus_fft_impl(st, fout);
}

pub fn opus_ifft(st: &KissFftState, fin: &[KissFftCpx], fout: &mut [KissFftCpx]) {
    assert!(fin.len() >= st.nfft);
    assert!(fout.len() >= st.nfft);
    for i in 0..st.nfft {
        fout[st.bitrev[i]] = fin[i];
    }
    for item in fout.iter_mut().take(st.nfft) {
        item.i = -item.i;
    }
    opus_fft_impl(st, fout);
    for item in fout.iter_mut().take(st.nfft) {
        item.i = -item.i;
    }
}
