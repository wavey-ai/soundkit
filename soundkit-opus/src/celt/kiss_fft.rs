//! Floating-point CELT KISS-FFT path, ported from the official Opus
//! `celt/kiss_fft.c` CUSTOM_MODES/non-FIXED_POINT implementation.

pub use crate::kernels::Complex32 as KissFftCpx;
use crate::kernels::InRangeIndices;

const MAX_FACTORS: usize = 8;
const PI: f64 = core::f64::consts::PI;

#[derive(Clone, Debug)]
pub struct KissFftState {
    nfft: usize,
    scale: f32,
    shift: i32,
    factors: [i16; 2 * MAX_FACTORS],
    bitrev: InRangeIndices,
    twiddles: Vec<KissFftCpx>,
    /// Use libopus trunk's association of the radix-5 butterfly sums. Set
    /// for the exact (libopus-arithmetic) tables, inherited by shifted FFTs.
    trunk_order: bool,
    #[cfg(target_arch = "aarch64")]
    radix5_twiddles: Vec<KissFftCpx>,
    #[cfg(target_arch = "wasm32")]
    stage_twiddles: Vec<Vec<KissFftCpx>>,
}

impl KissFftState {
    /// FFT whose twiddles carry the 8-digit rounding of libopus's static
    /// 48 kHz tables.
    pub fn new(nfft: usize) -> Option<Self> {
        Self::with_twiddles(nfft, None)
    }

    /// FFT whose twiddles are computed in double precision and rounded once
    /// to `f32`, as libopus does for a custom mode.
    pub fn new_exact(nfft: usize) -> Option<Self> {
        Self::build(nfft, None, TwiddleTable::Runtime)
    }

    /// FFT whose twiddles equal libopus's static 48 kHz table: each value
    /// printed with 8 significant digits, then read back as `f32`.
    pub fn new_static(nfft: usize) -> Option<Self> {
        Self::build(nfft, None, TwiddleTable::Static)
    }

    pub fn with_twiddles(nfft: usize, base: Option<&KissFftState>) -> Option<Self> {
        Self::build(nfft, base, TwiddleTable::Legacy)
    }

    pub(crate) fn build(
        nfft: usize,
        base: Option<&KissFftState>,
        table: TwiddleTable,
    ) -> Option<Self> {
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
            (compute_twiddles(nfft, table), -1)
        };
        let trunk_order = base.map_or(table != TwiddleTable::Legacy, |base| base.trunk_order);

        let mut bitrev = vec![0usize; nfft];
        compute_bitrev_table(0, &mut bitrev, 0, 1, 1, &factors);
        let bitrev = InRangeIndices::new(bitrev, nfft)?;
        #[cfg(target_arch = "aarch64")]
        let radix5_twiddles = pack_first_radix5_twiddles(&factors, shift, &twiddles);
        #[cfg(target_arch = "wasm32")]
        let stage_twiddles = pack_stage_twiddles(&factors, shift, &twiddles);

        Some(Self {
            nfft,
            scale: 1.0 / nfft as f32,
            shift,
            factors,
            bitrev,
            twiddles,
            trunk_order,
            #[cfg(target_arch = "aarch64")]
            radix5_twiddles,
            #[cfg(target_arch = "wasm32")]
            stage_twiddles,
        })
    }

    pub fn nfft(&self) -> usize {
        self.nfft
    }

    pub(crate) fn scale(&self) -> f32 {
        self.scale
    }

    pub(crate) fn bitrev(&self) -> &[usize] {
        self.bitrev.as_slice()
    }

    pub(crate) fn checked_bitrev(&self) -> &InRangeIndices {
        &self.bitrev
    }
}

#[cfg(target_arch = "aarch64")]
fn pack_first_radix5_twiddles(
    factors: &[i16; 2 * MAX_FACTORS],
    shift: i32,
    twiddles: &[KissFftCpx],
) -> Vec<KissFftCpx> {
    if factors[0] != 5 {
        return Vec::new();
    }
    let m = factors[1] as usize;
    if m == 0 || m % 4 != 0 {
        return Vec::new();
    }
    let stride = 1usize << shift.max(0) as usize;
    let mut packed = Vec::with_capacity(4 * m);
    for factor in 1..=4 {
        for u in 0..m {
            packed.push(twiddles[factor * u * stride]);
        }
    }
    packed
}

/// Packs the twiddles of each radix-3, radix-4, and radix-5 stage with a
/// multiple-of-four `m` into contiguous blocks, one block of `m` values per
/// twiddle power. The SIMD128 butterflies then load four twiddles at once.
/// Other stages get an empty block list.
#[cfg(target_arch = "wasm32")]
fn pack_stage_twiddles(
    factors: &[i16; 2 * MAX_FACTORS],
    shift: i32,
    twiddles: &[KissFftCpx],
) -> Vec<Vec<KissFftCpx>> {
    let shift = shift.max(0) as usize;
    let mut stages = Vec::new();
    let mut fstride = 1usize;
    for stage in 0..MAX_FACTORS {
        let p = factors[2 * stage] as usize;
        let m = factors[2 * stage + 1] as usize;
        let stride = fstride << shift;
        let packed = if (3..=5).contains(&p) && m != 0 && m % 4 == 0 {
            (1..p)
                .flat_map(|power| (0..m).map(move |u| twiddles[power * u * stride]))
                .collect()
        } else {
            Vec::new()
        };
        stages.push(packed);
        fstride *= p;
        if m <= 1 {
            break;
        }
    }
    stages
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

/// Reports whether an `nfft`-point FFT factors into the radices 2, 3, 4
/// and 5 that KISS FFT implements, as `opus_fft_alloc` requires.
pub(crate) fn fft_size_supported(nfft: usize) -> bool {
    let mut factors = [0i16; 2 * MAX_FACTORS];
    nfft != 0 && kf_factor(nfft, &mut factors).is_some()
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

/// How a twiddle table is rounded to `f32`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TwiddleTable {
    /// SoundKit's established approximation of libopus's static tables.
    Legacy,
    /// libopus's static tables exactly: 8 significant digits, read as `f32`.
    Static,
    /// Computed at run time in double precision, as for a custom mode.
    Runtime,
}

impl TwiddleTable {
    pub(crate) fn round(self, value: f64) -> f32 {
        match self {
            Self::Legacy => round_static_float(value),
            Self::Static => round_eight_digits(value),
            Self::Runtime => value as f32,
        }
    }
}

/// `printf("%#0.8g")` of a double, read back as `f32`, as the C compiler
/// reads libopus's static table literals.
pub(crate) fn round_eight_digits(value: f64) -> f32 {
    format!("{value:.7e}")
        .parse::<f32>()
        .expect("formatted float parses")
}

fn compute_twiddles(nfft: usize, table: TwiddleTable) -> Vec<KissFftCpx> {
    (0..nfft)
        .map(|i| {
            let phase = (-2.0 * PI / nfft as f64) * i as f64;
            KissFftCpx::new(table.round(phase.cos()), table.round(phase.sin()))
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

#[inline]
fn kf_bfly2(fout: &mut [KissFftCpx], m: usize, n: usize) {
    #[cfg(target_arch = "wasm32")]
    if m == 4 {
        crate::kernels::fft_radix2_m4(fout, n);
        return;
    }
    if m == 1 {
        for [left, right] in fout[..2 * n].as_chunks_mut::<2>().0 {
            let delta = c_sub(*left, *right);
            *left = c_add(*left, *right);
            *right = delta;
        }
    } else {
        debug_assert_eq!(m, 4);
        let tw = 0.707_106_77f32;
        for block in fout[..8 * n].as_chunks_mut::<8>().0 {
            let [a, b, c, d, e, f, g, h] = *block;
            let t1 = KissFftCpx::new((f.r + f.i) * tw, (f.i - f.r) * tw);
            let t2 = KissFftCpx::new(g.i, -g.r);
            let t3 = KissFftCpx::new((h.i - h.r) * tw, -(h.i + h.r) * tw);
            *block = [
                c_add(a, e),
                c_add(b, t1),
                c_add(c, t2),
                c_add(d, t3),
                c_sub(a, e),
                c_sub(b, t1),
                c_sub(c, t2),
                c_sub(d, t3),
            ];
        }
    }
}

#[inline]
fn kf_bfly4(
    fout: &mut [KissFftCpx],
    fstride: usize,
    st: &KissFftState,
    m: usize,
    n: usize,
    mm: usize,
    stage: usize,
) {
    #[cfg(target_arch = "wasm32")]
    if m > 1
        && simd_stage(fout, st, stage, 4, m, n, mm, |block, packed| {
            crate::kernels::fft_radix4(block, packed, m)
        })
    {
        return;
    }
    #[cfg(not(target_arch = "wasm32"))]
    let _ = stage;

    #[cfg(target_arch = "wasm32")]
    if m == 1 {
        crate::kernels::fft_radix4_m1(fout, n);
        return;
    }
    if m == 1 {
        for block in fout[..4 * n].as_chunks_mut::<4>().0 {
            let [a, b, c, d] = *block;
            let even_sum = c_add(a, c);
            let odd_sum = c_add(b, d);
            let even_diff = c_sub(a, c);
            let odd_diff = c_sub(b, d);
            *block = [
                c_add(even_sum, odd_sum),
                KissFftCpx::new(even_diff.r + odd_diff.i, even_diff.i - odd_diff.r),
                c_sub(even_sum, odd_sum),
                KissFftCpx::new(even_diff.r - odd_diff.i, even_diff.i + odd_diff.r),
            ];
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

#[inline]
fn kf_bfly3(
    fout: &mut [KissFftCpx],
    fstride: usize,
    st: &KissFftState,
    m: usize,
    n: usize,
    mm: usize,
    stage: usize,
) {
    let epi3 = st.twiddles[fstride * m];

    #[cfg(target_arch = "wasm32")]
    if simd_stage(fout, st, stage, 3, m, n, mm, |block, packed| {
        crate::kernels::fft_radix3(block, packed, epi3.i, m)
    }) {
        return;
    }
    #[cfg(not(target_arch = "wasm32"))]
    let _ = stage;

    let block_size = 3 * m;
    debug_assert!(n == 1 || mm == block_size);
    for block in fout[..n * block_size].chunks_exact_mut(block_size) {
        let (left, other) = block.split_at_mut(m);
        let (mid, right) = other.split_at_mut(m);
        let twiddles = &st.twiddles;
        for (((left, mid), right), (&tw1, &tw2)) in left.iter_mut().zip(mid).zip(right).zip(
            twiddles
                .iter()
                .step_by(fstride)
                .zip(twiddles.iter().step_by(2 * fstride)),
        ) {
            let a = c_mul(*mid, tw1);
            let b = c_mul(*right, tw2);
            let sum = c_add(a, b);
            let delta = c_mul_scalar(c_sub(a, b), epi3.i);
            let center = KissFftCpx::new(left.r - sum.r * 0.5, left.i - sum.i * 0.5);
            *left = c_add(*left, sum);
            *right = KissFftCpx::new(center.r + delta.i, center.i - delta.r);
            *mid = KissFftCpx::new(center.r - delta.i, center.i + delta.r);
        }
    }
}

/// `kf_bfly5` with libopus trunk's association: `a + (b + c)` for each
/// three-term sum.
#[allow(clippy::too_many_arguments)]
fn kf_bfly5_trunk(
    fout: &mut [KissFftCpx],
    fstride: usize,
    st: &KissFftState,
    m: usize,
    n: usize,
    mm: usize,
    ya: KissFftCpx,
    yb: KissFftCpx,
) {
    for i in 0..n {
        let base = i * mm;
        for u in 0..m {
            let fout0 = base + u;
            let fout1 = fout0 + m;
            let fout2 = fout0 + 2 * m;
            let fout3 = fout0 + 3 * m;
            let fout4 = fout0 + 4 * m;

            let s0 = fout[fout0];
            let s1 = c_mul(fout[fout1], st.twiddles[u * fstride]);
            let s2 = c_mul(fout[fout2], st.twiddles[2 * u * fstride]);
            let s3 = c_mul(fout[fout3], st.twiddles[3 * u * fstride]);
            let s4 = c_mul(fout[fout4], st.twiddles[4 * u * fstride]);

            let s7 = c_add(s1, s4);
            let s10 = c_sub(s1, s4);
            let s8 = c_add(s2, s3);
            let s9 = c_sub(s2, s3);

            fout[fout0] = KissFftCpx::new(s0.r + (s7.r + s8.r), s0.i + (s7.i + s8.i));

            let s5 = KissFftCpx::new(
                s0.r + (s7.r * ya.r + s8.r * yb.r),
                s0.i + (s7.i * ya.r + s8.i * yb.r),
            );
            let s6 = KissFftCpx::new(s10.i * ya.i + s9.i * yb.i, -(s10.r * ya.i + s9.r * yb.i));
            fout[fout1] = c_sub(s5, s6);
            fout[fout4] = c_add(s5, s6);

            let s11 = KissFftCpx::new(
                s0.r + (s7.r * yb.r + s8.r * ya.r),
                s0.i + (s7.i * yb.r + s8.i * ya.r),
            );
            let s12 = KissFftCpx::new(s9.i * ya.i - s10.i * yb.i, s10.r * yb.i - s9.r * ya.i);
            fout[fout2] = c_add(s11, s12);
            fout[fout3] = c_sub(s11, s12);
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
    stage: usize,
) {
    let ya = st.twiddles[fstride * m];
    let yb = st.twiddles[fstride * 2 * m];

    if st.trunk_order {
        kf_bfly5_trunk(fout, fstride, st, m, n, mm, ya, yb);
        return;
    }

    #[cfg(target_arch = "wasm32")]
    if simd_stage(fout, st, stage, 5, m, n, mm, |block, packed| {
        crate::kernels::fft_radix5(block, packed, ya, yb, m)
    }) {
        return;
    }
    #[cfg(not(target_arch = "wasm32"))]
    let _ = stage;

    #[cfg(target_arch = "aarch64")]
    if n == 1 && crate::kernels::fft_radix5_neon(fout, &st.radix5_twiddles, ya, yb, m) {
        return;
    }

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

/// Runs a SIMD128 butterfly on each of the `n` blocks of one stage. Returns
/// `false`, before it changes any value, when the stage has no packed
/// twiddles or a block does not fit.
#[cfg(target_arch = "wasm32")]
#[allow(clippy::too_many_arguments)]
fn simd_stage(
    fout: &mut [KissFftCpx],
    st: &KissFftState,
    stage: usize,
    radix: usize,
    m: usize,
    n: usize,
    mm: usize,
    kernel: impl Fn(&mut [KissFftCpx], &[KissFftCpx]) -> bool,
) -> bool {
    let Some(packed) = st.stage_twiddles.get(stage).filter(|p| !p.is_empty()) else {
        return false;
    };
    // Blocks are `radix * m` values long. The first stage passes `mm == 1`
    // with a single block; every later stage has `mm == radix * m`.
    let block_len = radix * m;
    if packed.len() != (radix - 1) * m
        || (n > 1 && mm != block_len)
        || n.checked_mul(block_len)
            .map_or(true, |len| len > fout.len())
    {
        return false;
    }
    for block in fout[..n * block_len].chunks_exact_mut(block_len) {
        let done = kernel(block, packed);
        debug_assert!(done);
    }
    true
}

pub fn opus_fft_impl(st: &KissFftState, fout: &mut [KissFftCpx]) {
    assert!(fout.len() >= st.nfft);
    let shift = st.shift.max(0) as usize;
    // A 48-sample CELT frame uses these three fixed FFT stages.
    #[cfg(not(target_arch = "wasm32"))]
    if st.nfft == 24 {
        debug_assert_eq!(&st.factors[..6], &[3, 8, 2, 4, 4, 1]);
        let f = &mut fout[..24];
        kf_bfly4(f, 6 << shift, st, 1, 6, 4, 2);
        kf_bfly2(f, 4, 3);
        kf_bfly3(f, 1 << shift, st, 8, 1, 1, 0);
        return;
    }
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
            3 => kf_bfly3(fout, fstride[i] << shift, st, m, fstride[i], m2, i),
            4 => kf_bfly4(fout, fstride[i] << shift, st, m, fstride[i], m2, i),
            5 => kf_bfly5(fout, fstride[i] << shift, st, m, fstride[i], m2, i),
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
        fout[st.bitrev()[i]] = KissFftCpx::new(x.r * st.scale, x.i * st.scale);
    }
    opus_fft_impl(st, fout);
}

pub fn opus_ifft(st: &KissFftState, fin: &[KissFftCpx], fout: &mut [KissFftCpx]) {
    assert!(fin.len() >= st.nfft);
    assert!(fout.len() >= st.nfft);
    for i in 0..st.nfft {
        fout[st.bitrev()[i]] = fin[i];
    }
    for item in fout.iter_mut().take(st.nfft) {
        item.i = -item.i;
    }
    opus_fft_impl(st, fout);
    for item in fout.iter_mut().take(st.nfft) {
        item.i = -item.i;
    }
}
