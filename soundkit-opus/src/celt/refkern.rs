//! Portable kernels that reproduce the floating-point summation order of
//! libopus on x86-64.
//!
//! A default x86-64 libopus build always uses its SSE kernels
//! (`celt_inner_prod_sse`, `dual_inner_prod_sse`, `xcorr_kernel_sse`,
//! `comb_filter_const_sse`) and, through run-time CPU detection on a CPU with
//! AVX2 and FMA, `celt_pitch_xcorr_avx2`. Floating-point addition is not
//! associative, so matching libopus sample for sample needs the same order of
//! operations, not only the same formula. These functions keep that order in
//! plain scalar code (fused multiply-adds through `f32::mul_add`), so they
//! give the same result on every target, WebAssembly included.
//!
//! They serve the paths that must match libopus exactly: packet-loss
//! concealment, and the codec when a mode selects libopus arithmetic
//! (`CeltMode::exact_math`).

use crate::celt::pitch::COMBFILTER_MINPERIOD;

/// `celt_inner_prod_sse`: four lanes, `(l0 + l2) + (l1 + l3)`, then the tail.
#[inline]
pub fn inner_prod(x: &[f32], y: &[f32], n: usize) -> f32 {
    let mut lanes = [0.0f32; 4];
    let mut i = 0usize;
    while i + 3 < n {
        for l in 0..4 {
            lanes[l] += x[i + l] * y[i + l];
        }
        i += 4;
    }
    let mut sum = (lanes[0] + lanes[2]) + (lanes[1] + lanes[3]);
    while i < n {
        sum += x[i] * y[i];
        i += 1;
    }
    sum
}

/// `dual_inner_prod_sse`.
#[inline]
pub fn dual_inner_prod(x: &[f32], y1: &[f32], y2: &[f32], n: usize) -> (f32, f32) {
    let mut a = [0.0f32; 4];
    let mut b = [0.0f32; 4];
    let mut i = 0usize;
    while i + 3 < n {
        for l in 0..4 {
            a[l] += x[i + l] * y1[i + l];
            b[l] += x[i + l] * y2[i + l];
        }
        i += 4;
    }
    let mut s1 = (a[0] + a[2]) + (a[1] + a[3]);
    let mut s2 = (b[0] + b[2]) + (b[1] + b[3]);
    while i < n {
        s1 += x[i] * y1[i];
        s2 += x[i] * y2[i];
        i += 1;
    }
    (s1, s2)
}

/// `xcorr_kernel_sse`: four correlations `sum[k] += sum_j x[j] * y[j + k]`
/// accumulated in two interleaved partial sums.
pub fn xcorr_kernel(x: &[f32], y: &[f32], sum: &mut [f32; 4], len: usize) {
    let mut s1 = *sum;
    let mut s2 = [0.0f32; 4];
    let mut j = 0usize;
    while j + 3 < len {
        for k in 0..4 {
            s1[k] += x[j] * y[j + k];
        }
        for k in 0..4 {
            s2[k] += x[j + 1] * y[j + 1 + k];
        }
        for k in 0..4 {
            s1[k] += x[j + 2] * y[j + 2 + k];
        }
        for k in 0..4 {
            s2[k] += x[j + 3] * y[j + 3 + k];
        }
        j += 4;
    }
    if j < len {
        for k in 0..4 {
            s1[k] += x[j] * y[j + k];
        }
        j += 1;
        if j < len {
            for k in 0..4 {
                s2[k] += x[j] * y[j + k];
            }
            j += 1;
            if j < len {
                for k in 0..4 {
                    s1[k] += x[j] * y[j + k];
                }
            }
        }
    }
    for k in 0..4 {
        sum[k] = s1[k] + s2[k];
    }
}

/// `celt_pitch_xcorr_avx2`: eight lags at a time with eight-lane fused
/// multiply-adds, the remaining lags through [`inner_prod`].
pub fn pitch_xcorr(x: &[f32], y: &[f32], xcorr: &mut [f32], len: usize, max_pitch: usize) {
    let mut i = 0usize;
    while i + 7 < max_pitch {
        let yi = &y[i..];
        let mut acc = [[0.0f32; 8]; 8];
        let mut j = 0usize;
        while j + 7 < len {
            for (k, lanes) in acc.iter_mut().enumerate() {
                for l in 0..8 {
                    lanes[l] = x[j + l].mul_add(yi[j + k + l], lanes[l]);
                }
            }
            j += 8;
        }
        if j != len {
            let active = len - j;
            for (k, lanes) in acc.iter_mut().enumerate() {
                for l in 0..active {
                    lanes[l] = x[j + l].mul_add(yi[j + k + l], lanes[l]);
                }
            }
        }
        for (k, s) in acc.iter().enumerate() {
            xcorr[i + k] = ((s[0] + s[4]) + (s[1] + s[5])) + ((s[2] + s[6]) + (s[3] + s[7]));
        }
        i += 8;
    }
    while i < max_pitch {
        xcorr[i] = inner_prod(x, &y[i..], len);
        i += 1;
    }
}

/// `_celt_autocorr` for the float build, with an optional window applied to
/// the first and last `overlap` samples.
pub fn autocorr(
    x: &[f32],
    ac: &mut [f32],
    window: Option<&[f32]>,
    overlap: usize,
    lag: usize,
    n: usize,
    scratch: &mut Vec<f32>,
) {
    let xptr: &[f32] = match window {
        Some(window) if overlap > 0 => {
            scratch.clear();
            scratch.extend_from_slice(&x[..n]);
            for i in 0..overlap {
                let w = window[i];
                scratch[i] = x[i] * w;
                scratch[n - i - 1] = x[n - i - 1] * w;
            }
            scratch
        }
        _ => x,
    };
    let fast_n = n - lag;
    pitch_xcorr(xptr, xptr, ac, fast_n, lag + 1);
    for k in 0..=lag {
        let mut d = 0.0f32;
        for i in k + fast_n..n {
            d += xptr[i] * xptr[i - k];
        }
        ac[k] += d;
    }
}

/// `_celt_lpc` for the float build.
pub fn lpc(lpc: &mut [f32], ac: &[f32], p: usize) {
    lpc[..p].fill(0.0);
    let mut error = ac[0];
    if ac[0] > 1e-10 {
        for i in 0..p {
            let mut rr = 0.0f32;
            for j in 0..i {
                rr += lpc[j] * ac[i - j];
            }
            rr += ac[i + 1];
            let r = -(rr / error);
            lpc[i] = r;
            for j in 0..(i + 1) >> 1 {
                let tmp1 = lpc[j];
                let tmp2 = lpc[i - 1 - j];
                lpc[j] = tmp1 + r * tmp2;
                lpc[i - 1 - j] = tmp2 + r * tmp1;
            }
            error -= (r * r) * error;
            if error <= 0.001 * ac[0] {
                break;
            }
        }
    }
}

/// `celt_fir_c` with the SSE `xcorr_kernel`. `x` holds `ord` samples of
/// history followed by the `n` samples to filter.
pub fn fir(x: &[f32], num: &[f32], y: &mut [f32], n: usize, ord: usize) {
    let mut rnum = [0.0f32; 32];
    for i in 0..ord {
        rnum[i] = num[ord - i - 1];
    }
    let mut i = 0usize;
    while i + 3 < n {
        let mut sum = [x[ord + i], x[ord + i + 1], x[ord + i + 2], x[ord + i + 3]];
        xcorr_kernel(&rnum[..ord], &x[i..], &mut sum, ord);
        y[i..i + 4].copy_from_slice(&sum);
        i += 4;
    }
    while i < n {
        let mut sum = x[ord + i];
        for j in 0..ord {
            sum += rnum[j] * x[i + j];
        }
        y[i] = sum;
        i += 1;
    }
}

/// `celt_iir` for the float build with the SSE `xcorr_kernel`. Filters
/// `x` in place.
pub fn iir(x: &mut [f32], den: &[f32], n: usize, ord: usize, mem: &mut [f32], y: &mut Vec<f32>) {
    let mut rden = [0.0f32; 32];
    for i in 0..ord {
        rden[i] = den[ord - i - 1];
    }
    y.clear();
    y.resize(n + ord, 0.0);
    for i in 0..ord {
        y[i] = -mem[ord - i - 1];
    }
    let mut i = 0usize;
    while i + 3 < n {
        let mut sum = [x[i], x[i + 1], x[i + 2], x[i + 3]];
        xcorr_kernel(&rden[..ord], &y[i..], &mut sum, ord);
        y[i + ord] = -sum[0];
        x[i] = sum[0];
        sum[1] += y[i + ord] * den[0];
        y[i + ord + 1] = -sum[1];
        x[i + 1] = sum[1];
        sum[2] += y[i + ord + 1] * den[0];
        sum[2] += y[i + ord] * den[1];
        y[i + ord + 2] = -sum[2];
        x[i + 2] = sum[2];
        sum[3] += y[i + ord + 2] * den[0];
        sum[3] += y[i + ord + 1] * den[1];
        sum[3] += y[i + ord] * den[2];
        y[i + ord + 3] = -sum[3];
        x[i + 3] = sum[3];
        i += 4;
    }
    while i < n {
        let mut sum = x[i];
        for j in 0..ord {
            sum -= rden[j] * y[i + j];
        }
        y[i + ord] = sum;
        x[i] = sum;
        i += 1;
    }
    for i in 0..ord {
        mem[i] = x[n - i - 1];
    }
}

/// Scratch buffers for [`pitch_downsample`] and [`pitch_search`].
#[derive(Clone, Debug, Default)]
pub struct PitchScratch {
    pub(crate) x_lp4: Vec<f32>,
    pub(crate) y_lp4: Vec<f32>,
    pub(crate) xcorr: Vec<f32>,
    pub(crate) window: Vec<f32>,
}

/// `pitch_downsample` with `factor`, writing `len` samples to `x_lp`.
pub fn pitch_downsample(
    x: &[&[f32]],
    x_lp: &mut [f32],
    len: usize,
    factor: usize,
    scratch: &mut PitchScratch,
) {
    let offset = factor / 2;
    for i in 1..len {
        x_lp[i] = 0.25 * x[0][factor * i - offset]
            + 0.25 * x[0][factor * i + offset]
            + 0.5 * x[0][factor * i];
    }
    x_lp[0] = 0.25 * x[0][offset] + 0.5 * x[0][0];
    if x.len() == 2 {
        for i in 1..len {
            x_lp[i] += 0.25 * x[1][factor * i - offset]
                + 0.25 * x[1][factor * i + offset]
                + 0.5 * x[1][factor * i];
        }
        x_lp[0] += 0.25 * x[1][offset] + 0.5 * x[1][0];
    }
    let mut ac = [0.0f32; 5];
    autocorr(x_lp, &mut ac, None, 0, 4, len, &mut scratch.window);
    ac[0] *= 1.0001;
    for (i, value) in ac.iter_mut().enumerate().skip(1) {
        *value -= *value * (0.008 * i as f32) * (0.008 * i as f32);
    }
    let mut coefs = [0.0f32; 4];
    lpc(&mut coefs, &ac, 4);
    let mut tmp = 1.0f32;
    for coef in &mut coefs {
        tmp *= 0.9;
        *coef *= tmp;
    }
    let c1 = 0.8f32;
    let num = [
        coefs[0] + 0.8,
        coefs[1] + c1 * coefs[0],
        coefs[2] + c1 * coefs[1],
        coefs[3] + c1 * coefs[2],
        c1 * coefs[3],
    ];
    let (mut m0, mut m1, mut m2, mut m3, mut m4) = (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for sample in x_lp.iter_mut().take(len) {
        let sum = *sample + num[0] * m0 + num[1] * m1 + num[2] * m2 + num[3] * m3 + num[4] * m4;
        m4 = m3;
        m3 = m2;
        m2 = m1;
        m1 = m0;
        m0 = *sample;
        *sample = sum;
    }
}

fn find_best_pitch(xcorr: &[f32], y: &[f32], len: usize, max_pitch: usize) -> [usize; 2] {
    let mut syy = 1.0f32;
    let mut best_num = [-1.0f32, -1.0];
    let mut best_den = [0.0f32, 0.0];
    let mut best_pitch = [0usize, 1];
    for &value in &y[..len] {
        syy += value * value;
    }
    for i in 0..max_pitch {
        if xcorr[i] > 0.0 {
            let xcorr16 = xcorr[i] * 1e-12;
            let num = xcorr16 * xcorr16;
            if num * best_den[1] > best_num[1] * syy {
                if num * best_den[0] > best_num[0] * syy {
                    best_num[1] = best_num[0];
                    best_den[1] = best_den[0];
                    best_pitch[1] = best_pitch[0];
                    best_num[0] = num;
                    best_den[0] = syy;
                    best_pitch[0] = i;
                } else {
                    best_num[1] = num;
                    best_den[1] = syy;
                    best_pitch[1] = i;
                }
            }
        }
        syy += y[i + len] * y[i + len] - y[i] * y[i];
        syy = syy.max(1.0);
    }
    best_pitch
}

/// `pitch_search`: returns the pitch lag in samples of `x_lp`.
pub fn pitch_search(
    x_lp: &[f32],
    y: &[f32],
    len: usize,
    max_pitch: usize,
    scratch: &mut PitchScratch,
) -> usize {
    let lag = len + max_pitch;
    scratch.x_lp4.clear();
    scratch.x_lp4.extend((0..len >> 2).map(|j| x_lp[2 * j]));
    scratch.y_lp4.clear();
    scratch.y_lp4.extend((0..lag >> 2).map(|j| y[2 * j]));
    scratch.xcorr.clear();
    scratch.xcorr.resize(max_pitch >> 1, 0.0);
    pitch_xcorr(
        &scratch.x_lp4,
        &scratch.y_lp4,
        &mut scratch.xcorr,
        len >> 2,
        max_pitch >> 2,
    );
    let best = find_best_pitch(&scratch.xcorr, &scratch.y_lp4, len >> 2, max_pitch >> 2);
    for i in 0..max_pitch >> 1 {
        scratch.xcorr[i] = 0.0;
        if (i as isize - 2 * best[0] as isize).abs() > 2
            && (i as isize - 2 * best[1] as isize).abs() > 2
        {
            continue;
        }
        let sum = inner_prod(x_lp, &y[i..], len >> 1);
        scratch.xcorr[i] = sum.max(-1.0);
    }
    let best = find_best_pitch(&scratch.xcorr, y, len >> 1, max_pitch >> 1);
    let offset = if best[0] > 0 && best[0] < (max_pitch >> 1) - 1 {
        let a = scratch.xcorr[best[0] - 1];
        let b = scratch.xcorr[best[0]];
        let c = scratch.xcorr[best[0] + 1];
        if (c - a) > 0.7 * (b - a) {
            1
        } else if (a - c) > 0.7 * (b - c) {
            -1
        } else {
            0
        }
    } else {
        0
    };
    (2 * best[0] as isize - offset) as usize
}

/// The steady-state part of `comb_filter` (`comb_filter_const_sse` and its
/// scalar tail), filtering `y[base + start..base + n]` in place with period
/// `t` (at least [`COMBFILTER_MINPERIOD`]).
#[allow(clippy::too_many_arguments)]
pub fn comb_filter_const_in_place(
    y: &mut [f32],
    base: usize,
    start: usize,
    n: usize,
    t: usize,
    g10: f32,
    g11: f32,
    g12: f32,
) {
    debug_assert!(t >= COMBFILTER_MINPERIOD);
    let mut i = start;
    while i + 3 < n {
        let mut out = [0.0f32; 4];
        for (l, value) in out.iter_mut().enumerate() {
            let k = base + i + l;
            let x0 = y[k - t - 2];
            let x1 = y[k - t - 1];
            let x2 = y[k - t];
            let x3 = y[k - t + 1];
            let x4 = y[k - t + 2];
            *value = (y[k] + g10 * x2) + (g11 * (x3 + x1) + g12 * (x4 + x0));
        }
        y[base + i..base + i + 4].copy_from_slice(&out);
        i += 4;
    }
    while i < n {
        let k = base + i;
        y[k] = y[k]
            + g10 * y[k - t]
            + g11 * (y[k - t + 1] + y[k - t - 1])
            + g12 * (y[k - t + 2] + y[k - t - 2]);
        i += 1;
    }
}

/// Out-of-place `comb_filter` with no cross-fade window (`window == NULL`,
/// `overlap == 0`), as the decoder's concealment calls it: reads
/// `x[x_base - t - 2..x_base + n]`, writes `y[..n]`.
#[allow(clippy::too_many_arguments)]
pub fn comb_filter_no_window(
    x: &[f32],
    x_base: usize,
    y: &mut [f32],
    t: usize,
    n: usize,
    g: f32,
    tapset: usize,
) {
    use crate::celt::pitch::GAINS;
    if g == 0.0 {
        y[..n].copy_from_slice(&x[x_base..x_base + n]);
        return;
    }
    let t = t.max(COMBFILTER_MINPERIOD);
    let g10 = g * GAINS[tapset][0];
    let g11 = g * GAINS[tapset][1];
    let g12 = g * GAINS[tapset][2];
    comb_filter_no_window_gains(x, x_base, y, t, n, g10, g11, g12);
}

/// The out-of-place steady-state comb filter with precomputed tap gains.
#[allow(clippy::too_many_arguments)]
pub fn comb_filter_no_window_gains(
    x: &[f32],
    x_base: usize,
    y: &mut [f32],
    t: usize,
    n: usize,
    g10: f32,
    g11: f32,
    g12: f32,
) {
    let mut i = 0usize;
    while i + 3 < n {
        for l in 0..4 {
            let k = x_base + i + l;
            y[i + l] = (x[k] + g10 * x[k - t])
                + (g11 * (x[k - t + 1] + x[k - t - 1]) + g12 * (x[k - t + 2] + x[k - t - 2]));
        }
        i += 4;
    }
    while i < n {
        let k = x_base + i;
        y[i] = x[k]
            + g10 * x[k - t]
            + g11 * (x[k - t + 1] + x[k - t - 1])
            + g12 * (x[k - t + 2] + x[k - t - 2]);
        i += 1;
    }
}
