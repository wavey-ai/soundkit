//! CELT pitch prefilter helpers ported from official `celt/pitch.c`,
//! `celt/celt.c`, and `celt/celt_encoder.c` float paths.

use crate::celt::modes::CeltMode;
use wide::f32x4;

pub const COMBFILTER_MAXPERIOD: usize = 1024;
pub const COMBFILTER_MINPERIOD: usize = 15;

const SECOND_CHECK: [usize; 16] = [0, 0, 3, 2, 3, 2, 5, 2, 3, 2, 3, 2, 5, 2, 3, 2];
const GAINS: [[f32; 3]; 3] = [
    [0.3066406250, 0.2170410156, 0.1296386719],
    [0.4638671875, 0.2680664062, 0.0],
    [0.7998046875, 0.1000976562, 0.0],
];

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PrefilterDecision {
    pub enabled: bool,
    pub pitch: i32,
    pub qgain: i32,
    pub tapset: i32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ToneAnalysis {
    pub frequency: f32,
    pub toneishness: f32,
}

impl Default for ToneAnalysis {
    fn default() -> Self {
        Self {
            frequency: -1.0,
            toneishness: 0.0,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct PrefilterScratch {
    pre: Vec<Vec<f32>>,
    pitch_buf: Vec<f32>,
    autocorr: Vec<f32>,
    lpc: Vec<f32>,
    x_lp4: Vec<f32>,
    y_lp4: Vec<f32>,
    xcorr: Vec<f32>,
    yy_lookup: Vec<f32>,
}

fn tone_lpc(x: &[f32], delay: usize) -> Option<[f32; 2]> {
    debug_assert!(x.len() > 2 * delay);
    let len = x.len();
    let mut r00 = 0.0f32;
    let mut r01 = 0.0f32;
    let mut r02 = 0.0f32;
    for i in 0..len - 2 * delay {
        r00 += x[i] * x[i];
        r01 += x[i] * x[i + delay];
        r02 += x[i] * x[i + 2 * delay];
    }

    let mut edges = 0.0f32;
    for i in 0..delay {
        edges += x[len + i - 2 * delay] * x[len + i - 2 * delay] - x[i] * x[i];
    }
    let r11 = r00 + edges;
    edges = 0.0;
    for i in 0..delay {
        edges += x[len + i - delay] * x[len + i - delay] - x[i + delay] * x[i + delay];
    }
    let r22 = r11 + edges;
    edges = 0.0;
    for i in 0..delay {
        edges += x[len + i - 2 * delay] * x[len + i - delay] - x[i] * x[i + delay];
    }
    let r12 = r01 + edges;

    let reverse_r00 = r00 + r22;
    let reverse_r01 = r01 + r12;
    let reverse_r11 = 2.0 * r11;
    let reverse_r02 = 2.0 * r02;
    let reverse_r12 = r12 + r01;
    r00 = reverse_r00;
    r01 = reverse_r01;
    let r11 = reverse_r11;
    r02 = reverse_r02;
    let r12 = reverse_r12;

    let product = r00 * r11;
    let denominator = product - r01 * r01;
    if denominator < 0.001 * product {
        return None;
    }

    let numerator1 = r02 * r11 - r01 * r12;
    let lpc1 = (numerator1 / denominator).clamp(-1.0, 1.0);
    let numerator0 = r00 * r12 - r02 * r01;
    let lpc0 = (numerator0 / denominator).clamp(-1.999_999, 1.999_999);
    Some([lpc0, lpc1])
}

pub fn tone_detect(
    input: &[Vec<f32>],
    channels: usize,
    len: usize,
    sample_rate: usize,
    scratch: &mut Vec<f32>,
) -> ToneAnalysis {
    assert!((1..=2).contains(&channels));
    assert!(input.len() >= channels);
    assert!(input
        .iter()
        .take(channels)
        .all(|channel| channel.len() >= len));

    scratch.resize(len, 0.0);
    let x = &mut scratch[..len];
    if channels == 2 {
        for (i, sample) in x.iter_mut().enumerate() {
            *sample = input[0][i] + input[1][i];
        }
    } else {
        x.copy_from_slice(&input[0][..len]);
    }

    let mut delay = 1usize;
    let mut lpc = tone_lpc(x, delay);
    while delay <= sample_rate / 3_000
        && match lpc {
            None => true,
            Some(coefficients) => coefficients[0] > 1.0 && coefficients[1] < 0.0,
        }
    {
        delay *= 2;
        lpc = tone_lpc(x, delay);
    }

    let Some([lpc0, lpc1]) = lpc else {
        return ToneAnalysis::default();
    };
    let discriminant = lpc0 * lpc0 + 3.999_999 * lpc1;
    if !discriminant.is_finite() || discriminant >= 0.0 {
        return ToneAnalysis::default();
    }

    ToneAnalysis {
        frequency: (0.5 * lpc0).acos() / delay as f32,
        toneishness: -lpc1,
    }
}

fn inner_prod(x: &[f32], y: &[f32], len: usize) -> f32 {
    let mut acc = f32x4::ZERO;
    let mut i = 0usize;
    while i + 4 <= len {
        let lhs = f32x4::new([x[i], x[i + 1], x[i + 2], x[i + 3]]);
        let rhs = f32x4::new([y[i], y[i + 1], y[i + 2], y[i + 3]]);
        acc += lhs * rhs;
        i += 4;
    }

    let mut sum = acc.reduce_add();
    while i < len {
        sum += x[i] * y[i];
        i += 1;
    }
    sum
}

fn autocorr(x: &[f32], lag: usize, n: usize, ac: &mut Vec<f32>) {
    ac.resize(lag + 1, 0.0);
    for k in 0..=lag {
        let mut sum = 0.0f32;
        for i in k..n {
            sum += x[i] * x[i - k];
        }
        ac[k] = sum;
    }
}

fn celt_lpc(ac: &[f32], p: usize, lpc: &mut Vec<f32>) {
    lpc.resize(p, 0.0);
    lpc[..p].fill(0.0);
    let mut error = ac[0];
    if ac[0] > 1e-10 {
        for i in 0..p {
            let mut rr = 0.0f32;
            for j in 0..i {
                rr += lpc[j] * ac[i - j];
            }
            rr += ac[i + 1];
            let r = -rr / error;
            lpc[i] = r;
            for j in 0..((i + 1) >> 1) {
                let tmp1 = lpc[j];
                let tmp2 = lpc[i - 1 - j];
                lpc[j] = tmp1 + r * tmp2;
                lpc[i - 1 - j] = tmp2 + r * tmp1;
            }
            error -= r * r * error;
            if error <= 0.001 * ac[0] {
                break;
            }
        }
    }
}

fn celt_fir5(x: &mut [f32], num: &[f32; 5]) {
    let mut mem0 = 0.0f32;
    let mut mem1 = 0.0f32;
    let mut mem2 = 0.0f32;
    let mut mem3 = 0.0f32;
    let mut mem4 = 0.0f32;
    for sample in x {
        let sum =
            *sample + num[0] * mem0 + num[1] * mem1 + num[2] * mem2 + num[3] * mem3 + num[4] * mem4;
        mem4 = mem3;
        mem3 = mem2;
        mem2 = mem1;
        mem1 = mem0;
        mem0 = *sample;
        *sample = sum;
    }
}

fn pitch_downsample(
    pre: &[Vec<f32>],
    len: usize,
    channels: usize,
    x_lp: &mut Vec<f32>,
    ac: &mut Vec<f32>,
    lpc: &mut Vec<f32>,
) {
    let half_len = len >> 1;
    x_lp.resize(half_len, 0.0);
    for i in 1..half_len {
        x_lp[i] = 0.25 * pre[0][2 * i - 1] + 0.25 * pre[0][2 * i + 1] + 0.5 * pre[0][2 * i];
    }
    x_lp[0] = 0.25 * pre[0][1] + 0.5 * pre[0][0];
    if channels == 2 {
        for i in 1..half_len {
            x_lp[i] += 0.25 * pre[1][2 * i - 1] + 0.25 * pre[1][2 * i + 1] + 0.5 * pre[1][2 * i];
        }
        x_lp[0] += 0.25 * pre[1][1] + 0.5 * pre[1][0];
    }

    autocorr(x_lp, 4, half_len, ac);
    ac[0] *= 1.0001;
    for i in 1..=4 {
        ac[i] -= ac[i] * (0.008 * i as f32) * (0.008 * i as f32);
    }
    celt_lpc(ac, 4, lpc);
    let mut tmp = 1.0f32;
    for coef in lpc.iter_mut() {
        tmp *= 0.9;
        *coef *= tmp;
    }
    let lpc2 = [
        lpc[0] + 0.8,
        lpc[1] + 0.8 * lpc[0],
        lpc[2] + 0.8 * lpc[1],
        lpc[3] + 0.8 * lpc[2],
        0.8 * lpc[3],
    ];
    celt_fir5(x_lp, &lpc2);
}

fn pitch_xcorr(x: &[f32], y: &[f32], len: usize, max_pitch: usize, xcorr: &mut Vec<f32>) {
    xcorr.resize(max_pitch, 0.0);
    for i in 0..max_pitch {
        xcorr[i] = inner_prod(x, &y[i..], len);
    }
}

fn find_best_pitch(xcorr: &[f32], y: &[f32], len: usize, max_pitch: usize) -> [usize; 2] {
    let mut syy = 1.0f32;
    let mut best_num = [-1.0f32, -1.0f32];
    let mut best_den = [0.0f32, 0.0f32];
    let mut best_pitch = [0usize, 1usize];

    for sample in y.iter().take(len) {
        syy += sample * sample;
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

fn pitch_search(
    x_lp: &[f32],
    y: &[f32],
    len: usize,
    max_pitch: usize,
    x_lp4: &mut Vec<f32>,
    y_lp4: &mut Vec<f32>,
    xcorr: &mut Vec<f32>,
) -> usize {
    let lag = len + max_pitch;
    x_lp4.resize(len >> 2, 0.0);
    y_lp4.resize(lag >> 2, 0.0);
    for j in 0..len >> 2 {
        x_lp4[j] = x_lp[2 * j];
    }
    for j in 0..lag >> 2 {
        y_lp4[j] = y[2 * j];
    }

    pitch_xcorr(x_lp4, y_lp4, len >> 2, max_pitch >> 2, xcorr);
    let best_pitch = find_best_pitch(xcorr, y_lp4, len >> 2, max_pitch >> 2);

    xcorr.resize(max_pitch >> 1, 0.0);
    xcorr[..max_pitch >> 1].fill(0.0);
    for i in 0..max_pitch >> 1 {
        if (i as isize - 2 * best_pitch[0] as isize).abs() > 2
            && (i as isize - 2 * best_pitch[1] as isize).abs() > 2
        {
            continue;
        }
        xcorr[i] = inner_prod(x_lp, &y[i..], len >> 1).max(-1.0);
    }
    let best_pitch = find_best_pitch(xcorr, y, len >> 1, max_pitch >> 1);

    let offset = if best_pitch[0] > 0 && best_pitch[0] < (max_pitch >> 1) - 1 {
        let a = xcorr[best_pitch[0] - 1];
        let b = xcorr[best_pitch[0]];
        let c = xcorr[best_pitch[0] + 1];
        if c - a > 0.7 * (b - a) {
            1
        } else if a - c > 0.7 * (b - c) {
            -1
        } else {
            0
        }
    } else {
        0
    };

    (2 * best_pitch[0] as i32 - offset) as usize
}

fn compute_pitch_gain(xy: f32, xx: f32, yy: f32) -> f32 {
    xy / (1.0 + xx * yy).sqrt()
}

fn dual_inner_prod(x: &[f32], y01: &[f32], y02: &[f32], n: usize) -> (f32, f32) {
    let mut xy01 = f32x4::ZERO;
    let mut xy02 = f32x4::ZERO;
    let mut i = 0usize;
    while i + 4 <= n {
        let lhs = f32x4::new([x[i], x[i + 1], x[i + 2], x[i + 3]]);
        let rhs01 = f32x4::new([y01[i], y01[i + 1], y01[i + 2], y01[i + 3]]);
        let rhs02 = f32x4::new([y02[i], y02[i + 1], y02[i + 2], y02[i + 3]]);
        xy01 += lhs * rhs01;
        xy02 += lhs * rhs02;
        i += 4;
    }

    let mut xy01_scalar = xy01.reduce_add();
    let mut xy02_scalar = xy02.reduce_add();
    while i < n {
        xy01_scalar += x[i] * y01[i];
        xy02_scalar += x[i] * y02[i];
        i += 1;
    }
    (xy01_scalar, xy02_scalar)
}

fn remove_doubling(
    x: &[f32],
    maxperiod: usize,
    minperiod: usize,
    n: usize,
    t0: &mut usize,
    prev_period: usize,
    prev_gain: f32,
    yy_lookup: &mut Vec<f32>,
) -> f32 {
    let minperiod0 = minperiod;
    let maxperiod = maxperiod / 2;
    let minperiod = minperiod / 2;
    *t0 /= 2;
    let prev_period = prev_period / 2;
    let n = n / 2;
    let base = maxperiod;
    if *t0 >= maxperiod {
        *t0 = maxperiod - 1;
    }

    let t0_initial = *t0;
    let mut t = t0_initial;
    let (xx, xy0) = dual_inner_prod(&x[base..], &x[base..], &x[base - t0_initial..], n);
    yy_lookup.resize(maxperiod + 1, 0.0);
    yy_lookup[0] = xx;
    let mut yy = xx;
    for i in 1..=maxperiod {
        yy += x[base - i] * x[base - i] - x[base + n - i] * x[base + n - i];
        yy_lookup[i] = yy.max(0.0);
    }

    let mut best_xy = xy0;
    let mut best_yy = yy_lookup[t0_initial];
    let g0 = compute_pitch_gain(xy0, xx, best_yy);
    let mut g = g0;

    for k in 2..=15 {
        let t1 = (2 * t0_initial + k) / (2 * k);
        if t1 < minperiod {
            break;
        }
        let t1b = if k == 2 {
            if t1 + t0_initial > maxperiod {
                t0_initial
            } else {
                t0_initial + t1
            }
        } else {
            (2 * SECOND_CHECK[k] * t0_initial + k) / (2 * k)
        };
        let (xy, xy2) = dual_inner_prod(&x[base..], &x[base - t1..], &x[base - t1b..], n);
        let xy = 0.5 * (xy + xy2);
        let yy = 0.5 * (yy_lookup[t1] + yy_lookup[t1b]);
        let g1 = compute_pitch_gain(xy, xx, yy);
        let cont = if (t1 as isize - prev_period as isize).abs() <= 1 {
            prev_gain
        } else if (t1 as isize - prev_period as isize).abs() <= 2 && 5 * k * k < t0_initial {
            0.5 * prev_gain
        } else {
            0.0
        };
        let mut thresh = 0.3f32.max(0.7 * g0 - cont);
        if t1 < 3 * minperiod {
            thresh = 0.4f32.max(0.85 * g0 - cont);
        } else if t1 < 2 * minperiod {
            thresh = 0.5f32.max(0.9 * g0 - cont);
        }
        if g1 > thresh {
            best_xy = xy;
            best_yy = yy;
            t = t1;
            g = g1;
        }
    }

    best_xy = best_xy.max(0.0);
    let mut pg = if best_yy <= best_xy {
        1.0
    } else {
        best_xy / (best_yy + 1.0)
    };

    let mut xcorr = [0.0f32; 3];
    for k in 0..3 {
        xcorr[k] = inner_prod(&x[base..], &x[base - (t + k - 1)..], n);
    }
    let offset = if xcorr[2] - xcorr[0] > 0.7 * (xcorr[1] - xcorr[0]) {
        1
    } else if xcorr[0] - xcorr[2] > 0.7 * (xcorr[1] - xcorr[2]) {
        -1
    } else {
        0
    };

    if pg > g {
        pg = g;
    }
    *t0 = (2 * t as i32 + offset) as usize;
    if *t0 < minperiod0 {
        *t0 = minperiod0;
    }
    pg
}

pub fn comb_filter(
    y: &mut [f32],
    y_base: usize,
    x: &[f32],
    x_base: usize,
    t0: usize,
    t1: usize,
    n: usize,
    g0: f32,
    g1: f32,
    tapset0: usize,
    tapset1: usize,
    window: Option<&[f32]>,
    mut overlap: usize,
) {
    if g0 == 0.0 && g1 == 0.0 {
        y[y_base..y_base + n].copy_from_slice(&x[x_base..x_base + n]);
        return;
    }

    let t0 = t0.max(COMBFILTER_MINPERIOD);
    let t1 = t1.max(COMBFILTER_MINPERIOD);
    let g00 = g0 * GAINS[tapset0][0];
    let g01 = g0 * GAINS[tapset0][1];
    let g02 = g0 * GAINS[tapset0][2];
    let g10 = g1 * GAINS[tapset1][0];
    let g11 = g1 * GAINS[tapset1][1];
    let g12 = g1 * GAINS[tapset1][2];

    if g0 == g1 && t0 == t1 && tapset0 == tapset1 {
        overlap = 0;
    }
    if window.is_none() {
        overlap = 0;
    }
    let window = window.unwrap_or(&[]);

    let mut i = 0usize;
    let mut x1 = x[x_base + 1 - t1];
    let mut x2 = x[x_base - t1];
    let mut x3 = x[x_base - t1 - 1];
    let mut x4 = x[x_base - t1 - 2];
    while i < overlap {
        let x0 = x[x_base + i - t1 + 2];
        let f = window[i] * window[i];
        y[y_base + i] = x[x_base + i]
            + (1.0 - f) * g00 * x[x_base + i - t0]
            + (1.0 - f) * g01 * (x[x_base + i - t0 + 1] + x[x_base + i - t0 - 1])
            + (1.0 - f) * g02 * (x[x_base + i - t0 + 2] + x[x_base + i - t0 - 2])
            + f * g10 * x2
            + f * g11 * (x1 + x3)
            + f * g12 * (x0 + x4);
        x4 = x3;
        x3 = x2;
        x2 = x1;
        x1 = x0;
        i += 1;
    }

    if g1 == 0.0 {
        y[y_base + i..y_base + n].copy_from_slice(&x[x_base + i..x_base + n]);
        return;
    }

    x4 = x[x_base + i - t1 - 2];
    x3 = x[x_base + i - t1 - 1];
    x2 = x[x_base + i - t1];
    x1 = x[x_base + i - t1 + 1];
    while i < n {
        let x0 = x[x_base + i - t1 + 2];
        y[y_base + i] = x[x_base + i] + g10 * x2 + g11 * (x1 + x3) + g12 * (x0 + x4);
        x4 = x3;
        x3 = x2;
        x2 = x1;
        x1 = x0;
        i += 1;
    }
}

pub fn comb_filter_in_place(
    y: &mut [f32],
    base: usize,
    t0: usize,
    t1: usize,
    n: usize,
    g0: f32,
    g1: f32,
    tapset0: usize,
    tapset1: usize,
    window: Option<&[f32]>,
    mut overlap: usize,
) {
    // libopus calls the decoder postfilter as comb_filter(out_syn, out_syn, ...).
    // For long pitches, later taps can read samples filtered earlier in the same frame.
    if g0 == 0.0 && g1 == 0.0 {
        return;
    }

    let t0 = t0.max(COMBFILTER_MINPERIOD);
    let t1 = t1.max(COMBFILTER_MINPERIOD);
    let g00 = g0 * GAINS[tapset0][0];
    let g01 = g0 * GAINS[tapset0][1];
    let g02 = g0 * GAINS[tapset0][2];
    let g10 = g1 * GAINS[tapset1][0];
    let g11 = g1 * GAINS[tapset1][1];
    let g12 = g1 * GAINS[tapset1][2];

    if g0 == g1 && t0 == t1 && tapset0 == tapset1 {
        overlap = 0;
    }
    if window.is_none() {
        overlap = 0;
    }
    let window = window.unwrap_or(&[]);

    let mut i = 0usize;
    let mut x1 = y[base + 1 - t1];
    let mut x2 = y[base - t1];
    let mut x3 = y[base - t1 - 1];
    let mut x4 = y[base - t1 - 2];
    while i < overlap {
        let x0 = y[base + i - t1 + 2];
        let f = window[i] * window[i];
        y[base + i] = y[base + i]
            + (1.0 - f) * g00 * y[base + i - t0]
            + (1.0 - f) * g01 * (y[base + i - t0 + 1] + y[base + i - t0 - 1])
            + (1.0 - f) * g02 * (y[base + i - t0 + 2] + y[base + i - t0 - 2])
            + f * g10 * x2
            + f * g11 * (x1 + x3)
            + f * g12 * (x0 + x4);
        x4 = x3;
        x3 = x2;
        x2 = x1;
        x1 = x0;
        i += 1;
    }

    if g1 == 0.0 {
        return;
    }

    x4 = y[base + i - t1 - 2];
    x3 = y[base + i - t1 - 1];
    x2 = y[base + i - t1];
    x1 = y[base + i - t1 + 1];
    while i < n {
        let x0 = y[base + i - t1 + 2];
        y[base + i] = y[base + i] + g10 * x2 + g11 * (x1 + x3) + g12 * (x0 + x4);
        x4 = x3;
        x3 = x2;
        x2 = x1;
        x1 = x0;
        i += 1;
    }
}

fn should_cancel_pitch_filter(
    before: &[f32; 2],
    after: &[f32; 2],
    channels: usize,
    gain: f32,
) -> bool {
    debug_assert!((1..=2).contains(&channels));
    if channels == 1 {
        return after[0] > before[0];
    }

    let threshold0 = 0.25 * gain * before[0] + 0.01 * before[1];
    let threshold1 = 0.25 * gain * before[1] + 0.01 * before[0];
    after[0] - before[0] > threshold0
        || after[1] - before[1] > threshold1
        || (before[0] - after[0] < threshold0 && before[1] - after[1] < threshold1)
}

#[allow(clippy::approx_constant, clippy::too_many_arguments)]
pub fn run_prefilter(
    mode: &CeltMode,
    input: &mut [Vec<f32>],
    prefilter_mem: &mut [Vec<f32>],
    prev_period: usize,
    prev_gain: f32,
    prev_tapset: usize,
    prefilter_tapset: usize,
    enabled: bool,
    tf_estimate: f32,
    mut tone_frequency: f32,
    toneishness: f32,
    max_pitch_ratio: Option<f32>,
    packet_bytes: usize,
    channels: usize,
    n: usize,
    scratch: &mut PrefilterScratch,
) -> (PrefilterDecision, f32) {
    let overlap = mode.overlap;
    scratch.pre.resize_with(channels, Vec::new);
    for c in 0..channels {
        scratch.pre[c].resize(n + COMBFILTER_MAXPERIOD, 0.0);
        scratch.pre[c][..COMBFILTER_MAXPERIOD].copy_from_slice(&prefilter_mem[c]);
        scratch.pre[c][COMBFILTER_MAXPERIOD..COMBFILTER_MAXPERIOD + n]
            .copy_from_slice(&input[c][overlap..overlap + n]);
    }

    let mut pitch_index = COMBFILTER_MINPERIOD;
    let mut gain1 = 0.0f32;
    if enabled && toneishness > 0.99 {
        let mut multiple = 1usize;
        // Keep this threshold slightly above PI for the 48 kHz path.
        if tone_frequency >= 3.1416 {
            tone_frequency = core::f32::consts::PI - tone_frequency;
        }
        while tone_frequency >= multiple as f32 * 0.39 {
            multiple += 1;
        }
        if tone_frequency > 0.006_148 {
            pitch_index = ((2.0 * core::f32::consts::PI * multiple as f32 / tone_frequency + 0.5)
                .floor() as usize)
                .min(COMBFILTER_MAXPERIOD - 2);
        } else {
            pitch_index = COMBFILTER_MINPERIOD;
        }
        gain1 = 0.75;
    } else if enabled {
        pitch_downsample(
            &scratch.pre,
            COMBFILTER_MAXPERIOD + n,
            channels,
            &mut scratch.pitch_buf,
            &mut scratch.autocorr,
            &mut scratch.lpc,
        );
        let pitch = pitch_search(
            &scratch.pitch_buf[COMBFILTER_MAXPERIOD >> 1..],
            &scratch.pitch_buf,
            n,
            COMBFILTER_MAXPERIOD - 3 * COMBFILTER_MINPERIOD,
            &mut scratch.x_lp4,
            &mut scratch.y_lp4,
            &mut scratch.xcorr,
        );
        pitch_index = COMBFILTER_MAXPERIOD - pitch;
        gain1 = remove_doubling(
            &scratch.pitch_buf,
            COMBFILTER_MAXPERIOD,
            COMBFILTER_MINPERIOD,
            n,
            &mut pitch_index,
            prev_period,
            prev_gain,
            &mut scratch.yy_lookup,
        );
        pitch_index = pitch_index.min(COMBFILTER_MAXPERIOD - 2);
        gain1 *= 0.7;
    }
    if let Some(max_pitch_ratio) = max_pitch_ratio {
        gain1 *= max_pitch_ratio;
    }

    let mut threshold = 0.2f32;
    if (pitch_index as isize - prev_period as isize).abs() * 10 > pitch_index as isize {
        threshold += 0.2;
        if tf_estimate > 0.98 {
            gain1 = 0.0;
        }
    }
    if packet_bytes < 25 {
        threshold += 0.1;
    }
    if packet_bytes < 35 {
        threshold += 0.1;
    }
    if prev_gain > 0.4 {
        threshold -= 0.1;
    }
    if prev_gain > 0.55 {
        threshold -= 0.1;
    }
    threshold = threshold.max(0.2);

    let (mut enabled, mut qgain) = if gain1 < threshold {
        gain1 = 0.0;
        (false, 0)
    } else {
        if (gain1 - prev_gain).abs() < 0.1 {
            gain1 = prev_gain;
        }
        let qgain = ((0.5 + gain1 * 32.0 / 3.0).floor() as i32 - 1).clamp(0, 7);
        gain1 = 0.09375 * (qgain + 1) as f32;
        (true, qgain)
    };

    let mut before = [0.0f32; 2];
    for c in 0..channels {
        before[c] = input[c][overlap..overlap + n]
            .iter()
            .map(|sample| sample.abs())
            .sum();
    }
    let offset = mode.short_mdct_size - overlap;
    for c in 0..channels {
        let period = prev_period.max(COMBFILTER_MINPERIOD);
        if offset > 0 {
            comb_filter(
                &mut input[c],
                overlap,
                &scratch.pre[c],
                COMBFILTER_MAXPERIOD,
                period,
                period,
                offset,
                -prev_gain,
                -prev_gain,
                prev_tapset,
                prev_tapset,
                None,
                0,
            );
        }
        comb_filter(
            &mut input[c],
            overlap + offset,
            &scratch.pre[c],
            COMBFILTER_MAXPERIOD + offset,
            period,
            pitch_index,
            n - offset,
            -prev_gain,
            -gain1,
            prev_tapset,
            prefilter_tapset,
            Some(&mode.window),
            overlap,
        );
    }

    let mut after = [0.0f32; 2];
    for c in 0..channels {
        after[c] = input[c][overlap..overlap + n]
            .iter()
            .map(|sample| sample.abs())
            .sum();
    }
    let cancel_pitch = should_cancel_pitch_filter(&before, &after, channels, gain1);

    if cancel_pitch {
        for c in 0..channels {
            input[c][overlap..overlap + n]
                .copy_from_slice(&scratch.pre[c][COMBFILTER_MAXPERIOD..COMBFILTER_MAXPERIOD + n]);
            comb_filter(
                &mut input[c],
                overlap + offset,
                &scratch.pre[c],
                COMBFILTER_MAXPERIOD + offset,
                prev_period.max(COMBFILTER_MINPERIOD),
                pitch_index,
                overlap,
                -prev_gain,
                0.0,
                prev_tapset,
                prefilter_tapset,
                Some(&mode.window),
                overlap,
            );
        }
        gain1 = 0.0;
        enabled = false;
        qgain = 0;
    }

    for c in 0..channels {
        if n > COMBFILTER_MAXPERIOD {
            prefilter_mem[c].copy_from_slice(&scratch.pre[c][n..n + COMBFILTER_MAXPERIOD]);
        } else {
            prefilter_mem[c].copy_within(n..COMBFILTER_MAXPERIOD, 0);
            prefilter_mem[c][COMBFILTER_MAXPERIOD - n..COMBFILTER_MAXPERIOD]
                .copy_from_slice(&scratch.pre[c][COMBFILTER_MAXPERIOD..COMBFILTER_MAXPERIOD + n]);
        }
    }

    (
        PrefilterDecision {
            enabled,
            pitch: pitch_index as i32,
            qgain,
            tapset: prefilter_tapset as i32,
        },
        gain1,
    )
}

#[cfg(test)]
mod tests {
    use super::{should_cancel_pitch_filter, tone_detect};

    #[test]
    fn pitch_filter_requires_a_material_channel_benefit() {
        let before = [100.0, 100.0];

        assert!(should_cancel_pitch_filter(&before, &[114.0, 80.0], 2, 0.5));
        assert!(should_cancel_pitch_filter(&before, &[90.0, 90.0], 2, 0.5));
        assert!(!should_cancel_pitch_filter(&before, &[80.0, 100.0], 2, 0.5));
        assert!(should_cancel_pitch_filter(&before, &[101.0, 0.0], 1, 0.5));
        assert!(!should_cancel_pitch_filter(&before, &[99.0, 0.0], 1, 0.5));
    }

    #[test]
    fn tone_detector_finds_a_pure_tone_and_rejects_silence() {
        let sample_rate = 48_000usize;
        let frequency_hz = 1_000.0f32;
        let angular_frequency = 2.0 * core::f32::consts::PI * frequency_hz / sample_rate as f32;
        let input = vec![(0..1_080)
            .map(|sample| (angular_frequency * sample as f32).sin())
            .collect::<Vec<_>>()];
        let mut scratch = Vec::new();

        let tone = tone_detect(&input, 1, input[0].len(), sample_rate, &mut scratch);
        assert!(tone.toneishness > 0.999);
        assert!((tone.frequency - angular_frequency).abs() < 1e-4);

        let silence = vec![vec![0.0; input[0].len()]];
        let tone = tone_detect(&silence, 1, silence[0].len(), sample_rate, &mut scratch);
        assert_eq!(tone.frequency, -1.0);
        assert_eq!(tone.toneishness, 0.0);
    }
}
