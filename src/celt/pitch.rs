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

fn autocorr(x: &[f32], lag: usize, n: usize) -> Vec<f32> {
    let mut ac = vec![0.0f32; lag + 1];
    for k in 0..=lag {
        let mut sum = 0.0f32;
        for i in k..n {
            sum += x[i] * x[i - k];
        }
        ac[k] = sum;
    }
    ac
}

fn celt_lpc(ac: &[f32], p: usize) -> Vec<f32> {
    let mut lpc = vec![0.0f32; p];
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
    lpc
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

fn pitch_downsample(pre: &[Vec<f32>], len: usize, channels: usize) -> Vec<f32> {
    let half_len = len >> 1;
    let mut x_lp = vec![0.0f32; half_len];
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

    let mut ac = autocorr(&x_lp, 4, half_len);
    ac[0] *= 1.0001;
    for i in 1..=4 {
        ac[i] -= ac[i] * (0.008 * i as f32) * (0.008 * i as f32);
    }
    let mut lpc = celt_lpc(&ac, 4);
    let mut tmp = 1.0f32;
    for coef in &mut lpc {
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
    celt_fir5(&mut x_lp, &lpc2);
    x_lp
}

fn pitch_xcorr(x: &[f32], y: &[f32], len: usize, max_pitch: usize) -> Vec<f32> {
    let mut xcorr = vec![0.0f32; max_pitch];
    for i in 0..max_pitch {
        xcorr[i] = inner_prod(x, &y[i..], len);
    }
    xcorr
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

fn pitch_search(x_lp: &[f32], y: &[f32], len: usize, max_pitch: usize) -> usize {
    let lag = len + max_pitch;
    let mut x_lp4 = vec![0.0f32; len >> 2];
    let mut y_lp4 = vec![0.0f32; lag >> 2];
    for j in 0..len >> 2 {
        x_lp4[j] = x_lp[2 * j];
    }
    for j in 0..lag >> 2 {
        y_lp4[j] = y[2 * j];
    }

    let mut xcorr = pitch_xcorr(&x_lp4, &y_lp4, len >> 2, max_pitch >> 2);
    let best_pitch = find_best_pitch(&xcorr, &y_lp4, len >> 2, max_pitch >> 2);

    xcorr = vec![0.0f32; max_pitch >> 1];
    for i in 0..max_pitch >> 1 {
        if (i as isize - 2 * best_pitch[0] as isize).abs() > 2
            && (i as isize - 2 * best_pitch[1] as isize).abs() > 2
        {
            continue;
        }
        xcorr[i] = inner_prod(x_lp, &y[i..], len >> 1).max(-1.0);
    }
    let best_pitch = find_best_pitch(&xcorr, y, len >> 1, max_pitch >> 1);

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
    let mut yy_lookup = vec![0.0f32; maxperiod + 1];
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

#[allow(clippy::too_many_arguments)]
pub fn run_prefilter(
    mode: &CeltMode,
    input: &mut [Vec<f32>],
    prefilter_mem: &mut [Vec<f32>],
    prev_period: usize,
    prev_gain: f32,
    prev_tapset: usize,
    prefilter_tapset: usize,
    enabled: bool,
    packet_bytes: usize,
    channels: usize,
    n: usize,
) -> (PrefilterDecision, f32) {
    let overlap = mode.overlap;
    let mut pre = vec![vec![0.0f32; n + COMBFILTER_MAXPERIOD]; channels];
    for c in 0..channels {
        pre[c][..COMBFILTER_MAXPERIOD].copy_from_slice(&prefilter_mem[c]);
        pre[c][COMBFILTER_MAXPERIOD..COMBFILTER_MAXPERIOD + n]
            .copy_from_slice(&input[c][overlap..overlap + n]);
    }

    let mut pitch_index = COMBFILTER_MINPERIOD;
    let mut gain1 = 0.0f32;
    if enabled {
        let pitch_buf = pitch_downsample(&pre, COMBFILTER_MAXPERIOD + n, channels);
        let pitch = pitch_search(
            &pitch_buf[COMBFILTER_MAXPERIOD >> 1..],
            &pitch_buf,
            n,
            COMBFILTER_MAXPERIOD - 3 * COMBFILTER_MINPERIOD,
        );
        pitch_index = COMBFILTER_MAXPERIOD - pitch;
        gain1 = remove_doubling(
            &pitch_buf,
            COMBFILTER_MAXPERIOD,
            COMBFILTER_MINPERIOD,
            n,
            &mut pitch_index,
            prev_period,
            prev_gain,
        );
        pitch_index = pitch_index.min(COMBFILTER_MAXPERIOD - 2);
        gain1 *= 0.7;
    }

    let mut threshold = 0.2f32;
    if (pitch_index as isize - prev_period as isize).abs() * 10 > pitch_index as isize {
        threshold += 0.2;
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

    let (enabled, qgain) = if gain1 < threshold {
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

    let offset = mode.short_mdct_size - overlap;
    for c in 0..channels {
        let period = prev_period.max(COMBFILTER_MINPERIOD);
        if offset > 0 {
            comb_filter(
                &mut input[c],
                overlap,
                &pre[c],
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
            &pre[c],
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

        if n > COMBFILTER_MAXPERIOD {
            prefilter_mem[c].copy_from_slice(&pre[c][n..n + COMBFILTER_MAXPERIOD]);
        } else {
            prefilter_mem[c].copy_within(n..COMBFILTER_MAXPERIOD, 0);
            prefilter_mem[c][COMBFILTER_MAXPERIOD - n..COMBFILTER_MAXPERIOD]
                .copy_from_slice(&pre[c][COMBFILTER_MAXPERIOD..COMBFILTER_MAXPERIOD + n]);
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
