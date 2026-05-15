//! CELT vector-quantization helpers, ported from official `celt/vq.c`.

use crate::celt::cwrs::{decode_pulses, encode_pulses};
use crate::celt::entropy::{RangeDecoder, RangeEncoder};
use crate::celt::mathops::{celt_cos_norm, celt_div, celt_rsqrt_norm};

pub const SPREAD_NONE: i32 = 0;
pub const SPREAD_LIGHT: i32 = 1;
pub const SPREAD_NORMAL: i32 = 2;
pub const SPREAD_AGGRESSIVE: i32 = 3;

fn exp_rotation1(x: &mut [f32], len: usize, stride: usize, c: f32, s: f32) {
    if len <= stride {
        return;
    }
    let ms = -s;
    for i in 0..len - stride {
        let x1 = x[i];
        let x2 = x[i + stride];
        x[i + stride] = c * x2 + s * x1;
        x[i] = c * x1 + ms * x2;
    }

    if len <= 2 * stride {
        return;
    }
    let mut i = len - 2 * stride - 1;
    loop {
        let x1 = x[i];
        let x2 = x[i + stride];
        x[i + stride] = c * x2 + s * x1;
        x[i] = c * x1 + ms * x2;
        if i == 0 {
            break;
        }
        i -= 1;
    }
}

pub fn exp_rotation(x: &mut [f32], len: usize, dir: i32, stride: usize, k: usize, spread: i32) {
    const SPREAD_FACTOR: [i32; 3] = [15, 10, 5];

    assert!(x.len() >= len);
    if 2 * k >= len || spread == SPREAD_NONE {
        return;
    }

    let factor = SPREAD_FACTOR[(spread - 1) as usize] as f32;
    let gain = celt_div(len as f32, len as f32 + factor * k as f32);
    let theta = 0.5 * gain * gain;
    let c = celt_cos_norm(theta);
    let s = celt_cos_norm(1.0 - theta);

    let mut stride2 = 0usize;
    if len >= 8 * stride {
        stride2 = 1;
        while (stride2 * stride2 + stride2) * stride + (stride >> 2) < len {
            stride2 += 1;
        }
    }

    let block_len = len / stride;
    for i in 0..stride {
        let start = i * block_len;
        let end = start + block_len;
        if dir < 0 {
            if stride2 != 0 {
                exp_rotation1(&mut x[start..end], block_len, stride2, s, c);
            }
            exp_rotation1(&mut x[start..end], block_len, 1, c, s);
        } else {
            exp_rotation1(&mut x[start..end], block_len, 1, c, -s);
            if stride2 != 0 {
                exp_rotation1(&mut x[start..end], block_len, stride2, s, -c);
            }
        }
    }
}

fn normalise_residual(iy: &[i32], x: &mut [f32], n: usize, ryy: f32, gain: f32) {
    let g = celt_rsqrt_norm(ryy) * gain;
    for i in 0..n {
        x[i] = g * iy[i] as f32;
    }
}

fn extract_collapse_mask(iy: &[i32], n: usize, b: usize) -> u32 {
    if b <= 1 {
        return 1;
    }
    let n0 = n / b;
    let mut collapse_mask = 0u32;
    for i in 0..b {
        let mut any = false;
        for j in 0..n0 {
            any |= iy[i * n0 + j] != 0;
        }
        collapse_mask |= u32::from(any) << i;
    }
    collapse_mask
}

pub fn op_pvq_search(x: &mut [f32], iy: &mut [i32], k: usize, n: usize) -> f32 {
    assert!(x.len() >= n);
    assert!(iy.len() >= n);

    let mut y = vec![0.0f32; n];
    let mut signx = vec![false; n];

    for j in 0..n {
        signx[j] = x[j] < 0.0;
        x[j] = x[j].abs();
        iy[j] = 0;
        y[j] = 0.0;
    }

    let mut xy = 0.0f32;
    let mut yy = 0.0f32;
    let mut pulses_left = k as i32;

    if k > (n >> 1) {
        let mut sum = 0.0f32;
        for value in x.iter().take(n) {
            sum += *value;
        }

        if !(sum > 1e-15 && sum < 64.0) {
            x[0] = 1.0;
            for value in x.iter_mut().take(n).skip(1) {
                *value = 0.0;
            }
            sum = 1.0;
        }

        let rcp = (k as f32 + 0.8) / sum;
        for j in 0..n {
            iy[j] = (rcp * x[j]).floor() as i32;
            y[j] = iy[j] as f32;
            yy += y[j] * y[j];
            xy += x[j] * y[j];
            y[j] *= 2.0;
            pulses_left -= iy[j];
        }
    }
    debug_assert!(pulses_left >= 0);

    if pulses_left > n as i32 + 3 {
        let tmp = pulses_left as f32;
        yy += tmp * tmp;
        yy += tmp * y[0];
        iy[0] += pulses_left;
        pulses_left = 0;
    }

    for _ in 0..pulses_left {
        let mut best_id = 0usize;
        yy += 1.0;

        let mut rxy = xy + x[0];
        let mut ryy = yy + y[0];
        rxy *= rxy;
        let mut best_den = ryy;
        let mut best_num = rxy;

        for j in 1..n {
            rxy = xy + x[j];
            ryy = yy + y[j];
            rxy *= rxy;
            if best_den * rxy > ryy * best_num {
                best_den = ryy;
                best_num = rxy;
                best_id = j;
            }
        }

        xy += x[best_id];
        yy += y[best_id];
        y[best_id] += 2.0;
        iy[best_id] += 1;
    }

    for j in 0..n {
        if signx[j] {
            iy[j] = -iy[j];
        }
    }

    yy
}

pub fn alg_quant(
    x: &mut [f32],
    n: usize,
    k: usize,
    spread: i32,
    b: usize,
    enc: &mut RangeEncoder,
    gain: f32,
    resynth: bool,
) -> u32 {
    assert!(k > 0);
    assert!(n > 1);
    assert!(x.len() >= n);

    let mut iy = vec![0i32; n + 3];
    exp_rotation(x, n, 1, b, k, spread);
    let yy = op_pvq_search(x, &mut iy, k, n);
    encode_pulses(&iy, n, k, enc);

    if resynth {
        normalise_residual(&iy, x, n, yy, gain);
        exp_rotation(x, n, -1, b, k, spread);
    }

    extract_collapse_mask(&iy, n, b)
}

pub fn alg_unquant(
    x: &mut [f32],
    n: usize,
    k: usize,
    spread: i32,
    b: usize,
    dec: &mut RangeDecoder,
    gain: f32,
) -> u32 {
    assert!(k > 0);
    assert!(n > 1);
    assert!(x.len() >= n);

    let mut iy = vec![0i32; n];
    let ryy = decode_pulses(&mut iy, n, k, dec) as f32;
    normalise_residual(&iy, x, n, ryy, gain);
    exp_rotation(x, n, -1, b, k, spread);
    extract_collapse_mask(&iy, n, b)
}

pub fn renormalise_vector(x: &mut [f32], n: usize, gain: f32) {
    assert!(x.len() >= n);
    let energy = 1e-15 + x.iter().take(n).map(|v| v * v).sum::<f32>();
    let g = celt_rsqrt_norm(energy) * gain;
    for value in x.iter_mut().take(n) {
        *value *= g;
    }
}
