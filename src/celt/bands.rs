//! CELT band helper routines, ported from official `celt/bands.c`.

use crate::celt::mathops::{
    bitexact_cos, bitexact_log2tan, celt_rsqrt_norm, fast_atan2f, frac_mul16,
};

pub const SPREAD_NONE: i32 = 0;
pub const SPREAD_LIGHT: i32 = 1;
pub const SPREAD_NORMAL: i32 = 2;
pub const SPREAD_AGGRESSIVE: i32 = 3;

const BITRES: i32 = 3;
const QTHETA_OFFSET: i32 = 4;
const QTHETA_OFFSET_TWOPHASE: i32 = 16;
const ORDERY_TABLE: [usize; 30] = [
    1, 0, 3, 0, 2, 1, 7, 0, 4, 3, 6, 1, 5, 2, 15, 0, 8, 7, 12, 3, 11, 4, 14, 1, 9, 6, 13, 2, 10, 5,
];

pub fn hysteresis_decision(
    val: f32,
    thresholds: &[f32],
    hysteresis: &[f32],
    n: usize,
    prev: usize,
) -> usize {
    let mut i = 0usize;
    while i < n {
        if val < thresholds[i] {
            break;
        }
        i += 1;
    }
    if i > prev && val < thresholds[prev] + hysteresis[prev] {
        i = prev;
    }
    if i < prev && val > thresholds[prev - 1] - hysteresis[prev - 1] {
        i = prev;
    }
    i
}

pub fn celt_lcg_rand(seed: u32) -> u32 {
    seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223)
}

pub fn stereo_split(x: &mut [f32], y: &mut [f32], n: usize) {
    assert!(x.len() >= n);
    assert!(y.len() >= n);
    let c = 0.707_106_77f32;
    for j in 0..n {
        let l = c * x[j];
        let r = c * y[j];
        x[j] = l + r;
        y[j] = r - l;
    }
}

pub fn stereo_merge(x: &mut [f32], y: &mut [f32], mid: f32, n: usize) {
    assert!(x.len() >= n);
    assert!(y.len() >= n);

    let mut xp = 0.0f32;
    let mut side = 0.0f32;
    for j in 0..n {
        xp += y[j] * x[j];
        side += y[j] * y[j];
    }

    xp = mid * xp;
    let el = mid * mid + side - 2.0 * xp;
    let er = mid * mid + side + 2.0 * xp;
    if er < 6e-4 || el < 6e-4 {
        y[..n].copy_from_slice(&x[..n]);
        return;
    }

    let lgain = celt_rsqrt_norm(el);
    let rgain = celt_rsqrt_norm(er);
    for j in 0..n {
        let l = mid * x[j];
        let r = y[j];
        x[j] = lgain * (l - r);
        y[j] = rgain * (l + r);
    }
}

pub fn deinterleave_hadamard(x: &mut [f32], n0: usize, stride: usize, hadamard: bool) {
    assert!(stride > 0);
    let n = n0 * stride;
    assert!(x.len() >= n);
    let mut tmp = vec![0.0f32; n];

    if hadamard {
        let ordery = &ORDERY_TABLE[stride - 2..];
        for i in 0..stride {
            for j in 0..n0 {
                tmp[ordery[i] * n0 + j] = x[j * stride + i];
            }
        }
    } else {
        for i in 0..stride {
            for j in 0..n0 {
                tmp[i * n0 + j] = x[j * stride + i];
            }
        }
    }
    x[..n].copy_from_slice(&tmp);
}

pub fn interleave_hadamard(x: &mut [f32], n0: usize, stride: usize, hadamard: bool) {
    assert!(stride > 0);
    let n = n0 * stride;
    assert!(x.len() >= n);
    let mut tmp = vec![0.0f32; n];

    if hadamard {
        let ordery = &ORDERY_TABLE[stride - 2..];
        for i in 0..stride {
            for j in 0..n0 {
                tmp[j * stride + i] = x[ordery[i] * n0 + j];
            }
        }
    } else {
        for i in 0..stride {
            for j in 0..n0 {
                tmp[j * stride + i] = x[i * n0 + j];
            }
        }
    }
    x[..n].copy_from_slice(&tmp);
}

pub fn haar1(x: &mut [f32], n0: usize, stride: usize) {
    assert!(x.len() >= n0 * stride);
    let n0 = n0 >> 1;
    let c = 0.707_106_77f32;
    for i in 0..stride {
        for j in 0..n0 {
            let a = stride * 2 * j + i;
            let b = stride * (2 * j + 1) + i;
            let tmp1 = c * x[a];
            let tmp2 = c * x[b];
            x[a] = tmp1 + tmp2;
            x[b] = tmp1 - tmp2;
        }
    }
}

pub fn stereo_itheta(x: &[f32], y: &[f32], stereo: bool, n: usize) -> i32 {
    assert!(x.len() >= n);
    assert!(y.len() >= n);

    let mut emid = 1e-15f32;
    let mut eside = 1e-15f32;
    if stereo {
        for i in 0..n {
            let mid = 0.5 * (x[i] + y[i]);
            let side = 0.5 * (x[i] - y[i]);
            emid += mid * mid;
            eside += side * side;
        }
    } else {
        for i in 0..n {
            emid += x[i] * x[i];
            eside += y[i] * y[i];
        }
    }

    let mid = emid.sqrt();
    let side = eside.sqrt();
    (0.5 + 16_384.0 * 0.63662 * fast_atan2f(side, mid)).floor() as i32
}

pub fn compute_qn(n: usize, b: i32, offset: i32, pulse_cap: i32, stereo: bool) -> i32 {
    const EXP2_TABLE8: [i32; 8] = [16384, 17866, 19483, 21247, 23170, 25267, 27554, 30048];
    let mut n2 = 2 * n as i32 - 1;
    if stereo && n == 2 {
        n2 -= 1;
    }

    let mut qb = (b + n2 * offset).div_euclid(n2);
    qb = qb.min(b - pulse_cap - (4 << BITRES));
    qb = qb.min(8 << BITRES);

    if qb < (1 << BITRES >> 1) {
        1
    } else {
        let qn = EXP2_TABLE8[(qb & 0x7) as usize] >> (14 - (qb >> BITRES));
        ((qn + 1) >> 1) << 1
    }
}

pub fn theta_metrics(
    n: usize,
    b: i32,
    pulse_cap: i32,
    stereo: bool,
    unquantized: i16,
) -> (i32, i32) {
    let offset = (pulse_cap >> 1)
        - if stereo && n == 2 {
            QTHETA_OFFSET_TWOPHASE
        } else {
            QTHETA_OFFSET
        };
    let qn = compute_qn(n, b, offset, pulse_cap, stereo);
    let imid = bitexact_cos(unquantized);
    let iside = bitexact_cos(16_384 - unquantized);
    let delta = frac_mul16(
        (n as i32 - 1) << 7,
        bitexact_log2tan(iside as i32, imid as i32),
    );
    (qn, delta)
}
