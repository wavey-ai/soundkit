//! CELT band helper routines, ported from official `celt/bands.c`.

use crate::celt::mathops::{
    bitexact_cos, bitexact_log2tan, celt_exp2, celt_exp2_db, celt_rsqrt_norm, celt_sqrt,
    fast_atan2f, frac_mul16,
};
use crate::celt::modes::CeltMode;
use crate::celt::quant_bands::E_MEANS;
use crate::celt::vq::renormalise_vector;

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

pub fn compute_band_energies(
    mode: &CeltMode,
    x: &[f32],
    band_e: &mut [f32],
    end: usize,
    channels: usize,
    lm: usize,
) {
    let n = mode.short_mdct_size << lm;
    assert!(end <= mode.nb_ebands);
    assert!(x.len() >= channels * n);
    assert!(band_e.len() >= channels * mode.nb_ebands);

    for c in 0..channels {
        for i in 0..end {
            let band_start = (mode.ebands[i] as usize) << lm;
            let band_end = (mode.ebands[i + 1] as usize) << lm;
            let base = c * n;
            let sum = 1e-27f32
                + x[base + band_start..base + band_end]
                    .iter()
                    .map(|v| v * v)
                    .sum::<f32>();
            band_e[i + c * mode.nb_ebands] = celt_sqrt(sum);
        }
    }
}

pub fn normalise_bands(
    mode: &CeltMode,
    freq: &[f32],
    x: &mut [f32],
    band_e: &[f32],
    end: usize,
    channels: usize,
    m: usize,
) {
    let n = m * mode.short_mdct_size;
    assert!(end <= mode.nb_ebands);
    assert!(freq.len() >= channels * n);
    assert!(x.len() >= channels * n);
    assert!(band_e.len() >= channels * mode.nb_ebands);

    for c in 0..channels {
        for i in 0..end {
            let g = 1.0 / (1e-27 + band_e[i + c * mode.nb_ebands]);
            for j in m * mode.ebands[i] as usize..m * mode.ebands[i + 1] as usize {
                x[j + c * n] = freq[j + c * n] * g;
            }
        }
    }
}

pub fn denormalise_bands(
    mode: &CeltMode,
    x: &[f32],
    freq: &mut [f32],
    band_log_e: &[f32],
    start: usize,
    end: usize,
    m: usize,
    downsample: usize,
    silence: bool,
) {
    let n = m * mode.short_mdct_size;
    assert!(end <= mode.nb_ebands);
    assert!(x.len() >= n);
    assert!(freq.len() >= n);
    assert!(band_log_e.len() >= mode.nb_ebands);

    let mut bound = m * mode.ebands[end] as usize;
    if downsample != 1 {
        bound = bound.min(n / downsample);
    }
    let mut start = start;
    let mut end = end;
    if silence {
        bound = 0;
        start = 0;
        end = 0;
    }

    freq[..m * mode.ebands[start] as usize].fill(0.0);
    for i in start..end {
        let band_start = m * mode.ebands[i] as usize;
        let band_end = m * mode.ebands[i + 1] as usize;
        let lg = band_log_e[i] + E_MEANS[i];
        let g = celt_exp2_db(lg.min(32.0));
        for j in band_start..band_end {
            freq[j] = x[j] * g;
        }
    }
    freq[bound..n].fill(0.0);
}

pub fn spreading_decision(
    mode: &CeltMode,
    x: &[f32],
    average: &mut i32,
    last_decision: i32,
    hf_average: &mut i32,
    tapset_decision: &mut i32,
    update_hf: bool,
    end: usize,
    channels: usize,
    m: usize,
    spread_weight: &[i32],
) -> i32 {
    assert!(end > 0);
    assert!(end <= mode.nb_ebands);
    let n0 = m * mode.short_mdct_size;
    assert!(x.len() >= channels * n0);
    assert!(spread_weight.len() >= end);

    if m * (mode.ebands[end] as usize - mode.ebands[end - 1] as usize) <= 8 {
        return SPREAD_NONE;
    }

    let mut sum = 0i32;
    let mut nb_bands = 0i32;
    let mut hf_sum = 0i32;
    for c in 0..channels {
        for i in 0..end {
            let n = m * (mode.ebands[i + 1] as usize - mode.ebands[i] as usize);
            if n <= 8 {
                continue;
            }
            let offset = m * mode.ebands[i] as usize + c * n0;
            let mut tcount = [0i32; 3];
            for j in 0..n {
                let x2n = x[offset + j] * x[offset + j] * n as f32;
                if x2n < 0.25 {
                    tcount[0] += 1;
                }
                if x2n < 0.0625 {
                    tcount[1] += 1;
                }
                if x2n < 0.015625 {
                    tcount[2] += 1;
                }
            }
            if i > mode.nb_ebands - 4 {
                hf_sum += 32 * (tcount[1] + tcount[0]) / n as i32;
            }
            let tmp = i32::from(2 * tcount[2] >= n as i32)
                + i32::from(2 * tcount[1] >= n as i32)
                + i32::from(2 * tcount[0] >= n as i32);
            sum += tmp * spread_weight[i];
            nb_bands += spread_weight[i];
        }
    }

    if update_hf {
        if hf_sum != 0 {
            hf_sum /= channels as i32 * (4 - mode.nb_ebands as i32 + end as i32);
        }
        *hf_average = (*hf_average + hf_sum) >> 1;
        hf_sum = *hf_average;
        if *tapset_decision == 2 {
            hf_sum += 4;
        } else if *tapset_decision == 0 {
            hf_sum -= 4;
        }
        *tapset_decision = if hf_sum > 22 {
            2
        } else if hf_sum > 18 {
            1
        } else {
            0
        };
    }

    debug_assert!(nb_bands > 0);
    sum = (sum << 8) / nb_bands;
    sum = (sum + *average) >> 1;
    *average = sum;
    sum = (3 * sum + (((3 - last_decision) << 7) + 64) + 2) >> 2;
    if sum < 80 {
        SPREAD_AGGRESSIVE
    } else if sum < 256 {
        SPREAD_NORMAL
    } else if sum < 384 {
        SPREAD_LIGHT
    } else {
        SPREAD_NONE
    }
}

#[allow(clippy::too_many_arguments)]
pub fn anti_collapse(
    mode: &CeltMode,
    x: &mut [f32],
    collapse_masks: &[u8],
    lm: usize,
    channels: usize,
    size: usize,
    start: usize,
    end: usize,
    log_e: &[f32],
    prev1_log_e: &[f32],
    prev2_log_e: &[f32],
    pulses: &[i32],
    mut seed: u32,
    encode: bool,
) -> u32 {
    assert!(x.len() >= channels * size);
    assert!(collapse_masks.len() >= end * channels);
    assert!(log_e.len() >= channels * mode.nb_ebands);
    assert!(prev1_log_e.len() >= channels * mode.nb_ebands);
    assert!(prev2_log_e.len() >= channels * mode.nb_ebands);
    assert!(pulses.len() >= end);

    for i in start..end {
        let n0 = mode.ebands[i + 1] as usize - mode.ebands[i] as usize;
        let depth = ((1 + pulses[i]) / n0 as i32) >> lm;
        let thresh = 0.5 * celt_exp2(-0.125 * depth as f32);
        let sqrt_1 = celt_rsqrt_norm((n0 << lm) as f32);

        for c in 0..channels {
            let mut prev1 = prev1_log_e[c * mode.nb_ebands + i];
            let mut prev2 = prev2_log_e[c * mode.nb_ebands + i];
            if !encode && channels == 1 && prev1_log_e.len() >= 2 * mode.nb_ebands {
                prev1 = prev1.max(prev1_log_e[mode.nb_ebands + i]);
                prev2 = prev2.max(prev2_log_e[mode.nb_ebands + i]);
            }
            let ediff = (log_e[c * mode.nb_ebands + i] - prev1.min(prev2)).max(0.0);
            let mut r = 2.0 * celt_exp2_db(-ediff);
            if lm == 3 {
                r *= core::f32::consts::SQRT_2;
            }
            r = r.min(thresh) * sqrt_1;

            let base = c * size + ((mode.ebands[i] as usize) << lm);
            let band_len = n0 << lm;
            let mut renormalize = false;
            for k in 0..1usize << lm {
                if collapse_masks[i * channels + c] & (1 << k) == 0 {
                    for j in 0..n0 {
                        seed = celt_lcg_rand(seed);
                        x[base + (j << lm) + k] = if seed & 0x8000 != 0 { r } else { -r };
                    }
                    renormalize = true;
                }
            }
            if renormalize {
                renormalise_vector(&mut x[base..base + band_len], band_len, 1.0);
            }
        }
    }
    seed
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
