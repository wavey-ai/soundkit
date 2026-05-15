//! CELT band helper routines, ported from official `celt/bands.c`.

use crate::celt::cwrs::get_pulses;
use crate::celt::entropy::{RangeDecoder, RangeEncoder};
use crate::celt::mathops::{
    bitexact_cos, bitexact_log2tan, celt_exp2, celt_exp2_db, celt_rsqrt_norm, celt_sqrt,
    fast_atan2f, frac_mul16, isqrt32,
};
use crate::celt::modes::{bits2pulses_signed, pulses2bits_signed, CeltMode};
use crate::celt::quant_bands::E_MEANS;
use crate::celt::vq::{alg_quant, alg_unquant, renormalise_vector};

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

fn intensity_stereo(
    mode: &CeltMode,
    x: &mut [f32],
    y: &[f32],
    band_e: &[f32],
    band: usize,
    n: usize,
) {
    let left = band_e[band];
    let right = band_e[band + mode.nb_ebands];
    let norm = 1e-15 + (1e-15 + left * left + right * right).sqrt();
    let a1 = left / norm;
    let a2 = right / norm;
    for j in 0..n {
        x[j] = a1 * x[j] + a2 * y[j];
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

    let mut qb = (b + n2 * offset) / n2;
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

pub enum BandCoder<'a> {
    Encode(&'a mut RangeEncoder),
    Decode(&'a mut RangeDecoder),
}

impl BandCoder<'_> {
    fn is_encode(&self) -> bool {
        matches!(self, Self::Encode(_))
    }

    fn tell_frac(&self) -> i32 {
        match self {
            Self::Encode(enc) => enc.tell_frac() as i32,
            Self::Decode(dec) => dec.tell_frac() as i32,
        }
    }

    fn encode_or_decode_uint(&mut self, value: usize, ft: u32) -> usize {
        match self {
            Self::Encode(enc) => {
                enc.encode_uint(value as u32, ft);
                value
            }
            Self::Decode(dec) => dec.decode_uint(ft) as usize,
        }
    }

    fn encode_or_decode_bit(&mut self, value: bool, logp: u32) -> bool {
        match self {
            Self::Encode(enc) => {
                enc.encode_bit_logp(value, logp);
                value
            }
            Self::Decode(dec) => dec.decode_bit_logp(logp),
        }
    }

    fn encode_or_decode_bits(&mut self, value: u32, bits: u32) -> u32 {
        match self {
            Self::Encode(enc) => {
                enc.encode_bits(value, bits);
                value
            }
            Self::Decode(dec) => dec.decode_bits(bits),
        }
    }

    fn encode_or_decode_range(&mut self, value: usize, fl: u32, fh: u32, ft: u32) -> usize {
        match self {
            Self::Encode(enc) => {
                enc.encode(fl, fh, ft);
                value
            }
            Self::Decode(dec) => {
                let fm = dec.decode(ft);
                let mut decoded = value;
                debug_assert!(fm >= fl && fm < fh);
                if !(fm >= fl && fm < fh) {
                    decoded = fl as usize;
                }
                dec.update(fl, fh, ft);
                decoded
            }
        }
    }

    fn alg_quant_or_unquant(
        &mut self,
        x: &mut [f32],
        n: usize,
        k: usize,
        spread: i32,
        b: usize,
        gain: f32,
        resynth: bool,
    ) -> u32 {
        match self {
            Self::Encode(enc) => alg_quant(x, n, k, spread, b, enc, gain, resynth),
            Self::Decode(dec) => alg_unquant(x, n, k, spread, b, dec, gain),
        }
    }

    fn clone_encoder(&mut self) -> Option<RangeEncoder> {
        match self {
            Self::Encode(enc) => Some((**enc).clone()),
            Self::Decode(_) => None,
        }
    }

    fn restore_encoder(&mut self, saved: RangeEncoder) {
        match self {
            Self::Encode(enc) => **enc = saved,
            Self::Decode(_) => unreachable!("theta RDO only runs while encoding"),
        }
    }
}

#[derive(Clone, Debug)]
struct BandContext<'a> {
    mode: &'a CeltMode,
    band_e: &'a [f32],
    band: usize,
    intensity: usize,
    spread: i32,
    tf_change: i32,
    remaining_bits: i32,
    seed: u32,
    resynth: bool,
    avoid_split_noise: bool,
    disable_inv: bool,
    theta_round: i32,
}

#[derive(Clone, Copy, Debug)]
struct SplitContext {
    inv: bool,
    imid: i32,
    iside: i32,
    delta: i32,
    itheta: i32,
    qalloc: i32,
}

fn quant_band_n1(
    ctx: &mut BandContext<'_>,
    coder: &mut BandCoder<'_>,
    x: &mut [f32],
    lowband_out: Option<&mut [f32]>,
) -> u32 {
    let mut sign = false;
    if ctx.remaining_bits >= 1 << BITRES {
        sign = coder.encode_or_decode_bits(u32::from(x[0] < 0.0), 1) != 0;
        ctx.remaining_bits -= 1 << BITRES;
    }
    if ctx.resynth {
        x[0] = if sign { -1.0 } else { 1.0 };
    }
    if let Some(out) = lowband_out {
        out[0] = x[0];
    }
    1
}

#[allow(clippy::too_many_arguments)]
fn compute_theta(
    ctx: &mut BandContext<'_>,
    coder: &mut BandCoder<'_>,
    x: &mut [f32],
    y: &mut [f32],
    n: usize,
    b: &mut i32,
    blocks: usize,
    blocks0: usize,
    lm: isize,
    stereo: bool,
    fill: &mut u32,
) -> SplitContext {
    let pulse_cap = ctx.mode.log_n[ctx.band] as i32 + (lm as i32) * (1 << BITRES);
    let offset = (pulse_cap >> 1)
        - if stereo && n == 2 {
            QTHETA_OFFSET_TWOPHASE
        } else {
            QTHETA_OFFSET
        };
    let mut qn = compute_qn(n, *b, offset, pulse_cap, stereo);
    if stereo && ctx.band >= ctx.intensity {
        qn = 1;
    }

    let mut inv = false;
    let mut itheta = if coder.is_encode() {
        stereo_itheta(x, y, stereo, n)
    } else {
        0
    };
    let tell = coder.tell_frac();

    if qn != 1 {
        if coder.is_encode() {
            if !stereo || ctx.theta_round == 0 {
                itheta = (itheta * qn + 8192) >> 14;
                if !stereo && ctx.avoid_split_noise && itheta > 0 && itheta < qn {
                    let unquantized = itheta * 16384 / qn;
                    let imid = bitexact_cos(unquantized as i16);
                    let iside = bitexact_cos((16384 - unquantized) as i16);
                    let delta = frac_mul16(
                        (n as i32 - 1) << 7,
                        bitexact_log2tan(iside as i32, imid as i32),
                    );
                    if delta > *b {
                        itheta = qn;
                    } else if delta < -*b {
                        itheta = 0;
                    }
                }
            } else {
                let bias = if itheta > 8192 {
                    32767 / qn
                } else {
                    -32767 / qn
                };
                let down = (itheta * qn + bias).clamp(0, (qn - 1) << 14) >> 14;
                itheta = if ctx.theta_round < 0 { down } else { down + 1 };
            }
        }

        if stereo && n > 2 {
            let p0 = 3i32;
            let x0 = qn / 2;
            let ft = p0 * (x0 + 1) + x0;
            match coder {
                BandCoder::Encode(enc) => {
                    let fl = if itheta <= x0 {
                        p0 * itheta
                    } else {
                        (itheta - 1 - x0) + (x0 + 1) * p0
                    };
                    let fh = if itheta <= x0 {
                        p0 * (itheta + 1)
                    } else {
                        (itheta - x0) + (x0 + 1) * p0
                    };
                    enc.encode(fl as u32, fh as u32, ft as u32);
                }
                BandCoder::Decode(dec) => {
                    let fs = dec.decode(ft as u32) as i32;
                    let xq = if fs < (x0 + 1) * p0 {
                        fs / p0
                    } else {
                        x0 + 1 + (fs - (x0 + 1) * p0)
                    };
                    let fl = if xq <= x0 {
                        p0 * xq
                    } else {
                        (xq - 1 - x0) + (x0 + 1) * p0
                    };
                    let fh = if xq <= x0 {
                        p0 * (xq + 1)
                    } else {
                        (xq - x0) + (x0 + 1) * p0
                    };
                    dec.update(fl as u32, fh as u32, ft as u32);
                    itheta = xq;
                }
            }
        } else if blocks0 > 1 || stereo {
            itheta = coder.encode_or_decode_uint(itheta as usize, (qn + 1) as u32) as i32;
        } else if coder.is_encode() {
            let half = qn >> 1;
            let fs = if itheta <= half {
                itheta + 1
            } else {
                qn + 1 - itheta
            };
            let fl = if itheta <= half {
                itheta * (itheta + 1) >> 1
            } else {
                ((half + 1) * (half + 1)) - ((qn + 1 - itheta) * (qn + 2 - itheta) >> 1)
            };
            let ft = (half + 1) * (half + 1);
            itheta = coder.encode_or_decode_range(
                itheta as usize,
                fl as u32,
                (fl + fs) as u32,
                ft as u32,
            ) as i32;
        } else {
            let half = qn >> 1;
            let ft = (half + 1) * (half + 1);
            let (fl, fs);
            let fm = match coder {
                BandCoder::Decode(dec) => dec.decode(ft as u32),
                BandCoder::Encode(_) => unreachable!("encode handled above"),
            };
            if fm < ((half * (half + 1)) >> 1) as u32 {
                itheta = ((isqrt32(8 * fm + 1) - 1) >> 1) as i32;
                fs = itheta + 1;
                fl = itheta * (itheta + 1) >> 1;
            } else {
                itheta = (2 * (qn + 1) - isqrt32(8 * (ft as u32 - fm - 1) + 1) as i32) >> 1;
                fs = qn + 1 - itheta;
                fl = ft - ((qn + 1 - itheta) * (qn + 2 - itheta) >> 1);
            }
            match coder {
                BandCoder::Decode(dec) => dec.update(fl as u32, (fl + fs) as u32, ft as u32),
                BandCoder::Encode(_) => unreachable!("encode handled above"),
            }
        }

        itheta = itheta * 16384 / qn;
        if coder.is_encode() && stereo {
            if itheta == 0 {
                intensity_stereo(ctx.mode, x, y, ctx.band_e, ctx.band, n);
            } else {
                stereo_split(x, y, n);
            }
        }
    } else if stereo {
        if coder.is_encode() {
            inv = itheta > 8192 && !ctx.disable_inv;
            if inv {
                for value in y.iter_mut().take(n) {
                    *value = -*value;
                }
            }
            intensity_stereo(ctx.mode, x, y, ctx.band_e, ctx.band, n);
        }
        if *b > 2 << BITRES && ctx.remaining_bits > 2 << BITRES {
            inv = coder.encode_or_decode_bit(inv, 2);
        } else {
            inv = false;
        }
        if ctx.disable_inv {
            inv = false;
        }
        itheta = 0;
    }

    let qalloc = coder.tell_frac() - tell;
    *b -= qalloc;

    let (imid, iside, delta);
    if itheta == 0 {
        imid = 32767;
        iside = 0;
        *fill &= (1 << blocks) - 1;
        delta = -16384;
    } else if itheta == 16384 {
        imid = 0;
        iside = 32767;
        *fill &= ((1 << blocks) - 1) << blocks;
        delta = 16384;
    } else {
        imid = bitexact_cos(itheta as i16) as i32;
        iside = bitexact_cos((16384 - itheta) as i16) as i32;
        delta = frac_mul16((n as i32 - 1) << 7, bitexact_log2tan(iside, imid));
    }

    SplitContext {
        inv,
        imid,
        iside,
        delta,
        itheta,
        qalloc,
    }
}

fn compute_channel_weights(ex: f32, ey: f32) -> (f32, f32) {
    let min_e = ex.min(ey);
    (ex + min_e / 3.0, ey + min_e / 3.0)
}

fn inner_prod(x: &[f32], y: &[f32], n: usize) -> f32 {
    x.iter()
        .zip(y.iter())
        .take(n)
        .map(|(left, right)| left * right)
        .sum()
}

fn shift_right_i32(value: i32, shift: isize) -> i32 {
    if shift >= 0 {
        value >> shift as u32
    } else {
        value << (-shift) as u32
    }
}

#[allow(clippy::too_many_arguments)]
fn quant_partition(
    ctx: &mut BandContext<'_>,
    coder: &mut BandCoder<'_>,
    x: &mut [f32],
    n: usize,
    b: i32,
    blocks: usize,
    lowband: Option<&[f32]>,
    lm: isize,
    gain: f32,
    mut fill: u32,
) -> u32 {
    let mode = ctx.mode;
    let cache_offset = mode.cache.index[(lm + 1) as usize * mode.nb_ebands + ctx.band] as usize;
    let cache = &mode.cache.bits[cache_offset..];

    if lm != -1 && b > cache[cache[0] as usize] as i32 + 12 && n > 2 {
        let mut b = b;
        let n2 = n >> 1;
        let (x0, x1) = x.split_at_mut(n2);
        let lm2 = lm - 1;
        let blocks0 = blocks;
        let mut blocks = blocks;
        if blocks == 1 {
            fill = (fill & 1) | (fill << 1);
        }
        blocks = (blocks + 1) >> 1;

        let mut low0 = None;
        let mut low1 = None;
        if let Some(low) = lowband {
            let (a, b_low) = low.split_at(n2);
            low0 = Some(a);
            low1 = Some(b_low);
        }

        let sctx = compute_theta(
            ctx, coder, x0, x1, n2, &mut b, blocks, blocks0, lm2, false, &mut fill,
        );
        let mid = (1.0 / 32768.0) * sctx.imid as f32;
        let side = (1.0 / 32768.0) * sctx.iside as f32;
        let mut delta = sctx.delta;

        if blocks0 > 1 && (sctx.itheta & 0x3fff) != 0 {
            if sctx.itheta > 8192 {
                delta -= shift_right_i32(delta, 4 - lm2);
            } else {
                delta = 0.min(delta + shift_right_i32((n2 as i32) << BITRES, 5 - lm2));
            }
        }

        let mut mbits = 0.max(b.min((b - delta) / 2));
        let mut sbits = b - mbits;
        ctx.remaining_bits -= sctx.qalloc;

        let rebalance = ctx.remaining_bits;
        if mbits >= sbits {
            let mut cm = quant_partition(
                ctx,
                coder,
                x0,
                n2,
                mbits,
                blocks,
                low0,
                lm2,
                gain * mid,
                fill,
            );
            let rebalance = mbits - (rebalance - ctx.remaining_bits);
            if rebalance > 3 << BITRES && sctx.itheta != 0 {
                sbits += rebalance - (3 << BITRES);
            }
            cm |= quant_partition(
                ctx,
                coder,
                x1,
                n2,
                sbits,
                blocks,
                low1,
                lm2,
                gain * side,
                fill >> blocks,
            ) << (blocks0 >> 1);
            cm
        } else {
            let mut cm = quant_partition(
                ctx,
                coder,
                x1,
                n2,
                sbits,
                blocks,
                low1,
                lm2,
                gain * side,
                fill >> blocks,
            ) << (blocks0 >> 1);
            let rebalance = sbits - (rebalance - ctx.remaining_bits);
            if rebalance > 3 << BITRES && sctx.itheta != 16384 {
                mbits += rebalance - (3 << BITRES);
            }
            cm |= quant_partition(
                ctx,
                coder,
                x0,
                n2,
                mbits,
                blocks,
                low0,
                lm2,
                gain * mid,
                fill,
            );
            cm
        }
    } else {
        let mut q = bits2pulses_signed(mode, ctx.band, lm, b);
        let mut curr_bits = pulses2bits_signed(mode, ctx.band, lm, q);
        ctx.remaining_bits -= curr_bits;
        while ctx.remaining_bits < 0 && q > 0 {
            ctx.remaining_bits += curr_bits;
            q -= 1;
            curr_bits = pulses2bits_signed(mode, ctx.band, lm, q);
            ctx.remaining_bits -= curr_bits;
        }

        if q != 0 {
            let k = get_pulses(q);
            coder.alg_quant_or_unquant(x, n, k, ctx.spread, blocks, gain, ctx.resynth)
        } else if ctx.resynth {
            let cm_mask = (1u32 << blocks) - 1;
            fill &= cm_mask;
            if fill == 0 {
                x[..n].fill(0.0);
                0
            } else if let Some(lowband) = lowband {
                for j in 0..n {
                    ctx.seed = celt_lcg_rand(ctx.seed);
                    let tmp = if ctx.seed & 0x8000 != 0 {
                        1.0 / 256.0
                    } else {
                        -1.0 / 256.0
                    };
                    x[j] = lowband[j] + tmp;
                }
                renormalise_vector(x, n, gain);
                fill
            } else {
                for value in x.iter_mut().take(n) {
                    ctx.seed = celt_lcg_rand(ctx.seed);
                    *value = (ctx.seed as i32 >> 20) as f32;
                }
                renormalise_vector(x, n, gain);
                cm_mask
            }
        } else {
            0
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn quant_band_mono(
    ctx: &mut BandContext<'_>,
    coder: &mut BandCoder<'_>,
    x: &mut [f32],
    n: usize,
    b: i32,
    mut blocks: usize,
    lowband: Option<&[f32]>,
    lm: isize,
    lowband_out: Option<&mut [f32]>,
    gain: f32,
    fill: u32,
) -> u32 {
    if n == 1 {
        return quant_band_n1(ctx, coder, x, lowband_out);
    }

    let n0 = n;
    let mut n_b = n / blocks;
    let mut blocks0 = blocks;
    let mut time_divide = 0usize;
    let recombine = ctx.tf_change.max(0) as usize;
    let mut tf_change = ctx.tf_change;
    let long_blocks = blocks0 == 1;
    let mut fill = fill;

    let mut lowband_storage = lowband.map(|lowband| lowband[..n].to_vec());
    let mut lowband = lowband_storage.as_deref_mut();

    const BIT_INTERLEAVE_TABLE: [u32; 16] = [0, 1, 1, 1, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3];
    for k in 0..recombine {
        if coder.is_encode() {
            haar1(x, n >> k, 1 << k);
        }
        if let Some(low) = lowband.as_deref_mut() {
            haar1(low, n >> k, 1 << k);
        }
        fill = BIT_INTERLEAVE_TABLE[(fill & 0xF) as usize]
            | (BIT_INTERLEAVE_TABLE[(fill >> 4) as usize] << 2);
    }
    blocks >>= recombine;
    n_b <<= recombine;

    while (n_b & 1) == 0 && tf_change < 0 {
        if coder.is_encode() {
            haar1(x, n_b, blocks);
        }
        if let Some(low) = lowband.as_deref_mut() {
            haar1(low, n_b, blocks);
        }
        fill |= fill << blocks;
        blocks <<= 1;
        n_b >>= 1;
        time_divide += 1;
        tf_change += 1;
    }

    blocks0 = blocks;
    let n_b0 = n_b;
    if blocks0 > 1 {
        if coder.is_encode() {
            deinterleave_hadamard(x, n_b >> recombine, blocks0 << recombine, long_blocks);
        }
        if let Some(low) = lowband.as_deref_mut() {
            deinterleave_hadamard(low, n_b >> recombine, blocks0 << recombine, long_blocks);
        }
    }

    let mut cm = quant_partition(
        ctx,
        coder,
        x,
        n,
        b,
        blocks,
        lowband.as_deref(),
        lm,
        gain,
        fill,
    );

    if ctx.resynth {
        if blocks0 > 1 {
            interleave_hadamard(x, n_b >> recombine, blocks0 << recombine, long_blocks);
        }

        n_b = n_b0;
        blocks = blocks0;
        for _ in 0..time_divide {
            blocks >>= 1;
            n_b <<= 1;
            cm |= cm >> blocks;
            haar1(x, n_b, blocks);
        }

        const BIT_DEINTERLEAVE_TABLE: [u32; 16] = [
            0x00, 0x03, 0x0C, 0x0F, 0x30, 0x33, 0x3C, 0x3F, 0xC0, 0xC3, 0xCC, 0xCF, 0xF0, 0xF3,
            0xFC, 0xFF,
        ];
        for k in 0..recombine {
            cm = BIT_DEINTERLEAVE_TABLE[cm as usize];
            haar1(x, n0 >> k, 1 << k);
        }
        blocks <<= recombine;

        if let Some(out) = lowband_out {
            let scale = celt_sqrt(n0 as f32);
            for j in 0..n0 {
                out[j] = scale * x[j];
            }
        }
        cm &= (1 << blocks) - 1;
    }

    cm
}

fn quant_band_n1_stereo(
    ctx: &mut BandContext<'_>,
    coder: &mut BandCoder<'_>,
    x: &mut [f32],
    y: &mut [f32],
    lowband_out: Option<&mut [f32]>,
) -> u32 {
    let mut sign = [false; 2];
    for c in 0..2 {
        if ctx.remaining_bits >= 1 << BITRES {
            let sample = if c == 0 { x[0] } else { y[0] };
            sign[c] = coder.encode_or_decode_bits(u32::from(sample < 0.0), 1) != 0;
            ctx.remaining_bits -= 1 << BITRES;
        }
    }
    if ctx.resynth {
        x[0] = if sign[0] { -1.0 } else { 1.0 };
        y[0] = if sign[1] { -1.0 } else { 1.0 };
    }
    if let Some(out) = lowband_out {
        out[0] = x[0];
    }
    1
}

#[allow(clippy::too_many_arguments)]
fn quant_band_stereo(
    ctx: &mut BandContext<'_>,
    coder: &mut BandCoder<'_>,
    x: &mut [f32],
    y: &mut [f32],
    n: usize,
    mut b: i32,
    blocks: usize,
    lowband: Option<&[f32]>,
    lm: isize,
    lowband_out: Option<&mut [f32]>,
    fill: u32,
) -> u32 {
    if n == 1 {
        return quant_band_n1_stereo(ctx, coder, x, y, lowband_out);
    }

    let orig_fill = fill;
    let mut fill = fill;
    let sctx = compute_theta(
        ctx, coder, x, y, n, &mut b, blocks, blocks, lm, true, &mut fill,
    );
    let mid = (1.0 / 32768.0) * sctx.imid as f32;
    let side = (1.0 / 32768.0) * sctx.iside as f32;

    let cm;
    if n == 2 {
        let sbits = if sctx.itheta != 0 && sctx.itheta != 16384 {
            1 << BITRES
        } else {
            0
        };
        let mbits = b - sbits;
        let side_first = sctx.itheta > 8192;
        ctx.remaining_bits -= sctx.qalloc + sbits;

        let mut sign = false;
        if side_first {
            if sbits != 0 {
                sign =
                    coder.encode_or_decode_bits(u32::from(y[0] * x[1] - y[1] * x[0] < 0.0), 1) != 0;
            }
            cm = quant_band_mono(
                ctx,
                coder,
                y,
                n,
                mbits,
                blocks,
                lowband,
                lm,
                lowband_out,
                1.0,
                orig_fill,
            );
            let sign = if sign { -1.0 } else { 1.0 };
            x[0] = -sign * y[1];
            x[1] = sign * y[0];
        } else {
            if sbits != 0 {
                sign =
                    coder.encode_or_decode_bits(u32::from(x[0] * y[1] - x[1] * y[0] < 0.0), 1) != 0;
            }
            cm = quant_band_mono(
                ctx,
                coder,
                x,
                n,
                mbits,
                blocks,
                lowband,
                lm,
                lowband_out,
                1.0,
                orig_fill,
            );
            let sign = if sign { -1.0 } else { 1.0 };
            y[0] = -sign * x[1];
            y[1] = sign * x[0];
        }

        if ctx.resynth {
            x[0] *= mid;
            x[1] *= mid;
            y[0] *= side;
            y[1] *= side;
            let tmp = x[0];
            x[0] = tmp - y[0];
            y[0] += tmp;
            let tmp = x[1];
            x[1] = tmp - y[1];
            y[1] += tmp;
        }
    } else {
        let mut mbits = 0.max(b.min((b - sctx.delta) / 2));
        let mut sbits = b - mbits;
        ctx.remaining_bits -= sctx.qalloc;

        let rebalance = ctx.remaining_bits;
        if mbits >= sbits {
            let mut cm_acc = quant_band_mono(
                ctx,
                coder,
                x,
                n,
                mbits,
                blocks,
                lowband,
                lm,
                lowband_out,
                1.0,
                fill,
            );
            let rebalance = mbits - (rebalance - ctx.remaining_bits);
            if rebalance > 3 << BITRES && sctx.itheta != 0 {
                sbits += rebalance - (3 << BITRES);
            }
            cm_acc |= quant_band_mono(
                ctx,
                coder,
                y,
                n,
                sbits,
                blocks,
                None,
                lm,
                None,
                side,
                fill >> blocks,
            );
            cm = cm_acc;
        } else {
            let mut cm_acc = quant_band_mono(
                ctx,
                coder,
                y,
                n,
                sbits,
                blocks,
                None,
                lm,
                None,
                side,
                fill >> blocks,
            );
            let rebalance = sbits - (rebalance - ctx.remaining_bits);
            if rebalance > 3 << BITRES && sctx.itheta != 16384 {
                mbits += rebalance - (3 << BITRES);
            }
            cm_acc |= quant_band_mono(
                ctx,
                coder,
                x,
                n,
                mbits,
                blocks,
                lowband,
                lm,
                lowband_out,
                1.0,
                fill,
            );
            cm = cm_acc;
        }
    }

    if ctx.resynth {
        if n != 2 {
            stereo_merge(x, y, mid, n);
        }
        if sctx.inv {
            for value in y.iter_mut().take(n) {
                *value = -*value;
            }
        }
    }
    cm
}

fn special_hybrid_folding_mono(mode: &CeltMode, norm: &mut [f32], start: usize, m: usize) {
    if start + 2 > mode.nb_ebands {
        return;
    }
    let n1 = m * (mode.ebands[start + 1] as usize - mode.ebands[start] as usize);
    let n2 = m * (mode.ebands[start + 2] as usize - mode.ebands[start + 1] as usize);
    if n2 > n1 {
        let copy_len = n2 - n1;
        let src_start = 2 * n1 - n2;
        norm.copy_within(src_start..src_start + copy_len, n1);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn quant_all_bands_mono(
    mode: &CeltMode,
    start: usize,
    end: usize,
    x: &mut [f32],
    collapse_masks: &mut [u8],
    band_e: &[f32],
    pulses: &[i32],
    short_blocks: bool,
    spread: i32,
    _intensity: usize,
    tf_res: &[i32],
    total_bits: i32,
    mut balance: i32,
    coder: &mut BandCoder<'_>,
    lm: usize,
    coded_bands: usize,
    seed: &mut u32,
    _complexity: i32,
    encode_resynth: bool,
) {
    assert!(start < end);
    assert!(end <= mode.nb_ebands);
    let m = 1usize << lm;
    let blocks = if short_blocks { m } else { 1 };
    let frame_len = m * mode.short_mdct_size;
    assert!(x.len() >= frame_len);
    assert!(collapse_masks.len() >= end);
    assert!(band_e.len() >= mode.nb_ebands);
    assert!(pulses.len() >= end);
    assert!(tf_res.len() >= end);

    let norm_offset = m * mode.ebands[start] as usize;
    let norm_limit = m * mode.ebands[mode.nb_ebands - 1] as usize;
    let mut norm = vec![0.0f32; norm_limit.saturating_sub(norm_offset)];

    let encode = coder.is_encode();
    let theta_rdo = false;
    let resynth = !encode || theta_rdo || encode_resynth;
    let mut lowband_offset = 0usize;
    let mut update_lowband = true;
    let mut ctx = BandContext {
        mode,
        band_e,
        band: start,
        intensity: mode.nb_ebands,
        spread,
        tf_change: 0,
        remaining_bits: 0,
        seed: *seed,
        resynth,
        avoid_split_noise: blocks > 1,
        disable_inv: false,
        theta_round: 0,
    };

    for i in start..end {
        ctx.band = i;
        let last = i == end - 1;
        let n = m * (mode.ebands[i + 1] as usize - mode.ebands[i] as usize);
        let tell = coder.tell_frac();

        if i != start {
            balance -= tell;
        }
        let remaining_bits = total_bits - tell - 1;
        ctx.remaining_bits = remaining_bits;
        let b = if i <= coded_bands.saturating_sub(1) {
            let curr_balance = balance / 3.min(coded_bands as i32 - i as i32);
            0.max(16383.min((remaining_bits + 1).min(pulses[i] + curr_balance)))
        } else {
            0
        };

        if resynth
            && (m * mode.ebands[i] as usize >= norm_offset + n || i == start + 1)
            && (update_lowband || lowband_offset == 0)
        {
            lowband_offset = i;
        }
        if resynth && i == start + 1 {
            special_hybrid_folding_mono(mode, &mut norm, start, m);
        }

        let tf_change = tf_res[i];
        ctx.tf_change = tf_change;

        let mut effective_lowband = None;
        let mut x_cm = if lowband_offset != 0
            && (spread != SPREAD_AGGRESSIVE || blocks > 1 || tf_change < 0)
        {
            let effective =
                (m * mode.ebands[lowband_offset] as usize).saturating_sub(norm_offset + n);
            let mut fold_start = lowband_offset;
            loop {
                fold_start -= 1;
                if m * mode.ebands[fold_start] as usize <= effective + norm_offset {
                    break;
                }
            }
            let mut fold_end = lowband_offset - 1;
            loop {
                fold_end += 1;
                if fold_end >= i
                    || m * mode.ebands[fold_end] as usize >= effective + norm_offset + n
                {
                    break;
                }
            }
            let mut cm = 0u32;
            for fold_i in fold_start..fold_end {
                cm |= collapse_masks[fold_i] as u32;
            }
            effective_lowband = Some(effective);
            cm
        } else {
            (1u32 << blocks) - 1
        };

        let band_start = m * mode.ebands[i] as usize;
        let band_end = band_start + n;
        let lowband_vec =
            effective_lowband.map(|effective| norm[effective..effective + n].to_vec());
        let mut lowband_out = if !last && resynth {
            Some(vec![0.0f32; n])
        } else {
            None
        };
        x_cm = quant_band_mono(
            &mut ctx,
            coder,
            &mut x[band_start..band_end],
            n,
            b,
            blocks,
            lowband_vec.as_deref(),
            lm as isize,
            lowband_out.as_deref_mut(),
            1.0,
            x_cm,
        );
        if let Some(out) = lowband_out {
            let norm_pos = band_start - norm_offset;
            if norm_pos + n <= norm.len() {
                norm[norm_pos..norm_pos + n].copy_from_slice(&out);
            }
        }

        collapse_masks[i] = x_cm as u8;
        balance += pulses[i] + tell;
        update_lowband = b > ((n as i32) << BITRES);
        ctx.avoid_split_noise = false;
    }

    *seed = ctx.seed;
}

#[allow(clippy::too_many_arguments)]
pub fn quant_all_bands_stereo(
    mode: &CeltMode,
    start: usize,
    end: usize,
    x: &mut [f32],
    y: &mut [f32],
    collapse_masks: &mut [u8],
    band_e: &[f32],
    pulses: &[i32],
    short_blocks: bool,
    spread: i32,
    mut dual_stereo: bool,
    intensity: usize,
    tf_res: &[i32],
    total_bits: i32,
    mut balance: i32,
    coder: &mut BandCoder<'_>,
    lm: usize,
    coded_bands: usize,
    seed: &mut u32,
    complexity: i32,
    disable_inv: bool,
    encode_resynth: bool,
) {
    assert!(start < end);
    assert!(end <= mode.nb_ebands);
    let m = 1usize << lm;
    let blocks = if short_blocks { m } else { 1 };
    let frame_len = m * mode.short_mdct_size;
    assert!(x.len() >= frame_len);
    assert!(y.len() >= frame_len);
    assert!(collapse_masks.len() >= end * 2);
    assert!(band_e.len() >= 2 * mode.nb_ebands);
    assert!(pulses.len() >= end);
    assert!(tf_res.len() >= end);

    let norm_offset = m * mode.ebands[start] as usize;
    let norm_limit = m * mode.ebands[mode.nb_ebands - 1] as usize;
    let norm_len = norm_limit.saturating_sub(norm_offset);
    let mut norm = vec![0.0f32; norm_len];
    let mut norm2 = vec![0.0f32; norm_len];

    let encode = coder.is_encode();
    let theta_rdo = encode && !dual_stereo && complexity >= 8;
    let resynth = !encode || theta_rdo || encode_resynth;
    let mut lowband_offset = 0usize;
    let mut update_lowband = true;
    let mut ctx = BandContext {
        mode,
        band_e,
        band: start,
        intensity,
        spread,
        tf_change: 0,
        remaining_bits: 0,
        seed: *seed,
        resynth,
        avoid_split_noise: blocks > 1,
        disable_inv,
        theta_round: 0,
    };

    for i in start..end {
        ctx.band = i;
        let last = i == end - 1;
        let n = m * (mode.ebands[i + 1] as usize - mode.ebands[i] as usize);
        let tell = coder.tell_frac();

        if i != start {
            balance -= tell;
        }
        let remaining_bits = total_bits - tell - 1;
        ctx.remaining_bits = remaining_bits;
        let b = if i <= coded_bands.saturating_sub(1) {
            let curr_balance = balance / 3.min(coded_bands as i32 - i as i32);
            0.max(16383.min((remaining_bits + 1).min(pulses[i] + curr_balance)))
        } else {
            0
        };

        if resynth
            && (m * mode.ebands[i] as usize >= norm_offset + n || i == start + 1)
            && (update_lowband || lowband_offset == 0)
        {
            lowband_offset = i;
        }
        if resynth && i == start + 1 {
            special_hybrid_folding_mono(mode, &mut norm, start, m);
            if dual_stereo {
                special_hybrid_folding_mono(mode, &mut norm2, start, m);
            }
        }

        let tf_change = tf_res[i];
        ctx.tf_change = tf_change;

        let mut effective_lowband = None;
        let (mut x_cm, mut y_cm) = if lowband_offset != 0
            && (spread != SPREAD_AGGRESSIVE || blocks > 1 || tf_change < 0)
        {
            let effective =
                (m * mode.ebands[lowband_offset] as usize).saturating_sub(norm_offset + n);
            let mut fold_start = lowband_offset;
            loop {
                fold_start -= 1;
                if m * mode.ebands[fold_start] as usize <= effective + norm_offset {
                    break;
                }
            }
            let mut fold_end = lowband_offset - 1;
            loop {
                fold_end += 1;
                if fold_end >= i
                    || m * mode.ebands[fold_end] as usize >= effective + norm_offset + n
                {
                    break;
                }
            }
            let mut x_mask = 0u32;
            let mut y_mask = 0u32;
            for fold_i in fold_start..fold_end {
                x_mask |= collapse_masks[fold_i * 2] as u32;
                y_mask |= collapse_masks[fold_i * 2 + 1] as u32;
            }
            effective_lowband = Some(effective);
            (x_mask, y_mask)
        } else {
            let mask = (1u32 << blocks) - 1;
            (mask, mask)
        };

        if dual_stereo && i == intensity {
            dual_stereo = false;
            if resynth {
                let upto = m * mode.ebands[i] as usize - norm_offset;
                for j in 0..upto.min(norm.len()) {
                    norm[j] = 0.5 * (norm[j] + norm2[j]);
                }
            }
        }

        let band_start = m * mode.ebands[i] as usize;
        let band_end = band_start + n;
        let lowband_vec =
            effective_lowband.map(|effective| norm[effective..effective + n].to_vec());
        let lowband_vec2 =
            effective_lowband.map(|effective| norm2[effective..effective + n].to_vec());
        let mut lowband_out = if !last && resynth {
            Some(vec![0.0f32; n])
        } else {
            None
        };
        let mut lowband_out2 = if !last && resynth && dual_stereo {
            Some(vec![0.0f32; n])
        } else {
            None
        };

        if dual_stereo {
            x_cm = quant_band_mono(
                &mut ctx,
                coder,
                &mut x[band_start..band_end],
                n,
                b / 2,
                blocks,
                lowband_vec.as_deref(),
                lm as isize,
                lowband_out.as_deref_mut(),
                1.0,
                x_cm,
            );
            y_cm = quant_band_mono(
                &mut ctx,
                coder,
                &mut y[band_start..band_end],
                n,
                b / 2,
                blocks,
                lowband_vec2.as_deref(),
                lm as isize,
                lowband_out2.as_deref_mut(),
                1.0,
                y_cm,
            );
        } else {
            let cm = x_cm | y_cm;
            if theta_rdo && i < intensity {
                let enc_save = coder
                    .clone_encoder()
                    .expect("theta RDO only runs while encoding");
                let ctx_save = ctx.clone();
                let x_save = x[band_start..band_end].to_vec();
                let y_save = y[band_start..band_end].to_vec();
                let (w0, w1) = compute_channel_weights(band_e[i], band_e[i + mode.nb_ebands]);

                let mut enc_down = enc_save.clone();
                let mut coder_down = BandCoder::Encode(&mut enc_down);
                let mut ctx_down = ctx_save.clone();
                ctx_down.theta_round = -1;
                let mut x_down = x_save.clone();
                let mut y_down = y_save.clone();
                let mut lowband_down = if !last && resynth {
                    Some(vec![0.0f32; n])
                } else {
                    None
                };
                let x_cm_down = quant_band_stereo(
                    &mut ctx_down,
                    &mut coder_down,
                    &mut x_down,
                    &mut y_down,
                    n,
                    b,
                    blocks,
                    lowband_vec.as_deref(),
                    lm as isize,
                    lowband_down.as_deref_mut(),
                    cm,
                );
                let dist_down =
                    w0 * inner_prod(&x_save, &x_down, n) + w1 * inner_prod(&y_save, &y_down, n);

                let mut enc_up = enc_save;
                let mut coder_up = BandCoder::Encode(&mut enc_up);
                let mut ctx_up = ctx_save;
                ctx_up.theta_round = 1;
                let mut x_up = x_save.clone();
                let mut y_up = y_save.clone();
                let mut lowband_up = if !last && resynth {
                    Some(vec![0.0f32; n])
                } else {
                    None
                };
                let x_cm_up = quant_band_stereo(
                    &mut ctx_up,
                    &mut coder_up,
                    &mut x_up,
                    &mut y_up,
                    n,
                    b,
                    blocks,
                    lowband_vec.as_deref(),
                    lm as isize,
                    lowband_up.as_deref_mut(),
                    cm,
                );
                let dist_up =
                    w0 * inner_prod(&x_save, &x_up, n) + w1 * inner_prod(&y_save, &y_up, n);

                if dist_down >= dist_up {
                    coder.restore_encoder(enc_down);
                    ctx = ctx_down;
                    x[band_start..band_end].copy_from_slice(&x_down);
                    y[band_start..band_end].copy_from_slice(&y_down);
                    lowband_out = lowband_down;
                    x_cm = x_cm_down;
                } else {
                    coder.restore_encoder(enc_up);
                    ctx = ctx_up;
                    x[band_start..band_end].copy_from_slice(&x_up);
                    y[band_start..band_end].copy_from_slice(&y_up);
                    lowband_out = lowband_up;
                    x_cm = x_cm_up;
                }
            } else {
                ctx.theta_round = 0;
                x_cm = quant_band_stereo(
                    &mut ctx,
                    coder,
                    &mut x[band_start..band_end],
                    &mut y[band_start..band_end],
                    n,
                    b,
                    blocks,
                    lowband_vec.as_deref(),
                    lm as isize,
                    lowband_out.as_deref_mut(),
                    cm,
                );
            }
            y_cm = x_cm;
        }

        if let Some(out) = lowband_out {
            let norm_pos = band_start - norm_offset;
            if norm_pos + n <= norm.len() {
                norm[norm_pos..norm_pos + n].copy_from_slice(&out);
            }
        }
        if let Some(out) = lowband_out2 {
            let norm_pos = band_start - norm_offset;
            if norm_pos + n <= norm2.len() {
                norm2[norm_pos..norm_pos + n].copy_from_slice(&out);
            }
        }

        collapse_masks[i * 2] = x_cm as u8;
        collapse_masks[i * 2 + 1] = y_cm as u8;
        balance += pulses[i] + tell;
        update_lowband = b > ((n as i32) << BITRES);
        ctx.avoid_split_noise = false;
    }

    *seed = ctx.seed;
}
