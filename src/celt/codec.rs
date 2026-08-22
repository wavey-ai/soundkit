//! CELT frame-control helpers ported from the official `celt/celt.c`,
//! `celt/celt_encoder.c`, and `celt/celt_decoder.c` control path.

use crate::celt::bands::{
    anti_collapse, haar1, quant_all_bands_mono_with_scratch, quant_all_bands_stereo_with_scratch,
    BandCoder, BandScratch, SPREAD_NORMAL,
};
use crate::celt::entropy::{RangeDecoder, RangeEncoder};
use crate::celt::mathops::{celt_exp2, ec_ilog};
use crate::celt::modes::CeltMode;
use crate::celt::pitch::PrefilterDecision;
use crate::celt::quant_bands::{
    amp2_log2, quant_coarse_energy, quant_energy_finalise, quant_fine_energy,
    unquant_coarse_energy, unquant_energy_finalise, unquant_fine_energy, E_MEANS,
};
use crate::celt::rate::{
    clt_compute_allocation_with_scratch, Allocation, AllocationCoder, AllocationInfo,
    AllocationScratch,
};
use crate::{Error, Result};

const BITRES: i32 = 3;
const ENERGY_FLOOR_DB: f32 = -28.0;

pub const TRIM_ICDF: [u8; 11] = [126, 124, 119, 109, 87, 41, 19, 9, 4, 2, 0];
pub const SPREAD_ICDF: [u8; 4] = [25, 23, 2, 0];
pub const TAPSET_ICDF: [u8; 3] = [2, 1, 0];

const TF_SELECT_TABLE: [[i32; 8]; 4] = [
    [0, -1, 0, -1, 0, -1, 0, -1],
    [0, -1, 0, -2, 1, 0, 1, -1],
    [0, -2, 0, -3, 2, 0, 1, -1],
    [0, -2, 0, -3, 3, 0, 1, -1],
];

pub fn init_caps(mode: &CeltMode, lm: usize, channels: usize) -> Vec<i32> {
    assert!(lm <= mode.max_lm);
    assert!((1..=2).contains(&channels));

    let mut cap = vec![0i32; mode.nb_ebands];
    init_caps_into(mode, lm, channels, &mut cap);
    cap
}

pub fn init_caps_into(mode: &CeltMode, lm: usize, channels: usize, cap: &mut Vec<i32>) {
    assert!(lm <= mode.max_lm);
    assert!((1..=2).contains(&channels));
    cap.resize(mode.nb_ebands, 0);
    for (i, value) in cap.iter_mut().enumerate().take(mode.nb_ebands) {
        let n = (mode.ebands[i + 1] as i32 - mode.ebands[i] as i32) << lm;
        let idx = mode.nb_ebands * (2 * lm + channels - 1) + i;
        *value = ((mode.cache.caps[idx] as i32 + 64) * channels as i32 * n) >> 2;
    }
}

pub fn encode_transient_flag(
    lm: usize,
    total_bits: i32,
    is_transient: bool,
    enc: &mut RangeEncoder,
) -> bool {
    if lm > 0 && enc.tell() + 3 <= total_bits {
        enc.encode_bit_logp(is_transient, 3);
        is_transient
    } else {
        false
    }
}

pub fn decode_transient_flag(lm: usize, total_bits: i32, dec: &mut RangeDecoder<'_>) -> bool {
    if lm > 0 && dec.tell() + 3 <= total_bits {
        dec.decode_bit_logp(3)
    } else {
        false
    }
}

pub fn tf_encode(
    start: usize,
    end: usize,
    is_transient: bool,
    tf_res: &mut [i32],
    lm: usize,
    mut tf_select: i32,
    enc: &mut RangeEncoder,
) {
    assert!(lm < TF_SELECT_TABLE.len());
    assert!(end <= tf_res.len());

    let mut curr = 0;
    let mut tf_changed = 0;
    let mut logp = if is_transient { 2 } else { 4 };
    let mut tell = enc.tell();
    let mut budget = enc.storage_bytes() as i32 * 8;
    let tf_select_rsv = lm > 0 && tell + logp + 1 <= budget;
    if tf_select_rsv {
        budget -= 1;
    }

    for value in tf_res.iter_mut().take(end).skip(start) {
        if tell + logp <= budget {
            enc.encode_bit_logp((*value ^ curr) != 0, logp as u32);
            tell = enc.tell();
            curr = *value;
            tf_changed |= curr;
        } else {
            *value = curr;
        }
        logp = if is_transient { 4 } else { 5 };
    }

    let transient_idx = usize::from(is_transient);
    if tf_select_rsv
        && TF_SELECT_TABLE[lm][4 * transient_idx + tf_changed as usize]
            != TF_SELECT_TABLE[lm][4 * transient_idx + 2 + tf_changed as usize]
    {
        enc.encode_bit_logp(tf_select != 0, 1);
    } else {
        tf_select = 0;
    }

    for value in tf_res.iter_mut().take(end).skip(start) {
        let idx = 4 * transient_idx + 2 * tf_select as usize + *value as usize;
        *value = TF_SELECT_TABLE[lm][idx];
    }
}

pub fn tf_decode(
    start: usize,
    end: usize,
    is_transient: bool,
    tf_res: &mut [i32],
    lm: usize,
    dec: &mut RangeDecoder<'_>,
) {
    assert!(lm < TF_SELECT_TABLE.len());
    assert!(end <= tf_res.len());

    let mut curr = 0;
    let mut tf_changed = 0;
    let mut logp = if is_transient { 2 } else { 4 };
    let mut budget = dec.storage_bytes() as i32 * 8;
    let mut tell = dec.tell();
    let tf_select_rsv = lm > 0 && tell + logp + 1 <= budget;
    if tf_select_rsv {
        budget -= 1;
    }

    for value in tf_res.iter_mut().take(end).skip(start) {
        if tell + logp <= budget {
            curr ^= i32::from(dec.decode_bit_logp(logp as u32));
            tell = dec.tell();
            tf_changed |= curr;
        }
        *value = curr;
        logp = if is_transient { 4 } else { 5 };
    }

    let transient_idx = usize::from(is_transient);
    let tf_select = if tf_select_rsv
        && TF_SELECT_TABLE[lm][4 * transient_idx + tf_changed as usize]
            != TF_SELECT_TABLE[lm][4 * transient_idx + 2 + tf_changed as usize]
    {
        i32::from(dec.decode_bit_logp(1))
    } else {
        0
    };

    for value in tf_res.iter_mut().take(end).skip(start) {
        let idx = 4 * transient_idx + 2 * tf_select as usize + *value as usize;
        *value = TF_SELECT_TABLE[lm][idx];
    }
}

pub fn encode_spread_decision(spread: i32, total_bits: i32, enc: &mut RangeEncoder) -> i32 {
    if enc.tell() + 4 <= total_bits {
        enc.encode_icdf(spread as usize, &SPREAD_ICDF, 5);
        spread
    } else {
        SPREAD_NORMAL
    }
}

pub fn decode_spread_decision(total_bits: i32, dec: &mut RangeDecoder<'_>) -> i32 {
    if dec.tell() + 4 <= total_bits {
        dec.decode_icdf(&SPREAD_ICDF, 5) as i32
    } else {
        SPREAD_NORMAL
    }
}

pub fn encode_prefilter(
    start: usize,
    total_bits: i32,
    prefilter: Option<PrefilterDecision>,
    enc: &mut RangeEncoder,
) -> bool {
    if start != 0 || enc.tell() + 16 > total_bits {
        return false;
    }
    let Some(prefilter) = prefilter.filter(|prefilter| prefilter.enabled) else {
        enc.encode_bit_logp(false, 1);
        return false;
    };

    enc.encode_bit_logp(true, 1);
    let pitch = (prefilter.pitch + 1) as u32;
    let octave = ec_ilog(pitch) - 5;
    enc.encode_uint(octave as u32, 6);
    enc.encode_bits(pitch - (16 << octave), (4 + octave) as u32);
    enc.encode_bits(prefilter.qgain as u32, 3);
    enc.encode_icdf(prefilter.tapset as usize, &TAPSET_ICDF, 2);
    true
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodedPrefilter {
    pub pitch: i32,
    pub qgain: i32,
    pub tapset: i32,
}

pub fn decode_prefilter(
    start: usize,
    total_bits: i32,
    dec: &mut RangeDecoder<'_>,
) -> Option<DecodedPrefilter> {
    if start != 0 || dec.tell() + 16 > total_bits {
        return None;
    }
    if !dec.decode_bit_logp(1) {
        return None;
    }

    let octave = dec.decode_uint(6) as i32;
    let pitch = ((16 << octave) + dec.decode_bits((4 + octave) as u32) as i32) - 1;
    let qgain = dec.decode_bits(3) as i32;
    let tapset = if dec.tell() + 2 <= total_bits {
        dec.decode_icdf(&TAPSET_ICDF, 2) as i32
    } else {
        0
    };
    Some(DecodedPrefilter {
        pitch,
        qgain,
        tapset,
    })
}

pub fn encode_dynalloc_offsets(
    mode: &CeltMode,
    start: usize,
    end: usize,
    offsets: &mut [i32],
    cap: &[i32],
    total_bits_frac: i32,
    channels: usize,
    lm: usize,
    enc: &mut RangeEncoder,
) -> i32 {
    assert!(end <= mode.nb_ebands);
    assert!(offsets.len() >= mode.nb_ebands);
    assert!(cap.len() >= mode.nb_ebands);

    let mut dynalloc_logp = 6;
    let mut total_boost = 0;
    let mut tell = enc.tell_frac() as i32;

    for i in start..end {
        let width = (channels as i32 * (mode.ebands[i + 1] as i32 - mode.ebands[i] as i32)) << lm;
        let quanta = (width << BITRES).min((6 << BITRES).max(width));
        let mut dynalloc_loop_logp = dynalloc_logp;
        let requested = offsets[i].max(0);
        let mut boost = 0;
        let mut j = 0;
        while tell + (dynalloc_loop_logp << BITRES) < total_bits_frac - total_boost
            && boost < cap[i]
        {
            let flag = j < requested;
            enc.encode_bit_logp(flag, dynalloc_loop_logp as u32);
            tell = enc.tell_frac() as i32;
            if !flag {
                break;
            }
            boost += quanta;
            total_boost += quanta;
            dynalloc_loop_logp = 1;
            j += 1;
        }
        if j > 0 {
            dynalloc_logp = 2.max(dynalloc_logp - 1);
        }
        offsets[i] = boost;
    }

    total_boost
}

pub fn decode_dynalloc_offsets(
    mode: &CeltMode,
    start: usize,
    end: usize,
    offsets: &mut [i32],
    cap: &[i32],
    mut total_bits_frac: i32,
    channels: usize,
    lm: usize,
    dec: &mut RangeDecoder<'_>,
) -> i32 {
    assert!(end <= mode.nb_ebands);
    assert!(offsets.len() >= mode.nb_ebands);
    assert!(cap.len() >= mode.nb_ebands);

    let mut dynalloc_logp = 6;
    let mut tell = dec.tell_frac() as i32;
    let mut total_boost = 0;

    for i in start..end {
        let width = (channels as i32 * (mode.ebands[i + 1] as i32 - mode.ebands[i] as i32)) << lm;
        let quanta = (width << BITRES).min((6 << BITRES).max(width));
        let mut dynalloc_loop_logp = dynalloc_logp;
        let mut boost = 0;
        while tell + (dynalloc_loop_logp << BITRES) < total_bits_frac && boost < cap[i] {
            if !dec.decode_bit_logp(dynalloc_loop_logp as u32) {
                tell = dec.tell_frac() as i32;
                break;
            }
            tell = dec.tell_frac() as i32;
            boost += quanta;
            total_boost += quanta;
            total_bits_frac -= quanta;
            dynalloc_loop_logp = 1;
        }
        offsets[i] = boost;
        if boost > 0 {
            dynalloc_logp = 2.max(dynalloc_logp - 1);
        }
    }

    total_boost
}

fn median_of_3(x: &[f32]) -> f32 {
    debug_assert!(x.len() >= 3);
    let (t0, t1) = if x[0] > x[1] {
        (x[1], x[0])
    } else {
        (x[0], x[1])
    };
    let t2 = x[2];
    if t1 < t2 {
        t1
    } else if t0 < t2 {
        t2
    } else {
        t0
    }
}

fn median_of_5(x: &[f32]) -> f32 {
    debug_assert!(x.len() >= 5);
    let t2 = x[2];
    let (mut t0, mut t1) = if x[0] > x[1] {
        (x[1], x[0])
    } else {
        (x[0], x[1])
    };
    let (mut t3, mut t4) = if x[3] > x[4] {
        (x[4], x[3])
    } else {
        (x[3], x[4])
    };
    if t0 > t3 {
        std::mem::swap(&mut t0, &mut t3);
        std::mem::swap(&mut t1, &mut t4);
    }
    if t2 > t1 {
        if t1 < t3 {
            t2.min(t3)
        } else {
            t4.min(t1)
        }
    } else if t2 < t3 {
        t1.min(t3)
    } else {
        t2.min(t4)
    }
}

fn l1_metric(tmp: &[f32], n: usize, lm: usize, bias: f32) -> f32 {
    let l1 = tmp.iter().take(n).map(|value| value.abs()).sum::<f32>();
    l1 + lm as f32 * bias * l1
}

#[derive(Clone, Debug, Default)]
struct TfAnalysisScratch {
    metric: Vec<i32>,
    path0: Vec<i32>,
    path1: Vec<i32>,
    tmp: Vec<f32>,
    tmp_1: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
fn tf_analysis(
    mode: &CeltMode,
    len: usize,
    is_transient: bool,
    tf_res: &mut [i32],
    lambda: i32,
    x: &[f32],
    lm: usize,
    tf_estimate: f32,
    importance: &[i32],
    scratch: &mut TfAnalysisScratch,
) -> i32 {
    let transient_idx = usize::from(is_transient);
    let bias = 0.04 * (-0.25f32).max(0.5 - tf_estimate);
    scratch.metric.resize(len, 0);
    scratch.path0.resize(len, 0);
    scratch.path1.resize(len, 0);
    let metric = &mut scratch.metric[..len];
    let path0 = &mut scratch.path0[..len];
    let path1 = &mut scratch.path1[..len];

    for i in 0..len {
        let band_width = mode.ebands[i + 1] as usize - mode.ebands[i] as usize;
        let n = band_width << lm;
        let narrow = band_width == 1;
        let offset = (mode.ebands[i] as usize) << lm;
        scratch.tmp.resize(n, 0.0);
        scratch.tmp[..n].copy_from_slice(&x[offset..offset + n]);
        let tmp = &mut scratch.tmp[..n];
        let mut best_l1 = l1_metric(tmp, n, if is_transient { lm } else { 0 }, bias);
        let mut best_level = 0i32;

        if is_transient && !narrow {
            scratch.tmp_1.resize(n, 0.0);
            scratch.tmp_1[..n].copy_from_slice(tmp);
            let tmp_1 = &mut scratch.tmp_1[..n];
            haar1(tmp_1, n >> lm, 1 << lm);
            let l1 = l1_metric(tmp_1, n, lm + 1, bias);
            if l1 < best_l1 {
                best_l1 = l1;
                best_level = -1;
            }
        }

        let extra_split = usize::from(!(is_transient || narrow));
        for k in 0..lm + extra_split {
            let b = if is_transient { lm - k - 1 } else { k + 1 };
            haar1(tmp, n >> k, 1 << k);
            let l1 = l1_metric(tmp, n, b, bias);
            if l1 < best_l1 {
                best_l1 = l1;
                best_level = k as i32 + 1;
            }
        }

        metric[i] = if is_transient {
            2 * best_level
        } else {
            -2 * best_level
        };
        if narrow && (metric[i] == 0 || metric[i] == -2 * lm as i32) {
            metric[i] -= 1;
        }
    }

    let mut selcost = [0i32; 2];
    for sel in 0..2 {
        let mut cost0 = importance[0]
            * (metric[0] - 2 * TF_SELECT_TABLE[lm][4 * transient_idx + 2 * sel]).abs();
        let mut cost1 = importance[0]
            * (metric[0] - 2 * TF_SELECT_TABLE[lm][4 * transient_idx + 2 * sel + 1]).abs()
            + if is_transient { 0 } else { lambda };
        for i in 1..len {
            let curr0 = cost0.min(cost1 + lambda);
            let curr1 = (cost0 + lambda).min(cost1);
            cost0 = curr0
                + importance[i]
                    * (metric[i] - 2 * TF_SELECT_TABLE[lm][4 * transient_idx + 2 * sel]).abs();
            cost1 = curr1
                + importance[i]
                    * (metric[i] - 2 * TF_SELECT_TABLE[lm][4 * transient_idx + 2 * sel + 1]).abs();
        }
        selcost[sel] = cost0.min(cost1);
    }

    let tf_select = i32::from(selcost[1] < selcost[0] && is_transient);
    let select_idx = tf_select as usize;
    let mut cost0 = importance[0]
        * (metric[0] - 2 * TF_SELECT_TABLE[lm][4 * transient_idx + 2 * select_idx]).abs();
    let mut cost1 = importance[0]
        * (metric[0] - 2 * TF_SELECT_TABLE[lm][4 * transient_idx + 2 * select_idx + 1]).abs()
        + if is_transient { 0 } else { lambda };

    for i in 1..len {
        let from0 = cost0;
        let from1 = cost1 + lambda;
        let curr0;
        if from0 < from1 {
            curr0 = from0;
            path0[i] = 0;
        } else {
            curr0 = from1;
            path0[i] = 1;
        }

        let from0 = cost0 + lambda;
        let from1 = cost1;
        let curr1;
        if from0 < from1 {
            curr1 = from0;
            path1[i] = 0;
        } else {
            curr1 = from1;
            path1[i] = 1;
        }

        cost0 = curr0
            + importance[i]
                * (metric[i] - 2 * TF_SELECT_TABLE[lm][4 * transient_idx + 2 * select_idx]).abs();
        cost1 = curr1
            + importance[i]
                * (metric[i] - 2 * TF_SELECT_TABLE[lm][4 * transient_idx + 2 * select_idx + 1])
                    .abs();
    }

    tf_res[len - 1] = if cost0 < cost1 { 0 } else { 1 };
    for i in (0..len - 1).rev() {
        tf_res[i] = if tf_res[i + 1] == 1 {
            path1[i + 1]
        } else {
            path0[i + 1]
        };
    }
    tf_select
}

#[derive(Clone, Debug, Default)]
struct DynallocAnalysisScratch {
    noise_floor: Vec<f32>,
    follower: Vec<f32>,
    band_log_e3: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
fn dynalloc_analysis_with_scratch(
    mode: &CeltMode,
    band_log_e: &[f32],
    band_log_e2: &[f32],
    old_band_e: &[f32],
    start: usize,
    end: usize,
    channels: usize,
    lm: usize,
    packet_bytes: usize,
    is_transient: bool,
    vbr: bool,
    constrained_vbr: bool,
    analysis_leak_boost: Option<&[u8; 19]>,
    tone_frequency: f32,
    toneishness: f32,
    offsets: &mut [i32],
    importance: &mut [i32],
    scratch: &mut DynallocAnalysisScratch,
) -> DynallocAnalysis {
    const LSB_DEPTH: i32 = 24;

    offsets[..mode.nb_ebands].fill(0);
    importance[..mode.nb_ebands].fill(13);
    let mut max_depth = -31.9f32;

    scratch.noise_floor.resize(mode.nb_ebands, 0.0);
    scratch.noise_floor[..mode.nb_ebands].fill(0.0);
    let noise_floor = &mut scratch.noise_floor[..mode.nb_ebands];
    for i in 0..end {
        noise_floor[i] = 0.0625 * mode.log_n[i] as f32 + 0.5 + (9 - LSB_DEPTH) as f32 - E_MEANS[i]
            + 0.0062 * (i as f32 + 5.0) * (i as f32 + 5.0);
    }

    let follower_len = channels * mode.nb_ebands;
    scratch.follower.resize(follower_len, 0.0);
    scratch.follower[..follower_len].fill(0.0);
    let follower = &mut scratch.follower[..follower_len];
    scratch.band_log_e3.resize(mode.nb_ebands, 0.0);
    scratch.band_log_e3[..mode.nb_ebands].fill(0.0);
    let band_log_e3 = &mut scratch.band_log_e3[..mode.nb_ebands];
    for c in 0..channels {
        for i in 0..end {
            max_depth = max_depth.max(band_log_e[c * mode.nb_ebands + i] - noise_floor[i]);
        }
    }
    if packet_bytes < 30 + 5 * lm {
        return DynallocAnalysis {
            total_boost: 0,
            max_depth,
        };
    }
    for c in 0..channels {
        let channel = c * mode.nb_ebands;
        band_log_e3[..end].copy_from_slice(&band_log_e2[channel..channel + end]);
        if lm == 0 {
            for i in 0..end.min(8) {
                band_log_e3[i] = band_log_e2[channel + i].max(old_band_e[channel + i]);
            }
        }

        let f = &mut follower[channel..channel + mode.nb_ebands];
        let mut last = 0usize;
        if end > 0 {
            f[0] = band_log_e3[0];
        }
        for i in 1..end {
            if band_log_e3[i] > band_log_e3[i - 1] + 0.5 {
                last = i;
            }
            f[i] = (f[i - 1] + 1.5).min(band_log_e3[i]);
        }
        for i in (0..last).rev() {
            f[i] = f[i].min((f[i + 1] + 2.0).min(band_log_e3[i]));
        }

        let offset = 1.0;
        if end >= 5 {
            for i in 2..end - 2 {
                f[i] = f[i].max(median_of_5(&band_log_e3[i - 2..]) - offset);
            }
        }
        if end >= 3 {
            let tmp = median_of_3(&band_log_e3[0..]) - offset;
            f[0] = f[0].max(tmp);
            if end > 1 {
                f[1] = f[1].max(tmp);
            }
            let tmp = median_of_3(&band_log_e3[end - 3..]) - offset;
            if end > 1 {
                f[end - 2] = f[end - 2].max(tmp);
            }
            f[end - 1] = f[end - 1].max(tmp);
        }

        for i in 0..end {
            f[i] = f[i].max(noise_floor[i]);
        }
    }

    if channels == 2 {
        for i in start..end {
            let right = mode.nb_ebands + i;
            follower[right] = follower[right].max(follower[i] - 4.0);
            follower[i] = follower[i].max(follower[right] - 4.0);
            follower[i] = 0.5
                * ((band_log_e[i] - follower[i]).max(0.0)
                    + (band_log_e[mode.nb_ebands + i] - follower[right]).max(0.0));
        }
    } else {
        for i in start..end {
            follower[i] = (band_log_e[i] - follower[i]).max(0.0);
        }
    }
    for i in start..end {
        importance[i] = (0.5 + 13.0 * celt_exp2(follower[i].min(4.0))).floor() as i32;
    }

    if (!vbr || constrained_vbr) && !is_transient {
        for value in follower.iter_mut().take(end).skip(start) {
            *value *= 0.5;
        }
    }
    for (i, value) in follower.iter_mut().enumerate().take(end).skip(start) {
        if i < 8 {
            *value *= 2.0;
        }
        if i >= 12 {
            *value *= 0.5;
        }
    }
    if toneishness > 0.98 {
        let frequency_bin = (0.5 + tone_frequency * 120.0 / core::f32::consts::PI).floor() as i32;
        for (i, value) in follower.iter_mut().enumerate().take(end).skip(start) {
            let band_start = mode.ebands[i] as i32;
            let band_end = mode.ebands[i + 1] as i32;
            if (band_start..=band_end).contains(&frequency_bin) {
                *value += 2.0;
            }
            if (band_start - 1..=band_end + 1).contains(&frequency_bin) {
                *value += 1.0;
            }
            if (band_start - 2..=band_end + 2).contains(&frequency_bin) {
                *value += 1.0;
            }
            if (band_start - 3..=band_end + 3).contains(&frequency_bin) {
                *value += 0.5;
            }
        }
        if end >= 2 && frequency_bin >= mode.ebands[end] as i32 {
            follower[end - 1] += 2.0;
            follower[end - 2] += 1.0;
        }
    }
    if let Some(leak_boost) = analysis_leak_boost {
        for i in start..end.min(19) {
            follower[i] += leak_boost[i] as f32 * (1.0 / 64.0);
        }
    }

    let mut total_boost = 0;
    for i in start..end {
        let width = (channels as i32 * (mode.ebands[i + 1] as i32 - mode.ebands[i] as i32)) << lm;
        let follower = follower[i].min(4.0);
        let (boost, boost_bits) = if width < 6 {
            let boost = follower as i32;
            (boost, boost * width << BITRES)
        } else if width > 48 {
            let boost = (follower * 8.0) as i32;
            (boost, (boost * width << BITRES) / 8)
        } else {
            let boost = (follower * width as f32 / 6.0) as i32;
            (boost, boost * 6 << BITRES)
        };

        if (!vbr || (constrained_vbr && !is_transient))
            && ((total_boost + boost_bits) >> BITRES >> 3) > 2 * packet_bytes as i32 / 3
        {
            let cap = (2 * packet_bytes as i32 / 3) << BITRES << 3;
            offsets[i] = cap - total_boost;
            break;
        }

        offsets[i] = boost;
        total_boost += boost_bits;
    }
    DynallocAnalysis {
        total_boost,
        max_depth,
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct DynallocAnalysis {
    total_boost: i32,
    max_depth: f32,
}

pub fn encode_alloc_trim(
    alloc_trim: i32,
    total_bits_frac: i32,
    total_boost: i32,
    enc: &mut RangeEncoder,
) -> i32 {
    if enc.tell_frac() as i32 + (6 << BITRES) <= total_bits_frac - total_boost {
        enc.encode_icdf(alloc_trim as usize, &TRIM_ICDF, 7);
        alloc_trim
    } else {
        5
    }
}

pub fn decode_alloc_trim(total_bits_frac: i32, dec: &mut RangeDecoder<'_>) -> i32 {
    if dec.tell_frac() as i32 + (6 << BITRES) <= total_bits_frac {
        dec.decode_icdf(&TRIM_ICDF, 7) as i32
    } else {
        5
    }
}

#[derive(Clone, Debug)]
pub struct CeltFrameConfig {
    pub start: usize,
    pub end: usize,
    pub lm: usize,
    pub channels: usize,
    pub packet_bytes: usize,
    pub is_transient: bool,
    pub spread: i32,
    pub alloc_trim: i32,
    pub intensity: usize,
    pub dual_stereo: bool,
    pub disable_inv: bool,
    pub last_coded_bands: usize,
    pub vbr: bool,
    pub constrained_vbr: bool,
    pub prefilter: Option<PrefilterDecision>,
    pub band_log_e2: Option<Vec<f32>>,
    pub tf_estimate: f32,
    pub tf_chan: usize,
    pub tone_frequency: f32,
    pub toneishness: f32,
    pub signal_bandwidth: usize,
    pub analysis_leak_boost: Option<[u8; 19]>,
    pub vbr_state: Option<CeltVbrConfig>,
}

impl CeltFrameConfig {
    pub fn new(mode: &CeltMode, lm: usize, channels: usize, packet_bytes: usize) -> Result<Self> {
        if lm > mode.max_lm || !(1..=2).contains(&channels) || !(2..=1275).contains(&packet_bytes) {
            return Err(Error::BadArg);
        }
        Ok(Self {
            start: 0,
            end: mode.nb_ebands,
            lm,
            channels,
            packet_bytes,
            is_transient: false,
            spread: SPREAD_NORMAL,
            alloc_trim: 5,
            intensity: 0,
            dual_stereo: false,
            disable_inv: false,
            last_coded_bands: 0,
            vbr: false,
            constrained_vbr: false,
            prefilter: None,
            band_log_e2: None,
            tf_estimate: 0.0,
            tf_chan: 0,
            tone_frequency: -1.0,
            toneishness: 0.0,
            signal_bandwidth: mode.nb_ebands - 1,
            analysis_leak_boost: None,
            vbr_state: None,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CeltVbrConfig {
    pub equiv_rate: i32,
    pub vbr_rate: i32,
    pub effective_bytes: usize,
    pub reservoir: i32,
    pub drift: i32,
    pub offset: i32,
    pub count: i32,
    pub stereo_saving: f32,
    pub temporal_vbr: f32,
    pub analysis_valid: bool,
    pub activity: f32,
    pub tonality: f32,
    pub pitch_change: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CeltVbrUpdate {
    pub packet_bytes: usize,
    pub reservoir: i32,
    pub drift: i32,
    pub offset: i32,
    pub count: i32,
}

#[derive(Clone, Debug)]
pub struct CeltFrameEncodeResult {
    pub data: Vec<u8>,
    pub allocation: Allocation,
    pub tf_res: Vec<i32>,
    pub collapse_masks: Vec<u8>,
    pub silence: bool,
    pub prefilter_symbol: bool,
    pub is_transient: bool,
    pub spread: i32,
    pub alloc_trim: i32,
    pub vbr_update: Option<CeltVbrUpdate>,
}

#[derive(Clone, Debug)]
pub struct CeltFrameDecodeResult {
    pub x: Vec<f32>,
    pub y: Option<Vec<f32>>,
    pub allocation: Allocation,
    pub tf_res: Vec<i32>,
    pub collapse_masks: Vec<u8>,
    pub silence: bool,
    pub prefilter: Option<DecodedPrefilter>,
    pub is_transient: bool,
    pub spread: i32,
    pub alloc_trim: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CeltFrameDecodeInfo {
    pub allocation: AllocationInfo,
    pub silence: bool,
    pub prefilter: Option<DecodedPrefilter>,
    pub is_transient: bool,
    pub spread: i32,
    pub alloc_trim: i32,
    pub samples: usize,
}

#[derive(Clone, Debug, Default)]
pub struct CeltFrameDecodeScratch {
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    anti_collapse_x: Vec<f32>,
    pub tf_res: Vec<i32>,
    pub collapse_masks: Vec<u8>,
    pub band_e: Vec<f32>,
    offsets: Vec<i32>,
    cap: Vec<i32>,
    pub allocation: AllocationScratch,
    bands: BandScratch,
}

#[derive(Clone, Debug, Default)]
pub struct CeltFrameEncodeScratch {
    band_log_e: Vec<f32>,
    offsets: Vec<i32>,
    importance: Vec<i32>,
    cap: Vec<i32>,
    error: Vec<f32>,
    tf_res: Vec<i32>,
    collapse_masks: Vec<u8>,
    allocation: AllocationScratch,
    bands: BandScratch,
    dynalloc: DynallocAnalysisScratch,
    tf_analysis: TfAnalysisScratch,
}

fn validate_spectral_args(
    mode: &CeltMode,
    config: &CeltFrameConfig,
    x_len: usize,
    y_len: Option<usize>,
    band_e_len: usize,
    old_band_e_len: usize,
    energy_error_len: usize,
) -> Result<usize> {
    if config.lm > mode.max_lm
        || config.start >= config.end
        || config.end > mode.nb_ebands
        || !(1..=2).contains(&config.channels)
        || !(2..=1275).contains(&config.packet_bytes)
        || config.signal_bandwidth >= mode.nb_ebands
    {
        return Err(Error::BadArg);
    }
    let n = mode.short_mdct_size << config.lm;
    if x_len < n
        || (config.channels == 2 && y_len.unwrap_or(0) < n)
        || band_e_len < config.channels * mode.nb_ebands
        || old_band_e_len < config.channels * mode.nb_ebands
        || energy_error_len < config.channels * mode.nb_ebands
    {
        return Err(Error::BadArg);
    }
    Ok(n)
}

fn anti_collapse_reservation(is_transient: bool, lm: usize, bits: i32) -> i32 {
    if is_transient && lm >= 2 && bits >= (((lm as i32) + 2) << BITRES) {
        1 << BITRES
    } else {
        0
    }
}

fn compute_vbr_target(
    mode: &CeltMode,
    config: &CeltFrameConfig,
    vbr: &CeltVbrConfig,
    dynalloc: DynallocAnalysis,
) -> i32 {
    let channels = config.channels as i32;
    let lm = config.lm as i32;
    let lm_diff = (mode.max_lm - config.lm) as i32;
    let mut base_target = vbr.vbr_rate - ((40 * channels + 20) << BITRES);
    base_target += vbr.offset >> lm_diff;

    let coded_bands = if config.last_coded_bands != 0 {
        config.last_coded_bands
    } else {
        mode.nb_ebands
    };
    let mut coded_bins = (mode.ebands[coded_bands] as i32) << lm;
    if config.channels == 2 {
        coded_bins += (mode.ebands[config.intensity.min(coded_bands)] as i32) << lm;
    }

    let mut target = base_target;
    if vbr.analysis_valid && vbr.activity < 0.4 {
        target -= (((coded_bins << BITRES) as f32) * (0.4 - vbr.activity)) as i32;
    }
    if config.channels == 2 && coded_bins > 0 {
        let coded_stereo_bands = config.intensity.min(coded_bands);
        let coded_stereo_dof =
            ((mode.ebands[coded_stereo_bands] as i32) << lm) - coded_stereo_bands as i32;
        let max_frac = 0.8 * coded_stereo_dof as f32 / coded_bins as f32;
        let stereo_saving = vbr.stereo_saving.min(1.0);
        let stereo_savings = (max_frac * target as f32)
            .min((stereo_saving - 0.1) * ((coded_stereo_dof << BITRES) as f32));
        target -= stereo_savings as i32;
    }

    target += dynalloc.total_boost - (19 << config.lm);
    target += (2.0 * (config.tf_estimate - 0.044) * target as f32) as i32;

    if vbr.analysis_valid {
        let tonal = (vbr.tonality - 0.15).max(0.0) - 0.12;
        target += (((coded_bins << BITRES) as f32) * 1.2 * tonal) as i32;
        if vbr.pitch_change {
            target += (((coded_bins << BITRES) as f32) * 0.8) as i32;
        }
    }

    let bins = (mode.ebands[mode.nb_ebands - 2] as i32) << lm;
    let floor_depth = (((channels * bins) << BITRES) as f32 * dynalloc.max_depth) as i32;
    target = target.min(floor_depth.max(target >> 2));

    let constrained_vbr_blend = if config.lm == 1 { 0.67 } else { 0.50 };
    target = base_target + (constrained_vbr_blend * (target - base_target) as f32) as i32;

    if config.tf_estimate < 0.2 {
        let amount = 0.0000031 * (96_000 - vbr.equiv_rate).clamp(0, 32_000) as f32;
        let tvbr_factor = vbr.temporal_vbr * amount;
        target += (tvbr_factor * target as f32) as i32;
    }

    target.min(2 * base_target)
}

fn apply_vbr_shrink(
    mode: &CeltMode,
    config: &CeltFrameConfig,
    vbr: CeltVbrConfig,
    dynalloc: DynallocAnalysis,
    encoded_total_boost: i32,
    enc: &mut RangeEncoder,
    total_bits: &mut i32,
    total_bits_frac: &mut i32,
) -> CeltVbrUpdate {
    let lm_diff = (mode.max_lm - config.lm) as i32;
    let target = compute_vbr_target(mode, config, &vbr, dynalloc) + enc.tell_frac() as i32;
    let min_allowed = ((enc.tell_frac() as i32 + encoded_total_boost + ((1 << (BITRES + 3)) - 1))
        >> (BITRES + 3))
        + 2;
    let max_packet_bytes = config.packet_bytes.min(1275usize >> (3 - config.lm));
    let mut packet_bytes = ((target + (1 << (BITRES + 2))) >> (BITRES + 3))
        .max(min_allowed)
        .max(2)
        .min(max_packet_bytes as i32) as usize;
    let delta = target - vbr.vbr_rate;

    let quantized_target = (packet_bytes as i32) << (BITRES + 3);
    let mut reservoir = vbr.reservoir + quantized_target - vbr.vbr_rate;
    if reservoir < 0 {
        let adjust = (-reservoir) / (8 << BITRES);
        packet_bytes = (packet_bytes + adjust as usize).min(max_packet_bytes);
        reservoir = 0;
    }

    enc.shrink(packet_bytes);
    *total_bits = packet_bytes as i32 * 8;
    *total_bits_frac = *total_bits << BITRES;

    let mut count = vbr.count;
    let alpha = if count < 970 {
        count += 1;
        1.0 / (count + 20) as f32
    } else {
        0.001
    };
    let mut drift = vbr.drift;
    drift += (alpha * (((delta << lm_diff) - vbr.offset - drift) as f32)) as i32;
    let offset = -drift;

    CeltVbrUpdate {
        packet_bytes,
        reservoir,
        drift,
        offset,
        count,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn encode_spectral_frame(
    mode: &CeltMode,
    config: &CeltFrameConfig,
    x: &mut [f32],
    y: Option<&mut [f32]>,
    band_e: &[f32],
    old_band_e: &mut [f32],
    energy_error: &mut [f32],
    delayed_intra: &mut f32,
    seed: &mut u32,
) -> Result<CeltFrameEncodeResult> {
    let mut scratch = CeltFrameEncodeScratch::default();
    encode_spectral_frame_with_scratch(
        mode,
        config,
        x,
        y,
        band_e,
        old_band_e,
        energy_error,
        delayed_intra,
        seed,
        &mut scratch,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn encode_spectral_frame_with_scratch(
    mode: &CeltMode,
    config: &CeltFrameConfig,
    x: &mut [f32],
    mut y: Option<&mut [f32]>,
    band_e: &[f32],
    old_band_e: &mut [f32],
    energy_error: &mut [f32],
    delayed_intra: &mut f32,
    seed: &mut u32,
    scratch: &mut CeltFrameEncodeScratch,
) -> Result<CeltFrameEncodeResult> {
    let n = validate_spectral_args(
        mode,
        config,
        x.len(),
        y.as_ref().map(|slice| slice.len()),
        band_e.len(),
        old_band_e.len(),
        energy_error.len(),
    )?;
    if config.channels == 1 && y.is_some() {
        return Err(Error::BadArg);
    }

    let mut total_bits = (config.packet_bytes * 8) as i32;
    let mut total_bits_frac = total_bits << BITRES;
    let effective_bytes = config
        .vbr_state
        .map_or(config.packet_bytes, |vbr| vbr.effective_bytes);
    let eff_end = config.end.min(mode.eff_ebands);
    let mut enc = RangeEncoder::with_extra_capacity(config.packet_bytes, 1);
    let silence = false;
    if enc.tell() == 1 && enc.tell() < total_bits {
        enc.encode_bit_logp(silence, 15);
    }
    let prefilter_symbol = encode_prefilter(config.start, total_bits, config.prefilter, &mut enc);
    let is_transient = encode_transient_flag(config.lm, total_bits, config.is_transient, &mut enc);

    let band_count = config.channels * mode.nb_ebands;
    scratch.band_log_e.resize(band_count, 0.0);
    scratch.band_log_e[..band_count].fill(0.0);
    let band_log_e = &mut scratch.band_log_e[..band_count];
    amp2_log2(
        mode,
        eff_end,
        config.end,
        band_e,
        band_log_e,
        config.channels,
    );
    scratch.offsets.resize(mode.nb_ebands, 0);
    scratch.importance.resize(mode.nb_ebands, 13);
    let dynalloc = {
        let band_log_e_read = &*band_log_e;
        let band_log_e2 = config.band_log_e2.as_deref().unwrap_or(band_log_e_read);
        if band_log_e2.len() < config.channels * mode.nb_ebands {
            return Err(Error::BadArg);
        }
        dynalloc_analysis_with_scratch(
            mode,
            band_log_e_read,
            band_log_e2,
            old_band_e,
            config.start,
            config.end,
            config.channels,
            config.lm,
            effective_bytes,
            is_transient,
            config.vbr,
            config.constrained_vbr,
            config.analysis_leak_boost.as_ref(),
            config.tone_frequency,
            config.toneishness,
            &mut scratch.offsets,
            &mut scratch.importance,
            &mut scratch.dynalloc,
        )
    };
    for c in 0..config.channels {
        for i in config.start..config.end {
            let idx = i + c * mode.nb_ebands;
            if (band_log_e[idx] - old_band_e[idx]).abs() < 2.0 {
                band_log_e[idx] -= 0.25 * energy_error[idx];
            }
        }
    }
    scratch.error.resize(band_count, 0.0);
    scratch.error[..band_count].fill(0.0);
    quant_coarse_energy(
        mode,
        config.start,
        config.end,
        eff_end,
        band_log_e,
        old_band_e,
        total_bits as u32,
        &mut scratch.error,
        &mut enc,
        config.channels,
        config.lm,
        config.packet_bytes as i32,
        false,
        delayed_intra,
        true,
        0,
        false,
    );
    scratch.tf_res.resize(mode.nb_ebands, 0);
    scratch.tf_res[..mode.nb_ebands].fill(0);
    let tf_select = if effective_bytes >= 15 * config.channels && config.toneishness < 0.98 {
        let tf_x = if config.tf_chan == 1 {
            y.as_ref().map(|right| &right[..]).unwrap_or(&x[..])
        } else {
            &x[..]
        };
        let lambda = 80.max(20480 / effective_bytes as i32 + 2);
        let tf_select = tf_analysis(
            mode,
            eff_end,
            is_transient,
            &mut scratch.tf_res,
            lambda,
            tf_x,
            config.lm,
            config.tf_estimate,
            &scratch.importance,
            &mut scratch.tf_analysis,
        );
        for i in eff_end..config.end {
            scratch.tf_res[i] = scratch.tf_res[eff_end - 1];
        }
        tf_select
    } else {
        for value in scratch.tf_res.iter_mut().take(config.end) {
            *value = i32::from(is_transient);
        }
        0
    };
    tf_encode(
        config.start,
        config.end,
        is_transient,
        &mut scratch.tf_res,
        config.lm,
        tf_select,
        &mut enc,
    );
    let spread = encode_spread_decision(config.spread, total_bits, &mut enc);

    init_caps_into(mode, config.lm, config.channels, &mut scratch.cap);
    let total_boost = encode_dynalloc_offsets(
        mode,
        config.start,
        config.end,
        &mut scratch.offsets,
        &scratch.cap,
        total_bits_frac,
        config.channels,
        config.lm,
        &mut enc,
    );
    let alloc_trim = encode_alloc_trim(config.alloc_trim, total_bits_frac, total_boost, &mut enc);
    let vbr_update = config.vbr_state.map(|vbr| {
        apply_vbr_shrink(
            mode,
            config,
            vbr,
            dynalloc,
            total_boost,
            &mut enc,
            &mut total_bits,
            &mut total_bits_frac,
        )
    });

    let mut bits = total_bits_frac - enc.tell_frac() as i32 - 1;
    let anti_collapse_rsv = anti_collapse_reservation(is_transient, config.lm, bits);
    bits -= anti_collapse_rsv;
    let allocation = {
        let mut allocation_coder = AllocationCoder::Encode(&mut enc);
        let info = clt_compute_allocation_with_scratch(
            mode,
            config.start,
            config.end,
            &scratch.offsets,
            &scratch.cap,
            alloc_trim,
            config.intensity,
            config.dual_stereo,
            bits,
            config.channels,
            config.lm,
            Some(&mut allocation_coder),
            config.last_coded_bands,
            config.signal_bandwidth.min(config.end.saturating_sub(1)),
            &mut scratch.allocation,
        );
        scratch.allocation.to_allocation(info)
    };

    quant_fine_energy(
        mode,
        config.start,
        config.end,
        old_band_e,
        &mut scratch.error,
        &allocation.ebits,
        &mut enc,
        config.channels,
    );

    let short_blocks = is_transient;
    scratch.collapse_masks.resize(band_count, 0);
    scratch.collapse_masks[..band_count].fill(0);
    {
        let mut band_coder = BandCoder::Encode(&mut enc);
        if config.channels == 1 {
            quant_all_bands_mono_with_scratch(
                mode,
                config.start,
                config.end,
                x,
                &mut scratch.collapse_masks,
                band_e,
                &allocation.pulses,
                short_blocks,
                spread,
                allocation.intensity,
                &scratch.tf_res,
                total_bits_frac - anti_collapse_rsv,
                allocation.balance,
                &mut band_coder,
                config.lm,
                allocation.coded_bands,
                seed,
                0,
                false,
                &mut scratch.bands,
            );
        } else {
            let y = y.as_deref_mut().ok_or(Error::BadArg)?;
            quant_all_bands_stereo_with_scratch(
                mode,
                config.start,
                config.end,
                x,
                y,
                &mut scratch.collapse_masks,
                band_e,
                &allocation.pulses,
                short_blocks,
                spread,
                allocation.dual_stereo,
                allocation.intensity,
                &scratch.tf_res,
                total_bits_frac - anti_collapse_rsv,
                allocation.balance,
                &mut band_coder,
                config.lm,
                allocation.coded_bands,
                seed,
                9,
                config.disable_inv,
                false,
                &mut scratch.bands,
            );
        }
    }

    if anti_collapse_rsv > 0 {
        enc.encode_bits(0, 1);
    }
    quant_energy_finalise(
        mode,
        config.start,
        config.end,
        old_band_e,
        &mut scratch.error,
        &allocation.ebits,
        &allocation.fine_priority,
        total_bits - enc.tell(),
        &mut enc,
        config.channels,
    );
    energy_error[..config.channels * mode.nb_ebands].fill(0.0);
    for c in 0..config.channels {
        for i in config.start..config.end {
            let idx = i + c * mode.nb_ebands;
            energy_error[idx] = scratch.error[idx].clamp(-0.5, 0.5);
        }
    }
    enc.finish();
    *seed = enc.final_range();
    if enc.error() != 0 {
        return Err(Error::BufferTooSmall);
    }

    let coded = mode.ebands[config.end] as usize * (1usize << config.lm);
    if x.len() > coded {
        x[coded..n].fill(0.0);
    }
    if let Some(y) = y {
        if y.len() > coded {
            y[coded..n].fill(0.0);
        }
    }

    Ok(CeltFrameEncodeResult {
        data: enc.into_range_data(),
        allocation,
        tf_res: scratch.tf_res.clone(),
        collapse_masks: scratch.collapse_masks.clone(),
        silence,
        prefilter_symbol,
        is_transient,
        spread,
        alloc_trim,
        vbr_update,
    })
}

pub fn decode_spectral_frame(
    mode: &CeltMode,
    config: &CeltFrameConfig,
    data: &[u8],
    old_band_e: &mut [f32],
    seed: &mut u32,
) -> Result<CeltFrameDecodeResult> {
    let mut scratch = CeltFrameDecodeScratch::default();
    let info = decode_spectral_frame_into(mode, config, data, old_band_e, seed, &mut scratch)?;
    Ok(CeltFrameDecodeResult {
        x: scratch.x[..info.samples].to_vec(),
        y: (config.channels == 2).then(|| scratch.y[..info.samples].to_vec()),
        allocation: scratch.allocation.to_allocation(info.allocation),
        tf_res: scratch.tf_res.clone(),
        collapse_masks: scratch.collapse_masks.clone(),
        silence: info.silence,
        prefilter: info.prefilter,
        is_transient: info.is_transient,
        spread: info.spread,
        alloc_trim: info.alloc_trim,
    })
}

pub fn decode_spectral_frame_into(
    mode: &CeltMode,
    config: &CeltFrameConfig,
    data: &[u8],
    old_band_e: &mut [f32],
    seed: &mut u32,
    scratch: &mut CeltFrameDecodeScratch,
) -> Result<CeltFrameDecodeInfo> {
    decode_spectral_frame_into_impl(mode, config, data, old_band_e, None, None, seed, scratch)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_spectral_frame_into_with_anti_collapse(
    mode: &CeltMode,
    config: &CeltFrameConfig,
    data: &[u8],
    old_band_e: &mut [f32],
    old_log_e: &[f32],
    old_log_e2: &[f32],
    seed: &mut u32,
    scratch: &mut CeltFrameDecodeScratch,
) -> Result<CeltFrameDecodeInfo> {
    decode_spectral_frame_into_impl(
        mode,
        config,
        data,
        old_band_e,
        Some(old_log_e),
        Some(old_log_e2),
        seed,
        scratch,
    )
}

#[allow(clippy::too_many_arguments)]
fn decode_spectral_frame_into_impl(
    mode: &CeltMode,
    config: &CeltFrameConfig,
    data: &[u8],
    old_band_e: &mut [f32],
    old_log_e: Option<&[f32]>,
    old_log_e2: Option<&[f32]>,
    seed: &mut u32,
    scratch: &mut CeltFrameDecodeScratch,
) -> Result<CeltFrameDecodeInfo> {
    if data.len() != config.packet_bytes {
        return Err(Error::InvalidPacket);
    }
    let n = validate_spectral_args(
        mode,
        config,
        mode.short_mdct_size << config.lm,
        Some(mode.short_mdct_size << config.lm),
        config.channels * mode.nb_ebands,
        old_band_e.len(),
        config.channels * mode.nb_ebands,
    )?;

    let total_bits = (data.len() * 8) as i32;
    let total_bits_frac = total_bits << BITRES;
    let mut dec = RangeDecoder::new(data);
    let silence = if dec.tell() >= total_bits {
        true
    } else if dec.tell() == 1 {
        dec.decode_bit_logp(15)
    } else {
        false
    };
    let prefilter = decode_prefilter(config.start, total_bits, &mut dec);
    let is_transient = decode_transient_flag(config.lm, total_bits, &mut dec);
    let intra = if dec.tell() + 3 <= total_bits {
        dec.decode_bit_logp(3)
    } else {
        false
    };
    unquant_coarse_energy(
        mode,
        config.start,
        config.end,
        old_band_e,
        intra,
        &mut dec,
        config.channels,
        config.lm,
    );

    scratch.tf_res.resize(mode.nb_ebands, 0);
    scratch.tf_res[..mode.nb_ebands].fill(0);
    tf_decode(
        config.start,
        config.end,
        is_transient,
        &mut scratch.tf_res,
        config.lm,
        &mut dec,
    );
    let spread = decode_spread_decision(total_bits, &mut dec);

    init_caps_into(mode, config.lm, config.channels, &mut scratch.cap);
    scratch.offsets.resize(mode.nb_ebands, 0);
    scratch.offsets[..mode.nb_ebands].fill(0);
    let total_boost = decode_dynalloc_offsets(
        mode,
        config.start,
        config.end,
        &mut scratch.offsets,
        &scratch.cap,
        total_bits_frac,
        config.channels,
        config.lm,
        &mut dec,
    );
    let alloc_trim = decode_alloc_trim(total_bits_frac - total_boost, &mut dec);

    let mut bits = total_bits_frac - dec.tell_frac() as i32 - 1;
    let anti_collapse_rsv = anti_collapse_reservation(is_transient, config.lm, bits);
    bits -= anti_collapse_rsv;
    let allocation = {
        let mut allocation_coder = AllocationCoder::Decode(&mut dec);
        clt_compute_allocation_with_scratch(
            mode,
            config.start,
            config.end,
            &scratch.offsets,
            &scratch.cap,
            alloc_trim,
            config.intensity,
            config.dual_stereo,
            bits,
            config.channels,
            config.lm,
            Some(&mut allocation_coder),
            mode.nb_ebands,
            config.end.saturating_sub(1),
            &mut scratch.allocation,
        )
    };

    unquant_fine_energy(
        mode,
        config.start,
        config.end,
        old_band_e,
        &scratch.allocation.ebits,
        &mut dec,
        config.channels,
    );

    let short_blocks = is_transient;
    let mask_len = config.channels * mode.nb_ebands;
    scratch.collapse_masks.resize(mask_len, 0);
    scratch.collapse_masks[..mask_len].fill(0);
    scratch.x.resize(n, 0.0);
    scratch.x[..n].fill(0.0);
    if config.channels == 2 {
        scratch.y.resize(n, 0.0);
        scratch.y[..n].fill(0.0);
    } else {
        scratch.y.clear();
    }
    scratch.band_e.resize(mask_len, 0.0);
    scratch.band_e[..mask_len].fill(0.0);
    {
        let mut band_coder = BandCoder::Decode(&mut dec);
        if config.channels == 1 {
            quant_all_bands_mono_with_scratch(
                mode,
                config.start,
                config.end,
                &mut scratch.x,
                &mut scratch.collapse_masks,
                &scratch.band_e,
                &scratch.allocation.pulses,
                short_blocks,
                spread,
                allocation.intensity,
                &scratch.tf_res,
                total_bits_frac - anti_collapse_rsv,
                allocation.balance,
                &mut band_coder,
                config.lm,
                allocation.coded_bands,
                seed,
                0,
                false,
                &mut scratch.bands,
            );
        } else {
            quant_all_bands_stereo_with_scratch(
                mode,
                config.start,
                config.end,
                &mut scratch.x,
                &mut scratch.y,
                &mut scratch.collapse_masks,
                &scratch.band_e,
                &scratch.allocation.pulses,
                short_blocks,
                spread,
                allocation.dual_stereo,
                allocation.intensity,
                &scratch.tf_res,
                total_bits_frac - anti_collapse_rsv,
                allocation.balance,
                &mut band_coder,
                config.lm,
                allocation.coded_bands,
                seed,
                0,
                config.disable_inv,
                false,
                &mut scratch.bands,
            );
        }
    }

    let anti_collapse_on = anti_collapse_rsv > 0 && dec.decode_bits(1) != 0;
    unquant_energy_finalise(
        mode,
        config.start,
        config.end,
        old_band_e,
        &scratch.allocation.ebits,
        &scratch.allocation.fine_priority,
        total_bits - dec.tell(),
        &mut dec,
        config.channels,
    );
    if anti_collapse_on {
        if let (Some(old_log_e), Some(old_log_e2)) = (old_log_e, old_log_e2) {
            if config.channels == 1 {
                let _ = anti_collapse(
                    mode,
                    &mut scratch.x[..n],
                    &scratch.collapse_masks,
                    config.lm,
                    config.channels,
                    n,
                    config.start,
                    config.end,
                    old_band_e,
                    old_log_e,
                    old_log_e2,
                    &scratch.allocation.pulses,
                    *seed,
                    false,
                );
            } else {
                scratch.anti_collapse_x.resize(config.channels * n, 0.0);
                scratch.anti_collapse_x[..n].copy_from_slice(&scratch.x[..n]);
                scratch.anti_collapse_x[n..2 * n].copy_from_slice(&scratch.y[..n]);
                let _ = anti_collapse(
                    mode,
                    &mut scratch.anti_collapse_x[..config.channels * n],
                    &scratch.collapse_masks,
                    config.lm,
                    config.channels,
                    n,
                    config.start,
                    config.end,
                    old_band_e,
                    old_log_e,
                    old_log_e2,
                    &scratch.allocation.pulses,
                    *seed,
                    false,
                );
                scratch.x[..n].copy_from_slice(&scratch.anti_collapse_x[..n]);
                scratch.y[..n].copy_from_slice(&scratch.anti_collapse_x[n..2 * n]);
            }
        }
    }
    if silence {
        old_band_e[..config.channels * mode.nb_ebands].fill(ENERGY_FLOOR_DB);
    }
    *seed = dec.final_range();
    if dec.error() != 0 {
        return Err(Error::InvalidPacket);
    }

    Ok(CeltFrameDecodeInfo {
        allocation,
        silence,
        prefilter,
        is_transient,
        spread,
        alloc_trim,
        samples: n,
    })
}
