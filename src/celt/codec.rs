//! CELT frame-control helpers ported from the official `celt/celt.c`,
//! `celt/celt_encoder.c`, and `celt/celt_decoder.c` control path.

use crate::celt::bands::{quant_all_bands_mono, quant_all_bands_stereo, BandCoder, SPREAD_NORMAL};
use crate::celt::entropy::{RangeDecoder, RangeEncoder};
use crate::celt::mathops::ec_ilog;
use crate::celt::modes::CeltMode;
use crate::celt::pitch::PrefilterDecision;
use crate::celt::quant_bands::{
    amp2_log2, quant_coarse_energy, quant_energy_finalise, quant_fine_energy,
    unquant_coarse_energy, unquant_energy_finalise, unquant_fine_energy, E_MEANS,
};
use crate::celt::rate::{clt_compute_allocation, Allocation, AllocationCoder};
use crate::{Error, Result};

const BITRES: i32 = 3;

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

    (0..mode.nb_ebands)
        .map(|i| {
            let n = (mode.ebands[i + 1] as i32 - mode.ebands[i] as i32) << lm;
            let idx = mode.nb_ebands * (2 * lm + channels - 1) + i;
            ((mode.cache.caps[idx] as i32 + 64) * channels as i32 * n) >> 2
        })
        .collect()
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

pub fn decode_transient_flag(lm: usize, total_bits: i32, dec: &mut RangeDecoder) -> bool {
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
    dec: &mut RangeDecoder,
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

pub fn decode_spread_decision(total_bits: i32, dec: &mut RangeDecoder) -> i32 {
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
    dec: &mut RangeDecoder,
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
    dec: &mut RangeDecoder,
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

#[allow(clippy::too_many_arguments)]
fn dynalloc_analysis(
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
    offsets: &mut [i32],
) {
    const LSB_DEPTH: i32 = 24;

    offsets[..mode.nb_ebands].fill(0);
    if packet_bytes < 30 + 5 * lm {
        return;
    }

    let mut noise_floor = vec![0.0f32; mode.nb_ebands];
    for i in 0..end {
        noise_floor[i] = 0.0625 * mode.log_n[i] as f32 + 0.5 + (9 - LSB_DEPTH) as f32 - E_MEANS[i]
            + 0.0062 * (i as f32 + 5.0) * (i as f32 + 5.0);
    }

    let mut follower = vec![0.0f32; channels * mode.nb_ebands];
    let mut band_log_e3 = vec![0.0f32; mode.nb_ebands];
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

pub fn decode_alloc_trim(total_bits_frac: i32, dec: &mut RangeDecoder) -> i32 {
    if dec.tell_frac() as i32 + (6 << BITRES) <= total_bits_frac {
        dec.decode_icdf(&TRIM_ICDF, 7) as i32
    } else {
        5
    }
}

#[derive(Clone, Copy, Debug)]
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
            intensity: mode.nb_ebands,
            dual_stereo: false,
            disable_inv: false,
            last_coded_bands: 0,
            vbr: false,
            constrained_vbr: false,
            prefilter: None,
        })
    }
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

#[allow(clippy::too_many_arguments)]
pub fn encode_spectral_frame(
    mode: &CeltMode,
    config: &CeltFrameConfig,
    x: &mut [f32],
    mut y: Option<&mut [f32]>,
    band_e: &[f32],
    old_band_e: &mut [f32],
    energy_error: &mut [f32],
    delayed_intra: &mut f32,
    seed: &mut u32,
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

    let total_bits = (config.packet_bytes * 8) as i32;
    let total_bits_frac = total_bits << BITRES;
    let eff_end = config.end.min(mode.eff_ebands);
    let mut enc = RangeEncoder::new(config.packet_bytes);
    let silence = false;
    if enc.tell() == 1 && enc.tell() < total_bits {
        enc.encode_bit_logp(silence, 15);
    }
    let prefilter_symbol = encode_prefilter(config.start, total_bits, config.prefilter, &mut enc);
    let is_transient = encode_transient_flag(config.lm, total_bits, config.is_transient, &mut enc);

    let mut band_log_e = vec![0.0f32; config.channels * mode.nb_ebands];
    amp2_log2(
        mode,
        eff_end,
        config.end,
        band_e,
        &mut band_log_e,
        config.channels,
    );
    let band_log_e2 = band_log_e.clone();
    let mut offsets = vec![0i32; mode.nb_ebands];
    dynalloc_analysis(
        mode,
        &band_log_e,
        &band_log_e2,
        old_band_e,
        config.start,
        config.end,
        config.channels,
        config.lm,
        config.packet_bytes,
        is_transient,
        config.vbr,
        config.constrained_vbr,
        &mut offsets,
    );
    for c in 0..config.channels {
        for i in config.start..config.end {
            let idx = i + c * mode.nb_ebands;
            if (band_log_e[idx] - old_band_e[idx]).abs() < 2.0 {
                band_log_e[idx] -= 0.25 * energy_error[idx];
            }
        }
    }
    let mut error = vec![0.0f32; config.channels * mode.nb_ebands];
    quant_coarse_energy(
        mode,
        config.start,
        config.end,
        eff_end,
        &band_log_e,
        old_band_e,
        total_bits as u32,
        &mut error,
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
    let mut tf_res = vec![0i32; mode.nb_ebands];
    tf_encode(
        config.start,
        config.end,
        is_transient,
        &mut tf_res,
        config.lm,
        i32::from(is_transient),
        &mut enc,
    );
    let spread = encode_spread_decision(config.spread, total_bits, &mut enc);

    let cap = init_caps(mode, config.lm, config.channels);
    let total_boost = encode_dynalloc_offsets(
        mode,
        config.start,
        config.end,
        &mut offsets,
        &cap,
        total_bits_frac,
        config.channels,
        config.lm,
        &mut enc,
    );
    let alloc_trim = encode_alloc_trim(config.alloc_trim, total_bits_frac, total_boost, &mut enc);

    let mut bits = total_bits_frac - enc.tell_frac() as i32 - 1;
    let anti_collapse_rsv = anti_collapse_reservation(is_transient, config.lm, bits);
    bits -= anti_collapse_rsv;
    let allocation = {
        let mut allocation_coder = AllocationCoder::Encode(&mut enc);
        clt_compute_allocation(
            mode,
            config.start,
            config.end,
            &offsets,
            &cap,
            alloc_trim,
            config.intensity,
            config.dual_stereo,
            bits,
            config.channels,
            config.lm,
            Some(&mut allocation_coder),
            config.last_coded_bands,
            config.end.saturating_sub(1),
        )
    };

    quant_fine_energy(
        mode,
        config.start,
        config.end,
        old_band_e,
        &mut error,
        &allocation.ebits,
        &mut enc,
        config.channels,
    );

    let short_blocks = is_transient;
    let mut collapse_masks = vec![0u8; config.channels * mode.nb_ebands];
    {
        let mut band_coder = BandCoder::Encode(&mut enc);
        if config.channels == 1 {
            quant_all_bands_mono(
                mode,
                config.start,
                config.end,
                x,
                &mut collapse_masks,
                band_e,
                &allocation.pulses,
                short_blocks,
                spread,
                allocation.intensity,
                &tf_res,
                total_bits_frac - anti_collapse_rsv,
                allocation.balance,
                &mut band_coder,
                config.lm,
                allocation.coded_bands,
                seed,
                0,
                false,
            );
        } else {
            let y = y.as_deref_mut().ok_or(Error::BadArg)?;
            quant_all_bands_stereo(
                mode,
                config.start,
                config.end,
                x,
                y,
                &mut collapse_masks,
                band_e,
                &allocation.pulses,
                short_blocks,
                spread,
                allocation.dual_stereo,
                allocation.intensity,
                &tf_res,
                total_bits_frac - anti_collapse_rsv,
                allocation.balance,
                &mut band_coder,
                config.lm,
                allocation.coded_bands,
                seed,
                9,
                config.disable_inv,
                false,
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
        &mut error,
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
            energy_error[idx] = error[idx].clamp(-0.5, 0.5);
        }
    }
    enc.finish();
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
        data: enc.range_data().to_vec(),
        allocation,
        tf_res,
        collapse_masks,
        silence,
        prefilter_symbol,
        is_transient,
        spread,
        alloc_trim,
    })
}

pub fn decode_spectral_frame(
    mode: &CeltMode,
    config: &CeltFrameConfig,
    data: &[u8],
    old_band_e: &mut [f32],
    seed: &mut u32,
) -> Result<CeltFrameDecodeResult> {
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

    let mut tf_res = vec![0i32; mode.nb_ebands];
    tf_decode(
        config.start,
        config.end,
        is_transient,
        &mut tf_res,
        config.lm,
        &mut dec,
    );
    let spread = decode_spread_decision(total_bits, &mut dec);

    let cap = init_caps(mode, config.lm, config.channels);
    let mut offsets = vec![0i32; mode.nb_ebands];
    let total_boost = decode_dynalloc_offsets(
        mode,
        config.start,
        config.end,
        &mut offsets,
        &cap,
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
        clt_compute_allocation(
            mode,
            config.start,
            config.end,
            &offsets,
            &cap,
            alloc_trim,
            config.intensity,
            config.dual_stereo,
            bits,
            config.channels,
            config.lm,
            Some(&mut allocation_coder),
            mode.nb_ebands,
            config.end.saturating_sub(1),
        )
    };

    unquant_fine_energy(
        mode,
        config.start,
        config.end,
        old_band_e,
        &allocation.ebits,
        &mut dec,
        config.channels,
    );

    let short_blocks = is_transient;
    let mut collapse_masks = vec![0u8; config.channels * mode.nb_ebands];
    let mut x = vec![0.0f32; n];
    let mut y = (config.channels == 2).then(|| vec![0.0f32; n]);
    let band_e = vec![0.0f32; config.channels * mode.nb_ebands];
    {
        let mut band_coder = BandCoder::Decode(&mut dec);
        if config.channels == 1 {
            quant_all_bands_mono(
                mode,
                config.start,
                config.end,
                &mut x,
                &mut collapse_masks,
                &band_e,
                &allocation.pulses,
                short_blocks,
                spread,
                allocation.intensity,
                &tf_res,
                total_bits_frac - anti_collapse_rsv,
                allocation.balance,
                &mut band_coder,
                config.lm,
                allocation.coded_bands,
                seed,
                0,
                false,
            );
        } else {
            let y_ref = y.as_mut().expect("stereo y buffer");
            quant_all_bands_stereo(
                mode,
                config.start,
                config.end,
                &mut x,
                y_ref,
                &mut collapse_masks,
                &band_e,
                &allocation.pulses,
                short_blocks,
                spread,
                allocation.dual_stereo,
                allocation.intensity,
                &tf_res,
                total_bits_frac - anti_collapse_rsv,
                allocation.balance,
                &mut band_coder,
                config.lm,
                allocation.coded_bands,
                seed,
                0,
                config.disable_inv,
                false,
            );
        }
    }

    if anti_collapse_rsv > 0 {
        let _anti_collapse_on = dec.decode_bits(1) != 0;
    }
    unquant_energy_finalise(
        mode,
        config.start,
        config.end,
        old_band_e,
        &allocation.ebits,
        &allocation.fine_priority,
        total_bits - dec.tell(),
        &mut dec,
        config.channels,
    );
    if dec.error() != 0 {
        return Err(Error::InvalidPacket);
    }

    Ok(CeltFrameDecodeResult {
        x,
        y,
        allocation,
        tf_res,
        collapse_masks,
        silence,
        prefilter,
        is_transient,
        spread,
        alloc_trim,
    })
}
