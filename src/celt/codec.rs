//! CELT frame-control helpers ported from the official `celt/celt.c`,
//! `celt/celt_encoder.c`, and `celt/celt_decoder.c` control path.

use crate::celt::bands::SPREAD_NORMAL;
use crate::celt::entropy::{RangeDecoder, RangeEncoder};
use crate::celt::modes::CeltMode;

const BITRES: i32 = 3;

pub const TRIM_ICDF: [u8; 11] = [126, 124, 119, 109, 87, 41, 19, 9, 4, 2, 0];
pub const SPREAD_ICDF: [u8; 4] = [25, 23, 2, 0];

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
