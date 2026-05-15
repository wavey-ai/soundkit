//! CELT energy quantization, ported from official `celt/quant_bands.c`.

use crate::celt::entropy::{RangeDecoder, RangeEncoder};
use crate::celt::laplace::{decode_laplace, encode_laplace};
use crate::celt::mathops::celt_log2_db;
use crate::celt::modes::CeltMode;

pub const E_MEANS: [f32; 25] = [
    6.437500, 6.250000, 5.750000, 5.312500, 5.062500, 4.812500, 4.500000, 4.375000, 4.875000,
    4.687500, 4.562500, 4.437500, 4.875000, 4.625000, 4.312500, 4.500000, 4.375000, 4.625000,
    4.750000, 4.437500, 3.750000, 3.750000, 3.750000, 3.750000, 3.750000,
];

const PRED_COEF: [f32; 4] = [
    29440.0 / 32768.0,
    26112.0 / 32768.0,
    21248.0 / 32768.0,
    16384.0 / 32768.0,
];
const BETA_COEF: [f32; 4] = [
    30147.0 / 32768.0,
    22282.0 / 32768.0,
    12124.0 / 32768.0,
    6554.0 / 32768.0,
];
const BETA_INTRA: f32 = 4915.0 / 32768.0;
const MAX_FINE_BITS: i32 = 8;

const E_PROB_MODEL: [[[u8; 42]; 2]; 4] = [
    [
        [
            72, 127, 65, 129, 66, 128, 65, 128, 64, 128, 62, 128, 64, 128, 64, 128, 92, 78, 92, 79,
            92, 78, 90, 79, 116, 41, 115, 40, 114, 40, 132, 26, 132, 26, 145, 17, 161, 12, 176, 10,
            177, 11,
        ],
        [
            24, 179, 48, 138, 54, 135, 54, 132, 53, 134, 56, 133, 55, 132, 55, 132, 61, 114, 70,
            96, 74, 88, 75, 88, 87, 74, 89, 66, 91, 67, 100, 59, 108, 50, 120, 40, 122, 37, 97, 43,
            78, 50,
        ],
    ],
    [
        [
            83, 78, 84, 81, 88, 75, 86, 74, 87, 71, 90, 73, 93, 74, 93, 74, 109, 40, 114, 36, 117,
            34, 117, 34, 143, 17, 145, 18, 146, 19, 162, 12, 165, 10, 178, 7, 189, 6, 190, 8, 177,
            9,
        ],
        [
            23, 178, 54, 115, 63, 102, 66, 98, 69, 99, 74, 89, 71, 91, 73, 91, 78, 89, 86, 80, 92,
            66, 93, 64, 102, 59, 103, 60, 104, 60, 117, 52, 123, 44, 138, 35, 133, 31, 97, 38, 77,
            45,
        ],
    ],
    [
        [
            61, 90, 93, 60, 105, 42, 107, 41, 110, 45, 116, 38, 113, 38, 112, 38, 124, 26, 132, 27,
            136, 19, 140, 20, 155, 14, 159, 16, 158, 18, 170, 13, 177, 10, 187, 8, 192, 6, 175, 9,
            159, 10,
        ],
        [
            21, 178, 59, 110, 71, 86, 75, 85, 84, 83, 91, 66, 88, 73, 87, 72, 92, 75, 98, 72, 105,
            58, 107, 54, 115, 52, 114, 55, 112, 56, 129, 51, 132, 40, 150, 33, 140, 29, 98, 35, 77,
            42,
        ],
    ],
    [
        [
            42, 121, 96, 66, 108, 43, 111, 40, 117, 44, 123, 32, 120, 36, 119, 33, 127, 33, 134,
            34, 139, 21, 147, 23, 152, 20, 158, 25, 154, 26, 166, 21, 173, 16, 184, 13, 184, 10,
            150, 13, 139, 15,
        ],
        [
            22, 178, 63, 114, 74, 82, 84, 83, 92, 82, 103, 62, 96, 72, 96, 67, 101, 73, 107, 72,
            113, 55, 118, 52, 125, 52, 118, 52, 117, 55, 135, 49, 137, 39, 157, 32, 145, 29, 97,
            33, 77, 40,
        ],
    ],
];

const SMALL_ENERGY_ICDF: [u8; 3] = [2, 1, 0];

fn loss_distortion(
    e_bands: &[f32],
    old_e_bands: &[f32],
    start: usize,
    end: usize,
    len: usize,
    channels: usize,
) -> f32 {
    let mut dist = 0.0f32;
    for c in 0..channels {
        for i in start..end {
            let d = e_bands[i + c * len] - old_e_bands[i + c * len];
            dist += d * d;
        }
    }
    dist.min(200.0)
}

fn quant_coarse_energy_impl(
    mode: &CeltMode,
    start: usize,
    end: usize,
    e_bands: &[f32],
    old_e_bands: &mut [f32],
    budget: i32,
    mut tell: i32,
    prob_model: &[u8; 42],
    error: &mut [f32],
    enc: &mut RangeEncoder,
    channels: usize,
    lm: usize,
    intra: bool,
    max_decay: f32,
    lfe: bool,
) -> i32 {
    let mut badness = 0i32;
    let mut prev = [0.0f32; 2];
    let (coef, beta) = if intra {
        (0.0, BETA_INTRA)
    } else {
        (PRED_COEF[lm], BETA_COEF[lm])
    };

    if tell + 3 <= budget {
        enc.encode_bit_logp(intra, 3);
    }

    for i in start..end {
        for c in 0..channels {
            let idx = i + c * mode.nb_ebands;
            let x = e_bands[idx];
            let old_e = old_e_bands[idx].max(-9.0);
            let f = x - coef * old_e - prev[c];
            let mut qi = (0.5 + f).floor() as i32;
            let qi0 = qi;
            let decay_bound = old_e_bands[idx].max(-28.0) - max_decay;
            if qi < 0 && x < decay_bound {
                qi += (decay_bound - x) as i32;
                if qi > 0 {
                    qi = 0;
                }
            }

            tell = enc.tell();
            let bits_left = budget - tell - 3 * channels as i32 * (end as i32 - i as i32);
            if i != start && bits_left < 30 {
                if bits_left < 24 {
                    qi = qi.min(1);
                }
                if bits_left < 16 {
                    qi = qi.max(-1);
                }
            }
            if lfe && i >= 2 {
                qi = qi.min(0);
            }

            if budget - tell >= 15 {
                let pi = 2 * i.min(20);
                qi = encode_laplace(
                    enc,
                    qi,
                    (prob_model[pi] as u32) << 7,
                    (prob_model[pi + 1] as i32) << 6,
                );
            } else if budget - tell >= 2 {
                qi = qi.clamp(-1, 1);
                let s = ((2 * qi) ^ -i32::from(qi < 0)) as usize;
                enc.encode_icdf(s, &SMALL_ENERGY_ICDF, 2);
            } else if budget - tell >= 1 {
                qi = qi.min(0);
                enc.encode_bit_logp(-qi != 0, 1);
            } else {
                qi = -1;
            }

            error[idx] = f - qi as f32;
            badness += (qi0 - qi).abs();
            let q = qi as f32;
            let tmp = coef * old_e + prev[c] + q;
            old_e_bands[idx] = tmp;
            prev[c] = prev[c] + q - beta * q;
        }
    }

    if lfe {
        0
    } else {
        badness
    }
}

#[allow(clippy::too_many_arguments)]
pub fn quant_coarse_energy(
    mode: &CeltMode,
    start: usize,
    end: usize,
    eff_end: usize,
    e_bands: &[f32],
    old_e_bands: &mut [f32],
    budget: u32,
    error: &mut [f32],
    enc: &mut RangeEncoder,
    channels: usize,
    lm: usize,
    nb_available_bytes: i32,
    force_intra: bool,
    delayed_intra: &mut f32,
    mut two_pass: bool,
    loss_rate: i32,
    lfe: bool,
) {
    let len = mode.nb_ebands;
    assert!(end <= len);
    assert!(eff_end <= len);
    assert!(e_bands.len() >= channels * len);
    assert!(old_e_bands.len() >= channels * len);
    assert!(error.len() >= channels * len);

    let budget = budget as i32;
    let mut intra = force_intra
        || (!two_pass
            && *delayed_intra > (2 * channels * (end - start)) as f32
            && nb_available_bytes > ((end - start) * channels) as i32);
    let intra_bias =
        ((budget as f32 * *delayed_intra * loss_rate as f32) / (channels as f32 * 512.0)) as i32;
    let new_distortion = loss_distortion(e_bands, old_e_bands, start, eff_end, len, channels);

    let tell = enc.tell();
    if tell + 3 > budget {
        two_pass = false;
        intra = false;
    }

    let mut max_decay = 16.0f32;
    if end - start > 10 {
        max_decay = max_decay.min(0.125 * nb_available_bytes as f32);
    }
    if lfe {
        max_decay = 3.0;
    }

    let enc_start_state = enc.clone();
    let mut old_e_bands_intra = old_e_bands.to_vec();
    let mut error_intra = error.to_vec();
    let mut badness1 = 0;
    let mut enc_intra_state = None;
    let mut tell_intra = 0i32;

    if two_pass || intra {
        badness1 = quant_coarse_energy_impl(
            mode,
            start,
            end,
            e_bands,
            &mut old_e_bands_intra,
            budget,
            tell,
            &E_PROB_MODEL[lm][1],
            &mut error_intra,
            enc,
            channels,
            lm,
            true,
            max_decay,
            lfe,
        );
        tell_intra = enc.tell_frac() as i32;
        enc_intra_state = Some(enc.clone());
    }

    if !intra {
        *enc = enc_start_state;
        let badness2 = quant_coarse_energy_impl(
            mode,
            start,
            end,
            e_bands,
            old_e_bands,
            budget,
            tell,
            &E_PROB_MODEL[lm][0],
            error,
            enc,
            channels,
            lm,
            false,
            max_decay,
            lfe,
        );

        if two_pass
            && (badness1 < badness2
                || (badness1 == badness2 && enc.tell_frac() as i32 + intra_bias > tell_intra))
        {
            *enc = enc_intra_state.expect("two-pass intra state exists");
            old_e_bands.copy_from_slice(&old_e_bands_intra[..old_e_bands.len()]);
            error.copy_from_slice(&error_intra[..error.len()]);
            intra = true;
        }
    } else {
        old_e_bands.copy_from_slice(&old_e_bands_intra[..old_e_bands.len()]);
        error.copy_from_slice(&error_intra[..error.len()]);
    }

    if intra {
        *delayed_intra = new_distortion;
    } else {
        *delayed_intra = PRED_COEF[lm] * PRED_COEF[lm] * *delayed_intra + new_distortion;
    }
}

pub fn quant_fine_energy(
    mode: &CeltMode,
    start: usize,
    end: usize,
    old_e_bands: &mut [f32],
    error: &mut [f32],
    fine_quant: &[i32],
    enc: &mut RangeEncoder,
    channels: usize,
) {
    for i in start..end {
        if fine_quant[i] <= 0 {
            continue;
        }
        let frac = 1 << fine_quant[i];
        for c in 0..channels {
            let idx = i + c * mode.nb_ebands;
            let mut q2 = ((error[idx] + 0.5) * frac as f32).floor() as i32;
            q2 = q2.clamp(0, frac - 1);
            enc.encode_bits(q2 as u32, fine_quant[i] as u32);
            let offset = (q2 as f32 + 0.5) * (1 << (14 - fine_quant[i])) as f32 / 16384.0 - 0.5;
            old_e_bands[idx] += offset;
            error[idx] -= offset;
        }
    }
}

pub fn quant_energy_finalise(
    mode: &CeltMode,
    start: usize,
    end: usize,
    old_e_bands: &mut [f32],
    error: &mut [f32],
    fine_quant: &[i32],
    fine_priority: &[i32],
    mut bits_left: i32,
    enc: &mut RangeEncoder,
    channels: usize,
) {
    for prio in 0..2 {
        for i in start..end {
            if bits_left < channels as i32 {
                break;
            }
            if fine_quant[i] >= MAX_FINE_BITS || fine_priority[i] != prio {
                continue;
            }
            for c in 0..channels {
                let idx = i + c * mode.nb_ebands;
                let q2 = i32::from(error[idx] >= 0.0);
                enc.encode_bits(q2 as u32, 1);
                let offset = (q2 as f32 - 0.5) * (1 << (14 - fine_quant[i] - 1)) as f32 / 16384.0;
                old_e_bands[idx] += offset;
                error[idx] -= offset;
                bits_left -= 1;
            }
        }
    }
}

pub fn unquant_coarse_energy(
    mode: &CeltMode,
    start: usize,
    end: usize,
    old_e_bands: &mut [f32],
    intra: bool,
    dec: &mut RangeDecoder,
    channels: usize,
    lm: usize,
) {
    let prob_model = &E_PROB_MODEL[lm][usize::from(intra)];
    let mut prev = [0.0f32; 2];
    let (coef, beta) = if intra {
        (0.0, BETA_INTRA)
    } else {
        (PRED_COEF[lm], BETA_COEF[lm])
    };
    let budget = dec.storage_bytes() as i32 * 8;

    for i in start..end {
        for c in 0..channels {
            let tell = dec.tell();
            let qi = if budget - tell >= 15 {
                let pi = 2 * i.min(20);
                decode_laplace(
                    dec,
                    (prob_model[pi] as u32) << 7,
                    (prob_model[pi + 1] as i32) << 6,
                )
            } else if budget - tell >= 2 {
                let qi = dec.decode_icdf(&SMALL_ENERGY_ICDF, 2) as i32;
                (qi >> 1) ^ -(qi & 1)
            } else if budget - tell >= 1 {
                -i32::from(dec.decode_bit_logp(1))
            } else {
                -1
            };
            let idx = i + c * mode.nb_ebands;
            old_e_bands[idx] = old_e_bands[idx].max(-9.0);
            let tmp = coef * old_e_bands[idx] + prev[c] + qi as f32;
            old_e_bands[idx] = tmp;
            prev[c] = prev[c] + qi as f32 - beta * qi as f32;
        }
    }
}

pub fn unquant_fine_energy(
    mode: &CeltMode,
    start: usize,
    end: usize,
    old_e_bands: &mut [f32],
    fine_quant: &[i32],
    dec: &mut RangeDecoder,
    channels: usize,
) {
    for i in start..end {
        if fine_quant[i] <= 0 {
            continue;
        }
        for c in 0..channels {
            let q2 = dec.decode_bits(fine_quant[i] as u32) as i32;
            let offset = (q2 as f32 + 0.5) * (1 << (14 - fine_quant[i])) as f32 / 16384.0 - 0.5;
            old_e_bands[i + c * mode.nb_ebands] += offset;
        }
    }
}

pub fn unquant_energy_finalise(
    mode: &CeltMode,
    start: usize,
    end: usize,
    old_e_bands: &mut [f32],
    fine_quant: &[i32],
    fine_priority: &[i32],
    mut bits_left: i32,
    dec: &mut RangeDecoder,
    channels: usize,
) {
    for prio in 0..2 {
        for i in start..end {
            if bits_left < channels as i32 {
                break;
            }
            if fine_quant[i] >= MAX_FINE_BITS || fine_priority[i] != prio {
                continue;
            }
            for c in 0..channels {
                let q2 = dec.decode_bits(1) as i32;
                let offset = (q2 as f32 - 0.5) * (1 << (14 - fine_quant[i] - 1)) as f32 / 16384.0;
                old_e_bands[i + c * mode.nb_ebands] += offset;
                bits_left -= 1;
            }
        }
    }
}

pub fn amp2_log2(
    mode: &CeltMode,
    eff_end: usize,
    end: usize,
    band_e: &[f32],
    band_log_e: &mut [f32],
    channels: usize,
) {
    assert!(end <= mode.nb_ebands);
    assert!(band_e.len() >= channels * mode.nb_ebands);
    assert!(band_log_e.len() >= channels * mode.nb_ebands);

    for c in 0..channels {
        for i in 0..eff_end {
            let idx = i + c * mode.nb_ebands;
            band_log_e[idx] = celt_log2_db(band_e[idx]) - E_MEANS[i];
        }
        for i in eff_end..end {
            band_log_e[i + c * mode.nb_ebands] = -14.0;
        }
    }
}
