//! CELT pulse/fine-energy allocation, ported from official `celt/rate.c`.

use crate::celt::entropy::{RangeDecoder, RangeEncoder};
use crate::celt::modes::CeltMode;

const BITRES: i32 = 3;
const ALLOC_STEPS: usize = 6;
const MAX_FINE_BITS: i32 = 8;
const FINE_OFFSET: i32 = 21;
const LOG2_FRAC_TABLE: [i32; 24] = [
    0, 8, 13, 16, 19, 21, 23, 24, 26, 27, 28, 29, 30, 31, 32, 32, 33, 34, 34, 35, 36, 36, 37, 37,
];

pub enum AllocationCoder<'a> {
    Encode(&'a mut RangeEncoder),
    Decode(&'a mut RangeDecoder),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Allocation {
    pub coded_bands: usize,
    pub balance: i32,
    pub pulses: Vec<i32>,
    pub ebits: Vec<i32>,
    pub fine_priority: Vec<i32>,
    pub intensity: usize,
    pub dual_stereo: bool,
}

#[allow(clippy::too_many_arguments)]
fn interp_bits2pulses(
    mode: &CeltMode,
    start: usize,
    end: usize,
    skip_start: usize,
    bits1: &[i32],
    bits2: &[i32],
    thresh: &[i32],
    cap: &[i32],
    mut total: i32,
    skip_rsv: i32,
    intensity: &mut usize,
    mut intensity_rsv: i32,
    dual_stereo: &mut bool,
    mut dual_stereo_rsv: i32,
    pulses: &mut [i32],
    ebits: &mut [i32],
    fine_priority: &mut [i32],
    c: usize,
    lm: usize,
    mut coder: Option<&mut AllocationCoder<'_>>,
    prev: usize,
    signal_bandwidth: usize,
) -> (usize, i32) {
    let alloc_floor = (c as i32) << BITRES;
    let stereo = c > 1;
    let log_m = (lm as i32) << BITRES;

    let mut lo = 0i32;
    let mut hi = 1 << ALLOC_STEPS;
    for _ in 0..ALLOC_STEPS {
        let mid = (lo + hi) >> 1;
        let mut psum = 0i32;
        let mut done = false;
        for j in (start..end).rev() {
            let tmp = bits1[j] + ((mid * bits2[j]) >> ALLOC_STEPS);
            if tmp >= thresh[j] || done {
                done = true;
                psum += tmp.min(cap[j]);
            } else if tmp >= alloc_floor {
                psum += alloc_floor;
            }
        }
        if psum > total {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    let mut psum = 0i32;
    let mut done = false;
    for j in (start..end).rev() {
        let mut tmp = bits1[j] + ((lo * bits2[j]) >> ALLOC_STEPS);
        if tmp < thresh[j] && !done {
            tmp = if tmp >= alloc_floor { alloc_floor } else { 0 };
        } else {
            done = true;
        }
        tmp = tmp.min(cap[j]);
        pulses[j] = tmp;
        psum += tmp;
    }

    let mut coded_bands = end;
    loop {
        let j = coded_bands - 1;
        if j <= skip_start {
            total += skip_rsv;
            break;
        }

        let left0 = total - psum;
        let denom = mode.ebands[coded_bands] as i32 - mode.ebands[start] as i32;
        let percoeff = left0 / denom;
        let left = left0 - denom * percoeff;
        let rem = (left - (mode.ebands[j] as i32 - mode.ebands[start] as i32)).max(0);
        let band_width = mode.ebands[coded_bands] as i32 - mode.ebands[j] as i32;
        let mut band_bits = pulses[j] + percoeff * band_width + rem;

        if band_bits >= thresh[j].max(alloc_floor + (1 << BITRES)) {
            match coder.as_deref_mut() {
                Some(AllocationCoder::Encode(enc)) => {
                    let depth_threshold = if coded_bands > 17 {
                        if j < prev {
                            7
                        } else {
                            9
                        }
                    } else {
                        0
                    };
                    if coded_bands <= start + 2
                        || (band_bits > ((depth_threshold * band_width) << lm << BITRES) >> 4
                            && j <= signal_bandwidth)
                    {
                        enc.encode_bit_logp(true, 1);
                        break;
                    }
                    enc.encode_bit_logp(false, 1);
                }
                Some(AllocationCoder::Decode(dec)) => {
                    if dec.decode_bit_logp(1) {
                        break;
                    }
                }
                None => {}
            }
            psum += 1 << BITRES;
            band_bits -= 1 << BITRES;
        }

        psum -= pulses[j] + intensity_rsv;
        if intensity_rsv > 0 {
            intensity_rsv = LOG2_FRAC_TABLE[j - start];
        }
        psum += intensity_rsv;
        if band_bits >= alloc_floor {
            psum += alloc_floor;
            pulses[j] = alloc_floor;
        } else {
            pulses[j] = 0;
        }

        coded_bands -= 1;
    }

    if intensity_rsv > 0 {
        match coder.as_deref_mut() {
            Some(AllocationCoder::Encode(enc)) => {
                *intensity = (*intensity).min(coded_bands);
                enc.encode_uint(
                    (*intensity - start) as u32,
                    (coded_bands + 1 - start) as u32,
                );
            }
            Some(AllocationCoder::Decode(dec)) => {
                *intensity = start + dec.decode_uint((coded_bands + 1 - start) as u32) as usize;
            }
            None => {
                *intensity = (*intensity).min(coded_bands);
            }
        }
    } else {
        *intensity = 0;
    }

    if *intensity <= start {
        total += dual_stereo_rsv;
        dual_stereo_rsv = 0;
    }
    if dual_stereo_rsv > 0 {
        match coder.as_deref_mut() {
            Some(AllocationCoder::Encode(enc)) => enc.encode_bit_logp(*dual_stereo, 1),
            Some(AllocationCoder::Decode(dec)) => *dual_stereo = dec.decode_bit_logp(1),
            None => {}
        }
    } else {
        *dual_stereo = false;
    }

    let left0 = total - psum;
    let denom = mode.ebands[coded_bands] as i32 - mode.ebands[start] as i32;
    let percoeff = left0 / denom;
    let mut left = left0 - denom * percoeff;
    for j in start..coded_bands {
        pulses[j] += percoeff * (mode.ebands[j + 1] as i32 - mode.ebands[j] as i32);
    }
    for j in start..coded_bands {
        let tmp = left.min(mode.ebands[j + 1] as i32 - mode.ebands[j] as i32);
        pulses[j] += tmp;
        left -= tmp;
    }

    let mut balance = 0i32;
    for j in start..coded_bands {
        let n0 = mode.ebands[j + 1] as i32 - mode.ebands[j] as i32;
        let n = n0 << lm;
        let bit = pulses[j] + balance;

        let excess;
        if n > 1 {
            excess = (bit - cap[j]).max(0);
            pulses[j] = bit - excess;
            let den = c as i32 * n + i32::from(c == 2 && n > 2 && !*dual_stereo && j < *intensity);
            let nclogn = den * (mode.log_n[j] as i32 + log_m);
            let mut offset = (nclogn >> 1) - den * FINE_OFFSET;
            if n == 2 {
                offset += den << BITRES >> 2;
            }
            if pulses[j] + offset < den * 2 << BITRES {
                offset += nclogn >> 2;
            } else if pulses[j] + offset < den * 3 << BITRES {
                offset += nclogn >> 3;
            }

            ebits[j] = (pulses[j] + offset + (den << (BITRES - 1))).max(0);
            ebits[j] = (ebits[j] / den) >> BITRES;
            if c as i32 * ebits[j] > pulses[j] >> BITRES {
                ebits[j] = pulses[j] >> i32::from(stereo) >> BITRES;
            }
            ebits[j] = ebits[j].min(MAX_FINE_BITS);
            fine_priority[j] = i32::from(ebits[j] * (den << BITRES) >= pulses[j] + offset);
            pulses[j] -= c as i32 * ebits[j] << BITRES;
        } else {
            excess = (bit - ((c as i32) << BITRES)).max(0);
            pulses[j] = bit - excess;
            ebits[j] = 0;
            fine_priority[j] = 1;
        }

        let mut excess = excess;
        if excess > 0 {
            let extra_fine = (excess >> (i32::from(stereo) + BITRES)).min(MAX_FINE_BITS - ebits[j]);
            ebits[j] += extra_fine;
            let extra_bits = extra_fine * (c as i32) << BITRES;
            fine_priority[j] = i32::from(extra_bits >= excess - balance);
            excess -= extra_bits;
        }
        balance = excess;
    }

    for j in coded_bands..end {
        ebits[j] = pulses[j] >> i32::from(stereo) >> BITRES;
        pulses[j] = 0;
        fine_priority[j] = i32::from(ebits[j] < 1);
    }

    (coded_bands, balance)
}

#[allow(clippy::too_many_arguments)]
pub fn clt_compute_allocation(
    mode: &CeltMode,
    start: usize,
    end: usize,
    offsets: &[i32],
    cap: &[i32],
    alloc_trim: i32,
    intensity: usize,
    dual_stereo: bool,
    total: i32,
    channels: usize,
    lm: usize,
    coder: Option<&mut AllocationCoder<'_>>,
    prev: usize,
    signal_bandwidth: usize,
) -> Allocation {
    let len = mode.nb_ebands;
    assert!(end <= len);
    assert!(offsets.len() >= len);
    assert!(cap.len() >= len);

    let mut total = total.max(0);
    let mut skip_start = start;
    let skip_rsv = if total >= 1 << BITRES { 1 << BITRES } else { 0 };
    total -= skip_rsv;

    let mut intensity_rsv = 0i32;
    let mut dual_stereo_rsv = 0i32;
    if channels == 2 {
        intensity_rsv = LOG2_FRAC_TABLE[end - start];
        if intensity_rsv > total {
            intensity_rsv = 0;
        } else {
            total -= intensity_rsv;
            dual_stereo_rsv = if total >= 1 << BITRES { 1 << BITRES } else { 0 };
            total -= dual_stereo_rsv;
        }
    }

    let mut bits1 = vec![0i32; len];
    let mut bits2 = vec![0i32; len];
    let mut thresh = vec![0i32; len];
    let mut trim_offset = vec![0i32; len];
    let mut pulses = vec![0i32; len];
    let mut ebits = vec![0i32; len];
    let mut fine_priority = vec![0i32; len];

    for j in start..end {
        let width = mode.ebands[j + 1] as i32 - mode.ebands[j] as i32;
        thresh[j] = ((channels as i32) << BITRES).max((3 * width << lm << BITRES) >> 4);
        trim_offset[j] = channels as i32
            * width
            * (alloc_trim - 5 - lm as i32)
            * (end as i32 - j as i32 - 1)
            * (1 << (lm + BITRES as usize))
            >> 6;
        if width << lm == 1 {
            trim_offset[j] -= (channels as i32) << BITRES;
        }
    }

    let mut lo = 1i32;
    let mut hi = mode.nb_alloc_vectors as i32 - 1;
    while lo <= hi {
        let mut done = false;
        let mut psum = 0i32;
        let mid = (lo + hi) >> 1;
        for j in (start..end).rev() {
            let width = mode.ebands[j + 1] as i32 - mode.ebands[j] as i32;
            let mut bitsj =
                channels as i32 * width * (mode.alloc_vectors[mid as usize * len + j] as i32) << lm
                    >> 2;
            if bitsj > 0 {
                bitsj = 0.max(bitsj + trim_offset[j]);
            }
            bitsj += offsets[j];
            if bitsj >= thresh[j] || done {
                done = true;
                psum += bitsj.min(cap[j]);
            } else if bitsj >= (channels as i32) << BITRES {
                psum += (channels as i32) << BITRES;
            }
        }
        if psum > total {
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
    }
    hi = lo;
    lo -= 1;

    for j in start..end {
        let width = mode.ebands[j + 1] as i32 - mode.ebands[j] as i32;
        let mut bits1j =
            channels as i32 * width * (mode.alloc_vectors[lo as usize * len + j] as i32) << lm >> 2;
        let mut bits2j = if hi as usize >= mode.nb_alloc_vectors {
            cap[j]
        } else {
            channels as i32 * width * (mode.alloc_vectors[hi as usize * len + j] as i32) << lm >> 2
        };
        if bits1j > 0 {
            bits1j = 0.max(bits1j + trim_offset[j]);
        }
        if bits2j > 0 {
            bits2j = 0.max(bits2j + trim_offset[j]);
        }
        if lo > 0 {
            bits1j += offsets[j];
        }
        bits2j += offsets[j];
        if offsets[j] > 0 {
            skip_start = j;
        }
        bits2j = 0.max(bits2j - bits1j);
        bits1[j] = bits1j;
        bits2[j] = bits2j;
    }

    let mut intensity = intensity;
    let mut dual_stereo = dual_stereo;
    let (coded_bands, balance) = interp_bits2pulses(
        mode,
        start,
        end,
        skip_start,
        &bits1,
        &bits2,
        &thresh,
        cap,
        total,
        skip_rsv,
        &mut intensity,
        intensity_rsv,
        &mut dual_stereo,
        dual_stereo_rsv,
        &mut pulses,
        &mut ebits,
        &mut fine_priority,
        channels,
        lm,
        coder,
        prev,
        signal_bandwidth,
    );

    Allocation {
        coded_bands,
        balance,
        pulses,
        ebits,
        fine_priority,
        intensity,
        dual_stereo,
    }
}
