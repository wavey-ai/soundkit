//! CELT mode construction and pulse cache, ported from official
//! `celt/modes.c` and `celt/rate.c`.

use crate::celt::cwrs::{get_pulses, get_required_bits, log2_frac, CELT_MAX_PULSES, MAX_PSEUDO};
use crate::celt::mdct::MdctLookup;
use crate::{Error, Result};

const BITRES: i32 = 3;
const BITALLOC_SIZE: usize = 11;
const BARK_BANDS: usize = 25;
const MAX_FINE_BITS: i32 = 8;
const FINE_OFFSET: i32 = 21;
const QTHETA_OFFSET: i32 = 4;
const QTHETA_OFFSET_TWOPHASE: i32 = 16;

pub const MAX_PERIOD: usize = 1024;

const EBAND_5MS: [i16; 22] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100,
];

const BAND_ALLOCATION: [u8; BITALLOC_SIZE * 21] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 90, 80, 75, 69, 63, 56, 49, 40,
    34, 29, 20, 18, 10, 0, 0, 0, 0, 0, 0, 0, 0, 110, 100, 90, 84, 78, 71, 65, 58, 51, 45, 39, 32,
    26, 20, 12, 0, 0, 0, 0, 0, 0, 118, 110, 103, 93, 86, 80, 75, 70, 65, 59, 53, 47, 40, 31, 23,
    15, 4, 0, 0, 0, 0, 126, 119, 112, 104, 95, 89, 83, 78, 72, 66, 60, 54, 47, 39, 32, 25, 17, 12,
    1, 0, 0, 134, 127, 120, 114, 103, 97, 91, 85, 78, 72, 66, 60, 54, 47, 41, 35, 29, 23, 16, 10,
    1, 144, 137, 130, 124, 113, 107, 101, 95, 88, 82, 76, 70, 64, 57, 51, 45, 39, 33, 26, 15, 1,
    152, 145, 138, 132, 123, 117, 111, 105, 98, 92, 86, 80, 74, 67, 61, 55, 49, 43, 36, 20, 1, 162,
    155, 148, 142, 133, 127, 121, 115, 108, 102, 96, 90, 84, 77, 71, 65, 59, 53, 46, 30, 1, 172,
    165, 158, 152, 143, 137, 131, 125, 118, 112, 106, 100, 94, 87, 81, 75, 69, 63, 56, 45, 20, 200,
    200, 200, 200, 200, 200, 200, 200, 198, 193, 188, 183, 178, 173, 168, 163, 158, 153, 148, 129,
    104,
];

const BARK_FREQ: [i32; BARK_BANDS + 1] = [
    0, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000, 2320, 2700, 3150,
    3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500, 20000,
];

const WINDOW_120: [f32; 120] = [
    6.7286966e-05,
    0.00060551348,
    0.0016815970,
    0.0032947962,
    0.0054439943,
    0.0081276923,
    0.011344001,
    0.015090633,
    0.019364886,
    0.024163635,
    0.029483315,
    0.035319905,
    0.041668911,
    0.048525347,
    0.055883718,
    0.063737999,
    0.072081616,
    0.080907428,
    0.090207705,
    0.099974111,
    0.11019769,
    0.12086883,
    0.13197729,
    0.14351214,
    0.15546177,
    0.16781389,
    0.18055550,
    0.19367290,
    0.20715171,
    0.22097682,
    0.23513243,
    0.24960208,
    0.26436860,
    0.27941419,
    0.29472040,
    0.31026818,
    0.32603788,
    0.34200931,
    0.35816177,
    0.37447407,
    0.39092462,
    0.40749142,
    0.42415215,
    0.44088423,
    0.45766484,
    0.47447104,
    0.49127978,
    0.50806798,
    0.52481261,
    0.54149077,
    0.55807973,
    0.57455701,
    0.59090049,
    0.60708841,
    0.62309951,
    0.63891306,
    0.65450896,
    0.66986776,
    0.68497077,
    0.69980010,
    0.71433873,
    0.72857055,
    0.74248043,
    0.75605424,
    0.76927895,
    0.78214257,
    0.79463430,
    0.80674445,
    0.81846456,
    0.82978733,
    0.84070669,
    0.85121779,
    0.86131698,
    0.87100183,
    0.88027111,
    0.88912479,
    0.89756398,
    0.90559094,
    0.91320904,
    0.92042270,
    0.92723738,
    0.93365955,
    0.93969656,
    0.94535671,
    0.95064907,
    0.95558353,
    0.96017067,
    0.96442171,
    0.96834849,
    0.97196334,
    0.97527906,
    0.97830883,
    0.98106616,
    0.98356480,
    0.98581869,
    0.98784191,
    0.98964856,
    0.99125274,
    0.99266849,
    0.99390969,
    0.99499004,
    0.99592297,
    0.99672162,
    0.99739874,
    0.99796667,
    0.99843728,
    0.99882195,
    0.99913147,
    0.99937606,
    0.99956527,
    0.99970802,
    0.99981248,
    0.99988613,
    0.99993565,
    0.99996697,
    0.99998518,
    0.99999457,
    0.99999859,
    0.99999982,
    1.0000000,
];

#[derive(Clone, Debug)]
pub struct PulseCache {
    pub size: usize,
    pub index: Vec<i16>,
    pub bits: Vec<u8>,
    pub caps: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct CeltMode {
    pub fs: i32,
    pub overlap: usize,
    pub nb_ebands: usize,
    pub eff_ebands: usize,
    pub preemph: [f32; 4],
    pub ebands: Vec<i16>,
    pub max_lm: usize,
    pub nb_short_mdcts: usize,
    pub short_mdct_size: usize,
    pub nb_alloc_vectors: usize,
    pub alloc_vectors: Vec<u8>,
    pub log_n: Vec<i16>,
    pub window: Vec<f32>,
    pub mdct: MdctLookup,
    pub cache: PulseCache,
}

impl CeltMode {
    pub fn new(fs: i32, frame_size: usize) -> Result<Self> {
        if !(8_000..=96_000).contains(&fs) {
            return Err(Error::BadArg);
        }
        if !(40..=1024).contains(&frame_size) || frame_size % 2 != 0 {
            return Err(Error::BadArg);
        }
        if frame_size as i32 * 1000 < fs {
            return Err(Error::BadArg);
        }

        let max_lm = if frame_size as i32 * 75 >= fs && frame_size % 16 == 0 {
            3
        } else if frame_size as i32 * 150 >= fs && frame_size % 8 == 0 {
            2
        } else if frame_size as i32 * 300 >= fs && frame_size % 4 == 0 {
            1
        } else {
            0
        };

        if ((frame_size >> max_lm) as i32) * 300 > fs {
            return Err(Error::BadArg);
        }

        let preemph = if fs < 12_000 {
            [0.3500061035, -0.1799926758, 0.2719968125, 3.6765136719]
        } else if fs < 24_000 {
            [0.6000061035, -0.1799926758, 0.4424998650, 2.2598876953]
        } else if fs < 40_000 {
            [0.7799987793, -0.1000061035, 0.7499771125, 1.3333740234]
        } else {
            [0.8500061035, 0.0, 1.0, 1.0]
        };

        let nb_short_mdcts = 1usize << max_lm;
        let short_mdct_size = frame_size / nb_short_mdcts;
        let res = (fs + short_mdct_size as i32) / (2 * short_mdct_size as i32);
        let ebands = compute_ebands(fs, short_mdct_size, res);
        let nb_ebands = ebands.len() - 1;

        let mut eff_ebands = nb_ebands;
        while ebands[eff_ebands] as usize > short_mdct_size {
            eff_ebands -= 1;
        }

        let overlap = (short_mdct_size >> 2) << 2;
        let alloc_vectors = compute_allocation_table(fs, short_mdct_size, &ebands);
        let window = compute_window(overlap);
        let log_n = (0..nb_ebands)
            .map(|i| log2_frac((ebands[i + 1] - ebands[i]) as u32, BITRES) as i16)
            .collect::<Vec<_>>();
        let mdct = MdctLookup::new(2 * short_mdct_size * nb_short_mdcts, max_lm)
            .ok_or(Error::AllocFail)?;

        let mut mode = Self {
            fs,
            overlap,
            nb_ebands,
            eff_ebands,
            preemph,
            ebands,
            max_lm,
            nb_short_mdcts,
            short_mdct_size,
            nb_alloc_vectors: BITALLOC_SIZE,
            alloc_vectors,
            log_n,
            window,
            mdct,
            cache: PulseCache {
                size: 0,
                index: Vec::new(),
                bits: Vec::new(),
                caps: Vec::new(),
            },
        };
        mode.cache = compute_pulse_cache(&mode, mode.max_lm);
        Ok(mode)
    }

    pub fn standard_48k() -> Self {
        Self::new(48_000, 960).expect("48 kHz CELT mode is valid")
    }
}

pub fn compute_window(overlap: usize) -> Vec<f32> {
    if overlap == 120 {
        return WINDOW_120.to_vec();
    }
    (0..overlap)
        .map(|i| {
            let inner = (0.5 * core::f32::consts::PI * (i as f32 + 0.5) / overlap as f32).sin();
            (0.5 * core::f32::consts::PI * inner * inner).sin()
        })
        .collect()
}

fn compute_ebands(fs: i32, frame_size: usize, res: i32) -> Vec<i16> {
    if fs == 400 * frame_size as i32 {
        return EBAND_5MS.to_vec();
    }

    let mut n_bark = 1usize;
    while n_bark < BARK_BANDS {
        if BARK_FREQ[n_bark + 1] * 2 >= fs {
            break;
        }
        n_bark += 1;
    }

    let mut lin = 0usize;
    while lin < n_bark {
        if BARK_FREQ[lin + 1] - BARK_FREQ[lin] >= res {
            break;
        }
        lin += 1;
    }

    let low = ((BARK_FREQ[lin] + res / 2) / res) as usize;
    let high = n_bark - lin;
    let mut ebands = vec![0i16; low + high + 1];
    let mut offset = 0;

    for (i, band) in ebands.iter_mut().enumerate().take(low) {
        *band = i as i16;
    }
    if low > 0 {
        offset = ebands[low - 1] as i32 * res - BARK_FREQ[lin - 1];
    }
    for i in 0..high {
        let target = BARK_FREQ[lin + i];
        ebands[i + low] = (((target + offset / 2 + res) / (2 * res)) * 2) as i16;
        offset = ebands[i + low] as i32 * res - target;
    }
    for (i, band) in ebands.iter_mut().enumerate().take(low + high) {
        if *band < i as i16 {
            *band = i as i16;
        }
    }
    ebands[low + high] = (((BARK_FREQ[n_bark] + res) / (2 * res)) * 2) as i16;
    if ebands[low + high] as usize > frame_size {
        ebands[low + high] = frame_size as i16;
    }

    for i in 1..low + high - 1 {
        if ebands[i + 1] - ebands[i] < ebands[i] - ebands[i - 1] {
            ebands[i] -= (2 * ebands[i] - ebands[i - 1] - ebands[i + 1]) / 2;
        }
    }

    let mut compact = vec![ebands[0]];
    for i in 0..low + high {
        if ebands[i + 1] > *compact.last().expect("nonempty") {
            compact.push(ebands[i + 1]);
        }
    }
    compact
}

fn compute_allocation_table(fs: i32, short_mdct_size: usize, ebands: &[i16]) -> Vec<u8> {
    let nb_ebands = ebands.len() - 1;
    if fs == 400 * short_mdct_size as i32 {
        return BAND_ALLOCATION.to_vec();
    }

    let max_bands = EBAND_5MS.len() - 1;
    let mut alloc_vectors = vec![0u8; BITALLOC_SIZE * nb_ebands];
    for i in 0..BITALLOC_SIZE {
        for j in 0..nb_ebands {
            let band_hz = ebands[j] as i32 * fs / short_mdct_size as i32;
            let mut k = 0usize;
            while k < max_bands {
                if 400 * EBAND_5MS[k] as i32 > band_hz {
                    break;
                }
                k += 1;
            }
            alloc_vectors[i * nb_ebands + j] = if k > max_bands - 1 {
                BAND_ALLOCATION[i * max_bands + max_bands - 1]
            } else {
                let a1 = band_hz - 400 * EBAND_5MS[k - 1] as i32;
                let a0 = 400 * EBAND_5MS[k] as i32 - band_hz;
                ((a0 * BAND_ALLOCATION[i * max_bands + k - 1] as i32
                    + a1 * BAND_ALLOCATION[i * max_bands + k] as i32)
                    / (a0 + a1)) as u8
            };
        }
    }
    alloc_vectors
}

fn fits_in32(n: usize, k: usize) -> bool {
    const MAX_N: [usize; 15] = [
        32767, 32767, 32767, 1476, 283, 109, 60, 40, 29, 24, 20, 18, 16, 14, 13,
    ];
    const MAX_K: [usize; 15] = [
        32767, 32767, 32767, 32767, 1172, 238, 95, 53, 36, 27, 22, 18, 16, 15, 13,
    ];
    if n >= 14 {
        k < 14 && n <= MAX_N[k]
    } else {
        k <= MAX_K[n]
    }
}

pub fn compute_pulse_cache(mode: &CeltMode, lm: usize) -> PulseCache {
    let mut curr = 0usize;
    let mut entries = Vec::<(usize, usize, usize)>::new();
    let mut cindex = vec![-1i16; mode.nb_ebands * (lm + 2)];

    for i in 0..=lm + 1 {
        for j in 0..mode.nb_ebands {
            let n = ((mode.ebands[j + 1] - mode.ebands[j]) as usize) << i >> 1;
            for k in 0..=i {
                for nband in 0..mode.nb_ebands {
                    if k == i && nband >= j {
                        break;
                    }
                    if n == ((mode.ebands[nband + 1] - mode.ebands[nband]) as usize) << k >> 1 {
                        cindex[i * mode.nb_ebands + j] = cindex[k * mode.nb_ebands + nband];
                        break;
                    }
                }
                if cindex[i * mode.nb_ebands + j] != -1 {
                    break;
                }
            }
            if cindex[i * mode.nb_ebands + j] == -1 && n != 0 {
                let mut k = 0usize;
                while fits_in32(n, get_pulses(k + 1)) && k < MAX_PSEUDO {
                    k += 1;
                }
                cindex[i * mode.nb_ebands + j] = curr as i16;
                entries.push((n, k, curr));
                curr += k + 1;
            }
        }
    }

    let mut bits = vec![0u8; curr];
    for (n, k, entry_i) in entries {
        let mut tmp = vec![0i16; CELT_MAX_PULSES + 1];
        get_required_bits(&mut tmp, n, get_pulses(k), BITRES);
        bits[entry_i] = k as u8;
        for j in 1..=k {
            bits[entry_i + j] = (tmp[get_pulses(j)] - 1) as u8;
        }
    }

    let mut caps = Vec::with_capacity((lm + 1) * 2 * mode.nb_ebands);
    for i in 0..=lm {
        for c in 1usize..=2 {
            for j in 0..mode.nb_ebands {
                let mut n0 = (mode.ebands[j + 1] - mode.ebands[j]) as usize;
                let max_bits = if n0 << i == 1 {
                    (c as i32 * (1 + MAX_FINE_BITS)) << BITRES
                } else {
                    let mut lm0 = 0isize;
                    if n0 > 2 {
                        n0 >>= 1;
                        lm0 -= 1;
                    } else if n0 <= 1 {
                        lm0 = i.min(1) as isize;
                        n0 <<= lm0 as usize;
                    }

                    let pcache_offset = cindex[(lm0 + 1) as usize * mode.nb_ebands + j] as usize;
                    let pcache_len = bits[pcache_offset] as usize;
                    let mut max_bits = bits[pcache_offset + pcache_len] as i32 + 1;

                    let mut n = n0 as i32;
                    for k in 0..(i as isize - lm0) as usize {
                        max_bits <<= 1;
                        let offset =
                            ((mode.log_n[j] as i32 + (((lm0 + k as isize) as i32) << BITRES)) >> 1)
                                - QTHETA_OFFSET;
                        let num = 459 * (((2 * n - 1) * offset) + max_bits);
                        let den = ((2 * n - 1) << 9) - 459;
                        let qb = ((num + (den >> 1)) / den).min(57);
                        max_bits += qb;
                        n <<= 1;
                    }

                    if c == 2 {
                        max_bits <<= 1;
                        let offset = ((mode.log_n[j] as i32 + ((i as i32) << BITRES)) >> 1)
                            - if n == 2 {
                                QTHETA_OFFSET_TWOPHASE
                            } else {
                                QTHETA_OFFSET
                            };
                        let ndof = 2 * n - 1 - i32::from(n == 2);
                        let scale = if n == 2 { 512 } else { 487 };
                        let num = scale * (max_bits + ndof * offset);
                        let den = (ndof << 9) - scale;
                        let qb = ((num + (den >> 1)) / den).min(if n == 2 { 64 } else { 61 });
                        max_bits += qb;
                    }

                    let ndof = c as i32 * n + i32::from(c == 2 && n > 2);
                    let mut offset =
                        ((mode.log_n[j] as i32 + ((i as i32) << BITRES)) >> 1) - FINE_OFFSET;
                    if n == 2 {
                        offset += 1 << BITRES >> 2;
                    }
                    let num = max_bits + ndof * offset;
                    let den = (ndof - 1) << BITRES;
                    let qb = ((num + (den >> 1)) / den).min(MAX_FINE_BITS);
                    max_bits + ((c as i32 * qb) << BITRES)
                };

                let width = c as i32 * (((mode.ebands[j + 1] - mode.ebands[j]) as i32) << i);
                let cap = 4 * max_bits / width - 64;
                caps.push(cap as u8);
            }
        }
    }

    PulseCache {
        size: curr,
        index: cindex,
        bits,
        caps,
    }
}

pub fn bits2pulses(mode: &CeltMode, band: usize, lm: usize, bits: i32) -> usize {
    bits2pulses_signed(mode, band, lm as isize, bits)
}

pub fn bits2pulses_signed(mode: &CeltMode, band: usize, lm: isize, bits: i32) -> usize {
    let lm = (lm + 1) as usize;
    let cache = &mode.cache.bits[mode.cache.index[lm * mode.nb_ebands + band] as usize..];
    let mut lo = 0usize;
    let mut hi = cache[0] as usize;
    let bits = bits - 1;
    for _ in 0..6 {
        let mid = (lo + hi + 1) >> 1;
        if cache[mid] as i32 >= bits {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    if bits - if lo == 0 { -1 } else { cache[lo] as i32 } <= cache[hi] as i32 - bits {
        lo
    } else {
        hi
    }
}

pub fn pulses2bits(mode: &CeltMode, band: usize, lm: usize, pulses: usize) -> i32 {
    pulses2bits_signed(mode, band, lm as isize, pulses)
}

pub fn pulses2bits_signed(mode: &CeltMode, band: usize, lm: isize, pulses: usize) -> i32 {
    if pulses == 0 {
        return 0;
    }
    let lm = (lm + 1) as usize;
    let cache = &mode.cache.bits[mode.cache.index[lm * mode.nb_ebands + band] as usize..];
    cache[pulses] as i32 + 1
}
