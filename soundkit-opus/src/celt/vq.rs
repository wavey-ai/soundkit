//! CELT vector-quantization helpers, ported from official `celt/vq.c`.

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
use wide::{f32x4, CmpGt};

use crate::celt::cwrs::{decode_pulses, decode_pulses_with_cache, encode_pulses, CwrsDecodeCache};
use crate::celt::entropy::{RangeDecoder, RangeEncoder};
use crate::celt::mathops::{celt_cos_norm, celt_div, celt_rsqrt_norm};

const MAX_STACK_DECODE_N: usize = 256;

pub const SPREAD_NONE: i32 = 0;
pub const SPREAD_LIGHT: i32 = 1;
pub const SPREAD_NORMAL: i32 = 2;
pub const SPREAD_AGGRESSIVE: i32 = 3;

fn exp_rotation1(x: &mut [f32], len: usize, stride: usize, c: f32, s: f32) {
    if len <= stride {
        return;
    }
    let ms = -s;

    // Unit-stride rotations form one dependent chain. Carry the shared value
    // between adjacent pairs in a register; the generic loop below remains
    // easier for LLVM to vectorise when pairs do not overlap.
    if stride == 1 {
        let mut x1 = x[0];
        let mut i = 0usize;
        while i + 1 < len {
            let x2 = x[i + 1];
            x[i] = c * x1 + ms * x2;
            x1 = c * x2 + s * x1;
            i += 1;
        }
        x[i] = x1;

        if len > 2 {
            i = len - 3;
            let mut x2 = x[i + 1];
            loop {
                x1 = x[i];
                x[i + 1] = c * x2 + s * x1;
                x2 = c * x1 + ms * x2;
                if i == 0 {
                    x[i] = x2;
                    break;
                }
                i -= 1;
            }
        }
        return;
    }

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
    let mut y = Vec::new();
    let mut signx = Vec::new();
    op_pvq_search_with_scratch(x, iy, k, n, &mut y, &mut signx)
}

pub fn op_pvq_search_with_scratch(
    x: &mut [f32],
    iy: &mut [i32],
    k: usize,
    n: usize,
    y_scratch: &mut Vec<f32>,
    signx_scratch: &mut Vec<u8>,
) -> f32 {
    assert!(x.len() >= n);
    assert!(iy.len() >= n);

    if y_scratch.len() < n {
        y_scratch.resize(n, 0.0);
    }
    if signx_scratch.len() < n {
        signx_scratch.resize(n, 0);
    }
    let y = &mut y_scratch[..n];
    let signx = &mut signx_scratch[..n];

    for j in 0..n {
        signx[j] = u8::from(x[j] < 0.0);
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
            let projected = rcp * x[j];
            debug_assert!(projected.is_finite() && projected >= 0.0);
            // `x` is finite and nonnegative after the guard above, and `rcp`
            // is positive. Truncation toward zero therefore equals `floor`.
            iy[j] = projected as i32;
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
        yy += 1.0;

        #[cfg(target_arch = "x86_64")]
        let best_id = crate::kernels::pvq_best_candidate_sse2(x, y, xy, yy, n);

        #[cfg(not(target_arch = "x86_64"))]
        let best_id = {
            let mut best_id = 0usize;
            let mut rxy = xy + x[0];
            let mut ryy = yy + y[0];
            rxy *= rxy;
            let mut best_den = ryy;
            let mut best_num = rxy;

            // Vector scan over candidates in groups of four. Lane arithmetic
            // matches the scalar expressions below. Consume hit lanes in
            // ascending order to preserve the strict scalar tie-break.
            #[cfg(not(target_arch = "aarch64"))]
            let j = {
                let mut j = 1usize;
                const LANES: usize = 4;
                if n >= 2 * LANES {
                    let xy_v = f32x4::splat(xy);
                    let yy_v = f32x4::splat(yy);
                    while j + LANES <= n {
                        let xv = f32x4::new([x[j], x[j + 1], x[j + 2], x[j + 3]]);
                        let yv = f32x4::new([y[j], y[j + 1], y[j + 2], y[j + 3]]);
                        let num_v = xy_v + xv;
                        let num_v = num_v * num_v;
                        let den_v = yy_v + yv;
                        let hit =
                            (f32x4::splat(best_den) * num_v).cmp_gt(den_v * f32x4::splat(best_num));
                        let mut mask = hit.move_mask();
                        if mask != 0 {
                            let num_arr = num_v.to_array();
                            let den_arr = den_v.to_array();
                            while mask != 0 {
                                let lane = mask.trailing_zeros() as usize;
                                let cand_j = j + lane;
                                if best_den * num_arr[lane] > den_arr[lane] * best_num {
                                    best_den = den_arr[lane];
                                    best_num = num_arr[lane];
                                    best_id = cand_j;
                                }
                                mask &= mask - 1;
                            }
                        }
                        j += LANES;
                    }
                }
                j
            };
            #[cfg(target_arch = "aarch64")]
            let j = 1usize;
            for j in j..n {
                rxy = xy + x[j];
                ryy = yy + y[j];
                rxy *= rxy;
                if best_den * rxy > ryy * best_num {
                    best_den = ryy;
                    best_num = rxy;
                    best_id = j;
                }
            }
            best_id
        };

        xy += x[best_id];
        yy += y[best_id];
        y[best_id] += 2.0;
        iy[best_id] += 1;
    }

    for j in 0..n {
        if signx[j] != 0 {
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
    let mut y = Vec::new();
    let mut signx = Vec::new();
    alg_quant_with_scratch(
        x, n, k, spread, b, enc, gain, resynth, &mut iy, &mut y, &mut signx,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn alg_quant_with_scratch(
    x: &mut [f32],
    n: usize,
    k: usize,
    spread: i32,
    b: usize,
    enc: &mut RangeEncoder,
    gain: f32,
    resynth: bool,
    iy_scratch: &mut Vec<i32>,
    y_scratch: &mut Vec<f32>,
    signx_scratch: &mut Vec<u8>,
) -> u32 {
    assert!(k > 0);
    assert!(n > 1);
    assert!(x.len() >= n);

    if iy_scratch.len() < n + 3 {
        iy_scratch.resize(n + 3, 0);
    }
    let iy = &mut iy_scratch[..n + 3];
    exp_rotation(x, n, 1, b, k, spread);
    let yy = op_pvq_search_with_scratch(x, iy, k, n, y_scratch, signx_scratch);
    encode_pulses(iy, n, k, enc);

    if resynth {
        normalise_residual(iy, x, n, yy, gain);
        exp_rotation(x, n, -1, b, k, spread);
    }

    extract_collapse_mask(iy, n, b)
}

pub fn alg_unquant(
    x: &mut [f32],
    n: usize,
    k: usize,
    spread: i32,
    b: usize,
    dec: &mut RangeDecoder<'_>,
    gain: f32,
) -> u32 {
    assert!(k > 0);
    assert!(n > 1);
    assert!(x.len() >= n);

    let mut iy_stack = [0i32; MAX_STACK_DECODE_N];
    let mut iy_heap = Vec::new();
    let iy = if n <= MAX_STACK_DECODE_N {
        &mut iy_stack[..n]
    } else {
        iy_heap.resize(n, 0);
        &mut iy_heap[..]
    };
    let ryy = decode_pulses(iy, n, k, dec) as f32;
    normalise_residual(iy, x, n, ryy, gain);
    exp_rotation(x, n, -1, b, k, spread);
    extract_collapse_mask(iy, n, b)
}

#[allow(clippy::too_many_arguments)]
pub fn alg_unquant_with_scratch(
    x: &mut [f32],
    n: usize,
    k: usize,
    spread: i32,
    b: usize,
    dec: &mut RangeDecoder<'_>,
    gain: f32,
    iy_scratch: &mut Vec<i32>,
    u_scratch: &mut Vec<u32>,
    row_cache: &mut CwrsDecodeCache,
) -> u32 {
    assert!(k > 0);
    assert!(n > 1);
    assert!(x.len() >= n);

    if iy_scratch.len() < n {
        iy_scratch.resize(n, 0);
    }
    let iy = &mut iy_scratch[..n];
    let ryy = decode_pulses_with_cache(iy, n, k, dec, u_scratch, row_cache) as f32;
    normalise_residual(iy, x, n, ryy, gain);
    exp_rotation(x, n, -1, b, k, spread);
    extract_collapse_mask(iy, n, b)
}

fn cubic_synthesis(
    x: &mut [f32],
    iy: &[i32],
    n: usize,
    k: i32,
    face: usize,
    negative_face: bool,
    gain: f32,
) {
    let kf = k as f32;
    for i in 0..n {
        x[i] = (1 + 2 * iy[i] - k) as f32;
    }
    x[face] = if negative_face { -kf } else { kf };

    let energy = x[..n].iter().map(|value| value * value).sum::<f32>();
    let scale = gain / energy.sqrt();
    for value in &mut x[..n] {
        *value *= scale;
    }
}

/// Quantizes a normalized band with the direct cubic codebook from Opus QEXT.
///
/// This is an experimental primitive. The interoperable CELT packet path still
/// uses PVQ. Direct cubic coding selects the cube face with the largest input
/// magnitude, codes its sign, scalar-quantizes the remaining coordinates, and
/// renormalizes the result. It therefore avoids the iterative pulse search and
/// CWRS index coding used by PVQ.
pub fn cubic_quant(
    x: &mut [f32],
    n: usize,
    resolution: u32,
    blocks: usize,
    enc: &mut RangeEncoder,
    gain: f32,
    resynth: bool,
) -> u32 {
    let mut iy = Vec::new();
    cubic_quant_with_scratch(x, n, resolution, blocks, enc, gain, resynth, &mut iy)
}

#[allow(clippy::too_many_arguments)]
pub fn cubic_quant_with_scratch(
    x: &mut [f32],
    n: usize,
    resolution: u32,
    blocks: usize,
    enc: &mut RangeEncoder,
    gain: f32,
    resynth: bool,
    iy_scratch: &mut Vec<i32>,
) -> u32 {
    assert!(n > 1 && x.len() >= n);
    assert!(resolution <= 14);
    assert!((1..u32::BITS as usize).contains(&blocks));
    debug_assert!(x[..n].iter().all(|value| value.is_finite()));

    let mut k = 1i32 << resolution;
    if blocks != 1 {
        k = (k - 1).max(1);
    }
    if k == 1 {
        if resynth {
            x[..n].fill(0.0);
        }
        return 0;
    }

    if iy_scratch.len() < n {
        iy_scratch.resize(n, 0);
    }
    let iy = &mut iy_scratch[..n];

    let mut face = 0usize;
    let mut face_value = -1.0f32;
    for (i, value) in x[..n].iter().enumerate() {
        let magnitude = value.abs();
        if magnitude > face_value {
            face_value = magnitude;
            face = i;
        }
    }
    let negative_face = x[face] < 0.0;
    enc.encode_uint(face as u32, n as u32);
    enc.encode_bits(u32::from(negative_face), 1);

    let scale = 0.5 * k as f32 / (face_value + 1e-15);
    for i in 0..n {
        let coordinate = ((x[i] + face_value) * scale) as i32;
        iy[i] = coordinate.clamp(0, k - 1);
        if i != face {
            enc.encode_bits(iy[i] as u32, resolution);
        }
    }

    if resynth {
        cubic_synthesis(x, iy, n, k, face, negative_face, gain);
    }
    (1u32 << blocks) - 1
}

pub fn cubic_unquant(
    x: &mut [f32],
    n: usize,
    resolution: u32,
    blocks: usize,
    dec: &mut RangeDecoder<'_>,
    gain: f32,
) -> u32 {
    let mut iy = Vec::new();
    cubic_unquant_with_scratch(x, n, resolution, blocks, dec, gain, &mut iy)
}

pub fn cubic_unquant_with_scratch(
    x: &mut [f32],
    n: usize,
    resolution: u32,
    blocks: usize,
    dec: &mut RangeDecoder<'_>,
    gain: f32,
    iy_scratch: &mut Vec<i32>,
) -> u32 {
    assert!(n > 1 && x.len() >= n);
    assert!(resolution <= 14);
    assert!((1..u32::BITS as usize).contains(&blocks));

    let mut k = 1i32 << resolution;
    if blocks != 1 {
        k = (k - 1).max(1);
    }
    if k == 1 {
        x[..n].fill(0.0);
        return 0;
    }

    if iy_scratch.len() < n {
        iy_scratch.resize(n, 0);
    }
    let iy = &mut iy_scratch[..n];
    let face = dec.decode_uint(n as u32) as usize;
    let negative_face = dec.decode_bits(1) != 0;
    for (i, coordinate) in iy.iter_mut().enumerate() {
        *coordinate = if i == face {
            0
        } else {
            dec.decode_bits(resolution) as i32
        };
    }
    cubic_synthesis(x, iy, n, k, face, negative_face, gain);
    (1u32 << blocks) - 1
}

pub fn renormalise_vector(x: &mut [f32], n: usize, gain: f32) {
    assert!(x.len() >= n);
    let energy = 1e-15 + x.iter().take(n).map(|v| v * v).sum::<f32>();
    let g = celt_rsqrt_norm(energy) * gain;
    for value in x.iter_mut().take(n) {
        *value *= g;
    }
}

#[cfg(test)]
mod rotation_tests {
    use super::{exp_rotation1, op_pvq_search_with_scratch};

    #[cfg(not(target_arch = "x86_64"))]
    fn scalar_pvq_search(x: &mut [f32], iy: &mut [i32], k: usize, n: usize) -> f32 {
        let mut y = vec![0.0f32; n];
        let mut signx = vec![0u8; n];
        for j in 0..n {
            signx[j] = u8::from(x[j] < 0.0);
            x[j] = x[j].abs();
            iy[j] = 0;
        }

        let mut xy = 0.0f32;
        let mut yy = 0.0f32;
        let mut pulses_left = k as i32;
        if k > (n >> 1) {
            let mut sum = x[..n].iter().sum::<f32>();
            if !(sum > 1e-15 && sum < 64.0) {
                x[0] = 1.0;
                x[1..n].fill(0.0);
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
            if signx[j] != 0 {
                iy[j] = -iy[j];
            }
        }
        yy
    }

    #[cfg(not(target_arch = "x86_64"))]
    #[test]
    fn pvq_candidate_scan_matches_scalar_decisions() {
        let mut seed = 0x51A7_9EEDu32;
        for n in 2..=64 {
            for k in 1..=(2 * n + 3) {
                let input = (0..n)
                    .map(|_| {
                        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        (seed as i32 as f32) * (0.9375 / i32::MAX as f32)
                    })
                    .collect::<Vec<_>>();
                let mut expected_x = input.clone();
                let mut actual_x = input;
                let mut expected_iy = vec![0i32; n];
                let mut actual_iy = vec![0i32; n];
                let expected = scalar_pvq_search(&mut expected_x, &mut expected_iy, k, n);
                let actual = op_pvq_search_with_scratch(
                    &mut actual_x,
                    &mut actual_iy,
                    k,
                    n,
                    &mut Vec::new(),
                    &mut Vec::new(),
                );
                assert_eq!(actual.to_bits(), expected.to_bits(), "n={n}, k={k}");
                assert_eq!(actual_x, expected_x, "n={n}, k={k}");
                assert_eq!(actual_iy, expected_iy, "n={n}, k={k}");
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn pvq_sse2_search_preserves_pulse_and_energy_invariants() {
        let mut seed = 0x51A7_9EEDu32;
        for n in 2..=64 {
            for k in 1..=(2 * n + 3) {
                let mut input = (0..n)
                    .map(|_| {
                        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        (seed as i32 as f32) * (0.9375 / i32::MAX as f32)
                    })
                    .collect::<Vec<_>>();
                let absolute = input.iter().map(|value| value.abs()).collect::<Vec<_>>();
                let mut iy = vec![0i32; n];
                let energy = op_pvq_search_with_scratch(
                    &mut input,
                    &mut iy,
                    k,
                    n,
                    &mut Vec::new(),
                    &mut Vec::new(),
                );
                assert_eq!(input, absolute, "n={n}, k={k}");
                assert_eq!(
                    iy.iter()
                        .map(|value| value.unsigned_abs() as usize)
                        .sum::<usize>(),
                    k
                );
                let expected_energy = iy.iter().map(|&value| value * value).sum::<i32>() as f32;
                assert_eq!(energy, expected_energy, "n={n}, k={k}");
            }
        }
    }

    #[test]
    fn unit_stride_register_chain_is_bit_exact() {
        for len in 2..=128 {
            let input: Vec<f32> = (0..len)
                .map(|i| {
                    let phase = (i * 37 + len * 11) as f32;
                    phase.sin() * 0.75 + phase.cos() * 0.125
                })
                .collect();
            let mut expected = input.clone();
            let mut actual = input;
            let c = 0.8125;
            let s = -0.3125;
            let ms = -s;

            for i in 0..len - 1 {
                let x1 = expected[i];
                let x2 = expected[i + 1];
                expected[i + 1] = c * x2 + s * x1;
                expected[i] = c * x1 + ms * x2;
            }
            if len > 2 {
                let mut i = len - 3;
                loop {
                    let x1 = expected[i];
                    let x2 = expected[i + 1];
                    expected[i + 1] = c * x2 + s * x1;
                    expected[i] = c * x1 + ms * x2;
                    if i == 0 {
                        break;
                    }
                    i -= 1;
                }
            }

            exp_rotation1(&mut actual, len, 1, c, s);
            for (index, (expected, actual)) in expected.iter().zip(&actual).enumerate() {
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "len={len}, index={index}"
                );
            }
        }
    }
}
