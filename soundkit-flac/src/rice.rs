// Copyright 2022-2024 Google LLC
// Copyright 2025- flacenc-rs developers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Functions for partitioned rice coding (PRC).

use super::constant::rice::MAX_PARTITION_ORDER as MAX_RICE_PARTITION_ORDER;
use super::constant::rice::MAX_RICE_PARAMETER;
use super::constant::rice::MIN_PARTITION_SIZE as MIN_RICE_PARTITION_SIZE;

import_simd!(as simd);

/// Encodes the sign bit into its LSB (for Rice coding).
#[inline]
pub const fn encode_signbit(v: i32) -> u32 {
    (v.unsigned_abs() << 1) - (v < 0) as u32
}

#[inline]
pub fn encode_signbit_simd<const N: usize>(v: simd::Simd<i32, N>) -> simd::Simd<u32, N> {
    (v.abs().cast() << simd::Simd::splat(1u32)) - (v.cast() >> simd::Simd::splat(31u32))
}

/// Recovers a sign bit from its LSB.
#[inline]
pub const fn decode_signbit(v: u32) -> i32 {
    let is_negative = v % 2 == 1;
    if is_negative {
        -(((v >> 1) + 1) as i32)
    } else {
        (v >> 1) as i32
    }
}

/// Computes the number of the finest partitions.
#[inline]
fn finest_partition_order(size: usize, min_part_size: usize) -> usize {
    assert!(min_part_size >= 1);
    let max_splits: u32 = (size / min_part_size) as u32;
    let max_order_for_min_part = (32 - max_splits.leading_zeros() - 1) as usize;
    std::cmp::min(
        MAX_RICE_PARTITION_ORDER,
        std::cmp::min(max_order_for_min_part, size.trailing_zeros() as usize),
    )
}

/// Bit estimate overhead of one partition header in the residual bitstream.
const PRC_HEADER_BITS: u64 = 4;

/// Helper object that holds pre-allocated buffers for PRC optimization.
///
/// The estimator keeps one accumulator per partition at the finest order.
/// As in libFLAC, the accumulator holds the sum of absolute residuals, which
/// is enough to estimate a Rice parameter and its encoded size in closed form:
///
///   bits(0) ~= header + n + 2 * sum - n / 2
///   bits(p) ~= header + n * (p + 1) + sum >> (p - 1) - n / 2
///
/// Coarser orders reuse the same accumulators by merging partition pairs,
/// so a block needs a single pass over the samples regardless of how many
/// partition orders get evaluated. This trades a small amount of ratio
/// against the exact per-parameter counting it replaces, matching the
/// strategy used by the reference encoder.
#[derive(Default)]
struct PrcParameterFinder {
    sums: Vec<u64>,
    counts: Vec<u64>,
    ps: Vec<u8>,
    best_ps: Vec<u8>,
}

impl PrcParameterFinder {
    /// Selects and scores one partition using libFLAC's default estimator.
    /// libFLAC derives the parameter from the mean absolute residual and,
    /// unless built with its optional Rice search, scores only that value.
    #[inline]
    fn select_param(sum: u64, n: u64, max_p: u64) -> (u8, u64) {
        if n == 0 {
            return (0, PRC_HEADER_BITS);
        }
        let mean = sum.saturating_sub(1) / n;
        let p = if sum < 2 || mean == 0 {
            0
        } else {
            u64::from(64 - mean.leading_zeros())
        }
        .min(max_p);
        let folded_sum = if p == 0 { sum << 1 } else { sum >> (p - 1) };
        let bits = PRC_HEADER_BITS + n * (p + 1) + folded_sum - (n >> 1);
        (p as u8, bits)
    }

    pub fn find(&mut self, signal: &[i32], warmup_length: usize, max_p: usize) -> PrcParameter {
        debug_assert!(max_p <= MAX_RICE_PARAMETER);

        let mut partition_order = finest_partition_order(
            signal.len(),
            std::cmp::max(MIN_RICE_PARTITION_SIZE, warmup_length),
        );
        let nparts = 1usize << partition_order;
        let part_size = signal.len() >> partition_order;
        debug_assert_eq!(nparts * part_size, signal.len());

        // Single streaming pass over the samples fills the finest-level
        // sums and sample counts. Partitions before the warm-up boundary
        // stay empty, mirroring the ranges that the bitstream writer uses.
        self.sums.clear();
        self.sums.resize(nparts, 0);
        self.counts.clear();
        self.counts.resize(nparts, 0);
        {
            let sums = &mut self.sums[..];
            let counts = &mut self.counts[..];
            for p in 0..nparts {
                let start =
                    std::cmp::min(std::cmp::max(p * part_size, warmup_length), signal.len());
                let end = (p + 1) * part_size;
                if end <= start {
                    continue;
                }
                let mut acc = 0_u64;
                for &x in &signal[start..end] {
                    acc += u64::from(x.unsigned_abs());
                }
                sums[p] = acc;
                counts[p] = (end - start) as u64;
            }
        }

        let max_p = max_p as u64;
        let mut min_bits = u64::MAX;
        let mut min_order: usize = 0;
        loop {
            let nparts = 1usize << partition_order;
            self.ps.clear();
            let mut total = 0_u64;
            for i in 0..nparts {
                let (p, bits) = Self::select_param(self.sums[i], self.counts[i], max_p);
                total += bits;
                self.ps.push(p);
            }
            if total < min_bits {
                min_bits = total;
                min_order = partition_order;
                std::mem::swap(&mut self.best_ps, &mut self.ps);
            }
            if partition_order == 0 {
                break;
            }
            // Merge partition pairs for the next coarser order.
            for i in 0..(nparts >> 1) {
                self.sums[i] = self.sums[2 * i] + self.sums[2 * i + 1];
                self.counts[i] = self.counts[2 * i] + self.counts[2 * i + 1];
            }
            partition_order -= 1;
        }

        PrcParameter::new(
            min_order,
            self.best_ps[..1 << min_order].to_vec(),
            min_bits as usize,
        )
    }
}

reusable!(PRC_FINDER: PrcParameterFinder);

/// Parameter for PRC (partitioned Rice-coding).
#[derive(Clone, Debug)]
pub struct PrcParameter {
    pub order: usize,
    pub ps: Vec<u8>,
    pub code_bits: usize,
}

impl PrcParameter {
    pub fn new(order: usize, ps: Vec<u8>, code_bits: usize) -> Self {
        Self {
            order,
            ps,
            code_bits,
        }
    }
}

pub fn find_partitioned_rice_parameter(
    signal: &[i32],
    warmup_length: usize,
    max_p: usize,
) -> PrcParameter {
    reuse!(PRC_FINDER, |finder: &mut PrcParameterFinder| {
        finder.find(signal, warmup_length, max_p)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Brute-force reference that mirrors the exact per-parameter bit
    /// accounting of the old table-based estimator.
    fn reference_find(signal: &[i32], warmup_length: usize, max_p: usize) -> PrcParameter {
        let order = finest_partition_order(
            signal.len(),
            std::cmp::max(MIN_RICE_PARTITION_SIZE, warmup_length),
        );
        let part_size = signal.len() >> order;
        let mut best_bits = usize::MAX;
        let mut best_order = 0;
        let mut best_ps = Vec::new();
        for level in 0..=order {
            let nparts = 1usize << (order - level);
            let mut ps = Vec::new();
            let mut total = 0usize;
            for p in 0..nparts {
                let start = std::cmp::min(
                    std::cmp::max((p * part_size) << level, warmup_length),
                    signal.len(),
                );
                let end = std::cmp::min((p + 1) * (part_size << level), signal.len());
                let n = end.saturating_sub(start);
                let mut best_p = 0usize;
                let mut best = usize::MAX;
                for cand in 0..=max_p {
                    let mut bits = 4 + n * (cand + 1);
                    for &x in &signal[start..end] {
                        bits += (encode_signbit(x) >> cand) as usize;
                    }
                    if bits < best {
                        best = bits;
                        best_p = cand;
                    }
                }
                total += best;
                ps.push(best_p as u8);
            }
            if total < best_bits {
                best_bits = total;
                best_order = order - level;
                best_ps = ps;
            }
        }
        PrcParameter::new(best_order, best_ps, best_bits)
    }

    /// Counts the exact bits that `param` would spend on `signal` using
    /// the same per-sample accounting as the bitstream writer.
    fn exact_cost(param: &PrcParameter, signal: &[i32], warmup_length: usize) -> usize {
        let size = signal.len() >> param.order;
        let mut total = 0usize;
        for (p, &choice) in param.ps.iter().enumerate() {
            let start = std::cmp::min(std::cmp::max(p * size, warmup_length), signal.len());
            let end = std::cmp::min((p + 1) * size, signal.len());
            total += 4 + (end - start) * (choice as usize + 1);
            for &x in &signal[start..end] {
                total += (encode_signbit(x) >> choice) as usize;
            }
        }
        total
    }

    #[test]
    fn stays_near_exact_optimum_on_structured_inputs() {
        let mut state = 0x12345678_u32;
        let mut next = move || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state >> 16) as i32) - 32768
        };
        for len in [128usize, 256, 512, 4096] {
            for warmup in [0usize, 4, 32, 128] {
                // Constant runs, alternating ramps, and pseudo-random noise
                // cover the degenerate and the typical cases.
                for kind in 0..3 {
                    let signal: Vec<i32> = match kind {
                        0 => vec![next() / 1048576; len],
                        1 => (0..len).map(|i| ((i as i32 % 97) - 48) * 512).collect(),
                        _ => (0..len).map(|_| next()).collect(),
                    };
                    let got = find_partitioned_rice_parameter(&signal, warmup, 30);
                    let want = reference_find(&signal, warmup, 30);
                    assert_eq!(got.ps.len(), 1 << got.order);
                    let spent = exact_cost(&got, &signal, warmup);
                    let optimum = exact_cost(&want, &signal, warmup);
                    assert!(
                        spent <= optimum + (optimum / 20) + 64,
                        "len={len} warmup={warmup} kind={kind}: spent {spent} vs optimum {optimum}"
                    );
                }
            }
        }
    }
}
