//! Allocation-free steady-state encoder for latency-sized raw FLAC frames.
//!
//! This is deliberately narrower than the generic component encoder. It
//! implements the fixed-predictor path used by libFLAC's low compression
//! levels and writes the selected representation directly to the packet.

use crate::crc::{crc16_flac, crc8_flac};
use crate::frame::{FlacFrameConfig, FlacProfile};

const MAX_FIXED_ORDER: usize = 4;
const MIN_RICE_PARTITION_SIZE: usize = 64;
const MAX_RICE_PARAMETER: u8 = 30;
// Five-millisecond 48/96 kHz blocks use at most four partitions. Keeping a
// little headroom makes the plan stack-only without imposing a large object.
const MAX_TARGET_PARTITIONS: usize = 8;

pub(crate) struct PacketEncoder {
    config: FlacFrameConfig,
    planar: Vec<i32>,
    mid: Vec<i32>,
    side: Vec<i32>,
    residuals: Vec<u32>,
}

impl PacketEncoder {
    pub(crate) fn new(config: FlacFrameConfig) -> Self {
        let frame_length = config.frame_length as usize;
        let sample_count = frame_length * usize::from(config.channels);
        Self {
            config,
            planar: vec![0; sample_count],
            mid: vec![0; frame_length],
            side: vec![0; frame_length],
            residuals: vec![0; sample_count.max(frame_length * 4)],
        }
    }

    /// Returns whether this specialized core covers the configured geometry.
    pub(crate) fn supports(&self) -> bool {
        matches!(
            (self.config.sample_rate, self.config.frame_length),
            (48_000, 240) | (96_000, 480)
        ) && !matches!(self.config.profile, FlacProfile::Maximum)
    }

    pub(crate) fn encode(
        &mut self,
        interleaved: &[i32],
        sequence: u32,
        output: &mut Vec<u8>,
    ) -> usize {
        debug_assert!(self.supports());
        let channels = usize::from(self.config.channels);
        let frame_length = self.config.frame_length as usize;
        debug_assert_eq!(interleaved.len(), channels * frame_length);

        let magnitude_bits = u32::from(self.config.bits_per_sample - 1);
        let minimum = -(1_i32 << magnitude_bits);
        let maximum = (1_i32 << magnitude_bits) - 1;
        match channels {
            1 => {
                for (target, &sample) in self.planar.iter_mut().zip(interleaved) {
                    *target = sample.clamp(minimum, maximum);
                }
            }
            2 => {
                let (left, right) = self.planar.split_at_mut(frame_length);
                for (frame, samples) in interleaved.chunks_exact(2).enumerate() {
                    let l = samples[0].clamp(minimum, maximum);
                    let r = samples[1].clamp(minimum, maximum);
                    left[frame] = l;
                    right[frame] = r;
                    self.mid[frame] = (l + r) >> 1;
                    self.side[frame] = l - r;
                }
            }
            _ => {
                for (frame, samples) in interleaved.chunks_exact(channels).enumerate() {
                    for (channel, &sample) in samples.iter().enumerate() {
                        self.planar[channel * frame_length + frame] =
                            sample.clamp(minimum, maximum);
                    }
                }
            }
        }

        let mut bytes = std::mem::take(output);
        bytes.clear();
        bytes.reserve(interleaved.len() * usize::from(self.config.bits_per_sample).div_ceil(8));

        // libFLAC levels 0 and 2 both search fixed predictors 0..=4. Level 2
        // additionally searches stereo decorrelation assignments below.
        let max_order = MAX_FIXED_ORDER;
        let bits_per_sample = self.config.bits_per_sample;

        if channels == 2 {
            let (left, right) = self.planar.split_at(frame_length);
            let right = &right[..frame_length];

            let (left_residual, residuals) = self.residuals.split_at_mut(frame_length);
            let (right_residual, residuals) = residuals.split_at_mut(frame_length);
            let (mid_residual, side_residual) = residuals.split_at_mut(frame_length);
            let side_residual = &mut side_residual[..frame_length];
            let left_plan = analyze_channel(left, bits_per_sample, max_order, left_residual);
            let right_plan = analyze_channel(right, bits_per_sample, max_order, right_residual);
            let search_stereo = matches!(self.config.profile, FlacProfile::Balanced);
            let (mid_plan, side_plan) = if search_stereo {
                (
                    analyze_channel(&self.mid, bits_per_sample, max_order, mid_residual),
                    analyze_channel(&self.side, bits_per_sample + 1, max_order, side_residual),
                )
            } else {
                // These plans are never written when channel assignment stays
                // independent; avoiding both analyses is the level-0 path.
                (left_plan, right_plan)
            };

            let mut assignment = StereoAssignment::Independent;
            let mut best_bits = left_plan.bits + right_plan.bits;
            if search_stereo {
                for (candidate, bits) in [
                    (StereoAssignment::LeftSide, left_plan.bits + side_plan.bits),
                    (
                        StereoAssignment::RightSide,
                        side_plan.bits + right_plan.bits,
                    ),
                    (StereoAssignment::MidSide, mid_plan.bits + side_plan.bits),
                ] {
                    if bits < best_bits {
                        assignment = candidate;
                        best_bits = bits;
                    }
                }
            }

            write_header(
                &mut bytes,
                self.config.frame_length,
                self.config.sample_rate,
                assignment.tag(),
                bits_per_sample,
                sequence,
            );
            let mut writer = BitWriter::from_aligned(bytes);
            match assignment {
                StereoAssignment::Independent => {
                    write_subframe(&mut writer, left, left_residual, bits_per_sample, left_plan);
                    write_subframe(
                        &mut writer,
                        right,
                        right_residual,
                        bits_per_sample,
                        right_plan,
                    );
                }
                StereoAssignment::LeftSide => {
                    write_subframe(&mut writer, left, left_residual, bits_per_sample, left_plan);
                    write_subframe(
                        &mut writer,
                        &self.side,
                        side_residual,
                        bits_per_sample + 1,
                        side_plan,
                    );
                }
                StereoAssignment::RightSide => {
                    write_subframe(
                        &mut writer,
                        &self.side,
                        side_residual,
                        bits_per_sample + 1,
                        side_plan,
                    );
                    write_subframe(
                        &mut writer,
                        right,
                        right_residual,
                        bits_per_sample,
                        right_plan,
                    );
                }
                StereoAssignment::MidSide => {
                    write_subframe(
                        &mut writer,
                        &self.mid,
                        mid_residual,
                        bits_per_sample,
                        mid_plan,
                    );
                    write_subframe(
                        &mut writer,
                        &self.side,
                        side_residual,
                        bits_per_sample + 1,
                        side_plan,
                    );
                }
            }
            writer.align_to_byte();
            bytes = writer.into_inner();
        } else {
            write_header(
                &mut bytes,
                self.config.frame_length,
                self.config.sample_rate,
                (channels - 1) as u8,
                bits_per_sample,
                sequence,
            );
            let mut writer = BitWriter::from_aligned(bytes);
            for channel in 0..channels {
                let start = channel * frame_length;
                let samples = &self.planar[start..start + frame_length];
                let residual = &mut self.residuals[start..start + frame_length];
                let plan = analyze_channel(samples, bits_per_sample, max_order, residual);
                write_subframe(&mut writer, samples, residual, bits_per_sample, plan);
            }
            writer.align_to_byte();
            bytes = writer.into_inner();
        }

        let checksum = crc16_flac(&bytes);
        bytes.extend_from_slice(&checksum.to_be_bytes());
        let written = bytes.len();
        *output = bytes;
        written
    }
}

#[derive(Clone, Copy)]
enum StereoAssignment {
    Independent,
    LeftSide,
    RightSide,
    MidSide,
}

impl StereoAssignment {
    const fn tag(self) -> u8 {
        match self {
            Self::Independent => 1,
            Self::LeftSide => 8,
            Self::RightSide => 9,
            Self::MidSide => 10,
        }
    }
}

#[derive(Clone, Copy)]
struct ChannelPlan {
    bits: usize,
    coding: ChannelCoding,
}

#[derive(Clone, Copy)]
enum ChannelCoding {
    Constant(i32),
    Verbatim,
    Fixed {
        order: u8,
        partition_order: u8,
        rice_parameters: [u8; MAX_TARGET_PARTITIONS],
    },
}

fn analyze_channel(
    samples: &[i32],
    bits_per_sample: u8,
    max_order: usize,
    residual: &mut [u32],
) -> ChannelPlan {
    let verbatim_bits = 8 + samples.len() * usize::from(bits_per_sample);
    if samples[1..].iter().all(|&sample| sample == samples[0]) {
        return ChannelPlan {
            bits: 8 + usize::from(bits_per_sample),
            coding: ChannelCoding::Constant(samples[0]),
        };
    }

    let max_order = max_order.min(samples.len() - 1);
    let order = select_fixed_order(samples, max_order);
    fill_fixed_residual(samples, order, residual);
    let rice = analyze_rice(residual, order);
    let fixed_bits = 8 + order * usize::from(bits_per_sample) + rice.exact_bits;
    if fixed_bits >= verbatim_bits {
        ChannelPlan {
            bits: verbatim_bits,
            coding: ChannelCoding::Verbatim,
        }
    } else {
        ChannelPlan {
            bits: fixed_bits,
            coding: ChannelCoding::Fixed {
                order: order as u8,
                partition_order: rice.partition_order,
                rice_parameters: rice.parameters,
            },
        }
    }
}

fn select_fixed_order(signal: &[i32], max_order: usize) -> usize {
    let mut totals = [0_u64; MAX_FIXED_ORDER + 1];
    for index in max_order..signal.len() {
        let x = signal[index];
        totals[0] += u64::from(x.unsigned_abs());
        if max_order >= 1 {
            totals[1] += u64::from(x.wrapping_sub(signal[index - 1]).unsigned_abs());
        }
        if max_order >= 2 {
            let error = x
                .wrapping_sub(signal[index - 1].wrapping_mul(2))
                .wrapping_add(signal[index - 2]);
            totals[2] += u64::from(error.unsigned_abs());
        }
        if max_order >= 3 {
            let error = x
                .wrapping_sub(signal[index - 1].wrapping_mul(3))
                .wrapping_add(signal[index - 2].wrapping_mul(3))
                .wrapping_sub(signal[index - 3]);
            totals[3] += u64::from(error.unsigned_abs());
        }
        if max_order >= 4 {
            let error = x
                .wrapping_sub(signal[index - 1].wrapping_mul(4))
                .wrapping_add(signal[index - 2].wrapping_mul(6))
                .wrapping_sub(signal[index - 3].wrapping_mul(4))
                .wrapping_add(signal[index - 4]);
            totals[4] += u64::from(error.unsigned_abs());
        }
    }
    totals[..=max_order]
        .iter()
        .enumerate()
        .min_by_key(|(order, total)| (**total, *order))
        .map_or(0, |(order, _)| order)
}

fn fill_fixed_residual(signal: &[i32], order: usize, residual: &mut [u32]) {
    residual[..order].fill(0);
    match order {
        0 => {
            for (target, &sample) in residual.iter_mut().zip(signal) {
                *target = encode_signbit(sample);
            }
        }
        1 => {
            for index in 1..signal.len() {
                residual[index] = encode_signbit(signal[index].wrapping_sub(signal[index - 1]));
            }
        }
        2 => {
            for index in 2..signal.len() {
                let error = signal[index]
                    .wrapping_sub(signal[index - 1].wrapping_mul(2))
                    .wrapping_add(signal[index - 2]);
                residual[index] = encode_signbit(error);
            }
        }
        3 => {
            for index in 3..signal.len() {
                let error = signal[index]
                    .wrapping_sub(signal[index - 1].wrapping_mul(3))
                    .wrapping_add(signal[index - 2].wrapping_mul(3))
                    .wrapping_sub(signal[index - 3]);
                residual[index] = encode_signbit(error);
            }
        }
        4 => {
            for index in 4..signal.len() {
                let error = signal[index]
                    .wrapping_sub(signal[index - 1].wrapping_mul(4))
                    .wrapping_add(signal[index - 2].wrapping_mul(6))
                    .wrapping_sub(signal[index - 3].wrapping_mul(4))
                    .wrapping_add(signal[index - 4]);
                residual[index] = encode_signbit(error);
            }
        }
        _ => unreachable!(),
    }
}

struct RicePlan {
    partition_order: u8,
    parameters: [u8; MAX_TARGET_PARTITIONS],
    exact_bits: usize,
}

fn analyze_rice(residual: &[u32], warmup: usize) -> RicePlan {
    let finest_order = finest_partition_order(residual.len(), warmup);
    let finest_parts = 1usize << finest_order;
    debug_assert!(finest_parts <= MAX_TARGET_PARTITIONS);
    let finest_size = residual.len() >> finest_order;
    let mut sums = [0_u64; MAX_TARGET_PARTITIONS];
    let mut counts = [0_u64; MAX_TARGET_PARTITIONS];

    for partition in 0..finest_parts {
        let start = (partition * finest_size).max(warmup);
        let end = (partition + 1) * finest_size;
        let mut sum = 0_u64;
        for &folded in &residual[start..end] {
            sum += u64::from((folded + 1) >> 1);
        }
        sums[partition] = sum;
        counts[partition] = (end - start) as u64;
    }

    let mut best_cost = u64::MAX;
    let mut best_order = 0;
    let mut best_parameters = [0_u8; MAX_TARGET_PARTITIONS];
    let mut candidate_parameters = [0_u8; MAX_TARGET_PARTITIONS];
    let mut order = finest_order;
    loop {
        let parts = 1usize << order;
        let mut sample_cost = 0_u64;
        let mut rice2 = false;
        for partition in 0..parts {
            let parameter = select_rice_parameter(sums[partition], counts[partition]);
            candidate_parameters[partition] = parameter;
            rice2 |= parameter > 14;
            sample_cost +=
                estimated_rice_sample_bits(sums[partition], counts[partition], parameter);
        }
        let parameter_bits = if rice2 { 5 } else { 4 };
        let cost = 6 + (parts * parameter_bits) as u64 + sample_cost;
        if cost < best_cost {
            best_cost = cost;
            best_order = order;
            best_parameters[..parts].copy_from_slice(&candidate_parameters[..parts]);
        }
        if order == 0 {
            break;
        }
        for partition in 0..(parts >> 1) {
            sums[partition] = sums[2 * partition] + sums[2 * partition + 1];
            counts[partition] = counts[2 * partition] + counts[2 * partition + 1];
        }
        order -= 1;
    }

    let parts = 1usize << best_order;
    let partition_size = residual.len() >> best_order;
    let parameter_bits = if best_parameters[..parts].iter().any(|&p| p > 14) {
        5
    } else {
        4
    };
    let mut exact_bits = 6 + parts * parameter_bits;
    for (partition, &parameter) in best_parameters[..parts].iter().enumerate() {
        let start = (partition * partition_size).max(warmup);
        let end = (partition + 1) * partition_size;
        for &folded in &residual[start..end] {
            exact_bits += (folded >> parameter) as usize + usize::from(parameter) + 1;
        }
    }

    RicePlan {
        partition_order: best_order as u8,
        parameters: best_parameters,
        exact_bits,
    }
}

fn finest_partition_order(block_size: usize, warmup: usize) -> usize {
    let minimum_size = MIN_RICE_PARTITION_SIZE.max(warmup);
    let split_count = block_size / minimum_size;
    let size_order = if split_count == 0 {
        0
    } else {
        split_count.ilog2() as usize
    };
    size_order
        .min(block_size.trailing_zeros() as usize)
        .min(MAX_TARGET_PARTITIONS.ilog2() as usize)
}

#[inline]
fn select_rice_parameter(sum: u64, count: u64) -> u8 {
    if count == 0 || sum < 2 {
        return 0;
    }
    let mean = (sum - 1) / count;
    if mean == 0 {
        0
    } else {
        (64 - mean.leading_zeros() as u8).min(MAX_RICE_PARAMETER)
    }
}

#[inline]
fn estimated_rice_sample_bits(sum: u64, count: u64, parameter: u8) -> u64 {
    let folded_sum = if parameter == 0 {
        sum << 1
    } else {
        sum >> (parameter - 1)
    };
    count * (u64::from(parameter) + 1) + folded_sum - (count >> 1)
}

#[inline]
const fn encode_signbit(value: i32) -> u32 {
    (value.unsigned_abs() << 1) - (value < 0) as u32
}

fn write_header(
    output: &mut Vec<u8>,
    block_size: u32,
    sample_rate: u32,
    channel_assignment: u8,
    bits_per_sample: u8,
    sequence: u32,
) {
    output.extend_from_slice(&[0xff, 0xf8]);
    let block_tag = if block_size <= 256 { 6 } else { 7 };
    let sample_rate_tag = match sample_rate {
        48_000 => 10,
        96_000 => 11,
        _ => unreachable!("packet encoder only accepts 48/96 kHz"),
    };
    output.push((block_tag << 4) | sample_rate_tag);
    let sample_size_tag = match bits_per_sample {
        16 => 4,
        24 => 6,
        _ => unreachable!("validated packet sample depth"),
    };
    output.push((channel_assignment << 4) | (sample_size_tag << 1));
    let encoded_sequence = crate::component::encode_to_utf8like(u64::from(sequence))
        .expect("31-bit frame sequence always has a FLAC UTF-8-like representation");
    output.extend_from_slice(&encoded_sequence);
    if block_size <= 256 {
        output.push((block_size - 1) as u8);
    } else {
        output.extend_from_slice(&((block_size - 1) as u16).to_be_bytes());
    }
    output.push(crc8_flac(output));
}

fn write_subframe(
    writer: &mut BitWriter,
    samples: &[i32],
    residual: &[u32],
    bits_per_sample: u8,
    plan: ChannelPlan,
) {
    match plan.coding {
        ChannelCoding::Constant(value) => {
            writer.write_bits(0, 8);
            writer.write_signed(value, bits_per_sample);
        }
        ChannelCoding::Verbatim => {
            writer.write_bits(0x02, 8);
            for &sample in samples {
                writer.write_signed(sample, bits_per_sample);
            }
        }
        ChannelCoding::Fixed {
            order,
            partition_order,
            rice_parameters,
        } => {
            writer.write_bits(u64::from(0x10 | (order << 1)), 8);
            for &sample in &samples[..usize::from(order)] {
                writer.write_signed(sample, bits_per_sample);
            }
            let parts = 1usize << partition_order;
            let parameter_bits = if rice_parameters[..parts].iter().any(|&p| p > 14) {
                writer.write_bits(1, 2);
                5
            } else {
                writer.write_bits(0, 2);
                4
            };
            writer.write_bits(u64::from(partition_order), 4);
            let partition_size = samples.len() >> partition_order;
            for (partition, &parameter) in rice_parameters[..parts].iter().enumerate() {
                writer.write_bits(u64::from(parameter), parameter_bits);
                let start = (partition * partition_size).max(usize::from(order));
                let end = (partition + 1) * partition_size;
                for &folded in &residual[start..end] {
                    writer.write_zeros((folded >> parameter) as usize);
                    let remainder = folded & ((1_u32 << parameter) - 1);
                    writer.write_bits(
                        u64::from((1_u32 << parameter) | remainder),
                        usize::from(parameter) + 1,
                    );
                }
            }
        }
    }
}

struct BitWriter {
    bytes: Vec<u8>,
    word: u64,
    used: u32,
}

impl BitWriter {
    fn from_aligned(bytes: Vec<u8>) -> Self {
        Self {
            bytes,
            word: 0,
            used: 0,
        }
    }

    #[inline]
    fn write_bits(&mut self, value: u64, bit_count: usize) {
        debug_assert!(bit_count <= 64);
        if bit_count == 0 {
            return;
        }
        let mut remaining = bit_count as u32;
        while remaining != 0 {
            let take = remaining.min(64 - self.used);
            let shift = remaining - take;
            let mask = if take == 64 {
                u64::MAX
            } else {
                (1_u64 << take) - 1
            };
            let chunk = (value >> shift) & mask;
            self.word |= chunk << (64 - self.used - take);
            self.used += take;
            remaining -= take;
            if self.used == 64 {
                self.flush_word();
            }
        }
    }

    #[inline]
    fn write_signed(&mut self, value: i32, bit_count: u8) {
        let mask = (1_u64 << bit_count) - 1;
        self.write_bits(u64::from(value as u32) & mask, usize::from(bit_count));
    }

    #[inline]
    fn write_zeros(&mut self, mut bit_count: usize) {
        if self.used != 0 {
            let take = bit_count.min((64 - self.used) as usize);
            self.used += take as u32;
            bit_count -= take;
            if self.used == 64 {
                self.flush_word();
            }
            if bit_count == 0 {
                return;
            }
        }
        if bit_count >= 64 {
            let byte_count = (bit_count >> 6) * 8;
            self.bytes.resize(self.bytes.len() + byte_count, 0);
            bit_count &= 63;
        }
        self.used = bit_count as u32;
    }

    fn align_to_byte(&mut self) {
        self.write_zeros(((8 - (self.used & 7)) & 7) as usize);
    }

    #[inline]
    fn flush_word(&mut self) {
        self.bytes.extend_from_slice(&self.word.to_be_bytes());
        self.word = 0;
        self.used = 0;
    }

    fn into_inner(mut self) -> Vec<u8> {
        if self.used != 0 {
            let byte_count = self.used.div_ceil(8) as usize;
            self.bytes
                .extend_from_slice(&self.word.to_be_bytes()[..byte_count]);
        }
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitsink::{BitSink, ByteSink};

    #[test]
    fn profiles_match_libflac_level_zero_and_two_stereo_search() {
        let samples = (0..240)
            .flat_map(|frame| {
                let sample = ((frame as f64 * 440.0 * std::f64::consts::TAU / 48_000.0).sin()
                    * 2_000_000.0) as i32;
                [sample, sample]
            })
            .collect::<Vec<_>>();
        let encode = |profile| {
            let config = FlacFrameConfig::new(48_000, 2, 24, 240, profile).unwrap();
            let mut encoder = PacketEncoder::new(config);
            let mut packet = Vec::new();
            encoder.encode(&samples, 0, &mut packet);
            packet
        };
        let realtime = encode(FlacProfile::Realtime);
        let balanced = encode(FlacProfile::Balanced);
        assert_eq!(realtime[3] >> 4, 1, "level 0 keeps stereo independent");
        assert!(
            (8..=10).contains(&(balanced[3] >> 4)),
            "level 2 searches stereo decorrelation"
        );
    }

    #[test]
    fn fixed_predictors_recover_polynomials() {
        let constant = vec![17; 240];
        let linear = (0_i32..240).map(|x| x * 31 - 100).collect::<Vec<_>>();
        let quadratic = (0_i32..240).map(|x| x * x - 50 * x + 7).collect::<Vec<_>>();
        assert_eq!(select_fixed_order(&constant, 4), 1);
        assert_eq!(select_fixed_order(&linear, 4), 2);
        assert_eq!(select_fixed_order(&quadratic, 4), 3);
    }

    #[test]
    fn target_geometries_fit_stack_rice_plan() {
        for (block_size, expected_max_order) in [(240, 1), (480, 2)] {
            assert_eq!(finest_partition_order(block_size, 4), expected_max_order);
            assert!((1usize << expected_max_order) <= MAX_TARGET_PARTITIONS);
        }
    }

    #[test]
    fn word_writer_matches_reference_bit_sink_across_boundaries() {
        for seed in 0..32_u64 {
            let mut state = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut writer = BitWriter::from_aligned(Vec::new());
            let mut reference = ByteSink::new();
            for operation in 0..200 {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let width = (state as usize >> 17) % 32;
                let value = state.rotate_left(operation % 64);
                writer.write_bits(value, width);
                reference.write_lsbs(value, width).unwrap();

                if operation % 5 == 0 {
                    let zeros = (state as usize >> 29) % 97;
                    writer.write_zeros(zeros);
                    reference.write_zeros(zeros).unwrap();
                }
            }
            writer.align_to_byte();
            reference.align_to_byte().unwrap();
            assert_eq!(writer.into_inner(), reference.into_inner(), "seed={seed}");
        }
    }
}
