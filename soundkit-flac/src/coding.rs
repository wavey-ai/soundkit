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

//! Controller connecting coding algorithms.

use super::arrayutils::find_sum_abs_f32;
use super::arrayutils::is_constant;
use super::arrayutils::SimdVec;
use super::component::BitRepr;
use super::component::BlockSizeSpec;
use super::component::ChannelAssignment;
use super::component::Constant;
use super::component::FixedLpc;
use super::component::Frame;
use super::component::FrameHeader;
use super::component::FrameOffset;
use super::component::Lpc;
use super::component::Residual;
use super::component::SampleRateSpec;
use super::component::SampleSizeSpec;
use super::component::StreamInfo;
use super::component::SubFrame;
use super::component::Verbatim;
use super::config;
use super::constant::fixed::MAX_LPC_ORDER as MAX_FIXED_LPC_ORDER;
use super::constant::panic_msg;
use super::constant::qlpc::MAX_ORDER as MAX_LPC_ORDER;
use super::constant::MIN_BLOCK_SIZE_FOR_PREDICTION;
use super::error::verify_range;
use super::error::verify_true;
use super::error::EncodeError;
use super::error::Verified;
use super::lpc;
#[cfg(feature = "par")]
use super::par;
use super::rice;
use super::source::FrameBuf;

import_simd!(as simd);

/// Computes rice encoding of a scalar (used in `encode_residual`.)
#[inline]
const fn quotients_and_remainders(err: i32, rice_p: u8) -> (u32, u32) {
    let remainder_mask = (1u32 << rice_p) - 1;
    let err = rice::encode_signbit(err);
    (err >> rice_p, err & remainder_mask)
}

/// Computes rice encoding of a SIMD vector (used in `encode_residual`.)
#[inline]
#[cfg(feature = "simd-nightly")]
fn quotients_and_remainders_simd<const N: usize>(
    err_v: simd::Simd<i32, N>,
    rice_p: u8,
    quotients: &mut [u32],
    remainders: &mut [u32],
) {
    let rice_p_v = simd::Simd::splat(u32::from(rice_p));
    let remainder_mask_v = simd::Simd::splat((1u32 << rice_p) - 1);
    let err_v = rice::encode_signbit_simd(err_v);
    quotients.copy_from_slice((err_v >> rice_p_v).as_ref());
    remainders.copy_from_slice((err_v & remainder_mask_v).as_ref());
}

/// Computes encoding of each residual partition.
///
/// This function is moved out from the main loop for avoiding messy conditoinal
/// compilation due to fakesimd. We had to resort conditional compilation
/// because `as_simd` operation that is provided as an extension to standard
/// types (slice/ Vec) is still there even if we use stable version, so the
/// approach we used in "fakesimd" is not suitable for mimicking this.
///
/// TODO: Probably, it's better to introduce another abstraction for `as_simd`
/// e.g. SIMD-version of `map` so we can do conditional compilation there.
#[cfg(feature = "simd-nightly")]
#[inline]
fn encode_residual_partition(
    start: usize,
    end: usize,
    rice_p: u8,
    errors: &[i32],
    quotients: &mut [u32],
    remainders: &mut [u32],
) {
    const SIMD_N: usize = 8;
    // note that t >= warmup_length because start >= warmup_length.
    let mut t = start;
    let (head, body, tail) = errors[start..end].as_simd::<SIMD_N>();
    for err in head {
        (quotients[t], remainders[t]) = quotients_and_remainders(*err, rice_p);
        t += 1;
    }
    for err_v in body {
        quotients_and_remainders_simd::<SIMD_N>(
            *err_v,
            rice_p,
            &mut quotients[t..t + SIMD_N],
            &mut remainders[t..t + SIMD_N],
        );
        t += SIMD_N;
    }
    for err in tail {
        (quotients[t], remainders[t]) = quotients_and_remainders(*err, rice_p);
        t += 1;
    }
}

/// Computes encoding of each residual partition. (without SIMD)
#[cfg(not(feature = "simd-nightly"))]
#[inline]
fn encode_residual_partition(
    start: usize,
    end: usize,
    rice_p: u8,
    errors: &[i32],
    quotients: &mut [u32],
    remainders: &mut [u32],
) {
    for (t, err) in (start..).zip(errors[start..end].iter()) {
        (quotients[t], remainders[t]) = quotients_and_remainders(*err, rice_p);
    }
}

/// Computes `Residual` from the given error signal and PRC parameters.
fn encode_residual_with_prc_parameter(
    _config: &config::Prc,
    errors: &[i32],
    warmup_length: usize,
    prc_p: rice::PrcParameter,
) -> Residual {
    let block_size = errors.len();
    let nparts = 1 << prc_p.order;
    let part_size = errors.len() >> prc_p.order;
    debug_assert!(part_size >= warmup_length);

    let mut quotients = vec![0u32; block_size];
    let mut remainders = vec![0u32; block_size];

    let mut offset = 0;
    for rice_p in &prc_p.ps[0..nparts] {
        let start = std::cmp::max(offset, warmup_length);
        offset += part_size;
        let end = offset;
        // ^ this is okay because partitions are larger than warmup_length
        encode_residual_partition(start, end, *rice_p, errors, &mut quotients, &mut remainders);
    }
    Residual::from_parts(
        prc_p.order as u8,
        block_size,
        warmup_length,
        prc_p.ps,
        quotients,
        remainders,
    )
}

/// Constructs `Residual` component given the error signal.
pub fn encode_residual(config: &config::Prc, errors: &[i32], warmup_length: usize) -> Residual {
    let prc_p = rice::find_partitioned_rice_parameter(errors, warmup_length, config.max_parameter);
    encode_residual_with_prc_parameter(config, errors, warmup_length, prc_p)
}

type FixedLpcErrors = [SimdVec<i32, 16>; MAX_FIXED_LPC_ORDER + 1];
reusable!(FIXED_LPC_ERRORS: FixedLpcErrors);

/// Resets `FixedLpcErrors` from the given signal.
fn reset_fixed_lpc_errors(errors: &mut FixedLpcErrors, signal: &[i32]) {
    errors[0].reset_from_slice(signal);

    for order in 0..MAX_FIXED_LPC_ORDER {
        let next_order = order + 1;

        let mut carry = 0i32;
        errors[next_order].resize(signal.len(), simd::Simd::default());
        for t in 0..errors[order].simd_len() {
            let x = errors[order].as_ref_simd()[t];
            let mut shifted = x.rotate_elements_right::<1>();
            (shifted[0], carry) = (carry, shifted[0]);
            errors[next_order].as_mut_simd()[t] = x - shifted;
        }
    }
}

/// Estimate bit count from the error.
fn estimate_entropy(errors: &[i32], warmup_len: usize, partitions: usize) -> usize {
    // this function computes partition average of:
    //   (1 + e) log (1 + e) - e * log e
    // where log-base is 2 and e is the average error.
    // This can further be approximated (by Stirling's formula) as:
    //   log(1 + e) + constant
    // given e >> 1, it can further be approximated as log(e); however we don't
    // use this formula as it is anyway cheap to compute.
    let block_size = errors.len();
    let partition_size = block_size.div_ceil(partitions);

    let mut offset = 0;
    let mut acc = 0;
    for _p in 0..partitions {
        let end = std::cmp::min(block_size, offset + partition_size);
        let partition_len = end - offset;
        if end >= warmup_len {
            let sample_count = std::cmp::min(end - warmup_len, partition_len);
            let sum_errors = find_sum_abs_f32::<16>(&errors[offset..end]);
            let avg_errors = sum_errors * 2.0 / (sample_count as f32 + 0.00001);
            let geom_p = 1.0 / (avg_errors + 1.0);
            let xent = avg_errors.mul_add(-(1.0 - geom_p).log2(), -geom_p.log2());
            acc += (xent * sample_count as f32) as usize;
        }
        offset = end;
    }
    acc
}

/// Selects the best LPC order from error signals and encode `Residual`.
fn select_order_and_encode_residual<'a, I>(
    order_sel: &config::OrderSel,
    prc_config: &config::Prc,
    errors: I,
    bits_per_sample: usize,
    baseline_bits: usize,
) -> Option<(usize, Residual)>
where
    I: Iterator<Item = (usize, &'a [i32])>,
{
    let max_rice_p = prc_config.max_parameter;
    match *order_sel {
        config::OrderSel::BitCount => errors
            .map(
                #[inline]
                |(order, err)| {
                    let prc_p = rice::find_partitioned_rice_parameter(err, order, max_rice_p);
                    let bits = bits_per_sample * order + prc_p.code_bits;
                    (order, err, prc_p, bits)
                },
            )
            .min_by_key(|(_order, _err, _prc_p, bits)| *bits)
            .and_then(
                #[inline]
                |(order, err, prc_p, bits)| {
                    (bits < baseline_bits).then(
                        #[inline]
                        || {
                            (
                                order,
                                encode_residual_with_prc_parameter(prc_config, err, order, prc_p),
                            )
                        },
                    )
                },
            ),
        config::OrderSel::ApproxEnt { partitions } => errors
            .map(
                #[inline]
                |(order, err)| {
                    (
                        order,
                        err,
                        estimate_entropy(err, order, partitions) + bits_per_sample * order,
                    )
                },
            )
            .min_by_key(
                #[inline]
                |(_order, _err, bits)| *bits,
            )
            .and_then(
                #[inline]
                |(order, err, bits)| {
                    (bits < baseline_bits).then(|| (order, encode_residual(prc_config, err, order)))
                },
            ),
    }
}

/// Tries `0..=4`-th order fixed LPC and returns the smallest `SubFrame`.
///
/// # Panics
///
/// The current implementation may cause overflow error if `bits_per_sample` is
/// larger than 29. Therefore, it panics when `bits_per_sample` is larger than
/// this.
#[inline]
fn fixed_lpc(
    config: &config::SubFrameCoding,
    signal: &[i32],
    bits_per_sample: u8,
    baseline_bits: usize,
) -> Option<SubFrame> {
    assert!(bits_per_sample < 30);
    let max_order = config.fixed.max_order;

    reuse!(FIXED_LPC_ERRORS, |errors: &mut FixedLpcErrors| {
        reset_fixed_lpc_errors(errors, signal);
        let errors = errors
            .iter()
            .map(SimdVec::as_ref)
            .take(max_order + 1)
            .enumerate();
        select_order_and_encode_residual(
            &config.fixed.order_sel,
            &config.prc,
            errors,
            bits_per_sample as usize,
            baseline_bits,
        )
        .map(|(order, residual)| {
            FixedLpc::from_parts(
                heapless::Vec::from_slice(&signal[..order])
                    .expect("Exceeded maximum order for FixedLpc component."),
                residual,
                bits_per_sample,
            )
            .into()
        })
    })
}

fn perform_qlpc(
    config: &config::SubFrameCoding,
    signal: &[i32],
) -> heapless::Vec<f64, MAX_LPC_ORDER> {
    if config.qlpc.use_direct_mse {
        if config.qlpc.mae_optimization_steps > 0 {
            lpc::lpc_with_irls_mae(
                signal,
                &config.qlpc.window,
                config.qlpc.lpc_order,
                config.qlpc.mae_optimization_steps,
            )
        } else {
            lpc::lpc_with_direct_mse(signal, &config.qlpc.window, config.qlpc.lpc_order)
        }
    } else {
        lpc::lpc_from_autocorr(signal, &config.qlpc.window, config.qlpc.lpc_order)
    }
}

reusable!(QLPC_ERROR_BUFFER: Vec<i32>);

/// Estimates the optimal LPC coefficients and returns `SubFrame`s with these.
///
/// # Panics
///
/// It panics if `signal` is shorter than `MAX_LPC_ORDER_PLUS_1`.
fn estimated_qlpc(
    config: &config::SubFrameCoding,
    signal: &[i32],
    bits_per_sample: u8,
) -> SubFrame {
    let lpc_order = config.qlpc.lpc_order;
    let lpc_coefs = perform_qlpc(config, signal);
    let qlpc = lpc::quantize_parameters(&lpc_coefs[0..lpc_order], config.qlpc.quant_precision);
    let residual = reuse!(QLPC_ERROR_BUFFER, |errors: &mut Vec<i32>| {
        errors.resize(signal.len(), 0i32);
        lpc::compute_error(&qlpc, signal, errors);
        encode_residual(&config.prc, errors, qlpc.order())
    });
    Lpc::from_parts(
        heapless::Vec::from_slice(&signal[0..qlpc.order()])
            .expect("LPC order exceeded the maximum"),
        qlpc,
        residual,
        bits_per_sample,
    )
    .into()
}

/// Finds the best method to encode the given samples, and returns `SubFrame`.
fn encode_subframe(
    config: &config::SubFrameCoding,
    samples: &[i32],
    bits_per_sample: u8,
) -> SubFrame {
    if config.use_constant && is_constant(samples) {
        // Assuming constant is always best if it's applicable.
        Constant::from_parts(samples.len(), samples[0], bits_per_sample).into()
    } else {
        let verbatim_bits =
            Verbatim::count_bits_from_metadata(samples.len(), bits_per_sample as usize);

        let too_short = samples.len() < MIN_BLOCK_SIZE_FOR_PREDICTION;
        let fixed = if !too_short && config.use_fixed {
            fixed_lpc(config, samples, bits_per_sample, verbatim_bits)
        } else {
            None
        };

        let baseline_bits = fixed.as_ref().map_or(verbatim_bits, |x| {
            std::cmp::min(verbatim_bits, x.count_bits())
        });
        let est_lpc = if !too_short && config.use_lpc {
            let candidate = estimated_qlpc(config, samples, bits_per_sample);
            (candidate.count_bits() < baseline_bits).then_some(candidate)
        } else {
            None
        };

        est_lpc
            .or(fixed)
            .filter(|sf| sf.count_bits() < verbatim_bits)
            .unwrap_or_else(|| Verbatim::from_samples(samples, bits_per_sample).into())
    }
}

/// Encode frame with the given channel assignment.
fn encode_frame_impl(
    config: &config::Encoder,
    framebuf: &FrameBuf,
    offset: u64,
    stream_info: &StreamInfo,
    ch_info: &ChannelAssignment,
) -> Frame {
    let nchannels = stream_info.channels();
    let bits_per_sample = stream_info.bits_per_sample();
    let mut frame = Frame::new_empty(
        BlockSizeSpec::from_size(framebuf.filled_size() as u16),
        ch_info.clone(),
        SampleSizeSpec::from_bits(bits_per_sample as u8).unwrap_or(SampleSizeSpec::Unspecified),
        SampleRateSpec::from_freq(stream_info.sample_rate() as u32)
            .unwrap_or(SampleRateSpec::Unspecified),
    );
    frame
        .header_mut()
        .set_frame_offset(FrameOffset::StartSample(offset));
    for ch in 0..nchannels {
        frame.add_subframe(encode_subframe(
            &config.subframe_coding,
            framebuf.channel_slice(ch),
            (bits_per_sample + ch_info.bits_per_sample_offset(ch)) as u8,
        ));
    }

    frame
}

// Recombines stereo frame.
#[allow(clippy::tuple_array_conversions)] // recommended conversion methods are not supported in MSRV
#[inline]
fn recombine_stereo_frame(header: FrameHeader, indep: Frame, ms: Frame) -> Frame {
    let (_header, l, r) = indep
        .into_stereo_channels()
        .expect(panic_msg::DATA_INCONSISTENT);
    let (_header, m, s) = ms
        .into_stereo_channels()
        .expect(panic_msg::DATA_INCONSISTENT);

    let chans = header.channel_assignment().select_channels(l, r, m, s);
    Frame::from_parts(header, vec![chans.0, chans.1])
}

reusable!(MSFRAMEBUF: FrameBuf = FrameBuf::new_stereo_buffer());

/// Tries several stereo channel recombinations and returns the best.
fn try_stereo_coding(
    config: &config::Encoder,
    framebuf: &FrameBuf,
    indep: Frame,
    offset: u64,
    stream_info: &StreamInfo,
) -> Frame {
    reuse!(MSFRAMEBUF, |ms_framebuf: &mut FrameBuf| {
        ms_framebuf.resize(framebuf.size());
        ms_framebuf.fill_stereo_with_iter(
            framebuf
                .channel_slice(0)
                .iter()
                .zip(framebuf.channel_slice(1).iter())
                .map(|(l, r)| ((l + r) >> 1, l - r)),
        );
        let ms_frame = encode_frame_impl(
            config,
            ms_framebuf,
            offset,
            stream_info,
            &ChannelAssignment::MidSide,
        );

        let (bits_l, bits_r, bits_m, bits_s) = (
            indep.subframe(0).unwrap().count_bits(),
            indep.subframe(1).unwrap().count_bits(),
            ms_frame.subframe(0).unwrap().count_bits(),
            ms_frame.subframe(1).unwrap().count_bits(),
        );

        let combinations = [
            config
                .stereo_coding
                .use_leftside
                .then_some((ChannelAssignment::LeftSide, bits_l + bits_s)),
            config
                .stereo_coding
                .use_rightside
                .then_some((ChannelAssignment::RightSide, bits_r + bits_s)),
            config
                .stereo_coding
                .use_midside
                .then_some((ChannelAssignment::MidSide, bits_m + bits_s)),
        ];

        let mut min_bits = bits_l + bits_r;
        let mut min_ch_info = ChannelAssignment::Independent(2);
        for (ch_info, bits) in combinations.iter().flatten() {
            if *bits < min_bits {
                min_bits = *bits;
                min_ch_info = ch_info.clone();
            }
        }
        let mut header = ms_frame.header().clone();
        header.reset_channel_assignment(min_ch_info);
        recombine_stereo_frame(header, indep, ms_frame)
    })
}

/// Finds the best configuration for encoding samples and returns a `Frame`.
fn encode_frame(
    config: &config::Encoder,
    framebuf: &FrameBuf,
    offset: u64,
    stream_info: &StreamInfo,
) -> Frame {
    let nchannels = stream_info.channels();
    let ch_info = ChannelAssignment::Independent(nchannels as u8);
    let mut ret = encode_frame_impl(config, framebuf, offset, stream_info, &ch_info);

    if nchannels == 2 {
        ret = try_stereo_coding(config, framebuf, ret, offset, stream_info);
    }
    ret
}

/// Encodes [`FrameBuf`] to [`Frame`].
///
/// The block size is taken from `FrameBuf::size`.
///
/// # Errors
///
/// Returns an error when an argument is invalid, e.g. when `frame_number` is
/// out of 31-bit range, or `framebuf` contains a sample that is out of range.
///
/// # Examples
///
/// ```
/// # use soundkit_flac::*;
/// use soundkit_flac::config;
/// use soundkit_flac::component::StreamInfo;
/// use soundkit_flac::error::Verify;
/// use soundkit_flac::source::{Fill, FrameBuf};
///
/// let (signal_len, block_size, channels, sample_rate) = (32000, 160, 2, 16000);
/// let signal = vec![0i32; signal_len * channels];
/// let bits_per_sample = 16;
///
/// let mut fb = FrameBuf::with_size(channels, block_size).unwrap();
/// let stream_info = StreamInfo::new(sample_rate, channels, bits_per_sample).unwrap();
/// Fill::fill_interleaved(&mut fb, &signal[..block_size * channels]).unwrap();
///
/// // NOTE: block-size in config will be overridden.
/// let frame = encode_fixed_size_frame(
///     &config::Encoder::default().into_verified().unwrap(),
///     &fb,
///     0,
///     &stream_info
/// );
/// ```
pub fn encode_fixed_size_frame(
    config: &Verified<config::Encoder>,
    framebuf: &FrameBuf,
    frame_number: usize,
    stream_info: &StreamInfo,
) -> Result<Frame, EncodeError> {
    verify_range!(
        "encode_fixed_size_frame (frame_number)",
        frame_number,
        ..(1usize << 31)
    )?;

    framebuf.verify_samples(stream_info.bits_per_sample())?;
    // NOTE: From expected use cases, wrapping `stream_info` is not practical
    // since it is mutable everywhere. On the other hand, verifying it here is
    // a bit redundant. Because broken `stream_info` actually harms nothing,
    // as long as it is consistent with `framebuf` (that is checked in the
    // previous line), we just leave as it is here.

    // A bit awkward, but this function is implemented by overwriting relevant
    // fields of `Frame` generated by `encode_frame`.
    let mut ret = encode_frame(config, framebuf, 0, stream_info);
    ret.header_mut()
        .set_frame_offset(FrameOffset::Frame(frame_number as u32));
    Ok(ret)
}
