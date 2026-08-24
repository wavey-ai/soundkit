// Claxon -- A FLAC decoding library in Rust
// Copyright 2014 Ruud van Asseldonk
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// A copy of the License has been included in the root of the repository.

//! The `subframe` module deals with subframes that make up a frame of the FLAC stream.

use crate::decode::bitstream::BitSource;
#[cfg(test)]
use crate::decode::bitstream::Bitstream;
use crate::decode::error::{fmt_err, Error, Result};

#[derive(Clone, Copy, Debug)]
enum SubframeType {
    Constant,
    Verbatim,
    Fixed(u8),
    Lpc(u8),
}

#[derive(Clone, Copy)]
struct SubframeHeader {
    sf_type: SubframeType,
    wasted_bits_per_sample: u32,
}

fn read_subframe_header<S: BitSource>(input: &mut S) -> Result<SubframeHeader> {
    // The first bit must be a 0 padding bit.
    if (input.read_bit())? {
        return fmt_err("invalid subframe header");
    }

    // Next is a 6-bit subframe type.
    let sf_type = match (input.read_leq_u8(6))? {
        0 => SubframeType::Constant,
        1 => SubframeType::Verbatim,

        // Bit patterns 00001x, 0001xx and 01xxxx are reserved, this library
        // would not know how to handle them, so this is an error. Values that
        // are reserved at the time of writing are a format error, the
        // `Unsupported` error type is for specified features that are not
        // implemented.
        n if (n & 0b111_110 == 0b000_010)
            || (n & 0b111_100 == 0b000_100)
            || (n & 0b110_000 == 0b010_000) =>
        {
            return fmt_err("invalid subframe header, encountered reserved value");
        }

        n if n & 0b111_000 == 0b001_000 => {
            let order = n & 0b000_111;

            // A fixed frame has order up to 4, other bit patterns are reserved.
            if order > 4 {
                return fmt_err("invalid subframe header, encountered reserved value");
            }

            SubframeType::Fixed(order)
        }

        // The only possibility left is bit pattern 1xxxxx, an LPC subframe.
        n => {
            // The xxxxx bits are the order minus one.
            let order_mo = n & 0b011_111;
            SubframeType::Lpc(order_mo + 1)
        }
    };

    // Next bits indicates whether there are wasted bits per sample.
    let wastes_bits = (input.read_bit())?;

    // If so, k - 1 zero bits follow, where k is the number of wasted bits.
    let wasted_bits = if !wastes_bits {
        0
    } else {
        1 + (input.read_unary())?
    };

    // The spec puts no bounds on the number of wasted bits per sample, but more
    // than 31 does not make sense, as it would remove all data even for 32-bit
    // samples.
    if wasted_bits > 31 {
        return fmt_err("wasted bits per sample must not exceed 31");
    }

    let subframe_header = SubframeHeader {
        sf_type: sf_type,
        wasted_bits_per_sample: wasted_bits,
    };
    Ok(subframe_header)
}

/// Given a signed two's complement integer in the `bits` least significant
/// bits of `val`, extends the sign bit to a valid 16-bit signed integer.
#[inline(always)]
fn extend_sign_u16(val: u16, bits: u32) -> i16 {
    // First shift the value so the desired sign bit is the actual sign bit,
    // then convert to a signed integer, and then do an arithmetic shift back,
    // which will extend the sign bit.
    return ((val << (16 - bits)) as i16) >> (16 - bits);
}

#[test]
fn verify_extend_sign_u16() {
    assert_eq!(5, extend_sign_u16(5, 4));
    assert_eq!(0x3ffe, extend_sign_u16(0x3ffe, 15));
    assert_eq!(-5, extend_sign_u16(16 - 5, 4));
    assert_eq!(-3, extend_sign_u16(512 - 3, 9));
    assert_eq!(-1, extend_sign_u16(0xffff, 16));
    assert_eq!(-2, extend_sign_u16(0xfffe, 16));
    assert_eq!(-1, extend_sign_u16(0x7fff, 15));
}

/// Given a signed two's complement integer in the `bits` least significant
/// bits of `val`, extends the sign bit to a valid 32-bit signed integer.
#[inline(always)]
pub fn extend_sign_u32(val: u32, bits: u32) -> i32 {
    // First shift the value so the desired sign bit is the actual sign bit,
    // then convert to a signed integer, and then do an arithmetic shift back,
    // which will extend the sign bit.
    ((val << (32 - bits)) as i32) >> (32 - bits)
}

#[test]
fn verify_extend_sign_u32() {
    assert_eq!(5, extend_sign_u32(5, 4));
    assert_eq!(0x3ffffffe, extend_sign_u32(0x3ffffffe, 31));
    assert_eq!(-5, extend_sign_u32(16 - 5, 4));
    assert_eq!(-3, extend_sign_u32(512 - 3, 9));
    assert_eq!(-2, extend_sign_u32(0xfffe, 16));
    assert_eq!(-1, extend_sign_u32(0xffffffff_u32, 32));
    assert_eq!(-2, extend_sign_u32(0xfffffffe_u32, 32));
    assert_eq!(-1, extend_sign_u32(0x7fffffff, 31));

    // The data below are samples from a real FLAC stream.
    assert_eq!(-6392, extend_sign_u32(124680, 17));
    assert_eq!(-6605, extend_sign_u32(124467, 17));
    assert_eq!(-6850, extend_sign_u32(124222, 17));
    assert_eq!(-7061, extend_sign_u32(124011, 17));
}

/// Decodes a signed number from Rice coding to the two's complement.
///
/// The Rice coding used by FLAC operates on unsigned integers, but the
/// residual is signed. The mapping is done as follows:
///
///  0 -> 0
/// -1 -> 1
///  1 -> 2
/// -2 -> 3
///  2 -> 4
///  etc.
///
/// This function takes the unsigned value and converts it into a signed
/// number.
#[inline(always)]
fn rice_to_signed(val: u32) -> i32 {
    // The following bit-level hackery compiles to only four instructions on
    // x64. It is equivalent to the following code:
    //
    //   if val & 1 == 1 {
    //       -1 - (val / 2) as i32
    //   } else {
    //       (val / 2) as i32
    //   }
    //
    let half = (val >> 1) as i32;
    let extended_bit_0 = ((val << 31) as i32) >> 31;
    half ^ extended_bit_0
}

#[test]
fn verify_rice_to_signed() {
    assert_eq!(rice_to_signed(0), 0);
    assert_eq!(rice_to_signed(1), -1);
    assert_eq!(rice_to_signed(2), 1);
    assert_eq!(rice_to_signed(3), -2);
    assert_eq!(rice_to_signed(4), 2);
}

/// Decodes a subframe into the provided block-size buffer.
///
/// It is assumed that the length of the buffer is the block size.
pub fn decode<S: BitSource>(input: &mut S, bps: u32, buffer: &mut [i32]) -> Result<()> {
    // The sample type i32 should be wide enough to accomodate for all bits of
    // the stream, but this can be verified at a higher level than here. Still,
    // it is a good idea to make the assumption explicit. FLAC supports up to
    // sample widths of 32 in theory, so with the delta between channels that
    // requires 33 bits, but the reference decoder supports only subset FLAC of
    // 24 bits per sample at most, so restricting ourselves to i32 is fine.
    debug_assert!(32 >= bps);

    let header = (read_subframe_header(input))?;

    if header.wasted_bits_per_sample >= bps {
        return fmt_err("subframe has no non-wasted bits");
    }

    // If there are wasted bits, the subframe stores samples with a lower bps
    // than the stream bps. We later shift all the samples left to correct this.
    let sf_bps = bps - header.wasted_bits_per_sample;

    match header.sf_type {
        SubframeType::Constant => (decode_constant(input, sf_bps, buffer))?,
        SubframeType::Verbatim => (decode_verbatim(input, sf_bps, buffer))?,
        SubframeType::Fixed(ord) => (decode_fixed(input, sf_bps, ord as u32, buffer))?,
        SubframeType::Lpc(ord) => (decode_lpc(input, sf_bps, ord as u32, buffer))?,
    }

    // Finally, everything must be shifted by 'wasted bits per sample' to
    // the left. Note: it might be better performance-wise to do this on
    // the fly while decoding. That could be done if this is a bottleneck.
    if header.wasted_bits_per_sample > 0 {
        debug_assert!(
            header.wasted_bits_per_sample <= 31,
            "Cannot shift by more than the sample width."
        );
        for s in buffer {
            // For a valid FLAC file, this shift does not overflow. For an
            // invalid file it might, and then we decode garbage, but we don't
            // crash the program in debug mode due to shift overflow.
            *s = s.wrapping_shl(header.wasted_bits_per_sample);
        }
    }

    Ok(())
}

#[derive(Copy, Clone, Debug)]
enum RicePartitionType {
    Rice,
    Rice2,
}

fn decode_residual<S: BitSource>(input: &mut S, block_size: u16, buffer: &mut [i32]) -> Result<()> {
    // Residual starts with two bits of coding method.
    let partition_type = match (input.read_leq_u8(2))? {
        0b00 => RicePartitionType::Rice,
        0b01 => RicePartitionType::Rice2,
        // 10 and 11 are reserved.
        _ => return fmt_err("invalid residual, encountered reserved value"),
    };

    // Next are 4 bits partition order.
    let order = (input.read_leq_u8(4))?;

    // There are 2^order partitions. Note: the specification states a 4-bit
    // partition order, so the order is at most 31, so there could be 2^31
    // partitions, but the block size is a 16-bit number, so there are at
    // most 2^16 - 1 samples in the block. No values have been marked as
    // invalid by the specification though.
    let n_partitions = 1u32 << order;
    let n_samples_per_partition = block_size >> order;

    // The partitions together must fill the block. If the block size is not a
    // multiple of 2^order; if we shifted off some bits, then we would not fill
    // the entire block. Such a partition order is invalid for this block size.
    if block_size & (n_partitions - 1) as u16 != 0 {
        return fmt_err("invalid partition order");
    }

    // NOTE: the check above checks that block_size is a multiple of n_partitions
    // (this works because n_partitions is a power of 2). The check below is
    // equivalent but more expensive.
    debug_assert_eq!(
        n_partitions * n_samples_per_partition as u32,
        block_size as u32
    );

    let n_warm_up = block_size - buffer.len() as u16;

    // The partition size must be at least as big as the number of warm-up
    // samples, otherwise the size of the first partition is negative.
    if n_warm_up > n_samples_per_partition {
        return fmt_err("invalid residual");
    }

    // Finally decode the partitions themselves.
    match partition_type {
        RicePartitionType::Rice => {
            let mut start = 0;
            let mut len = n_samples_per_partition - n_warm_up;
            for _ in 0..n_partitions {
                let slice = &mut buffer[start..start + len as usize];
                (decode_rice_partition(input, slice))?;
                start = start + len as usize;
                len = n_samples_per_partition;
            }
        }
        RicePartitionType::Rice2 => {
            let mut start = 0;
            let mut len = n_samples_per_partition - n_warm_up;
            for _ in 0..n_partitions {
                let slice = &mut buffer[start..start + len as usize];
                (decode_rice2_partition(input, slice))?;
                start = start + len as usize;
                len = n_samples_per_partition;
            }
        }
    }

    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn decode_residual_with<S: BitSource, F: FnMut(i32) -> i32>(
    input: &mut S,
    block_size: u16,
    buffer: &mut [i32],
    mut map: F,
) -> Result<()> {
    let partition_type = match input.read_leq_u8(2)? {
        0b00 => RicePartitionType::Rice,
        0b01 => RicePartitionType::Rice2,
        _ => return fmt_err("invalid residual, encountered reserved value"),
    };
    let order = input.read_leq_u8(4)?;
    let n_partitions = 1u32 << order;
    let n_samples_per_partition = block_size >> order;
    if block_size & (n_partitions - 1) as u16 != 0 {
        return fmt_err("invalid partition order");
    }
    let n_warm_up = block_size - buffer.len() as u16;
    if n_warm_up > n_samples_per_partition {
        return fmt_err("invalid residual");
    }

    let mut start = 0;
    let mut len = n_samples_per_partition - n_warm_up;
    for _ in 0..n_partitions {
        let slice = &mut buffer[start..start + len as usize];
        match partition_type {
            RicePartitionType::Rice => decode_rice_partition_with(input, slice, &mut map)?,
            RicePartitionType::Rice2 => decode_rice2_partition_with(input, slice, &mut map)?,
        }
        start += len as usize;
        len = n_samples_per_partition;
    }
    Ok(())
}

// Performance note: all Rice partitions in real-world FLAC files are Rice
// partitions, not Rice2 partitions. Therefore it makes sense to inline this
// function into decode_residual.
#[inline(always)]
fn decode_rice_partition<S: BitSource>(input: &mut S, buffer: &mut [i32]) -> Result<()> {
    // A Rice partition (not Rice2), starts with a 4-bit Rice parameter.
    let rice_param = (input.read_leq_u8(4))? as u32;

    // All ones is an escape code that indicates unencoded binary.
    if rice_param == 0b1111 {
        return decode_unencoded_partition(input, buffer);
    }

    // The fused `read_rice_unsigned` resolves quotient and remainder from a
    // single cache window when possible, which is where nearly all decode
    // time used to go.
    input.decode_rice_partition_into(rice_param, buffer, rice_to_signed)?;
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
#[inline(always)]
fn decode_rice_partition_with<S: BitSource, F: FnMut(i32) -> i32>(
    input: &mut S,
    buffer: &mut [i32],
    map: &mut F,
) -> Result<()> {
    let rice_param = input.read_leq_u8(4)? as u32;
    if rice_param == 0b1111 {
        return decode_unencoded_partition_with(input, buffer, map);
    }
    input.decode_rice_partition_into(rice_param, buffer, |value| map(rice_to_signed(value)))?;
    Ok(())
}

// Keep this out of line so the larger three-way remainder-width dispatch does
// not bloat the residual decoder. Rice2 was historically rare, but current
// FFmpeg versions emit it for high-resolution input, so it is not a cold path.
#[inline(never)]
fn decode_rice2_partition<S: BitSource>(input: &mut S, buffer: &mut [i32]) -> Result<()> {
    // A Rice2 partition, starts with a 5-bit Rice parameter.
    let rice_param = (input.read_leq_u8(5))? as u32;

    // All ones is an escape code that indicates unencoded binary.
    if rice_param == 0b11111 {
        return decode_unencoded_partition(input, buffer);
    }

    // The fused partition decode handles every remainder width, so no
    // dispatch on the parameter is needed here. Rice2 is emitted by current
    // FFmpeg encoders for high-resolution input, so it is not a cold path.
    input.decode_rice_partition_into(rice_param, buffer, rice_to_signed)?;
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
#[inline(never)]
fn decode_rice2_partition_with<S: BitSource, F: FnMut(i32) -> i32>(
    input: &mut S,
    buffer: &mut [i32],
    map: &mut F,
) -> Result<()> {
    let rice_param = input.read_leq_u8(5)? as u32;
    if rice_param == 0b11111 {
        return decode_unencoded_partition_with(input, buffer, map);
    }
    input.decode_rice_partition_into(rice_param, buffer, |value| map(rice_to_signed(value)))?;
    Ok(())
}

/// Decode a Rice escape partition, whose residuals are stored directly as
/// signed two's-complement integers of a shared width.
fn decode_unencoded_partition<S: BitSource>(input: &mut S, buffer: &mut [i32]) -> Result<()> {
    let raw_bits = input.read_leq_u8(5)? as u32;
    if raw_bits == 0 {
        buffer.fill(0);
    } else {
        for sample in buffer.iter_mut() {
            *sample = extend_sign_u32(input.read_leq_u32(raw_bits)?, raw_bits);
        }
    }
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn decode_unencoded_partition_with<S: BitSource, F: FnMut(i32) -> i32>(
    input: &mut S,
    buffer: &mut [i32],
    map: &mut F,
) -> Result<()> {
    let raw_bits = input.read_leq_u8(5)? as u32;
    if raw_bits == 0 {
        for sample in buffer {
            *sample = map(0);
        }
    } else {
        for sample in buffer {
            *sample = map(extend_sign_u32(input.read_leq_u32(raw_bits)?, raw_bits));
        }
    }
    Ok(())
}

#[test]
fn verify_decode_escaped_rice_partitions() {
    use crate::decode::bitstream::BufferedReader;
    use std::io;

    fn push_bits(bits: &mut Vec<bool>, value: u32, width: u32) {
        for bit in (0..width).rev() {
            bits.push(value & (1 << bit) != 0);
        }
    }

    for &rice2 in &[false, true] {
        for &(raw_bits, ref values, expected) in &[
            (0u32, vec![0u32, 0, 0, 0], [0i32, 0, 0, 0]),
            (1, vec![0, 1, 0, 1], [0, -1, 0, -1]),
            (5, vec![0, 15, 16, 31], [0, 15, -16, -1]),
            (
                31,
                vec![0, 0x3fff_ffff, 0x4000_0000, 0x7fff_ffff],
                [0, 0x3fff_ffff, -0x4000_0000, -1],
            ),
        ] {
            let parameter_bits = if rice2 { 5 } else { 4 };
            let mut bits = Vec::new();
            push_bits(&mut bits, (1 << parameter_bits) - 1, parameter_bits);
            push_bits(&mut bits, raw_bits, 5);
            for &value in values {
                push_bits(&mut bits, value, raw_bits);
            }

            let mut encoded = vec![0u8; (bits.len() + 7) / 8];
            for (index, &bit) in bits.iter().enumerate() {
                if bit {
                    encoded[index / 8] |= 1 << (7 - index % 8);
                }
            }

            let reader = BufferedReader::new(io::Cursor::new(encoded));
            let mut input = Bitstream::new(reader);
            let mut decoded = [99i32; 4];
            if rice2 {
                decode_rice2_partition(&mut input, &mut decoded).unwrap();
            } else {
                decode_rice_partition(&mut input, &mut decoded).unwrap();
            }
            assert_eq!(decoded, expected);
        }
    }
}

#[test]
fn verify_decode_rice2_partition_across_reader_widths() {
    use crate::decode::bitstream::BufferedReader;
    use std::io;

    for &rice_param in &[0u32, 8, 9, 16, 17, 30] {
        let unsigned = [0u32, 1, 2, (1u32 << rice_param) | 3];
        let mut bits = Vec::new();

        for bit in (0..5).rev() {
            bits.push(rice_param & (1 << bit) != 0);
        }
        for &value in &unsigned {
            let quotient = value >> rice_param;
            for _ in 0..quotient {
                bits.push(false);
            }
            bits.push(true);
            for bit in (0..rice_param).rev() {
                bits.push(value & (1 << bit) != 0);
            }
        }

        let mut encoded = vec![0u8; (bits.len() + 7) / 8];
        for (index, &bit) in bits.iter().enumerate() {
            if bit {
                encoded[index / 8] |= 1 << (7 - index % 8);
            }
        }

        let reader = BufferedReader::new(io::Cursor::new(encoded));
        let mut input = Bitstream::new(reader);
        let mut decoded = [0i32; 4];
        decode_rice2_partition(&mut input, &mut decoded).unwrap();

        assert_eq!(
            decoded,
            [
                rice_to_signed(unsigned[0]),
                rice_to_signed(unsigned[1]),
                rice_to_signed(unsigned[2]),
                rice_to_signed(unsigned[3]),
            ]
        );
    }
}

fn decode_constant<S: BitSource>(input: &mut S, bps: u32, buffer: &mut [i32]) -> Result<()> {
    let sample_u32 = (input.read_leq_u32(bps))?;
    let sample = extend_sign_u32(sample_u32, bps);

    for s in buffer {
        *s = sample;
    }

    Ok(())
}

#[cold]
fn decode_verbatim<S: BitSource>(input: &mut S, bps: u32, buffer: &mut [i32]) -> Result<()> {
    // This function must not be called for a sample wider than the sample type.
    // This has been verified at an earlier stage, but it is good to state the
    // assumption explicitly. FLAC supports up to 32-bit samples, so the
    // mid/side delta would require 33 bits per sample. But that is not subset
    // FLAC, and the reference decoder does not support it either.
    debug_assert!(bps <= 32);

    // A verbatim block stores samples without encoding whatsoever.
    for s in buffer {
        *s = extend_sign_u32((input.read_leq_u32(bps))?, bps);
    }

    Ok(())
}

#[cfg(any(test, target_arch = "wasm32"))]
fn predict_fixed(order: u32, buffer: &mut [i32]) -> Result<()> {
    // When this is called during decoding, the order as read from the subframe
    // header has already been verified, so it is safe to assume that
    // 0 <= order <= 4. Still, it is good to state that assumption explicitly.
    debug_assert!(order <= 4);

    // Keep the previous samples in registers. The iterator-and-coefficient
    // form used by the generic decoder repeatedly rebuilt a sliding window;
    // these are the same fixed predictors used by FFmpeg and libFLAC, with
    // wrapping arithmetic for malformed streams.
    match order {
        0 => {}
        1 => {
            let mut x1 = buffer[0];
            for delta in &mut buffer[1..] {
                let value = x1.wrapping_add(*delta);
                *delta = value;
                x1 = value;
            }
        }
        2 => {
            let mut x2 = buffer[0];
            let mut x1 = buffer[1];
            for delta in &mut buffer[2..] {
                let prediction = x1.wrapping_mul(2).wrapping_sub(x2);
                let value = prediction.wrapping_add(*delta);
                *delta = value;
                x2 = x1;
                x1 = value;
            }
        }
        3 => {
            let mut x3 = buffer[0];
            let mut x2 = buffer[1];
            let mut x1 = buffer[2];
            for delta in &mut buffer[3..] {
                let prediction = x1
                    .wrapping_mul(3)
                    .wrapping_sub(x2.wrapping_mul(3))
                    .wrapping_add(x3);
                let value = prediction.wrapping_add(*delta);
                *delta = value;
                x3 = x2;
                x2 = x1;
                x1 = value;
            }
        }
        4 => {
            let mut x4 = buffer[0];
            let mut x3 = buffer[1];
            let mut x2 = buffer[2];
            let mut x1 = buffer[3];
            for delta in &mut buffer[4..] {
                let prediction = x1
                    .wrapping_mul(4)
                    .wrapping_sub(x2.wrapping_mul(6))
                    .wrapping_add(x3.wrapping_mul(4))
                    .wrapping_sub(x4);
                let value = prediction.wrapping_add(*delta);
                *delta = value;
                x4 = x3;
                x3 = x2;
                x2 = x1;
                x1 = value;
            }
        }
        _ => unreachable!(),
    }

    Ok(())
}

#[test]
fn verify_predict_fixed() {
    // The following data is from an actual FLAC stream and has been verified
    // against the reference decoder. The data is from a 16-bit stream.
    let mut buffer = [
        -729, -722, -667, -19, -16, 17, -23, -7, 16, -16, -5, 3, -8, -13, -15, -1,
    ];
    assert!(predict_fixed(3, &mut buffer).is_ok());
    assert_eq!(
        &buffer,
        &[-729, -722, -667, -583, -486, -359, -225, -91, 59, 209, 354, 497, 630, 740, 812, 845]
    );

    // The following data causes overflow of i32 when not handled with care.
    let mut buffer = [21877, 27482, -6513];
    assert!(predict_fixed(2, &mut buffer).is_ok());
    assert_eq!(&buffer, &[21877, 27482, 26574]);
}

fn decode_fixed<S: BitSource>(
    input: &mut S,
    bps: u32,
    order: u32,
    buffer: &mut [i32],
) -> Result<()> {
    // The length of the buffer which is passed in, is the length of the block.
    // Thus, the number of warm-up samples must not exceed that length.
    if buffer.len() < order as usize {
        return fmt_err("invalid fixed subframe, order is larger than block size");
    }

    // There are order * bits per sample unencoded warm-up sample bits.
    (decode_verbatim(input, bps, &mut buffer[..order as usize]))?;

    #[cfg(target_arch = "wasm32")]
    {
        // The compact two-pass form is faster in current Wasm engines and
        // avoids cloning the Rice decoder for all five predictor orders.
        decode_residual(input, buffer.len() as u16, &mut buffer[order as usize..])?;
        predict_fixed(order, buffer)?;
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        // Native code benefits from reconstructing each sample inside the
        // Rice mapping callback, avoiding a second memory pass over the block.
        let block_size = buffer.len() as u16;
        match order {
            0 => decode_residual_with(input, block_size, buffer, |delta| delta)?,
            1 => {
                let mut x1 = buffer[0];
                decode_residual_with(input, block_size, &mut buffer[1..], |delta| {
                    let value = x1.wrapping_add(delta);
                    x1 = value;
                    value
                })?;
            }
            2 => {
                let mut x2 = buffer[0];
                let mut x1 = buffer[1];
                decode_residual_with(input, block_size, &mut buffer[2..], |delta| {
                    let prediction = x1.wrapping_mul(2).wrapping_sub(x2);
                    let value = prediction.wrapping_add(delta);
                    x2 = x1;
                    x1 = value;
                    value
                })?;
            }
            3 => {
                let mut x3 = buffer[0];
                let mut x2 = buffer[1];
                let mut x1 = buffer[2];
                decode_residual_with(input, block_size, &mut buffer[3..], |delta| {
                    let prediction = x1
                        .wrapping_mul(3)
                        .wrapping_sub(x2.wrapping_mul(3))
                        .wrapping_add(x3);
                    let value = prediction.wrapping_add(delta);
                    x3 = x2;
                    x2 = x1;
                    x1 = value;
                    value
                })?;
            }
            4 => {
                let mut x4 = buffer[0];
                let mut x3 = buffer[1];
                let mut x2 = buffer[2];
                let mut x1 = buffer[3];
                decode_residual_with(input, block_size, &mut buffer[4..], |delta| {
                    let prediction = x1
                        .wrapping_mul(4)
                        .wrapping_sub(x2.wrapping_mul(6))
                        .wrapping_add(x3.wrapping_mul(4))
                        .wrapping_sub(x4);
                    let value = prediction.wrapping_add(delta);
                    x4 = x3;
                    x3 = x2;
                    x2 = x1;
                    x1 = value;
                    value
                })?;
            }
            _ => unreachable!(),
        }
    }

    Ok(())
}

/// Apply LPC prediction for subframes with LPC order of at most 12.
///
/// This function takes advantage of the upper bound on the order. Virtually all
/// files that occur in the wild are subset-compliant files, which have an order
/// of at most 12, so it makes sense to optimize for this. A simpler (but
/// slower) fallback is implemented in `predict_lpc_high_order`.
fn predict_lpc_low_order(raw_coefficients: &[i16], qlp_shift: i16, buffer: &mut [i32]) {
    debug_assert!(
        qlp_shift >= 0,
        "Right-shift by negative value is not allowed."
    );
    debug_assert!(qlp_shift < 64, "Cannot shift by more than integer width.");
    debug_assert!(
        raw_coefficients.len() <= 12,
        "predict_lpc_low_order called with order > 12",
    );

    // The inner reduction keeps a runtime trip count on purpose. With a
    // compile-time constant order the optimizer unrolls the tap loop and then
    // fuses the exposed product tree into vector code; the horizontal combine
    // of that vector lands exactly on this loop's serial output dependency and
    // costs far more than the multiplies it saves. A runtime bound keeps one
    // plain scalar chain per output sample.
    let order = raw_coefficients.len();
    let mut coefficients = [0i64; 12];
    for j in 0..order {
        coefficients[j] = raw_coefficients[j] as i64;
    }
    for i in order..buffer.len() {
        let mut sum = 0i64;
        for j in 0..order {
            sum += coefficients[j] * buffer[i - order + j] as i64;
        }
        buffer[i] = ((sum >> qlp_shift) + buffer[i] as i64) as i32;
    }
}

/// The per-order LPC restore kernel.
///
/// A plain fully-unrolled inner product over the sliding window. Keeping the
/// loop free of auxiliary arrays lets the optimizer hold the recent samples
/// in registers across iterations; reloading them from memory makes each
/// output wait on the store-to-load forward of the previous one, and that
/// dependency was the dominant decode cost.
#[allow(dead_code)] // Kept alongside the runtime-order variant for reference.
fn predict_lpc_taps<const ORDER: usize>(
    raw_coefficients: &[i16],
    qlp_shift: i16,
    buffer: &mut [i32],
) {
    let mut coefficients = [0i64; ORDER];
    for j in 0..ORDER {
        coefficients[j] = raw_coefficients[j] as i64;
    }
    for i in ORDER..buffer.len() {
        let mut sum = 0i64;
        for j in 0..ORDER {
            sum += coefficients[j] * buffer[i - ORDER + j] as i64;
        }
        buffer[i] = ((sum >> qlp_shift) + buffer[i] as i64) as i32;
    }
}

/// Apply LPC prediction for non-subset subframes, with LPC order > 12.
fn predict_lpc_high_order(coefficients: &[i16], qlp_shift: i16, buffer: &mut [i32]) {
    // NOTE: See `predict_lpc_low_order` for more details. This function is a
    // copy that lifts the order restrictions (and specializations) at the cost
    // of performance. It is only used for subframes with a high LPC order,
    // which only occur in non-subset files. Such files are rare in the wild.

    let order = coefficients.len();

    debug_assert!(
        qlp_shift >= 0,
        "Right-shift by negative value is not allowed."
    );
    debug_assert!(qlp_shift < 64, "Cannot shift by more than integer width.");
    debug_assert!(
        order > 12,
        "Use the faster predict_lpc_low_order for LPC order <= 12."
    );
    debug_assert!(
        buffer.len() >= order,
        "Buffer must fit at least `order` warm-up samples."
    );

    // The linear prediction is essentially an inner product of the known
    // samples with the coefficients, followed by a shift. The first `order`
    // samples are stored as-is.
    for i in order..buffer.len() {
        let prediction = coefficients
            .iter()
            .zip(&buffer[i - order..i])
            .map(|(&c, &s)| c as i64 * s as i64)
            .sum::<i64>()
            >> qlp_shift;
        let delta = buffer[i] as i64;
        buffer[i] = (prediction + delta) as i32;
    }
}

#[test]
fn verify_predict_lpc() {
    // The following data is from an actual FLAC stream and has been verified
    // against the reference decoder. The data is from a 16-bit stream.
    let coefficients = [-75, 166, 121, -269, -75, -399, 1042];
    let mut buffer = [
        -796, -547, -285, -32, 199, 443, 670, -2, -23, 14, 6, 3, -4, 12, -2, 10,
    ];
    predict_lpc_low_order(&coefficients, 9, &mut buffer);
    assert_eq!(
        &buffer,
        &[
            -796, -547, -285, -32, 199, 443, 670, 875, 1046, 1208, 1343, 1454, 1541, 1616, 1663,
            1701
        ]
    );

    // The following data causes an overflow when not handled with care.
    let coefficients = [119, -255, 555, -836, 879, -1199, 1757];
    let mut buffer = [-21363, -21951, -22649, -24364, -27297, -26870, -30017, 3157];
    predict_lpc_low_order(&coefficients, 10, &mut buffer);
    assert_eq!(
        &buffer,
        &[-21363, -21951, -22649, -24364, -27297, -26870, -30017, -29718]
    );

    // The following data from a real-world file has a high LPC order, is has
    // more than 12 coefficients. The excepted output has been verified against
    // the reference decoder.
    let coefficients = [
        709, -2589, 4600, -4612, 1350, 4220, -9743, 12671, -12129, 8586, -3775, -645, 3904, -5543,
        4373, 182, -6873, 13265, -15417, 11550,
    ];
    let mut buffer = [
        213238, 210830, 234493, 209515, 235139, 201836, 208151, 186277, 157720, 148176, 115037,
        104836, 60794, 54523, 412, 17943, -6025, -3713, 8373, 11764, 30094,
    ];
    predict_lpc_high_order(&coefficients, 12, &mut buffer);
    assert_eq!(
        &buffer,
        &[
            213238, 210830, 234493, 209515, 235139, 201836, 208151, 186277, 157720, 148176, 115037,
            104836, 60794, 54523, 412, 17943, -6025, -3713, 8373, 11764, 33931,
        ]
    );
}

fn decode_lpc<S: BitSource>(input: &mut S, bps: u32, order: u32, buffer: &mut [i32]) -> Result<()> {
    // The order minus one fits in 5 bits, so the order is at most 32.
    debug_assert!(order <= 32);

    // On the frame decoding level it is ensured that the buffer is large
    // enough. If it can't even fit the warm-up samples, then there is a frame
    // smaller than its lpc order, which is invalid.
    if buffer.len() < order as usize {
        return fmt_err("invalid LPC subframe, lpc order is larger than block size");
    }

    // There are order * bits per sample unencoded warm-up sample bits.
    (decode_verbatim(input, bps, &mut buffer[..order as usize]))?;

    // Next are four bits quantised linear predictor coefficient precision - 1.
    let qlp_precision = (input.read_leq_u8(4))? as u32 + 1;

    // The bit pattern 1111 is invalid.
    if qlp_precision - 1 == 0b1111 {
        return fmt_err("invalid subframe, qlp precision value invalid");
    }

    // Next are five bits quantized linear predictor coefficient shift,
    // in signed two's complement. Read 5 bits and then extend the sign bit.
    let qlp_shift_unsig = (input.read_leq_u16(5))?;
    let qlp_shift = extend_sign_u16(qlp_shift_unsig, 5);

    // The spec does allow the qlp shift to be negative, but in practice this
    // does not happen. Fully supporting it would be a performance hit, as an
    // arithmetic shift by a negative amount is invalid, so this would incur a
    // branch. If a real-world file ever hits this case, then we should consider
    // making two LPC predictors, one for positive, and one for negative qlp.
    if qlp_shift < 0 {
        let msg = "a negative quantized linear predictor coefficient shift is \
                   not supported, please file a bug.";
        return Err(Error::Unsupported(msg));
    }

    // Finally, the coefficients themselves. The order is at most 32, so all
    // coefficients can be kept on the stack. Store them in reverse, because
    // that how they are used in prediction.
    let mut coefficients = [0; 32];
    for coef in coefficients[..order as usize].iter_mut().rev() {
        // We can safely read into a u16, qlp_precision is at most 15.
        let coef_unsig = (input.read_leq_u16(qlp_precision))?;
        *coef = extend_sign_u16(coef_unsig, qlp_precision);
    }

    // Next up is the residual. We decode it into the buffer directly, the
    // predictor contributions will be added in a second pass. The first
    // `order` samples have been decoded already, so continue after that.
    (decode_residual(input, buffer.len() as u16, &mut buffer[order as usize..]))?;

    // In "subset"-compliant files, the LPC order is at most 12. For LPC
    // prediction of such files we have a special fast path that takes advantage
    // of the low order. We can still decode non-subset file using a less
    // specialized implementation. Non-subset files are rare in the wild.
    if order <= 12 {
        // Dispatch on the exact order so the tap loop fully unrolls and the
        // optimizer can hold the sliding window in registers.
        macro_rules! lpc_taps {
            ($($order:literal),+) => {
                match order {
                    $($order => predict_lpc_taps::<$order>(
                        &coefficients[..order as usize],
                        qlp_shift,
                        buffer,
                    ),)+
                    _ => predict_lpc_low_order(
                        &coefficients[..order as usize],
                        qlp_shift,
                        buffer,
                    ),
                }
            };
        }
        lpc_taps!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
    } else {
        predict_lpc_high_order(&coefficients[..order as usize], qlp_shift, buffer);
    }

    Ok(())
}
