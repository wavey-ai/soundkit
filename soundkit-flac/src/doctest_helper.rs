// Copyright 2023-2024 Google LLC
// Copyright 2025- flacenc-rs developers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// A copy of the License has been included in the root of the repository.

#![allow(dead_code)]

// mimic clippy. This file acts as a part of the crate when checked by clippy.
// but it is outside of the crate when it is actually used by doctests.
#[cfg(clippy)]
use crate as soundkit_flac;

use soundkit_flac::component::{Frame, FrameHeader, Stream};
use soundkit_flac::config;
use soundkit_flac::encode_fixed_size_frame;
use soundkit_flac::error::Verify;
use soundkit_flac::source::{Fill, FrameBuf};

/// Makes a `Stream` for doctest.
pub fn make_example_stream(
    signal_len: usize,
    block_size: usize,
    channels: usize,
    sample_rate: usize,
) -> Stream {
    let signal = vec![0i32; signal_len * channels];
    let bits_per_sample = 16;
    let mut framebuf = FrameBuf::with_size(channels, block_size).expect("framebuf size error");
    let stream_info =
        config_stream(sample_rate, channels, bits_per_sample).expect("stream info error");
    let verified_config = config::Encoder::default()
        .into_verified()
        .expect("config value error");

    let mut stream = Stream::new(sample_rate, channels, bits_per_sample).expect("stream error");
    let mut offset = 0usize;
    let mut frame_number = 0usize;
    while offset < signal.len() {
        let end = (offset + block_size * channels).min(signal.len());
        Fill::fill_interleaved(&mut framebuf, &signal[offset..end]).expect("fill error");
        let frame = encode_fixed_size_frame(
            &verified_config,
            &framebuf,
            frame_number,
            &stream_info,
        )
        .expect("encoder error");
        stream.add_frame(frame);
        offset = end;
        frame_number += 1;
    }
    stream
}

fn config_stream(
    sample_rate: usize,
    channels: usize,
    bits_per_sample: usize,
) -> Result<soundkit_flac::component::StreamInfo, soundkit_flac::error::VerifyError> {
    use soundkit_flac::component::StreamInfo;
    StreamInfo::new(sample_rate, channels, bits_per_sample)
}

/// Makes a `Frame` for doctest.
pub fn make_example_frame(
    signal_len: usize,
    block_size: usize,
    channels: usize,
    sample_rate: usize,
) -> Frame {
    make_example_stream(signal_len, block_size, channels, sample_rate)
        .frame(0)
        .unwrap()
        .clone()
}

/// Makes a `FrameHeader` for doctest.
pub fn make_example_frame_header(
    signal_len: usize,
    block_size: usize,
    channels: usize,
    sample_rate: usize,
) -> FrameHeader {
    make_example_frame(signal_len, block_size, channels, sample_rate)
        .header()
        .clone()
}
