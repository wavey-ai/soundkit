// Vorbis decoder written in Rust
//
// Copyright (c) 2019 est31 <MTest31@outlook.com>
// and contributors. All rights reserved.
// Licensed under MIT license, or Apache 2 license,
// at your option. Please see the LICENSE file
// attached to this source distribution for details.
// Modified by SoundKit in 2026 for direct and vectorized PCM output.

/*!
Traits for sample formats
*/

/// Trait for a packet of multiple samples
pub trait Samples {
    fn num_samples(&self) -> usize;
    fn truncate(&mut self, limit: usize);
    fn from_floats(floats: Vec<Vec<f32>>) -> Self;
}

impl<S: Sample> Samples for Vec<Vec<S>> {
    fn num_samples(&self) -> usize {
        self[0].len()
    }
    fn truncate(&mut self, limit: usize) {
        for ch in self.iter_mut() {
            if limit < ch.len() {
                ch.truncate(limit);
            }
        }
    }

    fn from_floats(floats: Vec<Vec<f32>>) -> Self {
        floats
            .into_iter()
            .map(|samples| samples.into_iter().map(S::from_float).collect())
            .collect()
    }
}

/// A packet of multi-channel interleaved samples
pub struct InterleavedSamples<S: Sample> {
    pub samples: Vec<S>,
    pub channel_count: usize,
}

/// Interleaved signed 16-bit little-endian PCM produced directly by the core.
///
/// Keeping the byte representation here avoids a second full-buffer pass in
/// the streaming facade before constructing `AudioData`.
pub struct InterleavedPcm16 {
    pub bytes: Vec<u8>,
    pub channel_count: usize,
}

impl Samples for InterleavedPcm16 {
    fn num_samples(&self) -> usize {
        self.bytes.len() / (self.channel_count * 2)
    }

    fn truncate(&mut self, limit: usize) {
        self.bytes.truncate(limit * self.channel_count * 2);
    }

    fn from_floats(floats: Vec<Vec<f32>>) -> Self {
        let channel_count = floats.len();
        assert!(channel_count > 0);
        let frame_count = floats[0].len();
        let mut bytes = vec![0; frame_count * channel_count * 2];
        if channel_count == 2 {
            convert_stereo_pcm16(&floats[0], &floats[1], &mut bytes);
        } else {
            for frame in 0..frame_count {
                for (channel_index, channel) in floats.iter().enumerate() {
                    write_pcm16(
                        &mut bytes,
                        (frame * channel_count + channel_index) * 2,
                        channel[frame],
                    );
                }
            }
        }
        Self {
            bytes,
            channel_count,
        }
    }
}

#[inline]
fn convert_stereo_pcm16(left: &[f32], right: &[f32], bytes: &mut [u8]) {
    debug_assert_eq!(left.len(), right.len());
    debug_assert_eq!(bytes.len(), left.len() * 4);
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let converted = if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: Runtime detection guarantees AVX2 support and the helper
        // bounds every eight-frame load and store.
        unsafe { convert_stereo_pcm16_avx2(left, right, bytes) }
    } else {
        0
    };
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    let converted = 0usize;
    for frame in converted..left.len() {
        write_pcm16(bytes, frame * 4, left[frame]);
        write_pcm16(bytes, frame * 4 + 2, right[frame]);
    }
}

#[inline(always)]
fn write_pcm16(bytes: &mut [u8], offset: usize, sample: f32) {
    let scaled = sample * 32768.0;
    let sample = if scaled > 32767.0 {
        32767i16
    } else if scaled < -32768.0 {
        -32768i16
    } else {
        scaled as i16
    };
    let encoded = sample.to_le_bytes();
    bytes[offset] = encoded[0];
    bytes[offset + 1] = encoded[1];
}

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn convert_stereo_pcm16_avx2(left: &[f32], right: &[f32], bytes: &mut [u8]) -> usize {
    let frame_count = left.len() & !7;
    let scale = _mm256_set1_ps(32768.0);
    let minimum = _mm256_set1_ps(-32768.0);
    let maximum = _mm256_set1_ps(32767.0);
    let mut frame = 0usize;
    while frame < frame_count {
        // SAFETY: `frame_count` is rounded down to eight frames and the
        // destination contains four bytes for every input frame.
        unsafe {
            let left_samples = _mm256_min_ps(
                maximum,
                _mm256_max_ps(
                    minimum,
                    _mm256_mul_ps(_mm256_loadu_ps(left.as_ptr().add(frame)), scale),
                ),
            );
            let right_samples = _mm256_min_ps(
                maximum,
                _mm256_max_ps(
                    minimum,
                    _mm256_mul_ps(_mm256_loadu_ps(right.as_ptr().add(frame)), scale),
                ),
            );
            let left_i32 = _mm256_cvttps_epi32(left_samples);
            let right_i32 = _mm256_cvttps_epi32(right_samples);
            let low = _mm256_unpacklo_epi32(left_i32, right_i32);
            let high = _mm256_unpackhi_epi32(left_i32, right_i32);
            let interleaved = _mm256_packs_epi32(low, high);
            _mm256_storeu_si256(
                bytes.as_mut_ptr().add(frame * 4).cast::<__m256i>(),
                interleaved,
            );
        }
        frame += 8;
    }
    frame_count
}

impl<S: Sample> Samples for InterleavedSamples<S> {
    fn num_samples(&self) -> usize {
        self.samples.len() / self.channel_count
    }
    fn truncate(&mut self, limit: usize) {
        self.samples.truncate(limit * self.channel_count);
    }
    fn from_floats(floats: Vec<Vec<f32>>) -> Self {
        let channel_count = floats.len();
        // Note that a channel count of 0 is forbidden
        // by the spec and the header decoding code already
        // checks for that.
        assert!(floats.len() > 0);
        let samples_interleaved = if channel_count == 1 {
            // Because decoded_pck[0] doesn't work...
            <Vec<Vec<S>> as Samples>::from_floats(floats)
                .into_iter()
                .next()
                .unwrap()
        } else {
            let len = floats[0].len();
            let mut samples = Vec::with_capacity(len * channel_count);
            for i in 0..len {
                for ref chan in floats.iter() {
                    samples.push(S::from_float(chan[i]));
                }
            }
            samples
        };
        Self {
            samples: samples_interleaved,
            channel_count,
        }
    }
}

/// Trait representing a single sample
pub trait Sample {
    fn from_float(fl: f32) -> Self;
}

impl Sample for f32 {
    fn from_float(fl: f32) -> Self {
        fl
    }
}

impl Sample for i16 {
    fn from_float(fl: f32) -> Self {
        let fl = fl * 32768.0;
        if fl > 32767. {
            32767
        } else if fl < -32768. {
            -32768
        } else {
            fl as i16
        }
    }
}
