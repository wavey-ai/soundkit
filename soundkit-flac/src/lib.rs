//! Pure-Rust FLAC encoding and decoding for SoundKit.
//!
//! [`FlacFrameEncoder`] and [`FlacFrameDecoder`] are the primary low-latency
//! API. Each call encodes or decodes one raw FLAC frame, with stream geometry
//! carried out of band by the surrounding transport. The hot path is tuned for
//! 5 ms frames at 48 and 96 kHz.
//!
//! The SoundKit [`FlacEncoder`] adapter follows the packet contract and emits
//! raw frames. The [`stream`] module and [`FlacDecoder`] retain explicit native
//! stream compatibility for file import/export.

#![cfg_attr(feature = "simd-nightly", feature(portable_simd))]
#![cfg_attr(all(test, feature = "simd-nightly"), feature(test))]

use soundkit::audio_packet::{Decoder as PacketDecoder, Encoder as PacketEncoder};

/// Expands import statements for `fakesimd` or `std::simd`.
macro_rules! import_simd {
    (as $modalias:ident) => {
        #[cfg(feature = "simd-nightly")]
        use std::simd as $modalias;
        #[cfg(not(feature = "simd-nightly"))]
        use $crate::fakesimd as $modalias;

        #[allow(unused_imports)]
        use simd::prelude::*;

        #[allow(unused_imports)]
        use simd::StdFloat;
    };
}

/// Sets up the thread-local re-usable storage for avoiding reallocation.
///
/// This provides a short-cut for the common pattern using [`thread_local!`]
/// and [`RefCell`].
macro_rules! reusable {
    ($key:ident: $t:ty) => {
        thread_local! {
            static $key: std::cell::RefCell<$t> = std::cell::RefCell::new(Default::default());
        }
    };
    ($key:ident: $t:ty = $init:expr) => {
        thread_local! {
            static $key: std::cell::RefCell<$t> = std::cell::RefCell::new($init);
        }
    };
}

/// Macro used when using a storage declared using [`reusable!`].
macro_rules! reuse {
    ($key:ident, $fn:expr) => {{
        #[allow(clippy::redundant_closure_call)]
        $key.with(|cell| $fn(&mut cell.borrow_mut()))
    }};
}

#[cfg(not(feature = "simd-nightly"))]
mod fakesimd;

mod arrayutils;
pub mod bitsink;
mod coding;
pub mod component;
pub mod config;
pub mod constant;
pub mod crc;
pub mod decode;
pub mod error;
pub mod frame;
mod lpc;
mod packet;
mod repeat;
pub mod rice;
pub mod source;
pub mod stream;

pub use coding::encode_fixed_size_frame;
pub use frame::{
    DecodedFlacFrame, EncodedFlacFrame, FlacFrameConfig, FlacFrameDecoder, FlacFrameEncoder,
    FlacFrameError, FlacProfile,
};
pub use stream::{Decoder, Encoder};

fn profile_for_compression_level(compression_level: u32) -> FlacProfile {
    match compression_level {
        0 => FlacProfile::Realtime,
        1..=8 => FlacProfile::Balanced,
        _ => FlacProfile::Maximum,
    }
}

/// SoundKit's packet encoder contract backed by the raw frame codec.
pub struct FlacEncoder {
    inner: FlacFrameEncoder,
    scratch: Vec<u8>,
}

impl FlacEncoder {
    fn make_inner(
        sample_rate: u32,
        bits_per_sample: u32,
        channels: u32,
        frame_length: u32,
        compression_level: u32,
    ) -> Result<FlacFrameEncoder, String> {
        let channels =
            u16::try_from(channels).map_err(|_| format!("Channel count {channels} exceeds u16"))?;
        let bits_per_sample = u8::try_from(bits_per_sample)
            .map_err(|_| format!("Bits per sample {bits_per_sample} exceeds u8"))?;
        let config = FlacFrameConfig::new(
            sample_rate,
            channels,
            bits_per_sample,
            frame_length,
            profile_for_compression_level(compression_level),
        )
        .map_err(|error| error.to_string())?;
        FlacFrameEncoder::new(config).map_err(|error| error.to_string())
    }

    fn copy_encoded(&self, output: &mut [u8]) -> Result<usize, String> {
        if output.len() < self.scratch.len() {
            return Err(format!(
                "Output buffer of len {} too small for FLAC packet of len {}",
                output.len(),
                self.scratch.len()
            ));
        }
        output[..self.scratch.len()].copy_from_slice(&self.scratch);
        Ok(self.scratch.len())
    }

    /// Raw packet encoding has no buffered tail or stream metadata to finish.
    pub fn finish(&mut self, _output: &mut [u8]) -> Result<usize, String> {
        Ok(0)
    }
}

impl PacketEncoder for FlacEncoder {
    fn new(
        sample_rate: u32,
        bits_per_sample: u32,
        channels: u32,
        frame_length: u32,
        compression_level: u32,
    ) -> Self {
        let inner = Self::make_inner(
            sample_rate,
            bits_per_sample,
            channels,
            frame_length,
            compression_level,
        )
        .unwrap_or_else(|error| panic!("{error}"));
        Self {
            inner,
            scratch: Vec::new(),
        }
    }

    fn init(&mut self) -> Result<(), String> {
        self.reset()
    }

    fn encode_i16(&mut self, input: &[i16], output: &mut [u8]) -> Result<usize, String> {
        self.scratch.clear();
        self.inner
            .encode_i16_into(input, &mut self.scratch)
            .map_err(|error| error.to_string())?;
        self.copy_encoded(output)
    }

    fn encode_i32(&mut self, input: &[i32], output: &mut [u8]) -> Result<usize, String> {
        self.scratch.clear();
        self.inner
            .encode_i32_block_into(input, &mut self.scratch)
            .map_err(|error| error.to_string())?;
        self.copy_encoded(output)
    }

    fn reset(&mut self) -> Result<(), String> {
        self.scratch.clear();
        self.inner.reset();
        Ok(())
    }
}

/// SoundKit's incremental decoder contract backed by the vendored FLAC codec.
pub struct FlacDecoder {
    inner: stream::Decoder,
    scratch: Vec<i32>,
}

impl Default for FlacDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl FlacDecoder {
    pub fn new() -> Self {
        Self {
            inner: stream::Decoder::new(),
            scratch: Vec::new(),
        }
    }

    pub fn init(&mut self) -> Result<(), String> {
        self.reset();
        Ok(())
    }

    pub fn reset(&mut self) {
        self.inner.reset();
        self.scratch.clear();
    }

    pub fn finish(&self) -> Result<(), String> {
        self.inner.finish().map_err(|error| error.to_string())
    }

    pub fn sample_rate(&self) -> Option<u32> {
        self.inner.stream_info().map(|info| info.sample_rate)
    }

    pub fn channels(&self) -> Option<u8> {
        self.inner
            .stream_info()
            .and_then(|info| u8::try_from(info.channels).ok())
    }

    pub fn bits_per_sample(&self) -> Option<u8> {
        self.inner
            .stream_info()
            .and_then(|info| u8::try_from(info.bits_per_sample).ok())
    }

    pub fn minimum_block_size(&self) -> Option<u16> {
        self.inner.stream_info().map(|info| info.min_block_size)
    }

    pub fn maximum_block_size(&self) -> Option<u16> {
        self.inner.stream_info().map(|info| info.max_block_size)
    }

    pub fn buffered_bytes(&self) -> usize {
        self.inner.buffered_bytes()
    }

    pub fn pending_samples(&self) -> usize {
        self.inner.pending_samples()
    }
}

impl PacketDecoder for FlacDecoder {
    /// Renders the stream as 16-bit, whatever depth it was written at.
    ///
    /// A sample wider than 16 bits is shifted down by the difference, the
    /// way `decode_f32` divides by the stream's own full scale. Clamping
    /// instead — which this did — does not narrow a 24-bit sample, it
    /// saturates it: a quiet track sits around ±1,000,000 at that depth and
    /// every sample of it came back at ±32,767, a full-scale square wave
    /// whatever went in.
    fn decode_i16(
        &mut self,
        input: &[u8],
        output: &mut [i16],
        _fec: bool,
    ) -> Result<usize, String> {
        self.scratch.resize(output.len(), 0);
        let written = self
            .inner
            .decode_i32(input, &mut self.scratch)
            .map_err(|error| error.to_string())?;
        let shift = u32::from(self.bits_per_sample().unwrap_or(16).saturating_sub(16));
        for (target, &sample) in output.iter_mut().zip(&self.scratch[..written]) {
            *target = (sample >> shift)
                .clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
        }
        Ok(written)
    }

    fn decode_i32(
        &mut self,
        input: &[u8],
        output: &mut [i32],
        _fec: bool,
    ) -> Result<usize, String> {
        self.inner
            .decode_i32(input, output)
            .map_err(|error| error.to_string())
    }

    fn decode_f32(
        &mut self,
        input: &[u8],
        output: &mut [f32],
        _fec: bool,
    ) -> Result<usize, String> {
        self.inner
            .decode_f32(input, output)
            .map_err(|error| error.to_string())
    }
}

/// Legacy name for callers that selected a backend explicitly. There is one
/// codec now; this alias remains for source compatibility.
pub type FlacDecoderClaxon = FlacDecoder;

#[cfg(test)]
mod tests {
    use super::{FlacEncoder, FlacFrameConfig, FlacFrameDecoder, FlacProfile};
    use soundkit::audio_packet::Encoder;

    #[test]
    fn soundkit_adapter_round_trips_a_short_final_block() {
        let channels = 2usize;
        let frame_length = 128usize;
        let samples = (0..(frame_length * 2 + 64) * channels)
            .map(|index| ((index as i32 * 977) % 65_536) - 32_768)
            .collect::<Vec<_>>();
        let mut encoder = FlacEncoder::new(48_000, 16, channels as u32, frame_length as u32, 5);
        encoder.init().unwrap();

        let mut packets = Vec::new();
        for chunk in samples.chunks(frame_length * channels) {
            let mut output = vec![0_u8; chunk.len() * 8 + 4_096];
            let encoded = encoder.encode_i32(chunk, &mut output).unwrap();
            output.truncate(encoded);
            packets.push(output);
        }
        assert_eq!(encoder.finish(&mut []).unwrap(), 0);

        assert!(packets
            .iter()
            .all(|packet| packet.starts_with(&[0xff, 0xf8])));
        let config = FlacFrameConfig::new(
            48_000,
            channels as u16,
            16,
            frame_length as u32,
            FlacProfile::Balanced,
        )
        .unwrap();
        let mut decoder = FlacFrameDecoder::new(config).unwrap();
        let mut decoded = Vec::with_capacity(samples.len());
        for packet in &packets {
            let mut block = vec![0_i32; frame_length * channels];
            let written = decoder.decode_i32_block_into(packet, &mut block).unwrap();
            decoded.extend_from_slice(&block[..written]);
        }
        assert_eq!(decoded, samples);
    }
}

#[cfg(test)]
mod decode_width_tests {
    use super::*;
    use crate::frame::{FlacFrameConfig, FlacProfile};
    use crate::stream::Encoder;

    /// A complete FLAC file — marker, metadata, frames — at one depth.
    fn encode_file(bits: u8, samples: &[i32]) -> Vec<u8> {
        let config = FlacFrameConfig::new(48_000, 2, bits, 128, FlacProfile::Balanced).unwrap();
        let mut encoder = Encoder::new(config).unwrap();
        let mut packets = Vec::new();
        for chunk in samples.chunks(config.sample_count().unwrap()) {
            encoder.encode_i32(chunk, &mut packets).unwrap();
        }
        let header = encoder.finish().unwrap().to_vec();
        let metadata_len = header.len();
        let mut file = b"fLaC".to_vec();
        file.extend_from_slice(&header);
        file.extend_from_slice(&packets[metadata_len..]);
        file
    }

    /// Rendering a stream as 16-bit narrows it; it does not saturate it.
    ///
    /// A 24-bit stream carries samples far outside `i16`, and reducing them
    /// by clamping — which this did — returned full scale for every one of
    /// them: a square wave, the same whatever the audio was.
    #[test]
    fn a_wider_stream_renders_as_16_bit_at_its_own_level() {
        for bits in [16u8, 24] {
            let full = 1i64 << (bits - 1);
            // A sixth of full scale, so saturation is unmistakable.
            let amplitude = (full / 6) as f64;
            let samples: Vec<i32> = (0..640)
                .map(|index| ((index as f64 / 30.0).sin() * amplitude) as i32)
                .collect();
            let file = encode_file(bits, &samples);

            let mut decoder = FlacDecoder::new();
            decoder.init().unwrap();
            let mut out = vec![0i16; 1 << 12];
            let mut decoded: Vec<i16> = Vec::new();
            for chunk in file.chunks(64) {
                let written = decoder.decode_i16(chunk, &mut out, false).unwrap();
                decoded.extend_from_slice(&out[..written]);
                loop {
                    let written = decoder.decode_i16(&[], &mut out, false).unwrap();
                    if written == 0 {
                        break;
                    }
                    decoded.extend_from_slice(&out[..written]);
                }
            }

            let shift = u32::from(bits.saturating_sub(16));
            let peak = decoded.iter().map(|value| i32::from(*value).abs()).max().unwrap_or(0);
            let expected = samples.iter().map(|value| (value >> shift).abs()).max().unwrap_or(0);
            assert_eq!(
                decoded.len(),
                samples.len(),
                "{bits}-bit: got {} samples of {}",
                decoded.len(),
                samples.len()
            );
            assert!(
                (peak - expected).abs() <= 1,
                "{bits}-bit: went in peaking at {expected}, came back at {peak}"
            );
        }
    }
}
