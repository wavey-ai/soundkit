//! Pure-Rust FLAC encoding and decoding for SoundKit.
//!
//! This crate vendors a complete FLAC codec and adapts it to SoundKit's
//! audio-packet contracts. The codec is independently framed: each packet is
//! one raw FLAC frame, and stream geometry travels out of band through
//! [`FlacEncoder::stream_header`] so containers can backpatch `STREAMINFO`
//! after the final frame.

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
mod coding;
pub mod bitsink;
pub mod component;
pub mod config;
pub mod constant;
pub mod crc;
pub mod decode;
pub mod error;
pub mod frame;
mod lpc;
pub mod rice;
pub mod source;
pub mod stream;
mod repeat;

pub use coding::encode_fixed_size_frame;
pub use frame::{
    DecodedFlacFrame, EncodedFlacFrame, FlacFrameConfig, FlacFrameDecoder, FlacFrameEncoder,
    FlacFrameError, FlacProfile,
};
pub use stream::{Decoder, Encoder};

fn profile_for_compression_level(compression_level: u32) -> FlacProfile {
    match compression_level {
        0..=4 => FlacProfile::Realtime,
        5..=8 => FlacProfile::Balanced,
        _ => FlacProfile::Maximum,
    }
}

/// SoundKit's streaming encoder contract backed by the vendored FLAC codec.
pub struct FlacEncoder {
    inner: stream::Encoder,
    scratch: Vec<u8>,
}

impl FlacEncoder {
    fn make_inner(
        sample_rate: u32,
        bits_per_sample: u32,
        channels: u32,
        frame_length: u32,
        compression_level: u32,
    ) -> Result<stream::Encoder, String> {
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
        stream::Encoder::new(config).map_err(|error| error.to_string())
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

    /// Finalizes STREAMINFO. FLAC frames are emitted immediately, so no audio
    /// bytes are written during finalization.
    pub fn finish(&mut self, _output: &mut [u8]) -> Result<usize, String> {
        self.inner.finish().map_err(|error| error.to_string())?;
        Ok(0)
    }

    /// Returns the current STREAMINFO metadata without the `fLaC` marker.
    pub fn stream_header(&self) -> &[u8] {
        self.inner.stream_header()
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
            .encode_i16(input, &mut self.scratch)
            .map_err(|error| error.to_string())?;
        self.copy_encoded(output)
    }

    fn encode_i32(&mut self, input: &[i32], output: &mut [u8]) -> Result<usize, String> {
        self.scratch.clear();
        self.inner
            .encode_i32(input, &mut self.scratch)
            .map_err(|error| error.to_string())?;
        self.copy_encoded(output)
    }

    fn reset(&mut self) -> Result<(), String> {
        self.scratch.clear();
        self.inner.reset().map_err(|error| error.to_string())
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

    pub fn buffered_bytes(&self) -> usize {
        self.inner.buffered_bytes()
    }

    pub fn pending_samples(&self) -> usize {
        self.inner.pending_samples()
    }
}

impl PacketDecoder for FlacDecoder {
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
        for (target, &sample) in output.iter_mut().zip(&self.scratch[..written]) {
            *target = sample.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
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
    use super::{FlacDecoder, FlacEncoder};
    use soundkit::audio_packet::{Decoder, Encoder};

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

        let header = encoder.stream_header();
        let mut file = b"fLaC".to_vec();
        file.extend_from_slice(header);
        file.extend_from_slice(&packets[0][header.len()..]);
        for packet in &packets[1..] {
            file.extend_from_slice(packet);
        }

        let mut decoder = FlacDecoder::new();
        decoder.init().unwrap();
        let mut decoded = vec![0_i32; samples.len()];
        let written = decoder.decode_i32(&file, &mut decoded, false).unwrap();
        assert_eq!(written, samples.len());
        assert_eq!(decoded, samples);
        decoder.finish().unwrap();
        assert_eq!(decoder.sample_rate(), Some(48_000));
        assert_eq!(decoder.channels(), Some(2));
        assert_eq!(decoder.bits_per_sample(), Some(16));
    }
}
