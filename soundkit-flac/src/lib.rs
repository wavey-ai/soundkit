//! SoundKit adapters for the Wavey pure-Rust FLAC codec.

use soundkit::audio_packet::{Decoder as SoundKitDecoder, Encoder as SoundKitEncoder};
use wavey_flac::stream::{Decoder as WaveyDecoder, Encoder as WaveyEncoder};

pub use wavey_flac::frame::{
    DecodedFlacFrame, EncodedFlacFrame, FlacFrameConfig, FlacFrameDecoder, FlacFrameEncoder,
    FlacFrameError, FlacProfile,
};

/// SoundKit's streaming encoder contract backed by `wavey-flac`.
pub struct FlacEncoder {
    inner: WaveyEncoder,
    scratch: Vec<u8>,
}

impl FlacEncoder {
    fn profile(compression_level: u32) -> FlacProfile {
        match compression_level {
            0..=4 => FlacProfile::Realtime,
            5..=8 => FlacProfile::Balanced,
            _ => FlacProfile::Maximum,
        }
    }

    fn make_inner(
        sample_rate: u32,
        bits_per_sample: u32,
        channels: u32,
        frame_length: u32,
        compression_level: u32,
    ) -> Result<WaveyEncoder, String> {
        let channels =
            u16::try_from(channels).map_err(|_| format!("Channel count {channels} exceeds u16"))?;
        let bits_per_sample = u8::try_from(bits_per_sample)
            .map_err(|_| format!("Bits per sample {bits_per_sample} exceeds u8"))?;
        let config = FlacFrameConfig::new(
            sample_rate,
            channels,
            bits_per_sample,
            frame_length,
            Self::profile(compression_level),
        )
        .map_err(|error| error.to_string())?;
        WaveyEncoder::new(config).map_err(|error| error.to_string())
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

impl SoundKitEncoder for FlacEncoder {
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

/// SoundKit's incremental decoder contract backed by `wavey-flac`.
pub struct FlacDecoder {
    inner: WaveyDecoder,
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
            inner: WaveyDecoder::new(),
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

impl SoundKitDecoder for FlacDecoder {
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

/// Compatibility name for callers that previously selected the Claxon
/// backend directly. It now uses the unified Wavey codec.
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
