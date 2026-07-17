//! Pure-Rust, independently framed FLAC for latency-sensitive transports.
//!
//! This module deliberately keeps codec configuration out of the compressed
//! packet. The surrounding transport already carries sample rate, channel
//! count, sample format and frame count, so each packet can remain one raw
//! FLAC frame. A decoder is configured once per track and can decode every
//! subsequent packet independently.

use flacenc::bitsink::ByteSink;
use flacenc::component::{BitRepr, Stream, StreamInfo};
use flacenc::config;
use flacenc::error::{Verified, Verify};
use flacenc::source::{Fill, FrameBuf};
use oxideav_core::{
    CodecId, CodecParameters, Decoder as OxideDecoder, Frame as OxideFrame, Packet, SampleFormat,
    TimeBase,
};
use std::error::Error;
use std::fmt;

const FLAC_MAX_CHANNELS: u16 = 8;
const FLAC_MIN_BLOCK_SIZE: u32 = 32;
const FLAC_MAX_BLOCK_SIZE: u32 = 65_535;
const FLAC_MAX_SAMPLE_RATE: u32 = (1 << 20) - 1;
const MAX_PACKET_OVERHEAD_BYTES: usize = 4_096;
const MAX_PACKET_EXPANSION_RATIO: usize = 8;

/// Encoding effort for one independently framed packet.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FlacProfile {
    /// Bounded predictor search intended for live per-track encoding.
    #[default]
    Realtime,
    /// Evaluate the normal fixed predictors but skip the heavier LPC search.
    Balanced,
    /// Use the upstream encoder's complete default search.
    Maximum,
}

/// Immutable format contract shared by one track encoder and decoder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlacFrameConfig {
    pub sample_rate: u32,
    pub channels: u16,
    pub bits_per_sample: u8,
    pub frame_length: u32,
    pub profile: FlacProfile,
}

impl FlacFrameConfig {
    pub fn new(
        sample_rate: u32,
        channels: u16,
        bits_per_sample: u8,
        frame_length: u32,
        profile: FlacProfile,
    ) -> Result<Self, FlacFrameError> {
        let config = Self {
            sample_rate,
            channels,
            bits_per_sample,
            frame_length,
            profile,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), FlacFrameError> {
        if !(1..=FLAC_MAX_SAMPLE_RATE).contains(&self.sample_rate) {
            return Err(FlacFrameError::InvalidConfig(format!(
                "sample rate {} is outside FLAC's 1..={FLAC_MAX_SAMPLE_RATE} Hz range",
                self.sample_rate
            )));
        }
        if !(1..=FLAC_MAX_CHANNELS).contains(&self.channels) {
            return Err(FlacFrameError::InvalidConfig(format!(
                "channel count {} is outside the supported 1..={FLAC_MAX_CHANNELS} range",
                self.channels
            )));
        }
        if !matches!(self.bits_per_sample, 16 | 24) {
            return Err(FlacFrameError::InvalidConfig(format!(
                "only signed 16-bit and packed signed 24-bit PCM are supported, got {} bits",
                self.bits_per_sample
            )));
        }
        if !(FLAC_MIN_BLOCK_SIZE..=FLAC_MAX_BLOCK_SIZE).contains(&self.frame_length) {
            return Err(FlacFrameError::InvalidConfig(format!(
                "frame length {} is outside FLAC's {FLAC_MIN_BLOCK_SIZE}..={FLAC_MAX_BLOCK_SIZE} sample range",
                self.frame_length
            )));
        }
        self.sample_count()?;
        self.raw_pcm_bytes()?;
        Ok(())
    }

    pub fn sample_count(&self) -> Result<usize, FlacFrameError> {
        usize::try_from(self.frame_length)
            .ok()
            .and_then(|frames| frames.checked_mul(usize::from(self.channels)))
            .ok_or(FlacFrameError::Overflow("FLAC sample count"))
    }

    pub fn raw_pcm_bytes(&self) -> Result<usize, FlacFrameError> {
        self.sample_count()?
            .checked_mul(usize::from(self.bits_per_sample / 8))
            .ok_or(FlacFrameError::Overflow("FLAC PCM byte count"))
    }

    fn maximum_packet_bytes(&self) -> Result<usize, FlacFrameError> {
        self.raw_pcm_bytes()?
            .checked_mul(MAX_PACKET_EXPANSION_RATIO)
            .and_then(|bytes| bytes.checked_add(MAX_PACKET_OVERHEAD_BYTES))
            .ok_or(FlacFrameError::Overflow("FLAC packet size limit"))
    }

    fn sample_format(&self) -> SampleFormat {
        match self.bits_per_sample {
            16 => SampleFormat::S16,
            24 => SampleFormat::S24,
            _ => unreachable!("validated by FlacFrameConfig"),
        }
    }
}

/// Metadata and payload produced for one input frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedFlacFrame {
    pub sequence: u32,
    pub sample_rate: u32,
    pub channels: u16,
    pub bits_per_sample: u8,
    pub frame_count: u32,
    pub pcm_bytes: usize,
    pub payload: Vec<u8>,
}

impl EncodedFlacFrame {
    pub fn encoded_bytes(&self) -> usize {
        self.payload.len()
    }

    pub fn compression_ratio(&self) -> f64 {
        if self.pcm_bytes == 0 {
            return 0.0;
        }
        self.payload.len() as f64 / self.pcm_bytes as f64
    }
}

/// Decoded interleaved samples and the format that was actually checked.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodedFlacFrame {
    pub sample_rate: u32,
    pub channels: u16,
    pub bits_per_sample: u8,
    pub frame_count: u32,
    pub samples: Vec<i32>,
}

impl DecodedFlacFrame {
    pub fn to_s24le(&self) -> Result<Vec<u8>, FlacFrameError> {
        if self.bits_per_sample != 24 {
            return Err(FlacFrameError::FormatMismatch(format!(
                "cannot render {}-bit FLAC as a declared S24 frame",
                self.bits_per_sample
            )));
        }
        let mut output = Vec::with_capacity(
            self.samples
                .len()
                .checked_mul(3)
                .ok_or(FlacFrameError::Overflow("decoded S24 byte count"))?,
        );
        for &sample in &self.samples {
            let clipped = sample.clamp(-8_388_608, 8_388_607);
            output.push((clipped & 0xff) as u8);
            output.push(((clipped >> 8) & 0xff) as u8);
            output.push(((clipped >> 16) & 0xff) as u8);
        }
        Ok(output)
    }

    pub fn to_i16(&self) -> Result<Vec<i16>, FlacFrameError> {
        if self.bits_per_sample != 16 {
            return Err(FlacFrameError::FormatMismatch(format!(
                "cannot render {}-bit FLAC as a declared S16 frame",
                self.bits_per_sample
            )));
        }
        Ok(self
            .samples
            .iter()
            .map(|&sample| sample.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16)
            .collect())
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum FlacFrameError {
    InvalidConfig(String),
    InvalidInput(String),
    FormatMismatch(String),
    Encode(String),
    Decode(String),
    Overflow(&'static str),
}

impl fmt::Display for FlacFrameError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(formatter, "invalid FLAC config: {message}"),
            Self::InvalidInput(message) => write!(formatter, "invalid FLAC input: {message}"),
            Self::FormatMismatch(message) => write!(formatter, "FLAC format mismatch: {message}"),
            Self::Encode(message) => write!(formatter, "FLAC encode failed: {message}"),
            Self::Decode(message) => write!(formatter, "FLAC decode failed: {message}"),
            Self::Overflow(context) => write!(formatter, "integer overflow computing {context}"),
        }
    }
}

impl Error for FlacFrameError {}

/// Persistent per-track pure-Rust encoder.
pub struct FlacFrameEncoder {
    config: FlacFrameConfig,
    encoder_config: Verified<config::Encoder>,
    stream_info: StreamInfo,
    frame_buffer: FrameBuf,
    converted_samples: Vec<i32>,
    next_sequence: u32,
}

impl fmt::Debug for FlacFrameEncoder {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FlacFrameEncoder")
            .field("config", &self.config)
            .field("next_sequence", &self.next_sequence)
            .finish_non_exhaustive()
    }
}

impl FlacFrameEncoder {
    pub fn new(config: FlacFrameConfig) -> Result<Self, FlacFrameError> {
        config.validate()?;
        let encoder_config = verified_encoder_config(config)?;
        let stream_info = StreamInfo::new(
            config.sample_rate as usize,
            usize::from(config.channels),
            usize::from(config.bits_per_sample),
        )
        .map_err(|error| FlacFrameError::InvalidConfig(error.to_string()))?;
        let frame_buffer =
            FrameBuf::with_size(usize::from(config.channels), config.frame_length as usize)
                .map_err(|error| FlacFrameError::InvalidConfig(error.to_string()))?;
        Ok(Self {
            config,
            encoder_config,
            stream_info,
            frame_buffer,
            converted_samples: Vec::with_capacity(config.sample_count()?),
            next_sequence: 0,
        })
    }

    pub fn config(&self) -> FlacFrameConfig {
        self.config
    }

    pub fn next_sequence(&self) -> u32 {
        self.next_sequence
    }

    /// Resets only this track's FLAC continuity segment.
    pub fn reset(&mut self) {
        self.next_sequence = 0;
        self.converted_samples.clear();
    }

    pub fn encode_i16(&mut self, interleaved: &[i16]) -> Result<EncodedFlacFrame, FlacFrameError> {
        if self.config.bits_per_sample != 16 {
            return Err(FlacFrameError::FormatMismatch(format!(
                "encoder is configured for {}-bit samples, not S16",
                self.config.bits_per_sample
            )));
        }
        validate_sample_len(self.config, interleaved.len())?;
        self.converted_samples.clear();
        self.converted_samples
            .extend(interleaved.iter().copied().map(i32::from));
        self.encode_converted()
    }

    pub fn encode_s24le(&mut self, interleaved: &[u8]) -> Result<EncodedFlacFrame, FlacFrameError> {
        if self.config.bits_per_sample != 24 {
            return Err(FlacFrameError::FormatMismatch(format!(
                "encoder is configured for {}-bit samples, not packed S24",
                self.config.bits_per_sample
            )));
        }
        let expected_bytes = self.config.raw_pcm_bytes()?;
        if interleaved.len() != expected_bytes {
            return Err(FlacFrameError::InvalidInput(format!(
                "packed S24 frame has {} bytes, expected {expected_bytes}",
                interleaved.len()
            )));
        }
        self.converted_samples.clear();
        self.converted_samples.reserve(self.config.sample_count()?);
        for bytes in interleaved.chunks_exact(3) {
            let unsigned =
                u32::from(bytes[0]) | (u32::from(bytes[1]) << 8) | (u32::from(bytes[2]) << 16);
            let signed = if unsigned & 0x80_0000 != 0 {
                (unsigned | 0xff00_0000) as i32
            } else {
                unsigned as i32
            };
            self.converted_samples.push(signed);
        }
        self.encode_converted()
    }

    /// Encodes interleaved signed samples, clipping to the configured bit depth.
    pub fn encode_i32(&mut self, interleaved: &[i32]) -> Result<EncodedFlacFrame, FlacFrameError> {
        validate_sample_len(self.config, interleaved.len())?;
        let (minimum, maximum) = sample_limits(self.config.bits_per_sample);
        self.converted_samples.clear();
        self.converted_samples.extend(
            interleaved
                .iter()
                .map(|sample| sample.clamp(&minimum, &maximum)),
        );
        self.encode_converted()
    }

    fn encode_converted(&mut self) -> Result<EncodedFlacFrame, FlacFrameError> {
        self.frame_buffer
            .fill_interleaved(&self.converted_samples)
            .map_err(|error| FlacFrameError::InvalidInput(error.to_string()))?;
        let sequence = self.next_sequence;
        let frame = flacenc::encode_fixed_size_frame(
            &self.encoder_config,
            &self.frame_buffer,
            sequence as usize,
            &self.stream_info,
        )
        .map_err(|error| FlacFrameError::Encode(error.to_string()))?;
        let mut sink = ByteSink::with_capacity(frame.count_bits());
        frame
            .write(&mut sink)
            .map_err(|error| FlacFrameError::Encode(error.to_string()))?;
        let payload = sink.into_inner();
        if payload.is_empty() {
            return Err(FlacFrameError::Encode(
                "encoder returned an empty FLAC frame".to_string(),
            ));
        }
        if payload.len() > self.config.maximum_packet_bytes()? {
            return Err(FlacFrameError::Encode(format!(
                "encoded frame has {} bytes, exceeding the defensive {} byte limit",
                payload.len(),
                self.config.maximum_packet_bytes()?
            )));
        }
        self.next_sequence = (self.next_sequence + 1) & 0x7fff_ffff;
        Ok(EncodedFlacFrame {
            sequence,
            sample_rate: self.config.sample_rate,
            channels: self.config.channels,
            bits_per_sample: self.config.bits_per_sample,
            frame_count: self.config.frame_length,
            pcm_bytes: self.config.raw_pcm_bytes()?,
            payload,
        })
    }
}

/// Persistent per-track pure-Rust decoder.
pub struct FlacFrameDecoder {
    config: FlacFrameConfig,
    inner: Box<dyn OxideDecoder>,
}

impl fmt::Debug for FlacFrameDecoder {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FlacFrameDecoder")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl FlacFrameDecoder {
    pub fn new(config: FlacFrameConfig) -> Result<Self, FlacFrameError> {
        config.validate()?;
        Ok(Self {
            config,
            inner: make_decoder(config)?,
        })
    }

    pub fn config(&self) -> FlacFrameConfig {
        self.config
    }

    /// Recreates the decoder so a discontinuity cannot retain pending input.
    pub fn reset(&mut self) -> Result<(), FlacFrameError> {
        self.inner = make_decoder(self.config)?;
        Ok(())
    }

    pub fn decode(&mut self, payload: &[u8]) -> Result<DecodedFlacFrame, FlacFrameError> {
        if payload.is_empty() {
            return Err(FlacFrameError::InvalidInput(
                "compressed FLAC frame is empty".to_string(),
            ));
        }
        let maximum_packet_bytes = self.config.maximum_packet_bytes()?;
        if payload.len() > maximum_packet_bytes {
            return Err(FlacFrameError::InvalidInput(format!(
                "compressed FLAC frame has {} bytes, exceeding the defensive {maximum_packet_bytes} byte limit",
                payload.len()
            )));
        }
        let packet = Packet::new(
            0,
            TimeBase::new(1, i64::from(self.config.sample_rate)),
            payload.to_vec(),
        );
        self.inner
            .send_packet(&packet)
            .map_err(|error| FlacFrameError::Decode(error.to_string()))?;
        let frame = self
            .inner
            .receive_frame()
            .map_err(|error| FlacFrameError::Decode(error.to_string()))?;
        let OxideFrame::Audio(audio) = frame else {
            return Err(FlacFrameError::Decode(
                "FLAC decoder returned a non-audio frame".to_string(),
            ));
        };
        if audio.samples != self.config.frame_length {
            return Err(FlacFrameError::FormatMismatch(format!(
                "decoded frame contains {} samples per channel, expected {}",
                audio.samples, self.config.frame_length
            )));
        }
        if audio.data.len() != 1 {
            return Err(FlacFrameError::FormatMismatch(format!(
                "decoded interleaved FLAC has {} planes, expected one",
                audio.data.len()
            )));
        }
        let data = &audio.data[0];
        if data.len() != self.config.raw_pcm_bytes()? {
            return Err(FlacFrameError::FormatMismatch(format!(
                "decoded FLAC has {} PCM bytes, expected {}",
                data.len(),
                self.config.raw_pcm_bytes()?
            )));
        }
        let samples = match self.config.bits_per_sample {
            16 => data
                .chunks_exact(2)
                .map(|bytes| i32::from(i16::from_le_bytes([bytes[0], bytes[1]])))
                .collect(),
            24 => data
                .chunks_exact(3)
                .map(|bytes| {
                    let unsigned = u32::from(bytes[0])
                        | (u32::from(bytes[1]) << 8)
                        | (u32::from(bytes[2]) << 16);
                    if unsigned & 0x80_0000 != 0 {
                        (unsigned | 0xff00_0000) as i32
                    } else {
                        unsigned as i32
                    }
                })
                .collect(),
            _ => unreachable!("validated by FlacFrameConfig"),
        };
        Ok(DecodedFlacFrame {
            sample_rate: self.config.sample_rate,
            channels: self.config.channels,
            bits_per_sample: self.config.bits_per_sample,
            frame_count: audio.samples,
            samples,
        })
    }
}

fn validate_sample_len(config: FlacFrameConfig, actual: usize) -> Result<(), FlacFrameError> {
    let expected = config.sample_count()?;
    if actual != expected {
        return Err(FlacFrameError::InvalidInput(format!(
            "interleaved frame has {actual} samples, expected {expected}"
        )));
    }
    Ok(())
}

fn sample_limits(bits_per_sample: u8) -> (i32, i32) {
    let magnitude_bits = u32::from(bits_per_sample - 1);
    let maximum = (1i32 << magnitude_bits) - 1;
    let minimum = -(1i32 << magnitude_bits);
    (minimum, maximum)
}

fn verified_encoder_config(
    frame_config: FlacFrameConfig,
) -> Result<Verified<config::Encoder>, FlacFrameError> {
    let mut encoder = config::Encoder::default();
    encoder.block_size = frame_config.frame_length as usize;
    encoder.multithread = false;
    match frame_config.profile {
        FlacProfile::Realtime => {
            encoder.subframe_coding.use_lpc = false;
            encoder.subframe_coding.fixed.max_order = 2;
        }
        FlacProfile::Balanced => {
            encoder.subframe_coding.use_lpc = false;
        }
        FlacProfile::Maximum => {}
    }
    encoder
        .into_verified()
        .map_err(|(_, error)| FlacFrameError::InvalidConfig(error.to_string()))
}

fn make_decoder(config: FlacFrameConfig) -> Result<Box<dyn OxideDecoder>, FlacFrameError> {
    let stream = Stream::new(
        config.sample_rate as usize,
        usize::from(config.channels),
        usize::from(config.bits_per_sample),
    )
    .map_err(|error| FlacFrameError::InvalidConfig(error.to_string()))?;
    let mut sink = ByteSink::with_capacity(stream.count_bits());
    stream
        .write(&mut sink)
        .map_err(|error| FlacFrameError::InvalidConfig(error.to_string()))?;
    let stream_bytes = sink.into_inner();
    let extradata = stream_bytes
        .strip_prefix(b"fLaC")
        .ok_or_else(|| {
            FlacFrameError::InvalidConfig("encoder did not produce FLAC metadata".to_string())
        })?
        .to_vec();
    let mut params = CodecParameters::audio(CodecId::new("flac"));
    params.sample_rate = Some(config.sample_rate);
    params.channels = Some(config.channels);
    params.sample_format = Some(config.sample_format());
    params.extradata = extradata;
    oxideav_flac::decoder::make_decoder(&params)
        .map_err(|error| FlacFrameError::InvalidConfig(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(channels: u16, bits_per_sample: u8, frame_length: u32) -> FlacFrameConfig {
        FlacFrameConfig::new(
            48_000,
            channels,
            bits_per_sample,
            frame_length,
            FlacProfile::Realtime,
        )
        .unwrap()
    }

    fn deterministic_samples(count: usize, minimum: i32, maximum: i32) -> Vec<i32> {
        let width = i64::from(maximum) - i64::from(minimum) + 1;
        let mut state = 0x9e37_79b9_u32;
        (0..count)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (i64::from(minimum) + i64::from(state) % width) as i32
            })
            .collect()
    }

    fn s24le(samples: &[i32]) -> Vec<u8> {
        samples
            .iter()
            .flat_map(|sample| {
                let value = sample.clamp(&-8_388_608, &8_388_607);
                [
                    (*value & 0xff) as u8,
                    ((*value >> 8) & 0xff) as u8,
                    ((*value >> 16) & 0xff) as u8,
                ]
            })
            .collect()
    }

    #[test]
    fn validates_supported_transport_formats_and_short_blocks() {
        for frame_length in [32, 48, 120, 240, 480, 960] {
            assert!(
                FlacFrameConfig::new(48_000, 2, 24, frame_length, FlacProfile::Realtime).is_ok()
            );
        }
        assert!(FlacFrameConfig::new(0, 2, 24, 240, FlacProfile::Realtime).is_err());
        assert!(FlacFrameConfig::new(48_000, 0, 24, 240, FlacProfile::Realtime).is_err());
        assert!(FlacFrameConfig::new(48_000, 9, 24, 240, FlacProfile::Realtime).is_err());
        assert!(FlacFrameConfig::new(48_000, 2, 20, 240, FlacProfile::Realtime).is_err());
        assert!(FlacFrameConfig::new(48_000, 2, 24, 31, FlacProfile::Realtime).is_err());
    }

    #[test]
    fn round_trips_s16_silence_impulses_extrema_and_random_samples() {
        for channels in [1, 2, 8] {
            let config = config(channels, 16, 240);
            let count = config.sample_count().unwrap();
            let mut samples =
                deterministic_samples(count, i32::from(i16::MIN), i32::from(i16::MAX))
                    .into_iter()
                    .map(|sample| sample as i16)
                    .collect::<Vec<_>>();
            samples[0] = i16::MIN;
            samples[1] = -1;
            samples[2] = 0;
            samples[3] = 1;
            samples[4] = i16::MAX;
            samples[count / 2] = i16::MAX;

            let mut encoder = FlacFrameEncoder::new(config).unwrap();
            let encoded = encoder.encode_i16(&samples).unwrap();
            assert_eq!(encoded.frame_count, 240);
            assert_eq!(encoded.channels, channels);
            assert_eq!(encoded.sequence, 0);
            assert!(!encoded.payload.is_empty());

            let mut decoder = FlacFrameDecoder::new(config).unwrap();
            let decoded = decoder.decode(&encoded.payload).unwrap();
            assert_eq!(decoded.to_i16().unwrap(), samples);
        }
    }

    #[test]
    fn round_trips_packed_s24_with_correct_sign_extension_and_interleaving() {
        for channels in [1, 2, 8] {
            let config = config(channels, 24, 240);
            let count = config.sample_count().unwrap();
            let mut samples = deterministic_samples(count, -8_388_608, 8_388_607);
            samples[0] = -8_388_608;
            samples[1] = -1;
            samples[2] = 0;
            samples[3] = 1;
            samples[4] = 8_388_607;
            for (index, sample) in samples
                .iter_mut()
                .enumerate()
                .skip(5)
                .take(channels as usize)
            {
                *sample = index as i32 * 101 - 300;
            }
            let bytes = s24le(&samples);

            let mut encoder = FlacFrameEncoder::new(config).unwrap();
            let encoded = encoder.encode_s24le(&bytes).unwrap();
            let mut decoder = FlacFrameDecoder::new(config).unwrap();
            let decoded = decoder.decode(&encoded.payload).unwrap();
            assert_eq!(decoded.samples, samples);
            assert_eq!(decoded.to_s24le().unwrap(), bytes);
        }
    }

    #[test]
    fn defensive_packet_cap_allows_expanded_eight_channel_realtime_frames() {
        let config = config(8, 24, 240);
        let legacy_stereo_centric_cap = config.raw_pcm_bytes().unwrap() * 2 + 4_096;
        assert!(
            config.maximum_packet_bytes().unwrap() > legacy_stereo_centric_cap,
            "8-channel 5 ms frames can expand beyond the old stereo-centric cap"
        );
    }

    #[test]
    fn honors_each_requested_frame_length_including_five_milliseconds() {
        for frame_length in [32, 48, 120, 240, 480, 960] {
            let config = config(2, 24, frame_length);
            let samples = deterministic_samples(config.sample_count().unwrap(), -32_768, 32_767);
            let mut encoder = FlacFrameEncoder::new(config).unwrap();
            let encoded = encoder.encode_i32(&samples).unwrap();
            let mut decoder = FlacFrameDecoder::new(config).unwrap();
            let decoded = decoder.decode(&encoded.payload).unwrap();
            assert_eq!(decoded.frame_count, frame_length);
            assert_eq!(decoded.samples, samples);
        }
    }

    #[test]
    fn clips_i32_input_to_the_declared_sample_depth() {
        let config = config(1, 24, 32);
        let mut samples = vec![0; 32];
        samples[0] = i32::MIN;
        samples[1] = -8_388_609;
        samples[2] = -8_388_608;
        samples[3] = 8_388_607;
        samples[4] = 8_388_608;
        samples[5] = i32::MAX;
        let expected = samples
            .iter()
            .map(|sample| sample.clamp(&-8_388_608, &8_388_607))
            .copied()
            .collect::<Vec<_>>();
        let mut encoder = FlacFrameEncoder::new(config).unwrap();
        let encoded = encoder.encode_i32(&samples).unwrap();
        let mut decoder = FlacFrameDecoder::new(config).unwrap();
        assert_eq!(decoder.decode(&encoded.payload).unwrap().samples, expected);
    }

    #[test]
    fn rejects_wrong_dimensions_and_sample_api() {
        let config = config(2, 24, 240);
        let mut encoder = FlacFrameEncoder::new(config).unwrap();
        assert!(matches!(
            encoder.encode_s24le(&vec![0; config.raw_pcm_bytes().unwrap() - 1]),
            Err(FlacFrameError::InvalidInput(_))
        ));
        assert!(matches!(
            encoder.encode_i16(&vec![0; config.sample_count().unwrap()]),
            Err(FlacFrameError::FormatMismatch(_))
        ));
    }

    #[test]
    fn reset_starts_a_new_track_segment_without_cross_track_state() {
        let config = config(1, 16, 48);
        let first = vec![101i16; 48];
        let second = vec![-202i16; 48];
        let mut encoder_a = FlacFrameEncoder::new(config).unwrap();
        let mut encoder_b = FlacFrameEncoder::new(config).unwrap();

        let a0 = encoder_a.encode_i16(&first).unwrap();
        let a1 = encoder_a.encode_i16(&second).unwrap();
        let b0 = encoder_b.encode_i16(&second).unwrap();
        assert_eq!((a0.sequence, a1.sequence, b0.sequence), (0, 1, 0));

        encoder_a.reset();
        let reset = encoder_a.encode_i16(&first).unwrap();
        assert_eq!(reset.sequence, 0);

        let mut decoder_a = FlacFrameDecoder::new(config).unwrap();
        let mut decoder_b = FlacFrameDecoder::new(config).unwrap();
        assert_eq!(
            decoder_a.decode(&a0.payload).unwrap().to_i16().unwrap(),
            first
        );
        assert_eq!(
            decoder_b.decode(&b0.payload).unwrap().to_i16().unwrap(),
            second
        );
        decoder_a.reset().unwrap();
        assert_eq!(
            decoder_a.decode(&reset.payload).unwrap().to_i16().unwrap(),
            first
        );
    }

    #[test]
    fn malformed_truncated_and_corrupted_packets_fail_safely() {
        let config = config(2, 24, 240);
        let samples = deterministic_samples(config.sample_count().unwrap(), -8_388_608, 8_388_607);
        let mut encoder = FlacFrameEncoder::new(config).unwrap();
        let encoded = encoder.encode_i32(&samples).unwrap();

        let mut decoder = FlacFrameDecoder::new(config).unwrap();
        assert!(decoder.decode(&[]).is_err());
        assert!(decoder
            .decode(&encoded.payload[..encoded.payload.len() / 2])
            .is_err());

        let mut corrupted = encoded.payload.clone();
        let last = corrupted.len() - 1;
        corrupted[last] ^= 0x5a;
        assert!(decoder.decode(&corrupted).is_err());
    }

    #[test]
    fn rejects_packets_decoded_with_the_wrong_track_format() {
        let source_config = config(2, 24, 240);
        let samples = vec![123i32; source_config.sample_count().unwrap()];
        let encoded = FlacFrameEncoder::new(source_config)
            .unwrap()
            .encode_i32(&samples)
            .unwrap();
        let wrong_config = config(1, 24, 240);
        assert!(FlacFrameDecoder::new(wrong_config)
            .unwrap()
            .decode(&encoded.payload)
            .is_err());
    }
}
