//! Pure-Rust, independently framed FLAC for latency-sensitive transports.
//!
//! This module deliberately keeps codec configuration out of the compressed
//! packet. The surrounding transport already carries sample rate, channel
//! count, sample format and frame count, so each packet can remain one raw
//! FLAC frame. A decoder is configured once per track and can decode every
//! subsequent packet independently.

use crate::bitsink::ByteSink;
use crate::component::{BitRepr, Stream, StreamInfo};
use crate::config;
use crate::error::{Verified, Verify};
use crate::source::{Context, Fill, FrameBuf};
use std::error::Error;
use std::fmt;

const FLAC_MAX_CHANNELS: u16 = 8;
const FLAC_MIN_BLOCK_SIZE: u32 = 32;
const FLAC_MAX_BLOCK_SIZE: u32 = 65_535;
const FLAC_MAX_SAMPLE_RATE: u32 = crate::constant::MAX_SAMPLE_RATE as u32;
const MAX_PACKET_OVERHEAD_BYTES: usize = 4_096;
const MAX_PACKET_EXPANSION_RATIO: usize = 8;

/// Encoding effort for one independently framed packet.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FlacProfile {
    /// Match libFLAC compression level 0: fixed predictors through order four
    /// with independent channel coding.
    #[default]
    Realtime,
    /// Match libFLAC compression level 2: fixed predictors through order four
    /// plus per-frame stereo decorrelation.
    ///
    /// This is the recommended profile for the optimized 5 ms packet path.
    Balanced,
    /// Use the generic encoder's complete LPC search.
    ///
    /// This profile is retained for compatibility and does not use the
    /// specialized 5 ms packet encoder.
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

    /// The frame rendered as 16-bit, whatever depth it was written at.
    ///
    /// A wider sample is shifted down by the difference rather than refused:
    /// rendering as 16-bit is well defined from any depth, and every other
    /// way of asking — `decode_f32`, `FlacDecoder::decode_i16` — answers it.
    /// Refusing only moves the conversion out to the caller, which is where
    /// it went wrong before: a 24-bit stream reduced by clamping came back
    /// as a full-scale square wave.
    pub fn to_i16(&self) -> Result<Vec<i16>, FlacFrameError> {
        let shift = u32::from(self.bits_per_sample.saturating_sub(16));
        Ok(self
            .samples
            .iter()
            .map(|&sample| {
                (sample >> shift).clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
            })
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
    context: Context,
    packet_encoder: crate::packet::PacketEncoder,
    track_stream_info: bool,
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
            context: Context::new(
                usize::from(config.bits_per_sample),
                usize::from(config.channels),
            ),
            packet_encoder: crate::packet::PacketEncoder::new(config),
            track_stream_info: false,
            next_sequence: 0,
        })
    }

    pub fn config(&self) -> FlacFrameConfig {
        self.config
    }

    pub fn next_sequence(&self) -> u32 {
        self.next_sequence
    }

    /// Enables STREAMINFO statistics and PCM MD5 tracking for the native
    /// stream wrapper. Independently framed packet encoders leave this off so
    /// file-level bookkeeping is not part of each latency-sensitive call.
    pub(crate) fn enable_stream_info_tracking(&mut self) {
        self.track_stream_info = true;
    }

    /// Resets only this track's FLAC continuity segment.
    pub fn reset(&mut self) {
        self.next_sequence = 0;
        self.converted_samples.clear();
        self.stream_info = StreamInfo::new(
            self.config.sample_rate as usize,
            usize::from(self.config.channels),
            usize::from(self.config.bits_per_sample),
        )
        .expect("validated FLAC stream geometry");
        self.context = Context::new(
            usize::from(self.config.bits_per_sample),
            usize::from(self.config.channels),
        );
    }

    pub fn encode_i16(&mut self, interleaved: &[i16]) -> Result<EncodedFlacFrame, FlacFrameError> {
        let sequence = self.next_sequence;
        let mut payload = Vec::new();
        self.encode_i16_into(interleaved, &mut payload)?;
        self.owned_frame(sequence, self.config.frame_length, payload)
    }

    /// Encodes one S16 frame into caller-owned reusable packet storage.
    ///
    /// `output` is cleared before encoding. Its allocation is retained when it
    /// is large enough for the next packet.
    pub fn encode_i16_into(
        &mut self,
        interleaved: &[i16],
        output: &mut Vec<u8>,
    ) -> Result<usize, FlacFrameError> {
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
        self.encode_converted_into(self.config.frame_length, output)
    }

    pub fn encode_s24le(&mut self, interleaved: &[u8]) -> Result<EncodedFlacFrame, FlacFrameError> {
        let sequence = self.next_sequence;
        let mut payload = Vec::new();
        self.encode_s24le_into(interleaved, &mut payload)?;
        self.owned_frame(sequence, self.config.frame_length, payload)
    }

    /// Encodes one packed S24LE frame into caller-owned reusable packet
    /// storage.
    ///
    /// `output` is cleared before encoding. Its allocation is retained when it
    /// is large enough for the next packet.
    pub fn encode_s24le_into(
        &mut self,
        interleaved: &[u8],
        output: &mut Vec<u8>,
    ) -> Result<usize, FlacFrameError> {
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
        self.encode_converted_into(self.config.frame_length, output)
    }

    /// Encodes interleaved signed samples, clipping to the configured bit depth.
    pub fn encode_i32(&mut self, interleaved: &[i32]) -> Result<EncodedFlacFrame, FlacFrameError> {
        validate_sample_len(self.config, interleaved.len())?;
        self.encode_i32_block(interleaved)
    }

    /// Encodes one full i32 frame into caller-owned reusable packet storage.
    ///
    /// `output` is cleared before encoding. Its allocation is retained when it
    /// is large enough for the next packet.
    pub fn encode_i32_into(
        &mut self,
        interleaved: &[i32],
        output: &mut Vec<u8>,
    ) -> Result<usize, FlacFrameError> {
        validate_sample_len(self.config, interleaved.len())?;
        self.encode_i32_block_into(interleaved, output)
    }

    /// Encodes one complete stream block up to the configured frame length.
    ///
    /// A complete FLAC file can use a shorter final block. The caller must
    /// keep all non-final blocks at the configured frame length.
    pub fn encode_i32_block(
        &mut self,
        interleaved: &[i32],
    ) -> Result<EncodedFlacFrame, FlacFrameError> {
        let channels = usize::from(self.config.channels);
        let frame_length = u32::try_from(interleaved.len() / channels)
            .map_err(|_| FlacFrameError::Overflow("FLAC frame length"))?;
        let sequence = self.next_sequence;
        let mut payload = Vec::new();
        self.encode_i32_block_into(interleaved, &mut payload)?;
        self.owned_frame(sequence, frame_length, payload)
    }

    /// Encodes a complete stream block into caller-owned reusable packet
    /// storage. This compatibility variant accepts a shorter final block.
    pub fn encode_i32_block_into(
        &mut self,
        interleaved: &[i32],
        output: &mut Vec<u8>,
    ) -> Result<usize, FlacFrameError> {
        let channels = usize::from(self.config.channels);
        if interleaved.len() % channels != 0 {
            return Err(FlacFrameError::InvalidInput(format!(
                "interleaved block has {} samples, which is not divisible by {channels} channels",
                interleaved.len()
            )));
        }
        let frame_length = u32::try_from(interleaved.len() / channels)
            .map_err(|_| FlacFrameError::Overflow("FLAC frame length"))?;
        if !(FLAC_MIN_BLOCK_SIZE..=self.config.frame_length).contains(&frame_length) {
            return Err(FlacFrameError::InvalidInput(format!(
                "stream block has {frame_length} frames, expected {FLAC_MIN_BLOCK_SIZE}..={}",
                self.config.frame_length
            )));
        }
        if !self.track_stream_info
            && frame_length == self.config.frame_length
            && self.packet_encoder.supports()
        {
            self.packet_encoder
                .encode(interleaved, self.next_sequence, output);
            self.validate_packet_size(output, interleaved.len())?;
            self.next_sequence = (self.next_sequence + 1) & 0x7fff_ffff;
            return Ok(output.len());
        }
        let (minimum, maximum) = sample_limits(self.config.bits_per_sample);
        self.converted_samples.clear();
        self.converted_samples.extend(
            interleaved
                .iter()
                .map(|sample| sample.clamp(&minimum, &maximum)),
        );
        self.encode_converted_into(frame_length, output)
    }

    fn encode_converted_into(
        &mut self,
        frame_length: u32,
        output: &mut Vec<u8>,
    ) -> Result<usize, FlacFrameError> {
        if !self.track_stream_info
            && frame_length == self.config.frame_length
            && self.packet_encoder.supports()
        {
            self.packet_encoder
                .encode(&self.converted_samples, self.next_sequence, output);
            self.validate_packet_size(output, self.converted_samples.len())?;
            self.next_sequence = (self.next_sequence + 1) & 0x7fff_ffff;
            return Ok(output.len());
        }

        let mut final_frame_buffer;
        let frame_buffer = if frame_length == self.config.frame_length {
            self.frame_buffer
                .fill_interleaved(&self.converted_samples)
                .map_err(|error| FlacFrameError::InvalidInput(error.to_string()))?;
            &self.frame_buffer
        } else {
            final_frame_buffer =
                FrameBuf::with_size(usize::from(self.config.channels), frame_length as usize)
                    .map_err(|error| FlacFrameError::InvalidInput(error.to_string()))?;
            final_frame_buffer
                .fill_interleaved(&self.converted_samples)
                .map_err(|error| FlacFrameError::InvalidInput(error.to_string()))?;
            &final_frame_buffer
        };
        let sequence = self.next_sequence;
        let frame = crate::coding::encode_fixed_size_frame(
            &self.encoder_config,
            frame_buffer,
            sequence as usize,
            &self.stream_info,
        )
        .map_err(|error| FlacFrameError::Encode(error.to_string()))?;
        let mut sink = ByteSink::from_storage(std::mem::take(output));
        sink.reserve(frame.count_bits());
        let write_result = frame.write(&mut sink);
        *output = sink.into_inner();
        write_result.map_err(|error| FlacFrameError::Encode(error.to_string()))?;
        self.validate_packet_size(output, self.converted_samples.len())?;
        if self.track_stream_info {
            self.stream_info.update_frame_info(&frame);
            self.context
                .fill_interleaved(&self.converted_samples)
                .map_err(|error| FlacFrameError::Encode(error.to_string()))?;
        }
        self.next_sequence = (self.next_sequence + 1) & 0x7fff_ffff;
        Ok(output.len())
    }

    fn validate_packet_size(
        &self,
        output: &[u8],
        sample_count: usize,
    ) -> Result<(), FlacFrameError> {
        if output.is_empty() {
            return Err(FlacFrameError::Encode(
                "encoder returned an empty FLAC frame".to_string(),
            ));
        }
        let pcm_bytes = sample_count
            .checked_mul(usize::from(self.config.bits_per_sample / 8))
            .ok_or(FlacFrameError::Overflow("FLAC PCM byte count"))?;
        let maximum_packet_bytes = pcm_bytes
            .checked_mul(MAX_PACKET_EXPANSION_RATIO)
            .and_then(|bytes| bytes.checked_add(MAX_PACKET_OVERHEAD_BYTES))
            .ok_or(FlacFrameError::Overflow("FLAC packet size limit"))?;
        if output.len() > maximum_packet_bytes {
            return Err(FlacFrameError::Encode(format!(
                "encoded frame has {} bytes, exceeding the defensive {maximum_packet_bytes} byte limit",
                output.len()
            )));
        }
        Ok(())
    }

    fn owned_frame(
        &self,
        sequence: u32,
        frame_length: u32,
        payload: Vec<u8>,
    ) -> Result<EncodedFlacFrame, FlacFrameError> {
        let pcm_bytes = (frame_length as usize)
            .checked_mul(usize::from(self.config.channels))
            .and_then(|samples| samples.checked_mul(usize::from(self.config.bits_per_sample / 8)))
            .ok_or(FlacFrameError::Overflow("FLAC PCM byte count"))?;
        Ok(EncodedFlacFrame {
            sequence,
            sample_rate: self.config.sample_rate,
            channels: self.config.channels,
            bits_per_sample: self.config.bits_per_sample,
            frame_count: frame_length,
            pcm_bytes,
            payload,
        })
    }

    /// Returns metadata for the explicit native-stream compatibility wrapper.
    pub(crate) fn stream_header(&self) -> Result<Vec<u8>, FlacFrameError> {
        let mut stream_info = self.stream_info.clone();
        if self.context.total_samples() > 0 {
            stream_info.set_total_samples(self.context.total_samples());
            stream_info.set_md5_digest(&self.context.md5_digest());
        }
        let stream = Stream::with_stream_info(stream_info);
        let mut sink = ByteSink::with_capacity(stream.count_bits());
        stream
            .write(&mut sink)
            .map_err(|error| FlacFrameError::Encode(error.to_string()))?;
        let bytes = sink.into_inner();
        bytes
            .strip_prefix(b"fLaC")
            .map(<[u8]>::to_vec)
            .ok_or_else(|| FlacFrameError::Encode("FLAC metadata has no stream marker".to_string()))
    }
}

/// Persistent per-track pure-Rust decoder.
pub struct FlacFrameDecoder {
    config: FlacFrameConfig,
    frame_buffer: Vec<i32>,
    verify_checksums: bool,
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
            frame_buffer: Vec::with_capacity(config.sample_count()?),
            // FFmpeg's FLAC packet decoder does not verify the frame CRC. Raw
            // packets normally have integrity protection at the transport
            // layer; callers that need standalone FLAC validation can opt in.
            verify_checksums: false,
        })
    }

    pub fn config(&self) -> FlacFrameConfig {
        self.config
    }

    /// Enables or disables FLAC header and frame checksum verification.
    ///
    /// Independently framed packet decoding defaults to `false`, matching
    /// FFmpeg's FLAC decoder. Native stream decoding continues to verify by
    /// default.
    pub fn set_verify_checksums(&mut self, enabled: bool) {
        self.verify_checksums = enabled;
    }

    pub fn verify_checksums(&self) -> bool {
        self.verify_checksums
    }

    /// Resets the decoder at a packet boundary.
    pub fn reset(&mut self) -> Result<(), FlacFrameError> {
        self.frame_buffer.clear();
        Ok(())
    }

    pub fn decode(&mut self, payload: &[u8]) -> Result<DecodedFlacFrame, FlacFrameError> {
        let mut samples = vec![0; self.config.sample_count()?];
        self.decode_into(payload, &mut samples)?;
        Ok(DecodedFlacFrame {
            sample_rate: self.config.sample_rate,
            channels: self.config.channels,
            bits_per_sample: self.config.bits_per_sample,
            frame_count: self.config.frame_length,
            samples,
        })
    }

    /// Decodes one raw FLAC frame into caller-owned interleaved sample
    /// storage, returning the number of samples written.
    pub fn decode_into(
        &mut self,
        payload: &[u8],
        output: &mut [i32],
    ) -> Result<usize, FlacFrameError> {
        let expected_samples = self.config.sample_count()?;
        if output.len() < expected_samples {
            return Err(FlacFrameError::InvalidInput(format!(
                "decoded output has room for {} samples, expected at least {expected_samples}",
                output.len()
            )));
        }
        self.decode_i32_impl(payload, output, false)
    }

    /// Decodes one raw FLAC frame whose block length may be shorter than the
    /// configured packet length. This is intended for a transport's final
    /// packet; all ordinary packets should use [`Self::decode_into`].
    pub fn decode_i32_block_into(
        &mut self,
        payload: &[u8],
        output: &mut [i32],
    ) -> Result<usize, FlacFrameError> {
        self.decode_i32_impl(payload, output, true)
    }

    /// Decodes one full S16 FLAC packet directly into reusable interleaved
    /// sample storage.
    pub fn decode_i16_into(
        &mut self,
        payload: &[u8],
        output: &mut [i16],
    ) -> Result<usize, FlacFrameError> {
        let expected_samples = self.config.sample_count()?;
        if output.len() < expected_samples {
            return Err(FlacFrameError::InvalidInput(format!(
                "decoded S16 output has room for {} samples, expected at least {expected_samples}",
                output.len()
            )));
        }
        self.decode_i16_impl(payload, output, false)
    }

    /// Decodes a possibly short final S16 FLAC packet into reusable storage.
    pub fn decode_i16_block_into(
        &mut self,
        payload: &[u8],
        output: &mut [i16],
    ) -> Result<usize, FlacFrameError> {
        self.decode_i16_impl(payload, output, true)
    }

    /// Decodes one full S24 FLAC packet directly into caller-owned packed
    /// little-endian storage, returning the number of bytes written.
    pub fn decode_s24le_into(
        &mut self,
        payload: &[u8],
        output: &mut [u8],
    ) -> Result<usize, FlacFrameError> {
        if self.config.bits_per_sample != 24 {
            return Err(FlacFrameError::FormatMismatch(format!(
                "decoder is configured for {}-bit samples, not packed S24",
                self.config.bits_per_sample
            )));
        }
        let expected_bytes = self.config.raw_pcm_bytes()?;
        if output.len() < expected_bytes {
            return Err(FlacFrameError::InvalidInput(format!(
                "decoded S24 output has room for {} bytes, expected at least {expected_bytes}",
                output.len()
            )));
        }
        self.decode_s24le_impl(payload, output, false)
    }

    /// Decodes a possibly short final S24 FLAC packet into caller-owned packed
    /// little-endian storage, returning the number of bytes written.
    pub fn decode_s24le_block_into(
        &mut self,
        payload: &[u8],
        output: &mut [u8],
    ) -> Result<usize, FlacFrameError> {
        if self.config.bits_per_sample != 24 {
            return Err(FlacFrameError::FormatMismatch(format!(
                "decoder is configured for {}-bit samples, not packed S24",
                self.config.bits_per_sample
            )));
        }
        self.decode_s24le_impl(payload, output, true)
    }

    fn decode_i32_impl(
        &mut self,
        payload: &[u8],
        output: &mut [i32],
        allow_short_block: bool,
    ) -> Result<usize, FlacFrameError> {
        let block = self.decode_block(payload, allow_short_block)?;
        let samples = block.len() as usize;
        if output.len() < samples {
            self.frame_buffer = block.into_buffer();
            return Err(FlacFrameError::InvalidInput(format!(
                "decoded output has room for {} samples, packet contains {samples}",
                output.len()
            )));
        }
        for_each_interleaved_sample(&block, |index, sample| output[index] = sample);
        self.frame_buffer = block.into_buffer();
        Ok(samples)
    }

    fn decode_i16_impl(
        &mut self,
        payload: &[u8],
        output: &mut [i16],
        allow_short_block: bool,
    ) -> Result<usize, FlacFrameError> {
        let block = self.decode_block(payload, allow_short_block)?;
        let samples = block.len() as usize;
        if output.len() < samples {
            self.frame_buffer = block.into_buffer();
            return Err(FlacFrameError::InvalidInput(format!(
                "decoded S16 output has room for {} samples, packet contains {samples}",
                output.len()
            )));
        }
        let shift = u32::from(self.config.bits_per_sample.saturating_sub(16));
        for_each_interleaved_sample(&block, |index, sample| {
            output[index] =
                (sample >> shift).clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
        });
        self.frame_buffer = block.into_buffer();
        Ok(samples)
    }

    fn decode_s24le_impl(
        &mut self,
        payload: &[u8],
        output: &mut [u8],
        allow_short_block: bool,
    ) -> Result<usize, FlacFrameError> {
        let block = self.decode_block(payload, allow_short_block)?;
        let bytes = (block.len() as usize)
            .checked_mul(3)
            .ok_or(FlacFrameError::Overflow("decoded S24 byte count"))?;
        if output.len() < bytes {
            self.frame_buffer = block.into_buffer();
            return Err(FlacFrameError::InvalidInput(format!(
                "decoded S24 output has room for {} bytes, packet requires {bytes}",
                output.len()
            )));
        }
        for_each_interleaved_sample(&block, |index, sample| {
            let sample = sample.clamp(-8_388_608, 8_388_607);
            let offset = index * 3;
            output[offset] = (sample & 0xff) as u8;
            output[offset + 1] = ((sample >> 8) & 0xff) as u8;
            output[offset + 2] = ((sample >> 16) & 0xff) as u8;
        });
        self.frame_buffer = block.into_buffer();
        Ok(bytes)
    }

    fn decode_block(
        &mut self,
        payload: &[u8],
        allow_short_block: bool,
    ) -> Result<crate::decode::Block, FlacFrameError> {
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

        let frame_buffer = std::mem::take(&mut self.frame_buffer);
        let decoded = crate::decode::frame::decode_frame_slice(
            payload,
            Some(self.config.sample_rate),
            Some(u32::from(self.config.bits_per_sample)),
            frame_buffer,
            self.verify_checksums,
        );
        let (block, consumed) = match decoded {
            Ok(Some(decoded)) => decoded,
            Ok(None) => {
                self.frame_buffer = Vec::with_capacity(self.config.sample_count()?);
                return Err(FlacFrameError::Decode(
                    "FLAC packet has no frame".to_string(),
                ));
            }
            Err(error) => {
                self.frame_buffer = Vec::with_capacity(self.config.sample_count()?);
                return Err(FlacFrameError::Decode(error.to_string()));
            }
        };
        if consumed != payload.len() {
            self.frame_buffer = block.into_buffer();
            return Err(FlacFrameError::InvalidInput(format!(
                "FLAC packet contains {} trailing bytes",
                payload.len() - consumed
            )));
        }
        let valid_duration = if allow_short_block {
            (FLAC_MIN_BLOCK_SIZE..=self.config.frame_length).contains(&block.duration())
        } else {
            block.duration() == self.config.frame_length
        };
        if !valid_duration {
            let duration = block.duration();
            self.frame_buffer = block.into_buffer();
            return Err(FlacFrameError::FormatMismatch(format!(
                "decoded frame contains {duration} samples per channel, expected {}{}",
                if allow_short_block { "32..=" } else { "" },
                self.config.frame_length,
            )));
        }
        if block.channels() != u32::from(self.config.channels) {
            let channels = block.channels();
            self.frame_buffer = block.into_buffer();
            return Err(FlacFrameError::FormatMismatch(format!(
                "decoded frame contains {channels} channels, expected {}",
                self.config.channels,
            )));
        }
        Ok(block)
    }
}

#[inline]
fn for_each_interleaved_sample(block: &crate::decode::Block, mut write: impl FnMut(usize, i32)) {
    match block.channels() {
        1 => {
            for (index, &sample) in block.channel(0).iter().enumerate() {
                write(index, sample);
            }
        }
        2 => {
            for (frame, (left, right)) in block.stereo_samples().enumerate() {
                let index = frame * 2;
                write(index, left);
                write(index + 1, right);
            }
        }
        channels => {
            for frame in 0..block.duration() {
                for channel in 0..channels {
                    let index = frame as usize * channels as usize + channel as usize;
                    write(index, block.sample(channel, frame));
                }
            }
        }
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
    let mut encoder = config::Encoder {
        block_size: frame_config.frame_length as usize,
        multithread: false,
        ..config::Encoder::default()
    };
    // The default fixed-predictor search mirrors libFLAC's non-LPC compression
    // levels: choose one predictor before partitioned-Rice analysis, then
    // compare the completed subframe with verbatim.
    match frame_config.profile {
        FlacProfile::Realtime => {
            encoder.subframe_coding.use_lpc = false;
            encoder.stereo_coding.use_leftside = false;
            encoder.stereo_coding.use_rightside = false;
            encoder.stereo_coding.use_midside = false;
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
            let mut direct = vec![0_i16; samples.len()];
            assert_eq!(
                decoder
                    .decode_i16_into(&encoded.payload, &mut direct)
                    .unwrap(),
                samples.len()
            );
            assert_eq!(direct, samples);
        }
    }

    /// Asking a 24-bit frame for 16-bit narrows it rather than refusing or
    /// saturating. Both renderers answer, and both answer the same.
    #[test]
    fn renders_a_24_bit_frame_as_16_bit_at_its_own_level() {
        let channels = 2;
        let config = config(channels, 24, 240);
        let count = config.sample_count().unwrap();
        // A sixth of 24-bit full scale: far outside i16, so a clamp shows.
        let samples: Vec<i32> = (0..count)
            .map(|index| ((index as f64 / 30.0).sin() * 1_398_101.0) as i32)
            .collect();

        let mut encoder = FlacFrameEncoder::new(config).unwrap();
        let encoded = encoder.encode_i32(&samples).unwrap();

        let want: Vec<i16> = samples.iter().map(|sample| (sample >> 8) as i16).collect();
        let mut decoder = FlacFrameDecoder::new(config).unwrap();
        assert_eq!(
            decoder.decode(&encoded.payload).unwrap().to_i16().unwrap(),
            want,
            "to_i16 did not narrow a 24-bit frame"
        );
        let mut direct = vec![0_i16; samples.len()];
        assert_eq!(
            decoder
                .decode_i16_into(&encoded.payload, &mut direct)
                .unwrap(),
            samples.len()
        );
        assert_eq!(direct, want, "decode_i16_into did not narrow a 24-bit frame");
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
            let mut direct = vec![0_u8; bytes.len()];
            assert_eq!(
                decoder
                    .decode_s24le_into(&encoded.payload, &mut direct)
                    .unwrap(),
                bytes.len()
            );
            assert_eq!(direct, bytes);
        }
    }

    #[test]
    fn raw_decoder_accepts_a_declared_short_final_block_only_via_block_api() {
        let config = config(2, 16, 240);
        let frames = 73usize;
        let samples = deterministic_samples(
            frames * usize::from(config.channels),
            i32::from(i16::MIN),
            i32::from(i16::MAX),
        );
        let packet = FlacFrameEncoder::new(config)
            .unwrap()
            .encode_i32_block(&samples)
            .unwrap()
            .payload;
        let mut decoder = FlacFrameDecoder::new(config).unwrap();
        let mut output = vec![0_i32; config.sample_count().unwrap()];
        assert!(decoder.decode_into(&packet, &mut output).is_err());
        assert_eq!(
            decoder.decode_i32_block_into(&packet, &mut output).unwrap(),
            samples.len()
        );
        assert_eq!(&output[..samples.len()], samples);

        let mut direct = vec![0_i16; config.sample_count().unwrap()];
        assert_eq!(
            decoder.decode_i16_block_into(&packet, &mut direct).unwrap(),
            samples.len()
        );
        assert_eq!(
            &direct[..samples.len()],
            samples
                .iter()
                .map(|&sample| sample as i16)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn realtime_s24_uses_verbatim_when_capped_rice_would_expand() {
        let config = config(2, 24, 240);
        let mut samples = Vec::with_capacity(config.sample_count().unwrap());
        let mut left = -7_500_000_i32;
        let mut right = 7_000_000_i32;
        let mut state = 0x6d2b_79f5_u32;
        for _ in 0..config.frame_length {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let left_step = ((state >> 13) as i32 & 0x7ffff) - 0x40000;
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let right_step = ((state >> 13) as i32 & 0x7ffff) - 0x40000;
            left = (left + left_step).clamp(-8_388_608, 8_388_607);
            right = (right + right_step).clamp(-8_388_608, 8_388_607);
            samples.extend_from_slice(&[left, right]);
        }

        let mut encoder = FlacFrameEncoder::new(config).unwrap();
        let encoded = encoder.encode_i32(&samples).unwrap();
        assert!(
            encoded.encoded_bytes() <= encoded.pcm_bytes + 32,
            "FLAC frame expanded from {} PCM bytes to {} encoded bytes",
            encoded.pcm_bytes,
            encoded.encoded_bytes()
        );

        let mut decoder = FlacFrameDecoder::new(config).unwrap();
        assert_eq!(decoder.decode(&encoded.payload).unwrap().samples, samples);
    }

    #[test]
    fn pcm_step_to_silence_stays_within_verbatim_bounds() {
        for (bits_per_sample, left, right) in [(16, -20_000, -15_000), (24, -2_000_000, -1_500_000)]
        {
            for profile in [FlacProfile::Realtime, FlacProfile::Balanced] {
                let config =
                    FlacFrameConfig::new(48_000, 2, bits_per_sample, 240, profile).unwrap();
                let mut samples = vec![0; config.sample_count().unwrap()];
                for frame in 0..120 {
                    samples[frame * 2] = left + frame as i32 * 127;
                    samples[frame * 2 + 1] = right + frame as i32 * 63;
                }

                let mut encoder = FlacFrameEncoder::new(config).unwrap();
                let encoded = encoder.encode_i32(&samples).unwrap();
                assert!(encoded.encoded_bytes() <= encoded.pcm_bytes + 32);

                let mut decoder = FlacFrameDecoder::new(config).unwrap();
                assert_eq!(decoder.decode(&encoded.payload).unwrap().samples, samples);
            }
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
        decoder.set_verify_checksums(true);
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
    fn randomized_malformed_packet_matrix_never_panics_or_escapes_checksum_validation() {
        use std::panic::{catch_unwind, AssertUnwindSafe};

        for sample_rate in [48_000, 96_000] {
            for channels in [1, 2, 8] {
                for bits_per_sample in [16, 24] {
                    let config = FlacFrameConfig::new(
                        sample_rate,
                        channels,
                        bits_per_sample,
                        sample_rate / 200,
                        FlacProfile::Balanced,
                    )
                    .unwrap();
                    let (minimum, maximum) = sample_limits(bits_per_sample);
                    let samples =
                        deterministic_samples(config.sample_count().unwrap(), minimum, maximum);
                    let packet = FlacFrameEncoder::new(config)
                        .unwrap()
                        .encode_i32(&samples)
                        .unwrap()
                        .payload;
                    let mut decoder = FlacFrameDecoder::new(config).unwrap();
                    decoder.set_verify_checksums(true);
                    let mut output = vec![0_i32; samples.len()];

                    for length in [0, 1, 2, packet.len() / 3, packet.len() - 1] {
                        let result = catch_unwind(AssertUnwindSafe(|| {
                            decoder.decode_into(&packet[..length], &mut output)
                        }));
                        assert!(result.is_ok(), "truncated packet panicked: {config:?}");
                        assert!(
                            result.unwrap().is_err(),
                            "truncated packet decoded: {config:?}"
                        );
                    }

                    let mutation_stride = (packet.len() / 31).max(1);
                    for offset in (0..packet.len()).step_by(mutation_stride).take(32) {
                        let mut mutated = packet.clone();
                        mutated[offset] ^= 1 << (offset & 7);
                        let result = catch_unwind(AssertUnwindSafe(|| {
                            decoder.decode_into(&mutated, &mut output)
                        }));
                        assert!(result.is_ok(), "bit mutation panicked: {config:?}/{offset}");
                        assert!(
                            result.unwrap().is_err(),
                            "checksum-valid bit mutation escaped: {config:?}/{offset}"
                        );
                    }

                    let mut state = 0x6d2b_79f5_u32 ^ sample_rate ^ u32::from(channels);
                    for length in [3usize, 7, 31, 127, packet.len().min(1024)] {
                        let mut garbage = vec![0_u8; length];
                        for byte in &mut garbage {
                            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                            *byte = (state >> 24) as u8;
                        }
                        let result = catch_unwind(AssertUnwindSafe(|| {
                            decoder.decode_into(&garbage, &mut output)
                        }));
                        assert!(
                            result.is_ok(),
                            "random packet panicked: {config:?}/{length}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn reusable_packet_buffers_avoid_steady_state_allocations() {
        let config = config(2, 24, 240);
        let samples = deterministic_samples(config.sample_count().unwrap(), -500_000, 500_000);
        let mut encoder = FlacFrameEncoder::new(config).unwrap();
        let mut packet = Vec::with_capacity(config.maximum_packet_bytes().unwrap());

        let first_len = encoder.encode_i32_into(&samples, &mut packet).unwrap();
        assert_eq!(first_len, packet.len());
        let packet_allocation = packet.as_ptr();
        let second_len = encoder.encode_i32_into(&samples, &mut packet).unwrap();
        assert_eq!(second_len, packet.len());
        assert_eq!(packet.as_ptr(), packet_allocation);

        let mut decoder = FlacFrameDecoder::new(config).unwrap();
        let mut decoded = vec![0; config.sample_count().unwrap()];
        assert_eq!(
            decoder.decode_into(&packet, &mut decoded).unwrap(),
            decoded.len()
        );
        assert_eq!(decoded, samples);
        let decode_allocation = decoder.frame_buffer.as_ptr();
        decoder.decode_into(&packet, &mut decoded).unwrap();
        assert_eq!(decoder.frame_buffer.as_ptr(), decode_allocation);
    }

    #[test]
    fn raw_packet_checksum_policy_matches_ffmpeg_by_default() {
        let config = config(2, 24, 240);
        let samples = deterministic_samples(config.sample_count().unwrap(), -100_000, 100_000);
        let packet = FlacFrameEncoder::new(config)
            .unwrap()
            .encode_i32(&samples)
            .unwrap()
            .payload;
        let mut corrupted_footer = packet.clone();
        let last = corrupted_footer.len() - 1;
        corrupted_footer[last] ^= 0x5a;

        let mut decoder = FlacFrameDecoder::new(config).unwrap();
        assert!(!decoder.verify_checksums());
        assert_eq!(decoder.decode(&corrupted_footer).unwrap().samples, samples);
        decoder.set_verify_checksums(true);
        assert!(decoder.decode(&corrupted_footer).is_err());
    }

    #[test]
    fn packet_core_round_trips_target_geometry_matrix() {
        for sample_rate in [48_000, 96_000] {
            for profile in [FlacProfile::Realtime, FlacProfile::Balanced] {
                for bits_per_sample in [16, 24] {
                    for channels in [1, 2, 8] {
                        let config = FlacFrameConfig::new(
                            sample_rate,
                            channels,
                            bits_per_sample,
                            sample_rate / 200,
                            profile,
                        )
                        .unwrap();
                        let (minimum, maximum) = sample_limits(bits_per_sample);
                        let mut samples =
                            deterministic_samples(config.sample_count().unwrap(), minimum, maximum);
                        samples[0] = minimum;
                        samples[1] = maximum;

                        let mut packet = Vec::with_capacity(config.maximum_packet_bytes().unwrap());
                        let mut encoder = FlacFrameEncoder::new(config).unwrap();
                        encoder.encode_i32_into(&samples, &mut packet).unwrap();
                        let mut decoder = FlacFrameDecoder::new(config).unwrap();
                        decoder.set_verify_checksums(true);
                        let mut output = vec![0; samples.len()];
                        assert_eq!(
                            decoder.decode_into(&packet, &mut output).unwrap(),
                            samples.len()
                        );
                        assert_eq!(output, samples);
                    }
                }
            }
        }
    }

    #[test]
    fn packet_core_encodes_all_sequence_width_boundaries() {
        let config = config(2, 24, 240);
        let samples = deterministic_samples(config.sample_count().unwrap(), -50_000, 50_000);
        let mut encoder = FlacFrameEncoder::new(config).unwrap();
        let mut decoder = FlacFrameDecoder::new(config).unwrap();
        decoder.set_verify_checksums(true);
        let mut packet = Vec::new();
        let mut output = vec![0; samples.len()];

        for sequence in [0, 0x7f, 0x80, 0x7ff, 0x800, 0xffff, 0x1_0000, 0x7fff_ffff] {
            encoder.next_sequence = sequence;
            encoder.encode_i32_into(&samples, &mut packet).unwrap();
            decoder.decode_into(&packet, &mut output).unwrap();
            assert_eq!(output, samples, "sequence={sequence}");
            assert_eq!(
                encoder.next_sequence(),
                sequence.wrapping_add(1) & 0x7fff_ffff
            );
        }
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
