use frame_header::{EncodingFlag, Endianness};
use opus_rs_full::{
    Application, OpusDecoder as FixedWorkspaceOpusDecoder, OpusEncoder as FixedWorkspaceOpusEncoder,
};
use ropus::{Channels, DecodeMode, Decoder as FullOpusDecoder};
use soundkit::audio_packet::{Decoder, Encoder};
use soundkit::audio_types::AudioData;
use tracing::{debug, trace};

pub struct OpusEncoder {
    encoder: FixedWorkspaceOpusEncoder,
    sample_rate: u32,
    channels: u32,
    _bits_per_sample: u32,
    frame_size: u32,
    bitrate: u32,
    f32_input: Vec<f32>,
}

impl Encoder for OpusEncoder {
    fn new(
        sample_rate: u32,
        bits_per_sample: u32,
        channels: u32,
        frame_size: u32,
        bitrate: u32,
    ) -> Self {
        let encoder = create_pure_encoder(sample_rate, channels, bitrate)
            .expect("failed to create Opus encoder");

        Self {
            encoder,
            sample_rate,
            channels,
            _bits_per_sample: bits_per_sample,
            frame_size,
            bitrate,
            f32_input: vec![0.0; frame_size as usize * channels as usize],
        }
    }

    fn init(&mut self) -> Result<(), String> {
        self.reset()
    }

    fn encode_i16(&mut self, input: &[i16], output: &mut [u8]) -> Result<usize, String> {
        let required = self.frame_size as usize * self.channels as usize;
        if input.len() < required {
            return Err(format!(
                "opus input too small: {} < {}",
                input.len(),
                required
            ));
        }

        for (destination, &sample) in self.f32_input.iter_mut().zip(&input[..required]) {
            *destination = sample as f32 / 32_768.0;
        }
        self.encoder
            .encode(&self.f32_input, self.frame_size as usize, output)
            .map_err(str::to_string)
    }

    fn encode_i32(&mut self, _input: &[i32], _output: &mut [u8]) -> Result<usize, String> {
        Err("Not implemented.".to_string())
    }

    fn reset(&mut self) -> Result<(), String> {
        self.encoder = create_pure_encoder(self.sample_rate, self.channels, self.bitrate)?;
        self.f32_input.fill(0.0);
        Ok(())
    }
}

fn create_pure_encoder(
    sample_rate: u32,
    channels: u32,
    bitrate: u32,
) -> Result<FixedWorkspaceOpusEncoder, String> {
    let mut encoder = FixedWorkspaceOpusEncoder::new(
        sample_rate as i32,
        channels as usize,
        Application::RestrictedLowDelay,
    )
    .map_err(str::to_string)?;
    encoder.bitrate_bps = bitrate as i32;
    encoder.use_cbr = true;
    Ok(encoder)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OpusPacketMode {
    SilkOnly,
    Hybrid,
    CeltOnly,
}

fn opus_packet_mode(toc: u8) -> OpusPacketMode {
    if toc & 0x80 != 0 {
        OpusPacketMode::CeltOnly
    } else if toc & 0x60 == 0x60 {
        OpusPacketMode::Hybrid
    } else {
        OpusPacketMode::SilkOnly
    }
}

fn opus_packet_frame_duration_ms(toc: u8) -> i32 {
    match opus_packet_mode(toc) {
        OpusPacketMode::SilkOnly => match (toc >> 3) & 0x03 {
            0 => 10,
            1 => 20,
            2 => 40,
            3 => 60,
            _ => unreachable!(),
        },
        OpusPacketMode::Hybrid => {
            if (toc >> 3) & 0x01 == 0 {
                10
            } else {
                20
            }
        }
        OpusPacketMode::CeltOnly => match (toc >> 3) & 0x03 {
            0 => 2,
            1 => 5,
            2 => 10,
            3 => 20,
            _ => unreachable!(),
        },
    }
}

fn opus_packet_frame_samples(toc: u8, sample_rate: u32) -> Option<usize> {
    match opus_packet_mode(toc) {
        OpusPacketMode::CeltOnly => {
            let period = ((toc >> 3) & 0x03) as u32;
            let frame_rate = 400u32.checked_shr(period)?;
            if frame_rate == 0 || sample_rate % frame_rate != 0 {
                return None;
            }
            Some((sample_rate / frame_rate) as usize)
        }
        OpusPacketMode::SilkOnly | OpusPacketMode::Hybrid => {
            let duration_ms = opus_packet_frame_duration_ms(toc) as u32;
            Some((u64::from(sample_rate) * u64::from(duration_ms) / 1000) as usize)
        }
    }
}

fn opus_packet_samples_per_channel(packet: &[u8], sample_rate: u32) -> Option<usize> {
    let toc = *packet.first()?;
    let frames = match toc & 0x03 {
        0 => 1,
        1 | 2 => 2,
        3 => usize::from(*packet.get(1)? & 0x3f),
        _ => unreachable!(),
    };
    if frames == 0 {
        return None;
    }
    Some(opus_packet_frame_samples(toc, sample_rate)?.checked_mul(frames)?)
}

enum OpusDecoderBackend {
    /// SoundKit's restricted-low-delay encoder always emits 48 kHz CELT.
    /// This decoder retains all codec and output workspaces after warm-up.
    Celt {
        decoder: FixedWorkspaceOpusDecoder,
        f32_output: Vec<f32>,
    },
    /// General containers can switch between SILK, hybrid, and CELT. Keep one
    /// full decoder for the complete logical stream so transition state stays
    /// correct.
    Full(FullOpusDecoder),
}

pub struct OpusDecoder {
    backend: OpusDecoderBackend,
    sample_rate: u32,
    channels: u8,
    first_frame_logged: bool,
}

impl OpusDecoder {
    /// Creates a decoder for general Opus streams.
    ///
    /// Use [`Self::new_celt_only`] only when the stream contract guarantees
    /// that every packet uses the CELT mode.
    pub fn new(sample_rate: usize, channels: usize) -> Result<Self, String> {
        Self::new_full(sample_rate, channels)
    }

    /// Creates a decoder for streams that can contain SILK, hybrid, CELT, or
    /// legal mode transitions.
    pub fn new_full(sample_rate: usize, channels: usize) -> Result<Self, String> {
        validate_decoder_config(sample_rate, channels)?;
        let channel_layout = channel_layout(channels);
        let decoder = FullOpusDecoder::new(sample_rate as u32, channel_layout)
            .map_err(|error| format!("failed to create Opus decoder: {error}"))?;

        Ok(Self {
            backend: OpusDecoderBackend::Full(decoder),
            sample_rate: sample_rate as u32,
            channels: channels as u8,
            first_frame_logged: false,
        })
    }

    /// Creates a decoder that rejects non-CELT packets instead of silently
    /// losing Opus mode-transition state.
    pub fn new_celt_only(sample_rate: usize, channels: usize) -> Result<Self, String> {
        validate_decoder_config(sample_rate, channels)?;
        if sample_rate != 48_000 {
            return Err("the allocation-light CELT decoder requires 48 kHz Opus".to_string());
        }
        let decoder = FixedWorkspaceOpusDecoder::new(sample_rate as i32, channels)
            .map_err(|error| format!("failed to create CELT Opus decoder: {error}"))?;

        Ok(Self {
            backend: OpusDecoderBackend::Celt {
                decoder,
                f32_output: vec![0.0; MAX_OPUS_FRAME_SAMPLES * channels],
            },
            sample_rate: sample_rate as u32,
            channels: channels as u8,
            first_frame_logged: false,
        })
    }

    pub fn init(&mut self) -> Result<(), String> {
        Ok(())
    }

    /// Returns this decoder to its cold-stream state without replacing the
    /// caller-owned session or its reusable output workspaces.
    pub fn reset(&mut self) -> Result<(), String> {
        match &mut self.backend {
            OpusDecoderBackend::Celt {
                decoder,
                f32_output,
            } => {
                *decoder =
                    FixedWorkspaceOpusDecoder::new(self.sample_rate as i32, self.channels as usize)
                        .map_err(|error| format!("failed to reset CELT Opus decoder: {error}"))?;
                f32_output.fill(0.0);
            }
            OpusDecoderBackend::Full(decoder) => {
                *decoder =
                    FullOpusDecoder::new(self.sample_rate, channel_layout(self.channels as usize))
                        .map_err(|error| format!("failed to reset Opus decoder: {error}"))?;
            }
        }
        self.first_frame_logged = false;
        Ok(())
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn channels(&self) -> u8 {
        self.channels
    }
}

fn validate_decoder_config(sample_rate: usize, channels: usize) -> Result<(), String> {
    const SUPPORTED_SAMPLE_RATES: [usize; 5] = [8_000, 12_000, 16_000, 24_000, 48_000];

    if !SUPPORTED_SAMPLE_RATES.contains(&sample_rate) {
        return Err(format!(
                "unsupported Opus decode sample rate {sample_rate} Hz; expected 8000, 12000, 16000, 24000, or 48000 Hz"
            ));
    }
    if channels == 0 {
        return Err("Opus decode channel count must be 1 or 2".to_string());
    }
    if channels > 2 {
        return Err(format!(
                "unsupported Opus channel mapping for {channels} channels; the single-stream decoder supports mono or stereo"
            ));
    }
    Ok(())
}

fn channel_layout(channels: usize) -> Channels {
    if channels == 1 {
        Channels::Mono
    } else {
        Channels::Stereo
    }
}

fn validate_celt_output(
    input: &[u8],
    output_len: usize,
    sample_rate: u32,
    channels: usize,
) -> Result<(), String> {
    if !input
        .first()
        .is_some_and(|toc| opus_packet_mode(*toc) == OpusPacketMode::CeltOnly)
    {
        return Err(
            "the allocation-light SoundKit Opus decoder received a SILK or hybrid packet"
                .to_string(),
        );
    }
    let samples_per_channel = opus_packet_samples_per_channel(input, sample_rate)
        .ok_or_else(|| "invalid Opus packet duration".to_string())?;
    let sample_count = samples_per_channel
        .checked_mul(channels)
        .ok_or_else(|| "Opus decode output size overflowed".to_string())?;
    if sample_count > output_len {
        return Err(format!(
            "opus decode output too large: {sample_count} > {output_len}"
        ));
    }
    Ok(())
}

impl Decoder for OpusDecoder {
    fn decode_i16(&mut self, input: &[u8], output: &mut [i16], fec: bool) -> Result<usize, String> {
        let decoded_samples_per_channel = match &mut self.backend {
            OpusDecoderBackend::Celt {
                decoder,
                f32_output,
            } => {
                if fec {
                    return Err(
                        "Opus FEC decode is not implemented by the CELT backend".to_string()
                    );
                }
                validate_celt_output(
                    input,
                    output.len(),
                    self.sample_rate,
                    self.channels as usize,
                )?;
                let expected_frames = opus_packet_samples_per_channel(input, self.sample_rate)
                    .ok_or_else(|| "invalid Opus packet duration".to_string())?;
                let sample_count = expected_frames * self.channels as usize;
                let frames = decoder
                    .decode(input, expected_frames, &mut f32_output[..sample_count])
                    .map_err(str::to_string)?;
                let decoded_count = frames * self.channels as usize;
                for (destination, &sample) in output[..decoded_count]
                    .iter_mut()
                    .zip(&f32_output[..decoded_count])
                {
                    *destination = (sample * 32_768.0)
                        .round()
                        .clamp(i16::MIN as f32, i16::MAX as f32)
                        as i16;
                }
                frames
            }
            OpusDecoderBackend::Full(decoder) => {
                let decode_mode = if fec {
                    DecodeMode::Fec
                } else {
                    DecodeMode::Normal
                };
                decoder
                    .decode(input, output, decode_mode)
                    .map_err(|error| error.to_string())?
            }
        };
        let decoded_count = decoded_samples_per_channel * self.channels as usize;

        if !self.first_frame_logged {
            debug!(
                sample_rate_hz = self.sample_rate,
                channels = self.channels,
                packet_len = input.len(),
                pcm_samples_written = decoded_count,
                "decoded Opus packet"
            );
        } else {
            trace!(
                sample_rate_hz = self.sample_rate,
                channels = self.channels,
                packet_len = input.len(),
                pcm_samples_written = decoded_count,
                "decoded Opus packet"
            );
        }
        self.first_frame_logged = true;

        Ok(decoded_samples_per_channel)
    }
    fn decode_i32(
        &mut self,
        _input: &[u8],
        _output: &mut [i32],
        _fec: bool,
    ) -> Result<usize, String> {
        Err("not implemented.".to_string())
    }

    fn decode_f32(&mut self, input: &[u8], output: &mut [f32], fec: bool) -> Result<usize, String> {
        match &mut self.backend {
            OpusDecoderBackend::Celt { decoder, .. } => {
                if fec {
                    return Err(
                        "Opus FEC decode is not implemented by the CELT backend".to_string()
                    );
                }
                validate_celt_output(
                    input,
                    output.len(),
                    self.sample_rate,
                    self.channels as usize,
                )?;
                let expected_frames = opus_packet_samples_per_channel(input, self.sample_rate)
                    .ok_or_else(|| "invalid Opus packet duration".to_string())?;
                decoder
                    .decode(input, expected_frames, output)
                    .map_err(str::to_string)
            }
            OpusDecoderBackend::Full(decoder) => {
                let decode_mode = if fec {
                    DecodeMode::Fec
                } else {
                    DecodeMode::Normal
                };
                decoder
                    .decode_float(input, output, decode_mode)
                    .map_err(|error| error.to_string())
            }
        }
    }
}

const MAX_OPUS_FRAME_SAMPLES: usize = 5760; // 120 ms @ 48 kHz
const MAX_OPUS_STREAM_BUFFER_BYTES: usize = 4 * 1024 * 1024;

/// Streaming decoder for raw Opus format (OpusHead + length-prefixed packets)
pub struct OpusStreamDecoder {
    buffer: Vec<u8>,
    buffer_start: usize,
    decoder: Option<OpusDecoder>,
    scratch_buffer: Vec<i16>,
    sample_rate: Option<u32>,
    channels: Option<u8>,
    pre_skip_remaining: usize,
    header_parsed: bool,
}

impl Default for OpusStreamDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl OpusStreamDecoder {
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            buffer_start: 0,
            decoder: None,
            scratch_buffer: Vec::new(),
            sample_rate: None,
            channels: None,
            pre_skip_remaining: 0,
            header_parsed: false,
        }
    }

    pub fn init(&mut self) -> Result<(), String> {
        Ok(())
    }

    pub fn sample_rate(&self) -> Option<u32> {
        self.sample_rate
    }

    pub fn channels(&self) -> Option<u8> {
        self.channels
    }

    fn buffered_len(&self) -> usize {
        self.buffer.len().saturating_sub(self.buffer_start)
    }

    fn compact_before_append(&mut self) {
        if self.buffer_start == 0 {
            return;
        }
        let unread = self.buffered_len();
        if unread > 0 {
            self.buffer.copy_within(self.buffer_start.., 0);
        }
        self.buffer.truncate(unread);
        self.buffer_start = 0;
    }

    fn consume(&mut self, byte_count: usize) {
        self.buffer_start += byte_count;
        if self.buffer_start == self.buffer.len() {
            self.buffer.clear();
            self.buffer_start = 0;
        }
    }

    /// Add data and return decoded AudioData if a complete packet was decoded
    pub fn add(&mut self, data: &[u8]) -> Result<Option<AudioData>, String> {
        if self.buffered_len().saturating_add(data.len()) > MAX_OPUS_STREAM_BUFFER_BYTES {
            return Err(format!(
                "Opus stream exceeds the {MAX_OPUS_STREAM_BUFFER_BYTES} byte buffer budget"
            ));
        }
        if !data.is_empty() {
            self.compact_before_append();
            self.buffer.extend_from_slice(data);
        }

        // Parse header if not done yet
        if !self.header_parsed && self.buffered_len() >= 19 {
            let start = self.buffer_start;
            let header = &self.buffer[start..start + 19];
            if !header.starts_with(b"OpusHead") {
                return Err("Invalid Opus stream: missing OpusHead".to_string());
            }

            self.sample_rate = Some(u32::from_le_bytes([
                header[12], header[13], header[14], header[15],
            ]));

            let sample_rate = self.sample_rate.unwrap();
            if sample_rate == 0 {
                self.sample_rate = Some(48_000);
            }

            self.channels = Some(header[9]);
            let pre_skip = u16::from_le_bytes([header[10], header[11]]);
            self.pre_skip_remaining = pre_skip as usize * self.channels.unwrap() as usize;

            let channels = self.channels.unwrap();
            let decoder =
                OpusDecoder::new_full(self.sample_rate.unwrap() as usize, channels as usize)?;

            self.decoder = Some(decoder);
            self.scratch_buffer
                .resize(MAX_OPUS_FRAME_SAMPLES * channels as usize, 0);
            self.header_parsed = true;

            debug!(
                sample_rate_hz = self.sample_rate.unwrap(),
                channels = channels,
                pre_skip = pre_skip,
                "initialized Opus stream decoder"
            );

            self.consume(19);
        }

        // Try to decode a packet
        if self.buffered_len() >= 2 {
            let start = self.buffer_start;
            let packet_len =
                u16::from_le_bytes([self.buffer[start], self.buffer[start + 1]]) as usize;
            if packet_len == 0 {
                return Err("Opus stream contains a zero-length packet".to_string());
            }

            // Check if we have the complete packet
            if self.buffered_len() >= 2 + packet_len {
                let packet_start = start + 2;
                let packet_end = packet_start + packet_len;
                let (Some(decoder), Some(channels)) = (self.decoder.as_mut(), self.channels) else {
                    return Ok(None);
                };

                let decoded = decoder.decode_i16(
                    &self.buffer[packet_start..packet_end],
                    &mut self.scratch_buffer,
                    false,
                );
                match decoded {
                    Ok(samples_per_channel) if samples_per_channel > 0 => {
                        let mut frame_samples = samples_per_channel * channels as usize;
                        let mut start = 0;

                        // Handle pre-skip
                        if self.pre_skip_remaining > 0 {
                            let skip = self.pre_skip_remaining.min(frame_samples);
                            self.pre_skip_remaining -= skip;
                            start = skip;
                        }

                        frame_samples = frame_samples.saturating_sub(start);

                        self.consume(2 + packet_len);

                        if frame_samples > 0 {
                            // Convert to bytes
                            let mut pcm_bytes = Vec::with_capacity(frame_samples * 2);
                            for &sample in &self.scratch_buffer[start..start + frame_samples] {
                                pcm_bytes.extend_from_slice(&sample.to_le_bytes());
                            }

                            let audio_data = AudioData::new(
                                16,
                                channels,
                                self.sample_rate.unwrap(),
                                pcm_bytes,
                                EncodingFlag::PCMSigned,
                                Endianness::LittleEndian,
                            );

                            return Ok(Some(audio_data));
                        }
                    }
                    Ok(_) => {
                        self.consume(2 + packet_len);
                    }
                    Err(e) => {
                        self.consume(2 + packet_len);
                        return Err(e);
                    }
                }
            }
        }

        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use soundkit::audio_bytes::s16le_to_i16;
    use soundkit::test_utils::{print_waveform_with_header, DecodeResult};
    use soundkit::wav::WavStreamProcessor;
    use std::fs::{self, File};
    use std::io::Read;
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::sync::Once;
    use std::time::Instant;
    use tracing::debug;
    const TEST_FILE: &str = "A_Tusk_is_used_to_make_costly_gifts";

    #[derive(Debug)]
    struct RawOpusHeader {
        sample_rate: u32,
        channels: u8,
        pre_skip: u16,
    }

    fn init_tracing() {
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            let _ = tracing_subscriber::fmt()
                .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
                .with_test_writer()
                .try_init();
        });
    }

    fn testdata_path(file: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("testdata")
            .join(file)
    }

    fn golden_path(file: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("golden")
            .join(file)
    }

    #[test]
    fn opus_decoder_accepts_all_supported_decode_rates() {
        for sample_rate in [8_000, 12_000, 16_000, 24_000, 48_000] {
            let decoder = OpusDecoder::new(sample_rate, 1)
                .unwrap_or_else(|error| panic!("{sample_rate} Hz failed: {error}"));
            assert_eq!(decoder.sample_rate(), sample_rate as u32);
            assert_eq!(decoder.channels(), 1);
        }
    }

    #[test]
    fn opus_decoder_rejects_invalid_metadata_without_panicking() {
        for (sample_rate, channels) in [(44_100, 1), (48_000, 0), (48_000, 3)] {
            let result = std::panic::catch_unwind(|| OpusDecoder::new(sample_rate, channels));
            assert!(
                result.is_ok(),
                "OpusDecoder::new panicked for {sample_rate} Hz and {channels} channels"
            );
            assert!(
                result.unwrap().is_err(),
                "invalid Opus metadata was accepted for {sample_rate} Hz and {channels} channels"
            );
        }
    }

    fn parse_length_prefixed_opus(data: &[u8]) -> Result<(RawOpusHeader, Vec<&[u8]>), String> {
        if data.len() < 19 || !data.starts_with(b"OpusHead") {
            return Err("Missing OpusHead".to_string());
        }

        let header = RawOpusHeader {
            sample_rate: u32::from_le_bytes([data[12], data[13], data[14], data[15]]),
            channels: data[9],
            pre_skip: u16::from_le_bytes([data[10], data[11]]),
        };

        let mut packets = Vec::new();
        let mut cursor = &data[19..];
        while cursor.len() >= 2 {
            let len = u16::from_le_bytes([cursor[0], cursor[1]]) as usize;
            cursor = &cursor[2..];
            if len == 0 || cursor.len() < len {
                break;
            }
            let (packet, rest) = cursor.split_at(len);
            packets.push(packet);
            cursor = rest;
        }

        Ok((header, packets))
    }

    #[test]
    fn test_opus_roundtrip_48khz_synthetic() {
        const SAMPLE_RATE: u32 = 48_000;
        const CHANNELS: u32 = 2;
        const FRAME_SIZE: u32 = 960;

        let mut encoder = OpusEncoder::new(SAMPLE_RATE, 16, CHANNELS, FRAME_SIZE, 128_000);
        encoder.init().expect("Failed to initialize opus encoder");

        let mut decoder = OpusDecoder::new(SAMPLE_RATE as usize, CHANNELS as usize)
            .expect("failed to construct Opus decoder");
        decoder.init().expect("Decoder initialization failed");

        let input = (0..FRAME_SIZE as usize)
            .flat_map(|i| {
                let t = i as f32 / SAMPLE_RATE as f32;
                let left = (t * 440.0 * std::f32::consts::TAU).sin();
                let right = (t * 660.0 * std::f32::consts::TAU).sin();
                [
                    (left * i16::MAX as f32 * 0.25) as i16,
                    (right * i16::MAX as f32 * 0.25) as i16,
                ]
            })
            .collect::<Vec<_>>();

        let mut packet = vec![0u8; 4096];
        let encoded_len = encoder
            .encode_i16(&input, &mut packet)
            .expect("encoding failed");
        assert!(encoded_len > 0);

        let mut decoded = vec![0i16; input.len()];
        let samples_per_channel = decoder
            .decode_i16(&packet[..encoded_len], &mut decoded, false)
            .expect("decoding failed");

        assert_eq!(samples_per_channel, FRAME_SIZE as usize);
        assert!(decoded.iter().any(|sample| *sample != 0));
    }

    #[test]
    fn soundkit_cbr_packets_are_celt_and_full_decoder_compatible() {
        const SAMPLE_RATE: u32 = 48_000;
        const FRAME_SIZE: u32 = 960;
        const PACKETS: usize = 6;

        for bitrate in [64_000_u32, 192_000] {
            for channels in [1_u32, 2] {
                let mut encoder = OpusEncoder::new(SAMPLE_RATE, 16, channels, FRAME_SIZE, bitrate);
                encoder.init().unwrap();
                let mut fast_decoder =
                    OpusDecoder::new_celt_only(SAMPLE_RATE as usize, channels as usize).unwrap();
                let mut full_decoder =
                    OpusDecoder::new_full(SAMPLE_RATE as usize, channels as usize).unwrap();
                let mut packet = vec![0_u8; 4_096];
                let mut fast_output = vec![0.0_f32; FRAME_SIZE as usize * channels as usize];
                let mut full_output = vec![0.0_f32; FRAME_SIZE as usize * channels as usize];
                let expected_packet_bytes =
                    bitrate as usize * FRAME_SIZE as usize / SAMPLE_RATE as usize / 8;
                let mut first_packet = Vec::new();
                let mut fast_energy = 0.0_f64;
                let mut full_energy = 0.0_f64;

                for packet_index in 0..PACKETS {
                    let pcm = (0..FRAME_SIZE as usize)
                        .flat_map(|frame| {
                            let absolute_frame = packet_index * FRAME_SIZE as usize + frame;
                            let phase = absolute_frame as f32 * 440.0 * std::f32::consts::TAU
                                / SAMPLE_RATE as f32;
                            let sample = (phase.sin() * i16::MAX as f32 * 0.25) as i16;
                            std::iter::repeat_n(sample, channels as usize)
                        })
                        .collect::<Vec<_>>();
                    let packet_len = encoder.encode_i16(&pcm, &mut packet).unwrap();

                    assert_eq!(packet_len, expected_packet_bytes);
                    assert_eq!(opus_packet_mode(packet[0]), OpusPacketMode::CeltOnly);
                    assert_eq!(if packet[0] & 0x04 == 0 { 1 } else { 2 }, channels);
                    if packet_index == 0 {
                        first_packet.extend_from_slice(&packet[..packet_len]);
                    }
                    assert_eq!(
                        fast_decoder
                            .decode_f32(&packet[..packet_len], &mut fast_output, false)
                            .unwrap(),
                        FRAME_SIZE as usize
                    );
                    assert_eq!(
                        full_decoder
                            .decode_f32(&packet[..packet_len], &mut full_output, false)
                            .unwrap(),
                        FRAME_SIZE as usize
                    );
                    fast_energy += fast_output
                        .iter()
                        .map(|sample| f64::from(*sample) * f64::from(*sample))
                        .sum::<f64>();
                    full_energy += full_output
                        .iter()
                        .map(|sample| f64::from(*sample) * f64::from(*sample))
                        .sum::<f64>();
                }

                assert!(fast_energy.is_finite() && fast_energy > 0.01);
                assert!(full_energy.is_finite() && full_energy > 0.01);

                encoder.reset().unwrap();
                let first_pcm = (0..FRAME_SIZE as usize)
                    .flat_map(|frame| {
                        let phase =
                            frame as f32 * 440.0 * std::f32::consts::TAU / SAMPLE_RATE as f32;
                        let sample = (phase.sin() * i16::MAX as f32 * 0.25) as i16;
                        std::iter::repeat_n(sample, channels as usize)
                    })
                    .collect::<Vec<_>>();
                let reset_packet_len = encoder.encode_i16(&first_pcm, &mut packet).unwrap();
                assert_eq!(&packet[..reset_packet_len], first_packet);
            }
        }
    }

    #[test]
    fn test_opus_roundtrip_preserves_48khz_sine_pitch() {
        const SAMPLE_RATE: u32 = 48_000;
        const CHANNELS: u32 = 2;
        const FRAME_SIZE: u32 = 960;
        const FRAMES: usize = 50;

        for frequency_hz in [220.0_f64, 440.0, 1_000.0] {
            let mut encoder = OpusEncoder::new(SAMPLE_RATE, 16, CHANNELS, FRAME_SIZE, 128_000);
            encoder.init().expect("Failed to initialize opus encoder");

            let mut decoder = OpusDecoder::new(SAMPLE_RATE as usize, CHANNELS as usize)
                .expect("failed to construct Opus decoder");
            decoder.init().expect("Decoder initialization failed");

            let input = (0..FRAME_SIZE as usize * FRAMES)
                .flat_map(|i| {
                    let phase =
                        i as f64 * frequency_hz * std::f64::consts::TAU / SAMPLE_RATE as f64;
                    let sample = (phase.sin() * i16::MAX as f64 * 0.35).round() as i16;
                    [sample, sample]
                })
                .collect::<Vec<_>>();

            let mut decoded = Vec::with_capacity(input.len());
            for frame in input.chunks_exact(FRAME_SIZE as usize * CHANNELS as usize) {
                let mut packet = vec![0u8; 4096];
                let encoded_len = encoder
                    .encode_i16(frame, &mut packet)
                    .expect("encoding failed");
                assert!(encoded_len > 0);

                let mut pcm = vec![0i16; FRAME_SIZE as usize * CHANNELS as usize];
                let samples_per_channel = decoder
                    .decode_i16(&packet[..encoded_len], &mut pcm, false)
                    .expect("decoding failed");
                assert_eq!(samples_per_channel, FRAME_SIZE as usize);
                decoded.extend_from_slice(&pcm);
            }

            let left = decoded
                .chunks_exact(CHANNELS as usize)
                .map(|frame| frame[0] as f64 / i16::MAX as f64)
                .collect::<Vec<_>>();
            let estimated = estimate_frequency_hz(&left[FRAME_SIZE as usize..], SAMPLE_RATE);
            assert!(
                (estimated - frequency_hz).abs() < 2.0,
                "{frequency_hz}Hz sine decoded as {estimated:.2}Hz"
            );
        }
    }

    fn estimate_frequency_hz(samples: &[f64], sample_rate: u32) -> f64 {
        let mut crossings = Vec::new();
        for idx in 1..samples.len() {
            let previous = samples[idx - 1];
            let current = samples[idx];
            if previous <= 0.0 && current > 0.0 {
                let denom = current - previous;
                let frac = if denom.abs() <= f64::EPSILON {
                    0.0
                } else {
                    -previous / denom
                };
                crossings.push(idx as f64 - 1.0 + frac);
            }
        }

        if crossings.len() < 2 {
            return 0.0;
        }

        let mean_period = crossings
            .windows(2)
            .map(|pair| pair[1] - pair[0])
            .sum::<f64>()
            / (crossings.len() - 1) as f64;
        if mean_period <= f64::EPSILON {
            0.0
        } else {
            sample_rate as f64 / mean_period
        }
    }

    #[test]
    fn test_opus_decodes_silk_fixture_waveform() {
        let input_path = testdata_path(&format!("opus/{}.opus", TEST_FILE));
        let opus_bytes = fs::read(&input_path).unwrap();
        assert!(!opus_bytes.is_empty(), "fixture opus missing or empty");

        init_tracing();

        let (header, packets) =
            parse_length_prefixed_opus(&opus_bytes).expect("failed to parse opus fixture");
        assert!(
            packets
                .iter()
                .any(|packet| packet.first().is_some_and(|toc| toc & 0x80 == 0)),
            "fixture must exercise a SILK or hybrid Opus packet"
        );

        let mut decoder =
            OpusDecoder::new_full(header.sample_rate as usize, header.channels as usize)
                .expect("failed to construct Opus decoder");
        decoder.init().expect("Decoder initialization failed");

        let mut decoded = Vec::new();
        let mut scratch = vec![0i16; MAX_OPUS_FRAME_SAMPLES * header.channels as usize];
        let mut pre_skip = header.pre_skip as usize * header.channels as usize;

        for packet in packets {
            let samples_per_channel = decoder.decode_i16(packet, &mut scratch, false).unwrap();
            if samples_per_channel == 0 {
                continue;
            }

            let mut frame_samples = samples_per_channel * header.channels as usize;
            let mut start = 0;
            if pre_skip > 0 {
                let skip = pre_skip.min(frame_samples);
                pre_skip -= skip;
                start = skip;
            }

            frame_samples = frame_samples.saturating_sub(start);
            if frame_samples == 0 {
                continue;
            }

            decoded.extend_from_slice(&scratch[start..start + frame_samples]);
        }

        assert!(!decoded.is_empty(), "decoder produced no PCM samples");

        let result = DecodeResult::new(&decoded, decoder.sample_rate(), decoder.channels());
        print_waveform_with_header("Opus", &result);
    }

    #[test]
    fn default_48khz_decoder_accepts_silk_packets() {
        let input_path = testdata_path("opus/A_Tusk_is_used_to_make_costly_gifts.opus");
        let opus_bytes = fs::read(&input_path).unwrap();
        let (header, packets) =
            parse_length_prefixed_opus(&opus_bytes).expect("failed to parse opus fixture");
        assert!(packets.iter().any(|packet| {
            packet
                .first()
                .is_some_and(|toc| opus_packet_mode(*toc) != OpusPacketMode::CeltOnly)
        }));

        let mut decoder = OpusDecoder::new(48_000, header.channels as usize)
            .expect("failed to construct default 48 kHz Opus decoder");
        let mut scratch = vec![0i16; MAX_OPUS_FRAME_SAMPLES * header.channels as usize];
        let mut decoded_frames = 0usize;
        for packet in packets {
            decoded_frames += decoder.decode_i16(packet, &mut scratch, false).unwrap();
        }

        assert!(decoded_frames > 0);
    }

    #[test]
    fn test_opus_decoder_streaming_decode() {
        // decode the real fixture opus stream; it is already length-prefixed
        let input_path = testdata_path("opus/A_Tusk_is_used_to_make_costly_gifts.opus");
        let opus_bytes = fs::read(&input_path).unwrap();
        assert!(!opus_bytes.is_empty(), "fixture opus missing or empty");

        init_tracing();

        let (header, packets) =
            parse_length_prefixed_opus(&opus_bytes).expect("failed to parse opus fixture");

        const MAX_OPUS_FRAME_SAMPLES: usize = 5760; // 120 ms @ 48kHz
        let mut decoder = OpusDecoder::new(header.sample_rate as usize, header.channels as usize)
            .expect("failed to construct Opus decoder");
        decoder.init().expect("Decoder initialization failed");

        let mut decoded = Vec::new();
        let mut scratch = vec![0i16; MAX_OPUS_FRAME_SAMPLES * header.channels as usize];
        let mut pre_skip = header.pre_skip as usize * header.channels as usize;

        for packet in packets {
            let samples_per_channel = decoder.decode_i16(packet, &mut scratch, false).unwrap();
            if samples_per_channel == 0 {
                continue;
            }

            let mut frame_samples = samples_per_channel * header.channels as usize;
            let mut start = 0;
            if pre_skip > 0 {
                let skip = pre_skip.min(frame_samples);
                pre_skip -= skip;
                start = skip;
            }

            frame_samples = frame_samples.saturating_sub(start);
            if frame_samples == 0 {
                continue;
            }

            decoded.extend_from_slice(&scratch[start..start + frame_samples]);
        }

        assert!(!decoded.is_empty(), "decoder produced no PCM samples");
        assert_eq!(decoder.sample_rate(), 16_000);
        assert_eq!(decoder.channels(), 1);
        assert!(decoded.iter().any(|sample| *sample != 0));
    }

    fn run_opus_encoder_with_wav_file(
        file_path: &Path,
        encoded_output: &Path,
        decoded_output: &Path,
    ) {
        let mut file = File::open(file_path).unwrap();
        let mut file_buffer = Vec::new();
        file.read_to_end(&mut file_buffer).unwrap();

        let mut processor = WavStreamProcessor::new();
        let audio_data = processor.add(&file_buffer).unwrap().unwrap();

        init_tracing();
        debug!(
            bits_per_sample = audio_data.bits_per_sample(),
            sample_rate_hz = audio_data.sampling_rate(),
            channels = audio_data.channel_count(),
            "loaded WAV fixture"
        );

        let mut decoder = OpusDecoder::new(
            audio_data.sampling_rate() as usize,
            audio_data.channel_count() as usize,
        )
        .expect("failed to construct Opus decoder");
        decoder.init().expect("Decoder initialization failed");

        let frame_size = std::cmp::max(1, (audio_data.sampling_rate() / 50) as usize);

        let mut encoder = OpusEncoder::new(
            audio_data.sampling_rate(),
            audio_data.bits_per_sample() as u32,
            audio_data.channel_count() as u32,
            frame_size as u32,
            128_000,
        );
        encoder.init().expect("Failed to initialize opus encoder");

        let i16_samples = match audio_data.bits_per_sample() {
            16 => s16le_to_i16(audio_data.data()),
            _ => {
                unreachable!()
            }
        };

        let mut encoded_data = Vec::new();
        let chunk_size = frame_size * audio_data.channel_count() as usize;
        let mut decoded_samples = vec![0i16; chunk_size * 2];
        let mut output = Vec::new();
        for (i, chunk) in i16_samples.chunks(chunk_size).enumerate() {
            let start_time = Instant::now();
            let mut output_buffer = vec![0u8; chunk.len() * std::mem::size_of::<i32>() * 2];
            match encoder.encode_i16(chunk, &mut output_buffer) {
                Ok(encoded_len) => {
                    if encoded_len > 0 {
                        let elapsed_time = start_time.elapsed();
                        debug!(
                            chunk = i,
                            encoded_len,
                            elapsed_micros = elapsed_time.as_micros() as u64,
                            "encoded chunk"
                        );
                        match decoder.decode_i16(
                            &output_buffer[..encoded_len],
                            &mut decoded_samples,
                            false,
                        ) {
                            Ok(samples_read) => {
                                debug!(chunk = i, samples_read, encoded_len, "decoded opus chunk");
                                for sample in &decoded_samples
                                    [..samples_read * audio_data.channel_count() as usize]
                                {
                                    output.extend(sample.to_le_bytes());
                                }
                            }
                            Err(e) => panic!("Decoding failed: {}", e),
                        }
                    }
                    encoded_data.extend_from_slice(&output_buffer[..encoded_len]);
                }
                Err(e) => {
                    panic!("Failed to encode chunk {}: {:?}", i, e);
                }
            }
        }

        fs::create_dir_all(encoded_output.parent().unwrap()).unwrap();
        let mut file = File::create(encoded_output).expect("Failed to create output file");
        file.write_all(&encoded_data)
            .expect("Failed to write to output file");
        let mut file = File::create(decoded_output).expect("Failed to create output file");
        file.write_all(&output[..])
            .expect("Failed to write to output file");

        encoder.reset().expect("Failed to reset encoder");
    }

    #[test]
    #[ignore = "writes diagnostic golden files"]
    fn test_opus_encoder_with_wave_16bit() {
        run_opus_encoder_with_wav_file(
            &testdata_path("wav_stereo/A_Tusk_is_used_to_make_costly_gifts.wav"),
            &golden_path("opus/A_Tusk_is_used_to_make_costly_gifts_encoded.opus"),
            &golden_path("opus/A_Tusk_is_used_to_make_costly_gifts_decoded_from_opus.wav"),
        );
    }
}
