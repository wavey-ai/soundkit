#![deny(unsafe_op_in_unsafe_fn)]

mod analysis;
pub mod celt;
pub mod constants;
pub mod decoder;
pub mod encoder;
pub mod error;
mod kernels;
mod packet;
mod repacketizer;
mod soft_clip;
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
mod wasm;

pub use constants::{PCM_I24_MAX, PCM_I24_MIN};
pub use decoder::Decoder;
pub use encoder::{
    Application, Encoder, CELT_FRAME_SIZES_48K, CELT_MAX_BITRATE, CELT_MAX_FRAME_BYTES,
    CELT_MIN_BITRATE, CELT_MIN_FRAME_BYTES,
};
pub use error::{Error, Result};
pub use packet::*;
pub use repacketizer::*;
pub use soft_clip::*;

use frame_header::{EncodingFlag, Endianness};
use soundkit::audio_packet::Decoder as SoundkitDecoderTrait;
use soundkit::audio_types::AudioData;
use tracing::debug;

type AdapterResult<T> = std::result::Result<T, String>;

/// Backwards-compatible names for the authored codec types. These are aliases,
/// not adapter structs, so calls execute directly on the codec state.
pub type OpusEncoder = Encoder;
pub type OpusDecoder = Decoder;

/// Incremental encoder for SoundKit's raw Opus stream format
/// (`OpusHead` followed by little-endian `u16` packet lengths).
///
/// This is a framing layer over the authored [`Encoder`], not a second codec
/// object. PCM may arrive in arbitrary chunk sizes; complete codec frames are
/// encoded immediately into caller-owned stream storage.
pub struct OpusStreamEncoder {
    encoder: Encoder,
    sample_rate: u32,
    channels: usize,
    bits_per_sample: u32,
    frame_size: usize,
    pcm_i16: Vec<i16>,
    pcm_i24: Vec<i32>,
    pcm_start: usize,
    packet: Vec<u8>,
    header_emitted: bool,
}

impl OpusStreamEncoder {
    pub fn new(
        sample_rate: u32,
        bits_per_sample: u32,
        channels: u32,
        frame_size: u32,
        bitrate: u32,
    ) -> AdapterResult<Self> {
        let encoder =
            Encoder::try_new(sample_rate, bits_per_sample, channels, frame_size, bitrate)?;
        Ok(Self {
            encoder,
            sample_rate,
            channels: channels as usize,
            bits_per_sample,
            frame_size: frame_size as usize,
            pcm_i16: Vec::new(),
            pcm_i24: Vec::new(),
            pcm_start: 0,
            packet: Vec::with_capacity(CELT_MAX_FRAME_BYTES + 1),
            header_emitted: false,
        })
    }

    pub const fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub const fn channels(&self) -> usize {
        self.channels
    }

    pub const fn frame_size(&self) -> usize {
        self.frame_size
    }

    pub const fn pending_samples(&self) -> usize {
        match self.bits_per_sample {
            16 => self.pcm_i16.len().saturating_sub(self.pcm_start),
            24 => self.pcm_i24.len().saturating_sub(self.pcm_start),
            _ => 0,
        }
    }

    pub fn reset(&mut self) -> AdapterResult<()> {
        self.encoder.reset()?;
        self.pcm_i16.clear();
        self.pcm_i24.clear();
        self.pcm_start = 0;
        self.packet.clear();
        self.header_emitted = false;
        Ok(())
    }

    /// Adds interleaved PCM16 samples and appends every complete encoded packet
    /// to `output`. Returns the number of codec packets appended.
    pub fn add_i16(&mut self, input: &[i16], output: &mut Vec<u8>) -> AdapterResult<usize> {
        if self.bits_per_sample != 16 {
            return Err(format!(
                "PCM16 input does not match the configured {}-bit Opus stream",
                self.bits_per_sample
            ));
        }
        self.compact_i16();
        self.pcm_i16.extend_from_slice(input);
        self.emit_header(output);

        let samples_per_packet = self.frame_size * self.channels;
        let mut packets = 0;
        while self.pcm_i16.len() - self.pcm_start >= samples_per_packet {
            let end = self.pcm_start + samples_per_packet;
            let packet_len = self
                .encoder
                .encode_i16_into(
                    &self.pcm_i16[self.pcm_start..end],
                    self.frame_size,
                    &mut self.packet,
                )
                .map_err(|error| error.to_string())?;
            append_length_prefixed_packet(output, &self.packet[..packet_len])?;
            self.pcm_start = end;
            packets += 1;
        }
        self.clear_consumed_i16();
        Ok(packets)
    }

    /// Adds signed 24-bit samples stored sign-extended in `i32` values.
    pub fn add_i24(&mut self, input: &[i32], output: &mut Vec<u8>) -> AdapterResult<usize> {
        if self.bits_per_sample != 24 {
            return Err(format!(
                "24-bit input does not match the configured {}-bit Opus stream",
                self.bits_per_sample
            ));
        }
        self.compact_i24();
        self.pcm_i24.extend_from_slice(input);
        self.emit_header(output);

        let samples_per_packet = self.frame_size * self.channels;
        let mut packets = 0;
        while self.pcm_i24.len() - self.pcm_start >= samples_per_packet {
            let end = self.pcm_start + samples_per_packet;
            let packet_len = self
                .encoder
                .encode_i24_into(
                    &self.pcm_i24[self.pcm_start..end],
                    self.frame_size,
                    &mut self.packet,
                )
                .map_err(|error| error.to_string())?;
            append_length_prefixed_packet(output, &self.packet[..packet_len])?;
            self.pcm_start = end;
            packets += 1;
        }
        self.clear_consumed_i24();
        Ok(packets)
    }

    /// Finishes a stream without inventing padded audio. Callers must provide a
    /// whole number of interleaved codec frames.
    pub fn finish(&mut self, output: &mut Vec<u8>) -> AdapterResult<()> {
        self.emit_header(output);
        let pending = self.pending_samples();
        if pending != 0 {
            return Err(format!(
                "Opus stream ended with {pending} unencoded samples; a complete frame requires {}",
                self.frame_size * self.channels
            ));
        }
        Ok(())
    }

    fn emit_header(&mut self, output: &mut Vec<u8>) {
        if self.header_emitted {
            return;
        }
        output.extend_from_slice(b"OpusHead");
        output.push(1);
        output.push(self.channels as u8);
        output.extend_from_slice(&0_u16.to_le_bytes());
        output.extend_from_slice(&self.sample_rate.to_le_bytes());
        output.extend_from_slice(&0_i16.to_le_bytes());
        output.push(0);
        self.header_emitted = true;
    }

    fn compact_i16(&mut self) {
        if self.pcm_start == 0 {
            return;
        }
        let unread = self.pcm_i16.len() - self.pcm_start;
        self.pcm_i16.copy_within(self.pcm_start.., 0);
        self.pcm_i16.truncate(unread);
        self.pcm_start = 0;
    }

    fn clear_consumed_i16(&mut self) {
        if self.pcm_start == self.pcm_i16.len() {
            self.pcm_i16.clear();
            self.pcm_start = 0;
        }
    }

    fn compact_i24(&mut self) {
        if self.pcm_start == 0 {
            return;
        }
        let unread = self.pcm_i24.len() - self.pcm_start;
        self.pcm_i24.copy_within(self.pcm_start.., 0);
        self.pcm_i24.truncate(unread);
        self.pcm_start = 0;
    }

    fn clear_consumed_i24(&mut self) {
        if self.pcm_start == self.pcm_i24.len() {
            self.pcm_i24.clear();
            self.pcm_start = 0;
        }
    }
}

fn append_length_prefixed_packet(output: &mut Vec<u8>, packet: &[u8]) -> AdapterResult<()> {
    let packet_len = u16::try_from(packet.len()).map_err(|_| {
        format!(
            "Opus packet is too large for the raw stream: {} bytes",
            packet.len()
        )
    })?;
    output.extend_from_slice(&packet_len.to_le_bytes());
    output.extend_from_slice(packet);
    Ok(())
}

const MAX_OPUS_FRAME_SAMPLES: usize = 5760; // 120 ms @ 48 kHz
const MAX_OPUS_STREAM_BUFFER_BYTES: usize = 4 * 1024 * 1024;

/// Streaming decoder for raw Opus format (OpusHead + length-prefixed packets)
pub struct OpusStreamDecoder {
    buffer: Vec<u8>,
    buffer_start: usize,
    decoder: Option<Decoder>,
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
    /// Creates a streaming decoder for streams emitted by SoundKit's authored
    /// encoder. The decoder is fully SoundKit-authored and CELT-only.
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

    /// Explicit constructor for raw streams emitted by [`OpusEncoder`].
    pub fn for_soundkit_stream() -> Self {
        Self::new()
    }

    pub fn init(&mut self) -> AdapterResult<()> {
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

    /// Adds arbitrary raw-stream bytes and appends all complete decoded PCM16
    /// packets to caller-owned storage. Returns the interleaved sample count.
    pub fn decode_i16_into(&mut self, data: &[u8], output: &mut Vec<i16>) -> AdapterResult<usize> {
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
            let decoder = Decoder::new(self.sample_rate.unwrap() as i32, channels as usize)
                .map_err(|error| error.to_string())?;

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

        let output_start = output.len();
        loop {
            if self.buffered_len() < 2 {
                break;
            }
            let start = self.buffer_start;
            let packet_len =
                u16::from_le_bytes([self.buffer[start], self.buffer[start + 1]]) as usize;
            if packet_len == 0 {
                return Err("Opus stream contains a zero-length packet".to_string());
            }
            if self.buffered_len() < 2 + packet_len {
                break;
            }

            let packet_start = start + 2;
            let packet_end = packet_start + packet_len;
            let (Some(decoder), Some(channels)) = (self.decoder.as_mut(), self.channels) else {
                break;
            };
            let decoded = decoder
                .decode_i16(
                    &self.buffer[packet_start..packet_end],
                    &mut self.scratch_buffer,
                    false,
                )
                .map_err(|error| {
                    format!("failed to decode raw Opus packet at byte {packet_start}: {error}")
                })?;

            let frame_samples = decoded * channels as usize;
            let skip = self.pre_skip_remaining.min(frame_samples);
            self.pre_skip_remaining -= skip;
            output.extend_from_slice(&self.scratch_buffer[skip..frame_samples]);
            self.consume(2 + packet_len);
        }

        Ok(output.len() - output_start)
    }

    /// Legacy `AudioData` adapter. New hot paths should use
    /// [`Self::decode_i16_into`] and retain their output allocation.
    pub fn add(&mut self, data: &[u8]) -> AdapterResult<Option<AudioData>> {
        let mut pcm = Vec::new();
        if self.decode_i16_into(data, &mut pcm)? == 0 {
            return Ok(None);
        }
        let Some(channels) = self.channels else {
            return Ok(None);
        };
        let mut pcm_bytes = Vec::with_capacity(pcm.len() * 2);
        for sample in pcm {
            pcm_bytes.extend_from_slice(&sample.to_le_bytes());
        }
        Ok(Some(AudioData::new(
            16,
            channels,
            self.sample_rate.unwrap_or(48_000),
            pcm_bytes,
            EncodingFlag::PCMSigned,
            Endianness::LittleEndian,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use soundkit::audio_bytes::s16le_to_i16;
    use soundkit::wav::WavStreamProcessor;
    use std::fs::{self, File};
    use std::io::Read;
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::sync::Once;
    use std::time::Instant;
    use tracing::debug;

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

    #[test]
    fn decoder_is_owned_celt_only() {
        let decoder = OpusDecoder::new(48_000, 2).unwrap();
        assert_eq!(decoder.sample_rate(), 48_000);
        assert_eq!(decoder.channels(), 2);
        assert!(OpusDecoder::new(16_000, 1).is_err());
    }

    #[test]
    fn test_opus_roundtrip_48khz_synthetic() {
        const SAMPLE_RATE: u32 = 48_000;
        const CHANNELS: u32 = 2;
        const FRAME_SIZE: u32 = 960;

        let mut encoder = OpusEncoder::new(SAMPLE_RATE, 16, CHANNELS, FRAME_SIZE, 128_000);
        encoder.init().expect("Failed to initialize opus encoder");

        let mut decoder = OpusDecoder::new(SAMPLE_RATE as i32, CHANNELS as usize)
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
    fn authored_stream_encoder_and_decoder_accept_arbitrary_chunks() {
        const SAMPLE_RATE: u32 = 48_000;
        const CHANNELS: u32 = 2;
        const FRAME_SIZE: u32 = 240;
        const PACKETS: usize = 4;

        let input = (0..PACKETS * FRAME_SIZE as usize)
            .flat_map(|frame| {
                let phase = frame as f32 * 440.0 * std::f32::consts::TAU / SAMPLE_RATE as f32;
                let sample = (phase.sin() * i16::MAX as f32 * 0.25) as i16;
                [sample, sample]
            })
            .collect::<Vec<_>>();

        let mut encoder =
            OpusStreamEncoder::new(SAMPLE_RATE, 16, CHANNELS, FRAME_SIZE, 128_000).unwrap();
        let mut stream = Vec::new();
        let mut encoded_packets = 0;
        for chunk in input.chunks(137) {
            encoded_packets += encoder.add_i16(chunk, &mut stream).unwrap();
        }
        encoder.finish(&mut stream).unwrap();
        assert_eq!(encoded_packets, PACKETS);
        assert!(stream.starts_with(b"OpusHead"));

        let mut decoder = OpusStreamDecoder::new();
        let mut decoded = Vec::new();
        for chunk in stream.chunks(17) {
            decoder.decode_i16_into(chunk, &mut decoded).unwrap();
        }

        assert_eq!(decoder.sample_rate(), Some(SAMPLE_RATE));
        assert_eq!(decoder.channels(), Some(CHANNELS as u8));
        assert_eq!(
            decoded.len(),
            PACKETS * FRAME_SIZE as usize * CHANNELS as usize
        );
        assert!(decoded.iter().any(|sample| *sample != 0));
    }

    #[test]
    fn soundkit_trait_output_matches_low_level_api_for_all_pcm_types() {
        const SAMPLE_RATE: u32 = 48_000;
        const CHANNELS: usize = 2;
        const FRAME_SIZE: usize = 240;
        const PACKETS: usize = 8;

        let mut encoder =
            OpusEncoder::new(SAMPLE_RATE, 24, CHANNELS as u32, FRAME_SIZE as u32, 256_000);
        encoder.init().unwrap();
        let mut packets = Vec::with_capacity(PACKETS);
        for packet_index in 0..PACKETS {
            let input = (0..FRAME_SIZE)
                .flat_map(|frame| {
                    let absolute_frame = packet_index * FRAME_SIZE + frame;
                    let phase =
                        absolute_frame as f32 * 997.0 * std::f32::consts::TAU / SAMPLE_RATE as f32;
                    let sample = (phase.sin() * 4_000_000.0) as i32;
                    [sample, -sample]
                })
                .collect::<Vec<_>>();
            let mut packet = vec![0_u8; 4_096];
            let packet_len = encoder.encode_i32(&input, &mut packet).unwrap();
            packet.truncate(packet_len);
            packets.push(packet);
        }

        let mut core = Decoder::new(SAMPLE_RATE as i32, CHANNELS).unwrap();
        let mut soundkit = OpusDecoder::new(SAMPLE_RATE as i32, CHANNELS).unwrap();
        for packet in &packets {
            let mut expected = Vec::new();
            assert_eq!(
                core.decode_i16_into(packet, false, &mut expected).unwrap(),
                FRAME_SIZE
            );
            let mut actual = vec![i16::MIN; FRAME_SIZE * CHANNELS + 8];
            assert_eq!(
                soundkit.decode_i16(packet, &mut actual, false).unwrap(),
                FRAME_SIZE
            );
            assert_eq!(&actual[..FRAME_SIZE * CHANNELS], expected);
            assert!(actual[FRAME_SIZE * CHANNELS..]
                .iter()
                .all(|sample| *sample == i16::MIN));
        }

        let mut core = Decoder::new(SAMPLE_RATE as i32, CHANNELS).unwrap();
        let mut soundkit = OpusDecoder::new(SAMPLE_RATE as i32, CHANNELS).unwrap();
        for packet in &packets {
            let mut expected = Vec::new();
            assert_eq!(
                core.decode_i24_into(packet, false, &mut expected).unwrap(),
                FRAME_SIZE
            );
            let mut actual = vec![i32::MIN; FRAME_SIZE * CHANNELS + 8];
            assert_eq!(
                soundkit.decode_i32(packet, &mut actual, false).unwrap(),
                FRAME_SIZE
            );
            assert_eq!(&actual[..FRAME_SIZE * CHANNELS], expected);
            assert!(actual[FRAME_SIZE * CHANNELS..]
                .iter()
                .all(|sample| *sample == i32::MIN));
        }

        let mut core = Decoder::new(SAMPLE_RATE as i32, CHANNELS).unwrap();
        let mut soundkit = OpusDecoder::new(SAMPLE_RATE as i32, CHANNELS).unwrap();
        for packet in &packets {
            let mut expected = Vec::new();
            assert_eq!(
                core.decode_f32_into(packet, false, &mut expected).unwrap(),
                FRAME_SIZE
            );
            let mut actual = vec![f32::NAN; FRAME_SIZE * CHANNELS + 8];
            assert_eq!(
                soundkit.decode_f32(packet, &mut actual, false).unwrap(),
                FRAME_SIZE
            );
            assert_eq!(&actual[..FRAME_SIZE * CHANNELS], expected);
            assert!(actual[FRAME_SIZE * CHANNELS..]
                .iter()
                .all(|sample| sample.is_nan()));
        }
    }

    #[test]
    fn soundkit_cbr_packets_are_celt_and_owned_decoder_compatible() {
        const SAMPLE_RATE: u32 = 48_000;
        const FRAME_SIZE: u32 = 960;
        const PACKETS: usize = 6;

        for bitrate in [64_000_u32, 192_000] {
            for channels in [1_u32, 2] {
                let mut encoder = OpusEncoder::new(SAMPLE_RATE, 16, channels, FRAME_SIZE, bitrate);
                encoder.init().unwrap();
                let mut fast_decoder =
                    OpusDecoder::new(SAMPLE_RATE as i32, channels as usize).unwrap();
                let mut packet = vec![0_u8; 4_096];
                let mut fast_output = vec![0.0_f32; FRAME_SIZE as usize * channels as usize];
                let expected_packet_bytes =
                    bitrate as usize * FRAME_SIZE as usize / SAMPLE_RATE as usize / 8;
                let mut first_packet = Vec::new();
                let mut fast_energy = 0.0_f64;

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
                    assert_ne!(
                        packet[0] & 0x80,
                        0,
                        "authored encoder emitted non-CELT mode"
                    );
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
                    fast_energy += fast_output
                        .iter()
                        .map(|sample| f64::from(*sample) * f64::from(*sample))
                        .sum::<f64>();
                }

                assert!(fast_energy.is_finite() && fast_energy > 0.01);

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

            let mut decoder = OpusDecoder::new(SAMPLE_RATE as i32, CHANNELS as usize)
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
            audio_data.sampling_rate() as i32,
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
