//! Incremental FLAC encoding and decoding.
//!
//! The encoder emits complete FLAC frames and exposes the current `STREAMINFO`
//! metadata separately so seekable files and containers can backpatch it after
//! the final frame. The decoder accepts arbitrarily chunked native FLAC input.

use crate::decode::frame::FrameReader;
use crate::decode::metadata::{self, MetadataBlock, StreamInfo};
use crate::decode::{Block, Error as DecodeError};
use crate::frame::{EncodedFlacFrame, FlacFrameConfig, FlacFrameEncoder, FlacFrameError};
use std::io::{self, Cursor};

const FLAC_MARKER: &[u8; 4] = b"fLaC";
const MAX_INPUT_CHUNK_BYTES: usize = 4 * 1024 * 1024;
const MAX_BUFFERED_FRAME_BYTES: usize = 8 * 1024 * 1024;

/// Incremental encoder for a single FLAC stream.
pub struct Encoder {
    inner: FlacFrameEncoder,
    stream_header: Vec<u8>,
    emitted_stream_header: bool,
}

impl Encoder {
    /// Creates an encoder for a fixed stream geometry.
    pub fn new(config: FlacFrameConfig) -> Result<Self, FlacFrameError> {
        let inner = FlacFrameEncoder::new(config)?;
        let stream_header = inner.stream_header()?;
        Ok(Self {
            inner,
            stream_header,
            emitted_stream_header: false,
        })
    }

    /// Returns the immutable stream geometry.
    pub fn config(&self) -> FlacFrameConfig {
        self.inner.config()
    }

    /// Encodes one interleaved signed sample block.
    ///
    /// The first call prepends the metadata header, without the four-byte
    /// `fLaC` marker, to reserve exactly the bytes a container or file writer
    /// must eventually replace with [`Self::stream_header`]. Later calls append
    /// only raw FLAC frames.
    pub fn encode_i32(
        &mut self,
        interleaved: &[i32],
        output: &mut Vec<u8>,
    ) -> Result<usize, FlacFrameError> {
        let frame = self.inner.encode_i32_block(interleaved)?;
        self.append_frame(frame, output)
    }

    /// Encodes one interleaved signed 16-bit sample block.
    pub fn encode_i16(
        &mut self,
        interleaved: &[i16],
        output: &mut Vec<u8>,
    ) -> Result<usize, FlacFrameError> {
        let frame = self.inner.encode_i16(interleaved)?;
        self.append_frame(frame, output)
    }

    fn append_frame(
        &mut self,
        frame: EncodedFlacFrame,
        output: &mut Vec<u8>,
    ) -> Result<usize, FlacFrameError> {
        self.stream_header = self.inner.stream_header()?;
        let start = output.len();
        if !self.emitted_stream_header {
            output.extend_from_slice(&self.stream_header);
            self.emitted_stream_header = true;
        }
        output.extend_from_slice(&frame.payload);
        Ok(output.len() - start)
    }

    /// Finalizes and returns the metadata header without the `fLaC` marker.
    ///
    /// Encoding is frame-oriented, so this emits no additional audio bytes.
    pub fn finish(&mut self) -> Result<&[u8], FlacFrameError> {
        self.stream_header = self.inner.stream_header()?;
        Ok(&self.stream_header)
    }

    /// Returns the latest metadata header without the `fLaC` marker.
    pub fn stream_header(&self) -> &[u8] {
        &self.stream_header
    }

    /// Returns a native FLAC file header containing the marker and metadata.
    pub fn file_header(&self) -> Vec<u8> {
        let mut output = Vec::with_capacity(FLAC_MARKER.len() + self.stream_header.len());
        output.extend_from_slice(FLAC_MARKER);
        output.extend_from_slice(&self.stream_header);
        output
    }

    /// Starts a new independent stream with the same configuration.
    pub fn reset(&mut self) -> Result<(), FlacFrameError> {
        self.inner.reset();
        self.stream_header = self.inner.stream_header()?;
        self.emitted_stream_header = false;
        Ok(())
    }
}

#[derive(Debug)]
enum StreamState {
    Magic,
    MetadataHeader,
    MetadataPayload {
        block_type: u8,
        is_last: bool,
        remaining: usize,
        payload: Vec<u8>,
    },
    Frames,
}

/// Incremental decoder for a native FLAC stream.
pub struct Decoder {
    input: Vec<u8>,
    input_start: usize,
    frame_buffer: Vec<i32>,
    pending_samples: Vec<i32>,
    pending_start: usize,
    stream_info: Option<StreamInfo>,
    state: StreamState,
    /// Decode straight from contiguous buffered bytes. This is the default:
    /// the slice-backed reader with bulk CRC verification measures faster
    /// than the interleaved streaming reader and passes the formal corpus
    /// gate. Set `WAVEY_FLAC_STREAMING_DECODE=1` to fall back to the older
    /// interleaved path.
    slice_decode: bool,
    /// When false, frame checksums are neither accumulated nor verified;
    /// see `set_verify_checksums`.
    verify_checksums: bool,
}

impl Default for Decoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Decoder {
    /// Creates a decoder waiting for a native `fLaC` stream marker.
    pub fn new() -> Self {
        Self {
            input: Vec::new(),
            input_start: 0,
            frame_buffer: Vec::new(),
            pending_samples: Vec::new(),
            pending_start: 0,
            stream_info: None,
            state: StreamState::Magic,
            // The slice path decodes each buffered frame in place and then
            // verifies its CRC-16 in one pass, which is faster than
            // interleaving checksum accumulation with bit reads. Opt back
            // into the streaming reader with WAVEY_FLAC_STREAMING_DECODE.
            slice_decode: std::env::var_os("WAVEY_FLAC_STREAMING_DECODE").is_none(),
            verify_checksums: true,
        }
    }

    /// Enables or disables frame checksum verification.
    ///
    /// By default, checksums are computed and verified while decoding.
    /// Disabling this skips that work, which makes decoding measurably
    /// faster. Only disable verification for streams whose bytes are known
    /// to be intact already, for example frames produced by the encoder in
    /// the same process.
    pub fn set_verify_checksums(&mut self, enabled: bool) {
        self.verify_checksums = enabled;
    }

    /// Returns parsed `STREAMINFO` once it has arrived.
    pub fn stream_info(&self) -> Option<StreamInfo> {
        self.stream_info
    }

    /// Returns compressed bytes retained for an incomplete metadata block or frame.
    pub fn buffered_bytes(&self) -> usize {
        self.input.len().saturating_sub(self.input_start)
    }

    /// Returns decoded samples waiting for caller-provided output space.
    pub fn pending_samples(&self) -> usize {
        self.pending_samples
            .len()
            .saturating_sub(self.pending_start)
    }

    /// Appends compressed input and writes as many interleaved samples as fit.
    ///
    /// Passing an empty input slice drains already-decoded samples and attempts
    /// to decode additional complete frames already buffered.
    pub fn decode_i32(
        &mut self,
        input: &[u8],
        output: &mut [i32],
    ) -> Result<usize, FlacFrameError> {
        self.append_input(input)?;
        self.consume_metadata()?;

        let mut written = self.copy_pending(output);
        while written < output.len() && matches!(self.state, StreamState::Frames) {
            let Some(block) = self.decode_one_frame()? else {
                break;
            };
            written += self.copy_block(&block, &mut output[written..]);
            self.frame_buffer = block.into_buffer();
        }
        Ok(written)
    }

    /// Appends compressed input and writes normalized floating-point samples.
    pub fn decode_f32(
        &mut self,
        input: &[u8],
        output: &mut [f32],
    ) -> Result<usize, FlacFrameError> {
        let mut scratch = vec![0_i32; output.len()];
        let written = self.decode_i32(input, &mut scratch)?;
        let bits = self
            .stream_info
            .map_or(16, |stream_info| stream_info.bits_per_sample);
        let scale = (1_i64 << (bits - 1)) as f32;
        for (target, sample) in output.iter_mut().zip(scratch).take(written) {
            *target = sample as f32 / scale;
        }
        Ok(written)
    }

    /// Verifies that the compressed stream ended at a frame boundary and all
    /// decoded samples have been consumed.
    pub fn finish(&self) -> Result<(), FlacFrameError> {
        if !matches!(self.state, StreamState::Frames) {
            return Err(FlacFrameError::Decode(
                "FLAC stream ended before metadata was complete".to_string(),
            ));
        }
        if self.buffered_bytes() != 0 {
            return Err(FlacFrameError::Decode(format!(
                "FLAC stream ended with {} bytes of an incomplete frame",
                self.buffered_bytes()
            )));
        }
        if self.pending_samples() != 0 {
            return Err(FlacFrameError::Decode(format!(
                "FLAC stream has {} decoded samples awaiting output",
                self.pending_samples()
            )));
        }
        Ok(())
    }

    /// Discards all state and waits for a new native FLAC stream.
    pub fn reset(&mut self) {
        self.input.clear();
        self.input_start = 0;
        self.frame_buffer.clear();
        self.pending_samples.clear();
        self.pending_start = 0;
        self.stream_info = None;
        self.state = StreamState::Magic;
    }

    fn append_input(&mut self, input: &[u8]) -> Result<(), FlacFrameError> {
        if input.len() > MAX_INPUT_CHUNK_BYTES {
            return Err(FlacFrameError::InvalidInput(format!(
                "FLAC input chunk exceeds the {MAX_INPUT_CHUNK_BYTES} byte streaming budget"
            )));
        }
        if self.buffered_bytes().saturating_add(input.len()) > MAX_BUFFERED_FRAME_BYTES {
            return Err(FlacFrameError::InvalidInput(format!(
                "FLAC frame exceeds the {MAX_BUFFERED_FRAME_BYTES} byte buffer budget"
            )));
        }
        self.compact_input();
        self.input.extend_from_slice(input);
        Ok(())
    }

    fn compact_input(&mut self) {
        if self.input_start == 0 {
            return;
        }
        if self.input_start == self.input.len() {
            self.input.clear();
            self.input_start = 0;
        } else if self.input_start >= 64 * 1024
            || self.input_start.saturating_mul(2) >= self.input.len()
        {
            self.input.copy_within(self.input_start.., 0);
            self.input.truncate(self.input.len() - self.input_start);
            self.input_start = 0;
        }
    }

    fn available_input(&self) -> &[u8] {
        &self.input[self.input_start..]
    }

    fn consume_input(&mut self, count: usize) -> Result<(), FlacFrameError> {
        if count > self.buffered_bytes() {
            return Err(FlacFrameError::Decode(
                "decoder attempted to consume beyond buffered FLAC input".to_string(),
            ));
        }
        self.input_start += count;
        if self.input_start == self.input.len() {
            self.input.clear();
            self.input_start = 0;
        }
        Ok(())
    }

    fn consume_metadata(&mut self) -> Result<(), FlacFrameError> {
        loop {
            let state = std::mem::replace(&mut self.state, StreamState::Frames);
            match state {
                StreamState::Magic => {
                    if self.buffered_bytes() < FLAC_MARKER.len() {
                        self.state = StreamState::Magic;
                        return Ok(());
                    }
                    if &self.available_input()[..FLAC_MARKER.len()] != FLAC_MARKER {
                        return Err(FlacFrameError::Decode(
                            "FLAC stream has no fLaC marker".to_string(),
                        ));
                    }
                    self.consume_input(FLAC_MARKER.len())?;
                    self.state = StreamState::MetadataHeader;
                }
                StreamState::MetadataHeader => {
                    if self.buffered_bytes() < 4 {
                        self.state = StreamState::MetadataHeader;
                        return Ok(());
                    }
                    let header: [u8; 4] = self.available_input()[..4]
                        .try_into()
                        .expect("four-byte metadata header");
                    self.consume_input(4)?;
                    let block_type = header[0] & 0x7f;
                    let is_last = header[0] & 0x80 != 0;
                    let remaining = ((header[1] as usize) << 16)
                        | ((header[2] as usize) << 8)
                        | header[3] as usize;
                    if block_type == 0 && remaining != 34 {
                        return Err(FlacFrameError::Decode(
                            "FLAC STREAMINFO must contain 34 bytes".to_string(),
                        ));
                    }
                    if self.stream_info.is_none() && block_type != 0 {
                        return Err(FlacFrameError::Decode(
                            "FLAC STREAMINFO must be the first metadata block".to_string(),
                        ));
                    }
                    self.state = StreamState::MetadataPayload {
                        block_type,
                        is_last,
                        remaining,
                        payload: Vec::with_capacity(if block_type == 0 { 34 } else { 0 }),
                    };
                }
                StreamState::MetadataPayload {
                    block_type,
                    is_last,
                    mut remaining,
                    mut payload,
                } => {
                    let consumed = remaining.min(self.buffered_bytes());
                    if block_type == 0 {
                        payload.extend_from_slice(&self.available_input()[..consumed]);
                    }
                    self.consume_input(consumed)?;
                    remaining -= consumed;
                    if remaining > 0 {
                        self.state = StreamState::MetadataPayload {
                            block_type,
                            is_last,
                            remaining,
                            payload,
                        };
                        return Ok(());
                    }
                    if block_type == 0 {
                        self.install_stream_info(&payload)?;
                    }
                    self.state = if is_last {
                        StreamState::Frames
                    } else {
                        StreamState::MetadataHeader
                    };
                }
                StreamState::Frames => {
                    self.state = StreamState::Frames;
                    return Ok(());
                }
            }
        }
    }

    fn install_stream_info(&mut self, payload: &[u8]) -> Result<(), FlacFrameError> {
        let mut cursor = Cursor::new(payload);
        let metadata = metadata::read_metadata_block(&mut cursor, 0, payload.len() as u32)
            .map_err(|error| FlacFrameError::Decode(error.to_string()))?;
        let MetadataBlock::StreamInfo(stream_info) = metadata else {
            return Err(FlacFrameError::Decode(
                "FLAC metadata parser did not return STREAMINFO".to_string(),
            ));
        };
        if stream_info.sample_rate == 0
            || !(1..=8).contains(&stream_info.channels)
            || !(1..=32).contains(&stream_info.bits_per_sample)
        {
            return Err(FlacFrameError::Decode(
                "FLAC STREAMINFO contains invalid audio geometry".to_string(),
            ));
        }
        self.stream_info = Some(stream_info);
        Ok(())
    }

    fn copy_pending(&mut self, output: &mut [i32]) -> usize {
        let count = self.pending_samples().min(output.len());
        output[..count]
            .copy_from_slice(&self.pending_samples[self.pending_start..self.pending_start + count]);
        self.pending_start += count;
        if self.pending_start == self.pending_samples.len() {
            self.pending_samples.clear();
            self.pending_start = 0;
        }
        count
    }

    fn decode_one_frame(&mut self) -> Result<Option<Block>, FlacFrameError> {
        if self.buffered_bytes() == 0 {
            return Ok(None);
        }
        let stream_info = self.stream_info.ok_or_else(|| {
            FlacFrameError::Decode("FLAC stream has no STREAMINFO block".to_string())
        })?;
        let frame_buffer = std::mem::take(&mut self.frame_buffer);

        if self.slice_decode {
            let decoded = crate::decode::frame::decode_frame_slice(
                self.available_input(),
                Some(stream_info.sample_rate),
                Some(stream_info.bits_per_sample),
                frame_buffer,
                self.verify_checksums,
            );
            return match decoded {
                Ok(Some((block, consumed))) => {
                    if consumed == 0 || consumed > self.buffered_bytes() {
                        return Err(FlacFrameError::Decode(
                            "FLAC decoder reported an invalid consumed range".to_string(),
                        ));
                    }
                    self.validate_frame(block, &stream_info, consumed)
                }
                Ok(None) => Ok(None),
                Err(DecodeError::IoError(error))
                    if error.kind() == io::ErrorKind::UnexpectedEof =>
                {
                    Ok(None)
                }
                Err(error) => Err(FlacFrameError::Decode(error.to_string())),
            };
        }

        let cursor = Cursor::new(self.available_input());
        let mut reader = FrameReader::with_stream_info(
            cursor,
            stream_info.sample_rate,
            stream_info.bits_per_sample,
        );
        if !self.verify_checksums {
            reader.set_verify_checksums(false);
        }
        let block = match reader.read_next_or_eof(frame_buffer) {
            Ok(Some(block)) => block,
            Ok(None) => return Ok(None),
            Err(DecodeError::IoError(error)) if error.kind() == io::ErrorKind::UnexpectedEof => {
                return Ok(None);
            }
            Err(error) => return Err(FlacFrameError::Decode(error.to_string())),
        };
        let consumed = reader.into_inner().position() as usize;

        if consumed == 0 || consumed > self.buffered_bytes() {
            return Err(FlacFrameError::Decode(
                "FLAC decoder reported an invalid consumed range".to_string(),
            ));
        }
        self.validate_frame(block, &stream_info, consumed)
    }

    /// Checks a decoded frame against the stream's declared format, then
    /// retires the bytes the frame spanned.
    fn validate_frame(
        &mut self,
        block: Block,
        stream_info: &StreamInfo,
        consumed: usize,
    ) -> Result<Option<Block>, FlacFrameError> {
        if block.channels() != stream_info.channels {
            return Err(FlacFrameError::FormatMismatch(format!(
                "decoded frame contains {} channels, STREAMINFO declares {}",
                block.channels(),
                stream_info.channels
            )));
        }
        if block.duration() > u32::from(stream_info.max_block_size) {
            return Err(FlacFrameError::FormatMismatch(format!(
                "decoded frame contains {} samples per channel, STREAMINFO permits at most {}",
                block.duration(),
                stream_info.max_block_size
            )));
        }
        self.consume_input(consumed)?;
        Ok(Some(block))
    }

    fn copy_block(&mut self, block: &Block, output: &mut [i32]) -> usize {
        let sample_count = block.duration() as usize * block.channels() as usize;
        let written = sample_count.min(output.len());

        // Whole blocks are the normal streaming case. Specialize mono and
        // stereo so the compiler can eliminate repeated channel arithmetic,
        // bounds checks, and the per-sample pending-output branch.
        if output.len() >= sample_count {
            match block.channels() {
                1 => output[..sample_count].copy_from_slice(block.channel(0)),
                2 => {
                    for (target, (left, right)) in output[..sample_count]
                        .chunks_exact_mut(2)
                        .zip(block.stereo_samples())
                    {
                        target[0] = left;
                        target[1] = right;
                    }
                }
                _ => {
                    let channels = block.channels();
                    for frame in 0..block.duration() {
                        for channel in 0..channels {
                            output[frame as usize * channels as usize + channel as usize] =
                                block.sample(channel, frame);
                        }
                    }
                }
            }
            return sample_count;
        }

        self.pending_samples.reserve(sample_count - written);
        let mut index = 0;
        for frame in 0..block.duration() {
            for channel in 0..block.channels() {
                let sample = block.sample(channel, frame);
                if index < written {
                    output[index] = sample;
                } else {
                    self.pending_samples.push(sample);
                }
                index += 1;
            }
        }
        written
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{FlacFrameConfig, FlacProfile};

    fn fixture() -> (FlacFrameConfig, Vec<i32>) {
        let config = FlacFrameConfig::new(48_000, 2, 16, 128, FlacProfile::Balanced).unwrap();
        let samples = (0..640)
            .map(|index| ((index as i32 * 977) % 65_536) - 32_768)
            .collect();
        (config, samples)
    }

    fn encode_file(config: FlacFrameConfig, samples: &[i32]) -> Vec<u8> {
        let mut encoder = Encoder::new(config).unwrap();
        let mut packets = Vec::new();
        for chunk in samples.chunks(config.sample_count().unwrap()) {
            encoder.encode_i32(chunk, &mut packets).unwrap();
        }
        let final_header = encoder.finish().unwrap().to_vec();
        let metadata_len = final_header.len();
        let mut file = FLAC_MARKER.to_vec();
        file.extend_from_slice(&final_header);
        file.extend_from_slice(&packets[metadata_len..]);
        file
    }

    #[test]
    fn round_trips_arbitrarily_chunked_input_and_small_outputs() {
        let (config, samples) = fixture();
        let file = encode_file(config, &samples);
        let mut decoder = Decoder::new();
        let mut decoded = Vec::new();
        let mut scratch = [0_i32; 17];
        for chunk in file.chunks(7) {
            let written = decoder.decode_i32(chunk, &mut scratch).unwrap();
            decoded.extend_from_slice(&scratch[..written]);
            loop {
                let written = decoder.decode_i32(&[], &mut scratch).unwrap();
                if written == 0 {
                    break;
                }
                decoded.extend_from_slice(&scratch[..written]);
            }
        }
        assert_eq!(decoder.stream_info().unwrap().sample_rate, 48_000);
        assert_eq!(decoder.stream_info().unwrap().channels, 2);
        assert_eq!(decoded, samples);
        decoder.finish().unwrap();
    }

    #[test]
    fn decode_without_checksum_verification_matches_verified_output() {
        let (config, samples) = fixture();
        let file = encode_file(config, &samples);

        let decode_all = |verify: bool| -> Vec<i32> {
            let mut decoder = Decoder::new();
            decoder.set_verify_checksums(verify);
            let mut decoded = Vec::new();
            let mut scratch = [0_i32; 13];
            for chunk in file.chunks(11) {
                let written = decoder.decode_i32(chunk, &mut scratch).unwrap();
                decoded.extend_from_slice(&scratch[..written]);
                loop {
                    let written = decoder.decode_i32(&[], &mut scratch).unwrap();
                    if written == 0 {
                        break;
                    }
                    decoded.extend_from_slice(&scratch[..written]);
                }
            }
            decoded
        };

        assert_eq!(decode_all(true), samples);
        assert_eq!(decode_all(false), samples);
    }

    #[test]
    fn whole_block_copy_handles_mono_and_multichannel_streams() {
        for channels in [1_u16, 3, 8] {
            let config =
                FlacFrameConfig::new(48_000, channels, 16, 128, FlacProfile::Balanced).unwrap();
            let samples: Vec<_> = (0..channels as usize * 384)
                .map(|index| ((index as i32 * 977) % 65_536) - 32_768)
                .collect();
            let file = encode_file(config, &samples);
            let mut decoder = Decoder::new();
            let mut decoded = vec![0_i32; samples.len()];

            assert_eq!(
                decoder.decode_i32(&file, &mut decoded).unwrap(),
                samples.len()
            );
            assert_eq!(decoded, samples);
            decoder.finish().unwrap();
        }
    }

    #[test]
    fn reports_corruption_and_truncated_final_frames() {
        let (config, samples) = fixture();
        let file = encode_file(config, &samples);
        let mut corrupted = file.clone();
        *corrupted.last_mut().unwrap() ^= 0x5a;
        let mut decoder = Decoder::new();
        let mut output = vec![0_i32; samples.len() * 2];
        assert!(decoder.decode_i32(&corrupted, &mut output).is_err());

        let mut decoder = Decoder::new();
        decoder
            .decode_i32(&file[..file.len() - 1], &mut output)
            .unwrap();
        assert!(decoder.finish().is_err());
    }

    #[test]
    fn reset_starts_a_new_native_stream() {
        let (config, samples) = fixture();
        let file = encode_file(config, &samples);
        let mut decoder = Decoder::new();
        let mut output = vec![0_i32; samples.len()];
        assert_eq!(
            decoder.decode_i32(&file, &mut output).unwrap(),
            samples.len()
        );
        decoder.finish().unwrap();
        decoder.reset();
        assert_eq!(
            decoder.decode_i32(&file, &mut output).unwrap(),
            samples.len()
        );
        assert_eq!(output, samples);
    }
}
