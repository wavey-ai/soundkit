use crate::audio_types::{AudioData, PcmData};
use frame_header::{EncodingFlag, Endianness};

enum StreamWavState {
    Initial,
    ChunkHeader,
    ChunkPayload {
        kind: [u8; 4],
        remaining: usize,
        padding: bool,
        payload: Vec<u8>,
    },
    ReadingData {
        remaining: u64,
    },
    Finished,
}

const MAX_WAV_INPUT_CHUNK_BYTES: usize = 4 * 1024 * 1024;
const MAX_WAV_FMT_BYTES: usize = 4096;

pub struct WavStreamProcessor {
    state: StreamWavState,
    buffer: Vec<u8>,
    bits_per_sample: usize,
    channel_count: usize,
    sampling_rate: usize,
    audio_format: EncodingFlag,
    endianness: Endianness, // New field to track endianness
    data_chunk_size: usize,
    data_chunk_collected: u64,
    rf64: bool,
    rf64_data_size: Option<u64>,
}

impl Default for WavStreamProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl WavStreamProcessor {
    pub fn new() -> Self {
        Self {
            state: StreamWavState::Initial,
            buffer: Vec::new(),
            bits_per_sample: 0,
            channel_count: 0,
            sampling_rate: 0,
            audio_format: EncodingFlag::PCMSigned,
            endianness: Endianness::LittleEndian, // Default to little-endian
            data_chunk_size: 0,
            data_chunk_collected: 0,
            rf64: false,
            rf64_data_size: None,
        }
    }

    pub fn bits_per_sample(&self) -> usize {
        self.bits_per_sample
    }

    pub fn channel_count(&self) -> usize {
        self.channel_count
    }

    pub fn sampling_rate(&self) -> usize {
        self.sampling_rate
    }

    pub fn audio_format(&self) -> EncodingFlag {
        self.audio_format
    }

    pub fn endianness(&self) -> Endianness {
        self.endianness
    }

    pub fn buffered_len(&self) -> usize {
        self.buffer.len()
    }

    pub fn add(&mut self, chunk: &[u8]) -> Result<Option<AudioData>, String> {
        if chunk.len() > MAX_WAV_INPUT_CHUNK_BYTES {
            return Err(format!(
                "WAV input chunk exceeds the {MAX_WAV_INPUT_CHUNK_BYTES} byte streaming budget"
            ));
        }
        self.buffer.extend(chunk);

        loop {
            let state = std::mem::replace(&mut self.state, StreamWavState::Finished);
            match state {
                StreamWavState::Initial => {
                    if self.buffer.len() < 12 {
                        self.state = StreamWavState::Initial;
                        return Ok(None);
                    }

                    self.rf64 = &self.buffer[..4] == b"RF64";
                    if (!self.rf64 && &self.buffer[..4] != b"RIFF")
                        || &self.buffer[8..12] != b"WAVE"
                    {
                        return Err("Not a WAV file".to_string());
                    }

                    self.buffer.drain(..12);
                    self.state = StreamWavState::ChunkHeader;
                }
                StreamWavState::ChunkHeader => {
                    if self.buffer.len() < 8 {
                        self.state = StreamWavState::ChunkHeader;
                        return Ok(None);
                    }
                    let kind: [u8; 4] = self.buffer[..4]
                        .try_into()
                        .map_err(|_| "WAV chunk header is truncated".to_string())?;
                    let size = u32::from_le_bytes(
                        self.buffer[4..8]
                            .try_into()
                            .map_err(|_| "WAV chunk size is truncated".to_string())?,
                    ) as usize;
                    self.buffer.drain(..8);
                    if &kind == b"data" {
                        if self.bits_per_sample == 0
                            || self.channel_count == 0
                            || self.sampling_rate == 0
                        {
                            return Err("WAV data appears before a valid fmt chunk".to_string());
                        }
                        let data_size = if self.rf64 && size == u32::MAX as usize {
                            self.rf64_data_size.ok_or_else(|| {
                                "RF64 data chunk appears before a valid ds64 chunk".to_string()
                            })?
                        } else {
                            size as u64
                        };
                        self.data_chunk_size = usize::try_from(data_size).unwrap_or(usize::MAX);
                        self.data_chunk_collected = 0;
                        self.state = if data_size == 0 {
                            StreamWavState::Finished
                        } else {
                            StreamWavState::ReadingData {
                                remaining: data_size,
                            }
                        };
                    } else {
                        if (&kind == b"fmt " || &kind == b"ds64") && size > MAX_WAV_FMT_BYTES {
                            return Err(format!(
                                "WAV fmt chunk exceeds the {MAX_WAV_FMT_BYTES} byte metadata budget"
                            ));
                        }
                        self.state = StreamWavState::ChunkPayload {
                            kind,
                            remaining: size,
                            padding: size & 1 != 0,
                            payload: Vec::with_capacity(if &kind == b"fmt " || &kind == b"ds64" {
                                size
                            } else {
                                0
                            }),
                        };
                    }
                }
                StreamWavState::ChunkPayload {
                    kind,
                    mut remaining,
                    mut padding,
                    mut payload,
                } => {
                    let consumed = remaining.min(self.buffer.len());
                    if &kind == b"fmt " || &kind == b"ds64" {
                        payload.extend_from_slice(&self.buffer[..consumed]);
                    }
                    self.buffer.drain(..consumed);
                    remaining -= consumed;
                    if remaining > 0 {
                        self.state = StreamWavState::ChunkPayload {
                            kind,
                            remaining,
                            padding,
                            payload,
                        };
                        return Ok(None);
                    }
                    if padding {
                        if self.buffer.is_empty() {
                            self.state = StreamWavState::ChunkPayload {
                                kind,
                                remaining: 0,
                                padding,
                                payload,
                            };
                            return Ok(None);
                        }
                        self.buffer.drain(..1);
                        padding = false;
                    }
                    if &kind == b"fmt " {
                        self.install_fmt(&payload)?;
                    } else if &kind == b"ds64" {
                        self.install_ds64(&payload)?;
                    }
                    debug_assert!(!padding);
                    self.state = StreamWavState::ChunkHeader;
                }
                StreamWavState::ReadingData { mut remaining } => {
                    let bytes_per_sample = self.bits_per_sample / 8;
                    let bytes_per_frame = bytes_per_sample * self.channel_count;
                    if bytes_per_frame == 0 {
                        return Err("WAV fmt has zero bytes per frame".to_string());
                    }

                    let available = remaining.min(self.buffer.len() as u64) as usize;
                    let len = (available / bytes_per_frame) * bytes_per_frame;
                    if len == 0 {
                        if (self.buffer.len() as u64) >= remaining && remaining > 0 {
                            return Err("WAV data chunk is not frame-aligned".to_string());
                        }
                        self.state = StreamWavState::ReadingData { remaining };
                        return Ok(None); // Wait for more data.
                    }
                    let data_chunk: Vec<u8> = self.buffer.drain(..len).collect();
                    remaining -= len as u64;
                    self.data_chunk_collected += len as u64;
                    self.state = if remaining == 0 {
                        StreamWavState::Finished
                    } else {
                        StreamWavState::ReadingData { remaining }
                    };

                    let result = AudioData::new(
                        self.bits_per_sample as u8,
                        self.channel_count as u8,
                        self.sampling_rate as u32,
                        data_chunk,
                        self.audio_format,
                        self.endianness,
                    );

                    return Ok(Some(result));
                }

                StreamWavState::Finished => {
                    self.state = StreamWavState::Finished;
                    // Gracefully return None when finished - no more data available
                    return Ok(None);
                }
            }
        }
    }

    fn install_fmt(&mut self, payload: &[u8]) -> Result<(), String> {
        if payload.len() < 16 {
            return Err("WAV fmt chunk must contain at least 16 bytes".to_string());
        }
        let mut format = u16::from_le_bytes([payload[0], payload[1]]);
        if format == 0xfffe {
            if payload.len() < 40 {
                return Err("WAVE_FORMAT_EXTENSIBLE fmt chunk is truncated".to_string());
            }
            format = u16::from_le_bytes([payload[24], payload[25]]);
        }
        self.channel_count = u16::from_le_bytes([payload[2], payload[3]]) as usize;
        self.sampling_rate =
            u32::from_le_bytes([payload[4], payload[5], payload[6], payload[7]]) as usize;
        self.bits_per_sample = u16::from_le_bytes([payload[14], payload[15]]) as usize;
        self.audio_format = match format {
            1 => EncodingFlag::PCMSigned,
            3 => EncodingFlag::PCMFloat,
            other => return Err(format!("unsupported WAV format tag {other}")),
        };
        if self.channel_count == 0 || self.sampling_rate == 0 || self.bits_per_sample == 0 {
            return Err("WAV fmt contains invalid audio geometry".to_string());
        }
        if self.bits_per_sample % 8 != 0 {
            return Err("WAV sample width must be byte-aligned".to_string());
        }
        self.endianness = Endianness::LittleEndian;
        Ok(())
    }

    fn install_ds64(&mut self, payload: &[u8]) -> Result<(), String> {
        if !self.rf64 {
            return Err("ds64 chunk requires an RF64 header".to_string());
        }
        if payload.len() < 28 {
            return Err("RF64 ds64 chunk is truncated".to_string());
        }
        let data_size = u64::from_le_bytes(
            payload[8..16]
                .try_into()
                .map_err(|_| "RF64 data size is truncated".to_string())?,
        );
        let table_length = u32::from_le_bytes(
            payload[24..28]
                .try_into()
                .map_err(|_| "RF64 table length is truncated".to_string())?,
        ) as usize;
        let required = 28usize
            .checked_add(
                table_length
                    .checked_mul(12)
                    .ok_or_else(|| "RF64 ds64 table length overflows".to_string())?,
            )
            .ok_or_else(|| "RF64 ds64 size overflows".to_string())?;
        if payload.len() < required {
            return Err("RF64 ds64 table is truncated".to_string());
        }
        self.rf64_data_size = Some(data_size);
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WavSampleFormat {
    I16,
    I32,
    F32,
}

impl WavSampleFormat {
    fn bits_per_sample(self) -> u16 {
        match self {
            Self::I16 => 16,
            Self::I32 | Self::F32 => 32,
        }
    }

    fn format_tag(self) -> u16 {
        match self {
            Self::I16 | Self::I32 => 1,
            Self::F32 => 3,
        }
    }
}

/// Incremental PCM-to-WAV encoder with an exact RIFF or RF64 header.
///
/// The caller supplies the final frame count, writes `header()` once, then
/// forwards each bounded result from a `push_*` call. No complete PCM or WAV
/// allocation is required. Files beyond RIFF's 4 GiB limit use RF64.
pub struct WavStreamEncoder {
    format: WavSampleFormat,
    sampling_rate: u32,
    channel_count: usize,
    total_frames: u64,
    frames_written: u64,
    data_bytes: u64,
    header: Vec<u8>,
    finished: bool,
}

impl WavStreamEncoder {
    pub fn new(
        format: WavSampleFormat,
        sampling_rate: u32,
        channel_count: usize,
        total_frames: u64,
    ) -> Result<Self, String> {
        if sampling_rate == 0 {
            return Err("WAV sampling rate must be greater than zero".to_string());
        }
        if channel_count == 0 || channel_count > u16::MAX as usize {
            return Err("WAV channel count is outside the RIFF field range".to_string());
        }
        let bytes_per_sample = u64::from(format.bits_per_sample() / 8);
        let block_align = bytes_per_sample
            .checked_mul(channel_count as u64)
            .ok_or_else(|| "WAV block alignment overflows".to_string())?;
        if block_align > u16::MAX as u64 {
            return Err("WAV block alignment exceeds the RIFF field range".to_string());
        }
        let byte_rate = u64::from(sampling_rate)
            .checked_mul(block_align)
            .ok_or_else(|| "WAV byte rate overflows".to_string())?;
        if byte_rate > u32::MAX as u64 {
            return Err("WAV byte rate exceeds the RIFF field range".to_string());
        }
        let data_bytes = total_frames
            .checked_mul(block_align)
            .ok_or_else(|| "WAV data size overflows".to_string())?;
        let header = wav_header(
            format,
            sampling_rate,
            channel_count as u16,
            total_frames,
            data_bytes,
            block_align as u16,
            byte_rate as u32,
        )?;
        Ok(Self {
            format,
            sampling_rate,
            channel_count,
            total_frames,
            frames_written: 0,
            data_bytes,
            header,
            finished: false,
        })
    }

    pub fn header(&self) -> &[u8] {
        &self.header
    }

    pub fn is_rf64(&self) -> bool {
        self.header.starts_with(b"RF64")
    }

    pub fn frames_written(&self) -> u64 {
        self.frames_written
    }

    pub fn total_frames(&self) -> u64 {
        self.total_frames
    }

    pub fn data_bytes(&self) -> u64 {
        self.data_bytes
    }

    pub fn sampling_rate(&self) -> u32 {
        self.sampling_rate
    }

    pub fn channel_count(&self) -> usize {
        self.channel_count
    }

    pub fn push_planar_i16(
        &mut self,
        planar: &[i16],
        frames_per_channel: usize,
    ) -> Result<Vec<u8>, String> {
        self.require_format(WavSampleFormat::I16)?;
        self.reserve_chunk(planar.len(), frames_per_channel)?;
        let mut output = Vec::with_capacity(planar.len() * 2);
        for frame in 0..frames_per_channel {
            for channel in 0..self.channel_count {
                output
                    .extend_from_slice(&planar[channel * frames_per_channel + frame].to_le_bytes());
            }
        }
        self.frames_written += frames_per_channel as u64;
        Ok(output)
    }

    pub fn push_planar_i32(
        &mut self,
        planar: &[i32],
        frames_per_channel: usize,
    ) -> Result<Vec<u8>, String> {
        self.require_format(WavSampleFormat::I32)?;
        self.reserve_chunk(planar.len(), frames_per_channel)?;
        let mut output = Vec::with_capacity(planar.len() * 4);
        for frame in 0..frames_per_channel {
            for channel in 0..self.channel_count {
                output
                    .extend_from_slice(&planar[channel * frames_per_channel + frame].to_le_bytes());
            }
        }
        self.frames_written += frames_per_channel as u64;
        Ok(output)
    }

    pub fn push_planar_f32(
        &mut self,
        planar: &[f32],
        frames_per_channel: usize,
    ) -> Result<Vec<u8>, String> {
        self.require_format(WavSampleFormat::F32)?;
        self.reserve_chunk(planar.len(), frames_per_channel)?;
        let mut output = Vec::with_capacity(planar.len() * 4);
        for frame in 0..frames_per_channel {
            for channel in 0..self.channel_count {
                output
                    .extend_from_slice(&planar[channel * frames_per_channel + frame].to_le_bytes());
            }
        }
        self.frames_written += frames_per_channel as u64;
        Ok(output)
    }

    pub fn push(&mut self, pcm_data: &PcmData) -> Result<Vec<u8>, String> {
        match pcm_data {
            PcmData::I16(channels) => {
                let frames = validate_planar_channels(channels, self.channel_count)?;
                self.require_format(WavSampleFormat::I16)?;
                self.reserve_chunk(frames * self.channel_count, frames)?;
                let mut output = Vec::with_capacity(frames * self.channel_count * 2);
                for frame in 0..frames {
                    for channel in channels {
                        output.extend_from_slice(&channel[frame].to_le_bytes());
                    }
                }
                self.frames_written += frames as u64;
                Ok(output)
            }
            PcmData::I32(channels) => {
                let frames = validate_planar_channels(channels, self.channel_count)?;
                self.require_format(WavSampleFormat::I32)?;
                self.reserve_chunk(frames * self.channel_count, frames)?;
                let mut output = Vec::with_capacity(frames * self.channel_count * 4);
                for frame in 0..frames {
                    for channel in channels {
                        output.extend_from_slice(&channel[frame].to_le_bytes());
                    }
                }
                self.frames_written += frames as u64;
                Ok(output)
            }
            PcmData::F32(channels) => {
                let frames = validate_planar_channels(channels, self.channel_count)?;
                self.require_format(WavSampleFormat::F32)?;
                self.reserve_chunk(frames * self.channel_count, frames)?;
                let mut output = Vec::with_capacity(frames * self.channel_count * 4);
                for frame in 0..frames {
                    for channel in channels {
                        output.extend_from_slice(&channel[frame].to_le_bytes());
                    }
                }
                self.frames_written += frames as u64;
                Ok(output)
            }
        }
    }

    pub fn finish(&mut self) -> Result<(), String> {
        if self.finished {
            return Err("WAV encoder is already finished".to_string());
        }
        if self.frames_written != self.total_frames {
            return Err(format!(
                "WAV encoder expected {} frames but received {}",
                self.total_frames, self.frames_written
            ));
        }
        self.finished = true;
        Ok(())
    }

    fn require_format(&self, format: WavSampleFormat) -> Result<(), String> {
        if self.finished {
            return Err("WAV encoder is already finished".to_string());
        }
        if self.format != format {
            return Err(format!(
                "WAV encoder expects {:?} PCM, not {:?}",
                self.format, format
            ));
        }
        Ok(())
    }

    fn reserve_chunk(&self, sample_count: usize, frames_per_channel: usize) -> Result<(), String> {
        let expected = self
            .channel_count
            .checked_mul(frames_per_channel)
            .ok_or_else(|| "WAV input chunk geometry overflows".to_string())?;
        if sample_count != expected {
            return Err(format!(
                "WAV planar chunk needs {expected} samples, got {sample_count}"
            ));
        }
        let end = self
            .frames_written
            .checked_add(frames_per_channel as u64)
            .ok_or_else(|| "WAV frame count overflows".to_string())?;
        if end > self.total_frames {
            return Err(format!(
                "WAV chunk ends at frame {end}, beyond the declared {} frames",
                self.total_frames
            ));
        }
        Ok(())
    }
}

fn validate_planar_channels<T>(
    channels: &[Vec<T>],
    expected_channels: usize,
) -> Result<usize, String> {
    if channels.len() != expected_channels {
        return Err(format!(
            "WAV encoder expects {expected_channels} channels, got {}",
            channels.len()
        ));
    }
    let frames = channels.first().map_or(0, Vec::len);
    if channels.iter().any(|channel| channel.len() != frames) {
        return Err("WAV planar channels have different frame counts".to_string());
    }
    Ok(frames)
}

fn wav_header(
    format: WavSampleFormat,
    sampling_rate: u32,
    channel_count: u16,
    total_frames: u64,
    data_bytes: u64,
    block_align: u16,
    byte_rate: u32,
) -> Result<Vec<u8>, String> {
    let classic_riff_size = 36u64
        .checked_add(data_bytes)
        .ok_or_else(|| "WAV RIFF size overflows".to_string())?;
    let use_rf64 = classic_riff_size > u32::MAX as u64;
    let mut output = Vec::with_capacity(if use_rf64 { 80 } else { 44 });
    if use_rf64 {
        let riff_size = 72u64
            .checked_add(data_bytes)
            .ok_or_else(|| "RF64 RIFF size overflows".to_string())?;
        output.extend_from_slice(b"RF64");
        output.extend_from_slice(&u32::MAX.to_le_bytes());
        output.extend_from_slice(b"WAVE");
        output.extend_from_slice(b"ds64");
        output.extend_from_slice(&28u32.to_le_bytes());
        output.extend_from_slice(&riff_size.to_le_bytes());
        output.extend_from_slice(&data_bytes.to_le_bytes());
        output.extend_from_slice(&total_frames.to_le_bytes());
        output.extend_from_slice(&0u32.to_le_bytes());
    } else {
        output.extend_from_slice(b"RIFF");
        output.extend_from_slice(&(classic_riff_size as u32).to_le_bytes());
        output.extend_from_slice(b"WAVE");
    }
    output.extend_from_slice(b"fmt ");
    output.extend_from_slice(&16u32.to_le_bytes());
    output.extend_from_slice(&format.format_tag().to_le_bytes());
    output.extend_from_slice(&channel_count.to_le_bytes());
    output.extend_from_slice(&sampling_rate.to_le_bytes());
    output.extend_from_slice(&byte_rate.to_le_bytes());
    output.extend_from_slice(&block_align.to_le_bytes());
    output.extend_from_slice(&format.bits_per_sample().to_le_bytes());
    output.extend_from_slice(b"data");
    output.extend_from_slice(
        &if use_rf64 {
            u32::MAX
        } else {
            data_bytes as u32
        }
        .to_le_bytes(),
    );
    Ok(output)
}

/// Convenience wrapper for callers that already own complete planar PCM.
/// Streaming callers should use `WavStreamEncoder` directly.
pub fn generate_wav_buffer(pcm_data: &PcmData, sampling_rate: u32) -> Result<Vec<u8>, String> {
    let (format, channel_count, frame_count) = match pcm_data {
        PcmData::I16(channels) => (
            WavSampleFormat::I16,
            channels.len(),
            channels.first().map_or(0, Vec::len),
        ),
        PcmData::I32(channels) => (
            WavSampleFormat::I32,
            channels.len(),
            channels.first().map_or(0, Vec::len),
        ),
        PcmData::F32(channels) => (
            WavSampleFormat::F32,
            channels.len(),
            channels.first().map_or(0, Vec::len),
        ),
    };
    let mut encoder =
        WavStreamEncoder::new(format, sampling_rate, channel_count, frame_count as u64)?;
    let mut output = encoder.header().to_vec();
    output.extend_from_slice(&encoder.push(pcm_data)?);
    encoder.finish()?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Read;
    use std::path::PathBuf;

    fn testdata_path(file: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("testdata")
            .join(file)
    }

    #[test]
    fn test_wav_stream() {
        let file_path = testdata_path("wav_32f/A_Tusk_is_used_to_make_costly_gifts.wav");
        let mut file = File::open(&file_path).unwrap();

        let mut processor = WavStreamProcessor::new();
        let mut audio_packets = Vec::new();
        let mut buffer = [0u8; 128];

        loop {
            let bytes_read = file.read(&mut buffer).unwrap();
            if bytes_read == 0 {
                break;
            }

            let chunk = &buffer[..bytes_read];
            match processor.add(chunk) {
                Ok(Some(audio_data)) => audio_packets.push(audio_data),
                Ok(None) => continue,
                Err(err) => panic!("Error: {}", err),
            }
        }

        assert!(!audio_packets.is_empty(), "No audio packets processed");
    }

    #[test]
    fn test_wav_stream_24bit_pcm() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        let data_chunk_size = 3u32;
        let fmt_chunk_size = 16u32;
        let file_size = 4 + (8 + fmt_chunk_size) + (8 + data_chunk_size);
        buf.extend_from_slice(&file_size.to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&fmt_chunk_size.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes()); // audio format = PCM
        buf.extend_from_slice(&1u16.to_le_bytes()); // num channels = 1
        buf.extend_from_slice(&48_000u32.to_le_bytes()); // sample rate = 48000
        let byte_rate = 48_000 * 3;
        buf.extend_from_slice(&(byte_rate as u32).to_le_bytes());
        let block_align = 3;
        buf.extend_from_slice(&(block_align as u16).to_le_bytes());
        buf.extend_from_slice(&24u16.to_le_bytes()); // bits per sample = 24
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&data_chunk_size.to_le_bytes());
        buf.extend_from_slice(&[0x01, 0x02, 0x03]); // one 24-bit sample

        let mut proc = WavStreamProcessor::new();
        let out = proc.add(&buf).unwrap().unwrap();

        assert_eq!(out.bits_per_sample(), 24);
        assert_eq!(out.channel_count(), 1);
        assert_eq!(out.sampling_rate(), 48_000);
        assert_eq!(out.data(), &vec![1, 2, 3]);
    }

    #[test]
    fn wav_skips_large_unknown_chunks_incrementally() {
        let junk_size = 8 * 1024 * 1024u32;
        let mut processor = WavStreamProcessor::new();
        let mut prefix = Vec::new();
        prefix.extend_from_slice(b"RIFF");
        prefix.extend_from_slice(&(junk_size + 48).to_le_bytes());
        prefix.extend_from_slice(b"WAVEJUNK");
        prefix.extend_from_slice(&junk_size.to_le_bytes());
        assert!(processor.add(&prefix).unwrap().is_none());

        let zeros = [0u8; 64 * 1024];
        for _ in 0..128 {
            assert!(processor.add(&zeros).unwrap().is_none());
            assert!(processor.buffered_len() <= 8);
        }

        let mut tail = Vec::new();
        tail.extend_from_slice(b"fmt ");
        tail.extend_from_slice(&16u32.to_le_bytes());
        tail.extend_from_slice(&1u16.to_le_bytes());
        tail.extend_from_slice(&1u16.to_le_bytes());
        tail.extend_from_slice(&48_000u32.to_le_bytes());
        tail.extend_from_slice(&96_000u32.to_le_bytes());
        tail.extend_from_slice(&2u16.to_le_bytes());
        tail.extend_from_slice(&16u16.to_le_bytes());
        tail.extend_from_slice(b"data");
        tail.extend_from_slice(&2u32.to_le_bytes());
        tail.extend_from_slice(&123i16.to_le_bytes());
        let audio = processor.add(&tail).unwrap().unwrap();
        assert_eq!(audio.data(), &123i16.to_le_bytes());
    }

    #[test]
    fn wav_chunk_headers_are_safe_at_every_split() {
        let bytes = b"RIFF\x24\0\0\0WAVEfmt \x10\0\0\0\x01\0\x01\0\x80\xbb\0\0\0w\x01\0\x02\0\x10\0data\0\0\0\0";
        for split in 0..bytes.len() {
            let mut processor = WavStreamProcessor::new();
            processor.add(&bytes[..split]).unwrap();
            processor.add(&bytes[split..]).unwrap();
        }
    }

    #[test]
    fn wav_encoder_streams_exact_chunks_without_changing_output() {
        let complete = PcmData::I16(vec![vec![1, 2, 3, 4], vec![-1, -2, -3, -4]]);
        let expected = generate_wav_buffer(&complete, 48_000).unwrap();

        let mut encoder = WavStreamEncoder::new(WavSampleFormat::I16, 48_000, 2, 4).unwrap();
        let mut streamed = encoder.header().to_vec();
        streamed.extend_from_slice(&encoder.push_planar_i16(&[1, 2, -1, -2], 2).unwrap());
        streamed.extend_from_slice(&encoder.push_planar_i16(&[3, 4, -3, -4], 2).unwrap());
        encoder.finish().unwrap();

        assert_eq!(streamed, expected);
        assert_eq!(encoder.frames_written(), 4);
        assert!(!encoder.is_rf64());
    }

    #[test]
    fn wav_encoder_uses_exact_rf64_sizes_beyond_four_gibibytes() {
        let frames = (u32::MAX as u64 / 8) + 1;
        let encoder = WavStreamEncoder::new(WavSampleFormat::F32, 48_000, 2, frames).unwrap();
        let header = encoder.header();
        assert_eq!(&header[..4], b"RF64");
        assert_eq!(header.len(), 80);
        assert_eq!(
            u32::from_le_bytes(header[4..8].try_into().unwrap()),
            u32::MAX
        );
        assert_eq!(
            u64::from_le_bytes(header[28..36].try_into().unwrap()),
            frames * 8
        );
        assert_eq!(
            u64::from_le_bytes(header[36..44].try_into().unwrap()),
            frames
        );
        assert_eq!(&header[72..76], b"data");
        assert_eq!(
            u32::from_le_bytes(header[76..80].try_into().unwrap()),
            u32::MAX
        );
    }

    #[test]
    fn wav_decoder_streams_rf64_data() {
        let samples = [123i16, -456i16];
        let data_bytes = (samples.len() * 2) as u64;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RF64");
        bytes.extend_from_slice(&u32::MAX.to_le_bytes());
        bytes.extend_from_slice(b"WAVEds64");
        bytes.extend_from_slice(&28u32.to_le_bytes());
        bytes.extend_from_slice(&(72 + data_bytes).to_le_bytes());
        bytes.extend_from_slice(&data_bytes.to_le_bytes());
        bytes.extend_from_slice(&(samples.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&48_000u32.to_le_bytes());
        bytes.extend_from_slice(&96_000u32.to_le_bytes());
        bytes.extend_from_slice(&2u16.to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&u32::MAX.to_le_bytes());
        for sample in samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }

        let mut decoder = WavStreamProcessor::new();
        let mut decoded = Vec::new();
        for chunk in bytes.chunks(7) {
            if let Some(frame) = decoder.add(chunk).unwrap() {
                decoded.extend_from_slice(frame.data());
            }
        }
        assert_eq!(
            decoded,
            [123i16.to_le_bytes(), (-456i16).to_le_bytes()].concat()
        );
    }

    #[test]
    fn wav_encoder_rejects_geometry_and_incomplete_streams() {
        let mut encoder = WavStreamEncoder::new(WavSampleFormat::I16, 48_000, 2, 2).unwrap();
        assert!(encoder.push_planar_i16(&[1, 2, 3], 2).is_err());
        assert!(encoder.push_planar_f32(&[0.0; 4], 2).is_err());
        assert!(encoder.finish().is_err());
        encoder.push_planar_i16(&[1, 2, 3, 4], 2).unwrap();
        encoder.finish().unwrap();
        assert!(encoder.push_planar_i16(&[], 0).is_err());
    }
}
