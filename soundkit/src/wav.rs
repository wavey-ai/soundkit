use crate::audio_types::{AudioData, PcmData};
use frame_header::{EncodingFlag, Endianness};
use std::io::Write;

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
        remaining: usize,
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
    data_chunk_collected: usize,
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

                    if &self.buffer[..4] != b"RIFF" || &self.buffer[8..12] != b"WAVE" {
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
                        self.data_chunk_size = size;
                        self.data_chunk_collected = 0;
                        self.state = if size == 0 {
                            StreamWavState::Finished
                        } else {
                            StreamWavState::ReadingData { remaining: size }
                        };
                    } else {
                        if &kind == b"fmt " && size > MAX_WAV_FMT_BYTES {
                            return Err(format!(
                                "WAV fmt chunk exceeds the {MAX_WAV_FMT_BYTES} byte metadata budget"
                            ));
                        }
                        self.state = StreamWavState::ChunkPayload {
                            kind,
                            remaining: size,
                            padding: size & 1 != 0,
                            payload: Vec::with_capacity(if &kind == b"fmt " { size } else { 0 }),
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
                    if &kind == b"fmt " {
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

                    let available = remaining.min(self.buffer.len());
                    let len = (available / bytes_per_frame) * bytes_per_frame;
                    if len == 0 {
                        if self.buffer.len() >= remaining && remaining > 0 {
                            return Err("WAV data chunk is not frame-aligned".to_string());
                        }
                        self.state = StreamWavState::ReadingData { remaining };
                        return Ok(None); // Wait for more data.
                    }
                    let data_chunk: Vec<u8> = self.buffer.drain(..len).collect();
                    remaining -= len;
                    self.data_chunk_collected += len;
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
}

pub fn generate_wav_buffer(pcm_data: &PcmData, sampling_rate: u32) -> Result<Vec<u8>, String> {
    let mut cursor = Vec::new();

    let bits_per_sample = match pcm_data {
        PcmData::I16(_) => 16,
        PcmData::I32(_) => 32,
        PcmData::F32(_) => 32,
    };

    let channel_count = match pcm_data {
        PcmData::I16(data) => data.len(),
        PcmData::I32(data) => data.len(),
        PcmData::F32(data) => data.len(),
    };

    let sample_count = match pcm_data {
        PcmData::I16(data) => data[0].len(),
        PcmData::I32(data) => data[0].len(),
        PcmData::F32(data) => data[0].len(),
    };

    let audio_format = match pcm_data {
        PcmData::I16(_) => 1u16, // PCM
        PcmData::I32(_) => 1u16, // PCM
        PcmData::F32(_) => 3u16, // IEEE float
    };

    let bytes_per_sample = (bits_per_sample / 8) as usize;
    let byte_rate = sampling_rate as usize * bytes_per_sample * channel_count;
    let block_align = bytes_per_sample * channel_count;
    let sub_chunk_2_size = sample_count * bytes_per_sample * channel_count;

    cursor.write_all(b"RIFF").unwrap();
    cursor
        .write_all(&(36 + sub_chunk_2_size as u32).to_le_bytes())
        .unwrap();
    cursor.write_all(b"WAVE").unwrap();

    cursor.write_all(b"fmt ").unwrap();
    cursor.write_all(&16u32.to_le_bytes()).unwrap(); // fmt chunk size
    cursor.write_all(&audio_format.to_le_bytes()).unwrap(); // PCM or IEEE float
    cursor
        .write_all(&(channel_count as u16).to_le_bytes())
        .unwrap(); // Number of channels
    cursor.write_all(&sampling_rate.to_le_bytes()).unwrap(); // Sample rate
    cursor.write_all(&(byte_rate as u32).to_le_bytes()).unwrap(); // Byte rate
    cursor
        .write_all(&(block_align as u16).to_le_bytes())
        .unwrap(); // Block align
    cursor
        .write_all(&(bits_per_sample as u16).to_le_bytes())
        .unwrap(); // Bits per sample

    cursor.write_all(b"data").unwrap();
    cursor
        .write_all(&(sub_chunk_2_size as u32).to_le_bytes())
        .unwrap();

    for i in 0..sample_count {
        for ch in 0..channel_count {
            match pcm_data {
                PcmData::I16(data) => cursor.write_all(&data[ch][i].to_le_bytes()).unwrap(),
                PcmData::I32(data) => cursor.write_all(&data[ch][i].to_le_bytes()).unwrap(),
                PcmData::F32(data) => cursor.write_all(&f32::to_le_bytes(data[ch][i])).unwrap(),
            }
        }
    }

    Ok(cursor)
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
}
