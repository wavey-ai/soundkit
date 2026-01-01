use crate::audio_types::{AudioData, PcmData};
use frame_header::{EncodingFlag, Endianness};
use std::io::Write;

enum StreamWavState {
    Initial,
    ReadToFmt,
    ReadingFmt,
    ReadToData,
    ReadingData,
    Finished,
}

pub struct WavStreamProcessor {
    state: StreamWavState,
    buffer: Vec<u8>,
    idx: usize,
    bits_per_sample: usize,
    channel_count: usize,
    sampling_rate: usize,
    audio_format: EncodingFlag,
    endianness: Endianness, // New field to track endianness
    data_chunk_size: usize,
    data_chunk_collected: usize,
}

impl WavStreamProcessor {
    pub fn new() -> Self {
        Self {
            state: StreamWavState::Initial,
            buffer: Vec::new(),
            idx: 0,
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

    pub fn add(&mut self, chunk: &[u8]) -> Result<Option<AudioData>, String> {
        self.buffer.extend(chunk);

        loop {
            match &self.state {
                StreamWavState::Initial => {
                    if self.buffer.len() < 12 {
                        return Ok(None);
                    }

                    if &self.buffer[..4] != b"RIFF" || &self.buffer[8..12] != b"WAVE" {
                        return Err("Not a WAV file".to_string());
                    }

                    self.state = StreamWavState::ReadToFmt;
                    self.idx = 12;
                }

                StreamWavState::ReadToFmt => {
                    if self.buffer.len() < self.idx + 4 {
                        return Ok(None);
                    }

                    while &self.buffer[self.idx..self.idx + 4] != b"fmt " {
                        let chunk_size = u32::from_le_bytes(
                            self.buffer[self.idx + 4..self.idx + 8].try_into().unwrap(),
                        ) as usize;
                        self.idx += chunk_size + 8;

                        if self.buffer.len() < self.idx + 8 {
                            return Ok(None);
                        }
                    }

                    self.state = StreamWavState::ReadingFmt;
                }

                StreamWavState::ReadingFmt => {
                    if self.buffer.len() < self.idx + 24 {
                        return Ok(None);
                    }

                    let fmt_chunk = &self.buffer[self.idx..self.idx + 24];
                    self.sampling_rate =
                        u32::from_le_bytes(fmt_chunk[12..16].try_into().unwrap()) as usize;
                    self.bits_per_sample =
                        u16::from_le_bytes(fmt_chunk[22..24].try_into().unwrap()) as usize;
                    self.channel_count =
                        u16::from_le_bytes(fmt_chunk[10..12].try_into().unwrap()) as usize;

                    self.audio_format =
                        match u16::from_le_bytes(fmt_chunk[8..10].try_into().unwrap()) {
                            1 => EncodingFlag::PCMSigned,
                            3 => EncodingFlag::PCMFloat,
                            _ => EncodingFlag::PCMFloat,
                        };

                    self.endianness = Endianness::LittleEndian;
                    self.state = StreamWavState::ReadToData;

                    let chunk_size = u32::from_le_bytes(
                        self.buffer[self.idx + 4..self.idx + 8].try_into().unwrap(),
                    ) as usize;
                    self.idx += chunk_size + 8;
                }

                StreamWavState::ReadToData => {
                    if self.buffer.len() < self.idx + 4 {
                        return Ok(None);
                    }

                    while &self.buffer[self.idx..self.idx + 4] != b"data" {
                        let chunk_size = u32::from_le_bytes(
                            self.buffer[self.idx + 4..self.idx + 8].try_into().unwrap(),
                        ) as usize;
                        self.idx += chunk_size + 8;

                        if self.buffer.len() < self.idx + 8 {
                            return Ok(None);
                        }
                    }

                    let chunk_size = u32::from_le_bytes(
                        self.buffer[self.idx + 4..self.idx + 8].try_into().unwrap(),
                    ) as usize;

                    self.data_chunk_size = chunk_size;

                    self.state = StreamWavState::ReadingData;

                    self.buffer = self.buffer.split_off(self.idx + 8);
                }

                StreamWavState::ReadingData => {
                    let bytes_per_sample = (self.bits_per_sample / 8) as usize;
                    let bytes_per_frame = bytes_per_sample * self.channel_count as usize;

                    if self.buffer.len() < bytes_per_frame {
                        return Ok(None); // Wait for more data.
                    }

                    let frames_in_buffer = self.buffer.len() / bytes_per_frame;
                    let len = frames_in_buffer * bytes_per_frame;
                    let data_chunk = self.buffer[..len].to_vec();
                    self.buffer = self.buffer.split_off(len);

                    self.data_chunk_collected += len;
                    if self.data_chunk_collected == self.data_chunk_size {
                        self.state = StreamWavState::Finished;
                    }

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
                    // Gracefully return None when finished - no more data available
                    return Ok(None);
                }
            }
        }
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

        assert!(audio_packets.len() > 0, "No audio packets processed");
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
        let byte_rate = 48_000 * 1 * 3;
        buf.extend_from_slice(&(byte_rate as u32).to_le_bytes());
        let block_align = 1 * 3;
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
}
