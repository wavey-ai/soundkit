#![allow(dead_code)]

use crate::audio_types::{get_audio_config, AudioFileData};

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

    pub fn add(&mut self, chunk: &[u8]) -> Result<Option<AudioFileData>, String> {
        self.buffer.extend(chunk);

        loop {
            match &self.state {
                StreamWavState::Initial => {
                    if self.buffer.len() < 12 {
                        return Ok(None); // Wait for more data
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
                        self.idx += chunk_size + 8; // Advance to next chunk
                        if self.buffer.len() < self.idx + 8 {
                            return Ok(None);
                        }
                    }
                    self.state = StreamWavState::ReadingFmt;
                }

                StreamWavState::ReadingFmt => {
                    if self.buffer.len() < self.idx + 24 {
                        return Ok(None); // Wait for more data
                    }

                    let fmt_chunk = &self.buffer[self.idx..self.idx + 24];
                    self.sampling_rate =
                        u32::from_le_bytes(fmt_chunk[12..16].try_into().unwrap()) as usize;
                    self.bits_per_sample =
                        u16::from_le_bytes(fmt_chunk[22..24].try_into().unwrap()) as usize;
                    self.channel_count =
                        u16::from_le_bytes(fmt_chunk[10..12].try_into().unwrap()) as usize;

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
                        self.idx += chunk_size + 8; // Advance to next chunk
                        if self.buffer.len() < self.idx + 8 {
                            return Ok(None);
                        }
                    }

                    let chunk_size = u32::from_le_bytes(
                        self.buffer[self.idx + 4..self.idx + 8].try_into().unwrap(),
                    ) as usize;

                    self.data_chunk_size = chunk_size;

                    self.state = StreamWavState::ReadingData;
                    // Skip "data" and size
                    self.buffer = self.buffer.split_off(self.idx + 8);
                }

                StreamWavState::ReadingData => {
                    let bytes_per_sample = (self.bits_per_sample / 8) as usize;
                    let bytes_per_frame = bytes_per_sample * self.channel_count as usize;

                    if self.buffer.len() < bytes_per_frame as usize {
                        return Ok(None); // Wait for more data
                    }

                    let frames_in_buffer = self.buffer.len() / bytes_per_frame;
                    let len = frames_in_buffer * bytes_per_frame;
                    let data_chunk = self.buffer[..len].to_vec();
                    self.buffer = self.buffer.split_off(len);

                    self.data_chunk_collected += len;
                    if self.data_chunk_collected == self.data_chunk_size {
                        self.state = StreamWavState::Finished;
                    }

                    let result = AudioFileData::new(
                        self.bits_per_sample as u8,
                        self.channel_count as u8,
                        self.sampling_rate as u32,
                        data_chunk,
                    );

                    return Ok(Some(result));
                }

                StreamWavState::Finished => {
                    return Err("Already finished processing WAV file".to_string());
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Read;

    #[test]
    fn test_wav_stream() {
        let file_path = "testdata/f32le.wav";
        let mut file = File::open(&file_path).unwrap();

        let mut processor = WavStreamProcessor::new();
        let mut audio_packets = Vec::new();
        let mut buffer = [0u8; 1024];

        loop {
            let bytes_read = file.read(&mut buffer).unwrap();
            if bytes_read == 0 {
                break;
            }

            let chunk = &buffer[..bytes_read];
            match processor.add(chunk) {
                Ok(Some(audio_data)) => {
                    audio_packets.push(audio_data);
                }
                Ok(None) => continue,
                Err(err) => panic!("Error: {}", err),
            }
        }

        //        assert_eq!(
        //            &processor.data_chunk_size, &processor.data_chunk_collected,
        //            "processed more bytes than indicated in data header"
        //        );
        assert!(audio_packets.len() > 0, "No audio packets processed");
        dbg!(audio_packets.len());
    }
}
