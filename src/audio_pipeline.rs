use crate::audio_packet::*;
use crate::audio_types::*;
use crate::wav::*;

#[cfg(not(target_arch = "wasm32"))]
pub struct AudioEncoder {
    opus_encoder: libopus::encoder::Encoder,
    wav_reader: WavStreamProcessor,
    frame_size: usize,
    packets: Vec<Vec<u8>>,
    bitrate: usize,
    widow: Vec<AudioFileData>,
}

#[cfg(not(target_arch = "wasm32"))]
impl AudioEncoder {
    pub fn new(bitrate: usize, frame_size: usize) -> Self {
        let mut opus_encoder = libopus::encoder::Encoder::create(
            48000,
            2,
            1,
            1,
            &[0u8, 1u8],
            libopus::encoder::Application::Audio,
        )
        .unwrap();

        opus_encoder
            .set_option(libopus::encoder::OPUS_SET_BITRATE_REQUEST, bitrate as u32)
            .unwrap();

        let wav_reader = WavStreamProcessor::new();

        Self {
            opus_encoder,
            wav_reader,
            frame_size,
            packets: Vec::new(),
            bitrate,
            widow: Vec::new(),
        }
    }

    pub fn add(&mut self, data: &[u8]) -> Result<(), String> {
        match self.wav_reader.add(data) {
            Ok(Some(audio_data)) => self.encode(audio_data, false),
            Ok(None) => Ok(()),
            Err(err) => Err(err),
        }
    }

    pub fn flush(&mut self) -> Vec<u8> {
        if let Some(widow) = self.widow.pop() {
            let _ = self.encode(widow, true);
        }

        let mut offset = 0;
        let mut offsets = Vec::new();
        let mut encoded_data: Vec<u8> = Vec::new();
        for chunk in &self.packets {
            offsets.push(offset);
            offset += chunk.len();
            encoded_data.extend(chunk);
        }

        let mut final_encoded_data = Vec::new();
        for i in 0..4 {
            final_encoded_data.push(((offsets.len() >> (i * 8)) & 0xFF) as u8);
        }

        for offset in offsets {
            for i in 0..4 {
                final_encoded_data.push((offset >> (i * 8) & 0xFF) as u8);
            }
        }

        final_encoded_data.extend(encoded_data);

        self.reset();

        final_encoded_data
    }

    fn encode(&mut self, audio_data: AudioFileData, is_last: bool) -> Result<(), String> {
        let chunk_size = self.frame_size
            * audio_data.channel_count() as usize
            * audio_data.bits_per_sample() as usize;

        let mut data = audio_data.data().to_owned();
        if let Some(widow) = self.widow.pop() {
            data.extend_from_slice(&widow.data());
        }

        for chunk in audio_data.data().chunks(chunk_size) {
            let Some(config) =
                get_audio_config(audio_data.sampling_rate(), audio_data.bits_per_sample())
            else {
                return Err("Audio type not supported".to_string());
            };

            let chunk_size = self.frame_size as usize
                * audio_data.channel_count() as usize
                * audio_data.bits_per_sample() as usize;

            let flag = if chunk.len() < chunk_size {
                EncodingFlag::PCM
            } else {
                EncodingFlag::Opus
            };

            if flag == EncodingFlag::PCM && !is_last {
                let widow = AudioFileData::new(
                    audio_data.bits_per_sample(),
                    audio_data.channel_count(),
                    audio_data.sampling_rate(),
                    chunk.to_vec(),
                );
                self.widow.push(widow);
                return Ok(());
            };

            let packet = encode_audio_packet(
                config,
                &chunk.to_vec(),
                audio_data.channel_count() as usize,
                &flag,
                &mut self.opus_encoder,
            )?;

            self.packets.push(packet);
        }

        Ok(())
    }

    fn reset(&mut self) {
        self.opus_encoder
            .set_option(
                libopus::encoder::OPUS_SET_BITRATE_REQUEST,
                self.bitrate as u32,
            )
            .unwrap();

        self.wav_reader = WavStreamProcessor::new();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Read;

    #[test]
    fn test_opus_encoding() {
        let file_path = "testdata/f32le.wav";
        let mut file = File::open(&file_path).unwrap();

        let frame_size = 120;
        let bitrate = 128_000;
        let mut processor = AudioEncoder::new(bitrate, frame_size);

        let mut buffer = [0u8; 1024 * 100];
        loop {
            let bytes_read = file.read(&mut buffer).unwrap();
            if bytes_read == 0 {
                break;
            }

            let chunk = &buffer[..bytes_read];
            let _ = processor.add(chunk).unwrap();
        }
        let _encoded_data = processor.flush();
    }
}
