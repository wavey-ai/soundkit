use crate::audio_bytes::*;
use crate::audio_packet::*;
use crate::audio_types::*;
use crate::wav::*;

use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

pub fn vec_f32_to_i16(input: Vec<f32>) -> Vec<i16> {
    let mut output: Vec<i16> = Vec::with_capacity(input.len());

    for value in input {
        let clamped_value = value.max(-1.0).min(1.0);
        let scaled_value = (clamped_value * 32767.0) as i16;
        output.push(scaled_value);
    }

    output
}

pub fn vec_i16_to_f32(input: Vec<i16>) -> Vec<f32> {
    let mut output: Vec<f32> = Vec::with_capacity(input.len());

    for value in input {
        let scaled_value = value as f32 / 32768.0; // Division by 32768 instead of 32767 for better centering around 0
        output.push(scaled_value);
    }

    output
}

pub fn vec_i32_to_f32(input: Vec<i32>) -> Vec<f32> {
    let mut output: Vec<f32> = Vec::with_capacity(input.len());
    const MAX_I32: f32 = 2147483647.0; // or use `i32::MAX as f32`

    for value in input {
        let scaled_value = value as f32 / MAX_I32;
        output.push(scaled_value);
    }

    output
}

pub fn deserialize_audio(
    data: &[u8],
    bits_per_sample: u8,
    channel_count: u8,
) -> Result<PcmData, String> {
    match bits_per_sample {
        16 => Ok(PcmData::I16(deinterleave_vecs_i16(
            data,
            channel_count as usize,
        ))),
        24 => Ok(PcmData::I32(deinterleave_vecs_24bit(
            data,
            channel_count as usize,
        ))),
        32 => Ok(PcmData::F32(deinterleave_vecs_f32(
            data,
            channel_count as usize,
        ))),
        _ => Err("unsuporrted type".to_string()),
    }
}

pub fn downsample_audio(audio: &AudioData, sampling_rate: usize) -> Result<Vec<Vec<f32>>, String> {
    let audio_data =
        deserialize_audio(audio.data(), audio.bits_per_sample(), audio.channel_count());

    let data: Vec<Vec<f32>> = match audio_data {
        Ok(PcmData::I16(data)) => data.into_iter().map(vec_i16_to_f32).collect(),
        Ok(PcmData::I32(data)) => data.into_iter().map(vec_i32_to_f32).collect(),
        Ok(PcmData::F32(data)) => data,
        _ => Vec::new(),
    };

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let mut resampler = SincFixedIn::<f32>::new(
        sampling_rate as f64 / audio.sampling_rate() as f64,
        2.0,
        params,
        data[0].len(),
        2,
    )
    .unwrap();

    let out = resampler.process(&data, None).unwrap();
    Ok(out)
}

#[cfg(not(target_arch = "wasm32"))]
pub struct AudioEncoder {
    opus_encoder: libopus::encoder::Encoder,
    wav_reader: WavStreamProcessor,
    frame_size: usize,
    packets: Vec<Vec<u8>>,
    bitrate: usize,
    widow: Vec<AudioData>,
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

    pub fn encode(&mut self, audio_data: AudioData, is_last: bool) -> Result<(), String> {
        let chunk_size = self.frame_size
            * audio_data.channel_count() as usize
            * audio_data.bits_per_sample() as usize;

        let mut data = audio_data.data().to_owned();
        if let Some(widow) = self.widow.pop() {
            data.extend_from_slice(&widow.data());
        }

        for chunk in data.chunks(chunk_size) {
            let Some(config) = get_audio_config(
                audio_data.sampling_rate(),
                audio_data.bits_per_sample(),
                Some(audio_data.audio_format()),
                audio_data.endianness(), // Added endianness argument
            ) else {
                return Err("Audio type not supported".to_string());
            };

            let flag = if chunk.len() < chunk_size {
                EncodingFlag::PCM
            } else {
                EncodingFlag::Opus
            };

            if flag == EncodingFlag::PCM && !is_last {
                let widow = AudioData::new(
                    audio_data.bits_per_sample(),
                    audio_data.channel_count(),
                    audio_data.sampling_rate(),
                    chunk.to_vec(),
                    audio_data.audio_format(),
                    audio_data.endianness(),
                );
                self.widow.push(widow);
                return Ok(());
            }

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
    use std::io::Write;

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

    #[test]
    fn test_downsample_audio() {
        let file_path = "testdata/f32le.wav";
        let mut file = File::open(&file_path).unwrap();

        let mut processor = WavStreamProcessor::new();
        let mut buffer = [0u8; 1024 * 1024];

        let mut result = vec![Vec::new(); 2];
        loop {
            let bytes_read = file.read(&mut buffer).unwrap();
            if bytes_read == 0 {
                break;
            }

            let chunk = &buffer[..bytes_read];
            match processor.add(chunk) {
                Ok(Some(audio_data)) => {
                    let samples = downsample_audio(&audio_data, 8_000).unwrap();

                    assert!(samples[0].len() > 0);

                    for i in 0..processor.channel_count() {
                        result[i].extend_from_slice(&samples[i])
                    }
                }
                Ok(None) => continue,
                _ => panic!("Error"),
            }
        }

        match generate_wav_buffer(&PcmData::F32(result), 8_000) {
            Ok(wav_buffer) => {
                let mut file =
                    File::create("testdata/f32le_8kz.wav").expect("Could not create file");
                file.write_all(&wav_buffer)
                    .expect("Could not write to file");
            }
            Err(err) => {
                eprintln!("Error generating wav buffer: {}", err);
            }
        }
    }
}
