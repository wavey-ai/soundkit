use crate::audio_bytes::*;
use crate::audio_packet::{encode_audio_packet, Encoder};
use crate::audio_types::*;
use crate::wav::WavStreamProcessor;
use frame_header::{EncodingFlag, FrameHeader};
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

const COMMON_SAMPLE_RATES: [u32; 9] = [8000, 16000, 22050, 24000, 32000, 44100, 48000, 88200, 96000];
const COMMON_BITS_PER_SAMPLE: [u8; 3] = [16, 24, 32];

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
        24 => Ok(PcmData::I32(deinterleave_vecs_s24(
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
    let channel_count = audio.channel_count() as usize;
    if channel_count == 0 {
        return Err("Channel count must be > 0".to_string());
    }

    if !COMMON_BITS_PER_SAMPLE.contains(&audio.bits_per_sample()) {
        return Err(format!(
            "Unsupported bits_per_sample: {}",
            audio.bits_per_sample()
        ));
    }

    let output_rate = u32::try_from(sampling_rate)
        .map_err(|_| "sampling_rate out of range".to_string())?;
    let input_rate = audio.sampling_rate();

    if input_rate == 0 || output_rate == 0 {
        return Err("sampling_rate must be > 0".to_string());
    }

    if !COMMON_SAMPLE_RATES.contains(&input_rate) {
        return Err(format!("Unsupported input sample_rate: {}", input_rate));
    }

    if !COMMON_SAMPLE_RATES.contains(&output_rate) {
        return Err(format!("Unsupported output sample_rate: {}", output_rate));
    }

    let data: Vec<Vec<f32>> = if audio.bits_per_sample() == 32
        && audio.audio_format() != EncodingFlag::PCMFloat
    {
        let interleaved = s32le_to_i32(audio.data());
        let mut channels =
            vec![Vec::with_capacity(interleaved.len() / channel_count); channel_count];
        for (index, sample) in interleaved.into_iter().enumerate() {
            channels[index % channel_count].push(sample);
        }
        channels.into_iter().map(vec_i32_to_f32).collect()
    } else {
        let audio_data =
            deserialize_audio(audio.data(), audio.bits_per_sample(), audio.channel_count())
                .map_err(|e| format!("deserialize_audio failed: {}", e))?;

        match audio_data {
            PcmData::I16(data) => data.into_iter().map(vec_i16_to_f32).collect(),
            PcmData::I32(data) => data.into_iter().map(vec_i32_to_f32).collect(),
            PcmData::F32(data) => data,
        }
    };

    if data.is_empty() {
        return Ok(Vec::new());
    }

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let mut resampler = SincFixedIn::<f32>::new(
        output_rate as f64 / input_rate as f64,
        2.0,
        params,
        data[0].len(),
        data.len(),
    )
    .unwrap();

    let out = resampler.process(&data, None).unwrap();
    Ok(out)
}

pub struct AudioEncoder<E: Encoder> {
    encoder: E,
    encoding_flag: EncodingFlag,
    wav_reader: WavStreamProcessor,
    frame_size: usize,
    packets: Vec<Vec<u8>>,
    widow: Vec<AudioData>,
}

impl<E: Encoder> AudioEncoder<E> {
    pub fn new(encoding_flag: EncodingFlag, frame_size: usize, encoder: E) -> Self {
        let wav_reader = WavStreamProcessor::new();

        Self {
            encoder,
            encoding_flag,
            wav_reader,
            frame_size,
            packets: Vec::new(),
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
            let flag = if chunk.len() < chunk_size {
                EncodingFlag::PCMFloat
            } else {
                self.encoding_flag
            };

            if flag == EncodingFlag::PCMFloat || !is_last {
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

            let header = FrameHeader::new(
                audio_data.audio_format(),
                self.frame_size
                    .try_into()
                    .map_err(|_| "frame_size out of range".to_string())?,
                audio_data.sampling_rate(),
                audio_data.channel_count(),
                audio_data.bits_per_sample(),
                audio_data.endianness(),
                None,
                None,
            )?;

            let mut fullbuf = Vec::with_capacity(header.size() + chunk.len());
            header
                .encode(&mut fullbuf)
                .map_err(|e| format!("Failed to encode frame header: {}", e))?;
            fullbuf.extend_from_slice(chunk);

            let packet = encode_audio_packet(self.encoding_flag, &mut self.encoder, &fullbuf)?;

            self.packets.push(packet.to_vec());
        }

        Ok(())
    }

    fn reset(&mut self) {
        let _ = self.encoder.reset();

        self.wav_reader = WavStreamProcessor::new();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wav::generate_wav_buffer;
    use std::fs::File;
    use std::io::Read;
    use std::io::Write;
    use std::path::PathBuf;

    fn testdata_path(file: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("testdata")
            .join(file)
    }

    #[test]
    fn test_downsample_audio() {
        let file_path = testdata_path("wav_32f/A_Tusk_is_used_to_make_costly_gifts.wav");
        let mut file = File::open(&file_path).unwrap();

        let mut processor = WavStreamProcessor::new();
        let mut buffer = [0u8; 1024 * 1024];

        let mut result: Vec<Vec<f32>> = Vec::new();
        loop {
            let bytes_read = file.read(&mut buffer).unwrap();
            if bytes_read == 0 {
                break;
            }

            let chunk = &buffer[..bytes_read];
            match processor.add(chunk) {
                Ok(Some(audio_data)) => {
                    let samples = downsample_audio(&audio_data, 8_000).unwrap();

                    assert!(!samples.is_empty());
                    assert!(!samples[0].is_empty());

                    if result.is_empty() {
                        result = vec![Vec::new(); samples.len()];
                    }

                    assert_eq!(result.len(), samples.len());

                    for (channel_result, channel_samples) in
                        result.iter_mut().zip(samples.iter())
                    {
                        channel_result.extend_from_slice(channel_samples)
                    }
                }
                Ok(None) => continue,
                _ => panic!("Error"),
            }
        }

        match generate_wav_buffer(&PcmData::F32(result), 8_000) {
            Ok(wav_buffer) => {
                let output_path = file_path.with_file_name("A_Tusk_is_used_to_make_costly_gifts_8kz.wav");
                let mut file = File::create(output_path).expect("Could not create file");
                file.write_all(&wav_buffer)
                    .expect("Could not write to file");
            }
            Err(err) => {
                eprintln!("Error generating wav buffer: {}", err);
            }
        }
    }
}
