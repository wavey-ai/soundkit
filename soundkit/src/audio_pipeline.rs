use crate::audio_bytes::*;
use crate::audio_packet::{encode_audio_packet, Encoder};
use crate::audio_types::*;
use crate::wav::WavStreamProcessor;
use frame_header::{EncodingFlag, Endianness, FrameHeader};
#[cfg(feature = "sinc-resampler")]
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

#[cfg(feature = "sinc-resampler")]
const COMMON_SAMPLE_RATES: [u32; 9] =
    [8000, 16000, 22050, 24000, 32000, 44100, 48000, 88200, 96000];
#[cfg(feature = "sinc-resampler")]
const COMMON_BITS_PER_SAMPLE: [u8; 3] = [16, 24, 32];

pub fn vec_f32_to_i16(input: Vec<f32>) -> Vec<i16> {
    let mut output: Vec<i16> = Vec::with_capacity(input.len());

    for value in input {
        let clamped_value = value.clamp(-1.0, 1.0);
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

pub fn audio_to_f32_channels(audio: &AudioData) -> Result<Vec<Vec<f32>>, String> {
    let channel_count = audio.channel_count() as usize;
    if channel_count == 0 {
        return Err("Channel count must be > 0".to_string());
    }

    if audio.bits_per_sample() == 32 && audio.audio_format() != EncodingFlag::PCMFloat {
        let interleaved = s32le_to_i32(audio.data());
        let mut channels =
            vec![Vec::with_capacity(interleaved.len() / channel_count); channel_count];
        for (index, sample) in interleaved.into_iter().enumerate() {
            channels[index % channel_count].push(sample);
        }
        return Ok(channels.into_iter().map(vec_i32_to_f32).collect());
    }

    let pcm_data = deserialize_audio(audio.data(), audio.bits_per_sample(), audio.channel_count())
        .map_err(|error| format!("deserialize_audio failed: {error}"))?;

    match pcm_data {
        PcmData::I16(data) => Ok(data.into_iter().map(vec_i16_to_f32).collect()),
        // Scaled by the width the samples actually carry, not by the width
        // of the container they arrived in. A 24-bit source deserializes to
        // i32 — the values are 24-bit, the box is 32 — and dividing by the
        // box's full scale makes every sample 1/256 of itself. That is
        // silent enough to look like a decode failure and quiet enough to
        // be mistaken for a quiet master.
        PcmData::I32(data) => {
            let bits = u32::from(audio.bits_per_sample()).clamp(2, 32);
            let scale = 1.0f32 / (1u64 << (bits - 1)) as f32;
            Ok(data
                .into_iter()
                .map(|channel| channel.into_iter().map(|sample| sample as f32 * scale).collect())
                .collect())
        }
        PcmData::F32(data) => Ok(data),
    }
}

pub fn audio_to_mono_f32(audio: &AudioData) -> Result<Vec<f32>, String> {
    let channels = audio_to_f32_channels(audio)?;
    mixdown_to_mono_f32(&channels)
}

pub fn mixdown_to_mono_f32(channels: &[Vec<f32>]) -> Result<Vec<f32>, String> {
    if channels.is_empty() {
        return Ok(Vec::new());
    }
    if channels.len() == 1 {
        return Ok(channels[0].clone());
    }

    let sample_count = channels[0].len();
    if channels.iter().any(|channel| channel.len() != sample_count) {
        return Err("channel length mismatch".to_string());
    }

    let mut mono = vec![0.0f32; sample_count];
    for channel in channels {
        for (index, sample) in channel.iter().enumerate() {
            mono[index] += *sample;
        }
    }

    let scale = 1.0 / channels.len() as f32;
    for sample in &mut mono {
        *sample *= scale;
    }

    Ok(mono)
}

fn audio_to_stereo_f32(audio: &AudioData) -> Result<(Vec<f32>, Vec<f32>), String> {
    let channels = audio.channel_count() as usize;
    let sample_bytes = match audio.bits_per_sample() {
        16 => 2usize,
        24 => 3usize,
        32 => 4usize,
        bits => return Err(format!("unsupported PCM bit depth {bits}")),
    };
    if channels == 0 {
        return Err("decoded audio contained no channels".to_owned());
    }
    let frame_bytes = sample_bytes
        .checked_mul(channels)
        .ok_or_else(|| "decoded PCM frame size overflowed".to_owned())?;
    if audio.data().is_empty() || !audio.data().len().is_multiple_of(frame_bytes) {
        return Err("decoded audio contained an incomplete PCM frame".to_owned());
    }
    let frames = audio.data().len() / frame_bytes;
    let mut left = Vec::with_capacity(frames);
    let mut right = Vec::with_capacity(frames);
    for frame in audio.data().chunks_exact(frame_bytes) {
        let left_sample = pcm_sample_to_f32(
            &frame[..sample_bytes],
            audio.bits_per_sample(),
            audio.audio_format(),
            audio.endianness(),
        )?;
        let right_sample = if channels > 1 {
            pcm_sample_to_f32(
                &frame[sample_bytes..sample_bytes * 2],
                audio.bits_per_sample(),
                audio.audio_format(),
                audio.endianness(),
            )?
        } else {
            left_sample
        };
        left.push(finite_pcm(left_sample));
        right.push(finite_pcm(right_sample));
    }
    Ok((left, right))
}

fn pcm_sample_to_f32(
    bytes: &[u8],
    bits_per_sample: u8,
    encoding: EncodingFlag,
    endianness: Endianness,
) -> Result<f32, String> {
    let big_endian = endianness == Endianness::BigEndian;
    match (bits_per_sample, encoding) {
        (16, _) => {
            let bytes = [bytes[0], bytes[1]];
            let sample = if big_endian {
                i16::from_be_bytes(bytes)
            } else {
                i16::from_le_bytes(bytes)
            };
            Ok(sample as f32 / 32_768.0)
        }
        (24, _) => {
            let unsigned = if big_endian {
                u32::from_be_bytes([0, bytes[0], bytes[1], bytes[2]])
            } else {
                u32::from_le_bytes([bytes[0], bytes[1], bytes[2], 0])
            };
            let signed = if unsigned & 0x80_0000 != 0 {
                (unsigned | 0xff00_0000) as i32
            } else {
                unsigned as i32
            };
            Ok(signed as f32 / 8_388_608.0)
        }
        (32, EncodingFlag::PCMFloat) => {
            let bytes = [bytes[0], bytes[1], bytes[2], bytes[3]];
            Ok(if big_endian {
                f32::from_be_bytes(bytes)
            } else {
                f32::from_le_bytes(bytes)
            })
        }
        (32, _) => {
            let bytes = [bytes[0], bytes[1], bytes[2], bytes[3]];
            let sample = if big_endian {
                i32::from_be_bytes(bytes)
            } else {
                i32::from_le_bytes(bytes)
            };
            Ok(sample as f32 / 2_147_483_648.0)
        }
        _ => Err("unsupported PCM sample format".to_owned()),
    }
}

/// One bounded block of the library's canonical 48 kHz stereo PCM.
///
/// This type deliberately stays inside Rust. Browser adapters can feed its
/// samples straight into Opus and FLAC encoders without materializing a
/// complete Float32 programme in JavaScript.
#[derive(Debug)]
pub struct Stereo48kBlock {
    pub left: Vec<f32>,
    pub right: Vec<f32>,
}

impl Stereo48kBlock {
    pub fn frame_count(&self) -> usize {
        self.left.len()
    }

    pub fn is_empty(&self) -> bool {
        self.left.is_empty()
    }
}

/// Incrementally normalizes decoded SoundKit frames to 48 kHz stereo.
///
/// The interpolation and channel selection intentionally match the previous
/// browser implementation: mono is duplicated, sources with more channels
/// use their first two channels, and non-48 kHz sources use linear
/// interpolation over the complete stream timeline. Only the previous stereo
/// sample and the current decoded block are retained.
#[derive(Debug, Default)]
pub struct StreamingStereo48kNormalizer {
    source_sample_rate: u32,
    source_channels: u8,
    source_frames: u64,
    output_frames: u64,
    previous_left: f32,
    previous_right: f32,
    has_previous: bool,
    finished: bool,
}

impl StreamingStereo48kNormalizer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn source_sample_rate(&self) -> u32 {
        self.source_sample_rate
    }

    pub fn source_channels(&self) -> u8 {
        self.source_channels
    }

    pub fn source_frames(&self) -> u64 {
        self.source_frames
    }

    pub fn output_frames(&self) -> u64 {
        self.output_frames
    }

    pub fn push(&mut self, audio: &AudioData) -> Result<Option<Stereo48kBlock>, String> {
        if self.finished {
            return Err("48 kHz stereo normalizer is already finished".to_owned());
        }
        let sample_rate = audio.sampling_rate();
        let channel_count = audio.channel_count();
        if sample_rate == 0 || channel_count == 0 {
            return Err("decoded audio has invalid PCM geometry".to_owned());
        }
        if self.source_sample_rate == 0 {
            self.source_sample_rate = sample_rate;
            self.source_channels = channel_count;
        } else if self.source_sample_rate != sample_rate || self.source_channels != channel_count {
            return Err(format!(
                "decoded PCM geometry changed from {} Hz/{} ch to {sample_rate} Hz/{channel_count} ch",
                self.source_sample_rate, self.source_channels
            ));
        }

        let (left, right) = audio_to_stereo_f32(audio)?;
        if left.is_empty() || right.len() != left.len() {
            return Err("decoded audio contained an invalid PCM block".to_owned());
        }
        let frame_count = left.len() as u64;
        let start_frame = self.source_frames;
        let end_frame = start_frame
            .checked_add(frame_count)
            .ok_or_else(|| "decoded audio frame count overflowed".to_owned())?;
        let final_left = *left.last().expect("non-empty channel checked above");
        let final_right = *right.last().expect("non-empty channel checked above");

        let block = if sample_rate == 48_000 {
            self.output_frames = self
                .output_frames
                .checked_add(frame_count)
                .ok_or_else(|| "normalized audio frame count overflowed".to_owned())?;
            Stereo48kBlock { left, right }
        } else {
            let maximum_output = ((frame_count * 48_000).div_ceil(u64::from(sample_rate)) + 4)
                .try_into()
                .map_err(|_| "normalized audio block exceeds this address space".to_owned())?;
            let mut output_left = Vec::with_capacity(maximum_output);
            let mut output_right = Vec::with_capacity(maximum_output);
            loop {
                let source_position = self.output_frames as f64 * f64::from(sample_rate) / 48_000.0;
                let lower = source_position.floor() as u64;
                let upper = lower + 1;
                if upper >= end_frame {
                    break;
                }
                let fraction = (source_position - lower as f64) as f32;
                let left_lower = stream_sample(&left, self.previous_left, start_frame, lower);
                let right_lower = stream_sample(&right, self.previous_right, start_frame, lower);
                let left_upper = stream_sample(&left, self.previous_left, start_frame, upper);
                let right_upper = stream_sample(&right, self.previous_right, start_frame, upper);
                output_left.push(finite_pcm(
                    left_lower + ((left_upper - left_lower) * fraction),
                ));
                output_right.push(finite_pcm(
                    right_lower + ((right_upper - right_lower) * fraction),
                ));
                self.output_frames += 1;
            }
            Stereo48kBlock {
                left: output_left,
                right: output_right,
            }
        };

        self.source_frames = end_frame;
        self.previous_left = final_left;
        self.previous_right = final_right;
        self.has_previous = true;
        Ok((!block.is_empty()).then_some(block))
    }

    /// Completes the exact rounded 48 kHz duration by repeating the final
    /// sample, matching the old bounded browser stream.
    pub fn finish(&mut self) -> Result<Option<Stereo48kBlock>, String> {
        if self.finished {
            return Err("48 kHz stereo normalizer is already finished".to_owned());
        }
        self.finished = true;
        if !self.has_previous || self.source_frames == 0 || self.source_sample_rate == 0 {
            return Err("decoded source contained no PCM".to_owned());
        }
        if self.source_sample_rate == 48_000 {
            return Ok(None);
        }
        let target_frames = ((self.source_frames as f64 * 48_000.0)
            / f64::from(self.source_sample_rate))
        .round()
        .max(1.0) as u64;
        let remaining = target_frames.saturating_sub(self.output_frames);
        self.output_frames = target_frames;
        if remaining == 0 {
            return Ok(None);
        }
        let remaining: usize = remaining
            .try_into()
            .map_err(|_| "normalized audio tail exceeds this address space".to_owned())?;
        Ok(Some(Stereo48kBlock {
            left: vec![self.previous_left; remaining],
            right: vec![self.previous_right; remaining],
        }))
    }
}

fn finite_pcm(sample: f32) -> f32 {
    if sample.is_finite() {
        sample.clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

fn stream_sample(samples: &[f32], previous: f32, start_frame: u64, frame: u64) -> f32 {
    if frame < start_frame {
        return previous;
    }
    samples
        .get((frame - start_frame) as usize)
        .copied()
        .map(finite_pcm)
        .unwrap_or(previous)
}

pub fn f32s_to_le_bytes(samples: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(samples));
    for sample in samples {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    bytes
}

pub fn f32s_from_le_bytes(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if !bytes.len().is_multiple_of(std::mem::size_of::<f32>()) {
        return Err(format!(
            "invalid f32le byte length {}; expected multiple of 4",
            bytes.len()
        ));
    }

    let mut samples = Vec::with_capacity(bytes.len() / std::mem::size_of::<f32>());
    for chunk in bytes.chunks_exact(std::mem::size_of::<f32>()) {
        samples.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(samples)
}

#[cfg(feature = "sinc-resampler")]
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

    let output_rate =
        u32::try_from(sampling_rate).map_err(|_| "sampling_rate out of range".to_string())?;
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

    let data = audio_to_f32_channels(audio)?;

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
            data.extend_from_slice(widow.data());
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
    #[cfg(feature = "sinc-resampler")]
    use crate::wav::generate_wav_buffer;
    use frame_header::{EncodingFlag, Endianness};
    #[cfg(feature = "sinc-resampler")]
    use std::fs::File;
    #[cfg(feature = "sinc-resampler")]
    use std::io::Read;
    #[cfg(feature = "sinc-resampler")]
    use std::io::Write;
    #[cfg(feature = "sinc-resampler")]
    use std::path::PathBuf;

    #[cfg(feature = "sinc-resampler")]
    fn testdata_path(file: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("testdata")
            .join(file)
    }

    #[cfg(feature = "sinc-resampler")]
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

                    for (channel_result, channel_samples) in result.iter_mut().zip(samples.iter()) {
                        channel_result.extend_from_slice(channel_samples)
                    }
                }
                Ok(None) => continue,
                _ => panic!("Error"),
            }
        }

        match generate_wav_buffer(&PcmData::F32(result), 8_000) {
            Ok(wav_buffer) => {
                let output_path =
                    file_path.with_file_name("A_Tusk_is_used_to_make_costly_gifts_8kz.wav");
                let mut file = File::create(output_path).expect("Could not create file");
                file.write_all(&wav_buffer)
                    .expect("Could not write to file");
            }
            Err(err) => {
                eprintln!("Error generating wav buffer: {}", err);
            }
        }
    }

    #[test]
    fn test_audio_to_mono_f32_averages_channels() {
        let audio = AudioData::new(
            16,
            2,
            48_000,
            interleave_vecs_i16(&[vec![32767, -32768], vec![-32768, 32767]]),
            EncodingFlag::PCMSigned,
            Endianness::LittleEndian,
        );

        let mono = audio_to_mono_f32(&audio).unwrap();
        assert_eq!(mono.len(), 2);
        assert!(mono[0].abs() < 0.01);
        assert!(mono[1].abs() < 0.01);
    }

    #[test]
    fn streaming_normalizer_converts_pcm_without_intermediate_channel_planes() {
        let mut normalizer = StreamingStereo48kNormalizer::new();
        let big_endian = AudioData::new(
            16,
            2,
            48_000,
            [i16::MAX.to_be_bytes(), i16::MIN.to_be_bytes()].concat(),
            EncodingFlag::PCMSigned,
            Endianness::BigEndian,
        );
        let block = normalizer.push(&big_endian).unwrap().unwrap();
        assert!((block.left[0] - 32_767.0 / 32_768.0).abs() < f32::EPSILON);
        assert_eq!(block.right[0], -1.0);

        let mut normalizer = StreamingStereo48kNormalizer::new();
        let signed_24 = AudioData::new(
            24,
            2,
            48_000,
            vec![0xff, 0xff, 0x7f, 0x00, 0x00, 0x80],
            EncodingFlag::PCMSigned,
            Endianness::LittleEndian,
        );
        let block = normalizer.push(&signed_24).unwrap().unwrap();
        assert!((block.left[0] - 8_388_607.0 / 8_388_608.0).abs() < f32::EPSILON);
        assert_eq!(block.right[0], -1.0);

        let mut normalizer = StreamingStereo48kNormalizer::new();
        let float_mono = AudioData::new(
            32,
            1,
            48_000,
            f32::NAN.to_le_bytes().to_vec(),
            EncodingFlag::PCMFloat,
            Endianness::LittleEndian,
        );
        let block = normalizer.push(&float_mono).unwrap().unwrap();
        assert_eq!(block.left, vec![0.0]);
        assert_eq!(block.right, block.left);
    }

    #[test]
    fn test_f32_roundtrip_le_bytes() {
        let input = vec![0.0f32, 0.25, -0.5, 1.0];
        let bytes = f32s_to_le_bytes(&input);
        let output = f32s_from_le_bytes(&bytes).unwrap();
        assert_eq!(output, input);
    }

    #[test]
    fn test_f32_from_le_bytes_rejects_truncated_input() {
        let error = f32s_from_le_bytes(&[0, 1, 2]).unwrap_err();
        assert!(error.contains("multiple of 4"));
    }

    fn pcm16_audio(sample_rate: u32, channels: &[Vec<i16>]) -> AudioData {
        AudioData::new(
            16,
            channels.len() as u8,
            sample_rate,
            interleave_vecs_i16(channels),
            EncodingFlag::PCMSigned,
            Endianness::LittleEndian,
        )
    }

    fn normalize_blocks(blocks: &[AudioData]) -> (Vec<f32>, Vec<f32>) {
        let mut normalizer = StreamingStereo48kNormalizer::new();
        let mut left = Vec::new();
        let mut right = Vec::new();
        for block in blocks {
            if let Some(output) = normalizer.push(block).unwrap() {
                left.extend(output.left);
                right.extend(output.right);
            }
        }
        if let Some(output) = normalizer.finish().unwrap() {
            left.extend(output.left);
            right.extend(output.right);
        }
        (left, right)
    }

    #[test]
    fn streaming_stereo_normalizer_duplicates_mono_without_resampling() {
        let audio = pcm16_audio(48_000, &[vec![i16::MIN, 0, i16::MAX]]);
        let (left, right) = normalize_blocks(&[audio]);
        assert_eq!(left, right);
        assert_eq!(left.len(), 3);
        assert_eq!(left[0], -1.0);
        assert_eq!(left[1], 0.0);
    }

    #[test]
    fn streaming_stereo_normalizer_is_independent_of_decoder_chunking() {
        let left: Vec<i16> = (0..100).map(|sample| sample * 211 - 10_000).collect();
        let right: Vec<i16> = (0..100).map(|sample| 10_000 - sample * 173).collect();
        let whole = pcm16_audio(44_100, &[left.clone(), right.clone()]);
        let chunks = [
            pcm16_audio(44_100, &[left[..37].to_vec(), right[..37].to_vec()]),
            pcm16_audio(44_100, &[left[37..73].to_vec(), right[37..73].to_vec()]),
            pcm16_audio(44_100, &[left[73..].to_vec(), right[73..].to_vec()]),
        ];
        let expected = normalize_blocks(&[whole]);
        let actual = normalize_blocks(&chunks);
        assert_eq!(actual, expected);
        assert_eq!(actual.0.len(), 109);
    }
}
