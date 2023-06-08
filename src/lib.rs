mod graph;
mod types;

use libopus::decoder::*;
use libopus::encoder::*;
use std::convert::TryInto;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::process;

use crate::types::{
    get_audio_config, get_config, get_sampling_rate_and_bits_per_sample, AudioConfig, EncodingFlag,
};

use graph::*;

const HEADER_SIZE: usize = 4;

pub struct AudioList {
    channels: Vec<Vec<f32>>,
    sample_count: usize,
    sampling_rate: usize,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct AudioPacketHeader {
    encoding: EncodingFlag,
    sampling_rate: u32,
    bits_per_sample: u8,
    channel_count: u8,
    frame_size: u16,
}

pub struct AudioFileData {
    bits_per_sample: u8,
    channel_count: u8,
    data: Vec<u8>,
    sampling_rate: u32,
}

pub fn parse_wav<R: Read>(mut reader: R) -> Result<AudioFileData, String> {
    let mut buffer = Vec::new();
    reader
        .read_to_end(&mut buffer)
        .map_err(|err| err.to_string())?;

    if &buffer[..4] != b"RIFF" || &buffer[8..12] != b"WAVE" {
        return Err("Not a WAV file".to_string());
    }

    let mut position = 12; // After "WAVE"

    while &buffer[position..position + 4] != b"fmt " {
        let chunk_size =
            u32::from_le_bytes(buffer[position + 4..position + 8].try_into().unwrap()) as usize;
        position += chunk_size + 8; // Advance to next chunk
    }

    let fmt_chunk = &buffer[position..position + 24];
    let sample_rate = u32::from_le_bytes(fmt_chunk[12..16].try_into().unwrap()) as u32;
    let bits_per_sample = u16::from_le_bytes(fmt_chunk[22..24].try_into().unwrap()) as u8;
    let channel_count = u16::from_le_bytes(fmt_chunk[10..12].try_into().unwrap()) as u8;

    // Move position to after "fmt " chunk
    let chunk_size =
        u32::from_le_bytes(buffer[position + 4..position + 8].try_into().unwrap()) as usize;
    position += chunk_size + 8;

    while &buffer[position..position + 4] != b"data" {
        let chunk_size =
            u32::from_le_bytes(buffer[position + 4..position + 8].try_into().unwrap()) as usize;
        position += chunk_size + 8; // Advance to next chunk
    }

    let data_chunk = buffer[position + 8..].to_vec(); // Skip "data" and size

    let result = AudioFileData {
        bits_per_sample: bits_per_sample,
        channel_count: channel_count,
        sampling_rate: sample_rate,
        data: data_chunk,
    };

    Ok(result)
}

pub fn wav_to_opus_stream<R: Read>(mut reader: R) -> Result<Vec<u8>, String> {
    let result = parse_wav(reader)?;
    let frame_size: u16 = 240;
    let encoded_data = opus_stream_from_raw(
        &result.data,
        result.sampling_rate,
        result.bits_per_sample,
        result.channel_count as usize,
        frame_size,
    )?;

    Ok(encoded_data)
}

fn opus_stream_from_raw(
    data: &[u8],
    sampling_rate: u32,
    bits_per_sample: u8,
    channel_count: usize,
    frame_size: u16,
) -> Result<Vec<u8>, String> {
    let bytes_per_sample = bits_per_sample / 8;
    let chunk_size = frame_size as usize * channel_count * bytes_per_sample as usize;

    let d = deinterleave_vecs_i16(data, channel_count);
    save_waveform(&d, 1600, 48, "out/wave.png");

    const WIN: usize = 768;
    const HOP: usize = 256;
    const NORM: f32 = 0.0;

    let chunk_size = WIN * bytes_per_sample as usize * channel_count;
    let mut channel_frames: Vec<Vec<Vec<f32>>> = vec![Vec::new(); channel_count];

    for chunk in data.chunks(chunk_size) {
        let data: Vec<PcmData> = match bits_per_sample {
            16 => deinterleave_vecs_i16(chunk, channel_count)
                .into_iter()
                .map(|mut d| {
                    if d.len() % WIN != 0 {
                        d.extend(vec![0; WIN - d.len()]);
                    }
                    PcmData::I16(d)
                })
                .collect(),
            32 => deinterleave_vecs_f32(chunk, channel_count)
                .into_iter()
                .map(|mut d| {
                    if d.len() % WIN != 0 {
                        d.extend(vec![0.0; WIN - d.len()]);
                    }
                    PcmData::F32(d)
                })
                .collect(),
            _ => continue,
        };

        for (i, pcm_data) in data.into_iter().enumerate() {
            let frame_data = frame_spectrogram(pcm_data, WIN, HOP, NORM);
            channel_frames[i].extend(frame_data);
        }
    }

    save_spectrogram(channel_frames, true, "out/sono");

    let mut encoder = Encoder::create(48000, 2, 1, 1, &[0u8, 1u8], Application::Audio).unwrap();
    encoder
        .set_option(OPUS_SET_BITRATE_REQUEST, 128000)
        .unwrap();

    let mut encoded_data = Vec::new();
    let mut offset: u32 = 0;
    let mut offsets = Vec::new();

    let mut i = 0;
    for chunk in data.chunks(chunk_size) {
        let flag = if chunk.len() < chunk_size {
            EncodingFlag::PCM
        } else {
            EncodingFlag::Opus
        };
        let packets = encode_audio_packet(
            AudioConfig::Hz44100Bit16,
            chunk.to_vec(),
            channel_count,
            bits_per_sample,
            &flag,
            &mut encoder,
            0,
        )?;

        for packet_chunk in packets {
            let header = decode_audio_packet_header(&packet_chunk);
            offsets.push(offset);
            offset += HEADER_SIZE as u32 + header.frame_size as u32;
            encoded_data.extend(packet_chunk);
        }
    }

    // Prepend offset_count to encoded_data
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

    Ok(final_encoded_data)
}

fn encode_audio_packet(
    inputFormat: AudioConfig,
    buf: Vec<u8>,
    channel_count: usize,
    bits_per_sample: u8,
    format: &EncodingFlag,
    encoder: &mut Encoder,
    max_payload_size: usize,
) -> Result<Vec<Vec<u8>>, String> {
    let (sampling_rate, src_bits_per_sample) = get_sampling_rate_and_bits_per_sample(inputFormat);

    let config = match get_audio_config(sampling_rate, bits_per_sample) {
        Ok(c) => c,
        Err(err) => return Err(err),
    };

    let bytes_per_sample = bits_per_sample / 8;
    let frame_size = buf.len() as u32 / bytes_per_sample as u32;
    let len = frame_size as usize * bytes_per_sample as usize;

    let mut data = vec![0u8; len as usize];

    match &format {
        EncodingFlag::Opus => {
            let mut src: Vec<i16> = Vec::new();
            match src_bits_per_sample {
                16 => {
                    for bytes in buf.chunks_exact(2) {
                        let sample = i16::from_le_bytes([bytes[0], bytes[1]]);
                        src.push(sample);
                    }
                }
                32 => {
                    for bytes in buf.chunks_exact(4) {
                        let sample = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                        let scaled_sample = (sample * 32767.0) as i16;
                        src.push(scaled_sample);
                    }
                }
                _ => {
                    return Err(format!(
                        "Unsupported bits per_sample: {}",
                        src_bits_per_sample
                    ))
                }
            }

            let mut dst = vec![0u8; frame_size as usize * 2];
            let mut num_bytes = encoder.encode(&src[..], &mut data[..]).unwrap_or(0);
            if num_bytes == 0 {
                return Err("opus enc: got zero bytes".to_string());
            }
            data.truncate(num_bytes);
        }
        EncodingFlag::PCM => match bits_per_sample {
            16 => match src_bits_per_sample {
                16 => {
                    data.clone_from_slice(&buf[..]);
                }
                32 => {
                    for bytes in buf.chunks_exact(4) {
                        let sample = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                        let scaled_sample = (sample * 32767.0) as i16;
                        let bytes = scaled_sample.to_le_bytes();
                        data.extend_from_slice(&bytes);
                    }
                }
                _ => {
                    return Err(format!(
                        "Unsupported src bits per_sample: {}",
                        src_bits_per_sample
                    ))
                }
            },
            32 => match src_bits_per_sample {
                32 => {
                    data.clone_from_slice(&buf[..]);
                }
                _ => {
                    return Err(format!(
                        "Unsupported source bits per_sample: {}",
                        src_bits_per_sample
                    ))
                }
            },
            _ => return Err(format!("Unsupported bits per_sample: {}", bits_per_sample)),
        },
    };

    let mut result_chunks = Vec::new();

    if max_payload_size == 0 || len <= max_payload_size - HEADER_SIZE {
        let mut chunk = Vec::new();
        let header = encode_audio_packet_header(
            format,
            config.clone(),
            channel_count as u8,
            data.len() as u16,
        );
        chunk.extend_from_slice(&header);
        chunk.extend_from_slice(&data);

        result_chunks.push(chunk);
    } else {
        let max_audio_payload = max_payload_size - HEADER_SIZE;
        let total_size = len;
        let chunk_count = (total_size + max_audio_payload - 1) / max_audio_payload;

        for chunk_index in 0..chunk_count {
            let mut chunk_data = Vec::new();
            let mut frame_size = 0;

            let start = chunk_index * max_audio_payload / channel_count;
            let end = start + max_audio_payload.min(chunk_data.len() - start);
            chunk_data.extend_from_slice(&data[start..end]);
            frame_size = end - start;

            let header = encode_audio_packet_header(
                format,
                config.clone(),
                channel_count as u8,
                frame_size as u16,
            );

            let mut chunk = Vec::new();
            chunk.extend_from_slice(&header);
            chunk.extend_from_slice(&chunk_data);

            result_chunks.push(chunk);
        }
    }

    Ok(result_chunks)
}

fn decode_audio_packet(buffer: Vec<u8>, decoder: &mut Decoder) -> Option<AudioList> {
    let header = decode_audio_packet_header(&buffer);
    let channel_count = header.channel_count as usize;
    let encoding_flag = header.encoding;
    let bytes_per_sample = header.bits_per_sample as usize / 8;
    let len = buffer.len() - HEADER_SIZE;
    let mut sample_count: usize = len / (channel_count * bytes_per_sample);
    let data = &buffer[HEADER_SIZE..];
    let mut samples = vec![0.0f32; sample_count * channel_count];

    match encoding_flag {
        EncodingFlag::PCM => {
            if bytes_per_sample == 2 {
                for sample_bytes in data.chunks_exact(2) {
                    let sample_i16 = i16::from_le_bytes([sample_bytes[0], sample_bytes[1]]);
                    let sample_f32 = f32::from(sample_i16) / f32::from(std::i16::MAX);
                    samples.push(sample_f32);
                }
            } else if bytes_per_sample == 3 {
                for sample_bytes in data.chunks_exact(3) {
                    let sample_i24 =
                        i32::from_le_bytes([sample_bytes[0], sample_bytes[1], sample_bytes[2], 0]);
                    let sample_f32 = sample_i24 as f32 / (1 << 23) as f32;
                    samples.push(sample_f32);
                }
            } else if bytes_per_sample == 4 {
                for sample_bytes in data.chunks_exact(4) {
                    let sample_f32 = f32::from_le_bytes([
                        sample_bytes[0],
                        sample_bytes[1],
                        sample_bytes[2],
                        sample_bytes[3],
                    ]);
                    samples.push(sample_f32);
                }
            }
        }
        EncodingFlag::Opus => {
            let mut dst = vec![0i16; sample_count * channel_count];
            let num_samples = decoder.decode(&data[..], &mut dst[..], false).unwrap_or(0);

            for sample_i16 in dst {
                let sample_f32 = f32::from(sample_i16) / f32::from(std::i16::MAX);
                samples.push(sample_f32);
            }
        }
        _ => return None,
    };

    let mut deinterleaved_samples: Vec<Vec<f32>> =
        vec![Vec::with_capacity(samples.len() / channel_count); channel_count];
    for (i, sample) in samples.iter().enumerate() {
        deinterleaved_samples[i % channel_count].push(*sample);
    }

    Some(AudioList {
        channels: deinterleaved_samples,
        sampling_rate: header.sampling_rate as usize,
        sample_count: sample_count,
    })
}

fn encode_audio_packet_header(
    encoding: &EncodingFlag,
    config: AudioConfig,
    channel_count: u8,
    frame_size: u16,
) -> Vec<u8> {
    let mut flag: u8 = 0;
    if encoding == &EncodingFlag::Opus {
        flag = 1;
    };

    let mut id = config as u8;
    id |= flag << 5;
    let mut data = id.to_le_bytes().to_vec();
    data.push(channel_count);
    data.extend_from_slice(&frame_size.to_le_bytes());
    data
}

pub fn decode_audio_packet_header(data: &Vec<u8>) -> AudioPacketHeader {
    let flag = if (data[0] & 0x20) == 0 {
        EncodingFlag::PCM
    } else {
        EncodingFlag::Opus
    };
    let config_id = data[0] & 0x1F; // Extract the  config ID from the first byte of the header
    let config = get_config(config_id);
    let channel_count = data[1];
    let frame_size = u16::from_le_bytes([data[2], data[3]]);
    let (sampling_rate, bits_per_sample) = get_sampling_rate_and_bits_per_sample(config);
    return AudioPacketHeader {
        encoding: flag,
        sampling_rate: sampling_rate,
        bits_per_sample: bits_per_sample,
        channel_count: channel_count,
        frame_size: frame_size,
    };
}

fn deinterleave_vecs_i16(input: &[u8], channel_count: usize) -> Vec<Vec<i16>> {
    let sample_size = input.len() / (channel_count * 2);
    let mut result = vec![vec![0; sample_size]; channel_count];

    for i in 0..sample_size {
        for channel in 0..channel_count {
            let start = (i * channel_count + channel) * 2;
            let value = i16::from_le_bytes([input[start], input[start + 1]]);
            result[channel][i] = value;
        }
    }

    result
}

fn interleave_vecs_i16(channels: &[Vec<i16>]) -> Vec<u8> {
    let channel_count = channels.len();
    let sample_size = channels[0].len();
    let mut result = vec![0; channel_count * sample_size * 2];

    for i in 0..sample_size {
        for channel in 0..channel_count {
            let value = channels[channel][i];
            let bytes = value.to_le_bytes();
            let start = (i * channel_count + channel) * 2;
            result[start] = bytes[0];
            result[start + 1] = bytes[1];
        }
    }

    result
}

fn deinterleave_vecs_24bit(input: &[u8], channel_count: usize) -> Vec<Vec<i32>> {
    let sample_size = input.len() / (channel_count * 3);
    let mut result = vec![vec![0; sample_size]; channel_count];

    for i in 0..sample_size {
        for channel in 0..channel_count {
            let start = (i * channel_count + channel) * 3;
            let value = i32::from_le_bytes([input[start], input[start + 1], input[start + 2], 0]);
            result[channel][i] = value;
        }
    }

    result
}

fn deinterleave_vecs_f32(input: &[u8], channel_count: usize) -> Vec<Vec<f32>> {
    let sample_size = input.len() / (channel_count * 4);
    let mut result = vec![vec![0.0; sample_size]; channel_count];

    for i in 0..sample_size {
        for channel in 0..channel_count {
            let start = (i * channel_count + channel) * 4;
            let value = f32::from_le_bytes(input[start..start + 4].try_into().unwrap());
            result[channel][i] = value;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Read;

    fn read_test_file(filename: &str) -> Vec<u8> {
        let mut file = File::open(filename).unwrap();
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();
        buffer
    }

    #[test]
    fn test_deinterleave_vecs_i16() {
        let input = vec![1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0]; // Little Endian u16 values [1, 2, 3, 4, 5, 6]
        let result = deinterleave_vecs_i16(&input, 2);
        assert_eq!(result, vec![vec![1, 3, 5], vec![2, 4, 6]]);
    }

    #[test]
    fn test_interleave_vecs_i16() {
        let input = vec![vec![1, 3, 5], vec![2, 4, 6]];
        let result = interleave_vecs_i16(&input);
        assert_eq!(result, vec![1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0]);
    }

    #[test]
    fn test_deinterleave_vecs_24bit() {
        let input = vec![1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0, 5, 0, 0, 6, 0, 0]; // Little Endian u24 values [1, 2, 3, 4, 5, 6]
        let result = deinterleave_vecs_24bit(&input, 2);
        assert_eq!(result, vec![vec![1, 3, 5], vec![2, 4, 6]]);
    }

    #[test]
    fn test_deinterleave_vecs_f32() {
        let input = vec![
            0, 0, 128, 63, 0, 0, 0, 64, // f32: 1.0, 2.0
            0, 0, 64, 64, 0, 0, 128, 64, // f32: 3.0, 4.0
            0, 0, 160, 64, 0, 0, 192, 64, // f32: 5.0, 6.0
        ];
        let result = deinterleave_vecs_f32(&input, 2);
        assert_eq!(
            result,
            vec![
                vec![1.0, 3.0, 5.0], // channel 1
                vec![2.0, 4.0, 6.0], // channel 2
            ]
        );
    }

    #[test]
    fn test_decode_audio_packet_header() {
        let encoded =
            encode_audio_packet_header(&EncodingFlag::Opus, AudioConfig::Hz44100Bit32, 2, 400u16);
        let got = decode_audio_packet_header(&encoded);

        assert_eq!(got.encoding, EncodingFlag::Opus);
        assert_eq!(got.sampling_rate, 44100);
        assert_eq!(got.bits_per_sample, 32);
        assert_eq!(got.channel_count, 2);
        assert_eq!(got.frame_size, 400u16);
    }

    #[test]
    fn test_encode_audio_packet_header_pcm() {
        let got =
            encode_audio_packet_header(&EncodingFlag::PCM, AudioConfig::Hz44100Bit32, 2, 400u16);
        let flag = if (got[0] & 0x20) == 0 {
            EncodingFlag::PCM
        } else {
            EncodingFlag::Opus
        };
        assert_eq!(flag, EncodingFlag::PCM);
    }

    #[test]
    fn test_encode_audio_packet_header_opus() {
        let got =
            encode_audio_packet_header(&EncodingFlag::Opus, AudioConfig::Hz44100Bit32, 2, 400u16);
        let flag = if (got[0] & 0x20) == 0 {
            EncodingFlag::PCM
        } else {
            EncodingFlag::Opus
        };
        assert_eq!(flag, EncodingFlag::Opus);
    }

    #[test]
    fn test_encode_audio_packet_header() {
        let got =
            encode_audio_packet_header(&EncodingFlag::Opus, AudioConfig::Hz44100Bit32, 2, 400u16);
        let flag = if (got[0] & 0x20) == 0 {
            EncodingFlag::PCM
        } else {
            EncodingFlag::Opus
        };
        let config_id = got[0] & 0x1F; // Extract the audio config ID from the first byte of the header
        let decoded_config = match config_id {
            2 => AudioConfig::Hz44100Bit32,
            _ => panic!("Invalid config ID"),
        };
        assert_eq!(got.len() * std::mem::size_of::<u8>(), HEADER_SIZE);
        assert_eq!(flag, EncodingFlag::Opus);
        assert_eq!(decoded_config, AudioConfig::Hz44100Bit32);
        assert_eq!(got[2..4], 400u16.to_le_bytes(),);
    }

    fn test_encode_decode_common(bits_per_sample: u32) {
        let buffer = read_test_file("./test/f32.wav");
        let pcm_data = buffer[44..].to_vec();
        let channel_count = 2;
        let sampling_rate = 48000;
        let frame_count = pcm_data.len() as u32 / (channel_count * bits_per_sample / 8) as u32;
        let mut encoder = Encoder::create(48000, 2, 1, 1, &[0u8, 1u8], Application::Audio).unwrap();

        let encoded_buffers = encode_audio_packet(
            AudioConfig::Hz44100Bit32,
            pcm_data,
            channel_count as usize,
            bits_per_sample as u8,
            &EncodingFlag::PCM,
            &mut encoder,
            0,
        )
        .unwrap();

        let mut decoded_samples_total = Vec::new();
        let mut decoder = Decoder::create(48000, 2, 1, 1, &[0u8, 1u8]).unwrap();

        for encoded_buffer in encoded_buffers {
            let decoded_samples = decode_audio_packet(encoded_buffer, &mut decoder).unwrap();
            decoded_samples_total.push(decoded_samples);
        }
    }

    #[test]
    fn test_encode_decode_pcm32() {
        test_encode_decode_common(32);
    }

    #[test]
    fn test_encode_decode_pcm16() {
        test_encode_decode_common(16);
    }

    #[test]
    fn test_opus_encoding() {
        let file_path = "test/i16.wav";
        let file = File::open(&file_path).expect("unable to open file");
        let data = parse_wav(file).unwrap();

        let frame_size = 240;
        let chunk_size = frame_size as usize * data.channel_count as usize * 2 as usize;

        let mut enc = Encoder::create(48000, 2, 1, 1, &[0u8, 1u8], Application::Audio).unwrap();
        enc.set_option(OPUS_SET_BITRATE_REQUEST, 64000).unwrap();
        let mut dec = Decoder::create(48000, 2, 1, 1, &[0u8, 1u8]).unwrap();

        // Write WAV header
        let mut buffer: Vec<u8> = Vec::new();
        buffer.extend_from_slice(b"RIFF");
        buffer.extend_from_slice(&0u32.to_le_bytes()); // Placeholder for size
        buffer.extend_from_slice(b"WAVEfmt ");
        buffer.extend_from_slice(&16u32.to_le_bytes()); // Subchunk1 Size: 16 for PCM
        buffer.extend_from_slice(&1u16.to_le_bytes()); // Audio Format: 1 for PCM
        buffer.extend_from_slice(&2u16.to_le_bytes()); // Num Channels: 2 for stereo
        buffer.extend_from_slice(&44100u32.to_le_bytes()); // Sample Rate: 44100
        buffer.extend_from_slice(&(44100 * 2 * 16 / 8 as u32).to_le_bytes()); // Byte Rate: SampleRate * NumChannels * BitsPerSample/8
        buffer.extend_from_slice(&((2 * 16 / 8) as u16).to_le_bytes()); // Block Align: NumChannels * BitsPerSample/8
        buffer.extend_from_slice(&16u16.to_le_bytes()); // Bits Per Sample: 16
        buffer.extend_from_slice(b"data");
        buffer.extend_from_slice(&0u32.to_le_bytes()); // Placeholder for Subchunk2 Size

        for chunk in data.data.chunks(chunk_size) {
            let mut src: Vec<i16> = Vec::new();
            for bytes in chunk.chunks_exact(2) {
                let sample = i16::from_le_bytes([bytes[0], bytes[1]]);
                src.push(sample);
            }
            let mut compressed_dst = vec![0u8; frame_size as usize * 2];
            let mut num_bytes = enc.encode(&src[..], &mut compressed_dst[..]).unwrap_or(0);
            assert!(num_bytes > 1);
            compressed_dst.truncate(num_bytes);
            let mut dst = vec![0i16; frame_size * 2];
            let num_samples = dec
                .decode(&compressed_dst[..], &mut dst[..], false)
                .unwrap_or(0);
            assert_eq!(num_samples, frame_size);
            for sample in dst {
                buffer.extend_from_slice(&sample.to_le_bytes());
            }
        }

        let file_size = buffer.len() as u32;
        let data_size = file_size - 44;
        buffer[4..8].copy_from_slice(&file_size.to_le_bytes());
        buffer[40..44].copy_from_slice(&data_size.to_le_bytes());

        // Write buffer to file
        std::fs::write("test/opus.wav", &buffer).unwrap();
    }
}
