mod types;

use std::fs::File;
use std::io::Read;
use std::io::Seek;
use std::io::SeekFrom;
use std::io::{BufRead, BufReader};
use std::process;

use crate::types::{
    get_audio_config, get_config, get_sampling_rate_and_bits_per_sample, AudioConfig, EncodingFlag,
};

use opus::{Application, Channels, Decoder, Encoder};

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
    frame_sizes: Vec<u16>,
}

pub fn wav_to_opus_stream<R: Read>(mut reader: R) -> Result<Vec<u8>, String> {
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
    let sample_rate = u32::from_le_bytes(fmt_chunk[12..16].try_into().unwrap());
    let bits_per_sample = u16::from_le_bytes(fmt_chunk[22..24].try_into().unwrap()) as u8;
    let channel_count = u16::from_le_bytes(fmt_chunk[10..12].try_into().unwrap()) as usize;

    // Move position to after "fmt " chunk
    let chunk_size =
        u32::from_le_bytes(buffer[position + 4..position + 8].try_into().unwrap()) as usize;
    position += chunk_size + 8;

    while &buffer[position..position + 4] != b"data" {
        let chunk_size =
            u32::from_le_bytes(buffer[position + 4..position + 8].try_into().unwrap()) as usize;
        position += chunk_size + 8; // Advance to next chunk
    }

    let data_chunk = &buffer[position + 8..]; // Skip "data" and size

    let frame_size: u16 = 480;
    let encoded_data = opus_stream_from_raw(
        &data_chunk,
        sample_rate,
        bits_per_sample,
        channel_count,
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

    let mut encoder = Encoder::new(48000, Channels::Mono, Application::Audio).unwrap();
    encoder.set_bitrate(opus::Bitrate::Bits(96000));

    let mut encoded_data = Vec::new();
    let mut offset: u32 = 0;
    let mut offsets = Vec::new();

    for chunk in data.chunks(chunk_size) {
        let flag = if chunk.len() < 480 {
            EncodingFlag::PCM
        } else {
            EncodingFlag::PCM
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
            for len in header.frame_sizes {
                offset += len as u32;
            }
            offset += 2 + 2 * header.channel_count as u32;
            encoded_data.extend(packet_chunk);
        }
    }

    // Prepend offset_count to encoded_data
    let mut final_encoded_data = Vec::new();
    for i in 0..4 {
        final_encoded_data.push(((offsets.len() >> (i * 8)) & 0xFF) as u8);
    }

    // Prepend frame size to encoded_data
    for i in 0..2 {
        final_encoded_data.push(((frame_size >> (i * 8)) & 0xFF) as u8);
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
    src: Vec<u8>,
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

    let frame_count = src.len() as u32 / bytes_per_sample as u32 / channel_count as u32;
    let len = frame_count as usize * channel_count * bytes_per_sample as usize;
    let mut new_len = 0;
    let mut result = vec![0u8; len as usize];
    let mut frame_sizes = vec![0u16; channel_count];

    match &format {
        EncodingFlag::Opus => {
            let mut dsts = vec![Vec::new(); channel_count];
            for i in 0..channel_count {
                dsts[i as usize] = vec![0i16; frame_count as usize];
            }
            match src_bits_per_sample {
                16 => {
                    if channel_count == 1 {
                        for j in 0..(src.len() / 2) {
                            let sample = i16::from_le_bytes([src[2 * j], src[2 * j + 1]]);
                            dsts[0][j] = sample;
                        }
                    } else {
                        let deinterleaved_samples = deinterleave_vecs_i16(&src, channel_count);
                        for (i, channel) in deinterleaved_samples.iter().enumerate() {
                            for (j, &sample) in channel.iter().enumerate() {
                                dsts[i][j] = sample;
                            }
                        }
                    }
                }
                32 => {
                    for i in 0..channel_count {
                        for j in 0..frame_count {
                            let src_offset = (j * channel_count as u32 + i as u32) as usize * 4;
                            let sample = f32::from_le_bytes([
                                src[src_offset],
                                src[src_offset + 1],
                                src[src_offset + 2],
                                src[src_offset + 3],
                            ]);
                            let sample_i16 = (sample * i16::MAX as f32) as i16;
                            dsts[i][j as usize] = sample_i16;
                        }
                    }
                }
                _ => {
                    return Err(format!(
                        "Unsupported bits per_sample: {}",
                        src_bits_per_sample
                    ))
                }
            }
            let mut compressed_frames = vec![0u8; len];
            for i in 0..channel_count {
                match encoder.encode(&dsts[i as usize], &mut compressed_frames[new_len..]) {
                    Ok(num_bytes) => {
                        new_len += num_bytes as usize;
                        frame_sizes[i as usize] = num_bytes as u16;
                    }
                    Err(err) => {
                        if frame_count > 2880 {
                            return Err(
                                "Buffer size exceeds the maximum allowed frame size for Opus"
                                    .to_string(),
                            );
                        } else {
                            return Err(format!(
                                "Opus encoding failed for {} frames: {}",
                                frame_count, err
                            ));
                        }
                    }
                }
                result = compressed_frames[..new_len].to_vec();
            }
            let mut result_chunks = Vec::new();
            let chunk_header = encode_audio_packet_header(
                format,
                config.clone(),
                channel_count as u8,
                &frame_sizes,
            );
            result_chunks.push([&chunk_header[..], &result].concat());
            Ok(result_chunks)
        }
        EncodingFlag::PCM => {
            let mut dsts = vec![Vec::new(); channel_count as usize];
            match bits_per_sample {
                16 => match src_bits_per_sample {
                    16 => {
                        let deinterleaved_samples = deinterleave_vecs_i16(&src, channel_count);
                        for (i, channel) in deinterleaved_samples.iter().enumerate() {
                            for &sample in channel {
                                let sample_bytes = sample.to_le_bytes();
                                dsts[i].push(sample_bytes[0]);
                                dsts[i].push(sample_bytes[1]);
                            }
                        }
                    }
                    32 => {
                        let deinterleaved_samples = deinterleave_vecs_f32(&src, channel_count);
                        for (i, channel) in deinterleaved_samples.iter().enumerate() {
                            for &sample in channel {
                                let sample_i32 = (sample * i32::MAX as f32) as i32;
                                let sample_bytes = sample_i32.to_le_bytes();
                                dsts[i].extend_from_slice(&sample_bytes);
                            }
                        }
                    }
                    _ => {
                        return Err(format!(
                            "Unsupported source bits per sample: {}",
                            src_bits_per_sample
                        ))
                    }
                },
                32 => match src_bits_per_sample {
                    16 => {
                        let deinterleaved_samples = deinterleave_vecs_i16(&src, channel_count);
                        for (i, channel) in deinterleaved_samples.iter().enumerate() {
                            for &sample in channel {
                                let sample_f32 = sample as f32 / i16::MAX as f32;
                                let sample_bytes = sample_f32.to_le_bytes();
                                dsts[i].extend_from_slice(&sample_bytes);
                            }
                        }
                    }
                    32 => {
                        let deinterleaved_samples = deinterleave_vecs_f32(&src, channel_count);
                        for (i, channel) in deinterleaved_samples.iter().enumerate() {
                            for &sample in channel {
                                let sample_bytes = sample.to_le_bytes();
                                dsts[i].extend_from_slice(&sample_bytes);
                            }
                        }
                    }
                    _ => {
                        return Err(format!(
                            "Unsupported source bits per_sample: {}",
                            src_bits_per_sample
                        ))
                    }
                },
                _ => return Err(format!("Unsupported bits per_sample: {}", bits_per_sample)),
            }

            let header_size = 2 + 2 * channel_count;
            let mut result_chunks = Vec::new();

            if max_payload_size == 0 || len <= max_payload_size - header_size {
                for i in 0..channel_count {
                    frame_sizes[i] = dsts[i].len() as u16;
                }

                let mut chunk = Vec::new();
                let header = encode_audio_packet_header(
                    format,
                    config.clone(),
                    channel_count as u8,
                    &frame_sizes,
                );
                chunk.extend_from_slice(&header);
                for dst in dsts {
                    chunk.extend_from_slice(&dst);
                }

                result_chunks.push(chunk);
            } else {
                let max_audio_payload = max_payload_size - header_size;
                let total_size = len;
                let chunk_count = (total_size + max_audio_payload - 1) / max_audio_payload;

                for chunk_index in 0..chunk_count {
                    let mut chunk_data = Vec::new();
                    let mut frame_size = 0;

                    for channel in &dsts {
                        let start = chunk_index * max_audio_payload / channel_count;
                        let end = start + max_audio_payload.min(channel.len() - start);

                        chunk_data.extend_from_slice(&channel[start..end]);
                        frame_size = end - start;
                    }

                    let header = encode_audio_packet_header(
                        format,
                        config.clone(),
                        channel_count as u8,
                        &vec![frame_size as u16; channel_count],
                    );

                    let mut chunk = Vec::new();
                    chunk.extend_from_slice(&header);
                    chunk.extend_from_slice(&chunk_data);

                    result_chunks.push(chunk);
                }
            }

            Ok(result_chunks)
        }
    }
}

fn decode_audio_packet(buffer: Vec<u8>, decoder: &mut Decoder) -> Option<AudioList> {
    let header = decode_audio_packet_header(&buffer);
    let channel_count = header.channel_count as usize;
    let encoding_flag = header.encoding;
    let bytes_per_sample = header.bits_per_sample as usize / 8;
    let header_size = 2 + 2 * channel_count;
    let len = buffer.len() - header_size;
    let mut sample_count: usize = len / (channel_count * bytes_per_sample);

    let samples = match encoding_flag {
        EncodingFlag::PCM => {
            let sample_count: usize = len / (channel_count * bytes_per_sample);
            let mut samples = vec![vec![0.0f32; sample_count]; channel_count];
            if bytes_per_sample == 2 {
                for i in 0..channel_count {
                    for j in 0..sample_count {
                        let sample_bytes = &buffer[header_size
                            + j * channel_count * bytes_per_sample
                            + i * bytes_per_sample
                            ..header_size
                                + j * channel_count * bytes_per_sample
                                + (i + 1) * bytes_per_sample];
                        let sample_i16 = i16::from_le_bytes([sample_bytes[0], sample_bytes[1]]);
                        let sample_f32 = f32::from(sample_i16) / f32::from(std::i16::MAX);
                        samples[i][j] = sample_f32;
                    }
                }
            } else if bytes_per_sample == 3 {
                for i in 0..channel_count {
                    for j in 0..sample_count {
                        let sample_bytes = &buffer[header_size
                            + j * channel_count * bytes_per_sample
                            + i * bytes_per_sample
                            ..header_size
                                + j * channel_count * bytes_per_sample
                                + (i + 1) * bytes_per_sample];
                        let sample_i24 = i32::from_le_bytes([
                            sample_bytes[0],
                            sample_bytes[1],
                            sample_bytes[2],
                            0,
                        ]);
                        let sample_f32 = sample_i24 as f32 / (1 << 23) as f32;
                        samples[i][j] = sample_f32;
                    }
                }
            } else {
                for i in 0..channel_count {
                    for j in 0..sample_count {
                        let sample_bytes = &buffer[header_size
                            + j * channel_count * bytes_per_sample
                            + i * bytes_per_sample
                            ..header_size
                                + j * channel_count * bytes_per_sample
                                + (i + 1) * bytes_per_sample];
                        let sample_f32 = f32::from_le_bytes([
                            sample_bytes[0],
                            sample_bytes[1],
                            sample_bytes[2],
                            sample_bytes[3],
                        ]);
                        samples[i][j] = sample_f32;
                    }
                }
            }
            samples
        }
        EncodingFlag::Opus => {
            sample_count = 480;
            let mut samples = vec![vec![0f32; sample_count]; channel_count];
            let mut offset = header_size as usize;
            for i in 0..channel_count {
                let frame_size = header.frame_sizes[i as usize] as usize;
                let mut dst = vec![0i16; 480];
                let num_samples = decoder
                    .decode(&buffer[offset..offset + frame_size], &mut dst, false)
                    .unwrap_or(0);
                for j in 0..num_samples {
                    samples[i][j] = dst[j] as f32 / 32768.0;
                }
                offset += frame_size;
            }
            samples
        }
        _ => return None,
    };

    Some(AudioList {
        channels: samples,
        sampling_rate: header.sampling_rate as usize,
        sample_count: sample_count,
    })
}

fn encode_audio_packet_header(
    encoding: &EncodingFlag,
    config: AudioConfig,
    channel_count: u8,
    frame_sizes: &[u16],
) -> Vec<u8> {
    let mut flag: u8 = 0;
    if encoding == &EncodingFlag::Opus {
        flag = 1;
    };

    let mut id = config as u8;
    id |= flag << 5;
    let mut data = id.to_le_bytes().to_vec();
    data.push(channel_count);
    for size in frame_sizes {
        data.extend_from_slice(&size.to_le_bytes());
    }
    data
}

fn decode_audio_packet_header(data: &Vec<u8>) -> AudioPacketHeader {
    let flag = if (data[0] & 0x20) == 0 {
        EncodingFlag::PCM
    } else {
        EncodingFlag::Opus
    };
    let config_id = data[0] & 0x1F; // Extract the  config ID from the first byte of the header
    let config = get_config(config_id);
    let channel_count = data[1];

    let mut frame_sizes = vec![0; channel_count as usize];
    for i in 0..channel_count as usize {
        let offset = 2 + (i * 2);
        frame_sizes[i] = u16::from_le_bytes([data[offset], data[offset + 1]]);
    }
    let (sampling_rate, bits_per_sample) = get_sampling_rate_and_bits_per_sample(config);
    return AudioPacketHeader {
        encoding: flag,
        sampling_rate: sampling_rate,
        bits_per_sample: bits_per_sample,
        channel_count: channel_count,
        frame_sizes: frame_sizes,
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

    fn default_encoder(sampling_rate: u32) -> Encoder {
        Encoder::new(sampling_rate, Channels::Mono, Application::Audio).unwrap()
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
        let encoded = encode_audio_packet_header(
            &EncodingFlag::Opus,
            AudioConfig::Hz44100Bit32,
            2,
            &[400u16, 404u16],
        );
        let got = decode_audio_packet_header(&encoded);

        assert_eq!(got.encoding, EncodingFlag::Opus);
        assert_eq!(got.sampling_rate, 44100);
        assert_eq!(got.bits_per_sample, 32);
        assert_eq!(got.channel_count, 2);
        assert_eq!(got.frame_sizes, [400u16, 404u16]);
    }

    #[test]
    fn test_encode_audio_packet_header_pcm() {
        let got = encode_audio_packet_header(
            &EncodingFlag::PCM,
            AudioConfig::Hz44100Bit32,
            2,
            &[400u16, 404u16],
        );
        let flag = if (got[0] & 0x20) == 0 {
            EncodingFlag::PCM
        } else {
            EncodingFlag::Opus
        };
        assert_eq!(flag, EncodingFlag::PCM);
    }

    #[test]
    fn test_encode_audio_packet_header_opus() {
        let got = encode_audio_packet_header(
            &EncodingFlag::Opus,
            AudioConfig::Hz44100Bit32,
            2,
            &[400u16, 404u16],
        );
        let flag = if (got[0] & 0x20) == 0 {
            EncodingFlag::PCM
        } else {
            EncodingFlag::Opus
        };
        assert_eq!(flag, EncodingFlag::Opus);
    }

    #[test]
    fn test_encode_audio_packet_header() {
        let got = encode_audio_packet_header(
            &EncodingFlag::Opus,
            AudioConfig::Hz44100Bit32,
            2,
            &[400u16, 404u16],
        );
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
        assert_eq!(got.len() * std::mem::size_of::<u8>(), 6);
        assert_eq!(flag, EncodingFlag::Opus);
        assert_eq!(decoded_config, AudioConfig::Hz44100Bit32);
        assert_eq!(
            got[2..6],
            [400u16.to_le_bytes(), 404u16.to_le_bytes()].concat()
        );
    }

    #[ignore]
    #[test]
    fn test_encode_decode_opus_i16() {
        let buffer = read_test_file("./test/i16.wav");
        let mut pcm_data = buffer[44..].to_vec();
        let channel_count = 2;
        let sampling_rate = 48000;
        let bits_per_sample = 16;
        let byte_count_limit = 960 * 2;
        pcm_data.truncate(byte_count_limit as usize);
        let mut encoder = default_encoder(sampling_rate);

        let encoded_buffers = encode_audio_packet(
            AudioConfig::Hz44100Bit16,
            pcm_data,
            channel_count,
            bits_per_sample,
            &EncodingFlag::Opus,
            &mut encoder,
            0,
        )
        .unwrap();

        let mut decoded_samples_total = Vec::new();

        for encoded_buffer in encoded_buffers {
            let mut decoder = Decoder::new(sampling_rate, Channels::Mono).unwrap();
            let decoded_samples = decode_audio_packet(encoded_buffer, &mut decoder).unwrap();
            decoded_samples_total.push(decoded_samples);
        }

        // Concatenate channels from all decoded samples
        let mut channels_total = Vec::new();
        for i in 0..channel_count {
            let mut channel_data = Vec::new();
            for decoded_samples in &decoded_samples_total {
                channel_data.extend_from_slice(&decoded_samples.channels[i as usize]);
            }
            channels_total.push(channel_data);
        }

        assert_eq!(channels_total.len(), channel_count as usize);
        for i in 0..channel_count {
            assert_eq!(channels_total[i as usize].len(), 480);
        }
    }

    #[ignore]
    #[test]
    fn test_encode_decode_opus_f32() {
        let buffer = read_test_file("./test/f32.wav");
        let mut pcm_data = buffer[44..].to_vec();
        let channel_count = 2;
        let sampling_rate = 48000;
        let bits_per_sample = 32;
        let byte_count_limit = 960 * 4;
        pcm_data.truncate(byte_count_limit as usize);
        let mut encoder = default_encoder(sampling_rate);

        let encoded_buffers = encode_audio_packet(
            AudioConfig::Hz44100Bit32,
            pcm_data,
            channel_count,
            bits_per_sample,
            &EncodingFlag::Opus,
            &mut encoder,
            0,
        )
        .unwrap();

        let mut decoded_samples_total = Vec::new();

        for encoded_buffer in encoded_buffers {
            let mut decoder = Decoder::new(sampling_rate, Channels::Mono).unwrap();
            let decoded_samples = decode_audio_packet(encoded_buffer, &mut decoder).unwrap();
            decoded_samples_total.push(decoded_samples);
        }

        // Concatenate channels from all decoded samples
        let mut channels_total = Vec::new();
        for i in 0..channel_count {
            let mut channel_data = Vec::new();
            for decoded_samples in &decoded_samples_total {
                channel_data.extend_from_slice(&decoded_samples.channels[i as usize]);
            }
            channels_total.push(channel_data);
        }

        assert_eq!(channels_total.len(), channel_count as usize);
        for i in 0..channel_count {
            assert_eq!(channels_total[i as usize].len(), 480);
        }
    }

    fn test_encode_decode_common(bits_per_sample: u32) {
        let buffer = read_test_file("./test/f32.wav");
        let pcm_data = buffer[44..].to_vec();
        let channel_count = 2;
        let sampling_rate = 48000;
        let frame_count = pcm_data.len() as u32 / (channel_count * bits_per_sample / 8) as u32;
        let mut encoder = default_encoder(sampling_rate);

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

        for encoded_buffer in encoded_buffers {
            let mut decoder = Decoder::new(sampling_rate, Channels::Mono).unwrap();
            let decoded_samples = decode_audio_packet(encoded_buffer, &mut decoder).unwrap();
            decoded_samples_total.push(decoded_samples);
        }

        // Concatenate channels from all decoded samples
        let mut channels_total = Vec::new();
        for i in 0..channel_count {
            let mut channel_data = Vec::new();
            for decoded_samples in &decoded_samples_total {
                channel_data.extend_from_slice(&decoded_samples.channels[i as usize]);
            }
            channels_total.push(channel_data);
        }

        assert_eq!(channels_total.len(), channel_count as usize);
        for i in 0..channel_count {
            assert_eq!(channels_total[i as usize].len(), frame_count as usize);
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
    fn test_wav_to_pcm_stream() {
        let file_path = "test/i16.wav";
        let file = File::open(&file_path).expect("unable to open file");
        let data = wav_to_opus_stream(file).unwrap();

        let n_packets = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
        assert_eq!(n_packets, 735);

        let n_samples = u16::from_le_bytes(data[4..6].try_into().unwrap());
        assert_eq!(n_samples, 480);

        let mut i: usize = 0;
        let mut pkt_offsets = Vec::new();

        let cues_start: usize = 6;
        for i in 0..n_packets {
            let start = cues_start + i * 4;
            let u32_value = u32::from_le_bytes(data[start..start + 4].try_into().unwrap());
            pkt_offsets.push(u32_value);
        }
        assert_eq!(pkt_offsets.len(), n_packets);

        let n_channels = 2;
        let mut all_frames: Vec<Vec<Vec<i16>>> =
            vec![vec![vec![0; n_samples as usize]; n_channels]; n_packets];

        let mut all_frames: Vec<Vec<Vec<i16>>> = Vec::new();
        let cues_end: usize = (n_packets * 4) + 6;
        let pkt_header_size: usize = 6;

        // sanity check the first i16 in the payload
        let bytes: [u8; 2] = data[cues_end + 6..cues_end + 8]
            .try_into()
            .expect("Insufficient bytes to read");
        let value = i16::from_le_bytes(bytes);
        assert_eq!(value, 569);

        for i in pkt_offsets {
            let start = cues_end + i as usize;
            let b = Vec::from(&data[start..start + pkt_header_size]);
            let header = decode_audio_packet_header(&b);
            assert_eq!(header.bits_per_sample, 16);
            assert_eq!(header.frame_sizes[0], 960);
            assert_eq!(header.frame_sizes[1], 960);

            let mut offset = 0;
            let mut decoded_frames: Vec<Vec<i16>> = Vec::new();
            for size in header.frame_sizes {
                let fstart = start + pkt_header_size + offset;
                let frame = Vec::from(&data[fstart..fstart + size as usize]);
                assert_eq!(frame.len(), 960);
                let decoded_frame: Vec<i16> = frame
                    .chunks_exact(2)
                    .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect();

                decoded_frames.push(decoded_frame);
                offset += size as usize;
            }
            all_frames.push(decoded_frames);
        }

        // sanity check
        assert_eq!(all_frames[0][0][0], 569);

        let total_samples = n_samples as usize * n_packets as usize;
        let mut reduced_frames: Vec<Vec<i16>> = vec![vec![0; total_samples]; n_channels];
        for (i, packets) in all_frames.iter().enumerate() {
            for (j, frames) in packets.iter().enumerate() {
                for (k, &sample) in frames.iter().enumerate() {
                    reduced_frames[j][i * n_samples as usize + k] = sample;
                }
            }
        }

        // sanity check
        assert_eq!(reduced_frames[0][0], 569);
        assert_eq!(reduced_frames[1][0], 7028);

        // `sox test/i16.wav -t raw -e signed-integer -b 16 --channels 2 - | od -An -t d2 > test/i16.raw`
        let file = File::open("./test/i16.raw").expect("Failed to open file");
        let reader = BufReader::new(file);

        let mut samples: Vec<i16> = Vec::new();
        let mut samples: Vec<i16> = Vec::new();

        for line in reader
            .lines()
            .filter(|line| !line.as_ref().unwrap().contains('*'))
        {
            if let Ok(line) = line {
                let values: Vec<i16> = line
                    .split_whitespace()
                    .map(|value| value.parse::<i16>().expect("Failed to parse value"))
                    .collect();

                samples.extend(values);
            }
        }

        let interleaved_frames: Vec<i16> = (0..total_samples)
            .flat_map(|i| reduced_frames.iter().map(move |channel| channel[i]))
            .collect();


        // TODO: Debug zero padding at end
        assert_eq!(&interleaved_frames[0..10000], &samples[0..10000]);
    }
}
