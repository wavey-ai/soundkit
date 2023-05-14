mod types;

use crate::types::{AudioConfig, EncodingFlag, get_audio_config, get_config, get_sampling_rate_and_bits_per_sample};
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

fn encode_audio_packet(
    src: Vec<u8>,
    channel_count: u8,
    sampling_rate: u32,
    bits_per_sample: u8,
    format: &EncodingFlag,
    encoder: &mut Encoder,
) -> Result<Vec<Vec<u8>>, String> {
    let config = match get_audio_config(sampling_rate, bits_per_sample) {
        Ok(c) => c,
        Err(err) => return Err(err),
    };

    let frame_count = src.len() as u32 / 4 / channel_count as u32;

    let bytes_per_sample = bits_per_sample / 8;
    let len = frame_count as usize * channel_count as usize * bytes_per_sample as usize;
    let mut new_len = 0;
    let mut result = vec![0u8; len as usize];
    let mut frame_sizes = vec![0u16; channel_count as usize];

    match &format {
        EncodingFlag::Opus => {
            let mut dsts = vec![Vec::new(); channel_count as usize];
            for i in 0..channel_count {
                dsts[i as usize] = vec![0i16; frame_count as usize];
            }

            for i in 0..channel_count as usize {
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

            let mut compressed_frames = vec![0u8; len];
            for i in 0..channel_count {
                match encoder.encode(&dsts[i as usize], &mut compressed_frames[new_len..]) {
                    Ok(num_bytes) => {
                        if num_bytes < 4 {
                            eprintln!("Unable to encode audio frame");
                            return Err("Opus encoding failed".to_string());
                        }
                        new_len += num_bytes as usize;
                        frame_sizes[i as usize] = num_bytes as u16;
                    }
                    Err(err) => {
                        return Err(format!("Opus encoding failed: {}", err));
                    }
                }
                result = compressed_frames[..new_len].to_vec();
            }
            let mut result_chunks = Vec::new();
            let chunk_header =
                encode_audio_packet_header(format, config.clone(), channel_count, &frame_sizes);
            result_chunks.push([&chunk_header[..], &result].concat());
            Ok(result_chunks)
        }
        EncodingFlag::PCM => {
            if bits_per_sample == 16 {
                new_len = len;
                let mut dst = vec![0u8; new_len];
                // to i16
                for i in 0..dst.len() / 2 {
                    let src_offset = i * 4;
                    let sample = f32::from_le_bytes([
                        src[src_offset],
                        src[src_offset + 1],
                        src[src_offset + 2],
                        src[src_offset + 3],
                    ]);
                    let sample_i16 = (sample * i16::MAX as f32) as i16;
                    let sample_bytes = sample_i16.to_le_bytes();
                    dst[i * 2] = sample_bytes[0];
                    dst[i * 2 + 1] = sample_bytes[1];
                }
            } else if bits_per_sample == 24 {
                new_len = len;
                let mut dst = vec![0u8; new_len];
                // to i24
                for i in 0..dst.len() / 3 {
                    let src_offset = i * 4;
                    let sample = f32::from_le_bytes([
                        src[src_offset],
                        src[src_offset + 1],
                        src[src_offset + 2],
                        src[src_offset + 3],
                    ]);
                    let sample_i24 = (sample * (2i32.pow(23) - 1) as f32) as i32;
                    let sample_bytes = sample_i24.to_le_bytes();
                    dst[i * 3] = sample_bytes[0];
                    dst[i * 3 + 1] = sample_bytes[1];
                    dst[i * 3 + 2] = sample_bytes[2];
                }
            } else if bits_per_sample == 32 {
                result = src;
            }

            let mut result_chunks = Vec::new();
            let max_srt_payload_size = 1456;
            let header =
                encode_audio_packet_header(format, config.clone(), channel_count, &frame_sizes);
            let max_audio_payload = max_srt_payload_size - header.len();
            let mut bytes_per_chunk = max_audio_payload
                / (bytes_per_sample as usize * channel_count as usize)
                * (bytes_per_sample as usize * channel_count as usize);
            for chunk_start in (0..result.len()).step_by(bytes_per_chunk) {
                let chunk_end = usize::min(chunk_start + bytes_per_chunk, result.len());
                let chunk = &result[chunk_start..chunk_end];
                let chunk_header =
                    encode_audio_packet_header(format, config.clone(), channel_count, &frame_sizes);
                result_chunks.push([&chunk_header[..], chunk].concat());
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
    let header_size = channel_count + 2;
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
                if num_samples < 4 {
                    eprintln!("Unable to decode opus packet");
                    return None;
                }
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

fn deinterleave_vecs(input: &[u8], channel_count: usize) -> Vec<Vec<u8>> {
    let sample_size = input.len() / (channel_count * 4);
    let mut result = vec![vec![0; sample_size * 4]; channel_count];

    for i in 0..sample_size {
        for channel in 0..channel_count {
            let start = (i * channel_count + channel) * 4;
            let value = &input[start..start + 4];
            result[channel][i * 4..i * 4 + 4].copy_from_slice(value);
        }
    }

    result
}

fn interleave_vecs(v: &[&[u8]]) -> Vec<u8> {
    let len = v[0].len() / 4;
    let mut result = Vec::with_capacity(len * v.len() * 4);

    for i in 0..len {
        for inner_vec in v.iter() {
            let start = i * 4;
            let bytes = [
                inner_vec[start],
                inner_vec[start + 1],
                inner_vec[start + 2],
                inner_vec[start + 3],
            ];
            let value = f32::from_le_bytes(bytes);
            result.extend_from_slice(&value.to_le_bytes());
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
    fn test_deinterleave_vecs() {
        let input: &[u8] = &[
            1, 0, 0, 0, 3, 0, 0, 0, 5, 0, 0, 0,
            2, 0, 0, 0, 4, 0, 0, 0, 6, 0, 0, 0,
        ];
        let channel_count = 3;
        let expected_output: Vec<Vec<u8>> = vec![
            vec![1, 0, 0, 0, 2, 0, 0, 0],
            vec![3, 0, 0, 0, 4, 0, 0, 0],
            vec![5, 0, 0, 0, 6, 0, 0, 0],
        ];

        let result = deinterleave_vecs(input, channel_count);

        assert_eq!(result, expected_output);
    }

    #[test]
    fn test_interleave_vecs() {
        let input: &[&[u8]] = &[
            &[1, 0, 0, 0, 2, 0, 0, 0],
            &[3, 0, 0, 0, 4, 0, 0, 0],
            &[5, 0, 0, 0, 6, 0, 0, 0],
        ];
        let expected_output: Vec<u8> = vec![
            1, 0, 0, 0, 3, 0, 0, 0, 5, 0, 0, 0,
            2, 0, 0, 0, 4, 0, 0, 0, 6, 0, 0, 0,
        ];

        let result = interleave_vecs(input);

        assert_eq!(result, expected_output);
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

    fn test_encode_decode_common(bits_per_sample: u32) {
        let buffer = read_test_file("./test/f32.wav");
        let pcm_data = buffer[44..].to_vec();
        let channel_count = 2;
        let sampling_rate = 48000;
        let frame_count = pcm_data.len() as u32 / (channel_count * 32 / 8) as u32;
        let mut encoder = default_encoder(sampling_rate);

        let encoded_buffers = encode_audio_packet(
            pcm_data,
            channel_count,
            sampling_rate,
            bits_per_sample as u8,
            &EncodingFlag::PCM,
            &mut encoder,
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
    fn test_encode_decode_pcm24() {
        test_encode_decode_common(24);
    }

    #[test]
    fn test_encode_decode_pcm16() {
        test_encode_decode_common(16);
    }
}
