mod types;

use std::fs::File;
use std::io::Read;
use std::io::Seek;
use std::io::SeekFrom;

use crate::types::{
    AudioConfig,
    EncodingFlag,
    get_audio_config,
    get_config,
    get_sampling_rate_and_bits_per_sample,
};

use opus::{
    Application,
    Channels,
    Decoder,
    Encoder,
};

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

pub fn opus_stream_headers<R: Read + Seek>(mut reader: R) -> Result<(), String> {
    loop {
        let mut header_buffer = vec![0; 2]; // first 2 bytes for initial decoding
        match reader.read_exact(&mut header_buffer) {
            Ok(_) => {
                let channel_count = header_buffer[1]; // get the channel count
                let header_size = 2 + channel_count as usize * 2; // calculate total header size
                header_buffer.resize(header_size, 0); // resize the header_buffer
                reader.read_exact(&mut header_buffer[2..]).map_err(|err| err.to_string())?; // read the rest of the header

                let header = decode_audio_packet_header(&header_buffer);
                eprintln!("{:?}", header);
                let skip_size = header.frame_sizes.iter().sum::<u16>() as u64;

                // Skip the data of this frame
                match reader.seek(SeekFrom::Current(skip_size as i64)) {
                    Ok(_) => (),
                    Err(err) => return Err(err.to_string()),
                };
            }
            Err(err) => {
                if err.kind() == std::io::ErrorKind::UnexpectedEof {
                    // End of file, break the loop
                    break;
                } else {
                    // Another error occurred, return it
                    return Err(err.to_string());
                }
            }
        }
    }

    Ok(())
}

pub fn wav_to_opus_stream<R: Read>(mut reader: R) -> Result<Vec<u8>, String> {
    let mut buffer = Vec::new();
    reader.read_to_end(&mut buffer).map_err(|err| err.to_string())?;

    if &buffer[..4] != b"RIFF" || &buffer[8..12] != b"WAVE" {
        return Err("Not a WAV file".to_string());
    }

    let mut position = 12; // After "WAVE"

    while &buffer[position..position+4] != b"fmt " {
        let chunk_size = u32::from_le_bytes(buffer[position+4..position+8].try_into().unwrap()) as usize;
        position += chunk_size + 8; // Advance to next chunk
    }

    let fmt_chunk = &buffer[position..position+24];
    let sample_rate = u32::from_le_bytes(fmt_chunk[12..16].try_into().unwrap());
    let bits_per_sample = u16::from_le_bytes(fmt_chunk[22..24].try_into().unwrap()) as u8;
    let channel_count = u16::from_le_bytes(fmt_chunk[10..12].try_into().unwrap()) as usize;

    // Move position to after "fmt " chunk
    let chunk_size = u32::from_le_bytes(buffer[position+4..position+8].try_into().unwrap()) as usize;
    position += chunk_size + 8;

    while &buffer[position..position+4] != b"data" {
        let chunk_size = u32::from_le_bytes(buffer[position+4..position+8].try_into().unwrap()) as usize;
        position += chunk_size + 8; // Advance to next chunk
    }

    let data_chunk = &buffer[position+8..]; // Skip "data" and size
    let frame_size: u16 = 480;
    let encoded_data = opus_stream_from_raw(data_chunk, sample_rate, bits_per_sample, channel_count, frame_size)?;

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
    let mut offset_count: u32 = 0;

    for chunk in data.chunks(chunk_size) {
        let flag = if chunk.len() < 480 {
            EncodingFlag::PCM
        } else {
            EncodingFlag::Opus
        };

        let packet = encode_audio_packet(
            AudioConfig::Hz44100Bit16,
            chunk.to_vec(),
            channel_count,
            bits_per_sample,
            &flag,
            &mut encoder,
            0,
        )?;

        for packet_chunk in packet {
            let header = decode_audio_packet_header(&packet_chunk);
            // Convert the offset to bytes and append to the encoded data
            for i in 0..4 {
                encoded_data.push(((offset >> (i * 8)) & 0xFF) as u8);
            }
            for len in header.frame_sizes {
                offset += len as u32;
            }
            offset_count += 1;
            encoded_data.extend(packet_chunk);
        }
    }

    // Prepend offset_count to encoded_data
    let mut final_encoded_data = vec![0; 6];
    for i in 0..4 {
        final_encoded_data[i] = ((offset_count >> (i * 8)) & 0xFF) as u8;
    }
    // Prepend frame duration to encoded_data
    let frame_duration: u16 = 10;
    for i in 0..2 {
        final_encoded_data[4+i] = ((frame_duration >> (i * 8)) & 0xFF) as u8;
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

    let frame_count = src.len() as u32 / 4 / channel_count as u32;

    let bytes_per_sample = bits_per_sample / 8;
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

            let mut compressed_frames = vec![0u8; len];
            for i in 0..channel_count {
                match encoder.encode(&dsts[i as usize], &mut compressed_frames[new_len..]) {
                    Ok(num_bytes) => {
                        new_len += num_bytes as usize;
                        frame_sizes[i as usize] = num_bytes as u16;
                    }
                    Err(err) => {
                        if frame_count > 2880 {
                            return Err("Buffer size exceeds the maximum allowed frame size for Opus".to_string());
                        } else {
                            return Err(format!("Opus encoding failed for {} frames: {}", frame_count, err));
                        }
                    }
                }
                result = compressed_frames[..new_len].to_vec();
            }
            let mut result_chunks = Vec::new();
            let chunk_header =
                encode_audio_packet_header(format, config.clone(), channel_count as u8, &frame_sizes);
            result_chunks.push([&chunk_header[..], &result].concat());
            Ok(result_chunks)
        },
        EncodingFlag::PCM => {
            let mut dsts = vec![Vec::new(); channel_count as usize];
            match bits_per_sample {
                16 => {
                    match src_bits_per_sample {
                        16 => {
                            let deinterleaved_samples = deinterleave_vecs_u16(&src, channel_count);
                            for (i, channel) in deinterleaved_samples.iter().enumerate() {
                                for &sample in channel {
                                    let sample_bytes = sample.to_le_bytes();
                                    dsts[i].push(sample_bytes[0]);
                                    dsts[i].push(sample_bytes[1]);
                                }
                            }
                        },
                        32 => {
                            let deinterleaved_samples = deinterleave_vecs_f32(&src, channel_count);
                            for (i, channel) in deinterleaved_samples.iter().enumerate() {
                                for &sample in channel {
                                    let sample_i32 = (sample * i32::MAX as f32) as i32;
                                    let sample_bytes = sample_i32.to_le_bytes();
                                    dsts[i].extend_from_slice(&sample_bytes);
                                }
                            }
                        },
                        _ => return Err(format!("Unsupported source bits per sample: {}", src_bits_per_sample)),
                    }
                },
                32 => {
                    match src_bits_per_sample {
                        16 => {
                            let deinterleaved_samples = deinterleave_vecs_u16(&src, channel_count);
                            for (i, channel) in deinterleaved_samples.iter().enumerate() {
                                for &sample in channel {
                                    let sample_f32 = sample as f32 / i16::MAX as f32;
                                    let sample_bytes = sample_f32.to_le_bytes();
                                    dsts[i].extend_from_slice(&sample_bytes);
                                }
                            }
                        },
                        32 => {
                            let deinterleaved_samples = deinterleave_vecs_f32(&src, channel_count);
                            for (i, channel) in deinterleaved_samples.iter().enumerate() {
                                for &sample in channel {
                                    let sample_bytes = sample.to_le_bytes();
                                    dsts[i].extend_from_slice(&sample_bytes);
                                }
                            }
                        },
                        _ => return Err(format!("Unsupported source bits per_sample: {}", src_bits_per_sample)),
                    }
                },
                _ => return Err(format!("Unsupported bits per_sample: {}", bits_per_sample)),
            }

            let header_size = 2 + 2 * channel_count;
            let mut result_chunks = Vec::new();

            if max_payload_size == 0 || len <= max_payload_size - header_size {
                for i in 0..channel_count { 
                    frame_sizes[i] = dsts[i].len() as u16;
                };

                let mut chunk = Vec::new();
                let header = encode_audio_packet_header(format, config.clone(), channel_count as u8, &frame_sizes);
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

                    let header = encode_audio_packet_header(format, config.clone(), channel_count as u8, &vec![frame_size as u16; channel_count]);

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

fn deinterleave_vecs_u16(input: &[u8], channel_count: usize) -> Vec<Vec<u16>> {
    let sample_size = input.len() / (channel_count * 2);
    let mut result = vec![vec![0; sample_size]; channel_count];

    for i in 0..sample_size {
        for channel in 0..channel_count {
            let start = (i * channel_count + channel) * 2;
            let value = u16::from_le_bytes([input[start], input[start + 1]]);
            result[channel][i] = value;
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
    fn test_deinterleave_vecs_u16() {
        let input = vec![1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0]; // Little Endian u16 values [1, 2, 3, 4, 5, 6]
        let result = deinterleave_vecs_u16(&input, 2);
        assert_eq!(result, vec![vec![1, 3, 5], vec![2, 4, 6]]);
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

    #[test]
    fn test_encode_decode_opus() {
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
}
