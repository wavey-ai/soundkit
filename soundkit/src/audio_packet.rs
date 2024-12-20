use crate::audio_bytes::{f32le_to_i32, s16le_to_i32, s24le_to_i32, s32le_to_i32, s32le_to_s24};
use byteorder::{ByteOrder, LE};
use bytes::{Bytes, BytesMut};
use frame_header::{EncodingFlag, Endianness, FrameHeader};

pub trait Encoder {
    fn new(
        sample_rate: u32,
        bits_per_sample: u32,
        channels: u32,
        frame_size: u32,
        bitrate: u32,
    ) -> Self;
    fn init(&mut self) -> Result<(), String>;
    // used for libOpus
    fn encode_i16(&mut self, input: &[i16], output: &mut [u8]) -> Result<usize, String>;
    // used for libFLAC
    fn encode_i32(&mut self, input: &[i32], output: &mut [u8]) -> Result<usize, String>;
    fn reset(&mut self) -> Result<(), String>;
}

pub trait Decoder {
    fn decode_i16(&mut self, input: &[u8], output: &mut [i16], fec: bool) -> Result<usize, String>;
    fn decode_i32(&mut self, input: &[u8], output: &mut [i32], fec: bool) -> Result<usize, String>;
}

pub struct AudioList {
    pub channels: Vec<Vec<f32>>,
    pub sample_count: usize,
    pub sampling_rate: usize,
}

pub fn get_encoding_flag(header_bytes: &[u8]) -> Result<EncodingFlag, String> {
    if header_bytes.len() < 4 {
        return Err("Header too small to extract encoding flag".to_string());
    }

    // Extract the first 4 bytes and interpret as a big-endian u32
    let header = u32::from_be_bytes(header_bytes[..4].try_into().unwrap());

    // Extract the encoding flag (3 bits starting at bit 29)
    let encoding = match (header >> 29) & 0x7 {
        0 => EncodingFlag::PCMSigned,
        1 => EncodingFlag::PCMFloat,
        2 => EncodingFlag::Opus,
        3 => EncodingFlag::FLAC,
        4 => EncodingFlag::AAC,
        _ => return Err("Unknown encoding flag".to_string()),
    };

    Ok(encoding)
}

pub fn encode_audio_packet<E: Encoder>(
    encoding_format: EncodingFlag,
    encoder: &mut E,
    fullbuf: &[u8],
) -> Result<BytesMut, String> {
    let header = FrameHeader::decode(&mut &fullbuf[..]).unwrap();
    let buf = &fullbuf[header.size()..];
    let mut data = vec![0u8; buf.len() * 2];

    match encoding_format {
        EncodingFlag::FLAC => {
            let src = match header.bits_per_sample() {
                16 => s16le_to_i32(buf),
                24 => s24le_to_i32(buf),
                32 => {
                    if header.encoding() == &EncodingFlag::PCMSigned {
                        s32le_to_i32(buf)
                    } else {
                        f32le_to_i32(buf)
                    }
                }
                _ => {
                    unreachable!()
                }
            };

            let num_bytes;

            match encoder.encode_i32(&src[..], &mut data[..]) {
                Ok(n) => num_bytes = n,
                Err(e) => {
                    return Err(format!("Failed to encode chunk {:?}", e));
                }
            }

            if num_bytes == 0 {
                return Err("Flac encoding: zero bytes".to_string());
            }
            data.truncate(num_bytes);
        }
        EncodingFlag::Opus | EncodingFlag::AAC => {
            let mut src: Vec<i16> = Vec::new();

            match header.bits_per_sample() {
                16 => {
                    for bytes in buf.chunks_exact(2) {
                        src.push(i16::from_le_bytes([bytes[0], bytes[1]]));
                    }
                }
                24 => {
                    for bytes in buf.chunks_exact(3) {
                        src.push((LE::read_i24(&bytes) >> 8) as i16);
                    }
                }
                32 => {
                    for bytes in buf.chunks_exact(4) {
                        let sample = if header.encoding() == &EncodingFlag::PCMSigned {
                            // Read the 32-bit signed integer and scale to i16 range
                            let s32_sample =
                                i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                            (s32_sample as i64 * i16::MAX as i64 / i32::MAX as i64) as i32
                        } else {
                            // Read the 32-bit floating-point sample and scale to i16 range
                            let float_sample =
                                f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                            (float_sample * 32767.0) as i32
                        };

                        // Clamp the sample to i16 range to avoid overflow
                        let scaled_sample = sample.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                        src.push(scaled_sample);
                    }
                }
                _ => {
                    return Err(format!(
                        "Unsupported bits per sample: {}",
                        header.bits_per_sample()
                    ))
                }
            }

            let num_bytes = encoder
                .encode_i16(&src[..], &mut data[..])
                .to_owned()
                .map_err(|e| format!("Opus/AAC encoding failed: {}", e))?;

            if num_bytes == 0 {
                return Err("Opus/AAC encoding: zero bytes".to_string());
            }
            data.truncate(num_bytes);
        }
        EncodingFlag::PCMFloat => {
            data.clone_from_slice(&buf[..]);
        }
        _ => {}
    }

    let mut chunk = Vec::new();
    let header = FrameHeader::new(
        encoding_format,
        header.sample_size() as u16,
        header.sample_rate() as u32,
        header.channels() as u8,
        header.bits_per_sample() as u8,
        Endianness::LittleEndian,
        header.id(),
    );
    let mut buffer = Vec::new();
    header?.encode(&mut buffer).unwrap();
    chunk.extend_from_slice(&buffer);
    chunk.extend_from_slice(&data);

    Ok(BytesMut::from(&chunk[..]))
}

pub fn decode_audio_packet_scratch<D: Decoder>(
    buffer: Bytes,
    decoder: &mut D,
    scratch: &mut Vec<f32>,
) -> Result<FrameHeader, String> {
    let header = FrameHeader::decode(&mut &buffer[..20])
        .map_err(|e| format!("Failed to decode header: {}", e))?;
    let channel_count = header.channels() as usize;
    let data = &buffer[header.size()..];

    match header.encoding() {
        EncodingFlag::PCMSigned => match header.bits_per_sample() {
            16 => {
                for (i, sample_bytes) in data.chunks_exact(2).enumerate() {
                    let sample_i16 = i16::from_le_bytes([sample_bytes[0], sample_bytes[1]]);
                    scratch[i] = f32::from(sample_i16) / f32::from(std::i16::MAX);
                }
            }
            24 => {
                for (i, sample_bytes) in data.chunks_exact(3).enumerate() {
                    let sample_i24 = LE::read_i24(sample_bytes);
                    scratch[i] = sample_i24 as f32 / (1 << 23) as f32;
                }
            }
            32 => {
                for (i, sample_bytes) in data.chunks_exact(4).enumerate() {
                    let sample_i32 = i32::from_le_bytes([
                        sample_bytes[0],
                        sample_bytes[1],
                        sample_bytes[2],
                        sample_bytes[3],
                    ]);
                    scratch[i] = sample_i32 as f32 / std::i32::MAX as f32;
                }
            }
            _ => {
                return Err(format!(
                    "Unsupported bits per sample: {}",
                    header.bits_per_sample()
                ))
            }
        },
        EncodingFlag::PCMFloat => {
            for (i, sample_bytes) in data.chunks_exact(4).enumerate() {
                scratch[i] = f32::from_le_bytes([
                    sample_bytes[0],
                    sample_bytes[1],
                    sample_bytes[2],
                    sample_bytes[3],
                ]);
            }
        }
        EncodingFlag::Opus => {
            let mut dst = vec![0i16; header.sample_size() as usize * channel_count];
            let num_samples_decoded = decoder
                .decode_i16(data, &mut dst, false)
                .map_err(|e| format!("Opus decoding failed: {}", e))?;

            for (i, sample_i16) in dst[..num_samples_decoded].iter().enumerate() {
                scratch[i] = f32::from(*sample_i16) / f32::from(std::i16::MAX);
            }
        }
        _ => return Err("Unsupported encoding type".to_string()),
    }

    Ok(header)
}

pub fn decode_audio_packet<D: Decoder>(buffer: Vec<u8>, decoder: &mut D) -> Option<AudioList> {
    let header = FrameHeader::decode(&mut buffer.as_slice()).unwrap();
    let channel_count = header.channels() as usize;
    let data = &buffer[header.size()..];
    let mut samples = vec![0.0f32; header.sample_size() as usize * channel_count];

    match header.encoding() {
        EncodingFlag::PCMSigned => match header.bits_per_sample() {
            16 => {
                for sample_bytes in data.chunks_exact(2) {
                    let sample_i16 = i16::from_le_bytes([sample_bytes[0], sample_bytes[1]]);
                    let sample_f32 = f32::from(sample_i16) / f32::from(std::i16::MAX);
                    samples.push(sample_f32);
                }
            }
            24 => {
                for sample_bytes in data.chunks_exact(3) {
                    let sample_i24 = LE::read_i24(sample_bytes);
                    let sample_f32 = sample_i24 as f32 / (1 << 23) as f32;
                    samples.push(sample_f32);
                }
            }
            32 => {
                for sample_bytes in data.chunks_exact(4) {
                    let sample_i32: i32 = i32::from_le_bytes([
                        sample_bytes[0],
                        sample_bytes[1],
                        sample_bytes[2],
                        sample_bytes[3],
                    ]);
                    let sample_f32: f32 = sample_i32 as f32 / std::i32::MAX as f32;
                    samples.push(sample_f32);
                }
            }
            _ => todo!(),
        },
        EncodingFlag::PCMFloat => {
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
        EncodingFlag::Opus => {
            let mut dst = vec![0i16; header.sample_size() as usize * channel_count];
            let _num_samples_decoded = decoder
                .decode_i16(&data[..], &mut dst[..], false)
                .unwrap_or(0);

            for sample_i16 in dst {
                let sample_f32 = f32::from(sample_i16) / f32::from(std::i16::MAX);
                samples.push(sample_f32);
            }
        }
        _ => todo!(),
    }

    let mut deinterleaved_samples =
        vec![Vec::with_capacity(samples.len() / channel_count); header.channels() as usize];

    for (i, sample) in samples.iter().enumerate() {
        deinterleaved_samples[i % channel_count].push(*sample);
    }

    Some(AudioList {
        channels: deinterleaved_samples,
        sampling_rate: header.sample_rate() as usize,
        sample_count: header.sample_size() as usize,
    })
}
