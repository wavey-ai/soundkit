use crate::audio_bytes::{f32le_to_i32, s16le_to_i32, s24le_to_i32, s32le_to_i32, s32le_to_s24};
use crate::audio_types::{EncodingFlag, Endianness};
use byteorder::{ByteOrder, LE};
use bytes::{Bytes, BytesMut};
use std::io::{self, Read, Write};

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

pub fn patch_sample_size(header_bytes: &mut [u8], new_sample_size: u16) -> Result<(), String> {
    // Ensure the header is large enough to contain the sample size field
    if header_bytes.len() < 4 {
        return Err("Header too small to update sample size".to_string());
    }

    // Decode the existing header
    let mut header = u32::from_be_bytes(header_bytes[..4].try_into().unwrap());

    // Update the sample size (11 bits starting at bit 18)
    header &= !(0x7FF << 18); // Clear the existing sample size bits
    header |= (new_sample_size as u32 & 0x7FF) << 18; // Set the new sample size

    // Encode the updated header back into bytes
    header_bytes[..4].copy_from_slice(&header.to_be_bytes());

    Ok(())
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
        header.id,
    );
    let mut buffer = Vec::new();
    header.encode(&mut buffer).unwrap();
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
    let channel_count = header.channels as usize;
    let data = &buffer[header.size()..];

    match header.encoding {
        EncodingFlag::PCMSigned => match header.bits_per_sample {
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
                    header.bits_per_sample
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
            let mut dst = vec![0i16; header.sample_size as usize * channel_count];
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
    let channel_count = header.channels as usize;
    let data = &buffer[header.size()..];
    let mut samples = vec![0.0f32; header.sample_size as usize * channel_count];

    match header.encoding {
        EncodingFlag::PCMSigned => match header.bits_per_sample {
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
            let mut dst = vec![0i16; header.sample_size as usize * channel_count];
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
        vec![Vec::with_capacity(samples.len() / channel_count); header.channels as usize];

    for (i, sample) in samples.iter().enumerate() {
        deinterleaved_samples[i % channel_count].push(*sample);
    }

    Some(AudioList {
        channels: deinterleaved_samples,
        sampling_rate: header.sample_rate as usize,
        sample_count: header.sample_size as usize,
    })
}

#[derive(Debug)]
pub struct FrameHeader {
    encoding: EncodingFlag,
    sample_size: u16,
    sample_rate: u32,
    channels: u8,
    bits_per_sample: u8,
    endianness: Endianness,
    id: Option<u64>,
}

impl FrameHeader {
    pub fn new(
        encoding: EncodingFlag,
        sample_size: u16,
        sample_rate: u32,
        channels: u8,
        bits_per_sample: u8,
        endianness: Endianness,
        id: Option<u64>,
    ) -> Self {
        assert!(channels <= 16, "Channel count must not exceed 16");
        assert!(bits_per_sample <= 32, "Bits per sample must not exceed 32");
        FrameHeader {
            encoding,
            sample_size,
            sample_rate,
            channels,
            bits_per_sample,
            endianness,
            id,
        }
    }

    pub fn encoding(&self) -> &EncodingFlag {
        &self.encoding
    }

    pub fn sample_size(&self) -> u16 {
        self.sample_size
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn channels(&self) -> u8 {
        self.channels
    }

    pub fn bits_per_sample(&self) -> u8 {
        self.bits_per_sample
    }

    pub fn endianness(&self) -> &Endianness {
        &self.endianness
    }

    pub fn id(&self) -> Option<u64> {
        self.id
    }

    pub fn size(&self) -> usize {
        if self.id.is_some() {
            12
        } else {
            4
        }
    }

    pub fn encode<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        let mut header: u32 = 0;

        // Encoding flag (3 bits)
        header |= (self.encoding as u32) << 29;

        // Data length (11 bits)
        header |= (self.sample_size as u32) << 18;

        // Sample rate (3 bits)
        let sample_rate_code = match self.sample_rate {
            44100 => 0,
            48000 => 1,
            88200 => 2,
            96000 => 3,
            176400 => 4,
            192000 => 5,
            _ => 7, // Custom rate, will be written separately
        };
        header |= (sample_rate_code as u32) << 15;

        // Channels (4 bits)
        header |= ((self.channels - 1) as u32) << 11;

        // Bits per sample (5 bits)
        header |= ((self.bits_per_sample - 1) as u32) << 6;

        // Endianness (1 bit)
        header |= (self.endianness as u32) << 5;

        // Reserved bits (5 bits)
        // Use the first reserved bit to indicate presence of ID
        header |= (self.id.is_some() as u32) << 4;
        // The remaining 4 bits are still set to zero

        // Write the header
        writer.write_all(&header.to_be_bytes())?;

        // Write custom sample rate if needed
        if sample_rate_code == 7 {
            writer.write_all(&self.sample_rate.to_be_bytes())?;
        }

        // Write ID if present
        if self.id.is_some() {
            writer.write_all(&self.id.unwrap().to_be_bytes())?;
        }

        Ok(())
    }

    pub fn decode<R: Read>(reader: &mut R) -> io::Result<Self> {
        let mut header_bytes = [0u8; 4];
        reader.read_exact(&mut header_bytes)?;
        let header = u32::from_be_bytes(header_bytes);

        let encoding = match (header >> 29) & 0x7 {
            0 => EncodingFlag::PCMSigned,
            1 => EncodingFlag::PCMFloat,
            2 => EncodingFlag::Opus,
            3 => EncodingFlag::FLAC,
            4 => EncodingFlag::AAC,
            _ => unreachable!(),
        };

        let sample_size = ((header >> 18) & 0x7FF) as u16;

        let sample_rate_code = (header >> 15) & 0x7;
        let sample_rate = match sample_rate_code {
            0 => 44100,
            1 => 48000,
            2 => 88200,
            3 => 96000,
            4 => 176400,
            5 => 192000,
            7 => {
                let mut rate_bytes = [0u8; 4];
                reader.read_exact(&mut rate_bytes)?;
                u32::from_be_bytes(rate_bytes)
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid sample rate code",
                ))
            }
        };

        let channels = (((header >> 11) & 0xF) + 1) as u8;

        let bits_per_sample = (((header >> 6) & 0x1F) + 1) as u8;

        let endianness = if (header >> 5) & 0x1 == 0 {
            Endianness::LittleEndian
        } else {
            Endianness::BigEndian
        };

        let has_id = (header >> 4) & 0x1 == 1;

        let id = if has_id {
            let mut id_bytes = [0u8; 8];
            reader.read_exact(&mut id_bytes)?;
            Some(u64::from_be_bytes(id_bytes))
        } else {
            None
        };

        Ok(FrameHeader {
            encoding,
            sample_size,
            sample_rate,
            channels,
            bits_per_sample,
            endianness,
            id,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_bytes::{f32le_to_s24, s16le_to_i32, s24le_to_i32};
    use crate::wav::WavStreamProcessor;
    use std::fs::File;
    use std::io::Read;
    use std::io::Write;

    #[test]
    fn test_encode_decode_with_32bit_audio() {
        let original = FrameHeader::new(
            EncodingFlag::PCMSigned,
            1456,
            44100,
            2,
            32,
            Endianness::LittleEndian,
            None,
        );
        let mut buffer = Vec::new();
        original.encode(&mut buffer).unwrap();
        let decoded = FrameHeader::decode(&mut buffer.as_slice()).unwrap();
        assert_eq!(original.encoding, decoded.encoding);
        assert_eq!(original.sample_size, decoded.sample_size);
        assert_eq!(original.sample_rate, decoded.sample_rate);
        assert_eq!(original.channels, decoded.channels);
        assert_eq!(original.bits_per_sample, decoded.bits_per_sample);
        assert_eq!(original.endianness, decoded.endianness);
    }

    #[test]
    fn test_various_bit_depths() {
        for bits in [16, 24, 32] {
            let original = FrameHeader::new(
                EncodingFlag::PCMSigned,
                1024,
                48000,
                2,
                bits,
                Endianness::BigEndian,
                None,
            );
            let mut buffer = Vec::new();
            original.encode(&mut buffer).unwrap();
            let decoded = FrameHeader::decode(&mut buffer.as_slice()).unwrap();
            assert_eq!(original.bits_per_sample, decoded.bits_per_sample);
        }
    }

    #[test]
    #[should_panic(expected = "Bits per sample must not exceed 32")]
    fn test_invalid_bits_per_sample() {
        FrameHeader::new(
            EncodingFlag::PCMSigned,
            1024,
            44100,
            2,
            33,
            Endianness::LittleEndian,
            None,
        );
    }

    #[test]
    fn test_reserved_bits_are_zero() {
        let header = FrameHeader::new(
            EncodingFlag::PCMSigned,
            1024,
            48000,
            2,
            16,
            Endianness::LittleEndian,
            None,
        );
        let mut buffer = Vec::new();
        header.encode(&mut buffer).unwrap();

        // The last byte of the header should have its lower 6 bits as zero
        assert_eq!(buffer[3] & 0b00111111, 0);
    }

    #[test]
    fn test_patch_sample_size() {
        let original_header = FrameHeader::new(
            EncodingFlag::PCMSigned,
            1024,
            44100,
            2,
            16,
            Endianness::LittleEndian,
            None,
        );

        let mut header_bytes = Vec::new();
        original_header.encode(&mut header_bytes).unwrap();

        let new_sample_size = 512;
        patch_sample_size(&mut header_bytes, new_sample_size).expect("Failed to patch sample size");

        let patched_header = FrameHeader::decode(&mut header_bytes.as_slice()).unwrap();
        assert_eq!(patched_header.sample_size(), new_sample_size);

        // Verify that all other fields remain unchanged
        assert_eq!(patched_header.encoding(), original_header.encoding());
        assert_eq!(patched_header.sample_rate(), original_header.sample_rate());
        assert_eq!(patched_header.channels(), original_header.channels());
        assert_eq!(
            patched_header.bits_per_sample(),
            original_header.bits_per_sample()
        );
        assert_eq!(patched_header.endianness(), original_header.endianness());
        assert_eq!(patched_header.id(), original_header.id());
    }
}
