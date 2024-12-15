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
    const MAGIC_BIT: u32 = 1 << 31;
    const VALID_SAMPLE_RATES: [u32; 4] = [44100, 48000, 88200, 96000];

    pub fn new(
        encoding: EncodingFlag,
        sample_size: u16,
        sample_rate: u32,
        channels: u8,
        bits_per_sample: u8,
        endianness: Endianness,
        id: Option<u64>,
    ) -> Result<Self, String> {
        if channels == 0 || channels > 16 {
            return Err("Channel count must be between 1 and 16".to_string());
        }
        if bits_per_sample == 0 || bits_per_sample > 32 {
            return Err("Bits per sample must be between 1 and 32".to_string());
        }
        if sample_size > 0x3FFF {
            return Err("Sample size exceeds maximum value (16383)".to_string());
        }
        if !Self::VALID_SAMPLE_RATES.contains(&sample_rate) {
            return Err(format!(
                "Invalid sample rate: {}. Must be one of: {:?}",
                sample_rate,
                Self::VALID_SAMPLE_RATES
            ));
        }

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

    pub fn validate_header(header_bytes: &[u8]) -> Result<bool, String> {
        if header_bytes.len() < 4 {
            return Err("Header too small".to_string());
        }

        let header = u32::from_be_bytes(header_bytes[..4].try_into().unwrap());

        // Check magic bit
        if header & Self::MAGIC_BIT == 0 {
            return Ok(false);
        }

        // Validate encoding (3 bits)
        let encoding = (header >> 28) & 0x7;
        if encoding > 4 {
            return Ok(false);
        }

        // Validate sample rate (2 bits)
        let sample_rate_code = (header >> 26) & 0x3;
        if sample_rate_code > 3 {
            return Ok(false);
        }

        // Validate channels (4 bits)
        let channels = (((header >> 22) & 0xF) + 1) as u8;
        if channels == 0 || channels > 16 {
            return Ok(false);
        }

        // Validate bits per sample (5 bits)
        let bits = (((header >> 3) & 0x1F) + 1) as u8;
        if bits == 0 || bits > 32 {
            return Ok(false);
        }

        Ok(true)
    }

    pub fn encode<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        let mut header: u32 = Self::MAGIC_BIT;

        // Encoding flag (3 bits)
        header |= (self.encoding as u32) << 28;

        // Sample rate code (2 bits)
        let sample_rate_code = match self.sample_rate {
            44100 => 0,
            48000 => 1,
            88200 => 2,
            96000 => 3,
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "Invalid sample rate",
                ))
            }
        };
        header |= (sample_rate_code as u32) << 26;

        // Channels (4 bits)
        header |= ((self.channels - 1) as u32) << 22;

        // Sample size (14 bits)
        header |= (self.sample_size as u32) << 8;

        // Bits per sample (5 bits)
        header |= ((self.bits_per_sample - 1) as u32) << 3;

        // Endianness (1 bit)
        header |= (self.endianness as u32) << 2;

        // ID present flag (1 bit)
        header |= (self.id.is_some() as u32) << 1;

        writer.write_all(&header.to_be_bytes())?;

        if let Some(id) = self.id {
            writer.write_all(&id.to_be_bytes())?;
        }

        Ok(())
    }

    pub fn decode<R: Read>(reader: &mut R) -> io::Result<Self> {
        let mut header_bytes = [0u8; 4];
        reader.read_exact(&mut header_bytes)?;
        let header = u32::from_be_bytes(header_bytes);

        if header & Self::MAGIC_BIT == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid header magic bit",
            ));
        }

        let encoding = match (header >> 28) & 0x7 {
            0 => EncodingFlag::PCMSigned,
            1 => EncodingFlag::PCMFloat,
            2 => EncodingFlag::Opus,
            3 => EncodingFlag::FLAC,
            4 => EncodingFlag::AAC,
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid encoding flag",
                ))
            }
        };

        let sample_rate = match (header >> 26) & 0x3 {
            0 => 44100,
            1 => 48000,
            2 => 88200,
            3 => 96000,
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid sample rate code",
                ))
            }
        };

        let channels = (((header >> 22) & 0xF) + 1) as u8;
        let sample_size = ((header >> 8) & 0x3FFF) as u16;
        let bits_per_sample = (((header >> 3) & 0x1F) + 1) as u8;
        let endianness = if (header >> 2) & 0x1 == 0 {
            Endianness::LittleEndian
        } else {
            Endianness::BigEndian
        };

        let has_id = (header >> 1) & 0x1 == 1;
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

    pub fn patch_bits_per_sample(header_bytes: &mut [u8], bits: u8) -> Result<(), String> {
        if !Self::validate_header(header_bytes)? {
            return Err("Invalid header".to_string());
        }
        if bits == 0 || bits > 32 {
            return Err("Bits per sample must be between 1 and 32".to_string());
        }

        let mut header = u32::from_be_bytes(header_bytes[..4].try_into().unwrap());
        header &= !(0x1F << 3); // Clear bits per sample
        header |= ((bits - 1) as u32) << 3;
        header_bytes[..4].copy_from_slice(&header.to_be_bytes());
        Ok(())
    }

    pub fn patch_sample_size(header_bytes: &mut [u8], new_sample_size: u16) -> Result<(), String> {
        if !Self::validate_header(header_bytes)? {
            return Err("Invalid header".to_string());
        }

        if new_sample_size > 0x3FFF {
            return Err("Sample size exceeds maximum value (16383)".to_string());
        }

        let mut header = u32::from_be_bytes(header_bytes[..4].try_into().unwrap());
        header &= !(0x3FFF << 8); // Clear sample size bits
        header |= (new_sample_size as u32) << 8;
        header_bytes[..4].copy_from_slice(&header.to_be_bytes());
        Ok(())
    }

    pub fn patch_encoding(header_bytes: &mut [u8], encoding: EncodingFlag) -> Result<(), String> {
        if !Self::validate_header(header_bytes)? {
            return Err("Invalid header".to_string());
        }

        let mut header = u32::from_be_bytes(header_bytes[..4].try_into().unwrap());
        header &= !(0x7 << 28); // Clear encoding bits
        header |= (encoding as u32) << 28;
        header_bytes[..4].copy_from_slice(&header.to_be_bytes());
        Ok(())
    }

    pub fn patch_sample_rate(header_bytes: &mut [u8], sample_rate: u32) -> Result<(), String> {
        if header_bytes.len() < 4 {
            return Err("Header too small".to_string());
        }
        if !Self::VALID_SAMPLE_RATES.contains(&sample_rate) {
            return Err(format!(
                "Invalid sample rate: {}. Must be one of: {:?}",
                sample_rate,
                Self::VALID_SAMPLE_RATES
            ));
        }

        let rate_code = match sample_rate {
            44100 => 0,
            48000 => 1,
            88200 => 2,
            96000 => 3,
            _ => unreachable!(), // Already checked in contains()
        };

        let mut header = u32::from_be_bytes(header_bytes[..4].try_into().unwrap());
        header &= !(0x3 << 26); // Clear sample rate bits
        header |= rate_code << 26;
        header_bytes[..4].copy_from_slice(&header.to_be_bytes());
        Ok(())
    }

    // Getter methods
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn create_test_header() -> Vec<u8> {
        let header = FrameHeader::new(
            EncodingFlag::PCMSigned,
            1024,
            48000,
            2,
            24,
            Endianness::LittleEndian,
            None,
        )
        .unwrap();
        let mut buffer = Vec::new();
        header.encode(&mut buffer).unwrap();
        buffer
    }

    #[test]
    fn test_header_validation() {
        let valid_header = create_test_header();
        assert!(FrameHeader::validate_header(&valid_header).unwrap());

        // Test invalid magic bit
        let mut invalid_magic = valid_header.clone();
        invalid_magic[0] &= 0x7F; // Clear magic bit
        assert!(!FrameHeader::validate_header(&invalid_magic).unwrap());

        // Test invalid header size
        let short_header = vec![0; 2];
        assert!(FrameHeader::validate_header(&short_header).is_err());
    }

    #[test]
    fn test_new_header_validation() {
        // Test valid creation
        assert!(FrameHeader::new(
            EncodingFlag::PCMSigned,
            1024,
            48000,
            2,
            24,
            Endianness::LittleEndian,
            None,
        )
        .is_ok());

        // Test invalid sample size
        assert!(FrameHeader::new(
            EncodingFlag::PCMSigned,
            0x4000, // Too large
            48000,
            2,
            24,
            Endianness::LittleEndian,
            None,
        )
        .is_err());

        // Test invalid sample rate
        assert!(FrameHeader::new(
            EncodingFlag::PCMSigned,
            1024,
            192000, // Not in valid rates
            2,
            24,
            Endianness::LittleEndian,
            None,
        )
        .is_err());

        // Test invalid channels
        assert!(FrameHeader::new(
            EncodingFlag::PCMSigned,
            1024,
            48000,
            17, // Too many channels
            24,
            Endianness::LittleEndian,
            None,
        )
        .is_err());

        // Test invalid bits per sample
        assert!(FrameHeader::new(
            EncodingFlag::PCMSigned,
            1024,
            48000,
            2,
            33, // Too many bits
            Endianness::LittleEndian,
            None,
        )
        .is_err());
    }

    #[test]
    fn test_patch_sample_size() {
        let mut header_bytes = create_test_header();

        // Test valid sample size
        assert!(FrameHeader::patch_sample_size(&mut header_bytes, 2048).is_ok());
        let updated = FrameHeader::decode(&mut &header_bytes[..]).unwrap();
        assert_eq!(updated.sample_size(), 2048);

        // Test maximum value
        assert!(FrameHeader::patch_sample_size(&mut header_bytes, 0x3FFF).is_ok());

        // Test exceeding maximum value
        assert!(FrameHeader::patch_sample_size(&mut header_bytes, 0x4000).is_err());

        // Test with invalid header
        let mut invalid_header = header_bytes.clone();
        invalid_header[0] &= 0x7F; // Clear magic bit
        assert!(FrameHeader::patch_sample_size(&mut invalid_header, 1024).is_err());
    }

    #[test]
    fn test_patch_encoding() {
        let mut header_bytes = create_test_header();

        // Test all encoding types
        let encodings = vec![
            EncodingFlag::PCMSigned,
            EncodingFlag::PCMFloat,
            EncodingFlag::Opus,
            EncodingFlag::FLAC,
            EncodingFlag::AAC,
        ];

        for encoding in encodings {
            assert!(FrameHeader::patch_encoding(&mut header_bytes, encoding).is_ok());
            let updated = FrameHeader::decode(&mut &header_bytes[..]).unwrap();
            assert_eq!(*updated.encoding(), encoding);
        }
    }

    #[test]
    fn test_patch_bits_per_sample() {
        let mut header_bytes = create_test_header();

        // Test common bit depths
        for &bits in &[16, 24, 32] {
            assert!(FrameHeader::patch_bits_per_sample(&mut header_bytes, bits).is_ok());
            let updated = FrameHeader::decode(&mut &header_bytes[..]).unwrap();
            assert_eq!(updated.bits_per_sample(), bits);
        }

        // Test invalid bit depths
        assert!(FrameHeader::patch_bits_per_sample(&mut header_bytes, 0).is_err());
        assert!(FrameHeader::patch_bits_per_sample(&mut header_bytes, 33).is_err());
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let original = FrameHeader::new(
            EncodingFlag::PCMSigned,
            1024,
            48000,
            2,
            24,
            Endianness::LittleEndian,
            Some(12345),
        )
        .unwrap();

        let mut buffer = Vec::new();
        original.encode(&mut buffer).unwrap();

        let decoded = FrameHeader::decode(&mut &buffer[..]).unwrap();

        assert_eq!(*decoded.encoding(), *original.encoding());
        assert_eq!(decoded.sample_size(), original.sample_size());
        assert_eq!(decoded.sample_rate(), original.sample_rate());
        assert_eq!(decoded.channels(), original.channels());
        assert_eq!(decoded.bits_per_sample(), original.bits_per_sample());
        assert_eq!(*decoded.endianness(), *original.endianness());
        assert_eq!(decoded.id(), original.id());
    }

    #[test]
    fn test_field_preservation() {
        let mut header_bytes = create_test_header();
        let original = FrameHeader::decode(&mut &header_bytes[..]).unwrap();

        // Modify sample size and verify other fields remain unchanged
        FrameHeader::patch_sample_size(&mut header_bytes, 2048).unwrap();
        let updated = FrameHeader::decode(&mut &header_bytes[..]).unwrap();

        assert_eq!(updated.sample_size(), 2048); // Changed
        assert_eq!(updated.encoding(), original.encoding()); // Preserved
        assert_eq!(updated.sample_rate(), original.sample_rate()); // Preserved
        assert_eq!(updated.channels(), original.channels()); // Preserved
        assert_eq!(updated.bits_per_sample(), original.bits_per_sample()); // Preserved
        assert_eq!(updated.endianness(), original.endianness()); // Preserved
        assert_eq!(updated.id(), original.id()); // Preserved
    }
}
