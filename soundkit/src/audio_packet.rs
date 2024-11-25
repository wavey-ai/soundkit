use crate::audio_bytes::{f32le_to_i32, s16le_to_i32, s24le_to_i32, s32le_to_i32, s32le_to_s24};
use crate::audio_types::{EncodingFlag, Endianness};
use byteorder::{ByteOrder, LE};
use bytes::BytesMut;
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
        EncodingFlag::Opus => {
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
                            i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
                        } else {
                            f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as i32
                        };
                        let scaled_sample = (sample * 32767) as i16;
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

            let num_bytes = encoder.encode_i16(&src[..], &mut data[..]).unwrap_or(0);
            if num_bytes == 0 {
                return Err("Opus encoding: zero bytes".to_string());
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
        data.len() as u16,
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
    use soundkit_flac::FlacEncoder;
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
    fn test_flac_encoding() {
        let file_path = "testdata/f32le.wav";

        let frame_len = 480 * 2;
        let mut file = File::open(file_path.clone()).unwrap();
        let mut file_buffer = Vec::new();
        file.read_to_end(&mut file_buffer).unwrap();

        let mut processor = WavStreamProcessor::new();
        let audio_data = processor.add(&file_buffer).unwrap().unwrap();

        let mut encoder = FlacEncoder::new(
            audio_data.sampling_rate(),
            audio_data.bits_per_sample() as u32,
            audio_data.channel_count() as u32,
            frame_len as u32,
            5,
        );

        encoder.init().expect("Failed to initialize FLAC encoder");

        let i32_samples = match audio_data.bits_per_sample() {
            16 => {
                // this doesn't scale the 16 bit samples - important!
                s16le_to_i32(audio_data.data())
            }
            24 => s24le_to_i32(audio_data.data()),
            32 => f32le_to_s24(audio_data.data()),
            _ => {
                vec![0i32]
            }
        };

        let chunk_size = frame_len * audio_data.channel_count() as usize;

        let mut frames = Vec::new();

        let sample_rate = audio_data.sampling_rate() as u64;
        let channels = audio_data.channel_count() as u64;
        let samples_per_chunk = chunk_size as u64 / channels;
        let time_base = 1_000;

        let chunk_duration = samples_per_chunk * time_base / sample_rate;

        for (i, chunk) in i32_samples.chunks(chunk_size).enumerate() {
            let mut output_buffer = vec![0u8; 1024 * 10];

            match encoder.encode_i32(chunk, &mut output_buffer) {
                Ok(encoded_len) => {
                    if encoded_len > 0 {
                        let data = &output_buffer[..encoded_len];
                        frames.push(data.to_owned());
                    }
                }
                Err(e) => {
                    panic!("Failed to encode chunk {}: {:?}", i, e);
                }
            }
        }

        let flac_output_file_path = "testdata/test.flac";
        let mut flac_output_file =
            File::create(flac_output_file_path).expect("Failed to create output file");

        for frame in frames {
            flac_output_file
                .write_all(&frame)
                .expect("Failed to write data segment");
        }

        println!("FMP4 file written successfully");
    }
}
