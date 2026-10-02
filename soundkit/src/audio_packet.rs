use crate::audio_bytes::{f32le_to_i32, s16le_to_i32, s24le_to_i32, s32le_to_i32};
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
    fn decode_f32(&mut self, input: &[u8], output: &mut [f32], fec: bool) -> Result<usize, String>;
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

/// Interleaved f32 samples of a v1 packet payload: 16-bit divided by 32768,
/// 24-bit by 2^23, 32-bit integer by `i32::MAX`, float as stored.
fn payload_to_f32(header: &FrameHeader, data: &[u8]) -> Result<Vec<f32>, String> {
    let samples = match (header.bits_per_sample(), header.encoding()) {
        (16, _) => data
            .chunks_exact(2)
            .map(|bytes| f32::from(i16::from_le_bytes([bytes[0], bytes[1]])) / 32_768.0)
            .collect(),
        (24, _) => data
            .chunks_exact(3)
            .map(|bytes| LE::read_i24(bytes) as f32 / (1 << 23) as f32)
            .collect(),
        (32, EncodingFlag::PCMFloat) => data
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            .collect(),
        (32, _) => data
            .chunks_exact(4)
            .map(|bytes| {
                i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f32
                    / i32::MAX as f32
            })
            .collect(),
        (bits, _) => return Err(format!("Unsupported bits per sample: {bits}")),
    };
    Ok(samples)
}

/// Encodes the PCM payload of a v1 packet as `encoding_format` and returns
/// the new packet: a v1 header, then the encoded payload.
///
/// FLAC, Opus and AAC go through `encoder`. `PCMFloat` converts the samples
/// to 32-bit float. `PCMSigned` keeps the payload as it is. A packet whose
/// header cannot be read, or another target encoding, is an error.
pub fn encode_audio_packet<E: Encoder>(
    encoding_format: EncodingFlag,
    encoder: &mut E,
    fullbuf: &[u8],
) -> Result<BytesMut, String> {
    let header = FrameHeader::decode(&mut &fullbuf[..])
        .map_err(|error| format!("Failed to decode header: {error}"))?;
    let buf = fullbuf
        .get(header.size()..)
        .ok_or_else(|| "Packet is shorter than its header".to_string())?;
    // Room for an incompressible payload plus codec framing.
    let output_capacity = buf.len() * 2 + 4_096;
    let mut bits_per_sample = header.bits_per_sample();

    let data = match encoding_format {
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
                bits => return Err(format!("Unsupported bits per sample: {bits}")),
            };
            let mut data = vec![0u8; output_capacity];
            let num_bytes = encoder
                .encode_i32(&src, &mut data)
                .map_err(|e| format!("Failed to encode chunk {:?}", e))?;
            if num_bytes == 0 {
                return Err("Flac encoding: zero bytes".to_string());
            }
            data.truncate(num_bytes);
            data
        }
        EncodingFlag::Opus | EncodingFlag::AAC => {
            let src: Vec<i16> = match header.bits_per_sample() {
                16 => buf
                    .chunks_exact(2)
                    .map(|bytes| i16::from_le_bytes([bytes[0], bytes[1]]))
                    .collect(),
                24 => buf
                    .chunks_exact(3)
                    .map(|bytes| (LE::read_i24(bytes) >> 8) as i16)
                    .collect(),
                32 => buf
                    .chunks_exact(4)
                    .map(|bytes| {
                        let sample = if header.encoding() == &EncodingFlag::PCMSigned {
                            // Scale the 32-bit signed integer to the i16 range.
                            let s32_sample =
                                i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                            (s32_sample as i64 * i16::MAX as i64 / i32::MAX as i64) as i32
                        } else {
                            // Scale the 32-bit float to the i16 range.
                            let float_sample =
                                f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                            (float_sample * 32767.0) as i32
                        };
                        sample.clamp(i16::MIN as i32, i16::MAX as i32) as i16
                    })
                    .collect(),
                bits => return Err(format!("Unsupported bits per sample: {bits}")),
            };
            let mut data = vec![0u8; output_capacity];
            let num_bytes = encoder
                .encode_i16(&src, &mut data)
                .map_err(|e| format!("Opus/AAC encoding failed: {}", e))?;
            if num_bytes == 0 {
                return Err("Opus/AAC encoding: zero bytes".to_string());
            }
            data.truncate(num_bytes);
            data
        }
        EncodingFlag::PCMFloat => {
            bits_per_sample = 32;
            crate::audio_pipeline::f32s_to_le_bytes(&payload_to_f32(&header, buf)?)
        }
        EncodingFlag::PCMSigned => buf.to_vec(),
        other => return Err(format!("Unsupported target encoding: {other:?}")),
    };

    let header = FrameHeader::new(
        encoding_format,
        header.sample_size(),
        header.sample_rate(),
        header.channels(),
        bits_per_sample,
        Endianness::LittleEndian,
        header.id(),
        None,
    )?;
    let mut chunk = BytesMut::with_capacity(header.size() + data.len());
    let mut header_bytes = Vec::with_capacity(header.size());
    header
        .encode(&mut header_bytes)
        .map_err(|error| format!("Failed to encode header: {error}"))?;
    chunk.extend_from_slice(&header_bytes);
    chunk.extend_from_slice(&data);
    Ok(chunk)
}

/// Decodes the interleaved samples of a v1 packet as f32: PCM directly,
/// Opus through `decode_i16`, FLAC through `decode_i32`.
fn decode_packet_samples<D: Decoder>(
    header: &FrameHeader,
    data: &[u8],
    decoder: &mut D,
) -> Result<Vec<f32>, String> {
    let channel_count = header.channels() as usize;
    let capacity = header.sample_size() as usize * channel_count;
    match header.encoding() {
        EncodingFlag::PCMSigned => match header.bits_per_sample() {
            16 => Ok(data
                .chunks_exact(2)
                .map(|bytes| {
                    f32::from(i16::from_le_bytes([bytes[0], bytes[1]])) / f32::from(i16::MAX)
                })
                .collect()),
            24 | 32 => payload_to_f32(header, data),
            bits => Err(format!("Unsupported bits per sample: {bits}")),
        },
        EncodingFlag::PCMFloat => payload_to_f32(header, data),
        EncodingFlag::Opus => {
            let mut dst = vec![0i16; capacity];
            let decoded = decoder
                .decode_i16(data, &mut dst, false)
                .map_err(|e| format!("Opus decoding failed: {}", e))?;
            Ok(dst[..decoded.min(dst.len())]
                .iter()
                .map(|&sample| f32::from(sample) / f32::from(i16::MAX))
                .collect())
        }
        EncodingFlag::FLAC => {
            let mut dst = vec![0i32; capacity];
            let decoded = decoder
                .decode_i32(data, &mut dst, false)
                .map_err(|e| format!("FLAC decoding failed: {}", e))?;
            let scale = (1u64 << (header.bits_per_sample().clamp(2, 32) - 1)) as f32;
            Ok(dst[..decoded.min(dst.len())]
                .iter()
                .map(|&sample| sample as f32 / scale)
                .collect())
        }
        other => Err(format!("Unsupported encoding type: {other:?}")),
    }
}

/// Decodes a v1 packet into `scratch`, interleaved, and returns its header.
/// A packet that does not fit `scratch` is an error.
pub fn decode_audio_packet_scratch<D: Decoder>(
    buffer: Bytes,
    decoder: &mut D,
    scratch: &mut [f32],
) -> Result<FrameHeader, String> {
    let header = FrameHeader::decode(&mut &buffer[..])
        .map_err(|e| format!("Failed to decode header: {}", e))?;
    let data = buffer
        .get(header.size()..)
        .ok_or_else(|| "Packet is shorter than its header".to_string())?;
    let samples = decode_packet_samples(&header, data, decoder)?;
    let available = scratch.len();
    let target = scratch.get_mut(..samples.len()).ok_or_else(|| {
        format!(
            "Scratch holds {available} samples; the packet decodes to {}",
            samples.len()
        )
    })?;
    target.copy_from_slice(&samples);
    Ok(header)
}

/// Decodes a v1 packet into one f32 vector per channel. `None` when the
/// header cannot be read or the payload cannot be decoded.
pub fn decode_audio_packet<D: Decoder>(buffer: Vec<u8>, decoder: &mut D) -> Option<AudioList> {
    let header = FrameHeader::decode(&mut buffer.as_slice()).ok()?;
    let channel_count = header.channels() as usize;
    let data = buffer.get(header.size()..)?;
    let samples = decode_packet_samples(&header, data, decoder).ok()?;

    let mut deinterleaved_samples =
        vec![Vec::with_capacity(samples.len() / channel_count); channel_count];
    for (i, sample) in samples.iter().enumerate() {
        deinterleaved_samples[i % channel_count].push(*sample);
    }

    Some(AudioList {
        channels: deinterleaved_samples,
        sampling_rate: header.sample_rate() as usize,
        sample_count: header.sample_size() as usize,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A lossless stand-in codec: the samples as little-endian bytes.
    struct RawCodec;

    impl Encoder for RawCodec {
        fn new(_: u32, _: u32, _: u32, _: u32, _: u32) -> Self {
            RawCodec
        }
        fn init(&mut self) -> Result<(), String> {
            Ok(())
        }
        fn encode_i16(&mut self, input: &[i16], output: &mut [u8]) -> Result<usize, String> {
            for (target, sample) in output.chunks_exact_mut(2).zip(input) {
                target.copy_from_slice(&sample.to_le_bytes());
            }
            Ok(input.len() * 2)
        }
        fn encode_i32(&mut self, input: &[i32], output: &mut [u8]) -> Result<usize, String> {
            for (target, sample) in output.chunks_exact_mut(4).zip(input) {
                target.copy_from_slice(&sample.to_le_bytes());
            }
            Ok(input.len() * 4)
        }
        fn reset(&mut self) -> Result<(), String> {
            Ok(())
        }
    }

    impl Decoder for RawCodec {
        fn decode_i16(
            &mut self,
            input: &[u8],
            output: &mut [i16],
            _: bool,
        ) -> Result<usize, String> {
            let count = (input.len() / 2).min(output.len());
            for (target, bytes) in output.iter_mut().zip(input.chunks_exact(2)).take(count) {
                *target = i16::from_le_bytes([bytes[0], bytes[1]]);
            }
            Ok(count)
        }
        fn decode_i32(
            &mut self,
            input: &[u8],
            output: &mut [i32],
            _: bool,
        ) -> Result<usize, String> {
            let count = (input.len() / 4).min(output.len());
            for (target, bytes) in output.iter_mut().zip(input.chunks_exact(4)).take(count) {
                *target = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            }
            Ok(count)
        }
        fn decode_f32(&mut self, _: &[u8], _: &mut [f32], _: bool) -> Result<usize, String> {
            Err("unused".into())
        }
    }

    fn packet(encoding: EncodingFlag, bits: u8, frames: u16, payload: &[u8]) -> Vec<u8> {
        let header = FrameHeader::new(
            encoding,
            frames,
            48_000,
            2,
            bits,
            Endianness::LittleEndian,
            None,
            None,
        )
        .unwrap();
        let mut out = Vec::new();
        header.encode(&mut out).unwrap();
        out.extend_from_slice(payload);
        out
    }

    fn pcm16(frames: usize) -> (Vec<i16>, Vec<u8>) {
        let samples: Vec<i16> = (0..frames * 2)
            .map(|i| (i as i16).wrapping_mul(611))
            .collect();
        let bytes = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        (samples, bytes)
    }

    #[test]
    fn packets_round_trip_through_each_target() {
        let (samples, bytes) = pcm16(240);
        let input = packet(EncodingFlag::PCMSigned, 16, 240, &bytes);

        for (target, scale) in [
            (EncodingFlag::Opus, f32::from(i16::MAX)),
            (EncodingFlag::FLAC, 32_768.0),
            (EncodingFlag::PCMSigned, f32::from(i16::MAX)),
        ] {
            let encoded = encode_audio_packet(target, &mut RawCodec, &input).unwrap();
            let decoded = decode_audio_packet(encoded.to_vec(), &mut RawCodec).unwrap();
            let interleaved: Vec<f32> = (0..240)
                .flat_map(|frame| [decoded.channels[0][frame], decoded.channels[1][frame]])
                .collect();
            let expected: Vec<f32> = samples.iter().map(|&s| f32::from(s) / scale).collect();
            assert_eq!(
                decoded.channels[0].len(),
                240,
                "{target:?}: no leading zeros"
            );
            assert_eq!(interleaved, expected, "{target:?}");
        }

        let encoded = encode_audio_packet(EncodingFlag::PCMFloat, &mut RawCodec, &input).unwrap();
        let header = FrameHeader::decode(&mut &encoded[..]).unwrap();
        assert_eq!(header.bits_per_sample(), 32);
        assert_eq!(encoded.len(), header.size() + samples.len() * 4);
        let mut scratch = vec![0.0f32; 480];
        decode_audio_packet_scratch(encoded.freeze(), &mut RawCodec, &mut scratch).unwrap();
        let expected: Vec<f32> = samples.iter().map(|&s| f32::from(s) / 32_768.0).collect();
        assert_eq!(scratch, expected);
    }

    #[test]
    fn malformed_packets_are_errors_not_panics() {
        let (_, bytes) = pcm16(10);
        let input = packet(EncodingFlag::PCMSigned, 16, 10, &bytes);
        assert!(encode_audio_packet(EncodingFlag::FLAC, &mut RawCodec, &input[..3]).is_err());
        assert!(encode_audio_packet(EncodingFlag::H264, &mut RawCodec, &input).is_err());
        assert!(decode_audio_packet(input[..3].to_vec(), &mut RawCodec).is_none());
        let mut small = vec![0.0f32; 5];
        assert!(
            decode_audio_packet_scratch(Bytes::from(input.clone()), &mut RawCodec, &mut small)
                .is_err()
        );
        assert!(decode_audio_packet_scratch(
            Bytes::from_static(&[1, 2]),
            &mut RawCodec,
            &mut small
        )
        .is_err());
    }
}
