use crate::audio_types::{
    get_config, get_sampling_rate_and_bits_per_sample, AudioConfig, EncodingFlag,
};
use byteorder::{BigEndian, ByteOrder, LittleEndian};

pub trait Encoder {
    fn encode(&mut self, input: &[i16], output: &mut [u8]) -> Result<usize, String>;
    fn reset(&mut self);
}

pub trait Decoder {
    fn decode(&mut self, input: &[u8], output: &mut [i16], fec: bool) -> Result<usize, String>;
}

pub const HEADER_SIZE: usize = 5;

pub struct AudioList {
    pub channels: Vec<Vec<f32>>,
    pub sample_count: usize,
    pub sampling_rate: usize,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct AudioPacketHeader {
    encoding: EncodingFlag,
    config: AudioConfig,
    channel_count: u8,
    frame_size: u16,
}

impl AudioPacketHeader {
    pub fn bytes_per_sample(&self) -> u8 {
        match self.config {
            AudioConfig::Hz44100Bit16Le
            | AudioConfig::Hz44100Bit16Be
            | AudioConfig::Hz48000Bit16Le
            | AudioConfig::Hz48000Bit16Be
            | AudioConfig::Hz96000Bit16Le
            | AudioConfig::Hz96000Bit16Be => 2,

            AudioConfig::Hz44100Bit24Le
            | AudioConfig::Hz44100Bit24Be
            | AudioConfig::Hz48000Bit24Le
            | AudioConfig::Hz48000Bit24Be
            | AudioConfig::Hz96000Bit24Le
            | AudioConfig::Hz96000Bit24Be => 3,

            AudioConfig::Hz44100Bit32Le
            | AudioConfig::Hz44100Bit32Be
            | AudioConfig::Hz48000Bit32Le
            | AudioConfig::Hz48000Bit32Be
            | AudioConfig::Hz96000Bit32Le
            | AudioConfig::Hz96000Bit32Be => 4,

            AudioConfig::Hz44100Bit32FLe
            | AudioConfig::Hz44100Bit32FBe
            | AudioConfig::Hz48000Bit32FLe
            | AudioConfig::Hz48000Bit32FBe
            | AudioConfig::Hz96000Bit32FLe
            | AudioConfig::Hz96000Bit32FBe => 4,

            _ => 0,
        }
    }
}

pub fn encode_audio_packet<E: Encoder>(
    config: AudioConfig,
    buf: &Vec<u8>,
    channel_count: usize,
    format: &EncodingFlag,
    encoder: &mut E,
) -> Result<Vec<u8>, String> {
    let (_, bits_per_sample, audio_format) =
        get_sampling_rate_and_bits_per_sample(config.clone()).unwrap();
    let bytes_per_sample = bits_per_sample / 8;
    let frame_size = buf.len() as u32 / bytes_per_sample as u32;
    let len = frame_size as usize * bytes_per_sample as usize;

    let mut data = vec![0u8; len];

    match format {
        EncodingFlag::Opus => {
            let mut src: Vec<i16> = Vec::new();

            match bits_per_sample {
                16 => {
                    for bytes in buf.chunks_exact(2) {
                        let sample = if audio_format == 1 {
                            i16::from_le_bytes([bytes[0], bytes[1]])
                        } else {
                            i16::from_be_bytes([bytes[0], bytes[1]])
                        };
                        src.push(sample);
                    }
                }
                24 => {
                    for bytes in buf.chunks_exact(3) {
                        let sample = if audio_format == 1 {
                            (LittleEndian::read_i24(&bytes) >> 8) as i16
                        } else {
                            (BigEndian::read_i24(&bytes) >> 8) as i16
                        };
                        src.push(sample);
                    }
                }
                32 => {
                    for bytes in buf.chunks_exact(4) {
                        let sample = if audio_format == 1 {
                            f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
                        } else {
                            f32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
                        };
                        let scaled_sample = (sample * 32767.0) as i16;
                        src.push(scaled_sample);
                    }
                }
                _ => return Err(format!("Unsupported bits per sample: {}", bits_per_sample)),
            }

            let num_bytes = encoder.encode(&src[..], &mut data[..]).unwrap_or(0);
            if num_bytes == 0 {
                return Err("Opus encoding: zero bytes".to_string());
            }
            data.truncate(num_bytes);
        }
        EncodingFlag::PCM => {
            data.clone_from_slice(&buf[..]);
        }
    }

    let mut chunk = Vec::new();
    let header = encode_audio_packet_header(
        format,
        config.clone(),
        channel_count as u8,
        data.len() as u16,
    );
    chunk.extend_from_slice(&header);
    chunk.extend_from_slice(&data);

    Ok(chunk)
}

pub fn decode_audio_packet<D: Decoder>(buffer: Vec<u8>, decoder: &mut D) -> Option<AudioList> {
    let header = decode_audio_packet_header(&buffer);
    let channel_count = header.channel_count as usize;
    let bytes_per_sample = header.bytes_per_sample(); // Helper function
    let len = buffer.len() - HEADER_SIZE;
    let sample_count = len / (channel_count * bytes_per_sample as usize);
    let data = &buffer[HEADER_SIZE..];

    let mut samples = vec![0.0f32; sample_count * channel_count];

    match header.encoding {
        EncodingFlag::PCM => match header.config {
            AudioConfig::Hz44100Bit16Le | AudioConfig::Hz48000Bit16Le => {
                for sample_bytes in data.chunks_exact(2) {
                    let sample_i16 = i16::from_le_bytes([sample_bytes[0], sample_bytes[1]]);
                    let sample_f32 = f32::from(sample_i16) / f32::from(std::i16::MAX);
                    samples.push(sample_f32);
                }
            }
            AudioConfig::Hz44100Bit16Be | AudioConfig::Hz48000Bit16Be => {
                for sample_bytes in data.chunks_exact(2) {
                    let sample_i16 = i16::from_be_bytes([sample_bytes[0], sample_bytes[1]]);
                    let sample_f32 = f32::from(sample_i16) / f32::from(std::i16::MAX);
                    samples.push(sample_f32);
                }
            }
            AudioConfig::Hz44100Bit24Le | AudioConfig::Hz48000Bit24Le => {
                for sample_bytes in data.chunks_exact(3) {
                    let sample_i24 = LittleEndian::read_i24(sample_bytes);
                    let sample_f32 = sample_i24 as f32 / (1 << 23) as f32;
                    samples.push(sample_f32);
                }
            }
            AudioConfig::Hz44100Bit24Be | AudioConfig::Hz48000Bit24Be => {
                for sample_bytes in data.chunks_exact(3) {
                    let sample_i24 = BigEndian::read_i24(sample_bytes);
                    let sample_f32 = sample_i24 as f32 / (1 << 23) as f32;
                    samples.push(sample_f32);
                }
            }
            AudioConfig::Hz44100Bit32Le | AudioConfig::Hz48000Bit32Le => {
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
            AudioConfig::Hz44100Bit32Be | AudioConfig::Hz48000Bit32Be => {
                for sample_bytes in data.chunks_exact(4) {
                    let sample_f32 = f32::from_be_bytes([
                        sample_bytes[0],
                        sample_bytes[1],
                        sample_bytes[2],
                        sample_bytes[3],
                    ]);
                    samples.push(sample_f32);
                }
            }
            AudioConfig::Hz44100Bit32FLe | AudioConfig::Hz48000Bit32FLe => {
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
            AudioConfig::Hz44100Bit32FBe | AudioConfig::Hz48000Bit32FBe => {
                for sample_bytes in data.chunks_exact(4) {
                    let sample_f32 = f32::from_be_bytes([
                        sample_bytes[0],
                        sample_bytes[1],
                        sample_bytes[2],
                        sample_bytes[3],
                    ]);
                    samples.push(sample_f32);
                }
            }
            _ => return None, // Handle all remaining configurations similarly.
        },
        EncodingFlag::Opus => {
            let mut dst = vec![0i16; sample_count * channel_count];
            let _num_samples_decoded = decoder.decode(&data[..], &mut dst[..], false).unwrap_or(0);

            for sample_i16 in dst {
                let sample_f32 = f32::from(sample_i16) / f32::from(std::i16::MAX);
                samples.push(sample_f32);
            }
        }
    }

    let mut deinterleaved_samples =
        vec![Vec::with_capacity(samples.len() / channel_count); channel_count];

    for (i, sample) in samples.iter().enumerate() {
        deinterleaved_samples[i % channel_count].push(*sample);
    }

    Some(AudioList {
        channels: deinterleaved_samples,
        sampling_rate: get_sampling_rate_and_bits_per_sample(header.config)
            .unwrap()
            .0 as usize,
        sample_count,
    })
}

pub fn decode_audio_packet_header(data: &[u8]) -> AudioPacketHeader {
    let encoding = if (data[0] & 0x40) == 0 {
        EncodingFlag::PCM
    } else {
        EncodingFlag::Opus
    };
    let config_id = data[0] & 0x3F; // Use 6 bits for the config ID
    let config = get_config(config_id).unwrap();
    let channel_count = data[1];
    let frame_size = u16::from_le_bytes([data[2], data[3]]);

    AudioPacketHeader {
        encoding,
        config,
        channel_count,
        frame_size,
    }
}

pub fn encode_audio_packet_header(
    encoding: &EncodingFlag,
    config: AudioConfig,
    channel_count: u8,
    frame_size: u16,
) -> Vec<u8> {
    let mut flag = 0;
    if encoding == &EncodingFlag::Opus {
        flag = 1;
    }

    let mut id = config as u8;
    id |= flag << 6; // Use 6 bits for configuration ID and 1 bit for encoding

    let mut header = vec![id];
    header.push(channel_count);
    header.extend_from_slice(&frame_size.to_le_bytes());

    header
}
