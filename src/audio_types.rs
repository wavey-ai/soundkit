#![allow(dead_code)]

pub enum PcmData {
    I16(Vec<Vec<i16>>),
    I32(Vec<Vec<i32>>),
    F32(Vec<Vec<f32>>),
}

#[derive(Debug)]
pub struct AudioData {
    bits_per_sample: u8,
    channel_count: u8,
    data: Vec<u8>,
    sampling_rate: u32,
}

impl AudioData {
    pub fn new(bits_per_sample: u8, channel_count: u8, sampling_rate: u32, data: Vec<u8>) -> Self {
        AudioData {
            bits_per_sample,
            channel_count,
            sampling_rate,
            data,
        }
    }

    pub fn bits_per_sample(&self) -> u8 {
        self.bits_per_sample
    }

    pub fn channel_count(&self) -> u8 {
        self.channel_count
    }

    pub fn sampling_rate(&self) -> u32 {
        self.sampling_rate
    }

    pub fn data(&self) -> &Vec<u8> {
        &self.data
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum EncodingFlag {
    PCM = 0,
    Opus = 1,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum AudioConfig {
    Hz44100Bit16,
    Hz44100Bit24,
    Hz44100Bit32,
    Hz48000Bit16,
    Hz48000Bit24,
    Hz48000Bit32,
    Hz88200Bit16,
    Hz88200Bit24,
    Hz88200Bit32,
    Hz96000Bit16,
    Hz96000Bit24,
    Hz96000Bit32,
    Hz176400Bit16,
    Hz176400Bit24,
    Hz176400Bit32,
    Hz192000Bit16,
    Hz192000Bit24,
    Hz192000Bit32,
    Hz352800Bit16,
    Hz352800Bit24,
    Hz352800Bit32,
}

pub fn get_sampling_rate_and_bits_per_sample(config: AudioConfig) -> Option<(u32, u8)> {
    match config {
        AudioConfig::Hz44100Bit16 => Some((44100, 16)),
        AudioConfig::Hz44100Bit24 => Some((44100, 24)),
        AudioConfig::Hz44100Bit32 => Some((44100, 32)),
        AudioConfig::Hz48000Bit16 => Some((48000, 16)),
        AudioConfig::Hz48000Bit24 => Some((48000, 24)),
        AudioConfig::Hz48000Bit32 => Some((48000, 32)),
        AudioConfig::Hz88200Bit16 => Some((88200, 16)),
        AudioConfig::Hz88200Bit24 => Some((88200, 24)),
        AudioConfig::Hz88200Bit32 => Some((88200, 32)),
        AudioConfig::Hz96000Bit16 => Some((96000, 16)),
        AudioConfig::Hz96000Bit24 => Some((96000, 24)),
        AudioConfig::Hz96000Bit32 => Some((96000, 32)),
        AudioConfig::Hz176400Bit16 => Some((176400, 16)),
        AudioConfig::Hz176400Bit24 => Some((176400, 24)),
        AudioConfig::Hz176400Bit32 => Some((176400, 32)),
        AudioConfig::Hz192000Bit16 => Some((192000, 16)),
        AudioConfig::Hz192000Bit24 => Some((192000, 24)),
        AudioConfig::Hz192000Bit32 => Some((192000, 32)),
        AudioConfig::Hz352800Bit16 => Some((352800, 16)),
        AudioConfig::Hz352800Bit24 => Some((352800, 24)),
        AudioConfig::Hz352800Bit32 => Some((352800, 32)),
    }
}

pub fn get_config(id: u8) -> Option<AudioConfig> {
    match id {
        0 => Some(AudioConfig::Hz44100Bit16),
        1 => Some(AudioConfig::Hz44100Bit24),
        2 => Some(AudioConfig::Hz44100Bit32),
        3 => Some(AudioConfig::Hz48000Bit16),
        4 => Some(AudioConfig::Hz48000Bit24),
        5 => Some(AudioConfig::Hz48000Bit32),
        6 => Some(AudioConfig::Hz88200Bit16),
        7 => Some(AudioConfig::Hz88200Bit24),
        8 => Some(AudioConfig::Hz88200Bit32),
        9 => Some(AudioConfig::Hz96000Bit16),
        10 => Some(AudioConfig::Hz96000Bit24),
        11 => Some(AudioConfig::Hz96000Bit32),
        12 => Some(AudioConfig::Hz176400Bit16),
        13 => Some(AudioConfig::Hz176400Bit24),
        14 => Some(AudioConfig::Hz176400Bit32),
        15 => Some(AudioConfig::Hz192000Bit16),
        16 => Some(AudioConfig::Hz192000Bit24),
        17 => Some(AudioConfig::Hz192000Bit32),
        18 => Some(AudioConfig::Hz352800Bit16),
        19 => Some(AudioConfig::Hz352800Bit24),
        20 => Some(AudioConfig::Hz352800Bit32),
        _ => None,
    }
}

pub fn get_audio_config(sampling_rate: u32, bits_per_sample: u8) -> Option<AudioConfig> {
    match (sampling_rate, bits_per_sample) {
        (44100, 16) => Some(AudioConfig::Hz44100Bit16),
        (44100, 24) => Some(AudioConfig::Hz44100Bit24),
        (44100, 32) => Some(AudioConfig::Hz44100Bit32),
        (48000, 16) => Some(AudioConfig::Hz48000Bit16),
        (48000, 24) => Some(AudioConfig::Hz48000Bit24),
        (48000, 32) => Some(AudioConfig::Hz48000Bit32),
        (88200, 16) => Some(AudioConfig::Hz88200Bit16),
        (88200, 24) => Some(AudioConfig::Hz88200Bit24),
        (88200, 32) => Some(AudioConfig::Hz88200Bit32),
        (96000, 16) => Some(AudioConfig::Hz96000Bit16),
        (96000, 24) => Some(AudioConfig::Hz96000Bit24),
        (96000, 32) => Some(AudioConfig::Hz96000Bit32),
        (176400, 16) => Some(AudioConfig::Hz176400Bit16),
        (176400, 24) => Some(AudioConfig::Hz176400Bit24),
        (176400, 32) => Some(AudioConfig::Hz176400Bit32),
        (192000, 16) => Some(AudioConfig::Hz192000Bit16),
        (192000, 24) => Some(AudioConfig::Hz192000Bit24),
        (192000, 32) => Some(AudioConfig::Hz192000Bit32),
        (352800, 16) => Some(AudioConfig::Hz352800Bit16),
        (352800, 24) => Some(AudioConfig::Hz352800Bit24),
        (352800, 32) => Some(AudioConfig::Hz352800Bit32),
        _ => None,
    }
}
