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
    audio_format: u16,
    endianness: Endianness,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Endianness {
    LE,
    BE,
}

impl AudioData {
    pub fn new(
        bits_per_sample: u8,
        channel_count: u8,
        sampling_rate: u32,
        data: Vec<u8>,
        audio_format: u16,
        endianness: Endianness,
    ) -> Self {
        AudioData {
            bits_per_sample,
            channel_count,
            sampling_rate,
            data,
            audio_format,
            endianness,
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

    pub fn audio_format(&self) -> u16 {
        self.audio_format
    }

    pub fn endianness(&self) -> Endianness {
        self.endianness
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum EncodingFlag {
    PCM = 0,
    Opus = 1,
    FLAC = 2,
    AAC = 3,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum AudioConfig {
    Hz44100Bit16,
    Hz44100Bit24,
    Hz44100Bit32,
    Hz44100Bit32F,
    Hz48000Bit16,
    Hz48000Bit24,
    Hz48000Bit32,
    Hz48000Bit32F,
}

pub fn get_sampling_rate_and_bits_per_sample(config: AudioConfig) -> Option<(u32, u8, u16)> {
    match config {
        AudioConfig::Hz44100Bit16 => Some((44100, 16, 1)),
        AudioConfig::Hz44100Bit24 => Some((44100, 24, 1)),
        AudioConfig::Hz44100Bit32 => Some((44100, 32, 1)),
        AudioConfig::Hz44100Bit32F => Some((44100, 32, 3)),
        AudioConfig::Hz48000Bit16 => Some((48000, 16, 1)),
        AudioConfig::Hz48000Bit24 => Some((48000, 24, 1)),
        AudioConfig::Hz48000Bit32 => Some((48000, 32, 1)),
        AudioConfig::Hz48000Bit32F => Some((48000, 32, 3)),
    }
}

pub fn get_config(id: u8) -> Option<AudioConfig> {
    match id {
        0 => Some(AudioConfig::Hz44100Bit16),
        1 => Some(AudioConfig::Hz44100Bit24),
        2 => Some(AudioConfig::Hz44100Bit32),
        3 => Some(AudioConfig::Hz44100Bit32F),
        4 => Some(AudioConfig::Hz48000Bit16),
        5 => Some(AudioConfig::Hz48000Bit24),
        6 => Some(AudioConfig::Hz48000Bit32),
        7 => Some(AudioConfig::Hz48000Bit32F),
        _ => None,
    }
}

pub fn get_audio_config(
    sampling_rate: u32,
    bits_per_sample: u8,
    format_flag: Option<u16>,
) -> Option<AudioConfig> {
    let format_flag = match format_flag {
        Some(flag) if flag < 3 => None,
        _ => format_flag,
    };

    match (sampling_rate, bits_per_sample, format_flag) {
        (44100, 16, None) => Some(AudioConfig::Hz44100Bit16),
        (44100, 24, None) => Some(AudioConfig::Hz44100Bit24),
        (44100, 32, None) => Some(AudioConfig::Hz44100Bit32),
        (44100, 32, Some(3)) => Some(AudioConfig::Hz44100Bit32F),
        (48000, 16, None) => Some(AudioConfig::Hz48000Bit16),
        (48000, 24, None) => Some(AudioConfig::Hz48000Bit24),
        (48000, 32, None) => Some(AudioConfig::Hz48000Bit32),
        (48000, 32, Some(3)) => Some(AudioConfig::Hz48000Bit32F),
        _ => None,
    }
}
