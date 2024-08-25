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
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum AudioConfig {
    Hz44100Bit16Le,
    Hz44100Bit16Be,
    Hz44100Bit24Le,
    Hz44100Bit24Be,
    Hz44100Bit32Le,
    Hz44100Bit32Be,
    Hz44100Bit32FLe,
    Hz44100Bit32FBe,
    Hz48000Bit16Le,
    Hz48000Bit16Be,
    Hz48000Bit24Le,
    Hz48000Bit24Be,
    Hz48000Bit32Le,
    Hz48000Bit32Be,
    Hz48000Bit32FLe,
    Hz48000Bit32FBe,
    Hz88200Bit16Le,
    Hz88200Bit16Be,
    Hz88200Bit24Le,
    Hz88200Bit24Be,
    Hz88200Bit32Le,
    Hz88200Bit32Be,
    Hz96000Bit16Le,
    Hz96000Bit16Be,
    Hz96000Bit24Le,
    Hz96000Bit24Be,
    Hz96000Bit32Le,
    Hz96000Bit32Be,
    Hz96000Bit32FLe,
    Hz96000Bit32FBe,
    Hz176400Bit16Le,
    Hz176400Bit16Be,
    Hz176400Bit24Le,
    Hz176400Bit24Be,
    Hz176400Bit32Le,
    Hz176400Bit32Be,
    Hz176400Bit32FLe,
    Hz176400Bit32FBe,
    Hz192000Bit16Le,
    Hz192000Bit16Be,
    Hz192000Bit24Le,
    Hz192000Bit24Be,
    Hz192000Bit32Le,
    Hz192000Bit32Be,
    Hz192000Bit32FLe,
    Hz192000Bit32FBe,
    Hz352800Bit16Le,
    Hz352800Bit16Be,
    Hz352800Bit24Le,
    Hz352800Bit24Be,
    Hz352800Bit32Le,
    Hz352800Bit32Be,
    Hz352800Bit32FLe,
    Hz352800Bit32FBe,
}

pub fn get_sampling_rate_and_bits_per_sample(config: AudioConfig) -> Option<(u32, u8, u16)> {
    match config {
        AudioConfig::Hz44100Bit16Le | AudioConfig::Hz44100Bit16Be => Some((44100, 16, 1)),
        AudioConfig::Hz44100Bit24Le | AudioConfig::Hz44100Bit24Be => Some((44100, 24, 1)),
        AudioConfig::Hz44100Bit32Le | AudioConfig::Hz44100Bit32Be => Some((44100, 32, 1)),
        AudioConfig::Hz44100Bit32FLe | AudioConfig::Hz44100Bit32FBe => Some((44100, 32, 3)),
        AudioConfig::Hz48000Bit16Le | AudioConfig::Hz48000Bit16Be => Some((48000, 16, 1)),
        AudioConfig::Hz48000Bit24Le | AudioConfig::Hz48000Bit24Be => Some((48000, 24, 1)),
        AudioConfig::Hz48000Bit32Le | AudioConfig::Hz48000Bit32Be => Some((48000, 32, 1)),
        AudioConfig::Hz48000Bit32FLe | AudioConfig::Hz48000Bit32FBe => Some((48000, 32, 3)),
        AudioConfig::Hz88200Bit16Le | AudioConfig::Hz88200Bit16Be => Some((88200, 16, 1)),
        AudioConfig::Hz88200Bit24Le | AudioConfig::Hz88200Bit24Be => Some((88200, 24, 1)),
        AudioConfig::Hz88200Bit32Le | AudioConfig::Hz88200Bit32Be => Some((88200, 32, 1)),
        AudioConfig::Hz96000Bit16Le | AudioConfig::Hz96000Bit16Be => Some((96000, 16, 1)),
        AudioConfig::Hz96000Bit24Le | AudioConfig::Hz96000Bit24Be => Some((96000, 24, 1)),
        AudioConfig::Hz96000Bit32Le | AudioConfig::Hz96000Bit32Be => Some((96000, 32, 1)),
        AudioConfig::Hz96000Bit32FLe | AudioConfig::Hz96000Bit32FBe => Some((96000, 32, 3)),
        AudioConfig::Hz176400Bit16Le | AudioConfig::Hz176400Bit16Be => Some((176400, 16, 1)),
        AudioConfig::Hz176400Bit24Le | AudioConfig::Hz176400Bit24Be => Some((176400, 24, 1)),
        AudioConfig::Hz176400Bit32Le | AudioConfig::Hz176400Bit32Be => Some((176400, 32, 1)),
        AudioConfig::Hz176400Bit32FLe | AudioConfig::Hz176400Bit32FBe => Some((176400, 32, 3)),
        AudioConfig::Hz192000Bit16Le | AudioConfig::Hz192000Bit16Be => Some((192000, 16, 1)),
        AudioConfig::Hz192000Bit24Le | AudioConfig::Hz192000Bit24Be => Some((192000, 24, 1)),
        AudioConfig::Hz192000Bit32Le | AudioConfig::Hz192000Bit32Be => Some((192000, 32, 1)),
        AudioConfig::Hz192000Bit32FLe | AudioConfig::Hz192000Bit32FBe => Some((192000, 32, 3)),
        AudioConfig::Hz352800Bit16Le | AudioConfig::Hz352800Bit16Be => Some((352800, 16, 1)),
        AudioConfig::Hz352800Bit24Le | AudioConfig::Hz352800Bit24Be => Some((352800, 24, 1)),
        AudioConfig::Hz352800Bit32Le | AudioConfig::Hz352800Bit32Be => Some((352800, 32, 1)),
        AudioConfig::Hz352800Bit32FLe | AudioConfig::Hz352800Bit32FBe => Some((352800, 32, 3)),
    }
}

pub fn get_config(id: u8) -> Option<AudioConfig> {
    match id {
        0 => Some(AudioConfig::Hz44100Bit16Le),
        1 => Some(AudioConfig::Hz44100Bit16Be),
        2 => Some(AudioConfig::Hz44100Bit24Le),
        3 => Some(AudioConfig::Hz44100Bit24Be),
        4 => Some(AudioConfig::Hz44100Bit32Le),
        5 => Some(AudioConfig::Hz44100Bit32Be),
        6 => Some(AudioConfig::Hz44100Bit32FLe),
        7 => Some(AudioConfig::Hz44100Bit32FBe),
        8 => Some(AudioConfig::Hz48000Bit16Le),
        9 => Some(AudioConfig::Hz48000Bit16Be),
        10 => Some(AudioConfig::Hz48000Bit24Le),
        11 => Some(AudioConfig::Hz48000Bit24Be),
        12 => Some(AudioConfig::Hz48000Bit32Le),
        13 => Some(AudioConfig::Hz48000Bit32Be),
        14 => Some(AudioConfig::Hz48000Bit32FLe),
        15 => Some(AudioConfig::Hz48000Bit32FBe),
        16 => Some(AudioConfig::Hz88200Bit16Le),
        17 => Some(AudioConfig::Hz88200Bit16Be),
        18 => Some(AudioConfig::Hz88200Bit24Le),
        19 => Some(AudioConfig::Hz88200Bit24Be),
        20 => Some(AudioConfig::Hz88200Bit32Le),
        21 => Some(AudioConfig::Hz88200Bit32Be),
        22 => Some(AudioConfig::Hz96000Bit16Le),
        23 => Some(AudioConfig::Hz96000Bit16Be),
        24 => Some(AudioConfig::Hz96000Bit24Le),
        25 => Some(AudioConfig::Hz96000Bit24Be),
        26 => Some(AudioConfig::Hz96000Bit32Le),
        27 => Some(AudioConfig::Hz96000Bit32Be),
        28 => Some(AudioConfig::Hz96000Bit32FLe),
        29 => Some(AudioConfig::Hz96000Bit32FBe),
        30 => Some(AudioConfig::Hz176400Bit16Le),
        31 => Some(AudioConfig::Hz176400Bit16Be),
        32 => Some(AudioConfig::Hz176400Bit24Le),
        33 => Some(AudioConfig::Hz176400Bit24Be),
        34 => Some(AudioConfig::Hz176400Bit32Le),
        35 => Some(AudioConfig::Hz176400Bit32Be),
        36 => Some(AudioConfig::Hz176400Bit32FLe),
        37 => Some(AudioConfig::Hz176400Bit32FBe),
        38 => Some(AudioConfig::Hz192000Bit16Le),
        39 => Some(AudioConfig::Hz192000Bit16Be),
        40 => Some(AudioConfig::Hz192000Bit24Le),
        41 => Some(AudioConfig::Hz192000Bit24Be),
        42 => Some(AudioConfig::Hz192000Bit32Le),
        43 => Some(AudioConfig::Hz192000Bit32Be),
        44 => Some(AudioConfig::Hz192000Bit32FLe),
        45 => Some(AudioConfig::Hz192000Bit32FBe),
        46 => Some(AudioConfig::Hz352800Bit16Le),
        47 => Some(AudioConfig::Hz352800Bit16Be),
        48 => Some(AudioConfig::Hz352800Bit24Le),
        49 => Some(AudioConfig::Hz352800Bit24Be),
        50 => Some(AudioConfig::Hz352800Bit32Le),
        51 => Some(AudioConfig::Hz352800Bit32Be),
        52 => Some(AudioConfig::Hz352800Bit32FLe),
        53 => Some(AudioConfig::Hz352800Bit32FBe),
        _ => None,
    }
}

pub fn get_audio_config(
    sampling_rate: u32,
    bits_per_sample: u8,
    format_flag: Option<u16>,
    endian_flag: Endianness,
) -> Option<AudioConfig> {
    let format_flag = match format_flag {
        Some(flag) if flag < 3 => None,
        _ => format_flag,
    };

    match (sampling_rate, bits_per_sample, format_flag, endian_flag) {
        (44100, 16, None, Endianness::LE) => Some(AudioConfig::Hz44100Bit16Le),
        (44100, 16, None, Endianness::BE) => Some(AudioConfig::Hz44100Bit16Be),
        (44100, 24, None, Endianness::LE) => Some(AudioConfig::Hz44100Bit24Le),
        (44100, 24, None, Endianness::BE) => Some(AudioConfig::Hz44100Bit24Be),
        (44100, 32, None, Endianness::LE) => Some(AudioConfig::Hz44100Bit32Le),
        (44100, 32, None, Endianness::BE) => Some(AudioConfig::Hz44100Bit32Be),
        (44100, 32, Some(3), Endianness::LE) => Some(AudioConfig::Hz44100Bit32FLe),
        (44100, 32, Some(3), Endianness::BE) => Some(AudioConfig::Hz44100Bit32FBe),
        (48000, 16, None, Endianness::LE) => Some(AudioConfig::Hz48000Bit16Le),
        (48000, 16, None, Endianness::BE) => Some(AudioConfig::Hz48000Bit16Be),
        (48000, 24, None, Endianness::LE) => Some(AudioConfig::Hz48000Bit24Le),
        (48000, 24, None, Endianness::BE) => Some(AudioConfig::Hz48000Bit24Be),
        (48000, 32, None, Endianness::LE) => Some(AudioConfig::Hz48000Bit32Le),
        (48000, 32, None, Endianness::BE) => Some(AudioConfig::Hz48000Bit32Be),
        (48000, 32, Some(3), Endianness::LE) => Some(AudioConfig::Hz48000Bit32FLe),
        (48000, 32, Some(3), Endianness::BE) => Some(AudioConfig::Hz48000Bit32FBe),
        (88200, 16, None, Endianness::LE) => Some(AudioConfig::Hz88200Bit16Le),
        (88200, 16, None, Endianness::BE) => Some(AudioConfig::Hz88200Bit16Be),
        (88200, 24, None, Endianness::LE) => Some(AudioConfig::Hz88200Bit24Le),
        (88200, 24, None, Endianness::BE) => Some(AudioConfig::Hz88200Bit24Be),
        (88200, 32, None, Endianness::LE) => Some(AudioConfig::Hz88200Bit32Le),
        (88200, 32, None, Endianness::BE) => Some(AudioConfig::Hz88200Bit32Be),
        (96000, 16, None, Endianness::LE) => Some(AudioConfig::Hz96000Bit16Le),
        (96000, 16, None, Endianness::BE) => Some(AudioConfig::Hz96000Bit16Be),
        (96000, 24, None, Endianness::LE) => Some(AudioConfig::Hz96000Bit24Le),
        (96000, 24, None, Endianness::BE) => Some(AudioConfig::Hz96000Bit24Be),
        (96000, 32, None, Endianness::LE) => Some(AudioConfig::Hz96000Bit32Le),
        (96000, 32, None, Endianness::BE) => Some(AudioConfig::Hz96000Bit32Be),
        (96000, 32, Some(3), Endianness::LE) => Some(AudioConfig::Hz96000Bit32FLe),
        (96000, 32, Some(3), Endianness::BE) => Some(AudioConfig::Hz96000Bit32FBe),
        (176400, 16, None, Endianness::LE) => Some(AudioConfig::Hz176400Bit16Le),
        (176400, 16, None, Endianness::BE) => Some(AudioConfig::Hz176400Bit16Be),
        (176400, 24, None, Endianness::LE) => Some(AudioConfig::Hz176400Bit24Le),
        (176400, 24, None, Endianness::BE) => Some(AudioConfig::Hz176400Bit24Be),
        (176400, 32, None, Endianness::LE) => Some(AudioConfig::Hz176400Bit32Le),
        (176400, 32, None, Endianness::BE) => Some(AudioConfig::Hz176400Bit32Be),
        (176400, 32, Some(3), Endianness::LE) => Some(AudioConfig::Hz176400Bit32FLe),
        (176400, 32, Some(3), Endianness::BE) => Some(AudioConfig::Hz176400Bit32FBe),
        (192000, 16, None, Endianness::LE) => Some(AudioConfig::Hz192000Bit16Le),
        (192000, 16, None, Endianness::BE) => Some(AudioConfig::Hz192000Bit16Be),
        (192000, 24, None, Endianness::LE) => Some(AudioConfig::Hz192000Bit24Le),
        (192000, 24, None, Endianness::BE) => Some(AudioConfig::Hz192000Bit24Be),
        (192000, 32, None, Endianness::LE) => Some(AudioConfig::Hz192000Bit32Le),
        (192000, 32, None, Endianness::BE) => Some(AudioConfig::Hz192000Bit32Be),
        (192000, 32, Some(3), Endianness::LE) => Some(AudioConfig::Hz192000Bit32FLe),
        (192000, 32, Some(3), Endianness::BE) => Some(AudioConfig::Hz192000Bit32FBe),
        (352800, 16, None, Endianness::LE) => Some(AudioConfig::Hz352800Bit16Le),
        (352800, 16, None, Endianness::BE) => Some(AudioConfig::Hz352800Bit16Be),
        (352800, 24, None, Endianness::LE) => Some(AudioConfig::Hz352800Bit24Le),
        (352800, 24, None, Endianness::BE) => Some(AudioConfig::Hz352800Bit24Be),
        (352800, 32, None, Endianness::LE) => Some(AudioConfig::Hz352800Bit32Le),
        (352800, 32, None, Endianness::BE) => Some(AudioConfig::Hz352800Bit32Be),
        (352800, 32, Some(3), Endianness::LE) => Some(AudioConfig::Hz352800Bit32FLe),
        (352800, 32, Some(3), Endianness::BE) => Some(AudioConfig::Hz352800Bit32FBe),
        _ => None,
    }
}
