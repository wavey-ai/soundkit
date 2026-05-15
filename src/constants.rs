#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Bandwidth {
    Narrowband,
    Mediumband,
    Wideband,
    SuperWideband,
    Fullband,
}

impl Bandwidth {
    pub const fn opus_code(self) -> i32 {
        match self {
            Self::Narrowband => 1101,
            Self::Mediumband => 1102,
            Self::Wideband => 1103,
            Self::SuperWideband => 1104,
            Self::Fullband => 1105,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrameDuration {
    Argument,
    Ms2_5,
    Ms5,
    Ms10,
    Ms20,
    Ms40,
    Ms60,
    Ms80,
    Ms100,
    Ms120,
}

impl FrameDuration {
    pub const fn opus_code(self) -> i32 {
        match self {
            Self::Argument => 5000,
            Self::Ms2_5 => 5001,
            Self::Ms5 => 5002,
            Self::Ms10 => 5003,
            Self::Ms20 => 5004,
            Self::Ms40 => 5005,
            Self::Ms60 => 5006,
            Self::Ms80 => 5007,
            Self::Ms100 => 5008,
            Self::Ms120 => 5009,
        }
    }
}

pub fn valid_sample_rate(fs: i32) -> bool {
    matches!(fs, 8000 | 12000 | 16000 | 24000 | 48000)
}

pub fn valid_channels(channels: i32) -> bool {
    matches!(channels, 1 | 2)
}
