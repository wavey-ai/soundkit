use crate::constants::{valid_channels, valid_sample_rate};
use crate::{Error, Result};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Application {
    Voip,
    Audio,
    RestrictedLowDelay,
}

impl Application {
    pub const fn code(self) -> i32 {
        match self {
            Self::Voip => 2048,
            Self::Audio => 2049,
            Self::RestrictedLowDelay => 2051,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Encoder {
    sample_rate: i32,
    channels: usize,
    application: Application,
}

impl Encoder {
    pub fn new(sample_rate: i32, channels: usize, application: Application) -> Result<Self> {
        if !valid_sample_rate(sample_rate) || !valid_channels(channels as i32) {
            return Err(Error::BadArg);
        }
        Ok(Self {
            sample_rate,
            channels,
            application,
        })
    }

    pub const fn sample_rate(&self) -> i32 {
        self.sample_rate
    }

    pub const fn channels(&self) -> usize {
        self.channels
    }

    pub const fn application(&self) -> Application {
        self.application
    }

    pub fn encode_i16(&mut self, _pcm: &[i16], _frame_size: usize) -> Result<Vec<u8>> {
        Err(Error::Unimplemented)
    }

    pub fn encode_f32(&mut self, _pcm: &[f32], _frame_size: usize) -> Result<Vec<u8>> {
        Err(Error::Unimplemented)
    }
}
