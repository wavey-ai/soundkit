use crate::constants::{valid_channels, valid_sample_rate};
use crate::packet;
use crate::{Error, Result};

#[derive(Clone, Debug)]
pub struct Decoder {
    sample_rate: i32,
    channels: usize,
}

impl Decoder {
    pub fn new(sample_rate: i32, channels: usize) -> Result<Self> {
        if !valid_sample_rate(sample_rate) || !valid_channels(channels as i32) {
            return Err(Error::BadArg);
        }
        Ok(Self {
            sample_rate,
            channels,
        })
    }

    pub const fn sample_rate(&self) -> i32 {
        self.sample_rate
    }

    pub const fn channels(&self) -> usize {
        self.channels
    }

    pub fn validate_packet(&self, packet: &[u8]) -> Result<usize> {
        let samples = packet::sample_count(packet, self.sample_rate)?;
        Ok(samples as usize)
    }

    pub fn decode_i16(&mut self, packet: &[u8], _decode_fec: bool) -> Result<Vec<i16>> {
        self.validate_packet(packet)?;
        Err(Error::Unimplemented)
    }

    pub fn decode_f32(&mut self, packet: &[u8], _decode_fec: bool) -> Result<Vec<f32>> {
        self.validate_packet(packet)?;
        Err(Error::Unimplemented)
    }
}
