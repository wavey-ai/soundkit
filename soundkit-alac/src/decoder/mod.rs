// Decoder bootstrap derived from alac 0.5.0.
// Copyright (c) 2016 Edward Barnard.
// Modified and maintained in-tree by the SoundKit authors.

mod bitcursor;
mod dec;

pub use dec::Decoder;

use std::error;
use std::fmt;

/// An error indicating that an ALAC cookie or packet is invalid or truncated.
#[derive(Debug)]
pub struct InvalidData {
    message: &'static str,
}

impl error::Error for InvalidData {}

impl fmt::Display for InvalidData {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.message)
    }
}

impl From<bitcursor::NotEnoughData> for InvalidData {
    fn from(_: bitcursor::NotEnoughData) -> Self {
        invalid_data("packet is not long enough")
    }
}

fn invalid_data(message: &'static str) -> InvalidData {
    InvalidData { message }
}

/// Codec initialization parameters carried by an ALAC magic cookie.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StreamInfo {
    frame_length: u32,
    compatible_version: u8,
    bit_depth: u8,
    pb: u8,
    mb: u8,
    kb: u8,
    num_channels: u8,
    max_run: u16,
    max_frame_bytes: u32,
    avg_bit_rate: u32,
    sample_rate: u32,
}

impl StreamInfo {
    /// Parses a bare ALACSpecificConfig or an older atom-wrapped cookie.
    pub fn from_cookie(mut cookie: &[u8]) -> Result<Self, InvalidData> {
        if cookie.len() < 24 {
            return Err(invalid_data("magic cookie is not the correct length"));
        }
        if cookie.len() >= 28 && cookie[..4] == [0, 0, 0, 0] {
            cookie = &cookie[4..];
        }
        if &cookie[4..8] == b"frma" {
            cookie = &cookie[12..];
        }
        if cookie.len() >= 8 && &cookie[4..8] == b"alac" {
            cookie = cookie
                .get(12..)
                .ok_or_else(|| invalid_data("magic cookie is not the correct length"))?;
        }
        if cookie.len() < 24 {
            return Err(invalid_data("magic cookie is not the correct length"));
        }

        Ok(Self {
            frame_length: read_be_u32(&cookie[0..4]),
            compatible_version: cookie[4],
            bit_depth: cookie[5],
            pb: cookie[6],
            mb: cookie[7],
            kb: cookie[8],
            num_channels: cookie[9],
            max_run: read_be_u16(&cookie[10..12]),
            max_frame_bytes: read_be_u32(&cookie[12..16]),
            avg_bit_rate: read_be_u32(&cookie[16..20]),
            sample_rate: read_be_u32(&cookie[20..24]),
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn bit_depth(&self) -> u8 {
        self.bit_depth
    }

    pub fn channels(&self) -> u8 {
        self.num_channels
    }

    pub fn max_frames_per_packet(&self) -> u32 {
        self.frame_length
    }

    pub fn max_samples_per_packet(&self) -> u32 {
        self.frame_length * u32::from(self.num_channels)
    }
}

fn read_be_u16(bytes: &[u8]) -> u16 {
    u16::from_be_bytes(bytes.try_into().expect("two-byte ALAC field"))
}

fn read_be_u32(bytes: &[u8]) -> u32 {
    u32::from_be_bytes(bytes.try_into().expect("four-byte ALAC field"))
}

#[cfg(test)]
mod tests {
    use super::StreamInfo;

    #[test]
    fn parses_bare_and_full_box_cookies() {
        let mut config = Vec::new();
        config.extend_from_slice(&4096u32.to_be_bytes());
        config.extend_from_slice(&[0, 16, 40, 10, 14, 2]);
        config.extend_from_slice(&255u16.to_be_bytes());
        config.extend_from_slice(&0u32.to_be_bytes());
        config.extend_from_slice(&0u32.to_be_bytes());
        config.extend_from_slice(&192_000u32.to_be_bytes());

        let bare = StreamInfo::from_cookie(&config).unwrap();
        let boxed_cookie = [&[0u8; 4][..], config.as_slice()].concat();
        let boxed = StreamInfo::from_cookie(&boxed_cookie).unwrap();
        assert_eq!(bare, boxed);
        assert_eq!(bare.max_frames_per_packet(), 4096);
        assert_eq!(bare.sample_rate(), 192_000);
        assert_eq!(bare.channels(), 2);
        assert_eq!(bare.bit_depth(), 16);
    }
}
