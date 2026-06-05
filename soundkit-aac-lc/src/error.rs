use std::fmt;

pub type Result<T> = std::result::Result<T, AacLcError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AacLcError {
    UnexpectedEof {
        requested_bits: u8,
        remaining_bits: usize,
    },
    InvalidAudioObjectType(u8),
    UnsupportedAudioObjectType(u8),
    UnsupportedSamplingFrequencyIndex(u8),
    UnsupportedChannelConfig(u8),
    UnsupportedFeature(&'static str),
    InvalidConfig(&'static str),
    InvalidBitstream(&'static str),
    NotImplemented(&'static str),
}

impl fmt::Display for AacLcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AacLcError::UnexpectedEof {
                requested_bits,
                remaining_bits,
            } => write!(
                f,
                "unexpected end of AAC bitstream: requested {requested_bits} bits, {remaining_bits} bits remain"
            ),
            AacLcError::InvalidAudioObjectType(value) => {
                write!(f, "invalid AAC audio object type {value}")
            }
            AacLcError::UnsupportedAudioObjectType(value) => {
                write!(f, "unsupported AAC audio object type {value}")
            }
            AacLcError::UnsupportedSamplingFrequencyIndex(value) => {
                write!(f, "unsupported AAC sampling frequency index {value}")
            }
            AacLcError::UnsupportedChannelConfig(value) => {
                write!(f, "unsupported AAC channel configuration {value}")
            }
            AacLcError::UnsupportedFeature(feature) => {
                write!(f, "unsupported AAC feature: {feature}")
            }
            AacLcError::InvalidConfig(message) => write!(f, "invalid AAC config: {message}"),
            AacLcError::InvalidBitstream(message) => {
                write!(f, "invalid AAC bitstream: {message}")
            }
            AacLcError::NotImplemented(stage) => {
                write!(f, "AAC-LC decoder stage not implemented yet: {stage}")
            }
        }
    }
}

impl std::error::Error for AacLcError {}
