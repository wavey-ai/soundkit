use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    BadArg,
    BufferTooSmall,
    InternalError,
    InvalidPacket,
    Unimplemented,
    InvalidState,
    AllocFail,
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub const fn code(self) -> i32 {
        match self {
            Self::BadArg => -1,
            Self::BufferTooSmall => -2,
            Self::InternalError => -3,
            Self::InvalidPacket => -4,
            Self::Unimplemented => -5,
            Self::InvalidState => -6,
            Self::AllocFail => -7,
        }
    }

    pub const fn message(self) -> &'static str {
        match self {
            Self::BadArg => "invalid argument",
            Self::BufferTooSmall => "buffer too small",
            Self::InternalError => "internal error",
            Self::InvalidPacket => "invalid packet",
            Self::Unimplemented => "request not implemented",
            Self::InvalidState => "invalid state",
            Self::AllocFail => "memory allocation failed",
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.message())
    }
}

impl std::error::Error for Error {}
