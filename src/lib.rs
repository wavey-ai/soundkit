#![forbid(unsafe_code)]

pub mod celt;
pub mod constants;
pub mod decoder;
pub mod encoder;
pub mod error;
mod packet;
mod repacketizer;
mod soft_clip;

pub use decoder::Decoder;
pub use encoder::{Application, Encoder};
pub use error::{Error, Result};
pub use packet::*;
pub use repacketizer::*;
pub use soft_clip::*;
