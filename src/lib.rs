#![forbid(unsafe_code)]

pub mod celt;
pub mod constants;
pub mod decoder;
pub mod encoder;
pub mod error;
mod packet;
mod repacketizer;
mod soft_clip;
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
mod wasm;

pub use decoder::Decoder;
pub use encoder::{
    Application, Encoder, CELT_FRAME_SIZES_48K, CELT_MAX_BITRATE, CELT_MAX_FRAME_BYTES,
    CELT_MIN_BITRATE, CELT_MIN_FRAME_BYTES,
};
pub use error::{Error, Result};
pub use packet::*;
pub use repacketizer::*;
pub use soft_clip::*;
