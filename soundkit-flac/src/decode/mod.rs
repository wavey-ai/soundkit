//! Native FLAC stream and frame decoding.
//!
//! The decoder parses a native FLAC bitstream into interleaved samples. The
//! push-style entry point lives in [`crate::stream::Decoder`]; the seekable
//! reader lives in [`crate::decode`] consumers such as the packet decoder.

pub mod bitstream;
pub mod error;
pub mod frame;
pub mod metadata;
pub mod subframe;

pub use error::{Error, Result};
pub use frame::{Block, FrameReader};
