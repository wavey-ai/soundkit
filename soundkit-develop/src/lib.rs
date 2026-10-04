//! Photographic development for SoundKit: white balance by chromatic
//! adaptation, tone, local contrast, colour in OKLab with gamut mapping,
//! detail and effects. The same code serves the browser, through
//! WebAssembly, and native apps.

pub mod colour;
pub mod develop;
pub mod fast;
pub mod icc;
pub mod recipe;
#[cfg(feature = "tiff")]
pub mod tiff_file;
#[cfg(feature = "wasm")]
pub mod wasm;

pub use develop::{auto_tone, develop, from_rgba, from_rgba16, Camera, Curve, Developed, Encoded16, Frame, Options, Pixels, Samples};
#[cfg(feature = "tiff")]
pub use tiff_file::from_tiff;
pub use recipe::Recipe;
