//! Photographic development for SoundKit: white balance by chromatic
//! adaptation, tone, local contrast, colour in OKLab with gamut mapping,
//! detail and effects. The same code serves the browser, through
//! WebAssembly, and native apps.

pub mod colour;
pub mod develop;
pub mod fast;
pub mod recipe;
#[cfg(feature = "wasm")]
pub mod wasm;

pub use develop::{auto_tone, develop, from_rgba, Camera, Developed, Frame, Options, Pixels, Samples};
pub use recipe::Recipe;
