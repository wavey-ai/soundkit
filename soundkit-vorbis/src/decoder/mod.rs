//! In-tree Vorbis packet decoder core.
//!
//! The parser and scalar DSP were bootstrapped from the permissively licensed
//! Lewton implementation. SoundKit owns the integrated streaming state and the
//! optimized entropy, transform, and PCM paths maintained in this module.

#![allow(dead_code, deprecated)]

macro_rules! record_residue_pre_inverse {
    ($residue_vectors:expr) => {};
}

macro_rules! record_residue_post_inverse {
    ($residue_vectors:expr) => {};
}

macro_rules! record_pre_mdct {
    ($audio_spectri:expr) => {};
}

macro_rules! record_post_mdct {
    ($audio_spectri:expr) => {};
}

pub(crate) mod audio;
mod bitpacking;
mod fast_imdct;
pub(crate) mod header;
mod header_cached;
mod huffman_tree;
#[cfg(test)]
mod imdct_test;
pub(crate) mod samples;

fn ilog(value: u64) -> u8 {
    64 - value.leading_zeros() as u8
}
