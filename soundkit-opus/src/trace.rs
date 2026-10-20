//! Per-frame side information recorded by the decoder for encoder
//! comparisons. Built only with the `trace` feature.

use std::cell::RefCell;

#[derive(Clone, Copy, Debug)]
pub struct FrameTrace {
    pub prefilter_gain: i32,
    pub prefilter_pitch: i32,
    pub transient: bool,
    pub intra: bool,
    pub tf_changes: usize,
    pub spread: i32,
    pub total_boost: i32,
    pub alloc_trim: i32,
    pub coded_bands: usize,
    pub intensity: usize,
    pub dual_stereo: bool,
    pub fine_bits: i32,
    pub pulse_bits: i32,
}

thread_local! {
    static FRAMES: RefCell<Vec<FrameTrace>> = const { RefCell::new(Vec::new()) };
}

pub(crate) fn record(frame: FrameTrace) {
    FRAMES.with(|frames| frames.borrow_mut().push(frame));
}

/// Returns and clears the recorded frames.
pub fn take() -> Vec<FrameTrace> {
    FRAMES.with(|frames| std::mem::take(&mut *frames.borrow_mut()))
}

thread_local! {
    static COUNTERS: RefCell<[u64; 4]> = const { RefCell::new([0; 4]) };
}

/// Encoder counters: frames, analysis transients, patched transients, tone
/// overrides.
pub(crate) fn count(index: usize) {
    COUNTERS.with(|c| c.borrow_mut()[index] += 1);
}

/// Returns and clears the encoder counters.
pub fn take_counters() -> [u64; 4] {
    COUNTERS.with(|c| std::mem::take(&mut *c.borrow_mut()))
}
