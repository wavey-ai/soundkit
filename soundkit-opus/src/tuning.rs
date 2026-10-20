//! Encoder decision constants that a parameter search can change at run time.
//!
//! With the `tuning` feature off, `get` returns the default constant and the
//! compiler folds it into the caller.

pub const VBR_BLEND_LM1: usize = 0;
pub const TRIM_OFFSET: usize = 1;
pub const DUAL_STEREO_BIAS: usize = 2;
pub const SPREAD_FORCE: usize = 3;
pub const DYNALLOC_SCALE: usize = 4;
pub const TF_BIAS_SCALE: usize = 5;
pub const PREFILTER_ENABLE: usize = 6;
pub const TRANSIENT_THRESHOLD: usize = 7;
pub const TF_LAMBDA_SCALE: usize = 8;
pub const VBR_UNCONSTRAINED: usize = 9;
pub const PF_THRESHOLD_OFFSET: usize = 10;
pub const PF_GAIN_SCALE: usize = 11;
pub const DUAL_STEREO_SMOOTH: usize = 12;
pub const DUAL_STEREO_BANDS: usize = 13;
pub const ANALYSIS_OFF: usize = 14;
pub const KNOB_COUNT: usize = 16;

#[cfg(feature = "tuning")]
mod imp {
    use super::KNOB_COUNT;
    use std::sync::atomic::{AtomicU32, Ordering};

    static VALUES: [AtomicU32; KNOB_COUNT] = [const { AtomicU32::new(0) }; KNOB_COUNT];
    static SET: AtomicU32 = AtomicU32::new(0);

    /// Overrides one knob for every encoder in the process.
    pub fn set(knob: usize, value: f32) {
        VALUES[knob].store(value.to_bits(), Ordering::Relaxed);
        SET.fetch_or(1 << knob, Ordering::Relaxed);
    }

    /// Restores every knob to its default.
    pub fn clear() {
        SET.store(0, Ordering::Relaxed);
    }

    #[inline]
    pub(crate) fn get(knob: usize, default: f32) -> f32 {
        if SET.load(Ordering::Relaxed) & (1 << knob) != 0 {
            f32::from_bits(VALUES[knob].load(Ordering::Relaxed))
        } else {
            default
        }
    }
}

#[cfg(feature = "tuning")]
pub use imp::{clear, set};
#[cfg(feature = "tuning")]
pub(crate) use imp::get;

#[cfg(not(feature = "tuning"))]
#[inline(always)]
pub(crate) fn get(_knob: usize, default: f32) -> f32 {
    default
}
