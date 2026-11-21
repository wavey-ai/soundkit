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

/// The encoder decision constants that differ between SoundKit and libopus.
///
/// SoundKit's standard Opus encoder uses [`Decisions::SOUNDKIT`]: the four
/// tuned constants of the 2026-10-20 report, a constrained-VBR blend of 0.5
/// at frame sizes other than 5 ms, the fixed-point scale of the transient VBR
/// boost, and a drift correction in both VBR modes. Opus Custom encoders use
/// [`Decisions::LIBOPUS`], which reproduces the float build of libopus, so
/// their packets can match `opus_custom_encode` byte for byte. With the
/// `tuning` feature, a knob set through `tuning::set` still overrides either.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Decisions {
    /// Scale of the TF-change penalty in `tf_analysis` (libopus: 1).
    pub tf_bias_scale: f32,
    /// Constrained-VBR target blend at 5 ms frames (libopus: 0.67).
    pub vbr_blend_lm1: f32,
    /// Constrained-VBR target blend at other frame sizes (libopus: 0.67).
    pub vbr_blend: f32,
    /// Scale of the pitch pre-filter gain before quantisation (libopus: 1).
    pub pf_gain_scale: f32,
    /// Added to the pitch pre-filter threshold (libopus: 0).
    pub pf_threshold_offset: f32,
    /// Follow libopus's float encoder where SoundKit's differs: the VBR
    /// transient boost `(tf_estimate - 0.044) * target` (SoundKit doubles
    /// it, as the fixed-point build does), the VBR rate offset and drift
    /// updated in constrained VBR only, and trunk's check that rejects a
    /// pitch whose 5/8 or 6/8 multiple correlates as well.
    pub libopus_float: bool,
}

impl Decisions {
    pub const SOUNDKIT: Self = Self {
        tf_bias_scale: 3.0,
        vbr_blend_lm1: 0.4,
        vbr_blend: 0.5,
        pf_gain_scale: 0.8,
        pf_threshold_offset: 0.1,
        libopus_float: false,
    };
    pub const LIBOPUS: Self = Self {
        tf_bias_scale: 1.0,
        vbr_blend_lm1: 0.67,
        vbr_blend: 0.67,
        pf_gain_scale: 1.0,
        pf_threshold_offset: 0.0,
        libopus_float: true,
    };
}

impl Default for Decisions {
    fn default() -> Self {
        Self::SOUNDKIT
    }
}

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
pub(crate) use imp::get;
#[cfg(feature = "tuning")]
pub use imp::{clear, set};

#[cfg(not(feature = "tuning"))]
#[inline(always)]
pub(crate) fn get(_knob: usize, default: f32) -> f32 {
    default
}
