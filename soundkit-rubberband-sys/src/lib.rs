#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]

use std::os::raw::{c_double, c_int, c_uint};

pub const RUBBERBAND_VERSION: &str = "4.0.0";
pub const RUBBERBAND_API_MAJOR_VERSION: u32 = 3;
pub const RUBBERBAND_API_MINOR_VERSION: u32 = 0;

pub type RubberBandOptions = c_int;
pub type RubberBandLiveOptions = c_int;

pub const RubberBandOptionProcessOffline: RubberBandOptions = 0x00000000;
pub const RubberBandOptionProcessRealTime: RubberBandOptions = 0x00000001;
pub const RubberBandOptionStretchElastic: RubberBandOptions = 0x00000000;
pub const RubberBandOptionStretchPrecise: RubberBandOptions = 0x00000010;
pub const RubberBandOptionTransientsCrisp: RubberBandOptions = 0x00000000;
pub const RubberBandOptionTransientsMixed: RubberBandOptions = 0x00000100;
pub const RubberBandOptionTransientsSmooth: RubberBandOptions = 0x00000200;
pub const RubberBandOptionDetectorCompound: RubberBandOptions = 0x00000000;
pub const RubberBandOptionDetectorPercussive: RubberBandOptions = 0x00000400;
pub const RubberBandOptionDetectorSoft: RubberBandOptions = 0x00000800;
pub const RubberBandOptionPhaseLaminar: RubberBandOptions = 0x00000000;
pub const RubberBandOptionPhaseIndependent: RubberBandOptions = 0x00002000;
pub const RubberBandOptionThreadingAuto: RubberBandOptions = 0x00000000;
pub const RubberBandOptionThreadingNever: RubberBandOptions = 0x00010000;
pub const RubberBandOptionThreadingAlways: RubberBandOptions = 0x00020000;
pub const RubberBandOptionWindowStandard: RubberBandOptions = 0x00000000;
pub const RubberBandOptionWindowShort: RubberBandOptions = 0x00100000;
pub const RubberBandOptionWindowLong: RubberBandOptions = 0x00200000;
pub const RubberBandOptionSmoothingOff: RubberBandOptions = 0x00000000;
pub const RubberBandOptionSmoothingOn: RubberBandOptions = 0x00800000;
pub const RubberBandOptionFormantShifted: RubberBandOptions = 0x00000000;
pub const RubberBandOptionFormantPreserved: RubberBandOptions = 0x01000000;
pub const RubberBandOptionPitchHighSpeed: RubberBandOptions = 0x00000000;
pub const RubberBandOptionPitchHighQuality: RubberBandOptions = 0x02000000;
pub const RubberBandOptionPitchHighConsistency: RubberBandOptions = 0x04000000;
pub const RubberBandOptionChannelsApart: RubberBandOptions = 0x00000000;
pub const RubberBandOptionChannelsTogether: RubberBandOptions = 0x10000000;
pub const RubberBandOptionEngineFaster: RubberBandOptions = 0x00000000;
pub const RubberBandOptionEngineFiner: RubberBandOptions = 0x20000000;

pub const RubberBandLiveOptionWindowShort: RubberBandLiveOptions = 0x00000000;
pub const RubberBandLiveOptionWindowMedium: RubberBandLiveOptions = 0x00100000;
pub const RubberBandLiveOptionFormantShifted: RubberBandLiveOptions = 0x00000000;
pub const RubberBandLiveOptionFormantPreserved: RubberBandLiveOptions = 0x01000000;
pub const RubberBandLiveOptionChannelsApart: RubberBandLiveOptions = 0x00000000;
pub const RubberBandLiveOptionChannelsTogether: RubberBandLiveOptions = 0x10000000;

#[repr(C)]
pub struct RubberBandState_ {
    _private: [u8; 0],
}

pub type RubberBandState = *mut RubberBandState_;

#[repr(C)]
pub struct RubberBandLiveState_ {
    _private: [u8; 0],
}

pub type RubberBandLiveState = *mut RubberBandLiveState_;

extern "C" {
    pub fn rubberband_new(
        sampleRate: c_uint,
        channels: c_uint,
        options: RubberBandOptions,
        initialTimeRatio: c_double,
        initialPitchScale: c_double,
    ) -> RubberBandState;
    pub fn rubberband_delete(state: RubberBandState);
    pub fn rubberband_reset(state: RubberBandState);

    pub fn rubberband_get_engine_version(state: RubberBandState) -> c_int;

    pub fn rubberband_set_time_ratio(state: RubberBandState, ratio: c_double);
    pub fn rubberband_set_pitch_scale(state: RubberBandState, scale: c_double);
    pub fn rubberband_get_time_ratio(state: RubberBandState) -> c_double;
    pub fn rubberband_get_pitch_scale(state: RubberBandState) -> c_double;

    pub fn rubberband_set_formant_scale(state: RubberBandState, scale: c_double);
    pub fn rubberband_get_formant_scale(state: RubberBandState) -> c_double;

    pub fn rubberband_get_preferred_start_pad(state: RubberBandState) -> c_uint;
    pub fn rubberband_get_start_delay(state: RubberBandState) -> c_uint;
    pub fn rubberband_get_latency(state: RubberBandState) -> c_uint;

    pub fn rubberband_set_transients_option(state: RubberBandState, options: RubberBandOptions);
    pub fn rubberband_set_detector_option(state: RubberBandState, options: RubberBandOptions);
    pub fn rubberband_set_phase_option(state: RubberBandState, options: RubberBandOptions);
    pub fn rubberband_set_formant_option(state: RubberBandState, options: RubberBandOptions);
    pub fn rubberband_set_pitch_option(state: RubberBandState, options: RubberBandOptions);

    pub fn rubberband_set_expected_input_duration(state: RubberBandState, samples: c_uint);
    pub fn rubberband_get_samples_required(state: RubberBandState) -> c_uint;
    pub fn rubberband_set_max_process_size(state: RubberBandState, samples: c_uint);
    pub fn rubberband_get_process_size_limit(state: RubberBandState) -> c_uint;
    pub fn rubberband_set_key_frame_map(
        state: RubberBandState,
        keyframecount: c_uint,
        from: *mut c_uint,
        to: *mut c_uint,
    );

    pub fn rubberband_study(
        state: RubberBandState,
        input: *const *const f32,
        samples: c_uint,
        final_: c_int,
    );
    pub fn rubberband_process(
        state: RubberBandState,
        input: *const *const f32,
        samples: c_uint,
        final_: c_int,
    );

    pub fn rubberband_available(state: RubberBandState) -> c_int;
    pub fn rubberband_retrieve(
        state: RubberBandState,
        output: *mut *mut f32,
        samples: c_uint,
    ) -> c_uint;

    pub fn rubberband_get_channel_count(state: RubberBandState) -> c_uint;
    pub fn rubberband_calculate_stretch(state: RubberBandState);
    pub fn rubberband_set_debug_level(state: RubberBandState, level: c_int);
    pub fn rubberband_set_default_debug_level(level: c_int);

    pub fn rubberband_live_new(
        sampleRate: c_uint,
        channels: c_uint,
        options: RubberBandOptions,
    ) -> RubberBandLiveState;
    pub fn rubberband_live_delete(state: RubberBandLiveState);
    pub fn rubberband_live_reset(state: RubberBandLiveState);
    pub fn rubberband_live_set_pitch_scale(state: RubberBandLiveState, scale: c_double);
    pub fn rubberband_live_get_pitch_scale(state: RubberBandLiveState) -> c_double;
    pub fn rubberband_live_set_formant_scale(state: RubberBandLiveState, scale: c_double);
    pub fn rubberband_live_get_formant_scale(state: RubberBandLiveState) -> c_double;
    pub fn rubberband_live_get_start_delay(state: RubberBandLiveState) -> c_uint;
    pub fn rubberband_live_set_formant_option(
        state: RubberBandLiveState,
        options: RubberBandOptions,
    );
    pub fn rubberband_live_get_block_size(state: RubberBandLiveState) -> c_uint;
    pub fn rubberband_live_shift(
        state: RubberBandLiveState,
        input: *const *const f32,
        output: *mut *mut f32,
    );
    pub fn rubberband_live_get_channel_count(state: RubberBandLiveState) -> c_uint;
    pub fn rubberband_live_set_debug_level(state: RubberBandLiveState, level: c_int);
    pub fn rubberband_live_set_default_debug_level(level: c_int);
}
