//! Offline-first Rust wrapper around the Rubber Band time-stretcher.
//!
//! This crate targets remix/render workflows where quality matters more than
//! callback-oriented real-time ergonomics. Rubber Band's own integration notes
//! recommend an offline two-pass flow for maximum quality, which is what the
//! helper functions here implement.
//!
//! System setup:
//! - macOS/Homebrew: `brew install rubberband`
//! - Ubuntu/Debian: `apt-get install librubberband-dev`

use bitflags::bitflags;
use frame_header::{EncodingFlag, Endianness};
use soundkit::audio_bytes::s32le_to_i32;
use soundkit::audio_pipeline::{deserialize_audio, vec_f32_to_i16, vec_i16_to_f32, vec_i32_to_f32};
use soundkit::audio_types::{AudioData, PcmData};
use soundkit_rubberband_sys as ffi;
use std::error::Error;
use std::fmt;
use std::ptr::NonNull;

const DEFAULT_CHUNK_FRAMES: usize = 8192;
const EPSILON: f64 = 1.0e-9;

bitflags! {
    /// Non-zero Rubber Band option bits.
    ///
    /// Zero-valued defaults such as offline mode, standard windowing,
    /// `ChannelsApart`, and `EngineFaster` are intentionally not represented as
    /// flags here. Leave the corresponding flag unset to use those defaults.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct RubberBandOptions: u32 {
        const PROCESS_REAL_TIME = ffi::RubberBandOptionProcessRealTime as u32;
        const STRETCH_PRECISE = ffi::RubberBandOptionStretchPrecise as u32;
        const TRANSIENTS_MIXED = ffi::RubberBandOptionTransientsMixed as u32;
        const TRANSIENTS_SMOOTH = ffi::RubberBandOptionTransientsSmooth as u32;
        const DETECTOR_PERCUSSIVE = ffi::RubberBandOptionDetectorPercussive as u32;
        const DETECTOR_SOFT = ffi::RubberBandOptionDetectorSoft as u32;
        const PHASE_INDEPENDENT = ffi::RubberBandOptionPhaseIndependent as u32;
        const THREADING_NEVER = ffi::RubberBandOptionThreadingNever as u32;
        const THREADING_ALWAYS = ffi::RubberBandOptionThreadingAlways as u32;
        const WINDOW_SHORT = ffi::RubberBandOptionWindowShort as u32;
        const WINDOW_LONG = ffi::RubberBandOptionWindowLong as u32;
        const SMOOTHING_ON = ffi::RubberBandOptionSmoothingOn as u32;
        const FORMANT_PRESERVED = ffi::RubberBandOptionFormantPreserved as u32;
        const PITCH_HIGH_QUALITY = ffi::RubberBandOptionPitchHighQuality as u32;
        const PITCH_HIGH_CONSISTENCY = ffi::RubberBandOptionPitchHighConsistency as u32;
        const CHANNELS_TOGETHER = ffi::RubberBandOptionChannelsTogether as u32;
        const ENGINE_FINER = ffi::RubberBandOptionEngineFiner as u32;
    }
}

#[derive(Debug, Clone)]
pub struct OfflineStretchConfig {
    sample_rate: u32,
    channels: u32,
    time_ratio: f64,
    pitch_scale: f64,
    formant_scale: f64,
    options: RubberBandOptions,
    chunk_frames: usize,
}

impl OfflineStretchConfig {
    pub fn recommended_for_music(sample_rate: u32, channels: u32) -> Self {
        Self {
            sample_rate,
            channels,
            time_ratio: 1.0,
            pitch_scale: 1.0,
            formant_scale: 1.0,
            options: RubberBandOptions::ENGINE_FINER | RubberBandOptions::CHANNELS_TOGETHER,
            chunk_frames: DEFAULT_CHUNK_FRAMES,
        }
    }

    pub fn with_time_ratio(mut self, time_ratio: f64) -> Self {
        self.time_ratio = time_ratio;
        self
    }

    pub fn with_pitch_scale(mut self, pitch_scale: f64) -> Self {
        self.pitch_scale = pitch_scale;
        self
    }

    pub fn with_formant_scale(mut self, formant_scale: f64) -> Self {
        self.formant_scale = formant_scale;
        self
    }

    pub fn with_options(mut self, options: RubberBandOptions) -> Self {
        self.options = options;
        self
    }

    pub fn with_chunk_frames(mut self, chunk_frames: usize) -> Self {
        self.chunk_frames = chunk_frames;
        self
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn channels(&self) -> u32 {
        self.channels
    }

    pub fn time_ratio(&self) -> f64 {
        self.time_ratio
    }

    pub fn pitch_scale(&self) -> f64 {
        self.pitch_scale
    }

    pub fn formant_scale(&self) -> f64 {
        self.formant_scale
    }

    pub fn options(&self) -> RubberBandOptions {
        self.options
    }

    fn validate(&self) -> Result<(), StretchError> {
        if self.sample_rate == 0 {
            return Err(StretchError::InvalidSampleRate(self.sample_rate));
        }
        if self.channels == 0 {
            return Err(StretchError::InvalidChannelCount(self.channels));
        }
        if !self.time_ratio.is_finite() || self.time_ratio <= 0.0 {
            return Err(StretchError::InvalidRatio {
                field: "time_ratio",
                value: self.time_ratio,
            });
        }
        if !self.pitch_scale.is_finite() || self.pitch_scale <= 0.0 {
            return Err(StretchError::InvalidRatio {
                field: "pitch_scale",
                value: self.pitch_scale,
            });
        }
        if !self.formant_scale.is_finite() || self.formant_scale <= 0.0 {
            return Err(StretchError::InvalidRatio {
                field: "formant_scale",
                value: self.formant_scale,
            });
        }
        if self.chunk_frames == 0 {
            return Err(StretchError::InvalidChunkFrames(self.chunk_frames));
        }
        if self.options.contains(RubberBandOptions::PROCESS_REAL_TIME) {
            return Err(StretchError::UnsupportedOption(
                "offline wrapper does not support PROCESS_REAL_TIME".to_string(),
            ));
        }
        Ok(())
    }

    fn effective_options(&self) -> RubberBandOptions {
        let mut options = self.options;
        if (self.pitch_scale - 1.0).abs() > EPSILON
            && !options.intersects(
                RubberBandOptions::PITCH_HIGH_QUALITY | RubberBandOptions::PITCH_HIGH_CONSISTENCY,
            )
        {
            options |= RubberBandOptions::PITCH_HIGH_QUALITY;
        }
        options
    }
}

pub fn recommended_config_for_audio(audio: &AudioData) -> OfflineStretchConfig {
    OfflineStretchConfig::recommended_for_music(
        audio.sampling_rate(),
        u32::from(audio.channel_count()),
    )
}

#[derive(Debug, Clone, PartialEq)]
pub enum StretchError {
    InvalidSampleRate(u32),
    InvalidChannelCount(u32),
    InvalidChunkFrames(usize),
    InvalidRatio {
        field: &'static str,
        value: f64,
    },
    InputLengthNotInterleaved {
        sample_count: usize,
        channels: usize,
    },
    AudioMetadataMismatch {
        expected_sample_rate: u32,
        actual_sample_rate: u32,
        expected_channels: u32,
        actual_channels: u8,
    },
    InvalidAudioByteLength {
        byte_len: usize,
        bytes_per_frame: usize,
    },
    UnsupportedEndianness,
    UnsupportedAudioData(String),
    AudioDecodeFailed(String),
    ChannelCountMismatch {
        expected: usize,
        actual: usize,
    },
    ChannelLengthMismatch,
    FrameCountOverflow(usize),
    EngineCreationFailed,
    RetrieveReturnedZero(usize),
    UnexpectedAvailable(i32),
    UnsupportedOption(String),
}

impl fmt::Display for StretchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StretchError::InvalidSampleRate(rate) => {
                write!(f, "sample_rate must be > 0, got {}", rate)
            }
            StretchError::InvalidChannelCount(channels) => {
                write!(f, "channels must be > 0, got {}", channels)
            }
            StretchError::InvalidChunkFrames(frames) => {
                write!(f, "chunk_frames must be > 0, got {}", frames)
            }
            StretchError::InvalidRatio { field, value } => {
                write!(f, "{} must be finite and > 0, got {}", field, value)
            }
            StretchError::InputLengthNotInterleaved {
                sample_count,
                channels,
            } => write!(
                f,
                "input sample count {} is not divisible by channel count {}",
                sample_count, channels
            ),
            StretchError::AudioMetadataMismatch {
                expected_sample_rate,
                actual_sample_rate,
                expected_channels,
                actual_channels,
            } => write!(
                f,
                "config expects {} Hz / {} channels but audio is {} Hz / {} channels",
                expected_sample_rate, expected_channels, actual_sample_rate, actual_channels
            ),
            StretchError::InvalidAudioByteLength {
                byte_len,
                bytes_per_frame,
            } => write!(
                f,
                "audio byte length {} is not divisible by bytes per frame {}",
                byte_len, bytes_per_frame
            ),
            StretchError::UnsupportedEndianness => {
                write!(f, "only little-endian AudioData is currently supported")
            }
            StretchError::UnsupportedAudioData(message) => f.write_str(message),
            StretchError::AudioDecodeFailed(message) => {
                write!(f, "failed to decode AudioData: {}", message)
            }
            StretchError::ChannelCountMismatch { expected, actual } => write!(
                f,
                "channel count mismatch: expected {}, got {}",
                expected, actual
            ),
            StretchError::ChannelLengthMismatch => {
                write!(f, "all channels must contain the same number of frames")
            }
            StretchError::FrameCountOverflow(frames) => write!(
                f,
                "frame count {} exceeds Rubber Band's unsigned int API limit",
                frames
            ),
            StretchError::EngineCreationFailed => {
                write!(f, "rubberband_new returned a null stretcher")
            }
            StretchError::RetrieveReturnedZero(available) => write!(
                f,
                "rubberband_retrieve returned 0 while {} frames were reported available",
                available
            ),
            StretchError::UnexpectedAvailable(available) => {
                write!(f, "rubberband_available returned {}", available)
            }
            StretchError::UnsupportedOption(message) => f.write_str(message),
        }
    }
}

impl Error for StretchError {}

pub fn stretch_audio_data(
    audio: &AudioData,
    config: &OfflineStretchConfig,
) -> Result<AudioData, StretchError> {
    let stretched = stretch_audio_data_to_pcm(audio, config)?;
    Ok(audio_data_from_f32_pcm(&stretched, audio.sampling_rate()))
}

pub fn stretch_audio_data_to_pcm(
    audio: &AudioData,
    config: &OfflineStretchConfig,
) -> Result<PcmData, StretchError> {
    validate_audio_data_matches_config(audio, config)?;
    let input = audio_data_to_f32_channels(audio)?;
    let stretched = stretch_deinterleaved(&input, config)?;
    Ok(PcmData::F32(stretched))
}

pub fn stretch_audio_data_preserve_format(
    audio: &AudioData,
    config: &OfflineStretchConfig,
) -> Result<AudioData, StretchError> {
    validate_audio_data_matches_config(audio, config)?;
    let input = audio_data_to_f32_channels(audio)?;
    let stretched = stretch_deinterleaved(&input, config)?;
    audio_data_from_f32_channels(
        &stretched,
        audio.sampling_rate(),
        audio.bits_per_sample(),
        audio.audio_format(),
        audio.endianness(),
    )
}

pub fn stretch_interleaved(
    input: &[f32],
    config: &OfflineStretchConfig,
) -> Result<Vec<f32>, StretchError> {
    config.validate()?;
    let channels = config.channels as usize;
    if !input.len().is_multiple_of(channels) {
        return Err(StretchError::InputLengthNotInterleaved {
            sample_count: input.len(),
            channels,
        });
    }

    let deinterleaved = deinterleave_interleaved(input, channels);
    let stretched = stretch_deinterleaved(&deinterleaved, config)?;
    Ok(interleave_channels(&stretched))
}

pub fn stretch_deinterleaved(
    input: &[Vec<f32>],
    config: &OfflineStretchConfig,
) -> Result<Vec<Vec<f32>>, StretchError> {
    config.validate()?;

    let expected_channels = config.channels as usize;
    if input.len() != expected_channels {
        return Err(StretchError::ChannelCountMismatch {
            expected: expected_channels,
            actual: input.len(),
        });
    }

    let input_frames = validate_channel_layout(input)?;
    if input_frames == 0 {
        return Ok(vec![Vec::new(); expected_channels]);
    }

    let mut stretcher = RawStretcher::new(config)?;
    stretcher.set_expected_input_duration(input_frames)?;
    stretcher.study(input, config.chunk_frames)?;
    stretcher.calculate_stretch();
    stretcher.process(input, config.chunk_frames, config.time_ratio)
}

fn deinterleave_interleaved(input: &[f32], channels: usize) -> Vec<Vec<f32>> {
    let frames = input.len() / channels;
    let mut result = vec![Vec::with_capacity(frames); channels];

    for frame in input.chunks_exact(channels) {
        for (channel, sample) in frame.iter().copied().enumerate() {
            result[channel].push(sample);
        }
    }

    result
}

fn validate_audio_data_matches_config(
    audio: &AudioData,
    config: &OfflineStretchConfig,
) -> Result<(), StretchError> {
    match audio.audio_format() {
        EncodingFlag::PCMSigned | EncodingFlag::PCMFloat => {}
        other => {
            return Err(StretchError::UnsupportedAudioData(format!(
                "unsupported audio format {:?}; expected decoded PCM audio",
                other
            )));
        }
    }

    if audio.endianness() != Endianness::LittleEndian {
        return Err(StretchError::UnsupportedEndianness);
    }

    let bytes_per_frame =
        usize::from(audio.channel_count()) * usize::from(audio.bits_per_sample() / 8);
    if bytes_per_frame == 0 || !audio.data().len().is_multiple_of(bytes_per_frame) {
        return Err(StretchError::InvalidAudioByteLength {
            byte_len: audio.data().len(),
            bytes_per_frame,
        });
    }

    if audio.sampling_rate() != config.sample_rate
        || u32::from(audio.channel_count()) != config.channels
    {
        return Err(StretchError::AudioMetadataMismatch {
            expected_sample_rate: config.sample_rate,
            actual_sample_rate: audio.sampling_rate(),
            expected_channels: config.channels,
            actual_channels: audio.channel_count(),
        });
    }

    Ok(())
}

fn audio_data_to_f32_channels(audio: &AudioData) -> Result<Vec<Vec<f32>>, StretchError> {
    let channel_count = usize::from(audio.channel_count());

    if audio.bits_per_sample() == 32 && audio.audio_format() != EncodingFlag::PCMFloat {
        let interleaved = s32le_to_i32(audio.data());
        let mut channels =
            vec![Vec::with_capacity(interleaved.len() / channel_count); channel_count];
        for (index, sample) in interleaved.into_iter().enumerate() {
            channels[index % channel_count].push(sample);
        }
        return Ok(channels.into_iter().map(vec_i32_to_f32).collect());
    }

    let pcm = deserialize_audio(audio.data(), audio.bits_per_sample(), audio.channel_count())
        .map_err(StretchError::AudioDecodeFailed)?;

    match pcm {
        PcmData::I16(data) => Ok(data.into_iter().map(vec_i16_to_f32).collect()),
        PcmData::I32(data) => Ok(data.into_iter().map(vec_i32_to_f32).collect()),
        PcmData::F32(data) => Ok(data),
    }
}

fn interleave_channels(channels: &[Vec<f32>]) -> Vec<f32> {
    let frames = channels.first().map_or(0, Vec::len);
    let mut result = Vec::with_capacity(frames * channels.len());

    for frame_index in 0..frames {
        for channel in channels {
            result.push(channel[frame_index]);
        }
    }

    result
}

fn audio_data_from_f32_pcm(pcm: &PcmData, sample_rate: u32) -> AudioData {
    let PcmData::F32(channels) = pcm else {
        unreachable!("audio helper only produces f32 pcm");
    };

    AudioData::new(
        32,
        channels.len() as u8,
        sample_rate,
        interleave_f32_channels(channels),
        EncodingFlag::PCMFloat,
        Endianness::LittleEndian,
    )
}

fn audio_data_from_f32_channels(
    channels: &[Vec<f32>],
    sample_rate: u32,
    bits_per_sample: u8,
    audio_format: EncodingFlag,
    endianness: Endianness,
) -> Result<AudioData, StretchError> {
    if endianness != Endianness::LittleEndian {
        return Err(StretchError::UnsupportedEndianness);
    }

    let data = match (audio_format, bits_per_sample) {
        (EncodingFlag::PCMFloat, 32) => interleave_f32_channels(channels),
        (EncodingFlag::PCMSigned, 16) => interleave_i16_channels(
            &channels
                .iter()
                .cloned()
                .map(vec_f32_to_i16)
                .collect::<Vec<Vec<i16>>>(),
        ),
        (EncodingFlag::PCMSigned, 24) => interleave_s24_channels(channels),
        (EncodingFlag::PCMSigned, 32) => interleave_i32_channels(channels),
        (encoding, bits) => {
            return Err(StretchError::UnsupportedAudioData(format!(
                "cannot preserve format {:?} with {} bits per sample",
                encoding, bits
            )))
        }
    };

    Ok(AudioData::new(
        bits_per_sample,
        channels.len() as u8,
        sample_rate,
        data,
        audio_format,
        endianness,
    ))
}

fn interleave_f32_channels(channels: &[Vec<f32>]) -> Vec<u8> {
    let frames = channels.first().map_or(0, Vec::len);
    let mut result = Vec::with_capacity(frames * channels.len() * std::mem::size_of::<f32>());

    for frame_index in 0..frames {
        for channel in channels {
            result.extend_from_slice(&channel[frame_index].to_le_bytes());
        }
    }

    result
}

fn interleave_i16_channels(channels: &[Vec<i16>]) -> Vec<u8> {
    let frames = channels.first().map_or(0, Vec::len);
    let mut result = Vec::with_capacity(frames * channels.len() * std::mem::size_of::<i16>());

    for frame_index in 0..frames {
        for channel in channels {
            result.extend_from_slice(&channel[frame_index].to_le_bytes());
        }
    }

    result
}

fn interleave_i32_channels(channels: &[Vec<f32>]) -> Vec<u8> {
    let frames = channels.first().map_or(0, Vec::len);
    let mut result = Vec::with_capacity(frames * channels.len() * std::mem::size_of::<i32>());

    for frame_index in 0..frames {
        for channel in channels {
            result.extend_from_slice(&f32_to_i32_sample(channel[frame_index]).to_le_bytes());
        }
    }

    result
}

fn interleave_s24_channels(channels: &[Vec<f32>]) -> Vec<u8> {
    let frames = channels.first().map_or(0, Vec::len);
    let mut result = Vec::with_capacity(frames * channels.len() * 3);

    for frame_index in 0..frames {
        for channel in channels {
            let bytes = f32_to_s24_sample(channel[frame_index]).to_le_bytes();
            result.extend_from_slice(&bytes[..3]);
        }
    }

    result
}

fn f32_to_i32_sample(sample: f32) -> i32 {
    let clamped = sample.clamp(-1.0, 1.0);
    if clamped >= 0.0 {
        (clamped * i32::MAX as f32) as i32
    } else {
        (clamped * -(i32::MIN as f32)) as i32
    }
}

fn f32_to_s24_sample(sample: f32) -> i32 {
    let clamped = sample.clamp(-1.0, 1.0);
    let s24_max = 8_388_607_f32;
    if clamped >= 0.0 {
        (clamped * s24_max) as i32
    } else {
        (clamped * (s24_max + 1.0)) as i32
    }
}

fn validate_channel_layout(channels: &[Vec<f32>]) -> Result<usize, StretchError> {
    let Some(first) = channels.first() else {
        return Ok(0);
    };
    let frames = first.len();
    if channels.iter().any(|channel| channel.len() != frames) {
        return Err(StretchError::ChannelLengthMismatch);
    }
    Ok(frames)
}

fn to_c_frames(frames: usize) -> Result<u32, StretchError> {
    u32::try_from(frames).map_err(|_| StretchError::FrameCountOverflow(frames))
}

struct RawStretcher {
    state: NonNull<ffi::RubberBandState_>,
    channels: usize,
}

impl RawStretcher {
    fn new(config: &OfflineStretchConfig) -> Result<Self, StretchError> {
        let options = config.effective_options();
        let state = unsafe {
            ffi::rubberband_new(
                config.sample_rate,
                config.channels,
                options.bits() as ffi::RubberBandOptions,
                config.time_ratio,
                config.pitch_scale,
            )
        };
        let state = NonNull::new(state).ok_or(StretchError::EngineCreationFailed)?;

        if (config.formant_scale - 1.0).abs() > EPSILON {
            unsafe { ffi::rubberband_set_formant_scale(state.as_ptr(), config.formant_scale) };
        }

        Ok(Self {
            state,
            channels: config.channels as usize,
        })
    }

    fn set_expected_input_duration(&mut self, frames: usize) -> Result<(), StretchError> {
        unsafe {
            ffi::rubberband_set_expected_input_duration(self.state.as_ptr(), to_c_frames(frames)?)
        };
        Ok(())
    }

    fn study(&mut self, input: &[Vec<f32>], chunk_frames: usize) -> Result<(), StretchError> {
        self.feed_chunks(
            input,
            chunk_frames,
            |state, ptrs, frames, is_final| unsafe {
                ffi::rubberband_study(state, ptrs, frames, is_final as i32);
            },
        )
    }

    fn calculate_stretch(&mut self) {
        unsafe { ffi::rubberband_calculate_stretch(self.state.as_ptr()) };
    }

    fn process(
        &mut self,
        input: &[Vec<f32>],
        chunk_frames: usize,
        time_ratio: f64,
    ) -> Result<Vec<Vec<f32>>, StretchError> {
        let input_frames = validate_channel_layout(input)?;
        let reserve = (((input_frames as f64) * time_ratio.max(1.0)).ceil() as usize)
            .saturating_add(chunk_frames * 2);
        let mut output = vec![Vec::with_capacity(reserve); self.channels];

        let total_frames = validate_channel_layout(input)?;
        for start in (0..total_frames).step_by(chunk_frames) {
            let end = (start + chunk_frames).min(total_frames);
            let mut ptrs = Vec::with_capacity(self.channels);
            for channel in input {
                ptrs.push(channel[start..end].as_ptr());
            }

            unsafe {
                ffi::rubberband_process(
                    self.state.as_ptr(),
                    ptrs.as_ptr(),
                    to_c_frames(end - start)?,
                    (end == total_frames) as i32,
                );
            }
            self.retrieve_all_available(&mut output)?;
        }

        Ok(output)
    }

    fn feed_chunks<F>(
        &mut self,
        input: &[Vec<f32>],
        chunk_frames: usize,
        mut f: F,
    ) -> Result<(), StretchError>
    where
        F: FnMut(ffi::RubberBandState, *const *const f32, u32, bool),
    {
        let total_frames = validate_channel_layout(input)?;
        for start in (0..total_frames).step_by(chunk_frames) {
            let end = (start + chunk_frames).min(total_frames);
            let mut ptrs = Vec::with_capacity(self.channels);
            for channel in input {
                ptrs.push(channel[start..end].as_ptr());
            }
            f(
                self.state.as_ptr(),
                ptrs.as_ptr(),
                to_c_frames(end - start)?,
                end == total_frames,
            );
        }
        Ok(())
    }

    fn retrieve_all_available(&mut self, output: &mut [Vec<f32>]) -> Result<(), StretchError> {
        loop {
            let Some(available) = self.available()? else {
                break;
            };
            if available == 0 {
                break;
            }

            let mut buffers = vec![vec![0.0_f32; available]; self.channels];
            let mut ptrs: Vec<*mut f32> = buffers
                .iter_mut()
                .map(|channel| channel.as_mut_ptr())
                .collect();
            let retrieved = unsafe {
                ffi::rubberband_retrieve(
                    self.state.as_ptr(),
                    ptrs.as_mut_ptr(),
                    to_c_frames(available)?,
                )
            } as usize;
            if retrieved == 0 {
                return Err(StretchError::RetrieveReturnedZero(available));
            }

            for (destination, channel) in output.iter_mut().zip(buffers.iter()) {
                destination.extend_from_slice(&channel[..retrieved]);
            }
        }
        Ok(())
    }

    fn available(&self) -> Result<Option<usize>, StretchError> {
        let available = unsafe { ffi::rubberband_available(self.state.as_ptr()) };
        if available == -1 {
            Ok(None)
        } else if available < 0 {
            Err(StretchError::UnexpectedAvailable(available))
        } else {
            Ok(Some(available as usize))
        }
    }
}

impl Drop for RawStretcher {
    fn drop(&mut self) {
        unsafe { ffi::rubberband_delete(self.state.as_ptr()) };
    }
}

#[cfg(test)]
mod tests {
    use super::{
        recommended_config_for_audio, stretch_audio_data, stretch_audio_data_preserve_format,
        stretch_audio_data_to_pcm, stretch_interleaved, OfflineStretchConfig, RubberBandOptions,
        StretchError,
    };
    use frame_header::{EncodingFlag, Endianness};
    use soundkit::audio_pipeline::vec_f32_to_i16;
    use soundkit::audio_types::{AudioData, PcmData};

    fn sine_wave(frames: usize, sample_rate: usize, frequency: f32) -> Vec<f32> {
        (0..frames)
            .map(|index| {
                let phase = index as f32 * frequency * std::f32::consts::TAU / sample_rate as f32;
                phase.sin()
            })
            .collect()
    }

    #[test]
    fn recommended_music_config_uses_high_quality_defaults() {
        let config = OfflineStretchConfig::recommended_for_music(48_000, 2);
        assert!(config.options().contains(RubberBandOptions::ENGINE_FINER));
        assert!(config
            .options()
            .contains(RubberBandOptions::CHANNELS_TOGETHER));
        assert_eq!(config.time_ratio(), 1.0);
        assert_eq!(config.pitch_scale(), 1.0);
    }

    #[test]
    fn stretch_interleaved_slows_audio_down() {
        let input = sine_wave(48_000, 48_000, 440.0);
        let config = OfflineStretchConfig::recommended_for_music(48_000, 1)
            .with_time_ratio(1.5)
            .with_chunk_frames(2048);

        let output = stretch_interleaved(&input, &config).expect("stretch should succeed");
        let ratio = output.len() as f64 / input.len() as f64;

        assert!(output.len() > input.len());
        assert!((ratio - 1.5).abs() < 0.2, "unexpected ratio: {}", ratio);
    }

    #[test]
    fn stretch_audio_data_returns_float_audio_data() {
        let input = sine_wave(24_000, 48_000, 220.0);
        let pcm = vec_f32_to_i16(input);
        let bytes: Vec<u8> = pcm.iter().flat_map(|sample| sample.to_le_bytes()).collect();
        let audio = AudioData::new(
            16,
            1,
            48_000,
            bytes,
            EncodingFlag::PCMSigned,
            Endianness::LittleEndian,
        );

        let config = recommended_config_for_audio(&audio)
            .with_time_ratio(1.25)
            .with_chunk_frames(2048);

        let stretched = stretch_audio_data(&audio, &config).expect("AudioData stretch should work");

        assert_eq!(stretched.bits_per_sample(), 32);
        assert_eq!(stretched.channel_count(), 1);
        assert_eq!(stretched.sampling_rate(), 48_000);
        assert_eq!(stretched.audio_format(), EncodingFlag::PCMFloat);
        assert_eq!(stretched.endianness(), Endianness::LittleEndian);
        assert!(!stretched.data().is_empty());
    }

    #[test]
    fn stretch_audio_data_to_pcm_returns_f32_channels() {
        let input = sine_wave(24_000, 48_000, 330.0);
        let pcm = vec_f32_to_i16(input);
        let bytes: Vec<u8> = pcm.iter().flat_map(|sample| sample.to_le_bytes()).collect();
        let audio = AudioData::new(
            16,
            1,
            48_000,
            bytes,
            EncodingFlag::PCMSigned,
            Endianness::LittleEndian,
        );

        let config = recommended_config_for_audio(&audio)
            .with_time_ratio(0.8)
            .with_chunk_frames(2048);

        let stretched = stretch_audio_data_to_pcm(&audio, &config)
            .expect("AudioData-to-PCM stretch should work");

        match stretched {
            PcmData::F32(channels) => {
                assert_eq!(channels.len(), 1);
                assert!(!channels[0].is_empty());
                assert!(channels[0].len() < pcm.len());
            }
            _ => panic!("expected f32 output"),
        }
    }

    #[test]
    fn stretch_audio_data_preserves_16_bit_pcm_format() {
        let input = sine_wave(24_000, 48_000, 440.0);
        let pcm = vec_f32_to_i16(input);
        let bytes: Vec<u8> = pcm.iter().flat_map(|sample| sample.to_le_bytes()).collect();
        let audio = AudioData::new(
            16,
            1,
            48_000,
            bytes,
            EncodingFlag::PCMSigned,
            Endianness::LittleEndian,
        );

        let config = recommended_config_for_audio(&audio)
            .with_time_ratio(1.2)
            .with_chunk_frames(2048);

        let stretched = stretch_audio_data_preserve_format(&audio, &config)
            .expect("preserve-format stretch should work");

        assert_eq!(stretched.bits_per_sample(), 16);
        assert_eq!(stretched.audio_format(), EncodingFlag::PCMSigned);
        assert_eq!(stretched.endianness(), Endianness::LittleEndian);
        assert_eq!(stretched.data().len() % 2, 0);
        assert!(stretched.data().len() > audio.data().len());
    }

    #[test]
    fn stretch_audio_data_preserves_24_bit_pcm_format() {
        let input = sine_wave(24_000, 48_000, 550.0);
        let mut bytes = Vec::with_capacity(input.len() * 3);
        for sample in input {
            let value = if sample >= 0.0 {
                (sample.clamp(-1.0, 1.0) * 8_388_607.0) as i32
            } else {
                (sample.clamp(-1.0, 1.0) * 8_388_608.0) as i32
            };
            let le = value.to_le_bytes();
            bytes.extend_from_slice(&le[..3]);
        }
        let audio = AudioData::new(
            24,
            1,
            48_000,
            bytes,
            EncodingFlag::PCMSigned,
            Endianness::LittleEndian,
        );

        let config = recommended_config_for_audio(&audio)
            .with_time_ratio(0.9)
            .with_chunk_frames(2048);

        let stretched = stretch_audio_data_preserve_format(&audio, &config)
            .expect("24-bit preserve-format stretch should work");

        assert_eq!(stretched.bits_per_sample(), 24);
        assert_eq!(stretched.audio_format(), EncodingFlag::PCMSigned);
        assert_eq!(stretched.endianness(), Endianness::LittleEndian);
        assert_eq!(stretched.data().len() % 3, 0);
        assert!(stretched.data().len() < audio.data().len());
    }

    #[test]
    fn stretch_audio_data_rejects_non_pcm_audio() {
        let audio = AudioData::new(
            16,
            1,
            48_000,
            vec![0, 0],
            EncodingFlag::Opus,
            Endianness::LittleEndian,
        );
        let config = recommended_config_for_audio(&audio);

        let error = stretch_audio_data(&audio, &config).expect_err("non-PCM audio should fail");
        assert!(matches!(error, StretchError::UnsupportedAudioData(_)));
    }

    #[test]
    fn stretch_audio_data_rejects_big_endian_audio() {
        let audio = AudioData::new(
            16,
            1,
            48_000,
            vec![0, 0],
            EncodingFlag::PCMSigned,
            Endianness::BigEndian,
        );
        let config = recommended_config_for_audio(&audio);

        let error = stretch_audio_data_preserve_format(&audio, &config)
            .expect_err("big-endian audio should fail");
        assert_eq!(error, StretchError::UnsupportedEndianness);
    }

    #[test]
    fn stretch_audio_data_rejects_truncated_frames() {
        let audio = AudioData::new(
            16,
            2,
            48_000,
            vec![0, 0, 0],
            EncodingFlag::PCMSigned,
            Endianness::LittleEndian,
        );
        let config = recommended_config_for_audio(&audio);

        let error = stretch_audio_data(&audio, &config).expect_err("truncated audio should fail");
        assert_eq!(
            error,
            StretchError::InvalidAudioByteLength {
                byte_len: 3,
                bytes_per_frame: 4,
            }
        );
    }

    #[test]
    fn stretch_interleaved_rejects_invalid_channel_config() {
        let config = OfflineStretchConfig::recommended_for_music(48_000, 1)
            .with_time_ratio(1.1)
            .with_chunk_frames(1024);
        let invalid = OfflineStretchConfig {
            channels: 0,
            ..config
        };

        let error = stretch_interleaved(&[0.0, 0.0], &invalid)
            .expect_err("zero-channel config should fail");
        assert_eq!(error, StretchError::InvalidChannelCount(0));
    }
}
