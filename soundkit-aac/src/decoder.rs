use crate::{MAX_AAC_BUFFERED_BYTES, MAX_INPUT_CHUNK_BYTES};
#[cfg(feature = "fdk")]
use fdk_aac::dec::{Decoder as FdkDecoder, DecoderError, Transport};
use soundkit::audio_packet::Decoder;
#[cfg(feature = "owned-lc")]
use soundkit_aac_lc::{AacLcDecoder, AacLcError};
use tracing::{debug, trace};

#[cfg(feature = "fdk")]
const MAX_FDK_PCM_SAMPLES: usize = 16_384;
const COMPACT_THRESHOLD: usize = 16 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AacDecoderBackend {
    Pending,
    SoundKitAacLc,
    FdkAac,
}

pub struct AacDecoder {
    backend: DecodeBackend,
    allow_fdk_fallback: bool,
    input_buffer: Vec<u8>,
    buffer_start: usize,
    sample_rate: Option<u32>,
    channels: Option<u8>,
    decoded_frames: u64,
    #[cfg(feature = "fdk")]
    fdk_pcm: Vec<i16>,
    #[cfg(feature = "fdk")]
    fdk_pending_samples: usize,
}

enum DecodeBackend {
    Pending,
    #[cfg(feature = "owned-lc")]
    SoundKit(OwnedDecoder),
    #[cfg(feature = "fdk")]
    Fdk(FdkDecoder),
}

#[cfg(feature = "owned-lc")]
struct OwnedDecoder {
    decoder: AacLcDecoder,
    config: StreamConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StreamConfig {
    object_type: u8,
    sample_rate: u32,
    channels: u8,
}

#[derive(Debug, Clone, Copy)]
struct AdtsFrame {
    #[cfg(feature = "owned-lc")]
    raw_start: usize,
    #[cfg(feature = "owned-lc")]
    frame_end: usize,
    config: StreamConfig,
    raw_data_blocks: u8,
}

#[cfg(feature = "owned-lc")]
enum OwnedDecodeResult {
    Complete(usize),
    RetryWithFdk,
}

trait OutputSample: Copy {
    #[cfg(feature = "owned-lc")]
    fn write_owned(channels: &[Vec<f32>], frames: usize, output: &mut [Self]);
    #[cfg(feature = "fdk")]
    fn write_fdk(input: &[i16], output: &mut [Self]);
}

impl OutputSample for i16 {
    #[cfg(feature = "owned-lc")]
    fn write_owned(channels: &[Vec<f32>], frames: usize, output: &mut [Self]) {
        write_owned_i16(channels, frames, output);
    }

    #[cfg(feature = "fdk")]
    fn write_fdk(input: &[i16], output: &mut [Self]) {
        output.copy_from_slice(input);
    }
}

impl OutputSample for f32 {
    #[cfg(feature = "owned-lc")]
    fn write_owned(channels: &[Vec<f32>], frames: usize, output: &mut [Self]) {
        let mut written = 0;
        for frame in 0..frames {
            for channel in channels {
                output[written] = channel[frame];
                written += 1;
            }
        }
    }

    #[cfg(feature = "fdk")]
    fn write_fdk(input: &[i16], output: &mut [Self]) {
        for (output, input) in output.iter_mut().zip(input) {
            *output = f32::from(*input) / 32_768.0;
        }
    }
}

impl AacDecoder {
    pub fn new() -> Self {
        Self::pending(true)
    }

    #[cfg(feature = "owned-lc")]
    pub fn new_soundkit_aac_lc() -> Self {
        Self::pending(false)
    }

    #[cfg(feature = "fdk")]
    pub fn new_fdk() -> Self {
        let mut decoder = Self::pending(true);
        decoder
            .activate_fdk("FDK backend was explicitly requested")
            .expect("FDK feature is enabled");
        decoder
    }

    fn pending(allow_fdk_fallback: bool) -> Self {
        Self {
            backend: DecodeBackend::Pending,
            allow_fdk_fallback,
            input_buffer: Vec::with_capacity(16 * 1024),
            buffer_start: 0,
            sample_rate: None,
            channels: None,
            decoded_frames: 0,
            #[cfg(feature = "fdk")]
            fdk_pcm: Vec::new(),
            #[cfg(feature = "fdk")]
            fdk_pending_samples: 0,
        }
    }

    pub fn init(&mut self) -> Result<(), String> {
        Ok(())
    }

    pub fn sample_rate(&self) -> Option<u32> {
        self.sample_rate
    }

    pub fn channels(&self) -> Option<u8> {
        self.channels
    }

    pub fn backend(&self) -> AacDecoderBackend {
        match self.backend {
            DecodeBackend::Pending => AacDecoderBackend::Pending,
            #[cfg(feature = "owned-lc")]
            DecodeBackend::SoundKit(_) => AacDecoderBackend::SoundKitAacLc,
            #[cfg(feature = "fdk")]
            DecodeBackend::Fdk(_) => AacDecoderBackend::FdkAac,
        }
    }

    /// Select a decoder from container-provided MPEG-4 AudioSpecificConfig.
    /// ADTS users do not need to call this; the first complete ADTS header is
    /// enough to select the backend.
    pub fn set_audio_specific_config(&mut self, data: &[u8]) -> Result<(), String> {
        if data.is_empty() || self.backend() == AacDecoderBackend::FdkAac {
            return Ok(());
        }
        if self.decoded_frames != 0 || self.buffered_len() != 0 {
            return Err("AAC decoder configuration arrived after stream data".to_string());
        }

        #[cfg(feature = "owned-lc")]
        {
            match AacLcDecoder::from_audio_specific_config(data) {
                Ok(decoder) => {
                    let info = decoder.frame_info();
                    let channels = u8::try_from(info.channels)
                        .map_err(|_| "AAC-LC channel count exceeds u8".to_string())?;
                    self.sample_rate = Some(info.sample_rate);
                    self.channels = Some(channels);
                    self.backend = DecodeBackend::SoundKit(OwnedDecoder {
                        decoder,
                        config: StreamConfig {
                            object_type: 2,
                            sample_rate: info.sample_rate,
                            channels,
                        },
                    });
                    debug!(
                        sample_rate_hz = info.sample_rate,
                        channels, "selected SoundKit AAC-LC decoder from AudioSpecificConfig"
                    );
                    return Ok(());
                }
                Err(error) if self.allow_fdk_fallback => {
                    return self.activate_fdk(&format!(
                        "AudioSpecificConfig is outside the owned AAC-LC profile: {error}"
                    ));
                }
                Err(error) => return Err(error.to_string()),
            }
        }

        #[cfg(not(feature = "owned-lc"))]
        self.activate_fdk("owned AAC-LC support is disabled")
    }

    fn decode_samples<T: OutputSample>(
        &mut self,
        input: &[u8],
        output: &mut [T],
    ) -> Result<usize, String> {
        self.append_input(input)?;
        if output.is_empty() {
            return Ok(0);
        }

        loop {
            if self.backend() == AacDecoderBackend::Pending && !self.select_backend_from_adts()? {
                return Ok(0);
            }

            match self.backend() {
                AacDecoderBackend::Pending => return Ok(0),
                AacDecoderBackend::SoundKitAacLc => {
                    #[cfg(feature = "owned-lc")]
                    match self.decode_owned(output)? {
                        OwnedDecodeResult::Complete(written) => return Ok(written),
                        OwnedDecodeResult::RetryWithFdk => continue,
                    }
                    #[cfg(not(feature = "owned-lc"))]
                    unreachable!("owned backend cannot be selected without owned-lc");
                }
                AacDecoderBackend::FdkAac => {
                    #[cfg(feature = "fdk")]
                    return self.decode_fdk(output);
                    #[cfg(not(feature = "fdk"))]
                    unreachable!("FDK backend cannot be selected without fdk");
                }
            }
        }
    }

    fn append_input(&mut self, input: &[u8]) -> Result<(), String> {
        if input.len() > MAX_INPUT_CHUNK_BYTES {
            return Err(format!(
                "AAC input chunk exceeds the {MAX_INPUT_CHUNK_BYTES} byte streaming budget"
            ));
        }
        if input.is_empty() {
            return Ok(());
        }
        self.compact_buffer(false);
        if self.buffered_len().saturating_add(input.len()) > MAX_AAC_BUFFERED_BYTES {
            return Err(format!(
                "AAC decoder buffer exceeds the {MAX_AAC_BUFFERED_BYTES} byte streaming budget"
            ));
        }
        self.input_buffer.extend_from_slice(input);
        Ok(())
    }

    fn select_backend_from_adts(&mut self) -> Result<bool, String> {
        let Some(frame) = self.next_adts_frame()? else {
            return Ok(false);
        };

        if frame_supports_owned(frame) {
            #[cfg(feature = "owned-lc")]
            {
                let asc = audio_specific_config(frame.config)?;
                let decoder = AacLcDecoder::from_audio_specific_config(&asc)
                    .map_err(|error| format!("initialize SoundKit AAC-LC decoder: {error}"))?;
                self.sample_rate = Some(frame.config.sample_rate);
                self.channels = Some(frame.config.channels);
                self.backend = DecodeBackend::SoundKit(OwnedDecoder {
                    decoder,
                    config: frame.config,
                });
                debug!(
                    sample_rate_hz = frame.config.sample_rate,
                    channels = frame.config.channels,
                    "selected SoundKit AAC-LC decoder from ADTS"
                );
                return Ok(true);
            }
        }

        self.activate_fdk(&format!(
            "ADTS profile object_type={} sample_rate={} channels={} raw_blocks={} is outside the owned AAC-LC profile",
            frame.config.object_type,
            frame.config.sample_rate,
            frame.config.channels,
            frame.raw_data_blocks,
        ))?;
        Ok(true)
    }

    #[cfg(feature = "owned-lc")]
    fn decode_owned<T: OutputSample>(
        &mut self,
        output: &mut [T],
    ) -> Result<OwnedDecodeResult, String> {
        let mut written = 0usize;

        loop {
            let Some(frame) = self.next_adts_frame()? else {
                break;
            };
            if !frame_supports_owned(frame) {
                if self.allow_fdk_fallback && self.decoded_frames == 0 && written == 0 {
                    self.activate_fdk("ADTS stream changed to an unsupported AAC profile")?;
                    return Ok(OwnedDecodeResult::RetryWithFdk);
                }
                return Err("AAC format changed outside the owned AAC-LC profile".to_string());
            }

            let configured = match &self.backend {
                DecodeBackend::SoundKit(state) => state.config,
                _ => unreachable!("owned decode requires the SoundKit backend"),
            };
            if configured != frame.config {
                return Err(format!(
                    "AAC-LC format changed during the stream: {:?} -> {:?}",
                    configured, frame.config
                ));
            }

            let frame_samples = usize::from(frame.config.channels) * 1024;
            let remaining = output.len().saturating_sub(written);
            if remaining < frame_samples {
                if written == 0 {
                    return Err(format!(
                        "Output buffer too small for decoded AAC-LC frame (needed {frame_samples}, had {remaining})"
                    ));
                }
                break;
            }

            enum FrameResult {
                Decoded { frames: usize, channels: usize },
                Failed { message: String, can_fallback: bool },
            }

            let result = {
                let state = match &mut self.backend {
                    DecodeBackend::SoundKit(state) => state,
                    _ => unreachable!("owned decode requires the SoundKit backend"),
                };
                match state
                    .decoder
                    .decode_access_unit(&self.input_buffer[frame.raw_start..frame.frame_end])
                {
                    Ok(decoded) => {
                        let frames = decoded.frames();
                        let channels = decoded.channels().len();
                        let sample_count = frames * channels;
                        T::write_owned(
                            decoded.channels(),
                            frames,
                            &mut output[written..written + sample_count],
                        );
                        written += sample_count;
                        FrameResult::Decoded { frames, channels }
                    }
                    Err(error) => FrameResult::Failed {
                        can_fallback: owned_error_supports_fdk_fallback(&error),
                        message: error.to_string(),
                    },
                }
            };

            match result {
                FrameResult::Decoded { frames, channels } => {
                    if frames != 1024 || channels != usize::from(frame.config.channels) {
                        return Err(format!(
                            "SoundKit AAC-LC returned {frames} frames/{channels} channels for expected 1024/{}",
                            frame.config.channels
                        ));
                    }
                    let first_frame = self.decoded_frames == 0;
                    self.decoded_frames += 1;
                    self.sample_rate = Some(frame.config.sample_rate);
                    self.channels = Some(frame.config.channels);
                    self.buffer_start = frame.frame_end;
                    if first_frame {
                        debug!(
                            sample_rate_hz = frame.config.sample_rate,
                            channels = frame.config.channels,
                            frame_samples,
                            "decoded AAC frame with SoundKit AAC-LC"
                        );
                    } else {
                        trace!(
                            sample_rate_hz = frame.config.sample_rate,
                            channels = frame.config.channels,
                            frame_samples,
                            "decoded AAC frame with SoundKit AAC-LC"
                        );
                    }
                }
                FrameResult::Failed {
                    message,
                    can_fallback,
                } => {
                    if can_fallback
                        && self.allow_fdk_fallback
                        && self.decoded_frames == 0
                        && written == 0
                    {
                        self.activate_fdk(&format!(
                            "owned AAC-LC decoder rejected the stream: {message}"
                        ))?;
                        return Ok(OwnedDecodeResult::RetryWithFdk);
                    }
                    return Err(format!(
                        "SoundKit AAC-LC decode failed at frame {}: {message}",
                        self.decoded_frames
                    ));
                }
            }
        }

        self.clear_buffer_if_empty();
        Ok(OwnedDecodeResult::Complete(written))
    }

    #[cfg(feature = "fdk")]
    fn decode_fdk<T: OutputSample>(&mut self, output: &mut [T]) -> Result<usize, String> {
        let mut written = 0usize;

        if self.fdk_pending_samples != 0 {
            if output.len() < self.fdk_pending_samples {
                return Err(format!(
                    "Output buffer too small for decoded FDK AAC frame (needed {}, had {})",
                    self.fdk_pending_samples,
                    output.len()
                ));
            }
            T::write_fdk(
                &self.fdk_pcm[..self.fdk_pending_samples],
                &mut output[..self.fdk_pending_samples],
            );
            written = self.fdk_pending_samples;
            self.fdk_pending_samples = 0;
        }

        loop {
            if written == output.len() {
                break;
            }

            let consumed = if self.buffered_len() == 0 {
                0
            } else {
                let bytes = &self.input_buffer[self.buffer_start..];
                let decoder = match &mut self.backend {
                    DecodeBackend::Fdk(decoder) => decoder,
                    _ => unreachable!("FDK decode requires the FDK backend"),
                };
                decoder
                    .fill(bytes)
                    .map_err(|error| format!("Error filling FDK AAC decoder: {error}"))?
            };
            self.buffer_start += consumed;

            enum FdkFrameResult {
                Decoded {
                    sample_rate: u32,
                    channels: u8,
                    frame_samples: usize,
                },
                NeedMore,
                Failed(String),
            }

            let decoded = {
                let decoder = match &mut self.backend {
                    DecodeBackend::Fdk(decoder) => decoder,
                    _ => unreachable!("FDK decode requires the FDK backend"),
                };
                match decoder.decode_frame(&mut self.fdk_pcm) {
                    Ok(()) => {
                        let info = decoder.stream_info();
                        let channels = u8::try_from(info.numChannels).unwrap_or_default();
                        let frame_size = usize::try_from(info.frameSize).unwrap_or_default();
                        FdkFrameResult::Decoded {
                            sample_rate: u32::try_from(info.sampleRate).unwrap_or_default(),
                            channels,
                            frame_samples: usize::from(channels) * frame_size,
                        }
                    }
                    Err(error) if error == DecoderError::NOT_ENOUGH_BITS => {
                        FdkFrameResult::NeedMore
                    }
                    Err(error) => FdkFrameResult::Failed(error.to_string()),
                }
            };

            match decoded {
                FdkFrameResult::Decoded {
                    sample_rate,
                    channels,
                    frame_samples,
                } => {
                    if sample_rate == 0 || channels == 0 || frame_samples == 0 {
                        return Err("FDK AAC returned invalid stream metadata".to_string());
                    }
                    if frame_samples > self.fdk_pcm.len() {
                        return Err(format!(
                            "FDK AAC frame exceeds internal PCM capacity: {frame_samples}"
                        ));
                    }
                    let first_frame = self.decoded_frames == 0;
                    self.decoded_frames += 1;
                    self.sample_rate = Some(sample_rate);
                    self.channels = Some(channels);
                    if first_frame {
                        debug!(
                            sample_rate_hz = sample_rate,
                            channels, frame_samples, "decoded AAC frame with FDK fallback"
                        );
                    } else {
                        trace!(
                            sample_rate_hz = sample_rate,
                            channels,
                            frame_samples,
                            "decoded AAC frame with FDK fallback"
                        );
                    }
                    let remaining = output.len() - written;
                    if remaining < frame_samples {
                        self.fdk_pending_samples = frame_samples;
                        if written == 0 {
                            return Err(format!(
                                "Output buffer too small for decoded FDK AAC frame (needed {frame_samples}, had {remaining})"
                            ));
                        }
                        break;
                    }
                    T::write_fdk(
                        &self.fdk_pcm[..frame_samples],
                        &mut output[written..written + frame_samples],
                    );
                    written += frame_samples;
                }
                FdkFrameResult::NeedMore => {
                    if consumed > 0 && self.buffered_len() > 0 {
                        continue;
                    }
                    break;
                }
                FdkFrameResult::Failed(error) => {
                    return Err(format!("FDK AAC decoding error: {error}"));
                }
            }
        }

        self.clear_buffer_if_empty();
        Ok(written)
    }

    fn next_adts_frame(&mut self) -> Result<Option<AdtsFrame>, String> {
        let remaining = &self.input_buffer[self.buffer_start..];
        let Some(sync) = remaining
            .windows(2)
            .position(|bytes| bytes[0] == 0xff && (bytes[1] & 0xf6) == 0xf0)
        else {
            let keep = usize::from(remaining.last() == Some(&0xff));
            self.buffer_start = self.input_buffer.len().saturating_sub(keep);
            self.compact_buffer(false);
            return Ok(None);
        };
        self.buffer_start += sync;

        if self.buffered_len() < 7 {
            return Ok(None);
        }
        let base = self.buffer_start;
        let protection_absent = self.input_buffer[base + 1] & 1 != 0;
        let header_len = if protection_absent { 7 } else { 9 };
        let frame_len = (((self.input_buffer[base + 3] & 3) as usize) << 11)
            | ((self.input_buffer[base + 4] as usize) << 3)
            | ((self.input_buffer[base + 5] as usize) >> 5);
        if frame_len <= header_len || frame_len > 8191 {
            return Err(format!(
                "invalid ADTS frame length {frame_len} at stream offset {base}"
            ));
        }
        if self.buffered_len() < frame_len {
            return Ok(None);
        }

        let object_type = ((self.input_buffer[base + 2] & 0xc0) >> 6) + 1;
        let sample_rate_index = (self.input_buffer[base + 2] & 0x3c) >> 2;
        let sample_rate = adts_sample_rate(sample_rate_index)
            .ok_or_else(|| format!("unsupported ADTS sample-rate index {sample_rate_index}"))?;
        let channels =
            ((self.input_buffer[base + 2] & 1) << 2) | ((self.input_buffer[base + 3] & 0xc0) >> 6);
        let raw_data_blocks = (self.input_buffer[base + 6] & 3) + 1;

        Ok(Some(AdtsFrame {
            #[cfg(feature = "owned-lc")]
            raw_start: base + header_len,
            #[cfg(feature = "owned-lc")]
            frame_end: base + frame_len,
            config: StreamConfig {
                object_type,
                sample_rate,
                channels,
            },
            raw_data_blocks,
        }))
    }

    fn buffered_len(&self) -> usize {
        self.input_buffer.len().saturating_sub(self.buffer_start)
    }

    fn compact_buffer(&mut self, force: bool) {
        if self.buffer_start == 0 {
            return;
        }
        if self.buffer_start == self.input_buffer.len() {
            self.input_buffer.clear();
            self.buffer_start = 0;
            return;
        }
        if force
            || (self.buffer_start >= COMPACT_THRESHOLD
                && self.buffer_start.saturating_mul(2) >= self.input_buffer.len())
        {
            let remaining = self.buffered_len();
            self.input_buffer.copy_within(self.buffer_start.., 0);
            self.input_buffer.truncate(remaining);
            self.buffer_start = 0;
        }
    }

    fn clear_buffer_if_empty(&mut self) {
        if self.buffer_start == self.input_buffer.len() {
            self.input_buffer.clear();
            self.buffer_start = 0;
        }
    }

    #[cfg(feature = "fdk")]
    fn activate_fdk(&mut self, reason: &str) -> Result<(), String> {
        if !self.allow_fdk_fallback && self.backend() != AacDecoderBackend::FdkAac {
            return Err(format!(
                "SoundKit AAC-LC decoder cannot handle stream: {reason}"
            ));
        }
        if self.backend() != AacDecoderBackend::FdkAac {
            self.backend = DecodeBackend::Fdk(FdkDecoder::new(Transport::Adts));
            self.fdk_pcm.resize(MAX_FDK_PCM_SAMPLES, 0);
            self.fdk_pending_samples = 0;
            debug!(reason, "selected FDK AAC fallback decoder");
        }
        Ok(())
    }

    #[cfg(not(feature = "fdk"))]
    fn activate_fdk(&mut self, reason: &str) -> Result<(), String> {
        Err(format!(
            "AAC stream requires the disabled FDK fallback: {reason}"
        ))
    }
}

impl Default for AacDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Decoder for AacDecoder {
    fn decode_i16(
        &mut self,
        input: &[u8],
        output: &mut [i16],
        _fec: bool,
    ) -> Result<usize, String> {
        self.decode_samples(input, output)
    }

    fn decode_i32(
        &mut self,
        _input: &[u8],
        _output: &mut [i32],
        _fec: bool,
    ) -> Result<usize, String> {
        Err("Not implemented.".to_string())
    }

    fn decode_f32(
        &mut self,
        input: &[u8],
        output: &mut [f32],
        _fec: bool,
    ) -> Result<usize, String> {
        self.decode_samples(input, output)
    }
}

#[cfg(feature = "owned-lc")]
fn write_owned_i16(channels: &[Vec<f32>], frames: usize, output: &mut [i16]) {
    debug_assert_eq!(output.len(), channels.len() * frames);

    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        match channels {
            [mono] => unsafe {
                write_mono_i16_avx2(&mono[..frames], output);
                return;
            },
            [left, right] => unsafe {
                write_stereo_i16_avx2(&left[..frames], &right[..frames], output);
                return;
            },
            _ => {}
        }
    }

    let mut written = 0;
    for frame in 0..frames {
        for channel in channels {
            output[written] = float_to_i16(channel[frame]);
            written += 1;
        }
    }
}

#[cfg(feature = "owned-lc")]
#[inline]
fn float_to_i16(sample: f32) -> i16 {
    let sample = if !sample.is_nan() {
        sample.clamp(-1.0, 1.0)
    } else {
        0.0
    };
    ((sample * 32_768.0).round_ties_even() as i32).clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

#[cfg(all(feature = "owned-lc", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn convert_f32x8_to_i32_avx2(
    value: std::arch::x86_64::__m256,
) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;

    let ordered = _mm256_cmp_ps(value, value, _CMP_ORD_Q);
    let value = _mm256_and_ps(value, ordered);
    let value = _mm256_max_ps(
        _mm256_set1_ps(-1.0),
        _mm256_min_ps(value, _mm256_set1_ps(1.0)),
    );
    let rounded = _mm256_round_ps(
        _mm256_mul_ps(value, _mm256_set1_ps(32_768.0)),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
    );
    _mm256_cvtps_epi32(rounded)
}

#[cfg(all(feature = "owned-lc", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn pack_i32x8_to_i16_avx2(
    first: std::arch::x86_64::__m256i,
    second: std::arch::x86_64::__m256i,
) -> std::arch::x86_64::__m256i {
    let packed;
    std::arch::asm!(
        "vpackssdw {packed}, {first}, {second}",
        packed = lateout(ymm_reg) packed,
        first = in(ymm_reg) first,
        second = in(ymm_reg) second,
        options(pure, nomem, nostack, preserves_flags),
    );
    packed
}

#[cfg(all(feature = "owned-lc", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn write_stereo_i16_avx2(left: &[f32], right: &[f32], output: &mut [i16]) {
    use std::arch::x86_64::*;

    let mut frame = 0;
    while frame + 8 <= left.len() {
        let left_i32 = convert_f32x8_to_i32_avx2(_mm256_loadu_ps(left.as_ptr().add(frame)));
        let right_i32 = convert_f32x8_to_i32_avx2(_mm256_loadu_ps(right.as_ptr().add(frame)));
        let interleaved_lo = _mm256_unpacklo_epi32(left_i32, right_i32);
        let interleaved_hi = _mm256_unpackhi_epi32(left_i32, right_i32);
        let packed = pack_i32x8_to_i16_avx2(interleaved_lo, interleaved_hi);
        _mm256_storeu_si256(output.as_mut_ptr().add(frame * 2).cast(), packed);
        frame += 8;
    }
    for index in frame..left.len() {
        output[index * 2] = float_to_i16(left[index]);
        output[index * 2 + 1] = float_to_i16(right[index]);
    }
}

#[cfg(all(feature = "owned-lc", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn write_mono_i16_avx2(input: &[f32], output: &mut [i16]) {
    use std::arch::x86_64::*;

    let mut frame = 0;
    while frame + 16 <= input.len() {
        let first = convert_f32x8_to_i32_avx2(_mm256_loadu_ps(input.as_ptr().add(frame)));
        let second = convert_f32x8_to_i32_avx2(_mm256_loadu_ps(input.as_ptr().add(frame + 8)));
        let packed = pack_i32x8_to_i16_avx2(first, second);
        let ordered = _mm256_permute4x64_epi64(packed, 0xd8);
        _mm256_storeu_si256(output.as_mut_ptr().add(frame).cast(), ordered);
        frame += 16;
    }
    for index in frame..input.len() {
        output[index] = float_to_i16(input[index]);
    }
}

fn frame_supports_owned(frame: AdtsFrame) -> bool {
    frame.config.object_type == 2
        && matches!(frame.config.channels, 1 | 2)
        && frame.raw_data_blocks == 1
}

#[cfg(feature = "owned-lc")]
fn audio_specific_config(config: StreamConfig) -> Result<[u8; 2], String> {
    let sample_rate_index = sample_rate_index(config.sample_rate)
        .ok_or_else(|| format!("unsupported AAC-LC sample rate {}", config.sample_rate))?;
    Ok([
        (config.object_type << 3) | (sample_rate_index >> 1),
        ((sample_rate_index & 1) << 7) | (config.channels << 3),
    ])
}

#[cfg(feature = "owned-lc")]
fn owned_error_supports_fdk_fallback(error: &AacLcError) -> bool {
    matches!(
        error,
        AacLcError::UnsupportedAudioObjectType(_)
            | AacLcError::UnsupportedSamplingFrequencyIndex(_)
            | AacLcError::UnsupportedChannelConfig(_)
            | AacLcError::UnsupportedFeature(_)
            | AacLcError::NotImplemented(_)
    )
}

fn adts_sample_rate(index: u8) -> Option<u32> {
    const RATES: [u32; 13] = [
        96_000, 88_200, 64_000, 48_000, 44_100, 32_000, 24_000, 22_050, 16_000, 12_000, 11_025,
        8_000, 7_350,
    ];
    RATES.get(index as usize).copied()
}

#[cfg(feature = "owned-lc")]
fn sample_rate_index(sample_rate: u32) -> Option<u8> {
    const RATES: [u32; 13] = [
        96_000, 88_200, 64_000, 48_000, 44_100, 32_000, 24_000, 22_050, 16_000, 12_000, 11_025,
        8_000, 7_350,
    ];
    RATES
        .iter()
        .position(|rate| *rate == sample_rate)
        .and_then(|index| u8::try_from(index).ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_split_sync_word_until_more_input_arrives() {
        let mut decoder = AacDecoder::new();
        let mut output = [0i16; 2048];

        assert_eq!(
            decoder
                .decode_i16(&[0x00, 0xff], &mut output, false)
                .unwrap(),
            0
        );
        assert_eq!(decoder.buffered_len(), 1);
    }

    #[test]
    #[cfg(feature = "owned-lc")]
    fn rejects_too_small_owned_output_without_consuming_frame() {
        let fixture =
            include_bytes!("../../golden/aac/A_Tusk_is_used_to_make_costly_gifts_encoded.aac");
        let mut decoder = AacDecoder::new_soundkit_aac_lc();
        let mut small = [0i16; 1024];
        let error = decoder.decode_i16(fixture, &mut small, false).unwrap_err();

        assert!(error.contains("Output buffer too small"));
        assert_eq!(decoder.decoded_frames, 0);
        assert!(decoder.buffered_len() > 0);
    }

    #[test]
    #[cfg(all(feature = "owned-lc", feature = "fdk"))]
    fn implicit_he_aac_config_uses_fdk_unless_owned_is_forced() {
        let implicit_he_aac = [0x15, 0x10, 0x56, 0xe5, 0xb8];

        let mut automatic = AacDecoder::new();
        automatic
            .set_audio_specific_config(&implicit_he_aac)
            .unwrap();
        assert_eq!(automatic.backend(), AacDecoderBackend::FdkAac);

        let mut forced_owned = AacDecoder::new_soundkit_aac_lc();
        let error = forced_owned
            .set_audio_specific_config(&implicit_he_aac)
            .unwrap_err();
        assert!(error.contains("SBR/HE-AAC"));
        assert_eq!(forced_owned.backend(), AacDecoderBackend::Pending);
    }

    #[test]
    #[cfg(all(feature = "owned-lc", not(feature = "fdk")))]
    fn implicit_he_aac_config_reports_disabled_fdk_fallback() {
        let mut decoder = AacDecoder::new();
        let error = decoder
            .set_audio_specific_config(&[0x15, 0x10, 0x56, 0xe5, 0xb8])
            .unwrap_err();

        assert!(error.contains("disabled FDK fallback"));
        assert!(error.contains("SBR/HE-AAC"));
        assert_eq!(decoder.backend(), AacDecoderBackend::Pending);
    }

    #[test]
    #[cfg(feature = "fdk")]
    fn retains_fdk_frame_when_output_is_too_small() {
        let fixture =
            include_bytes!("../../golden/aac/A_Tusk_is_used_to_make_costly_gifts_encoded.aac");
        let mut decoder = AacDecoder::new_fdk();
        let mut small = [0i16; 1024];
        let error = decoder.decode_i16(fixture, &mut small, false).unwrap_err();
        assert!(error.contains("Output buffer too small"));

        let mut recovered = [0i16; 2048];
        assert_eq!(
            decoder.decode_i16(&[], &mut recovered, false).unwrap(),
            recovered.len()
        );

        let mut control = AacDecoder::new_fdk();
        let mut expected = [0i16; 2048];
        assert_eq!(
            control.decode_i16(fixture, &mut expected, false).unwrap(),
            expected.len()
        );
        assert_eq!(recovered, expected);
    }

    #[test]
    #[cfg(all(feature = "owned-lc", target_arch = "x86_64"))]
    fn avx2_i16_conversion_matches_scalar_and_preserves_channel_order() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }

        let values = [
            -f32::INFINITY,
            -1.25,
            -1.0,
            -0.75,
            -16_383.5 / 32_768.0,
            -0.0,
            0.0,
            16_383.5 / 32_768.0,
            0.75,
            1.0,
            1.25,
            f32::INFINITY,
            f32::NAN,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            0.123_456_7,
            -0.987_654_3,
        ];
        let left: Vec<f32> = (0..67).map(|index| values[index % values.len()]).collect();
        let right: Vec<f32> = (0..67)
            .map(|index| values[(index * 7 + 3) % values.len()])
            .collect();

        let mut mono = vec![0i16; left.len()];
        unsafe { write_mono_i16_avx2(&left, &mut mono) };
        let expected_mono: Vec<i16> = left.iter().copied().map(float_to_i16).collect();
        assert_eq!(mono, expected_mono);

        let mut stereo = vec![0i16; left.len() * 2];
        unsafe { write_stereo_i16_avx2(&left, &right, &mut stereo) };
        let expected_stereo: Vec<i16> = left
            .iter()
            .zip(&right)
            .flat_map(|(&left, &right)| [float_to_i16(left), float_to_i16(right)])
            .collect();
        assert_eq!(stereo, expected_stereo);
    }
}
