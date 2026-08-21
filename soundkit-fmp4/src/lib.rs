//! Zero-transcode Soundkit v2 audio to fragmented MP4 reboxing.
//!
//! Transport recovery is intentionally out of scope. Callers recover RaptorQ
//! objects and unwrap AEP1 before passing one contiguous, codec-homogeneous
//! Soundkit v2 stream to this crate.

use access_unit::{AccessUnit, PSI_STREAM_AUDIO_OPUS, PSI_STREAM_PRIVATE_DATA};
use boxer::fmp4::{
    box_fmp4_with_init_and_audio_config, opus_packet_info, AudioTrackConfig, Config,
    OpusAudioConfig,
};
use bytes::Bytes;
use frame_header::EncodingFlag;
use soundkit::frame_stream::SoundKitFrameStream;
use std::error::Error;
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SoundKitFmp4Codec {
    Flac,
    Opus,
}

impl SoundKitFmp4Codec {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Flac => "flac",
            Self::Opus => "opus",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReboxOptions {
    pub sequence: u32,
    pub include_init: bool,
    /// Override the embedded Soundkit timeline with an outer publication
    /// timeline. Packet-to-packet spacing is still validated from Soundkit.
    pub start_pts_ms: Option<u64>,
}

impl Default for ReboxOptions {
    fn default() -> Self {
        Self {
            sequence: 0,
            include_init: true,
            start_pts_ms: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ReboxedFmp4 {
    pub codec: SoundKitFmp4Codec,
    pub sample_rate: u32,
    pub channels: u8,
    pub bits_per_sample: u8,
    pub first_pts_samples: u64,
    pub next_pts_samples: u64,
    pub frame_count: u64,
    pub packet_count: usize,
    pub duration_ms: u32,
    pub init: Option<Bytes>,
    pub media: Bytes,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReboxError(String);

impl ReboxError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for ReboxError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for ReboxError {}

/// Rebox one complete, codec-homogeneous Soundkit v2 byte stream as fMP4.
///
/// The codec payloads are copied into `mdat` unchanged. This function verifies
/// Soundkit framing and CRCs, rejects encrypted input, requires contiguous PTS,
/// and emits an initialization segment when requested.
pub fn rebox_soundkit_v2(input: &[u8], options: ReboxOptions) -> Result<ReboxedFmp4, ReboxError> {
    let mut parser = SoundKitFrameStream::default();
    let frames = parser.push(input).map_err(ReboxError::new)?;
    parser.finish().map_err(ReboxError::new)?;
    let first = frames
        .first()
        .ok_or_else(|| ReboxError::new("Soundkit v2 stream is empty"))?;
    if first.encrypted || first.header.is_encrypted() {
        return Err(ReboxError::new(
            "encrypted Soundkit v2 cannot be published without a playback key",
        ));
    }

    let codec = codec_from_encoding(first.header.encoding())?;
    let sample_rate = first.header.sample_rate();
    let channels = first.header.channels();
    let bits_per_sample = first.header.bits_per_sample();
    if sample_rate == 0 || channels == 0 {
        return Err(ReboxError::new("invalid Soundkit v2 audio geometry"));
    }
    if codec == SoundKitFmp4Codec::Opus && !matches!(channels, 1 | 2) {
        return Err(ReboxError::new(
            "Opus fMP4 supports one or two output channels",
        ));
    }

    let embedded_first_pts_samples = first.header.pts().unwrap_or(0);
    let first_pts_samples = options
        .start_pts_ms
        .map_or(embedded_first_pts_samples, |pts_ms| {
            millis_to_samples(pts_ms, sample_rate)
        });
    let mut expected_embedded_pts_samples = embedded_first_pts_samples;
    let mut expected_output_pts_samples = first_pts_samples;
    let mut total_frames = 0_u64;
    let mut access_units = Vec::with_capacity(frames.len());

    for (index, frame) in frames.iter().enumerate() {
        if frame.encrypted || frame.header.is_encrypted() {
            return Err(ReboxError::new(format!(
                "Soundkit v2 packet {index} is encrypted"
            )));
        }
        if codec_from_encoding(frame.header.encoding())? != codec
            || frame.header.sample_rate() != sample_rate
            || frame.header.channels() != channels
            || frame.header.bits_per_sample() != bits_per_sample
        {
            return Err(ReboxError::new(format!(
                "Soundkit v2 packet {index} changes codec or audio geometry"
            )));
        }

        let pts_samples = frame.header.pts().unwrap_or(expected_embedded_pts_samples);
        if pts_samples != expected_embedded_pts_samples {
            return Err(ReboxError::new(format!(
                "Soundkit v2 packet {index} PTS is {pts_samples}, expected {expected_embedded_pts_samples}"
            )));
        }
        let frame_count = frame.header.frame_count();
        if frame_count == 0 {
            return Err(ReboxError::new(format!(
                "Soundkit v2 packet {index} declares zero frames"
            )));
        }
        if codec == SoundKitFmp4Codec::Opus {
            validate_opus_packet(&frame.payload, frame_count, sample_rate, index)?;
        }

        let pts_ms = options.start_pts_ms.map_or_else(
            || samples_to_millis(pts_samples, sample_rate),
            |start_pts_ms| {
                start_pts_ms.saturating_add(samples_to_millis(
                    pts_samples.saturating_sub(embedded_first_pts_samples),
                    sample_rate,
                ))
            },
        );
        access_units.push(AccessUnit {
            key: true,
            pts: pts_ms,
            dts: pts_ms,
            data: Bytes::copy_from_slice(&frame.payload),
            stream_type: match codec {
                SoundKitFmp4Codec::Flac => PSI_STREAM_PRIVATE_DATA,
                SoundKitFmp4Codec::Opus => PSI_STREAM_AUDIO_OPUS,
            },
            id: frame.header.id().unwrap_or(index as u64),
        });
        expected_embedded_pts_samples = expected_embedded_pts_samples
            .checked_add(u64::from(frame_count))
            .ok_or_else(|| ReboxError::new("Soundkit v2 PTS overflow"))?;
        expected_output_pts_samples = expected_output_pts_samples
            .checked_add(u64::from(frame_count))
            .ok_or_else(|| ReboxError::new("Soundkit v2 output PTS overflow"))?;
        total_frames = total_frames
            .checked_add(u64::from(frame_count))
            .ok_or_else(|| ReboxError::new("Soundkit v2 frame-count overflow"))?;
    }

    let audio_config = match codec {
        SoundKitFmp4Codec::Flac => None,
        SoundKitFmp4Codec::Opus => Some(AudioTrackConfig::Opus(OpusAudioConfig {
            input_sample_rate: sample_rate,
            channel_count: u16::from(channels),
            pre_skip: 0,
            output_gain: 0,
        })),
    };
    let next_dts_ms = options.start_pts_ms.map_or_else(
        || samples_to_millis(expected_output_pts_samples, sample_rate),
        |start_pts_ms| start_pts_ms.saturating_add(samples_to_millis(total_frames, sample_rate)),
    );
    let packet_count = access_units.len();
    let boxed = box_fmp4_with_init_and_audio_config(
        options.sequence,
        Config {
            width: 0,
            height: 0,
            avcc: None,
        },
        Vec::new(),
        access_units,
        next_dts_ms,
        options.include_init,
        audio_config,
    );
    if boxed.data.is_empty() {
        return Err(ReboxError::new(
            "fMP4 boxer did not recognize the Soundkit codec payload",
        ));
    }
    if options.include_init && boxed.init.is_none() {
        return Err(ReboxError::new(
            "fMP4 boxer did not produce an initialization segment",
        ));
    }

    Ok(ReboxedFmp4 {
        codec,
        sample_rate,
        channels,
        bits_per_sample,
        first_pts_samples,
        next_pts_samples: expected_output_pts_samples,
        frame_count: total_frames,
        packet_count,
        duration_ms: boxed.duration,
        init: boxed.init,
        media: boxed.data,
    })
}

fn codec_from_encoding(encoding: &EncodingFlag) -> Result<SoundKitFmp4Codec, ReboxError> {
    match encoding {
        EncodingFlag::FLAC => Ok(SoundKitFmp4Codec::Flac),
        EncodingFlag::Opus => Ok(SoundKitFmp4Codec::Opus),
        other => Err(ReboxError::new(format!(
            "Soundkit v2 encoding {other:?} is not supported by the audio fMP4 reboxer"
        ))),
    }
}

fn validate_opus_packet(
    payload: &[u8],
    frame_count: u32,
    sample_rate: u32,
    index: usize,
) -> Result<(), ReboxError> {
    let info = opus_packet_info(payload).ok_or_else(|| {
        ReboxError::new(format!(
            "Soundkit v2 packet {index} is not a valid raw Opus packet"
        ))
    })?;
    if u64::from(frame_count).saturating_mul(48_000)
        != u64::from(info.duration_samples).saturating_mul(u64::from(sample_rate))
    {
        return Err(ReboxError::new(format!(
            "Soundkit v2 packet {index} Opus duration disagrees with its frame count"
        )));
    }
    Ok(())
}

fn samples_to_millis(samples: u64, sample_rate: u32) -> u64 {
    samples
        .saturating_mul(1_000)
        .saturating_add(u64::from(sample_rate) / 2)
        / u64::from(sample_rate)
}

fn millis_to_samples(milliseconds: u64, sample_rate: u32) -> u64 {
    milliseconds
        .saturating_mul(u64::from(sample_rate))
        .saturating_add(500)
        / 1_000
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_stream_is_rejected() {
        assert_eq!(
            rebox_soundkit_v2(&[], ReboxOptions::default())
                .unwrap_err()
                .to_string(),
            "Soundkit v2 stream is empty"
        );
    }
}
