//! Seekable MOV/MP4 video keyframe timeline decoding.
//!
//! A timeline lists every sync sample (keyframe) of one video track with its
//! presentation time. Callers that need pixels decode a capped, evenly spaced
//! subset so a long source does not force a full-film decode.

use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

use crate::{inspect_mp4_top_level_box, MediaSampleIndex, MediaTrackConfig, MediaTrackKind, Mp4MediaIndex};
use soundkit_video::{VideoCodec, VideoDecoder, VideoFrame};

/// The decode budget for one MOV/MP4 video track.
///
/// `stride` thins the timeline: a stride of 3 keeps one keyframe in every
/// three and is useful for all-intra sources such as ProRes. `max_keyframes`
/// caps how many timeline entries carry decoded pixels; the entries are spread
/// evenly across the timeline so the whole cut stays visible.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mp4KeyframeOptions {
    /// Decode the first video track when `None`.
    pub track_id: Option<u64>,
    /// Keep every `stride`-th keyframe in the timeline. Zero and one are
    /// equivalent and keep every keyframe.
    pub stride: usize,
    /// Decode at most this many keyframes. Zero returns a metadata-only
    /// timeline with no decoded pixels.
    pub max_keyframes: usize,
}

impl Default for Mp4KeyframeOptions {
    fn default() -> Self {
        Self {
            track_id: None,
            stride: 1,
            max_keyframes: 16,
        }
    }
}

impl Mp4KeyframeOptions {
    fn effective_stride(&self) -> usize {
        self.stride.max(1)
    }
}

/// One keyframe in a MOV/MP4 video track.
#[derive(Clone, Debug, PartialEq)]
pub struct Mp4Keyframe {
    pub sample_id: u32,
    pub decode_time: u64,
    pub presentation_time: i64,
    pub duration: u32,
    pub presentation_seconds: f64,
    pub byte_offset: u64,
    pub byte_size: u32,
    pub frame: Option<VideoFrame>,
}

/// The complete keyframe timeline for one video track.
#[derive(Clone, Debug, PartialEq)]
pub struct Mp4KeyframeTimeline {
    pub track_id: u64,
    pub codec: String,
    pub codec_id: String,
    pub timescale: u32,
    pub width: u32,
    pub height: u32,
    /// Number of sync samples in the source track before `stride`.
    pub total_keyframes: usize,
    /// Number of timeline entries that carried decoded pixels.
    pub decoded_keyframes: usize,
    pub keyframes: Vec<Mp4Keyframe>,
}

/// Build a decoded keyframe timeline from an in-memory MOV or MP4 source.
pub fn decode_mp4_keyframes(
    bytes: &[u8],
    options: &Mp4KeyframeOptions,
) -> Result<Mp4KeyframeTimeline, String> {
    let index = Mp4MediaIndex::from_file(bytes)?;
    mp4_keyframes_from_index(index, |offset, len| {
        let start = usize::try_from(offset)
            .map_err(|_| "MP4 sample offset exceeds this platform".to_string())?;
        let end = start
            .checked_add(len)
            .ok_or_else(|| "MP4 sample byte range overflow".to_string())?;
        bytes
            .get(start..end)
            .map(|slice| slice.to_vec())
            .ok_or_else(|| "MP4 sample range extends past the source".to_string())
    }, options)
}

/// Build a decoded keyframe timeline from a MOV or MP4 file on disk.
///
/// Only the `moov` box and the selected keyframe sample ranges are read into
/// memory, so multi-gigabyte sources remain practical.
pub fn decode_mp4_keyframes_from_file(
    path: &Path,
    options: &Mp4KeyframeOptions,
) -> Result<Mp4KeyframeTimeline, String> {
    let mut file = File::open(path)
        .map_err(|error| format!("open {}: {error}", path.display()))?;
    let file_size = file
        .metadata()
        .map_err(|error| format!("metadata {}: {error}", path.display()))?
        .len();
    let moov = read_moov_payload(&mut file, file_size)?;
    let index = Mp4MediaIndex::from_moov_payload(&moov)?;
    mp4_keyframes_from_index(index, |offset, len| {
        let mut buffer = vec![0_u8; len];
        file.seek(SeekFrom::Start(offset))
            .map_err(|error| format!("seek to MP4 sample at byte {offset}: {error}"))?;
        file.read_exact(&mut buffer)
            .map_err(|error| format!("read MP4 sample at byte {offset}: {error}"))?;
        Ok(buffer)
    }, options)
}

/// Decode every sync sample of one video track through `read_sample`.
///
/// `read_sample` must return exactly `len` bytes at absolute file offset
/// `offset`. Bytes may come from memory or from a seekable reader so the same
/// timeline logic serves both native and browser callers.
pub fn mp4_keyframes_from_index<R>(
    index: Mp4MediaIndex,
    mut read_sample: R,
    options: &Mp4KeyframeOptions,
) -> Result<Mp4KeyframeTimeline, String>
where
    R: FnMut(u64, usize) -> Result<Vec<u8>, String>,
{
    let stride = options.effective_stride();

    let track = select_video_track(&index, options.track_id)?;
    let mut source_indices = index
        .samples
        .iter()
        .enumerate()
        .filter(|(_, sample)| {
            sample.kind == MediaTrackKind::Video
                && sample.track_id == track.track_id
                && sample.is_sync
        })
        .map(|(position, _)| position)
        .collect::<Vec<_>>();
    source_indices.sort_unstable_by_key(|&position| index.samples[position].decode_time);
    let total_keyframes = source_indices.len();

    let timeline_indices = source_indices
        .iter()
        .enumerate()
        .filter(|(position, _)| position % stride == 0)
        .map(|(_, &position)| position)
        .collect::<Vec<_>>();

    let max_keyframes = options.max_keyframes;
    let decode_count = timeline_indices.len().min(max_keyframes);
    let decode_positions = evenly_spaced_indices(timeline_indices.len(), decode_count);

    let codec = VideoCodec::parse(&track.codec)
        .ok_or_else(|| format!("unsupported keyframe codec {}", track.codec))?;
    let width = track.width.unwrap_or(0);
    let height = track.height.unwrap_or(0);

    let mut keyframes = Vec::with_capacity(timeline_indices.len());
    let mut decoded_keyframes = 0usize;
    for (timeline_position, &source_index) in timeline_indices.iter().enumerate() {
        let sample = &index.samples[source_index];
        let frame = if decode_positions.contains(&timeline_position) {
            decoded_keyframes += 1;
            decode_sync_sample(&index, track, sample, source_index, codec, &mut read_sample)?
        } else {
            None
        };
        keyframes.push(Mp4Keyframe {
            sample_id: sample.sample_id,
            decode_time: sample.decode_time,
            presentation_time: sample.presentation_time,
            duration: sample.duration,
            presentation_seconds: seconds_at(track, sample),
            byte_offset: sample.absolute_offset,
            byte_size: sample.size,
            frame,
        });
    }

    Ok(Mp4KeyframeTimeline {
        track_id: track.track_id,
        codec: track.codec.clone(),
        codec_id: track.codec_id.clone(),
        timescale: track.timescale,
        width,
        height,
        total_keyframes,
        decoded_keyframes,
        keyframes,
    })
}

/// Decode into pixels the keyframe at `timeline_position`, on demand.
///
/// The timeline entries carry no decoded pixels when they were built with
/// `Mp4KeyframeOptions::max_keyframes` set to zero. This decodes one entry
/// then, so a browser can walk a timeline asking for frames as it needs them
/// instead of holding a whole film in memory.
pub fn decode_mp4_keyframe_at<R>(
    index: &Mp4MediaIndex,
    timeline: &Mp4KeyframeTimeline,
    timeline_position: usize,
    mut read_sample: R,
) -> Result<Option<VideoFrame>, String>
where
    R: FnMut(u64, usize) -> Result<Vec<u8>, String>,
{
    let track = select_video_track(index, Some(timeline.track_id))?;
    let entry = timeline
        .keyframes
        .get(timeline_position)
        .ok_or_else(|| format!("keyframe timeline position {timeline_position} is out of range"))?;
    let source_index = index
        .samples
        .iter()
        .position(|sample| {
            sample.kind == MediaTrackKind::Video
                && sample.track_id == track.track_id
                && sample.is_sync
                && sample.sample_id == entry.sample_id
        })
        .ok_or_else(|| {
            format!(
                "keyframe sample {} is missing from the source index",
                entry.sample_id
            )
        })?;
    let codec = VideoCodec::parse(&track.codec)
        .ok_or_else(|| format!("unsupported keyframe codec {}", track.codec))?;
    decode_sync_sample(
        index,
        track,
        &index.samples[source_index],
        source_index,
        codec,
        &mut read_sample,
    )
}

fn decode_sync_sample<R>(
    index: &Mp4MediaIndex,
    track: &MediaTrackConfig,
    sample: &MediaSampleIndex,
    source_index: usize,
    codec: VideoCodec,
    read_sample: &mut R,
) -> Result<Option<VideoFrame>, String>
where
    R: FnMut(u64, usize) -> Result<Vec<u8>, String>,
{
    let raw = read_sample(sample.absolute_offset, sample.size as usize)?;
    let packet = index.packet_from_sample_bytes(source_index, &raw)?;
    let mut access_unit = Vec::with_capacity(track.decoder_configuration.len() + packet.data.len());
    access_unit.extend_from_slice(&track.decoder_configuration);
    access_unit.extend_from_slice(&packet.data);

    let mut decoder = VideoDecoder::new(codec)?;
    let frames = decoder.decode(
        &access_unit,
        Some(packet.presentation_time),
        Some(u64::from(packet.duration)),
    )?;
    Ok(frames.into_iter().next())
}

fn select_video_track(
    index: &Mp4MediaIndex,
    requested_track_id: Option<u64>,
) -> Result<&MediaTrackConfig, String> {
    let video_tracks = index
        .tracks
        .iter()
        .filter(|track| track.kind == MediaTrackKind::Video)
        .collect::<Vec<_>>();
    match requested_track_id {
        Some(track_id) => video_tracks
            .into_iter()
            .find(|track| track.track_id == track_id)
            .ok_or_else(|| format!("MOV/MP4 has no video track {track_id}")),
        None => video_tracks
            .first()
            .copied()
            .ok_or_else(|| "MOV/MP4 has no video track".to_string()),
    }
}

fn seconds_at(track: &MediaTrackConfig, sample: &MediaSampleIndex) -> f64 {
    if track.timescale == 0 {
        return 0.0;
    }
    sample.presentation_time as f64 / track.timescale as f64
}

fn evenly_spaced_indices(len: usize, count: usize) -> Vec<usize> {
    if count == 0 || len == 0 {
        return Vec::new();
    }
    if count >= len {
        return (0..len).collect();
    }
    (0..count).map(|index| index * len / count).collect()
}

/// Read the `moov` payload from a seekable source without reading `mdat`.
fn read_moov_payload(file: &mut File, file_size: u64) -> Result<Vec<u8>, String> {
    let mut offset = 0u64;
    loop {
        let remaining = (file_size - offset).min(16);
        if remaining < 8 {
            return Err("MOV/MP4 source has no moov box".to_string());
        }
        let mut header = vec![0_u8; remaining as usize];
        file.seek(SeekFrom::Start(offset))
            .map_err(|error| format!("seek to MOV/MP4 box at byte {offset}: {error}"))?;
        file.read_exact(&mut header)
            .map_err(|error| format!("read MOV/MP4 box header at byte {offset}: {error}"))?;
        let top_level = inspect_mp4_top_level_box(&header, offset, file_size)?;
        if top_level.box_type == *b"moov" {
            let mut payload = vec![0_u8; top_level.payload_size as usize];
            file.seek(SeekFrom::Start(top_level.payload_offset))
                .map_err(|error| format!("seek to moov payload: {error}"))?;
            file.read_exact(&mut payload)
                .map_err(|error| format!("read moov payload: {error}"))?;
            return Ok(payload);
        }
        if top_level.end <= offset {
            return Err("MOV/MP4 box range does not advance".to_string());
        }
        offset = top_level.end;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    fn fixture(path: &str) -> Vec<u8> {
        fs::read(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("testdata")
                .join(path),
        )
        .unwrap()
    }

    #[test]
    fn h264_timeline_decodes_a_capped_evenly_spread_subset() {
        let bytes = fixture("mp4/heat.mp4");
        let timeline = decode_mp4_keyframes(&bytes, &Mp4KeyframeOptions::default()).unwrap();
        assert_eq!(timeline.codec, "h264");
        assert_eq!(timeline.codec_id, "avc1");
        assert_eq!(timeline.width, 1280);
        assert_eq!(timeline.height, 720);
        assert_eq!(timeline.timescale, 30_000);
        assert!(timeline.total_keyframes >= 64, "keyframes={}", timeline.total_keyframes);
        assert_eq!(timeline.keyframes.len(), timeline.total_keyframes);
        assert_eq!(timeline.decoded_keyframes, 16);
        let decoded = timeline
            .keyframes
            .iter()
            .filter(|keyframe| keyframe.frame.is_some())
            .collect::<Vec<_>>();
        assert_eq!(decoded.len(), 16);
        for keyframe in &decoded {
            let frame = keyframe.frame.as_ref().unwrap();
            assert_eq!(frame.width, timeline.width);
            assert_eq!(frame.height, timeline.height);
            assert!(frame.validate().is_ok());
        }
        assert!(timeline.keyframes[0].presentation_seconds
            < timeline.keyframes[1].presentation_seconds);
        assert_eq!(timeline.keyframes[0].decode_time, 0);
    }

    #[test]
    fn h264_metadata_only_timeline_decodes_nothing() {
        let bytes = fixture("mp4/heat.mp4");
        let options = Mp4KeyframeOptions {
            max_keyframes: 0,
            ..Mp4KeyframeOptions::default()
        };
        let timeline = decode_mp4_keyframes(&bytes, &options).unwrap();
        assert_eq!(timeline.decoded_keyframes, 0);
        assert!(timeline
            .keyframes
            .iter()
            .all(|keyframe| keyframe.frame.is_none()));
        assert_eq!(timeline.keyframes.len(), timeline.total_keyframes);
    }

    #[test]
    fn h264_stride_thins_only_the_timeline() {
        let bytes = fixture("mp4/heat.mp4");
        let all = decode_mp4_keyframes(&bytes, &Mp4KeyframeOptions::default()).unwrap();
        let thinned = decode_mp4_keyframes(
            &bytes,
            &Mp4KeyframeOptions {
                stride: 2,
                ..Mp4KeyframeOptions::default()
            },
        )
        .unwrap();
        assert_eq!(all.total_keyframes, thinned.total_keyframes);
        assert_eq!(thinned.keyframes.len(), all.keyframes.len().div_ceil(2));
        assert_eq!(thinned.decoded_keyframes, 16);
        assert_eq!(
            thinned
                .keyframes
                .iter()
                .filter(|keyframe| keyframe.frame.is_some())
                .count(),
            16
        );
    }

    #[test]
    fn hevc_timeline_decodes_keyframes() {
        let bytes = fixture("video-compat/never-final/hevc-main-aac.mov");
        let options = Mp4KeyframeOptions {
            max_keyframes: 4,
            ..Mp4KeyframeOptions::default()
        };
        let timeline = decode_mp4_keyframes(&bytes, &options).unwrap();
        assert_eq!(timeline.codec, "hevc");
        assert!(timeline.total_keyframes >= 1);
        assert_eq!(
            timeline.decoded_keyframes,
            timeline.keyframes.len().min(4)
        );
        let decoded = timeline
            .keyframes
            .iter()
            .filter_map(|keyframe| keyframe.frame.as_ref())
            .collect::<Vec<_>>();
        assert!(decoded.iter().all(|frame| frame.validate().is_ok()));
    }

    #[test]
    fn all_intra_prores_timeline_marks_every_sample_sync() {
        let bytes = fixture("video-compat/never-final/prores-422-hq-pcm.mov");
        let timeline = decode_mp4_keyframes(&bytes, &Mp4KeyframeOptions::default()).unwrap();
        assert_eq!(timeline.codec, "prores");
        assert_eq!(timeline.codec_id, "apch");
        assert_eq!(timeline.total_keyframes, 75);
        assert_eq!(timeline.keyframes.len(), 75);
        assert_eq!(timeline.decoded_keyframes, 16);
        assert_eq!(
            timeline
                .keyframes
                .iter()
                .filter(|keyframe| keyframe.frame.is_some())
                .count(),
            16
        );
    }

    #[test]
    fn missing_video_track_is_rejected() {
        let bytes = fixture("mov-pcm/pcm-s24le.mov");
        let error = decode_mp4_keyframes(&bytes, &Mp4KeyframeOptions::default()).unwrap_err();
        assert!(error.contains("no video track"), "{error}");
    }

    #[test]
    fn metadata_timeline_decodes_any_single_keyframe_on_demand() {
        let bytes = fixture("mp4/heat.mp4");
        let options = Mp4KeyframeOptions {
            max_keyframes: 0,
            ..Mp4KeyframeOptions::default()
        };
        let index = Mp4MediaIndex::from_file(&bytes).unwrap();
        let timeline = mp4_keyframes_from_index(index.clone(), |offset, len| {
            let start = usize::try_from(offset).unwrap();
            Ok(bytes[start..start + len].to_vec())
        }, &options)
        .unwrap();
        assert_eq!(timeline.decoded_keyframes, 0);

        for position in [0, timeline.keyframes.len() / 2, timeline.keyframes.len() - 1] {
            let frame = decode_mp4_keyframe_at(&index, &timeline, position, |offset, len| {
                let start = usize::try_from(offset).unwrap();
                Ok(bytes[start..start + len].to_vec())
            })
            .unwrap()
            .expect("an in-range keyframe must decode");
            assert_eq!(frame.width, timeline.width);
            assert_eq!(frame.height, timeline.height);
            assert!(frame.validate().is_ok());
        }

        let error = decode_mp4_keyframe_at(&index, &timeline, timeline.keyframes.len(), |_, _| {
            unreachable!("out-of-range positions must fail before any read")
        })
        .unwrap_err();
        assert!(error.contains("out of range"), "{error}");
    }
}