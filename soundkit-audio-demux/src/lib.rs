#[cfg(feature = "mpeg-ts")]
use access_unit::{bluray::parse_lpcm_access_unit, dts::parse_core_access_unit};
#[cfg(feature = "webm")]
use soundkit_webm::{WebmAudioDemuxEvent, WebmAudioDemuxer};
#[cfg(feature = "mpeg-ts")]
use std::collections::HashMap;
#[cfg(feature = "mp4")]
use std::collections::VecDeque;

#[cfg(feature = "mxf")]
mod mxf;
#[cfg(feature = "mxf")]
pub use mxf::{
    unpack_aes3_pcm, MxfMediaDemuxEvent, MxfMediaDemuxer, MxfMediaIndex, MxfPartition,
    MxfPartitionKind, MxfPcmSourcePacking, MxfTrackSourcePacking,
};
mod caf;
pub use caf::{inspect_caf_chunk, validate_caf_file_header, CafAudioIndex, CafChunkRange};

const MIN_DETECTION_BYTES: usize = 8192;
const MAX_GENERIC_DETECTION_BYTES: usize = 65_536;
const MAX_MP4_DETECTION_BYTES: usize = 64 * 1024 * 1024;
pub(crate) const MAX_CONTAINER_INPUT_CHUNK_BYTES: usize = 4 * 1024 * 1024;
#[cfg(any(feature = "mp4", feature = "mxf"))]
const MAX_MEDIA_PACKET_BYTES: u32 = 128 * 1024 * 1024;
#[cfg(feature = "mpeg-ts")]
const MAX_MPEG_TS_PES_BYTES: usize = 1024 * 1024;

fn validate_container_input_chunk(bytes: &[u8], container: &str) -> Result<(), String> {
    if bytes.len() > MAX_CONTAINER_INPUT_CHUNK_BYTES {
        return Err(format!(
            "{container} input chunk exceeds the {MAX_CONTAINER_INPUT_CHUNK_BYTES} byte streaming budget"
        ));
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AudioContainer {
    Caf,
    Mp4,
    WebM,
    MpegTs,
    Mxf,
}

impl AudioContainer {
    pub fn as_str(&self) -> &'static str {
        match self {
            AudioContainer::Caf => "caf",
            AudioContainer::Mp4 => "mp4",
            AudioContainer::WebM => "webm",
            AudioContainer::MpegTs => "mpeg-ts",
            AudioContainer::Mxf => "mxf",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AudioCodec {
    Aac,
    Pcm,
    Flac,
    Alac,
    Opus,
    Vorbis,
    Mp3,
    Ac3,
    Dts,
    Unknown(String),
}

impl AudioCodec {
    pub fn as_str(&self) -> &str {
        match self {
            AudioCodec::Aac => "aac",
            AudioCodec::Pcm => "pcm",
            AudioCodec::Flac => "flac",
            AudioCodec::Alac => "alac",
            AudioCodec::Opus => "opus",
            AudioCodec::Vorbis => "vorbis",
            AudioCodec::Mp3 => "mp3",
            AudioCodec::Ac3 => "ac3",
            AudioCodec::Dts => "dts",
            AudioCodec::Unknown(codec) => codec.as_str(),
        }
    }
}

/// A container track category. This is the shared demux contract; codecs and
/// renderers remain separate from container parsing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MediaTrackKind {
    Audio,
    Video,
}

/// The single linear presentation edit applied to a media track. Values use
/// the track timescale. Samples before `media_start` are decoder preroll;
/// samples after `media_start + duration` are decoder padding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MediaTrackTimeline {
    pub presentation_start: u64,
    pub media_start: u64,
    pub duration: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PcmPacketTrim {
    pub source_frame_start: u32,
    pub frame_count: u32,
}

/// Resolve the exact decoded-frame intersection between one packet and the
/// Rust-validated presentation timeline. This removes codec preroll and tail
/// padding without asking a platform adapter to interpret MP4 edit lists.
pub fn resolve_pcm_packet_trim(
    timeline: MediaTrackTimeline,
    packet_presentation_time: i64,
    packet_duration: u32,
    decoded_frames: u32,
    track_timescale: u32,
    sample_rate: u32,
) -> Result<Option<PcmPacketTrim>, String> {
    if packet_duration == 0 {
        return Err("PCM trim packet duration is zero".to_string());
    }
    if track_timescale == 0 || sample_rate == 0 {
        return Err("PCM trim requires non-zero track and sample rates".to_string());
    }
    if decoded_frames == 0 || timeline.duration == 0 {
        return Ok(None);
    }
    let packet_start = i128::from(packet_presentation_time);
    let packet_end = packet_start
        .checked_add(i128::from(packet_duration))
        .ok_or_else(|| "PCM trim packet time overflow".to_string())?;
    let timeline_start = i128::from(timeline.presentation_start);
    let timeline_end = timeline_start
        .checked_add(i128::from(timeline.duration))
        .ok_or_else(|| "PCM trim timeline overflow".to_string())?;
    let intersection_start = packet_start.max(timeline_start);
    let intersection_end = packet_end.min(timeline_end);
    if intersection_end <= intersection_start {
        return Ok(None);
    }

    let timescale = u128::from(track_timescale);
    let rate = u128::from(sample_rate);
    let relative_start = u128::try_from(intersection_start - packet_start)
        .map_err(|_| "PCM trim start precedes packet".to_string())?;
    let relative_end = u128::try_from(intersection_end - packet_start)
        .map_err(|_| "PCM trim end precedes packet".to_string())?;
    let source_start = relative_start
        .checked_mul(rate)
        .and_then(|value| value.checked_add(timescale - 1))
        .ok_or_else(|| "PCM trim start overflow".to_string())?
        / timescale;
    let source_end = relative_end
        .checked_mul(rate)
        .ok_or_else(|| "PCM trim end overflow".to_string())?
        / timescale;
    let source_start = source_start.min(u128::from(decoded_frames));
    let source_end = source_end.min(u128::from(decoded_frames));
    if source_end <= source_start {
        return Ok(None);
    }
    Ok(Some(PcmPacketTrim {
        source_frame_start: u32::try_from(source_start)
            .map_err(|_| "PCM trim start exceeds u32".to_string())?,
        frame_count: u32::try_from(source_end - source_start)
            .map_err(|_| "PCM trim frame count exceeds u32".to_string())?,
    }))
}

impl MediaTrackKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Audio => "audio",
            Self::Video => "video",
        }
    }
}

/// Codec and sample-table metadata for one selected container track.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MediaTrackConfig {
    pub container: AudioContainer,
    pub kind: MediaTrackKind,
    pub track_id: u64,
    /// The normalized SoundKit codec name, such as `h264`, `prores`, or `aac`.
    pub codec: String,
    /// The exact container sample-entry identifier, such as `hvc1` or `ap4h`.
    pub codec_id: String,
    pub timescale: u32,
    /// First media edit retained for source compatibility with older callers.
    pub timeline: Option<MediaTrackTimeline>,
    /// Every linear media edit, with empty edits represented by gaps between
    /// adjacent `presentation_start` values.
    pub edit_timeline: Vec<MediaTrackTimeline>,
    pub sample_count: u32,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub sample_rate: Option<u32>,
    pub channels: Option<u8>,
    pub bits_per_sample: Option<u8>,
    pub pcm_endianness: Option<PcmEndianness>,
    pub pcm_float: Option<bool>,
    pub pcm_signed: Option<bool>,
    pub pcm_packed: Option<bool>,
    pub pcm_aligned_high: Option<bool>,
    pub pcm_interleaved: Option<bool>,
    pub pcm_bytes_per_frame: Option<u32>,
    pub pcm_frames_per_packet: Option<u32>,
    /// Container-native codec configuration (`avcC`, `hvcC`, `av1C`, etc.).
    pub codec_private: Vec<u8>,
    /// Decoder-ready configuration. AVC and HEVC parameter sets use Annex B.
    pub decoder_configuration: Vec<u8>,
    pub nal_length_size: Option<u8>,
}

/// One decoder-ready compressed access unit or PCM sample chunk.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MediaTrackPacket {
    pub track_id: u64,
    pub kind: MediaTrackKind,
    pub codec: String,
    pub sample_id: u32,
    pub data: Vec<u8>,
    pub decode_time: u64,
    pub presentation_time: i64,
    pub duration: u32,
    pub is_sync: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DemuxedMediaFile {
    pub tracks: Vec<MediaTrackConfig>,
    /// Packets are ordered by their byte position in the source file. This
    /// preserves streaming locality for interleaved audio and video tracks.
    pub packets: Vec<MediaTrackPacket>,
}

/// A validated byte range and timestamp record from a container sample table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MediaSampleIndex {
    pub track_id: u64,
    pub kind: MediaTrackKind,
    pub codec: String,
    pub sample_id: u32,
    pub absolute_offset: u64,
    pub size: u32,
    pub decode_time: u64,
    pub presentation_time: i64,
    pub duration: u32,
    pub is_sync: bool,
}

/// Seekable MOV/MP4 sample index. Browser and native adapters can read `moov`
/// once, then fetch only each requested sample range from the source.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mp4MediaIndex {
    pub tracks: Vec<MediaTrackConfig>,
    pub samples: Vec<MediaSampleIndex>,
}

const MAX_MP4_METADATA_BYTES: u64 = 64 * 1024 * 1024;
/// Maximum number of compressed access-unit records materialized from one MP4
/// track. Constant-size PCM uses a compact descriptor and bounded packet runs.
#[cfg(feature = "mp4")]
const MAX_MP4_MATERIALIZED_ACCESS_UNITS: usize = 8_000_000;

/// A Rust-validated top-level MOV/MP4 box range.
///
/// Seekable adapters read at most 16 bytes at `offset`, call
/// [`inspect_mp4_top_level_box`], and skip directly to `end`. This permits a
/// multi-gigabyte `mdat` to remain outside WASM memory while Rust owns all
/// container size and range validation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mp4TopLevelBox {
    pub box_type: [u8; 4],
    pub offset: u64,
    pub payload_offset: u64,
    pub payload_size: u64,
    pub end: u64,
}

/// Inspect one top-level MOV/MP4 box from its 8-byte or 16-byte header.
///
/// `header` must start at `absolute_offset`. A 16-byte read handles both
/// regular and extended-size boxes. A zero box size extends to `file_size`.
pub fn inspect_mp4_top_level_box(
    header: &[u8],
    absolute_offset: u64,
    file_size: u64,
) -> Result<Mp4TopLevelBox, String> {
    if absolute_offset > file_size || file_size - absolute_offset < 8 {
        return Err("MOV/MP4 source ends before a top-level box header".to_string());
    }
    if header.len() < 8 {
        return Err("MOV/MP4 top-level box header needs at least 8 bytes".to_string());
    }

    let short_size = u32::from_be_bytes([header[0], header[1], header[2], header[3]]);
    let box_type = [header[4], header[5], header[6], header[7]];
    let (size, header_size) = match short_size {
        0 => (file_size - absolute_offset, 8u64),
        1 => {
            if header.len() < 16 {
                return Err("extended MOV/MP4 box header needs 16 bytes".to_string());
            }
            (
                u64::from_be_bytes([
                    header[8], header[9], header[10], header[11], header[12], header[13],
                    header[14], header[15],
                ]),
                16,
            )
        }
        value => (u64::from(value), 8),
    };
    if size < header_size {
        return Err(format!(
            "MOV/MP4 box {} is shorter than its header",
            String::from_utf8_lossy(&box_type)
        ));
    }
    let end = absolute_offset
        .checked_add(size)
        .ok_or_else(|| "MOV/MP4 top-level box range overflows u64".to_string())?;
    if end > file_size {
        return Err(format!(
            "MOV/MP4 box {} exceeds the source length",
            String::from_utf8_lossy(&box_type)
        ));
    }
    let payload_offset = absolute_offset + header_size;
    let payload_size = end - payload_offset;
    if box_type == *b"moov" && payload_size > MAX_MP4_METADATA_BYTES {
        return Err(format!(
            "MOV/MP4 moov metadata exceeds the {} byte budget",
            MAX_MP4_METADATA_BYTES
        ));
    }
    Ok(Mp4TopLevelBox {
        box_type,
        offset: absolute_offset,
        payload_offset,
        payload_size,
        end,
    })
}

#[cfg(feature = "mp4")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Mp4MediaDemuxEvent {
    Config(MediaTrackConfig),
    Packet(MediaTrackPacket),
}

/// Incremental fragmented-MP4/CMAF demuxer for all supported audio and video
/// tracks. Container parsing and AVC/HEVC normalization remain entirely Rust
/// owned; callers only transport byte chunks and consume typed events.
#[cfg(feature = "mp4")]
pub struct Mp4MediaDemuxer {
    buffer: Vec<u8>,
    cursor: usize,
    absolute_start: u64,
    skip_remaining: u64,
    active_mdat: Option<RegularMdatRange>,
    tracks: Vec<MediaTrackConfig>,
    track_defaults: Vec<Fmp4TrackDefaults>,
    pending_fragments: Vec<Fmp4Fragment>,
    next_sample_ids: Vec<(u64, u32)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PcmEndianness {
    Little,
    Big,
}

impl PcmEndianness {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Little => "little",
            Self::Big => "big",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AudioPacketFormat {
    Adts,
    Latm,
    Raw,
}

impl AudioPacketFormat {
    pub fn as_str(&self) -> &'static str {
        match self {
            AudioPacketFormat::Adts => "adts",
            AudioPacketFormat::Latm => "latm",
            AudioPacketFormat::Raw => "raw",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AudioTrackConfig {
    pub container: AudioContainer,
    pub codec: AudioCodec,
    pub packet_format: Option<AudioPacketFormat>,
    pub codec_id: Option<String>,
    pub track_id: Option<u64>,
    pub pid: Option<u16>,
    pub stream_type: Option<u8>,
    pub timescale: Option<u32>,
    pub transport_packet_stride: Option<u16>,
    pub transport_prefix_bytes: Option<u8>,
    pub program_number: Option<u16>,
    pub sample_rate: Option<u32>,
    pub channels: Option<u8>,
    pub bits_per_sample: Option<u8>,
    pub pcm_endianness: Option<PcmEndianness>,
    pub pcm_float: Option<bool>,
    pub pcm_signed: Option<bool>,
    pub pcm_packed: Option<bool>,
    pub pcm_aligned_high: Option<bool>,
    pub pcm_interleaved: Option<bool>,
    pub pcm_bytes_per_frame: Option<u32>,
    pub pcm_frames_per_packet: Option<u32>,
    pub sample_count: Option<u32>,
    pub codec_private: Vec<u8>,
    pub pre_skip: Option<u16>,
    pub output_gain: Option<i16>,
    pub mapping_family: Option<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AudioTrackPacket {
    pub container: AudioContainer,
    pub codec: AudioCodec,
    pub format: AudioPacketFormat,
    pub data: Vec<u8>,
    pub raw_data: Option<Vec<u8>>,
    pub track_id: Option<u64>,
    pub pid: Option<u16>,
    pub stream_type: Option<u8>,
    pub timescale: Option<u32>,
    pub continuity_counter: Option<u8>,
    pub discontinuity: bool,
    pub decode_time: Option<u64>,
    pub sample_id: Option<u32>,
    pub start_time: Option<u64>,
    pub duration: Option<u32>,
    pub rendering_offset: Option<i32>,
    pub is_sync: Option<bool>,
    pub timecode: Option<i64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AudioDemuxEvent {
    Config(AudioTrackConfig),
    Packet(AudioTrackPacket),
}

pub struct AudioTrackDemuxer {
    state: DemuxerState,
}

enum DemuxerState {
    Detecting {
        buffer: Vec<u8>,
    },
    #[cfg(feature = "mp4")]
    Mp4(Mp4AudioDemuxer),
    #[cfg(feature = "webm")]
    WebM(WebmAudioDemuxer),
    #[cfg(feature = "mpeg-ts")]
    MpegTs(MpegTsAudioDemuxer),
    Finished,
}

impl AudioTrackDemuxer {
    pub fn new_auto() -> Self {
        Self {
            state: DemuxerState::Detecting { buffer: Vec::new() },
        }
    }

    pub fn new_with_format(format: &str) -> Result<Self, String> {
        Ok(Self {
            state: demuxer_for_format(format)?,
        })
    }

    pub fn push(&mut self, bytes: &[u8]) -> Result<Vec<AudioDemuxEvent>, String> {
        validate_container_input_chunk(bytes, "media container")?;
        let state = std::mem::replace(&mut self.state, DemuxerState::Finished);
        match state {
            DemuxerState::Detecting { mut buffer } => {
                #[cfg(feature = "mp4")]
                let recognized_mp4 =
                    looks_like_mp4(&buffer) || (buffer.is_empty() && looks_like_mp4(bytes));
                #[cfg(not(feature = "mp4"))]
                let recognized_mp4 = false;
                let detection_limit = if recognized_mp4 {
                    MAX_MP4_DETECTION_BYTES
                } else {
                    MAX_GENERIC_DETECTION_BYTES
                };
                let probe_bytes = detection_limit
                    .saturating_sub(buffer.len())
                    .min(bytes.len());
                buffer.extend_from_slice(&bytes[..probe_bytes]);
                let new_bytes_collected = buffer.len();

                if new_bytes_collected < MIN_DETECTION_BYTES {
                    self.state = DemuxerState::Detecting { buffer };
                    return Ok(Vec::new());
                }

                match detect_and_init_demuxer(&buffer) {
                    Ok(mut demuxer) => {
                        let mut events = process_state(&mut demuxer, &buffer, false)?;
                        if probe_bytes < bytes.len() {
                            events.extend(process_state(
                                &mut demuxer,
                                &bytes[probe_bytes..],
                                false,
                            )?);
                        }
                        self.state = demuxer;
                        Ok(events)
                    }
                    Err(error) if new_bytes_collected < detection_limit => {
                        self.state = DemuxerState::Detecting { buffer };
                        if bytes.is_empty() {
                            self.state = DemuxerState::Finished;
                            Err(error)
                        } else {
                            Ok(Vec::new())
                        }
                    }
                    Err(error) => {
                        self.state = DemuxerState::Finished;
                        Err(error)
                    }
                }
            }
            state @ DemuxerState::Finished => {
                self.state = state;
                Err("demuxer is already finished".to_string())
            }
            #[cfg(all(feature = "mp4", feature = "webm", feature = "mpeg-ts"))]
            mut
            state @ (DemuxerState::Mp4(_) | DemuxerState::WebM(_) | DemuxerState::MpegTs(_)) => {
                let events = process_state(&mut state, bytes, false)?;
                self.state = state;
                Ok(events)
            }
            #[cfg(all(feature = "mp4", feature = "webm", not(feature = "mpeg-ts")))]
            mut state @ (DemuxerState::Mp4(_) | DemuxerState::WebM(_)) => {
                let events = process_state(&mut state, bytes, false)?;
                self.state = state;
                Ok(events)
            }
            #[cfg(all(feature = "mp4", feature = "mpeg-ts", not(feature = "webm")))]
            mut state @ (DemuxerState::Mp4(_) | DemuxerState::MpegTs(_)) => {
                let events = process_state(&mut state, bytes, false)?;
                self.state = state;
                Ok(events)
            }
            #[cfg(all(feature = "webm", feature = "mpeg-ts", not(feature = "mp4")))]
            mut state @ (DemuxerState::WebM(_) | DemuxerState::MpegTs(_)) => {
                let events = process_state(&mut state, bytes, false)?;
                self.state = state;
                Ok(events)
            }
            #[cfg(all(feature = "mp4", not(any(feature = "webm", feature = "mpeg-ts"))))]
            mut state @ DemuxerState::Mp4(_) => {
                let events = process_state(&mut state, bytes, false)?;
                self.state = state;
                Ok(events)
            }
            #[cfg(all(feature = "webm", not(any(feature = "mp4", feature = "mpeg-ts"))))]
            mut state @ DemuxerState::WebM(_) => {
                let events = process_state(&mut state, bytes, false)?;
                self.state = state;
                Ok(events)
            }
            #[cfg(all(feature = "mpeg-ts", not(any(feature = "mp4", feature = "webm"))))]
            mut state @ DemuxerState::MpegTs(_) => {
                let events = process_state(&mut state, bytes, false)?;
                self.state = state;
                Ok(events)
            }
        }
    }

    pub fn flush(&mut self) -> Result<Vec<AudioDemuxEvent>, String> {
        let state = std::mem::replace(&mut self.state, DemuxerState::Finished);
        match state {
            DemuxerState::Detecting { buffer } => {
                let mut demuxer = detect_and_init_demuxer(&buffer)?;
                process_state(&mut demuxer, &buffer, true)
            }
            DemuxerState::Finished => Ok(Vec::new()),
            #[cfg(all(feature = "mp4", feature = "webm", feature = "mpeg-ts"))]
            mut
            state @ (DemuxerState::Mp4(_) | DemuxerState::WebM(_) | DemuxerState::MpegTs(_)) => {
                process_state(&mut state, &[], true)
            }
            #[cfg(all(feature = "mp4", feature = "webm", not(feature = "mpeg-ts")))]
            mut state @ (DemuxerState::Mp4(_) | DemuxerState::WebM(_)) => {
                process_state(&mut state, &[], true)
            }
            #[cfg(all(feature = "mp4", feature = "mpeg-ts", not(feature = "webm")))]
            mut state @ (DemuxerState::Mp4(_) | DemuxerState::MpegTs(_)) => {
                process_state(&mut state, &[], true)
            }
            #[cfg(all(feature = "webm", feature = "mpeg-ts", not(feature = "mp4")))]
            mut state @ (DemuxerState::WebM(_) | DemuxerState::MpegTs(_)) => {
                process_state(&mut state, &[], true)
            }
            #[cfg(all(feature = "mp4", not(any(feature = "webm", feature = "mpeg-ts"))))]
            mut state @ DemuxerState::Mp4(_) => process_state(&mut state, &[], true),
            #[cfg(all(feature = "webm", not(any(feature = "mp4", feature = "mpeg-ts"))))]
            mut state @ DemuxerState::WebM(_) => process_state(&mut state, &[], true),
            #[cfg(all(feature = "mpeg-ts", not(any(feature = "mp4", feature = "webm"))))]
            mut state @ DemuxerState::MpegTs(_) => process_state(&mut state, &[], true),
        }
    }
}

impl Default for AudioTrackDemuxer {
    fn default() -> Self {
        Self::new_auto()
    }
}

fn demuxer_for_format(format: &str) -> Result<DemuxerState, String> {
    match normalize_format(format).as_str() {
        #[cfg(feature = "mp4")]
        "mp4" | "m4a" | "m4v" | "mov" | "quicktime" | "aac-mp4" | "mp4-aac" => {
            Ok(DemuxerState::Mp4(Mp4AudioDemuxer::regular()?))
        }
        #[cfg(feature = "mp4")]
        "fmp4" | "fragmented-mp4" | "cmaf" | "cmf" => {
            Ok(DemuxerState::Mp4(Mp4AudioDemuxer::fragmented()))
        }
        #[cfg(feature = "webm")]
        "webm" | "matroska" | "mkv" => {
            let mut demuxer = WebmAudioDemuxer::new();
            demuxer.init()?;
            Ok(DemuxerState::WebM(demuxer))
        }
        #[cfg(feature = "mpeg-ts")]
        "ts" | "mpeg-ts" | "mpegts" | "hls-ts" | "m2ts" | "bdav" => {
            Ok(DemuxerState::MpegTs(MpegTsAudioDemuxer::new()))
        }
        other => Err(format!("unsupported audio demux format: {other}")),
    }
}

fn detect_and_init_demuxer(bytes: &[u8]) -> Result<DemuxerState, String> {
    let _ = bytes;
    #[cfg(feature = "mpeg-ts")]
    if looks_like_mpeg_ts(bytes) {
        return demuxer_for_format("mpeg-ts");
    }

    #[cfg(feature = "webm")]
    if bytes.starts_with(&[0x1a, 0x45, 0xdf, 0xa3]) {
        return demuxer_for_format("webm");
    }

    #[cfg(feature = "mp4")]
    if looks_like_mp4(bytes) {
        return match classify_mp4_layout(bytes)? {
            Some(Mp4Layout::Fragmented) => demuxer_for_format("fmp4"),
            Some(Mp4Layout::Regular) => demuxer_for_format("mp4"),
            None => Err(format!(
                "MOV/MP4 classification is incomplete within {} collected bytes",
                bytes.len()
            )),
        };
    }

    Err("could not detect supported audio/video container".to_string())
}

#[cfg(feature = "mp4")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mp4Layout {
    Regular,
    Fragmented,
}

#[cfg(feature = "mp4")]
fn classify_mp4_layout(bytes: &[u8]) -> Result<Option<Mp4Layout>, String> {
    let mut pos = 0usize;
    while pos < bytes.len() {
        let remaining = &bytes[pos..];
        if remaining.len() < 8 || (remaining.starts_with(&[0, 0, 0, 1]) && remaining.len() < 16) {
            return Ok(None);
        }
        let header = Mp4BoxHeader::read(remaining)
            .ok_or_else(|| format!("invalid MOV/MP4 box header at byte {pos}"))?;
        let end = pos
            .checked_add(header.size)
            .ok_or_else(|| "MOV/MP4 classification offset overflow".to_string())?;
        if end > bytes.len() {
            return Ok(None);
        }
        match &header.name {
            b"moof" => return Ok(Some(Mp4Layout::Fragmented)),
            b"moov" => {
                let payload = &bytes[pos + header.header_size..end];
                return if mp4_moov_contains_mvex(payload)? {
                    Ok(Some(Mp4Layout::Fragmented))
                } else {
                    Ok(Some(Mp4Layout::Regular))
                };
            }
            _ => {}
        }
        pos = end;
    }
    Ok(None)
}

#[cfg(feature = "mp4")]
fn mp4_moov_contains_mvex(moov: &[u8]) -> Result<bool, String> {
    let mut found = false;
    for_each_child_box(moov, |header, _, _| {
        found |= header.name == *b"mvex";
        Ok(())
    })?;
    Ok(found)
}

fn process_state(
    state: &mut DemuxerState,
    bytes: &[u8],
    finalizing: bool,
) -> Result<Vec<AudioDemuxEvent>, String> {
    let _ = (bytes, finalizing);
    match state {
        #[cfg(feature = "mp4")]
        DemuxerState::Mp4(demuxer) => {
            if finalizing {
                demuxer.finish(bytes)
            } else {
                demuxer.add(bytes)
            }
        }
        #[cfg(feature = "webm")]
        DemuxerState::WebM(demuxer) => {
            if finalizing {
                convert_webm_events(demuxer.finish()?)
            } else {
                convert_webm_events(demuxer.add(bytes)?)
            }
        }
        #[cfg(feature = "mpeg-ts")]
        DemuxerState::MpegTs(demuxer) => {
            if finalizing {
                demuxer.finish(bytes)
            } else {
                demuxer.add(bytes)
            }
        }
        DemuxerState::Detecting { .. } => Ok(Vec::new()),
        DemuxerState::Finished => Err("demuxer is already finished".to_string()),
    }
}

#[cfg(feature = "mp4")]
enum Mp4AudioDemuxer {
    Regular(RegularMp4AudioDemuxer),
    Fragmented(FragmentedMp4AudioDemuxer),
}

#[cfg(feature = "mp4")]
impl Mp4AudioDemuxer {
    fn regular() -> Result<Self, String> {
        Ok(Self::Regular(RegularMp4AudioDemuxer::new()))
    }

    fn fragmented() -> Self {
        Self::Fragmented(FragmentedMp4AudioDemuxer::new())
    }

    fn add(&mut self, bytes: &[u8]) -> Result<Vec<AudioDemuxEvent>, String> {
        match self {
            Self::Regular(demuxer) => demuxer.add(bytes),
            Self::Fragmented(demuxer) => demuxer.add(bytes),
        }
    }

    fn finish(&mut self, bytes: &[u8]) -> Result<Vec<AudioDemuxEvent>, String> {
        match self {
            Self::Regular(demuxer) => demuxer.finish(bytes),
            Self::Fragmented(demuxer) => demuxer.finish(bytes),
        }
    }
}

#[cfg(feature = "mp4")]
#[derive(Clone, Debug)]
struct RegularMp4AudioTrack {
    track_id: u32,
    timescale: u32,
    sample_rate: u32,
    channels: u8,
    bits_per_sample: Option<u8>,
    codec: AudioCodec,
    codec_id: String,
    packet_format: AudioPacketFormat,
    pcm_endianness: Option<PcmEndianness>,
    pcm_float: Option<bool>,
    pcm_signed: Option<bool>,
    pcm_packed: Option<bool>,
    pcm_aligned_high: Option<bool>,
    pcm_interleaved: Option<bool>,
    pcm_bytes_per_frame: Option<u32>,
    pcm_frames_per_packet: Option<u32>,
    codec_private: Vec<u8>,
    samples: Vec<RegularMp4Sample>,
}

#[cfg(feature = "mp4")]
#[derive(Clone, Debug)]
struct RegularMp4Sample {
    sample_id: u32,
    absolute_offset: u64,
    size: u32,
    duration: u32,
    start_time: u64,
    rendering_offset: i32,
    is_sync: bool,
}

#[cfg(feature = "mp4")]
#[derive(Clone, Debug)]
struct Mp4MediaTrackIndex {
    config: MediaTrackConfig,
    samples: Vec<RegularMp4Sample>,
}

#[cfg(feature = "mp4")]
#[derive(Clone, Debug)]
struct Mp4VideoSampleEntry {
    codec: String,
    codec_id: String,
    width: u32,
    height: u32,
    codec_private: Vec<u8>,
    decoder_configuration: Vec<u8>,
    nal_length_size: Option<u8>,
}

#[cfg(feature = "mp4")]
#[derive(Clone, Debug, Default)]
struct Mp4MediaTrakTables {
    track_id: Option<u32>,
    kind: Option<MediaTrackKind>,
    timescale: Option<u32>,
    edit_list: Vec<Mp4EditListEntry>,
    audio_entry: Option<RegularAudioSampleEntry>,
    video_entry: Option<Mp4VideoSampleEntry>,
    stts: Vec<SttsEntry>,
    ctts: Vec<CttsEntry>,
    stsc: Vec<StscEntry>,
    sample_sizes: Mp4SampleSizes,
    chunk_offsets: Vec<u64>,
    sync_samples: Vec<u32>,
}

#[cfg(feature = "mp4")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Mp4EditListEntry {
    segment_duration: u64,
    media_time: i64,
    media_rate_integer: i16,
    media_rate_fraction: i16,
}

/// Demux every supported audio and video track from a complete MOV or MP4
/// source. The returned packets are decoder-ready: length-prefixed AVC/HEVC
/// samples are converted to Annex B in Rust.
#[cfg(feature = "mp4")]
pub fn demux_mp4_media_file(bytes: &[u8]) -> Result<DemuxedMediaFile, String> {
    let moov = find_top_level_mp4_box(bytes, b"moov")?
        .ok_or_else(|| "MP4 source has no moov box".to_string())?;
    let index = Mp4MediaIndex::from_moov_payload(moov)?;
    let mut packets = Vec::with_capacity(index.samples.len());
    for (sample_index, sample) in index.samples.iter().enumerate() {
        let start = usize::try_from(sample.absolute_offset)
            .map_err(|_| "MP4 sample offset exceeds this platform".to_string())?;
        let end = start
            .checked_add(sample.size as usize)
            .ok_or_else(|| "MP4 sample byte range overflow".to_string())?;
        let raw = bytes.get(start..end).ok_or_else(|| {
            format!(
                "MP4 track {} sample {} extends past the source",
                sample.track_id, sample.sample_id
            )
        })?;
        packets.push(index.packet_from_sample_bytes(sample_index, raw)?);
    }
    Ok(DemuxedMediaFile {
        tracks: index.tracks,
        packets,
    })
}

#[cfg(feature = "mp4")]
impl Mp4MediaIndex {
    /// Parse a complete MOV/MP4 source. Prefer [`Self::from_moov_payload`]
    /// with seekable files so a large leading `mdat` does not cross the WASM
    /// boundary.
    pub fn from_file(bytes: &[u8]) -> Result<Self, String> {
        let moov = find_top_level_mp4_box(bytes, b"moov")?
            .ok_or_else(|| "MP4 source has no moov box".to_string())?;
        Self::from_moov_payload(moov)
    }

    /// Parse a `moov` payload without reading `mdat`. All absolute sample
    /// offsets remain relative to the complete source file.
    pub fn from_moov_payload(moov: &[u8]) -> Result<Self, String> {
        if moov.len() as u64 > MAX_MP4_METADATA_BYTES {
            return Err(format!(
                "MOV/MP4 moov metadata exceeds the {} byte budget",
                MAX_MP4_METADATA_BYTES
            ));
        }
        let indexes = parse_mp4_media_indexes(moov)?;
        let mut samples = Vec::new();
        for track in &indexes {
            let source_samples = if track.config.codec == "pcm" {
                coalesce_regular_pcm_samples(track.samples.clone())?
            } else {
                track.samples.clone()
            };
            let mut track_samples = Vec::with_capacity(source_samples.len());
            for sample in &source_samples {
                let media_presentation_time = i128::from(sample.start_time)
                    .checked_add(i128::from(sample.rendering_offset))
                    .ok_or_else(|| "MP4 presentation timestamp overflow".to_string())?;
                let presentation_time = map_mp4_presentation_time(
                    &track.config.edit_timeline,
                    media_presentation_time,
                )?;
                track_samples.push(MediaSampleIndex {
                    track_id: track.config.track_id,
                    kind: track.config.kind,
                    codec: track.config.codec.clone(),
                    sample_id: sample.sample_id,
                    absolute_offset: sample.absolute_offset,
                    size: sample.size,
                    decode_time: sample.start_time,
                    presentation_time,
                    duration: sample.duration,
                    is_sync: sample.is_sync,
                });
            }
            samples.extend(track_samples);
        }
        samples.sort_unstable_by_key(|sample| sample.absolute_offset);
        Ok(Self {
            tracks: indexes.into_iter().map(|track| track.config).collect(),
            samples,
        })
    }

    /// Validate and normalize one source sample. Callers must pass exactly the
    /// byte range described by `samples[sample_index]`.
    pub fn packet_from_sample_bytes(
        &self,
        sample_index: usize,
        raw: &[u8],
    ) -> Result<MediaTrackPacket, String> {
        let sample = self
            .samples
            .get(sample_index)
            .ok_or_else(|| format!("MP4 sample index {sample_index} is out of range"))?;
        if sample.size > MAX_MEDIA_PACKET_BYTES {
            return Err(format!(
                "MP4 sample exceeds the {MAX_MEDIA_PACKET_BYTES} byte packet budget"
            ));
        }
        if raw.len() != sample.size as usize {
            return Err(format!(
                "MP4 track {} sample {} expected {} bytes, got {}",
                sample.track_id,
                sample.sample_id,
                sample.size,
                raw.len()
            ));
        }
        let track = self
            .tracks
            .iter()
            .find(|track| track.track_id == sample.track_id)
            .ok_or_else(|| format!("MP4 sample references unknown track {}", sample.track_id))?;
        let data = match track.nal_length_size {
            Some(length_size) => mp4_nals_to_annex_b(raw, length_size)?,
            None => raw.to_vec(),
        };
        Ok(MediaTrackPacket {
            track_id: sample.track_id,
            kind: sample.kind,
            codec: sample.codec.clone(),
            sample_id: sample.sample_id,
            data,
            decode_time: sample.decode_time,
            presentation_time: sample.presentation_time,
            duration: sample.duration,
            is_sync: sample.is_sync,
        })
    }

    pub fn pcm_packet_trim(
        &self,
        sample_index: usize,
        decoded_frames: u32,
    ) -> Result<Option<PcmPacketTrim>, String> {
        let sample = self
            .samples
            .get(sample_index)
            .ok_or_else(|| format!("MP4 sample index {sample_index} is out of range"))?;
        if sample.kind != MediaTrackKind::Audio {
            return Err(format!("MP4 sample {sample_index} is not an audio packet"));
        }
        let track = self
            .tracks
            .iter()
            .find(|track| track.track_id == sample.track_id)
            .ok_or_else(|| format!("MP4 sample references unknown track {}", sample.track_id))?;
        match track.timeline {
            Some(timeline) => resolve_pcm_packet_trim(
                timeline,
                sample.presentation_time,
                sample.duration,
                decoded_frames,
                track.timescale,
                track.sample_rate.unwrap_or(track.timescale),
            ),
            None if decoded_frames == 0 => Ok(None),
            None => Ok(Some(PcmPacketTrim {
                source_frame_start: 0,
                frame_count: decoded_frames,
            })),
        }
    }
}

#[cfg(feature = "mp4")]
fn parse_mp4_media_indexes(moov: &[u8]) -> Result<Vec<Mp4MediaTrackIndex>, String> {
    let mut movie_timescale = None;
    for_each_child_box(moov, |header, payload, _| {
        if header.name == *b"mvhd" {
            movie_timescale = parse_mdhd_timescale(payload);
        }
        Ok(())
    })?;
    let mut indexes = Vec::new();
    for_each_child_box(moov, |header, payload, _| {
        if header.name == *b"trak" {
            if let Some(track) = parse_mp4_media_trak(payload, movie_timescale)? {
                indexes.push(track);
            }
        }
        Ok(())
    })?;
    if indexes.is_empty() {
        return Err("MP4 source has no supported audio or video tracks".to_string());
    }
    Ok(indexes)
}

#[cfg(feature = "mp4")]
fn find_top_level_mp4_box<'a>(
    bytes: &'a [u8],
    target: &[u8; 4],
) -> Result<Option<&'a [u8]>, String> {
    let mut pos = 0usize;
    while pos + 8 <= bytes.len() {
        let header = Mp4BoxHeader::read(&bytes[pos..])
            .ok_or_else(|| format!("invalid MP4 box header at byte {pos}"))?;
        let end = pos
            .checked_add(header.size)
            .ok_or_else(|| "MP4 box byte range overflow".to_string())?;
        if end > bytes.len() {
            return Err(format!(
                "truncated MP4 box {} at byte {pos}",
                String::from_utf8_lossy(&header.name)
            ));
        }
        if &header.name == target {
            return Ok(Some(&bytes[pos + header.header_size..end]));
        }
        pos = end;
    }
    Ok(None)
}

#[cfg(feature = "mp4")]
fn parse_mp4_media_trak(
    data: &[u8],
    movie_timescale: Option<u32>,
) -> Result<Option<Mp4MediaTrackIndex>, String> {
    let mut tables = Mp4MediaTrakTables::default();
    walk_boxes(data, &mut |header, payload| {
        match &header.name {
            b"tkhd" => tables.track_id = parse_tkhd_track_id(payload),
            b"mdhd" => tables.timescale = parse_mdhd_timescale(payload),
            b"elst" => tables.edit_list = parse_elst(payload)?,
            b"hdlr" => {
                if let Some(kind) = parse_media_track_kind(payload) {
                    tables.kind = Some(kind);
                }
            }
            b"stsd" => match tables.kind {
                Some(MediaTrackKind::Audio) => tables.audio_entry = parse_stsd_audio(payload)?,
                Some(MediaTrackKind::Video) => tables.video_entry = parse_stsd_video(payload),
                None => {}
            },
            b"stts" => tables.stts = parse_stts(payload)?,
            b"ctts" => tables.ctts = parse_ctts(payload)?,
            b"stsc" => tables.stsc = parse_stsc(payload)?,
            b"stsz" => tables.sample_sizes = parse_stsz(payload)?,
            b"stco" => tables.chunk_offsets = parse_stco(payload)?,
            b"co64" => tables.chunk_offsets = parse_co64(payload)?,
            b"stss" => tables.sync_samples = parse_stss(payload)?,
            _ => {}
        }
        Ok(())
    })?;

    let Some(kind) = tables.kind else {
        return Ok(None);
    };
    let track_id = tables
        .track_id
        .ok_or_else(|| "MP4 media track is missing tkhd track id".to_string())?;
    let timescale = tables
        .timescale
        .ok_or_else(|| format!("MP4 track {track_id} is missing mdhd timescale"))?;
    if timescale == 0 {
        return Err(format!("MP4 track {track_id} has zero timescale"));
    }
    let edit_timeline = resolve_media_timeline(&tables.edit_list, movie_timescale, timescale)?;
    let timeline = edit_timeline.first().copied();
    let sample_tables = RegularTrakTables {
        track_id: tables.track_id,
        is_audio: kind == MediaTrackKind::Audio,
        timescale: tables.timescale,
        sample_entry: tables.audio_entry.clone(),
        stts: tables.stts,
        ctts: tables.ctts,
        stsc: tables.stsc,
        sample_sizes: tables.sample_sizes,
        chunk_offsets: tables.chunk_offsets,
        sync_samples: tables.sync_samples,
    };
    let is_fragmented_init = sample_tables.sample_sizes.is_empty()
        && sample_tables.chunk_offsets.is_empty()
        && sample_tables.stsc.is_empty();
    let samples = if is_fragmented_init {
        Vec::new()
    } else {
        build_regular_samples(&sample_tables)?
    };
    let sample_count = u32::try_from(samples.len())
        .map_err(|_| format!("MP4 track {track_id} has too many samples"))?;

    let config = match kind {
        MediaTrackKind::Audio => {
            let entry = tables
                .audio_entry
                .ok_or_else(|| format!("MP4 audio track {track_id} has no supported codec"))?;
            let decoder_configuration = match entry.codec {
                AudioCodec::Flac => normalize_mp4_flac_decoder_configuration(&entry.codec_private)?,
                _ => entry.codec_private.clone(),
            };
            MediaTrackConfig {
                container: AudioContainer::Mp4,
                kind,
                track_id: track_id as u64,
                codec: entry.codec.as_str().to_string(),
                codec_id: entry.codec_id,
                timescale,
                timeline,
                edit_timeline: edit_timeline.clone(),
                sample_count,
                width: None,
                height: None,
                sample_rate: Some(entry.sample_rate),
                channels: Some(entry.channels),
                bits_per_sample: entry.bits_per_sample,
                pcm_endianness: entry.pcm_endianness,
                pcm_float: entry.pcm_float,
                pcm_signed: entry.pcm_signed,
                pcm_packed: entry.pcm_packed,
                pcm_aligned_high: entry.pcm_aligned_high,
                pcm_interleaved: entry.pcm_interleaved,
                pcm_bytes_per_frame: entry.pcm_bytes_per_frame,
                pcm_frames_per_packet: entry.pcm_frames_per_packet,
                decoder_configuration,
                codec_private: entry.codec_private,
                nal_length_size: None,
            }
        }
        MediaTrackKind::Video => {
            let entry = tables
                .video_entry
                .ok_or_else(|| format!("MP4 video track {track_id} has no supported codec"))?;
            MediaTrackConfig {
                container: AudioContainer::Mp4,
                kind,
                track_id: track_id as u64,
                codec: entry.codec,
                codec_id: entry.codec_id,
                timescale,
                timeline,
                edit_timeline,
                sample_count,
                width: Some(entry.width),
                height: Some(entry.height),
                sample_rate: None,
                channels: None,
                bits_per_sample: None,
                pcm_endianness: None,
                pcm_float: None,
                pcm_signed: None,
                pcm_packed: None,
                pcm_aligned_high: None,
                pcm_interleaved: None,
                pcm_bytes_per_frame: None,
                pcm_frames_per_packet: None,
                codec_private: entry.codec_private,
                decoder_configuration: entry.decoder_configuration,
                nal_length_size: entry.nal_length_size,
            }
        }
    };
    Ok(Some(Mp4MediaTrackIndex { config, samples }))
}

#[cfg(feature = "mp4")]
fn normalize_mp4_flac_decoder_configuration(data: &[u8]) -> Result<Vec<u8>, String> {
    if data.starts_with(b"fLaC") {
        return Ok(data.to_vec());
    }
    if data.len() == 34 {
        let mut stream = Vec::with_capacity(42);
        stream.extend_from_slice(b"fLaC");
        stream.extend_from_slice(&[0x80, 0, 0, 34]);
        stream.extend_from_slice(data);
        return Ok(stream);
    }
    let metadata = data
        .get(4..)
        .ok_or_else(|| "MP4 dfLa decoder configuration is truncated".to_string())?;
    if metadata.len() < 38 || metadata[0] & 0x7f != 0 {
        return Err("MP4 dfLa has no leading FLAC STREAMINFO block".to_string());
    }
    let streaminfo_size = u32::from_be_bytes([0, metadata[1], metadata[2], metadata[3]]) as usize;
    if streaminfo_size != 34 || metadata.len() < 4 + streaminfo_size {
        return Err("MP4 dfLa has an invalid FLAC STREAMINFO block".to_string());
    }
    let mut stream = Vec::with_capacity(4 + metadata.len());
    stream.extend_from_slice(b"fLaC");
    stream.extend_from_slice(metadata);
    Ok(stream)
}

#[cfg(feature = "mp4")]
fn parse_media_track_kind(data: &[u8]) -> Option<MediaTrackKind> {
    match data.get(8..12)? {
        b"soun" => Some(MediaTrackKind::Audio),
        b"vide" => Some(MediaTrackKind::Video),
        _ => None,
    }
}

#[cfg(feature = "mp4")]
fn parse_stsd_video(data: &[u8]) -> Option<Mp4VideoSampleEntry> {
    let entry_count = be_u32(data, 4)?;
    let mut pos = 8usize;
    for _ in 0..entry_count {
        let header = Mp4BoxHeader::read(data.get(pos..)?)?;
        let end = pos.checked_add(header.size)?;
        let payload = data.get(pos + header.header_size..end)?;
        if payload.len() < 78 {
            pos = end;
            continue;
        }
        let codec_id = String::from_utf8_lossy(&header.name).into_owned();
        let codec = match &header.name {
            b"avc1" | b"avc3" => "h264",
            b"hvc1" | b"hev1" => "hevc",
            b"av01" => "av1",
            b"vp09" => "vp9",
            b"apco" | b"apcs" | b"apcn" | b"apch" | b"ap4h" | b"ap4x" => "prores",
            b"AVdn" | b"AVdh" => "dnxhr",
            _ => {
                pos = end;
                continue;
            }
        };
        let width = be_u16(payload, 24)? as u32;
        let height = be_u16(payload, 26)? as u32;
        if width == 0 || height == 0 {
            return None;
        }
        let mut codec_private = Vec::new();
        let mut decoder_configuration = Vec::new();
        let mut nal_length_size = None;
        let _ = for_each_child_box(&payload[78..], |child, child_payload, _| {
            let is_config = matches!(&child.name, b"avcC" | b"hvcC" | b"av1C" | b"vpcC");
            if is_config && codec_private.is_empty() {
                codec_private = child_payload.to_vec();
                match &child.name {
                    b"avcC" => {
                        let config =
                            soundkit_video::parse_avc_decoder_configuration(child_payload)?;
                        nal_length_size = Some(config.length_size);
                        decoder_configuration = config.annex_b;
                    }
                    b"hvcC" => {
                        let config =
                            soundkit_video::parse_hevc_decoder_configuration(child_payload)?;
                        nal_length_size = Some(config.length_size);
                        decoder_configuration = config.annex_b;
                    }
                    b"av1C" => {
                        decoder_configuration =
                            soundkit_video::parse_av1_decoder_configuration(child_payload)?;
                    }
                    // A vpcC box is codec metadata, not parameter sets: VP9
                    // frames carry everything the decoder needs.
                    _ => decoder_configuration = Vec::new(),
                }
            }
            Ok(())
        });
        return Some(Mp4VideoSampleEntry {
            codec: codec.to_string(),
            codec_id,
            width,
            height,
            codec_private,
            decoder_configuration,
            nal_length_size,
        });
    }
    None
}

#[cfg(feature = "mp4")]
fn mp4_nals_to_annex_b(data: &[u8], length_size: u8) -> Result<Vec<u8>, String> {
    soundkit_video::length_prefixed_nals_to_annex_b(data, length_size)
}

#[cfg(feature = "mp4")]
#[derive(Clone, Debug)]
struct RegularMdatRange {
    payload_start: u64,
    payload_end: u64,
}

#[cfg(feature = "mp4")]
struct RegularMp4AudioDemuxer {
    buffer: Vec<u8>,
    absolute_start: u64,
    skip_remaining: u64,
    track: Option<RegularMp4AudioTrack>,
    active_mdat: Option<RegularMdatRange>,
    emitted_config: bool,
    next_sample_index: usize,
}

#[cfg(feature = "mp4")]
impl RegularMp4AudioDemuxer {
    fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(128 * 1024),
            absolute_start: 0,
            skip_remaining: 0,
            track: None,
            active_mdat: None,
            emitted_config: false,
            next_sample_index: 0,
        }
    }

    fn add(&mut self, bytes: &[u8]) -> Result<Vec<AudioDemuxEvent>, String> {
        validate_container_input_chunk(bytes, "MP4")?;
        self.buffer.extend_from_slice(bytes);
        self.parse_available(false)
    }

    fn finish(&mut self, bytes: &[u8]) -> Result<Vec<AudioDemuxEvent>, String> {
        validate_container_input_chunk(bytes, "MP4")?;
        self.buffer.extend_from_slice(bytes);
        self.parse_available(true)
    }

    fn parse_available(&mut self, finalizing: bool) -> Result<Vec<AudioDemuxEvent>, String> {
        let mut events = Vec::new();

        loop {
            if self.skip_remaining > 0 {
                let remaining = usize::try_from(self.skip_remaining).unwrap_or(usize::MAX);
                let available = self.buffer.len().min(remaining);
                self.drain_front(available);
                self.skip_remaining -= available as u64;
                if self.skip_remaining > 0 {
                    break;
                }
                continue;
            }
            if self.active_mdat.is_some() {
                let before = self.absolute_start;
                events.extend(self.emit_active_mdat_samples()?);
                if self.active_mdat.is_some() && self.absolute_start == before {
                    break;
                }
                continue;
            }

            let Some(header) = Mp4BoxHeader::read(&self.buffer) else {
                if finalizing && !self.buffer.is_empty() {
                    return Err("truncated MP4 box header".to_string());
                }
                break;
            };

            if header.name == *b"mdat" {
                let box_start = self.absolute_start;
                let payload_start = box_start + header.header_size as u64;
                let payload_end = box_start + header.size as u64;

                if self.track.is_some() {
                    self.drain_front(header.header_size);
                    self.active_mdat = Some(RegularMdatRange {
                        payload_start,
                        payload_end,
                    });
                    continue;
                }

                return Err(
                    "MP4 mdat appears before moov; use the seekable MP4 packet API".to_string(),
                );
            }

            if header.name != *b"moov" {
                self.drain_front(header.header_size);
                self.skip_remaining = (header.size - header.header_size) as u64;
                continue;
            }

            let payload_size = header.size - header.header_size;
            if payload_size as u64 > MAX_MP4_METADATA_BYTES {
                return Err(format!(
                    "MP4 moov metadata exceeds the {} byte budget",
                    MAX_MP4_METADATA_BYTES
                ));
            }
            if self.buffer.len() < header.size {
                if finalizing {
                    return Err(format!(
                        "truncated MP4 box {}",
                        String::from_utf8_lossy(&header.name)
                    ));
                }
                break;
            }

            let payload = self.buffer[header.header_size..header.size].to_vec();
            if let Some(track) = parse_regular_moov(&payload)? {
                self.track = Some(track);
                self.emit_config_if_needed(&mut events);
            }
            self.drain_front(header.size);
        }

        if finalizing && self.skip_remaining > 0 {
            return Err("truncated MP4 top-level box".to_string());
        }

        Ok(events)
    }

    fn drain_front(&mut self, bytes: usize) {
        self.buffer.drain(..bytes);
        self.absolute_start += bytes as u64;
    }

    fn emit_config_if_needed(&mut self, events: &mut Vec<AudioDemuxEvent>) {
        if self.emitted_config {
            return;
        }
        let Some(track) = self.track.as_ref() else {
            return;
        };
        events.push(AudioDemuxEvent::Config(AudioTrackConfig {
            container: AudioContainer::Mp4,
            codec: track.codec.clone(),
            packet_format: Some(track.packet_format.clone()),
            codec_id: Some(track.codec_id.clone()),
            track_id: Some(track.track_id as u64),
            pid: None,
            stream_type: None,
            timescale: Some(track.timescale),
            transport_packet_stride: None,
            transport_prefix_bytes: None,
            program_number: None,
            sample_rate: Some(track.sample_rate),
            channels: Some(track.channels),
            bits_per_sample: track.bits_per_sample,
            pcm_endianness: track.pcm_endianness,
            pcm_float: track.pcm_float,
            pcm_signed: track.pcm_signed,
            pcm_packed: track.pcm_packed,
            pcm_aligned_high: track.pcm_aligned_high,
            pcm_interleaved: track.pcm_interleaved,
            pcm_bytes_per_frame: track.pcm_bytes_per_frame,
            pcm_frames_per_packet: track.pcm_frames_per_packet,
            sample_count: Some(track.samples.len() as u32),
            codec_private: track.codec_private.clone(),
            pre_skip: None,
            output_gain: None,
            mapping_family: None,
        }));
        self.emitted_config = true;
    }

    fn emit_active_mdat_samples(&mut self) -> Result<Vec<AudioDemuxEvent>, String> {
        let mut events = Vec::new();
        let Some(mdat) = self.active_mdat.clone() else {
            return Ok(events);
        };
        self.emit_config_if_needed(&mut events);

        let available_end = (self.absolute_start + self.buffer.len() as u64).min(mdat.payload_end);
        if available_end <= self.absolute_start {
            return Ok(events);
        }

        while let Some(sample) = self.next_sample_in_mdat(&mdat).cloned() {
            let sample_end = sample.absolute_offset + sample.size as u64;
            if sample_end > available_end {
                break;
            }
            if sample.absolute_offset < self.absolute_start {
                return Err("MP4 sample offset was already discarded".to_string());
            }

            let start = (sample.absolute_offset - self.absolute_start) as usize;
            let end = start + sample.size as usize;
            if end > self.buffer.len() {
                break;
            }
            let raw = self.buffer[start..end].to_vec();
            self.next_sample_index += 1;
            events.push(self.packet_event(&sample, raw));
        }

        let drain_until = self.safe_mdat_drain_until(&mdat, available_end);
        if drain_until > self.absolute_start {
            self.drain_front((drain_until - self.absolute_start) as usize);
        }
        if self.absolute_start >= mdat.payload_end {
            self.active_mdat = None;
        }

        Ok(events)
    }

    fn next_sample_in_mdat(&self, mdat: &RegularMdatRange) -> Option<&RegularMp4Sample> {
        self.next_sample_in_range(mdat.payload_start, mdat.payload_end)
    }

    fn next_sample_in_range(&self, start: u64, end: u64) -> Option<&RegularMp4Sample> {
        let track = self.track.as_ref()?;
        let mut index = self.next_sample_index;
        while let Some(sample) = track.samples.get(index) {
            let sample_end = sample.absolute_offset + sample.size as u64;
            if sample_end <= start {
                index += 1;
                continue;
            }
            if sample.absolute_offset >= end {
                return None;
            }
            return Some(sample);
        }
        None
    }

    fn safe_mdat_drain_until(&self, mdat: &RegularMdatRange, available_end: u64) -> u64 {
        let Some(next_sample) = self.next_sample_in_mdat(mdat) else {
            return available_end;
        };
        next_sample
            .absolute_offset
            .min(available_end)
            .max(self.absolute_start)
    }

    fn packet_event(&self, sample: &RegularMp4Sample, raw: Vec<u8>) -> AudioDemuxEvent {
        let track = self
            .track
            .as_ref()
            .expect("track is set before emitting samples");
        let data = if track.codec == AudioCodec::Aac {
            let mut data = create_adts_header(
                track.sample_rate,
                track.channels,
                raw.len(),
                &track.codec_private,
            );
            data.extend_from_slice(&raw);
            data
        } else {
            raw.clone()
        };
        AudioDemuxEvent::Packet(AudioTrackPacket {
            container: AudioContainer::Mp4,
            codec: track.codec.clone(),
            format: track.packet_format.clone(),
            data,
            raw_data: Some(raw),
            track_id: Some(track.track_id as u64),
            pid: None,
            stream_type: None,
            timescale: Some(track.timescale),
            continuity_counter: None,
            discontinuity: false,
            decode_time: Some(sample.start_time),
            sample_id: Some(sample.sample_id),
            start_time: Some(sample.start_time),
            duration: Some(sample.duration),
            rendering_offset: Some(sample.rendering_offset),
            is_sync: Some(sample.is_sync),
            timecode: Some(sample.start_time as i64),
        })
    }
}

#[cfg(feature = "webm")]
fn convert_webm_events(events: Vec<WebmAudioDemuxEvent>) -> Result<Vec<AudioDemuxEvent>, String> {
    let mut output = Vec::new();
    for event in events {
        match event {
            WebmAudioDemuxEvent::Config(config) => {
                let codec = webm_codec(&config.codec_id);
                output.push(AudioDemuxEvent::Config(AudioTrackConfig {
                    container: AudioContainer::WebM,
                    codec,
                    packet_format: Some(AudioPacketFormat::Raw),
                    codec_id: Some(config.codec_id),
                    track_id: Some(config.track_number),
                    pid: None,
                    stream_type: None,
                    timescale: Some(1_000_000_000),
                    transport_packet_stride: None,
                    transport_prefix_bytes: None,
                    program_number: None,
                    sample_rate: Some(config.sample_rate),
                    channels: Some(config.channels),
                    bits_per_sample: None,
                    pcm_endianness: None,
                    pcm_float: None,
                    pcm_signed: None,
                    pcm_packed: None,
                    pcm_aligned_high: None,
                    pcm_interleaved: None,
                    pcm_bytes_per_frame: None,
                    pcm_frames_per_packet: None,
                    sample_count: None,
                    codec_private: config.codec_private,
                    pre_skip: config.pre_skip,
                    output_gain: config.output_gain,
                    mapping_family: config.mapping_family,
                }));
            }
            WebmAudioDemuxEvent::Packet {
                track_number,
                codec_id,
                data,
                timestamp_ns,
                duration_ns,
                discard_padding_ns: _,
            } => {
                let codec = webm_codec(&codec_id);
                output.push(AudioDemuxEvent::Packet(AudioTrackPacket {
                    container: AudioContainer::WebM,
                    codec,
                    format: AudioPacketFormat::Raw,
                    data,
                    raw_data: None,
                    track_id: Some(track_number),
                    pid: None,
                    stream_type: None,
                    timescale: Some(1_000_000_000),
                    continuity_counter: None,
                    discontinuity: false,
                    decode_time: u64::try_from(timestamp_ns).ok(),
                    sample_id: None,
                    start_time: u64::try_from(timestamp_ns).ok(),
                    duration: duration_ns.and_then(|value| u32::try_from(value).ok()),
                    rendering_offset: None,
                    is_sync: None,
                    timecode: Some(timestamp_ns),
                }));
            }
        }
    }
    Ok(output)
}

#[cfg(feature = "webm")]
fn webm_codec(codec_id: &str) -> AudioCodec {
    match codec_id {
        "A_OPUS" => AudioCodec::Opus,
        "A_VORBIS" => AudioCodec::Vorbis,
        "A_AAC" | "A_AAC/MPEG2/LC" | "A_AAC/MPEG4/LC" => AudioCodec::Aac,
        "A_MPEG/L3" => AudioCodec::Mp3,
        "A_AC3" => AudioCodec::Ac3,
        other => AudioCodec::Unknown(other.to_ascii_lowercase()),
    }
}

fn normalize_format(format: &str) -> String {
    format.trim().to_ascii_lowercase().replace('_', "-")
}

#[cfg(feature = "mp4")]
fn looks_like_mp4(bytes: &[u8]) -> bool {
    bytes.len() >= 12 && &bytes[4..8] == b"ftyp"
}

#[cfg(feature = "mp4")]
#[derive(Clone, Copy, Debug)]
struct Mp4BoxHeader {
    name: [u8; 4],
    size: usize,
    header_size: usize,
}

#[cfg(feature = "mp4")]
impl Mp4BoxHeader {
    fn read(data: &[u8]) -> Option<Self> {
        if data.len() < 8 {
            return None;
        }

        let short_size = u32::from_be_bytes([data[0], data[1], data[2], data[3]]) as u64;
        let name = [data[4], data[5], data[6], data[7]];
        let (size, header_size) = if short_size == 1 {
            if data.len() < 16 {
                return None;
            }
            (
                u64::from_be_bytes([
                    data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15],
                ]),
                16usize,
            )
        } else if short_size == 0 {
            return None;
        } else {
            (short_size, 8usize)
        };

        if size < header_size as u64 || size > usize::MAX as u64 {
            return None;
        }

        Some(Self {
            name,
            size: size as usize,
            header_size,
        })
    }
}

#[cfg(feature = "mp4")]
#[derive(Clone, Debug, Default)]
struct Fmp4TrackDefaults {
    track_id: u32,
    sample_description_index: u32,
    default_sample_duration: Option<u32>,
    default_sample_size: Option<u32>,
}

#[cfg(feature = "mp4")]
#[derive(Clone, Debug)]
struct Fmp4Sample {
    absolute_offset: u64,
    size: u32,
    duration: u32,
    start_time: u64,
    rendering_offset: i32,
    is_sync: bool,
}

#[cfg(feature = "mp4")]
#[derive(Clone, Debug)]
struct Fmp4Fragment {
    track_id: u32,
    samples: Vec<Fmp4Sample>,
}

#[cfg(feature = "mp4")]
impl Mp4MediaDemuxer {
    pub fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(128 * 1024),
            cursor: 0,
            absolute_start: 0,
            skip_remaining: 0,
            active_mdat: None,
            tracks: Vec::new(),
            track_defaults: Vec::new(),
            pending_fragments: Vec::new(),
            next_sample_ids: Vec::new(),
        }
    }

    pub fn push(&mut self, bytes: &[u8]) -> Result<Vec<Mp4MediaDemuxEvent>, String> {
        validate_container_input_chunk(bytes, "fragmented MP4")?;
        self.buffer.extend_from_slice(bytes);
        self.parse_available(false)
    }

    pub fn flush(&mut self) -> Result<Vec<Mp4MediaDemuxEvent>, String> {
        self.parse_available(true)
    }

    pub fn pcm_packet_trim(
        &self,
        track_id: u64,
        presentation_time: i64,
        packet_duration: u32,
        decoded_frames: u32,
    ) -> Result<Option<PcmPacketTrim>, String> {
        let track = self
            .tracks
            .iter()
            .find(|track| track.track_id == track_id)
            .ok_or_else(|| format!("fragmented MP4 references unknown track {track_id}"))?;
        if track.kind != MediaTrackKind::Audio {
            return Err(format!("fragmented MP4 track {track_id} is not audio"));
        }
        match track.timeline {
            Some(timeline) => resolve_pcm_packet_trim(
                timeline,
                presentation_time,
                packet_duration,
                decoded_frames,
                track.timescale,
                track.sample_rate.unwrap_or(track.timescale),
            ),
            None if decoded_frames == 0 => Ok(None),
            None => Ok(Some(PcmPacketTrim {
                source_frame_start: 0,
                frame_count: decoded_frames,
            })),
        }
    }

    fn parse_available(&mut self, finalizing: bool) -> Result<Vec<Mp4MediaDemuxEvent>, String> {
        let mut events = Vec::new();
        loop {
            if self.skip_remaining > 0 {
                let remaining = usize::try_from(self.skip_remaining).unwrap_or(usize::MAX);
                let available = self.remaining_len().min(remaining);
                self.drain_front(available);
                self.skip_remaining -= available as u64;
                if self.skip_remaining > 0 {
                    break;
                }
                continue;
            }
            if self.active_mdat.is_some() {
                let before = self.absolute_start;
                events.extend(self.emit_active_fragmented_mdat_samples()?);
                if self.active_mdat.is_some() && before == self.absolute_start {
                    break;
                }
                continue;
            }
            let Some(header) = Mp4BoxHeader::read(self.remaining()) else {
                break;
            };
            if header.name == *b"mdat" {
                let box_start = self.absolute_start;
                let payload_start = box_start + header.header_size as u64;
                let payload_end = box_start + header.size as u64;
                self.drain_front(header.header_size);
                self.active_mdat = Some(RegularMdatRange {
                    payload_start,
                    payload_end,
                });
                continue;
            }
            if !matches!(&header.name, b"moov" | b"moof") {
                self.drain_front(header.header_size);
                self.skip_remaining = (header.size - header.header_size) as u64;
                continue;
            }
            let payload_size = header.size - header.header_size;
            if payload_size as u64 > MAX_MP4_METADATA_BYTES {
                return Err(format!(
                    "fragmented MP4 {} metadata exceeds the {} byte budget",
                    String::from_utf8_lossy(&header.name),
                    MAX_MP4_METADATA_BYTES
                ));
            }
            if self.remaining_len() < header.size {
                if finalizing {
                    return Err(format!(
                        "truncated fragmented MP4 box {}",
                        String::from_utf8_lossy(&header.name)
                    ));
                }
                break;
            }

            let box_start = self.absolute_start;
            match &header.name {
                b"moov" => {
                    let payload = self.remaining()[header.header_size..header.size].to_vec();
                    events.extend(self.parse_moov(&payload)?);
                    self.drain_front(header.size);
                }
                b"moof" => {
                    let payload = self.remaining()[header.header_size..header.size].to_vec();
                    let fragments = self.parse_moof(&payload, box_start)?;
                    self.pending_fragments.extend(fragments);
                    self.drain_front(header.size);
                }
                _ => self.drain_front(header.size),
            }
        }
        if finalizing && self.remaining_len() != 0 {
            return Err(format!(
                "truncated fragmented MP4 box header ({} bytes remain)",
                self.remaining_len()
            ));
        }
        if finalizing && self.skip_remaining > 0 {
            return Err("truncated fragmented MP4 top-level box".to_string());
        }
        if finalizing && self.active_mdat.is_some() {
            return Err("truncated fragmented MP4 mdat".to_string());
        }
        if finalizing && !self.pending_fragments.is_empty() {
            return Err("fragmented MP4 ended before all indexed samples arrived".to_string());
        }
        Ok(events)
    }

    fn drain_front(&mut self, bytes: usize) {
        self.cursor += bytes;
        self.absolute_start += bytes as u64;
        if self.cursor > 64 * 1024 || self.cursor == self.buffer.len() {
            self.buffer.drain(..self.cursor);
            self.cursor = 0;
        }
    }

    fn remaining(&self) -> &[u8] {
        &self.buffer[self.cursor..]
    }

    fn remaining_len(&self) -> usize {
        self.buffer.len() - self.cursor
    }

    fn parse_moov(&mut self, data: &[u8]) -> Result<Vec<Mp4MediaDemuxEvent>, String> {
        self.track_defaults = parse_trex_defaults(data)?;
        let indexes = parse_mp4_media_indexes(data)?;
        self.tracks = indexes.into_iter().map(|index| index.config).collect();
        self.next_sample_ids = self
            .tracks
            .iter()
            .map(|track| (track.track_id, 1))
            .collect();
        Ok(self
            .tracks
            .iter()
            .cloned()
            .map(Mp4MediaDemuxEvent::Config)
            .collect())
    }

    fn parse_moof(
        &self,
        data: &[u8],
        moof_absolute_start: u64,
    ) -> Result<Vec<Fmp4Fragment>, String> {
        let mut fragments = Vec::new();
        let mut inherited_base_data_offset = None;
        for_each_child_box(data, |header, payload, _| {
            if header.name == *b"traf" {
                let (fragment, data_end) =
                    self.parse_traf(payload, moof_absolute_start, inherited_base_data_offset)?;
                inherited_base_data_offset = Some(data_end);
                if let Some(fragment) = fragment {
                    fragments.push(fragment);
                }
            }
            Ok(())
        })?;
        Ok(fragments)
    }

    fn parse_traf(
        &self,
        data: &[u8],
        moof_absolute_start: u64,
        inherited_base_data_offset: Option<u64>,
    ) -> Result<(Option<Fmp4Fragment>, u64), String> {
        let mut tfhd = None;
        let mut base_decode_time = 0u64;
        let mut truns = Vec::new();
        for_each_child_box(data, |header, payload, _| {
            match &header.name {
                b"tfhd" => tfhd = Some(parse_tfhd(payload)?),
                b"tfdt" => base_decode_time = parse_tfdt(payload).unwrap_or(0),
                b"trun" => {
                    truns
                        .try_reserve(1)
                        .map_err(|_| "fragmented MP4 trun list allocation failed".to_string())?;
                    truns.push(parse_trun(payload)?);
                }
                _ => {}
            }
            Ok(())
        })?;
        let Some(tfhd) = tfhd else {
            return Err("fragmented MP4 traf is missing tfhd".to_string());
        };
        if !self
            .tracks
            .iter()
            .any(|track| track.track_id == u64::from(tfhd.track_id))
        {
            return Err(format!(
                "fragmented MP4 references unknown track {}",
                tfhd.track_id
            ));
        }
        let defaults = self
            .track_defaults
            .iter()
            .find(|defaults| defaults.track_id == tfhd.track_id)
            .cloned()
            .unwrap_or_default();
        let sample_description_index = tfhd
            .sample_description_index
            .unwrap_or(defaults.sample_description_index.max(1));
        if sample_description_index != 1 {
            return Err(format!(
                "fragmented MP4 track {} references unsupported sample description {sample_description_index}",
                tfhd.track_id
            ));
        }
        let default_duration = tfhd
            .default_sample_duration
            .or(defaults.default_sample_duration);
        let default_size = tfhd.default_sample_size.or(defaults.default_sample_size);
        let base_data_offset = match tfhd.base_data_offset {
            Some(offset) => offset,
            None if tfhd.default_base_is_moof => moof_absolute_start,
            None => inherited_base_data_offset.unwrap_or(moof_absolute_start),
        };
        let total_sample_count = truns.iter().try_fold(0usize, |total, trun| {
            total
                .checked_add(trun.sample_count as usize)
                .ok_or_else(|| "fragmented MP4 sample count overflow".to_string())
        })?;
        if total_sample_count > MAX_MP4_MATERIALIZED_ACCESS_UNITS {
            return Err(format!(
                "fragmented MP4 traf has {total_sample_count} access units; the materialized limit is {MAX_MP4_MATERIALIZED_ACCESS_UNITS}"
            ));
        }
        let mut samples = Vec::new();
        samples.try_reserve_exact(total_sample_count).map_err(|_| {
            format!("fragmented MP4 sample allocation failed for {total_sample_count} records")
        })?;
        let mut decode_time = base_decode_time;
        let mut fallback_data_offset = base_data_offset;
        for trun in truns {
            let mut sample_offset = match trun.data_offset {
                Some(offset) => add_signed_offset(base_data_offset, offset)?,
                None => fallback_data_offset,
            };
            for index in 0..trun.sample_count as usize {
                let size = trun
                    .sample_sizes
                    .get(index)
                    .copied()
                    .or(default_size)
                    .ok_or_else(|| "fragmented MP4 trun sample has no size".to_string())?;
                if size > MAX_MEDIA_PACKET_BYTES {
                    return Err(format!(
                        "fragmented MP4 sample exceeds the {MAX_MEDIA_PACKET_BYTES} byte packet budget"
                    ));
                }
                let duration = trun
                    .sample_durations
                    .get(index)
                    .copied()
                    .or(default_duration)
                    .ok_or_else(|| "fragmented MP4 trun sample has no duration".to_string())?;
                let rendering_offset = trun.sample_cts.get(index).copied().unwrap_or(0);
                let is_sync = trun
                    .sample_flags
                    .get(index)
                    .copied()
                    .or(trun.first_sample_flags.filter(|_| index == 0))
                    .map(|flags| flags & 0x0001_0000 == 0)
                    .unwrap_or(true);
                samples.push(Fmp4Sample {
                    absolute_offset: sample_offset,
                    size,
                    duration,
                    start_time: decode_time,
                    rendering_offset,
                    is_sync,
                });
                sample_offset = sample_offset
                    .checked_add(u64::from(size))
                    .ok_or_else(|| "fragmented MP4 sample offset overflow".to_string())?;
                decode_time = decode_time
                    .checked_add(u64::from(duration))
                    .ok_or_else(|| "fragmented MP4 decode time overflow".to_string())?;
            }
            fallback_data_offset = sample_offset;
        }
        Ok((
            Some(Fmp4Fragment {
                track_id: tfhd.track_id,
                samples,
            }),
            fallback_data_offset,
        ))
    }

    fn emit_active_fragmented_mdat_samples(&mut self) -> Result<Vec<Mp4MediaDemuxEvent>, String> {
        let mdat = self
            .active_mdat
            .clone()
            .ok_or_else(|| "fragmented MP4 has no active mdat".to_string())?;
        let mut pending = Vec::new();
        for fragment in std::mem::take(&mut self.pending_fragments) {
            for sample in fragment.samples {
                pending.push((fragment.track_id, sample));
            }
        }
        pending.sort_unstable_by_key(|(_, sample)| sample.absolute_offset);
        let mut pending: VecDeque<_> = pending.into();

        let mut events = Vec::new();
        loop {
            let Some((track_id, sample)) = pending.front().cloned() else {
                let available_end = self.absolute_start + self.remaining_len() as u64;
                let drain_end = available_end.min(mdat.payload_end);
                if drain_end > self.absolute_start {
                    self.drain_front((drain_end - self.absolute_start) as usize);
                }
                if self.absolute_start == mdat.payload_end {
                    self.active_mdat = None;
                }
                break;
            };
            let sample_end = sample
                .absolute_offset
                .checked_add(u64::from(sample.size))
                .ok_or_else(|| "fragmented MP4 sample range overflow".to_string())?;
            if sample.absolute_offset < mdat.payload_start {
                return Err("fragmented MP4 sample precedes its mdat payload".to_string());
            }
            if sample.absolute_offset >= mdat.payload_end || sample_end > mdat.payload_end {
                let available_end = self.absolute_start + self.remaining_len() as u64;
                let drain_end = available_end.min(mdat.payload_end);
                if drain_end > self.absolute_start {
                    self.drain_front((drain_end - self.absolute_start) as usize);
                }
                if self.absolute_start == mdat.payload_end {
                    self.active_mdat = None;
                }
                break;
            }
            if sample.absolute_offset < self.absolute_start {
                return Err("fragmented MP4 samples overlap or arrive out of order".to_string());
            }
            if sample.absolute_offset > self.absolute_start {
                let gap = usize::try_from(sample.absolute_offset - self.absolute_start)
                    .map_err(|_| "fragmented MP4 sample gap exceeds this platform".to_string())?;
                if self.remaining_len() < gap {
                    let available = self.remaining_len();
                    self.drain_front(available);
                    break;
                }
                self.drain_front(gap);
            }
            if self.remaining_len() < sample.size as usize {
                break;
            }
            let next_sample_id = self
                .next_sample_ids
                .iter_mut()
                .find(|(candidate, _)| *candidate == u64::from(track_id))
                .ok_or_else(|| format!("fragmented MP4 references unknown track {track_id}"))?;
            let sample_id = next_sample_id.1;
            next_sample_id.1 = next_sample_id
                .1
                .checked_add(1)
                .ok_or_else(|| "fragmented MP4 sample id overflow".to_string())?;
            let track = self
                .tracks
                .iter()
                .find(|track| track.track_id == u64::from(track_id))
                .ok_or_else(|| format!("fragmented MP4 references unknown track {track_id}"))?;
            let raw = self
                .remaining()
                .get(..sample.size as usize)
                .ok_or_else(|| "fragmented MP4 sample is outside the buffered mdat".to_string())?;
            let data = match track.nal_length_size {
                Some(length_size) => mp4_nals_to_annex_b(raw, length_size)?,
                None => raw.to_vec(),
            };
            let media_presentation = i128::from(sample.start_time)
                .checked_add(i128::from(sample.rendering_offset))
                .ok_or_else(|| "fragmented MP4 presentation time overflow".to_string())?;
            let presentation_time =
                map_mp4_presentation_time(&track.edit_timeline, media_presentation)?;
            events.push(Mp4MediaDemuxEvent::Packet(MediaTrackPacket {
                track_id: u64::from(track_id),
                kind: track.kind,
                codec: track.codec.clone(),
                sample_id,
                data,
                decode_time: sample.start_time,
                presentation_time,
                duration: sample.duration,
                is_sync: sample.is_sync,
            }));
            self.drain_front(sample.size as usize);
            pending.pop_front();
            if self.absolute_start == mdat.payload_end {
                self.active_mdat = None;
                break;
            }
        }
        self.pending_fragments = pending
            .into_iter()
            .map(|(track_id, sample)| Fmp4Fragment {
                track_id,
                samples: vec![sample],
            })
            .collect();
        Ok(events)
    }
}

#[cfg(feature = "mp4")]
impl Default for Mp4MediaDemuxer {
    fn default() -> Self {
        Self::new()
    }
}

/// Compatibility adapter for the audio-only API. Fragment parsing is owned by
/// `Mp4MediaDemuxer`; this layer only selects the AAC track and preserves the
/// historical ADTS packet contract.
#[cfg(feature = "mp4")]
struct FragmentedMp4AudioDemuxer {
    media: Mp4MediaDemuxer,
    selected_track: Option<MediaTrackConfig>,
}

#[cfg(feature = "mp4")]
impl FragmentedMp4AudioDemuxer {
    fn new() -> Self {
        Self {
            media: Mp4MediaDemuxer::new(),
            selected_track: None,
        }
    }

    fn add(&mut self, bytes: &[u8]) -> Result<Vec<AudioDemuxEvent>, String> {
        let events = self.media.push(bytes)?;
        self.audio_events(events)
    }

    fn finish(&mut self, bytes: &[u8]) -> Result<Vec<AudioDemuxEvent>, String> {
        let mut output = if bytes.is_empty() {
            Vec::new()
        } else {
            let events = self.media.push(bytes)?;
            self.audio_events(events)?
        };
        let events = self.media.flush()?;
        output.extend(self.audio_events(events)?);
        Ok(output)
    }

    fn audio_events(
        &mut self,
        events: Vec<Mp4MediaDemuxEvent>,
    ) -> Result<Vec<AudioDemuxEvent>, String> {
        let mut output = Vec::new();
        for event in events {
            match event {
                Mp4MediaDemuxEvent::Config(track)
                    if self.selected_track.is_none()
                        && track.kind == MediaTrackKind::Audio
                        && track.codec == "aac" =>
                {
                    output.push(AudioDemuxEvent::Config(AudioTrackConfig {
                        container: AudioContainer::Mp4,
                        codec: AudioCodec::Aac,
                        packet_format: Some(AudioPacketFormat::Adts),
                        codec_id: Some(track.codec_id.clone()),
                        track_id: Some(track.track_id),
                        pid: None,
                        stream_type: None,
                        timescale: Some(track.timescale),
                        transport_packet_stride: None,
                        transport_prefix_bytes: None,
                        program_number: None,
                        sample_rate: track.sample_rate,
                        channels: track.channels,
                        bits_per_sample: None,
                        pcm_endianness: None,
                        pcm_float: None,
                        pcm_signed: None,
                        pcm_packed: None,
                        pcm_aligned_high: None,
                        pcm_interleaved: None,
                        pcm_bytes_per_frame: None,
                        pcm_frames_per_packet: None,
                        sample_count: None,
                        codec_private: track.decoder_configuration.clone(),
                        pre_skip: None,
                        output_gain: None,
                        mapping_family: None,
                    }));
                    self.selected_track = Some(track);
                }
                Mp4MediaDemuxEvent::Packet(packet)
                    if self
                        .selected_track
                        .as_ref()
                        .is_some_and(|track| track.track_id == packet.track_id) =>
                {
                    let track = self.selected_track.as_ref().expect("matched above");
                    let sample_rate = track
                        .sample_rate
                        .ok_or_else(|| "fragmented MP4 AAC track has no sample rate".to_string())?;
                    let channels = track.channels.ok_or_else(|| {
                        "fragmented MP4 AAC track has no channel count".to_string()
                    })?;
                    let raw = packet.data;
                    let mut data = create_adts_header(
                        sample_rate,
                        channels,
                        raw.len(),
                        &track.decoder_configuration,
                    );
                    data.extend_from_slice(&raw);
                    let rendering_offset = i128::from(packet.presentation_time)
                        .checked_sub(i128::from(packet.decode_time))
                        .and_then(|value| i32::try_from(value).ok())
                        .ok_or_else(|| {
                            "fragmented MP4 AAC rendering offset exceeds i32".to_string()
                        })?;
                    output.push(AudioDemuxEvent::Packet(AudioTrackPacket {
                        container: AudioContainer::Mp4,
                        codec: AudioCodec::Aac,
                        format: AudioPacketFormat::Adts,
                        data,
                        raw_data: Some(raw),
                        track_id: Some(packet.track_id),
                        pid: None,
                        stream_type: None,
                        timescale: Some(track.timescale),
                        continuity_counter: None,
                        discontinuity: false,
                        decode_time: Some(packet.decode_time),
                        sample_id: Some(packet.sample_id),
                        start_time: Some(packet.decode_time),
                        duration: Some(packet.duration),
                        rendering_offset: Some(rendering_offset),
                        is_sync: Some(packet.is_sync),
                        timecode: Some(packet.presentation_time),
                    }));
                }
                _ => {}
            }
        }
        Ok(output)
    }
}

#[cfg(feature = "mp4")]
#[derive(Clone, Debug, Default)]
struct ParsedTfhd {
    track_id: u32,
    base_data_offset: Option<u64>,
    sample_description_index: Option<u32>,
    default_sample_duration: Option<u32>,
    default_sample_size: Option<u32>,
    default_base_is_moof: bool,
}

#[cfg(feature = "mp4")]
#[derive(Clone, Debug, Default)]
struct ParsedTrun {
    sample_count: u32,
    data_offset: Option<i32>,
    first_sample_flags: Option<u32>,
    sample_durations: Vec<u32>,
    sample_sizes: Vec<u32>,
    sample_flags: Vec<u32>,
    sample_cts: Vec<i32>,
}

#[cfg(feature = "mp4")]
fn for_each_child_box<F>(data: &[u8], mut f: F) -> Result<(), String>
where
    F: FnMut(Mp4BoxHeader, &[u8], usize) -> Result<(), String>,
{
    let mut pos = 0usize;
    while pos + 8 <= data.len() {
        let Some(header) = Mp4BoxHeader::read(&data[pos..]) else {
            break;
        };
        if header.size == 0 || pos + header.size > data.len() {
            break;
        }
        let payload_start = pos + header.header_size;
        let payload_end = pos + header.size;
        f(header, &data[payload_start..payload_end], pos)?;
        pos += header.size;
    }
    Ok(())
}

#[cfg(feature = "mp4")]
#[derive(Clone, Debug, Default)]
struct RegularTrakTables {
    track_id: Option<u32>,
    is_audio: bool,
    timescale: Option<u32>,
    sample_entry: Option<RegularAudioSampleEntry>,
    stts: Vec<SttsEntry>,
    ctts: Vec<CttsEntry>,
    stsc: Vec<StscEntry>,
    sample_sizes: Mp4SampleSizes,
    chunk_offsets: Vec<u64>,
    sync_samples: Vec<u32>,
}

#[cfg(feature = "mp4")]
#[derive(Clone, Debug)]
struct SttsEntry {
    sample_count: u32,
    sample_duration: u32,
}

#[cfg(feature = "mp4")]
#[derive(Clone, Debug)]
struct CttsEntry {
    sample_count: u32,
    sample_offset: i32,
}

#[cfg(feature = "mp4")]
#[derive(Clone, Debug)]
struct StscEntry {
    first_chunk: u32,
    samples_per_chunk: u32,
    sample_description_index: u32,
}

#[cfg(feature = "mp4")]
#[derive(Clone, Debug, Default)]
enum Mp4SampleSizes {
    #[default]
    Missing,
    Constant {
        size: u32,
        count: u32,
    },
    Variable(Vec<u32>),
}

#[cfg(feature = "mp4")]
impl Mp4SampleSizes {
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn len(&self) -> usize {
        match self {
            Self::Missing => 0,
            Self::Constant { count, .. } => *count as usize,
            Self::Variable(sizes) => sizes.len(),
        }
    }

    fn get(&self, index: usize) -> Option<u32> {
        match self {
            Self::Missing => None,
            Self::Constant { size, count } => (index < *count as usize).then_some(*size),
            Self::Variable(sizes) => sizes.get(index).copied(),
        }
    }

    fn constant(&self) -> Option<(u32, u32)> {
        match self {
            Self::Constant { size, count } => Some((*size, *count)),
            _ => None,
        }
    }
}

#[cfg(feature = "mp4")]
fn parse_regular_moov(data: &[u8]) -> Result<Option<RegularMp4AudioTrack>, String> {
    let mut selected = None;

    for_each_child_box(data, |header, payload, _| {
        if header.name == *b"trak" {
            if let Some(track) = parse_regular_trak(payload)? {
                selected = Some(track);
            }
        }
        Ok(())
    })?;

    Ok(selected)
}

#[cfg(feature = "mp4")]
fn parse_regular_trak(data: &[u8]) -> Result<Option<RegularMp4AudioTrack>, String> {
    let mut tables = RegularTrakTables::default();

    walk_boxes(data, &mut |header, payload| {
        match &header.name {
            b"tkhd" => tables.track_id = parse_tkhd_track_id(payload),
            b"mdhd" => tables.timescale = parse_mdhd_timescale(payload),
            b"hdlr" => tables.is_audio |= parse_hdlr_is_audio(payload),
            b"stsd" => tables.sample_entry = parse_stsd_audio(payload)?,
            b"stts" => tables.stts = parse_stts(payload)?,
            b"ctts" => tables.ctts = parse_ctts(payload)?,
            b"stsc" => tables.stsc = parse_stsc(payload)?,
            b"stsz" => tables.sample_sizes = parse_stsz(payload)?,
            b"stco" => tables.chunk_offsets = parse_stco(payload)?,
            b"co64" => tables.chunk_offsets = parse_co64(payload)?,
            b"stss" => tables.sync_samples = parse_stss(payload)?,
            _ => {}
        }
        Ok(())
    })?;

    if !tables.is_audio {
        return Ok(None);
    }

    let track_id = tables
        .track_id
        .ok_or_else(|| "MP4 audio track is missing tkhd track id".to_string())?;
    let mut samples = build_regular_samples(&tables)?;
    if samples.is_empty() {
        return Err("MP4 audio track has no samples".to_string());
    }

    let mut sample_entry = tables
        .sample_entry
        .ok_or_else(|| "MP4 audio track has no supported sample entry".to_string())?;
    if sample_entry.codec == AudioCodec::Aac {
        if let Some((sample_rate, channels)) = parse_asc_audio_config(&sample_entry.codec_private) {
            sample_entry.sample_rate = sample_rate;
            sample_entry.channels = channels;
        }
    }
    if sample_entry.codec == AudioCodec::Pcm {
        samples = coalesce_regular_pcm_samples(samples)?;
    }

    Ok(Some(RegularMp4AudioTrack {
        track_id,
        timescale: tables
            .timescale
            .ok_or_else(|| "MP4 audio track has no timescale".to_string())?,
        sample_rate: sample_entry.sample_rate,
        channels: sample_entry.channels,
        bits_per_sample: sample_entry.bits_per_sample,
        codec: sample_entry.codec,
        codec_id: sample_entry.codec_id,
        packet_format: sample_entry.packet_format,
        pcm_endianness: sample_entry.pcm_endianness,
        pcm_float: sample_entry.pcm_float,
        pcm_signed: sample_entry.pcm_signed,
        pcm_packed: sample_entry.pcm_packed,
        pcm_aligned_high: sample_entry.pcm_aligned_high,
        pcm_interleaved: sample_entry.pcm_interleaved,
        pcm_bytes_per_frame: sample_entry.pcm_bytes_per_frame,
        pcm_frames_per_packet: sample_entry.pcm_frames_per_packet,
        codec_private: sample_entry.codec_private,
        samples,
    }))
}

#[cfg(feature = "mp4")]
fn coalesce_regular_pcm_samples(
    samples: Vec<RegularMp4Sample>,
) -> Result<Vec<RegularMp4Sample>, String> {
    const MAX_PCM_PACKET_FRAMES: u32 = 4096;
    const MAX_PCM_PACKET_BYTES: u32 = 1024 * 1024;

    let mut output: Vec<RegularMp4Sample> =
        Vec::with_capacity(samples.len().div_ceil(MAX_PCM_PACKET_FRAMES as usize));
    for sample in samples {
        let can_merge = output.last().is_some_and(|current| {
            let current_presentation = i128::from(current.start_time)
                + i128::from(current.rendering_offset)
                + i128::from(current.duration);
            let sample_presentation =
                i128::from(sample.start_time) + i128::from(sample.rendering_offset);
            current.absolute_offset.checked_add(current.size as u64) == Some(sample.absolute_offset)
                && current.start_time.checked_add(current.duration as u64)
                    == Some(sample.start_time)
                && current_presentation == sample_presentation
                && current
                    .duration
                    .checked_add(sample.duration)
                    .is_some_and(|duration| duration <= MAX_PCM_PACKET_FRAMES)
                && current
                    .size
                    .checked_add(sample.size)
                    .is_some_and(|size| size <= MAX_PCM_PACKET_BYTES)
        });
        if can_merge {
            let current = output.last_mut().expect("checked above");
            current.size = current
                .size
                .checked_add(sample.size)
                .ok_or_else(|| "PCM packet size overflow".to_string())?;
            current.duration = current
                .duration
                .checked_add(sample.duration)
                .ok_or_else(|| "PCM packet duration overflow".to_string())?;
            current.is_sync &= sample.is_sync;
        } else {
            output.push(sample);
        }
    }
    output.shrink_to_fit();
    Ok(output)
}

#[cfg(feature = "mp4")]
fn build_regular_samples(tables: &RegularTrakTables) -> Result<Vec<RegularMp4Sample>, String> {
    if tables.sample_sizes.is_empty() {
        return Err("MP4 audio track is missing stsz sample sizes".to_string());
    }
    if tables.chunk_offsets.is_empty() {
        return Err("MP4 audio track is missing stco/co64 chunk offsets".to_string());
    }
    if tables.stsc.is_empty() {
        return Err("MP4 audio track is missing stsc sample-to-chunk table".to_string());
    }

    let sample_count = tables.sample_sizes.len();
    u32::try_from(sample_count).map_err(|_| "MP4 sample count exceeds u32".to_string())?;
    let timing_sample_count = tables.stts.iter().try_fold(0u64, |total, entry| {
        total
            .checked_add(u64::from(entry.sample_count))
            .ok_or_else(|| "MP4 stts sample count overflow".to_string())
    })?;
    if timing_sample_count != sample_count as u64 {
        return Err(format!(
            "MP4 stts describes {timing_sample_count} samples, but stsz describes {sample_count}"
        ));
    }
    if !tables.ctts.is_empty() {
        let composition_sample_count = tables.ctts.iter().try_fold(0u64, |total, entry| {
            total
                .checked_add(u64::from(entry.sample_count))
                .ok_or_else(|| "MP4 ctts sample count overflow".to_string())
        })?;
        if composition_sample_count != sample_count as u64 {
            return Err(format!(
                "MP4 ctts describes {composition_sample_count} samples, but stsz describes {sample_count}"
            ));
        }
    }
    if tables
        .sync_samples
        .last()
        .is_some_and(|sample_id| *sample_id as usize > sample_count)
    {
        return Err("MP4 stss references a sample beyond stsz".to_string());
    }
    if tables
        .stsc
        .iter()
        .any(|entry| entry.sample_description_index != 1)
    {
        return Err(
            "MP4 stsc references a sample description other than the parsed first entry"
                .to_string(),
        );
    }

    let chunk_count = tables.chunk_offsets.len();
    let mapped_sample_count =
        tables
            .stsc
            .iter()
            .enumerate()
            .try_fold(0u64, |total, (index, entry)| {
                let first_chunk = entry.first_chunk as usize;
                if first_chunk > chunk_count {
                    return Err("MP4 stsc references a chunk beyond stco/co64".to_string());
                }
                let end_chunk_exclusive = tables
                    .stsc
                    .get(index + 1)
                    .map(|next| next.first_chunk as usize)
                    .unwrap_or(chunk_count + 1);
                let described_chunks = end_chunk_exclusive
                    .checked_sub(first_chunk)
                    .ok_or_else(|| "MP4 stsc chunk range underflow".to_string())?;
                let described_samples = (described_chunks as u64)
                    .checked_mul(u64::from(entry.samples_per_chunk))
                    .ok_or_else(|| "MP4 stsc sample count overflow".to_string())?;
                total
                    .checked_add(described_samples)
                    .ok_or_else(|| "MP4 stsc sample count overflow".to_string())
            })?;
    if mapped_sample_count != sample_count as u64 {
        return Err(format!(
            "MP4 stsc maps {mapped_sample_count} samples, but stsz describes {sample_count}"
        ));
    }

    let is_pcm = tables
        .sample_entry
        .as_ref()
        .is_some_and(|entry| entry.codec == AudioCodec::Pcm);
    if is_pcm
        && tables.sample_sizes.constant().is_some()
        && tables.stts.len() == 1
        && tables.ctts.is_empty()
        && tables.sync_samples.is_empty()
    {
        return build_constant_pcm_samples(tables);
    }
    if sample_count > MAX_MP4_MATERIALIZED_ACCESS_UNITS {
        return Err(format!(
            "MP4 track has {sample_count} access units; the materialized limit is {MAX_MP4_MATERIALIZED_ACCESS_UNITS}"
        ));
    }

    let mut samples = Vec::new();
    samples
        .try_reserve_exact(sample_count)
        .map_err(|_| format!("MP4 sample index allocation failed for {sample_count} records"))?;
    let mut sample_index = 0usize;
    let mut stsc_index = 0usize;
    let mut decode_time = 0u64;
    let mut stts_reader = TimeToSampleReader::new(&tables.stts);
    let mut ctts_reader = CompositionOffsetReader::new(&tables.ctts);

    for (chunk_index, chunk_offset) in tables.chunk_offsets.iter().copied().enumerate() {
        let chunk_number = chunk_index as u32 + 1;
        while stsc_index + 1 < tables.stsc.len()
            && tables.stsc[stsc_index + 1].first_chunk <= chunk_number
        {
            stsc_index += 1;
        }
        let samples_per_chunk = tables.stsc[stsc_index].samples_per_chunk as usize;
        let mut sample_offset = chunk_offset;

        for _ in 0..samples_per_chunk {
            if sample_index >= tables.sample_sizes.len() {
                break;
            }
            let size = tables
                .sample_sizes
                .get(sample_index)
                .ok_or_else(|| "MP4 stsz ended before stsc".to_string())?;
            if size > MAX_MEDIA_PACKET_BYTES {
                return Err(format!(
                    "MP4 sample exceeds the {MAX_MEDIA_PACKET_BYTES} byte packet budget"
                ));
            }
            let duration = stts_reader
                .next_duration()
                .ok_or_else(|| "MP4 stts ended before stsz".to_string())?;
            let rendering_offset = ctts_reader
                .next_offset()
                .ok_or_else(|| "MP4 ctts ended before stsz".to_string())?;
            let sample_id = u32::try_from(sample_index + 1)
                .map_err(|_| "MP4 sample identifier exceeds u32".to_string())?;
            let is_sync = tables.sync_samples.is_empty()
                || tables.sync_samples.binary_search(&sample_id).is_ok();

            samples.push(RegularMp4Sample {
                sample_id,
                absolute_offset: sample_offset,
                size,
                duration,
                start_time: decode_time,
                rendering_offset,
                is_sync,
            });

            sample_offset = sample_offset
                .checked_add(size as u64)
                .ok_or_else(|| "MP4 sample offset overflow".to_string())?;
            decode_time = decode_time
                .checked_add(duration as u64)
                .ok_or_else(|| "MP4 sample timestamp overflow".to_string())?;
            sample_index += 1;
        }
    }

    if samples.len() != tables.sample_sizes.len() {
        return Err(format!(
            "MP4 sample tables described {} samples but only {} were mapped to chunks",
            tables.sample_sizes.len(),
            samples.len()
        ));
    }

    Ok(samples)
}

#[cfg(feature = "mp4")]
fn build_constant_pcm_samples(tables: &RegularTrakTables) -> Result<Vec<RegularMp4Sample>, String> {
    const MAX_PCM_PACKET_FRAMES: u32 = 4096;
    const MAX_PCM_PACKET_BYTES: u32 = 1024 * 1024;

    let (sample_size, sample_count) = tables
        .sample_sizes
        .constant()
        .ok_or_else(|| "constant PCM index requires a constant stsz".to_string())?;
    if sample_size == 0 {
        return Err("PCM stsz contains a zero sample size".to_string());
    }
    let sample_duration = tables
        .stts
        .first()
        .ok_or_else(|| "PCM track is missing stts timing".to_string())?
        .sample_duration;
    let frames_by_bytes = (MAX_PCM_PACKET_BYTES / sample_size).max(1);
    let frames_by_duration = u32::MAX
        .checked_div(sample_duration)
        .unwrap_or(MAX_PCM_PACKET_FRAMES)
        .max(1);
    let max_packet_frames = MAX_PCM_PACKET_FRAMES
        .min(frames_by_bytes)
        .min(frames_by_duration);
    let estimated_packets = (sample_count as usize)
        .div_ceil(max_packet_frames as usize)
        .checked_add(tables.chunk_offsets.len())
        .ok_or_else(|| "PCM packet index size overflow".to_string())?;
    if estimated_packets > MAX_MP4_MATERIALIZED_ACCESS_UNITS {
        return Err(format!(
            "PCM track requires more than {MAX_MP4_MATERIALIZED_ACCESS_UNITS} indexed packet runs"
        ));
    }

    let mut samples = Vec::new();
    samples
        .try_reserve_exact(estimated_packets.min(sample_count as usize))
        .map_err(|_| "PCM packet index allocation failed".to_string())?;
    let mut sample_index = 0u32;
    let mut stsc_index = 0usize;
    let mut decode_time = 0u64;

    for (chunk_index, chunk_offset) in tables.chunk_offsets.iter().copied().enumerate() {
        let chunk_number = chunk_index as u32 + 1;
        while stsc_index + 1 < tables.stsc.len()
            && tables.stsc[stsc_index + 1].first_chunk <= chunk_number
        {
            stsc_index += 1;
        }
        let mut frames_remaining = tables.stsc[stsc_index].samples_per_chunk;
        let mut sample_offset = chunk_offset;
        while frames_remaining > 0 {
            let frame_count = frames_remaining.min(max_packet_frames);
            let size = sample_size
                .checked_mul(frame_count)
                .ok_or_else(|| "PCM packet size overflow".to_string())?;
            let duration = sample_duration
                .checked_mul(frame_count)
                .ok_or_else(|| "PCM packet duration overflow".to_string())?;
            let sample_id = sample_index
                .checked_add(1)
                .ok_or_else(|| "PCM sample identifier overflow".to_string())?;
            samples.push(RegularMp4Sample {
                sample_id,
                absolute_offset: sample_offset,
                size,
                duration,
                start_time: decode_time,
                rendering_offset: 0,
                is_sync: true,
            });
            sample_offset = sample_offset
                .checked_add(u64::from(size))
                .ok_or_else(|| "PCM sample offset overflow".to_string())?;
            decode_time = decode_time
                .checked_add(u64::from(duration))
                .ok_or_else(|| "PCM sample timestamp overflow".to_string())?;
            sample_index = sample_index
                .checked_add(frame_count)
                .ok_or_else(|| "PCM sample count overflow".to_string())?;
            frames_remaining -= frame_count;
        }
    }

    if sample_index != sample_count {
        return Err(format!(
            "MP4 stsc maps {sample_index} PCM frames, but stsz describes {sample_count}"
        ));
    }
    Ok(samples)
}

#[cfg(feature = "mp4")]
struct TimeToSampleReader<'a> {
    entries: &'a [SttsEntry],
    index: usize,
    remaining: u32,
}

#[cfg(feature = "mp4")]
impl<'a> TimeToSampleReader<'a> {
    fn new(entries: &'a [SttsEntry]) -> Self {
        Self {
            entries,
            index: 0,
            remaining: 0,
        }
    }

    fn next_duration(&mut self) -> Option<u32> {
        if self.remaining == 0 {
            let entry = self.entries.get(self.index)?;
            self.index += 1;
            self.remaining = entry.sample_count;
        }
        self.remaining = self.remaining.saturating_sub(1);
        self.entries
            .get(self.index.saturating_sub(1))
            .map(|entry| entry.sample_duration)
    }
}

#[cfg(feature = "mp4")]
struct CompositionOffsetReader<'a> {
    entries: &'a [CttsEntry],
    index: usize,
    remaining: u32,
}

#[cfg(feature = "mp4")]
impl<'a> CompositionOffsetReader<'a> {
    fn new(entries: &'a [CttsEntry]) -> Self {
        Self {
            entries,
            index: 0,
            remaining: 0,
        }
    }

    fn next_offset(&mut self) -> Option<i32> {
        if self.entries.is_empty() {
            return Some(0);
        }
        if self.remaining == 0 {
            let entry = self.entries.get(self.index)?;
            self.index += 1;
            self.remaining = entry.sample_count;
        }
        self.remaining = self.remaining.saturating_sub(1);
        self.entries
            .get(self.index.saturating_sub(1))
            .map(|entry| entry.sample_offset)
    }
}

#[cfg(feature = "mp4")]
fn parse_trex_defaults(data: &[u8]) -> Result<Vec<Fmp4TrackDefaults>, String> {
    let mut defaults = Vec::new();
    walk_boxes(data, &mut |header, payload| {
        if header.name == *b"trex" {
            if payload.len() < 24 {
                return Err("fragmented MP4 trex box is truncated".to_string());
            }
            defaults
                .try_reserve(1)
                .map_err(|_| "fragmented MP4 trex allocation failed".to_string())?;
            defaults.push(Fmp4TrackDefaults {
                track_id: be_u32(payload, 4)
                    .ok_or_else(|| "fragmented MP4 trex track id is truncated".to_string())?,
                sample_description_index: be_u32(payload, 8).ok_or_else(|| {
                    "fragmented MP4 trex sample description is truncated".to_string()
                })?,
                default_sample_duration: Some(be_u32(payload, 12).ok_or_else(|| {
                    "fragmented MP4 trex sample duration is truncated".to_string()
                })?)
                .filter(|value| *value > 0),
                default_sample_size: Some(
                    be_u32(payload, 16).ok_or_else(|| {
                        "fragmented MP4 trex sample size is truncated".to_string()
                    })?,
                )
                .filter(|value| *value > 0),
            });
        }
        Ok(())
    })?;
    Ok(defaults)
}

#[cfg(feature = "mp4")]
#[derive(Clone, Debug)]
struct RegularAudioSampleEntry {
    sample_rate: u32,
    channels: u8,
    bits_per_sample: Option<u8>,
    codec: AudioCodec,
    codec_id: String,
    packet_format: AudioPacketFormat,
    pcm_endianness: Option<PcmEndianness>,
    pcm_float: Option<bool>,
    pcm_signed: Option<bool>,
    pcm_packed: Option<bool>,
    pcm_aligned_high: Option<bool>,
    pcm_interleaved: Option<bool>,
    pcm_bytes_per_frame: Option<u32>,
    pcm_frames_per_packet: Option<u32>,
    codec_private: Vec<u8>,
}

#[cfg(feature = "mp4")]
fn parse_tkhd_track_id(data: &[u8]) -> Option<u32> {
    let version = *data.first()?;
    let offset = if version == 1 { 20 } else { 12 };
    be_u32(data, offset)
}

#[cfg(feature = "mp4")]
fn parse_mdhd_timescale(data: &[u8]) -> Option<u32> {
    let version = *data.first()?;
    let offset = if version == 1 { 20 } else { 12 };
    be_u32(data, offset)
}

#[cfg(feature = "mp4")]
fn parse_elst(data: &[u8]) -> Result<Vec<Mp4EditListEntry>, String> {
    let version = *data
        .first()
        .ok_or_else(|| "MP4 elst box is truncated".to_string())?;
    let entry_count =
        be_u32(data, 4).ok_or_else(|| "MP4 elst entry count is truncated".to_string())?;
    let entry_size = match version {
        0 => 12usize,
        1 => 20usize,
        _ => return Err(format!("unsupported MP4 elst version {version}")),
    };
    let required = 8usize
        .checked_add(
            (entry_count as usize)
                .checked_mul(entry_size)
                .ok_or_else(|| "MP4 elst entry size overflow".to_string())?,
        )
        .ok_or_else(|| "MP4 elst size overflow".to_string())?;
    if data.len() < required {
        return Err(format!(
            "MP4 elst expected {required} bytes, got {}",
            data.len()
        ));
    }

    let mut entries = Vec::new();
    entries
        .try_reserve_exact(entry_count as usize)
        .map_err(|_| format!("MP4 elst allocation failed for {entry_count} entries"))?;
    let mut pos = 8usize;
    for _ in 0..entry_count {
        let (segment_duration, media_time) = if version == 1 {
            (
                be_u64(data, pos).ok_or_else(|| "MP4 elst duration is truncated".to_string())?,
                be_u64(data, pos + 8)
                    .ok_or_else(|| "MP4 elst media time is truncated".to_string())?
                    as i64,
            )
        } else {
            (
                u64::from(
                    be_u32(data, pos)
                        .ok_or_else(|| "MP4 elst duration is truncated".to_string())?,
                ),
                i64::from(
                    be_i32(data, pos + 4)
                        .ok_or_else(|| "MP4 elst media time is truncated".to_string())?,
                ),
            )
        };
        let rate_pos = pos + if version == 1 { 16 } else { 8 };
        entries.push(Mp4EditListEntry {
            segment_duration,
            media_time,
            media_rate_integer: be_u16(data, rate_pos)
                .ok_or_else(|| "MP4 elst media rate is truncated".to_string())?
                as i16,
            media_rate_fraction: be_u16(data, rate_pos + 2)
                .ok_or_else(|| "MP4 elst media rate fraction is truncated".to_string())?
                as i16,
        });
        pos += entry_size;
    }
    Ok(entries)
}

#[cfg(feature = "mp4")]
fn resolve_media_timeline(
    entries: &[Mp4EditListEntry],
    movie_timescale: Option<u32>,
    track_timescale: u32,
) -> Result<Vec<MediaTrackTimeline>, String> {
    if entries.is_empty() {
        return Ok(Vec::new());
    }
    let movie_timescale = movie_timescale
        .filter(|value| *value > 0)
        .ok_or_else(|| "MP4 edit list requires a non-zero movie timescale".to_string())?;
    let mut presentation_start = 0u64;
    let mut timeline = Vec::new();
    timeline
        .try_reserve_exact(entries.len())
        .map_err(|_| "MP4 edit timeline allocation failed".to_string())?;
    for entry in entries {
        if entry.media_rate_integer != 1 || entry.media_rate_fraction != 0 {
            return Err(format!(
                "unsupported MP4 edit rate {}+{}/65536",
                entry.media_rate_integer, entry.media_rate_fraction
            ));
        }
        let duration = scale_mp4_time(entry.segment_duration, movie_timescale, track_timescale)?;
        if entry.media_time == -1 {
            presentation_start = presentation_start
                .checked_add(duration)
                .ok_or_else(|| "MP4 edit presentation time overflow".to_string())?;
            continue;
        }
        if entry.media_time < 0 {
            return Err(format!(
                "unsupported MP4 edit media time {}",
                entry.media_time
            ));
        }
        timeline.push(MediaTrackTimeline {
            presentation_start,
            media_start: entry.media_time as u64,
            duration,
        });
        presentation_start = presentation_start
            .checked_add(duration)
            .ok_or_else(|| "MP4 edit presentation time overflow".to_string())?;
    }
    if timeline.is_empty() {
        return Err("MP4 edit list contains no media edit".to_string());
    }
    Ok(timeline)
}

#[cfg(feature = "mp4")]
fn map_mp4_presentation_time(
    edits: &[MediaTrackTimeline],
    media_presentation_time: i128,
) -> Result<i64, String> {
    let Some(first) = edits.first() else {
        return i64::try_from(media_presentation_time)
            .map_err(|_| "MP4 presentation timestamp overflow".to_string());
    };
    let selected = edits
        .iter()
        .find(|edit| {
            let start = i128::from(edit.media_start);
            let end = start + i128::from(edit.duration);
            media_presentation_time >= start && media_presentation_time < end
        })
        .or_else(|| {
            edits
                .iter()
                .find(|edit| media_presentation_time < i128::from(edit.media_start))
        })
        .unwrap_or_else(|| edits.last().unwrap_or(first));
    media_presentation_time
        .checked_sub(i128::from(selected.media_start))
        .and_then(|value| value.checked_add(i128::from(selected.presentation_start)))
        .and_then(|value| i64::try_from(value).ok())
        .ok_or_else(|| "MP4 edited presentation timestamp overflow".to_string())
}

#[cfg(feature = "mp4")]
fn scale_mp4_time(value: u64, from_timescale: u32, to_timescale: u32) -> Result<u64, String> {
    let numerator = u128::from(value)
        .checked_mul(u128::from(to_timescale))
        .ok_or_else(|| "MP4 edit duration overflow".to_string())?;
    let rounded = numerator
        .checked_add(u128::from(from_timescale / 2))
        .ok_or_else(|| "MP4 edit duration overflow".to_string())?
        / u128::from(from_timescale);
    u64::try_from(rounded).map_err(|_| "MP4 edit duration exceeds u64".to_string())
}

#[cfg(feature = "mp4")]
fn parse_hdlr_is_audio(data: &[u8]) -> bool {
    data.len() >= 12 && &data[8..12] == b"soun"
}

#[cfg(feature = "mp4")]
fn parse_stsd_audio(data: &[u8]) -> Result<Option<RegularAudioSampleEntry>, String> {
    if data.len() < 16 {
        return Err("MP4 stsd audio table is truncated".to_string());
    }
    let entry_count =
        be_u32(data, 4).ok_or_else(|| "MP4 stsd entry count is truncated".to_string())?;
    let mut pos = 8usize;
    for _ in 0..entry_count {
        let remaining = data
            .get(pos..)
            .ok_or_else(|| "MP4 stsd entry offset is invalid".to_string())?;
        let header = Mp4BoxHeader::read(remaining)
            .ok_or_else(|| "MP4 stsd audio entry header is truncated".to_string())?;
        let end = pos
            .checked_add(header.size)
            .ok_or_else(|| "MP4 stsd audio entry size overflow".to_string())?;
        if end > data.len() {
            return Err("MP4 stsd audio entry is truncated".to_string());
        }
        let payload = &data[pos + header.header_size..end];
        if !matches!(
            &header.name,
            b"mp4a"
                | b"fLaC"
                | b"alac"
                | b"sowt"
                | b"in24"
                | b"in32"
                | b"twos"
                | b"fl32"
                | b"fl64"
                | b"lpcm"
        ) {
            pos += header.size;
            continue;
        }
        if payload.len() >= 28 {
            let version = be_u16(payload, 8)
                .ok_or_else(|| "QuickTime audio sample-entry version is truncated".to_string())?;
            let base_channels = u8::try_from(
                be_u16(payload, 16)
                    .ok_or_else(|| "QuickTime audio channel count is truncated".to_string())?,
            )
            .map_err(|_| "QuickTime audio channel count exceeds u8".to_string())?;
            let base_bits = be_u16(payload, 18)
                .and_then(|value| u8::try_from(value).ok())
                .filter(|value| *value > 0);
            let base_sample_rate = be_u32(payload, 24)
                .ok_or_else(|| "QuickTime audio sample rate is truncated".to_string())?
                >> 16;
            let (
                sample_rate,
                channels,
                declared_bits,
                bytes_per_frame,
                frames_per_packet,
                lpcm_flags,
            ) = match version {
                0 => (
                    base_sample_rate,
                    base_channels,
                    base_bits,
                    None,
                    Some(1),
                    None,
                ),
                1 => {
                    if payload.len() < 44 {
                        return Err(
                            "QuickTime version-one audio sample entry is truncated".to_string()
                        );
                    }
                    let frames_per_packet = be_u32(payload, 28).filter(|value| *value > 0);
                    let bytes_per_frame = be_u32(payload, 36).filter(|value| *value > 0);
                    let bits = base_bits.or_else(|| {
                        be_u32(payload, 40)
                            .and_then(|value| value.checked_mul(8))
                            .and_then(|value| u8::try_from(value).ok())
                    });
                    (
                        base_sample_rate,
                        base_channels,
                        bits,
                        bytes_per_frame,
                        frames_per_packet,
                        None,
                    )
                }
                2 => {
                    if payload.len() < 64 {
                        return Err(
                            "QuickTime version-two audio sample entry is truncated".to_string()
                        );
                    }
                    let rate = f64::from_bits(be_u64(payload, 32).ok_or_else(|| {
                        "QuickTime version-two audio sample rate is truncated".to_string()
                    })?);
                    if !rate.is_finite() || rate <= 0.0 || rate > u32::MAX as f64 {
                        return Err(format!("invalid QuickTime version-two sample rate {rate}"));
                    }
                    let channels = u8::try_from(be_u32(payload, 40).ok_or_else(|| {
                        "QuickTime version-two channel count is truncated".to_string()
                    })?)
                    .map_err(|_| "QuickTime version-two channel count exceeds u8".to_string())?;
                    let bits = u8::try_from(be_u32(payload, 48).ok_or_else(|| {
                        "QuickTime version-two sample depth is truncated".to_string()
                    })?)
                    .map_err(|_| "QuickTime version-two sample depth exceeds u8".to_string())?;
                    let flags = be_u32(payload, 52).ok_or_else(|| {
                        "QuickTime version-two LPCM flags are truncated".to_string()
                    })?;
                    let bytes_per_packet = be_u32(payload, 56).filter(|value| *value > 0);
                    let frames_per_packet = be_u32(payload, 60).filter(|value| *value > 0);
                    let bytes_per_frame =
                        bytes_per_packet
                            .zip(frames_per_packet)
                            .and_then(|(bytes, frames)| {
                                (bytes % frames == 0).then_some(bytes / frames)
                            });
                    (
                        rate.round() as u32,
                        channels,
                        Some(bits),
                        bytes_per_frame,
                        frames_per_packet,
                        Some(flags),
                    )
                }
                _ => {
                    return Err(format!(
                        "unsupported QuickTime audio sample-entry version {version}"
                    ))
                }
            };
            let mp4a_codec_private = (header.name == *b"mp4a").then(|| {
                find_audio_sample_entry_private(payload, b"esds")
                    .and_then(parse_esds_audio_specific_config)
                    .unwrap_or_default()
            });
            let (sample_rate, channels) = mp4a_codec_private
                .as_deref()
                .and_then(parse_asc_audio_config)
                .unwrap_or((sample_rate, channels));
            if sample_rate == 0 || channels == 0 {
                return Err(
                    "QuickTime audio sample entry has zero sample rate or channels".to_string(),
                );
            }
            if header.name == *b"mp4a" {
                let codec_private = mp4a_codec_private.unwrap_or_default();
                return Ok(Some(RegularAudioSampleEntry {
                    sample_rate,
                    channels,
                    bits_per_sample: None,
                    codec: AudioCodec::Aac,
                    codec_id: "mp4a".to_string(),
                    packet_format: AudioPacketFormat::Adts,
                    pcm_endianness: None,
                    pcm_float: None,
                    pcm_signed: None,
                    pcm_packed: None,
                    pcm_aligned_high: None,
                    pcm_interleaved: None,
                    pcm_bytes_per_frame: None,
                    pcm_frames_per_packet: None,
                    codec_private,
                }));
            }
            if header.name == *b"fLaC" {
                let codec_private = find_audio_sample_entry_private(payload, b"dfLa")
                    .map(ToOwned::to_owned)
                    .unwrap_or_default();
                return Ok(Some(RegularAudioSampleEntry {
                    sample_rate,
                    channels,
                    bits_per_sample: declared_bits,
                    codec: AudioCodec::Flac,
                    codec_id: "fLaC".to_string(),
                    packet_format: AudioPacketFormat::Raw,
                    pcm_endianness: None,
                    pcm_float: None,
                    pcm_signed: None,
                    pcm_packed: None,
                    pcm_aligned_high: None,
                    pcm_interleaved: None,
                    pcm_bytes_per_frame: None,
                    pcm_frames_per_packet: None,
                    codec_private,
                }));
            }
            if header.name == *b"alac" {
                let codec_private = find_audio_sample_entry_private(payload, b"alac")
                    .map(ToOwned::to_owned)
                    .unwrap_or_default();
                return Ok(Some(RegularAudioSampleEntry {
                    sample_rate,
                    channels,
                    bits_per_sample: declared_bits,
                    codec: AudioCodec::Alac,
                    codec_id: "alac".to_string(),
                    packet_format: AudioPacketFormat::Raw,
                    pcm_endianness: None,
                    pcm_float: None,
                    pcm_signed: None,
                    pcm_packed: None,
                    pcm_aligned_high: None,
                    pcm_interleaved: None,
                    pcm_bytes_per_frame: None,
                    pcm_frames_per_packet: None,
                    codec_private,
                }));
            }
            let enda = find_audio_sample_entry_private(payload, b"enda")
                .map(|data| {
                    let flag = be_u16(data, 0)
                        .ok_or_else(|| "QuickTime enda atom is truncated".to_string())?;
                    match flag {
                        0 => Ok(PcmEndianness::Big),
                        1 => Ok(PcmEndianness::Little),
                        _ => Err(format!("invalid QuickTime enda flag {flag}")),
                    }
                })
                .transpose()?;
            let (endianness, float, signed, packed, aligned_high, interleaved, bits_per_sample) =
                match &header.name {
                    b"sowt" => (
                        PcmEndianness::Little,
                        false,
                        true,
                        true,
                        false,
                        true,
                        declared_bits,
                    ),
                    b"in24" => (
                        enda.unwrap_or(PcmEndianness::Big),
                        false,
                        true,
                        true,
                        false,
                        true,
                        Some(24),
                    ),
                    b"in32" => (
                        enda.unwrap_or(PcmEndianness::Big),
                        false,
                        true,
                        true,
                        false,
                        true,
                        Some(32),
                    ),
                    b"twos" => (
                        PcmEndianness::Big,
                        false,
                        true,
                        true,
                        false,
                        true,
                        declared_bits,
                    ),
                    b"fl32" => (
                        enda.unwrap_or(PcmEndianness::Big),
                        true,
                        false,
                        true,
                        false,
                        true,
                        Some(32),
                    ),
                    b"fl64" => (
                        enda.unwrap_or(PcmEndianness::Big),
                        true,
                        false,
                        true,
                        false,
                        true,
                        Some(64),
                    ),
                    b"lpcm" => {
                        let flags = lpcm_flags.ok_or_else(|| {
                            "QuickTime lpcm requires a version-two sample entry".to_string()
                        })?;
                        let interleaved = flags & (1 << 5) == 0;
                        if !interleaved {
                            return Err("non-interleaved QuickTime LPCM is unsupported".to_string());
                        }
                        (
                            if flags & (1 << 1) != 0 {
                                PcmEndianness::Big
                            } else {
                                PcmEndianness::Little
                            },
                            flags & 1 != 0,
                            flags & (1 << 2) != 0,
                            flags & (1 << 3) != 0,
                            flags & (1 << 4) != 0,
                            interleaved,
                            declared_bits,
                        )
                    }
                    _ => {
                        pos += header.size;
                        continue;
                    }
                };
            let bytes_per_frame = bytes_per_frame.or_else(|| {
                bits_per_sample
                    .and_then(|bits| u32::from(bits).div_ceil(8).checked_mul(u32::from(channels)))
            });
            return Ok(Some(RegularAudioSampleEntry {
                sample_rate,
                channels,
                bits_per_sample,
                codec: AudioCodec::Pcm,
                codec_id: String::from_utf8_lossy(&header.name).into_owned(),
                packet_format: AudioPacketFormat::Raw,
                pcm_endianness: Some(endianness),
                pcm_float: Some(float),
                pcm_signed: Some(signed),
                pcm_packed: Some(packed),
                pcm_aligned_high: Some(aligned_high),
                pcm_interleaved: Some(interleaved),
                pcm_bytes_per_frame: bytes_per_frame,
                pcm_frames_per_packet: frames_per_packet,
                codec_private: Vec::new(),
            }));
        }
        pos += header.size;
    }
    Ok(None)
}

#[cfg(feature = "mp4")]
fn find_audio_sample_entry_private<'a>(payload: &'a [u8], target: &[u8; 4]) -> Option<&'a [u8]> {
    let version = be_u16(payload, 8).unwrap_or(0);
    let children_offset = match version {
        0 => 28,
        1 => 44,
        2 => 64,
        _ => 28,
    };
    find_sample_entry_box(payload.get(children_offset..)?, target)
}

#[cfg(feature = "mp4")]
fn find_sample_entry_box<'a>(data: &'a [u8], target: &[u8; 4]) -> Option<&'a [u8]> {
    let mut pos = 0usize;
    while pos + 8 <= data.len() {
        let header = Mp4BoxHeader::read(&data[pos..])?;
        let end = pos.checked_add(header.size)?;
        if end > data.len() {
            return None;
        }
        let payload = &data[pos + header.header_size..end];
        if &header.name == target {
            return Some(payload);
        }
        if matches!(&header.name, b"wave" | b"sinf" | b"schi") {
            if let Some(found) = find_sample_entry_box(payload, target) {
                return Some(found);
            }
        }
        pos = end;
    }
    None
}

#[cfg(feature = "mp4")]
fn validate_mp4_table_payload(
    data: &[u8],
    header_size: usize,
    entry_count: u32,
    entry_size: usize,
    name: &str,
) -> Result<usize, String> {
    let entries_size = (entry_count as usize)
        .checked_mul(entry_size)
        .ok_or_else(|| format!("MP4 {name} table size overflow"))?;
    let required = header_size
        .checked_add(entries_size)
        .ok_or_else(|| format!("MP4 {name} table size overflow"))?;
    if data.len() < required {
        return Err(format!(
            "MP4 {name} declares {entry_count} entries requiring {required} bytes, but the box has {}",
            data.len()
        ));
    }
    Ok(entry_count as usize)
}

#[cfg(feature = "mp4")]
fn reserve_mp4_table<T>(entries: &mut Vec<T>, count: usize, name: &str) -> Result<(), String> {
    entries
        .try_reserve_exact(count)
        .map_err(|_| format!("MP4 {name} table allocation failed for {count} entries"))
}

#[cfg(feature = "mp4")]
fn parse_stts(data: &[u8]) -> Result<Vec<SttsEntry>, String> {
    let entry_count = be_u32(data, 4).ok_or_else(|| "MP4 stts header is truncated".to_string())?;
    let count = validate_mp4_table_payload(data, 8, entry_count, 8, "stts")?;
    let mut entries = Vec::new();
    reserve_mp4_table(&mut entries, count, "stts")?;
    let mut pos = 8usize;
    for _ in 0..entry_count {
        let sample_count = be_u32(data, pos).ok_or_else(|| "MP4 stts is truncated".to_string())?;
        let sample_duration =
            be_u32(data, pos + 4).ok_or_else(|| "MP4 stts is truncated".to_string())?;
        if sample_count == 0 {
            return Err("MP4 stts contains a zero sample count".to_string());
        }
        entries.push(SttsEntry {
            sample_count,
            sample_duration,
        });
        pos += 8;
    }
    Ok(entries)
}

#[cfg(feature = "mp4")]
fn parse_ctts(data: &[u8]) -> Result<Vec<CttsEntry>, String> {
    let version = data
        .first()
        .copied()
        .ok_or_else(|| "MP4 ctts header is truncated".to_string())?;
    if version > 1 {
        return Err(format!("unsupported MP4 ctts version {version}"));
    }
    let entry_count = be_u32(data, 4).ok_or_else(|| "MP4 ctts header is truncated".to_string())?;
    let count = validate_mp4_table_payload(data, 8, entry_count, 8, "ctts")?;
    let mut entries = Vec::new();
    reserve_mp4_table(&mut entries, count, "ctts")?;
    let mut pos = 8usize;
    for _ in 0..entry_count {
        let sample_count = be_u32(data, pos).ok_or_else(|| "MP4 ctts is truncated".to_string())?;
        let raw_offset =
            be_u32(data, pos + 4).ok_or_else(|| "MP4 ctts is truncated".to_string())?;
        if sample_count == 0 {
            return Err("MP4 ctts contains a zero sample count".to_string());
        }
        let sample_offset = if version == 1 {
            raw_offset as i32
        } else {
            i32::try_from(raw_offset)
                .map_err(|_| "MP4 ctts version 0 offset exceeds i32".to_string())?
        };
        entries.push(CttsEntry {
            sample_count,
            sample_offset,
        });
        pos += 8;
    }
    Ok(entries)
}

#[cfg(feature = "mp4")]
fn parse_stsc(data: &[u8]) -> Result<Vec<StscEntry>, String> {
    let entry_count = be_u32(data, 4).ok_or_else(|| "MP4 stsc header is truncated".to_string())?;
    let count = validate_mp4_table_payload(data, 8, entry_count, 12, "stsc")?;
    let mut entries = Vec::new();
    reserve_mp4_table(&mut entries, count, "stsc")?;
    let mut pos = 8usize;
    let mut previous_first_chunk = 0u32;
    for _ in 0..entry_count {
        let first_chunk = be_u32(data, pos).ok_or_else(|| "MP4 stsc is truncated".to_string())?;
        let samples_per_chunk =
            be_u32(data, pos + 4).ok_or_else(|| "MP4 stsc is truncated".to_string())?;
        let sample_description_index =
            be_u32(data, pos + 8).ok_or_else(|| "MP4 stsc is truncated".to_string())?;
        if first_chunk == 0 || first_chunk <= previous_first_chunk {
            return Err("MP4 stsc first_chunk values must increase from one".to_string());
        }
        if samples_per_chunk == 0 || sample_description_index == 0 {
            return Err(
                "MP4 stsc contains a zero samples-per-chunk or description index".to_string(),
            );
        }
        entries.push(StscEntry {
            first_chunk,
            samples_per_chunk,
            sample_description_index,
        });
        previous_first_chunk = first_chunk;
        pos += 12;
    }
    if entries.first().is_some_and(|entry| entry.first_chunk != 1) {
        return Err("MP4 stsc must begin with chunk one".to_string());
    }
    Ok(entries)
}

#[cfg(feature = "mp4")]
fn parse_stsz(data: &[u8]) -> Result<Mp4SampleSizes, String> {
    let sample_size = be_u32(data, 4).ok_or_else(|| "MP4 stsz header is truncated".to_string())?;
    let sample_count = be_u32(data, 8).ok_or_else(|| "MP4 stsz header is truncated".to_string())?;
    if sample_size > 0 {
        return Ok(Mp4SampleSizes::Constant {
            size: sample_size,
            count: sample_count,
        });
    }
    let count = validate_mp4_table_payload(data, 12, sample_count, 4, "stsz")?;
    if count > MAX_MP4_MATERIALIZED_ACCESS_UNITS {
        return Err(format!(
            "MP4 stsz has {count} variable-size samples; the materialized access-unit limit is {MAX_MP4_MATERIALIZED_ACCESS_UNITS}"
        ));
    }
    let mut sizes = Vec::new();
    reserve_mp4_table(&mut sizes, count, "stsz")?;
    let mut pos = 12usize;
    for _ in 0..sample_count {
        let size = be_u32(data, pos).ok_or_else(|| "MP4 stsz is truncated".to_string())?;
        sizes.push(size);
        pos += 4;
    }
    Ok(Mp4SampleSizes::Variable(sizes))
}

#[cfg(feature = "mp4")]
fn parse_stco(data: &[u8]) -> Result<Vec<u64>, String> {
    let entry_count = be_u32(data, 4).ok_or_else(|| "MP4 stco header is truncated".to_string())?;
    let count = validate_mp4_table_payload(data, 8, entry_count, 4, "stco")?;
    let mut offsets = Vec::new();
    reserve_mp4_table(&mut offsets, count, "stco")?;
    let mut pos = 8usize;
    for _ in 0..entry_count {
        let offset = be_u32(data, pos).ok_or_else(|| "MP4 stco is truncated".to_string())?;
        offsets.push(offset as u64);
        pos += 4;
    }
    Ok(offsets)
}

#[cfg(feature = "mp4")]
fn parse_co64(data: &[u8]) -> Result<Vec<u64>, String> {
    let entry_count = be_u32(data, 4).ok_or_else(|| "MP4 co64 header is truncated".to_string())?;
    let count = validate_mp4_table_payload(data, 8, entry_count, 8, "co64")?;
    let mut offsets = Vec::new();
    reserve_mp4_table(&mut offsets, count, "co64")?;
    let mut pos = 8usize;
    for _ in 0..entry_count {
        let offset = be_u64(data, pos).ok_or_else(|| "MP4 co64 is truncated".to_string())?;
        offsets.push(offset);
        pos += 8;
    }
    Ok(offsets)
}

#[cfg(feature = "mp4")]
fn parse_stss(data: &[u8]) -> Result<Vec<u32>, String> {
    let entry_count = be_u32(data, 4).ok_or_else(|| "MP4 stss header is truncated".to_string())?;
    let count = validate_mp4_table_payload(data, 8, entry_count, 4, "stss")?;
    let mut samples = Vec::new();
    reserve_mp4_table(&mut samples, count, "stss")?;
    let mut pos = 8usize;
    let mut previous = 0u32;
    for _ in 0..entry_count {
        let sample_id = be_u32(data, pos).ok_or_else(|| "MP4 stss is truncated".to_string())?;
        if sample_id == 0 || sample_id <= previous {
            return Err("MP4 stss sample identifiers must increase from one".to_string());
        }
        samples.push(sample_id);
        previous = sample_id;
        pos += 4;
    }
    Ok(samples)
}

#[cfg(feature = "mp4")]
fn parse_tfhd(data: &[u8]) -> Result<ParsedTfhd, String> {
    if data.len() < 8 {
        return Err("fragmented MP4 tfhd box is truncated".to_string());
    }

    let flags =
        be_u24(data, 1).ok_or_else(|| "fragmented MP4 tfhd flags are truncated".to_string())?;
    let track_id =
        be_u32(data, 4).ok_or_else(|| "fragmented MP4 tfhd track id is truncated".to_string())?;
    let mut pos = 8usize;
    let base_data_offset = if flags & 0x000001 != 0 {
        let value = be_u64(data, pos)
            .ok_or_else(|| "fragmented MP4 tfhd base data offset is truncated".to_string())?;
        pos += 8;
        Some(value)
    } else {
        None
    };
    let sample_description_index = if flags & 0x000002 != 0 {
        let value = be_u32(data, pos).ok_or_else(|| {
            "fragmented MP4 tfhd sample description index is truncated".to_string()
        })?;
        pos += 4;
        Some(value)
    } else {
        None
    };
    let default_sample_duration = if flags & 0x000008 != 0 {
        let value = be_u32(data, pos)
            .ok_or_else(|| "fragmented MP4 tfhd sample duration is truncated".to_string())?;
        pos += 4;
        Some(value)
    } else {
        None
    };
    let default_sample_size = if flags & 0x000010 != 0 {
        Some(
            be_u32(data, pos)
                .ok_or_else(|| "fragmented MP4 tfhd sample size is truncated".to_string())?,
        )
    } else {
        None
    };

    Ok(ParsedTfhd {
        track_id,
        base_data_offset,
        sample_description_index,
        default_sample_duration,
        default_sample_size,
        default_base_is_moof: flags & 0x020000 != 0,
    })
}

#[cfg(feature = "mp4")]
fn parse_tfdt(data: &[u8]) -> Option<u64> {
    if data.len() < 8 {
        return None;
    }
    match data[0] {
        1 => be_u64(data, 4),
        0 => be_u32(data, 4).map(|value| value as u64),
        _ => None,
    }
}

#[cfg(feature = "mp4")]
fn parse_trun(data: &[u8]) -> Result<ParsedTrun, String> {
    if data.len() < 8 {
        return Err("fragmented MP4 trun box is truncated".to_string());
    }

    let version = data[0];
    if version > 1 {
        return Err(format!("unsupported fragmented MP4 trun version {version}"));
    }
    let flags =
        be_u24(data, 1).ok_or_else(|| "fragmented MP4 trun flags are truncated".to_string())?;
    let sample_count = be_u32(data, 4)
        .ok_or_else(|| "fragmented MP4 trun sample count is truncated".to_string())?;
    let count = sample_count as usize;
    if count > MAX_MP4_MATERIALIZED_ACCESS_UNITS {
        return Err(format!(
            "fragmented MP4 trun has {count} access units; the materialized limit is {MAX_MP4_MATERIALIZED_ACCESS_UNITS}"
        ));
    }
    let mut pos = 8usize;

    let data_offset = if flags & 0x000001 != 0 {
        let value = be_i32(data, pos)
            .ok_or_else(|| "fragmented MP4 trun data offset is truncated".to_string())?;
        pos += 4;
        Some(value)
    } else {
        None
    };
    let first_sample_flags = if flags & 0x000004 != 0 {
        let value = be_u32(data, pos)
            .ok_or_else(|| "fragmented MP4 trun first sample flags are truncated".to_string())?;
        pos += 4;
        Some(value)
    } else {
        None
    };

    let per_sample_fields = [0x000100, 0x000200, 0x000400, 0x000800]
        .iter()
        .filter(|flag| flags & **flag != 0)
        .count();
    validate_mp4_table_payload(data, pos, sample_count, per_sample_fields * 4, "trun")?;

    let mut sample_durations = Vec::new();
    let mut sample_sizes = Vec::new();
    let mut sample_flags = Vec::new();
    let mut sample_cts = Vec::new();
    if flags & 0x000100 != 0 {
        reserve_mp4_table(&mut sample_durations, count, "trun duration")?;
    }
    if flags & 0x000200 != 0 {
        reserve_mp4_table(&mut sample_sizes, count, "trun size")?;
    }
    if flags & 0x000400 != 0 {
        reserve_mp4_table(&mut sample_flags, count, "trun flags")?;
    }
    if flags & 0x000800 != 0 {
        reserve_mp4_table(&mut sample_cts, count, "trun composition time")?;
    }

    for _ in 0..sample_count {
        if flags & 0x000100 != 0 {
            sample_durations.push(
                be_u32(data, pos)
                    .ok_or_else(|| "fragmented MP4 trun duration is truncated".to_string())?,
            );
            pos += 4;
        }
        if flags & 0x000200 != 0 {
            sample_sizes.push(
                be_u32(data, pos)
                    .ok_or_else(|| "fragmented MP4 trun size is truncated".to_string())?,
            );
            pos += 4;
        }
        if flags & 0x000400 != 0 {
            sample_flags.push(
                be_u32(data, pos)
                    .ok_or_else(|| "fragmented MP4 trun flags are truncated".to_string())?,
            );
            pos += 4;
        }
        if flags & 0x000800 != 0 {
            let value = if version == 1 {
                be_i32(data, pos).ok_or_else(|| {
                    "fragmented MP4 trun composition time is truncated".to_string()
                })?
            } else {
                let value = be_u32(data, pos).ok_or_else(|| {
                    "fragmented MP4 trun composition time is truncated".to_string()
                })?;
                i32::try_from(value).map_err(|_| {
                    "fragmented MP4 version 0 composition time exceeds i32".to_string()
                })?
            };
            sample_cts.push(value);
            pos += 4;
        }
    }

    Ok(ParsedTrun {
        sample_count,
        data_offset,
        first_sample_flags,
        sample_durations,
        sample_sizes,
        sample_flags,
        sample_cts,
    })
}

#[cfg(feature = "mp4")]
fn walk_boxes<F>(data: &[u8], f: &mut F) -> Result<(), String>
where
    F: FnMut(Mp4BoxHeader, &[u8]) -> Result<(), String>,
{
    for_each_child_box(data, |header, payload, _| {
        f(header, payload)?;
        if is_mp4_container_box(&header.name) {
            walk_boxes(payload, f)?;
        }
        Ok(())
    })
}

#[cfg(feature = "mp4")]
fn is_mp4_container_box(name: &[u8; 4]) -> bool {
    matches!(
        name,
        b"moov" | b"trak" | b"edts" | b"mdia" | b"minf" | b"stbl" | b"mvex"
    )
}

#[cfg(feature = "mp4")]
fn parse_esds_audio_specific_config(data: &[u8]) -> Option<Vec<u8>> {
    if data.len() < 4 {
        return None;
    }
    find_mpeg4_descriptor(&data[4..], 0x05)
}

#[cfg(feature = "mp4")]
fn find_mpeg4_descriptor(data: &[u8], tag: u8) -> Option<Vec<u8>> {
    let mut pos = 0usize;
    while pos + 2 <= data.len() {
        let descriptor_tag = data[pos];
        pos += 1;
        let (len, len_len) = read_descriptor_len(&data[pos..])?;
        pos += len_len;
        if pos + len > data.len() {
            return None;
        }
        let body = &data[pos..pos + len];
        if descriptor_tag == tag {
            return Some(body.to_vec());
        }
        let nested_start = match descriptor_tag {
            0x03 if body.len() >= 3 => 3,
            0x04 if body.len() >= 13 => 13,
            _ => 0,
        };
        if nested_start > 0 {
            if let Some(found) = find_mpeg4_descriptor(&body[nested_start..], tag) {
                return Some(found);
            }
        }
        pos += len;
    }
    None
}

#[cfg(feature = "mp4")]
fn read_descriptor_len(data: &[u8]) -> Option<(usize, usize)> {
    let mut value = 0usize;
    for (index, byte) in data.iter().take(4).enumerate() {
        value = (value << 7) | (byte & 0x7f) as usize;
        if byte & 0x80 == 0 {
            return Some((value, index + 1));
        }
    }
    None
}

#[cfg(feature = "mp4")]
fn parse_asc_audio_config(data: &[u8]) -> Option<(u32, u8)> {
    if data.len() < 2 {
        return None;
    }
    let freq_index = ((data[0] & 0x07) << 1) | (data[1] >> 7);
    let sample_rate = adts_sample_rate(freq_index)?;
    let channels = (data[1] >> 3) & 0x0f;
    Some((sample_rate, channels))
}

#[cfg(feature = "mp4")]
fn create_adts_header(sample_rate: u32, channels: u8, raw_len: usize, asc: &[u8]) -> Vec<u8> {
    let profile = asc
        .first()
        .map(|first| ((*first >> 3).saturating_sub(1)).min(3))
        .unwrap_or(1);
    let sample_rate_index = sample_rate_index(sample_rate);
    let channel_config = channels.min(7);
    let frame_length = raw_len + 7;

    vec![
        0xff,
        0xf1,
        (profile << 6) | (sample_rate_index << 2) | (channel_config >> 2),
        ((channel_config & 0x03) << 6) | (((frame_length >> 11) & 0x03) as u8),
        ((frame_length >> 3) & 0xff) as u8,
        (((frame_length & 0x07) << 5) as u8) | 0x1f,
        0xfc,
    ]
}

#[cfg(feature = "mp4")]
fn sample_rate_index(sample_rate: u32) -> u8 {
    match sample_rate {
        96_000 => 0,
        88_200 => 1,
        64_000 => 2,
        48_000 => 3,
        44_100 => 4,
        32_000 => 5,
        24_000 => 6,
        22_050 => 7,
        16_000 => 8,
        12_000 => 9,
        11_025 => 10,
        8_000 => 11,
        7_350 => 12,
        _ => 15,
    }
}

#[cfg(feature = "mp4")]
fn add_signed_offset(base: u64, offset: i32) -> Result<u64, String> {
    if offset >= 0 {
        base.checked_add(offset as u64)
            .ok_or_else(|| "fMP4 data offset overflow".to_string())
    } else {
        base.checked_sub(offset.unsigned_abs() as u64)
            .ok_or_else(|| "fMP4 negative data offset underflow".to_string())
    }
}

#[cfg(feature = "mp4")]
fn be_u16(data: &[u8], pos: usize) -> Option<u16> {
    Some(u16::from_be_bytes([*data.get(pos)?, *data.get(pos + 1)?]))
}

#[cfg(feature = "mp4")]
fn be_u24(data: &[u8], pos: usize) -> Option<u32> {
    Some(
        ((*data.get(pos)? as u32) << 16)
            | ((*data.get(pos + 1)? as u32) << 8)
            | *data.get(pos + 2)? as u32,
    )
}

#[cfg(feature = "mp4")]
fn be_u32(data: &[u8], pos: usize) -> Option<u32> {
    Some(u32::from_be_bytes([
        *data.get(pos)?,
        *data.get(pos + 1)?,
        *data.get(pos + 2)?,
        *data.get(pos + 3)?,
    ]))
}

#[cfg(feature = "mp4")]
fn be_i32(data: &[u8], pos: usize) -> Option<i32> {
    be_u32(data, pos).map(|value| value as i32)
}

#[cfg(feature = "mp4")]
fn be_u64(data: &[u8], pos: usize) -> Option<u64> {
    Some(u64::from_be_bytes([
        *data.get(pos)?,
        *data.get(pos + 1)?,
        *data.get(pos + 2)?,
        *data.get(pos + 3)?,
        *data.get(pos + 4)?,
        *data.get(pos + 5)?,
        *data.get(pos + 6)?,
        *data.get(pos + 7)?,
    ]))
}

#[cfg(feature = "mpeg-ts")]
fn looks_like_mpeg_ts(bytes: &[u8]) -> bool {
    detect_ts_layout(bytes).is_some()
}

#[cfg(feature = "mpeg-ts")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TsPacketLayout {
    stride: usize,
    prefix: usize,
}

#[cfg(feature = "mpeg-ts")]
fn detect_ts_layout(bytes: &[u8]) -> Option<TsPacketLayout> {
    [
        TsPacketLayout {
            stride: 188,
            prefix: 0,
        },
        TsPacketLayout {
            stride: 192,
            prefix: 4,
        },
        TsPacketLayout {
            stride: 204,
            prefix: 0,
        },
    ]
    .into_iter()
    .find(|layout| {
        bytes.len() >= layout.stride * 5
            && (0..5).all(|index| bytes[layout.prefix + index * layout.stride] == 0x47)
    })
}

#[cfg(feature = "mpeg-ts")]
struct MpegTsAudioDemuxer {
    buffer: Vec<u8>,
    cursor: usize,
    layout: Option<TsPacketLayout>,
    pmt_pid: Option<u16>,
    audio_pid: Option<u16>,
    audio_codec: Option<AudioCodec>,
    packet_format: Option<AudioPacketFormat>,
    stream_type: Option<u8>,
    current_pes: Vec<u8>,
    emitted_config: bool,
    sample_rate: Option<u32>,
    channels: Option<u8>,
    continuity: HashMap<u16, u8>,
    current_pes_continuity: Option<u8>,
    current_pes_discontinuity: bool,
    current_pts: Option<u64>,
    current_dts: Option<u64>,
    pts_epoch: u64,
    last_raw_pts: Option<u64>,
    selected_program: Option<u16>,
    psi_sections: HashMap<u16, PsiSectionAssembler>,
    pat_version: Option<u8>,
    pmt_version: Option<u8>,
}

#[cfg(feature = "mpeg-ts")]
impl MpegTsAudioDemuxer {
    fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(188 * 32),
            cursor: 0,
            layout: None,
            pmt_pid: None,
            audio_pid: None,
            audio_codec: None,
            packet_format: None,
            stream_type: None,
            current_pes: Vec::new(),
            emitted_config: false,
            sample_rate: None,
            channels: None,
            continuity: HashMap::new(),
            current_pes_continuity: None,
            current_pes_discontinuity: false,
            current_pts: None,
            current_dts: None,
            pts_epoch: 0,
            last_raw_pts: None,
            selected_program: None,
            psi_sections: HashMap::new(),
            pat_version: None,
            pmt_version: None,
        }
    }

    fn add(&mut self, data: &[u8]) -> Result<Vec<AudioDemuxEvent>, String> {
        validate_container_input_chunk(data, "MPEG-TS")?;
        self.buffer.extend_from_slice(data);
        self.parse_available_packets()
    }

    fn finish(&mut self, data: &[u8]) -> Result<Vec<AudioDemuxEvent>, String> {
        validate_container_input_chunk(data, "MPEG-TS")?;
        self.buffer.extend_from_slice(data);
        let mut events = self.parse_available_packets()?;
        events.extend(self.flush_current_pes()?);
        Ok(events)
    }

    fn parse_available_packets(&mut self) -> Result<Vec<AudioDemuxEvent>, String> {
        let mut events = Vec::new();

        loop {
            let remaining = &self.buffer[self.cursor..];
            let layout = match self.layout.or_else(|| detect_ts_layout(remaining)) {
                Some(layout) => {
                    self.layout = Some(layout);
                    layout
                }
                None if remaining.len() >= 204 * 5 => {
                    self.cursor += 1;
                    continue;
                }
                None => break,
            };
            if remaining.len() < layout.stride {
                break;
            }
            let sync = self.cursor + layout.prefix;
            if self.buffer.get(sync) != Some(&0x47) || sync + 188 > self.buffer.len() {
                self.layout = None;
                self.cursor += 1;
                continue;
            }
            let mut packet = [0_u8; 188];
            packet.copy_from_slice(&self.buffer[sync..sync + 188]);
            self.cursor += layout.stride;
            self.parse_packet(&packet, &mut events)?;
        }

        if self.cursor > 64 * 1024 || self.cursor == self.buffer.len() {
            self.buffer.drain(..self.cursor);
            self.cursor = 0;
        }

        Ok(events)
    }

    fn parse_packet(
        &mut self,
        packet: &[u8],
        events: &mut Vec<AudioDemuxEvent>,
    ) -> Result<(), String> {
        if packet.len() != 188 || packet[0] != 0x47 {
            return Ok(());
        }
        if packet[1] & 0x80 != 0 {
            return Err("MPEG-TS packet has the transport-error indicator".to_string());
        }
        if packet[3] & 0xC0 != 0 {
            return Err("scrambled MPEG-TS packets are unsupported".to_string());
        }

        let payload_unit_start = packet[1] & 0x40 != 0;
        let pid = (((packet[1] & 0x1f) as u16) << 8) | packet[2] as u16;
        let adaptation_field_control = (packet[3] & 0x30) >> 4;
        let continuity_counter = packet[3] & 0x0f;

        if adaptation_field_control == 0 {
            return Err("MPEG-TS packet has reserved adaptation-field control".to_string());
        }
        if adaptation_field_control == 2 {
            return Ok(());
        }

        let mut payload_start = 4usize;
        let mut discontinuity = false;
        if adaptation_field_control == 3 {
            let adaptation_len = packet[payload_start] as usize;
            let adaptation_end = payload_start
                .checked_add(1 + adaptation_len)
                .ok_or_else(|| "MPEG-TS adaptation field size overflow".to_string())?;
            if adaptation_end > packet.len() {
                return Err("MPEG-TS adaptation field exceeds its packet".to_string());
            }
            if adaptation_len > 0 {
                discontinuity = packet[payload_start + 1] & 0x80 != 0;
            }
            payload_start = adaptation_end;
        }

        if payload_start >= packet.len() {
            return Ok(());
        }

        if let Some(previous) = self.continuity.insert(pid, continuity_counter) {
            if continuity_counter == previous {
                return Ok(());
            }
            let expected = (previous + 1) & 0x0f;
            if continuity_counter != expected && !discontinuity {
                discontinuity = true;
                if Some(pid) == self.audio_pid {
                    self.current_pes.clear();
                }
                if let Some(assembler) = self.psi_sections.get_mut(&pid) {
                    assembler.reset();
                }
            }
        }
        if discontinuity {
            if let Some(assembler) = self.psi_sections.get_mut(&pid) {
                assembler.reset();
            }
        }

        let payload = &packet[payload_start..];
        if pid == 0 {
            for section in self.push_psi(pid, payload, payload_unit_start)? {
                self.parse_pat_section(&section)?;
            }
        } else if Some(pid) == self.pmt_pid {
            for section in self.push_psi(pid, payload, payload_unit_start)? {
                self.parse_pmt_section(&section)?;
            }
        } else if Some(pid) == self.audio_pid {
            if payload_unit_start {
                events.extend(self.flush_current_pes()?);
                self.current_pes.clear();
                self.current_pes_discontinuity = discontinuity;
                self.current_pes_continuity = Some(continuity_counter);
            }
            self.current_pes_discontinuity |= discontinuity;
            if self.current_pes.len().saturating_add(payload.len()) > MAX_MPEG_TS_PES_BYTES {
                return Err(format!(
                    "MPEG-TS PES exceeds the {MAX_MPEG_TS_PES_BYTES} byte packet budget"
                ));
            }
            self.current_pes.extend_from_slice(payload);
        }

        Ok(())
    }

    fn push_psi(
        &mut self,
        pid: u16,
        payload: &[u8],
        payload_unit_start: bool,
    ) -> Result<Vec<Vec<u8>>, String> {
        self.psi_sections
            .entry(pid)
            .or_default()
            .push(payload, payload_unit_start)
    }

    fn parse_pat_section(&mut self, section: &[u8]) -> Result<(), String> {
        if section.len() < 12 || section[0] != 0x00 {
            return Ok(());
        }

        let section_length = (((section[1] & 0x0f) as usize) << 8) | section[2] as usize;
        let section_end = 3 + section_length;
        if section_end > section.len() {
            return Ok(());
        }
        if section_end < 12 {
            return Ok(());
        }
        validate_psi_crc(&section[..section_end])?;
        if section[5] & 0x01 == 0 {
            return Ok(());
        }
        let version = (section[5] >> 1) & 0x1f;
        if self.pat_version == Some(version) {
            return Ok(());
        }

        let entries_end = section_end.saturating_sub(4);
        let mut pos = 8usize;
        let mut selected = None;
        while pos + 4 <= entries_end {
            let program_number = u16::from_be_bytes([section[pos], section[pos + 1]]);
            let pid = (((section[pos + 2] & 0x1f) as u16) << 8) | section[pos + 3] as u16;
            if program_number != 0 {
                selected = match selected {
                    Some((current, _)) if current <= program_number => selected,
                    _ => Some((program_number, pid)),
                };
            }
            pos += 4;
        }
        if let Some((program, pid)) = selected {
            if self.pmt_pid != Some(pid) {
                self.audio_pid = None;
                self.audio_codec = None;
                self.packet_format = None;
                self.stream_type = None;
                self.pmt_version = None;
            }
            self.selected_program = Some(program);
            self.pmt_pid = Some(pid);
            self.pat_version = Some(version);
        }

        Ok(())
    }

    fn parse_pmt_section(&mut self, section: &[u8]) -> Result<(), String> {
        if section.len() < 16 || section[0] != 0x02 {
            return Ok(());
        }

        let section_length = (((section[1] & 0x0f) as usize) << 8) | section[2] as usize;
        let section_end = 3 + section_length;
        if section_end > section.len() {
            return Ok(());
        }
        if section_end < 16 {
            return Ok(());
        }
        validate_psi_crc(&section[..section_end])?;
        if section[5] & 0x01 == 0 {
            return Ok(());
        }
        let version = (section[5] >> 1) & 0x1f;
        if self.pmt_version == Some(version) {
            return Ok(());
        }
        let program_number = u16::from_be_bytes([section[3], section[4]]);
        if self
            .selected_program
            .is_some_and(|selected| selected != program_number)
        {
            return Ok(());
        }

        let program_info_length = (((section[10] & 0x0f) as usize) << 8) | section[11] as usize;
        let entries_end = section_end.saturating_sub(4);
        let program_descriptors_end = 12usize
            .checked_add(program_info_length)
            .ok_or_else(|| "MPEG-TS PMT program descriptor size overflow".to_string())?;
        if program_descriptors_end > entries_end {
            return Err("MPEG-TS PMT program descriptors exceed their section".to_string());
        }
        let program_descriptors = &section[12..program_descriptors_end];
        let mut pos = program_descriptors_end;

        while pos + 5 <= entries_end {
            let stream_type = section[pos];
            let pid = (((section[pos + 1] & 0x1f) as u16) << 8) | section[pos + 2] as u16;
            let es_info_length =
                (((section[pos + 3] & 0x0f) as usize) << 8) | section[pos + 4] as usize;

            let descriptors_end = pos
                .checked_add(5 + es_info_length)
                .ok_or_else(|| "MPEG-TS PMT descriptor size overflow".to_string())?;
            if descriptors_end > entries_end {
                return Err("MPEG-TS PMT descriptor exceeds its section".to_string());
            }
            if let Some((codec, packet_format)) = ts_stream_codec(
                stream_type,
                program_descriptors,
                &section[pos + 5..descriptors_end],
            ) {
                if self.audio_pid.is_none() {
                    self.audio_pid = Some(pid);
                    self.audio_codec = Some(codec);
                    self.packet_format = Some(packet_format);
                    self.stream_type = Some(stream_type);
                }
                self.pmt_version = Some(version);
                break;
            }

            pos = descriptors_end;
        }

        Ok(())
    }

    fn flush_current_pes(&mut self) -> Result<Vec<AudioDemuxEvent>, String> {
        if self.current_pes.is_empty() {
            return Ok(Vec::new());
        }

        let pes = std::mem::take(&mut self.current_pes);
        let parsed = parse_pes(&pes)?;
        let payload = parsed.payload;
        let pts = parsed.pts.map(|value| self.expand_pts(value));
        let dts = parsed
            .dts
            .map(|value| self.expand_timestamp_near_pts(value));
        self.current_pts = pts;
        self.current_dts = dts;

        let codec = match self.audio_codec.clone() {
            Some(codec) => codec,
            None => return Ok(Vec::new()),
        };

        if self.stream_type == Some(0x80) {
            return self.emit_bluray_lpcm(payload, pts, dts);
        }
        if self.stream_type == Some(0x82) {
            return self.emit_dts_frames(payload, pts, dts);
        }

        match self.packet_format.clone().unwrap_or(AudioPacketFormat::Raw) {
            AudioPacketFormat::Adts => self.emit_adts_frames(payload, pts, dts),
            AudioPacketFormat::Latm => self.emit_loas_frames(payload, pts, dts),
            AudioPacketFormat::Raw if matches!(self.stream_type, Some(0x03 | 0x04)) => {
                self.emit_mpeg_audio_frames(payload, pts, dts)
            }
            AudioPacketFormat::Raw => {
                let mut events = Vec::new();
                self.ensure_config(&mut events, None, None);
                if !payload.is_empty() {
                    events.push(AudioDemuxEvent::Packet(AudioTrackPacket {
                        container: AudioContainer::MpegTs,
                        codec,
                        format: self.packet_format.clone().unwrap_or(AudioPacketFormat::Raw),
                        data: payload.to_vec(),
                        raw_data: None,
                        track_id: None,
                        pid: self.audio_pid,
                        stream_type: self.stream_type,
                        timescale: Some(90_000),
                        continuity_counter: self.current_pes_continuity,
                        discontinuity: self.current_pes_discontinuity,
                        decode_time: self.current_dts,
                        sample_id: None,
                        start_time: self.current_pts,
                        duration: None,
                        rendering_offset: None,
                        is_sync: None,
                        timecode: pts.and_then(|value| i64::try_from(value).ok()),
                    }));
                }
                Ok(events)
            }
        }
    }

    fn emit_bluray_lpcm(
        &mut self,
        payload: &[u8],
        pts: Option<u64>,
        dts: Option<u64>,
    ) -> Result<Vec<AudioDemuxEvent>, String> {
        let unit = parse_lpcm_access_unit(payload).map_err(|error| error.to_string())?;
        let duration_ticks = u64::try_from(unit.frames)
            .ok()
            .and_then(|frames| frames.checked_mul(90_000))
            .ok_or_else(|| "Blu-ray LPCM duration calculation overflow".to_string())?
            / u64::from(unit.sample_rate);
        let duration = u32::try_from(duration_ticks)
            .map_err(|_| "Blu-ray LPCM duration exceeds u32".to_string())?;
        self.audio_codec = Some(AudioCodec::Pcm);
        self.sample_rate = Some(unit.sample_rate);
        self.channels = Some(unit.channels);

        let mut events = Vec::new();
        if !self.emitted_config {
            events.push(AudioDemuxEvent::Config(AudioTrackConfig {
                container: AudioContainer::MpegTs,
                codec: AudioCodec::Pcm,
                packet_format: Some(AudioPacketFormat::Raw),
                codec_id: Some("pcm-bluray".to_string()),
                track_id: None,
                pid: self.audio_pid,
                stream_type: self.stream_type,
                timescale: Some(90_000),
                transport_packet_stride: self.layout.map(|layout| layout.stride as u16),
                transport_prefix_bytes: self.layout.map(|layout| layout.prefix as u8),
                program_number: self.selected_program,
                sample_rate: self.sample_rate,
                channels: self.channels,
                bits_per_sample: Some(unit.bits_per_sample),
                pcm_endianness: Some(PcmEndianness::Big),
                pcm_float: Some(false),
                pcm_signed: Some(true),
                pcm_packed: Some(true),
                pcm_aligned_high: Some(false),
                pcm_interleaved: Some(true),
                pcm_bytes_per_frame: Some(
                    u32::try_from(unit.bytes_per_frame)
                        .map_err(|_| "Blu-ray LPCM frame size exceeds u32".to_string())?,
                ),
                pcm_frames_per_packet: Some(
                    u32::try_from(unit.frames)
                        .map_err(|_| "Blu-ray LPCM packet frame count exceeds u32".to_string())?,
                ),
                sample_count: None,
                codec_private: unit.header.to_vec(),
                pre_skip: None,
                output_gain: None,
                mapping_family: None,
            }));
            self.emitted_config = true;
        }
        events.push(AudioDemuxEvent::Packet(AudioTrackPacket {
            container: AudioContainer::MpegTs,
            codec: AudioCodec::Pcm,
            format: AudioPacketFormat::Raw,
            data: unit.payload.to_vec(),
            raw_data: None,
            track_id: None,
            pid: self.audio_pid,
            stream_type: self.stream_type,
            timescale: Some(90_000),
            continuity_counter: self.current_pes_continuity,
            discontinuity: self.current_pes_discontinuity,
            decode_time: dts,
            sample_id: None,
            start_time: pts,
            duration: Some(duration),
            rendering_offset: None,
            is_sync: None,
            timecode: pts.and_then(|value| i64::try_from(value).ok()),
        }));
        Ok(events)
    }

    fn emit_dts_frames(
        &mut self,
        payload: &[u8],
        pts: Option<u64>,
        dts: Option<u64>,
    ) -> Result<Vec<AudioDemuxEvent>, String> {
        let mut events = Vec::new();
        let mut position = 0usize;
        let mut timestamp_offset = 0u64;
        while position < payload.len() {
            let unit =
                parse_core_access_unit(&payload[position..]).map_err(|error| error.to_string())?;
            let end = position
                .checked_add(unit.data.len())
                .ok_or_else(|| "DTS access-unit byte range overflow".to_string())?;
            self.ensure_config(&mut events, Some(unit.sample_rate), Some(unit.channels));
            let duration =
                u32::try_from(u64::from(unit.samples) * 90_000 / u64::from(unit.sample_rate))
                    .map_err(|_| "DTS timestamp duration exceeds u32".to_string())?;
            let frame_pts = pts.and_then(|value| value.checked_add(timestamp_offset));
            let frame_dts = dts.and_then(|value| value.checked_add(timestamp_offset));
            events.push(AudioDemuxEvent::Packet(AudioTrackPacket {
                container: AudioContainer::MpegTs,
                codec: AudioCodec::Dts,
                format: AudioPacketFormat::Raw,
                data: payload[position..end].to_vec(),
                raw_data: None,
                track_id: None,
                pid: self.audio_pid,
                stream_type: self.stream_type,
                timescale: Some(90_000),
                continuity_counter: self.current_pes_continuity,
                discontinuity: self.current_pes_discontinuity && position == 0,
                decode_time: frame_dts,
                sample_id: None,
                start_time: frame_pts,
                duration: Some(duration),
                rendering_offset: None,
                is_sync: None,
                timecode: frame_pts.and_then(|value| i64::try_from(value).ok()),
            }));
            timestamp_offset = timestamp_offset.saturating_add(u64::from(duration));
            position = end;
        }
        Ok(events)
    }

    fn emit_loas_frames(
        &mut self,
        payload: &[u8],
        pts: Option<u64>,
        dts: Option<u64>,
    ) -> Result<Vec<AudioDemuxEvent>, String> {
        let mut events = Vec::new();
        let mut position = 0usize;
        while position + 3 <= payload.len() {
            if payload[position] != 0x56 || payload[position + 1] & 0xe0 != 0xe0 {
                position += 1;
                continue;
            }
            let payload_length = (usize::from(payload[position + 1] & 0x1f) << 8)
                | usize::from(payload[position + 2]);
            let frame_length = 3usize
                .checked_add(payload_length)
                .ok_or_else(|| "LOAS/LATM frame size overflow".to_string())?;
            if position + frame_length > payload.len() {
                return Err("LOAS/LATM access unit is truncated at the PES boundary".to_string());
            }
            self.ensure_config(&mut events, None, None);
            let first_frame = !events
                .iter()
                .any(|event| matches!(event, AudioDemuxEvent::Packet(_)));
            events.push(AudioDemuxEvent::Packet(AudioTrackPacket {
                container: AudioContainer::MpegTs,
                codec: AudioCodec::Aac,
                format: AudioPacketFormat::Latm,
                data: payload[position..position + frame_length].to_vec(),
                raw_data: None,
                track_id: None,
                pid: self.audio_pid,
                stream_type: self.stream_type,
                timescale: Some(90_000),
                continuity_counter: self.current_pes_continuity,
                discontinuity: self.current_pes_discontinuity && first_frame,
                decode_time: dts,
                sample_id: None,
                start_time: pts,
                duration: None,
                rendering_offset: None,
                is_sync: None,
                timecode: pts.and_then(|value| i64::try_from(value).ok()),
            }));
            position += frame_length;
        }
        if position != payload.len() && payload[position..].iter().any(|byte| *byte != 0xff) {
            return Err("LOAS/LATM PES ends with an incomplete access unit".to_string());
        }
        Ok(events)
    }

    fn emit_mpeg_audio_frames(
        &mut self,
        payload: &[u8],
        pts: Option<u64>,
        dts: Option<u64>,
    ) -> Result<Vec<AudioDemuxEvent>, String> {
        let mut events = Vec::new();
        let mut position = 0usize;
        let mut timestamp_offset = 0u64;
        while position + 4 <= payload.len() {
            let Some(header) = parse_mpeg_audio_header(&payload[position..]) else {
                position += 1;
                continue;
            };
            if position + header.frame_length > payload.len() {
                return Err("MPEG audio frame is truncated at the PES boundary".to_string());
            }
            let codec = if header.layer == 3 {
                AudioCodec::Mp3
            } else {
                AudioCodec::Unknown(format!("mp{}", header.layer))
            };
            if !self.emitted_config {
                self.audio_codec = Some(codec.clone());
            }
            self.ensure_config(&mut events, Some(header.sample_rate), Some(header.channels));
            let duration = u32::try_from(
                u64::from(header.samples_per_frame) * 90_000 / u64::from(header.sample_rate),
            )
            .map_err(|_| "MPEG audio timestamp duration exceeds u32".to_string())?;
            let frame_pts = pts.and_then(|value| value.checked_add(timestamp_offset));
            let frame_dts = dts.and_then(|value| value.checked_add(timestamp_offset));
            events.push(AudioDemuxEvent::Packet(AudioTrackPacket {
                container: AudioContainer::MpegTs,
                codec,
                format: AudioPacketFormat::Raw,
                data: payload[position..position + header.frame_length].to_vec(),
                raw_data: None,
                track_id: None,
                pid: self.audio_pid,
                stream_type: self.stream_type,
                timescale: Some(90_000),
                continuity_counter: self.current_pes_continuity,
                discontinuity: self.current_pes_discontinuity && timestamp_offset == 0,
                decode_time: frame_dts,
                sample_id: None,
                start_time: frame_pts,
                duration: Some(duration),
                rendering_offset: None,
                is_sync: None,
                timecode: frame_pts.and_then(|value| i64::try_from(value).ok()),
            }));
            timestamp_offset = timestamp_offset.saturating_add(u64::from(duration));
            position += header.frame_length;
        }
        if position != payload.len() && payload[position..].iter().any(|byte| *byte != 0xff) {
            return Err("MPEG audio PES ends with an incomplete frame".to_string());
        }
        Ok(events)
    }

    fn emit_adts_frames(
        &mut self,
        payload: &[u8],
        pts: Option<u64>,
        dts: Option<u64>,
    ) -> Result<Vec<AudioDemuxEvent>, String> {
        let mut events = Vec::new();
        let mut pos = 0usize;
        while pos + 7 <= payload.len() {
            if payload[pos] != 0xff || payload[pos + 1] & 0xf0 != 0xf0 {
                pos += 1;
                continue;
            }

            let Some(header) = parse_adts_header(&payload[pos..]) else {
                break;
            };

            if pos + header.frame_length > payload.len() {
                break;
            }

            self.ensure_config(&mut events, Some(header.sample_rate), Some(header.channels));
            let duration = u32::try_from(
                (90_000_u64 * u64::from(header.samples_per_frame)) / u64::from(header.sample_rate),
            )
            .ok();
            let frame_index = events
                .iter()
                .filter(|event| matches!(event, AudioDemuxEvent::Packet(_)))
                .count() as u64;
            let timestamp_offset = u64::from(duration.unwrap_or(0)).saturating_mul(frame_index);
            let frame_pts = pts.and_then(|value| value.checked_add(timestamp_offset));
            let frame_dts = dts.and_then(|value| value.checked_add(timestamp_offset));
            events.push(AudioDemuxEvent::Packet(AudioTrackPacket {
                container: AudioContainer::MpegTs,
                codec: AudioCodec::Aac,
                format: AudioPacketFormat::Adts,
                data: payload[pos..pos + header.frame_length].to_vec(),
                raw_data: None,
                track_id: None,
                pid: self.audio_pid,
                stream_type: self.stream_type,
                timescale: Some(90_000),
                continuity_counter: self.current_pes_continuity,
                discontinuity: self.current_pes_discontinuity && frame_index == 0,
                decode_time: frame_dts,
                sample_id: None,
                start_time: frame_pts,
                duration,
                rendering_offset: None,
                is_sync: None,
                timecode: frame_pts.and_then(|value| i64::try_from(value).ok()),
            }));
            pos += header.frame_length;
        }

        Ok(events)
    }

    fn expand_pts(&mut self, raw: u64) -> u64 {
        if let Some(previous) = self.last_raw_pts {
            if previous > raw && previous - raw > (1_u64 << 32) {
                self.pts_epoch = self.pts_epoch.saturating_add(1_u64 << 33);
            }
        }
        self.last_raw_pts = Some(raw);
        self.pts_epoch.saturating_add(raw)
    }

    fn expand_timestamp_near_pts(&self, raw: u64) -> u64 {
        self.pts_epoch.saturating_add(raw)
    }

    fn ensure_config(
        &mut self,
        events: &mut Vec<AudioDemuxEvent>,
        sample_rate: Option<u32>,
        channels: Option<u8>,
    ) {
        if self.emitted_config {
            return;
        }

        self.sample_rate = sample_rate.or(self.sample_rate);
        self.channels = channels.or(self.channels);
        events.push(AudioDemuxEvent::Config(AudioTrackConfig {
            container: AudioContainer::MpegTs,
            codec: self
                .audio_codec
                .clone()
                .unwrap_or_else(|| AudioCodec::Unknown("unknown".to_string())),
            packet_format: self.packet_format.clone(),
            codec_id: None,
            track_id: None,
            pid: self.audio_pid,
            stream_type: self.stream_type,
            timescale: Some(90_000),
            transport_packet_stride: self.layout.map(|layout| layout.stride as u16),
            transport_prefix_bytes: self.layout.map(|layout| layout.prefix as u8),
            program_number: self.selected_program,
            sample_rate: self.sample_rate,
            channels: self.channels,
            bits_per_sample: None,
            pcm_endianness: None,
            pcm_float: None,
            pcm_signed: None,
            pcm_packed: None,
            pcm_aligned_high: None,
            pcm_interleaved: None,
            pcm_bytes_per_frame: None,
            pcm_frames_per_packet: None,
            sample_count: None,
            codec_private: Vec::new(),
            pre_skip: None,
            output_gain: None,
            mapping_family: None,
        }));
        self.emitted_config = true;
    }
}

#[cfg(feature = "mpeg-ts")]
#[derive(Default)]
struct PsiSectionAssembler {
    bytes: Vec<u8>,
    expected: Option<usize>,
}

#[cfg(feature = "mpeg-ts")]
impl PsiSectionAssembler {
    fn push(&mut self, payload: &[u8], payload_unit_start: bool) -> Result<Vec<Vec<u8>>, String> {
        let mut completed = Vec::new();
        let mut data = payload;
        if payload_unit_start {
            let pointer = usize::from(
                *data
                    .first()
                    .ok_or_else(|| "truncated PSI pointer field".to_string())?,
            );
            data = data
                .get(1..)
                .ok_or_else(|| "truncated PSI pointer field".to_string())?;
            if pointer > data.len() {
                return Err("PSI pointer exceeds payload".to_string());
            }
            if !self.bytes.is_empty() {
                self.append(&data[..pointer], &mut completed)?;
                if !self.bytes.is_empty() {
                    return Err(
                        "PSI pointer starts a new section before the previous section ends"
                            .to_string(),
                    );
                }
            }
            data = &data[pointer..];
        }
        self.append(data, &mut completed)?;
        Ok(completed)
    }

    fn append(&mut self, mut data: &[u8], completed: &mut Vec<Vec<u8>>) -> Result<(), String> {
        while !data.is_empty() {
            if self.bytes.is_empty() && data[0] == 0xff {
                return Ok(());
            }
            if self.expected.is_none() {
                let header_needed = 3usize.saturating_sub(self.bytes.len());
                let take = header_needed.min(data.len());
                self.bytes.extend_from_slice(&data[..take]);
                data = &data[take..];
                if self.bytes.len() < 3 {
                    return Ok(());
                }
                let section_length =
                    ((usize::from(self.bytes[1] & 0x0f)) << 8) | usize::from(self.bytes[2]);
                let expected = 3usize
                    .checked_add(section_length)
                    .ok_or_else(|| "PSI section size overflow".to_string())?;
                if !(4..=1024).contains(&expected) {
                    self.reset();
                    return Err(format!("PSI section size {expected} is out of range"));
                }
                self.expected = Some(expected);
            }
            let expected = self.expected.unwrap();
            let needed = expected - self.bytes.len();
            let take = needed.min(data.len());
            self.bytes.extend_from_slice(&data[..take]);
            data = &data[take..];
            if self.bytes.len() == expected {
                completed.push(std::mem::take(&mut self.bytes));
                self.expected = None;
            }
        }
        Ok(())
    }

    fn reset(&mut self) {
        self.bytes.clear();
        self.expected = None;
    }
}

#[cfg(feature = "mpeg-ts")]
fn ts_stream_codec(
    stream_type: u8,
    program_descriptors: &[u8],
    descriptors: &[u8],
) -> Option<(AudioCodec, AudioPacketFormat)> {
    match stream_type {
        0x0f => Some((AudioCodec::Aac, AudioPacketFormat::Adts)),
        0x11 => Some((AudioCodec::Aac, AudioPacketFormat::Latm)),
        0x03 | 0x04 => Some((
            AudioCodec::Unknown("mpeg-audio".to_string()),
            AudioPacketFormat::Raw,
        )),
        0x81 => Some((AudioCodec::Ac3, AudioPacketFormat::Raw)),
        0x80 => Some((AudioCodec::Pcm, AudioPacketFormat::Raw)),
        0x82 => Some((AudioCodec::Dts, AudioPacketFormat::Raw)),
        0x87 => Some((AudioCodec::Ac3, AudioPacketFormat::Raw)),
        0x06 if descriptors_have_tag(descriptors, 0x6A)
            || descriptors_have_registration(descriptors, b"AC-3") =>
        {
            Some((AudioCodec::Ac3, AudioPacketFormat::Raw))
        }
        // FFmpeg signals AAC as private PES in M2TS mode and identifies the
        // program with the HDMV registration descriptor.
        0x06 if descriptors_have_registration(program_descriptors, b"HDMV") => {
            Some((AudioCodec::Aac, AudioPacketFormat::Adts))
        }
        _ => None,
    }
}

#[cfg(feature = "mpeg-ts")]
fn descriptors_have_tag(mut data: &[u8], target: u8) -> bool {
    while data.len() >= 2 {
        let length = data[1] as usize;
        if data.len() < 2 + length {
            return false;
        }
        if data[0] == target {
            return true;
        }
        data = &data[2 + length..];
    }
    false
}

#[cfg(feature = "mpeg-ts")]
fn descriptors_have_registration(mut data: &[u8], target: &[u8; 4]) -> bool {
    while data.len() >= 2 {
        let length = data[1] as usize;
        if data.len() < 2 + length {
            return false;
        }
        if data[0] == 0x05 && length >= 4 && &data[2..6] == target {
            return true;
        }
        data = &data[2 + length..];
    }
    false
}

#[cfg(feature = "mpeg-ts")]
fn validate_psi_crc(section: &[u8]) -> Result<(), String> {
    let mut crc = 0xffff_ffff_u32;
    for byte in section {
        crc ^= u32::from(*byte) << 24;
        for _ in 0..8 {
            crc = if crc & 0x8000_0000 != 0 {
                (crc << 1) ^ 0x04C1_1DB7
            } else {
                crc << 1
            };
        }
    }
    if crc != 0 {
        return Err("MPEG-TS PSI CRC mismatch".to_string());
    }
    Ok(())
}

#[cfg(feature = "mpeg-ts")]
struct ParsedPes<'a> {
    payload: &'a [u8],
    pts: Option<u64>,
    dts: Option<u64>,
}

#[cfg(feature = "mpeg-ts")]
fn parse_pes(pes: &[u8]) -> Result<ParsedPes<'_>, String> {
    if pes.len() < 9 || pes[0] != 0x00 || pes[1] != 0x00 || pes[2] != 0x01 {
        return Err("invalid MPEG-TS PES header".to_string());
    }

    let header_data_len = pes[8] as usize;
    let payload_start = 9 + header_data_len;
    if payload_start > pes.len() {
        return Err("MPEG-TS PES optional header is truncated".to_string());
    }
    let timestamp_flags = (pes[7] >> 6) & 0x03;
    let pts = if timestamp_flags & 0x02 != 0 {
        Some(parse_pes_timestamp(&pes[9..])?)
    } else {
        None
    };
    let dts = if timestamp_flags == 0x03 {
        Some(parse_pes_timestamp(&pes[14..])?)
    } else {
        pts
    };
    let declared_length = u16::from_be_bytes([pes[4], pes[5]]) as usize;
    let payload_end = if declared_length == 0 {
        pes.len()
    } else {
        6usize
            .checked_add(declared_length)
            .ok_or_else(|| "MPEG-TS PES length overflow".to_string())?
            .min(pes.len())
    };
    if payload_end < payload_start {
        return Err("MPEG-TS PES length ends inside its header".to_string());
    }
    Ok(ParsedPes {
        payload: &pes[payload_start..payload_end],
        pts,
        dts,
    })
}

#[cfg(feature = "mpeg-ts")]
fn parse_pes_timestamp(data: &[u8]) -> Result<u64, String> {
    if data.len() < 5 || data[0] & 1 == 0 || data[2] & 1 == 0 || data[4] & 1 == 0 {
        return Err("invalid or truncated MPEG-TS PES timestamp".to_string());
    }
    Ok((u64::from((data[0] >> 1) & 0x07) << 30)
        | (u64::from(data[1]) << 22)
        | (u64::from(data[2] >> 1) << 15)
        | (u64::from(data[3]) << 7)
        | u64::from(data[4] >> 1))
}

#[cfg(feature = "mpeg-ts")]
struct MpegAudioHeader {
    frame_length: usize,
    sample_rate: u32,
    channels: u8,
    samples_per_frame: u32,
    layer: u8,
}

#[cfg(feature = "mpeg-ts")]
fn parse_mpeg_audio_header(data: &[u8]) -> Option<MpegAudioHeader> {
    if data.len() < 4 || data[0] != 0xff || data[1] & 0xe0 != 0xe0 {
        return None;
    }
    let version_bits = (data[1] >> 3) & 0x03;
    let layer_bits = (data[1] >> 1) & 0x03;
    if version_bits == 1 || layer_bits == 0 {
        return None;
    }
    let layer = 4 - layer_bits;
    let bitrate_index = usize::from(data[2] >> 4);
    let sample_rate_index = usize::from((data[2] >> 2) & 0x03);
    if bitrate_index == 0 || bitrate_index == 15 || sample_rate_index == 3 {
        return None;
    }
    const MPEG1_LAYER1: [u16; 14] = [
        32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448,
    ];
    const MPEG1_LAYER2: [u16; 14] = [
        32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384,
    ];
    const MPEG1_LAYER3: [u16; 14] = [
        32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320,
    ];
    const MPEG2_LAYER1: [u16; 14] = [
        32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256,
    ];
    const MPEG2_LAYER23: [u16; 14] = [8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160];
    let mpeg1 = version_bits == 3;
    let bitrate_kbps = match (mpeg1, layer) {
        (true, 1) => MPEG1_LAYER1[bitrate_index - 1],
        (true, 2) => MPEG1_LAYER2[bitrate_index - 1],
        (true, 3) => MPEG1_LAYER3[bitrate_index - 1],
        (false, 1) => MPEG2_LAYER1[bitrate_index - 1],
        (false, 2 | 3) => MPEG2_LAYER23[bitrate_index - 1],
        _ => return None,
    };
    let base_rate = [44_100u32, 48_000, 32_000][sample_rate_index];
    let sample_rate = match version_bits {
        3 => base_rate,
        2 => base_rate / 2,
        0 => base_rate / 4,
        _ => return None,
    };
    let bitrate = u32::from(bitrate_kbps) * 1000;
    let padding = u32::from((data[2] >> 1) & 1);
    let (frame_length, samples_per_frame) = match layer {
        1 => (((12 * bitrate / sample_rate) + padding) * 4, 384),
        2 => ((144 * bitrate / sample_rate) + padding, 1152),
        3 if mpeg1 => ((144 * bitrate / sample_rate) + padding, 1152),
        3 => ((72 * bitrate / sample_rate) + padding, 576),
        _ => return None,
    };
    Some(MpegAudioHeader {
        frame_length: usize::try_from(frame_length).ok()?,
        sample_rate,
        channels: if data[3] >> 6 == 3 { 1 } else { 2 },
        samples_per_frame,
        layer,
    })
}

#[cfg(feature = "mpeg-ts")]
struct AdtsHeader {
    frame_length: usize,
    sample_rate: u32,
    channels: u8,
    samples_per_frame: u32,
}

#[cfg(feature = "mpeg-ts")]
fn parse_adts_header(data: &[u8]) -> Option<AdtsHeader> {
    if data.len() < 7 || data[0] != 0xff || data[1] & 0xf0 != 0xf0 {
        return None;
    }

    let layer = (data[1] & 0x06) >> 1;
    if layer != 0 {
        return None;
    }

    let sample_rate_index = (data[2] & 0x3c) >> 2;
    let sample_rate = adts_sample_rate(sample_rate_index)?;
    let channels = ((data[2] & 0x01) << 2) | ((data[3] & 0xc0) >> 6);
    let frame_length =
        ((data[3] as usize & 0x03) << 11) | ((data[4] as usize) << 3) | (data[5] as usize >> 5);

    if frame_length < 7 {
        return None;
    }

    Some(AdtsHeader {
        frame_length,
        sample_rate,
        channels,
        samples_per_frame: 1024 * (u32::from(data[6] & 0x03) + 1),
    })
}

#[cfg(any(feature = "mp4", feature = "mpeg-ts"))]
fn adts_sample_rate(index: u8) -> Option<u32> {
    match index {
        0 => Some(96_000),
        1 => Some(88_200),
        2 => Some(64_000),
        3 => Some(48_000),
        4 => Some(44_100),
        5 => Some(32_000),
        6 => Some(24_000),
        7 => Some(22_050),
        8 => Some(16_000),
        9 => Some(12_000),
        10 => Some(11_025),
        11 => Some(8_000),
        12 => Some(7_350),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(any(feature = "mp4", feature = "mpeg-ts"))]
    use sha2::{Digest, Sha256};
    #[cfg(any(feature = "mp4", feature = "webm", feature = "mpeg-ts"))]
    use std::fs;
    #[cfg(any(feature = "mp4", feature = "webm", feature = "mpeg-ts"))]
    use std::path::PathBuf;

    #[cfg(any(feature = "mp4", feature = "mpeg-ts"))]
    fn fixture(path: &str) -> Vec<u8> {
        fs::read(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("testdata")
                .join(path),
        )
        .unwrap()
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn mp4_tables_reject_hostile_counts_before_allocation() {
        let mut counted_header = vec![0u8; 12];
        counted_header[4..8].copy_from_slice(&u32::MAX.to_be_bytes());
        counted_header[8..12].copy_from_slice(&u32::MAX.to_be_bytes());

        let parsers = [
            std::panic::catch_unwind(|| parse_stts(&counted_header).map(|_| ())),
            std::panic::catch_unwind(|| parse_ctts(&counted_header).map(|_| ())),
            std::panic::catch_unwind(|| parse_stsc(&counted_header).map(|_| ())),
            std::panic::catch_unwind(|| parse_stco(&counted_header).map(|_| ())),
            std::panic::catch_unwind(|| parse_co64(&counted_header).map(|_| ())),
            std::panic::catch_unwind(|| parse_stss(&counted_header).map(|_| ())),
            std::panic::catch_unwind(|| parse_elst(&counted_header).map(|_| ())),
            std::panic::catch_unwind(|| parse_trun(&counted_header).map(|_| ())),
        ];
        for result in parsers {
            assert!(result.is_ok(), "MP4 table parser panicked");
            assert!(result.unwrap().is_err(), "hostile table count was accepted");
        }

        let mut variable_stsz = vec![0u8; 12];
        variable_stsz[8..12].copy_from_slice(&u32::MAX.to_be_bytes());
        let result = std::panic::catch_unwind(|| parse_stsz(&variable_stsz));
        assert!(result.is_ok(), "MP4 stsz parser panicked");
        assert!(result.unwrap().is_err());
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn auto_detects_fragmented_mp4_after_large_top_level_boxes() {
        let baseline = synthetic_fmp4_segment();
        let ftyp_size = Mp4BoxHeader::read(&baseline).unwrap().size;
        let mut padded = baseline[..ftyp_size].to_vec();
        padded.extend(mp4_box(b"free", &vec![0u8; 64 * 1024]));
        padded.extend_from_slice(&baseline[ftyp_size..]);
        assert_eq!(
            classify_mp4_layout(&padded).unwrap(),
            Some(Mp4Layout::Fragmented)
        );

        let decode = |chunk_size: usize| {
            let mut demuxer = AudioTrackDemuxer::new_auto();
            let mut events = Vec::new();
            for chunk in padded.chunks(chunk_size) {
                events.extend(demuxer.push(chunk).unwrap());
            }
            events.extend(demuxer.flush().unwrap());
            events
        };
        let expected = decode(4 * 1024 * 1024);
        assert!(!expected.is_empty());
        for chunk_size in [1, 188, 4 * 1024, 64 * 1024] {
            assert_eq!(decode(chunk_size), expected, "chunk size {chunk_size}");
        }
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn real_mp4_and_fmp4_match_ffprobe_reference() {
        let regular = fixture("video-compat/never-final/h264-high-aac.mp4");
        let index = Mp4MediaIndex::from_file(&regular).unwrap();
        let video = index
            .tracks
            .iter()
            .find(|track| track.kind == MediaTrackKind::Video)
            .unwrap();
        let audio = index
            .tracks
            .iter()
            .find(|track| track.kind == MediaTrackKind::Audio)
            .unwrap();
        assert_eq!(
            (video.codec.as_str(), video.timescale, video.sample_count),
            ("h264", 12_800, 75)
        );
        assert_eq!(
            (audio.codec.as_str(), audio.timescale, audio.sample_count),
            ("aac", 48_000, 142)
        );
        assert_eq!(index.samples.len(), 217);
        assert!(index.samples.iter().all(|sample| {
            sample
                .absolute_offset
                .checked_add(u64::from(sample.size))
                .is_some_and(|end| end <= regular.len() as u64)
        }));

        let fragmented = fixture("video-compat/never-final/h264-aac-fragmented.mp4");
        let mut demuxer = Mp4MediaDemuxer::new();
        let mut events = Vec::new();
        for chunk in fragmented.chunks(4_093) {
            events.extend(demuxer.push(chunk).unwrap());
        }
        events.extend(demuxer.flush().unwrap());
        let packets = events
            .iter()
            .filter_map(|event| match event {
                Mp4MediaDemuxEvent::Packet(packet) => Some(packet),
                _ => None,
            })
            .collect::<Vec<_>>();
        let video_packets = packets
            .iter()
            .copied()
            .filter(|packet| packet.kind == MediaTrackKind::Video)
            .collect::<Vec<_>>();
        let audio_packets = packets
            .iter()
            .copied()
            .filter(|packet| packet.kind == MediaTrackKind::Audio)
            .collect::<Vec<_>>();
        assert_eq!(video_packets.len(), 75);
        assert_eq!(audio_packets.len(), 142);
        assert_eq!(
            (
                audio_packets[0].presentation_time,
                audio_packets[0].decode_time,
                audio_packets[0].duration,
            ),
            (0, 0, 3_840)
        );
        assert_eq!(
            (
                audio_packets[141].presentation_time,
                audio_packets[141].decode_time,
                audio_packets[141].duration,
            ),
            (147_200, 147_200, 640)
        );
        assert_eq!(
            (
                video_packets[0].presentation_time,
                video_packets[0].decode_time,
                video_packets[0].duration,
            ),
            (1_024, 0, 512)
        );
        assert_eq!(
            (
                video_packets[74].presentation_time,
                video_packets[74].decode_time,
                video_packets[74].duration,
            ),
            (38_400, 37_888, 512)
        );
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn mvex_classifies_an_initialization_segment_without_moof() {
        let ftyp = mp4_box(
            b"ftyp",
            &[b"isom".as_slice(), &0u32.to_be_bytes(), b"iso6", b"cmfc"].concat(),
        );
        let initialization = [ftyp, synthetic_init_moov()].concat();
        assert_eq!(
            classify_mp4_layout(&initialization).unwrap(),
            Some(Mp4Layout::Fragmented)
        );
        let mut demuxer = AudioTrackDemuxer::new_auto();
        let mut events = demuxer.push(&initialization).unwrap();
        events.extend(demuxer.flush().unwrap());
        assert!(events
            .iter()
            .any(|event| matches!(event, AudioDemuxEvent::Config(_))));
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn quicktime_pcm_enda_controls_integer_and_float_byte_order() {
        for (codec, bits, float) in [
            (*b"in24", 24, false),
            (*b"in32", 32, false),
            (*b"fl32", 32, true),
            (*b"fl64", 64, true),
        ] {
            for (enda, expected) in [(0, PcmEndianness::Big), (1, PcmEndianness::Little)] {
                let entry = quicktime_pcm_entry(codec, 0, bits, Some(enda), None);
                let parsed = parse_stsd_audio(&stsd_payload(entry)).unwrap().unwrap();
                assert_eq!(parsed.pcm_endianness, Some(expected), "{codec:?}");
                assert_eq!(parsed.bits_per_sample, Some(bits), "{codec:?}");
                assert_eq!(parsed.pcm_float, Some(float), "{codec:?}");
                assert_eq!(parsed.pcm_interleaved, Some(true), "{codec:?}");
            }
        }

        let default_in24 = parse_stsd_audio(&stsd_payload(quicktime_pcm_entry(
            *b"in24", 0, 24, None, None,
        )))
        .unwrap()
        .unwrap();
        assert_eq!(default_in24.pcm_endianness, Some(PcmEndianness::Big));
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn committed_mov_pcm_endian_fixtures_match_ffmpeg_hashes() {
        let cases = [
            (
                "pcm-s24be.mov",
                "in24",
                24,
                PcmEndianness::Big,
                false,
                "c4e54b143defc7ce84e84c3b6fe96b1f87423477a8ca0f6676085b35eba7d087",
            ),
            (
                "pcm-s24le.mov",
                "in24",
                24,
                PcmEndianness::Little,
                false,
                "c4e54b143defc7ce84e84c3b6fe96b1f87423477a8ca0f6676085b35eba7d087",
            ),
            (
                "pcm-s32be.mov",
                "in32",
                32,
                PcmEndianness::Big,
                false,
                "1568245a441f20f5c94b0810fa460c820f0e7d65ab2c6091100bf8c23ad7913c",
            ),
            (
                "pcm-s32le.mov",
                "in32",
                32,
                PcmEndianness::Little,
                false,
                "1568245a441f20f5c94b0810fa460c820f0e7d65ab2c6091100bf8c23ad7913c",
            ),
            (
                "pcm-f32be.mov",
                "fl32",
                32,
                PcmEndianness::Big,
                true,
                "31522f2b9f09614ab868408fd95f5d826028b61407827b7b112d0ab9981ad65e",
            ),
            (
                "pcm-f32le.mov",
                "fl32",
                32,
                PcmEndianness::Little,
                true,
                "31522f2b9f09614ab868408fd95f5d826028b61407827b7b112d0ab9981ad65e",
            ),
            (
                "pcm-f64be.mov",
                "fl64",
                64,
                PcmEndianness::Big,
                true,
                "0a01dbaba912df543bf79d7780f4853fccfb612c8c5276d11ba01c0563879d7e",
            ),
            (
                "pcm-f64le.mov",
                "fl64",
                64,
                PcmEndianness::Little,
                true,
                "0a01dbaba912df543bf79d7780f4853fccfb612c8c5276d11ba01c0563879d7e",
            ),
        ];

        for (name, codec_id, bits, endianness, float, expected_hash) in cases {
            let file = fixture(&format!("mov-pcm/{name}"));
            let index = Mp4MediaIndex::from_file(&file).unwrap();
            let track = index
                .tracks
                .iter()
                .find(|track| track.kind == MediaTrackKind::Audio)
                .unwrap();
            assert_eq!(track.codec_id, codec_id, "{name}");
            assert_eq!(track.sample_rate, Some(48_000), "{name}");
            assert_eq!(track.channels, Some(2), "{name}");
            assert_eq!(track.bits_per_sample, Some(bits), "{name}");
            assert_eq!(track.pcm_endianness, Some(endianness), "{name}");
            assert_eq!(track.pcm_float, Some(float), "{name}");
            assert_eq!(track.sample_count, 2, "{name}");

            let samples = index
                .samples
                .iter()
                .enumerate()
                .filter(|(_, sample)| sample.track_id == track.track_id)
                .collect::<Vec<_>>();
            assert_eq!(
                samples
                    .iter()
                    .map(|(_, sample)| u64::from(sample.duration))
                    .sum::<u64>(),
                4_800,
                "{name}"
            );
            let mut canonical = Vec::new();
            for (sample_index, sample) in samples {
                let start = usize::try_from(sample.absolute_offset).unwrap();
                let end = start + sample.size as usize;
                let packet = index
                    .packet_from_sample_bytes(sample_index, &file[start..end])
                    .unwrap();
                canonical.extend_from_slice(&packet.data);
            }
            let bytes_per_sample = usize::from(bits.div_ceil(8));
            if endianness == PcmEndianness::Big {
                for sample in canonical.chunks_exact_mut(bytes_per_sample) {
                    sample.reverse();
                }
            }
            assert_eq!(
                format!("{:x}", Sha256::digest(&canonical)),
                expected_hash,
                "{name}"
            );
        }
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn parses_quicktime_version_one_and_two_lpcm_geometry() {
        let version_one = parse_stsd_audio(&stsd_payload(quicktime_pcm_entry(
            *b"sowt", 1, 16, None, None,
        )))
        .unwrap()
        .unwrap();
        assert_eq!(version_one.pcm_bytes_per_frame, Some(4));
        assert_eq!(version_one.pcm_frames_per_packet, Some(1));

        const SIGNED_PACKED_LITTLE: u32 = (1 << 2) | (1 << 3);
        let version_two = parse_stsd_audio(&stsd_payload(quicktime_pcm_entry(
            *b"lpcm",
            2,
            24,
            None,
            Some(SIGNED_PACKED_LITTLE),
        )))
        .unwrap()
        .unwrap();
        assert_eq!(version_two.sample_rate, 48_000);
        assert_eq!(version_two.channels, 2);
        assert_eq!(version_two.bits_per_sample, Some(24));
        assert_eq!(version_two.pcm_endianness, Some(PcmEndianness::Little));
        assert_eq!(version_two.pcm_signed, Some(true));
        assert_eq!(version_two.pcm_packed, Some(true));
        assert_eq!(version_two.pcm_aligned_high, Some(false));
        assert_eq!(version_two.pcm_interleaved, Some(true));
        assert_eq!(version_two.pcm_bytes_per_frame, Some(6));

        const NON_INTERLEAVED: u32 = SIGNED_PACKED_LITTLE | (1 << 5);
        let error = parse_stsd_audio(&stsd_payload(quicktime_pcm_entry(
            *b"lpcm",
            2,
            24,
            None,
            Some(NON_INTERLEAVED),
        )))
        .unwrap_err();
        assert!(error.contains("non-interleaved"));
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn fragmented_mp4_inherits_base_offsets_across_traf_boxes() {
        let first = synthetic_traf(1, Some(2_000), false, None, Some(5), 10);
        let second = synthetic_traf(2, None, false, None, Some(7), 12);
        let moof_payload = [first, second].concat();
        let mut demuxer = Mp4MediaDemuxer::new();
        demuxer.tracks = vec![dummy_fragment_track(1), dummy_fragment_track(2)];

        let fragments = demuxer.parse_moof(&moof_payload, 1_000).unwrap();
        assert_eq!(fragments.len(), 2);
        assert_eq!(fragments[0].samples[0].absolute_offset, 2_000);
        assert_eq!(fragments[1].samples[0].absolute_offset, 2_010);

        let moof_relative = synthetic_traf(1, None, true, None, Some(5), 10);
        let fragments = demuxer.parse_moof(&moof_relative, 1_000).unwrap();
        assert_eq!(fragments[0].samples[0].absolute_offset, 1_000);

        let missing_duration = synthetic_traf(1, None, true, None, None, 10);
        assert!(demuxer
            .parse_moof(&missing_duration, 1_000)
            .unwrap_err()
            .contains("no duration"));
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn constant_pcm_stsz_stays_compact_and_builds_packet_runs() {
        const FRAME_COUNT: u32 = 1_000_000;
        let mut stsz = vec![0u8; 12];
        stsz[4..8].copy_from_slice(&4u32.to_be_bytes());
        stsz[8..12].copy_from_slice(&FRAME_COUNT.to_be_bytes());
        let sizes = parse_stsz(&stsz).unwrap();
        assert!(matches!(
            &sizes,
            Mp4SampleSizes::Constant {
                size: 4,
                count: FRAME_COUNT
            }
        ));

        let tables = RegularTrakTables {
            track_id: Some(1),
            is_audio: true,
            timescale: Some(48_000),
            sample_entry: Some(RegularAudioSampleEntry {
                sample_rate: 48_000,
                channels: 2,
                bits_per_sample: Some(16),
                codec: AudioCodec::Pcm,
                codec_id: "lpcm".to_string(),
                packet_format: AudioPacketFormat::Raw,
                pcm_endianness: Some(PcmEndianness::Little),
                pcm_float: Some(false),
                pcm_signed: Some(true),
                pcm_packed: Some(true),
                pcm_aligned_high: Some(false),
                pcm_interleaved: Some(true),
                pcm_bytes_per_frame: Some(4),
                pcm_frames_per_packet: Some(1),
                codec_private: Vec::new(),
            }),
            stts: vec![SttsEntry {
                sample_count: FRAME_COUNT,
                sample_duration: 1,
            }],
            ctts: Vec::new(),
            stsc: vec![StscEntry {
                first_chunk: 1,
                samples_per_chunk: FRAME_COUNT,
                sample_description_index: 1,
            }],
            sample_sizes: sizes,
            chunk_offsets: vec![1_024],
            sync_samples: Vec::new(),
        };
        let samples = build_regular_samples(&tables).unwrap();
        assert_eq!(samples.len(), FRAME_COUNT.div_ceil(4096) as usize);
        assert_eq!(samples.first().unwrap().sample_id, 1);
        assert_eq!(samples.last().unwrap().sample_id, 999_425);
        assert_eq!(
            samples
                .iter()
                .map(|sample| sample.duration as u64)
                .sum::<u64>(),
            u64::from(FRAME_COUNT)
        );
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn inspects_large_mp4_box_ranges_without_payload_bytes() {
        let file_size = 8_000_000_032u64;
        let mut extended = Vec::from(&[0, 0, 0, 1, b'm', b'd', b'a', b't'][..]);
        extended.extend_from_slice(&8_000_000_016u64.to_be_bytes());
        let mdat = inspect_mp4_top_level_box(&extended, 16, file_size).unwrap();
        assert_eq!(mdat.box_type, *b"mdat");
        assert_eq!(mdat.payload_offset, 32);
        assert_eq!(mdat.payload_size, 8_000_000_000);
        assert_eq!(mdat.end, file_size);

        let moov = inspect_mp4_top_level_box(&[0, 0, 0, 16, b'm', b'o', b'o', b'v'], 0, file_size)
            .unwrap();
        assert_eq!(moov.payload_offset, 8);
        assert_eq!(moov.payload_size, 8);
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn rejects_mp4_box_ranges_outside_the_source() {
        assert!(
            inspect_mp4_top_level_box(&[0, 0, 0, 32, b'm', b'o', b'o', b'v'], 90, 100,)
                .unwrap_err()
                .contains("exceeds")
        );
        assert!(
            inspect_mp4_top_level_box(&[0, 0, 0, 4, b'f', b'r', b'e', b'e'], 0, 100,)
                .unwrap_err()
                .contains("shorter")
        );
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn seekable_mp4_rejects_oversized_samples_before_reading_payload() {
        let index = Mp4MediaIndex {
            tracks: Vec::new(),
            samples: vec![MediaSampleIndex {
                track_id: 1,
                kind: MediaTrackKind::Video,
                codec: "avc1".to_string(),
                sample_id: 1,
                absolute_offset: 128,
                size: MAX_MEDIA_PACKET_BYTES + 1,
                decode_time: 0,
                presentation_time: 0,
                duration: 1,
                is_sync: true,
            }],
        };
        let error = index.packet_from_sample_bytes(0, &[]).unwrap_err();
        assert!(error.contains("packet budget"));
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn parses_and_resolves_linear_mp4_edit_timeline() {
        let mut payload = vec![0, 0, 0, 0];
        payload.extend_from_slice(&2u32.to_be_bytes());
        payload.extend_from_slice(&500u32.to_be_bytes());
        payload.extend_from_slice(&(-1i32).to_be_bytes());
        payload.extend_from_slice(&1i16.to_be_bytes());
        payload.extend_from_slice(&0i16.to_be_bytes());
        payload.extend_from_slice(&3_000u32.to_be_bytes());
        payload.extend_from_slice(&1_024i32.to_be_bytes());
        payload.extend_from_slice(&1i16.to_be_bytes());
        payload.extend_from_slice(&0i16.to_be_bytes());

        let entries = parse_elst(&payload).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(
            resolve_media_timeline(&entries, Some(1_000), 48_000).unwrap(),
            vec![MediaTrackTimeline {
                presentation_start: 24_000,
                media_start: 1_024,
                duration: 144_000,
            }]
        );
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn preserves_multiple_media_edits_and_empty_gaps() {
        let entries = [
            Mp4EditListEntry {
                segment_duration: 100,
                media_time: 0,
                media_rate_integer: 1,
                media_rate_fraction: 0,
            },
            Mp4EditListEntry {
                segment_duration: 50,
                media_time: -1,
                media_rate_integer: 1,
                media_rate_fraction: 0,
            },
            Mp4EditListEntry {
                segment_duration: 200,
                media_time: 500,
                media_rate_integer: 1,
                media_rate_fraction: 0,
            },
        ];
        assert_eq!(
            resolve_media_timeline(&entries, Some(1_000), 1_000).unwrap(),
            [
                MediaTrackTimeline {
                    presentation_start: 0,
                    media_start: 0,
                    duration: 100,
                },
                MediaTrackTimeline {
                    presentation_start: 150,
                    media_start: 500,
                    duration: 200,
                },
            ]
        );
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn rejects_non_linear_mp4_edit_rate() {
        let entries = [Mp4EditListEntry {
            segment_duration: 1_000,
            media_time: 0,
            media_rate_integer: 0,
            media_rate_fraction: 16_384,
        }];
        assert!(resolve_media_timeline(&entries, Some(1_000), 48_000)
            .unwrap_err()
            .contains("unsupported MP4 edit rate"));
    }

    #[test]
    fn trims_pcm_preroll_and_tail_to_the_rust_timeline() {
        let timeline = MediaTrackTimeline {
            presentation_start: 0,
            media_start: 1_024,
            duration: 144_000,
        };
        assert_eq!(
            resolve_pcm_packet_trim(timeline, -1_024, 1_024, 1_024, 48_000, 48_000).unwrap(),
            None
        );
        assert_eq!(
            resolve_pcm_packet_trim(timeline, 0, 1_024, 1_024, 48_000, 48_000).unwrap(),
            Some(PcmPacketTrim {
                source_frame_start: 0,
                frame_count: 1_024,
            })
        );
        assert_eq!(
            resolve_pcm_packet_trim(timeline, 143_360, 640, 1_024, 48_000, 48_000).unwrap(),
            Some(PcmPacketTrim {
                source_frame_start: 0,
                frame_count: 640,
            })
        );
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn tail_moov_mp4_requires_seekable_packet_api() {
        let data = fixture("mac_aac/A_Tusk_is_used_to_make_costly_gifts.m4a");
        let mut demuxer = AudioTrackDemuxer::new_with_format("mp4").unwrap();
        let mut error = None;
        for chunk in data.chunks(997) {
            if let Err(found) = demuxer.push(chunk) {
                error = Some(found);
                break;
            }
        }
        assert!(error.unwrap().contains("seekable MP4 packet API"));
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn demuxes_regular_mp4_with_quicktime_alias_handler() {
        let data = synthetic_regular_mp4_with_alias_handler();
        let mut demuxer = AudioTrackDemuxer::new_with_format("mp4").unwrap();
        let mut events = Vec::new();
        for chunk in data.chunks(23) {
            events.extend(demuxer.push(chunk).unwrap());
        }
        events.extend(demuxer.flush().unwrap());

        assert!(events.iter().any(|event| matches!(
            event,
            AudioDemuxEvent::Config(AudioTrackConfig {
                container: AudioContainer::Mp4,
                codec: AudioCodec::Aac,
                packet_format: Some(AudioPacketFormat::Adts),
                sample_rate: Some(44_100),
                channels: Some(2),
                track_id: Some(1),
                sample_count: Some(2),
                ..
            })
        )));

        let packets: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                AudioDemuxEvent::Packet(packet) => Some(packet),
                _ => None,
            })
            .collect();
        assert_eq!(packets.len(), 2);
        assert_eq!(packets[0].sample_id, Some(1));
        assert_eq!(packets[0].start_time, Some(0));
        assert_eq!(
            packets[0].raw_data.as_deref(),
            Some(&[0x11, 0x22, 0x33][..])
        );
        assert_eq!(packets[1].sample_id, Some(2));
        assert_eq!(packets[1].start_time, Some(1024));
        assert_eq!(packets[1].raw_data.as_deref(), Some(&[0x44, 0x55][..]));
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn regular_mp4_rejects_mdat_before_moov_without_buffering_payload() {
        let mut demuxer = RegularMp4AudioDemuxer::new();
        let ftyp = mp4_box(
            b"ftyp",
            &[b"isom".as_slice(), &0u32.to_be_bytes(), b"isom"].concat(),
        );
        demuxer.add(&ftyp).unwrap();

        let mut mdat_header = Vec::new();
        mdat_header.extend_from_slice(&8_000_000u32.to_be_bytes());
        mdat_header.extend_from_slice(b"mdat");
        let error = demuxer.add(&mdat_header).unwrap_err();
        assert!(error.contains("seekable MP4 packet API"));
        assert!(demuxer.buffer.len() < 64);
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn fragmented_mp4_skips_large_unknown_boxes_incrementally() {
        let box_size = 8 * 1024 * 1024u32;
        let mut header = Vec::new();
        header.extend_from_slice(&box_size.to_be_bytes());
        header.extend_from_slice(b"free");
        let mut demuxer = Mp4MediaDemuxer::new();
        demuxer.push(&header).unwrap();
        let zeros = [0u8; 1024];
        let mut remaining = box_size as usize - header.len();
        while remaining > 0 {
            let count = remaining.min(zeros.len());
            demuxer.push(&zeros[..count]).unwrap();
            assert!(demuxer.buffer.len() <= zeros.len());
            remaining -= count;
        }
        demuxer.flush().unwrap();
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn fragmented_mp4_rejects_oversized_metadata_from_header() {
        let size = u32::try_from(MAX_MP4_METADATA_BYTES + 9).unwrap();
        let mut header = Vec::new();
        header.extend_from_slice(&size.to_be_bytes());
        header.extend_from_slice(b"moof");
        let error = Mp4MediaDemuxer::new().push(&header).unwrap_err();
        assert!(error.contains("metadata"));
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn fragmented_mp4_rejects_oversized_samples_from_metadata() {
        let mfhd = full_box(b"mfhd", 0, 0, &1u32.to_be_bytes());
        let tfhd = full_box(b"tfhd", 0, 0x020000, &1u32.to_be_bytes());
        let tfdt = full_box(b"tfdt", 0, 0, &0u32.to_be_bytes());
        let trun_payload = [
            &1u32.to_be_bytes()[..],
            &1024u32.to_be_bytes(),
            &(MAX_MEDIA_PACKET_BYTES + 1).to_be_bytes(),
        ]
        .concat();
        let trun = full_box(b"trun", 0, 0x000300, &trun_payload);
        let traf = mp4_box(b"traf", &[tfhd, tfdt, trun].concat());
        let moof = mp4_box(b"moof", &[mfhd, traf].concat());
        let mut demuxer = Mp4MediaDemuxer::new();
        demuxer.push(&synthetic_init_moov()).unwrap();
        let error = demuxer.push(&moof).unwrap_err();
        assert!(error.contains("packet budget"));
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn converts_mp4_dfla_to_a_decoder_ready_flac_stream() {
        let mut dfla = vec![0, 0, 0, 0, 0x80, 0, 0, 34];
        dfla.extend_from_slice(&[0x55; 34]);
        let stream = normalize_mp4_flac_decoder_configuration(&dfla).unwrap();
        assert_eq!(&stream[..8], b"fLaC\x80\0\0\x22");
        assert_eq!(&stream[8..], &[0x55; 34]);

        let bare_streaminfo = normalize_mp4_flac_decoder_configuration(&[0x33; 34]).unwrap();
        assert_eq!(&bare_streaminfo[..8], b"fLaC\x80\0\0\x22");
        assert!(normalize_mp4_flac_decoder_configuration(&[0; 7]).is_err());
    }

    #[cfg(feature = "webm")]
    #[test]
    fn demuxes_webm_audio() {
        let data = fs::read(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("soundkit-webm")
                .join("testdata")
                .join("test.webm"),
        )
        .unwrap();
        let mut demuxer = AudioTrackDemuxer::new_with_format("webm").unwrap();
        let mut events = Vec::new();
        for chunk in data.chunks(997) {
            events.extend(demuxer.push(chunk).unwrap());
        }
        events.extend(demuxer.flush().unwrap());

        assert!(events.iter().any(|event| matches!(
            event,
            AudioDemuxEvent::Config(AudioTrackConfig {
                container: AudioContainer::WebM,
                codec: AudioCodec::Opus,
                sample_rate: Some(48_000),
                channels: Some(1),
                ..
            })
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            AudioDemuxEvent::Packet(AudioTrackPacket {
                container: AudioContainer::WebM,
                format: AudioPacketFormat::Raw,
                ..
            })
        )));
    }

    #[cfg(feature = "webm")]
    #[test]
    fn webm_audio_flush_validates_the_container_tail() {
        let mut data = fs::read(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("soundkit-webm")
                .join("testdata")
                .join("test.webm"),
        )
        .unwrap();
        data.truncate(data.len() - 1);
        let mut demuxer = AudioTrackDemuxer::new_with_format("webm").unwrap();
        for chunk in data.chunks(997) {
            demuxer.push(chunk).unwrap();
        }
        assert!(demuxer
            .flush()
            .unwrap_err()
            .contains("truncated WebM element"));
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn demuxes_fragmented_mp4_cmaf_aac() {
        let data = synthetic_fmp4_segment();
        let mut demuxer = AudioTrackDemuxer::new_with_format("fmp4").unwrap();
        let mut events = Vec::new();
        for chunk in data.chunks(31) {
            events.extend(demuxer.push(chunk).unwrap());
        }
        events.extend(demuxer.flush().unwrap());

        assert!(events.iter().any(|event| matches!(
            event,
            AudioDemuxEvent::Config(AudioTrackConfig {
                container: AudioContainer::Mp4,
                codec: AudioCodec::Aac,
                packet_format: Some(AudioPacketFormat::Adts),
                sample_rate: Some(44_100),
                channels: Some(2),
                track_id: Some(1),
                sample_count: None,
                ..
            })
        )));

        let packets: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                AudioDemuxEvent::Packet(packet) => Some(packet),
                _ => None,
            })
            .collect();
        assert_eq!(packets.len(), 2);
        assert_eq!(packets[0].sample_id, Some(1));
        assert_eq!(packets[0].start_time, Some(0));
        assert_eq!(packets[0].duration, Some(1024));
        assert!(packets[0].data.starts_with(&[0xff, 0xf1]));
        assert_eq!(
            packets[0].raw_data.as_deref(),
            Some(&[0x11, 0x22, 0x33][..])
        );
        assert_eq!(packets[1].sample_id, Some(2));
        assert_eq!(packets[1].start_time, Some(1024));
        assert_eq!(packets[1].raw_data.as_deref(), Some(&[0x44, 0x55][..]));
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn streams_fragmented_mp4_video_and_audio_from_one_rust_demuxer() {
        let data = fixture("video-compat/never-final/h264-aac-fragmented.mp4");
        let mut demuxer = Mp4MediaDemuxer::new();
        let mut events = Vec::new();
        let mut first_packet_before_eof = false;
        for (index, chunk) in data.chunks(4093).enumerate() {
            let emitted = demuxer.push(chunk).unwrap();
            if (index + 1) * 4093 < data.len()
                && emitted
                    .iter()
                    .any(|event| matches!(event, Mp4MediaDemuxEvent::Packet(_)))
            {
                first_packet_before_eof = true;
            }
            events.extend(emitted);
        }
        events.extend(demuxer.flush().unwrap());
        assert!(
            first_packet_before_eof,
            "fragmented mdat must release samples before EOF"
        );

        let configs: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                Mp4MediaDemuxEvent::Config(config) => Some(config),
                _ => None,
            })
            .collect();
        assert_eq!(configs.len(), 2);
        assert!(configs.iter().any(|config| {
            config.kind == MediaTrackKind::Video
                && config.codec == "h264"
                && !config.decoder_configuration.is_empty()
        }));
        assert!(configs.iter().any(|config| {
            config.kind == MediaTrackKind::Audio
                && config.codec == "aac"
                && config.sample_rate == Some(48_000)
                && config.channels == Some(2)
        }));

        let packets: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                Mp4MediaDemuxEvent::Packet(packet) => Some(packet),
                _ => None,
            })
            .collect();
        assert_eq!(
            packets
                .iter()
                .filter(|packet| packet.kind == MediaTrackKind::Video)
                .count(),
            75
        );
        assert_eq!(
            packets
                .iter()
                .filter(|packet| packet.kind == MediaTrackKind::Audio)
                .count(),
            142
        );
        assert!(packets.iter().all(|packet| !packet.data.is_empty()));
        for kind in [MediaTrackKind::Video, MediaTrackKind::Audio] {
            assert!(packets
                .iter()
                .filter(|packet| packet.kind == kind)
                .enumerate()
                .all(|(index, packet)| packet.sample_id == index as u32 + 1));
        }
    }

    #[cfg(feature = "mpeg-ts")]
    #[test]
    fn demuxes_mpeg_ts_aac_adts() {
        let data = synthetic_ts_segment();
        let mut demuxer = AudioTrackDemuxer::new_with_format("mpeg-ts").unwrap();
        let mut events = Vec::new();
        for chunk in data.chunks(113) {
            events.extend(demuxer.push(chunk).unwrap());
        }
        events.extend(demuxer.flush().unwrap());

        assert!(events.iter().any(|event| matches!(
            event,
            AudioDemuxEvent::Config(AudioTrackConfig {
                container: AudioContainer::MpegTs,
                codec: AudioCodec::Aac,
                packet_format: Some(AudioPacketFormat::Adts),
                sample_rate: Some(44_100),
                channels: Some(2),
                pid: Some(0x0101),
                stream_type: Some(0x0f),
                ..
            })
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            AudioDemuxEvent::Packet(AudioTrackPacket {
                container: AudioContainer::MpegTs,
                codec: AudioCodec::Aac,
                format: AudioPacketFormat::Adts,
                data,
                ..
            }) if data.starts_with(&[0xff, 0xf1])
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            AudioDemuxEvent::Packet(AudioTrackPacket {
                timescale: Some(90_000),
                start_time: Some(90_000),
                decode_time: Some(90_000),
                duration: Some(_),
                ..
            })
        )));
    }

    #[cfg(feature = "mpeg-ts")]
    #[test]
    fn detects_m2ts_stride_and_is_chunk_invariant() {
        let ts = synthetic_ts_segment();
        let mut m2ts = Vec::new();
        for packet in ts.chunks_exact(188) {
            m2ts.extend_from_slice(&0_u32.to_be_bytes());
            m2ts.extend_from_slice(packet);
        }
        assert_eq!(
            detect_ts_layout(&m2ts),
            Some(TsPacketLayout {
                stride: 192,
                prefix: 4,
            })
        );

        let collect = |chunk_size: usize| {
            let mut demuxer = AudioTrackDemuxer::new_with_format("m2ts").unwrap();
            let mut events = Vec::new();
            for chunk in m2ts.chunks(chunk_size) {
                events.extend(demuxer.push(chunk).unwrap());
            }
            events.extend(demuxer.flush().unwrap());
            events
        };
        let reference = collect(MAX_CONTAINER_INPUT_CHUNK_BYTES);
        assert_eq!(reference, collect(1));
        assert_eq!(reference, collect(188));
        assert_eq!(reference, collect(4 * 1024));
        assert_eq!(reference, collect(64 * 1024));
        assert!(reference.iter().any(|event| matches!(
            event,
            AudioDemuxEvent::Config(AudioTrackConfig {
                transport_packet_stride: Some(192),
                transport_prefix_bytes: Some(4),
                ..
            })
        )));
    }

    #[cfg(feature = "mpeg-ts")]
    #[test]
    fn real_ts_and_m2ts_match_ffprobe_packet_counts_and_chunking() {
        let cases = [
            ("mpeg-ts/aac-stereo-48k.ts", "mpeg-ts", "aac", 48usize),
            ("mpeg-ts/aac-stereo-48k.m2ts", "m2ts", "aac", 48usize),
            ("mpeg-ts/mp2-stereo-48k.ts", "mpeg-ts", "mp2", 42usize),
        ];

        for (path, format, codec, expected_packets) in cases {
            let data = fixture(path);
            let collect = |chunk_size: usize| {
                let mut demuxer = AudioTrackDemuxer::new_with_format(format).unwrap();
                let mut events = Vec::new();
                for chunk in data.chunks(chunk_size) {
                    events.extend(demuxer.push(chunk).unwrap());
                }
                events.extend(demuxer.flush().unwrap());
                events
            };
            let reference = collect(data.len());
            assert_eq!(reference, collect(1), "one-byte chunks changed {path}");
            assert_eq!(reference, collect(188), "packet chunks changed {path}");
            assert_eq!(reference, collect(4 * 1024), "4 KiB chunks changed {path}");
            assert_eq!(
                reference,
                collect(64 * 1024),
                "64 KiB chunks changed {path}"
            );
            assert_eq!(
                reference,
                collect(4 * 1024 * 1024),
                "4 MiB chunks changed {path}"
            );

            let config = reference
                .iter()
                .find_map(|event| match event {
                    AudioDemuxEvent::Config(config) => Some(config),
                    _ => None,
                })
                .unwrap_or_else(|| panic!("audio configuration for {path}"));
            assert_eq!(config.codec.as_str(), codec);
            assert_eq!(config.sample_rate, Some(48_000));
            assert_eq!(config.channels, Some(2));

            let packets = reference
                .iter()
                .filter_map(|event| match event {
                    AudioDemuxEvent::Packet(packet) => Some(packet),
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert_eq!(packets.len(), expected_packets, "FFprobe count for {path}");
            assert!(packets
                .iter()
                .all(|packet| packet.timescale == Some(90_000)));
            assert!(packets.windows(2).all(|pair| {
                match (pair[0].start_time, pair[1].start_time) {
                    (Some(left), Some(right)) => left <= right,
                    _ => true,
                }
            }));
        }
    }

    #[cfg(feature = "mpeg-ts")]
    #[test]
    fn real_m2ts_lpcm_and_dts_match_ffmpeg_references() {
        struct Case {
            path: &'static str,
            codec: AudioCodec,
            packets: usize,
            reference_sha256: &'static str,
            swap_s16_to_little_endian: bool,
        }
        let cases = [
            Case {
                path: "mpeg-ts/lpcm-stereo-48k.m2ts",
                codec: AudioCodec::Pcm,
                packets: 20,
                reference_sha256:
                    "7126a0d8077e0433c1e78df6c0f1074f78daf3800240ef48c936114ef7cbb563",
                swap_s16_to_little_endian: true,
            },
            Case {
                path: "mpeg-ts/dts-stereo-48k.m2ts",
                codec: AudioCodec::Dts,
                packets: 10,
                reference_sha256:
                    "a5472bde5d532950bdb12c27bb63bf190147c77e70f54d7e558090875a15ccd4",
                swap_s16_to_little_endian: false,
            },
        ];

        for case in cases {
            let data = fixture(case.path);
            let collect = |chunk_size: usize| {
                let mut demuxer = AudioTrackDemuxer::new_with_format("m2ts").unwrap();
                let mut events = Vec::new();
                for chunk in data.chunks(chunk_size) {
                    events.extend(demuxer.push(chunk).unwrap());
                }
                events.extend(demuxer.flush().unwrap());
                events
            };
            let reference = collect(data.len());
            for chunk_size in [1, 188, 4 * 1024, 64 * 1024, 4 * 1024 * 1024] {
                assert_eq!(
                    reference,
                    collect(chunk_size),
                    "chunk size {chunk_size} changed {}",
                    case.path
                );
            }

            let config = reference
                .iter()
                .find_map(|event| match event {
                    AudioDemuxEvent::Config(config) => Some(config),
                    _ => None,
                })
                .unwrap_or_else(|| panic!("audio configuration for {}", case.path));
            assert_eq!(config.codec, case.codec);
            assert_eq!(config.sample_rate, Some(48_000));
            assert_eq!(config.channels, Some(2));
            if case.codec == AudioCodec::Pcm {
                assert_eq!(config.bits_per_sample, Some(16));
                assert_eq!(config.pcm_endianness, Some(PcmEndianness::Big));
                assert_eq!(config.pcm_bytes_per_frame, Some(4));
            }

            let packets = reference
                .iter()
                .filter_map(|event| match event {
                    AudioDemuxEvent::Packet(packet) => Some(packet),
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert_eq!(
                packets.len(),
                case.packets,
                "FFprobe count for {}",
                case.path
            );
            assert!(packets.iter().all(|packet| packet.codec == case.codec));
            assert!(packets.iter().all(|packet| packet.duration.is_some()));
            let mut elementary = packets
                .iter()
                .flat_map(|packet| packet.data.iter().copied())
                .collect::<Vec<_>>();
            if case.swap_s16_to_little_endian {
                for sample in elementary.chunks_exact_mut(2) {
                    sample.swap(0, 1);
                }
            }
            assert_eq!(
                format!("{:x}", Sha256::digest(&elementary)),
                case.reference_sha256,
                "FFmpeg elementary-stream hash for {}",
                case.path
            );
        }
    }

    #[cfg(feature = "mpeg-ts")]
    #[test]
    fn reassembles_psi_sections_across_transport_packets() {
        let pat = pat_section(0x1000);
        let split = 7;
        let mut assembler = PsiSectionAssembler::default();
        assert!(assembler.push(&pat[..split], true).unwrap().is_empty());
        let sections = assembler.push(&pat[split..], false).unwrap();
        assert_eq!(sections, vec![pat[1..].to_vec()]);
        validate_psi_crc(&sections[0]).unwrap();

        let mut two_sections = vec![0];
        two_sections.extend_from_slice(&pat[1..]);
        two_sections.extend_from_slice(&pat[1..]);
        let sections = PsiSectionAssembler::default()
            .push(&two_sections, true)
            .unwrap();
        assert_eq!(sections.len(), 2);
    }

    #[cfg(feature = "mpeg-ts")]
    #[test]
    fn splits_loas_and_mpeg_layer_audio_access_units() {
        let mut demuxer = MpegTsAudioDemuxer::new();
        demuxer.audio_pid = Some(0x101);
        demuxer.audio_codec = Some(AudioCodec::Aac);
        demuxer.packet_format = Some(AudioPacketFormat::Latm);
        let loas = [0x56, 0xe0, 0x03, 1, 2, 3, 0x56, 0xe0, 0x02, 4, 5];
        let events = demuxer.emit_loas_frames(&loas, Some(90_000), None).unwrap();
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, AudioDemuxEvent::Packet(_)))
                .count(),
            2
        );

        let mp3 = parse_mpeg_audio_header(&[0xff, 0xfb, 0x90, 0x00]).unwrap();
        assert_eq!(
            (mp3.layer, mp3.sample_rate, mp3.frame_length),
            (3, 44_100, 417)
        );
        let mp2 = parse_mpeg_audio_header(&[0xff, 0xfd, 0x80, 0x00]).unwrap();
        assert_eq!(
            (mp2.layer, mp2.sample_rate, mp2.frame_length),
            (2, 44_100, 417)
        );

        let mut payload = vec![0; mp3.frame_length * 2];
        payload[..4].copy_from_slice(&[0xff, 0xfb, 0x90, 0x00]);
        payload[mp3.frame_length..mp3.frame_length + 4].copy_from_slice(&[0xff, 0xfb, 0x90, 0x00]);
        demuxer.emitted_config = false;
        demuxer.audio_codec = Some(AudioCodec::Mp3);
        demuxer.packet_format = Some(AudioPacketFormat::Raw);
        let events = demuxer
            .emit_mpeg_audio_frames(&payload, Some(180_000), Some(180_000))
            .unwrap();
        let packets = events
            .iter()
            .filter_map(|event| match event {
                AudioDemuxEvent::Packet(packet) => Some(packet),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(packets.len(), 2);
        assert!(packets[1].start_time > packets[0].start_time);
    }

    #[cfg(feature = "mpeg-ts")]
    #[test]
    fn mpeg_ts_rejects_unterminated_oversized_pes() {
        let mut demuxer = MpegTsAudioDemuxer::new();
        demuxer.layout = Some(TsPacketLayout {
            stride: 188,
            prefix: 0,
        });
        demuxer.audio_pid = Some(0x0101);
        demuxer.current_pes = vec![0; MAX_MPEG_TS_PES_BYTES];
        let packet = ts_packet(0x0101, false, &[0x11]);
        let error = demuxer.add(&packet).unwrap_err();
        assert!(error.contains("packet budget"));
    }

    #[cfg(feature = "mp4")]
    fn synthetic_fmp4_segment() -> Vec<u8> {
        let mut out = Vec::new();
        out.extend(mp4_box(
            b"ftyp",
            &[
                b"isom".as_slice(),
                &0u32.to_be_bytes(),
                b"iso6",
                b"cmfc",
                b"mp41",
            ]
            .concat(),
        ));
        out.extend(synthetic_init_moov());
        out.extend(synthetic_fragment());
        out
    }

    #[cfg(feature = "mp4")]
    fn synthetic_regular_mp4_with_alias_handler() -> Vec<u8> {
        let raw_a = vec![0x11, 0x22, 0x33];
        let raw_b = vec![0x44, 0x55];
        let ftyp = mp4_box(
            b"ftyp",
            &[b"isom".as_slice(), &0u32.to_be_bytes(), b"isom", b"mp42"].concat(),
        );
        let moov = synthetic_regular_moov(0);
        let chunk_offset = ftyp.len() + moov.len() + 8;
        let moov = synthetic_regular_moov(chunk_offset as u32);
        let mdat = mp4_box(b"mdat", &[raw_a, raw_b].concat());
        [ftyp, moov, mdat].concat()
    }

    #[cfg(feature = "mp4")]
    fn synthetic_regular_moov(chunk_offset: u32) -> Vec<u8> {
        let mvhd = full_box(b"mvhd", 0, 0, &[0u8; 20]);
        let tkhd_payload = [
            &0u32.to_be_bytes()[..],
            &0u32.to_be_bytes(),
            &1u32.to_be_bytes(),
            &0u32.to_be_bytes(),
            &0u32.to_be_bytes(),
            &[0u8; 60],
        ]
        .concat();
        let tkhd = full_box(b"tkhd", 0, 0x000007, &tkhd_payload);
        let mdhd_payload = [
            &0u32.to_be_bytes()[..],
            &0u32.to_be_bytes(),
            &44_100u32.to_be_bytes(),
            &0u32.to_be_bytes(),
            &0u16.to_be_bytes(),
            &0u16.to_be_bytes(),
        ]
        .concat();
        let mdhd = full_box(b"mdhd", 0, 0, &mdhd_payload);
        let hdlr = full_box(
            b"hdlr",
            0,
            0,
            &[&0u32.to_be_bytes()[..], b"soun", &[0u8; 12].as_slice()].concat(),
        );
        let smhd = full_box(b"smhd", 0, 0, &[0u8; 4]);
        let alias_hdlr = full_box(
            b"hdlr",
            0,
            0,
            &[&0u32.to_be_bytes()[..], b"alis", &[0u8; 12].as_slice()].concat(),
        );
        let dinf = mp4_box(b"dinf", &[]);
        let stsd = full_box(
            b"stsd",
            0,
            0,
            &[&1u32.to_be_bytes()[..], &synthetic_mp4a_entry()].concat(),
        );
        let stts = full_box(
            b"stts",
            0,
            0,
            &[
                &1u32.to_be_bytes()[..],
                &2u32.to_be_bytes(),
                &1024u32.to_be_bytes(),
            ]
            .concat(),
        );
        let stsc = full_box(
            b"stsc",
            0,
            0,
            &[
                &1u32.to_be_bytes()[..],
                &1u32.to_be_bytes(),
                &2u32.to_be_bytes(),
                &1u32.to_be_bytes(),
            ]
            .concat(),
        );
        let stsz = full_box(
            b"stsz",
            0,
            0,
            &[
                &0u32.to_be_bytes()[..],
                &2u32.to_be_bytes(),
                &3u32.to_be_bytes(),
                &2u32.to_be_bytes(),
            ]
            .concat(),
        );
        let stco = full_box(
            b"stco",
            0,
            0,
            &[&1u32.to_be_bytes()[..], &chunk_offset.to_be_bytes()].concat(),
        );
        let stbl = mp4_box(b"stbl", &[stsd, stts, stsc, stsz, stco].concat());
        let minf = mp4_box(b"minf", &[smhd, alias_hdlr, dinf, stbl].concat());
        let mdia = mp4_box(b"mdia", &[mdhd, hdlr, minf].concat());
        let trak = mp4_box(b"trak", &[tkhd, mdia].concat());
        mp4_box(b"moov", &[mvhd, trak].concat())
    }

    #[cfg(feature = "mp4")]
    fn synthetic_init_moov() -> Vec<u8> {
        let mvhd = full_box(b"mvhd", 0, 0, &[0u8; 20]);
        let tkhd_payload = [
            &0u32.to_be_bytes()[..],
            &0u32.to_be_bytes(),
            &1u32.to_be_bytes(),
            &0u32.to_be_bytes(),
            &0u32.to_be_bytes(),
            &[0u8; 60],
        ]
        .concat();
        let tkhd = full_box(b"tkhd", 0, 0x000007, &tkhd_payload);
        let mdhd_payload = [
            &0u32.to_be_bytes()[..],
            &0u32.to_be_bytes(),
            &44_100u32.to_be_bytes(),
            &0u32.to_be_bytes(),
            &0u16.to_be_bytes(),
            &0u16.to_be_bytes(),
        ]
        .concat();
        let mdhd = full_box(b"mdhd", 0, 0, &mdhd_payload);
        let hdlr = full_box(
            b"hdlr",
            0,
            0,
            &[&0u32.to_be_bytes()[..], b"soun", &[0u8; 12].as_slice()].concat(),
        );
        let smhd = full_box(b"smhd", 0, 0, &[0u8; 4]);
        let alias_hdlr = full_box(
            b"hdlr",
            0,
            0,
            &[&0u32.to_be_bytes()[..], b"alis", &[0u8; 12].as_slice()].concat(),
        );
        let dinf = mp4_box(b"dinf", &[]);
        let stsd = full_box(
            b"stsd",
            0,
            0,
            &[&1u32.to_be_bytes()[..], &synthetic_mp4a_entry()].concat(),
        );
        let stbl = mp4_box(b"stbl", &stsd);
        let minf = mp4_box(b"minf", &[smhd, alias_hdlr, dinf, stbl].concat());
        let mdia = mp4_box(b"mdia", &[mdhd, hdlr, minf].concat());
        let trak = mp4_box(b"trak", &[tkhd, mdia].concat());
        let trex_payload = [
            &1u32.to_be_bytes()[..],
            &1u32.to_be_bytes(),
            &1024u32.to_be_bytes(),
            &0u32.to_be_bytes(),
            &0u32.to_be_bytes(),
        ]
        .concat();
        let trex = full_box(b"trex", 0, 0, &trex_payload);
        let mvex = mp4_box(b"mvex", &trex);
        mp4_box(b"moov", &[mvhd, trak, mvex].concat())
    }

    #[cfg(feature = "mp4")]
    fn synthetic_mp4a_entry() -> Vec<u8> {
        let asc = [0x12, 0x10];
        let dec_specific = descriptor(0x05, &asc);
        let dec_config = descriptor(
            0x04,
            &[
                &[0x40, 0x15, 0x00, 0x00, 0x00][..],
                &0u32.to_be_bytes(),
                &0u32.to_be_bytes(),
                &dec_specific,
            ]
            .concat(),
        );
        let es = descriptor(
            0x03,
            &[&0u16.to_be_bytes()[..], &[0x00], &dec_config].concat(),
        );
        let esds = full_box(b"esds", 0, 0, &es);
        let payload = [
            &[0u8; 6][..],
            &1u16.to_be_bytes(),
            &[0u8; 8],
            &2u16.to_be_bytes(),
            &16u16.to_be_bytes(),
            &0u16.to_be_bytes(),
            &0u16.to_be_bytes(),
            &(44_100u32 << 16).to_be_bytes(),
            &esds,
        ]
        .concat();
        mp4_box(b"mp4a", &payload)
    }

    #[cfg(feature = "mp4")]
    fn synthetic_fragment() -> Vec<u8> {
        let raw_a = vec![0x11, 0x22, 0x33];
        let raw_b = vec![0x44, 0x55];
        let mfhd = full_box(b"mfhd", 0, 0, &1u32.to_be_bytes());
        let tfhd = full_box(b"tfhd", 0, 0x020000, &1u32.to_be_bytes());
        let tfdt = full_box(b"tfdt", 0, 0, &0u32.to_be_bytes());
        let trun_size = 8 + 4 + 4 + 4 + (2 * 8);
        let traf_size = 8 + tfhd.len() + tfdt.len() + trun_size;
        let moof_size = 8 + mfhd.len() + traf_size;
        let data_offset = (moof_size + 8) as i32;
        let trun_payload = [
            &2u32.to_be_bytes()[..],
            &data_offset.to_be_bytes(),
            &1024u32.to_be_bytes(),
            &(raw_a.len() as u32).to_be_bytes(),
            &1024u32.to_be_bytes(),
            &(raw_b.len() as u32).to_be_bytes(),
        ]
        .concat();
        let trun = full_box(b"trun", 0, 0x000301, &trun_payload);
        let traf = mp4_box(b"traf", &[tfhd, tfdt, trun].concat());
        let moof = mp4_box(b"moof", &[mfhd, traf].concat());
        assert_eq!(moof.len(), moof_size);
        let mdat = mp4_box(b"mdat", &[raw_a, raw_b].concat());
        [moof, mdat].concat()
    }

    #[cfg(feature = "mp4")]
    fn descriptor(tag: u8, payload: &[u8]) -> Vec<u8> {
        assert!(payload.len() < 128);
        let mut out = vec![tag, payload.len() as u8];
        out.extend_from_slice(payload);
        out
    }

    #[cfg(feature = "mp4")]
    fn full_box(name: &[u8; 4], version: u8, flags: u32, payload: &[u8]) -> Vec<u8> {
        let mut full_payload = vec![
            version,
            ((flags >> 16) & 0xff) as u8,
            ((flags >> 8) & 0xff) as u8,
            (flags & 0xff) as u8,
        ];
        full_payload.extend_from_slice(payload);
        mp4_box(name, &full_payload)
    }

    #[cfg(feature = "mp4")]
    fn mp4_box(name: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let size = 8 + payload.len();
        let mut out = Vec::with_capacity(size);
        out.extend_from_slice(&(size as u32).to_be_bytes());
        out.extend_from_slice(name);
        out.extend_from_slice(payload);
        out
    }

    #[cfg(feature = "mp4")]
    fn stsd_payload(entry: Vec<u8>) -> Vec<u8> {
        [&[0u8; 4][..], &1u32.to_be_bytes(), &entry].concat()
    }

    #[cfg(feature = "mp4")]
    fn quicktime_pcm_entry(
        codec: [u8; 4],
        version: u16,
        bits: u8,
        enda: Option<u16>,
        lpcm_flags: Option<u32>,
    ) -> Vec<u8> {
        let payload_size = match version {
            0 => 28,
            1 => 44,
            2 => 64,
            _ => unreachable!(),
        };
        let mut payload = vec![0u8; payload_size];
        payload[6..8].copy_from_slice(&1u16.to_be_bytes());
        payload[8..10].copy_from_slice(&version.to_be_bytes());
        payload[16..18].copy_from_slice(&2u16.to_be_bytes());
        payload[18..20].copy_from_slice(&u16::from(bits).to_be_bytes());
        payload[24..28].copy_from_slice(&(48_000u32 << 16).to_be_bytes());
        match version {
            1 => {
                payload[28..32].copy_from_slice(&1u32.to_be_bytes());
                payload[32..36].copy_from_slice(&u32::from(bits).div_ceil(8).to_be_bytes());
                payload[36..40].copy_from_slice(&(u32::from(bits).div_ceil(8) * 2).to_be_bytes());
                payload[40..44].copy_from_slice(&u32::from(bits).div_ceil(8).to_be_bytes());
            }
            2 => {
                payload[28..32].copy_from_slice(&72u32.to_be_bytes());
                payload[32..40].copy_from_slice(&48_000f64.to_bits().to_be_bytes());
                payload[40..44].copy_from_slice(&2u32.to_be_bytes());
                payload[44..48].copy_from_slice(&0x7f00_0000u32.to_be_bytes());
                payload[48..52].copy_from_slice(&u32::from(bits).to_be_bytes());
                payload[52..56].copy_from_slice(&lpcm_flags.unwrap_or_default().to_be_bytes());
                payload[56..60].copy_from_slice(&(u32::from(bits).div_ceil(8) * 2).to_be_bytes());
                payload[60..64].copy_from_slice(&1u32.to_be_bytes());
            }
            _ => {}
        }
        if let Some(flag) = enda {
            payload.extend(mp4_box(b"enda", &flag.to_be_bytes()));
        }
        mp4_box(&codec, &payload)
    }

    #[cfg(feature = "mp4")]
    fn synthetic_traf(
        track_id: u32,
        base_data_offset: Option<u64>,
        default_base_is_moof: bool,
        data_offset: Option<i32>,
        duration: Option<u32>,
        size: u32,
    ) -> Vec<u8> {
        let mut tfhd_flags = 0u32;
        let mut tfhd_payload = track_id.to_be_bytes().to_vec();
        if let Some(offset) = base_data_offset {
            tfhd_flags |= 0x000001;
            tfhd_payload.extend_from_slice(&offset.to_be_bytes());
        }
        if default_base_is_moof {
            tfhd_flags |= 0x020000;
        }
        let tfhd = full_box(b"tfhd", 0, tfhd_flags, &tfhd_payload);

        let mut trun_flags = 0x000200u32;
        let mut trun_payload = 1u32.to_be_bytes().to_vec();
        if let Some(offset) = data_offset {
            trun_flags |= 0x000001;
            trun_payload.extend_from_slice(&offset.to_be_bytes());
        }
        if let Some(duration) = duration {
            trun_flags |= 0x000100;
            trun_payload.extend_from_slice(&duration.to_be_bytes());
        }
        trun_payload.extend_from_slice(&size.to_be_bytes());
        let trun = full_box(b"trun", 0, trun_flags, &trun_payload);
        mp4_box(b"traf", &[tfhd, trun].concat())
    }

    #[cfg(feature = "mp4")]
    fn dummy_fragment_track(track_id: u64) -> MediaTrackConfig {
        MediaTrackConfig {
            container: AudioContainer::Mp4,
            kind: MediaTrackKind::Audio,
            track_id,
            codec: "aac".to_string(),
            codec_id: "mp4a".to_string(),
            timescale: 48_000,
            timeline: None,
            edit_timeline: Vec::new(),
            sample_count: 0,
            width: None,
            height: None,
            sample_rate: Some(48_000),
            channels: Some(2),
            bits_per_sample: None,
            pcm_endianness: None,
            pcm_float: None,
            pcm_signed: None,
            pcm_packed: None,
            pcm_aligned_high: None,
            pcm_interleaved: None,
            pcm_bytes_per_frame: None,
            pcm_frames_per_packet: None,
            codec_private: Vec::new(),
            decoder_configuration: Vec::new(),
            nal_length_size: None,
        }
    }

    #[cfg(feature = "mpeg-ts")]
    fn synthetic_ts_segment() -> Vec<u8> {
        let mut out = Vec::new();
        out.extend(ts_packet(0x0000, true, &pat_section(0x1000)));
        out.extend(ts_packet(0x1000, true, &pmt_section(0x0101, 0x0f)));
        out.extend(ts_packet(0x0101, true, &pes_packet(&adts_frame())));
        out.extend(ts_packet(0x1fff, false, &[]));
        out.extend(ts_packet(0x1fff, false, &[]));
        out
    }

    #[cfg(feature = "mpeg-ts")]
    fn ts_packet(pid: u16, payload_unit_start: bool, payload: &[u8]) -> Vec<u8> {
        let mut packet = vec![0xff; 188];
        packet[0] = 0x47;
        packet[1] = ((pid >> 8) as u8 & 0x1f) | if payload_unit_start { 0x40 } else { 0 };
        packet[2] = pid as u8;
        packet[3] = 0x10;
        let copy_len = payload.len().min(184);
        packet[4..4 + copy_len].copy_from_slice(&payload[..copy_len]);
        packet
    }

    #[cfg(feature = "mpeg-ts")]
    fn pat_section(pmt_pid: u16) -> Vec<u8> {
        let mut section = vec![
            0x00,
            0x00,
            0xb0,
            0x0d,
            0x00,
            0x01,
            0xc1,
            0x00,
            0x00,
            0x00,
            0x01,
            0xe0 | ((pmt_pid >> 8) as u8 & 0x1f),
            pmt_pid as u8,
            0,
            0,
            0,
            0,
        ];
        set_test_psi_crc(&mut section);
        section
    }

    #[cfg(feature = "mpeg-ts")]
    fn pmt_section(audio_pid: u16, stream_type: u8) -> Vec<u8> {
        let mut section = vec![
            0x00,
            0x02,
            0xb0,
            0x12,
            0x00,
            0x01,
            0xc1,
            0x00,
            0x00,
            0xe1,
            0x00,
            0xf0,
            0x00,
            stream_type,
            0xe0 | ((audio_pid >> 8) as u8 & 0x1f),
            audio_pid as u8,
            0xf0,
            0x00,
            0,
            0,
            0,
            0,
        ];
        set_test_psi_crc(&mut section);
        section
    }

    #[cfg(feature = "mpeg-ts")]
    fn set_test_psi_crc(section_with_pointer: &mut [u8]) {
        let end = section_with_pointer.len() - 4;
        let mut crc = 0xffff_ffff_u32;
        for byte in &section_with_pointer[1..end] {
            crc ^= u32::from(*byte) << 24;
            for _ in 0..8 {
                crc = if crc & 0x8000_0000 != 0 {
                    (crc << 1) ^ 0x04C1_1DB7
                } else {
                    crc << 1
                };
            }
        }
        section_with_pointer[end..].copy_from_slice(&crc.to_be_bytes());
    }

    #[cfg(feature = "mpeg-ts")]
    fn pes_packet(payload: &[u8]) -> Vec<u8> {
        let pes_len = payload.len() + 8;
        let pts = 90_000_u64;
        let mut pes = vec![
            0x00,
            0x00,
            0x01,
            0xc0,
            (pes_len >> 8) as u8,
            pes_len as u8,
            0x80,
            0x80,
            0x05,
            0x21 | (((pts >> 29) as u8) & 0x0e),
            (pts >> 22) as u8,
            (((pts >> 14) as u8) & 0xfe) | 1,
            (pts >> 7) as u8,
            ((pts << 1) as u8) | 1,
        ];
        pes.extend_from_slice(payload);
        pes
    }

    #[cfg(feature = "mpeg-ts")]
    fn adts_frame() -> Vec<u8> {
        let payload_len = 8usize;
        let frame_len = 7 + payload_len;
        let mut frame = vec![0u8; frame_len];
        frame[0] = 0xff;
        frame[1] = 0xf1;
        frame[2] = (1 << 6) | (4 << 2);
        frame[3] = (2 << 6) | (((frame_len >> 11) & 0x03) as u8);
        frame[4] = ((frame_len >> 3) & 0xff) as u8;
        frame[5] = (((frame_len & 0x07) << 5) as u8) | 0x1f;
        frame[6] = 0xfc;
        for (idx, byte) in frame[7..].iter_mut().enumerate() {
            *byte = idx as u8;
        }
        frame
    }
}
