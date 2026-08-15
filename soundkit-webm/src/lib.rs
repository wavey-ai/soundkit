use frame_header::{EncodingFlag, Endianness};
#[cfg(feature = "opus")]
use soundkit::audio_packet::Decoder;
use soundkit::audio_types::AudioData;
#[cfg(feature = "opus")]
use soundkit_opus::OpusDecoder;
#[cfg(feature = "vorbis")]
use soundkit_vorbis::VorbisPacketDecoder;
use std::collections::VecDeque;
use tracing::{debug, trace};

#[cfg(feature = "opus")]
const MAX_OPUS_FRAME_SAMPLES: usize = 5760; // 120 ms @ 48 kHz
const MAX_WEBM_INPUT_CHUNK_BYTES: usize = 4 * 1024 * 1024;
const MAX_WEBM_METADATA_ELEMENT_BYTES: u64 = 16 * 1024 * 1024;
const MAX_WEBM_PACKET_ELEMENT_BYTES: u64 = 64 * 1024 * 1024;
const MAX_WEBM_BUFFER_BYTES: usize = MAX_WEBM_PACKET_ELEMENT_BYTES as usize + 64 * 1024;

// EBML Element IDs (variable length encoded in files)
const EBML_ID: u32 = 0x1A45DFA3;
const SEGMENT_ID: u32 = 0x18538067;
const TRACKS_ID: u32 = 0x1654AE6B;
const TRACK_ENTRY_ID: u32 = 0xAE;
const TRACK_NUMBER_ID: u32 = 0xD7;
const TRACK_TYPE_ID: u32 = 0x83;
const CODEC_ID_ID: u32 = 0x86;
const CODEC_PRIVATE_ID: u32 = 0x63A2;
const AUDIO_ID: u32 = 0xE1;
const VIDEO_ID: u32 = 0xE0;
const SAMPLING_FREQUENCY_ID: u32 = 0xB5;
const CHANNELS_ID: u32 = 0x9F;
const PIXEL_WIDTH_ID: u32 = 0xB0;
const PIXEL_HEIGHT_ID: u32 = 0xBA;
const CLUSTER_ID: u32 = 0x1F43B675;
const CLUSTER_TIMECODE_ID: u32 = 0xE7;
const INFO_ID: u32 = 0x1549A966;
const TIMECODE_SCALE_ID: u32 = 0x2AD7B1;
const SIMPLE_BLOCK_ID: u32 = 0xA3;
const BLOCK_ID: u32 = 0xA1;
const BLOCK_GROUP_ID: u32 = 0xA0;
const BLOCK_DURATION_ID: u32 = 0x9B;
const REFERENCE_BLOCK_ID: u32 = 0xFB;
const DEFAULT_DURATION_ID: u32 = 0x23E383;

// Track types
const TRACK_TYPE_AUDIO: u64 = 2;
const TRACK_TYPE_VIDEO: u64 = 1;

/// Read a variable-length integer (VINT) used in EBML
/// Returns (value, bytes_consumed) or None if not enough data
fn read_vint(data: &[u8]) -> Option<(u64, usize)> {
    if data.is_empty() {
        return None;
    }

    let first = data[0];
    if first == 0 {
        return None; // Invalid VINT
    }

    // Count leading zeros to determine length
    let len = first.leading_zeros() as usize + 1;
    if len > 8 || data.len() < len {
        return None;
    }

    // Read the value - mask off the length marker bits
    let mask = if len < 8 { 0xFFu8 >> len } else { 0 };
    let mut value = (first & mask) as u64;
    for byte in data.iter().take(len).skip(1) {
        value = (value << 8) | *byte as u64;
    }

    Some((value, len))
}

/// Read an EBML element ID
/// Returns (id, bytes_consumed) or None if not enough data
fn read_element_id(data: &[u8]) -> Option<(u32, usize)> {
    if data.is_empty() {
        return None;
    }

    let first = data[0];
    if first == 0 {
        return None;
    }

    let len = first.leading_zeros() as usize + 1;
    if len > 4 || data.len() < len {
        return None;
    }

    let mut id = 0u32;
    for byte in data.iter().take(len) {
        id = (id << 8) | *byte as u32;
    }

    Some((id, len))
}

/// Read an EBML unsigned integer value
fn read_uint(data: &[u8], size: usize) -> u64 {
    let mut value = 0u64;
    for byte in data.iter().take(size.min(8).min(data.len())) {
        value = (value << 8) | *byte as u64;
    }
    value
}

/// Read an EBML float value
fn read_float(data: &[u8], size: usize) -> f64 {
    match size {
        4 if data.len() >= 4 => {
            let bytes: [u8; 4] = [data[0], data[1], data[2], data[3]];
            f32::from_be_bytes(bytes) as f64
        }
        8 if data.len() >= 8 => {
            let bytes: [u8; 8] = [
                data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
            ];
            f64::from_be_bytes(bytes)
        }
        _ => 0.0,
    }
}

fn append_webm_stream_chunk(buffer: &mut Vec<u8>, data: &[u8]) -> Result<(), String> {
    if data.len() > MAX_WEBM_INPUT_CHUNK_BYTES {
        return Err(format!(
            "WebM input chunk exceeds the {MAX_WEBM_INPUT_CHUNK_BYTES} byte streaming budget"
        ));
    }
    let new_len = buffer
        .len()
        .checked_add(data.len())
        .ok_or_else(|| "WebM buffer size overflow".to_string())?;
    if new_len > MAX_WEBM_BUFFER_BYTES {
        return Err(format!(
            "WebM parser buffer exceeds the {MAX_WEBM_BUFFER_BYTES} byte budget"
        ));
    }
    buffer.extend_from_slice(data);
    Ok(())
}

fn bounded_webm_element_size(id: u32, size: u64) -> Result<usize, String> {
    let budget = if matches!(id, SIMPLE_BLOCK_ID | BLOCK_ID | BLOCK_GROUP_ID) {
        MAX_WEBM_PACKET_ELEMENT_BYTES
    } else {
        MAX_WEBM_METADATA_ELEMENT_BYTES
    };
    if size > budget {
        return Err(format!(
            "WebM element 0x{id:X} exceeds the {budget} byte budget"
        ));
    }
    usize::try_from(size).map_err(|_| "WebM element size exceeds this platform".to_string())
}

fn read_signed_vint(data: &[u8]) -> Option<(i64, usize)> {
    let (value, len) = read_vint(data)?;
    let bias = (1i64 << (7 * len - 1)) - 1;
    Some((value as i64 - bias, len))
}

fn read_xiph_lace_size(data: &[u8], pos: &mut usize) -> Result<usize, String> {
    let mut size = 0usize;
    loop {
        let byte = *data
            .get(*pos)
            .ok_or_else(|| "Truncated Xiph lacing size".to_string())?;
        *pos += 1;
        size = size
            .checked_add(byte as usize)
            .ok_or_else(|| "Xiph lacing size overflow".to_string())?;
        if byte != 255 {
            return Ok(size);
        }
    }
}

fn split_laced_frames(
    payload: &[u8],
    data_start: usize,
    sizes: &[usize],
) -> Result<Vec<Vec<u8>>, String> {
    let mut frames = Vec::with_capacity(sizes.len());
    let mut pos = data_start;
    for size in sizes {
        let end = pos
            .checked_add(*size)
            .ok_or_else(|| "Laced frame size overflow".to_string())?;
        if end > payload.len() {
            return Err("Laced frame extends past block payload".to_string());
        }
        frames.push(payload[pos..end].to_vec());
        pos = end;
    }
    if pos != payload.len() {
        return Err("Laced frame sizes do not consume block payload".to_string());
    }
    Ok(frames)
}

fn parse_block_frames(flags: u8, payload: &[u8]) -> Result<Vec<Vec<u8>>, String> {
    match flags & 0x06 {
        0x00 => Ok(vec![payload.to_vec()]),
        0x02 => parse_xiph_laced_frames(payload),
        0x04 => parse_fixed_laced_frames(payload),
        0x06 => parse_ebml_laced_frames(payload),
        _ => unreachable!(),
    }
}

fn parse_xiph_laced_frames(payload: &[u8]) -> Result<Vec<Vec<u8>>, String> {
    let frame_count = payload
        .first()
        .map(|count| *count as usize + 1)
        .ok_or_else(|| "Truncated Xiph-laced block".to_string())?;
    let mut pos = 1usize;
    let mut sizes = Vec::with_capacity(frame_count);
    let mut known_size = 0usize;

    for _ in 0..frame_count.saturating_sub(1) {
        let size = read_xiph_lace_size(payload, &mut pos)?;
        known_size = known_size
            .checked_add(size)
            .ok_or_else(|| "Xiph-laced block size overflow".to_string())?;
        sizes.push(size);
    }

    let remaining = payload
        .len()
        .checked_sub(pos)
        .ok_or_else(|| "Invalid Xiph-laced block".to_string())?;
    if known_size > remaining {
        return Err("Xiph-laced frame sizes exceed block payload".to_string());
    }
    sizes.push(remaining - known_size);
    split_laced_frames(payload, pos, &sizes)
}

fn parse_fixed_laced_frames(payload: &[u8]) -> Result<Vec<Vec<u8>>, String> {
    let frame_count = payload
        .first()
        .map(|count| *count as usize + 1)
        .ok_or_else(|| "Truncated fixed-laced block".to_string())?;
    let data_start = 1usize;
    let remaining = payload.len() - data_start;
    if frame_count == 0 || remaining % frame_count != 0 {
        return Err("Invalid fixed-laced block size".to_string());
    }
    let frame_size = remaining / frame_count;
    let sizes = vec![frame_size; frame_count];
    split_laced_frames(payload, data_start, &sizes)
}

fn parse_ebml_laced_frames(payload: &[u8]) -> Result<Vec<Vec<u8>>, String> {
    let frame_count = payload
        .first()
        .map(|count| *count as usize + 1)
        .ok_or_else(|| "Truncated EBML-laced block".to_string())?;
    let mut pos = 1usize;
    let (first_size, first_len) = read_vint(&payload[pos..])
        .ok_or_else(|| "Truncated EBML-laced first frame size".to_string())?;
    pos += first_len;

    let mut sizes = Vec::with_capacity(frame_count);
    let mut previous = first_size as i64;
    sizes.push(first_size as usize);

    for _ in 1..frame_count.saturating_sub(1) {
        let (delta, delta_len) = read_signed_vint(&payload[pos..])
            .ok_or_else(|| "Truncated EBML-laced frame size delta".to_string())?;
        pos += delta_len;
        previous = previous
            .checked_add(delta)
            .ok_or_else(|| "EBML-laced frame size overflow".to_string())?;
        if previous < 0 {
            return Err("Negative EBML-laced frame size".to_string());
        }
        sizes.push(previous as usize);
    }

    let remaining = payload
        .len()
        .checked_sub(pos)
        .ok_or_else(|| "Invalid EBML-laced block".to_string())?;
    let known_size = sizes.iter().try_fold(0usize, |total, size| {
        total
            .checked_add(*size)
            .ok_or_else(|| "EBML-laced block size overflow".to_string())
    })?;
    if known_size > remaining {
        return Err("EBML-laced frame sizes exceed block payload".to_string());
    }
    sizes.push(remaining - known_size);
    split_laced_frames(payload, pos, &sizes)
}

#[cfg(feature = "vorbis")]
fn parse_vorbis_codec_private(codec_private: &[u8]) -> Result<Vec<Vec<u8>>, String> {
    let header_count = codec_private
        .first()
        .map(|count| *count as usize + 1)
        .ok_or_else(|| "Missing Vorbis CodecPrivate".to_string())?;
    if header_count != 3 {
        return Err(format!(
            "Expected 3 Vorbis headers in CodecPrivate, got {}",
            header_count
        ));
    }

    let mut pos = 1usize;
    let ident_size = read_xiph_lace_size(codec_private, &mut pos)?;
    let comment_size = read_xiph_lace_size(codec_private, &mut pos)?;
    let setup_size = codec_private
        .len()
        .checked_sub(pos + ident_size + comment_size)
        .ok_or_else(|| "Vorbis CodecPrivate header sizes exceed payload".to_string())?;

    split_laced_frames(codec_private, pos, &[ident_size, comment_size, setup_size])
}

fn parse_opus_head_metadata(codec_private: &[u8]) -> Option<(u16, i16, u8)> {
    if codec_private.len() < 19 || !codec_private.starts_with(b"OpusHead") {
        return None;
    }

    Some((
        u16::from_le_bytes([codec_private[10], codec_private[11]]),
        i16::from_le_bytes([codec_private[16], codec_private[17]]),
        codec_private[18],
    ))
}

/// Audio track information extracted from WebM
#[derive(Clone, Debug)]
struct AudioTrackInfo {
    track_number: u64,
    codec_id: String,
    sample_rate: u32,
    channels: u8,
    codec_private: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WebmOpusConfig {
    pub sample_rate: u32,
    pub channels: u8,
    pub pre_skip: u16,
    pub output_gain: i16,
    pub mapping_family: u8,
    pub codec_private: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WebmOpusDemuxEvent {
    Config(WebmOpusConfig),
    Packet { data: Vec<u8>, timecode: i16 },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WebmAudioConfig {
    pub track_number: u64,
    pub codec_id: String,
    pub sample_rate: u32,
    pub channels: u8,
    pub codec_private: Vec<u8>,
    pub pre_skip: Option<u16>,
    pub output_gain: Option<i16>,
    pub mapping_family: Option<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WebmAudioDemuxEvent {
    Config(WebmAudioConfig),
    Packet {
        track_number: u64,
        codec_id: String,
        data: Vec<u8>,
        timecode: i16,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WebmTrackKind {
    Audio,
    Video,
}

impl WebmTrackKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Audio => "audio",
            Self::Video => "video",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WebmMediaTrackConfig {
    pub track_number: u64,
    pub kind: WebmTrackKind,
    pub codec_id: String,
    pub codec_private: Vec<u8>,
    /// Decoder-ready codec initialization bytes. AVC and HEVC parameter sets
    /// are normalized to Annex B in Rust.
    pub decoder_configuration: Vec<u8>,
    /// Length-field width for AVC or HEVC blocks stored in Matroska.
    pub nal_length_size: Option<u8>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub sample_rate: Option<u32>,
    pub channels: Option<u8>,
    /// Matroska DefaultDuration, in nanoseconds per decoded frame.
    pub default_duration_ns: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WebmMediaDemuxEvent {
    Config {
        timecode_scale_ns: u64,
        track: WebmMediaTrackConfig,
    },
    Packet {
        track_number: u64,
        kind: WebmTrackKind,
        codec_id: String,
        data: Vec<u8>,
        /// Signed time in nanoseconds from the Segment time origin.
        timestamp_ns: i64,
        /// Duration of this decoded frame in nanoseconds, when declared.
        duration_ns: Option<u64>,
        is_keyframe: bool,
    },
}

/// One streaming WebM/Matroska demuxer for both video and audio tracks.
pub struct WebmMediaDemuxer {
    buffer: Vec<u8>,
    state: ParserState,
    tracks: Vec<WebmMediaTrackConfig>,
    pending_events: VecDeque<WebmMediaDemuxEvent>,
    emitted_configs: bool,
    timecode_scale_ns: u64,
    cluster_timecode: i64,
}

enum WebmAudioDecoder {
    #[cfg(feature = "opus")]
    Opus(OpusDecoder),
    #[cfg(feature = "vorbis")]
    Vorbis(VorbisPacketDecoder),
}

/// Parser state
#[derive(Debug, Clone, Copy, PartialEq)]
enum ParserState {
    Header,
    Tracks,
    Clusters,
}

impl WebmMediaDemuxer {
    pub fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(65_536),
            state: ParserState::Header,
            tracks: Vec::new(),
            pending_events: VecDeque::new(),
            emitted_configs: false,
            timecode_scale_ns: 1_000_000,
            cluster_timecode: 0,
        }
    }

    pub fn add(&mut self, data: &[u8]) -> Result<Vec<WebmMediaDemuxEvent>, String> {
        append_webm_stream_chunk(&mut self.buffer, data)?;
        loop {
            let previous_state = self.state;
            let consumed = match self.state {
                ParserState::Header => self.parse_media_header()?,
                ParserState::Tracks => self.parse_media_tracks()?,
                ParserState::Clusters => self.parse_media_clusters()?,
            };
            if consumed > 0 {
                self.buffer.drain(..consumed);
            } else if self.state == previous_state {
                break;
            }
        }
        Ok(self.pending_events.drain(..).collect())
    }

    pub fn finish(&mut self) -> Result<Vec<WebmMediaDemuxEvent>, String> {
        let events = self.add(&[])?;
        if !self.buffer.is_empty() {
            return Err(format!(
                "truncated WebM element: {} buffered bytes remain",
                self.buffer.len()
            ));
        }
        Ok(events)
    }

    fn parse_media_header(&mut self) -> Result<usize, String> {
        if self.buffer.len() < 4 {
            return Ok(0);
        }
        let (id, id_len) = read_element_id(&self.buffer)
            .ok_or_else(|| "invalid WebM EBML element ID".to_string())?;
        if id != EBML_ID {
            return Err(format!("invalid WebM: expected EBML header, got 0x{id:X}"));
        }
        let (size, size_len) = match read_vint(&self.buffer[id_len..]) {
            Some(value) => value,
            None => return Ok(0),
        };
        let size = bounded_webm_element_size(id, size)?;
        let total_len = id_len
            .checked_add(size_len)
            .and_then(|value| value.checked_add(size))
            .ok_or_else(|| "WebM EBML header size overflow".to_string())?;
        if self.buffer.len() < total_len {
            return Ok(0);
        }
        let after_ebml = &self.buffer[total_len..];
        let Some((segment_id, segment_id_len)) = read_element_id(after_ebml) else {
            return Ok(total_len);
        };
        if segment_id != SEGMENT_ID {
            return Err(format!(
                "invalid WebM: expected Segment, got 0x{segment_id:X}"
            ));
        }
        let Some((_, segment_size_len)) = read_vint(&after_ebml[segment_id_len..]) else {
            return Ok(total_len);
        };
        self.state = ParserState::Tracks;
        Ok(total_len + segment_id_len + segment_size_len)
    }

    fn parse_media_tracks(&mut self) -> Result<usize, String> {
        let mut pos = 0usize;
        while pos + 2 <= self.buffer.len() {
            let Some((id, id_len)) = read_element_id(&self.buffer[pos..]) else {
                break;
            };
            let Some((size, size_len)) = read_vint(&self.buffer[pos + id_len..]) else {
                break;
            };
            let header_len = id_len + size_len;
            if id == CLUSTER_ID {
                self.emit_media_configs()?;
                self.state = ParserState::Clusters;
                return Ok(pos);
            }
            if id == TRACKS_ID {
                pos += header_len;
                continue;
            }
            if is_unknown_ebml_size(size, size_len) && matches!(id, SEGMENT_ID) {
                pos += header_len;
                continue;
            }
            let size = bounded_webm_element_size(id, size)?;
            let data_start = pos
                .checked_add(header_len)
                .ok_or_else(|| "WebM element offset overflow".to_string())?;
            let data_end = data_start
                .checked_add(size)
                .ok_or_else(|| "WebM element size overflow".to_string())?;
            if data_end > self.buffer.len() {
                break;
            }
            match id {
                TRACK_ENTRY_ID => {
                    let entry = self.buffer[data_start..data_end].to_vec();
                    if let Some(track) = parse_media_track_entry(&entry)? {
                        if self
                            .tracks
                            .iter()
                            .all(|existing| existing.track_number != track.track_number)
                        {
                            self.tracks.push(track);
                        }
                    }
                    pos = data_end;
                }
                INFO_ID => {
                    self.timecode_scale_ns = parse_media_info(
                        &self.buffer[data_start..data_end],
                        self.timecode_scale_ns,
                    )?;
                    pos = data_end;
                }
                _ => pos = data_end,
            }
        }
        Ok(pos)
    }

    fn emit_media_configs(&mut self) -> Result<(), String> {
        if self.emitted_configs {
            return Ok(());
        }
        if !self
            .tracks
            .iter()
            .any(|track| track.kind == WebmTrackKind::Video)
        {
            return Err("WebM source has no supported video track".to_string());
        }
        if !self
            .tracks
            .iter()
            .any(|track| track.kind == WebmTrackKind::Audio)
        {
            return Err("WebM source has no supported audio track".to_string());
        }
        for track in &self.tracks {
            self.pending_events.push_back(WebmMediaDemuxEvent::Config {
                timecode_scale_ns: self.timecode_scale_ns,
                track: track.clone(),
            });
        }
        self.emitted_configs = true;
        Ok(())
    }

    fn parse_media_clusters(&mut self) -> Result<usize, String> {
        let mut pos = 0usize;
        while pos + 2 <= self.buffer.len() {
            let Some((id, id_len)) = read_element_id(&self.buffer[pos..]) else {
                break;
            };
            let Some((size, size_len)) = read_vint(&self.buffer[pos + id_len..]) else {
                break;
            };
            let header_len = id_len + size_len;
            if id == CLUSTER_ID {
                self.cluster_timecode = 0;
                pos += header_len;
                continue;
            }
            let size = bounded_webm_element_size(id, size)?;
            let data_start = pos
                .checked_add(header_len)
                .ok_or_else(|| "WebM cluster offset overflow".to_string())?;
            let data_end = data_start
                .checked_add(size)
                .ok_or_else(|| "WebM cluster element size overflow".to_string())?;
            if data_end > self.buffer.len() {
                break;
            }
            match id {
                CLUSTER_TIMECODE_ID => {
                    self.cluster_timecode =
                        i64::try_from(read_uint(&self.buffer[data_start..data_end], size))
                            .map_err(|_| "WebM cluster timecode exceeds i64".to_string())?;
                    pos = data_end;
                }
                SIMPLE_BLOCK_ID | BLOCK_ID => {
                    let block = self.buffer[data_start..data_end].to_vec();
                    self.emit_media_block(&block, id == SIMPLE_BLOCK_ID, None, None)?;
                    pos = data_end;
                }
                BLOCK_GROUP_ID => {
                    let group = self.buffer[data_start..data_end].to_vec();
                    self.parse_media_block_group(&group)?;
                    pos = data_end;
                }
                _ => pos = data_end,
            }
        }
        Ok(pos)
    }

    fn parse_media_block_group(&mut self, data: &[u8]) -> Result<(), String> {
        let mut pos = 0usize;
        let mut block = None;
        let mut has_reference = false;
        let mut duration_ticks = None;
        while pos + 2 <= data.len() {
            let Some((id, id_len)) = read_element_id(&data[pos..]) else {
                break;
            };
            let Some((size, size_len)) = read_vint(&data[pos + id_len..]) else {
                break;
            };
            let start = pos + id_len + size_len;
            let end = start
                .checked_add(size as usize)
                .ok_or_else(|| "WebM BlockGroup size overflow".to_string())?;
            if end > data.len() {
                return Err("truncated WebM BlockGroup".to_string());
            }
            match id {
                BLOCK_ID => block = Some(data[start..end].to_vec()),
                BLOCK_DURATION_ID => {
                    duration_ticks = Some(read_uint(&data[start..end], size as usize))
                }
                REFERENCE_BLOCK_ID => has_reference = true,
                _ => {}
            }
            pos = end;
        }
        if let Some(block) = block {
            self.emit_media_block(&block, false, Some(!has_reference), duration_ticks)?;
        }
        Ok(())
    }

    fn emit_media_block(
        &mut self,
        data: &[u8],
        simple_block: bool,
        keyframe_override: Option<bool>,
        block_duration_ticks: Option<u64>,
    ) -> Result<(), String> {
        let (track_number, track_len) =
            read_vint(data).ok_or_else(|| "invalid WebM block track number".to_string())?;
        let frame_start = track_len
            .checked_add(3)
            .ok_or_else(|| "WebM block header overflow".to_string())?;
        if data.len() < frame_start {
            return Err("truncated WebM block header".to_string());
        }
        let track = self
            .tracks
            .iter()
            .find(|track| track.track_number == track_number)
            .cloned();
        let Some(track) = track else {
            return Ok(());
        };
        let relative = i16::from_be_bytes([data[track_len], data[track_len + 1]]) as i64;
        let flags = data[track_len + 2];
        let ticks = self
            .cluster_timecode
            .checked_add(relative)
            .ok_or_else(|| "WebM block timestamp overflow".to_string())?;
        let timestamp_ns = i128::from(ticks)
            .checked_mul(i128::from(self.timecode_scale_ns))
            .and_then(|value| i64::try_from(value).ok())
            .ok_or_else(|| "WebM nanosecond timestamp overflow".to_string())?;
        let is_keyframe = keyframe_override.unwrap_or(simple_block && flags & 0x80 != 0);
        let frames = parse_block_frames(flags, &data[frame_start..])?;
        let frame_count = frames.len().max(1) as u64;
        let frame_duration_ns = block_duration_ticks
            .map(|ticks| {
                ticks
                    .checked_mul(self.timecode_scale_ns)
                    .ok_or_else(|| "WebM BlockDuration overflow".to_string())
            })
            .transpose()?
            .map(|duration| duration / frame_count)
            .or(track.default_duration_ns);
        for (frame_index, frame) in frames.into_iter().enumerate() {
            if !frame.is_empty() {
                let frame_timestamp_ns = match frame_duration_ns {
                    Some(duration) => {
                        let offset = duration
                            .checked_mul(frame_index as u64)
                            .and_then(|value| i64::try_from(value).ok())
                            .ok_or_else(|| "WebM laced frame timestamp overflow".to_string())?;
                        timestamp_ns
                            .checked_add(offset)
                            .ok_or_else(|| "WebM laced frame timestamp overflow".to_string())?
                    }
                    None => timestamp_ns,
                };
                let data = match track.nal_length_size {
                    Some(length_size) => {
                        soundkit_video::length_prefixed_nals_to_annex_b(&frame, length_size)?
                    }
                    None => frame,
                };
                self.pending_events.push_back(WebmMediaDemuxEvent::Packet {
                    track_number,
                    kind: track.kind,
                    codec_id: track.codec_id.clone(),
                    data,
                    timestamp_ns: frame_timestamp_ns,
                    duration_ns: frame_duration_ns,
                    is_keyframe,
                });
            }
        }
        Ok(())
    }
}

impl Default for WebmMediaDemuxer {
    fn default() -> Self {
        Self::new()
    }
}

fn is_unknown_ebml_size(size: u64, encoded_len: usize) -> bool {
    (1..=8).contains(&encoded_len) && size == (1u64 << (encoded_len * 7)) - 1
}

fn parse_media_info(data: &[u8], default_scale_ns: u64) -> Result<u64, String> {
    let mut scale_ns = default_scale_ns;
    parse_media_settings(data, |id, value| {
        if id == TIMECODE_SCALE_ID {
            scale_ns = read_uint(value, value.len());
        }
    })?;
    if scale_ns == 0 {
        return Err("WebM TimecodeScale must be positive".to_string());
    }
    Ok(scale_ns)
}

fn parse_media_track_entry(data: &[u8]) -> Result<Option<WebmMediaTrackConfig>, String> {
    let mut pos = 0usize;
    let mut track_number = None;
    let mut kind = None;
    let mut codec_id = String::new();
    let mut codec_private = Vec::new();
    let mut width = None;
    let mut height = None;
    let mut sample_rate = None;
    let mut channels = None;
    let mut default_duration_ns = None;
    while pos + 2 <= data.len() {
        let Some((id, id_len)) = read_element_id(&data[pos..]) else {
            break;
        };
        let Some((size, size_len)) = read_vint(&data[pos + id_len..]) else {
            break;
        };
        let start = pos + id_len + size_len;
        let end = start
            .checked_add(size as usize)
            .ok_or_else(|| "WebM TrackEntry size overflow".to_string())?;
        if end > data.len() {
            return Err("truncated WebM TrackEntry".to_string());
        }
        let value = &data[start..end];
        match id {
            TRACK_NUMBER_ID => track_number = Some(read_uint(value, size as usize)),
            TRACK_TYPE_ID => {
                kind = match read_uint(value, size as usize) {
                    TRACK_TYPE_AUDIO => Some(WebmTrackKind::Audio),
                    TRACK_TYPE_VIDEO => Some(WebmTrackKind::Video),
                    _ => None,
                }
            }
            CODEC_ID_ID => codec_id = String::from_utf8_lossy(value).into_owned(),
            CODEC_PRIVATE_ID => codec_private = value.to_vec(),
            DEFAULT_DURATION_ID => {
                default_duration_ns =
                    Some(read_uint(value, size as usize)).filter(|value| *value > 0)
            }
            AUDIO_ID => parse_media_audio_settings(value, &mut sample_rate, &mut channels)?,
            VIDEO_ID => parse_media_video_settings(value, &mut width, &mut height)?,
            _ => {}
        }
        pos = end;
    }
    let (Some(track_number), Some(kind)) = (track_number, kind) else {
        return Ok(None);
    };
    let supported = match kind {
        WebmTrackKind::Audio => matches!(
            codec_id.as_str(),
            "A_OPUS" | "A_VORBIS" | "A_AAC" | "A_FLAC" | "A_ALAC"
        ),
        WebmTrackKind::Video => matches!(
            codec_id.as_str(),
            "V_VP9" | "V_AV1" | "V_MPEG4/ISO/AVC" | "V_MPEGH/ISO/HEVC"
        ),
    };
    if !supported {
        return Ok(None);
    }
    let (decoder_configuration, nal_length_size) = match codec_id.as_str() {
        "V_MPEG4/ISO/AVC" => {
            let config = soundkit_video::parse_avc_decoder_configuration(&codec_private)?;
            (config.annex_b, Some(config.length_size))
        }
        "V_MPEGH/ISO/HEVC" => {
            let config = soundkit_video::parse_hevc_decoder_configuration(&codec_private)?;
            (config.annex_b, Some(config.length_size))
        }
        // Matroska carries the AV1CodecConfigurationRecord as CodecPrivate.
        "V_AV1" if !codec_private.is_empty() => (
            soundkit_video::parse_av1_decoder_configuration(&codec_private)?,
            None,
        ),
        _ => (codec_private.clone(), None),
    };
    Ok(Some(WebmMediaTrackConfig {
        track_number,
        kind,
        codec_id,
        codec_private,
        decoder_configuration,
        nal_length_size,
        width,
        height,
        sample_rate,
        channels,
        default_duration_ns,
    }))
}

fn parse_media_audio_settings(
    data: &[u8],
    sample_rate: &mut Option<u32>,
    channels: &mut Option<u8>,
) -> Result<(), String> {
    parse_media_settings(data, |id, value| match id {
        SAMPLING_FREQUENCY_ID => {
            let rate = read_float(value, value.len());
            if rate.is_finite() && rate > 0.0 && rate <= u32::MAX as f64 {
                *sample_rate = Some(rate.round() as u32);
            }
        }
        CHANNELS_ID => {
            let count = read_uint(value, value.len());
            if let Ok(count) = u8::try_from(count) {
                if count > 0 {
                    *channels = Some(count);
                }
            }
        }
        _ => {}
    })
}

fn parse_media_video_settings(
    data: &[u8],
    width: &mut Option<u32>,
    height: &mut Option<u32>,
) -> Result<(), String> {
    parse_media_settings(data, |id, value| {
        let parsed = u32::try_from(read_uint(value, value.len())).ok();
        match id {
            PIXEL_WIDTH_ID => *width = parsed.filter(|value| *value > 0),
            PIXEL_HEIGHT_ID => *height = parsed.filter(|value| *value > 0),
            _ => {}
        }
    })
}

fn parse_media_settings<F>(data: &[u8], mut visit: F) -> Result<(), String>
where
    F: FnMut(u32, &[u8]),
{
    let mut pos = 0usize;
    while pos + 2 <= data.len() {
        let Some((id, id_len)) = read_element_id(&data[pos..]) else {
            break;
        };
        let Some((size, size_len)) = read_vint(&data[pos + id_len..]) else {
            break;
        };
        let start = pos + id_len + size_len;
        let end = start
            .checked_add(size as usize)
            .ok_or_else(|| "WebM settings size overflow".to_string())?;
        if end > data.len() {
            return Err("truncated WebM track settings".to_string());
        }
        visit(id, &data[start..end]);
        pos = end;
    }
    Ok(())
}

pub struct WebmOpusDemuxer {
    buffer: Vec<u8>,
    state: ParserState,
    audio_track: Option<AudioTrackInfo>,
    pending_events: VecDeque<WebmOpusDemuxEvent>,
    emitted_config: bool,
}

impl WebmOpusDemuxer {
    pub fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(65536),
            state: ParserState::Header,
            audio_track: None,
            pending_events: VecDeque::new(),
            emitted_config: false,
        }
    }

    pub fn init(&mut self) -> Result<(), String> {
        Ok(())
    }

    pub fn add(&mut self, data: &[u8]) -> Result<Vec<WebmOpusDemuxEvent>, String> {
        append_webm_stream_chunk(&mut self.buffer, data)?;

        loop {
            let previous_state = self.state;
            let consumed = match self.state {
                ParserState::Header => self.parse_header()?,
                ParserState::Tracks => self.parse_tracks()?,
                ParserState::Clusters => self.parse_clusters()?,
            };

            if consumed > 0 {
                self.buffer.drain(..consumed);
            } else if self.state == previous_state {
                break;
            }
        }

        Ok(self.pending_events.drain(..).collect())
    }

    pub fn sample_rate(&self) -> Option<u32> {
        self.audio_track.as_ref().map(|track| track.sample_rate)
    }

    pub fn channels(&self) -> Option<u8> {
        self.audio_track.as_ref().map(|track| track.channels)
    }

    fn parse_header(&mut self) -> Result<usize, String> {
        if self.buffer.len() < 4 {
            return Ok(0);
        }

        let (id, id_len) = match read_element_id(&self.buffer) {
            Some(x) => x,
            None => return Ok(0),
        };

        if id != EBML_ID {
            return Err(format!(
                "Invalid WebM: expected EBML header, got 0x{:X}",
                id
            ));
        }

        let (size, size_len) = match read_vint(&self.buffer[id_len..]) {
            Some(x) => x,
            None => return Ok(0),
        };

        let header_len = id_len + size_len;
        let size = bounded_webm_element_size(id, size)?;
        let total_len = header_len
            .checked_add(size)
            .ok_or_else(|| "WebM EBML header size overflow".to_string())?;

        if self.buffer.len() < total_len {
            return Ok(0);
        }

        let after_ebml = &self.buffer[total_len..];
        if after_ebml.len() < 8 {
            return Ok(total_len);
        }

        let (seg_id, seg_id_len) = match read_element_id(after_ebml) {
            Some(x) => x,
            None => return Ok(total_len),
        };

        if seg_id == SEGMENT_ID {
            let (_, seg_size_len) = match read_vint(&after_ebml[seg_id_len..]) {
                Some(x) => x,
                None => return Ok(total_len),
            };

            self.state = ParserState::Tracks;
            return Ok(total_len + seg_id_len + seg_size_len);
        }

        Ok(total_len)
    }

    fn parse_tracks(&mut self) -> Result<usize, String> {
        let mut pos = 0;

        while pos + 4 <= self.buffer.len() {
            let (id, id_len) = match read_element_id(&self.buffer[pos..]) {
                Some(x) => x,
                None => break,
            };

            let (size, size_len) = match read_vint(&self.buffer[pos + id_len..]) {
                Some(x) => x,
                None => break,
            };

            let header_len = id_len + size_len;
            let is_master = matches!(
                id,
                TRACKS_ID | TRACK_ENTRY_ID | AUDIO_ID | SEGMENT_ID | CLUSTER_ID
            );
            if id == CLUSTER_ID {
                self.emit_config()?;
                self.state = ParserState::Clusters;
                return Ok(pos);
            }
            if id == TRACKS_ID {
                pos += header_len;
                continue;
            }
            if size == 0x00FFFFFFFFFFFFFF && is_master {
                pos += header_len;
                continue;
            }
            let size = bounded_webm_element_size(id, size)?;
            let data_start = pos
                .checked_add(header_len)
                .ok_or_else(|| "WebM element offset overflow".to_string())?;
            let data_end = data_start
                .checked_add(size)
                .ok_or_else(|| "WebM element size overflow".to_string())?;

            if data_end > self.buffer.len() {
                break;
            }

            match id {
                TRACK_ENTRY_ID => {
                    let element_data = self.buffer[data_start..data_end].to_vec();
                    self.parse_track_entry(&element_data)?;
                    pos = data_end;
                    continue;
                }
                _ => {
                    pos = data_end;
                    continue;
                }
            }
        }

        Ok(pos)
    }

    fn parse_track_entry(&mut self, data: &[u8]) -> Result<(), String> {
        let mut pos = 0;
        let mut track_number = 0u64;
        let mut track_type = 0u64;
        let mut codec_id = String::new();
        let mut codec_private = Vec::new();
        let mut sample_rate = 48000u32;
        let mut channels = 2u8;

        while pos + 2 <= data.len() {
            let (id, id_len) = match read_element_id(&data[pos..]) {
                Some(x) => x,
                None => break,
            };

            let (size, size_len) = match read_vint(&data[pos + id_len..]) {
                Some(x) => x,
                None => break,
            };

            let header_len = id_len + size_len;
            let elem_start = pos + header_len;
            let size = bounded_webm_element_size(id, size)?;
            let elem_end = elem_start
                .checked_add(size)
                .ok_or_else(|| "WebM track element size overflow".to_string())?;

            if elem_end > data.len() {
                break;
            }

            let elem_data = &data[elem_start..elem_end];

            match id {
                TRACK_NUMBER_ID => track_number = read_uint(elem_data, size),
                TRACK_TYPE_ID => track_type = read_uint(elem_data, size),
                CODEC_ID_ID => codec_id = String::from_utf8_lossy(elem_data).to_string(),
                CODEC_PRIVATE_ID => codec_private = elem_data.to_vec(),
                AUDIO_ID => {
                    let mut audio_pos = 0;
                    while audio_pos + 2 <= elem_data.len() {
                        let (audio_id, audio_id_len) =
                            match read_element_id(&elem_data[audio_pos..]) {
                                Some(x) => x,
                                None => break,
                            };

                        let (audio_size, audio_size_len) =
                            match read_vint(&elem_data[audio_pos + audio_id_len..]) {
                                Some(x) => x,
                                None => break,
                            };

                        let audio_header_len = audio_id_len + audio_size_len;
                        let audio_elem_start = audio_pos + audio_header_len;
                        let audio_size = bounded_webm_element_size(audio_id, audio_size)?;
                        let audio_elem_end = audio_elem_start
                            .checked_add(audio_size)
                            .ok_or_else(|| "WebM audio element size overflow".to_string())?;

                        if audio_elem_end > elem_data.len() {
                            break;
                        }

                        let audio_elem_data = &elem_data[audio_elem_start..audio_elem_end];
                        match audio_id {
                            SAMPLING_FREQUENCY_ID => {
                                sample_rate = read_float(audio_elem_data, audio_size) as u32;
                                if sample_rate == 0 {
                                    sample_rate = 48000;
                                }
                            }
                            CHANNELS_ID => {
                                channels = read_uint(audio_elem_data, audio_size) as u8;
                                if channels == 0 {
                                    channels = 2;
                                }
                            }
                            _ => {}
                        }

                        audio_pos = audio_elem_end;
                    }
                }
                _ => {}
            }

            pos = elem_end;
        }

        if track_type == TRACK_TYPE_AUDIO && codec_id == "A_OPUS" {
            self.audio_track = Some(AudioTrackInfo {
                track_number,
                codec_id,
                sample_rate,
                channels,
                codec_private,
            });
        }

        Ok(())
    }

    fn emit_config(&mut self) -> Result<(), String> {
        if self.emitted_config {
            return Ok(());
        }

        let track = self
            .audio_track
            .as_ref()
            .ok_or_else(|| "No WebM Opus audio track found".to_string())?;
        if track.codec_id != "A_OPUS" {
            return Err(format!(
                "Unsupported WebM codec for Opus debox: {}",
                track.codec_id
            ));
        }

        let (pre_skip, output_gain, mapping_family) =
            parse_opus_head_metadata(&track.codec_private).unwrap_or((0, 0, 0));
        self.pending_events
            .push_back(WebmOpusDemuxEvent::Config(WebmOpusConfig {
                sample_rate: track.sample_rate,
                channels: track.channels,
                pre_skip,
                output_gain,
                mapping_family,
                codec_private: track.codec_private.clone(),
            }));
        self.emitted_config = true;
        Ok(())
    }

    fn parse_clusters(&mut self) -> Result<usize, String> {
        let mut pos = 0;

        while pos + 4 <= self.buffer.len() {
            let (id, id_len) = match read_element_id(&self.buffer[pos..]) {
                Some(x) => x,
                None => break,
            };

            let (size, size_len) = match read_vint(&self.buffer[pos + id_len..]) {
                Some(x) => x,
                None => break,
            };

            let header_len = id_len + size_len;
            if id == CLUSTER_ID {
                pos += header_len;
                continue;
            }
            if is_unknown_ebml_size(size, size_len) {
                return Err(format!("WebM element 0x{id:X} has an unknown payload size"));
            }
            let size = bounded_webm_element_size(id, size)?;
            let data_start = pos
                .checked_add(header_len)
                .ok_or_else(|| "WebM cluster offset overflow".to_string())?;
            let data_end = data_start
                .checked_add(size)
                .ok_or_else(|| "WebM cluster element size overflow".to_string())?;
            if data_end > self.buffer.len() {
                break;
            }

            match id {
                SIMPLE_BLOCK_ID | BLOCK_ID => {
                    let block_data = self.buffer[data_start..data_end].to_vec();
                    self.parse_block(&block_data)?;
                    pos = data_end;
                    continue;
                }
                _ => {
                    pos = data_end;
                    continue;
                }
            }
        }

        Ok(pos)
    }

    fn parse_block(&mut self, data: &[u8]) -> Result<(), String> {
        if data.len() < 4 {
            return Ok(());
        }

        let (track_num, track_len) = match read_vint(data) {
            Some(x) => x,
            None => return Ok(()),
        };

        let is_audio_track = self
            .audio_track
            .as_ref()
            .map(|track| track.track_number == track_num)
            .unwrap_or(false);
        if !is_audio_track {
            return Ok(());
        }

        let frame_start = track_len + 3;
        if data.len() <= frame_start {
            return Ok(());
        }

        let timecode = i16::from_be_bytes([data[track_len], data[track_len + 1]]);
        let flags = data[track_len + 2];
        let frame_data = &data[frame_start..];

        for frame in parse_block_frames(flags, frame_data)? {
            if !frame.is_empty() {
                self.pending_events.push_back(WebmOpusDemuxEvent::Packet {
                    data: frame,
                    timecode,
                });
            }
        }

        Ok(())
    }
}

impl Default for WebmOpusDemuxer {
    fn default() -> Self {
        Self::new()
    }
}

pub struct WebmAudioDemuxer {
    buffer: Vec<u8>,
    state: ParserState,
    audio_track: Option<AudioTrackInfo>,
    pending_events: VecDeque<WebmAudioDemuxEvent>,
    emitted_config: bool,
}

impl WebmAudioDemuxer {
    pub fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(65536),
            state: ParserState::Header,
            audio_track: None,
            pending_events: VecDeque::new(),
            emitted_config: false,
        }
    }

    pub fn init(&mut self) -> Result<(), String> {
        Ok(())
    }

    pub fn add(&mut self, data: &[u8]) -> Result<Vec<WebmAudioDemuxEvent>, String> {
        append_webm_stream_chunk(&mut self.buffer, data)?;

        loop {
            let previous_state = self.state;
            let consumed = match self.state {
                ParserState::Header => self.parse_header()?,
                ParserState::Tracks => self.parse_tracks()?,
                ParserState::Clusters => self.parse_clusters()?,
            };

            if consumed > 0 {
                self.buffer.drain(..consumed);
            } else if self.state == previous_state {
                break;
            }
        }

        Ok(self.pending_events.drain(..).collect())
    }

    pub fn sample_rate(&self) -> Option<u32> {
        self.audio_track.as_ref().map(|track| track.sample_rate)
    }

    pub fn channels(&self) -> Option<u8> {
        self.audio_track.as_ref().map(|track| track.channels)
    }

    fn parse_header(&mut self) -> Result<usize, String> {
        if self.buffer.len() < 4 {
            return Ok(0);
        }

        let (id, id_len) = match read_element_id(&self.buffer) {
            Some(x) => x,
            None => return Ok(0),
        };

        if id != EBML_ID {
            return Err(format!(
                "Invalid WebM/Matroska: expected EBML header, got 0x{:X}",
                id
            ));
        }

        let (size, size_len) = match read_vint(&self.buffer[id_len..]) {
            Some(x) => x,
            None => return Ok(0),
        };

        let header_len = id_len + size_len;
        let size = bounded_webm_element_size(id, size)?;
        let total_len = header_len
            .checked_add(size)
            .ok_or_else(|| "WebM EBML header size overflow".to_string())?;

        if self.buffer.len() < total_len {
            return Ok(0);
        }

        let after_ebml = &self.buffer[total_len..];
        if after_ebml.len() < 8 {
            return Ok(total_len);
        }

        let (seg_id, seg_id_len) = match read_element_id(after_ebml) {
            Some(x) => x,
            None => return Ok(total_len),
        };

        if seg_id == SEGMENT_ID {
            let (_, seg_size_len) = match read_vint(&after_ebml[seg_id_len..]) {
                Some(x) => x,
                None => return Ok(total_len),
            };

            self.state = ParserState::Tracks;
            return Ok(total_len + seg_id_len + seg_size_len);
        }

        Ok(total_len)
    }

    fn parse_tracks(&mut self) -> Result<usize, String> {
        let mut pos = 0;

        while pos + 4 <= self.buffer.len() {
            let (id, id_len) = match read_element_id(&self.buffer[pos..]) {
                Some(x) => x,
                None => break,
            };

            let (size, size_len) = match read_vint(&self.buffer[pos + id_len..]) {
                Some(x) => x,
                None => break,
            };

            let header_len = id_len + size_len;
            let is_master = matches!(
                id,
                TRACKS_ID | TRACK_ENTRY_ID | AUDIO_ID | SEGMENT_ID | CLUSTER_ID
            );
            if id == CLUSTER_ID {
                self.emit_config()?;
                self.state = ParserState::Clusters;
                return Ok(pos);
            }
            if id == TRACKS_ID {
                pos += header_len;
                continue;
            }
            if size == 0x00FFFFFFFFFFFFFF && is_master {
                pos += header_len;
                continue;
            }
            let size = bounded_webm_element_size(id, size)?;
            let data_start = pos
                .checked_add(header_len)
                .ok_or_else(|| "WebM element offset overflow".to_string())?;
            let data_end = data_start
                .checked_add(size)
                .ok_or_else(|| "WebM element size overflow".to_string())?;

            if data_end > self.buffer.len() {
                break;
            }

            match id {
                TRACK_ENTRY_ID => {
                    let element_data = self.buffer[data_start..data_end].to_vec();
                    self.parse_track_entry(&element_data)?;
                    pos = data_end;
                    continue;
                }
                _ => {
                    pos = data_end;
                    continue;
                }
            }
        }

        Ok(pos)
    }

    fn parse_track_entry(&mut self, data: &[u8]) -> Result<(), String> {
        let mut pos = 0;
        let mut track_number = 0u64;
        let mut track_type = 0u64;
        let mut codec_id = String::new();
        let mut codec_private = Vec::new();
        let mut sample_rate = 48000u32;
        let mut channels = 2u8;

        while pos + 2 <= data.len() {
            let (id, id_len) = match read_element_id(&data[pos..]) {
                Some(x) => x,
                None => break,
            };

            let (size, size_len) = match read_vint(&data[pos + id_len..]) {
                Some(x) => x,
                None => break,
            };

            let header_len = id_len + size_len;
            let elem_start = pos + header_len;
            let size = bounded_webm_element_size(id, size)?;
            let elem_end = elem_start
                .checked_add(size)
                .ok_or_else(|| "WebM track element size overflow".to_string())?;

            if elem_end > data.len() {
                break;
            }

            let elem_data = &data[elem_start..elem_end];

            match id {
                TRACK_NUMBER_ID => track_number = read_uint(elem_data, size),
                TRACK_TYPE_ID => track_type = read_uint(elem_data, size),
                CODEC_ID_ID => codec_id = String::from_utf8_lossy(elem_data).to_string(),
                CODEC_PRIVATE_ID => codec_private = elem_data.to_vec(),
                AUDIO_ID => {
                    let mut audio_pos = 0;
                    while audio_pos + 2 <= elem_data.len() {
                        let (audio_id, audio_id_len) =
                            match read_element_id(&elem_data[audio_pos..]) {
                                Some(x) => x,
                                None => break,
                            };

                        let (audio_size, audio_size_len) =
                            match read_vint(&elem_data[audio_pos + audio_id_len..]) {
                                Some(x) => x,
                                None => break,
                            };

                        let audio_header_len = audio_id_len + audio_size_len;
                        let audio_elem_start = audio_pos + audio_header_len;
                        let audio_size = bounded_webm_element_size(audio_id, audio_size)?;
                        let audio_elem_end = audio_elem_start
                            .checked_add(audio_size)
                            .ok_or_else(|| "WebM audio element size overflow".to_string())?;

                        if audio_elem_end > elem_data.len() {
                            break;
                        }

                        let audio_elem_data = &elem_data[audio_elem_start..audio_elem_end];
                        match audio_id {
                            SAMPLING_FREQUENCY_ID => {
                                sample_rate = read_float(audio_elem_data, audio_size) as u32;
                                if sample_rate == 0 {
                                    sample_rate = 48000;
                                }
                            }
                            CHANNELS_ID => {
                                channels = read_uint(audio_elem_data, audio_size) as u8;
                                if channels == 0 {
                                    channels = 2;
                                }
                            }
                            _ => {}
                        }

                        audio_pos = audio_elem_end;
                    }
                }
                _ => {}
            }

            pos = elem_end;
        }

        if track_type == TRACK_TYPE_AUDIO && !codec_id.is_empty() && self.audio_track.is_none() {
            self.audio_track = Some(AudioTrackInfo {
                track_number,
                codec_id,
                sample_rate,
                channels,
                codec_private,
            });
        }

        Ok(())
    }

    fn emit_config(&mut self) -> Result<(), String> {
        if self.emitted_config {
            return Ok(());
        }

        let track = self
            .audio_track
            .as_ref()
            .ok_or_else(|| "No WebM/Matroska audio track found".to_string())?;
        let (pre_skip, output_gain, mapping_family) =
            parse_opus_head_metadata(&track.codec_private)
                .map(|(skip, gain, family)| (Some(skip), Some(gain), Some(family)))
                .unwrap_or((None, None, None));

        self.pending_events
            .push_back(WebmAudioDemuxEvent::Config(WebmAudioConfig {
                track_number: track.track_number,
                codec_id: track.codec_id.clone(),
                sample_rate: track.sample_rate,
                channels: track.channels,
                codec_private: track.codec_private.clone(),
                pre_skip,
                output_gain,
                mapping_family,
            }));
        self.emitted_config = true;
        Ok(())
    }

    fn parse_clusters(&mut self) -> Result<usize, String> {
        let mut pos = 0;

        while pos + 4 <= self.buffer.len() {
            let (id, id_len) = match read_element_id(&self.buffer[pos..]) {
                Some(x) => x,
                None => break,
            };

            let (size, size_len) = match read_vint(&self.buffer[pos + id_len..]) {
                Some(x) => x,
                None => break,
            };

            let header_len = id_len + size_len;
            if id == CLUSTER_ID {
                pos += header_len;
                continue;
            }
            if is_unknown_ebml_size(size, size_len) {
                return Err(format!("WebM element 0x{id:X} has an unknown payload size"));
            }
            let size = bounded_webm_element_size(id, size)?;
            let data_start = pos
                .checked_add(header_len)
                .ok_or_else(|| "WebM cluster offset overflow".to_string())?;
            let data_end = data_start
                .checked_add(size)
                .ok_or_else(|| "WebM cluster element size overflow".to_string())?;
            if data_end > self.buffer.len() {
                break;
            }

            match id {
                SIMPLE_BLOCK_ID | BLOCK_ID => {
                    let block_data = self.buffer[data_start..data_end].to_vec();
                    self.parse_block(&block_data)?;
                    pos = data_end;
                    continue;
                }
                _ => {
                    pos = data_end;
                    continue;
                }
            }
        }

        Ok(pos)
    }

    fn parse_block(&mut self, data: &[u8]) -> Result<(), String> {
        if data.len() < 4 {
            return Ok(());
        }

        let (track_num, track_len) = match read_vint(data) {
            Some(x) => x,
            None => return Ok(()),
        };

        let audio_track = self
            .audio_track
            .as_ref()
            .filter(|track| track.track_number == track_num);
        let Some(audio_track) = audio_track else {
            return Ok(());
        };
        let codec_id = audio_track.codec_id.clone();

        let frame_start = track_len + 3;
        if data.len() <= frame_start {
            return Ok(());
        }

        let timecode = i16::from_be_bytes([data[track_len], data[track_len + 1]]);
        let flags = data[track_len + 2];
        let frame_data = &data[frame_start..];

        for frame in parse_block_frames(flags, frame_data)? {
            if !frame.is_empty() {
                self.pending_events.push_back(WebmAudioDemuxEvent::Packet {
                    track_number: track_num,
                    codec_id: codec_id.clone(),
                    data: frame,
                    timecode,
                });
            }
        }

        Ok(())
    }
}

impl Default for WebmAudioDemuxer {
    fn default() -> Self {
        Self::new()
    }
}

/// Streaming WebM audio decoder
pub struct WebmDecoder {
    buffer: Vec<u8>,
    state: ParserState,
    audio_track: Option<AudioTrackInfo>,
    audio_decoder: Option<WebmAudioDecoder>,
    pending_audio: VecDeque<Vec<u8>>,
    #[cfg(feature = "opus")]
    scratch_buffer: Vec<i16>,
    #[cfg(feature = "opus")]
    pre_skip_remaining: usize,
    logged_first_audio: bool,
    header_complete: bool,
}

impl WebmDecoder {
    pub fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(65536),
            state: ParserState::Header,
            audio_track: None,
            audio_decoder: None,
            pending_audio: VecDeque::new(),
            #[cfg(feature = "opus")]
            scratch_buffer: Vec::with_capacity(MAX_OPUS_FRAME_SAMPLES * 2),
            #[cfg(feature = "opus")]
            pre_skip_remaining: 0,
            logged_first_audio: false,
            header_complete: false,
        }
    }

    pub fn init(&mut self) -> Result<(), String> {
        Ok(())
    }

    /// Feed more bytes of a WebM stream. Returns decoded PCM when available.
    pub fn add(&mut self, data: &[u8]) -> Result<Option<AudioData>, String> {
        append_webm_stream_chunk(&mut self.buffer, data)?;

        // Try to parse elements
        loop {
            let consumed = match self.state {
                ParserState::Header => self.parse_header()?,
                ParserState::Tracks => self.parse_tracks()?,
                ParserState::Clusters => self.parse_clusters()?,
            };

            if consumed == 0 {
                break;
            }

            self.buffer.drain(..consumed);
        }

        // Try to decode pending audio packets
        self.decode_pending_audio()
    }

    fn parse_header(&mut self) -> Result<usize, String> {
        if self.buffer.len() < 4 {
            return Ok(0);
        }

        // Check for EBML header
        let (id, id_len) = match read_element_id(&self.buffer) {
            Some(x) => x,
            None => return Ok(0),
        };

        if id != EBML_ID {
            return Err(format!(
                "Invalid WebM: expected EBML header, got 0x{:X}",
                id
            ));
        }

        let (size, size_len) = match read_vint(&self.buffer[id_len..]) {
            Some(x) => x,
            None => return Ok(0),
        };

        let header_len = id_len + size_len;
        let size = bounded_webm_element_size(id, size)?;
        let total_len = header_len
            .checked_add(size)
            .ok_or_else(|| "WebM EBML header size overflow".to_string())?;

        if self.buffer.len() < total_len {
            return Ok(0);
        }

        debug!(size = size, "parsed EBML header");

        // After EBML header, look for Segment
        let after_ebml = &self.buffer[total_len..];
        if after_ebml.len() < 8 {
            return Ok(total_len);
        }

        let (seg_id, seg_id_len) = match read_element_id(after_ebml) {
            Some(x) => x,
            None => return Ok(total_len),
        };

        if seg_id == SEGMENT_ID {
            let (_, seg_size_len) = match read_vint(&after_ebml[seg_id_len..]) {
                Some(x) => x,
                None => return Ok(total_len),
            };

            debug!("found Segment element");
            self.state = ParserState::Tracks;
            return Ok(total_len + seg_id_len + seg_size_len);
        }

        Ok(total_len)
    }

    fn parse_tracks(&mut self) -> Result<usize, String> {
        let mut pos = 0;

        while pos + 4 <= self.buffer.len() {
            let (id, id_len) = match read_element_id(&self.buffer[pos..]) {
                Some(x) => x,
                None => break,
            };

            let (size, size_len) = match read_vint(&self.buffer[pos + id_len..]) {
                Some(x) => x,
                None => break,
            };

            let header_len = id_len + size_len;
            let is_master = matches!(
                id,
                TRACKS_ID | TRACK_ENTRY_ID | AUDIO_ID | SEGMENT_ID | CLUSTER_ID
            );
            if id == CLUSTER_ID {
                if self.audio_track.is_some() {
                    self.init_decoder()?;
                    self.header_complete = true;
                }
                self.state = ParserState::Clusters;
                return Ok(pos);
            }
            if id == TRACKS_ID {
                trace!(size = size, "found Tracks element");
                pos += header_len;
                continue;
            }
            if size == 0x00FFFFFFFFFFFFFF && is_master {
                pos += header_len;
                continue;
            }
            let size = bounded_webm_element_size(id, size)?;
            let data_start = pos
                .checked_add(header_len)
                .ok_or_else(|| "WebM element offset overflow".to_string())?;
            let data_end = data_start
                .checked_add(size)
                .ok_or_else(|| "WebM element size overflow".to_string())?;

            if data_end > self.buffer.len() {
                break;
            }

            match id {
                TRACK_ENTRY_ID => {
                    // Copy the data to avoid borrow checker issues
                    let element_data = self.buffer[data_start..data_end].to_vec();
                    self.parse_track_entry(&element_data)?;
                    pos = data_end;
                    continue;
                }
                _ => {
                    // Skip unknown elements at this level
                    pos = data_end;
                    continue;
                }
            }
        }

        Ok(pos)
    }

    fn parse_track_entry(&mut self, data: &[u8]) -> Result<(), String> {
        let mut pos = 0;
        let mut track_number = 0u64;
        let mut track_type = 0u64;
        let mut codec_id = String::new();
        let mut codec_private = Vec::new();
        let mut sample_rate = 48000u32;
        let mut channels = 2u8;

        while pos + 2 <= data.len() {
            let (id, id_len) = match read_element_id(&data[pos..]) {
                Some(x) => x,
                None => break,
            };

            let (size, size_len) = match read_vint(&data[pos + id_len..]) {
                Some(x) => x,
                None => break,
            };

            let header_len = id_len + size_len;
            let elem_start = pos + header_len;
            let size = bounded_webm_element_size(id, size)?;
            let elem_end = elem_start
                .checked_add(size)
                .ok_or_else(|| "WebM track element size overflow".to_string())?;

            if elem_end > data.len() {
                break;
            }

            let elem_data = &data[elem_start..elem_end];

            match id {
                TRACK_NUMBER_ID => {
                    track_number = read_uint(elem_data, size);
                }
                TRACK_TYPE_ID => {
                    track_type = read_uint(elem_data, size);
                }
                CODEC_ID_ID => {
                    codec_id = String::from_utf8_lossy(elem_data).to_string();
                }
                CODEC_PRIVATE_ID => {
                    codec_private = elem_data.to_vec();
                }
                AUDIO_ID => {
                    // Parse Audio element children
                    let mut audio_pos = 0;
                    while audio_pos + 2 <= elem_data.len() {
                        let (audio_id, audio_id_len) =
                            match read_element_id(&elem_data[audio_pos..]) {
                                Some(x) => x,
                                None => break,
                            };

                        let (audio_size, audio_size_len) =
                            match read_vint(&elem_data[audio_pos + audio_id_len..]) {
                                Some(x) => x,
                                None => break,
                            };

                        let audio_header_len = audio_id_len + audio_size_len;
                        let audio_elem_start = audio_pos + audio_header_len;
                        let audio_size = bounded_webm_element_size(audio_id, audio_size)?;
                        let audio_elem_end = audio_elem_start
                            .checked_add(audio_size)
                            .ok_or_else(|| "WebM audio element size overflow".to_string())?;

                        if audio_elem_end > elem_data.len() {
                            break;
                        }

                        let audio_elem_data = &elem_data[audio_elem_start..audio_elem_end];

                        match audio_id {
                            SAMPLING_FREQUENCY_ID => {
                                sample_rate = read_float(audio_elem_data, audio_size) as u32;
                                if sample_rate == 0 {
                                    sample_rate = 48000;
                                }
                            }
                            CHANNELS_ID => {
                                channels = read_uint(audio_elem_data, audio_size) as u8;
                                if channels == 0 {
                                    channels = 2;
                                }
                            }
                            _ => {}
                        }

                        audio_pos = audio_elem_end;
                    }
                }
                _ => {}
            }

            pos = elem_end;
        }

        // Only store audio tracks
        if track_type == TRACK_TYPE_AUDIO && !codec_id.is_empty() {
            debug!(
                track_number = track_number,
                codec_id = %codec_id,
                sample_rate = sample_rate,
                channels = channels,
                codec_private_len = codec_private.len(),
                "found audio track"
            );

            let supported_codec = (cfg!(feature = "opus") && codec_id == "A_OPUS")
                || (cfg!(feature = "vorbis") && codec_id == "A_VORBIS");

            if supported_codec {
                self.audio_track = Some(AudioTrackInfo {
                    track_number,
                    codec_id,
                    sample_rate,
                    channels,
                    codec_private,
                });
            }
        }

        Ok(())
    }

    fn init_decoder(&mut self) -> Result<(), String> {
        let (codec_id, sample_rate, channels, codec_private) = {
            let track = self.audio_track.as_ref().ok_or("No audio track found")?;
            (
                track.codec_id.clone(),
                track.sample_rate,
                track.channels,
                track.codec_private.clone(),
            )
        };

        match codec_id.as_str() {
            #[cfg(feature = "opus")]
            "A_OPUS" => {
                if codec_private.len() >= 12 && codec_private.starts_with(b"OpusHead") {
                    let pre_skip = u16::from_le_bytes([codec_private[10], codec_private[11]]);
                    self.pre_skip_remaining = pre_skip as usize;
                    debug!(pre_skip = pre_skip, "parsed OpusHead pre-skip");
                }

                let mut decoder = OpusDecoder::new(sample_rate as usize, channels as usize);
                decoder.init()?;

                debug!(
                    sample_rate = sample_rate,
                    channels = channels,
                    "initialized Opus decoder for WebM"
                );

                self.audio_decoder = Some(WebmAudioDecoder::Opus(decoder));
            }
            #[cfg(feature = "vorbis")]
            "A_VORBIS" => {
                let headers = parse_vorbis_codec_private(&codec_private)?;
                let mut decoder = VorbisPacketDecoder::new();
                decoder.init()?;
                for header in headers {
                    decoder.decode_packet(&header)?;
                }

                if let Some(track) = self.audio_track.as_mut() {
                    if let Some(sample_rate) = decoder.sample_rate() {
                        track.sample_rate = sample_rate;
                    }
                    if let Some(channels) = decoder.channels() {
                        track.channels = channels;
                    }
                }

                debug!(
                    sample_rate = decoder.sample_rate().unwrap_or(sample_rate),
                    channels = decoder.channels().unwrap_or(channels),
                    "initialized Vorbis decoder for WebM"
                );

                self.audio_decoder = Some(WebmAudioDecoder::Vorbis(decoder));
            }
            _ => return Err(format!("Unsupported or disabled WebM codec: {}", codec_id)),
        }
        Ok(())
    }

    fn parse_clusters(&mut self) -> Result<usize, String> {
        let mut pos = 0;

        while pos + 4 <= self.buffer.len() {
            let (id, id_len) = match read_element_id(&self.buffer[pos..]) {
                Some(x) => x,
                None => break,
            };

            let (size, size_len) = match read_vint(&self.buffer[pos + id_len..]) {
                Some(x) => x,
                None => break,
            };

            let header_len = id_len + size_len;
            if id == CLUSTER_ID {
                pos += header_len;
                continue;
            }
            if is_unknown_ebml_size(size, size_len) {
                return Err(format!("WebM element 0x{id:X} has an unknown payload size"));
            }
            let size = bounded_webm_element_size(id, size)?;
            let data_start = pos
                .checked_add(header_len)
                .ok_or_else(|| "WebM cluster offset overflow".to_string())?;
            let data_end = data_start
                .checked_add(size)
                .ok_or_else(|| "WebM cluster element size overflow".to_string())?;

            if data_end > self.buffer.len() {
                break;
            }

            match id {
                SIMPLE_BLOCK_ID | BLOCK_ID => {
                    // Copy the data to avoid borrow checker issues
                    let block_data = self.buffer[data_start..data_end].to_vec();
                    self.parse_block(&block_data)?;
                    pos = data_end;
                    continue;
                }
                _ => {
                    // Skip other elements
                    pos = data_end;
                    continue;
                }
            }
        }

        Ok(pos)
    }

    fn parse_block(&mut self, data: &[u8]) -> Result<(), String> {
        if data.len() < 4 {
            return Ok(());
        }

        // Read track number (variable length)
        let (track_num, track_len) = match read_vint(data) {
            Some(x) => x,
            None => return Ok(()),
        };

        // Check if this is our audio track
        let is_audio_track = self
            .audio_track
            .as_ref()
            .map(|t| t.track_number == track_num)
            .unwrap_or(false);

        if !is_audio_track {
            return Ok(());
        }

        // Skip track number + 2 bytes timecode + 1 byte flags
        let frame_start = track_len + 3;
        if data.len() <= frame_start {
            return Ok(());
        }

        let flags = data[track_len + 2];
        let frame_data = &data[frame_start..];

        for frame in parse_block_frames(flags, frame_data)? {
            if !frame.is_empty() {
                self.pending_audio.push_back(frame);
            }
        }

        Ok(())
    }

    fn decode_pending_audio(&mut self) -> Result<Option<AudioData>, String> {
        let mut decoder = match self.audio_decoder.take() {
            Some(decoder) => decoder,
            None => return Ok(None),
        };

        let result = match &mut decoder {
            #[cfg(feature = "opus")]
            WebmAudioDecoder::Opus(opus_decoder) => self.decode_pending_opus(opus_decoder),
            #[cfg(feature = "vorbis")]
            WebmAudioDecoder::Vorbis(vorbis_decoder) => self.decode_pending_vorbis(vorbis_decoder),
        };

        self.audio_decoder = Some(decoder);
        result
    }

    #[cfg(feature = "opus")]
    fn decode_pending_opus(
        &mut self,
        decoder: &mut OpusDecoder,
    ) -> Result<Option<AudioData>, String> {
        let (sample_rate, channels) = match self.audio_track.as_ref() {
            Some(track) => (track.sample_rate, track.channels),
            None => return Ok(None),
        };

        let mut pcm_bytes = Vec::new();

        while let Some(packet) = self.pending_audio.pop_front() {
            // Reuse scratch buffer
            let required_size = MAX_OPUS_FRAME_SAMPLES * channels as usize;
            if self.scratch_buffer.len() < required_size {
                self.scratch_buffer.resize(required_size, 0);
            }

            let samples = match decoder.decode_i16(&packet, &mut self.scratch_buffer, false) {
                Ok(s) => s,
                Err(e) => {
                    trace!(error = %e, "Opus decode error, skipping packet");
                    continue;
                }
            };

            if samples == 0 {
                continue;
            }

            if !self.logged_first_audio {
                debug!(
                    packet_len = packet.len(),
                    samples_per_channel = samples,
                    pre_skip_remaining = self.pre_skip_remaining,
                    "decoded WebM Opus packet"
                );
                self.logged_first_audio = true;
            } else {
                trace!(
                    packet_len = packet.len(),
                    samples_per_channel = samples,
                    "decoded WebM Opus packet"
                );
            }

            let mut start = 0;
            if self.pre_skip_remaining > 0 {
                let skip = self.pre_skip_remaining.min(samples);
                self.pre_skip_remaining -= skip;
                start = skip * channels as usize;
            }

            let end = samples * channels as usize;
            for sample in &self.scratch_buffer[start..end] {
                pcm_bytes.extend_from_slice(&sample.to_le_bytes());
            }
        }

        if pcm_bytes.is_empty() {
            return Ok(None);
        }

        let audio = AudioData::new(
            16,
            channels,
            sample_rate,
            pcm_bytes,
            EncodingFlag::PCMSigned,
            Endianness::LittleEndian,
        );

        Ok(Some(audio))
    }

    #[cfg(feature = "vorbis")]
    fn decode_pending_vorbis(
        &mut self,
        decoder: &mut VorbisPacketDecoder,
    ) -> Result<Option<AudioData>, String> {
        let (fallback_sample_rate, fallback_channels) = match self.audio_track.as_ref() {
            Some(track) => (track.sample_rate, track.channels),
            None => return Ok(None),
        };
        let sample_rate = decoder.sample_rate().unwrap_or(fallback_sample_rate);
        let channels = decoder.channels().unwrap_or(fallback_channels);
        let mut pcm_bytes = Vec::new();

        while let Some(packet) = self.pending_audio.pop_front() {
            let samples = match decoder.decode_packet(&packet) {
                Ok(Some(samples)) => samples,
                Ok(None) => continue,
                Err(error) => {
                    trace!(error = %error, "Vorbis decode error, skipping packet");
                    continue;
                }
            };

            if samples.is_empty() {
                continue;
            }

            if !self.logged_first_audio {
                debug!(
                    packet_len = packet.len(),
                    pcm_samples_written = samples.len(),
                    "decoded WebM Vorbis packet"
                );
                self.logged_first_audio = true;
            } else {
                trace!(
                    packet_len = packet.len(),
                    pcm_samples_written = samples.len(),
                    "decoded WebM Vorbis packet"
                );
            }

            for sample in samples {
                pcm_bytes.extend_from_slice(&sample.to_le_bytes());
            }
        }

        if pcm_bytes.is_empty() {
            return Ok(None);
        }

        let audio = AudioData::new(
            16,
            channels,
            sample_rate,
            pcm_bytes,
            EncodingFlag::PCMSigned,
            Endianness::LittleEndian,
        );

        Ok(Some(audio))
    }

    pub fn sample_rate(&self) -> Option<u32> {
        self.audio_track.as_ref().map(|t| t.sample_rate)
    }

    pub fn channels(&self) -> Option<u8> {
        self.audio_track.as_ref().map(|t| t.channels)
    }
}

impl Default for WebmDecoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "opus")]
    use soundkit::audio_bytes::s16le_to_i16;
    #[cfg(feature = "opus")]
    use soundkit::test_utils::{print_waveform_with_header, DecodeResult};
    use std::fs;
    use std::path::PathBuf;

    #[cfg(feature = "opus")]
    const TEST_FILE: &str = "A_Tusk_is_used_to_make_costly_gifts";

    #[test]
    fn recognizes_unknown_ebml_sizes_for_every_vint_width() {
        for encoded_len in 1..=8 {
            let unknown = (1u64 << (encoded_len * 7)) - 1;
            assert!(is_unknown_ebml_size(unknown, encoded_len));
            assert!(!is_unknown_ebml_size(unknown - 1, encoded_len));
        }
        assert!(!is_unknown_ebml_size(0x7f, 2));
        assert!(!is_unknown_ebml_size(0, 0));
    }

    #[test]
    fn reads_non_default_timecode_scale_from_info_children() {
        let info_payload = [0x2A, 0xD7, 0xB1, 0x83, 0x1E, 0x84, 0x80];
        assert_eq!(
            parse_media_info(&info_payload, 1_000_000).unwrap(),
            2_000_000
        );

        let invalid = [0x2A, 0xD7, 0xB1, 0x81, 0x00];
        assert!(parse_media_info(&invalid, 1_000_000)
            .unwrap_err()
            .contains("positive"));
    }

    #[test]
    fn reads_video_default_duration_in_rust() {
        let entry = vec![
            0xD7, 0x81, 0x01, // TrackNumber = 1
            0x83, 0x81, 0x01, // TrackType = video
            0x86, 0x85, b'V', b'_', b'V', b'P', b'9', 0x23, 0xE3, 0x83, 0x84, 0x01, 0xFC, 0xA0,
            0x55, // 33,333,333 ns
            0xE0, 0x88, 0xB0, 0x82, 0x02, 0x80, 0xBA, 0x82, 0x01, 0x68,
        ];
        let track = parse_media_track_entry(&entry).unwrap().unwrap();
        assert_eq!(track.kind, WebmTrackKind::Video);
        assert_eq!(track.codec_id, "V_VP9");
        assert_eq!(track.width, Some(640));
        assert_eq!(track.height, Some(360));
        assert_eq!(track.default_duration_ns, Some(33_333_333));
    }

    #[test]
    fn normalizes_matroska_avc_configuration_and_blocks_in_rust() {
        let avcc = [1, 100, 0, 31, 0xff, 0xe1, 0, 2, 0x67, 0x64, 1, 0, 1, 0x68];
        let mut entry = vec![
            0xD7, 0x81, 0x01, // TrackNumber = 1
            0x83, 0x81, 0x01, // TrackType = video
            0x86, 0x8F, b'V', b'_', b'M', b'P', b'E', b'G', b'4', b'/', b'I', b'S', b'O', b'/',
            b'A', b'V', b'C', 0x63, 0xA2, 0x8E,
        ];
        entry.extend_from_slice(&avcc);
        let track = parse_media_track_entry(&entry).unwrap().unwrap();
        assert_eq!(track.nal_length_size, Some(4));
        assert_eq!(
            track.decoder_configuration,
            [0, 0, 0, 1, 0x67, 0x64, 0, 0, 0, 1, 0x68]
        );

        let mut demuxer = WebmMediaDemuxer::new();
        demuxer.tracks.push(track);
        demuxer
            .emit_media_block(
                &[0x81, 0, 0, 0x80, 0, 0, 0, 2, 0x65, 0x88],
                true,
                None,
                None,
            )
            .unwrap();
        assert!(matches!(
            demuxer.pending_events.pop_front(),
            Some(WebmMediaDemuxEvent::Packet { data, .. })
                if data == [0, 0, 0, 1, 0x65, 0x88]
        ));
    }

    #[test]
    fn known_size_cluster_emits_packets_before_cluster_end() {
        let data = fs::read(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("testdata/video-compat/never-final/vp9-profile0-opus.webm"),
        )
        .unwrap();
        let mut demuxer = WebmMediaDemuxer::new();
        let mut first_packet_at = None;
        let mut maximum_buffer = 0usize;
        for (index, chunk) in data.chunks(997).enumerate() {
            let events = demuxer.add(chunk).unwrap();
            maximum_buffer = maximum_buffer.max(demuxer.buffer.len());
            if first_packet_at.is_none()
                && events
                    .iter()
                    .any(|event| matches!(event, WebmMediaDemuxEvent::Packet { .. }))
            {
                first_packet_at = Some((index + 1) * 997);
            }
        }
        demuxer.finish().unwrap();
        assert!(first_packet_at.unwrap() < data.len());
        assert!(maximum_buffer < 64 * 1024);
    }

    #[test]
    fn oversized_block_is_rejected_from_its_header() {
        let mut demuxer = WebmMediaDemuxer::new();
        demuxer.state = ParserState::Clusters;
        let size = MAX_WEBM_PACKET_ELEMENT_BYTES + 1;
        let mut header = vec![0xA3, 0x01];
        header.extend_from_slice(&size.to_be_bytes()[1..]);
        let error = demuxer.add(&header).unwrap_err();
        assert!(error.contains("element 0xA3"));
        assert!(demuxer.buffer.len() < 16);
    }

    #[cfg(feature = "opus")]
    fn testdata_path(file: &str) -> PathBuf {
        // First try local testdata, then parent testdata
        let local = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("testdata")
            .join(file);
        if local.exists() {
            return local;
        }
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("soundkit-decoder")
            .join("testdata")
            .join(file)
    }

    #[cfg(feature = "opus")]
    fn outputs_path(file: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("outputs")
            .join(file)
    }

    #[test]
    fn test_webm_opus_demux() {
        let test_file = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("testdata")
            .join("test.webm");
        if !test_file.exists() {
            println!("Skipping test: no test.webm file found");
            return;
        }

        let data = fs::read(&test_file).unwrap();
        let mut demuxer = WebmOpusDemuxer::new();
        let mut config = None;
        let mut packets = 0usize;

        for chunk in data.chunks(997) {
            for event in demuxer.add(chunk).unwrap() {
                match event {
                    WebmOpusDemuxEvent::Config(next) => config = Some(next),
                    WebmOpusDemuxEvent::Packet { data, .. } => {
                        assert!(!data.is_empty());
                        packets += 1;
                    }
                }
            }
        }

        for event in demuxer.add(&[]).unwrap() {
            match event {
                WebmOpusDemuxEvent::Config(next) => config = Some(next),
                WebmOpusDemuxEvent::Packet { data, .. } => {
                    assert!(!data.is_empty());
                    packets += 1;
                }
            }
        }

        let config = config.expect("WebM Opus config event");
        assert_eq!(config.channels, 1);
        assert_eq!(config.sample_rate, 48_000);
        assert!(config.codec_private.starts_with(b"OpusHead"));
        assert!(packets > 0);
    }

    #[test]
    fn test_webm_audio_demux() {
        let test_file = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("testdata")
            .join("test.webm");
        if !test_file.exists() {
            println!("Skipping test: no test.webm file found");
            return;
        }

        let data = fs::read(&test_file).unwrap();
        let mut demuxer = WebmAudioDemuxer::new();
        let mut config = None;
        let mut packets = 0usize;

        for chunk in data.chunks(997) {
            for event in demuxer.add(chunk).unwrap() {
                match event {
                    WebmAudioDemuxEvent::Config(next) => config = Some(next),
                    WebmAudioDemuxEvent::Packet { data, .. } => {
                        assert!(!data.is_empty());
                        packets += 1;
                    }
                }
            }
        }

        for event in demuxer.add(&[]).unwrap() {
            match event {
                WebmAudioDemuxEvent::Config(next) => config = Some(next),
                WebmAudioDemuxEvent::Packet { data, .. } => {
                    assert!(!data.is_empty());
                    packets += 1;
                }
            }
        }

        let config = config.expect("WebM audio config event");
        assert_eq!(config.codec_id, "A_OPUS");
        assert_eq!(config.channels, 1);
        assert_eq!(config.sample_rate, 48_000);
        assert!(config.codec_private.starts_with(b"OpusHead"));
        assert!(packets > 0);
    }

    #[cfg(feature = "opus")]
    #[test]
    fn test_webm_decode_waveform() {
        let test_file = testdata_path(&format!("webm/{}.webm", TEST_FILE));
        if !test_file.exists() {
            println!("Skipping test: no webm file found at {:?}", test_file);
            return;
        }

        let data = fs::read(&test_file).unwrap();
        let mut decoder = WebmDecoder::new();
        decoder.init().unwrap();

        let mut decoded_bytes = Vec::new();

        for chunk in data.chunks(4096) {
            if let Some(audio) = decoder.add(chunk).unwrap() {
                decoded_bytes.extend_from_slice(audio.data());
            }
        }

        // Drain remaining
        loop {
            match decoder.add(&[]) {
                Ok(Some(audio)) => decoded_bytes.extend_from_slice(audio.data()),
                Ok(None) => break,
                Err(_) => break,
            }
        }

        assert!(!decoded_bytes.is_empty(), "decoder produced no PCM samples");

        // Get sample_rate and channels from decoder after parsing is complete
        let sample_rate = decoder.sample_rate().unwrap_or(48000);
        let channels = decoder.channels().unwrap_or(1);

        let decoded = s16le_to_i16(&decoded_bytes);
        let result = DecodeResult::new(&decoded, sample_rate, channels);
        print_waveform_with_header("WebM", &result);
    }

    #[cfg(feature = "opus")]
    #[test]
    fn test_webm_opus_decode() {
        let test_file = testdata_path("test.webm");
        if !test_file.exists() {
            println!("Skipping test: no test.webm file found");
            return;
        }

        let data = fs::read(&test_file).unwrap();
        let mut decoder = WebmDecoder::new();
        decoder.init().unwrap();

        let mut decoded = Vec::new();
        for chunk in data.chunks(4096) {
            if let Some(audio) = decoder.add(chunk).unwrap() {
                assert_eq!(audio.bits_per_sample(), 16);
                decoded.extend_from_slice(audio.data());
            }
        }

        // Drain remaining
        loop {
            match decoder.add(&[]) {
                Ok(Some(audio)) => decoded.extend_from_slice(audio.data()),
                Ok(None) => break,
                Err(_) => break,
            }
        }

        if !decoded.is_empty() {
            let output_path = outputs_path("test.s16le");
            fs::create_dir_all(output_path.parent().unwrap()).unwrap();
            fs::write(&output_path, &decoded).unwrap();
            println!("Decoded {} bytes from WebM", decoded.len());
        }
    }

    #[test]
    fn test_webm_vorbis_itag_171_decode() {
        let test_file = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("testdata")
            .join("itag171/yt_itag_171_vorbis.webm");
        let data = fs::read(&test_file).unwrap();
        let mut decoder = WebmDecoder::new();
        decoder.init().unwrap();

        let mut decoded = Vec::new();
        for chunk in data.chunks(997) {
            if let Some(audio) = decoder.add(chunk).unwrap() {
                assert_eq!(audio.bits_per_sample(), 16);
                assert_eq!(audio.channel_count(), 2);
                assert_eq!(audio.sampling_rate(), 44_100);
                decoded.extend_from_slice(audio.data());
            }
        }

        loop {
            match decoder.add(&[]) {
                Ok(Some(audio)) => decoded.extend_from_slice(audio.data()),
                Ok(None) => break,
                Err(error) => panic!("drain WebM Vorbis fixture: {}", error),
            }
        }

        assert_eq!(decoder.sample_rate(), Some(44_100));
        assert_eq!(decoder.channels(), Some(2));
        assert!(!decoded.is_empty(), "WebM Vorbis fixture decoded no PCM");
        assert!(decoded
            .chunks_exact(2)
            .map(|bytes| i16::from_le_bytes([bytes[0], bytes[1]]))
            .any(|sample| sample != 0));
    }
}
