use std::collections::BTreeMap;

use soundkit_video::{inspect_dnx_frame, DnxFrameInfo};

use crate::{AudioContainer, MediaTrackConfig, MediaTrackKind, MediaTrackPacket, PcmEndianness};

const UL_PREFIX: [u8; 4] = [0x06, 0x0e, 0x2b, 0x34];
const ESSENCE_ELEMENT_PREFIX: [u8; 12] = [
    0x06, 0x0e, 0x2b, 0x34, 0x01, 0x02, 0x01, 0x01, 0x0d, 0x01, 0x03, 0x01,
];
const PRIMER_PACK_KEY: [u8; 16] = [
    0x06, 0x0e, 0x2b, 0x34, 0x02, 0x05, 0x01, 0x01, 0x0d, 0x01, 0x02, 0x01, 0x01, 0x05, 0x01, 0x00,
];
const MAX_RUN_IN_BYTES: usize = 65_536;
const MAX_KLV_VALUE_BYTES: usize = 64 * 1024 * 1024;
const MAX_LOCAL_SET_ITEMS: usize = 65_536;
const MAX_TRACKS: usize = 64;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MxfMediaDemuxEvent {
    Config(MediaTrackConfig),
    Packet(MediaTrackPacket),
}

#[derive(Clone, Debug, Default)]
struct TrackMetadata {
    track_id: Option<u32>,
    track_number: Option<[u8; 4]>,
    edit_rate_numerator: Option<u32>,
    edit_rate_denominator: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DescriptorKind {
    Picture,
    WaveAudio,
    Aes3Audio,
    GenericAudio,
}

#[derive(Clone, Debug)]
struct DescriptorMetadata {
    kind: DescriptorKind,
    linked_track_id: Option<u32>,
    duration: Option<u64>,
    width: Option<u32>,
    height: Option<u32>,
    component_depth: Option<u32>,
    horizontal_subsampling: Option<u32>,
    vertical_subsampling: Option<u32>,
    sample_rate: Option<(u32, u32)>,
    channels: Option<u32>,
    bits_per_sample: Option<u32>,
    essence_container_ul: Option<[u8; 16]>,
    essence_codec_ul: Option<[u8; 16]>,
}

impl DescriptorMetadata {
    fn new(kind: DescriptorKind) -> Self {
        Self {
            kind,
            linked_track_id: None,
            duration: None,
            width: None,
            height: None,
            component_depth: None,
            horizontal_subsampling: None,
            vertical_subsampling: None,
            sample_rate: None,
            channels: None,
            bits_per_sample: None,
            essence_container_ul: None,
            essence_codec_ul: None,
        }
    }
}

#[derive(Clone, Debug)]
struct ActiveTrack {
    config: MediaTrackConfig,
    track_number: [u8; 4],
    video_packet_duration: Option<u32>,
    next_sample_id: u32,
    next_decode_time: u64,
}

/// Incremental, bounded SMPTE MXF KLV demuxer.
///
/// The first implementation supports frame-wrapped Generic Container picture
/// essence and Wave PCM sound essence, including DNxHD/DNxHR inspection. It
/// rejects clip-wrapped or unknown essence explicitly instead of asking the
/// browser to infer media semantics.
pub struct MxfMediaDemuxer {
    buffer: Vec<u8>,
    absolute_start: u64,
    started: bool,
    finished: bool,
    primer: BTreeMap<u16, [u8; 16]>,
    tracks: Vec<TrackMetadata>,
    descriptors: Vec<DescriptorMetadata>,
    active_tracks: Vec<ActiveTrack>,
}

impl MxfMediaDemuxer {
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            absolute_start: 0,
            started: false,
            finished: false,
            primer: BTreeMap::new(),
            tracks: Vec::new(),
            descriptors: Vec::new(),
            active_tracks: Vec::new(),
        }
    }

    pub fn push(&mut self, bytes: &[u8]) -> Result<Vec<MxfMediaDemuxEvent>, String> {
        if self.finished {
            return Err("MXF demuxer cannot accept bytes after flush".to_string());
        }
        self.buffer.extend_from_slice(bytes);
        self.drain(false)
    }

    pub fn flush(&mut self) -> Result<Vec<MxfMediaDemuxEvent>, String> {
        if self.finished {
            return Err("MXF demuxer was already flushed".to_string());
        }
        self.finished = true;
        let events = self.drain(true)?;
        if !self.buffer.is_empty() {
            return Err(format!(
                "truncated MXF KLV at byte {} ({} bytes remain)",
                self.absolute_start,
                self.buffer.len()
            ));
        }
        if self.active_tracks.is_empty() {
            return Err("MXF contains no supported frame-wrapped media essence".to_string());
        }
        Ok(events)
    }

    fn drain(&mut self, final_input: bool) -> Result<Vec<MxfMediaDemuxEvent>, String> {
        let mut output = Vec::new();
        loop {
            if !self.started {
                let Some(offset) = find_ul_prefix(&self.buffer) else {
                    if self.buffer.len() > MAX_RUN_IN_BYTES {
                        return Err("MXF run-in exceeds 65536 bytes".to_string());
                    }
                    if final_input && !self.buffer.is_empty() {
                        return Err("input does not contain an MXF universal label".to_string());
                    }
                    break;
                };
                if offset > MAX_RUN_IN_BYTES {
                    return Err("MXF run-in exceeds 65536 bytes".to_string());
                }
                self.consume(offset);
                self.started = true;
            }

            if self.buffer.len() < 17 {
                break;
            }
            if self.buffer[..4] != UL_PREFIX {
                return Err(format!(
                    "invalid MXF KLV key at byte {}",
                    self.absolute_start
                ));
            }
            let mut key = [0_u8; 16];
            key.copy_from_slice(&self.buffer[..16]);
            let Some((length, length_bytes)) = parse_ber_length(&self.buffer[16..])? else {
                break;
            };
            if length > MAX_KLV_VALUE_BYTES {
                return Err(format!(
                    "MXF KLV value of {length} bytes exceeds the SoundKit packet budget"
                ));
            }
            let header_bytes = 16_usize
                .checked_add(length_bytes)
                .ok_or_else(|| "MXF KLV header size overflow".to_string())?;
            let total_bytes = header_bytes
                .checked_add(length)
                .ok_or_else(|| "MXF KLV size overflow".to_string())?;
            if self.buffer.len() < total_bytes {
                break;
            }
            let value = self.buffer[header_bytes..total_bytes].to_vec();
            self.consume(total_bytes);
            self.process_klv(key, value, &mut output)?;
        }
        Ok(output)
    }

    fn process_klv(
        &mut self,
        key: [u8; 16],
        value: Vec<u8>,
        output: &mut Vec<MxfMediaDemuxEvent>,
    ) -> Result<(), String> {
        if key == PRIMER_PACK_KEY {
            self.parse_primer(&value)?;
            return Ok(());
        }
        if let Some(kind) = metadata_set_kind(&key) {
            let items = parse_local_set(&value)?;
            match kind {
                MetadataSetKind::Track => self.add_track(&items)?,
                MetadataSetKind::Descriptor(kind) => self.add_descriptor(kind, &items)?,
            }
            return Ok(());
        }
        if key[..12] == ESSENCE_ELEMENT_PREFIX {
            let mut track_number = [0_u8; 4];
            track_number.copy_from_slice(&key[12..]);
            self.emit_essence(track_number, value, output)?;
        }
        Ok(())
    }

    fn parse_primer(&mut self, value: &[u8]) -> Result<(), String> {
        if value.len() < 8 {
            return Err("truncated MXF primer pack".to_string());
        }
        let count = read_be_u32(value, 0)? as usize;
        let item_size = read_be_u32(value, 4)? as usize;
        if item_size != 18 || count > MAX_LOCAL_SET_ITEMS {
            return Err(format!(
                "unsupported MXF primer layout: {count} entries of {item_size} bytes"
            ));
        }
        let required = 8_usize
            .checked_add(
                count
                    .checked_mul(item_size)
                    .ok_or_else(|| "MXF primer size overflow".to_string())?,
            )
            .ok_or_else(|| "MXF primer size overflow".to_string())?;
        if required > value.len() {
            return Err("truncated MXF primer entries".to_string());
        }
        self.primer.clear();
        for entry in value[8..required].chunks_exact(18) {
            let tag = u16::from_be_bytes([entry[0], entry[1]]);
            let mut ul = [0_u8; 16];
            ul.copy_from_slice(&entry[2..18]);
            self.primer.insert(tag, ul);
        }
        Ok(())
    }

    fn add_track(&mut self, items: &[LocalItem<'_>]) -> Result<(), String> {
        if self.tracks.len() >= MAX_TRACKS {
            return Err("MXF track count exceeds the SoundKit limit".to_string());
        }
        let mut track = TrackMetadata::default();
        for item in items {
            match item.tag {
                0x4801 => track.track_id = Some(read_item_u32(item)?),
                0x4804 if item.value.len() == 4 => {
                    let mut number = [0_u8; 4];
                    number.copy_from_slice(item.value);
                    track.track_number = Some(number);
                }
                0x4b01 if item.value.len() == 8 => {
                    track.edit_rate_numerator = Some(read_be_u32(item.value, 0)?);
                    track.edit_rate_denominator = Some(read_be_u32(item.value, 4)?);
                }
                _ => {}
            }
        }
        if track.track_id.is_some() && track.track_number.is_some() {
            self.tracks.push(track);
        }
        Ok(())
    }

    fn add_descriptor(
        &mut self,
        kind: DescriptorKind,
        items: &[LocalItem<'_>],
    ) -> Result<(), String> {
        if self.descriptors.len() >= MAX_TRACKS {
            return Err("MXF descriptor count exceeds the SoundKit limit".to_string());
        }
        let mut descriptor = DescriptorMetadata::new(kind);
        for item in items {
            match item.tag {
                0x3002 => descriptor.duration = Some(read_item_u64(item)?),
                0x3004 => descriptor.essence_container_ul = read_item_ul(item),
                0x3006 => descriptor.linked_track_id = Some(read_item_u32(item)?),
                0x3201 | 0x3d06 => descriptor.essence_codec_ul = read_item_ul(item),
                0x3202 => descriptor.height = Some(read_item_u32(item)?),
                0x3203 => descriptor.width = Some(read_item_u32(item)?),
                0x3301 => descriptor.component_depth = Some(read_item_u32(item)?),
                0x3302 => descriptor.horizontal_subsampling = Some(read_item_u32(item)?),
                0x3308 => descriptor.vertical_subsampling = Some(read_item_u32(item)?),
                0x3d03 if item.value.len() == 8 => {
                    descriptor.sample_rate =
                        Some((read_be_u32(item.value, 0)?, read_be_u32(item.value, 4)?));
                }
                0x3d07 => descriptor.channels = Some(read_item_u32(item)?),
                0x3d01 => descriptor.bits_per_sample = Some(read_item_u32(item)?),
                _ => {}
            }
        }
        if descriptor.linked_track_id.is_some() {
            self.descriptors.push(descriptor);
        }
        Ok(())
    }

    fn emit_essence(
        &mut self,
        track_number: [u8; 4],
        value: Vec<u8>,
        output: &mut Vec<MxfMediaDemuxEvent>,
    ) -> Result<(), String> {
        let active_index = match self
            .active_tracks
            .iter()
            .position(|track| track.track_number == track_number)
        {
            Some(index) => index,
            None => {
                let active = self.resolve_track(track_number, &value)?;
                output.push(MxfMediaDemuxEvent::Config(active.config.clone()));
                self.active_tracks.push(active);
                self.active_tracks.len() - 1
            }
        };
        let active = &mut self.active_tracks[active_index];
        let duration = packet_duration(&active.config, active.video_packet_duration, &value)?;
        let packet = MediaTrackPacket {
            track_id: active.config.track_id,
            kind: active.config.kind,
            codec: active.config.codec.clone(),
            sample_id: active.next_sample_id,
            data: value,
            decode_time: active.next_decode_time,
            presentation_time: i64::try_from(active.next_decode_time)
                .map_err(|_| "MXF presentation time exceeds i64".to_string())?,
            duration,
            is_sync: true,
        };
        active.next_sample_id = active
            .next_sample_id
            .checked_add(1)
            .ok_or_else(|| "MXF sample id overflow".to_string())?;
        active.next_decode_time = active
            .next_decode_time
            .checked_add(u64::from(duration))
            .ok_or_else(|| "MXF decode time overflow".to_string())?;
        output.push(MxfMediaDemuxEvent::Packet(packet));
        Ok(())
    }

    fn resolve_track(
        &self,
        track_number: [u8; 4],
        first_packet: &[u8],
    ) -> Result<ActiveTrack, String> {
        let track = self
            .tracks
            .iter()
            .rev()
            .find(|track| track.track_number == Some(track_number))
            .ok_or_else(|| {
                format!(
                    "MXF essence track {:02x?} has no header metadata",
                    track_number
                )
            })?;
        let track_id = track
            .track_id
            .ok_or_else(|| "MXF track has no id".to_string())?;
        let descriptor = self
            .descriptors
            .iter()
            .rev()
            .find(|descriptor| descriptor.linked_track_id == Some(track_id))
            .ok_or_else(|| format!("MXF track {track_id} has no linked descriptor"))?;

        let (config, video_packet_duration) = match descriptor.kind {
            DescriptorKind::Picture => {
                let info = inspect_dnx_frame(first_packet).map_err(|error| {
                    format!("unsupported MXF picture essence on track {track_id}: {error}")
                })?;
                let packet_duration = track
                    .edit_rate_denominator
                    .filter(|value| *value > 0)
                    .ok_or_else(|| {
                        "MXF picture track has no valid edit-rate denominator".to_string()
                    })?;
                (
                    picture_config(track, descriptor, info)?,
                    Some(packet_duration),
                )
            }
            DescriptorKind::WaveAudio | DescriptorKind::Aes3Audio => {
                (pcm_audio_config(track, descriptor)?, None)
            }
            DescriptorKind::GenericAudio => {
                return Err("unknown MXF audio descriptor is not supported".to_string())
            }
        };
        Ok(ActiveTrack {
            config,
            track_number,
            video_packet_duration,
            next_sample_id: 0,
            next_decode_time: 0,
        })
    }

    fn consume(&mut self, bytes: usize) {
        self.buffer.drain(..bytes);
        self.absolute_start += bytes as u64;
    }
}

impl Default for MxfMediaDemuxer {
    fn default() -> Self {
        Self::new()
    }
}

fn picture_config(
    track: &TrackMetadata,
    descriptor: &DescriptorMetadata,
    info: DnxFrameInfo,
) -> Result<MediaTrackConfig, String> {
    // StoredWidth/StoredHeight describe the padded container raster. The DNx
    // coding-unit header describes the visible decoded raster and is therefore
    // authoritative here (for example, 368 stored lines can carry 360 visible
    // lines).
    if let Some(depth) = descriptor.component_depth {
        if depth != u32::from(info.bit_depth) {
            return Err(format!(
                "MXF descriptor depth {depth} disagrees with DNx depth {}",
                info.bit_depth
            ));
        }
    }
    let timescale = track
        .edit_rate_numerator
        .filter(|value| *value > 0)
        .ok_or_else(|| "MXF picture track has no valid edit rate".to_string())?;
    let sample_count = descriptor
        .duration
        .and_then(|value| u32::try_from(value).ok())
        .unwrap_or(0);
    Ok(MediaTrackConfig {
        container: AudioContainer::Mxf,
        kind: MediaTrackKind::Video,
        track_id: u64::from(track.track_id.unwrap_or_default()),
        codec: if info.profile.as_str().starts_with("dnxhr") {
            "dnxhr".to_string()
        } else {
            "dnxhd".to_string()
        },
        codec_id: info.profile.as_str().to_string(),
        timescale,
        timeline: None,
        sample_count,
        width: Some(info.width),
        height: Some(info.height),
        sample_rate: None,
        channels: None,
        bits_per_sample: Some(info.bit_depth),
        pcm_endianness: None,
        pcm_float: None,
        codec_private: Vec::new(),
        decoder_configuration: Vec::new(),
        nal_length_size: None,
    })
}

fn pcm_audio_config(
    track: &TrackMetadata,
    descriptor: &DescriptorMetadata,
) -> Result<MediaTrackConfig, String> {
    let container_ul = descriptor
        .essence_container_ul
        .ok_or_else(|| "MXF PCM descriptor has no essence-container label".to_string())?;
    const BWF_OR_AES3_PREFIX: [u8; 14] = [
        0x06, 0x0e, 0x2b, 0x34, 0x04, 0x01, 0x01, 0x01, 0x0d, 0x01, 0x03, 0x01, 0x02, 0x06,
    ];
    if container_ul[..14] != BWF_OR_AES3_PREFIX || !matches!(container_ul[14], 0x01 | 0x03) {
        return Err(format!(
            "unsupported MXF PCM essence-container label {:02x?}",
            container_ul
        ));
    }
    let (sample_rate_numerator, sample_rate_denominator) = descriptor
        .sample_rate
        .ok_or_else(|| "MXF PCM descriptor has no sample rate".to_string())?;
    if sample_rate_denominator == 0 || sample_rate_numerator % sample_rate_denominator != 0 {
        return Err(format!(
            "MXF PCM sample rate {sample_rate_numerator}/{sample_rate_denominator} is not integral"
        ));
    }
    let sample_rate = sample_rate_numerator / sample_rate_denominator;
    if !(8_000..=384_000).contains(&sample_rate) {
        return Err(format!("MXF PCM sample rate {sample_rate} is out of range"));
    }
    let channels = u8::try_from(
        descriptor
            .channels
            .ok_or_else(|| "MXF PCM descriptor has no channel count".to_string())?,
    )
    .map_err(|_| "MXF PCM channel count exceeds u8".to_string())?;
    if channels == 0 || channels > 32 {
        return Err(format!("MXF PCM channel count {channels} is out of range"));
    }
    let bits = u8::try_from(
        descriptor
            .bits_per_sample
            .ok_or_else(|| "MXF PCM descriptor has no sample depth".to_string())?,
    )
    .map_err(|_| "MXF PCM sample depth exceeds u8".to_string())?;
    if !matches!(bits, 8 | 16 | 20 | 24 | 32) {
        return Err(format!("unsupported MXF PCM sample depth {bits}"));
    }
    Ok(MediaTrackConfig {
        container: AudioContainer::Mxf,
        kind: MediaTrackKind::Audio,
        track_id: u64::from(track.track_id.unwrap_or_default()),
        codec: "pcm".to_string(),
        codec_id: format!("pcm_s{bits}le"),
        timescale: sample_rate,
        timeline: None,
        sample_count: descriptor
            .duration
            .and_then(|value| u32::try_from(value).ok())
            .unwrap_or(0),
        width: None,
        height: None,
        sample_rate: Some(sample_rate),
        channels: Some(channels),
        bits_per_sample: Some(bits),
        pcm_endianness: Some(PcmEndianness::Little),
        pcm_float: Some(false),
        codec_private: Vec::new(),
        decoder_configuration: Vec::new(),
        nal_length_size: None,
    })
}

fn packet_duration(
    config: &MediaTrackConfig,
    video_packet_duration: Option<u32>,
    packet: &[u8],
) -> Result<u32, String> {
    match config.kind {
        MediaTrackKind::Video => video_packet_duration
            .filter(|duration| *duration > 0)
            .ok_or_else(|| "MXF picture track has no packet duration".to_string()),
        MediaTrackKind::Audio => {
            let channels = usize::from(
                config
                    .channels
                    .ok_or_else(|| "MXF PCM track has no channel count".to_string())?,
            );
            let bits = usize::from(
                config
                    .bits_per_sample
                    .ok_or_else(|| "MXF PCM track has no sample depth".to_string())?,
            );
            let bytes_per_sample = bits.div_ceil(8);
            let frame_bytes = channels
                .checked_mul(bytes_per_sample)
                .ok_or_else(|| "MXF PCM frame size overflow".to_string())?;
            if frame_bytes == 0 || packet.len() % frame_bytes != 0 {
                return Err(format!(
                    "MXF PCM packet of {} bytes is not aligned to {frame_bytes}-byte frames",
                    packet.len()
                ));
            }
            u32::try_from(packet.len() / frame_bytes)
                .map_err(|_| "MXF PCM packet duration exceeds u32".to_string())
        }
    }
}

#[derive(Clone, Copy)]
enum MetadataSetKind {
    Track,
    Descriptor(DescriptorKind),
}

fn metadata_set_kind(key: &[u8; 16]) -> Option<MetadataSetKind> {
    const SET_PREFIX: [u8; 13] = [
        0x06, 0x0e, 0x2b, 0x34, 0x02, 0x53, 0x01, 0x01, 0x0d, 0x01, 0x01, 0x01, 0x01,
    ];
    if key[..13] != SET_PREFIX || key[13] != 0x01 || key[15] != 0x00 {
        return None;
    }
    match key[14] {
        0x3a | 0x3b => Some(MetadataSetKind::Track),
        0x28 | 0x29 | 0x51 => Some(MetadataSetKind::Descriptor(DescriptorKind::Picture)),
        0x48 => Some(MetadataSetKind::Descriptor(DescriptorKind::WaveAudio)),
        0x47 => Some(MetadataSetKind::Descriptor(DescriptorKind::Aes3Audio)),
        0x42 | 0x5e => Some(MetadataSetKind::Descriptor(DescriptorKind::GenericAudio)),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug)]
struct LocalItem<'a> {
    tag: u16,
    value: &'a [u8],
}

fn parse_local_set(value: &[u8]) -> Result<Vec<LocalItem<'_>>, String> {
    let mut items = Vec::new();
    let mut cursor = 0_usize;
    while cursor < value.len() {
        if items.len() >= MAX_LOCAL_SET_ITEMS {
            return Err("MXF local set has too many items".to_string());
        }
        let header = value
            .get(cursor..cursor + 4)
            .ok_or_else(|| "truncated MXF local-set item header".to_string())?;
        let tag = u16::from_be_bytes([header[0], header[1]]);
        let size = usize::from(u16::from_be_bytes([header[2], header[3]]));
        cursor += 4;
        let end = cursor
            .checked_add(size)
            .ok_or_else(|| "MXF local-set item size overflow".to_string())?;
        let item_value = value
            .get(cursor..end)
            .ok_or_else(|| format!("MXF local-set tag 0x{tag:04x} exceeds its KLV"))?;
        items.push(LocalItem {
            tag,
            value: item_value,
        });
        cursor = end;
    }
    Ok(items)
}

fn parse_ber_length(bytes: &[u8]) -> Result<Option<(usize, usize)>, String> {
    let Some(&first) = bytes.first() else {
        return Ok(None);
    };
    if first & 0x80 == 0 {
        return Ok(Some((usize::from(first), 1)));
    }
    let count = usize::from(first & 0x7f);
    if count == 0 {
        return Err("indefinite MXF BER lengths are not allowed".to_string());
    }
    if count > 8 {
        return Err(format!("MXF BER length uses {count} bytes"));
    }
    if bytes.len() < 1 + count {
        return Ok(None);
    }
    let mut value = 0_u64;
    for byte in &bytes[1..=count] {
        value = value
            .checked_shl(8)
            .and_then(|current| current.checked_add(u64::from(*byte)))
            .ok_or_else(|| "MXF BER length overflow".to_string())?;
    }
    let value = usize::try_from(value).map_err(|_| "MXF BER length exceeds usize".to_string())?;
    Ok(Some((value, 1 + count)))
}

fn find_ul_prefix(bytes: &[u8]) -> Option<usize> {
    bytes
        .windows(UL_PREFIX.len())
        .position(|window| window == UL_PREFIX)
}

fn read_be_u32(bytes: &[u8], offset: usize) -> Result<u32, String> {
    let value = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| "truncated MXF u32".to_string())?;
    Ok(u32::from_be_bytes([value[0], value[1], value[2], value[3]]))
}

fn read_item_u32(item: &LocalItem<'_>) -> Result<u32, String> {
    if item.value.len() != 4 {
        return Err(format!(
            "MXF local tag 0x{:04x} expected 4 bytes, got {}",
            item.tag,
            item.value.len()
        ));
    }
    read_be_u32(item.value, 0)
}

fn read_item_u64(item: &LocalItem<'_>) -> Result<u64, String> {
    if item.value.len() != 8 {
        return Err(format!(
            "MXF local tag 0x{:04x} expected 8 bytes, got {}",
            item.tag,
            item.value.len()
        ));
    }
    Ok(u64::from_be_bytes([
        item.value[0],
        item.value[1],
        item.value[2],
        item.value[3],
        item.value[4],
        item.value[5],
        item.value[6],
        item.value[7],
    ]))
}

fn read_item_ul(item: &LocalItem<'_>) -> Option<[u8; 16]> {
    if item.value.len() != 16 {
        return None;
    }
    let mut value = [0_u8; 16];
    value.copy_from_slice(item.value);
    Some(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn parses_short_and_long_ber_lengths() {
        assert_eq!(parse_ber_length(&[0x7f]).unwrap(), Some((127, 1)));
        assert_eq!(
            parse_ber_length(&[0x82, 0x01, 0x00]).unwrap(),
            Some((256, 3))
        );
        assert_eq!(
            parse_ber_length(&[0x83, 0x00, 0x01, 0x00]).unwrap(),
            Some((256, 4))
        );
        assert!(parse_ber_length(&[0x80]).is_err());
    }

    #[test]
    fn rejects_local_items_that_cross_their_set() {
        assert!(parse_local_set(&[0x48, 0x01, 0x00, 0x04, 0x00])
            .unwrap_err()
            .contains("exceeds"));
    }

    #[test]
    fn streams_real_dnxhr_hqx_and_pcm_op1a() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../testdata/video-compat/never-final/dnxhr-hqx-pcm.mxf"
        );
        let bytes = fs::read(path).unwrap();
        let mut demuxer = MxfMediaDemuxer::new();
        let mut events = Vec::new();
        for chunk in bytes.chunks(32_749) {
            events.extend(demuxer.push(chunk).unwrap());
        }
        events.extend(demuxer.flush().unwrap());

        let video_config = events.iter().find_map(|event| match event {
            MxfMediaDemuxEvent::Config(config) if config.kind == MediaTrackKind::Video => {
                Some(config)
            }
            _ => None,
        });
        let audio_config = events.iter().find_map(|event| match event {
            MxfMediaDemuxEvent::Config(config) if config.kind == MediaTrackKind::Audio => {
                Some(config)
            }
            _ => None,
        });
        let video_config = video_config.unwrap();
        assert_eq!(video_config.codec, "dnxhr");
        assert_eq!(video_config.codec_id, "dnxhr-hqx");
        assert_eq!(
            (video_config.width, video_config.height),
            (Some(640), Some(360))
        );
        let audio_config = audio_config.unwrap();
        assert_eq!(audio_config.codec, "pcm");
        assert_eq!(audio_config.sample_rate, Some(48_000));
        assert_eq!(audio_config.channels, Some(2));
        assert_eq!(audio_config.bits_per_sample, Some(24));

        let video_packets = events
            .iter()
            .filter(|event| {
                matches!(event, MxfMediaDemuxEvent::Packet(packet) if packet.kind == MediaTrackKind::Video)
            })
            .count();
        let audio_packets: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                MxfMediaDemuxEvent::Packet(packet) if packet.kind == MediaTrackKind::Audio => {
                    Some(packet)
                }
                _ => None,
            })
            .collect();
        assert_eq!(video_packets, 75);
        assert_eq!(audio_packets.len(), 75);
        assert_eq!(
            audio_packets
                .iter()
                .map(|packet| u64::from(packet.duration))
                .sum::<u64>(),
            144_000
        );
    }
}
