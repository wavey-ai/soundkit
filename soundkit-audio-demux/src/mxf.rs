use std::collections::BTreeMap;

use crate::MAX_CONTAINER_INPUT_CHUNK_BYTES;

use soundkit_video::{inspect_dnx_frame, DnxFrameInfo};

use crate::{
    AudioContainer, MediaSampleIndex, MediaTrackConfig, MediaTrackKind, MediaTrackPacket,
    PcmEndianness,
};

const UL_PREFIX: [u8; 4] = [0x06, 0x0e, 0x2b, 0x34];
const ESSENCE_ELEMENT_PREFIX: [u8; 12] = [
    0x06, 0x0e, 0x2b, 0x34, 0x01, 0x02, 0x01, 0x01, 0x0d, 0x01, 0x03, 0x01,
];
const PRIMER_PACK_KEY: [u8; 16] = [
    0x06, 0x0e, 0x2b, 0x34, 0x02, 0x05, 0x01, 0x01, 0x0d, 0x01, 0x02, 0x01, 0x01, 0x05, 0x01, 0x00,
];
const AVID_OP_ATOM_PCM_CONTAINER_UL: [u8; 16] = [
    0x06, 0x0e, 0x2b, 0x34, 0x01, 0x01, 0x01, 0xff, 0x4b, 0x46, 0x41, 0x41, 0x00, 0x0d, 0x4d, 0x4f,
];
const MAX_RUN_IN_BYTES: usize = 65_536;
const MAX_KLV_VALUE_BYTES: usize = 64 * 1024 * 1024;
const MAX_LOCAL_SET_ITEMS: usize = 65_536;
const MAX_TRACKS: usize = 64;
const MAX_MXF_INDEX_SAMPLES: usize = 8_000_000;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MxfMediaDemuxEvent {
    Config(MediaTrackConfig),
    Packet(MediaTrackPacket),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MxfPartitionKind {
    Header,
    Body,
    Footer,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MxfPartition {
    pub kind: MxfPartitionKind,
    pub offset: u64,
    pub this_partition: u64,
    pub previous_partition: u64,
    pub footer_partition: u64,
    pub header_byte_count: u64,
    pub index_byte_count: u64,
    pub index_sid: u32,
    pub body_offset: u64,
    pub body_sid: u32,
    pub closed: bool,
    pub complete: bool,
}

/// Seekable MXF sample index. Essence bytes remain in the source and are
/// represented by validated absolute ranges.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MxfMediaIndex {
    pub tracks: Vec<MediaTrackConfig>,
    pub samples: Vec<MediaSampleIndex>,
    pub partitions: Vec<MxfPartition>,
    pub random_index_offsets: Vec<(u32, u64)>,
    pub index_table_segments: u32,
    pub used_klv_fallback: bool,
    /// Source packing required when a seekable adapter reads a PCM range.
    pub pcm_source_packings: Vec<MxfTrackSourcePacking>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MxfPcmSourcePacking {
    Packed {
        frame_bytes: u32,
    },
    Aes3 {
        bits_per_sample: u8,
        channels: u8,
        stored_channels: u8,
        header_bytes: u8,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MxfTrackSourcePacking {
    pub track_id: u64,
    pub packing: MxfPcmSourcePacking,
}

impl MxfMediaIndex {
    /// Read one indexed sample and apply any MXF-specific PCM word unpacking.
    pub fn sample_data(&self, file: &[u8], sample: &MediaSampleIndex) -> Result<Vec<u8>, String> {
        let start = usize::try_from(sample.absolute_offset)
            .map_err(|_| "MXF sample offset exceeds this platform".to_string())?;
        let end = start
            .checked_add(sample.size as usize)
            .ok_or_else(|| "MXF sample range overflow".to_string())?;
        let source = file
            .get(start..end)
            .ok_or_else(|| "MXF sample range exceeds the source".to_string())?;
        match self
            .pcm_source_packings
            .iter()
            .find(|entry| entry.track_id == sample.track_id)
            .map(|entry| entry.packing)
        {
            Some(MxfPcmSourcePacking::Aes3 {
                bits_per_sample,
                channels,
                stored_channels,
                header_bytes,
            }) => unpack_aes3_pcm(
                source,
                bits_per_sample,
                channels,
                stored_channels,
                usize::from(header_bytes),
            ),
            _ => Ok(source.to_vec()),
        }
    }
}

impl MxfMediaIndex {
    /// Index a seekable MXF source. IndexTableSegments and the Random Index
    /// Pack are validated when present; bounded KLV scanning supplies exact
    /// essence ranges when an index table is absent or incomplete.
    pub fn from_file(file: &[u8]) -> Result<Self, String> {
        let run_in = find_ul_prefix(file).ok_or_else(|| {
            "input does not contain an MXF universal label within its run-in".to_string()
        })?;
        if run_in > MAX_RUN_IN_BYTES {
            return Err("MXF run-in exceeds 65536 bytes".to_string());
        }

        let mut metadata = MxfMediaDemuxer::new();
        let mut active_tracks: Vec<ActiveTrack> = Vec::new();
        let mut active_by_number = BTreeMap::new();
        let mut tracks = Vec::new();
        let mut samples = Vec::new();
        let mut partitions = Vec::new();
        let mut random_index_offsets = Vec::new();
        let mut index_table_segments = 0u32;
        let mut pcm_source_packings = Vec::new();
        let mut position = run_in;
        while position < file.len() {
            let key_bytes = file
                .get(position..position + 16)
                .ok_or_else(|| format!("truncated MXF KLV key at byte {position}"))?;
            // MXF permits privately registered 16-byte KLV keys. Avid uses
            // them for descriptive metadata alongside SMPTE UL-keyed sets.
            let mut key = [0u8; 16];
            key.copy_from_slice(key_bytes);
            let (length, length_bytes) = parse_ber_length(
                file.get(position + 16..)
                    .ok_or_else(|| "truncated MXF BER length".to_string())?,
            )?
            .ok_or_else(|| "truncated MXF BER length".to_string())?;
            let header_size = 16usize
                .checked_add(length_bytes)
                .ok_or_else(|| "MXF KLV header size overflow".to_string())?;
            let value_offset = position
                .checked_add(header_size)
                .ok_or_else(|| "MXF KLV value offset overflow".to_string())?;
            let end = value_offset
                .checked_add(length)
                .ok_or_else(|| "MXF KLV range overflow".to_string())?;
            let value = file
                .get(value_offset..end)
                .ok_or_else(|| format!("MXF KLV at byte {position} exceeds the source"))?;
            let is_essence = key[..12] == ESSENCE_ELEMENT_PREFIX;
            if (!is_essence && length > MAX_KLV_VALUE_BYTES)
                || (is_essence && length > crate::MAX_MEDIA_PACKET_BYTES as usize)
            {
                return Err(format!(
                    "MXF KLV value of {length} bytes exceeds its SoundKit budget"
                ));
            }

            if key == PRIMER_PACK_KEY {
                metadata.parse_primer(value)?;
            } else if let Some(kind) = metadata_set_kind(&key) {
                let items = parse_local_set(value)?;
                match kind {
                    MetadataSetKind::Track => metadata.add_track(&items)?,
                    MetadataSetKind::Descriptor(kind) => metadata.add_descriptor(kind, &items)?,
                    MetadataSetKind::Package(kind) => metadata.add_package(kind, &items)?,
                    MetadataSetKind::Sequence => metadata.add_sequence(&items)?,
                    MetadataSetKind::SourceClip => metadata.add_source_clip(&items)?,
                    MetadataSetKind::Preface => metadata.add_preface(&items)?,
                    MetadataSetKind::ContentStorage => metadata.add_content_storage(&items)?,
                }
            } else if let Some(kind) = partition_kind(&key) {
                partitions.push(parse_partition_pack(kind, &key, position as u64, value)?);
            } else if is_index_table_segment(&key) {
                validate_index_table_segment(value)?;
                index_table_segments = index_table_segments
                    .checked_add(1)
                    .ok_or_else(|| "MXF index-table count overflow".to_string())?;
            } else if is_random_index_pack(&key) {
                random_index_offsets = parse_random_index_pack(value, file.len() as u64)?;
            } else if is_essence {
                let mut track_number = [0u8; 4];
                track_number.copy_from_slice(&key[12..]);
                if metadata.is_known_undescribed_track(track_number)
                    || metadata.is_known_unsupported_picture_track(track_number)
                {
                    position = end;
                    continue;
                }
                let active_index = match active_by_number.get(&track_number).copied() {
                    Some(index) => index,
                    None => {
                        let active = metadata.resolve_track(track_number, value)?;
                        let index = active_tracks.len();
                        tracks.push(active.config.clone());
                        if let Some(packing) = active.pcm_source_packing {
                            let packing = match packing {
                                MxfPcmSourcePacking::Aes3 {
                                    bits_per_sample,
                                    channels,
                                    stored_channels,
                                    header_bytes: _,
                                } if active.clip_wrapped => MxfPcmSourcePacking::Aes3 {
                                    bits_per_sample,
                                    channels,
                                    stored_channels,
                                    header_bytes: 0,
                                },
                                packing => packing,
                            };
                            pcm_source_packings.push(MxfTrackSourcePacking {
                                track_id: active.config.track_id,
                                packing,
                            });
                        }
                        active_tracks.push(active);
                        active_by_number.insert(track_number, index);
                        index
                    }
                };
                let active = &mut active_tracks[active_index];
                if samples.len() >= MAX_MXF_INDEX_SAMPLES {
                    return Err(format!(
                        "MXF sample count exceeds the {MAX_MXF_INDEX_SAMPLES} index budget"
                    ));
                }
                append_indexed_essence_samples(active, &mut samples, value_offset as u64, value)?;
            }
            if end <= position {
                return Err("MXF KLV scanner made no progress".to_string());
            }
            position = end;
        }
        if samples.is_empty() {
            return Err("MXF contains no supported essence samples".to_string());
        }
        for track in &mut tracks {
            let count = samples
                .iter()
                .filter(|sample| sample.track_id == track.track_id)
                .count();
            track.sample_count = u32::try_from(count)
                .map_err(|_| "MXF track sample count exceeds u32".to_string())?;
        }
        Ok(Self {
            tracks,
            samples,
            partitions,
            random_index_offsets,
            index_table_segments,
            used_klv_fallback: index_table_segments == 0,
            pcm_source_packings,
        })
    }
}

#[derive(Clone, Debug, Default)]
struct TrackMetadata {
    instance_uid: Option<[u8; 16]>,
    track_id: Option<u32>,
    track_number: Option<[u8; 4]>,
    edit_rate_numerator: Option<u32>,
    edit_rate_denominator: Option<u32>,
    origin: Option<i64>,
    sequence_ref: Option<[u8; 16]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PackageKind {
    Material,
    Source,
}

#[derive(Clone, Debug)]
struct PackageMetadata {
    kind: PackageKind,
    instance_uid: [u8; 16],
    package_uid: [u8; 32],
    track_refs: Vec<[u8; 16]>,
    descriptor_ref: Option<[u8; 16]>,
}

#[derive(Clone, Debug)]
struct SequenceMetadata {
    instance_uid: [u8; 16],
    duration: Option<i64>,
    component_refs: Vec<[u8; 16]>,
}

#[derive(Clone, Debug)]
struct SourceClipMetadata {
    instance_uid: [u8; 16],
    duration: Option<i64>,
    start_position: Option<i64>,
    source_package_uid: [u8; 32],
    source_track_id: u32,
}

#[derive(Clone, Debug)]
struct PrefaceMetadata {
    instance_uid: [u8; 16],
    content_storage_ref: [u8; 16],
}

#[derive(Clone, Debug)]
struct ContentStorageMetadata {
    instance_uid: [u8; 16],
    package_refs: Vec<[u8; 16]>,
}

#[derive(Clone, Copy, Debug)]
struct ResolvedMaterialTimeline {
    edit_rate_numerator: u32,
    edit_rate_denominator: u32,
    wrapping_rate_numerator: u32,
    wrapping_rate_denominator: u32,
    material_origin: i64,
    source_origin: i64,
    source_clip_start: i64,
    duration: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DescriptorKind {
    Multiple,
    Picture,
    WaveAudio,
    Aes3Audio,
    GenericAudio,
}

#[derive(Clone, Debug)]
struct DescriptorMetadata {
    kind: DescriptorKind,
    instance_uid: Option<[u8; 16]>,
    subdescriptor_refs: Vec<[u8; 16]>,
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
    block_align: Option<u32>,
    essence_container_ul: Option<[u8; 16]>,
    essence_codec_ul: Option<[u8; 16]>,
}

impl DescriptorMetadata {
    fn new(kind: DescriptorKind) -> Self {
        Self {
            kind,
            instance_uid: None,
            subdescriptor_refs: Vec::new(),
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
            block_align: None,
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
    pcm_source_packing: Option<MxfPcmSourcePacking>,
    clip_wrapped: bool,
    edit_unit_frames: Option<u32>,
}

/// Incremental, bounded SMPTE MXF KLV demuxer.
///
/// The first implementation supports frame-wrapped Generic Container picture
/// essence and Wave PCM sound essence, including DNxHD/DNxHR inspection. It
/// exposes clip-wrapped PCM as a validated clip-level packet and rejects
/// unknown essence instead of asking the browser to infer media semantics.
pub struct MxfMediaDemuxer {
    buffer: Vec<u8>,
    cursor: usize,
    absolute_start: u64,
    started: bool,
    finished: bool,
    primer: BTreeMap<u16, [u8; 16]>,
    tracks: Vec<TrackMetadata>,
    descriptors: Vec<DescriptorMetadata>,
    packages: Vec<PackageMetadata>,
    sequences: Vec<SequenceMetadata>,
    source_clips: Vec<SourceClipMetadata>,
    prefaces: Vec<PrefaceMetadata>,
    content_storages: Vec<ContentStorageMetadata>,
    active_tracks: Vec<ActiveTrack>,
}

impl MxfMediaDemuxer {
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            cursor: 0,
            absolute_start: 0,
            started: false,
            finished: false,
            primer: BTreeMap::new(),
            tracks: Vec::new(),
            descriptors: Vec::new(),
            packages: Vec::new(),
            sequences: Vec::new(),
            source_clips: Vec::new(),
            prefaces: Vec::new(),
            content_storages: Vec::new(),
            active_tracks: Vec::new(),
        }
    }

    pub fn push(&mut self, bytes: &[u8]) -> Result<Vec<MxfMediaDemuxEvent>, String> {
        if self.finished {
            return Err("MXF demuxer cannot accept bytes after flush".to_string());
        }
        if bytes.len() > MAX_CONTAINER_INPUT_CHUNK_BYTES {
            return Err(format!(
                "MXF input chunk exceeds the {MAX_CONTAINER_INPUT_CHUNK_BYTES} byte streaming budget"
            ));
        }
        if self
            .buffer
            .len()
            .saturating_sub(self.cursor)
            .saturating_add(bytes.len())
            > MAX_KLV_VALUE_BYTES + 32
        {
            return Err("MXF streaming buffer exceeds the SoundKit KLV budget".to_string());
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
        if self.buffer.len() != self.cursor {
            return Err(format!(
                "truncated MXF KLV at byte {} ({} bytes remain)",
                self.absolute_start,
                self.buffer.len() - self.cursor
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
                let remaining = &self.buffer[self.cursor..];
                let Some(offset) = find_ul_prefix(remaining) else {
                    if remaining.len() > MAX_RUN_IN_BYTES {
                        return Err("MXF run-in exceeds 65536 bytes".to_string());
                    }
                    if final_input && !remaining.is_empty() {
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

            let remaining = &self.buffer[self.cursor..];
            if remaining.len() < 17 {
                break;
            }
            // Private metadata KLVs need not use the SMPTE UL prefix. Their
            // BER ranges are still bounded and skipped like other unknowns.
            let mut key = [0_u8; 16];
            key.copy_from_slice(&remaining[..16]);
            let Some((length, length_bytes)) = parse_ber_length(&remaining[16..])? else {
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
            if remaining.len() < total_bytes {
                break;
            }
            let value = remaining[header_bytes..total_bytes].to_vec();
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
                MetadataSetKind::Package(kind) => self.add_package(kind, &items)?,
                MetadataSetKind::Sequence => self.add_sequence(&items)?,
                MetadataSetKind::SourceClip => self.add_source_clip(&items)?,
                MetadataSetKind::Preface => self.add_preface(&items)?,
                MetadataSetKind::ContentStorage => self.add_content_storage(&items)?,
            }
            return Ok(());
        }
        if is_index_table_segment(&key) {
            validate_index_table_segment(&value)?;
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
            match local_property(&self.primer, item.tag) {
                Some(LocalProperty::InstanceUid) => track.instance_uid = read_item_uid(item),
                Some(LocalProperty::TrackId) => track.track_id = Some(read_item_u32(item)?),
                Some(LocalProperty::TrackNumber) if item.value.len() == 4 => {
                    let mut number = [0_u8; 4];
                    number.copy_from_slice(item.value);
                    track.track_number = Some(number);
                }
                Some(LocalProperty::EditRate) if item.value.len() == 8 => {
                    track.edit_rate_numerator = Some(read_be_u32(item.value, 0)?);
                    track.edit_rate_denominator = Some(read_be_u32(item.value, 4)?);
                }
                Some(LocalProperty::Origin) => track.origin = Some(read_item_i64(item)?),
                Some(LocalProperty::SequenceRef) => track.sequence_ref = read_item_uid(item),
                _ => {}
            }
        }
        if track.track_id.is_some() && track.track_number.is_some() {
            self.tracks.push(track);
        }
        Ok(())
    }

    fn add_package(&mut self, kind: PackageKind, items: &[LocalItem<'_>]) -> Result<(), String> {
        if self.packages.len() >= MAX_TRACKS * 4 {
            return Err("MXF package count exceeds the SoundKit limit".to_string());
        }
        let mut instance_uid = None;
        let mut package_uid = None;
        let mut track_refs = Vec::new();
        let mut descriptor_ref = None;
        for item in items {
            match local_property(&self.primer, item.tag) {
                Some(LocalProperty::InstanceUid) => instance_uid = read_item_uid(item),
                Some(LocalProperty::PackageUid) if item.value.len() == 32 => {
                    let mut uid = [0u8; 32];
                    uid.copy_from_slice(item.value);
                    package_uid = Some(uid);
                }
                Some(LocalProperty::PackageTracks) => {
                    track_refs = parse_strong_ref_batch(item.value, MAX_TRACKS, "package tracks")?
                }
                Some(LocalProperty::DescriptorRef) => descriptor_ref = read_item_uid(item),
                _ => {}
            }
        }
        if let (Some(instance_uid), Some(package_uid)) = (instance_uid, package_uid) {
            self.packages.push(PackageMetadata {
                kind,
                instance_uid,
                package_uid,
                track_refs,
                descriptor_ref,
            });
        }
        Ok(())
    }

    fn add_sequence(&mut self, items: &[LocalItem<'_>]) -> Result<(), String> {
        let mut instance_uid = None;
        let mut duration = None;
        let mut component_refs = Vec::new();
        for item in items {
            match local_property(&self.primer, item.tag) {
                Some(LocalProperty::InstanceUid) => instance_uid = read_item_uid(item),
                Some(LocalProperty::StructuralDuration) => duration = Some(read_item_i64(item)?),
                Some(LocalProperty::StructuralComponents) => {
                    component_refs = parse_strong_ref_batch(
                        item.value,
                        MAX_LOCAL_SET_ITEMS,
                        "sequence components",
                    )?
                }
                _ => {}
            }
        }
        if let Some(instance_uid) = instance_uid {
            self.sequences.push(SequenceMetadata {
                instance_uid,
                duration,
                component_refs,
            });
        }
        Ok(())
    }

    fn add_source_clip(&mut self, items: &[LocalItem<'_>]) -> Result<(), String> {
        let mut instance_uid = None;
        let mut duration = None;
        let mut start_position = None;
        let mut source_package_uid = None;
        let mut source_track_id = None;
        for item in items {
            match local_property(&self.primer, item.tag) {
                Some(LocalProperty::InstanceUid) => instance_uid = read_item_uid(item),
                Some(LocalProperty::StructuralDuration) => duration = Some(read_item_i64(item)?),
                Some(LocalProperty::StartPosition) => start_position = Some(read_item_i64(item)?),
                Some(LocalProperty::SourcePackageUid) if item.value.len() == 32 => {
                    let mut uid = [0u8; 32];
                    uid.copy_from_slice(item.value);
                    source_package_uid = Some(uid);
                }
                Some(LocalProperty::SourceTrackId) => source_track_id = Some(read_item_u32(item)?),
                _ => {}
            }
        }
        if let (Some(instance_uid), Some(source_package_uid), Some(source_track_id)) =
            (instance_uid, source_package_uid, source_track_id)
        {
            self.source_clips.push(SourceClipMetadata {
                instance_uid,
                duration,
                start_position,
                source_package_uid,
                source_track_id,
            });
        }
        Ok(())
    }

    fn add_preface(&mut self, items: &[LocalItem<'_>]) -> Result<(), String> {
        let mut instance_uid = None;
        let mut content_storage_ref = None;
        for item in items {
            match local_property(&self.primer, item.tag) {
                Some(LocalProperty::InstanceUid) => instance_uid = read_item_uid(item),
                Some(LocalProperty::ContentStorageRef) => content_storage_ref = read_item_uid(item),
                _ => {}
            }
        }
        if let (Some(instance_uid), Some(content_storage_ref)) = (instance_uid, content_storage_ref)
        {
            self.prefaces.push(PrefaceMetadata {
                instance_uid,
                content_storage_ref,
            });
        }
        Ok(())
    }

    fn add_content_storage(&mut self, items: &[LocalItem<'_>]) -> Result<(), String> {
        let mut instance_uid = None;
        let mut package_refs = Vec::new();
        for item in items {
            match local_property(&self.primer, item.tag) {
                Some(LocalProperty::InstanceUid) => instance_uid = read_item_uid(item),
                Some(LocalProperty::ContentPackages) => {
                    package_refs =
                        parse_strong_ref_batch(item.value, MAX_TRACKS * 4, "content packages")?
                }
                _ => {}
            }
        }
        if let Some(instance_uid) = instance_uid {
            self.content_storages.push(ContentStorageMetadata {
                instance_uid,
                package_refs,
            });
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
            match local_property(&self.primer, item.tag) {
                Some(LocalProperty::InstanceUid) => descriptor.instance_uid = read_item_uid(item),
                Some(LocalProperty::SubDescriptors) => {
                    descriptor.subdescriptor_refs =
                        parse_strong_ref_batch(item.value, MAX_TRACKS, "descriptor references")?
                }
                Some(LocalProperty::Duration) => descriptor.duration = Some(read_item_u64(item)?),
                Some(LocalProperty::EssenceContainer) => {
                    descriptor.essence_container_ul = read_item_ul(item)
                }
                Some(LocalProperty::LinkedTrackId) => {
                    descriptor.linked_track_id = Some(read_item_u32(item)?)
                }
                Some(LocalProperty::EssenceCodec) => {
                    descriptor.essence_codec_ul = read_item_ul(item)
                }
                Some(LocalProperty::StoredHeight) => descriptor.height = Some(read_item_u32(item)?),
                Some(LocalProperty::StoredWidth) => descriptor.width = Some(read_item_u32(item)?),
                Some(LocalProperty::ComponentDepth) => {
                    descriptor.component_depth = Some(read_item_u32(item)?)
                }
                Some(LocalProperty::HorizontalSubsampling) => {
                    descriptor.horizontal_subsampling = Some(read_item_u32(item)?)
                }
                Some(LocalProperty::VerticalSubsampling) => {
                    descriptor.vertical_subsampling = Some(read_item_u32(item)?)
                }
                Some(LocalProperty::AudioSampleRate) if item.value.len() == 8 => {
                    descriptor.sample_rate =
                        Some((read_be_u32(item.value, 0)?, read_be_u32(item.value, 4)?));
                }
                Some(LocalProperty::ChannelCount) => {
                    descriptor.channels = Some(read_item_u32(item)?)
                }
                Some(LocalProperty::QuantizationBits) => {
                    descriptor.bits_per_sample = Some(read_item_u32(item)?)
                }
                Some(LocalProperty::BlockAlign) if item.value.len() == 2 => {
                    descriptor.block_align = Some(u32::from(u16::from_be_bytes([
                        item.value[0],
                        item.value[1],
                    ])));
                }
                _ => {}
            }
        }
        if descriptor.instance_uid.is_some()
            || descriptor.linked_track_id.is_some()
            || descriptor.kind == DescriptorKind::Multiple
        {
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
                if self.is_known_undescribed_track(track_number)
                    || self.is_known_unsupported_picture_track(track_number)
                {
                    return Ok(());
                }
                let active = self.resolve_track(track_number, &value)?;
                output.push(MxfMediaDemuxEvent::Config(active.config.clone()));
                self.active_tracks.push(active);
                self.active_tracks.len() - 1
            }
        };
        let active = &mut self.active_tracks[active_index];
        let duration = packet_duration(active, &value)?;
        let data = match active.pcm_source_packing {
            Some(MxfPcmSourcePacking::Aes3 {
                bits_per_sample,
                channels,
                stored_channels,
                header_bytes,
            }) => unpack_aes3_pcm(
                &value,
                bits_per_sample,
                channels,
                stored_channels,
                usize::from(header_bytes),
            )?,
            _ => value,
        };
        if active.config.kind == MediaTrackKind::Video && active.clip_wrapped {
            let coding_unit_duration = active
                .video_packet_duration
                .filter(|duration| *duration > 0)
                .ok_or_else(|| "MXF picture track has no packet duration".to_string())?;
            for range in dnx_coding_unit_ranges(&data)? {
                let packet = MediaTrackPacket {
                    track_id: active.config.track_id,
                    kind: active.config.kind,
                    codec: active.config.codec.clone(),
                    sample_id: active.next_sample_id,
                    data: data[range].to_vec(),
                    decode_time: active.next_decode_time,
                    presentation_time: i64::try_from(active.next_decode_time)
                        .map_err(|_| "MXF presentation time exceeds i64".to_string())?,
                    duration: coding_unit_duration,
                    is_sync: true,
                };
                active.next_sample_id = active
                    .next_sample_id
                    .checked_add(1)
                    .ok_or_else(|| "MXF sample id overflow".to_string())?;
                active.next_decode_time = active
                    .next_decode_time
                    .checked_add(u64::from(coding_unit_duration))
                    .ok_or_else(|| "MXF decode time overflow".to_string())?;
                output.push(MxfMediaDemuxEvent::Packet(packet));
            }
            return Ok(());
        }
        let split_frames = active
            .edit_unit_frames
            .filter(|frames| active.clip_wrapped && *frames > 0);
        let packed_frame_bytes = active
            .config
            .pcm_bytes_per_frame
            .map(|bytes| bytes as usize);
        let mut remaining_frames = duration;
        let mut data_offset = 0usize;
        loop {
            let packet_duration = split_frames
                .map(|frames| frames.min(remaining_frames))
                .unwrap_or(remaining_frames);
            let data_size = match (split_frames, packed_frame_bytes) {
                (Some(_), Some(frame_bytes)) => usize::try_from(packet_duration)
                    .ok()
                    .and_then(|frames| frames.checked_mul(frame_bytes))
                    .ok_or_else(|| "MXF clip-wrapped PCM packet size overflow".to_string())?,
                _ => data.len(),
            };
            let packet_end = data_offset
                .checked_add(data_size)
                .ok_or_else(|| "MXF clip-wrapped PCM data range overflow".to_string())?;
            let packet_data = data
                .get(data_offset..packet_end)
                .ok_or_else(|| "MXF clip-wrapped PCM data is truncated".to_string())?
                .to_vec();
            let packet = MediaTrackPacket {
                track_id: active.config.track_id,
                kind: active.config.kind,
                codec: active.config.codec.clone(),
                sample_id: active.next_sample_id,
                data: packet_data,
                decode_time: active.next_decode_time,
                presentation_time: i64::try_from(active.next_decode_time)
                    .map_err(|_| "MXF presentation time exceeds i64".to_string())?,
                duration: packet_duration,
                is_sync: true,
            };
            active.next_sample_id = active
                .next_sample_id
                .checked_add(1)
                .ok_or_else(|| "MXF sample id overflow".to_string())?;
            active.next_decode_time = active
                .next_decode_time
                .checked_add(u64::from(packet_duration))
                .ok_or_else(|| "MXF decode time overflow".to_string())?;
            output.push(MxfMediaDemuxEvent::Packet(packet));
            if split_frames.is_none() || packet_duration == remaining_frames {
                break;
            }
            remaining_frames -= packet_duration;
            data_offset = packet_end;
        }
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
        let source_package = self.source_package_for_track(track);
        let descriptor = self
            .descriptors
            .iter()
            .rev()
            .find(|descriptor| {
                if !descriptor_matches_essence(descriptor.kind, track_number[0]) {
                    return false;
                }
                let linked_by_id = descriptor.linked_track_id == Some(track_id);
                let linked_by_package = source_package
                    .is_some_and(|package| self.package_references_descriptor(package, descriptor));
                (linked_by_id || linked_by_package)
                    && source_package.is_none_or(|package| {
                        self.package_references_descriptor(package, descriptor)
                    })
            })
            .ok_or_else(|| format!("MXF track {track_id} has no linked descriptor"))?;

        let resolved_timeline = self.resolve_material_timeline(track)?;
        let (mut config, video_packet_duration, pcm_source_packing) = match descriptor.kind {
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
                    None,
                )
            }
            DescriptorKind::WaveAudio | DescriptorKind::Aes3Audio => {
                let (config, packing) = pcm_audio_config(track, descriptor, track_number)?;
                (config, None, Some(packing))
            }
            DescriptorKind::GenericAudio if track_number[..3] == [0x06, 0x01, 0x10] => {
                let (config, packing) = pcm_audio_config(track, descriptor, track_number)?;
                (config, None, Some(packing))
            }
            DescriptorKind::GenericAudio => {
                return Err("unknown MXF audio descriptor is not supported".to_string())
            }
            DescriptorKind::Multiple => {
                return Err("MXF essence track resolved to a MultipleDescriptor".to_string())
            }
        };
        let clip_wrapped = descriptor
            .essence_container_ul
            .is_some_and(|label| match config.kind {
                MediaTrackKind::Audio => {
                    label == AVID_OP_ATOM_PCM_CONTAINER_UL
                        || (label[8..14] == [0x0d, 0x01, 0x03, 0x01, 0x02, 0x06]
                            && label[14] == 0x01)
                }
                MediaTrackKind::Video => label[14] == 0x02,
            });
        if let Some(timeline) = resolved_timeline {
            apply_material_timeline(&mut config, track, timeline)?;
        }
        let edit_unit_frames = if config.kind == MediaTrackKind::Audio && clip_wrapped {
            let (edit_rate_numerator, edit_rate_denominator) = resolved_timeline
                .map(|timeline| {
                    (
                        timeline.wrapping_rate_numerator,
                        timeline.wrapping_rate_denominator,
                    )
                })
                .unwrap_or((
                    track
                        .edit_rate_numerator
                        .filter(|value| *value > 0)
                        .ok_or_else(|| "MXF audio track has no valid edit rate".to_string())?,
                    track
                        .edit_rate_denominator
                        .filter(|value| *value > 0)
                        .ok_or_else(|| {
                            "MXF audio track has no valid edit-rate denominator".to_string()
                        })?,
                ));
            let numerator = u64::from(edit_rate_numerator);
            let denominator = u64::from(edit_rate_denominator);
            let sample_rate = u64::from(config.sample_rate.unwrap_or(config.timescale));
            let scaled = sample_rate
                .checked_mul(denominator)
                .ok_or_else(|| "MXF audio edit-unit duration overflow".to_string())?;
            if scaled % numerator != 0 {
                return Err(format!(
                    "MXF audio rate {sample_rate} is not integral at edit rate {numerator}/{denominator}"
                ));
            }
            Some(
                u32::try_from(scaled / numerator)
                    .map_err(|_| "MXF audio edit unit exceeds u32 frames".to_string())?,
            )
        } else {
            None
        };
        Ok(ActiveTrack {
            config,
            track_number,
            video_packet_duration,
            next_sample_id: 0,
            next_decode_time: 0,
            pcm_source_packing,
            clip_wrapped,
            edit_unit_frames,
        })
    }

    fn is_known_undescribed_track(&self, track_number: [u8; 4]) -> bool {
        let Some(track) = self
            .tracks
            .iter()
            .rev()
            .find(|track| track.track_number == Some(track_number))
        else {
            return false;
        };
        let track_id = track.track_id;
        let source_package = self.source_package_for_track(track);
        !self.descriptors.iter().any(|descriptor| {
            descriptor_matches_essence(descriptor.kind, track_number[0])
                && (descriptor.linked_track_id == track_id
                    || source_package.is_some_and(|package| {
                        self.package_references_descriptor(package, descriptor)
                    }))
        })
    }

    fn is_known_unsupported_picture_track(&self, track_number: [u8; 4]) -> bool {
        if !matches!(track_number[0], 0x05 | 0x15) {
            return false;
        }
        let Some(track) = self
            .tracks
            .iter()
            .rev()
            .find(|track| track.track_number == Some(track_number))
        else {
            return false;
        };
        let source_package = self.source_package_for_track(track);
        self.descriptors.iter().rev().any(|descriptor| {
            let linked = descriptor.linked_track_id == track.track_id
                || source_package
                    .is_some_and(|package| self.package_references_descriptor(package, descriptor));
            linked
                && descriptor.kind == DescriptorKind::Picture
                && descriptor.essence_container_ul.is_some_and(|label| {
                    label[..4] != UL_PREFIX || label[8..14] != [0x0d, 0x01, 0x03, 0x01, 0x02, 0x11]
                })
        })
    }

    fn resolve_material_timeline(
        &self,
        source_track: &TrackMetadata,
    ) -> Result<Option<ResolvedMaterialTimeline>, String> {
        let Some(source_track_uid) = source_track.instance_uid else {
            return Ok(None);
        };
        let Some(source_track_id) = source_track.track_id else {
            return Ok(None);
        };
        let Some(source_package) = self.packages.iter().rev().find(|package| {
            package.kind == PackageKind::Source && package.track_refs.contains(&source_track_uid)
        }) else {
            return Ok(None);
        };

        for source_clip in self.source_clips.iter().rev().filter(|clip| {
            clip.source_package_uid == source_package.package_uid
                && clip.source_track_id == source_track_id
        }) {
            let Some(sequence) = self
                .sequences
                .iter()
                .rev()
                .find(|sequence| sequence.component_refs.contains(&source_clip.instance_uid))
            else {
                continue;
            };
            let Some(material_track) = self
                .tracks
                .iter()
                .rev()
                .find(|track| track.sequence_ref == Some(sequence.instance_uid))
            else {
                continue;
            };
            let Some(material_track_uid) = material_track.instance_uid else {
                continue;
            };
            let Some(material_package) = self.packages.iter().rev().find(|package| {
                package.kind == PackageKind::Material
                    && package.track_refs.contains(&material_track_uid)
            }) else {
                continue;
            };
            self.validate_content_storage(source_package, material_package)?;
            let edit_rate_numerator = material_track
                .edit_rate_numerator
                .filter(|value| *value > 0)
                .ok_or_else(|| "MXF material track has no valid edit rate".to_string())?;
            let edit_rate_denominator = material_track
                .edit_rate_denominator
                .filter(|value| *value > 0)
                .ok_or_else(|| {
                    "MXF material track has no valid edit-rate denominator".to_string()
                })?;
            let duration = source_clip
                .duration
                .or(sequence.duration)
                .ok_or_else(|| "MXF material sequence has no duration".to_string())?;
            if duration < 0 {
                return Err("MXF material sequence has a negative duration".to_string());
            }
            let graph_wrapping_rate = material_package
                .track_refs
                .iter()
                .filter_map(|track_uid| {
                    self.tracks
                        .iter()
                        .rev()
                        .find(|track| track.instance_uid == Some(*track_uid))
                })
                .filter_map(|track| {
                    Some((
                        track.edit_rate_numerator.filter(|value| *value > 0)?,
                        track.edit_rate_denominator.filter(|value| *value > 0)?,
                    ))
                })
                .min_by(|left, right| {
                    (u64::from(left.0) * u64::from(right.1))
                        .cmp(&(u64::from(right.0) * u64::from(left.1)))
                })
                .unwrap_or((edit_rate_numerator, edit_rate_denominator));
            let (wrapping_rate_numerator, wrapping_rate_denominator) = graph_wrapping_rate;
            return Ok(Some(ResolvedMaterialTimeline {
                edit_rate_numerator,
                edit_rate_denominator,
                wrapping_rate_numerator,
                wrapping_rate_denominator,
                material_origin: material_track.origin.unwrap_or(0),
                source_origin: source_track.origin.unwrap_or(0),
                source_clip_start: source_clip.start_position.unwrap_or(0),
                duration,
            }));
        }

        Err(format!(
            "MXF source track {source_track_id} is not connected to a material-package timeline"
        ))
    }

    fn source_package_for_track(&self, track: &TrackMetadata) -> Option<&PackageMetadata> {
        let track_uid = track.instance_uid?;
        self.packages.iter().rev().find(|package| {
            package.kind == PackageKind::Source && package.track_refs.contains(&track_uid)
        })
    }

    fn package_references_descriptor(
        &self,
        package: &PackageMetadata,
        descriptor: &DescriptorMetadata,
    ) -> bool {
        let Some(package_descriptor_uid) = package.descriptor_ref else {
            return true;
        };
        if descriptor.instance_uid == Some(package_descriptor_uid) {
            return true;
        }
        self.descriptors.iter().rev().any(|multiple| {
            multiple.kind == DescriptorKind::Multiple
                && multiple.instance_uid == Some(package_descriptor_uid)
                && descriptor
                    .instance_uid
                    .is_some_and(|uid| multiple.subdescriptor_refs.contains(&uid))
        })
    }

    fn validate_content_storage(
        &self,
        source_package: &PackageMetadata,
        material_package: &PackageMetadata,
    ) -> Result<(), String> {
        if self.prefaces.is_empty() && self.content_storages.is_empty() {
            return Ok(());
        }
        let valid = self.prefaces.iter().rev().any(|preface| {
            preface.instance_uid != [0; 16]
                && self.content_storages.iter().rev().any(|storage| {
                    storage.instance_uid == preface.content_storage_ref
                        && storage.package_refs.contains(&source_package.instance_uid)
                        && storage
                            .package_refs
                            .contains(&material_package.instance_uid)
                })
        });
        if valid {
            Ok(())
        } else {
            Err("MXF package graph is not reachable from Preface ContentStorage".to_string())
        }
    }

    fn consume(&mut self, bytes: usize) {
        self.cursor += bytes;
        self.absolute_start += bytes as u64;
        if self.cursor > 64 * 1024 || self.cursor == self.buffer.len() {
            self.buffer.drain(..self.cursor);
            self.cursor = 0;
        }
    }
}

impl Default for MxfMediaDemuxer {
    fn default() -> Self {
        Self::new()
    }
}

fn descriptor_matches_essence(kind: DescriptorKind, item_type: u8) -> bool {
    match kind {
        DescriptorKind::Picture => matches!(item_type, 0x05 | 0x15),
        DescriptorKind::WaveAudio | DescriptorKind::Aes3Audio | DescriptorKind::GenericAudio => {
            matches!(item_type, 0x06 | 0x16)
        }
        DescriptorKind::Multiple => false,
    }
}

fn scale_edit_units(
    value: u64,
    edit_rate_numerator: u32,
    edit_rate_denominator: u32,
    timescale: u32,
    name: &str,
) -> Result<u64, String> {
    let scaled = u128::from(value)
        .checked_mul(u128::from(timescale))
        .and_then(|value| value.checked_mul(u128::from(edit_rate_denominator)))
        .ok_or_else(|| format!("MXF {name} conversion overflow"))?;
    let denominator = u128::from(edit_rate_numerator);
    if denominator == 0 || !scaled.is_multiple_of(denominator) {
        return Err(format!(
            "MXF {name} is not integral at edit rate {edit_rate_numerator}/{edit_rate_denominator} and timescale {timescale}"
        ));
    }
    u64::try_from(scaled / denominator)
        .map_err(|_| format!("MXF {name} exceeds u64 after conversion"))
}

fn apply_material_timeline(
    config: &mut MediaTrackConfig,
    source_track: &TrackMetadata,
    timeline: ResolvedMaterialTimeline,
) -> Result<(), String> {
    let source_rate_numerator = source_track
        .edit_rate_numerator
        .filter(|value| *value > 0)
        .ok_or_else(|| "MXF source track has no valid edit rate".to_string())?;
    let source_rate_denominator = source_track
        .edit_rate_denominator
        .filter(|value| *value > 0)
        .ok_or_else(|| "MXF source track has no valid edit-rate denominator".to_string())?;
    let presentation_start = scale_edit_units(
        timeline.material_origin.max(0) as u64,
        timeline.edit_rate_numerator,
        timeline.edit_rate_denominator,
        config.timescale,
        "material origin",
    )?;
    let media_start_units = timeline
        .source_clip_start
        .checked_add(timeline.source_origin)
        .ok_or_else(|| "MXF source timeline origin overflow".to_string())?
        .max(0);
    let media_start = scale_edit_units(
        media_start_units as u64,
        source_rate_numerator,
        source_rate_denominator,
        config.timescale,
        "source start position",
    )?;
    let duration = scale_edit_units(
        timeline.duration as u64,
        timeline.edit_rate_numerator,
        timeline.edit_rate_denominator,
        config.timescale,
        "material duration",
    )?;
    let resolved = crate::MediaTrackTimeline {
        presentation_start,
        media_start,
        duration,
    };
    config.timeline = Some(resolved);
    config.edit_timeline = vec![resolved];
    Ok(())
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
        edit_timeline: Vec::new(),
        sample_count,
        width: Some(info.width),
        height: Some(info.height),
        sample_rate: None,
        channels: None,
        bits_per_sample: Some(info.bit_depth),
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
    })
}

fn pcm_audio_config(
    track: &TrackMetadata,
    descriptor: &DescriptorMetadata,
    track_number: [u8; 4],
) -> Result<(MediaTrackConfig, MxfPcmSourcePacking), String> {
    let container_ul = descriptor
        .essence_container_ul
        .ok_or_else(|| "MXF PCM descriptor has no essence-container label".to_string())?;
    const BWF_OR_AES3_PREFIX: [u8; 14] = [
        0x06, 0x0e, 0x2b, 0x34, 0x04, 0x01, 0x01, 0x01, 0x0d, 0x01, 0x03, 0x01, 0x02, 0x06,
    ];
    const D10_CONTAINER_PREFIX: [u8; 14] = [
        0x06, 0x0e, 0x2b, 0x34, 0x04, 0x01, 0x01, 0x01, 0x0d, 0x01, 0x03, 0x01, 0x02, 0x01,
    ];
    if container_ul != AVID_OP_ATOM_PCM_CONTAINER_UL
        && container_ul[..14] != D10_CONTAINER_PREFIX
        && (container_ul[..14] != BWF_OR_AES3_PREFIX || !matches!(container_ul[14], 0x01 | 0x03))
    {
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
    let packed_frame_bytes = u32::from(channels)
        .checked_mul(u32::from(bits).div_ceil(8))
        .ok_or_else(|| "MXF PCM packed frame size overflow".to_string())?;
    let frame_bytes = descriptor.block_align.unwrap_or(packed_frame_bytes);
    // SMPTE 382 frame/clip-wrapped AES3 descriptors carry ordinary packed
    // PCM. Only the D-10 AES3 element mapping stores eight 32-bit subframes
    // behind a SMPTE 331M header and therefore requires word unpacking.
    let is_aes3 = track_number[..3] == [0x06, 0x01, 0x10];
    if is_aes3 && !matches!(bits, 16 | 24) {
        return Err(format!(
            "MXF AES3 unpacking supports 16-bit and 24-bit PCM, got {bits} bits"
        ));
    }
    if !is_aes3 && frame_bytes != packed_frame_bytes {
        return Err(format!(
            "unsupported MXF PCM block alignment {frame_bytes} for {channels} channels at {bits} bits"
        ));
    }
    let packing = if is_aes3 {
        MxfPcmSourcePacking::Aes3 {
            bits_per_sample: bits,
            channels,
            stored_channels: 8,
            header_bytes: 4,
        }
    } else {
        MxfPcmSourcePacking::Packed {
            frame_bytes: packed_frame_bytes,
        }
    };
    Ok((
        MediaTrackConfig {
            container: AudioContainer::Mxf,
            kind: MediaTrackKind::Audio,
            track_id: u64::from(track.track_id.unwrap_or_default()),
            codec: "pcm".to_string(),
            codec_id: format!("pcm_s{bits}le"),
            timescale: sample_rate,
            timeline: None,
            edit_timeline: Vec::new(),
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
            pcm_signed: Some(true),
            pcm_packed: Some(true),
            pcm_aligned_high: Some(false),
            pcm_interleaved: Some(true),
            pcm_bytes_per_frame: Some(packed_frame_bytes),
            pcm_frames_per_packet: Some(1),
            codec_private: Vec::new(),
            decoder_configuration: Vec::new(),
            nal_length_size: None,
        },
        packing,
    ))
}

fn packet_duration(active: &ActiveTrack, packet: &[u8]) -> Result<u32, String> {
    match active.config.kind {
        MediaTrackKind::Video => active
            .video_packet_duration
            .filter(|duration| *duration > 0)
            .ok_or_else(|| "MXF picture track has no packet duration".to_string()),
        MediaTrackKind::Audio => {
            let packing = active
                .pcm_source_packing
                .ok_or_else(|| "MXF PCM track has no source packing".to_string())?;
            packing.frame_count(packet)
        }
    }
}

fn append_indexed_essence_samples(
    active: &mut ActiveTrack,
    samples: &mut Vec<MediaSampleIndex>,
    absolute_offset: u64,
    packet: &[u8],
) -> Result<(), String> {
    if active.config.kind == MediaTrackKind::Video && active.clip_wrapped {
        let duration = active
            .video_packet_duration
            .filter(|duration| *duration > 0)
            .ok_or_else(|| "MXF picture track has no packet duration".to_string())?;
        for range in dnx_coding_unit_ranges(packet)? {
            if samples.len() >= MAX_MXF_INDEX_SAMPLES {
                return Err(format!(
                    "MXF sample count exceeds the {MAX_MXF_INDEX_SAMPLES} index budget"
                ));
            }
            samples.push(MediaSampleIndex {
                track_id: active.config.track_id,
                kind: active.config.kind,
                codec: active.config.codec.clone(),
                sample_id: active.next_sample_id,
                absolute_offset: absolute_offset
                    .checked_add(range.start as u64)
                    .ok_or_else(|| "MXF DNx sample offset overflow".to_string())?,
                size: u32::try_from(range.len())
                    .map_err(|_| "MXF DNx coding unit exceeds u32".to_string())?,
                decode_time: active.next_decode_time,
                presentation_time: i64::try_from(active.next_decode_time)
                    .map_err(|_| "MXF presentation time exceeds i64".to_string())?,
                duration,
                is_sync: true,
            });
            active.next_sample_id = active
                .next_sample_id
                .checked_add(1)
                .ok_or_else(|| "MXF sample id overflow".to_string())?;
            active.next_decode_time = active
                .next_decode_time
                .checked_add(u64::from(duration))
                .ok_or_else(|| "MXF decode time overflow".to_string())?;
        }
        return Ok(());
    }
    let packet_duration = packet_duration(active, packet)?;
    let split_frames = active
        .edit_unit_frames
        .filter(|frames| active.clip_wrapped && *frames > 0);
    let (source_frame_bytes, payload_offset) = match active.pcm_source_packing {
        Some(packing) => {
            let header_bytes = match (split_frames, packing) {
                (Some(_), MxfPcmSourcePacking::Aes3 { header_bytes, .. }) => {
                    u64::from(header_bytes)
                }
                (_, MxfPcmSourcePacking::Packed { .. }) => 0,
                _ => 0,
            };
            (Some(packing.source_frame_bytes()), header_bytes)
        }
        None => (None, 0),
    };
    let mut remaining_frames = packet_duration;
    let mut local_offset = payload_offset;
    loop {
        let duration = split_frames
            .map(|frames| frames.min(remaining_frames))
            .unwrap_or(remaining_frames);
        let size = match (split_frames, source_frame_bytes) {
            (Some(_), Some(frame_bytes)) => usize::try_from(duration)
                .ok()
                .and_then(|frames| frames.checked_mul(frame_bytes))
                .ok_or_else(|| "MXF clip-wrapped edit-unit size overflow".to_string())?,
            _ => packet.len(),
        };
        let sample_offset = absolute_offset
            .checked_add(local_offset)
            .ok_or_else(|| "MXF sample offset overflow".to_string())?;
        samples.push(MediaSampleIndex {
            track_id: active.config.track_id,
            kind: active.config.kind,
            codec: active.config.codec.clone(),
            sample_id: active.next_sample_id,
            absolute_offset: sample_offset,
            size: u32::try_from(size)
                .map_err(|_| "MXF essence element size exceeds u32".to_string())?,
            decode_time: active.next_decode_time,
            presentation_time: i64::try_from(active.next_decode_time)
                .map_err(|_| "MXF presentation time exceeds i64".to_string())?,
            duration,
            is_sync: true,
        });
        active.next_sample_id = active
            .next_sample_id
            .checked_add(1)
            .ok_or_else(|| "MXF sample id overflow".to_string())?;
        active.next_decode_time = active
            .next_decode_time
            .checked_add(u64::from(duration))
            .ok_or_else(|| "MXF decode time overflow".to_string())?;
        if split_frames.is_none() || duration == remaining_frames {
            break;
        }
        remaining_frames -= duration;
        local_offset = local_offset
            .checked_add(size as u64)
            .ok_or_else(|| "MXF clip-wrapped sample offset overflow".to_string())?;
    }
    Ok(())
}

fn dnx_coding_unit_ranges(packet: &[u8]) -> Result<Vec<std::ops::Range<usize>>, String> {
    let first_info = inspect_dnx_frame(packet)
        .map_err(|error| format!("invalid clip-wrapped DNx coding unit: {error}"))?;
    let mut starts = packet
        .windows(5)
        .enumerate()
        .filter_map(|(offset, window)| {
            if window[..2] != [0, 0] {
                return None;
            }
            let data_offset = u16::from_be_bytes([window[2], window[3]]);
            let plausible_prefix = (data_offset == 0x0280 && matches!(window[4], 1 | 2))
                || (window[4] == 3
                    && (0x0280..=0x2170).contains(&data_offset)
                    && data_offset.is_multiple_of(4));
            (plausible_prefix
                && inspect_dnx_frame(&packet[offset..]).is_ok_and(|info| info == first_info))
            .then_some(offset)
        })
        .collect::<Vec<_>>();
    if starts.first() != Some(&0) {
        return Err("MXF clip-wrapped DNx essence does not begin with a coding unit".to_string());
    }
    starts.push(packet.len());
    let ranges = starts
        .windows(2)
        .map(|bounds| bounds[0]..bounds[1])
        .collect::<Vec<_>>();
    if ranges.is_empty() {
        return Err("MXF clip-wrapped DNx essence contains no coding units".to_string());
    }
    Ok(ranges)
}

impl MxfPcmSourcePacking {
    fn source_frame_bytes(self) -> usize {
        match self {
            Self::Packed { frame_bytes } => frame_bytes as usize,
            Self::Aes3 {
                stored_channels, ..
            } => usize::from(stored_channels) * 4,
        }
    }

    fn payload(self, packet: &[u8]) -> Result<&[u8], String> {
        let header_bytes = match self {
            Self::Aes3 { header_bytes, .. } => usize::from(header_bytes),
            Self::Packed { .. } => 0,
        };
        packet
            .get(header_bytes..)
            .ok_or_else(|| "MXF AES3 packet is shorter than its SMPTE 331M header".to_string())
    }

    fn frame_count(self, packet: &[u8]) -> Result<u32, String> {
        let payload = self.payload(packet)?;
        let frame_bytes = self.source_frame_bytes();
        if frame_bytes == 0 || !payload.len().is_multiple_of(frame_bytes) {
            return Err(format!(
                "MXF PCM packet of {} payload bytes is not aligned to {frame_bytes}-byte source frames",
                payload.len()
            ));
        }
        let frames = payload.len() / frame_bytes;
        if let Self::Aes3 {
            header_bytes: 4, ..
        } = self
        {
            let declared = usize::from(u16::from_le_bytes([packet[1], packet[2]]));
            if declared != frames {
                return Err(format!(
                    "SMPTE 331M header declares {declared} AES3 frames but {frames} are present"
                ));
            }
        }
        u32::try_from(frames).map_err(|_| "MXF PCM packet duration exceeds u32".to_string())
    }
}

/// Unpack SMPTE 331M/382M AES3 subframes into interleaved little-endian PCM.
pub fn unpack_aes3_pcm(
    packet: &[u8],
    bits_per_sample: u8,
    channels: u8,
    stored_channels: u8,
    header_bytes: usize,
) -> Result<Vec<u8>, String> {
    if !matches!(bits_per_sample, 16 | 24) {
        return Err(format!(
            "MXF AES3 unpacking supports 16-bit and 24-bit PCM, got {bits_per_sample} bits"
        ));
    }
    if channels == 0 || stored_channels < channels || stored_channels > 32 {
        return Err("MXF AES3 channel geometry is invalid".to_string());
    }
    let packing = MxfPcmSourcePacking::Aes3 {
        bits_per_sample,
        channels,
        stored_channels,
        header_bytes: u8::try_from(header_bytes)
            .map_err(|_| "MXF AES3 header length exceeds u8".to_string())?,
    };
    let frames = usize::try_from(packing.frame_count(packet)?)
        .map_err(|_| "MXF AES3 frame count exceeds this platform".to_string())?;
    let bytes_per_sample = usize::from(bits_per_sample / 8);
    let output_bytes = frames
        .checked_mul(usize::from(channels))
        .and_then(|samples| samples.checked_mul(bytes_per_sample))
        .ok_or_else(|| "MXF AES3 output size overflow".to_string())?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(output_bytes)
        .map_err(|_| "MXF AES3 output allocation failed".to_string())?;
    for frame in packing
        .payload(packet)?
        .chunks_exact(packing.source_frame_bytes())
    {
        for word in frame.chunks_exact(4).take(usize::from(channels)) {
            let word = u32::from_le_bytes([word[0], word[1], word[2], word[3]]);
            if bits_per_sample == 24 {
                let sample = (word >> 4).to_le_bytes();
                output.extend_from_slice(&sample[..3]);
            } else {
                output.extend_from_slice(&((word >> 12) as u16).to_le_bytes());
            }
        }
    }
    Ok(output)
}

fn partition_kind(key: &[u8; 16]) -> Option<MxfPartitionKind> {
    const PREFIX: [u8; 13] = [
        0x06, 0x0e, 0x2b, 0x34, 0x02, 0x05, 0x01, 0x01, 0x0d, 0x01, 0x02, 0x01, 0x01,
    ];
    if key[..13] != PREFIX || key[15] != 0 {
        return None;
    }
    match key[13] {
        0x02 => Some(MxfPartitionKind::Header),
        0x03 => Some(MxfPartitionKind::Body),
        0x04 => Some(MxfPartitionKind::Footer),
        _ => None,
    }
}

fn parse_partition_pack(
    kind: MxfPartitionKind,
    key: &[u8; 16],
    offset: u64,
    value: &[u8],
) -> Result<MxfPartition, String> {
    if value.len() < 88 {
        return Err("truncated MXF partition pack".to_string());
    }
    let major = u16::from_be_bytes([value[0], value[1]]);
    let minor = u16::from_be_bytes([value[2], value[3]]);
    if major != 1 || minor > 3 {
        return Err(format!("unsupported MXF partition version {major}.{minor}"));
    }
    let status = key[14];
    if !(1..=4).contains(&status) {
        return Err(format!("invalid MXF partition status {status}"));
    }
    let this_partition = read_be_u64(value, 8)?;
    if this_partition != offset {
        return Err(format!(
            "MXF partition at {offset} declares offset {this_partition}"
        ));
    }
    Ok(MxfPartition {
        kind,
        offset,
        this_partition,
        previous_partition: read_be_u64(value, 16)?,
        footer_partition: read_be_u64(value, 24)?,
        header_byte_count: read_be_u64(value, 32)?,
        index_byte_count: read_be_u64(value, 40)?,
        index_sid: read_be_u32(value, 48)?,
        body_offset: read_be_u64(value, 52)?,
        body_sid: read_be_u32(value, 60)?,
        closed: matches!(status, 2 | 4),
        complete: matches!(status, 3 | 4),
    })
}

fn is_index_table_segment(key: &[u8; 16]) -> bool {
    *key == [
        0x06, 0x0e, 0x2b, 0x34, 0x02, 0x53, 0x01, 0x01, 0x0d, 0x01, 0x02, 0x01, 0x01, 0x10, 0x01,
        0x00,
    ]
}

fn validate_index_table_segment(value: &[u8]) -> Result<(), String> {
    let items = parse_local_set(value)?;
    let mut slice_count = 0usize;
    let mut pos_table_count = 0usize;
    for item in &items {
        match item.tag {
            0x3f05..=0x3f07 if item.value.len() != 4 => {
                return Err(format!(
                    "MXF index local tag 0x{:04x} must contain four bytes",
                    item.tag
                ));
            }
            0x3f0b if item.value.len() != 8 => {
                return Err("MXF index edit rate must contain eight bytes".to_string());
            }
            0x3f0b => {
                let numerator = read_be_u32(item.value, 0)?;
                let denominator = read_be_u32(item.value, 4)?;
                if numerator == 0 || denominator == 0 {
                    return Err("MXF index edit rate must be non-zero".to_string());
                }
            }
            0x3f0c | 0x3f0d if item.value.len() != 8 => {
                return Err("MXF index position or duration must contain eight bytes".to_string());
            }
            0x3f08 | 0x3f0e if item.value.len() != 1 => {
                return Err("MXF index count field must contain one byte".to_string());
            }
            0x3f08 => slice_count = usize::from(item.value[0]),
            0x3f0e => pos_table_count = usize::from(item.value[0]),
            _ => {}
        }
    }
    for item in &items {
        match item.tag {
            0x3f09 => validate_mxf_batch(item.value, 6, "DeltaEntryArray")?,
            0x3f0a => {
                let slice_width = slice_count
                    .checked_mul(4)
                    .ok_or_else(|| "MXF index slice-entry width overflow".to_string())?;
                let pos_width = pos_table_count
                    .checked_mul(8)
                    .ok_or_else(|| "MXF index position-table width overflow".to_string())?;
                let item_size = 11usize
                    .checked_add(slice_width)
                    .and_then(|value| value.checked_add(pos_width))
                    .ok_or_else(|| "MXF index-entry width overflow".to_string())?;
                validate_mxf_batch(item.value, item_size, "IndexEntryArray")?;
            }
            _ => {}
        }
    }
    Ok(())
}

fn validate_mxf_batch(value: &[u8], expected_item_size: usize, name: &str) -> Result<(), String> {
    if value.len() < 8 {
        return Err(format!("MXF {name} header is truncated"));
    }
    let count = read_be_u32(value, 0)? as usize;
    let item_size = read_be_u32(value, 4)? as usize;
    if count == 0 && item_size == 0 && value.len() == 8 {
        return Ok(());
    }
    if item_size != expected_item_size || count > MAX_MXF_INDEX_SAMPLES {
        return Err(format!(
            "unsupported MXF {name}: {count} entries of {item_size} bytes"
        ));
    }
    let required = count
        .checked_mul(item_size)
        .and_then(|size| size.checked_add(8))
        .ok_or_else(|| format!("MXF {name} size overflow"))?;
    if required != value.len() {
        return Err(format!("MXF {name} size disagrees with its entry count"));
    }
    Ok(())
}

fn is_random_index_pack(key: &[u8; 16]) -> bool {
    *key == [
        0x06, 0x0e, 0x2b, 0x34, 0x02, 0x05, 0x01, 0x01, 0x0d, 0x01, 0x02, 0x01, 0x01, 0x11, 0x01,
        0x00,
    ]
}

fn parse_random_index_pack(value: &[u8], file_size: u64) -> Result<Vec<(u32, u64)>, String> {
    if value.len() < 4 || !(value.len() - 4).is_multiple_of(12) {
        return Err("MXF Random Index Pack has an invalid size".to_string());
    }
    let entry_count = (value.len() - 4) / 12;
    if entry_count > MAX_TRACKS * 1024 {
        return Err("MXF Random Index Pack exceeds its entry budget".to_string());
    }
    let declared_size = read_be_u32(value, value.len() - 4)? as usize;
    if declared_size < value.len() + 17 {
        return Err("MXF Random Index Pack declares an invalid pack size".to_string());
    }
    let mut entries = Vec::new();
    entries
        .try_reserve_exact(entry_count)
        .map_err(|_| "MXF Random Index Pack allocation failed".to_string())?;
    for entry in value[..value.len() - 4].chunks_exact(12) {
        let body_sid = read_be_u32(entry, 0)?;
        let offset = read_be_u64(entry, 4)?;
        if offset >= file_size {
            return Err("MXF Random Index Pack offset exceeds the source".to_string());
        }
        entries.push((body_sid, offset));
    }
    Ok(entries)
}

#[derive(Clone, Copy)]
enum MetadataSetKind {
    Track,
    Descriptor(DescriptorKind),
    Package(PackageKind),
    Sequence,
    SourceClip,
    Preface,
    ContentStorage,
}

fn metadata_set_kind(key: &[u8; 16]) -> Option<MetadataSetKind> {
    const SET_PREFIX: [u8; 13] = [
        0x06, 0x0e, 0x2b, 0x34, 0x02, 0x53, 0x01, 0x01, 0x0d, 0x01, 0x01, 0x01, 0x01,
    ];
    if key[..13] != SET_PREFIX || key[13] != 0x01 || key[15] != 0x00 {
        return None;
    }
    match key[14] {
        0x2f => Some(MetadataSetKind::Preface),
        0x18 => Some(MetadataSetKind::ContentStorage),
        0x36 => Some(MetadataSetKind::Package(PackageKind::Material)),
        0x37 => Some(MetadataSetKind::Package(PackageKind::Source)),
        0x0f => Some(MetadataSetKind::Sequence),
        0x11 => Some(MetadataSetKind::SourceClip),
        0x3a | 0x3b => Some(MetadataSetKind::Track),
        0x44 => Some(MetadataSetKind::Descriptor(DescriptorKind::Multiple)),
        0x28 | 0x29 | 0x51 => Some(MetadataSetKind::Descriptor(DescriptorKind::Picture)),
        0x48 => Some(MetadataSetKind::Descriptor(DescriptorKind::WaveAudio)),
        0x47 => Some(MetadataSetKind::Descriptor(DescriptorKind::Aes3Audio)),
        0x42 | 0x5e => Some(MetadataSetKind::Descriptor(DescriptorKind::GenericAudio)),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LocalProperty {
    InstanceUid,
    ContentStorageRef,
    ContentPackages,
    PackageUid,
    PackageTracks,
    DescriptorRef,
    SubDescriptors,
    TrackId,
    TrackNumber,
    EditRate,
    Origin,
    SequenceRef,
    StructuralDuration,
    StructuralComponents,
    StartPosition,
    SourcePackageUid,
    SourceTrackId,
    Duration,
    EssenceContainer,
    LinkedTrackId,
    EssenceCodec,
    StoredHeight,
    StoredWidth,
    ComponentDepth,
    HorizontalSubsampling,
    VerticalSubsampling,
    AudioSampleRate,
    ChannelCount,
    QuantizationBits,
    BlockAlign,
}

fn local_property(primer: &BTreeMap<u16, [u8; 16]>, tag: u16) -> Option<LocalProperty> {
    if let Some(ul) = primer.get(&tag) {
        return match *ul {
            [0x06, 0x0e, 0x2b, 0x34, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x15, 0x02, 0, 0, 0, 0] => {
                Some(LocalProperty::InstanceUid)
            }
            [0x06, 0x0e, 0x2b, 0x34, 0x01, 0x01, 0x01, 0x02, 0x06, 0x01, 0x01, 0x04, 0x02, 0x01, 0, 0] => {
                Some(LocalProperty::ContentStorageRef)
            }
            [0x06, 0x0e, 0x2b, 0x34, 0x01, 0x01, 0x01, 0x02, 0x06, 0x01, 0x01, 0x04, 0x05, 0x01, 0, 0] => {
                Some(LocalProperty::ContentPackages)
            }
            [0x06, 0x0e, 0x2b, 0x34, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x15, 0x10, 0, 0, 0, 0] => {
                Some(LocalProperty::PackageUid)
            }
            [0x06, 0x0e, 0x2b, 0x34, 0x01, 0x01, 0x01, 0x02, 0x06, 0x01, 0x01, 0x04, 0x06, 0x05, 0, 0] => {
                Some(LocalProperty::PackageTracks)
            }
            [0x06, 0x0e, 0x2b, 0x34, 0x01, 0x01, 0x01, 0x02, 0x06, 0x01, 0x01, 0x04, 0x02, 0x03, 0, 0] => {
                Some(LocalProperty::DescriptorRef)
            }
            [0x06, 0x0e, 0x2b, 0x34, 0x01, 0x01, 0x01, 0x04, 0x06, 0x01, 0x01, 0x04, 0x06, 0x0b, 0, 0] => {
                Some(LocalProperty::SubDescriptors)
            }
            [0x06, 0x0e, 0x2b, 0x34, 0x01, 0x01, 0x01, 0x02, 0x01, 0x07, 0x01, 0x01, 0, 0, 0, 0] => {
                Some(LocalProperty::TrackId)
            }
            [0x06, 0x0e, 0x2b, 0x34, 0x01, 0x01, 0x01, 0x02, 0x01, 0x04, 0x01, 0x03, 0, 0, 0, 0] => {
                Some(LocalProperty::TrackNumber)
            }
            [0x06, 0x0e, 0x2b, 0x34, 0x01, 0x01, 0x01, 0x02, 0x05, 0x30, 0x04, 0x05, 0, 0, 0, 0] => {
                Some(LocalProperty::EditRate)
            }
            [0x06, 0x0e, 0x2b, 0x34, 0x01, 0x01, 0x01, 0x02, 0x07, 0x02, 0x01, 0x03, 0x01, 0x03, 0, 0] => {
                Some(LocalProperty::Origin)
            }
            [0x06, 0x0e, 0x2b, 0x34, 0x01, 0x01, 0x01, 0x02, 0x06, 0x01, 0x01, 0x04, 0x02, 0x04, 0, 0] => {
                Some(LocalProperty::SequenceRef)
            }
            [0x06, 0x0e, 0x2b, 0x34, 0x01, 0x01, 0x01, 0x02, 0x07, 0x02, 0x02, 0x01, 0x01, 0x03, 0, 0] => {
                Some(LocalProperty::StructuralDuration)
            }
            [0x06, 0x0e, 0x2b, 0x34, 0x01, 0x01, 0x01, 0x02, 0x06, 0x01, 0x01, 0x04, 0x06, 0x09, 0, 0] => {
                Some(LocalProperty::StructuralComponents)
            }
            [0x06, 0x0e, 0x2b, 0x34, 0x01, 0x01, 0x01, 0x02, 0x07, 0x02, 0x01, 0x03, 0x01, 0x04, 0, 0] => {
                Some(LocalProperty::StartPosition)
            }
            [0x06, 0x0e, 0x2b, 0x34, 0x01, 0x01, 0x01, 0x02, 0x06, 0x01, 0x01, 0x03, 0x01, 0, 0, 0] => {
                Some(LocalProperty::SourcePackageUid)
            }
            [0x06, 0x0e, 0x2b, 0x34, 0x01, 0x01, 0x01, 0x02, 0x06, 0x01, 0x01, 0x03, 0x02, 0, 0, 0] => {
                Some(LocalProperty::SourceTrackId)
            }
            [0x06, 0x0e, 0x2b, 0x34, 0x01, 0x01, 0x01, 0x01, 0x04, 0x06, 0x01, 0x02, 0, 0, 0, 0] => {
                Some(LocalProperty::Duration)
            }
            [0x06, 0x0e, 0x2b, 0x34, 0x01, 0x01, 0x01, 0x02, 0x06, 0x01, 0x01, 0x04, 0x01, 0x02, 0, 0] => {
                Some(LocalProperty::EssenceContainer)
            }
            [0x06, 0x0e, 0x2b, 0x34, 0x01, 0x01, 0x01, 0x05, 0x06, 0x01, 0x01, 0x03, 0x05, 0, 0, 0] => {
                Some(LocalProperty::LinkedTrackId)
            }
            [0x06, 0x0e, 0x2b, 0x34, 0x01, 0x01, 0x01, 0x02, 0x04, 0x01, 0x06, 0x01, 0, 0, 0, 0]
            | [0x06, 0x0e, 0x2b, 0x34, 0x01, 0x01, 0x01, 0x02, 0x04, 0x02, 0x04, 0x02, 0, 0, 0, 0] => {
                Some(LocalProperty::EssenceCodec)
            }
            [0x06, 0x0e, 0x2b, 0x34, 0x01, 0x01, 0x01, 0x01, 0x04, 0x01, 0x05, 0x02, 0x01, 0, 0, 0] => {
                Some(LocalProperty::StoredHeight)
            }
            [0x06, 0x0e, 0x2b, 0x34, 0x01, 0x01, 0x01, 0x01, 0x04, 0x01, 0x05, 0x02, 0x02, 0, 0, 0] => {
                Some(LocalProperty::StoredWidth)
            }
            [0x06, 0x0e, 0x2b, 0x34, 0x01, 0x01, 0x01, 0x02, 0x04, 0x01, 0x05, 0x03, 0x0a, 0, 0, 0] => {
                Some(LocalProperty::ComponentDepth)
            }
            [0x06, 0x0e, 0x2b, 0x34, 0x01, 0x01, 0x01, 0x01, 0x04, 0x01, 0x05, 0x01, 0x05, 0, 0, 0] => {
                Some(LocalProperty::HorizontalSubsampling)
            }
            [0x06, 0x0e, 0x2b, 0x34, 0x01, 0x01, 0x01, 0x02, 0x04, 0x01, 0x05, 0x01, 0x10, 0, 0, 0] => {
                Some(LocalProperty::VerticalSubsampling)
            }
            [0x06, 0x0e, 0x2b, 0x34, 0x01, 0x01, 0x01, 0x05, 0x04, 0x02, 0x03, 0x01, 0x01, 0x01, 0, 0] => {
                Some(LocalProperty::AudioSampleRate)
            }
            [0x06, 0x0e, 0x2b, 0x34, 0x01, 0x01, 0x01, 0x05, 0x04, 0x02, 0x01, 0x01, 0x04, 0, 0, 0] => {
                Some(LocalProperty::ChannelCount)
            }
            [0x06, 0x0e, 0x2b, 0x34, 0x01, 0x01, 0x01, 0x04, 0x04, 0x02, 0x03, 0x03, 0x04, 0, 0, 0] => {
                Some(LocalProperty::QuantizationBits)
            }
            [0x06, 0x0e, 0x2b, 0x34, 0x01, 0x01, 0x01, 0x05, 0x04, 0x02, 0x03, 0x02, 0x01, 0, 0, 0] => {
                Some(LocalProperty::BlockAlign)
            }
            _ => None,
        };
    }
    match tag {
        0x3c0a => Some(LocalProperty::InstanceUid),
        0x3b03 => Some(LocalProperty::ContentStorageRef),
        0x1901 => Some(LocalProperty::ContentPackages),
        0x4401 => Some(LocalProperty::PackageUid),
        0x4403 => Some(LocalProperty::PackageTracks),
        0x4701 => Some(LocalProperty::DescriptorRef),
        0x3f01 => Some(LocalProperty::SubDescriptors),
        0x4801 => Some(LocalProperty::TrackId),
        0x4804 => Some(LocalProperty::TrackNumber),
        0x4b01 => Some(LocalProperty::EditRate),
        0x4b02 => Some(LocalProperty::Origin),
        0x4803 => Some(LocalProperty::SequenceRef),
        0x0202 => Some(LocalProperty::StructuralDuration),
        0x1001 => Some(LocalProperty::StructuralComponents),
        0x1201 => Some(LocalProperty::StartPosition),
        0x1101 => Some(LocalProperty::SourcePackageUid),
        0x1102 => Some(LocalProperty::SourceTrackId),
        0x3002 => Some(LocalProperty::Duration),
        0x3004 => Some(LocalProperty::EssenceContainer),
        0x3006 => Some(LocalProperty::LinkedTrackId),
        0x3201 | 0x3d06 => Some(LocalProperty::EssenceCodec),
        0x3202 => Some(LocalProperty::StoredHeight),
        0x3203 => Some(LocalProperty::StoredWidth),
        0x3301 => Some(LocalProperty::ComponentDepth),
        0x3302 => Some(LocalProperty::HorizontalSubsampling),
        0x3308 => Some(LocalProperty::VerticalSubsampling),
        0x3d03 => Some(LocalProperty::AudioSampleRate),
        0x3d07 => Some(LocalProperty::ChannelCount),
        0x3d01 => Some(LocalProperty::QuantizationBits),
        0x3d0a => Some(LocalProperty::BlockAlign),
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

fn read_be_u64(bytes: &[u8], offset: usize) -> Result<u64, String> {
    let value = bytes
        .get(offset..offset + 8)
        .ok_or_else(|| "truncated MXF u64".to_string())?;
    Ok(u64::from_be_bytes([
        value[0], value[1], value[2], value[3], value[4], value[5], value[6], value[7],
    ]))
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

fn read_item_i64(item: &LocalItem<'_>) -> Result<i64, String> {
    if item.value.len() != 8 {
        return Err(format!(
            "MXF local tag 0x{:04x} expected 8 bytes, got {}",
            item.tag,
            item.value.len()
        ));
    }
    Ok(i64::from_be_bytes([
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

fn read_item_uid(item: &LocalItem<'_>) -> Option<[u8; 16]> {
    item.value.try_into().ok()
}

fn parse_strong_ref_batch(
    value: &[u8],
    max_items: usize,
    name: &str,
) -> Result<Vec<[u8; 16]>, String> {
    if value.len() < 8 {
        return Err(format!("truncated MXF {name} strong-reference batch"));
    }
    let count = usize::try_from(read_be_u32(value, 0)?)
        .map_err(|_| format!("MXF {name} count exceeds this platform"))?;
    let item_size = usize::try_from(read_be_u32(value, 4)?)
        .map_err(|_| format!("MXF {name} item size exceeds this platform"))?;
    if item_size != 16 || count > max_items {
        return Err(format!(
            "unsupported MXF {name} layout: {count} entries of {item_size} bytes"
        ));
    }
    let required = count
        .checked_mul(item_size)
        .and_then(|bytes| bytes.checked_add(8))
        .ok_or_else(|| format!("MXF {name} size overflow"))?;
    if value.len() != required {
        return Err(format!(
            "MXF {name} strong-reference batch has {} bytes, expected {required}",
            value.len()
        ));
    }
    let mut references = Vec::new();
    references
        .try_reserve_exact(count)
        .map_err(|_| format!("MXF {name} allocation failed"))?;
    for reference in value[8..].chunks_exact(16) {
        let mut uid = [0u8; 16];
        uid.copy_from_slice(reference);
        references.push(uid);
    }
    Ok(references)
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
    use sha2::{Digest, Sha256};
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
    fn resolves_remapped_local_tags_through_primer_uls() {
        let mut primer = BTreeMap::new();
        primer.insert(
            0x9001,
            [
                0x06, 0x0e, 0x2b, 0x34, 0x01, 0x01, 0x01, 0x02, 0x01, 0x07, 0x01, 0x01, 0, 0, 0, 0,
            ],
        );
        primer.insert(0x4801, [0xff; 16]);
        assert_eq!(
            local_property(&primer, 0x9001),
            Some(LocalProperty::TrackId)
        );
        assert_eq!(local_property(&primer, 0x4801), None);
    }

    #[test]
    fn unpacks_d10_aes3_words_to_packed_pcm() {
        fn aes3_packet(bits: u8, samples: &[[u32; 2]]) -> Vec<u8> {
            let mut packet = vec![0, samples.len() as u8, 0, 0];
            for frame in samples {
                for channel in 0..8 {
                    let sample = frame.get(channel).copied().unwrap_or(0);
                    let word = if bits == 24 {
                        sample << 4
                    } else {
                        sample << 12
                    };
                    packet.extend_from_slice(&word.to_le_bytes());
                }
            }
            packet
        }

        let packet_24 = aes3_packet(24, &[[0x12_34_56, 0xab_cd_ef], [1, 0x7f_ff_ff]]);
        assert_eq!(
            unpack_aes3_pcm(&packet_24, 24, 2, 8, 4).unwrap(),
            [0x56, 0x34, 0x12, 0xef, 0xcd, 0xab, 1, 0, 0, 0xff, 0xff, 0x7f]
        );

        let packet_16 = aes3_packet(16, &[[0x1234, 0xabcd], [1, 0x7fff]]);
        assert_eq!(
            unpack_aes3_pcm(&packet_16, 16, 2, 8, 4).unwrap(),
            [0x34, 0x12, 0xcd, 0xab, 1, 0, 0xff, 0x7f]
        );
    }

    #[test]
    fn rejects_malformed_aes3_and_reference_batches() {
        let mut bad_count = vec![0, 2, 0, 0];
        bad_count.extend_from_slice(&[0; 32]);
        assert!(unpack_aes3_pcm(&bad_count, 24, 2, 8, 4)
            .unwrap_err()
            .contains("declares 2"));
        assert!(unpack_aes3_pcm(&[0; 7], 24, 2, 8, 4)
            .unwrap_err()
            .contains("aligned"));
        assert!(unpack_aes3_pcm(&[0; 36], 20, 2, 8, 4)
            .unwrap_err()
            .contains("supports 16-bit and 24-bit"));

        let mut batch = Vec::new();
        batch.extend_from_slice(&2u32.to_be_bytes());
        batch.extend_from_slice(&16u32.to_be_bytes());
        batch.extend_from_slice(&[0; 16]);
        assert!(parse_strong_ref_batch(&batch, 4, "test")
            .unwrap_err()
            .contains("expected 40"));
    }

    #[test]
    fn rejects_a_dangling_source_package_timeline() {
        let track_uid = [1; 16];
        let track = TrackMetadata {
            instance_uid: Some(track_uid),
            track_id: Some(7),
            track_number: Some([0x16, 1, 1, 0]),
            edit_rate_numerator: Some(48_000),
            edit_rate_denominator: Some(1),
            origin: Some(0),
            sequence_ref: None,
        };
        let mut demuxer = MxfMediaDemuxer::new();
        demuxer.packages.push(PackageMetadata {
            kind: PackageKind::Source,
            instance_uid: [2; 16],
            package_uid: [3; 32],
            track_refs: vec![track_uid],
            descriptor_ref: None,
        });
        assert!(demuxer
            .resolve_material_timeline(&track)
            .unwrap_err()
            .contains("not connected"));
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

    #[test]
    fn indexes_real_op1a_ranges_partitions_and_sample_counts() {
        let bytes = include_bytes!("../../testdata/video-compat/never-final/dnxhr-hqx-pcm.mxf");
        let index = MxfMediaIndex::from_file(bytes).unwrap();
        assert_eq!(index.tracks.len(), 2);
        assert_eq!(index.samples.len(), 150);
        assert_eq!(
            index
                .tracks
                .iter()
                .map(|track| (track.kind, track.sample_count))
                .collect::<Vec<_>>(),
            vec![(MediaTrackKind::Video, 75), (MediaTrackKind::Audio, 75)]
        );
        assert!(index
            .partitions
            .iter()
            .any(|partition| partition.kind == MxfPartitionKind::Header));
        assert!(index
            .partitions
            .iter()
            .any(|partition| partition.kind == MxfPartitionKind::Footer));
        for sample in [
            &index.samples[0],
            &index.samples[index.samples.len() / 2],
            index.samples.last().unwrap(),
        ] {
            let start = sample.absolute_offset as usize;
            let end = start + sample.size as usize;
            assert!(end <= bytes.len());
            assert!(!bytes[start..end].is_empty());
        }
    }

    #[test]
    fn indexes_op_atom_picture_and_audio_without_companion_files() {
        let cases = [
            (
                include_bytes!("../../testdata/mxf-opatom/dnxhr-hqx-one-frame.mxf").as_slice(),
                MediaTrackKind::Video,
                1usize,
                "dnxhr",
            ),
            (
                include_bytes!("../../testdata/mxf-opatom/pcm24-mono-48k.mxf").as_slice(),
                MediaTrackKind::Audio,
                25usize,
                "pcm",
            ),
        ];
        for (bytes, kind, expected_samples, codec) in cases {
            let index = MxfMediaIndex::from_file(bytes).unwrap();
            assert_eq!(index.tracks.len(), 1);
            assert_eq!(index.tracks[0].kind, kind);
            assert_eq!(index.tracks[0].codec, codec);
            assert_eq!(index.tracks[0].sample_count as usize, expected_samples);
            assert_eq!(index.samples.len(), expected_samples);
            assert!(index.samples.iter().all(|sample| {
                let start = sample.absolute_offset as usize;
                start
                    .checked_add(sample.size as usize)
                    .is_some_and(|end| end <= bytes.len())
            }));

            let mut demuxer = MxfMediaDemuxer::new();
            let mut events = Vec::new();
            for chunk in bytes.chunks(997) {
                events.extend(demuxer.push(chunk).unwrap());
            }
            events.extend(demuxer.flush().unwrap());
            assert_eq!(
                events
                    .iter()
                    .filter(|event| matches!(event, MxfMediaDemuxEvent::Packet(_)))
                    .count(),
                expected_samples
            );
            if kind == MediaTrackKind::Audio {
                assert_eq!(
                    index.tracks[0].timeline,
                    Some(crate::MediaTrackTimeline {
                        presentation_start: 0,
                        media_start: 0,
                        duration: 48_000,
                    })
                );
                assert!(index
                    .samples
                    .iter()
                    .all(|sample| sample.duration == 1_920 && sample.size == 5_760));
                assert!(index.samples.windows(2).all(|samples| {
                    samples[0].absolute_offset + u64::from(samples[0].size)
                        == samples[1].absolute_offset
                        && samples[0].decode_time + u64::from(samples[0].duration)
                            == samples[1].decode_time
                }));
                let indexed_pcm = index
                    .samples
                    .iter()
                    .map(|sample| index.sample_data(bytes, sample).unwrap())
                    .flatten()
                    .collect::<Vec<_>>();
                let streamed_pcm = events
                    .iter()
                    .filter_map(|event| match event {
                        MxfMediaDemuxEvent::Packet(packet) => Some(packet.data.as_slice()),
                        MxfMediaDemuxEvent::Config(_) => None,
                    })
                    .flatten()
                    .copied()
                    .collect::<Vec<_>>();
                assert_eq!(indexed_pcm, streamed_pcm);
                assert_eq!(indexed_pcm.len(), 48_000 * 3);
            }
        }
    }

    #[test]
    fn indexes_real_avid_media_composer_op_atom_files() {
        let cases = [
            (
                include_bytes!("../../testdata/mxf-avid/track_01_v02.mxf").as_slice(),
                MediaTrackKind::Video,
                10usize,
                "dnxhr",
            ),
            (
                include_bytes!("../../testdata/mxf-avid/track_02_a01.mxf").as_slice(),
                MediaTrackKind::Audio,
                11usize,
                "pcm",
            ),
        ];
        for (bytes, kind, expected_samples, codec) in cases {
            let index = MxfMediaIndex::from_file(bytes).unwrap();
            let track = index
                .tracks
                .iter()
                .find(|track| track.kind == kind)
                .expect("expected Avid essence track");
            assert_eq!(track.codec, codec);
            assert_eq!(track.sample_count as usize, expected_samples);
            let samples = index
                .samples
                .iter()
                .filter(|sample| sample.kind == kind)
                .collect::<Vec<_>>();
            assert_eq!(samples.len(), expected_samples);
            assert!(samples.iter().all(|sample| {
                let start = sample.absolute_offset as usize;
                start
                    .checked_add(sample.size as usize)
                    .is_some_and(|end| end <= bytes.len())
            }));
            if kind == MediaTrackKind::Audio {
                assert!(samples
                    .iter()
                    .all(|sample| sample.duration == 2_002 && sample.size == 6_006));
                assert_eq!(
                    samples
                        .iter()
                        .map(|sample| u64::from(sample.duration))
                        .sum::<u64>(),
                    22_022
                );
                let pcm = samples
                    .iter()
                    .flat_map(|sample| index.sample_data(bytes, sample).unwrap())
                    .collect::<Vec<_>>();
                assert_eq!(
                    format!("{:x}", Sha256::digest(&pcm)),
                    "a52dbfc5af845dce23f28bc5b7b95bd75aabd55bd1641a90e64075973e8381e4"
                );
            }
        }
    }

    #[test]
    fn indexes_real_avid_multitrack_pcm() {
        let bytes = include_bytes!("../../testdata/mxf-avid/Avid-00005.mxf").as_slice();
        let index = MxfMediaIndex::from_file(bytes).unwrap();
        let audio_tracks = index
            .tracks
            .iter()
            .filter(|track| track.kind == MediaTrackKind::Audio)
            .collect::<Vec<_>>();
        assert_eq!(audio_tracks.len(), 2);
        assert!(audio_tracks.iter().all(|track| {
            track.sample_count == 25
                && track.sample_rate == Some(48_000)
                && track.channels == Some(1)
                && track.bits_per_sample == Some(16)
        }));
        for track in audio_tracks {
            let samples = index
                .samples
                .iter()
                .filter(|sample| sample.track_id == track.track_id)
                .collect::<Vec<_>>();
            assert_eq!(samples.len(), 25);
            assert!(samples
                .iter()
                .all(|sample| sample.duration == 1_920 && sample.size == 3_840));
            let pcm = samples
                .iter()
                .flat_map(|sample| index.sample_data(bytes, sample).unwrap())
                .collect::<Vec<_>>();
            assert_eq!(
                format!("{:x}", Sha256::digest(&pcm)),
                "c6649bfc9efb59ac9753e1704b6ee8e7c27e0dfdedc7d2b56227964a664a0649"
            );
        }
    }

    #[test]
    fn indexes_and_unpacks_real_d10_aes3_audio() {
        let bytes = include_bytes!("../../testdata/mxf-avid/d10-aes3-stereo-16bit.mxf").as_slice();
        let index = MxfMediaIndex::from_file(bytes).unwrap();
        let track = index
            .tracks
            .iter()
            .find(|track| track.kind == MediaTrackKind::Audio)
            .unwrap();
        assert_eq!(track.sample_rate, Some(48_000));
        assert_eq!(track.channels, Some(2));
        assert_eq!(track.bits_per_sample, Some(16));
        let samples = index
            .samples
            .iter()
            .filter(|sample| sample.kind == MediaTrackKind::Audio)
            .collect::<Vec<_>>();
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].duration, 1_920);
        assert_eq!(samples[0].size, 61_444);
        let pcm = index.sample_data(bytes, samples[0]).unwrap();
        assert_eq!(pcm.len(), 7_680);
        assert_eq!(
            format!("{:x}", Sha256::digest(&pcm)),
            "0a538ff65fd57a22526103706538ec42b1370e9fb5cf89924a143023d7b4f93c"
        );

        let mut demuxer = MxfMediaDemuxer::new();
        let mut events = Vec::new();
        for chunk in bytes.chunks(997) {
            events.extend(demuxer.push(chunk).unwrap());
        }
        events.extend(demuxer.flush().unwrap());
        let packet = events
            .iter()
            .find_map(|event| match event {
                MxfMediaDemuxEvent::Packet(packet) if packet.kind == MediaTrackKind::Audio => {
                    Some(packet)
                }
                _ => None,
            })
            .unwrap();
        assert_eq!(packet.duration, 1_920);
        assert_eq!(packet.data, pcm);
    }

    #[test]
    fn rejects_truncated_op_atom_essence_fixture() {
        let bytes =
            include_bytes!("../../testdata/mxf-avid/malformed-opatom-truncated.mxf").as_slice();
        let error = MxfMediaIndex::from_file(bytes).unwrap_err();
        assert!(error.contains("exceeds the source"), "{error}");

        for chunk_size in [1, 997, 4 * 1024, 64 * 1024] {
            let mut demuxer = MxfMediaDemuxer::new();
            let mut error = None;
            for chunk in bytes.chunks(chunk_size) {
                if let Err(candidate) = demuxer.push(chunk) {
                    error = Some(candidate);
                    break;
                }
            }
            let error = error.unwrap_or_else(|| demuxer.flush().unwrap_err());
            assert!(error.contains("truncated MXF KLV"), "{error}");
        }
    }
}
