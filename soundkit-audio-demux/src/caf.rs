use crate::{
    AudioCodec, AudioContainer, AudioPacketFormat, AudioTrackConfig, MediaSampleIndex,
    MediaTrackKind, MediaTrackPacket, PcmEndianness, PcmPacketTrim,
};

const MAX_CAF_DESCRIPTION_BYTES: u64 = 32;
const MAX_CAF_COOKIE_BYTES: u64 = 1024 * 1024;
const MAX_CAF_PACKET_TABLE_BYTES: u64 = 64 * 1024 * 1024;
const MAX_CAF_CHANNEL_LAYOUT_BYTES: u64 = 1024 * 1024;
const MAX_CAF_MATERIALIZED_PACKETS: usize = 8_000_000;
const MAX_CAF_PACKET_BYTES: u32 = 128 * 1024 * 1024;

const LPCM_IS_FLOAT: u32 = 1 << 0;
const LPCM_IS_BIG_ENDIAN: u32 = 1 << 1;
const LPCM_IS_SIGNED_INTEGER: u32 = 1 << 2;
const LPCM_IS_PACKED: u32 = 1 << 3;
const LPCM_IS_ALIGNED_HIGH: u32 = 1 << 4;
const LPCM_IS_NON_INTERLEAVED: u32 = 1 << 5;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CafChunkRange {
    pub chunk_type: [u8; 4],
    pub payload_offset: u64,
    pub payload_size: u64,
    pub end: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CafAudioIndex {
    pub config: AudioTrackConfig,
    pub packets: Vec<MediaSampleIndex>,
    pub valid_frames: u64,
    pub priming_frames: u32,
    pub remainder_frames: u32,
    pub channel_layout: Vec<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CafDescription {
    sample_rate: u32,
    format_id: [u8; 4],
    format_flags: u32,
    bytes_per_packet: u32,
    frames_per_packet: u32,
    channels: u8,
    bits_per_channel: u8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CafPacketTable {
    packet_count: u64,
    valid_frames: u64,
    priming_frames: u32,
    remainder_frames: u32,
    packet_sizes: Vec<u32>,
}

pub fn validate_caf_file_header(header: &[u8], file_size: u64) -> Result<(), String> {
    if file_size < 8 || header.len() < 8 {
        return Err("CAF source ends before its 8-byte file header".to_string());
    }
    if &header[..4] != b"caff" {
        return Err("CAF source does not start with caff".to_string());
    }
    let version = u16::from_be_bytes([header[4], header[5]]);
    let flags = u16::from_be_bytes([header[6], header[7]]);
    if version != 1 || flags != 0 {
        return Err(format!(
            "unsupported CAF header version={version} flags={flags}"
        ));
    }
    Ok(())
}

/// Inspect one CAF chunk without reading its payload.
///
/// A size of `-1` is accepted only for the final `data` chunk, as required by
/// the CAF specification. Metadata chunks are bounded before callers read
/// their payloads.
pub fn inspect_caf_chunk(
    header: &[u8],
    absolute_offset: u64,
    file_size: u64,
) -> Result<CafChunkRange, String> {
    if absolute_offset > file_size || file_size - absolute_offset < 12 {
        return Err("CAF source ends before a chunk header".to_string());
    }
    if header.len() < 12 {
        return Err("CAF chunk header needs 12 bytes".to_string());
    }
    let chunk_type: [u8; 4] = header[..4].try_into().unwrap();
    let payload_offset = absolute_offset
        .checked_add(12)
        .ok_or_else(|| "CAF chunk payload offset overflows u64".to_string())?;
    let signed_size = i64::from_be_bytes(header[4..12].try_into().unwrap());
    let payload_size = match signed_size {
        -1 if &chunk_type == b"data" => file_size - payload_offset,
        -1 => {
            return Err(format!(
                "CAF chunk {} cannot have an unknown size",
                String::from_utf8_lossy(&chunk_type)
            ))
        }
        value if value >= 0 => value as u64,
        value => return Err(format!("CAF chunk has invalid negative size {value}")),
    };
    let end = payload_offset
        .checked_add(payload_size)
        .ok_or_else(|| "CAF chunk range overflows u64".to_string())?;
    if end > file_size {
        return Err(format!(
            "CAF chunk {} exceeds the source length",
            String::from_utf8_lossy(&chunk_type)
        ));
    }
    let budget = match &chunk_type {
        b"desc" => Some(MAX_CAF_DESCRIPTION_BYTES),
        b"kuki" => Some(MAX_CAF_COOKIE_BYTES),
        b"pakt" => Some(MAX_CAF_PACKET_TABLE_BYTES),
        b"chan" => Some(MAX_CAF_CHANNEL_LAYOUT_BYTES),
        _ => None,
    };
    if let Some(budget) = budget {
        if payload_size > budget {
            return Err(format!(
                "CAF chunk {} exceeds the {budget} byte metadata budget",
                String::from_utf8_lossy(&chunk_type)
            ));
        }
    }
    Ok(CafChunkRange {
        chunk_type,
        payload_offset,
        payload_size,
        end,
    })
}

impl CafAudioIndex {
    /// Build a validated seekable packet index from a complete CAF file.
    /// Packet payloads are never copied into the index.
    pub fn from_file(file: &[u8]) -> Result<Self, String> {
        let file_size =
            u64::try_from(file.len()).map_err(|_| "CAF source length exceeds u64".to_string())?;
        validate_caf_file_header(file, file_size)?;

        let mut description = None;
        let mut packet_table = None;
        let mut magic_cookie = None;
        let mut channel_layout = None;
        let mut data_range = None;
        let mut position = 8u64;
        while position < file_size {
            let start = usize::try_from(position)
                .map_err(|_| "CAF chunk offset exceeds this platform".to_string())?;
            let range = inspect_caf_chunk(&file[start..], position, file_size)?;
            let payload_start = usize::try_from(range.payload_offset)
                .map_err(|_| "CAF payload offset exceeds this platform".to_string())?;
            let payload_end = usize::try_from(range.end)
                .map_err(|_| "CAF payload end exceeds this platform".to_string())?;
            let payload = &file[payload_start..payload_end];
            match &range.chunk_type {
                b"desc" => set_once(&mut description, payload.to_vec(), "desc")?,
                b"pakt" => set_once(&mut packet_table, payload.to_vec(), "pakt")?,
                b"kuki" => set_once(&mut magic_cookie, payload.to_vec(), "kuki")?,
                b"chan" => set_once(&mut channel_layout, payload.to_vec(), "chan")?,
                b"data" => {
                    if data_range.replace(range.clone()).is_some() {
                        return Err("CAF source contains multiple data chunks".to_string());
                    }
                }
                _ => {}
            }
            if range.end <= position {
                return Err("CAF chunk scanner made no progress".to_string());
            }
            position = range.end;
        }
        if position != file_size {
            return Err("CAF source ends inside a chunk header".to_string());
        }

        let description = description
            .as_deref()
            .ok_or_else(|| "CAF source is missing its desc chunk".to_string())?;
        let data = data_range.ok_or_else(|| "CAF source is missing its data chunk".to_string())?;
        if data.payload_size < 4 {
            return Err("CAF data chunk is shorter than its edit count".to_string());
        }
        let data_start = usize::try_from(data.payload_offset)
            .map_err(|_| "CAF data offset exceeds this platform".to_string())?;
        let edit_count = u32::from_be_bytes(file[data_start..data_start + 4].try_into().unwrap());
        Self::from_metadata(
            description,
            magic_cookie.as_deref().unwrap_or_default(),
            packet_table.as_deref(),
            channel_layout.as_deref().unwrap_or_default(),
            data.payload_offset,
            data.payload_size,
            edit_count,
        )
    }

    /// Build an index from bounded metadata reads and the seekable `data`
    /// range. `data_payload_offset` points at the four-byte edit count.
    pub fn from_metadata(
        description: &[u8],
        magic_cookie: &[u8],
        packet_table: Option<&[u8]>,
        channel_layout: &[u8],
        data_payload_offset: u64,
        data_payload_size: u64,
        _edit_count: u32,
    ) -> Result<Self, String> {
        if data_payload_size < 4 {
            return Err("CAF data chunk is shorter than its edit count".to_string());
        }
        if magic_cookie.len() as u64 > MAX_CAF_COOKIE_BYTES
            || channel_layout.len() as u64 > MAX_CAF_CHANNEL_LAYOUT_BYTES
            || packet_table.is_some_and(|data| data.len() as u64 > MAX_CAF_PACKET_TABLE_BYTES)
        {
            return Err("CAF metadata exceeds its bounded read budget".to_string());
        }
        let description = parse_description(description)?;
        let packet_start = data_payload_offset
            .checked_add(4)
            .ok_or_else(|| "CAF packet offset overflows u64".to_string())?;
        let packet_bytes = data_payload_size - 4;
        let parsed_table = packet_table.map(parse_packet_table).transpose()?;
        let packet_sizes = resolve_packet_sizes(description, parsed_table.as_ref(), packet_bytes)?;
        let packet_count = u64::try_from(packet_sizes.len())
            .map_err(|_| "CAF packet count exceeds u64".to_string())?;

        let total_packet_frames = packet_count
            .checked_mul(u64::from(description.frames_per_packet))
            .ok_or_else(|| "CAF packet frame count overflows u64".to_string())?;
        let (valid_frames, priming_frames, remainder_frames) = match parsed_table {
            Some(table) => {
                if table.packet_count != packet_count {
                    return Err("CAF packet table count disagrees with packet data".to_string());
                }
                let accounted = table
                    .valid_frames
                    .checked_add(u64::from(table.priming_frames))
                    .and_then(|value| value.checked_add(u64::from(table.remainder_frames)))
                    .ok_or_else(|| "CAF packet-table frame counts overflow u64".to_string())?;
                if accounted > total_packet_frames {
                    return Err("CAF packet-table frame counts exceed packet capacity".to_string());
                }
                (
                    table.valid_frames,
                    table.priming_frames,
                    table.remainder_frames,
                )
            }
            None => (total_packet_frames, 0, 0),
        };

        let (codec, codec_id, pcm) = resolve_format(description)?;
        let sample_count = u32::try_from(packet_count)
            .map_err(|_| "CAF packet count exceeds u32 sample identifiers".to_string())?;
        let mut packets = Vec::new();
        packets
            .try_reserve_exact(packet_sizes.len())
            .map_err(|_| "CAF packet index allocation failed".to_string())?;
        let mut offset = packet_start;
        let mut decode_time = 0u64;
        for (index, size) in packet_sizes.into_iter().enumerate() {
            let sample_id = u32::try_from(index + 1)
                .map_err(|_| "CAF sample identifier exceeds u32".to_string())?;
            packets.push(MediaSampleIndex {
                track_id: 1,
                kind: MediaTrackKind::Audio,
                codec: codec.as_str().to_string(),
                sample_id,
                absolute_offset: offset,
                size,
                decode_time,
                presentation_time: i64::try_from(decode_time)
                    .map_err(|_| "CAF presentation timestamp exceeds i64".to_string())?,
                duration: description.frames_per_packet,
                is_sync: true,
            });
            offset = offset
                .checked_add(u64::from(size))
                .ok_or_else(|| "CAF packet range overflows u64".to_string())?;
            decode_time = decode_time
                .checked_add(u64::from(description.frames_per_packet))
                .ok_or_else(|| "CAF packet timestamp overflows u64".to_string())?;
        }
        if offset != packet_start + packet_bytes {
            return Err("CAF packet ranges do not cover the data chunk".to_string());
        }

        Ok(Self {
            config: AudioTrackConfig {
                container: AudioContainer::Caf,
                codec,
                packet_format: Some(AudioPacketFormat::Raw),
                codec_id: Some(codec_id),
                track_id: Some(1),
                pid: None,
                stream_type: None,
                timescale: Some(description.sample_rate),
                transport_packet_stride: None,
                transport_prefix_bytes: None,
                program_number: None,
                sample_rate: Some(description.sample_rate),
                channels: Some(description.channels),
                bits_per_sample: (description.bits_per_channel != 0)
                    .then_some(description.bits_per_channel),
                pcm_endianness: pcm.map(|value| value.0),
                pcm_float: pcm.map(|value| value.1),
                pcm_signed: pcm.map(|value| value.2),
                pcm_packed: pcm.map(|value| value.3),
                pcm_aligned_high: pcm.map(|value| value.4),
                pcm_interleaved: pcm.map(|value| value.5),
                pcm_bytes_per_frame: pcm
                    .map(|_| description.bytes_per_packet / description.frames_per_packet),
                pcm_frames_per_packet: Some(description.frames_per_packet),
                sample_count: Some(sample_count),
                codec_private: magic_cookie.to_vec(),
                pre_skip: u16::try_from(priming_frames).ok(),
                output_gain: None,
                mapping_family: None,
            },
            packets,
            valid_frames,
            priming_frames,
            remainder_frames,
            channel_layout: channel_layout.to_vec(),
        })
    }

    /// Validate and normalize one indexed CAF packet.
    ///
    /// Callers must read exactly the range in `packets[sample_index]`.
    pub fn packet_from_sample_bytes(
        &self,
        sample_index: usize,
        raw: &[u8],
    ) -> Result<MediaTrackPacket, String> {
        let sample = self
            .packets
            .get(sample_index)
            .ok_or_else(|| format!("CAF sample index {sample_index} is out of range"))?;
        if raw.len() != sample.size as usize {
            return Err(format!(
                "CAF sample {} expected {} bytes, got {}",
                sample.sample_id,
                sample.size,
                raw.len()
            ));
        }
        Ok(MediaTrackPacket {
            track_id: sample.track_id,
            kind: sample.kind,
            codec: sample.codec.clone(),
            sample_id: sample.sample_id,
            data: raw.to_vec(),
            decode_time: sample.decode_time,
            presentation_time: sample.presentation_time,
            duration: sample.duration,
            is_sync: sample.is_sync,
        })
    }

    /// Select the decoded frames that belong to the CAF presentation.
    /// Priming and remainder frames stay outside the returned range.
    pub fn pcm_packet_trim(
        &self,
        sample_index: usize,
        decoded_frames: u32,
    ) -> Result<Option<PcmPacketTrim>, String> {
        let sample = self
            .packets
            .get(sample_index)
            .ok_or_else(|| format!("CAF sample index {sample_index} is out of range"))?;
        let packet_start = sample.decode_time;
        let packet_end = packet_start
            .checked_add(u64::from(decoded_frames))
            .ok_or_else(|| "CAF decoded packet timeline overflows u64".to_string())?;
        let programme_start = u64::from(self.priming_frames);
        let programme_end = programme_start
            .checked_add(self.valid_frames)
            .ok_or_else(|| "CAF presentation timeline overflows u64".to_string())?;
        let start = packet_start.max(programme_start);
        let end = packet_end.min(programme_end);
        if end <= start {
            return Ok(None);
        }
        Ok(Some(PcmPacketTrim {
            source_frame_start: u32::try_from(start - packet_start)
                .map_err(|_| "CAF packet trim start exceeds u32".to_string())?,
            frame_count: u32::try_from(end - start)
                .map_err(|_| "CAF packet trim length exceeds u32".to_string())?,
        }))
    }
}

fn set_once(slot: &mut Option<Vec<u8>>, value: Vec<u8>, name: &str) -> Result<(), String> {
    if slot.replace(value).is_some() {
        return Err(format!("CAF source contains multiple {name} chunks"));
    }
    Ok(())
}

fn parse_description(data: &[u8]) -> Result<CafDescription, String> {
    if data.len() != 32 {
        return Err(format!(
            "CAF desc must contain 32 bytes, got {}",
            data.len()
        ));
    }
    let raw_rate = f64::from_be_bytes(data[..8].try_into().unwrap());
    if !raw_rate.is_finite()
        || raw_rate <= 0.0
        || raw_rate > f64::from(u32::MAX)
        || raw_rate.fract() != 0.0
    {
        return Err(format!("invalid CAF sample rate: {raw_rate}"));
    }
    let channels = u32::from_be_bytes(data[24..28].try_into().unwrap());
    let bits = u32::from_be_bytes(data[28..32].try_into().unwrap());
    Ok(CafDescription {
        sample_rate: raw_rate as u32,
        format_id: data[8..12].try_into().unwrap(),
        format_flags: u32::from_be_bytes(data[12..16].try_into().unwrap()),
        bytes_per_packet: u32::from_be_bytes(data[16..20].try_into().unwrap()),
        frames_per_packet: u32::from_be_bytes(data[20..24].try_into().unwrap()),
        channels: u8::try_from(channels)
            .ok()
            .filter(|value| *value != 0)
            .ok_or_else(|| "CAF channel count is outside 1...255".to_string())?,
        bits_per_channel: u8::try_from(bits)
            .map_err(|_| "CAF bits per channel exceeds 255".to_string())?,
    })
}

fn parse_packet_table(data: &[u8]) -> Result<CafPacketTable, String> {
    if data.len() < 24 {
        return Err("CAF packet table is shorter than 24 bytes".to_string());
    }
    let packet_count = i64::from_be_bytes(data[..8].try_into().unwrap());
    let valid_frames = i64::from_be_bytes(data[8..16].try_into().unwrap());
    let priming_frames = i32::from_be_bytes(data[16..20].try_into().unwrap());
    let remainder_frames = i32::from_be_bytes(data[20..24].try_into().unwrap());
    if packet_count < 0 || valid_frames < 0 || priming_frames < 0 || remainder_frames < 0 {
        return Err("CAF packet table contains a negative count".to_string());
    }
    let packet_count_usize = usize::try_from(packet_count)
        .map_err(|_| "CAF packet count exceeds this platform".to_string())?;
    if packet_count_usize > MAX_CAF_MATERIALIZED_PACKETS {
        return Err(format!(
            "CAF packet count exceeds the {MAX_CAF_MATERIALIZED_PACKETS} packet index budget"
        ));
    }
    if packet_count_usize > data.len() - 24 {
        return Err("CAF packet count exceeds its variable-length table".to_string());
    }
    let mut packet_sizes = Vec::new();
    packet_sizes
        .try_reserve_exact(packet_count_usize)
        .map_err(|_| "CAF packet-size allocation failed".to_string())?;
    let mut position = 24usize;
    while position < data.len() {
        if packet_sizes.len() == packet_count_usize {
            return Err("CAF packet table has trailing bytes".to_string());
        }
        let mut value = 0u64;
        let mut complete = false;
        for _ in 0..10 {
            let byte = *data
                .get(position)
                .ok_or_else(|| "CAF packet size VLQ is truncated".to_string())?;
            position += 1;
            value = value
                .checked_mul(128)
                .and_then(|value| value.checked_add(u64::from(byte & 0x7f)))
                .ok_or_else(|| "CAF packet size VLQ overflows u64".to_string())?;
            if byte & 0x80 == 0 {
                complete = true;
                break;
            }
        }
        if !complete {
            return Err("CAF packet size VLQ exceeds 10 bytes".to_string());
        }
        let size = u32::try_from(value)
            .ok()
            .filter(|value| *value != 0 && *value <= MAX_CAF_PACKET_BYTES)
            .ok_or_else(|| format!("CAF packet size {value} exceeds its budget"))?;
        packet_sizes.push(size);
    }
    Ok(CafPacketTable {
        packet_count: packet_count as u64,
        valid_frames: valid_frames as u64,
        priming_frames: priming_frames as u32,
        remainder_frames: remainder_frames as u32,
        packet_sizes,
    })
}

fn resolve_packet_sizes(
    description: CafDescription,
    packet_table: Option<&CafPacketTable>,
    packet_bytes: u64,
) -> Result<Vec<u32>, String> {
    if description.frames_per_packet == 0 {
        return Err("CAF frames per packet is zero".to_string());
    }
    if description.bytes_per_packet == 0 {
        let table = packet_table
            .ok_or_else(|| "variable-packet CAF source is missing its pakt chunk".to_string())?;
        if table.packet_sizes.len() != table.packet_count as usize {
            return Err("CAF variable packet table omits packet sizes".to_string());
        }
        let described_bytes = table.packet_sizes.iter().try_fold(0u64, |total, size| {
            total
                .checked_add(u64::from(*size))
                .ok_or_else(|| "CAF packet byte count overflows u64".to_string())
        })?;
        if described_bytes != packet_bytes {
            return Err(format!(
                "CAF packet table describes {described_bytes} bytes, data contains {packet_bytes}"
            ));
        }
        return Ok(table.packet_sizes.clone());
    }
    if description.bytes_per_packet > MAX_CAF_PACKET_BYTES {
        return Err(format!(
            "CAF packet size {} exceeds its budget",
            description.bytes_per_packet
        ));
    }
    if packet_bytes % u64::from(description.bytes_per_packet) != 0 {
        return Err("CAF constant packet size does not divide the data chunk".to_string());
    }
    let packet_count = packet_bytes / u64::from(description.bytes_per_packet);
    let packet_count = usize::try_from(packet_count)
        .map_err(|_| "CAF packet count exceeds this platform".to_string())?;
    if packet_count > MAX_CAF_MATERIALIZED_PACKETS {
        return Err(format!(
            "CAF packet count exceeds the {MAX_CAF_MATERIALIZED_PACKETS} packet index budget"
        ));
    }
    if let Some(table) = packet_table {
        if table.packet_count != packet_count as u64 {
            return Err("CAF packet table count disagrees with constant packet data".to_string());
        }
        if !table.packet_sizes.is_empty() {
            return Err("CAF constant packet table unexpectedly contains packet sizes".to_string());
        }
    }
    let mut sizes = Vec::new();
    sizes
        .try_reserve_exact(packet_count)
        .map_err(|_| "CAF packet index allocation failed".to_string())?;
    sizes.resize(packet_count, description.bytes_per_packet);
    Ok(sizes)
}

type PcmFormat = (PcmEndianness, bool, bool, bool, bool, bool);

fn resolve_format(
    description: CafDescription,
) -> Result<(AudioCodec, String, Option<PcmFormat>), String> {
    let codec_id = String::from_utf8_lossy(&description.format_id).into_owned();
    match &description.format_id {
        b"lpcm" => {
            if description.bits_per_channel == 0 {
                return Err("CAF LPCM bits per channel is zero".to_string());
            }
            let is_float = description.format_flags & LPCM_IS_FLOAT != 0;
            if is_float && !matches!(description.bits_per_channel, 32 | 64) {
                return Err("CAF float PCM must use 32-bit or 64-bit samples".to_string());
            }
            if description.bytes_per_packet == 0 {
                return Err("CAF LPCM requires a constant packet size".to_string());
            }
            let bytes_per_frame = description.bytes_per_packet / description.frames_per_packet;
            if bytes_per_frame == 0
                || description.bytes_per_packet % description.frames_per_packet != 0
            {
                return Err("CAF LPCM packet geometry is not frame-aligned".to_string());
            }
            let minimum_frame_bytes = u32::from(description.bits_per_channel)
                .div_ceil(8)
                .checked_mul(u32::from(description.channels))
                .ok_or_else(|| "CAF LPCM frame size overflows u32".to_string())?;
            if bytes_per_frame < minimum_frame_bytes {
                return Err("CAF LPCM bytes per frame cannot hold its channels".to_string());
            }
            Ok((
                AudioCodec::Pcm,
                codec_id,
                Some((
                    if description.format_flags & LPCM_IS_BIG_ENDIAN != 0 {
                        PcmEndianness::Big
                    } else {
                        PcmEndianness::Little
                    },
                    is_float,
                    description.format_flags & LPCM_IS_SIGNED_INTEGER != 0,
                    description.format_flags & LPCM_IS_PACKED != 0,
                    description.format_flags & LPCM_IS_ALIGNED_HIGH != 0,
                    description.format_flags & LPCM_IS_NON_INTERLEAVED == 0,
                )),
            ))
        }
        b"alac" => Ok((AudioCodec::Alac, codec_id, None)),
        other => Err(format!(
            "unsupported CAF audio format {}",
            String::from_utf8_lossy(other)
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(kind);
        bytes.extend_from_slice(&(payload.len() as i64).to_be_bytes());
        bytes.extend_from_slice(payload);
        bytes
    }

    fn desc(
        rate: u32,
        format: &[u8; 4],
        flags: u32,
        bytes_per_packet: u32,
        frames_per_packet: u32,
        channels: u32,
        bits: u32,
    ) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&f64::from(rate).to_be_bytes());
        bytes.extend_from_slice(format);
        bytes.extend_from_slice(&flags.to_be_bytes());
        bytes.extend_from_slice(&bytes_per_packet.to_be_bytes());
        bytes.extend_from_slice(&frames_per_packet.to_be_bytes());
        bytes.extend_from_slice(&channels.to_be_bytes());
        bytes.extend_from_slice(&bits.to_be_bytes());
        bytes
    }

    fn pcm_caf(bits: u32, flags: u32, packet_count: usize, metadata_after_data: bool) -> Vec<u8> {
        let bytes_per_packet = bits.div_ceil(8) * 2;
        let description = chunk(
            b"desc",
            &desc(48_000, b"lpcm", flags, bytes_per_packet, 1, 2, bits),
        );
        let mut data = vec![0; 4 + bytes_per_packet as usize * packet_count];
        data[..4].copy_from_slice(&0u32.to_be_bytes());
        let data = chunk(b"data", &data);
        let channel_layout = chunk(b"chan", &[0, 1, 2, 3]);
        let mut file = b"caff\0\x01\0\0".to_vec();
        if metadata_after_data {
            file.extend_from_slice(&data);
            file.extend_from_slice(&description);
        } else {
            file.extend_from_slice(&description);
            file.extend_from_slice(&data);
        }
        file.extend_from_slice(&channel_layout);
        file
    }

    #[test]
    fn indexes_common_pcm_widths_endianness_and_float_flags() {
        for (bits, flags, endianness, float) in [
            (
                16,
                LPCM_IS_SIGNED_INTEGER | LPCM_IS_PACKED,
                PcmEndianness::Little,
                false,
            ),
            (
                24,
                LPCM_IS_BIG_ENDIAN | LPCM_IS_SIGNED_INTEGER | LPCM_IS_PACKED,
                PcmEndianness::Big,
                false,
            ),
            (
                32,
                LPCM_IS_SIGNED_INTEGER | LPCM_IS_PACKED,
                PcmEndianness::Little,
                false,
            ),
            (
                32,
                LPCM_IS_FLOAT | LPCM_IS_PACKED,
                PcmEndianness::Little,
                true,
            ),
            (
                64,
                LPCM_IS_FLOAT | LPCM_IS_BIG_ENDIAN | LPCM_IS_PACKED,
                PcmEndianness::Big,
                true,
            ),
        ] {
            for metadata_after_data in [false, true] {
                let file = pcm_caf(bits, flags, 3, metadata_after_data);
                let index = CafAudioIndex::from_file(&file).unwrap();
                assert_eq!(index.config.codec, AudioCodec::Pcm);
                assert_eq!(index.config.bits_per_sample, Some(bits as u8));
                assert_eq!(index.config.pcm_endianness, Some(endianness));
                assert_eq!(index.config.pcm_float, Some(float));
                assert_eq!(index.config.sample_count, Some(3));
                assert_eq!(index.valid_frames, 3);
                assert_eq!(index.channel_layout, vec![0, 1, 2, 3]);
                assert_eq!(index.packets.len(), 3);
                assert_eq!(index.packets[2].decode_time, 2);
            }
        }
    }

    #[test]
    fn accepts_apple_style_nonzero_data_edit_count() {
        let mut file = pcm_caf(32, LPCM_IS_FLOAT | LPCM_IS_PACKED, 3, false);
        let data_chunk = file
            .windows(4)
            .position(|bytes| bytes == b"data")
            .expect("generated CAF has a data chunk");
        let edit_count = data_chunk + 12;
        file[edit_count..edit_count + 4].copy_from_slice(&1u32.to_be_bytes());

        let index = CafAudioIndex::from_file(&file).unwrap();
        assert_eq!(index.config.codec, AudioCodec::Pcm);
        assert_eq!(index.packets.len(), 3);
    }

    #[test]
    fn indexes_variable_alac_packets_and_preserves_trim() {
        let description = desc(48_000, b"alac", 0, 0, 4096, 2, 24);
        let mut packet_table = Vec::new();
        packet_table.extend_from_slice(&2i64.to_be_bytes());
        packet_table.extend_from_slice(&7_000i64.to_be_bytes());
        packet_table.extend_from_slice(&128i32.to_be_bytes());
        packet_table.extend_from_slice(&1_064i32.to_be_bytes());
        packet_table.extend_from_slice(&[3, 5]);
        let mut data = vec![0; 12];
        data[..4].copy_from_slice(&0u32.to_be_bytes());
        let mut file = b"caff\0\x01\0\0".to_vec();
        file.extend_from_slice(&chunk(b"data", &data));
        file.extend_from_slice(&chunk(b"kuki", &[1, 2, 3]));
        file.extend_from_slice(&chunk(b"pakt", &packet_table));
        file.extend_from_slice(&chunk(b"desc", &description));

        let index = CafAudioIndex::from_file(&file).unwrap();
        assert_eq!(index.config.codec, AudioCodec::Alac);
        assert_eq!(index.config.codec_private, vec![1, 2, 3]);
        assert_eq!(index.valid_frames, 7_000);
        assert_eq!(index.priming_frames, 128);
        assert_eq!(index.remainder_frames, 1_064);
        assert_eq!(
            index
                .packets
                .iter()
                .map(|packet| packet.size)
                .collect::<Vec<_>>(),
            vec![3, 5]
        );
    }

    #[test]
    fn rejects_unbounded_metadata_and_malformed_counts() {
        let mut header = *b"pakt\0\0\0\0\0\0\0\0";
        header[4..12].copy_from_slice(&((MAX_CAF_PACKET_TABLE_BYTES + 1) as i64).to_be_bytes());
        let error = inspect_caf_chunk(&header, 8, MAX_CAF_PACKET_TABLE_BYTES + 21).unwrap_err();
        assert!(error.contains("metadata budget"));

        let mut table = vec![0; 24];
        table[..8].copy_from_slice(&i64::MAX.to_be_bytes());
        let error = parse_packet_table(&table).unwrap_err();
        assert!(error.contains("packet count"));
    }

    #[test]
    fn indexes_committed_alac_caf_through_generic_path() {
        let file = include_bytes!("../../testdata/alac/A_Tusk_is_used_to_make_costly_gifts.caf");
        let index = CafAudioIndex::from_file(file).unwrap();
        assert_eq!(index.config.codec, AudioCodec::Alac);
        assert!(!index.config.codec_private.is_empty());
        assert!(!index.packets.is_empty());
        assert!(index.valid_frames > 0);
        assert_eq!(index.config.sample_count, Some(index.packets.len() as u32));
    }
}
