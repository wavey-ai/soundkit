#[cfg(test)]
use alac::Reader as AlacReader;
use alac::{Decoder as CodecDecoder, StreamInfo};
use frame_header::{EncodingFlag, Endianness};
use soundkit::audio_types::AudioData;
#[cfg(test)]
use std::io::Cursor;

const MAX_ALAC_CHANNELS: u8 = 8;
const MAX_ALAC_FRAMES_PER_PACKET: u32 = 65_536;
const MAX_ALAC_PACKET_BYTES: usize = 16 * 1024 * 1024;
const MAX_CAF_COOKIE_BYTES: u64 = 1024 * 1024;
const MAX_CAF_PACKET_TABLE_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CafChunkRange {
    pub chunk_type: [u8; 4],
    pub payload_offset: u64,
    pub payload_size: u64,
    pub end: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CafPacketRange {
    pub offset: u64,
    pub size: u32,
}

/// Rust-owned packet index for a seekable ALAC CAF source.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CafAlacPacketIndex {
    pub magic_cookie: Vec<u8>,
    pub sample_rate: u32,
    pub channels: u8,
    pub bit_depth: u8,
    pub frames_per_packet: u32,
    pub valid_frames: u64,
    pub priming_frames: u32,
    pub remainder_frames: u32,
    pub packets: Vec<CafPacketRange>,
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

/// Inspect one CAF chunk header without reading its payload.
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
    let signed_size = i64::from_be_bytes(header[4..12].try_into().unwrap());
    let payload_offset = absolute_offset + 12;
    let payload_size = match signed_size {
        -1 => file_size - payload_offset,
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
        b"desc" => Some(32),
        b"kuki" => Some(MAX_CAF_COOKIE_BYTES),
        b"pakt" => Some(MAX_CAF_PACKET_TABLE_BYTES),
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

impl CafAlacPacketIndex {
    pub fn new(
        description: &[u8],
        magic_cookie: &[u8],
        packet_table: &[u8],
        data_payload_offset: u64,
        data_payload_size: u64,
    ) -> Result<Self, String> {
        if description.len() != 32 {
            return Err(format!(
                "CAF desc must contain 32 bytes, got {}",
                description.len()
            ));
        }
        let sample_rate_value = f64::from_be_bytes(description[..8].try_into().unwrap());
        if &description[8..12] != b"alac" {
            return Err("CAF audio description is not ALAC".to_string());
        }
        let bytes_per_packet = u32::from_be_bytes(description[16..20].try_into().unwrap());
        let frames_per_packet = u32::from_be_bytes(description[20..24].try_into().unwrap());
        let channels = u32::from_be_bytes(description[24..28].try_into().unwrap());
        let described_bits_per_channel =
            u32::from_be_bytes(description[28..32].try_into().unwrap());
        let sample_rate = finite_sample_rate(sample_rate_value)?;
        let channels = u8::try_from(channels)
            .ok()
            .filter(|value| (1..=MAX_ALAC_CHANNELS).contains(value))
            .ok_or_else(|| "CAF ALAC channel count exceeds the supported range".to_string())?;
        if data_payload_size < 4 {
            return Err("CAF data chunk is shorter than its edit count".to_string());
        }
        let packet_bytes = data_payload_size - 4;
        let packet_start = data_payload_offset
            .checked_add(4)
            .ok_or_else(|| "CAF packet offset overflow".to_string())?;
        let stream_info =
            StreamInfo::from_cookie(magic_cookie).map_err(|error| error.to_string())?;
        validate_stream_info(&stream_info)?;
        let bit_depth = stream_info.bit_depth();
        if stream_info.sample_rate() != sample_rate
            || stream_info.channels() != channels
            || (described_bits_per_channel != 0
                && u32::from(bit_depth) != described_bits_per_channel)
        {
            return Err("CAF desc and ALAC magic cookie disagree".to_string());
        }
        if frames_per_packet == 0 || frames_per_packet > MAX_ALAC_FRAMES_PER_PACKET {
            return Err(format!(
                "CAF ALAC frames-per-packet {frames_per_packet} exceeds the decoder budget"
            ));
        }

        let (valid_frames, priming_frames, remainder_frames, lengths) = if bytes_per_packet == 0 {
            parse_caf_packet_table(packet_table)?
        } else {
            if packet_bytes % u64::from(bytes_per_packet) != 0 {
                return Err("CAF constant packet size does not divide the data chunk".to_string());
            }
            let packet_count = packet_bytes / u64::from(bytes_per_packet);
            let valid_frames = packet_count
                .checked_mul(u64::from(frames_per_packet))
                .ok_or_else(|| "CAF valid frame count overflow".to_string())?;
            let packet_count = usize::try_from(packet_count)
                .map_err(|_| "CAF packet count exceeds this platform".to_string())?;
            (valid_frames, 0, 0, vec![bytes_per_packet; packet_count])
        };
        let mut offset = packet_start;
        let mut packets = Vec::with_capacity(lengths.len());
        for length in lengths {
            if length == 0 || length as usize > MAX_ALAC_PACKET_BYTES {
                return Err(format!("CAF ALAC packet size {length} exceeds its budget"));
            }
            packets.push(CafPacketRange {
                offset,
                size: length,
            });
            offset = offset
                .checked_add(u64::from(length))
                .ok_or_else(|| "CAF packet range overflow".to_string())?;
        }
        let expected_end = packet_start
            .checked_add(packet_bytes)
            .ok_or_else(|| "CAF data range overflow".to_string())?;
        if offset != expected_end {
            return Err(format!(
                "CAF packet table describes {} bytes, data contains {packet_bytes}",
                offset - packet_start
            ));
        }
        Ok(Self {
            magic_cookie: magic_cookie.to_vec(),
            sample_rate,
            channels,
            bit_depth,
            frames_per_packet,
            valid_frames,
            priming_frames,
            remainder_frames,
            packets,
        })
    }

    pub fn validate_packet_bytes(&self, index: usize, bytes: &[u8]) -> Result<(), String> {
        let packet = self
            .packets
            .get(index)
            .ok_or_else(|| format!("CAF packet index {index} is out of range"))?;
        if bytes.len() != packet.size as usize {
            return Err(format!(
                "CAF packet {index} expected {} bytes, got {}",
                packet.size,
                bytes.len()
            ));
        }
        Ok(())
    }
}

fn finite_sample_rate(value: f64) -> Result<u32, String> {
    if !value.is_finite() || value <= 0.0 || value > u32::MAX as f64 {
        return Err(format!("invalid CAF sample rate: {value}"));
    }
    Ok(value.round() as u32)
}

fn parse_caf_packet_table(data: &[u8]) -> Result<(u64, u32, u32, Vec<u32>), String> {
    if data.len() < 24 {
        return Err("CAF packet table is shorter than 24 bytes".to_string());
    }
    let packet_count = i64::from_be_bytes(data[..8].try_into().unwrap());
    let valid_frames = i64::from_be_bytes(data[8..16].try_into().unwrap());
    let priming = i32::from_be_bytes(data[16..20].try_into().unwrap());
    let remainder = i32::from_be_bytes(data[20..24].try_into().unwrap());
    if packet_count < 0 || valid_frames < 0 || priming < 0 || remainder < 0 {
        return Err("CAF packet table contains a negative count".to_string());
    }
    let packet_count = usize::try_from(packet_count)
        .map_err(|_| "CAF packet count exceeds this platform".to_string())?;
    if packet_count > data.len() - 24 {
        return Err("CAF packet count exceeds its variable-length table".to_string());
    }
    let mut position = 24usize;
    let mut lengths = Vec::with_capacity(packet_count);
    for _ in 0..packet_count {
        let mut value = 0u64;
        let mut complete = false;
        for _ in 0..10 {
            let byte = *data
                .get(position)
                .ok_or_else(|| "CAF packet size VLQ is truncated".to_string())?;
            position += 1;
            value = value
                .checked_shl(7)
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
        lengths.push(
            u32::try_from(value)
                .map_err(|_| "CAF packet size exceeds the u32 range".to_string())?,
        );
    }
    if position != data.len() {
        return Err("CAF packet table has trailing bytes".to_string());
    }
    Ok((
        valid_frames as u64,
        priming as u32,
        remainder as u32,
        lengths,
    ))
}

/// Bounded ALAC access-unit decoder.
///
/// A container demuxer supplies the magic cookie once and one encoded packet
/// per call. The decoder retains only codec state and one maximum-size PCM
/// packet. It never retains source-container bytes.
pub struct AlacPacketDecoder {
    decoder: CodecDecoder,
    scratch: Vec<i32>,
    sample_rate: u32,
    channels: u8,
    bit_depth: u8,
}

impl AlacPacketDecoder {
    pub fn new(magic_cookie: &[u8]) -> Result<Self, String> {
        // MP4 sample entries commonly expose the `alac` FullBox payload. Its
        // four version/flags bytes precede the 24-byte ALACSpecificConfig.
        let cookie = if magic_cookie.len() >= 28 && magic_cookie[..4] == [0, 0, 0, 0] {
            &magic_cookie[4..]
        } else {
            magic_cookie
        };
        let info = StreamInfo::from_cookie(cookie).map_err(|error| error.to_string())?;
        validate_stream_info(&info)?;
        let scratch_len = usize::try_from(info.max_samples_per_packet())
            .map_err(|_| "ALAC packet sample count exceeds this platform".to_string())?;
        let sample_rate = info.sample_rate();
        let channels = info.channels();
        let bit_depth = info.bit_depth();
        Ok(Self {
            decoder: CodecDecoder::new(info),
            scratch: vec![0; scratch_len],
            sample_rate,
            channels,
            bit_depth,
        })
    }

    pub fn decode_packet(&mut self, packet: &[u8]) -> Result<AudioData, String> {
        if packet.is_empty() {
            return Err("ALAC packet is empty".to_string());
        }
        if packet.len() > MAX_ALAC_PACKET_BYTES {
            return Err(format!(
                "ALAC packet exceeds the {MAX_ALAC_PACKET_BYTES} byte budget"
            ));
        }
        let samples = self
            .decoder
            .decode_packet(packet, &mut self.scratch)
            .map_err(|error| error.to_string())?;
        let mut pcm = Vec::with_capacity(
            samples
                .len()
                .checked_mul(usize::from(self.bit_depth.div_ceil(8)))
                .ok_or_else(|| "ALAC PCM output size overflow".to_string())?,
        );
        append_left_aligned_i32_samples(samples, self.bit_depth, &mut pcm)?;
        Ok(AudioData::new(
            self.bit_depth,
            self.channels,
            self.sample_rate,
            pcm,
            EncodingFlag::PCMSigned,
            Endianness::LittleEndian,
        ))
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn channels(&self) -> u8 {
        self.channels
    }

    pub fn bit_depth(&self) -> u8 {
        self.bit_depth
    }

    pub fn maximum_pcm_samples(&self) -> usize {
        self.scratch.len()
    }
}

fn validate_stream_info(info: &StreamInfo) -> Result<(), String> {
    if info.sample_rate() == 0 {
        return Err("ALAC stream reports a zero sample rate".to_string());
    }
    if !(1..=MAX_ALAC_CHANNELS).contains(&info.channels()) {
        return Err(format!(
            "ALAC channel count {} exceeds the supported range 1-{MAX_ALAC_CHANNELS}",
            info.channels()
        ));
    }
    if !matches!(info.bit_depth(), 16 | 24 | 32) {
        return Err(format!("Unsupported ALAC bit depth: {}", info.bit_depth()));
    }
    if info.max_frames_per_packet() == 0
        || info.max_frames_per_packet() > MAX_ALAC_FRAMES_PER_PACKET
    {
        return Err(format!(
            "ALAC frame length {} exceeds the 1-{MAX_ALAC_FRAMES_PER_PACKET} frame budget",
            info.max_frames_per_packet()
        ));
    }
    Ok(())
}

pub const SEEKABLE_ALAC_REQUIRED: &str =
    "ALAC containers require the seekable M4A/MP4 or CAF packet API";

/// Compatibility wrapper that rejects sequential ALAC containers.
///
/// M4A/MP4 and CAF can place required metadata after a large media extent.
/// Use the seekable packet index instead of retaining the complete source.
pub struct AlacDecoder;

impl AlacDecoder {
    pub fn new() -> Self {
        Self
    }

    pub fn init(&mut self) -> Result<(), String> {
        Err(SEEKABLE_ALAC_REQUIRED.to_string())
    }

    pub fn add(&mut self, _data: &[u8]) -> Result<Option<AudioData>, String> {
        Err(SEEKABLE_ALAC_REQUIRED.to_string())
    }
}

impl Default for AlacDecoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
fn decode_alac_container(data: &[u8]) -> Result<AudioData, String> {
    if data.is_empty() {
        return Err("ALAC input is empty".to_string());
    }

    let reader = AlacReader::new(Cursor::new(data.to_vec())).map_err(|error| format!("{error}"))?;
    let info = reader.stream_info();
    let sample_rate = info.sample_rate();
    let channels = info.channels();
    let bit_depth = info.bit_depth();
    let max_samples = info.max_samples_per_packet() as usize;

    if channels == 0 {
        return Err("ALAC stream reports zero channels".to_string());
    }
    if !matches!(bit_depth, 16 | 24 | 32) {
        return Err(format!("Unsupported ALAC bit depth: {bit_depth}"));
    }

    let mut packets = reader.into_packets::<i32>();
    let mut packet_samples = vec![0i32; max_samples];
    let mut pcm = Vec::new();

    while let Some(samples) = packets
        .next_into(&mut packet_samples)
        .map_err(|error| format!("{error}"))?
    {
        append_left_aligned_i32_samples(samples, bit_depth, &mut pcm)?;
    }

    Ok(AudioData::new(
        bit_depth,
        channels,
        sample_rate,
        pcm,
        EncodingFlag::PCMSigned,
        Endianness::LittleEndian,
    ))
}

fn append_left_aligned_i32_samples(
    samples: &[i32],
    bit_depth: u8,
    out: &mut Vec<u8>,
) -> Result<(), String> {
    let shift = 32u8
        .checked_sub(bit_depth)
        .ok_or_else(|| format!("Invalid ALAC bit depth: {bit_depth}"))?;

    match bit_depth {
        16 => {
            out.reserve(samples.len() * 2);
            for &sample in samples {
                let right_aligned = sample >> shift;
                out.extend_from_slice(&(right_aligned as i16).to_le_bytes());
            }
        }
        24 => {
            out.reserve(samples.len() * 3);
            for &sample in samples {
                let right_aligned = sample >> shift;
                out.extend_from_slice(&right_aligned.to_le_bytes()[..3]);
            }
        }
        32 => {
            out.reserve(samples.len() * 4);
            for &sample in samples {
                out.extend_from_slice(&sample.to_le_bytes());
            }
        }
        _ => return Err(format!("Unsupported ALAC bit depth: {bit_depth}")),
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use soundkit_audio_demux::Mp4MediaIndex;
    use std::fs;
    use std::path::PathBuf;
    use std::process::Command;

    fn testdata_path(file: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("testdata")
            .join(file)
    }

    fn golden_path(file: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("golden")
            .join(file)
    }

    fn caf_index_from_file(fixture: &[u8]) -> CafAlacPacketIndex {
        let file_size = fixture.len() as u64;
        validate_caf_file_header(&fixture[..8], file_size).unwrap();
        let mut offset = 8u64;
        let mut description = None;
        let mut magic_cookie = None;
        let mut packet_table = None;
        let mut data = None;
        while offset < file_size {
            let header_end = usize::try_from(offset + 12).unwrap();
            let range = inspect_caf_chunk(
                &fixture[usize::try_from(offset).unwrap()..header_end],
                offset,
                file_size,
            )
            .unwrap();
            let payload = &fixture[usize::try_from(range.payload_offset).unwrap()
                ..usize::try_from(range.end).unwrap()];
            match &range.chunk_type {
                b"desc" => description = Some(payload.to_vec()),
                b"kuki" => magic_cookie = Some(payload.to_vec()),
                b"pakt" => packet_table = Some(payload.to_vec()),
                b"data" => data = Some((range.payload_offset, range.payload_size)),
                _ => {}
            }
            offset = range.end;
        }
        let (data_offset, data_size) = data.expect("CAF data chunk");
        CafAlacPacketIndex::new(
            &description.expect("CAF desc chunk"),
            &magic_cookie.expect("CAF kuki chunk"),
            &packet_table.expect("CAF pakt chunk"),
            data_offset,
            data_size,
        )
        .unwrap()
    }

    #[test]
    fn caf_packet_index_streams_exact_pcm_with_bounded_state() {
        let fixture = fs::read(testdata_path(
            "alac/A_Tusk_is_used_to_make_costly_gifts.caf",
        ))
        .unwrap();
        let index = caf_index_from_file(&fixture);
        assert_eq!(index.sample_rate, 8_000);
        assert_eq!(index.channels, 1);
        assert_eq!(index.bit_depth, 16);
        assert_eq!(index.valid_frames, 23_680);
        assert_eq!(index.packets.len(), 6);

        let mut decoder = AlacPacketDecoder::new(&index.magic_cookie).unwrap();
        assert!(decoder.maximum_pcm_samples() <= 65_536 * 8);
        let mut streamed_pcm = Vec::new();
        for (packet_index, packet) in index.packets.iter().enumerate() {
            let start = usize::try_from(packet.offset).unwrap();
            let end = start + packet.size as usize;
            let encoded = &fixture[start..end];
            index.validate_packet_bytes(packet_index, encoded).unwrap();
            streamed_pcm.extend_from_slice(decoder.decode_packet(encoded).unwrap().data());
        }

        let whole = decode_alac_container(&fixture).unwrap();
        assert_eq!(streamed_pcm.as_slice(), whole.data().as_slice());
        assert!(
            index
                .packets
                .iter()
                .map(|packet| packet.size as usize)
                .max()
                .unwrap()
                < fixture.len()
        );
    }

    #[test]
    fn caf_range_validation_rejects_unbounded_or_inconsistent_metadata() {
        let mut chunk = [0u8; 12];
        chunk[..4].copy_from_slice(b"pakt");
        chunk[4..].copy_from_slice(&(MAX_CAF_PACKET_TABLE_BYTES as i64 + 1).to_be_bytes());
        let error = inspect_caf_chunk(&chunk, 8, MAX_CAF_PACKET_TABLE_BYTES + 21).unwrap_err();
        assert!(error.contains("metadata budget"));

        chunk[4..].copy_from_slice(&24i64.to_be_bytes());
        let error = inspect_caf_chunk(&chunk, 8, 43).unwrap_err();
        assert!(error.contains("source length"));

        let fixture = fs::read(testdata_path(
            "alac/A_Tusk_is_used_to_make_costly_gifts.caf",
        ))
        .unwrap();
        let index = caf_index_from_file(&fixture);
        let mut truncated_table = Vec::new();
        truncated_table.extend_from_slice(&2i64.to_be_bytes());
        truncated_table.extend_from_slice(&0i64.to_be_bytes());
        truncated_table.extend_from_slice(&0i32.to_be_bytes());
        truncated_table.extend_from_slice(&0i32.to_be_bytes());
        truncated_table.push(1);
        let error = CafAlacPacketIndex::new(
            &fixture[20..52],
            &index.magic_cookie,
            &truncated_table,
            index.packets[0].offset - 4,
            5,
        )
        .unwrap_err();
        assert!(error.contains("packet count exceeds"));
    }

    #[test]
    fn caf_sparse_large_data_chunk_needs_only_its_header() {
        let file_size = 8 * 1024 * 1024 * 1024u64;
        let mut chunk = [0u8; 12];
        chunk[..4].copy_from_slice(b"data");
        chunk[4..].copy_from_slice(&(-1i64).to_be_bytes());
        let range = inspect_caf_chunk(&chunk, 8, file_size).unwrap();
        assert_eq!(range.payload_offset, 20);
        assert_eq!(range.payload_size, file_size - 20);
        assert_eq!(range.end, file_size);
    }

    #[test]
    fn packet_decoder_emits_each_indexed_packet_with_bounded_state() {
        let fixture = fs::read(testdata_path(
            "alac/A_Tusk_is_used_to_make_costly_gifts.m4a",
        ))
        .unwrap();
        let index = Mp4MediaIndex::from_file(&fixture).unwrap();
        let track = index
            .tracks
            .iter()
            .find(|track| track.codec == "alac")
            .expect("ALAC track");
        let mut decoder = AlacPacketDecoder::new(&track.codec_private).unwrap();
        assert!(decoder.maximum_pcm_samples() <= 65_536 * 8);

        let mut streamed_pcm = Vec::new();
        let mut packet_count = 0usize;
        for (sample_index, sample) in index.samples.iter().enumerate() {
            if sample.track_id != track.track_id {
                continue;
            }
            let start = sample.absolute_offset as usize;
            let end = start + sample.size as usize;
            let packet = index
                .packet_from_sample_bytes(sample_index, &fixture[start..end])
                .unwrap();
            let frame = decoder.decode_packet(&packet.data).unwrap();
            assert!(!frame.data().is_empty());
            streamed_pcm.extend_from_slice(frame.data());
            packet_count += 1;
        }

        let whole = decode_alac_container(&fixture).unwrap();
        assert!(packet_count > 1);
        assert_eq!(streamed_pcm.as_slice(), whole.data().as_slice());
    }

    #[test]
    fn packet_decoder_rejects_unbounded_stream_contracts() {
        let fixture = fs::read(testdata_path(
            "alac/A_Tusk_is_used_to_make_costly_gifts.m4a",
        ))
        .unwrap();
        let index = Mp4MediaIndex::from_file(&fixture).unwrap();
        let mut cookie = index
            .tracks
            .iter()
            .find(|track| track.codec == "alac")
            .unwrap()
            .codec_private
            .clone();
        let config_start = if cookie.get(4..8) == Some(b"alac") {
            12
        } else if cookie.get(..4) == Some(&[0, 0, 0, 0]) {
            4
        } else {
            0
        };
        cookie[config_start..config_start + 4].copy_from_slice(&u32::MAX.to_be_bytes());
        let error = match AlacPacketDecoder::new(&cookie) {
            Ok(_) => panic!("unbounded ALAC stream contract was accepted"),
            Err(error) => error,
        };
        assert!(error.contains("frame budget"));
    }

    #[test]
    #[ignore = "regenerates the committed ALAC fixture using ffmpeg"]
    fn generate_alac_fixture_with_ffmpeg() {
        let input = testdata_path("linear16_8/A_Tusk_is_used_to_make_costly_gifts.s16le");
        let output = testdata_path("alac/A_Tusk_is_used_to_make_costly_gifts.m4a");
        fs::create_dir_all(output.parent().unwrap()).unwrap();
        let status = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "s16le",
                "-ar",
                "8000",
                "-ac",
                "1",
                "-i",
            ])
            .arg(&input)
            .args(["-c:a", "alac", "-f", "ipod"])
            .arg(&output)
            .status()
            .unwrap();
        assert!(status.success());
    }

    #[test]
    fn sequential_container_decoder_requires_seekable_ranges() {
        let fixture = fs::read(testdata_path(
            "alac/A_Tusk_is_used_to_make_costly_gifts.m4a",
        ))
        .unwrap();
        assert!(!fixture.is_empty(), "ALAC fixture missing or empty");
        let mut decoder = AlacDecoder::new();
        assert_eq!(decoder.init().unwrap_err(), SEEKABLE_ALAC_REQUIRED);
        assert_eq!(
            decoder.add(&fixture[..997]).unwrap_err(),
            SEEKABLE_ALAC_REQUIRED
        );
    }

    #[test]
    fn decode_alac_fixture_and_write_golden_wav() {
        let fixture = fs::read(testdata_path(
            "alac/A_Tusk_is_used_to_make_costly_gifts.m4a",
        ))
        .unwrap();
        let audio = decode_alac_container(&fixture).unwrap();
        assert_eq!(audio.bits_per_sample(), 16);
        assert_eq!(audio.channel_count(), 1);
        assert_eq!(audio.sampling_rate(), 8_000);
        assert!(audio
            .data()
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
            .any(|sample| sample != 0));

        let samples: Vec<i16> = audio
            .data()
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
            .collect();
        let wav = soundkit::wav::generate_wav_buffer(
            &soundkit::audio_types::PcmData::I16(vec![samples]),
            8_000,
        )
        .unwrap();
        let output_path = golden_path("alac/A_Tusk_is_used_to_make_costly_gifts.decoded.wav");
        fs::create_dir_all(output_path.parent().unwrap()).unwrap();
        fs::write(output_path, wav).unwrap();
    }

    #[test]
    fn native_decode_matches_ffmpeg_pcm() {
        let input = testdata_path("alac/A_Tusk_is_used_to_make_costly_gifts.m4a");
        let ffmpeg_pcm = std::env::temp_dir().join("soundkit-alac-ffmpeg.s16le");
        let status = Command::new("ffmpeg")
            .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
            .arg(&input)
            .args(["-f", "s16le", "-acodec", "pcm_s16le"])
            .arg(&ffmpeg_pcm)
            .status()
            .unwrap();
        assert!(status.success());

        let fixture = fs::read(input).unwrap();
        let audio = decode_alac_container(&fixture).unwrap();
        assert_eq!(audio.data(), &fs::read(ffmpeg_pcm).unwrap());
    }
}
