//! SoundKit v2 frame-stream encoding and sidecar-index utilities.
//!
//! A stream is a concatenation of [`frame_header::FrameHeaderV2`] frames, one
//! per encoded packet. A sidecar index ([`SoundKitIndex`]) maps a decoded PCM
//! frame to the byte offset of the packet that starts at or before it, so a
//! player can range-fetch and seek without holding the whole stream.
//!
//! Two packet codecs are emitted from one interleaved i16 PCM pass:
//!
//! - Opus (`soundkit-opus`), the lossy house profile.
//! - FLAC (`soundkit-flac`), the lossless copy.
//!
//! The Opus-only entry points are kept for callers that want one stream.

use frame_header::{EncodingFlag, Endianness, FrameHeaderV2};
use soundkit_flac::{FlacFrameConfig, FlacFrameEncoder, FlacProfile};
use soundkit_opus::{Encoder as OpusEncoder, CELT_FRAME_SIZES_48K};

pub const SOUNDKIT_INDEX_MAGIC: [u8; 8] = *b"SKIDX2\0\0";
pub const SOUNDKIT_INDEX_VERSION: u16 = 1;
pub const SOUNDKIT_INDEX_ENTRY_BYTES: u16 = 16;
pub const SOUNDKIT_INDEX_HEADER_BYTES: usize = 32;

/// The shortest FLAC block the encoder will write.
const FLAC_MIN_BLOCK_SIZE: usize = 32;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SoundKitIndexEntry {
    pub byte_offset: u64,
    pub start_frame: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SoundKitIndex {
    pub timescale: u32,
    pub duration_frames: u64,
    pub entries: Vec<SoundKitIndexEntry>,
}

/// Options for the Opus-only entry point.
#[derive(Debug, Clone)]
pub struct PcmOpusStreamOptions {
    pub sample_rate: u32,
    pub channels: u8,
    pub frame_size: u32,
    pub bitrate: u32,
    pub start_pts: u64,
    pub include_packet_crc32: bool,
}

impl Default for PcmOpusStreamOptions {
    fn default() -> Self {
        Self {
            sample_rate: 48_000,
            channels: 2,
            frame_size: 960,
            bitrate: 128_000,
            start_pts: 0,
            include_packet_crc32: true,
        }
    }
}

/// Options for encoding both streams from one PCM pass.
#[derive(Debug, Clone)]
pub struct PcmI16StreamOptions {
    pub sample_rate: u32,
    pub channels: u8,
    pub bitrate: u32,
    /// Opus packet length in PCM frames. One of [`CELT_FRAME_SIZES_48K`].
    pub opus_frame_size: u32,
    pub start_pts: u64,
    pub include_packet_crc32: bool,
}

impl Default for PcmI16StreamOptions {
    fn default() -> Self {
        Self {
            sample_rate: 48_000,
            channels: 2,
            bitrate: 128_000,
            opus_frame_size: 960,
            start_pts: 0,
            include_packet_crc32: true,
        }
    }
}

/// One encoded codec stream and the sidecar index that addresses it.
#[derive(Debug, Clone)]
pub struct EncodedSoundKitStream {
    pub stream: Vec<u8>,
    pub index: SoundKitIndex,
    pub packet_count: u64,
}

impl EncodedSoundKitStream {
    pub fn index_bytes(&self) -> Result<Vec<u8>, String> {
        encode_soundkit_index(&self.index)
    }
}

/// The Opus stream and the FLAC stream cut from one PCM pass.
#[derive(Debug, Clone)]
pub struct EncodedSoundKitStreams {
    pub opus: EncodedSoundKitStream,
    pub flac: EncodedSoundKitStream,
    /// The stereo, three-band waveform sidecar — `soundkit_visuals`'s bytes —
    /// summed from the same PCM pass, so a player draws the side without
    /// reading the audio again. Empty when the PCM was one channel.
    pub visuals: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamCodec {
    Opus,
    Flac,
}

/// One framed packet emitted by [`StreamEncoder`].
#[derive(Debug, Clone)]
pub struct StreamPacket {
    pub codec: StreamCodec,
    pub bytes: Vec<u8>,
    pub start_frame: u64,
    pub frame_count: u32,
}

/// Incremental SoundKit v2 stream encoder.
///
/// PCM is pushed in arbitrary chunks and complete packets are emitted as they
/// fill, so a caller that decodes a container incrementally never retains the
/// whole programme. Opus is 48 kHz, 1–2 channels. FLAC carries whatever
/// geometry the caller declares, in interleaved 24-bit samples, and is only
/// encoded when the encoder was opened for lossless preservation.
pub struct StreamEncoder {
    sample_rate: u32,
    opus_channels: usize,
    opus_frame_size: usize,
    include_crc: bool,
    start_pts: u64,
    opus: OpusEncoder,
    opus_frame: Vec<i16>,
    opus_stream: Vec<u8>,
    opus_entries: Vec<SoundKitIndexEntry>,
    opus_frames: u64,
    preserve_lossless: bool,
    flac: Option<FlacFrameEncoder>,
    flac_frame_size: usize,
    flac_sample_rate: u32,
    flac_channels: u8,
    flac_frame: Vec<i32>,
    flac_stream: Vec<u8>,
    flac_entries: Vec<SoundKitIndexEntry>,
    flac_frames: u64,
}

impl StreamEncoder {
    pub fn new(preserve_lossless: bool, options: &PcmOpusStreamOptions) -> Result<Self, String> {
        validate_pcm_opus_options(options)?;
        let mut opus = OpusEncoder::try_new(
            options.sample_rate,
            16,
            options.channels as u32,
            options.frame_size,
            options.bitrate,
        )?;
        opus.init()?;
        Ok(Self {
            sample_rate: options.sample_rate,
            opus_channels: options.channels as usize,
            opus_frame_size: options.frame_size as usize,
            include_crc: options.include_packet_crc32,
            start_pts: options.start_pts,
            opus,
            opus_frame: Vec::with_capacity(options.frame_size as usize * options.channels as usize),
            opus_stream: Vec::new(),
            opus_entries: Vec::new(),
            opus_frames: 0,
            preserve_lossless,
            flac: None,
            flac_frame_size: 0,
            flac_sample_rate: 0,
            flac_channels: 0,
            flac_frame: Vec::new(),
            flac_stream: Vec::new(),
            flac_entries: Vec::new(),
            flac_frames: 0,
        })
    }

    pub const fn preserve_lossless(&self) -> bool {
        self.preserve_lossless
    }

    /// Push interleaved 48 kHz PCM into the Opus stream.
    pub fn push_opus_i16(&mut self, pcm: &[i16]) -> Result<Vec<StreamPacket>, String> {
        let channels = self.opus_channels;
        if pcm.len() % channels != 0 {
            return Err("Opus PCM sample count must be divisible by channel count".to_string());
        }
        let mut packets = Vec::new();
        for frame in pcm.chunks_exact(channels) {
            self.opus_frame.extend_from_slice(frame);
            if self.opus_frame.len() == self.opus_frame_size * channels {
                self.emit_opus(&mut packets, false)?;
            }
        }
        Ok(packets)
    }

    /// Declare the FLAC geometry. The block length defaults to the low-latency
    /// one for the rate; a caller that knows the total length can pass the
    /// exact size so the final block lands on a boundary.
    pub fn ensure_flac_geometry(
        &mut self,
        sample_rate: u32,
        channels: u8,
        frame_size: Option<usize>,
    ) -> Result<(), String> {
        if sample_rate == 0 || channels == 0 || channels > 8 {
            return Err(format!(
                "FLAC preservation has unsupported PCM geometry {sample_rate} Hz/{channels} ch"
            ));
        }
        if self.flac.is_some() {
            if self.flac_sample_rate != sample_rate || self.flac_channels != channels {
                return Err(format!(
                    "FLAC geometry changed from {} Hz/{} ch to {sample_rate} Hz/{channels} ch",
                    self.flac_sample_rate, self.flac_channels
                ));
            }
            return Ok(());
        }
        let size = frame_size.unwrap_or_else(|| low_latency_flac_frame_size(sample_rate));
        if size < FLAC_MIN_BLOCK_SIZE || size > 65_535 {
            return Err(format!("FLAC block length {size} is out of range"));
        }
        let config = FlacFrameConfig::new(
            sample_rate,
            u16::from(channels),
            24,
            size as u32,
            FlacProfile::Balanced,
        )
        .map_err(|error| error.to_string())?;
        self.flac = Some(FlacFrameEncoder::new(config).map_err(|error| error.to_string())?);
        self.flac_frame_size = size;
        self.flac_sample_rate = sample_rate;
        self.flac_channels = channels;
        self.flac_frame = Vec::with_capacity((size + FLAC_MIN_BLOCK_SIZE) * channels as usize);
        Ok(())
    }

    /// Push interleaved 24-bit samples into the FLAC stream.
    pub fn push_flac_i32(&mut self, pcm: &[i32]) -> Result<Vec<StreamPacket>, String> {
        let channels = self.flac_channels as usize;
        if self.flac.is_none() || channels == 0 {
            return Err("FLAC geometry has not been declared".to_string());
        }
        if pcm.len() % channels != 0 {
            return Err("FLAC PCM sample count must be divisible by channel count".to_string());
        }
        self.flac_frame.extend_from_slice(pcm);
        let mut packets = Vec::new();
        while self.flac_frame.len() / channels >= self.flac_frame_size + FLAC_MIN_BLOCK_SIZE {
            self.emit_flac(&mut packets, self.flac_frame_size)?;
        }
        Ok(packets)
    }

    /// Flush the final partial Opus block, zero-padding the encoder input.
    pub fn finish_opus(&mut self) -> Result<Vec<StreamPacket>, String> {
        let mut packets = Vec::new();
        if !self.opus_frame.is_empty() {
            self.emit_opus(&mut packets, true)?;
        }
        Ok(packets)
    }

    /// Flush the final partial FLAC block, keeping it at or above the minimum.
    pub fn finish_flac(&mut self) -> Result<Vec<StreamPacket>, String> {
        let mut packets = Vec::new();
        if self.flac.is_none() {
            return Ok(packets);
        }
        let channels = self.flac_channels as usize;
        let mut remaining = self.flac_frame.len() / channels;
        if remaining == 0 {
            return Ok(packets);
        }
        if self.flac_frames == 0 && remaining < FLAC_MIN_BLOCK_SIZE {
            return Err(format!(
                "FLAC requires at least {FLAC_MIN_BLOCK_SIZE} PCM frames"
            ));
        }
        while remaining > self.flac_frame_size {
            let after_full = remaining - self.flac_frame_size;
            let count = if after_full < FLAC_MIN_BLOCK_SIZE {
                remaining - FLAC_MIN_BLOCK_SIZE
            } else {
                self.flac_frame_size
            };
            self.emit_flac(&mut packets, count)?;
            remaining = self.flac_frame.len() / channels;
        }
        if remaining > 0 {
            self.emit_flac(&mut packets, remaining)?;
        }
        Ok(packets)
    }

    pub fn opus_stream(&self) -> &[u8] {
        &self.opus_stream
    }

    pub fn flac_stream(&self) -> &[u8] {
        &self.flac_stream
    }

    pub fn opus_entries(&self) -> &[SoundKitIndexEntry] {
        &self.opus_entries
    }

    pub fn flac_entries(&self) -> &[SoundKitIndexEntry] {
        &self.flac_entries
    }

    pub fn opus_packet_count(&self) -> u64 {
        self.opus_entries.len() as u64
    }

    pub fn flac_packet_count(&self) -> u64 {
        self.flac_entries.len() as u64
    }

    pub fn opus_frames(&self) -> u64 {
        self.opus_frames
    }

    pub fn flac_frames(&self) -> u64 {
        self.flac_frames
    }

    pub fn flac_sample_rate(&self) -> u32 {
        self.flac_sample_rate
    }

    pub fn flac_channels(&self) -> u8 {
        self.flac_channels
    }

    fn opus_index(&self) -> SoundKitIndex {
        SoundKitIndex {
            timescale: self.sample_rate,
            duration_frames: self.start_pts + self.opus_frames,
            entries: self.opus_entries.clone(),
        }
    }

    fn flac_index(&self) -> SoundKitIndex {
        SoundKitIndex {
            timescale: self.flac_sample_rate,
            duration_frames: self.start_pts + self.flac_frames,
            entries: self.flac_entries.clone(),
        }
    }

    fn emit_opus(
        &mut self,
        packets: &mut Vec<StreamPacket>,
        final_packet: bool,
    ) -> Result<(), String> {
        let channels = self.opus_channels;
        let frame_count = (self.opus_frame.len() / channels) as u32;
        if frame_count == 0 {
            return Ok(());
        }
        if (frame_count as usize) < self.opus_frame_size {
            if !final_packet {
                return Err("short Opus block appeared before EOF".to_string());
            }
            self.opus_frame.resize(self.opus_frame_size * channels, 0);
        }
        let mut output = vec![0u8; 4096];
        let written = self.opus.encode_i16(&self.opus_frame, &mut output)?;
        if written == 0 {
            return Err("Opus emitted an empty packet".to_string());
        }
        output.truncate(written);
        let start_frame = self.start_pts + self.opus_frames;
        let bytes = frame_soundkit_packet(
            EncodingFlag::Opus,
            output,
            frame_count,
            self.sample_rate,
            channels as u8,
            0,
            self.opus_entries.len() as u64,
            start_frame,
            self.include_crc,
        )?;
        self.opus_entries.push(SoundKitIndexEntry {
            byte_offset: self.opus_stream.len() as u64,
            start_frame,
        });
        self.opus_stream.extend_from_slice(&bytes);
        packets.push(StreamPacket {
            codec: StreamCodec::Opus,
            bytes,
            start_frame,
            frame_count,
        });
        self.opus_frames += u64::from(frame_count);
        self.opus_frame.clear();
        Ok(())
    }

    fn emit_flac(
        &mut self,
        packets: &mut Vec<StreamPacket>,
        frame_count: usize,
    ) -> Result<(), String> {
        let channels = self.flac_channels as usize;
        if !(FLAC_MIN_BLOCK_SIZE..=self.flac_frame_size).contains(&frame_count) {
            return Err(format!(
                "FLAC block has {frame_count} frames, expected {FLAC_MIN_BLOCK_SIZE}..={}",
                self.flac_frame_size
            ));
        }
        let sample_count = frame_count * channels;
        if self.flac_frame.len() < sample_count {
            return Err("FLAC block is incomplete".to_string());
        }
        let mut output = Vec::with_capacity(sample_count.saturating_mul(4).saturating_add(64));
        self.flac
            .as_mut()
            .ok_or_else(|| "FLAC encoder is unavailable".to_string())?
            .encode_i32_block_into(&self.flac_frame[..sample_count], &mut output)
            .map_err(|error| error.to_string())?;
        if output.is_empty() {
            return Err("FLAC emitted an empty packet".to_string());
        }
        let start_frame = self.start_pts + self.flac_frames;
        let bytes = frame_soundkit_packet(
            EncodingFlag::FLAC,
            output,
            frame_count as u32,
            self.flac_sample_rate,
            self.flac_channels,
            24,
            self.flac_entries.len() as u64,
            start_frame,
            self.include_crc,
        )?;
        self.flac_entries.push(SoundKitIndexEntry {
            byte_offset: self.flac_stream.len() as u64,
            start_frame,
        });
        self.flac_stream.extend_from_slice(&bytes);
        packets.push(StreamPacket {
            codec: StreamCodec::Flac,
            bytes,
            start_frame,
            frame_count: frame_count as u32,
        });
        self.flac_frames += frame_count as u64;
        self.flac_frame.copy_within(sample_count.., 0);
        self.flac_frame.truncate(self.flac_frame.len() - sample_count);
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn frame_soundkit_packet(
    encoding: EncodingFlag,
    payload: Vec<u8>,
    frame_count: u32,
    sample_rate: u32,
    channels: u8,
    bits_per_sample: u8,
    packet_sequence: u64,
    pts: u64,
    include_crc: bool,
) -> Result<Vec<u8>, String> {
    if payload.is_empty() || frame_count == 0 {
        return Err("A SoundKit stream packet is empty".to_string());
    }
    let mut header = FrameHeaderV2::new(
        encoding,
        payload.len() as u32,
        frame_count,
        sample_rate,
        channels,
        bits_per_sample,
        Endianness::LittleEndian,
        Some(packet_sequence),
        Some(pts),
        None,
    )?;
    if include_crc {
        header = header.with_packet_crc32(&payload)?;
    }
    let mut framed = Vec::with_capacity(header.size() + payload.len());
    header
        .encode(&mut framed)
        .map_err(|error| error.to_string())?;
    framed.extend_from_slice(&payload);
    Ok(framed)
}

/// Encodes interleaved i16 PCM into an Opus stream and a lossless FLAC stream.
pub fn encode_interleaved_i16_to_soundkit_streams(
    pcm: &[i16],
    options: PcmI16StreamOptions,
) -> Result<EncodedSoundKitStreams, String> {
    validate_pcm_stream_options(&options)?;
    let opus = encode_interleaved_i16_to_opus_soundkit_stream(
        pcm,
        PcmOpusStreamOptions {
            sample_rate: options.sample_rate,
            channels: options.channels,
            frame_size: options.opus_frame_size,
            bitrate: options.bitrate,
            start_pts: options.start_pts,
            include_packet_crc32: options.include_packet_crc32,
        },
    )?;
    let flac = encode_interleaved_i16_to_flac_soundkit_stream(pcm, &options)?;
    let visuals = soundkit_visuals::compute_waveform(
        pcm,
        options.sample_rate,
        options.channels,
        &soundkit_visuals::WaveformOptions::default(),
    )
    .map(|waveform| waveform.encode())
    .unwrap_or_default();
    Ok(EncodedSoundKitStreams { opus, flac, visuals })
}

/// Encodes interleaved i16 PCM into a SoundKit v2 Opus stream.
pub fn encode_interleaved_i16_to_opus_soundkit_stream(
    pcm: &[i16],
    options: PcmOpusStreamOptions,
) -> Result<EncodedSoundKitStream, String> {
    let channels = options.channels as usize;
    if pcm.len() % channels != 0 {
        return Err("PCM sample count must be divisible by channel count".to_string());
    }
    let mut encoder = StreamEncoder::new(false, &options)?;
    encoder.push_opus_i16(pcm)?;
    encoder.finish_opus()?;
    Ok(EncodedSoundKitStream {
        stream: encoder.opus_stream().to_vec(),
        index: encoder.opus_index(),
        packet_count: encoder.opus_packet_count(),
    })
}

/// Encodes interleaved i16 PCM into a lossless SoundKit v2 FLAC stream.
///
/// The i16 samples are widened to 24-bit FLAC blocks; every non-final block is
/// the configured frame length and the final block may be shorter.
fn encode_interleaved_i16_to_flac_soundkit_stream(
    pcm: &[i16],
    options: &PcmI16StreamOptions,
) -> Result<EncodedSoundKitStream, String> {
    let channels = options.channels as usize;
    if pcm.len() % channels != 0 {
        return Err("PCM sample count must be divisible by channel count".to_string());
    }
    let total_frames = pcm.len() / channels;
    let frame_size = flac_frame_size(total_frames, low_latency_flac_frame_size(options.sample_rate))?;
    let opus_options = PcmOpusStreamOptions {
        sample_rate: options.sample_rate,
        channels: options.channels,
        frame_size: options.opus_frame_size,
        bitrate: options.bitrate,
        start_pts: options.start_pts,
        include_packet_crc32: options.include_packet_crc32,
    };
    let mut encoder = StreamEncoder::new(true, &opus_options)?;
    encoder.ensure_flac_geometry(options.sample_rate, options.channels, Some(frame_size))?;
    let samples: Vec<i32> = pcm.iter().map(|&sample| flac_sample(sample)).collect();
    encoder.push_flac_i32(&samples)?;
    encoder.finish_flac()?;
    Ok(EncodedSoundKitStream {
        stream: encoder.flac_stream().to_vec(),
        index: encoder.flac_index(),
        packet_count: encoder.flac_packet_count(),
    })
}

/// Widen an i16 sample to the 24-bit range the FLAC stream carries.
fn flac_sample(sample: i16) -> i32 {
    (f64::from(sample) * 8_388_607.0 / 32_768.0).round() as i32
}

/// The low-latency FLAC block length for a rate: roughly one 200th of a second,
/// clamped to the legal block range.
fn low_latency_flac_frame_size(sample_rate: u32) -> usize {
    ((sample_rate as usize).saturating_add(100) / 200).clamp(FLAC_MIN_BLOCK_SIZE, 65_535)
}

/// The largest block length no longer than `requested` whose final block is
/// either exact or still at least the minimum block size.
fn flac_frame_size(total_frames: usize, requested: usize) -> Result<usize, String> {
    let maximum = 32_767usize.min(total_frames).min(requested);
    if maximum < FLAC_MIN_BLOCK_SIZE {
        return Err(format!(
            "FLAC stream requires at least {FLAC_MIN_BLOCK_SIZE} PCM frames"
        ));
    }
    (FLAC_MIN_BLOCK_SIZE..=maximum)
        .rev()
        .find(|candidate| {
            let final_block = total_frames % candidate;
            final_block == 0 || final_block >= FLAC_MIN_BLOCK_SIZE
        })
        .ok_or_else(|| "Could not select a valid FLAC frame size".to_string())
}

pub fn encode_soundkit_index(index: &SoundKitIndex) -> Result<Vec<u8>, String> {
    validate_index(index)?;

    let byte_len = SOUNDKIT_INDEX_HEADER_BYTES
        .checked_add(
            index
                .entries
                .len()
                .checked_mul(SOUNDKIT_INDEX_ENTRY_BYTES as usize)
                .ok_or_else(|| "SoundKit index byte length overflow".to_string())?,
        )
        .ok_or_else(|| "SoundKit index byte length overflow".to_string())?;
    let mut output = vec![0u8; byte_len];
    output[..8].copy_from_slice(&SOUNDKIT_INDEX_MAGIC);
    write_u16_le(&mut output, 8, SOUNDKIT_INDEX_VERSION);
    write_u16_le(&mut output, 10, SOUNDKIT_INDEX_ENTRY_BYTES);
    write_u32_le(&mut output, 12, index.timescale);
    write_u64_le(&mut output, 16, index.entries.len() as u64);
    write_u64_le(&mut output, 24, index.duration_frames);

    let mut offset = SOUNDKIT_INDEX_HEADER_BYTES;
    for entry in &index.entries {
        write_u64_le(&mut output, offset, entry.byte_offset);
        write_u64_le(&mut output, offset + 8, entry.start_frame);
        offset += SOUNDKIT_INDEX_ENTRY_BYTES as usize;
    }

    Ok(output)
}

pub fn decode_soundkit_index(bytes: &[u8]) -> Result<SoundKitIndex, String> {
    if bytes.len() < SOUNDKIT_INDEX_HEADER_BYTES {
        return Err("SoundKit index is too small".to_string());
    }
    if bytes[..8] != SOUNDKIT_INDEX_MAGIC {
        return Err("Invalid SoundKit index magic".to_string());
    }

    let version = read_u16_le(bytes, 8)?;
    if version != SOUNDKIT_INDEX_VERSION {
        return Err(format!("Unsupported SoundKit index version {version}"));
    }
    let entry_bytes = read_u16_le(bytes, 10)?;
    if entry_bytes != SOUNDKIT_INDEX_ENTRY_BYTES {
        return Err(format!(
            "Unsupported SoundKit index entry size {entry_bytes}"
        ));
    }

    let timescale = read_u32_le(bytes, 12)?;
    let entry_count = read_u64_le(bytes, 16)?;
    let duration_frames = read_u64_le(bytes, 24)?;
    let entry_count_usize: usize = entry_count
        .try_into()
        .map_err(|_| "SoundKit index entry count is too large".to_string())?;
    let expected_bytes = SOUNDKIT_INDEX_HEADER_BYTES
        .checked_add(
            entry_count_usize
                .checked_mul(entry_bytes as usize)
                .ok_or_else(|| "SoundKit index byte length overflow".to_string())?,
        )
        .ok_or_else(|| "SoundKit index byte length overflow".to_string())?;
    if bytes.len() != expected_bytes {
        return Err("SoundKit index byte length does not match entry count".to_string());
    }

    let mut entries = Vec::with_capacity(entry_count_usize);
    let mut offset = SOUNDKIT_INDEX_HEADER_BYTES;
    for _ in 0..entry_count_usize {
        entries.push(SoundKitIndexEntry {
            byte_offset: read_u64_le(bytes, offset)?,
            start_frame: read_u64_le(bytes, offset + 8)?,
        });
        offset += entry_bytes as usize;
    }

    let index = SoundKitIndex {
        timescale,
        duration_frames,
        entries,
    };
    validate_index(&index)?;
    Ok(index)
}

pub fn seek_entry_for_frame(
    index: &SoundKitIndex,
    target_frame: u64,
) -> Option<&SoundKitIndexEntry> {
    if index.entries.is_empty() {
        return None;
    }

    let mut low = 0usize;
    let mut high = index.entries.len() - 1;
    let mut found = 0usize;
    while low <= high {
        let mid = low + ((high - low) / 2);
        if index.entries[mid].start_frame <= target_frame {
            found = mid;
            low = mid.saturating_add(1);
        } else if mid == 0 {
            break;
        } else {
            high = mid - 1;
        }
    }
    Some(&index.entries[found])
}

fn validate_pcm_stream_options(options: &PcmI16StreamOptions) -> Result<(), String> {
    if options.sample_rate != 48_000 {
        return Err("The SoundKit stream encoder requires 48 kHz PCM input".to_string());
    }
    if !(1..=2).contains(&options.channels) {
        return Err("The SoundKit stream encoder supports 1 or 2 channels".to_string());
    }
    if !CELT_FRAME_SIZES_48K.contains(&(options.opus_frame_size as usize)) {
        return Err(format!(
            "Opus frame size must be one of {:?} at 48 kHz",
            CELT_FRAME_SIZES_48K
        ));
    }
    if options.bitrate == 0 || options.bitrate > 512_000 {
        return Err("Opus bitrate must be between 1 and 512000".to_string());
    }
    Ok(())
}

fn validate_pcm_opus_options(options: &PcmOpusStreamOptions) -> Result<(), String> {
    if options.sample_rate != 48_000 {
        return Err("The SoundKit stream encoder requires 48 kHz PCM input".to_string());
    }
    if !(1..=2).contains(&options.channels) {
        return Err("The SoundKit stream encoder supports 1 or 2 channels".to_string());
    }
    if !CELT_FRAME_SIZES_48K.contains(&(options.frame_size as usize)) {
        return Err(format!(
            "Opus frame_size must be one of {:?} at 48 kHz",
            CELT_FRAME_SIZES_48K
        ));
    }
    if options.bitrate == 0 || options.bitrate > 512_000 {
        return Err("Opus bitrate must be between 1 and 512000".to_string());
    }
    Ok(())
}

fn validate_index(index: &SoundKitIndex) -> Result<(), String> {
    if index.timescale == 0 {
        return Err("SoundKit index timescale must be non-zero".to_string());
    }
    let mut previous = None;
    for entry in &index.entries {
        if let Some(previous_start_frame) = previous {
            if entry.start_frame <= previous_start_frame {
                return Err("SoundKit index entries must be sorted by start_frame".to_string());
            }
        }
        previous = Some(entry.start_frame);
    }
    Ok(())
}

fn read_u16_le(bytes: &[u8], offset: usize) -> Result<u16, String> {
    let window = bytes
        .get(offset..offset + 2)
        .ok_or_else(|| "SoundKit index truncated while reading u16".to_string())?;
    Ok(u16::from_le_bytes([window[0], window[1]]))
}

fn read_u32_le(bytes: &[u8], offset: usize) -> Result<u32, String> {
    let window = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| "SoundKit index truncated while reading u32".to_string())?;
    Ok(u32::from_le_bytes([
        window[0], window[1], window[2], window[3],
    ]))
}

fn read_u64_le(bytes: &[u8], offset: usize) -> Result<u64, String> {
    let window = bytes
        .get(offset..offset + 8)
        .ok_or_else(|| "SoundKit index truncated while reading u64".to_string())?;
    Ok(u64::from_le_bytes([
        window[0], window[1], window[2], window[3], window[4], window[5], window[6], window[7],
    ]))
}

fn write_u16_le(bytes: &mut [u8], offset: usize, value: u16) {
    bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn write_u32_le(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn write_u64_le(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_and_decodes_sidecar_index() {
        let index = SoundKitIndex {
            timescale: 48_000,
            duration_frames: 1920,
            entries: vec![
                SoundKitIndexEntry {
                    byte_offset: 0,
                    start_frame: 0,
                },
                SoundKitIndexEntry {
                    byte_offset: 123,
                    start_frame: 960,
                },
            ],
        };

        let bytes = encode_soundkit_index(&index).unwrap();
        assert_eq!(bytes.len(), SOUNDKIT_INDEX_HEADER_BYTES + 32);
        assert_eq!(decode_soundkit_index(&bytes).unwrap(), index);
        assert_eq!(seek_entry_for_frame(&index, 959).unwrap().byte_offset, 0);
        assert_eq!(seek_entry_for_frame(&index, 960).unwrap().byte_offset, 123);
    }

    #[test]
    fn encodes_pcm_to_soundkit_v2_opus_stream() {
        let pcm = vec![0i16; 960 * 2 * 2];
        let encoded = encode_interleaved_i16_to_opus_soundkit_stream(
            &pcm,
            PcmOpusStreamOptions {
                sample_rate: 48_000,
                channels: 2,
                frame_size: 960,
                bitrate: 128_000,
                start_pts: 0,
                include_packet_crc32: true,
            },
        )
        .unwrap();

        assert_eq!(encoded.packet_count, 2);
        assert_eq!(encoded.index.entries.len(), 2);
        assert_eq!(encoded.index.entries[0].byte_offset, 0);
        assert_eq!(encoded.index.entries[0].start_frame, 0);
        assert_eq!(encoded.index.entries[1].start_frame, 960);
        assert!(!encoded.stream.is_empty());
        let index_bytes = encoded.index_bytes().unwrap();
        assert_eq!(decode_soundkit_index(&index_bytes).unwrap(), encoded.index);
    }

    #[test]
    fn encodes_pcm_to_both_soundkit_v2_streams() {
        let pcm = vec![0i16; 48_000 * 2];
        let encoded = encode_interleaved_i16_to_soundkit_streams(
            &pcm,
            PcmI16StreamOptions::default(),
        )
        .unwrap();

        assert_eq!(encoded.opus.packet_count, 50);
        assert_eq!(encoded.opus.index.duration_frames, 48_000);
        assert!(!encoded.opus.stream.is_empty());

        assert!(encoded.flac.packet_count > 0);
        assert_eq!(encoded.flac.index.duration_frames, 48_000);
        assert_eq!(
            encoded.flac.index.entries[0],
            SoundKitIndexEntry {
                byte_offset: 0,
                start_frame: 0
            }
        );
        assert!(!encoded.flac.stream.is_empty());

        for stream in [&encoded.opus, &encoded.flac] {
            let bytes = stream.index_bytes().unwrap();
            assert_eq!(decode_soundkit_index(&bytes).unwrap(), stream.index);
        }
    }
}
