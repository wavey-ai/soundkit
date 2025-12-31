use frame_header::{EncodingFlag, Endianness};
use soundkit::audio_packet::Decoder;
use soundkit::audio_types::AudioData;
use soundkit_opus::OpusDecoder;
use std::collections::VecDeque;
use tracing::{debug, trace};

const MAX_OPUS_FRAME_SAMPLES: usize = 5760; // 120 ms @ 48 kHz

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
const SAMPLING_FREQUENCY_ID: u32 = 0xB5;
const CHANNELS_ID: u32 = 0x9F;
const CLUSTER_ID: u32 = 0x1F43B675;
const SIMPLE_BLOCK_ID: u32 = 0xA3;
const BLOCK_ID: u32 = 0xA1;

// Track types
const TRACK_TYPE_AUDIO: u64 = 2;

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
    for i in 1..len {
        value = (value << 8) | data[i] as u64;
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
    for i in 0..len {
        id = (id << 8) | data[i] as u32;
    }

    Some((id, len))
}

/// Read an EBML unsigned integer value
fn read_uint(data: &[u8], size: usize) -> u64 {
    let mut value = 0u64;
    for i in 0..size.min(8).min(data.len()) {
        value = (value << 8) | data[i] as u64;
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

/// Audio track information extracted from WebM
#[derive(Clone, Debug)]
struct AudioTrackInfo {
    track_number: u64,
    codec_id: String,
    sample_rate: u32,
    channels: u8,
    codec_private: Vec<u8>,
}

/// Parser state
#[derive(Debug, Clone, Copy, PartialEq)]
enum ParserState {
    ReadingHeader,
    ReadingTracks,
    ReadingClusters,
}

/// Streaming WebM audio decoder
pub struct WebmDecoder {
    buffer: Vec<u8>,
    state: ParserState,
    audio_track: Option<AudioTrackInfo>,
    opus_decoder: Option<OpusDecoder>,
    pending_audio: VecDeque<Vec<u8>>,
    scratch_buffer: Vec<i16>,
    pre_skip_remaining: usize,
    logged_first_audio: bool,
    header_complete: bool,
}

impl WebmDecoder {
    pub fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(65536),
            state: ParserState::ReadingHeader,
            audio_track: None,
            opus_decoder: None,
            pending_audio: VecDeque::new(),
            scratch_buffer: Vec::with_capacity(MAX_OPUS_FRAME_SAMPLES * 2),
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
        self.buffer.extend_from_slice(data);

        // Try to parse elements
        loop {
            let consumed = match self.state {
                ParserState::ReadingHeader => self.parse_header()?,
                ParserState::ReadingTracks => self.parse_tracks()?,
                ParserState::ReadingClusters => self.parse_clusters()?,
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
            return Err(format!("Invalid WebM: expected EBML header, got 0x{:X}", id));
        }

        let (size, size_len) = match read_vint(&self.buffer[id_len..]) {
            Some(x) => x,
            None => return Ok(0),
        };

        let header_len = id_len + size_len;
        let total_len = header_len + size as usize;

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
            self.state = ParserState::ReadingTracks;
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
            let data_start = pos + header_len;
            let data_end = data_start + size as usize;

            // Check for unknown/infinite size
            let is_master = matches!(
                id,
                TRACKS_ID | TRACK_ENTRY_ID | AUDIO_ID | SEGMENT_ID | CLUSTER_ID
            );

            // Handle infinite size for master elements
            if size == 0x00FFFFFFFFFFFFFF && is_master {
                pos += header_len;
                continue;
            }

            if data_end > self.buffer.len() {
                break;
            }

            match id {
                TRACKS_ID => {
                    trace!(size = size, "found Tracks element");
                    // Parse children inline
                    pos += header_len;
                    continue;
                }
                TRACK_ENTRY_ID => {
                    // Copy the data to avoid borrow checker issues
                    let element_data = self.buffer[data_start..data_end].to_vec();
                    self.parse_track_entry(&element_data)?;
                    pos = data_end;
                    continue;
                }
                CLUSTER_ID => {
                    // We've reached clusters, tracks parsing is done
                    if self.audio_track.is_some() {
                        self.init_decoder()?;
                        self.header_complete = true;
                    }
                    self.state = ParserState::ReadingClusters;
                    return Ok(pos);
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
            let elem_end = elem_start + size as usize;

            if elem_end > data.len() {
                break;
            }

            let elem_data = &data[elem_start..elem_end];

            match id {
                TRACK_NUMBER_ID => {
                    track_number = read_uint(elem_data, size as usize);
                }
                TRACK_TYPE_ID => {
                    track_type = read_uint(elem_data, size as usize);
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
                        let (audio_id, audio_id_len) = match read_element_id(&elem_data[audio_pos..])
                        {
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
                        let audio_elem_end = audio_elem_start + audio_size as usize;

                        if audio_elem_end > elem_data.len() {
                            break;
                        }

                        let audio_elem_data = &elem_data[audio_elem_start..audio_elem_end];

                        match audio_id {
                            SAMPLING_FREQUENCY_ID => {
                                sample_rate = read_float(audio_elem_data, audio_size as usize) as u32;
                                if sample_rate == 0 {
                                    sample_rate = 48000;
                                }
                            }
                            CHANNELS_ID => {
                                channels = read_uint(audio_elem_data, audio_size as usize) as u8;
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

            // Only support Opus for now
            if codec_id == "A_OPUS" {
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
        let track = self.audio_track.as_ref().ok_or("No audio track found")?;

        if track.codec_id != "A_OPUS" {
            return Err(format!("Unsupported codec: {}", track.codec_id));
        }

        // Parse codec private data for pre-skip if available (OpusHead format)
        if track.codec_private.len() >= 10 && track.codec_private.starts_with(b"OpusHead") {
            let pre_skip = u16::from_le_bytes([track.codec_private[10], track.codec_private[11]]);
            self.pre_skip_remaining = pre_skip as usize;
            debug!(pre_skip = pre_skip, "parsed OpusHead pre-skip");
        }

        let mut decoder = OpusDecoder::new(track.sample_rate as usize, track.channels as usize);
        decoder.init()?;

        debug!(
            sample_rate = track.sample_rate,
            channels = track.channels,
            "initialized Opus decoder for WebM"
        );

        self.opus_decoder = Some(decoder);
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
            let data_start = pos + header_len;

            // Handle infinite size clusters
            if size == 0x00FFFFFFFFFFFFFF {
                pos += header_len;
                continue;
            }

            let data_end = data_start + size as usize;

            if data_end > self.buffer.len() {
                break;
            }

            match id {
                CLUSTER_ID => {
                    // Parse cluster children
                    pos += header_len;
                    continue;
                }
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

        let frame_data = &data[frame_start..];

        if !frame_data.is_empty() {
            self.pending_audio.push_back(frame_data.to_vec());
        }

        Ok(())
    }

    fn decode_pending_audio(&mut self) -> Result<Option<AudioData>, String> {
        let decoder = match self.opus_decoder.as_mut() {
            Some(d) => d,
            None => return Ok(None),
        };

        let track = match self.audio_track.as_ref() {
            Some(t) => t,
            None => return Ok(None),
        };

        let mut pcm_bytes = Vec::new();

        while let Some(packet) = self.pending_audio.pop_front() {
            // Reuse scratch buffer
            let required_size = MAX_OPUS_FRAME_SAMPLES * track.channels as usize;
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
                start = skip * track.channels as usize;
            }

            let end = samples * track.channels as usize;
            for sample in &self.scratch_buffer[start..end] {
                pcm_bytes.extend_from_slice(&sample.to_le_bytes());
            }
        }

        if pcm_bytes.is_empty() {
            return Ok(None);
        }

        let audio = AudioData::new(
            16,
            track.channels,
            track.sample_rate,
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
    use soundkit::audio_bytes::s16le_to_i16;
    use soundkit::test_utils::{print_waveform_with_header, DecodeResult};
    use std::fs;
    use std::path::PathBuf;

    const TEST_FILE: &str = "A_Tusk_is_used_to_make_costly_gifts";

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

    fn outputs_path(file: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("outputs")
            .join(file)
    }

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
            println!(
                "Decoded {} bytes from WebM",
                decoded.len()
            );
        }
    }
}
