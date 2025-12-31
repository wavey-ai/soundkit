use frame_header::{EncodingFlag, Endianness};
use memchr::memmem;
use soundkit::audio_packet::Decoder;
use soundkit::audio_types::AudioData;
use soundkit_opus::OpusDecoder;
use std::collections::VecDeque;
use tracing::{debug, trace};

const MAX_OPUS_FRAME_SAMPLES: usize = 5760; // 120 ms @ 48 kHz

// Zero-copy Ogg page header
#[derive(Debug)]
struct OggPageHeader {
    _version: u8,
    header_type: u8,
    _granule_position: u64,
    serial: u32,
    _sequence: u32,
    _checksum: u32,
    segment_count: u8,
}

impl OggPageHeader {
    fn parse(data: &[u8]) -> Option<Self> {
        if data.len() < 27 || &data[0..4] != b"OggS" {
            return None;
        }

        Some(Self {
            _version: data[4],
            header_type: data[5],
            _granule_position: u64::from_le_bytes(data[6..14].try_into().ok()?),
            serial: u32::from_le_bytes(data[14..18].try_into().ok()?),
            _sequence: u32::from_le_bytes(data[18..22].try_into().ok()?),
            _checksum: u32::from_le_bytes(data[22..26].try_into().ok()?),
            segment_count: data[26],
        })
    }

    fn is_first_page(&self) -> bool {
        self.header_type & 0x02 != 0
    }
}

// Lightweight packet wrapper with metadata
struct Packet {
    data: Vec<u8>,
    serial: u32,
    first_in_stream: bool,
}

struct FastOggParser {
    buffer: Vec<u8>,
    pos: usize,
    packet_buffer: Vec<u8>,
    pending_packets: VecDeque<Packet>,
}

impl FastOggParser {
    fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(8192),
            pos: 0,
            packet_buffer: Vec::with_capacity(4096),
            pending_packets: VecDeque::new(),
        }
    }

    fn push<'a>(&'a mut self, data: &[u8]) -> FastOggPackets<'a> {
        self.buffer.extend_from_slice(data);
        FastOggPackets { parser: self }
    }

    fn compact(&mut self) {
        if self.pos > 0 {
            self.buffer.drain(..self.pos);
            self.pos = 0;
        }
    }
}

struct FastOggPackets<'a> {
    parser: &'a mut FastOggParser,
}

impl<'a> Iterator for FastOggPackets<'a> {
    type Item = Packet;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(packet) = self.parser.pending_packets.pop_front() {
            return Some(packet);
        }

        loop {
            // Find next OggS
            let search_start = self.parser.pos;
            let oggs_pos = memmem::find(&self.parser.buffer[search_start..], b"OggS")?;
            self.parser.pos = search_start + oggs_pos;

            // Need at least 27 bytes for header
            if self.parser.buffer.len() - self.parser.pos < 27 {
                self.parser.compact();
                return None;
            }

            // Parse header
            let header = OggPageHeader::parse(&self.parser.buffer[self.parser.pos..])?;
            let header_size = 27 + header.segment_count as usize;

            if self.parser.buffer.len() - self.parser.pos < header_size {
                self.parser.compact();
                return None;
            }

            // Calculate body size from segment table
            let segment_table =
                &self.parser.buffer[self.parser.pos + 27..self.parser.pos + header_size];
            let body_size: usize = segment_table.iter().map(|&x| x as usize).sum();

            let total_size = header_size + body_size;
            if self.parser.buffer.len() - self.parser.pos < total_size {
                self.parser.compact();
                return None;
            }

            // Parse segments one by one to extract individual packets
            let body_start = self.parser.pos + header_size;
            let mut seg_offset = 0;

            for &seg_size in segment_table.iter() {
                let seg_start = body_start + seg_offset;
                let seg_end = seg_start + seg_size as usize;
                self.parser
                    .packet_buffer
                    .extend_from_slice(&self.parser.buffer[seg_start..seg_end]);
                seg_offset += seg_size as usize;

                // Packet complete when segment < 255
                if seg_size < 255 {
                    let mut packet_data = Vec::new();
                    std::mem::swap(&mut packet_data, &mut self.parser.packet_buffer);
                    self.parser.pending_packets.push_back(Packet {
                        data: packet_data,
                        serial: header.serial,
                        first_in_stream: header.is_first_page(),
                    });
                }
            }

            self.parser.pos += total_size;

            self.parser.compact();
            if let Some(packet) = self.parser.pending_packets.pop_front() {
                return Some(packet);
            }
        }
    }
}

#[derive(Clone, Copy)]
struct OpusStreamInfo {
    sample_rate: u32,
    channels: u8,
    pre_skip: u16,
    serial: u32,
}

pub struct OggOpusDecoder {
    parser: FastOggParser,
    opus: Option<OpusDecoder>,
    info: Option<OpusStreamInfo>,
    seen_tags: bool,
    pre_skip_remaining: usize,
    logged_first_audio: bool,
    scratch_buffer: Vec<i16>,
}

impl OggOpusDecoder {
    pub fn new() -> Self {
        Self {
            parser: FastOggParser::new(),
            opus: None,
            info: None,
            seen_tags: false,
            pre_skip_remaining: 0,
            logged_first_audio: false,
            scratch_buffer: Vec::with_capacity(MAX_OPUS_FRAME_SAMPLES * 2), // Pre-allocate for stereo
        }
    }

    pub fn init(&mut self) -> Result<(), String> {
        Ok(())
    }

    /// Feed more bytes of an Ogg Opus stream. Returns decoded PCM when available.
    pub fn add(&mut self, data: &[u8]) -> Result<Option<AudioData>, String> {
        let packets = self.parser.push(data);
        let mut pcm_bytes = Vec::new();

        for packet in packets {
            if packet.data.is_empty() {
                continue;
            }

            if self.info.is_none() {
                if !packet.first_in_stream {
                    return Err("Expected OpusHead as first packet".to_string());
                }
                let info = Self::parse_head(&packet)?;
                let mut opus = OpusDecoder::new(info.sample_rate as usize, info.channels as usize);
                opus.init()?;
                let scaled_skip =
                    ((info.pre_skip as u64 * info.sample_rate as u64) / 48_000) as usize;
                self.pre_skip_remaining = scaled_skip;
                self.opus = Some(opus);
                self.info = Some(info);
                debug!(
                    sample_rate_hz = info.sample_rate,
                    channels = info.channels,
                    pre_skip = info.pre_skip,
                    scaled_skip = scaled_skip,
                    "parsed OpusHead"
                );
                continue;
            }

            let info = self.info.expect("info set after head");
            if packet.serial != info.serial {
                return Err("Unexpected second logical bitstream".to_string());
            }

            if !self.seen_tags {
                if packet.data.starts_with(b"OpusTags") {
                    self.seen_tags = true;
                    continue;
                } else {
                    return Err("Expected OpusTags packet after OpusHead".to_string());
                }
            }

            let decoder = self
                .opus
                .as_mut()
                .ok_or_else(|| "Opus decoder not initialized".to_string())?;

            // Reuse scratch buffer - resize if needed
            let required_size = MAX_OPUS_FRAME_SAMPLES * info.channels as usize;
            if self.scratch_buffer.len() < required_size {
                self.scratch_buffer.resize(required_size, 0);
            }

            let samples = decoder
                .decode_i16(packet.data.as_slice(), &mut self.scratch_buffer, false)
                .map_err(|e| format!("Opus decode error: {e}"))?;

            if samples == 0 {
                continue;
            }

            if !self.logged_first_audio {
                debug!(
                    packet_len = packet.data.len(),
                    samples_per_channel = samples,
                    pre_skip_remaining = self.pre_skip_remaining,
                    "decoded Opus packet"
                );
            } else {
                trace!(
                    packet_len = packet.data.len(),
                    samples_per_channel = samples,
                    pre_skip_remaining = self.pre_skip_remaining,
                    "decoded Opus packet"
                );
            }

            let mut start = 0;
            if self.pre_skip_remaining > 0 {
                let skip = self.pre_skip_remaining.min(samples);
                self.pre_skip_remaining -= skip;
                start = skip * info.channels as usize;
            }

            let end = samples * info.channels as usize;
            trace!(
                pcm_samples_written = end.saturating_sub(start),
                "appending decoded PCM"
            );
            for sample in &self.scratch_buffer[start..end] {
                pcm_bytes.extend_from_slice(&sample.to_le_bytes());
            }

            self.logged_first_audio = true;
        }

        if pcm_bytes.is_empty() {
            return Ok(None);
        }

        let info = self.info.expect("info set when pcm present");
        let audio = AudioData::new(
            16,
            info.channels,
            info.sample_rate,
            pcm_bytes,
            EncodingFlag::PCMSigned,
            Endianness::LittleEndian,
        );

        Ok(Some(audio))
    }

    fn parse_head(packet: &Packet) -> Result<OpusStreamInfo, String> {
        let data = packet.data.as_slice();
        if data.len() < 19 || !data.starts_with(b"OpusHead") {
            return Err("Invalid OpusHead packet".to_string());
        }

        let channels = data[9];
        let pre_skip = u16::from_le_bytes([data[10], data[11]]);
        let mut sample_rate = u32::from_le_bytes([data[12], data[13], data[14], data[15]]);
        if sample_rate == 0 {
            sample_rate = 48_000;
        }

        Ok(OpusStreamInfo {
            sample_rate,
            channels,
            pre_skip,
            serial: packet.serial,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use soundkit::audio_bytes::{deinterleave_vecs_i16, s16le_to_i16};
    use soundkit::audio_types::PcmData;
    use soundkit::test_utils::{print_waveform_with_header, DecodeResult};
    use soundkit::wav::generate_wav_buffer;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::Once;

    fn init_tracing() {
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            let _ = tracing_subscriber::fmt()
                .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
                .with_test_writer()
                .try_init();
        });
    }

    const TEST_FILE: &str = "A_Tusk_is_used_to_make_costly_gifts";

    fn testdata_path(file: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("testdata")
            .join(file)
    }

    fn outputs_path(file: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("outputs")
            .join(file)
    }

    fn golden_path(file: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("golden")
            .join(file)
    }

    #[test]
    fn test_ogg_opus_decode_waveform() {
        let input_path = testdata_path(&format!("ogg_opus/{}.ogg", TEST_FILE));
        let data = fs::read(&input_path).unwrap();
        assert!(!data.is_empty(), "fixture ogg opus missing or empty");

        init_tracing();

        let mut decoder = OggOpusDecoder::new();
        decoder.init().unwrap();

        let mut decoded_bytes = Vec::new();
        let mut sample_rate = 0u32;
        let mut channels = 0u8;

        for chunk in data.chunks(1024) {
            if let Some(audio) = decoder.add(chunk).unwrap() {
                sample_rate = audio.sampling_rate();
                channels = audio.channel_count();
                decoded_bytes.extend_from_slice(audio.data());
            }
        }

        // Drain remaining
        loop {
            match decoder.add(&[]) {
                Ok(Some(audio)) => {
                    decoded_bytes.extend_from_slice(audio.data());
                }
                Ok(None) => break,
                Err(_) => break,
            }
        }

        assert!(!decoded_bytes.is_empty(), "decoder produced no PCM samples");

        let decoded = s16le_to_i16(&decoded_bytes);
        let result = DecodeResult::new(&decoded, sample_rate, channels);
        print_waveform_with_header("Ogg Opus", &result);
    }

    #[test]
    fn decode_ogg_opus_stream() {
        let data = fs::read(testdata_path(
            "ogg_opus/A_Tusk_is_used_to_make_costly_gifts.ogg",
        ))
        .unwrap();

        init_tracing();

        let mut decoder = OggOpusDecoder::new();
        decoder.init().unwrap();

        let mut decoded = Vec::new();
        for chunk in data.chunks(1024) {
            if let Some(audio) = decoder.add(chunk).unwrap() {
                assert_eq!(audio.bits_per_sample(), 16);
                assert!(audio.channel_count() >= 1);
                decoded.extend_from_slice(audio.data());
            }
        }

        // Drain any remaining buffered data
        loop {
            match decoder.add(&[]) {
                Ok(Some(audio)) => {
                    decoded.extend_from_slice(audio.data());
                }
                Ok(None) => break,
                Err(_) => break,
            }
        }

        assert!(!decoded.is_empty(), "no samples decoded from ogg opus stream");

        // Save PCM output for inspection
        let output_path = outputs_path("A_Tusk_is_used_to_make_costly_gifts.s16le");
        fs::create_dir_all(output_path.parent().unwrap()).unwrap();
        fs::write(&output_path, decoded).unwrap();
    }

    #[test]
    fn decode_ogg_opus_and_write_wav() {
        let input_path = testdata_path("ogg_opus/A_Tusk_is_used_to_make_costly_gifts.ogg");
        let data = fs::read(&input_path).unwrap();

        init_tracing();

        let mut decoder = OggOpusDecoder::new();
        decoder.init().unwrap();

        let mut sample_rate = 0u32;
        let mut pcm_channels: Option<Vec<Vec<i16>>> = None;
        let mut decoded_packets = 0usize;

        for chunk in data.chunks(1024) {
            if let Some(audio) = decoder.add(chunk).unwrap() {
                if sample_rate == 0 {
                    sample_rate = audio.sampling_rate();
                }

                let channel_count = audio.channel_count() as usize;
                let channels = pcm_channels.get_or_insert_with(|| vec![Vec::new(); channel_count]);
                assert_eq!(
                    channels.len(),
                    channel_count,
                    "channel count changed mid-stream"
                );

                let samples = deinterleave_vecs_i16(audio.data(), channel_count);
                for (dst, src) in channels.iter_mut().zip(samples.iter()) {
                    dst.extend_from_slice(src);
                }

                decoded_packets += 1;
            }
        }

        let channels = pcm_channels.expect("no audio decoded from ogg stream");
        assert!(decoded_packets > 0, "no opus packets were decoded");

        let wav_bytes = generate_wav_buffer(&PcmData::I16(channels), sample_rate).unwrap();
        let output_path =
            golden_path("ogg_opus/A_Tusk_is_used_to_make_costly_gifts.ogg.decoded.wav");
        fs::create_dir_all(output_path.parent().unwrap()).unwrap();
        fs::write(&output_path, &wav_bytes).unwrap();

        assert!(
            wav_bytes.starts_with(b"RIFF"),
            "decoded WAV output did not start with RIFF header"
        );
    }
}
