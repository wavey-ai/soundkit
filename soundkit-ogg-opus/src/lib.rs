#[cfg(feature = "decode")]
use frame_header::{EncodingFlag, Endianness};
#[cfg(feature = "decode")]
use soundkit::audio_packet::Decoder;
#[cfg(feature = "decode")]
use soundkit::audio_types::AudioData;
#[cfg(test)]
use soundkit::ogg::ogg_page_crc;
use soundkit::ogg::{OggPacket, OggPacketParser};
#[cfg(feature = "decode")]
use soundkit_opus::OpusDecoder;
#[cfg(feature = "decode")]
use tracing::{debug, trace};

#[cfg(feature = "decode")]
const MAX_OPUS_FRAME_SAMPLES: usize = 5760; // 120 ms @ 48 kHz
#[cfg(test)]
const MAX_OGG_INPUT_CHUNK_BYTES: usize = 4 * 1024 * 1024;

#[derive(Clone, Copy, Debug)]
struct OpusStreamInfo {
    sample_rate: u32,
    channels: u8,
    pre_skip: u16,
    output_gain: i16,
    mapping_family: u8,
    serial: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OggOpusConfig {
    pub sample_rate: u32,
    pub channels: u8,
    pub pre_skip: u16,
    pub output_gain: i16,
    pub mapping_family: u8,
    pub head: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OggOpusDemuxEvent {
    Config(OggOpusConfig),
    Tags(Vec<u8>),
    Packet(Vec<u8>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OggOpusPacket {
    pub data: Vec<u8>,
    /// Ogg's 48 kHz end-granule for the final packet completed on this page.
    pub granule_position: Option<u64>,
    pub end_of_stream: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OggOpusTimedDemuxEvent {
    Config(OggOpusConfig),
    Tags(Vec<u8>),
    Packet(OggOpusPacket),
}

pub struct OggOpusDemuxer {
    parser: OggPacketParser,
    info: Option<OpusStreamInfo>,
    seen_tags: bool,
}

impl OggOpusDemuxer {
    pub fn new() -> Self {
        Self {
            parser: OggPacketParser::new(),
            info: None,
            seen_tags: false,
        }
    }

    pub fn init(&mut self) -> Result<(), String> {
        Ok(())
    }

    pub fn add(&mut self, data: &[u8]) -> Result<Vec<OggOpusDemuxEvent>, String> {
        self.add_timed(data).map(|events| {
            events
                .into_iter()
                .map(|event| match event {
                    OggOpusTimedDemuxEvent::Config(config) => OggOpusDemuxEvent::Config(config),
                    OggOpusTimedDemuxEvent::Tags(tags) => OggOpusDemuxEvent::Tags(tags),
                    OggOpusTimedDemuxEvent::Packet(packet) => {
                        OggOpusDemuxEvent::Packet(packet.data)
                    }
                })
                .collect()
        })
    }

    pub fn add_timed(&mut self, data: &[u8]) -> Result<Vec<OggOpusTimedDemuxEvent>, String> {
        let packets = self.parser.add(data)?;
        self.process_packets(packets)
    }

    fn process_packets(
        &mut self,
        packets: Vec<OggPacket>,
    ) -> Result<Vec<OggOpusTimedDemuxEvent>, String> {
        let mut events = Vec::new();

        for packet in packets {
            if packet.data.is_empty() {
                continue;
            }

            if self.info.is_none() {
                if !packet.first_in_stream {
                    return Err("Expected OpusHead as first packet".to_string());
                }
                let info = parse_head(&packet)?;
                events.push(OggOpusTimedDemuxEvent::Config(OggOpusConfig {
                    sample_rate: info.sample_rate,
                    channels: info.channels,
                    pre_skip: info.pre_skip,
                    output_gain: info.output_gain,
                    mapping_family: info.mapping_family,
                    head: packet.data,
                }));
                self.info = Some(info);
                continue;
            }

            let info = self
                .info
                .ok_or_else(|| "Opus stream metadata disappeared".to_string())?;
            if packet.serial != info.serial {
                return Err("Unexpected second logical bitstream".to_string());
            }

            if !self.seen_tags {
                if packet.data.starts_with(b"OpusTags") {
                    self.seen_tags = true;
                    events.push(OggOpusTimedDemuxEvent::Tags(packet.data));
                    continue;
                } else {
                    return Err("Expected OpusTags packet after OpusHead".to_string());
                }
            }

            events.push(OggOpusTimedDemuxEvent::Packet(OggOpusPacket {
                data: packet.data,
                granule_position: packet.granule_position,
                end_of_stream: packet.last_in_stream,
            }));
        }

        Ok(events)
    }

    pub fn finish(&mut self) -> Result<Vec<OggOpusDemuxEvent>, String> {
        self.finish_timed().map(|events| {
            events
                .into_iter()
                .map(|event| match event {
                    OggOpusTimedDemuxEvent::Config(config) => OggOpusDemuxEvent::Config(config),
                    OggOpusTimedDemuxEvent::Tags(tags) => OggOpusDemuxEvent::Tags(tags),
                    OggOpusTimedDemuxEvent::Packet(packet) => {
                        OggOpusDemuxEvent::Packet(packet.data)
                    }
                })
                .collect()
        })
    }

    pub fn finish_timed(&mut self) -> Result<Vec<OggOpusTimedDemuxEvent>, String> {
        let packets = self.parser.finish()?;
        self.process_packets(packets)
    }

    pub fn sample_rate(&self) -> Option<u32> {
        self.info.map(|info| info.sample_rate)
    }

    pub fn channels(&self) -> Option<u8> {
        self.info.map(|info| info.channels)
    }
}

impl Default for OggOpusDemuxer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "decode")]
pub struct OggOpusDecoder {
    parser: OggPacketParser,
    opus: Option<OpusDecoder>,
    info: Option<OpusStreamInfo>,
    seen_tags: bool,
    pre_skip_remaining: usize,
    logged_first_audio: bool,
    scratch_buffer: Vec<i16>,
    decoded_samples: u64,
}

#[cfg(feature = "decode")]
impl OggOpusDecoder {
    pub fn new() -> Self {
        Self {
            parser: OggPacketParser::new(),
            opus: None,
            info: None,
            seen_tags: false,
            pre_skip_remaining: 0,
            logged_first_audio: false,
            scratch_buffer: Vec::with_capacity(MAX_OPUS_FRAME_SAMPLES * 2), // Pre-allocate for stereo
            decoded_samples: 0,
        }
    }

    pub fn init(&mut self) -> Result<(), String> {
        Ok(())
    }

    /// Feed more bytes of an Ogg Opus stream. Returns decoded PCM when available.
    pub fn add(&mut self, data: &[u8]) -> Result<Option<AudioData>, String> {
        let packets = self.parser.add(data)?;
        self.decode_packets(packets)
    }

    fn decode_packets(&mut self, packets: Vec<OggPacket>) -> Result<Option<AudioData>, String> {
        let mut pcm_bytes = Vec::new();

        for packet in packets {
            if packet.data.is_empty() {
                continue;
            }

            if self.info.is_none() {
                if !packet.first_in_stream {
                    return Err("Expected OpusHead as first packet".to_string());
                }
                let info = parse_head(&packet)?;
                // RFC 7845 defines the OpusHead input rate as informational.
                // Ogg Opus granule positions and pre-skip use the 48 kHz clock.
                let mut opus = OpusDecoder::new_full(48_000, info.channels as usize)?;
                opus.init()?;
                self.pre_skip_remaining = info.pre_skip as usize;
                self.opus = Some(opus);
                self.info = Some(info);
                debug!(
                    sample_rate_hz = info.sample_rate,
                    channels = info.channels,
                    pre_skip = info.pre_skip,
                    decode_sample_rate_hz = 48_000,
                    "parsed OpusHead"
                );
                continue;
            }

            let info = self
                .info
                .ok_or_else(|| "Opus stream metadata disappeared".to_string())?;
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
            let decoded_start = self.decoded_samples;
            self.decoded_samples = self
                .decoded_samples
                .checked_add(samples as u64)
                .ok_or_else(|| "Ogg Opus decoded sample count overflow".to_string())?;

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

            let mut start = 0usize;
            if self.pre_skip_remaining > 0 {
                let skip = self.pre_skip_remaining.min(samples);
                self.pre_skip_remaining -= skip;
                start = skip * info.channels as usize;
            }

            let end_sample = match packet.granule_position {
                Some(granule) if granule < decoded_start => {
                    return Err(format!(
                        "Ogg Opus granule {granule} precedes decoded position {decoded_start}"
                    ))
                }
                Some(granule) => usize::try_from(granule - decoded_start)
                    .unwrap_or(usize::MAX)
                    .min(samples),
                None => samples,
            };
            let end = end_sample * info.channels as usize;
            let start = start.min(end);
            trace!(
                pcm_samples_written = end.saturating_sub(start),
                "appending decoded PCM"
            );
            for sample in &self.scratch_buffer[start..end] {
                let sample = apply_opus_output_gain(*sample, info.output_gain);
                pcm_bytes.extend_from_slice(&sample.to_le_bytes());
            }

            self.logged_first_audio = true;
        }
        if pcm_bytes.is_empty() {
            return Ok(None);
        }

        let info = self
            .info
            .ok_or_else(|| "Opus stream metadata disappeared".to_string())?;
        let audio = AudioData::new(
            16,
            info.channels,
            48_000,
            pcm_bytes,
            EncodingFlag::PCMSigned,
            Endianness::LittleEndian,
        );

        Ok(Some(audio))
    }

    pub fn finish(&mut self) -> Result<Option<AudioData>, String> {
        let packets = self.parser.finish()?;
        self.decode_packets(packets)
    }
}

#[cfg(feature = "decode")]
fn apply_opus_output_gain(sample: i16, gain_q8_db: i16) -> i16 {
    if gain_q8_db == 0 {
        return sample;
    }
    let gain = 10_f64.powf(f64::from(gain_q8_db) / (20.0 * 256.0));
    (f64::from(sample) * gain)
        .round()
        .clamp(f64::from(i16::MIN), f64::from(i16::MAX)) as i16
}

fn parse_head(packet: &OggPacket) -> Result<OpusStreamInfo, String> {
    let data = packet.data.as_slice();
    if data.len() < 19 || !data.starts_with(b"OpusHead") {
        return Err("Invalid OpusHead packet".to_string());
    }

    let version = data[8];
    if version > 15 {
        return Err(format!("unsupported OpusHead version {version}"));
    }
    let channels = data[9];
    if channels == 0 {
        return Err("OpusHead channel count must be positive".to_string());
    }
    let pre_skip = u16::from_le_bytes([data[10], data[11]]);
    let sample_rate = u32::from_le_bytes([data[12], data[13], data[14], data[15]]);
    let mapping_family = data[18];
    match mapping_family {
        0 if channels <= 2 => {}
        0 => {
            return Err(format!(
                "Opus mapping family 0 does not support {channels} channels"
            ))
        }
        1 | 255 => {
            let required = 21usize
                .checked_add(channels as usize)
                .ok_or_else(|| "OpusHead mapping size overflow".to_string())?;
            if data.len() < required {
                return Err(format!(
                    "OpusHead mapping family {mapping_family} requires {required} bytes"
                ));
            }
            let streams = data[19];
            let coupled = data[20];
            if streams == 0
                || coupled > streams
                || u16::from(streams) + u16::from(coupled) != u16::from(channels)
            {
                return Err("invalid OpusHead multistream mapping counts".to_string());
            }
            if data[21..required]
                .iter()
                .any(|mapping| *mapping != 255 && *mapping >= streams + coupled)
            {
                return Err("invalid OpusHead channel mapping index".to_string());
            }
            return Err(format!(
                "unsupported Opus multistream mapping family {mapping_family}"
            ));
        }
        _ => {
            return Err(format!(
                "unsupported Opus channel mapping family {mapping_family}"
            ))
        }
    }

    Ok(OpusStreamInfo {
        sample_rate,
        channels,
        pre_skip,
        output_gain: i16::from_le_bytes([data[16], data[17]]),
        mapping_family,
        serial: packet.serial,
    })
}

#[cfg(feature = "decode")]
impl Default for OggOpusDecoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "decode")]
    use soundkit::audio_bytes::{deinterleave_vecs_i16, s16le_to_i16};
    #[cfg(feature = "decode")]
    use soundkit::audio_types::PcmData;
    #[cfg(feature = "decode")]
    use soundkit::test_utils::{print_waveform_with_header, DecodeResult};
    #[cfg(feature = "decode")]
    use soundkit::wav::generate_wav_buffer;
    use std::fs;
    use std::path::PathBuf;
    #[cfg(feature = "decode")]
    use std::sync::Once;

    fn test_page(
        header_type: u8,
        granule: u64,
        serial: u32,
        sequence: u32,
        lacing: &[u8],
        body: &[u8],
    ) -> Vec<u8> {
        assert_eq!(
            lacing.iter().map(|value| *value as usize).sum::<usize>(),
            body.len()
        );
        let mut page = vec![0_u8; 27 + lacing.len() + body.len()];
        page[..4].copy_from_slice(b"OggS");
        page[5] = header_type;
        page[6..14].copy_from_slice(&granule.to_le_bytes());
        page[14..18].copy_from_slice(&serial.to_le_bytes());
        page[18..22].copy_from_slice(&sequence.to_le_bytes());
        page[26] = lacing.len() as u8;
        page[27..27 + lacing.len()].copy_from_slice(lacing);
        page[27 + lacing.len()..].copy_from_slice(body);
        let crc = ogg_page_crc(&page);
        page[22..26].copy_from_slice(&crc.to_le_bytes());
        page
    }

    fn test_opus_head(
        version: u8,
        channels: u8,
        input_rate: u32,
        gain: i16,
        family: u8,
    ) -> OggPacket {
        let mut data = b"OpusHead".to_vec();
        data.push(version);
        data.push(channels);
        data.extend_from_slice(&312_u16.to_le_bytes());
        data.extend_from_slice(&input_rate.to_le_bytes());
        data.extend_from_slice(&gain.to_le_bytes());
        data.push(family);
        OggPacket {
            data,
            serial: 1,
            first_in_stream: true,
            last_in_page: true,
            last_in_stream: false,
            granule_position: Some(0),
        }
    }

    #[test]
    fn validates_opus_head_versions_rates_and_mappings() {
        let info = parse_head(&test_opus_head(0, 1, 44_100, 256, 0)).unwrap();
        assert_eq!(info.sample_rate, 44_100);
        assert_eq!(info.output_gain, 256);

        assert!(parse_head(&test_opus_head(16, 1, 48_000, 0, 0))
            .unwrap_err()
            .contains("version"));
        assert!(parse_head(&test_opus_head(0, 0, 48_000, 0, 0))
            .unwrap_err()
            .contains("channel count"));
        assert!(parse_head(&test_opus_head(0, 3, 48_000, 0, 0))
            .unwrap_err()
            .contains("mapping family 0"));
        assert!(parse_head(&test_opus_head(0, 2, 48_000, 0, 1))
            .unwrap_err()
            .contains("requires"));
    }

    #[cfg(feature = "decode")]
    #[test]
    fn applies_opus_output_gain_with_saturation() {
        assert_eq!(apply_opus_output_gain(1234, 0), 1234);
        assert!(apply_opus_output_gain(1000, 256) > 1000);
        assert_eq!(apply_opus_output_gain(i16::MAX, 10 * 256), i16::MAX);
    }

    #[test]
    fn validates_crc_sequence_continuation_and_eos() {
        let first = test_page(0x02, 0, 7, 0, &[1], &[0xAA]);
        let last = test_page(0x04, 1, 7, 1, &[1], &[0xBB]);
        let mut parser = OggPacketParser::new();
        assert_eq!(parser.add(&first).unwrap().len(), 1);
        assert_eq!(parser.add(&last).unwrap().len(), 1);
        assert!(parser.finish().unwrap().is_empty());

        let mut corrupt = first.clone();
        *corrupt.last_mut().unwrap() ^= 1;
        let mut parser = OggPacketParser::new();
        assert!(parser.add(&corrupt).unwrap_err().contains("CRC"));

        let wrong_sequence = test_page(0x04, 1, 7, 2, &[1], &[0xBB]);
        let mut parser = OggPacketParser::new();
        assert_eq!(parser.add(&first).unwrap().len(), 1);
        assert!(parser
            .add(&wrong_sequence)
            .unwrap_err()
            .contains("sequence"));

        let partial = test_page(0x02, 0, 7, 0, &[255], &[0; 255]);
        let missing_continuation = test_page(0x04, 1, 7, 1, &[1], &[0xBB]);
        let mut parser = OggPacketParser::new();
        assert!(parser.add(&partial).unwrap().is_empty());
        assert!(parser
            .add(&missing_continuation)
            .unwrap_err()
            .contains("continued flag"));

        let orphan_continuation = test_page(0x03, 0, 7, 0, &[1], &[0xAA]);
        let mut parser = OggPacketParser::new();
        assert!(parser
            .add(&orphan_continuation)
            .unwrap_err()
            .contains("no preceding packet"));
    }

    #[test]
    fn ogg_parser_bounds_garbage_and_input_chunks() {
        let mut parser = OggPacketParser::new();
        assert!(parser.add(&vec![0x55; 65_536]).is_err());

        let mut demuxer = OggOpusDemuxer::new();
        let error = demuxer
            .add(&vec![0; MAX_OGG_INPUT_CHUNK_BYTES + 1])
            .unwrap_err();
        assert!(error.contains("streaming budget"));
    }

    #[cfg(feature = "decode")]
    fn init_tracing() {
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            let _ = tracing_subscriber::fmt()
                .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
                .with_test_writer()
                .try_init();
        });
    }

    #[cfg(feature = "decode")]
    const TEST_FILE: &str = "A_Tusk_is_used_to_make_costly_gifts";

    fn testdata_path(file: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("testdata")
            .join(file)
    }

    #[cfg(feature = "decode")]
    fn golden_path(file: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("golden")
            .join(file)
    }

    #[test]
    fn demux_ogg_opus_packets() {
        let data = fs::read(testdata_path(
            "ogg_opus/A_Tusk_is_used_to_make_costly_gifts_48khz.ogg",
        ))
        .unwrap();
        let mut demuxer = OggOpusDemuxer::new();
        let mut config = None;
        let mut packets = 0usize;

        for chunk in data.chunks(333) {
            for event in demuxer.add(chunk).unwrap() {
                match event {
                    OggOpusDemuxEvent::Config(next) => config = Some(next),
                    OggOpusDemuxEvent::Tags(tags) => assert!(tags.starts_with(b"OpusTags")),
                    OggOpusDemuxEvent::Packet(packet) => {
                        assert!(!packet.is_empty());
                        packets += 1;
                    }
                }
            }
        }

        let config = config.expect("OpusHead event");
        assert_eq!(config.channels, 1);
        assert_eq!(config.sample_rate, 48_000);
        assert!(packets > 0);
    }

    #[test]
    fn timed_demux_is_chunk_invariant_and_preserves_final_granule() {
        let data = fs::read(testdata_path(
            "ogg_opus/A_Tusk_is_used_to_make_costly_gifts_48khz.ogg",
        ))
        .unwrap();
        let collect = |chunk_size: usize| {
            let mut demuxer = OggOpusDemuxer::new();
            let mut events = Vec::new();
            for chunk in data.chunks(chunk_size) {
                events.extend(demuxer.add_timed(chunk).unwrap());
            }
            events.extend(demuxer.finish_timed().unwrap());
            events
        };
        let reference = collect(MAX_OGG_INPUT_CHUNK_BYTES);
        assert_eq!(reference, collect(1));
        assert_eq!(reference, collect(4 * 1024));
        assert_eq!(reference, collect(64 * 1024));
        assert!(matches!(
            reference.last(),
            Some(OggOpusTimedDemuxEvent::Packet(OggOpusPacket {
                granule_position: Some(_),
                end_of_stream: true,
                ..
            }))
        ));
    }

    #[cfg(feature = "decode")]
    #[test]
    fn decoded_length_and_quality_match_source_after_pre_skip() {
        let data = fs::read(testdata_path(
            "ogg_opus/A_Tusk_is_used_to_make_costly_gifts_48khz.ogg",
        ))
        .unwrap();
        let mut demuxer = OggOpusDemuxer::new();
        let mut final_granule = None;
        let mut pre_skip = 0_u16;
        let mut channels = 0_u8;
        for event in demuxer.add_timed(&data).unwrap() {
            match event {
                OggOpusTimedDemuxEvent::Config(config) => {
                    pre_skip = config.pre_skip;
                    channels = config.channels;
                }
                OggOpusTimedDemuxEvent::Packet(packet) if packet.end_of_stream => {
                    final_granule = packet.granule_position;
                }
                _ => {}
            }
        }
        demuxer.finish_timed().unwrap();

        let mut decoder = OggOpusDecoder::new();
        let mut pcm_bytes = Vec::new();
        for chunk in data.chunks(997) {
            if let Some(audio) = decoder.add(chunk).unwrap() {
                pcm_bytes.extend_from_slice(audio.data());
            }
        }
        if let Some(audio) = decoder.finish().unwrap() {
            pcm_bytes.extend_from_slice(audio.data());
        }
        let decoded_frames = pcm_bytes.len() / (2 * channels as usize);
        assert_eq!(
            decoded_frames as u64,
            final_granule.unwrap() - u64::from(pre_skip)
        );
        let reference =
            include_bytes!("../../testdata/linear16_48/A_Tusk_is_used_to_make_costly_gifts.s16le");
        assert_eq!(pcm_bytes.len(), reference.len());
        let mut signal_energy = 0.0_f64;
        let mut decoded_energy = 0.0_f64;
        let mut error_energy = 0.0_f64;
        let mut dot_product = 0.0_f64;
        let mut max_error = 0_i32;
        for (decoded, expected) in pcm_bytes.chunks_exact(2).zip(reference.chunks_exact(2)) {
            let decoded = f64::from(i16::from_le_bytes([decoded[0], decoded[1]]));
            let expected = f64::from(i16::from_le_bytes([expected[0], expected[1]]));
            let error = decoded - expected;
            signal_energy += expected * expected;
            decoded_energy += decoded * decoded;
            error_energy += error * error;
            dot_product += decoded * expected;
            max_error = max_error.max(error.abs() as i32);
        }
        let gain_delta_db = 10.0 * (decoded_energy / signal_energy).log10();
        let signal_to_noise_db = 10.0 * (signal_energy / error_energy).log10();
        let correlation = dot_product / (decoded_energy * signal_energy).sqrt();
        assert!(
            gain_delta_db.abs() < 0.1,
            "decoded gain changed by {gain_delta_db:.3} dB"
        );
        assert!(
            signal_to_noise_db > 20.0,
            "decoded signal-to-noise ratio fell to {signal_to_noise_db:.2} dB"
        );
        assert!(
            correlation > 0.995,
            "decoded waveform correlation fell to {correlation:.6}"
        );
        assert!(
            max_error <= 1_024,
            "decoded sample error increased to {max_error}"
        );
    }

    #[cfg(feature = "decode")]
    #[test]
    fn test_ogg_opus_decode_waveform() {
        let input_path = testdata_path(&format!("ogg_opus/{}_48khz.ogg", TEST_FILE));
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

    #[cfg(feature = "decode")]
    #[test]
    fn decode_ogg_opus_stream() {
        let data = fs::read(testdata_path(
            "ogg_opus/A_Tusk_is_used_to_make_costly_gifts_48khz.ogg",
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

        assert!(
            !decoded.is_empty(),
            "no samples decoded from ogg opus stream"
        );

        assert!(decoded.iter().any(|sample| *sample != 0));
    }

    #[cfg(feature = "decode")]
    #[test]
    fn decode_ogg_opus_and_write_wav() {
        let input_path = testdata_path("ogg_opus/A_Tusk_is_used_to_make_costly_gifts_48khz.ogg");
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
