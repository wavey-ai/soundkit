use frame_header::{EncodingFlag, Endianness};
use oxideav_speex::SpeexDecoder as OxideSpeexDecoder;
use soundkit::audio_types::AudioData;
use soundkit::ogg::{OggPacket, OggPacketParser};

#[cfg(test)]
const MAX_OGG_INPUT_CHUNK_BYTES: usize = 4 * 1024 * 1024;
use std::fmt;

const SPEEX_SIGNATURE: &[u8; 8] = b"Speex   ";
const SPEEX_HEADER_MIN_SIZE: usize = 80;

#[derive(Debug, Clone, Copy)]
struct SpeexHeaderInfo {
    rate: u32,
    mode: u32,
    channels: u8,
    extra_headers: usize,
}

fn read_le_u32(data: &[u8], offset: usize) -> Result<u32, String> {
    let bytes: [u8; 4] = data
        .get(offset..offset + 4)
        .ok_or_else(|| "Truncated Speex header".to_string())?
        .try_into()
        .map_err(|_| "Invalid Speex header field".to_string())?;
    Ok(u32::from_le_bytes(bytes))
}

fn parse_speex_header(data: &[u8]) -> Result<SpeexHeaderInfo, String> {
    if data.len() < SPEEX_HEADER_MIN_SIZE || !data.starts_with(SPEEX_SIGNATURE) {
        return Err("Invalid Speex header packet".to_string());
    }

    let rate = read_le_u32(data, 36)?;
    let mode = read_le_u32(data, 40)?;
    let channels = read_le_u32(data, 48)?;
    let frames_per_packet = read_le_u32(data, 64)?;
    let extra_headers = read_le_u32(data, 68)?;
    if !(1..=64).contains(&frames_per_packet) {
        return Err(format!(
            "Speex frames-per-packet {frames_per_packet} is outside the supported range"
        ));
    }

    let fallback_rate = match mode {
        0 => 8_000,
        1 => 16_000,
        2 => 32_000,
        _ => 8_000,
    };

    Ok(SpeexHeaderInfo {
        rate: if rate == 0 { fallback_rate } else { rate },
        mode,
        channels: channels.clamp(1, u8::MAX as u32) as u8,
        extra_headers: extra_headers as usize,
    })
}

fn push_i16le_samples(samples: &[i16], out: &mut Vec<u8>) {
    out.reserve(samples.len() * 2);
    for &sample in samples {
        out.extend_from_slice(&sample.to_le_bytes());
    }
}

fn error_to_string(error: impl fmt::Display) -> String {
    error.to_string()
}

#[derive(Clone, Copy)]
struct SpeexStreamInfo {
    sample_rate: u32,
    channels: u8,
    serial: u32,
}

/// Streaming Ogg Speex decoder.
///
/// Speex payloads are packetized by Ogg in normal files and archives. This
/// decoder parses Ogg pages incrementally, skips the Speex header/comment
/// packets, and decodes each audio packet with the pure Rust `oxideav-speex`
/// codec core.
pub struct SpeexDecoder {
    parser: OggPacketParser,
    decoder: Option<OxideSpeexDecoder>,
    info: Option<SpeexStreamInfo>,
    headers_to_skip: usize,
}

impl SpeexDecoder {
    pub fn new() -> Self {
        Self {
            parser: OggPacketParser::new(),
            decoder: None,
            info: None,
            headers_to_skip: 0,
        }
    }

    pub fn init(&mut self) -> Result<(), String> {
        Ok(())
    }

    /// Feed more Ogg Speex bytes. Returns decoded interleaved S16LE PCM when
    /// a complete packet or page has produced audio.
    pub fn add(&mut self, data: &[u8]) -> Result<Option<AudioData>, String> {
        let packets = self.parser.add(data)?;
        self.process_packets(packets)
    }

    pub fn finish(&mut self) -> Result<Option<AudioData>, String> {
        let packets = self.parser.finish()?;
        self.process_packets(packets)
    }

    fn process_packets(&mut self, packets: Vec<OggPacket>) -> Result<Option<AudioData>, String> {
        let mut pcm_bytes = Vec::new();

        for packet in packets {
            if packet.data.is_empty() {
                continue;
            }

            if self.info.is_none() {
                if !packet.first_in_stream {
                    return Err("Expected Speex header as first Ogg packet".to_string());
                }
                self.init_from_header(packet)?;
                continue;
            }

            let info = self.info.expect("stream info initialized");
            if packet.serial != info.serial {
                return Err("Unexpected second Ogg logical bitstream".to_string());
            }

            if self.headers_to_skip > 0 {
                self.headers_to_skip -= 1;
                continue;
            }

            self.decode_packet(packet.data, &mut pcm_bytes)?;
        }

        if pcm_bytes.is_empty() {
            return Ok(None);
        }

        let info = self.info.expect("stream info initialized before audio");
        Ok(Some(AudioData::new(
            16,
            info.channels,
            info.sample_rate,
            pcm_bytes,
            EncodingFlag::PCMSigned,
            Endianness::LittleEndian,
        )))
    }

    fn init_from_header(&mut self, packet: OggPacket) -> Result<(), String> {
        let header = parse_speex_header(&packet.data)?;

        if header.mode > 1 {
            return Err(format!(
                "Unsupported Speex mode {}: this decoder currently supports narrowband and wideband",
                header.mode
            ));
        }

        self.decoder = Some(OxideSpeexDecoder::new());
        self.info = Some(SpeexStreamInfo {
            sample_rate: header.rate,
            channels: header.channels,
            serial: packet.serial,
        });
        self.headers_to_skip = 1 + header.extra_headers;
        Ok(())
    }

    fn decode_packet(&mut self, data: Vec<u8>, pcm_bytes: &mut Vec<u8>) -> Result<(), String> {
        let info = self
            .info
            .ok_or_else(|| "Speex stream info missing".to_string())?;
        let decoder = self
            .decoder
            .as_mut()
            .ok_or_else(|| "Speex decoder not initialized".to_string())?;

        let samples = decoder
            .decode_packet_pcm_i16(&data)
            .map_err(error_to_string)?;

        // oxideav-speex's public top-level decoder returns the decoded full-rate
        // Speex PCM for each packet. Ogg Speex is normally mono at this layer;
        // if the container declares multiple channels, preserve the declared
        // channel count by duplicating the mono stream rather than returning a
        // byte count that disagrees with AudioData's channel metadata.
        if info.channels <= 1 {
            push_i16le_samples(&samples, pcm_bytes);
        } else {
            for sample in samples {
                for _ in 0..info.channels {
                    pcm_bytes.extend_from_slice(&sample.to_le_bytes());
                }
            }
        }

        Ok(())
    }
}

impl Default for SpeexDecoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use soundkit::audio_bytes::{deinterleave_vecs_i16, s16le_to_i16};
    use soundkit::audio_types::PcmData;
    use soundkit::wav::generate_wav_buffer;
    use std::fs;
    use std::path::PathBuf;

    #[test]
    fn ogg_parser_bounds_garbage_and_input_chunks() {
        let mut parser = OggPacketParser::new();
        assert!(parser.add(&vec![0x55; 65_536]).is_err());

        let mut decoder = SpeexDecoder::new();
        let error = decoder
            .add(&vec![0; MAX_OGG_INPUT_CHUNK_BYTES + 1])
            .unwrap_err();
        assert!(error.contains("streaming budget"));
    }

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

    fn decode_chunks(data: &[u8], chunk_size: usize) -> (Vec<u8>, u32, u8) {
        let mut decoder = SpeexDecoder::new();
        decoder.init().unwrap();

        let mut decoded = Vec::new();
        let mut sample_rate = 0;
        let mut channels = 0;

        for chunk in data.chunks(chunk_size) {
            if let Some(audio) = decoder.add(chunk).unwrap() {
                sample_rate = audio.sampling_rate();
                channels = audio.channel_count();
                decoded.extend_from_slice(audio.data());
            }
        }

        if let Some(audio) = decoder.finish().unwrap() {
            sample_rate = audio.sampling_rate();
            channels = audio.channel_count();
            decoded.extend_from_slice(audio.data());
        }

        (decoded, sample_rate, channels)
    }

    #[test]
    fn streaming_decoder_matches_whole_decode() {
        let fixture = fs::read(testdata_path(
            "speex/A_Tusk_is_used_to_make_costly_gifts.spx",
        ))
        .unwrap();
        assert!(!fixture.is_empty(), "Speex fixture missing or empty");

        let (whole, whole_rate, whole_channels) = decode_chunks(&fixture, fixture.len());
        let (chunked, chunked_rate, chunked_channels) = decode_chunks(&fixture, 997);

        assert_eq!(chunked, whole);
        assert_eq!(chunked_rate, whole_rate);
        assert_eq!(chunked_channels, whole_channels);
        assert_eq!(chunked_rate, 8_000);
        assert_eq!(chunked_channels, 1);
        assert!(s16le_to_i16(&chunked).iter().any(|&sample| sample != 0));
    }

    #[test]
    fn decode_speex_fixture_and_write_golden_wav() {
        let fixture = fs::read(testdata_path(
            "speex/A_Tusk_is_used_to_make_costly_gifts.spx",
        ))
        .unwrap();
        assert!(!fixture.is_empty(), "Speex fixture missing or empty");

        let (decoded, sample_rate, channels) = decode_chunks(&fixture, 509);
        assert!(!decoded.is_empty(), "no samples decoded from Speex stream");
        assert_eq!(sample_rate, 8_000);
        assert_eq!(channels, 1);

        let channel_samples = deinterleave_vecs_i16(&decoded, channels as usize);
        assert!(channel_samples[0].iter().any(|&sample| sample != 0));

        let wav = generate_wav_buffer(&PcmData::I16(channel_samples), sample_rate).unwrap();
        let output_path = golden_path("speex/A_Tusk_is_used_to_make_costly_gifts.decoded.wav");
        fs::create_dir_all(output_path.parent().unwrap()).unwrap();
        fs::write(output_path, wav).unwrap();
    }
}
