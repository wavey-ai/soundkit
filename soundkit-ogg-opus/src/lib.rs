use frame_header::{EncodingFlag, Endianness};
use ogg::reading::{BasePacketReader, OggReadError, PageParser};
use ogg::Packet;
use soundkit::audio_packet::Decoder;
use soundkit::audio_types::AudioData;
use soundkit_opus::OpusDecoder;

const MAX_OPUS_FRAME_SAMPLES: usize = 5760; // 120 ms @ 48 kHz

struct OggPager {
    buffer: Vec<u8>,
    base: BasePacketReader,
}

impl OggPager {
    fn new() -> Self {
        Self {
            buffer: Vec::new(),
            base: BasePacketReader::new(),
        }
    }

    fn push(&mut self, data: &[u8]) -> Result<Vec<Packet>, OggReadError> {
        self.buffer.extend_from_slice(data);
        let mut packets = Vec::new();

        loop {
            let header_pos = match find_capture_pattern(&self.buffer) {
                Some(pos) => pos,
                None => {
                    // keep last 3 bytes in case a header spans the boundary
                    if self.buffer.len() > 3 {
                        let keep = 3;
                        let drop = self.buffer.len() - keep;
                        self.buffer.drain(..drop);
                    }
                    break;
                }
            };

            if header_pos > 0 {
                self.buffer.drain(..header_pos);
            }

            if self.buffer.len() < 27 {
                break;
            }

            let mut header = [0u8; 27];
            header.copy_from_slice(&self.buffer[..27]);
            let (mut parser, seg_len) = PageParser::new(header)?;

            if self.buffer.len() < 27 + seg_len {
                break;
            }

            let segments = self.buffer[27..27 + seg_len].to_vec();
            let body_len = parser.parse_segments(segments);

            if self.buffer.len() < 27 + seg_len + body_len {
                break;
            }

            let packet_data = self.buffer[27 + seg_len..27 + seg_len + body_len].to_vec();
            let page = parser.parse_packet_data(packet_data)?;

            self.base.push_page(page)?;
            self.buffer.drain(..27 + seg_len + body_len);

            while let Some(packet) = self.base.read_packet() {
                packets.push(packet);
            }
        }

        Ok(packets)
    }
}

fn find_capture_pattern(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"OggS")
}

#[derive(Clone, Copy)]
struct OpusStreamInfo {
    sample_rate: u32,
    channels: u8,
    pre_skip: u16,
    serial: u32,
}

pub struct OggOpusDecoder {
    pager: OggPager,
    opus: Option<OpusDecoder>,
    info: Option<OpusStreamInfo>,
    seen_tags: bool,
    pre_skip_remaining: usize,
}

impl OggOpusDecoder {
    pub fn new() -> Self {
        Self {
            pager: OggPager::new(),
            opus: None,
            info: None,
            seen_tags: false,
            pre_skip_remaining: 0,
        }
    }

    pub fn init(&mut self) -> Result<(), String> {
        Ok(())
    }

    /// Feed more bytes of an Ogg Opus stream. Returns decoded PCM when available.
    pub fn add(&mut self, data: &[u8]) -> Result<Option<AudioData>, String> {
        let packets = self.pager.push(data).map_err(|e| format!("{:?}", e))?;
        let mut pcm_bytes = Vec::new();

        for packet in packets {
            if packet.data.is_empty() {
                continue;
            }

            if self.info.is_none() {
                if !packet.first_in_stream() {
                    return Err("Expected OpusHead as first packet".to_string());
                }
                let info = Self::parse_head(&packet)?;
                let mut opus =
                    OpusDecoder::new(info.sample_rate as usize, info.channels as usize);
                opus.init()?;
                let scaled_skip =
                    ((info.pre_skip as u64 * info.sample_rate as u64) / 48_000) as usize;
                self.pre_skip_remaining = scaled_skip;
                self.opus = Some(opus);
                self.info = Some(info);
                continue;
            }

            let info = self.info.expect("info set after head");
            if packet.stream_serial() != info.serial {
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

            let mut tmp = vec![0i16; MAX_OPUS_FRAME_SAMPLES * info.channels as usize];
            let samples = decoder
                .decode_i16(&packet.data, &mut tmp, false)
                .map_err(|e| format!("Opus decode error: {e}"))?;

            if samples == 0 {
                continue;
            }

            let mut start = 0;
            if self.pre_skip_remaining > 0 {
                let skip = self.pre_skip_remaining.min(samples);
                self.pre_skip_remaining -= skip;
                start = skip * info.channels as usize;
            }

            let end = samples * info.channels as usize;
            for sample in &tmp[start..end] {
                pcm_bytes.extend_from_slice(&sample.to_le_bytes());
            }
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
        let data = &packet.data;
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
            serial: packet.stream_serial(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use soundkit::audio_bytes::deinterleave_vecs_i16;
    use soundkit::audio_types::PcmData;
    use soundkit::wav::generate_wav_buffer;
    use std::fs;
    use std::path::{Path, PathBuf};

    fn testdata_path(file: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("testdata")
            .join(file)
    }

    fn append_suffix(path: &Path, suffix: &str) -> PathBuf {
        path.with_file_name(format!(
            "{}{}",
            path.file_name().unwrap().to_string_lossy(),
            suffix
        ))
    }

    #[test]
    fn decode_ogg_opus_stream() {
        let data = fs::read(testdata_path("ogg_opus/A_Tusk_is_used_to_make_costly_gifts.ogg"))
            .unwrap();

        let mut decoder = OggOpusDecoder::new();
        decoder.init().unwrap();

        let mut total_samples = 0usize;
        for chunk in data.chunks(1024) {
            if let Some(audio) = decoder.add(chunk).unwrap() {
                assert_eq!(audio.bits_per_sample(), 16);
                assert!(audio.channel_count() >= 1);
                total_samples +=
                    audio.data().len() / 2 / audio.channel_count() as usize;
            }
        }

        assert!(total_samples > 0, "no samples decoded from ogg opus stream");
    }

    #[test]
    fn decode_ogg_opus_and_write_wav() {
        let input_path = testdata_path("ogg_opus/A_Tusk_is_used_to_make_costly_gifts.ogg");
        let data = fs::read(&input_path).unwrap();

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
                let channels =
                    pcm_channels.get_or_insert_with(|| vec![Vec::new(); channel_count]);
                assert_eq!(channels.len(), channel_count, "channel count changed mid-stream");

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
        let output_path = append_suffix(&input_path, ".decoded.wav");
        fs::write(&output_path, &wav_bytes).unwrap();

        assert!(
            wav_bytes.starts_with(b"RIFF"),
            "decoded WAV output did not start with RIFF header"
        );
    }
}
