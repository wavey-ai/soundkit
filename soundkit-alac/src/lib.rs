use alac::{Decoder as CodecDecoder, Reader as AlacReader, StreamInfo};
use frame_header::{EncodingFlag, Endianness};
use soundkit::audio_types::AudioData;
use std::io::Cursor;

const MAX_ALAC_CHANNELS: u8 = 8;
const MAX_ALAC_FRAMES_PER_PACKET: u32 = 65_536;
const MAX_ALAC_PACKET_BYTES: usize = 16 * 1024 * 1024;

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

/// ALAC decoder for M4A/MP4 and CAF containers.
///
/// The underlying pure Rust ALAC container reader requires `Read + Seek`, so
/// this wrapper accepts streaming chunks but decodes once EOF is signalled with
/// an empty chunk.
pub struct AlacDecoder {
    buffer: Vec<u8>,
    decoded: bool,
}

impl AlacDecoder {
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            decoded: false,
        }
    }

    pub fn init(&mut self) -> Result<(), String> {
        Ok(())
    }

    pub fn add(&mut self, data: &[u8]) -> Result<Option<AudioData>, String> {
        if self.decoded {
            return Ok(None);
        }

        if !data.is_empty() {
            self.buffer.extend_from_slice(data);
            return Ok(None);
        }

        self.decoded = true;
        decode_alac_container(&self.buffer).map(Some)
    }
}

impl Default for AlacDecoder {
    fn default() -> Self {
        Self::new()
    }
}

pub fn decode_alac_container(data: &[u8]) -> Result<AudioData, String> {
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

    fn decode_chunks(data: &[u8], chunk_size: usize) -> AudioData {
        let mut decoder = AlacDecoder::new();
        for chunk in data.chunks(chunk_size) {
            assert!(decoder.add(chunk).unwrap().is_none());
        }
        decoder.add(&[]).unwrap().expect("ALAC decode at EOF")
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
    fn chunked_decoder_matches_whole_decode() {
        let fixture = fs::read(testdata_path(
            "alac/A_Tusk_is_used_to_make_costly_gifts.m4a",
        ))
        .unwrap();
        assert!(!fixture.is_empty(), "ALAC fixture missing or empty");

        let whole = decode_chunks(&fixture, fixture.len());
        let chunked = decode_chunks(&fixture, 997);
        assert_eq!(chunked.bits_per_sample(), whole.bits_per_sample());
        assert_eq!(chunked.channel_count(), whole.channel_count());
        assert_eq!(chunked.sampling_rate(), whole.sampling_rate());
        assert_eq!(chunked.data(), whole.data());
    }

    #[test]
    fn decode_alac_fixture_and_write_golden_wav() {
        let fixture = fs::read(testdata_path(
            "alac/A_Tusk_is_used_to_make_costly_gifts.m4a",
        ))
        .unwrap();
        let audio = decode_chunks(&fixture, 641);
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
        let audio = decode_chunks(&fixture, 1024);
        assert_eq!(audio.data(), &fs::read(ffmpeg_pcm).unwrap());
    }
}
