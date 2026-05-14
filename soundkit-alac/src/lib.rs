use alac::Reader as AlacReader;
use frame_header::{EncodingFlag, Endianness};
use soundkit::audio_types::AudioData;
use std::io::Cursor;

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
