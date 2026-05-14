use aifc::{AifcReader, Sample, SampleFormat};
use frame_header::{EncodingFlag, Endianness};
use soundkit::audio_types::AudioData;
use std::io::Cursor;

/// AIFF/AIFF-C decoder.
///
/// The container parser is seek-based, so this wrapper accepts streaming
/// chunks and emits decoded PCM once EOF is signalled with an empty chunk.
pub struct AiffDecoder {
    buffer: Vec<u8>,
    decoded: bool,
}

impl AiffDecoder {
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
        decode_aiff_container(&self.buffer).map(Some)
    }
}

impl Default for AiffDecoder {
    fn default() -> Self {
        Self::new()
    }
}

pub fn decode_aiff_container(data: &[u8]) -> Result<AudioData, String> {
    if data.is_empty() {
        return Err("AIFF input is empty".to_string());
    }

    let mut reader =
        AifcReader::new(Cursor::new(data.to_vec())).map_err(|error| format!("{error:?}"))?;
    let info = reader.info();
    let sample_rate = parse_sample_rate(info.sample_rate)?;
    let channels = parse_channels(info.channels)?;
    let (bits_per_sample, audio_format) = output_format(info.sample_format)?;

    let mut pcm = Vec::new();
    for sample in reader.samples().map_err(|error| format!("{error:?}"))? {
        append_sample(sample.map_err(|error| format!("{error:?}"))?, &mut pcm);
    }

    Ok(AudioData::new(
        bits_per_sample,
        channels,
        sample_rate,
        pcm,
        audio_format,
        Endianness::LittleEndian,
    ))
}

fn parse_sample_rate(value: f64) -> Result<u32, String> {
    if !value.is_finite() || value <= 0.0 || value > u32::MAX as f64 {
        return Err(format!("Invalid AIFF sample rate: {value}"));
    }
    Ok(value.round() as u32)
}

fn parse_channels(value: i16) -> Result<u8, String> {
    if value <= 0 || value > u8::MAX as i16 {
        return Err(format!("Invalid AIFF channel count: {value}"));
    }
    Ok(value as u8)
}

fn output_format(sample_format: SampleFormat) -> Result<(u8, EncodingFlag), String> {
    match sample_format {
        SampleFormat::U8
        | SampleFormat::I8
        | SampleFormat::I16
        | SampleFormat::I16LE
        | SampleFormat::CompressedUlaw
        | SampleFormat::CompressedAlaw
        | SampleFormat::CompressedIma4 => Ok((16, EncodingFlag::PCMSigned)),
        SampleFormat::I24 => Ok((24, EncodingFlag::PCMSigned)),
        SampleFormat::I32 | SampleFormat::I32LE => Ok((32, EncodingFlag::PCMSigned)),
        SampleFormat::F32 | SampleFormat::F64 => Ok((32, EncodingFlag::PCMFloat)),
        SampleFormat::Custom(tag) => Err(format!(
            "Unsupported AIFF compression type: {}",
            String::from_utf8_lossy(&tag)
        )),
    }
}

fn append_sample(sample: Sample, out: &mut Vec<u8>) {
    match sample {
        Sample::U8(value) => {
            let signed = (i16::from(value) - 128) << 8;
            out.extend_from_slice(&signed.to_le_bytes());
        }
        Sample::I8(value) => {
            let widened = i16::from(value) << 8;
            out.extend_from_slice(&widened.to_le_bytes());
        }
        Sample::I16(value) => out.extend_from_slice(&value.to_le_bytes()),
        Sample::I24(value) => out.extend_from_slice(&value.to_le_bytes()[..3]),
        Sample::I32(value) => out.extend_from_slice(&value.to_le_bytes()),
        Sample::F32(value) => out.extend_from_slice(&value.to_le_bytes()),
        Sample::F64(value) => out.extend_from_slice(&(value as f32).to_le_bytes()),
    }
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
        let mut decoder = AiffDecoder::new();
        for chunk in data.chunks(chunk_size) {
            assert!(decoder.add(chunk).unwrap().is_none());
        }
        decoder.add(&[]).unwrap().expect("AIFF decode at EOF")
    }

    #[test]
    #[ignore = "regenerates committed AIFF and AIFF-C fixtures using ffmpeg"]
    fn generate_aiff_fixtures_with_ffmpeg() {
        let input = testdata_path("linear16_8/A_Tusk_is_used_to_make_costly_gifts.s16le");
        let aiff_output = testdata_path("aiff/A_Tusk_is_used_to_make_costly_gifts.aiff");
        let aifc_output = testdata_path("aifc/A_Tusk_is_used_to_make_costly_gifts.aifc");
        fs::create_dir_all(aiff_output.parent().unwrap()).unwrap();
        fs::create_dir_all(aifc_output.parent().unwrap()).unwrap();

        let aiff_status = Command::new("ffmpeg")
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
            .args(["-c:a", "pcm_s16be", "-f", "aiff"])
            .arg(&aiff_output)
            .status()
            .unwrap();
        assert!(aiff_status.success());

        let aifc_status = Command::new("ffmpeg")
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
            .args(["-c:a", "pcm_s16le", "-f", "aiff"])
            .arg(&aifc_output)
            .status()
            .unwrap();
        assert!(aifc_status.success());
    }

    #[test]
    fn chunked_decoder_matches_whole_decode() {
        for fixture_name in [
            "aiff/A_Tusk_is_used_to_make_costly_gifts.aiff",
            "aifc/A_Tusk_is_used_to_make_costly_gifts.aifc",
        ] {
            let fixture = fs::read(testdata_path(fixture_name)).unwrap();
            assert!(!fixture.is_empty(), "AIFF fixture missing or empty");

            let whole = decode_chunks(&fixture, fixture.len());
            let chunked = decode_chunks(&fixture, 997);
            assert_eq!(chunked.bits_per_sample(), whole.bits_per_sample());
            assert_eq!(chunked.channel_count(), whole.channel_count());
            assert_eq!(chunked.sampling_rate(), whole.sampling_rate());
            assert_eq!(chunked.data(), whole.data());
        }
    }

    #[test]
    fn decode_aiff_fixtures_and_write_golden_wavs() {
        for (fixture_name, golden_name) in [
            (
                "aiff/A_Tusk_is_used_to_make_costly_gifts.aiff",
                "aiff/A_Tusk_is_used_to_make_costly_gifts.decoded.wav",
            ),
            (
                "aifc/A_Tusk_is_used_to_make_costly_gifts.aifc",
                "aifc/A_Tusk_is_used_to_make_costly_gifts.decoded.wav",
            ),
        ] {
            let fixture = fs::read(testdata_path(fixture_name)).unwrap();
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
            let output_path = golden_path(golden_name);
            fs::create_dir_all(output_path.parent().unwrap()).unwrap();
            fs::write(output_path, wav).unwrap();
        }
    }

    #[test]
    fn native_decode_matches_ffmpeg_pcm() {
        for fixture_name in [
            "aiff/A_Tusk_is_used_to_make_costly_gifts.aiff",
            "aifc/A_Tusk_is_used_to_make_costly_gifts.aifc",
        ] {
            let input = testdata_path(fixture_name);
            let ffmpeg_pcm = std::env::temp_dir().join(format!(
                "soundkit-{}-ffmpeg.s16le",
                fixture_name.replace('/', "-")
            ));
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
}
