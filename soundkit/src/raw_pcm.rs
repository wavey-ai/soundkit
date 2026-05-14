use crate::audio_types::AudioData;
use frame_header::{EncodingFlag, Endianness};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RawPcmSampleFormat {
    I16,
    I24,
    I32,
    F32,
}

impl RawPcmSampleFormat {
    pub fn bits_per_sample(self) -> u8 {
        match self {
            RawPcmSampleFormat::I16 => 16,
            RawPcmSampleFormat::I24 => 24,
            RawPcmSampleFormat::I32 | RawPcmSampleFormat::F32 => 32,
        }
    }

    pub fn bytes_per_sample(self) -> usize {
        usize::from(self.bits_per_sample() / 8)
    }

    pub fn encoding_flag(self) -> EncodingFlag {
        match self {
            RawPcmSampleFormat::F32 => EncodingFlag::PCMFloat,
            RawPcmSampleFormat::I16 | RawPcmSampleFormat::I24 | RawPcmSampleFormat::I32 => {
                EncodingFlag::PCMSigned
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RawPcmFormat {
    sample_rate: u32,
    channels: u8,
    sample_format: RawPcmSampleFormat,
    endianness: Endianness,
}

impl RawPcmFormat {
    pub fn new(
        sample_rate: u32,
        channels: u8,
        sample_format: RawPcmSampleFormat,
        endianness: Endianness,
    ) -> Result<Self, String> {
        let format = Self {
            sample_rate,
            channels,
            sample_format,
            endianness,
        };
        format.validate()?;
        Ok(format)
    }

    pub fn linear16(sample_rate: u32, channels: u8) -> Result<Self, String> {
        Self::new(
            sample_rate,
            channels,
            RawPcmSampleFormat::I16,
            Endianness::LittleEndian,
        )
    }

    pub fn l16(sample_rate: u32, channels: u8) -> Result<Self, String> {
        Self::new(
            sample_rate,
            channels,
            RawPcmSampleFormat::I16,
            Endianness::BigEndian,
        )
    }

    pub fn linear32(sample_rate: u32, channels: u8) -> Result<Self, String> {
        Self::new(
            sample_rate,
            channels,
            RawPcmSampleFormat::F32,
            Endianness::LittleEndian,
        )
    }

    pub fn bytes_per_frame(self) -> usize {
        self.sample_format.bytes_per_sample() * self.channels as usize
    }

    pub fn sample_rate(self) -> u32 {
        self.sample_rate
    }

    pub fn channels(self) -> u8 {
        self.channels
    }

    pub fn sample_format(self) -> RawPcmSampleFormat {
        self.sample_format
    }

    pub fn endianness(self) -> Endianness {
        self.endianness
    }

    pub fn bits_per_sample(self) -> u8 {
        self.sample_format.bits_per_sample()
    }

    pub fn encoding_flag(self) -> EncodingFlag {
        self.sample_format.encoding_flag()
    }

    pub fn validate(self) -> Result<(), String> {
        if self.sample_rate == 0 {
            return Err("Raw PCM sample rate must be > 0".to_string());
        }
        if self.channels == 0 {
            return Err("Raw PCM channel count must be > 0".to_string());
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct RawPcmStreamProcessor {
    format: RawPcmFormat,
    buffer: Vec<u8>,
}

impl RawPcmStreamProcessor {
    pub fn new(format: RawPcmFormat) -> Self {
        Self {
            format,
            buffer: Vec::new(),
        }
    }

    pub fn format(&self) -> RawPcmFormat {
        self.format
    }

    pub fn buffered_len(&self) -> usize {
        self.buffer.len()
    }

    pub fn add(&mut self, chunk: &[u8]) -> Result<Option<AudioData>, String> {
        self.buffer.extend_from_slice(chunk);

        let bytes_per_frame = self.format.bytes_per_frame();
        if bytes_per_frame == 0 {
            return Err("Raw PCM bytes per frame must be > 0".to_string());
        }

        let complete_len = (self.buffer.len() / bytes_per_frame) * bytes_per_frame;
        if complete_len == 0 {
            return Ok(None);
        }

        let remaining = self.buffer.split_off(complete_len);
        let data = std::mem::replace(&mut self.buffer, remaining);

        Ok(Some(AudioData::new(
            self.format.bits_per_sample(),
            self.format.channels,
            self.format.sample_rate,
            data,
            self.format.encoding_flag(),
            self.format.endianness,
        )))
    }

    pub fn flush(&mut self) -> Result<Option<AudioData>, String> {
        if self.buffer.is_empty() {
            return Ok(None);
        }

        Err(format!(
            "Raw PCM stream ended with {} trailing partial-frame byte(s)",
            self.buffer.len()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_types::PcmData;
    use crate::wav::generate_wav_buffer;
    use std::fs;
    use std::path::PathBuf;

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

    #[test]
    fn linear16_buffers_partial_frames_until_complete() {
        let format = RawPcmFormat::linear16(8_000, 1).unwrap();
        let mut processor = RawPcmStreamProcessor::new(format);

        assert!(processor.add(&[0x34]).unwrap().is_none());
        assert_eq!(processor.buffered_len(), 1);

        let audio = processor.add(&[0x12, 0x78, 0x56]).unwrap().unwrap();
        assert_eq!(audio.bits_per_sample(), 16);
        assert_eq!(audio.channel_count(), 1);
        assert_eq!(audio.sampling_rate(), 8_000);
        assert_eq!(audio.audio_format(), EncodingFlag::PCMSigned);
        assert_eq!(audio.endianness(), Endianness::LittleEndian);
        assert_eq!(audio.data(), &vec![0x34, 0x12, 0x78, 0x56]);
        assert_eq!(processor.buffered_len(), 0);
    }

    #[test]
    fn stereo_streams_only_emit_complete_interleaved_frames() {
        let format = RawPcmFormat::linear16(16_000, 2).unwrap();
        let mut processor = RawPcmStreamProcessor::new(format);

        assert!(processor.add(&[1, 2, 3]).unwrap().is_none());
        assert_eq!(processor.buffered_len(), 3);

        let audio = processor.add(&[4, 5, 6, 7, 8, 9]).unwrap().unwrap();
        assert_eq!(audio.data(), &vec![1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(processor.buffered_len(), 1);
        assert!(processor.flush().unwrap_err().contains("partial-frame"));
    }

    #[test]
    fn l16_uses_big_endian_metadata() {
        let format = RawPcmFormat::l16(8_000, 1).unwrap();
        let mut processor = RawPcmStreamProcessor::new(format);

        let audio = processor.add(&[0x12, 0x34]).unwrap().unwrap();
        assert_eq!(audio.endianness(), Endianness::BigEndian);
        assert_eq!(audio.data(), &vec![0x12, 0x34]);
    }

    #[test]
    fn linear32_is_float_pcm() {
        let format = RawPcmFormat::linear32(48_000, 1).unwrap();
        let mut processor = RawPcmStreamProcessor::new(format);
        let sample = 0.25f32.to_le_bytes();

        let audio = processor.add(&sample).unwrap().unwrap();
        assert_eq!(audio.bits_per_sample(), 32);
        assert_eq!(audio.audio_format(), EncodingFlag::PCMFloat);
        assert_eq!(audio.data(), &sample.to_vec());
    }

    #[test]
    fn rejects_invalid_format_metadata() {
        assert!(RawPcmFormat::linear16(0, 1).is_err());
        assert!(RawPcmFormat::linear16(8_000, 0).is_err());
    }

    #[test]
    fn decode_linear16_fixture_and_write_golden_wav() {
        let fixture = fs::read(testdata_path(
            "linear16/A_Tusk_is_used_to_make_costly_gifts.s16le",
        ))
        .unwrap();
        assert!(!fixture.is_empty(), "linear16 fixture missing or empty");

        let format = RawPcmFormat::linear16(16_000, 1).unwrap();
        let mut processor = RawPcmStreamProcessor::new(format);
        let mut decoded = Vec::new();

        for chunk in fixture.chunks(333) {
            if let Some(audio) = processor.add(chunk).unwrap() {
                assert_eq!(audio.bits_per_sample(), 16);
                assert_eq!(audio.channel_count(), 1);
                assert_eq!(audio.sampling_rate(), 16_000);
                decoded.extend_from_slice(audio.data());
            }
        }
        assert!(processor.flush().unwrap().is_none());

        assert_eq!(decoded, fixture);

        let samples: Vec<i16> = decoded
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
            .collect();
        let wav = generate_wav_buffer(&PcmData::I16(vec![samples]), 16_000).unwrap();
        let output_path = golden_path("linear16/A_Tusk_is_used_to_make_costly_gifts.decoded.wav");
        fs::create_dir_all(output_path.parent().unwrap()).unwrap();
        fs::write(output_path, wav).unwrap();
    }
}
