use g729_sys::{
    Decoder as InnerDecoder, Encoder as InnerEncoder, FRAME_SAMPLES, VOICE_FRAME_BYTES,
};
use soundkit::audio_packet::{Decoder, Encoder};

pub const G729_SAMPLE_RATE: u32 = 8_000;
pub const G729_CHANNELS: u8 = 1;

pub struct G729Encoder {
    inner: InnerEncoder,
    pending_samples: Vec<i16>,
}

impl G729Encoder {
    pub fn try_new() -> Result<Self, String> {
        Ok(Self {
            inner: InnerEncoder::new(false).map_err(|error| error.to_string())?,
            pending_samples: Vec::with_capacity(FRAME_SAMPLES),
        })
    }

    pub fn new_voice() -> Self {
        Self::try_new().expect("failed to initialize G.729 encoder")
    }

    pub fn encode_to_vec(&mut self, input: &[i16], output: &mut Vec<u8>) -> Result<usize, String> {
        let start = output.len();
        let required =
            ((self.pending_samples.len() + input.len()) / FRAME_SAMPLES) * VOICE_FRAME_BYTES;
        output.resize(start + required, 0);
        let written = self.encode_i16(input, &mut output[start..])?;
        output.truncate(start + written);
        Ok(written)
    }

    pub fn flush_into(&mut self, output: &mut [u8]) -> Result<usize, String> {
        if self.pending_samples.is_empty() {
            return Ok(0);
        }

        if output.len() < VOICE_FRAME_BYTES {
            return Err(format!(
                "Output buffer too small for G.729 flush: need {}, have {}",
                VOICE_FRAME_BYTES,
                output.len()
            ));
        }

        let mut frame = [0i16; FRAME_SAMPLES];
        frame[..self.pending_samples.len()].copy_from_slice(&self.pending_samples);
        self.pending_samples.clear();

        let encoded = self.inner.encode(&frame);
        output[..encoded.len()].copy_from_slice(&encoded);
        Ok(encoded.len())
    }

    pub fn flush_to_vec(&mut self, output: &mut Vec<u8>) -> Result<usize, String> {
        let start = output.len();
        output.resize(start + VOICE_FRAME_BYTES, 0);
        let written = self.flush_into(&mut output[start..])?;
        output.truncate(start + written);
        Ok(written)
    }
}

impl Default for G729Encoder {
    fn default() -> Self {
        Self::new_voice()
    }
}

impl Encoder for G729Encoder {
    fn new(
        _sample_rate: u32,
        _bits_per_sample: u32,
        _channels: u32,
        _frame_size: u32,
        _bitrate: u32,
    ) -> Self {
        Self::new_voice()
    }

    fn init(&mut self) -> Result<(), String> {
        Ok(())
    }

    fn encode_i16(&mut self, input: &[i16], output: &mut [u8]) -> Result<usize, String> {
        let total_samples = self.pending_samples.len() + input.len();
        let complete_samples = (total_samples / FRAME_SAMPLES) * FRAME_SAMPLES;
        let frame_count = complete_samples / FRAME_SAMPLES;
        let required = frame_count * VOICE_FRAME_BYTES;
        if output.len() < required {
            return Err(format!(
                "Output buffer too small for G.729 encode: need {}, have {}",
                required,
                output.len()
            ));
        }

        let mut samples = Vec::with_capacity(total_samples);
        samples.extend_from_slice(&self.pending_samples);
        samples.extend_from_slice(input);

        let mut written = 0;
        for chunk in samples[..complete_samples].chunks_exact(FRAME_SAMPLES) {
            let mut frame = [0i16; FRAME_SAMPLES];
            frame.copy_from_slice(chunk);
            let encoded = self.inner.encode(&frame);
            output[written..written + encoded.len()].copy_from_slice(&encoded);
            written += encoded.len();
        }

        self.pending_samples.clear();
        self.pending_samples
            .extend_from_slice(&samples[complete_samples..]);

        Ok(written)
    }

    fn encode_i32(&mut self, input: &[i32], output: &mut [u8]) -> Result<usize, String> {
        let samples: Vec<i16> = input.iter().map(|sample| (sample >> 16) as i16).collect();
        self.encode_i16(&samples, output)
    }

    fn reset(&mut self) -> Result<(), String> {
        self.inner = InnerEncoder::new(false).map_err(|error| error.to_string())?;
        self.pending_samples.clear();
        Ok(())
    }
}

pub struct G729Decoder {
    inner: InnerDecoder,
    pending_bytes: Vec<u8>,
}

impl G729Decoder {
    pub fn try_new() -> Result<Self, String> {
        Ok(Self {
            inner: InnerDecoder::new().map_err(|error| error.to_string())?,
            pending_bytes: Vec::with_capacity(VOICE_FRAME_BYTES),
        })
    }

    pub fn new_voice() -> Self {
        Self::try_new().expect("failed to initialize G.729 decoder")
    }

    pub fn sample_rate(&self) -> u32 {
        G729_SAMPLE_RATE
    }

    pub fn channels(&self) -> u8 {
        G729_CHANNELS
    }

    pub fn flush(&mut self) -> Result<(), String> {
        if self.pending_bytes.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "G.729 stream ended with {} trailing partial-frame byte(s)",
                self.pending_bytes.len()
            ))
        }
    }
}

impl Default for G729Decoder {
    fn default() -> Self {
        Self::new_voice()
    }
}

impl Decoder for G729Decoder {
    fn decode_i16(
        &mut self,
        input: &[u8],
        output: &mut [i16],
        _fec: bool,
    ) -> Result<usize, String> {
        let total_bytes = self.pending_bytes.len() + input.len();
        let complete_bytes = (total_bytes / VOICE_FRAME_BYTES) * VOICE_FRAME_BYTES;
        let frame_count = complete_bytes / VOICE_FRAME_BYTES;
        let required = frame_count * FRAME_SAMPLES;
        if output.len() < required {
            return Err(format!(
                "Output buffer too small for G.729 decode: need {}, have {}",
                required,
                output.len()
            ));
        }

        let mut bytes = Vec::with_capacity(total_bytes);
        bytes.extend_from_slice(&self.pending_bytes);
        bytes.extend_from_slice(input);

        let mut written = 0;
        for frame in bytes[..complete_bytes].chunks_exact(VOICE_FRAME_BYTES) {
            let decoded = self.inner.decode(frame, false, false, false);
            output[written..written + FRAME_SAMPLES].copy_from_slice(&decoded);
            written += FRAME_SAMPLES;
        }

        self.pending_bytes.clear();
        self.pending_bytes
            .extend_from_slice(&bytes[complete_bytes..]);

        Ok(written)
    }

    fn decode_i32(
        &mut self,
        input: &[u8],
        output: &mut [i32],
        _fec: bool,
    ) -> Result<usize, String> {
        let required =
            ((self.pending_bytes.len() + input.len()) / VOICE_FRAME_BYTES) * FRAME_SAMPLES;
        if output.len() < required {
            return Err(format!(
                "Output buffer too small for G.729 decode: need {}, have {}",
                required,
                output.len()
            ));
        }

        let mut samples = vec![0i16; required];
        let written = self.decode_i16(input, &mut samples, false)?;
        for (dst, sample) in output.iter_mut().zip(samples.into_iter()).take(written) {
            *dst = i32::from(sample) << 16;
        }
        Ok(written)
    }

    fn decode_f32(
        &mut self,
        input: &[u8],
        output: &mut [f32],
        _fec: bool,
    ) -> Result<usize, String> {
        let required =
            ((self.pending_bytes.len() + input.len()) / VOICE_FRAME_BYTES) * FRAME_SAMPLES;
        if output.len() < required {
            return Err(format!(
                "Output buffer too small for G.729 decode: need {}, have {}",
                required,
                output.len()
            ));
        }

        let mut samples = vec![0i16; required];
        let written = self.decode_i16(input, &mut samples, false)?;
        for (dst, sample) in output.iter_mut().zip(samples.into_iter()).take(written) {
            *dst = f32::from(sample) / 32768.0;
        }
        Ok(written)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

    fn read_s16le_fixture(path: &str) -> Vec<i16> {
        let bytes = fs::read(testdata_path(path)).unwrap();
        assert!(!bytes.is_empty(), "{path} missing or empty");
        bytes
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
            .collect()
    }

    fn samples() -> Vec<i16> {
        (0..205)
            .map(|index| {
                let phase = index as f32 / 80.0 * std::f32::consts::TAU;
                (phase.sin() * 8_000.0) as i16
            })
            .collect()
    }

    #[test]
    fn streaming_encoder_matches_padded_whole_encode() {
        let input = samples();
        let mut padded = input.clone();
        padded.resize(input.len().div_ceil(FRAME_SAMPLES) * FRAME_SAMPLES, 0);

        let mut whole_encoder = G729Encoder::new_voice();
        let mut whole = vec![0u8; padded.len() / FRAME_SAMPLES * VOICE_FRAME_BYTES];
        let whole_len = whole_encoder.encode_i16(&padded, &mut whole).unwrap();
        whole.truncate(whole_len);

        let mut stream_encoder = G729Encoder::new_voice();
        let mut chunked = Vec::new();
        let mut scratch = [0u8; VOICE_FRAME_BYTES * 2];
        for chunk in input.chunks(37) {
            let written = stream_encoder.encode_i16(chunk, &mut scratch).unwrap();
            chunked.extend_from_slice(&scratch[..written]);
        }
        stream_encoder.flush_to_vec(&mut chunked).unwrap();

        assert_eq!(chunked, whole);
    }

    #[test]
    fn streaming_decoder_matches_whole_decode() {
        let input = samples();
        let mut encoder = G729Encoder::new_voice();
        let mut encoded = Vec::new();
        encoder.encode_to_vec(&input, &mut encoded).unwrap();
        encoder.flush_to_vec(&mut encoded).unwrap();

        let mut whole_decoder = G729Decoder::new_voice();
        let mut whole = vec![0i16; encoded.len() / VOICE_FRAME_BYTES * FRAME_SAMPLES];
        let whole_len = whole_decoder
            .decode_i16(&encoded, &mut whole, false)
            .unwrap();
        whole.truncate(whole_len);

        let mut stream_decoder = G729Decoder::new_voice();
        let mut chunked = Vec::new();
        let mut scratch = [0i16; FRAME_SAMPLES * 2];
        for chunk in encoded.chunks(7) {
            let written = stream_decoder
                .decode_i16(chunk, &mut scratch, false)
                .unwrap();
            chunked.extend_from_slice(&scratch[..written]);
        }

        assert_eq!(chunked, whole);
    }

    #[test]
    fn decoder_reports_partial_trailing_frame_on_flush() {
        let mut decoder = G729Decoder::new_voice();
        let mut output = [0i16; FRAME_SAMPLES];
        assert_eq!(
            decoder.decode_i16(&[1, 2, 3], &mut output, false).unwrap(),
            0
        );
        assert!(decoder.flush().unwrap_err().contains("partial-frame"));
    }

    #[test]
    fn decoder_trait_supports_i16_i32_and_f32_output() {
        let input = samples();
        let mut encoder = G729Encoder::new_voice();
        let mut encoded = Vec::new();
        encoder.encode_to_vec(&input, &mut encoded).unwrap();
        encoder.flush_to_vec(&mut encoded).unwrap();

        let sample_count = encoded.len() / VOICE_FRAME_BYTES * FRAME_SAMPLES;
        let mut decoder_i16 = G729Decoder::new_voice();
        let mut decoder_i32 = G729Decoder::new_voice();
        let mut decoder_f32 = G729Decoder::new_voice();
        let mut i16_out = vec![0i16; sample_count];
        let mut i32_out = vec![0i32; sample_count];
        let mut f32_out = vec![0.0f32; sample_count];

        let i16_len = decoder_i16
            .decode_i16(&encoded, &mut i16_out, false)
            .unwrap();
        let i32_len = decoder_i32
            .decode_i32(&encoded, &mut i32_out, false)
            .unwrap();
        let f32_len = decoder_f32
            .decode_f32(&encoded, &mut f32_out, false)
            .unwrap();

        assert_eq!(i16_len, i32_len);
        assert_eq!(i16_len, f32_len);
        for i in 0..i16_len {
            assert_eq!(i32_out[i], i32::from(i16_out[i]) << 16);
            assert!((f32_out[i] - f32::from(i16_out[i]) / 32768.0).abs() < f32::EPSILON);
        }
    }

    #[test]
    #[ignore = "regenerates the committed G.729 fixture from the 8 kHz linear16 fixture"]
    fn generate_g729_fixture_from_linear16_8() {
        let samples = read_s16le_fixture("linear16_8/A_Tusk_is_used_to_make_costly_gifts.s16le");

        let mut encoder = G729Encoder::new_voice();
        let mut encoded = Vec::new();
        for chunk in samples.chunks(173) {
            encoder.encode_to_vec(chunk, &mut encoded).unwrap();
        }
        encoder.flush_to_vec(&mut encoded).unwrap();

        let output_path = testdata_path("g729/A_Tusk_is_used_to_make_costly_gifts.g729");
        fs::create_dir_all(output_path.parent().unwrap()).unwrap();
        fs::write(output_path, encoded).unwrap();
    }

    #[test]
    fn decode_g729_fixture_and_write_golden_wav() {
        let fixture = fs::read(testdata_path(
            "g729/A_Tusk_is_used_to_make_costly_gifts.g729",
        ))
        .unwrap();
        assert!(!fixture.is_empty(), "G.729 fixture missing or empty");

        let mut decoder = G729Decoder::new_voice();
        let mut decoded = Vec::new();
        let mut scratch = [0i16; FRAME_SAMPLES * 3];

        for chunk in fixture.chunks(13) {
            let written = decoder.decode_i16(chunk, &mut scratch, false).unwrap();
            decoded.extend_from_slice(&scratch[..written]);
        }
        decoder.flush().unwrap();

        assert_eq!(
            decoded.len(),
            fixture.len() / VOICE_FRAME_BYTES * FRAME_SAMPLES
        );
        assert!(decoded.iter().any(|&sample| sample != 0));

        let wav = soundkit::wav::generate_wav_buffer(
            &soundkit::audio_types::PcmData::I16(vec![decoded]),
            G729_SAMPLE_RATE,
        )
        .unwrap();
        let output_path = golden_path("g729/A_Tusk_is_used_to_make_costly_gifts.decoded.wav");
        fs::create_dir_all(output_path.parent().unwrap()).unwrap();
        fs::write(output_path, wav).unwrap();
    }
}
