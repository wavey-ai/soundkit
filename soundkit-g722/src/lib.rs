use ezk_g722::libg722::{
    decoder::Decoder as InnerDecoder, encoder::Encoder as InnerEncoder, Bitrate,
};
use soundkit::audio_packet::{Decoder, Encoder};

pub const G722_SAMPLE_RATE: u32 = 16_000;
pub const G722_CHANNELS: u8 = 1;

pub struct G722Encoder {
    inner: InnerEncoder,
    pending_sample: Option<i16>,
}

impl G722Encoder {
    pub fn new_64k() -> Self {
        Self {
            inner: InnerEncoder::new(Bitrate::Mode1_64000, false, false),
            pending_sample: None,
        }
    }

    pub fn encode_to_vec(&mut self, input: &[i16], output: &mut Vec<u8>) -> Result<usize, String> {
        let start = output.len();
        let mut scratch =
            vec![0u8; max_encoded_len(input.len() + self.pending_sample.is_some() as usize)];
        let written = self.encode_i16(input, &mut scratch)?;
        output.extend_from_slice(&scratch[..written]);
        Ok(output.len() - start)
    }

    pub fn flush_into(&mut self, output: &mut [u8]) -> Result<usize, String> {
        let Some(sample) = self.pending_sample.take() else {
            return Ok(0);
        };

        if output.is_empty() {
            self.pending_sample = Some(sample);
            return Err("Output buffer too small for G.722 flush: need 1, have 0".to_string());
        }

        let encoded = self.inner.encode(&[sample, 0]);
        output[..encoded.len()].copy_from_slice(&encoded);
        Ok(encoded.len())
    }

    pub fn flush_to_vec(&mut self, output: &mut Vec<u8>) -> Result<usize, String> {
        let start = output.len();
        output.resize(start + 1, 0);
        let written = self.flush_into(&mut output[start..])?;
        output.truncate(start + written);
        Ok(written)
    }
}

impl Default for G722Encoder {
    fn default() -> Self {
        Self::new_64k()
    }
}

impl Encoder for G722Encoder {
    fn new(
        _sample_rate: u32,
        _bits_per_sample: u32,
        _channels: u32,
        _frame_size: u32,
        _bitrate: u32,
    ) -> Self {
        Self::new_64k()
    }

    fn init(&mut self) -> Result<(), String> {
        Ok(())
    }

    fn encode_i16(&mut self, input: &[i16], output: &mut [u8]) -> Result<usize, String> {
        let total_samples = input.len() + self.pending_sample.is_some() as usize;
        let complete_samples = total_samples - (total_samples % 2);
        let required = complete_samples / 2;
        if output.len() < required {
            return Err(format!(
                "Output buffer too small for G.722 encode: need {}, have {}",
                required,
                output.len()
            ));
        }

        if total_samples < 2 {
            if self.pending_sample.is_none() {
                self.pending_sample = input.first().copied();
            }
            return Ok(0);
        }

        let mut samples = Vec::with_capacity(complete_samples);
        if let Some(sample) = self.pending_sample.take() {
            samples.push(sample);
        }
        samples.extend_from_slice(input);

        if samples.len() % 2 != 0 {
            self.pending_sample = samples.pop();
        }

        let encoded = self.inner.encode(&samples);
        output[..encoded.len()].copy_from_slice(&encoded);
        Ok(encoded.len())
    }

    fn encode_i32(&mut self, input: &[i32], output: &mut [u8]) -> Result<usize, String> {
        let samples: Vec<i16> = input.iter().map(|sample| (sample >> 16) as i16).collect();
        self.encode_i16(&samples, output)
    }

    fn reset(&mut self) -> Result<(), String> {
        self.inner = InnerEncoder::new(Bitrate::Mode1_64000, false, false);
        self.pending_sample = None;
        Ok(())
    }
}

pub struct G722Decoder {
    inner: InnerDecoder,
}

impl G722Decoder {
    pub fn new_64k() -> Self {
        Self {
            inner: InnerDecoder::new(Bitrate::Mode1_64000, false, false),
        }
    }

    pub fn sample_rate(&self) -> u32 {
        G722_SAMPLE_RATE
    }

    pub fn channels(&self) -> u8 {
        G722_CHANNELS
    }
}

impl Default for G722Decoder {
    fn default() -> Self {
        Self::new_64k()
    }
}

impl Decoder for G722Decoder {
    fn decode_i16(
        &mut self,
        input: &[u8],
        output: &mut [i16],
        _fec: bool,
    ) -> Result<usize, String> {
        let required = input.len() * 2;
        if output.len() < required {
            return Err(format!(
                "Output buffer too small for G.722 decode: need {}, have {}",
                required,
                output.len()
            ));
        }

        let decoded = self.inner.decode(input);
        output[..decoded.len()].copy_from_slice(&decoded);
        Ok(decoded.len())
    }

    fn decode_i32(
        &mut self,
        input: &[u8],
        output: &mut [i32],
        _fec: bool,
    ) -> Result<usize, String> {
        let required = input.len() * 2;
        if output.len() < required {
            return Err(format!(
                "Output buffer too small for G.722 decode: need {}, have {}",
                required,
                output.len()
            ));
        }

        let decoded = self.inner.decode(input);
        for (dst, sample) in output.iter_mut().zip(decoded) {
            *dst = i32::from(sample) << 16;
        }
        Ok(required)
    }

    fn decode_f32(
        &mut self,
        input: &[u8],
        output: &mut [f32],
        _fec: bool,
    ) -> Result<usize, String> {
        let required = input.len() * 2;
        if output.len() < required {
            return Err(format!(
                "Output buffer too small for G.722 decode: need {}, have {}",
                required,
                output.len()
            ));
        }

        let decoded = self.inner.decode(input);
        for (dst, sample) in output.iter_mut().zip(decoded) {
            *dst = f32::from(sample) / 32768.0;
        }
        Ok(required)
    }
}

fn max_encoded_len(input_samples: usize) -> usize {
    input_samples.div_ceil(2)
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

    fn samples() -> Vec<i16> {
        (0..161)
            .map(|index| {
                let phase = index as f32 / 160.0 * std::f32::consts::TAU * 3.0;
                (phase.sin() * 12_000.0) as i16
            })
            .collect()
    }

    #[test]
    fn streaming_encoder_matches_padded_whole_encode() {
        let input = samples();
        let mut padded = input.clone();
        padded.push(0);

        let mut whole_encoder = G722Encoder::new_64k();
        let mut whole = vec![0u8; padded.len() / 2];
        let whole_len = whole_encoder.encode_i16(&padded, &mut whole).unwrap();
        whole.truncate(whole_len);

        let mut stream_encoder = G722Encoder::new_64k();
        let mut chunked = Vec::new();
        let mut scratch = [0u8; 8];
        for chunk in input.chunks(5) {
            let written = stream_encoder.encode_i16(chunk, &mut scratch).unwrap();
            chunked.extend_from_slice(&scratch[..written]);
        }
        stream_encoder.flush_to_vec(&mut chunked).unwrap();

        assert_eq!(chunked, whole);
    }

    #[test]
    fn streaming_decoder_matches_whole_decode() {
        let input = samples();
        let mut encoder = G722Encoder::new_64k();
        let mut encoded = Vec::new();
        encoder.encode_to_vec(&input, &mut encoded).unwrap();
        encoder.flush_to_vec(&mut encoded).unwrap();

        let mut whole_decoder = G722Decoder::new_64k();
        let mut whole = vec![0i16; encoded.len() * 2];
        let whole_len = whole_decoder
            .decode_i16(&encoded, &mut whole, false)
            .unwrap();
        whole.truncate(whole_len);

        let mut stream_decoder = G722Decoder::new_64k();
        let mut chunked = Vec::new();
        let mut scratch = [0i16; 6];
        for chunk in encoded.chunks(3) {
            let written = stream_decoder
                .decode_i16(chunk, &mut scratch, false)
                .unwrap();
            chunked.extend_from_slice(&scratch[..written]);
        }

        assert_eq!(chunked, whole);
    }

    #[test]
    fn decoder_trait_supports_i16_i32_and_f32_output() {
        let input = samples();
        let mut encoder = G722Encoder::new_64k();
        let mut encoded = Vec::new();
        encoder.encode_to_vec(&input, &mut encoded).unwrap();
        encoder.flush_to_vec(&mut encoded).unwrap();

        let mut decoder_i16 = G722Decoder::new_64k();
        let mut decoder_i32 = G722Decoder::new_64k();
        let mut decoder_f32 = G722Decoder::new_64k();
        let mut i16_out = vec![0i16; encoded.len() * 2];
        let mut i32_out = vec![0i32; encoded.len() * 2];
        let mut f32_out = vec![0.0f32; encoded.len() * 2];

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
    fn decode_g722_fixture_and_write_golden_wav() {
        let fixture = fs::read(testdata_path(
            "g722/A_Tusk_is_used_to_make_costly_gifts.g722",
        ))
        .unwrap();
        assert!(!fixture.is_empty(), "G.722 fixture missing or empty");

        let mut decoder = G722Decoder::new_64k();
        let mut decoded = Vec::new();
        let mut scratch = [0i16; 254];

        for chunk in fixture.chunks(127) {
            let written = decoder.decode_i16(chunk, &mut scratch, false).unwrap();
            decoded.extend_from_slice(&scratch[..written]);
        }

        assert_eq!(decoded.len(), fixture.len() * 2);
        assert!(decoded.iter().any(|&sample| sample != 0));

        let wav = soundkit::wav::generate_wav_buffer(
            &soundkit::audio_types::PcmData::I16(vec![decoded]),
            G722_SAMPLE_RATE,
        )
        .unwrap();
        let output_path = golden_path("g722/A_Tusk_is_used_to_make_costly_gifts.decoded.wav");
        fs::create_dir_all(output_path.parent().unwrap()).unwrap();
        fs::write(output_path, wav).unwrap();
    }
}
