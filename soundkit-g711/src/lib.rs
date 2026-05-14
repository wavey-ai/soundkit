use soundkit::audio_packet::{Decoder, Encoder};

const ULAW_BIAS: i32 = 0x84;
const ULAW_CLIP: i32 = 32635;
const SEGMENT_ENDS: [i32; 8] = [
    0x0000_00ff,
    0x0000_01ff,
    0x0000_03ff,
    0x0000_07ff,
    0x0000_0fff,
    0x0000_1fff,
    0x0000_3fff,
    0x0000_7fff,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum G711Law {
    MuLaw,
    ALaw,
}

pub fn encode_sample(law: G711Law, sample: i16) -> u8 {
    match law {
        G711Law::MuLaw => encode_mulaw_sample(sample),
        G711Law::ALaw => encode_alaw_sample(sample),
    }
}

pub fn decode_sample(law: G711Law, sample: u8) -> i16 {
    match law {
        G711Law::MuLaw => decode_mulaw_sample(sample),
        G711Law::ALaw => decode_alaw_sample(sample),
    }
}

pub fn encode_i16(law: G711Law, input: &[i16], output: &mut [u8]) -> Result<usize, String> {
    if output.len() < input.len() {
        return Err(format!(
            "Output buffer too small for G.711 encode: need {}, have {}",
            input.len(),
            output.len()
        ));
    }

    for (dst, &sample) in output.iter_mut().zip(input) {
        *dst = encode_sample(law, sample);
    }

    Ok(input.len())
}

pub fn decode_i16(law: G711Law, input: &[u8], output: &mut [i16]) -> Result<usize, String> {
    if output.len() < input.len() {
        return Err(format!(
            "Output buffer too small for G.711 decode: need {}, have {}",
            input.len(),
            output.len()
        ));
    }

    for (dst, &sample) in output.iter_mut().zip(input) {
        *dst = decode_sample(law, sample);
    }

    Ok(input.len())
}

#[derive(Debug)]
pub struct G711Encoder {
    law: G711Law,
    sample_rate: u32,
    channels: u8,
}

impl G711Encoder {
    pub fn new_with_law(law: G711Law, sample_rate: u32, channels: u8) -> Self {
        Self {
            law,
            sample_rate,
            channels,
        }
    }

    pub fn new_mulaw(sample_rate: u32, channels: u8) -> Self {
        Self::new_with_law(G711Law::MuLaw, sample_rate, channels)
    }

    pub fn new_alaw(sample_rate: u32, channels: u8) -> Self {
        Self::new_with_law(G711Law::ALaw, sample_rate, channels)
    }

    pub fn law(&self) -> G711Law {
        self.law
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn channels(&self) -> u8 {
        self.channels
    }
}

impl Encoder for G711Encoder {
    fn new(
        sample_rate: u32,
        _bits_per_sample: u32,
        channels: u32,
        _frame_size: u32,
        _bitrate: u32,
    ) -> Self {
        Self::new_mulaw(sample_rate, channels as u8)
    }

    fn init(&mut self) -> Result<(), String> {
        Ok(())
    }

    fn encode_i16(&mut self, input: &[i16], output: &mut [u8]) -> Result<usize, String> {
        encode_i16(self.law, input, output)
    }

    fn encode_i32(&mut self, input: &[i32], output: &mut [u8]) -> Result<usize, String> {
        if output.len() < input.len() {
            return Err(format!(
                "Output buffer too small for G.711 encode: need {}, have {}",
                input.len(),
                output.len()
            ));
        }

        for (dst, &sample) in output.iter_mut().zip(input) {
            *dst = encode_sample(self.law, (sample >> 16) as i16);
        }

        Ok(input.len())
    }

    fn reset(&mut self) -> Result<(), String> {
        Ok(())
    }
}

#[derive(Debug)]
pub struct G711Decoder {
    law: G711Law,
    sample_rate: u32,
    channels: u8,
}

impl G711Decoder {
    pub fn new_with_law(law: G711Law, sample_rate: u32, channels: u8) -> Self {
        Self {
            law,
            sample_rate,
            channels,
        }
    }

    pub fn new_mulaw(sample_rate: u32, channels: u8) -> Self {
        Self::new_with_law(G711Law::MuLaw, sample_rate, channels)
    }

    pub fn new_alaw(sample_rate: u32, channels: u8) -> Self {
        Self::new_with_law(G711Law::ALaw, sample_rate, channels)
    }

    pub fn law(&self) -> G711Law {
        self.law
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn channels(&self) -> u8 {
        self.channels
    }
}

impl Decoder for G711Decoder {
    fn decode_i16(
        &mut self,
        input: &[u8],
        output: &mut [i16],
        _fec: bool,
    ) -> Result<usize, String> {
        decode_i16(self.law, input, output)
    }

    fn decode_i32(
        &mut self,
        input: &[u8],
        output: &mut [i32],
        _fec: bool,
    ) -> Result<usize, String> {
        if output.len() < input.len() {
            return Err(format!(
                "Output buffer too small for G.711 decode: need {}, have {}",
                input.len(),
                output.len()
            ));
        }

        for (dst, &sample) in output.iter_mut().zip(input) {
            *dst = i32::from(decode_sample(self.law, sample)) << 16;
        }

        Ok(input.len())
    }

    fn decode_f32(
        &mut self,
        input: &[u8],
        output: &mut [f32],
        _fec: bool,
    ) -> Result<usize, String> {
        if output.len() < input.len() {
            return Err(format!(
                "Output buffer too small for G.711 decode: need {}, have {}",
                input.len(),
                output.len()
            ));
        }

        for (dst, &sample) in output.iter_mut().zip(input) {
            *dst = f32::from(decode_sample(self.law, sample)) / 32768.0;
        }

        Ok(input.len())
    }
}

fn encode_mulaw_sample(sample: i16) -> u8 {
    let mut pcm = i32::from(sample);
    let mask = if pcm < 0 {
        pcm = ULAW_BIAS - pcm;
        0x7f
    } else {
        pcm += ULAW_BIAS;
        0xff
    };

    let pcm = pcm.min(ULAW_CLIP);
    let segment = segment_for(pcm);
    let encoded = if segment >= 8 {
        0x7f
    } else {
        ((segment as u8) << 4) | (((pcm >> (segment + 3)) as u8) & 0x0f)
    };

    encoded ^ mask
}

fn decode_mulaw_sample(sample: u8) -> i16 {
    let sample = !sample;
    let mut magnitude = i32::from(sample & 0x0f) << 3;
    magnitude += ULAW_BIAS;
    magnitude <<= u32::from((sample & 0x70) >> 4);

    let decoded = if sample & 0x80 != 0 {
        ULAW_BIAS - magnitude
    } else {
        magnitude - ULAW_BIAS
    };

    decoded as i16
}

fn encode_alaw_sample(sample: i16) -> u8 {
    let mut pcm = i32::from(sample);
    let mask = if pcm >= 0 {
        0xd5
    } else {
        pcm = -pcm - 1;
        0x55
    };

    let segment = segment_for(pcm);
    let encoded = if segment >= 8 {
        0x7f
    } else {
        let mut value = (segment as u8) << 4;
        if segment < 2 {
            value |= ((pcm >> 4) as u8) & 0x0f;
        } else {
            value |= ((pcm >> (segment + 3)) as u8) & 0x0f;
        }
        value
    };

    encoded ^ mask
}

fn decode_alaw_sample(sample: u8) -> i16 {
    let sample = sample ^ 0x55;
    let segment = (sample & 0x70) >> 4;
    let mut magnitude = i32::from(sample & 0x0f) << 4;

    match segment {
        0 => magnitude += 8,
        1 => magnitude += 0x108,
        _ => {
            magnitude += 0x108;
            magnitude <<= u32::from(segment - 1);
        }
    }

    if sample & 0x80 != 0 {
        magnitude as i16
    } else {
        -(magnitude as i16)
    }
}

fn segment_for(sample: i32) -> usize {
    SEGMENT_ENDS
        .iter()
        .position(|&end| sample <= end)
        .unwrap_or(8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    const SAMPLES: [i16; 17] = [
        i16::MIN,
        -30000,
        -20000,
        -12000,
        -4096,
        -1024,
        -32,
        -1,
        0,
        1,
        32,
        1024,
        4096,
        12000,
        20000,
        30000,
        i16::MAX,
    ];

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

    fn load_linear16_fixture() -> Vec<i16> {
        let bytes = fs::read(testdata_path(
            "linear16/A_Tusk_is_used_to_make_costly_gifts.s16le",
        ))
        .unwrap();
        assert!(!bytes.is_empty(), "linear16 fixture missing or empty");
        bytes
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
            .collect()
    }

    fn decode_fixture_to_wav(law: G711Law, fixture_path: &str, golden_file: &str) {
        let fixture = fs::read(testdata_path(fixture_path)).unwrap();
        assert!(!fixture.is_empty(), "{fixture_path} missing or empty");

        let mut decoder = G711Decoder::new_with_law(law, 8_000, 1);
        let mut decoded = Vec::new();
        let mut scratch = [0i16; 137];

        for chunk in fixture.chunks(137) {
            let written = decoder.decode_i16(chunk, &mut scratch, false).unwrap();
            decoded.extend_from_slice(&scratch[..written]);
        }

        assert_eq!(decoded.len(), fixture.len());
        assert!(decoded.iter().any(|&sample| sample != 0));

        let wav = soundkit::wav::generate_wav_buffer(
            &soundkit::audio_types::PcmData::I16(vec![decoded]),
            8_000,
        )
        .unwrap();
        let output_path = golden_path(golden_file);
        fs::create_dir_all(output_path.parent().unwrap()).unwrap();
        fs::write(output_path, wav).unwrap();
    }

    #[test]
    fn mulaw_known_zero_codes_decode_to_zero() {
        assert_eq!(decode_sample(G711Law::MuLaw, 0xff), 0);
        assert_eq!(decode_sample(G711Law::MuLaw, 0x7f), 0);
        assert_eq!(encode_sample(G711Law::MuLaw, 0), 0xff);
    }

    #[test]
    fn alaw_known_zero_codes_decode_to_small_quantized_values() {
        assert_eq!(decode_sample(G711Law::ALaw, 0xd5), 8);
        assert_eq!(decode_sample(G711Law::ALaw, 0x55), -8);
        assert_eq!(encode_sample(G711Law::ALaw, 0), 0xd5);
    }

    #[test]
    fn mulaw_roundtrip_is_stable_after_quantization() {
        for sample in SAMPLES {
            let encoded = encode_sample(G711Law::MuLaw, sample);
            let decoded = decode_sample(G711Law::MuLaw, encoded);
            let redecoded = decode_sample(G711Law::MuLaw, encode_sample(G711Law::MuLaw, decoded));
            assert_eq!(redecoded, decoded);
        }
    }

    #[test]
    fn alaw_roundtrip_is_stable_after_quantization() {
        for sample in SAMPLES {
            let encoded = encode_sample(G711Law::ALaw, sample);
            let decoded = decode_sample(G711Law::ALaw, encoded);
            let redecoded = decode_sample(G711Law::ALaw, encode_sample(G711Law::ALaw, decoded));
            assert_eq!(redecoded, decoded);
        }
    }

    #[test]
    fn streaming_chunk_boundaries_do_not_change_encoded_output() {
        let mut encoder = G711Encoder::new_mulaw(8_000, 1);
        let mut whole = vec![0u8; SAMPLES.len()];
        let whole_len = encoder.encode_i16(&SAMPLES, &mut whole).unwrap();
        whole.truncate(whole_len);

        let mut chunked = Vec::new();
        let mut scratch = [0u8; 4];
        for chunk in SAMPLES.chunks(3) {
            let written = encoder.encode_i16(chunk, &mut scratch).unwrap();
            chunked.extend_from_slice(&scratch[..written]);
        }

        assert_eq!(chunked, whole);
    }

    #[test]
    fn decoder_trait_supports_i16_i32_and_f32_output() {
        let encoded = [0xff, 0xd7, 0xb7, 0x37, 0x17, 0x7f];
        let mut decoder = G711Decoder::new_mulaw(8_000, 1);

        let mut i16_out = [0i16; 6];
        let mut i32_out = [0i32; 6];
        let mut f32_out = [0.0f32; 6];

        assert_eq!(
            decoder.decode_i16(&encoded, &mut i16_out, false).unwrap(),
            6
        );
        assert_eq!(
            decoder.decode_i32(&encoded, &mut i32_out, false).unwrap(),
            6
        );
        assert_eq!(
            decoder.decode_f32(&encoded, &mut f32_out, false).unwrap(),
            6
        );

        for i in 0..encoded.len() {
            assert_eq!(i32_out[i], i32::from(i16_out[i]) << 16);
            assert!((f32_out[i] - f32::from(i16_out[i]) / 32768.0).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn fixture_roundtrips_through_mulaw_and_alaw_streaming() {
        let samples = load_linear16_fixture();

        for law in [G711Law::MuLaw, G711Law::ALaw] {
            let mut encoder = G711Encoder::new_with_law(law, 16_000, 1);
            let mut encoded = Vec::with_capacity(samples.len());
            let mut encode_scratch = [0u8; 257];
            for chunk in samples.chunks(257) {
                let written = encoder.encode_i16(chunk, &mut encode_scratch).unwrap();
                encoded.extend_from_slice(&encode_scratch[..written]);
            }
            assert_eq!(encoded.len(), samples.len());

            let mut decoder = G711Decoder::new_with_law(law, 16_000, 1);
            let mut decoded = Vec::with_capacity(samples.len());
            let mut decode_scratch = [0i16; 113];
            for chunk in encoded.chunks(113) {
                let written = decoder
                    .decode_i16(chunk, &mut decode_scratch, false)
                    .unwrap();
                decoded.extend_from_slice(&decode_scratch[..written]);
            }

            assert_eq!(decoded.len(), samples.len());
            assert!(
                decoded.iter().any(|&sample| sample != 0),
                "fixture decoded to silence for {:?}",
                law
            );
        }
    }

    #[test]
    fn decode_mulaw_fixture_and_write_golden_wav() {
        decode_fixture_to_wav(
            G711Law::MuLaw,
            "g711_ulaw/A_Tusk_is_used_to_make_costly_gifts.ulaw",
            "g711_ulaw/A_Tusk_is_used_to_make_costly_gifts.decoded.wav",
        );
    }

    #[test]
    fn decode_alaw_fixture_and_write_golden_wav() {
        decode_fixture_to_wav(
            G711Law::ALaw,
            "g711_alaw/A_Tusk_is_used_to_make_costly_gifts.alaw",
            "g711_alaw/A_Tusk_is_used_to_make_costly_gifts.decoded.wav",
        );
    }
}
