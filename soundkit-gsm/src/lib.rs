use soundkit::audio_packet::{Decoder, Encoder};
use std::ptr::NonNull;

pub const GSM_SAMPLE_RATE: u32 = 8_000;
pub const GSM_CHANNELS: u8 = 1;
pub const GSM_FRAME_SAMPLES: usize = 160;
pub const GSM_FRAME_BYTES: usize = 33;
pub const GSM_MS_FRAME_BYTES: usize = 65;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GsmVariant {
    /// Standard ETSI GSM 06.10 framing: one 33-byte packet per 160 samples.
    Standard,
    /// Microsoft WAV-49 framing: one 65-byte packet per two 160-sample frames.
    Microsoft,
}

impl GsmVariant {
    fn packet_bytes(self) -> usize {
        match self {
            Self::Standard => GSM_FRAME_BYTES,
            Self::Microsoft => GSM_MS_FRAME_BYTES,
        }
    }

    fn samples_per_packet(self) -> usize {
        match self {
            Self::Standard => GSM_FRAME_SAMPLES,
            Self::Microsoft => GSM_FRAME_SAMPLES * 2,
        }
    }
}

struct GsmState {
    ptr: NonNull<gsm_sys::GsmState>,
}

impl GsmState {
    fn new() -> Result<Self, String> {
        let ptr = unsafe { gsm_sys::gsm_create() };
        let ptr = NonNull::new(ptr).ok_or_else(|| "failed to initialize GSM state".to_string())?;
        Ok(Self { ptr })
    }

    fn encode_standard(&mut self, samples: &[i16; GSM_FRAME_SAMPLES]) -> [u8; GSM_FRAME_BYTES] {
        let mut input = *samples;
        let mut output = [0u8; GSM_FRAME_BYTES];
        unsafe { gsm_sys::gsm_encode(self.ptr.as_ptr(), input.as_mut_ptr(), output.as_mut_ptr()) };
        output
    }

    fn decode_standard(
        &mut self,
        frame: &[u8; GSM_FRAME_BYTES],
    ) -> Result<[i16; GSM_FRAME_SAMPLES], String> {
        let mut input = *frame;
        let mut output = [0i16; GSM_FRAME_SAMPLES];
        let status = unsafe {
            gsm_sys::gsm_decode(self.ptr.as_ptr(), input.as_mut_ptr(), output.as_mut_ptr())
        };
        if status < 0 {
            return Err("GSM decoder rejected frame".to_string());
        }
        Ok(output)
    }
}

impl Drop for GsmState {
    fn drop(&mut self) {
        unsafe { gsm_sys::gsm_destroy(self.ptr.as_ptr()) };
    }
}

// The libgsm state is owned exclusively and accessed only via `&mut self`.
unsafe impl Send for GsmState {}

pub struct GsmEncoder {
    state: GsmState,
    variant: GsmVariant,
    pending_samples: Vec<i16>,
    pending_ms_frame: Option<[u8; GSM_FRAME_BYTES]>,
}

impl GsmEncoder {
    pub fn try_new(variant: GsmVariant) -> Result<Self, String> {
        Ok(Self {
            state: GsmState::new()?,
            variant,
            pending_samples: Vec::with_capacity(GSM_FRAME_SAMPLES),
            pending_ms_frame: None,
        })
    }

    pub fn new_standard() -> Self {
        Self::try_new(GsmVariant::Standard).expect("failed to initialize GSM encoder")
    }

    pub fn new_microsoft() -> Self {
        Self::try_new(GsmVariant::Microsoft).expect("failed to initialize GSM-MS encoder")
    }

    pub fn variant(&self) -> GsmVariant {
        self.variant
    }

    pub fn encode_to_vec(&mut self, input: &[i16], output: &mut Vec<u8>) -> Result<usize, String> {
        let start = output.len();
        let required = self.required_encode_bytes(input.len(), false);
        output.resize(start + required, 0);
        let written = self.encode_i16(input, &mut output[start..])?;
        output.truncate(start + written);
        Ok(written)
    }

    pub fn flush_into(&mut self, output: &mut [u8]) -> Result<usize, String> {
        let required = self.required_encode_bytes(0, true);
        if output.len() < required {
            return Err(format!(
                "Output buffer too small for GSM flush: need {}, have {}",
                required,
                output.len()
            ));
        }

        let mut written = 0;
        if !self.pending_samples.is_empty() {
            let mut frame = [0i16; GSM_FRAME_SAMPLES];
            frame[..self.pending_samples.len()].copy_from_slice(&self.pending_samples);
            self.pending_samples.clear();
            let encoded = self.state.encode_standard(&frame);
            self.emit_encoded_standard(encoded, output, &mut written)?;
        }

        if self.variant == GsmVariant::Microsoft {
            if let Some(first) = self.pending_ms_frame.take() {
                let silence = [0i16; GSM_FRAME_SAMPLES];
                let second = self.state.encode_standard(&silence);
                let packet = pack_ms_pair(&first, &second);
                let end = written + GSM_MS_FRAME_BYTES;
                output[written..end].copy_from_slice(&packet);
                written = end;
            }
        }

        Ok(written)
    }

    pub fn flush_to_vec(&mut self, output: &mut Vec<u8>) -> Result<usize, String> {
        let start = output.len();
        let required = self.required_encode_bytes(0, true);
        output.resize(start + required, 0);
        let written = self.flush_into(&mut output[start..])?;
        output.truncate(start + written);
        Ok(written)
    }

    fn required_encode_bytes(&self, input_samples: usize, flush: bool) -> usize {
        let total_samples = self.pending_samples.len() + input_samples;
        let mut complete_frames = total_samples / GSM_FRAME_SAMPLES;
        if flush && total_samples % GSM_FRAME_SAMPLES != 0 {
            complete_frames += 1;
        }

        match self.variant {
            GsmVariant::Standard => complete_frames * GSM_FRAME_BYTES,
            GsmVariant::Microsoft => {
                let pending = usize::from(self.pending_ms_frame.is_some());
                let frames = pending + complete_frames;
                let packets = if flush {
                    frames.div_ceil(2)
                } else {
                    frames / 2
                };
                packets * GSM_MS_FRAME_BYTES
            }
        }
    }

    fn emit_encoded_standard(
        &mut self,
        frame: [u8; GSM_FRAME_BYTES],
        output: &mut [u8],
        written: &mut usize,
    ) -> Result<(), String> {
        match self.variant {
            GsmVariant::Standard => {
                let end = *written + GSM_FRAME_BYTES;
                if end > output.len() {
                    return Err(format!(
                        "Output buffer too small for GSM encode: need {}, have {}",
                        end,
                        output.len()
                    ));
                }
                output[*written..end].copy_from_slice(&frame);
                *written = end;
            }
            GsmVariant::Microsoft => {
                if let Some(first) = self.pending_ms_frame.take() {
                    let packet = pack_ms_pair(&first, &frame);
                    let end = *written + GSM_MS_FRAME_BYTES;
                    if end > output.len() {
                        return Err(format!(
                            "Output buffer too small for GSM-MS encode: need {}, have {}",
                            end,
                            output.len()
                        ));
                    }
                    output[*written..end].copy_from_slice(&packet);
                    *written = end;
                } else {
                    self.pending_ms_frame = Some(frame);
                }
            }
        }
        Ok(())
    }
}

impl Default for GsmEncoder {
    fn default() -> Self {
        Self::new_standard()
    }
}

impl Encoder for GsmEncoder {
    fn new(
        _sample_rate: u32,
        _bits_per_sample: u32,
        _channels: u32,
        _frame_size: u32,
        _bitrate: u32,
    ) -> Self {
        Self::new_standard()
    }

    fn init(&mut self) -> Result<(), String> {
        Ok(())
    }

    fn encode_i16(&mut self, input: &[i16], output: &mut [u8]) -> Result<usize, String> {
        let required = self.required_encode_bytes(input.len(), false);
        if output.len() < required {
            return Err(format!(
                "Output buffer too small for GSM encode: need {}, have {}",
                required,
                output.len()
            ));
        }

        let total_samples = self.pending_samples.len() + input.len();
        let complete_samples = (total_samples / GSM_FRAME_SAMPLES) * GSM_FRAME_SAMPLES;

        let mut samples = Vec::with_capacity(total_samples);
        samples.extend_from_slice(&self.pending_samples);
        samples.extend_from_slice(input);

        let mut written = 0;
        for chunk in samples[..complete_samples].chunks_exact(GSM_FRAME_SAMPLES) {
            let mut frame = [0i16; GSM_FRAME_SAMPLES];
            frame.copy_from_slice(chunk);
            let encoded = self.state.encode_standard(&frame);
            self.emit_encoded_standard(encoded, output, &mut written)?;
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
        self.state = GsmState::new()?;
        self.pending_samples.clear();
        self.pending_ms_frame = None;
        Ok(())
    }
}

pub struct GsmDecoder {
    state: GsmState,
    variant: GsmVariant,
    pending_bytes: Vec<u8>,
}

impl GsmDecoder {
    pub fn try_new(variant: GsmVariant) -> Result<Self, String> {
        Ok(Self {
            state: GsmState::new()?,
            variant,
            pending_bytes: Vec::with_capacity(variant.packet_bytes()),
        })
    }

    pub fn new_standard() -> Self {
        Self::try_new(GsmVariant::Standard).expect("failed to initialize GSM decoder")
    }

    pub fn new_microsoft() -> Self {
        Self::try_new(GsmVariant::Microsoft).expect("failed to initialize GSM-MS decoder")
    }

    pub fn variant(&self) -> GsmVariant {
        self.variant
    }

    pub fn sample_rate(&self) -> u32 {
        GSM_SAMPLE_RATE
    }

    pub fn channels(&self) -> u8 {
        GSM_CHANNELS
    }

    pub fn flush(&mut self) -> Result<(), String> {
        if self.pending_bytes.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "GSM stream ended with {} trailing partial-frame byte(s)",
                self.pending_bytes.len()
            ))
        }
    }
}

impl Default for GsmDecoder {
    fn default() -> Self {
        Self::new_standard()
    }
}

impl Decoder for GsmDecoder {
    fn decode_i16(
        &mut self,
        input: &[u8],
        output: &mut [i16],
        _fec: bool,
    ) -> Result<usize, String> {
        let packet_bytes = self.variant.packet_bytes();
        let total_bytes = self.pending_bytes.len() + input.len();
        let complete_bytes = (total_bytes / packet_bytes) * packet_bytes;
        let packet_count = complete_bytes / packet_bytes;
        let required = packet_count * self.variant.samples_per_packet();

        if output.len() < required {
            return Err(format!(
                "Output buffer too small for GSM decode: need {}, have {}",
                required,
                output.len()
            ));
        }

        let mut bytes = Vec::with_capacity(total_bytes);
        bytes.extend_from_slice(&self.pending_bytes);
        bytes.extend_from_slice(input);

        let mut written = 0;
        for packet_data in bytes[..complete_bytes].chunks_exact(packet_bytes) {
            match self.variant {
                GsmVariant::Standard => {
                    let mut frame = [0u8; GSM_FRAME_BYTES];
                    frame.copy_from_slice(packet_data);
                    let decoded = self.state.decode_standard(&frame)?;
                    output[written..written + GSM_FRAME_SAMPLES].copy_from_slice(&decoded);
                    written += GSM_FRAME_SAMPLES;
                }
                GsmVariant::Microsoft => {
                    let mut packet = [0u8; GSM_MS_FRAME_BYTES];
                    packet.copy_from_slice(packet_data);
                    let [first, second] = unpack_ms_pair(&packet);
                    for frame in [first, second] {
                        let decoded = self.state.decode_standard(&frame)?;
                        output[written..written + GSM_FRAME_SAMPLES].copy_from_slice(&decoded);
                        written += GSM_FRAME_SAMPLES;
                    }
                }
            }
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
        let required = ((self.pending_bytes.len() + input.len()) / self.variant.packet_bytes())
            * self.variant.samples_per_packet();
        if output.len() < required {
            return Err(format!(
                "Output buffer too small for GSM decode: need {}, have {}",
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
        let required = ((self.pending_bytes.len() + input.len()) / self.variant.packet_bytes())
            * self.variant.samples_per_packet();
        if output.len() < required {
            return Err(format!(
                "Output buffer too small for GSM decode: need {}, have {}",
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

fn read_bit(input: &[u8], bit_index: usize) -> u8 {
    (input[bit_index / 8] >> (7 - (bit_index % 8))) & 1
}

fn write_bit(output: &mut [u8], bit_index: usize, bit: u8) {
    if bit != 0 {
        output[bit_index / 8] |= 1 << (7 - (bit_index % 8));
    }
}

fn pack_ms_pair(
    first: &[u8; GSM_FRAME_BYTES],
    second: &[u8; GSM_FRAME_BYTES],
) -> [u8; GSM_MS_FRAME_BYTES] {
    let mut output = [0u8; GSM_MS_FRAME_BYTES];
    for index in 0..260 {
        write_bit(&mut output, index, read_bit(first, index + 4));
        write_bit(&mut output, index + 260, read_bit(second, index + 4));
    }
    output
}

fn unpack_ms_pair(packet: &[u8; GSM_MS_FRAME_BYTES]) -> [[u8; GSM_FRAME_BYTES]; 2] {
    let mut first = [0u8; GSM_FRAME_BYTES];
    let mut second = [0u8; GSM_FRAME_BYTES];
    first[0] = 0xD0;
    second[0] = 0xD0;
    for index in 0..260 {
        write_bit(&mut first, index + 4, read_bit(packet, index));
        write_bit(&mut second, index + 4, read_bit(packet, index + 260));
    }
    [first, second]
}

#[cfg(test)]
fn i16_to_s16le(input: &[i16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(input.len() * 2);
    for &sample in input {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    bytes
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

    fn samples() -> Vec<i16> {
        (0..397)
            .map(|index| {
                let phase = index as f32 / 80.0 * std::f32::consts::TAU;
                (phase.sin() * 10_000.0) as i16
            })
            .collect()
    }

    fn encode_fixture(samples: &[i16], variant: GsmVariant) -> Vec<u8> {
        let mut encoder = GsmEncoder::try_new(variant).unwrap();
        let mut encoded = Vec::new();
        for chunk in samples.chunks(173) {
            encoder.encode_to_vec(chunk, &mut encoded).unwrap();
        }
        encoder.flush_to_vec(&mut encoded).unwrap();
        encoded
    }

    #[test]
    fn streaming_encoder_matches_padded_whole_encode() {
        let input = samples();
        let mut padded = input.clone();
        padded.resize(
            input.len().div_ceil(GSM_FRAME_SAMPLES) * GSM_FRAME_SAMPLES,
            0,
        );

        let mut whole_encoder = GsmEncoder::new_standard();
        let mut whole = vec![0u8; padded.len() / GSM_FRAME_SAMPLES * GSM_FRAME_BYTES];
        let whole_len = whole_encoder.encode_i16(&padded, &mut whole).unwrap();
        whole.truncate(whole_len);

        let mut stream_encoder = GsmEncoder::new_standard();
        let mut chunked = Vec::new();
        let mut scratch = [0u8; GSM_FRAME_BYTES * 3];
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
        let encoded = encode_fixture(&input, GsmVariant::Standard);

        let mut whole_decoder = GsmDecoder::new_standard();
        let mut whole = vec![0i16; encoded.len() / GSM_FRAME_BYTES * GSM_FRAME_SAMPLES];
        let whole_len = whole_decoder
            .decode_i16(&encoded, &mut whole, false)
            .unwrap();
        whole.truncate(whole_len);

        let mut stream_decoder = GsmDecoder::new_standard();
        let mut chunked = Vec::new();
        let mut scratch = [0i16; GSM_FRAME_SAMPLES * 2];
        for chunk in encoded.chunks(19) {
            let written = stream_decoder
                .decode_i16(chunk, &mut scratch, false)
                .unwrap();
            chunked.extend_from_slice(&scratch[..written]);
        }
        stream_decoder.flush().unwrap();

        assert_eq!(chunked, whole);
    }

    #[test]
    fn microsoft_framing_roundtrips_in_chunks() {
        let input = samples();
        let encoded = encode_fixture(&input, GsmVariant::Microsoft);
        assert_eq!(encoded.len() % GSM_MS_FRAME_BYTES, 0);

        let mut decoder = GsmDecoder::new_microsoft();
        let mut decoded = Vec::new();
        let mut scratch = [0i16; GSM_FRAME_SAMPLES * 4];
        for chunk in encoded.chunks(17) {
            let written = decoder.decode_i16(chunk, &mut scratch, false).unwrap();
            decoded.extend_from_slice(&scratch[..written]);
        }
        decoder.flush().unwrap();

        assert!(!decoded.is_empty());
        assert_eq!(
            decoded.len(),
            encoded.len() / GSM_MS_FRAME_BYTES * GSM_FRAME_SAMPLES * 2
        );
    }

    #[test]
    fn decoder_reports_partial_trailing_frame_on_flush() {
        let mut decoder = GsmDecoder::new_standard();
        let mut output = [0i16; GSM_FRAME_SAMPLES];
        assert_eq!(
            decoder
                .decode_i16(&[0xD0, 0, 0], &mut output, false)
                .unwrap(),
            0
        );
        assert!(decoder.flush().unwrap_err().contains("partial-frame"));
    }

    #[test]
    fn decoder_trait_supports_i16_i32_and_f32_output() {
        let input = samples();
        let encoded = encode_fixture(&input, GsmVariant::Standard);

        let sample_count = encoded.len() / GSM_FRAME_BYTES * GSM_FRAME_SAMPLES;
        let mut decoder_i16 = GsmDecoder::new_standard();
        let mut decoder_i32 = GsmDecoder::new_standard();
        let mut decoder_f32 = GsmDecoder::new_standard();
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
    #[ignore = "regenerates the committed GSM fixture from the 8 kHz linear16 fixture using SoX/libgsm"]
    fn generate_gsm_fixture_from_linear16_8() {
        let input = testdata_path("linear16_8/A_Tusk_is_used_to_make_costly_gifts.s16le");
        let output = testdata_path("gsm/A_Tusk_is_used_to_make_costly_gifts.gsm");
        fs::create_dir_all(output.parent().unwrap()).unwrap();

        let status = Command::new("sox")
            .args([
                "-t",
                "raw",
                "-r",
                "8000",
                "-e",
                "signed-integer",
                "-b",
                "16",
                "-c",
                "1",
            ])
            .arg(&input)
            .args(["-t", "gsm"])
            .arg(&output)
            .status()
            .unwrap();
        assert!(status.success());
    }

    #[test]
    fn decode_gsm_fixture_and_write_golden_wav() {
        let fixture =
            fs::read(testdata_path("gsm/A_Tusk_is_used_to_make_costly_gifts.gsm")).unwrap();
        assert!(!fixture.is_empty(), "GSM fixture missing or empty");

        let mut decoder = GsmDecoder::new_standard();
        let mut decoded = Vec::new();
        let mut scratch = [0i16; GSM_FRAME_SAMPLES * 3];

        for chunk in fixture.chunks(41) {
            let written = decoder.decode_i16(chunk, &mut scratch, false).unwrap();
            decoded.extend_from_slice(&scratch[..written]);
        }
        decoder.flush().unwrap();

        assert_eq!(
            decoded.len(),
            fixture.len() / GSM_FRAME_BYTES * GSM_FRAME_SAMPLES
        );
        assert!(decoded.iter().any(|&sample| sample != 0));

        let wav = soundkit::wav::generate_wav_buffer(
            &soundkit::audio_types::PcmData::I16(vec![decoded]),
            GSM_SAMPLE_RATE,
        )
        .unwrap();
        let output_path = golden_path("gsm/A_Tusk_is_used_to_make_costly_gifts.decoded.wav");
        fs::create_dir_all(output_path.parent().unwrap()).unwrap();
        fs::write(output_path, wav).unwrap();
    }

    #[test]
    fn ffmpeg_can_decode_gsm_fixture_when_available() {
        let input_path = testdata_path("gsm/A_Tusk_is_used_to_make_costly_gifts.gsm");
        let output_path = std::env::temp_dir().join("soundkit-gsm-fixture.s16le");
        let status = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "gsm",
                "-i",
            ])
            .arg(&input_path)
            .args(["-f", "s16le", "-acodec", "pcm_s16le"])
            .arg(&output_path)
            .status();

        match status {
            Ok(status) if status.success() => {}
            _ => {
                eprintln!("skipping: ffmpeg GSM decoder unavailable");
                return;
            }
        }

        let fixture = fs::read(&input_path).unwrap();
        let mut decoder = GsmDecoder::new_standard();
        let mut decoded = vec![0i16; fixture.len() / GSM_FRAME_BYTES * GSM_FRAME_SAMPLES];
        let decoded_len = decoder.decode_i16(&fixture, &mut decoded, false).unwrap();
        decoded.truncate(decoded_len);
        let native = i16_to_s16le(&decoded);
        let ffmpeg = fs::read(&output_path).unwrap();

        assert!(!ffmpeg.is_empty(), "ffmpeg produced empty GSM decode");
        assert_eq!(native, ffmpeg, "native GSM decode differs from ffmpeg");
    }
}
