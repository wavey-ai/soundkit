use soundkit::audio_packet::{Decoder, Encoder};
use std::ffi::c_void;
use std::os::raw::c_int;

pub const AMR_NB_SAMPLE_RATE: u32 = 8_000;
pub const AMR_NB_CHANNELS: u8 = 1;
pub const AMR_NB_FRAME_SAMPLES: usize = 160;
pub const AMR_NB_FILE_MAGIC: &[u8; 6] = b"#!AMR\n";

const AMR_NB_FRAME_BYTES: [usize; 16] = [13, 14, 16, 18, 20, 21, 27, 32, 6, 1, 1, 1, 1, 1, 1, 1];
const AMR_NB_MAX_FRAME_BYTES: usize = 32;

#[link(name = "opencore-amrnb")]
extern "C" {
    fn Decoder_Interface_init() -> *mut c_void;
    fn Decoder_Interface_exit(state: *mut c_void);
    fn Decoder_Interface_Decode(state: *mut c_void, input: *const u8, output: *mut i16, bfi: c_int);

    fn Encoder_Interface_init(dtx: c_int) -> *mut c_void;
    fn Encoder_Interface_exit(state: *mut c_void);
    fn Encoder_Interface_Encode(
        state: *mut c_void,
        mode: c_int,
        speech: *const i16,
        output: *mut u8,
        force_speech: c_int,
    ) -> c_int;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum AmrNbMode {
    Mr475 = 0,
    Mr515 = 1,
    Mr59 = 2,
    Mr67 = 3,
    Mr74 = 4,
    Mr795 = 5,
    Mr102 = 6,
    Mr122 = 7,
}

impl AmrNbMode {
    pub fn from_bitrate(bit_rate: u32) -> Self {
        match bit_rate {
            0..=4_950 => Self::Mr475,
            4_951..=5_500 => Self::Mr515,
            5_501..=6_300 => Self::Mr59,
            6_301..=7_050 => Self::Mr67,
            7_051..=7_675 => Self::Mr74,
            7_676..=8_875 => Self::Mr795,
            8_876..=11_200 => Self::Mr102,
            _ => Self::Mr122,
        }
    }

    fn as_c_int(self) -> c_int {
        self as c_int
    }
}

pub struct AmrNbEncoder {
    state: *mut c_void,
    mode: AmrNbMode,
    pending_samples: Vec<i16>,
}

// The OpenCORE state is owned exclusively by this wrapper and is only accessed
// through `&mut self`, so moving the wrapper between threads is safe.
unsafe impl Send for AmrNbEncoder {}

impl AmrNbEncoder {
    pub fn try_new(mode: AmrNbMode) -> Result<Self, String> {
        let state = unsafe { Encoder_Interface_init(0) };
        if state.is_null() {
            return Err("failed to initialize AMR-NB encoder".to_string());
        }
        Ok(Self {
            state,
            mode,
            pending_samples: Vec::with_capacity(AMR_NB_FRAME_SAMPLES),
        })
    }

    pub fn new_mr122() -> Self {
        Self::try_new(AmrNbMode::Mr122).expect("failed to initialize AMR-NB encoder")
    }

    pub fn mode(&self) -> AmrNbMode {
        self.mode
    }

    pub fn encode_to_vec(&mut self, input: &[i16], output: &mut Vec<u8>) -> Result<usize, String> {
        let start = output.len();
        let required = ((self.pending_samples.len() + input.len()) / AMR_NB_FRAME_SAMPLES)
            * AMR_NB_MAX_FRAME_BYTES;
        output.resize(start + required, 0);
        let written = self.encode_i16(input, &mut output[start..])?;
        output.truncate(start + written);
        Ok(written)
    }

    pub fn flush_into(&mut self, output: &mut [u8]) -> Result<usize, String> {
        if self.pending_samples.is_empty() {
            return Ok(0);
        }

        if output.len() < AMR_NB_MAX_FRAME_BYTES {
            return Err(format!(
                "Output buffer too small for AMR-NB flush: need {}, have {}",
                AMR_NB_MAX_FRAME_BYTES,
                output.len()
            ));
        }

        let mut frame = [0i16; AMR_NB_FRAME_SAMPLES];
        frame[..self.pending_samples.len()].copy_from_slice(&self.pending_samples);
        self.pending_samples.clear();

        let written = unsafe {
            Encoder_Interface_Encode(
                self.state,
                self.mode.as_c_int(),
                frame.as_ptr(),
                output.as_mut_ptr(),
                1,
            )
        };
        if written < 0 {
            return Err("AMR-NB encoder returned an error".to_string());
        }
        Ok(written as usize)
    }

    pub fn flush_to_vec(&mut self, output: &mut Vec<u8>) -> Result<usize, String> {
        let start = output.len();
        output.resize(start + AMR_NB_MAX_FRAME_BYTES, 0);
        let written = self.flush_into(&mut output[start..])?;
        output.truncate(start + written);
        Ok(written)
    }
}

impl Default for AmrNbEncoder {
    fn default() -> Self {
        Self::new_mr122()
    }
}

impl Drop for AmrNbEncoder {
    fn drop(&mut self) {
        if !self.state.is_null() {
            unsafe { Encoder_Interface_exit(self.state) };
            self.state = std::ptr::null_mut();
        }
    }
}

impl Encoder for AmrNbEncoder {
    fn new(
        _sample_rate: u32,
        _bits_per_sample: u32,
        _channels: u32,
        _frame_size: u32,
        bitrate: u32,
    ) -> Self {
        Self::try_new(AmrNbMode::from_bitrate(bitrate))
            .expect("failed to initialize AMR-NB encoder")
    }

    fn init(&mut self) -> Result<(), String> {
        Ok(())
    }

    fn encode_i16(&mut self, input: &[i16], output: &mut [u8]) -> Result<usize, String> {
        let total_samples = self.pending_samples.len() + input.len();
        let complete_samples = (total_samples / AMR_NB_FRAME_SAMPLES) * AMR_NB_FRAME_SAMPLES;
        let frame_count = complete_samples / AMR_NB_FRAME_SAMPLES;
        let required = frame_count * AMR_NB_MAX_FRAME_BYTES;
        if output.len() < required {
            return Err(format!(
                "Output buffer too small for AMR-NB encode: need {}, have {}",
                required,
                output.len()
            ));
        }

        let mut samples = Vec::with_capacity(total_samples);
        samples.extend_from_slice(&self.pending_samples);
        samples.extend_from_slice(input);

        let mut written = 0;
        for frame in samples[..complete_samples].chunks_exact(AMR_NB_FRAME_SAMPLES) {
            let out = &mut output[written..written + AMR_NB_MAX_FRAME_BYTES];
            let frame_len = unsafe {
                Encoder_Interface_Encode(
                    self.state,
                    self.mode.as_c_int(),
                    frame.as_ptr(),
                    out.as_mut_ptr(),
                    1,
                )
            };
            if frame_len < 0 {
                return Err("AMR-NB encoder returned an error".to_string());
            }
            written += frame_len as usize;
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
        if !self.state.is_null() {
            unsafe { Encoder_Interface_exit(self.state) };
        }
        self.state = unsafe { Encoder_Interface_init(0) };
        if self.state.is_null() {
            return Err("failed to reinitialize AMR-NB encoder".to_string());
        }
        self.pending_samples.clear();
        Ok(())
    }
}

pub struct AmrNbDecoder {
    state: *mut c_void,
    pending_bytes: Vec<u8>,
    checked_magic: bool,
}

// The OpenCORE state is owned exclusively by this wrapper and is only accessed
// through `&mut self`, so moving the wrapper between threads is safe.
unsafe impl Send for AmrNbDecoder {}

impl AmrNbDecoder {
    pub fn try_new() -> Result<Self, String> {
        let state = unsafe { Decoder_Interface_init() };
        if state.is_null() {
            return Err("failed to initialize AMR-NB decoder".to_string());
        }
        Ok(Self {
            state,
            pending_bytes: Vec::with_capacity(AMR_NB_MAX_FRAME_BYTES),
            checked_magic: false,
        })
    }

    pub fn new_decoder() -> Self {
        Self::try_new().expect("failed to initialize AMR-NB decoder")
    }

    pub fn sample_rate(&self) -> u32 {
        AMR_NB_SAMPLE_RATE
    }

    pub fn channels(&self) -> u8 {
        AMR_NB_CHANNELS
    }

    pub fn flush(&mut self) -> Result<(), String> {
        if self.pending_bytes.is_empty()
            || (!self.checked_magic && AMR_NB_FILE_MAGIC.starts_with(&self.pending_bytes))
        {
            Ok(())
        } else {
            Err(format!(
                "AMR-NB stream ended with {} trailing partial-frame byte(s)",
                self.pending_bytes.len()
            ))
        }
    }

    fn strip_magic_if_needed(&mut self) {
        strip_magic(&mut self.pending_bytes, &mut self.checked_magic);
    }

    fn decoded_samples_available(&self, input: &[u8]) -> Result<usize, String> {
        let mut pending = self.pending_bytes.clone();
        let mut checked_magic = self.checked_magic;
        pending.extend_from_slice(input);
        strip_magic(&mut pending, &mut checked_magic);

        let mut samples = 0;
        while !pending.is_empty() {
            let frame_len = amr_nb_frame_len(pending[0])?;
            if pending.len() < frame_len {
                break;
            }
            if frame_len > 1 {
                samples += AMR_NB_FRAME_SAMPLES;
            }
            pending.drain(..frame_len);
        }
        Ok(samples)
    }
}

impl Default for AmrNbDecoder {
    fn default() -> Self {
        Self::new_decoder()
    }
}

impl Drop for AmrNbDecoder {
    fn drop(&mut self) {
        if !self.state.is_null() {
            unsafe { Decoder_Interface_exit(self.state) };
            self.state = std::ptr::null_mut();
        }
    }
}

impl Decoder for AmrNbDecoder {
    fn decode_i16(
        &mut self,
        input: &[u8],
        output: &mut [i16],
        _fec: bool,
    ) -> Result<usize, String> {
        let required = self.decoded_samples_available(input)?;
        if output.len() < required {
            return Err(format!(
                "Output buffer too small for AMR-NB decode: need {}, have {}",
                required,
                output.len()
            ));
        }

        self.pending_bytes.extend_from_slice(input);
        self.strip_magic_if_needed();

        let mut written = 0;
        loop {
            if self.pending_bytes.is_empty() {
                break;
            }

            let frame_len = amr_nb_frame_len(self.pending_bytes[0])?;
            if self.pending_bytes.len() < frame_len {
                break;
            }

            if frame_len == 1 {
                self.pending_bytes.drain(..1);
                continue;
            }

            unsafe {
                Decoder_Interface_Decode(
                    self.state,
                    self.pending_bytes.as_ptr(),
                    output[written..].as_mut_ptr(),
                    0,
                );
            }
            written += AMR_NB_FRAME_SAMPLES;
            self.pending_bytes.drain(..frame_len);
        }

        Ok(written)
    }

    fn decode_i32(
        &mut self,
        input: &[u8],
        output: &mut [i32],
        _fec: bool,
    ) -> Result<usize, String> {
        let required = self.decoded_samples_available(input)?;
        if output.len() < required {
            return Err(format!(
                "Output buffer too small for AMR-NB decode: need {}, have {}",
                required,
                output.len()
            ));
        }

        let mut samples = vec![0i16; required];
        let written = self.decode_i16(input, &mut samples, false)?;
        if output.len() < written {
            return Err(format!(
                "Output buffer too small for AMR-NB decode: need {}, have {}",
                written,
                output.len()
            ));
        }
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
        let required = self.decoded_samples_available(input)?;
        if output.len() < required {
            return Err(format!(
                "Output buffer too small for AMR-NB decode: need {}, have {}",
                required,
                output.len()
            ));
        }

        let mut samples = vec![0i16; required];
        let written = self.decode_i16(input, &mut samples, false)?;
        if output.len() < written {
            return Err(format!(
                "Output buffer too small for AMR-NB decode: need {}, have {}",
                written,
                output.len()
            ));
        }
        for (dst, sample) in output.iter_mut().zip(samples.into_iter()).take(written) {
            *dst = f32::from(sample) / 32768.0;
        }
        Ok(written)
    }
}

fn strip_magic(pending: &mut Vec<u8>, checked_magic: &mut bool) {
    if *checked_magic {
        return;
    }

    if pending.len() >= AMR_NB_FILE_MAGIC.len() {
        if pending.starts_with(AMR_NB_FILE_MAGIC) {
            pending.drain(..AMR_NB_FILE_MAGIC.len());
        }
        *checked_magic = true;
    } else if !AMR_NB_FILE_MAGIC.starts_with(pending) {
        *checked_magic = true;
    }
}

pub fn amr_nb_frame_len(frame_header: u8) -> Result<usize, String> {
    let frame_type = (frame_header >> 3) & 0x0f;
    AMR_NB_FRAME_BYTES
        .get(frame_type as usize)
        .copied()
        .ok_or_else(|| format!("Invalid AMR-NB frame type: {frame_type}"))
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

    fn decode_fixture(data: &[u8], chunk_size: usize) -> Vec<i16> {
        let mut decoder = AmrNbDecoder::new_decoder();
        let mut decoded = Vec::new();
        for chunk in data.chunks(chunk_size) {
            let max_frames = chunk.len().max(1);
            let mut scratch = vec![0i16; max_frames * AMR_NB_FRAME_SAMPLES];
            let written = decoder.decode_i16(chunk, &mut scratch, false).unwrap();
            decoded.extend_from_slice(&scratch[..written]);
        }
        decoder.flush().unwrap();
        decoded
    }

    #[test]
    fn streaming_encoder_matches_padded_whole_encode() {
        let input = samples();
        let mut padded = input.clone();
        padded.resize(
            input.len().div_ceil(AMR_NB_FRAME_SAMPLES) * AMR_NB_FRAME_SAMPLES,
            0,
        );

        let mut whole_encoder = AmrNbEncoder::new_mr122();
        let mut whole = vec![0u8; padded.len() / AMR_NB_FRAME_SAMPLES * AMR_NB_MAX_FRAME_BYTES];
        let whole_len = whole_encoder.encode_i16(&padded, &mut whole).unwrap();
        whole.truncate(whole_len);

        let mut stream_encoder = AmrNbEncoder::new_mr122();
        let mut chunked = Vec::new();
        let mut scratch = [0u8; AMR_NB_MAX_FRAME_BYTES * 3];
        for chunk in input.chunks(37) {
            let written = stream_encoder.encode_i16(chunk, &mut scratch).unwrap();
            chunked.extend_from_slice(&scratch[..written]);
        }
        stream_encoder.flush_to_vec(&mut chunked).unwrap();

        assert_eq!(chunked, whole);
    }

    #[test]
    fn streaming_decoder_matches_whole_decode() {
        let fixture = fs::read(testdata_path(
            "amr_nb/A_Tusk_is_used_to_make_costly_gifts.amr",
        ))
        .unwrap();
        let whole = decode_fixture(&fixture, fixture.len());
        let chunked = decode_fixture(&fixture, 17);
        assert_eq!(chunked, whole);
        assert!(chunked.iter().any(|&sample| sample != 0));
    }

    #[test]
    fn decoder_trait_supports_i16_i32_and_f32_output() {
        let input = samples();
        let mut encoder = AmrNbEncoder::new_mr122();
        let mut encoded = Vec::new();
        encoder.encode_to_vec(&input, &mut encoded).unwrap();
        encoder.flush_to_vec(&mut encoded).unwrap();

        let mut with_magic = AMR_NB_FILE_MAGIC.to_vec();
        with_magic.extend_from_slice(&encoded);

        let sample_count = encoded
            .iter()
            .filter(|&&byte| amr_nb_frame_len(byte).unwrap_or(0) > 1)
            .count()
            * AMR_NB_FRAME_SAMPLES;
        let mut decoder_i16 = AmrNbDecoder::new_decoder();
        let mut decoder_i32 = AmrNbDecoder::new_decoder();
        let mut decoder_f32 = AmrNbDecoder::new_decoder();
        let mut i16_out = vec![0i16; sample_count];
        let mut i32_out = vec![0i32; sample_count];
        let mut f32_out = vec![0.0f32; sample_count];

        let i16_len = decoder_i16
            .decode_i16(&with_magic, &mut i16_out, false)
            .unwrap();
        let i32_len = decoder_i32
            .decode_i32(&with_magic, &mut i32_out, false)
            .unwrap();
        let f32_len = decoder_f32
            .decode_f32(&with_magic, &mut f32_out, false)
            .unwrap();

        assert_eq!(i16_len, i32_len);
        assert_eq!(i16_len, f32_len);
        for i in 0..i16_len {
            assert_eq!(i32_out[i], i32::from(i16_out[i]) << 16);
            assert!((f32_out[i] - f32::from(i16_out[i]) / 32768.0).abs() < f32::EPSILON);
        }
    }

    #[test]
    #[ignore = "regenerates the committed AMR-NB fixture using ffmpeg/libopencore_amrnb"]
    fn generate_amr_nb_fixture_with_ffmpeg() {
        let input = testdata_path("linear16_8/A_Tusk_is_used_to_make_costly_gifts.s16le");
        let output = testdata_path("amr_nb/A_Tusk_is_used_to_make_costly_gifts.amr");
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
            .args(["-c:a", "libopencore_amrnb", "-b:a", "12.2k", "-f", "amr"])
            .arg(&output)
            .status()
            .unwrap();
        assert!(status.success());
    }

    #[test]
    fn decode_amr_nb_fixture_and_write_golden_wav() {
        let fixture = fs::read(testdata_path(
            "amr_nb/A_Tusk_is_used_to_make_costly_gifts.amr",
        ))
        .unwrap();
        assert!(fixture.starts_with(AMR_NB_FILE_MAGIC));

        let decoded = decode_fixture(&fixture, 23);
        assert!(
            !decoded.is_empty(),
            "no samples decoded from AMR-NB fixture"
        );
        assert!(decoded.iter().any(|&sample| sample != 0));

        let wav = soundkit::wav::generate_wav_buffer(
            &soundkit::audio_types::PcmData::I16(vec![decoded]),
            AMR_NB_SAMPLE_RATE,
        )
        .unwrap();
        let output_path = golden_path("amr_nb/A_Tusk_is_used_to_make_costly_gifts.decoded.wav");
        fs::create_dir_all(output_path.parent().unwrap()).unwrap();
        fs::write(output_path, wav).unwrap();
    }
}
