use core::slice;
use libflac_sys as ffi;
use libflac_sys::*;
use soundkit::audio_packet::{Decoder, Encoder};
use std::cell::RefCell;
use std::rc::Rc;
use tracing::{debug, error, trace};

pub struct FlacEncoder {
    encoder: *mut ffi::FLAC__StreamEncoder,
    sample_rate: u32,
    channels: u32,
    bits_per_sample: u32,
    buffer: Rc<RefCell<Vec<u8>>>,
    frame_length: u32,
    compression_level: u32,
}

extern "C" fn write_callback(
    _encoder: *const ffi::FLAC__StreamEncoder,
    buffer: *const ffi::FLAC__byte,
    bytes: usize,
    _samples: u32,
    _current_frame: u32,
    client_data: *mut libc::c_void,
) -> ffi::FLAC__StreamEncoderWriteStatus {
    unsafe {
        let output = &mut *(client_data as *mut RefCell<Vec<u8>>);
        let slice = std::slice::from_raw_parts(buffer, bytes);
        output.borrow_mut().extend_from_slice(slice);
    }
    ffi::FLAC__STREAM_ENCODER_WRITE_STATUS_OK
}

impl Encoder for FlacEncoder {
    fn new(
        sample_rate: u32,
        bits_per_sample: u32,
        channels: u32,
        frame_length: u32,
        compression_level: u32,
    ) -> Self {
        let buffer = Rc::new(RefCell::new(Vec::new()));

        let encoder = unsafe {
            let encoder = ffi::FLAC__stream_encoder_new();
            encoder
        };

        Self {
            encoder,
            sample_rate,
            channels,
            bits_per_sample,
            buffer,
            frame_length,
            compression_level,
        }
    }

    fn init(&mut self) -> Result<(), String> {
        return self.reset();
    }

    fn encode_i16(&mut self, _input: &[i16], _output: &mut [u8]) -> Result<usize, String> {
        Err("Not implemented.".to_string())
    }

    fn encode_i32(&mut self, input: &[i32], output: &mut [u8]) -> Result<usize, String> {
        self.buffer.borrow_mut().clear(); // Clear previous encoded data

        unsafe {
            let success = ffi::FLAC__stream_encoder_process_interleaved(
                self.encoder,
                input.as_ptr() as *const libflac_sys::FLAC__int32,
                (input.len() / self.channels as usize) as u32,
            );

            if success == 0 {
                let state = ffi::FLAC__stream_encoder_get_state(self.encoder);
                return Err(format!(
                    "Failed to process samples, encoder state: {:?}",
                    state
                ));
            }
        }

        let encoded_data = self.buffer.borrow();
        let encoded_len = encoded_data.len();

        if output.len() < encoded_len {
            return Err(format!(
                "Output buffer of len {} too small for encoded data of len {}; input len was {}",
                output.len(),
                encoded_len,
                input.len(),
            ));
        }

        output[..encoded_len].copy_from_slice(&encoded_data);
        Ok(encoded_len)
    }

    fn reset(&mut self) -> Result<(), String> {
        unsafe {
            ffi::FLAC__stream_encoder_finish(self.encoder);
            ffi::FLAC__stream_encoder_delete(self.encoder);

            self.encoder = ffi::FLAC__stream_encoder_new();
            ffi::FLAC__stream_encoder_set_blocksize(self.encoder, self.frame_length);
            ffi::FLAC__stream_encoder_set_verify(self.encoder, true as i32);
            ffi::FLAC__stream_encoder_set_compression_level(self.encoder, self.compression_level);
            ffi::FLAC__stream_encoder_set_channels(self.encoder, self.channels);

            ffi::FLAC__stream_encoder_set_bits_per_sample(self.encoder, self.bits_per_sample);
            ffi::FLAC__stream_encoder_set_sample_rate(self.encoder, self.sample_rate);

            let status = ffi::FLAC__stream_encoder_init_stream(
                self.encoder,
                Some(write_callback),
                None, // seek callback
                None, // tell callback
                None,
                Rc::into_raw(self.buffer.clone()) as *mut libc::c_void,
            );

            if status != ffi::FLAC__STREAM_ENCODER_INIT_STATUS_OK {
                let state: u32 = ffi::FLAC__stream_encoder_get_state(self.encoder);
                return Err(format!(
                    "Failed to reset encoder, encoder state: {:?}",
                    state
                ));
            }
        }

        Ok(())
    }
}

impl Drop for FlacEncoder {
    fn drop(&mut self) {
        unsafe {
            ffi::FLAC__stream_encoder_finish(self.encoder);
            ffi::FLAC__stream_encoder_delete(self.encoder);
        }
    }
}

pub struct FlacDecoder {
    decoder: *mut ffi::FLAC__StreamDecoder,
    output_buffer: Vec<i32>,
    input_buffer: Vec<u8>,
    input_position: usize,
    sample_rate: Option<u32>,
    channels: Option<u8>,
    bits_per_sample: Option<u8>,
}

impl FlacDecoder {
    pub fn new() -> Self {
        let decoder = unsafe { ffi::FLAC__stream_decoder_new() };
        FlacDecoder {
            decoder,
            output_buffer: Vec::new(),
            input_buffer: Vec::new(),
            input_position: 0,
            sample_rate: None,
            channels: None,
            bits_per_sample: None,
        }
    }

    pub fn init(&mut self) -> Result<(), String> {
        unsafe {
            ffi::FLAC__stream_decoder_set_metadata_ignore_all(self.decoder);
            let decoder_status = ffi::FLAC__stream_decoder_init_stream(
                self.decoder,
                Some(read_callback_decode),
                None,
                None,
                None,
                None,
                Some(write_callback_decode),
                None,
                Some(error_callback_decode),
                self as *mut _ as *mut libc::c_void,
            );

            match decoder_status {
                ffi::FLAC__STREAM_DECODER_INIT_STATUS_OK => Ok(()),
                _ => Err(format!(
                    "Failed to initialize the decoder. Status: {:?}",
                    decoder_status
                )),
            }
        }
    }

    pub fn sample_rate(&self) -> Option<u32> {
        self.sample_rate
    }

    pub fn channels(&self) -> Option<u8> {
        self.channels
    }

    pub fn bits_per_sample(&self) -> Option<u8> {
        self.bits_per_sample
    }
}
impl Decoder for FlacDecoder {
    fn decode_i16(
        &mut self,
        _input: &[u8],
        _output: &mut [i16],
        _fec: bool,
    ) -> Result<usize, String> {
        Err("not implemented.".to_string())
    }

    fn decode_i32(
        &mut self,
        input: &[u8],
        output: &mut [i32],
        _fec: bool,
    ) -> Result<usize, String> {
        // Reset decoded buffer for this call
        self.output_buffer.clear();

        if !input.is_empty() {
            self.input_buffer.extend_from_slice(input);
        }

        let mut total_written = 0usize;

        // Process as many frames as we can with the buffered data.
        while !self.input_buffer.is_empty() {
            self.input_position = 0;

            let result = unsafe { ffi::FLAC__stream_decoder_process_single(self.decoder) };
            let mut end_of_stream = false;
            if result == 0 {
                let state = unsafe { ffi::FLAC__stream_decoder_get_state(self.decoder) };
                if state == ffi::FLAC__STREAM_DECODER_END_OF_STREAM {
                    end_of_stream = true;
                } else {
                    return Err(format!(
                        "Failed to decode FLAC block, decoder state: {:?}",
                        state
                    ));
                }
            }

            let consumed = self.input_position.min(self.input_buffer.len());
            if consumed > 0 {
                self.input_buffer.drain(..consumed);
                self.input_position = 0;
            }

            if self.output_buffer.is_empty() {
                // Need more data to form a full frame, or we've reached end-of-stream.
                if consumed == 0 || end_of_stream {
                    break;
                }
                continue;
            }

            let decoded_len = self.output_buffer.len();
            if output.len().saturating_sub(total_written) < decoded_len {
                return Err(format!(
                    "Output buffer too small for decoded frame (needed {}, had {})",
                    decoded_len,
                    output.len().saturating_sub(total_written)
                ));
            }

            output[total_written..total_written + decoded_len]
                .copy_from_slice(&self.output_buffer);
            total_written += decoded_len;
            self.output_buffer.clear();

            // Stop early if caller's buffer is nearly full; let them call again.
            if output.len().saturating_sub(total_written) < 1024 {
                break;
            }

            if end_of_stream {
                break;
            }
        }

        Ok(total_written)
    }

    fn decode_f32(
        &mut self,
        input: &[u8],
        output: &mut [f32],
        fec: bool,
    ) -> Result<usize, String> {
        // Decode to i32 then convert to f32
        let mut i32_buf = vec![0i32; output.len()];
        let samples = self.decode_i32(input, &mut i32_buf, fec)?;

        for i in 0..samples {
            // FLAC uses full 32-bit range
            output[i] = (i32_buf[i] as f64 / i32::MAX as f64) as f32;
        }

        Ok(samples)
    }
}

impl Drop for FlacDecoder {
    fn drop(&mut self) {
        unsafe {
            ffi::FLAC__stream_decoder_finish(self.decoder);
            ffi::FLAC__stream_decoder_delete(self.decoder);
        }
    }
}

unsafe extern "C" fn read_callback_decode(
    _decoder: *const ffi::FLAC__StreamDecoder,
    buffer: *mut ffi::FLAC__byte,
    bytes: *mut usize,
    client_data: *mut std::ffi::c_void,
) -> ffi::FLAC__StreamDecoderReadStatus {
    let decoder = &mut *(client_data as *mut FlacDecoder);
    // Avoid underflow if libFLAC asks for more bytes than we buffered.
    let remaining = decoder
        .input_buffer
        .len()
        .saturating_sub(decoder.input_position);
    let to_read = std::cmp::min(*bytes, remaining);

    if to_read == 0 {
        *bytes = 0;
        return ffi::FLAC__STREAM_DECODER_READ_STATUS_END_OF_STREAM;
    }

    let src = decoder.input_buffer[decoder.input_position..].as_ptr();
    std::ptr::copy_nonoverlapping(src, buffer, to_read);

    decoder.input_position += to_read;
    *bytes = to_read;

    ffi::FLAC__STREAM_DECODER_READ_STATUS_CONTINUE
}

unsafe extern "C" fn write_callback_decode(
    _decoder: *const FLAC__StreamDecoder,
    frame: *const FLAC__Frame,
    buffer: *const *const FLAC__int32,
    client_data: *mut std::ffi::c_void,
) -> FLAC__StreamDecoderWriteStatus {
    let decoder = &mut *(client_data as *mut FlacDecoder);

    let channels = (*frame).header.channels as usize;
    let blocksize = (*frame).header.blocksize as usize;
    let first_frame = decoder.sample_rate.is_none() || decoder.channels.is_none();
    decoder.sample_rate.get_or_insert((*frame).header.sample_rate);
    decoder.channels.get_or_insert((*frame).header.channels as u8);
    decoder.bits_per_sample.get_or_insert((*frame).header.bits_per_sample as u8);

    let buffer = slice::from_raw_parts(buffer, channels);
    let buffer = buffer
        .iter()
        .map(|x| slice::from_raw_parts(*x, blocksize))
        .collect::<Vec<&[i32]>>();

    if first_frame {
        debug!(
            sample_rate_hz = (*frame).header.sample_rate,
            channels = (*frame).header.channels,
            bits_per_sample = (*frame).header.bits_per_sample,
            blocksize,
            pcm_samples_written = blocksize * channels,
            "decoded FLAC frame"
        );
    } else {
        trace!(
            sample_rate_hz = (*frame).header.sample_rate,
            channels = (*frame).header.channels,
            blocksize,
            pcm_samples_written = blocksize * channels,
            "decoded FLAC frame"
        );
    }

    for i in 0..blocksize {
        for j in 0..channels {
            decoder.output_buffer.push(buffer[j][i]);
        }
    }

    FLAC__STREAM_DECODER_WRITE_STATUS_CONTINUE
}

unsafe extern "C" fn error_callback_decode(
    _decoder: *const ffi::FLAC__StreamDecoder,
    status: ffi::FLAC__StreamDecoderErrorStatus,
    _client_data: *mut std::ffi::c_void,
) {
    match status {
        ffi::FLAC__STREAM_DECODER_ERROR_STATUS_LOST_SYNC => {
            debug!("Decoder error: Lost sync with FLAC stream");
        }
        ffi::FLAC__STREAM_DECODER_ERROR_STATUS_BAD_HEADER => {
            error!("Decoder error: Bad FLAC stream header");
        }
        ffi::FLAC__STREAM_DECODER_ERROR_STATUS_FRAME_CRC_MISMATCH => {
            error!("Decoder error: Frame CRC mismatch");
        }
        ffi::FLAC__STREAM_DECODER_ERROR_STATUS_UNPARSEABLE_STREAM => {
            error!("Decoder error: Unparseable stream");
        }
        _ => {
            error!("Decoder error: Unknown error");
        }
    }
}

#[cfg(feature = "claxon-decoder")]
mod claxon_decoder {
    use claxon::{Block, FlacReader};
    use soundkit::audio_packet::Decoder;
    use std::io::Cursor;
    use tracing::debug;

    pub struct FlacDecoderClaxon {
        input_buffer: Vec<u8>,
        pending_samples_i32: Vec<i32>,
        pending_samples_f32: Vec<f32>,
        sample_rate: Option<u32>,
        channels: Option<u8>,
        bits_per_sample: Option<u8>,
    }

    impl FlacDecoderClaxon {
        pub fn new() -> Self {
            Self {
                input_buffer: Vec::new(),
                pending_samples_i32: Vec::new(),
                pending_samples_f32: Vec::new(),
                sample_rate: None,
                channels: None,
                bits_per_sample: None,
            }
        }

        pub fn init(&mut self) -> Result<(), String> {
            Ok(())
        }

        pub fn sample_rate(&self) -> Option<u32> {
            self.sample_rate
        }

        pub fn channels(&self) -> Option<u8> {
            self.channels
        }

        fn extract_metadata_if_needed(&mut self) -> Result<(), String> {
            if self.sample_rate.is_some() {
                return Ok(());
            }

            let cursor = Cursor::new(&self.input_buffer[..]);
            match FlacReader::new(cursor) {
                Ok(reader) => {
                    let streaminfo = reader.streaminfo();
                    self.sample_rate = Some(streaminfo.sample_rate);
                    self.channels = Some(streaminfo.channels as u8);
                    self.bits_per_sample = Some(streaminfo.bits_per_sample as u8);

                    debug!(
                        sample_rate_hz = streaminfo.sample_rate,
                        channels = streaminfo.channels,
                        bits_per_sample = streaminfo.bits_per_sample,
                        "initialized Claxon FLAC decoder"
                    );

                    Ok(())
                }
                Err(e) => Err(format!("Failed to read FLAC metadata: {:?}", e)),
            }
        }
    }

    impl Decoder for FlacDecoderClaxon {
        fn decode_i16(
            &mut self,
            _input: &[u8],
            _output: &mut [i16],
            _fec: bool,
        ) -> Result<usize, String> {
            Err("Not implemented - FLAC uses i32 or f32".to_string())
        }

        fn decode_i32(
            &mut self,
            input: &[u8],
            output: &mut [i32],
            _fec: bool,
        ) -> Result<usize, String> {
            // Append new input data
            if !input.is_empty() {
                self.input_buffer.extend_from_slice(input);
            }

            // Extract metadata if we haven't yet
            if self.sample_rate.is_none() && self.input_buffer.len() >= 42 {
                self.extract_metadata_if_needed()?;
            }

            let mut written = 0;

            // First, drain any pending samples
            if !self.pending_samples_i32.is_empty() {
                let to_copy = self.pending_samples_i32.len().min(output.len());
                output[..to_copy].copy_from_slice(&self.pending_samples_i32[..to_copy]);
                self.pending_samples_i32.drain(..to_copy);
                written += to_copy;
            }

            // If we don't have metadata yet, can't decode
            if self.sample_rate.is_none() {
                return Ok(written);
            }

            // Try to decode frames from the buffer
            let cursor = Cursor::new(&self.input_buffer[..]);
            match FlacReader::new(cursor) {
                Ok(mut reader) => {
                    let channels = self.channels.unwrap() as usize;
                    let mut blocks = reader.blocks();
                    let mut current_block = Block::empty();

                    loop {
                        if written >= output.len() {
                            break;
                        }

                        match blocks.read_next_or_eof(current_block.into_buffer()) {
                            Ok(Some(block)) => {
                                current_block = block;
                                let duration = current_block.duration() as usize;

                                // Interleave samples
                                let mut frame_samples = Vec::with_capacity(duration * channels);
                                for i in 0..duration {
                                    for ch in 0..channels {
                                        frame_samples.push(current_block.sample(ch as u32, i as u32));
                                    }
                                }

                                // Copy what we can to output
                                let remaining = output.len() - written;
                                let to_copy = frame_samples.len().min(remaining);

                                output[written..written + to_copy].copy_from_slice(&frame_samples[..to_copy]);
                                written += to_copy;

                                // Store any overflow in pending buffer
                                if to_copy < frame_samples.len() {
                                    self.pending_samples_i32.extend_from_slice(&frame_samples[to_copy..]);
                                }
                            }
                            Ok(None) => {
                                // End of stream
                                break;
                            }
                            Err(_e) => {
                                // Need more data, stop for now
                                break;
                            }
                        }
                    }
                }
                Err(_e) => {
                    // Can't create reader yet, need more data
                }
            }

            Ok(written)
        }

        fn decode_f32(
            &mut self,
            input: &[u8],
            output: &mut [f32],
            _fec: bool,
        ) -> Result<usize, String> {
            // Append new input data
            if !input.is_empty() {
                self.input_buffer.extend_from_slice(input);
            }

            // Extract metadata if we haven't yet
            if self.sample_rate.is_none() && self.input_buffer.len() >= 42 {
                self.extract_metadata_if_needed()?;
            }

            let mut written = 0;

            // First, drain any pending samples
            if !self.pending_samples_f32.is_empty() {
                let to_copy = self.pending_samples_f32.len().min(output.len());
                output[..to_copy].copy_from_slice(&self.pending_samples_f32[..to_copy]);
                self.pending_samples_f32.drain(..to_copy);
                written += to_copy;
            }

            // If we don't have metadata yet, can't decode
            if self.sample_rate.is_none() {
                return Ok(written);
            }

            // Try to decode frames from the buffer
            let cursor = Cursor::new(&self.input_buffer[..]);
            match FlacReader::new(cursor) {
                Ok(mut reader) => {
                    let channels = self.channels.unwrap() as usize;
                    let bits_per_sample = self.bits_per_sample.unwrap() as i32;
                    let scale = (1i64 << (bits_per_sample - 1)) as f32;
                    let mut blocks = reader.blocks();
                    let mut current_block = Block::empty();

                    loop {
                        if written >= output.len() {
                            break;
                        }

                        match blocks.read_next_or_eof(current_block.into_buffer()) {
                            Ok(Some(block)) => {
                                current_block = block;
                                let duration = current_block.duration() as usize;

                                // Interleave and convert to f32
                                let mut frame_samples = Vec::with_capacity(duration * channels);
                                for i in 0..duration {
                                    for ch in 0..channels {
                                        let sample = current_block.sample(ch as u32, i as u32);
                                        frame_samples.push((sample as f32) / scale);
                                    }
                                }

                                // Copy what we can to output
                                let remaining = output.len() - written;
                                let to_copy = frame_samples.len().min(remaining);

                                output[written..written + to_copy].copy_from_slice(&frame_samples[..to_copy]);
                                written += to_copy;

                                // Store any overflow in pending buffer
                                if to_copy < frame_samples.len() {
                                    self.pending_samples_f32.extend_from_slice(&frame_samples[to_copy..]);
                                }
                            }
                            Ok(None) => {
                                // End of stream
                                break;
                            }
                            Err(_e) => {
                                // Need more data, stop for now
                                break;
                            }
                        }
                    }
                }
                Err(_e) => {
                    // Can't create reader yet, need more data
                }
            }

            Ok(written)
        }
    }
}

#[cfg(feature = "claxon-decoder")]
pub use claxon_decoder::FlacDecoderClaxon;

#[cfg(test)]
mod tests {
    use super::*;
    use soundkit::audio_bytes::{f32le_to_s24, s16le_to_i32, s24le_to_i32};
    use soundkit::test_utils::{print_waveform_with_header, DecodeResult};
    use soundkit::wav::WavStreamProcessor;
    use std::fs::{self, File};
    use std::io::Read;
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::sync::Once;
    use tracing::trace;

    fn init_tracing() {
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            let _ = tracing_subscriber::fmt()
                .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
                .with_test_writer()
                .try_init();
        });
    }

    const TEST_FILE: &str = "A_Tusk_is_used_to_make_costly_gifts";

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

    fn outputs_path(file: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("outputs")
            .join(file)
    }

    #[test]
    fn test_flac_decode_waveform() {
        let input_path = testdata_path(&format!("flac/{}.flac", TEST_FILE));
        let flac_bytes = fs::read(&input_path).unwrap();
        assert!(!flac_bytes.is_empty(), "fixture flac missing or empty");

        init_tracing();

        let mut decoder = FlacDecoder::new();
        decoder.init().expect("Decoder initialization failed");

        let mut decoded = Vec::new();
        let mut scratch = vec![0i32; 8192];

        for chunk in flac_bytes.chunks(4096) {
            let written = decoder.decode_i32(chunk, &mut scratch, false).unwrap();
            decoded.extend_from_slice(&scratch[..written]);
        }

        // Drain remaining
        loop {
            let written = decoder.decode_i32(&[], &mut scratch, false).unwrap();
            if written == 0 {
                break;
            }
            decoded.extend_from_slice(&scratch[..written]);
        }

        assert!(!decoded.is_empty(), "decoder produced no PCM samples");

        // FLAC stores samples as i32 but scaled to original bit depth (16-bit for this test file)
        let result = DecodeResult::from_i32_with_bits(
            &decoded,
            decoder.sample_rate().unwrap_or(16000),
            decoder.channels().unwrap_or(1),
            16, // 16-bit FLAC test file
        );
        print_waveform_with_header("FLAC", &result);
    }

    #[test]
    fn test_flac_decoder_streaming_decode() {
        // decode the real fixture FLAC, not a freshly encoded one
        let input_path = testdata_path("flac/A_Tusk_is_used_to_make_costly_gifts.flac");
        let flac_bytes = fs::read(&input_path).unwrap();
        assert!(!flac_bytes.is_empty(), "fixture flac missing or empty");

        init_tracing();

        let mut decoder = FlacDecoder::new();
        decoder.init().expect("Decoder initialization failed");

        let mut decoded = Vec::new();
        let mut scratch = vec![0i32; 8192];

        for chunk in flac_bytes.chunks(4096) {
            let written = decoder.decode_i32(chunk, &mut scratch, false).unwrap();
            decoded.extend_from_slice(&scratch[..written]);
        }

        loop {
            let written = decoder.decode_i32(&[], &mut scratch, false).unwrap();
            if written == 0 {
                break;
            }
            decoded.extend_from_slice(&scratch[..written]);
        }

        assert!(!decoded.is_empty(), "decoder produced no PCM samples");
        assert_eq!(decoder.sample_rate(), Some(16_000), "fixture sample rate");
        assert_eq!(decoder.channels(), Some(1), "fixture channel count");

        let output_path = outputs_path("A_Tusk_is_used_to_make_costly_gifts.s32le");
        fs::create_dir_all(output_path.parent().unwrap()).unwrap();
        let pcm_bytes: Vec<u8> = decoded.iter().flat_map(|s| s.to_le_bytes()).collect();
        fs::write(&output_path, pcm_bytes).unwrap();
    }

    fn run_flac_encoder_with_wav_file(file_path: &Path, output_path: &Path) {
        init_tracing();

        let mut decoder = FlacDecoder::new();
        decoder.init().expect("Decoder initialization failed");

        let frame_size = 3600;
        let mut file = File::open(file_path).unwrap();
        let mut file_buffer = Vec::new();
        file.read_to_end(&mut file_buffer).unwrap();

        let mut processor = WavStreamProcessor::new();
        let audio_data = processor.add(&file_buffer).unwrap().unwrap();

        trace!(
            file = ?file_path,
            bits_per_sample = audio_data.bits_per_sample(),
            "loaded WAV for FLAC encoding"
        );

        let mut encoder = FlacEncoder::new(
            audio_data.sampling_rate(),
            audio_data.bits_per_sample() as u32,
            audio_data.channel_count() as u32,
            0 as u32,
            5,
        );
        encoder.init().expect("Failed to initialize FLAC encoder");

        let i32_samples = match audio_data.bits_per_sample() {
            16 => {
                // this doesn't scale the 16 bit samples - important!
                s16le_to_i32(audio_data.data())
            }
            24 => s24le_to_i32(audio_data.data()),
            32 => f32le_to_s24(audio_data.data()),
            _ => {
                unreachable!()
            }
        };

        let mut encoded_data = Vec::new();
        let chunk_size = frame_size * audio_data.channel_count() as usize;
        let mut decoded_samples = vec![0i32; chunk_size * 4];
        let mut n = 0;
        for (i, chunk) in i32_samples.chunks(chunk_size).enumerate() {
            let mut output_buffer = vec![0u8; chunk.len() * std::mem::size_of::<i32>() * 2];
            match encoder.encode_i32(chunk, &mut output_buffer) {
                Ok(encoded_len) => {
                    if encoded_len > 0 {
                        n += 1;
                        match decoder.decode_i32(
                            &output_buffer[..encoded_len],
                            &mut decoded_samples,
                            false,
                        ) {
                            Ok(samples_read) => {
                                trace!(
                                    chunk = i,
                                    samples_read,
                                    encoded_len,
                                    "decoded FLAC chunk"
                                );
                            }
                            Err(e) => panic!("Decoding failed: {}", e),
                        }
                    }
                    encoded_data.extend_from_slice(&output_buffer[..encoded_len]);
                }
                Err(e) => {
                    panic!("Failed to encode chunk {}: {:?}", i, e);
                }
            }
        }

        trace!(chunks_encoded = n, "FLAC encoding complete");

        fs::create_dir_all(output_path.parent().unwrap()).unwrap();
        let mut file = File::create(output_path).expect("Failed to create output file");
        file.write_all(&encoded_data)
            .expect("Failed to write to output file");

        encoder.reset().expect("Failed to reset encoder");
    }

    #[test]
    fn test_flac_encoder_with_wave_16bit() {
        run_flac_encoder_with_wav_file(
            &testdata_path("wav_stereo/A_Tusk_is_used_to_make_costly_gifts.wav"),
            &golden_path("flac/A_Tusk_is_used_to_make_costly_gifts_16bit.flac"),
        );
    }

    #[test]
    fn test_flac_encoder_with_wave_24bit() {
        run_flac_encoder_with_wav_file(
            &testdata_path("wav_24/A_Tusk_is_used_to_make_costly_gifts.wav"),
            &golden_path("flac/A_Tusk_is_used_to_make_costly_gifts_24bit.flac"),
        );
    }

    #[test]
    fn test_flac_encoder_with_wave_32bit() {
        run_flac_encoder_with_wav_file(
            &testdata_path("wav_32f/A_Tusk_is_used_to_make_costly_gifts.wav"),
            &golden_path("flac/A_Tusk_is_used_to_make_costly_gifts_32float.flac"),
        );
    }
}
