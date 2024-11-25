use core::slice;
use libflac_sys as ffi;
use libflac_sys::*;
use soundkit::audio_packet::{Decoder, Encoder};
use std::cell::RefCell;
use std::rc::Rc;
use tracing::{debug, error, info};

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

    fn encode_i16(&mut self, input: &[i16], output: &mut [u8]) -> Result<usize, String> {
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
}

impl FlacDecoder {
    pub fn new() -> Self {
        let decoder = unsafe { ffi::FLAC__stream_decoder_new() };
        FlacDecoder {
            decoder,
            output_buffer: Vec::new(),
            input_buffer: Vec::new(),
            input_position: 0,
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
}
impl Decoder for FlacDecoder {
    fn decode_i16(
        &mut self,
        input: &[u8],
        output: &mut [i16],
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
        // Reset internal buffers
        self.output_buffer.clear();
        self.input_buffer.clear();
        self.input_position = 0;

        // Copy input data to internal buffer
        self.input_buffer.extend_from_slice(input);

        unsafe {
            // Process the entire input
            let result = ffi::FLAC__stream_decoder_process_single(self.decoder);
            if result == 0 {
                let state = ffi::FLAC__stream_decoder_get_state(self.decoder);
                return Err(format!(
                    "Failed to decode FLAC block, decoder state: {:?}",
                    state
                ));
            }
        }

        // Copy decoded samples to the output buffer
        let decoded_len = self.output_buffer.len();
        if output.len() < decoded_len {
            return Err("Output buffer too small".to_string());
        }

        output[..decoded_len].copy_from_slice(&self.output_buffer);
        Ok(decoded_len)
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
    let remaining = decoder.input_buffer.len() - decoder.input_position;
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

    let buffer = slice::from_raw_parts(buffer, channels);
    let buffer = buffer
        .iter()
        .map(|x| slice::from_raw_parts(*x, blocksize))
        .collect::<Vec<&[i32]>>();

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

#[cfg(test)]
mod tests {
    use super::*;
    use soundkit::audio_bytes::{f32le_to_s24, s16le_to_i32, s24le_to_i32};
    use soundkit::wav::WavStreamProcessor;
    use std::fs::File;
    use std::io::Read;
    use std::io::Write;

    fn run_flac_encoder_with_wav_file(file_path: &str) {
        let mut decoder = FlacDecoder::new();
        decoder.init().expect("Decoder initialization failed");

        let frame_size = 3600;
        let mut file = File::open(file_path).unwrap();
        let mut file_buffer = Vec::new();
        file.read_to_end(&mut file_buffer).unwrap();

        let mut processor = WavStreamProcessor::new();
        let audio_data = processor.add(&file_buffer).unwrap().unwrap();

        dbg!(file_path, audio_data.sampling_rate());

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
                                println!(
                                    "Decoded {} samples of {} data successfully.",
                                    samples_read, encoded_len
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

        dbg!(n);

        let mut file =
            File::create(file_path.to_owned() + ".flac").expect("Failed to create output file");
        file.write_all(&encoded_data)
            .expect("Failed to write to output file");

        encoder.reset().expect("Failed to reset encoder");
    }

    #[test]
    fn test_flac_encoder_with_wave_16bit() {
        run_flac_encoder_with_wav_file("../testdata/s16le.wav");
    }

    #[test]
    fn test_flac_encoder_with_wave_24bit() {
        run_flac_encoder_with_wav_file("../testdata/s24le.wav");
    }

    #[test]
    fn test_flac_encoder_with_wave_32bit() {
        run_flac_encoder_with_wav_file("../testdata/f32le.wav");
    }

    fn test_flac_encoder_with_wave_s32bit() {
        run_flac_encoder_with_wav_file("../testdata/s32le.wav");
    }
}
