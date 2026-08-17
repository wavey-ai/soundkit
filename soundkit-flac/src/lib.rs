#[cfg(feature = "libflac")]
use core::slice;
#[cfg(feature = "libflac")]
use libflac_sys as ffi;
#[cfg(feature = "libflac")]
use libflac_sys::*;
#[cfg(all(feature = "oxideav-encoder", not(feature = "flacenc-encoder")))]
use oxideav_core::registry::codec::Encoder as OxideEncoder;
#[cfg(all(feature = "oxideav-encoder", not(feature = "flacenc-encoder")))]
use oxideav_core::{AudioFrame as OxideAudioFrame, CodecId, CodecParameters, Error as OxideError};
#[cfg(all(feature = "oxideav-encoder", not(feature = "flacenc-encoder")))]
use oxideav_core::{Frame as OxideFrame, SampleFormat};
#[cfg(feature = "libflac")]
use soundkit::audio_packet::Decoder;
#[cfg(any(
    feature = "libflac",
    feature = "oxideav-encoder",
    feature = "flacenc-encoder"
))]
use soundkit::audio_packet::Encoder;
#[cfg(all(
    feature = "libflac",
    not(feature = "oxideav-encoder"),
    not(feature = "flacenc-encoder")
))]
use std::cell::RefCell;
#[cfg(all(feature = "oxideav-encoder", not(feature = "flacenc-encoder")))]
use std::collections::VecDeque;
#[cfg(all(
    feature = "libflac",
    not(feature = "oxideav-encoder"),
    not(feature = "flacenc-encoder")
))]
use std::rc::Rc;
#[cfg(feature = "libflac")]
use tracing::{debug, error, trace};

#[cfg(any(feature = "packet-codec", feature = "flacenc-encoder"))]
mod frame_codec;
#[cfg(feature = "packet-codec")]
pub use frame_codec::{DecodedFlacFrame, FlacFrameDecoder};
#[cfg(any(feature = "packet-codec", feature = "flacenc-encoder"))]
pub use frame_codec::{
    EncodedFlacFrame, FlacFrameConfig, FlacFrameEncoder, FlacFrameError, FlacProfile,
};

#[cfg(feature = "flacenc-encoder")]
pub struct FlacEncoder {
    sample_rate: u32,
    channels: u32,
    bits_per_sample: u32,
    frame_length: u32,
    compression_level: u32,
    inner: FlacFrameEncoder,
    stream_header: Vec<u8>,
    emitted_stream_header: bool,
}

#[cfg(feature = "flacenc-encoder")]
impl FlacEncoder {
    fn profile(compression_level: u32) -> FlacProfile {
        match compression_level {
            0..=4 => FlacProfile::Realtime,
            5..=8 => FlacProfile::Balanced,
            _ => FlacProfile::Maximum,
        }
    }

    fn make_inner(
        sample_rate: u32,
        bits_per_sample: u32,
        channels: u32,
        frame_length: u32,
        compression_level: u32,
    ) -> Result<(FlacFrameEncoder, Vec<u8>), String> {
        let channels =
            u16::try_from(channels).map_err(|_| format!("Channel count {channels} exceeds u16"))?;
        let bits_per_sample = u8::try_from(bits_per_sample)
            .map_err(|_| format!("Bits per sample {bits_per_sample} exceeds u8"))?;
        let config = FlacFrameConfig::new(
            sample_rate,
            channels,
            bits_per_sample,
            frame_length,
            Self::profile(compression_level),
        )
        .map_err(|error| error.to_string())?;
        let inner = FlacFrameEncoder::new(config).map_err(|error| error.to_string())?;
        let stream_header = inner.stream_header().map_err(|error| error.to_string())?;
        Ok((inner, stream_header))
    }

    /// Finish the FLAC stream.
    ///
    /// The flacenc backend emits each complete block immediately. This call
    /// finalizes STREAMINFO and therefore does not emit another frame.
    pub fn finish(&mut self, _output: &mut [u8]) -> Result<usize, String> {
        self.stream_header = self
            .inner
            .stream_header()
            .map_err(|error| error.to_string())?;
        Ok(0)
    }

    pub fn stream_header(&self) -> &[u8] {
        &self.stream_header
    }
}

#[cfg(feature = "flacenc-encoder")]
impl Encoder for FlacEncoder {
    fn new(
        sample_rate: u32,
        bits_per_sample: u32,
        channels: u32,
        frame_length: u32,
        compression_level: u32,
    ) -> Self {
        let (inner, stream_header) = Self::make_inner(
            sample_rate,
            bits_per_sample,
            channels,
            frame_length,
            compression_level,
        )
        .unwrap_or_else(|error| panic!("{error}"));
        Self {
            sample_rate,
            channels,
            bits_per_sample,
            frame_length,
            compression_level,
            inner,
            stream_header,
            emitted_stream_header: false,
        }
    }

    fn init(&mut self) -> Result<(), String> {
        self.reset()
    }

    fn encode_i16(&mut self, _input: &[i16], _output: &mut [u8]) -> Result<usize, String> {
        Err("Not implemented - FLAC uses i32 input".to_string())
    }

    fn encode_i32(&mut self, input: &[i32], output: &mut [u8]) -> Result<usize, String> {
        let encoded = self
            .inner
            .encode_i32_block(input)
            .map_err(|error| error.to_string())?;
        self.stream_header = self
            .inner
            .stream_header()
            .map_err(|error| error.to_string())?;
        let header_length = if self.emitted_stream_header {
            0
        } else {
            self.stream_header.len()
        };
        let encoded_length = header_length
            .checked_add(encoded.payload.len())
            .ok_or_else(|| "FLAC packet length overflow".to_string())?;
        if output.len() < encoded_length {
            return Err(format!(
                "Output buffer of len {} too small for FLAC packet of len {encoded_length}",
                output.len()
            ));
        }
        if header_length > 0 {
            output[..header_length].copy_from_slice(&self.stream_header);
        }
        output[header_length..encoded_length].copy_from_slice(&encoded.payload);
        self.emitted_stream_header = true;
        Ok(encoded_length)
    }

    fn reset(&mut self) -> Result<(), String> {
        let (inner, stream_header) = Self::make_inner(
            self.sample_rate,
            self.bits_per_sample,
            self.channels,
            self.frame_length,
            self.compression_level,
        )?;
        self.inner = inner;
        self.stream_header = stream_header;
        self.emitted_stream_header = false;
        Ok(())
    }
}

#[cfg(all(feature = "oxideav-encoder", not(feature = "flacenc-encoder")))]
pub struct FlacEncoder {
    sample_rate: u32,
    channels: u32,
    bits_per_sample: u32,
    frame_length: u32,
    bitrate: u32,
    inner: Box<dyn OxideEncoder>,
    pending_packets: VecDeque<Vec<u8>>,
    stream_header: Vec<u8>,
    emitted_stream_header: bool,
}

#[cfg(all(feature = "oxideav-encoder", not(feature = "flacenc-encoder")))]
impl FlacEncoder {
    fn sample_format(bits_per_sample: u32) -> Result<SampleFormat, String> {
        match bits_per_sample {
            1..=16 => Ok(SampleFormat::S16),
            17..=24 => Ok(SampleFormat::S24),
            25..=32 => Ok(SampleFormat::S32),
            _ => Err(format!(
                "Unsupported FLAC bits per sample: {}",
                bits_per_sample
            )),
        }
    }

    fn build_params(
        sample_rate: u32,
        bits_per_sample: u32,
        channels: u32,
        bitrate: u32,
    ) -> Result<CodecParameters, String> {
        let sample_format = Self::sample_format(bits_per_sample)?;
        let channels = u16::try_from(channels)
            .map_err(|_| format!("Channel count {} exceeds u16", channels))?;
        let mut params = CodecParameters::audio(CodecId::new("flac"));
        params.sample_rate = Some(sample_rate);
        params.channels = Some(channels);
        params.sample_format = Some(sample_format);
        if bitrate > 0 {
            params.bit_rate = Some(u64::from(bitrate));
        }
        Ok(params)
    }

    fn make_inner(
        sample_rate: u32,
        bits_per_sample: u32,
        channels: u32,
        bitrate: u32,
    ) -> Result<(Box<dyn OxideEncoder>, Vec<u8>), String> {
        let params = Self::build_params(sample_rate, bits_per_sample, channels, bitrate)?;
        let inner = oxideav_flac::encoder::make_encoder(&params)
            .map_err(|error| format!("Failed to create oxideav FLAC encoder: {error}"))?;
        let stream_header = inner.output_params().extradata.clone();
        Ok((inner, stream_header))
    }

    fn queue_ready_packets(&mut self) -> Result<(), String> {
        loop {
            match self.inner.receive_packet() {
                Ok(packet) => {
                    let mut data = packet.data;
                    if !self.emitted_stream_header {
                        let mut first_packet =
                            Vec::with_capacity(self.stream_header.len() + data.len());
                        first_packet.extend_from_slice(&self.stream_header);
                        first_packet.append(&mut data);
                        data = first_packet;
                        self.emitted_stream_header = true;
                    }
                    self.pending_packets.push_back(data);
                }
                Err(OxideError::NeedMore) => return Ok(()),
                Err(error) => return Err(format!("Failed to receive FLAC packet: {error}")),
            }
        }
    }

    fn copy_next_packet(&mut self, output: &mut [u8]) -> Result<usize, String> {
        let Some(packet) = self.pending_packets.pop_front() else {
            // OxideAV buffers a trailing block shorter than its configured
            // block size until EOF. An empty packet is therefore a normal
            // streaming result; callers must finish() once after the final
            // input block to drain it.
            return Ok(0);
        };
        if output.len() < packet.len() {
            return Err(format!(
                "Output buffer of len {} too small for FLAC packet of len {}",
                output.len(),
                packet.len()
            ));
        }
        output[..packet.len()].copy_from_slice(&packet);
        Ok(packet.len())
    }

    /// Signal EOF and return the next packet produced by the encoder.
    ///
    /// The Wasm streaming wrapper feeds complete blocks as they arrive and
    /// calls this once at the end to recover a final short block.
    pub fn finish(&mut self, output: &mut [u8]) -> Result<usize, String> {
        self.inner
            .flush()
            .map_err(|error| format!("Failed to finish FLAC stream: {error}"))?;
        self.stream_header = self.inner.output_params().extradata.clone();
        self.queue_ready_packets()?;
        self.copy_next_packet(output)
    }

    pub fn stream_header(&self) -> &[u8] {
        &self.stream_header
    }
}

#[cfg(all(feature = "oxideav-encoder", not(feature = "flacenc-encoder")))]
impl Encoder for FlacEncoder {
    fn new(
        sample_rate: u32,
        bits_per_sample: u32,
        channels: u32,
        frame_length: u32,
        bitrate: u32,
    ) -> Self {
        let (inner, stream_header) =
            Self::make_inner(sample_rate, bits_per_sample, channels, bitrate)
                .unwrap_or_else(|error| panic!("{error}"));

        Self {
            sample_rate,
            channels,
            bits_per_sample,
            frame_length,
            bitrate,
            inner,
            pending_packets: VecDeque::new(),
            stream_header,
            emitted_stream_header: false,
        }
    }

    fn init(&mut self) -> Result<(), String> {
        self.reset()
    }

    fn encode_i16(&mut self, _input: &[i16], _output: &mut [u8]) -> Result<usize, String> {
        Err("Not implemented - FLAC uses i32 input".to_string())
    }

    fn encode_i32(&mut self, input: &[i32], output: &mut [u8]) -> Result<usize, String> {
        let channels = self.channels as usize;
        if channels == 0 {
            return Err("FLAC encoder requires at least one channel".to_string());
        }
        if !input.len().is_multiple_of(channels) {
            return Err(format!(
                "FLAC encoder input length {} is not divisible by channel count {}",
                input.len(),
                channels
            ));
        }

        if self.pending_packets.is_empty() {
            let bytes_per_sample = match self.bits_per_sample {
                1..=16 => 2,
                17..=24 => 3,
                25..=32 => 4,
                _ => {
                    return Err(format!(
                        "Unsupported FLAC bits per sample: {}",
                        self.bits_per_sample
                    ));
                }
            };
            let mut audio_bytes = Vec::with_capacity(input.len() * bytes_per_sample);
            for &sample in input {
                let bytes = sample.to_le_bytes();
                audio_bytes.extend_from_slice(&bytes[..bytes_per_sample]);
            }

            let frame = OxideFrame::Audio(OxideAudioFrame {
                samples: (input.len() / channels) as u32,
                pts: None,
                data: vec![audio_bytes],
            });

            self.inner
                .send_frame(&frame)
                .map_err(|error| format!("Failed to encode FLAC frame: {error}"))?;
            self.queue_ready_packets()?;
        }

        self.copy_next_packet(output)
    }

    fn reset(&mut self) -> Result<(), String> {
        let (inner, stream_header) = Self::make_inner(
            self.sample_rate,
            self.bits_per_sample,
            self.channels,
            self.bitrate,
        )?;
        self.inner = inner;
        self.stream_header = stream_header;
        self.pending_packets.clear();
        self.emitted_stream_header = false;
        let _ = self.frame_length;
        Ok(())
    }
}

#[cfg(all(
    feature = "libflac",
    not(feature = "oxideav-encoder"),
    not(feature = "flacenc-encoder")
))]
pub struct FlacEncoder {
    encoder: *mut ffi::FLAC__StreamEncoder,
    sample_rate: u32,
    channels: u32,
    bits_per_sample: u32,
    buffer: Rc<RefCell<Vec<u8>>>,
    frame_length: u32,
    compression_level: u32,
}

#[cfg(all(
    feature = "libflac",
    not(feature = "oxideav-encoder"),
    not(feature = "flacenc-encoder")
))]
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

#[cfg(all(
    feature = "libflac",
    not(feature = "oxideav-encoder"),
    not(feature = "flacenc-encoder")
))]
impl Encoder for FlacEncoder {
    fn new(
        sample_rate: u32,
        bits_per_sample: u32,
        channels: u32,
        frame_length: u32,
        compression_level: u32,
    ) -> Self {
        let buffer = Rc::new(RefCell::new(Vec::new()));

        let encoder = unsafe { ffi::FLAC__stream_encoder_new() };

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
        self.reset()
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

#[cfg(all(
    feature = "libflac",
    not(feature = "oxideav-encoder"),
    not(feature = "flacenc-encoder")
))]
impl Drop for FlacEncoder {
    fn drop(&mut self) {
        unsafe {
            ffi::FLAC__stream_encoder_finish(self.encoder);
            ffi::FLAC__stream_encoder_delete(self.encoder);
        }
    }
}

#[cfg(feature = "libflac")]
pub struct FlacDecoder {
    decoder: *mut ffi::FLAC__StreamDecoder,
    output_buffer: Vec<i32>,
    input_buffer: Vec<u8>,
    input_position: usize,
    sample_rate: Option<u32>,
    channels: Option<u8>,
    bits_per_sample: Option<u8>,
}

#[cfg(feature = "libflac")]
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

#[cfg(feature = "libflac")]
impl Default for FlacDecoder {
    fn default() -> Self {
        Self::new()
    }
}
#[cfg(feature = "libflac")]
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

            output[total_written..total_written + decoded_len].copy_from_slice(&self.output_buffer);
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

    fn decode_f32(&mut self, input: &[u8], output: &mut [f32], fec: bool) -> Result<usize, String> {
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

#[cfg(feature = "libflac")]
impl Drop for FlacDecoder {
    fn drop(&mut self) {
        unsafe {
            ffi::FLAC__stream_decoder_finish(self.decoder);
            ffi::FLAC__stream_decoder_delete(self.decoder);
        }
    }
}

#[cfg(feature = "libflac")]
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

#[cfg(feature = "libflac")]
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
    decoder
        .sample_rate
        .get_or_insert((*frame).header.sample_rate);
    decoder
        .channels
        .get_or_insert((*frame).header.channels as u8);
    decoder
        .bits_per_sample
        .get_or_insert((*frame).header.bits_per_sample as u8);

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

    for sample_index in 0..blocksize {
        for channel_buffer in buffer.iter().take(channels) {
            decoder.output_buffer.push(channel_buffer[sample_index]);
        }
    }

    FLAC__STREAM_DECODER_WRITE_STATUS_CONTINUE
}

#[cfg(feature = "libflac")]
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
    use claxon::{frame::FrameReader, Block};
    use soundkit::audio_packet::Decoder;
    use std::io::Cursor;
    use tracing::debug;

    const MAX_FLAC_INPUT_CHUNK_BYTES: usize = 4 * 1024 * 1024;
    const MAX_FLAC_FRAME_BUFFER_BYTES: usize = 8 * 1024 * 1024;

    enum FlacStreamState {
        Magic,
        MetadataHeader,
        MetadataPayload {
            block_type: u8,
            is_last: bool,
            remaining: usize,
            payload: Vec<u8>,
        },
        Frames,
    }

    pub struct FlacDecoderClaxon {
        input_buffer: Vec<u8>,
        pending_samples_i32: Vec<i32>,
        sample_rate: Option<u32>,
        channels: Option<u8>,
        bits_per_sample: Option<u8>,
        state: FlacStreamState,
    }

    impl Default for FlacDecoderClaxon {
        fn default() -> Self {
            Self::new()
        }
    }

    impl FlacDecoderClaxon {
        pub fn new() -> Self {
            Self {
                input_buffer: Vec::new(),
                pending_samples_i32: Vec::new(),
                sample_rate: None,
                channels: None,
                bits_per_sample: None,
                state: FlacStreamState::Magic,
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

        pub fn bits_per_sample(&self) -> Option<u8> {
            self.bits_per_sample
        }

        pub fn buffered_bytes(&self) -> usize {
            self.input_buffer.len()
        }

        fn append_input(&mut self, input: &[u8]) -> Result<(), String> {
            if input.len() > MAX_FLAC_INPUT_CHUNK_BYTES {
                return Err(format!(
                    "FLAC input chunk exceeds the {MAX_FLAC_INPUT_CHUNK_BYTES} byte streaming budget"
                ));
            }
            if self.input_buffer.len().saturating_add(input.len()) > MAX_FLAC_FRAME_BUFFER_BYTES {
                return Err(format!(
                    "FLAC frame exceeds the {MAX_FLAC_FRAME_BUFFER_BYTES} byte buffer budget"
                ));
            }
            self.input_buffer.extend_from_slice(input);
            Ok(())
        }

        fn consume_metadata(&mut self) -> Result<(), String> {
            loop {
                let state = std::mem::replace(&mut self.state, FlacStreamState::Frames);
                match state {
                    FlacStreamState::Magic => {
                        if self.input_buffer.len() < 4 {
                            self.state = FlacStreamState::Magic;
                            return Ok(());
                        }
                        if &self.input_buffer[..4] != b"fLaC" {
                            return Err("FLAC stream has no fLaC marker".to_string());
                        }
                        self.input_buffer.drain(..4);
                        self.state = FlacStreamState::MetadataHeader;
                    }
                    FlacStreamState::MetadataHeader => {
                        if self.input_buffer.len() < 4 {
                            self.state = FlacStreamState::MetadataHeader;
                            return Ok(());
                        }
                        let header = [
                            self.input_buffer[0],
                            self.input_buffer[1],
                            self.input_buffer[2],
                            self.input_buffer[3],
                        ];
                        self.input_buffer.drain(..4);
                        let block_type = header[0] & 0x7f;
                        let is_last = header[0] & 0x80 != 0;
                        let remaining = ((header[1] as usize) << 16)
                            | ((header[2] as usize) << 8)
                            | header[3] as usize;
                        if block_type == 0 && remaining != 34 {
                            return Err("FLAC STREAMINFO must contain 34 bytes".to_string());
                        }
                        self.state = FlacStreamState::MetadataPayload {
                            block_type,
                            is_last,
                            remaining,
                            payload: Vec::with_capacity(if block_type == 0 { 34 } else { 0 }),
                        };
                    }
                    FlacStreamState::MetadataPayload {
                        block_type,
                        is_last,
                        mut remaining,
                        mut payload,
                    } => {
                        let consumed = remaining.min(self.input_buffer.len());
                        if block_type == 0 {
                            payload.extend_from_slice(&self.input_buffer[..consumed]);
                        }
                        self.input_buffer.drain(..consumed);
                        remaining -= consumed;
                        if remaining > 0 {
                            self.state = FlacStreamState::MetadataPayload {
                                block_type,
                                is_last,
                                remaining,
                                payload,
                            };
                            return Ok(());
                        }
                        if block_type == 0 {
                            self.install_streaminfo(&payload)?;
                        }
                        self.state = if is_last {
                            FlacStreamState::Frames
                        } else {
                            FlacStreamState::MetadataHeader
                        };
                    }
                    FlacStreamState::Frames => {
                        self.state = FlacStreamState::Frames;
                        return Ok(());
                    }
                }
            }
        }

        fn install_streaminfo(&mut self, payload: &[u8]) -> Result<(), String> {
            let packed = u64::from_be_bytes(
                payload[10..18]
                    .try_into()
                    .map_err(|_| "FLAC STREAMINFO is truncated".to_string())?,
            );
            let sample_rate = (packed >> 44) as u32;
            let channels = ((packed >> 41) & 0x07) as u8 + 1;
            let bits_per_sample = ((packed >> 36) & 0x1f) as u8 + 1;
            if sample_rate == 0 || channels == 0 || bits_per_sample == 0 {
                return Err("FLAC STREAMINFO contains invalid audio geometry".to_string());
            }
            self.sample_rate = Some(sample_rate);
            self.channels = Some(channels);
            self.bits_per_sample = Some(bits_per_sample);
            debug!(
                sample_rate_hz = sample_rate,
                channels, bits_per_sample, "initialized streaming Claxon FLAC decoder"
            );
            Ok(())
        }

        fn decode_available(&mut self, target_samples: usize) -> Result<bool, String> {
            self.consume_metadata()?;
            if !matches!(self.state, FlacStreamState::Frames) || self.input_buffer.is_empty() {
                return Ok(false);
            }
            let channels = self
                .channels
                .ok_or_else(|| "FLAC stream has no STREAMINFO block".to_string())?
                as usize;
            let mut made_progress = false;
            while self.pending_samples_i32.len() < target_samples && !self.input_buffer.is_empty() {
                let cursor = Cursor::new(&self.input_buffer[..]);
                let mut reader = FrameReader::new(cursor);
                let block = match reader.read_next_or_eof(Block::empty().into_buffer()) {
                    Ok(Some(block)) => block,
                    Ok(None) | Err(_) => break,
                };
                let consumed = reader.into_inner().position() as usize;
                if consumed == 0 || consumed > self.input_buffer.len() {
                    return Err("FLAC decoder reported an invalid consumed range".to_string());
                }
                let duration = block.duration() as usize;
                for frame in 0..duration {
                    for channel in 0..channels {
                        self.pending_samples_i32
                            .push(block.sample(channel as u32, frame as u32));
                    }
                }
                self.input_buffer.drain(..consumed);
                made_progress = true;
            }
            Ok(made_progress)
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
            self.append_input(input)?;
            self.decode_available(output.len())?;

            // Return samples from pending buffer
            let to_copy = self.pending_samples_i32.len().min(output.len());
            if to_copy > 0 {
                output[..to_copy].copy_from_slice(&self.pending_samples_i32[..to_copy]);
                self.pending_samples_i32.drain(..to_copy);
            }

            Ok(to_copy)
        }

        fn decode_f32(
            &mut self,
            input: &[u8],
            output: &mut [f32],
            _fec: bool,
        ) -> Result<usize, String> {
            // Decode to i32 first, then convert
            let mut i32_output = vec![0i32; output.len()];
            let samples = self.decode_i32(input, &mut i32_output, _fec)?;

            if samples > 0 {
                let bits = self.bits_per_sample.unwrap_or(16) as i32;
                let scale = (1i64 << (bits - 1)) as f32;

                for i in 0..samples {
                    output[i] = (i32_output[i] as f32) / scale;
                }
            }

            Ok(samples)
        }
    }
}

#[cfg(feature = "claxon-decoder")]
pub use claxon_decoder::FlacDecoderClaxon;

#[cfg(test)]
mod tests {
    #[cfg(any(
        feature = "libflac",
        feature = "oxideav-encoder",
        feature = "flacenc-encoder"
    ))]
    use super::*;
    #[cfg(all(feature = "libflac", not(feature = "flacenc-encoder")))]
    use soundkit::audio_bytes::{f32le_to_s24, s16le_to_i32, s24le_to_i32};
    #[cfg(any(feature = "libflac", feature = "claxon-decoder"))]
    use soundkit::test_utils::{print_waveform_with_header, DecodeResult};
    #[cfg(all(feature = "libflac", not(feature = "flacenc-encoder")))]
    use soundkit::wav::WavStreamProcessor;
    #[cfg(any(feature = "libflac", feature = "claxon-decoder"))]
    use std::fs;
    #[cfg(all(feature = "libflac", not(feature = "flacenc-encoder")))]
    use std::fs::File;
    #[cfg(all(feature = "libflac", not(feature = "flacenc-encoder")))]
    use std::io::Read;
    #[cfg(all(
        feature = "libflac",
        not(feature = "oxideav-encoder"),
        not(feature = "flacenc-encoder")
    ))]
    use std::io::Write;
    #[cfg(all(feature = "libflac", not(feature = "flacenc-encoder")))]
    use std::path::Path;
    #[cfg(any(
        feature = "libflac",
        feature = "oxideav-encoder",
        feature = "claxon-decoder"
    ))]
    use std::path::PathBuf;
    #[cfg(any(
        feature = "libflac",
        feature = "oxideav-encoder",
        feature = "claxon-decoder"
    ))]
    use std::sync::Once;
    #[cfg(all(feature = "libflac", not(feature = "flacenc-encoder")))]
    use tracing::trace;

    #[cfg(any(
        feature = "libflac",
        feature = "oxideav-encoder",
        feature = "claxon-decoder"
    ))]
    fn init_tracing() {
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            let _ = tracing_subscriber::fmt()
                .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
                .with_test_writer()
                .try_init();
        });
    }

    #[cfg(any(feature = "libflac", feature = "claxon-decoder"))]
    const TEST_FILE: &str = "A_Tusk_is_used_to_make_costly_gifts";

    #[cfg(any(
        feature = "libflac",
        feature = "oxideav-encoder",
        feature = "claxon-decoder"
    ))]
    fn testdata_path(file: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("testdata")
            .join(file)
    }

    #[cfg(all(
        feature = "libflac",
        not(feature = "oxideav-encoder"),
        not(feature = "flacenc-encoder")
    ))]
    fn golden_path(file: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("golden")
            .join(file)
    }

    #[cfg(feature = "libflac")]
    fn outputs_path(file: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("outputs")
            .join(file)
    }

    #[cfg(feature = "libflac")]
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

    #[cfg(feature = "libflac")]
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

    #[cfg(all(feature = "libflac", not(feature = "flacenc-encoder")))]
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
            0_u32,
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
            let mut output_buffer = vec![0u8; std::mem::size_of_val(chunk) * 2];
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
                                trace!(chunk = i, samples_read, encoded_len, "decoded FLAC chunk");
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

        if !output_path.as_os_str().is_empty() {
            fs::create_dir_all(output_path.parent().unwrap()).unwrap();
            fs::write(output_path, &encoded_data).expect("Failed to write encoded FLAC output");
        }

        assert!(!encoded_data.is_empty(), "FLAC encoder produced no bytes");

        encoder.reset().expect("Failed to reset encoder");
    }

    #[cfg(all(
        feature = "libflac",
        not(feature = "oxideav-encoder"),
        not(feature = "flacenc-encoder")
    ))]
    #[test]
    fn test_flac_encoder_with_wave_16bit() {
        run_flac_encoder_with_wav_file(
            &testdata_path("wav_stereo/A_Tusk_is_used_to_make_costly_gifts.wav"),
            &golden_path("flac/A_Tusk_is_used_to_make_costly_gifts_16bit.flac"),
        );
    }

    #[cfg(all(
        feature = "libflac",
        not(feature = "oxideav-encoder"),
        not(feature = "flacenc-encoder")
    ))]
    #[test]
    fn test_flac_encoder_with_wave_24bit() {
        run_flac_encoder_with_wav_file(
            &testdata_path("wav_24/A_Tusk_is_used_to_make_costly_gifts.wav"),
            &golden_path("flac/A_Tusk_is_used_to_make_costly_gifts_24bit.flac"),
        );
    }

    #[cfg(all(
        feature = "libflac",
        not(feature = "oxideav-encoder"),
        not(feature = "flacenc-encoder")
    ))]
    #[test]
    fn test_flac_encoder_with_wave_32bit() {
        run_flac_encoder_with_wav_file(
            &testdata_path("wav_32f/A_Tusk_is_used_to_make_costly_gifts.wav"),
            &golden_path("flac/A_Tusk_is_used_to_make_costly_gifts_32float.flac"),
        );
    }

    #[cfg(all(
        feature = "oxideav-encoder",
        feature = "libflac",
        not(feature = "flacenc-encoder")
    ))]
    #[test]
    fn test_oxideav_flac_encoder_streaming_packets_roundtrip() {
        run_flac_encoder_with_wav_file(
            &testdata_path("wav_stereo/A_Tusk_is_used_to_make_costly_gifts.wav"),
            Path::new(""),
        );
    }

    #[cfg(all(feature = "flacenc-encoder", feature = "claxon-decoder"))]
    #[test]
    fn test_flacenc_stream_encoder_roundtrip_with_a_short_final_block() {
        use std::io::Cursor;

        let channels = 2usize;
        let frame_length = 128usize;
        let frame_count = frame_length * 2 + 64;
        let samples = (0..frame_count * channels)
            .map(|index| ((index as i32 * 977) % 65_536) - 32_768)
            .collect::<Vec<_>>();
        let mut encoder = FlacEncoder::new(48_000, 16, channels as u32, frame_length as u32, 5);
        encoder.init().unwrap();

        let mut packets = Vec::new();
        for chunk in samples.chunks(frame_length * channels) {
            let mut output = vec![0u8; chunk.len() * 8 + 4_096];
            let encoded = encoder.encode_i32(chunk, &mut output).unwrap();
            output.truncate(encoded);
            packets.push(output);
        }
        let mut flush = vec![0u8; frame_length * channels * 8 + 4_096];
        assert_eq!(encoder.finish(&mut flush).unwrap(), 0);

        let header = encoder.stream_header();
        let mut file = b"fLaC".to_vec();
        file.extend_from_slice(header);
        file.extend_from_slice(&packets[0][header.len()..]);
        for packet in &packets[1..] {
            file.extend_from_slice(packet);
        }

        let mut reader = claxon::FlacReader::new(Cursor::new(file)).unwrap();
        let decoded = reader.samples().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(decoded, samples);
    }

    #[cfg(feature = "claxon-decoder")]
    mod claxon_tests {
        use super::*;
        use crate::FlacDecoderClaxon;
        use soundkit::audio_packet::Decoder;
        use std::io::Cursor;

        /// Decode FLAC using claxon decoder
        fn decode_with_claxon(
            flac_bytes: &[u8],
        ) -> (Vec<i32>, Option<u32>, Option<u8>, Option<u8>) {
            let mut decoder = FlacDecoderClaxon::new();
            decoder
                .init()
                .expect("Claxon decoder initialization failed");

            let mut decoded = Vec::new();
            let mut scratch = vec![0i32; 8192];

            // Feed one bounded source chunk.
            let written = decoder.decode_i32(flac_bytes, &mut scratch, false).unwrap();
            decoded.extend_from_slice(&scratch[..written]);

            // Drain remaining
            loop {
                let written = decoder.decode_i32(&[], &mut scratch, false).unwrap();
                if written == 0 {
                    break;
                }
                decoded.extend_from_slice(&scratch[..written]);
            }

            (
                decoded,
                decoder.sample_rate(),
                decoder.channels(),
                decoder.bits_per_sample(),
            )
        }

        /// Decode FLAC using libflac decoder
        #[cfg(feature = "libflac")]
        fn decode_with_libflac(
            flac_bytes: &[u8],
        ) -> (Vec<i32>, Option<u32>, Option<u8>, Option<u8>) {
            let mut decoder = FlacDecoder::new();
            decoder
                .init()
                .expect("libFLAC decoder initialization failed");

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

            (
                decoded,
                decoder.sample_rate(),
                decoder.channels(),
                decoder.bits_per_sample(),
            )
        }

        #[test]
        fn test_claxon_decode_waveform() {
            let input_path = testdata_path(&format!("flac/{}.flac", TEST_FILE));
            let flac_bytes = fs::read(&input_path).unwrap();
            assert!(!flac_bytes.is_empty(), "fixture flac missing or empty");

            init_tracing();

            let (decoded, sample_rate, channels, bits) = decode_with_claxon(&flac_bytes);

            assert!(
                !decoded.is_empty(),
                "claxon decoder produced no PCM samples"
            );
            assert_eq!(sample_rate, Some(16_000), "sample rate");
            assert_eq!(channels, Some(1), "channels");
            assert_eq!(bits, Some(16), "bits per sample");

            let result = DecodeResult::from_i32_with_bits(
                &decoded,
                sample_rate.unwrap_or(16000),
                channels.unwrap_or(1),
                bits.unwrap_or(16),
            );
            print_waveform_with_header("FLAC (claxon)", &result);
        }

        #[test]
        fn claxon_stream_stays_open_after_metadata_and_drains_without_duplicates() {
            let input_path = testdata_path(&format!("flac/{}.flac", TEST_FILE));
            let flac_bytes = fs::read(&input_path).unwrap();
            let metadata_end = flac_metadata_end(&flac_bytes);

            let mut reference = claxon::FlacReader::new(Cursor::new(&flac_bytes)).unwrap();
            let expected = reference.samples().collect::<Result<Vec<_>, _>>().unwrap();

            let mut decoder = FlacDecoderClaxon::new();
            decoder.init().unwrap();
            let mut scratch = vec![0i32; 257];
            assert_eq!(
                decoder
                    .decode_i32(&flac_bytes[..metadata_end], &mut scratch, false)
                    .unwrap(),
                0
            );

            let mut actual = Vec::new();
            for chunk in flac_bytes[metadata_end..].chunks(997) {
                drain_claxon(&mut decoder, chunk, &mut scratch, &mut actual);
            }
            drain_claxon(&mut decoder, &[], &mut scratch, &mut actual);
            assert_eq!(actual, expected);
        }

        #[test]
        fn claxon_skips_large_metadata_without_retaining_it() {
            let input_path = testdata_path(&format!("flac/{}.flac", TEST_FILE));
            let flac_bytes = fs::read(&input_path).unwrap();
            assert_eq!(&flac_bytes[..4], b"fLaC");
            assert_eq!(flac_bytes[4] & 0x7f, 0);

            let mut decoder = FlacDecoderClaxon::new();
            let mut scratch = [0i32; 64];
            let mut prefix = Vec::from(&b"fLaC"[..]);
            prefix.extend_from_slice(&[4, 0x80, 0, 0]);
            assert_eq!(decoder.decode_i32(&prefix, &mut scratch, false).unwrap(), 0);

            let zeros = [0u8; 64 * 1024];
            for _ in 0..128 {
                assert_eq!(decoder.decode_i32(&zeros, &mut scratch, false).unwrap(), 0);
                assert!(decoder.buffered_bytes() <= 4);
            }

            let mut streaminfo = vec![0x80, 0, 0, 34];
            streaminfo.extend_from_slice(&flac_bytes[8..42]);
            assert_eq!(
                decoder
                    .decode_i32(&streaminfo, &mut scratch, false)
                    .unwrap(),
                0
            );
            assert_eq!(decoder.sample_rate(), Some(16_000));
            assert!(decoder.buffered_bytes() <= 4);
        }

        fn drain_claxon(
            decoder: &mut FlacDecoderClaxon,
            input: &[u8],
            scratch: &mut [i32],
            output: &mut Vec<i32>,
        ) {
            let mut next = input;
            loop {
                let written = decoder.decode_i32(next, scratch, false).unwrap();
                output.extend_from_slice(&scratch[..written]);
                next = &[];
                if written == 0 {
                    break;
                }
            }
        }

        fn flac_metadata_end(bytes: &[u8]) -> usize {
            assert!(bytes.starts_with(b"fLaC"));
            let mut pos = 4usize;
            loop {
                let header = bytes[pos];
                let size = u32::from_be_bytes([0, bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]])
                    as usize;
                pos += 4 + size;
                if header & 0x80 != 0 {
                    return pos;
                }
            }
        }

        #[cfg(feature = "libflac")]
        #[test]
        fn test_compare_libflac_vs_claxon() {
            let input_path = testdata_path(&format!("flac/{}.flac", TEST_FILE));
            let flac_bytes = fs::read(&input_path).unwrap();
            assert!(!flac_bytes.is_empty(), "fixture flac missing or empty");

            init_tracing();

            let (libflac_samples, libflac_sr, libflac_ch, libflac_bits) =
                decode_with_libflac(&flac_bytes);
            let (claxon_samples, claxon_sr, claxon_ch, claxon_bits) =
                decode_with_claxon(&flac_bytes);

            // Compare metadata
            assert_eq!(libflac_sr, claxon_sr, "sample rate mismatch");
            assert_eq!(libflac_ch, claxon_ch, "channel count mismatch");
            assert_eq!(libflac_bits, claxon_bits, "bits per sample mismatch");

            println!(
                "  Sample counts: libflac={}, claxon={}",
                libflac_samples.len(),
                claxon_samples.len()
            );

            // libflac has issues draining the last frame, so claxon may have more samples
            // Compare the overlapping portion
            let min_len = libflac_samples.len().min(claxon_samples.len());

            // Compare actual samples (should be bit-exact for lossless codec)
            let mut mismatches = 0;
            let mut max_diff: i64 = 0;
            for i in 0..min_len {
                let a = libflac_samples[i];
                let b = claxon_samples[i];
                if a != b {
                    mismatches += 1;
                    let diff = (a as i64 - b as i64).abs();
                    if diff > max_diff {
                        max_diff = diff;
                    }
                    if mismatches <= 10 {
                        trace!(
                            sample = i,
                            libflac = a,
                            claxon = b,
                            diff = a - b,
                            "sample mismatch"
                        );
                    }
                }
            }

            if mismatches > 0 {
                println!(
                    "  Decoder comparison: {} mismatches out of {} samples compared, max diff: {}",
                    mismatches, min_len, max_diff
                );
            }

            // Both decoders should produce identical output for the overlapping portion
            assert_eq!(
                mismatches, 0,
                "libflac and claxon produced different output in overlapping range: {} mismatches",
                mismatches
            );

            // Claxon should produce at least as many samples as libflac
            assert!(
                claxon_samples.len() >= libflac_samples.len(),
                "claxon produced fewer samples than libflac"
            );

            println!(
                "  ✓ Decoders match on {} overlapping samples (claxon has {} extra samples)",
                min_len,
                claxon_samples.len().saturating_sub(libflac_samples.len())
            );
        }
    }
}
