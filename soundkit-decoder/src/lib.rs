use access_unit::{detect_audio, AudioType};
pub use bytes::Bytes;
use bytes::BytesMut;
use frame_header::{EncodingFlag, Endianness};
use rtrb::{Consumer, Producer, RingBuffer};
use soundkit::audio_bytes::{interleave_vecs_i16, s32le_to_i32};
use soundkit::audio_pipeline::{
    deserialize_audio, downsample_audio, vec_f32_to_i16, vec_i16_to_f32, vec_i32_to_f32,
};
use soundkit::audio_packet::Decoder;
use soundkit::audio_types::{AudioData, PcmData};
use soundkit_aac::{AacDecoder, AacDecoderMp4};
use soundkit_flac::FlacDecoderClaxon;
use soundkit_mp3::Mp3Decoder;
use soundkit_ogg_opus::OggOpusDecoder;
use soundkit_opus::OpusStreamDecoder;
use soundkit::wav::WavStreamProcessor;
use soundkit_webm::WebmDecoder;
use std::thread;

/// Unified streaming decoder trait - all decoders implement this interface.
/// This eliminates the need for codec-specific process_* and flush_* functions.
trait StreamingDecoder {
    /// Process a chunk of input data and return decoded audio frames.
    /// An empty chunk signals EOF but does not trigger flush.
    fn process(&mut self, chunk: &[u8]) -> Result<Vec<AudioData>, String>;

    /// Flush any remaining buffered data after EOF.
    fn flush(&mut self) -> Result<Vec<AudioData>, String>;
}

const MIN_DETECTION_BYTES: usize = 8192; // Increased for M4A/MP4 container detection
const MAX_DETECTION_BYTES: usize = 65_536;
const DEFAULT_INPUT_BUFFER: usize = 128;
const DEFAULT_OUTPUT_BUFFER: usize = 128;

/// Error types for decode pipeline
#[derive(Debug, Clone)]
pub enum DecodeError {
    FormatDetectionFailed,
    DecoderInitFailed(String),
    DecodingFailed(String),
    InputBufferFull,
    UnsupportedFormat(AudioType),
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecodeError::FormatDetectionFailed => write!(f, "Failed to detect audio format"),
            DecodeError::DecoderInitFailed(msg) => {
                write!(f, "Decoder initialization failed: {}", msg)
            }
            DecodeError::DecodingFailed(msg) => write!(f, "Decoding failed: {}", msg),
            DecodeError::InputBufferFull => write!(f, "Input buffer full"),
            DecodeError::UnsupportedFormat(fmt) => write!(f, "Unsupported format: {:?}", fmt),
        }
    }
}

impl std::error::Error for DecodeError {}

/// Output type for the pipeline
pub type DecodeOutput = Result<AudioData, DecodeError>;

/// Output transformation options for the decoder pipeline
#[derive(Debug, Clone, Copy)]
pub struct DecodeOptions {
    pub output_bits_per_sample: Option<u8>,
    pub output_sample_rate: Option<u32>,
    pub output_channels: Option<u8>,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self {
            output_bits_per_sample: None,
            output_sample_rate: None,
            output_channels: None,
        }
    }
}

/// Internal state machine for the pipeline
enum PipelineState {
    Detecting {
        buffer: BytesMut,
        bytes_collected: usize,
    },
    Decoding {
        decoder: FormatDecoder,
    },
}

/// Wrapper enum for different decoder types.
/// All variants implement StreamingDecoder through the enum's impl.
enum FormatDecoder {
    /// MP3 decoder using minimp3
    Mp3(Mp3Decoder),
    /// Raw AAC (ADTS) decoder
    Aac(AacDecoder),
    /// AAC decoder for M4A/MP4 containers
    M4a(AacDecoderMp4),
    /// FLAC decoder using pure-Rust claxon (avoids libFLAC FFI release-mode bug)
    Flac(FlacDecoderClaxon),
    /// Raw Opus stream decoder
    Opus(OpusStreamDecoder),
    /// Ogg-wrapped Opus decoder
    OggOpus(OggOpusDecoder),
    /// WebM container decoder (typically Opus audio)
    WebM(WebmDecoder),
    /// WAV decoder (raw PCM in RIFF container)
    Wav(WavStreamProcessor),
}

/// Helper to decode using the Decoder trait and drain all buffered frames.
/// Works for MP3, AAC, FLAC which use decode_i16/decode_i32 API.
fn decode_with_drain<D, F>(
    decoder: &mut D,
    chunk: &[u8],
    decode_fn: F,
) -> Result<Vec<AudioData>, String>
where
    D: Decoder,
    F: Fn(&D, usize, &[i32]) -> Option<AudioData>,
{
    let mut results = Vec::new();
    let mut output = vec![0i32; 262144];

    // First call with actual data
    let samples = decoder.decode_i32(chunk, &mut output, false)?;
    if samples > 0 {
        if let Some(audio_data) = decode_fn(decoder, samples, &output) {
            results.push(audio_data);
        }
    }

    // Drain remaining buffered frames
    loop {
        let samples = decoder.decode_i32(&[], &mut output, false)?;
        if samples == 0 {
            break;
        }
        if let Some(audio_data) = decode_fn(decoder, samples, &output) {
            results.push(audio_data);
        }
    }

    Ok(results)
}

/// Helper to decode using decode_i16 and drain all buffered frames.
fn decode_i16_with_drain<D, F>(
    decoder: &mut D,
    chunk: &[u8],
    decode_fn: F,
) -> Result<Vec<AudioData>, String>
where
    D: Decoder,
    F: Fn(&D, usize, &[i16]) -> Option<AudioData>,
{
    let mut results = Vec::new();
    let mut output = vec![0i16; 262144];

    // First call with actual data
    let samples = decoder.decode_i16(chunk, &mut output, false)?;
    if samples > 0 {
        if let Some(audio_data) = decode_fn(decoder, samples, &output) {
            results.push(audio_data);
        }
    }

    // Drain remaining buffered frames
    loop {
        let samples = decoder.decode_i16(&[], &mut output, false)?;
        if samples == 0 {
            break;
        }
        if let Some(audio_data) = decode_fn(decoder, samples, &output) {
            results.push(audio_data);
        }
    }

    Ok(results)
}

/// Helper to process using the add() API and drain all buffered packets.
/// Works for Opus, OggOpus, WebM which return Option<AudioData>.
fn process_with_add_api<D, F>(decoder: &mut D, chunk: &[u8], add_fn: F) -> Result<Vec<AudioData>, String>
where
    F: Fn(&mut D, &[u8]) -> Result<Option<AudioData>, String>,
{
    let mut results = Vec::new();

    // First call with actual data
    if let Some(audio_data) = add_fn(decoder, chunk)? {
        results.push(audio_data);
    } else {
        // No data yet, return early (don't drain if we haven't produced anything)
        return Ok(results);
    }

    // Drain remaining buffered packets
    loop {
        match add_fn(decoder, &[])? {
            Some(audio_data) => results.push(audio_data),
            None => break,
        }
    }

    Ok(results)
}

impl StreamingDecoder for FormatDecoder {
    fn process(&mut self, chunk: &[u8]) -> Result<Vec<AudioData>, String> {
        match self {
            FormatDecoder::Mp3(dec) => {
                decode_i16_with_drain(dec, chunk, |d, samples, output| {
                    let (sample_rate, channels) = (d.sample_rate()?, d.channels()?);
                    Some(create_audio_data_i16(sample_rate, channels, &output[..samples]))
                })
            }
            FormatDecoder::Aac(dec) => {
                decode_i16_with_drain(dec, chunk, |d, samples, output| {
                    let (sample_rate, channels) = (d.sample_rate()?, d.channels()?);
                    Some(create_audio_data_i16(sample_rate, channels, &output[..samples]))
                })
            }
            FormatDecoder::M4a(dec) => {
                decode_i16_with_drain(dec, chunk, |d, samples, output| {
                    let (sample_rate, channels) = (d.sample_rate()?, d.channels()?);
                    Some(create_audio_data_i16(sample_rate, channels, &output[..samples]))
                })
            }
            FormatDecoder::Flac(dec) => {
                decode_with_drain(dec, chunk, |d, samples, output| {
                    let (sample_rate, channels, bits) =
                        (d.sample_rate()?, d.channels()?, d.bits_per_sample()?);
                    Some(create_audio_data_i32_with_bits(sample_rate, channels, bits, &output[..samples]))
                })
            }
            FormatDecoder::Opus(dec) => {
                process_with_add_api(dec, chunk, |d, data| d.add(data))
            }
            FormatDecoder::OggOpus(dec) => {
                process_with_add_api(dec, chunk, |d, data| d.add(data))
            }
            FormatDecoder::WebM(dec) => {
                process_with_add_api(dec, chunk, |d, data| d.add(data))
            }
            FormatDecoder::Wav(dec) => {
                process_with_add_api(dec, chunk, |d, data| d.add(data))
            }
        }
    }

    fn flush(&mut self) -> Result<Vec<AudioData>, String> {
        // Flush is the same as process with empty input for all decoders
        self.process(&[])
    }
}

/// Main pipeline entry point
pub struct DecodePipeline;

impl DecodePipeline {
    /// Create and spawn a new decode pipeline with default buffer sizes
    pub fn spawn() -> DecodePipelineHandle {
        Self::spawn_with_buffers_and_options(
            DEFAULT_INPUT_BUFFER,
            DEFAULT_OUTPUT_BUFFER,
            DecodeOptions::default(),
        )
    }

    /// Create and spawn a new decode pipeline with output options
    pub fn spawn_with_options(options: DecodeOptions) -> DecodePipelineHandle {
        Self::spawn_with_buffers_and_options(DEFAULT_INPUT_BUFFER, DEFAULT_OUTPUT_BUFFER, options)
    }

    /// Spawn with custom ring buffer sizes
    ///
    /// - `input_buffer`: Number of input chunks that can be buffered
    /// - `output_buffer`: Number of decoded AudioData frames that can be buffered
    pub fn spawn_with_buffers(
        input_buffer: usize,
        output_buffer: usize,
    ) -> DecodePipelineHandle {
        Self::spawn_with_buffers_and_options(input_buffer, output_buffer, DecodeOptions::default())
    }

    /// Spawn with custom ring buffer sizes and output options
    ///
    /// - `input_buffer`: Number of input chunks that can be buffered
    /// - `output_buffer`: Number of decoded AudioData frames that can be buffered
    pub fn spawn_with_buffers_and_options(
        input_buffer: usize,
        output_buffer: usize,
        options: DecodeOptions,
    ) -> DecodePipelineHandle {
        let (input_tx, input_rx) = RingBuffer::<Bytes>::new(input_buffer);
        let (output_tx, output_rx) = RingBuffer::<DecodeOutput>::new(output_buffer);

        let worker = thread::spawn(move || {
            pipeline_worker(input_rx, output_tx, options);
        });

        DecodePipelineHandle {
            input_tx,
            output_rx,
            _worker: Some(worker),
        }
    }
}

/// Handle for interacting with the pipeline
pub struct DecodePipelineHandle {
    input_tx: Producer<Bytes>,
    output_rx: Consumer<DecodeOutput>,
    _worker: Option<thread::JoinHandle<()>>,
}

impl DecodePipelineHandle {
    /// Send encoded audio bytes to the pipeline (non-blocking)
    ///
    /// Returns `Err` if the ring buffer is full (backpressure)
    pub fn send(&mut self, data: Bytes) -> Result<(), DecodeError> {
        self.input_tx
            .push(data)
            .map_err(|_| DecodeError::InputBufferFull)
    }

    /// Try to receive a decoded audio frame without blocking
    ///
    /// Returns `None` if no data is available
    pub fn try_recv(&mut self) -> Option<DecodeOutput> {
        self.output_rx.pop().ok()
    }

    /// Receive a decoded audio frame, blocking until available
    ///
    /// Spins until data is available or the pipeline is closed
    pub fn recv(&mut self) -> Option<DecodeOutput> {
        loop {
            if let Ok(output) = self.output_rx.pop() {
                return Some(output);
            }
            if let Some(worker) = self._worker.as_ref() {
                if worker.is_finished() {
                    return None;
                }
            }
            // Small sleep to avoid busy-waiting
            std::thread::sleep(std::time::Duration::from_micros(100));
        }
    }

    /// Get the input producer (for sharing with other threads)
    ///
    /// Note: rtrb doesn't support cloning producers, so this consumes self
    pub fn split(self) -> (Producer<Bytes>, Consumer<DecodeOutput>) {
        (self.input_tx, self.output_rx)
    }
}

/// Main worker thread function
fn pipeline_worker(
    mut input_rx: Consumer<Bytes>,
    mut output_tx: Producer<DecodeOutput>,
    options: DecodeOptions,
) {
    let mut state = PipelineState::Detecting {
        buffer: BytesMut::new(),
        bytes_collected: 0,
    };

    loop {
        // Try to get next chunk
        let chunk = match input_rx.pop() {
            Ok(chunk) => chunk,
            Err(_) => {
                // Small sleep to avoid busy-waiting
                std::thread::sleep(std::time::Duration::from_micros(100));
                continue;
            }
        };

        // Empty chunk signals end-of-stream, initiate flush
        let is_eof = chunk.is_empty();

        let next_state = match state {
            PipelineState::Detecting {
                mut buffer,
                bytes_collected,
            } => {
                if is_eof {
                    // EOF during detection - try to decode with whatever we have
                    // Some formats (like Opus) can be detected with very little data
                    match detect_and_init_decoder(buffer.as_ref()) {
                        Ok(mut decoder) => {
                            process_with_decoder(
                                &mut decoder,
                                buffer.as_ref(),
                                &mut output_tx,
                                &options,
                            );
                            flush_decoder(&mut decoder, &mut output_tx, &options);
                        }
                        Err(e) => {
                            push_output(&mut output_tx, Err(e.clone()));
                        }
                    }
                    None
                } else {
                    buffer.extend_from_slice(&chunk);
                    let new_bytes_collected = bytes_collected + chunk.len();

                    // Try early detection for formats with clear magic bytes
                    // This allows smaller files (like short Opus) to be processed
                    // without waiting for MIN_DETECTION_BYTES
                    if new_bytes_collected >= MIN_DETECTION_BYTES {
                        match detect_and_init_decoder(buffer.as_ref()) {
                            Ok(mut decoder) => {
                                // Feed accumulated buffer to decoder
                                process_with_decoder(
                                    &mut decoder,
                                    buffer.as_ref(),
                                    &mut output_tx,
                                    &options,
                                );
                                Some(PipelineState::Decoding { decoder })
                            }
                            Err(_e) if new_bytes_collected < MAX_DETECTION_BYTES => {
                                // Need more data
                                Some(PipelineState::Detecting {
                                    buffer,
                                    bytes_collected: new_bytes_collected,
                                })
                            }
                            Err(e) => {
                                // Failed detection
                                push_output(&mut output_tx, Err(e.clone()));
                                None
                            }
                        }
                    } else {
                        Some(PipelineState::Detecting {
                            buffer,
                            bytes_collected: new_bytes_collected,
                        })
                    }
                }
            }

            PipelineState::Decoding { mut decoder } => {
                if is_eof {
                    flush_decoder(&mut decoder, &mut output_tx, &options);
                    None
                } else {
                    process_with_decoder(&mut decoder, chunk.as_ref(), &mut output_tx, &options);
                    Some(PipelineState::Decoding { decoder })
                }
            }

        };

        match next_state {
            Some(next_state) => state = next_state,
            None => break,
        }
    }
}

/// Detect format and initialize appropriate decoder
fn detect_and_init_decoder(buffer: &[u8]) -> Result<FormatDecoder, DecodeError> {
    let audio_type = detect_audio(buffer);
    match audio_type {
        AudioType::MP3 => {
            let decoder = Mp3Decoder::new();
            Ok(FormatDecoder::Mp3(decoder))
        }
        AudioType::AAC => {
            // Raw AAC (ADTS format)
            let mut decoder = AacDecoder::new();
            decoder
                .init()
                .map_err(|e| DecodeError::DecoderInitFailed(e))?;
            Ok(FormatDecoder::Aac(decoder))
        }
        AudioType::M4A => {
            // AAC in M4A/MP4 container
            let mut decoder = AacDecoderMp4::new();
            decoder
                .init()
                .map_err(|e| DecodeError::DecoderInitFailed(e))?;
            Ok(FormatDecoder::M4a(decoder))
        }
        AudioType::FLAC => {
            // Use pure-Rust claxon decoder (avoids libFLAC FFI release-mode bug)
            let mut decoder = FlacDecoderClaxon::new();
            decoder
                .init()
                .map_err(|e| DecodeError::DecoderInitFailed(e))?;
            Ok(FormatDecoder::Flac(decoder))
        }
        AudioType::Opus => {
            let mut decoder = OpusStreamDecoder::new();
            decoder
                .init()
                .map_err(|e| DecodeError::DecoderInitFailed(e))?;
            Ok(FormatDecoder::Opus(decoder))
        }
        AudioType::OggOpus => {
            let mut decoder = OggOpusDecoder::new();
            decoder
                .init()
                .map_err(|e| DecodeError::DecoderInitFailed(e))?;
            Ok(FormatDecoder::OggOpus(decoder))
        }
        AudioType::WebM => {
            let mut decoder = WebmDecoder::new();
            decoder
                .init()
                .map_err(|e| DecodeError::DecoderInitFailed(e))?;
            Ok(FormatDecoder::WebM(decoder))
        }
        AudioType::Wav => {
            let decoder = WavStreamProcessor::new();
            Ok(FormatDecoder::Wav(decoder))
        }
        AudioType::Unknown => Err(DecodeError::FormatDetectionFailed),
        other => Err(DecodeError::UnsupportedFormat(other)),
    }
}

/// Process a chunk through the decoder using the unified StreamingDecoder trait.
/// All codec-specific logic is handled inside the trait implementation.
fn process_with_decoder(
    decoder: &mut FormatDecoder,
    chunk: &[u8],
    output_tx: &mut Producer<DecodeOutput>,
    options: &DecodeOptions,
) {
    match decoder.process(chunk) {
        Ok(audio_frames) => {
            for audio_data in audio_frames {
                push_audio_data(output_tx, audio_data, options);
            }
        }
        Err(e) => {
            push_output(output_tx, Err(DecodeError::DecodingFailed(e)));
        }
    }
}

/// Flush remaining samples from decoder using the unified StreamingDecoder trait.
fn flush_decoder(
    decoder: &mut FormatDecoder,
    output_tx: &mut Producer<DecodeOutput>,
    options: &DecodeOptions,
) {
    match decoder.flush() {
        Ok(audio_frames) => {
            for audio_data in audio_frames {
                push_audio_data(output_tx, audio_data, options);
            }
        }
        Err(e) => {
            push_output(output_tx, Err(DecodeError::DecodingFailed(e)));
        }
    }
}

/// Create AudioData from i16 samples
fn create_audio_data_i16(sample_rate: u32, channels: u8, samples: &[i16]) -> AudioData {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for &sample in samples {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }

    AudioData::new(
        16, // bits_per_sample
        channels,
        sample_rate,
        bytes,
        EncodingFlag::PCMSigned,
        Endianness::LittleEndian,
    )
}

/// Create AudioData from i32 samples with specified bit depth.
/// FLAC stores samples as i32 but with values in the original bit depth range.
fn create_audio_data_i32_with_bits(sample_rate: u32, channels: u8, bits_per_sample: u8, samples: &[i32]) -> AudioData {
    let mut bytes = Vec::with_capacity(samples.len() * 4);
    for &sample in samples {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }

    AudioData::new(
        bits_per_sample,
        channels,
        sample_rate,
        bytes,
        EncodingFlag::PCMSigned,
        Endianness::LittleEndian,
    )
}

fn push_output(output_tx: &mut Producer<DecodeOutput>, output: DecodeOutput) {
    // Retry push with backoff instead of silently dropping frames
    let mut item = output;
    loop {
        match output_tx.push(item) {
            Ok(_) => return,
            Err(rtrb::PushError::Full(returned_item)) => {
                // Buffer is full, retry with the returned item
                item = returned_item;
                std::thread::sleep(std::time::Duration::from_micros(100));
            }
        }
    }
}

fn push_audio_data(
    output_tx: &mut Producer<DecodeOutput>,
    audio_data: AudioData,
    options: &DecodeOptions,
) {
    let output = match apply_output_options(audio_data, options) {
        Ok(audio_data) => Ok(audio_data),
        Err(err) => Err(err),
    };

    push_output(output_tx, output);
}

fn apply_output_options(
    audio_data: AudioData,
    options: &DecodeOptions,
) -> Result<AudioData, DecodeError> {
    let target_sample_rate = options
        .output_sample_rate
        .unwrap_or(audio_data.sampling_rate());
    let target_bits_per_sample = options
        .output_bits_per_sample
        .unwrap_or(audio_data.bits_per_sample());
    let target_channels = options
        .output_channels
        .unwrap_or(audio_data.channel_count());

    // Fast path: no transformations needed
    if target_sample_rate == audio_data.sampling_rate()
        && target_bits_per_sample == audio_data.bits_per_sample()
        && target_channels == audio_data.channel_count()
    {
        return Ok(audio_data);
    }

    if target_sample_rate == 0 {
        return Err(DecodeError::DecodingFailed(
            "Output sample rate must be > 0".to_string(),
        ));
    }

    if !matches!(target_bits_per_sample, 16 | 24 | 32) {
        return Err(DecodeError::DecodingFailed(format!(
            "Unsupported output bits per sample: {}",
            target_bits_per_sample
        )));
    }

    if target_channels == 0 {
        return Err(DecodeError::DecodingFailed(
            "Output channels must be > 0".to_string(),
        ));
    }

    // Convert to f32 channels, resampling if needed
    let mut channels = if target_sample_rate != audio_data.sampling_rate() {
        if audio_data.sampling_rate() == 0 {
            return Err(DecodeError::DecodingFailed(
                "Input sample rate must be > 0".to_string(),
            ));
        }

        downsample_audio(&audio_data, target_sample_rate as usize).map_err(|e| {
            DecodeError::DecodingFailed(format!("Output resample failed: {}", e))
        })?
    } else {
        audio_data_to_f32_channels(&audio_data).map_err(|e| {
            DecodeError::DecodingFailed(format!("Output conversion failed: {}", e))
        })?
    };

    // Downmix channels if needed
    let output_channel_count = if target_channels < channels.len() as u8 {
        channels = downmix_channels(&channels, target_channels);
        target_channels
    } else {
        channels.len() as u8
    };

    let output_format = if target_bits_per_sample == 32
        && audio_data.audio_format() == EncodingFlag::PCMFloat
    {
        EncodingFlag::PCMFloat
    } else {
        EncodingFlag::PCMSigned
    };

    let bytes = f32_channels_to_bytes(&channels, target_bits_per_sample, output_format)
        .map_err(|e| DecodeError::DecodingFailed(format!("Output conversion failed: {}", e)))?;

    Ok(AudioData::new(
        target_bits_per_sample,
        output_channel_count,
        target_sample_rate,
        bytes,
        output_format,
        Endianness::LittleEndian,
    ))
}

/// Downmix multiple channels to target channel count
fn downmix_channels(channels: &[Vec<f32>], target_channels: u8) -> Vec<Vec<f32>> {
    if channels.is_empty() || target_channels == 0 {
        return Vec::new();
    }

    let sample_count = channels[0].len();

    // Mono downmix: average all channels
    if target_channels == 1 {
        let mut mono = vec![0.0f32; sample_count];
        let scale = 1.0 / channels.len() as f32;
        for channel in channels {
            for (i, &sample) in channel.iter().enumerate() {
                mono[i] += sample * scale;
            }
        }
        return vec![mono];
    }

    // Stereo downmix from surround (5.1, 7.1, etc.)
    if target_channels == 2 && channels.len() > 2 {
        let mut left = vec![0.0f32; sample_count];
        let mut right = vec![0.0f32; sample_count];

        // Standard downmix coefficients
        // L' = L + 0.707*C + 0.707*Ls
        // R' = R + 0.707*C + 0.707*Rs
        let center_coef = 0.707f32;
        let surround_coef = 0.707f32;

        for i in 0..sample_count {
            left[i] = channels[0][i]; // Front Left
            right[i] = channels[1][i]; // Front Right

            if channels.len() > 2 {
                // Add center to both
                left[i] += center_coef * channels[2][i];
                right[i] += center_coef * channels[2][i];
            }
            if channels.len() > 4 {
                // Add surround channels
                left[i] += surround_coef * channels[4][i]; // Ls
                if channels.len() > 5 {
                    right[i] += surround_coef * channels[5][i]; // Rs
                }
            }
        }

        // Normalize to prevent clipping
        let max_val = left
            .iter()
            .chain(right.iter())
            .map(|&x| x.abs())
            .fold(0.0f32, f32::max);
        if max_val > 1.0 {
            let scale = 1.0 / max_val;
            for sample in &mut left {
                *sample *= scale;
            }
            for sample in &mut right {
                *sample *= scale;
            }
        }

        return vec![left, right];
    }

    // Fallback: just take first N channels
    channels[..target_channels as usize].to_vec()
}

fn audio_data_to_f32_channels(audio_data: &AudioData) -> Result<Vec<Vec<f32>>, String> {
    let channel_count = audio_data.channel_count() as usize;
    if channel_count == 0 {
        return Err("Channel count must be > 0".to_string());
    }

    if audio_data.bits_per_sample() == 32 && audio_data.audio_format() != EncodingFlag::PCMFloat {
        let interleaved = s32le_to_i32(audio_data.data());
        let mut channels =
            vec![Vec::with_capacity(interleaved.len() / channel_count); channel_count];
        for (index, sample) in interleaved.into_iter().enumerate() {
            channels[index % channel_count].push(sample);
        }
        return Ok(channels.into_iter().map(vec_i32_to_f32).collect());
    }

    let pcm_data = deserialize_audio(
        audio_data.data(),
        audio_data.bits_per_sample(),
        audio_data.channel_count(),
    )
    .map_err(|e| format!("deserialize_audio failed: {}", e))?;

    match pcm_data {
        PcmData::I16(data) => Ok(data.into_iter().map(vec_i16_to_f32).collect()),
        PcmData::I32(data) => Ok(data.into_iter().map(vec_i32_to_f32).collect()),
        PcmData::F32(data) => Ok(data),
    }
}

fn f32_channels_to_bytes(
    channels: &[Vec<f32>],
    bits_per_sample: u8,
    output_format: EncodingFlag,
) -> Result<Vec<u8>, String> {
    if channels.is_empty() {
        return Ok(Vec::new());
    }

    let sample_count = channels[0].len();
    if channels.iter().any(|channel| channel.len() != sample_count) {
        return Err("Channel length mismatch".to_string());
    }

    if output_format == EncodingFlag::PCMFloat {
        if bits_per_sample != 32 {
            return Err("PCMFloat output requires 32-bit samples".to_string());
        }
        return Ok(interleave_vecs_f32(channels));
    }

    match bits_per_sample {
        16 => {
            let channels_i16: Vec<Vec<i16>> =
                channels.iter().map(|c| vec_f32_to_i16(c.clone())).collect();
            Ok(interleave_vecs_i16(&channels_i16))
        }
        24 => {
            let channels_i32: Vec<Vec<i32>> =
                channels.iter().map(|c| vec_f32_to_s24(c)).collect();
            Ok(interleave_vecs_s24(channels_i32.as_slice()))
        }
        32 => {
            let channels_i32: Vec<Vec<i32>> =
                channels.iter().map(|c| vec_f32_to_i32(c)).collect();
            Ok(interleave_vecs_i32(channels_i32.as_slice()))
        }
        bits => Err(format!("Unsupported output bits per sample: {}", bits)),
    }
}

fn vec_f32_to_i32(input: &[f32]) -> Vec<i32> {
    let mut output = Vec::with_capacity(input.len());
    for &value in input {
        let clamped = value.max(-1.0).min(1.0);
        let sample = if clamped >= 0.0 {
            (clamped * i32::MAX as f32) as i32
        } else {
            (clamped * -(i32::MIN as f32)) as i32
        };
        output.push(sample);
    }
    output
}

fn vec_f32_to_s24(input: &[f32]) -> Vec<i32> {
    let mut output = Vec::with_capacity(input.len());
    let s24_max = 8_388_607.0;

    for &value in input {
        let clamped = value.max(-1.0).min(1.0);
        let sample = if clamped >= 0.0 {
            (clamped * s24_max) as i32
        } else {
            (clamped * (s24_max + 1.0)) as i32
        };
        output.push(sample);
    }
    output
}

fn interleave_vecs_i32(channels: &[Vec<i32>]) -> Vec<u8> {
    if channels.is_empty() {
        return Vec::new();
    }

    let channel_count = channels.len();
    let sample_count = channels[0].len();
    let mut result = Vec::with_capacity(channel_count * sample_count * 4);

    for i in 0..sample_count {
        for channel in channels {
            result.extend_from_slice(&channel[i].to_le_bytes());
        }
    }

    result
}

fn interleave_vecs_s24(channels: &[Vec<i32>]) -> Vec<u8> {
    if channels.is_empty() {
        return Vec::new();
    }

    let channel_count = channels.len();
    let sample_count = channels[0].len();
    let mut result = Vec::with_capacity(channel_count * sample_count * 3);

    for i in 0..sample_count {
        for channel in channels {
            let sample = channel[i].clamp(-8_388_608, 8_388_607);
            let bytes = sample.to_le_bytes();
            result.extend_from_slice(&bytes[..3]);
        }
    }

    result
}

fn interleave_vecs_f32(channels: &[Vec<f32>]) -> Vec<u8> {
    if channels.is_empty() {
        return Vec::new();
    }

    let channel_count = channels.len();
    let sample_count = channels[0].len();
    let mut result = Vec::with_capacity(channel_count * sample_count * 4);

    for i in 0..sample_count {
        for channel in channels {
            result.extend_from_slice(&channel[i].to_le_bytes());
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use std::path::PathBuf;

    fn testdata_path(file: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("testdata")
            .join(file)
    }

    fn golden_path(file: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("testdata")
            .join("golden")
            .join(file)
    }

    #[test]
    fn test_decode_mp3() {
        let data = Bytes::from(
            fs::read(testdata_path("mp3/A_Tusk_is_used_to_make_costly_gifts.mp3")).unwrap(),
        );

        let mut pipeline = DecodePipeline::spawn();
        pipeline.send(data).unwrap();
        pipeline.send(Bytes::new()).unwrap(); // EOF signal

        let mut frame_count = 0;
        let mut total_samples = 0;

        for _ in 0..100 {
            if let Some(result) = pipeline.try_recv() {
                match result {
                    Ok(audio_data) => {
                        assert_eq!(audio_data.bits_per_sample(), 16);
                        assert!(audio_data.sampling_rate() > 0);
                        assert!(audio_data.channel_count() > 0);
                        total_samples += audio_data.data().len() / 2;
                        frame_count += 1;

                        if frame_count >= 5 {
                            break;
                        }
                    }
                    Err(e) => panic!("Decode error: {:?}", e),
                }
            } else {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        }

        assert!(frame_count > 0, "No frames decoded");
        assert!(total_samples > 0, "No samples decoded");
    }

    #[test]
    fn test_decode_flac() {
        let data = Bytes::from(
            fs::read(testdata_path("flac/A_Tusk_is_used_to_make_costly_gifts.flac")).unwrap(),
        );

        let mut pipeline = DecodePipeline::spawn();
        pipeline.send(data).unwrap();
        pipeline.send(Bytes::new()).unwrap(); // EOF signal

        let mut frame_count = 0;

        for _ in 0..100 {
            if let Some(result) = pipeline.try_recv() {
                match result {
                    Ok(audio_data) => {
                        // FLAC correctly reports actual bit depth (16-bit test file)
                        assert_eq!(audio_data.bits_per_sample(), 16);
                        assert!(audio_data.sampling_rate() > 0);
                        frame_count += 1;

                        if frame_count >= 5 {
                            break;
                        }
                    }
                    Err(e) => panic!("Decode error: {:?}", e),
                }
            } else {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        }

        assert!(frame_count > 0, "No FLAC frames decoded");
    }

    #[test]
    fn test_decode_opus() {
        let data = Bytes::from(
            fs::read(testdata_path("opus/A_Tusk_is_used_to_make_costly_gifts.opus")).unwrap(),
        );

        let mut pipeline = DecodePipeline::spawn();
        pipeline.send(data).unwrap();
        pipeline.send(Bytes::new()).unwrap(); // EOF signal - required for small files

        let mut frame_count = 0;

        for _ in 0..100 {
            if let Some(result) = pipeline.try_recv() {
                match result {
                    Ok(audio_data) => {
                        assert_eq!(audio_data.bits_per_sample(), 16);
                        assert!(audio_data.sampling_rate() > 0);
                        frame_count += 1;

                        if frame_count >= 5 {
                            break;
                        }
                    }
                    Err(e) => panic!("Decode error: {:?}", e),
                }
            } else {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        }

        assert!(frame_count > 0, "No Opus frames decoded");
    }

    #[test]
    fn test_decode_ogg_opus() {
        let data = Bytes::from(
            fs::read(testdata_path("ogg_opus/A_Tusk_is_used_to_make_costly_gifts.ogg")).unwrap(),
        );

        let mut pipeline = DecodePipeline::spawn();
        pipeline.send(data).unwrap();
        pipeline.send(Bytes::new()).unwrap(); // EOF signal

        let mut frame_count = 0;

        for _ in 0..100 {
            if let Some(result) = pipeline.try_recv() {
                match result {
                    Ok(audio_data) => {
                        assert_eq!(audio_data.bits_per_sample(), 16);
                        assert!(audio_data.sampling_rate() > 0);
                        frame_count += 1;

                        if frame_count >= 5 {
                            break;
                        }
                    }
                    Err(e) => panic!("Decode error: {:?}", e),
                }
            } else {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        }

        assert!(frame_count > 0, "No Ogg Opus frames decoded");
    }

    #[test]
    fn test_decode_webm() {
        let data = Bytes::from(
            fs::read(testdata_path("webm/A_Tusk_is_used_to_make_costly_gifts.webm")).unwrap(),
        );

        let mut pipeline = DecodePipeline::spawn();
        pipeline.send(data).unwrap();
        pipeline.send(Bytes::new()).unwrap(); // EOF signal

        let mut frame_count = 0;

        for _ in 0..100 {
            if let Some(result) = pipeline.try_recv() {
                match result {
                    Ok(audio_data) => {
                        assert_eq!(audio_data.bits_per_sample(), 16);
                        assert!(audio_data.sampling_rate() > 0);
                        frame_count += 1;

                        if frame_count >= 5 {
                            break;
                        }
                    }
                    Err(e) => panic!("Decode error: {:?}", e),
                }
            } else {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        }

        assert!(frame_count > 0, "No WebM frames decoded");
    }

    #[test]
    fn test_chunked_input() {
        let data = Bytes::from(
            fs::read(testdata_path("mp3/A_Tusk_is_used_to_make_costly_gifts.mp3")).unwrap(),
        );

        let mut pipeline = DecodePipeline::spawn();

        // Send in small chunks
        for start in (0..data.len()).step_by(256) {
            let end = (start + 256).min(data.len());
            pipeline.send(data.slice(start..end)).unwrap();
        }

        let mut frame_count = 0;

        for _ in 0..100 {
            if let Some(result) = pipeline.try_recv() {
                match result {
                    Ok(_audio_data) => {
                        frame_count += 1;
                        if frame_count >= 3 {
                            break;
                        }
                    }
                    Err(e) => panic!("Decode error: {:?}", e),
                }
            } else {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        }

        assert!(frame_count > 0, "No frames decoded from chunked input");
    }


    #[test]
    fn test_detection_failure() {
        let garbage_data = Bytes::from(vec![0u8; 5000]);

        let mut pipeline = DecodePipeline::spawn();
        pipeline.send(garbage_data).unwrap();
        pipeline.send(Bytes::new()).unwrap(); // EOF to trigger detection

        for _ in 0..100 {
            if let Some(result) = pipeline.try_recv() {
                match result {
                    Err(DecodeError::FormatDetectionFailed) => {
                        // Expected
                        return;
                    }
                    other => panic!("Expected FormatDetectionFailed, got: {:?}", other),
                }
            } else {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        }

        panic!("Never received FormatDetectionFailed error");
    }

    fn select_sample_input(format_name: &str) -> PathBuf {
        let base_path = testdata_path(format_name);
        let mut files: Vec<PathBuf> = fs::read_dir(&base_path)
            .unwrap_or_else(|_| panic!("Missing testdata folder: {:?}", base_path))
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| path.is_file())
            .collect();

        files.sort();
        if files.is_empty() {
            panic!("No files found in {:?}", base_path);
        }

        let preferred = "A_Tusk_is_used_to_make_costly_gifts";
        if let Some(path) = files.iter().find(|path| {
            path.file_stem()
                .map(|stem| stem.to_string_lossy() == preferred)
                .unwrap_or(false)
        }) {
            return path.clone();
        }

        files[0].clone()
    }

    /// Decode audio file to 16kHz mono s16le format
    fn decode_to_s16le_16k_mono(
        input_path: &PathBuf,
        output_path: &PathBuf,
    ) -> Result<DecodeResult, DecodeError> {
        let data = Bytes::from(fs::read(input_path).unwrap_or_else(|e| {
            panic!("Failed to read {:?}: {}", input_path, e);
        }));

        let options = DecodeOptions {
            output_bits_per_sample: Some(16),
            output_sample_rate: Some(16_000),
            output_channels: Some(1), // Mono output
        };
        // Use large buffers to ensure we don't lose decoded frames during flush
        let mut pipeline = DecodePipeline::spawn_with_buffers_and_options(1024, 4096, options);

        // Send all data at once - decoders buffer internally and process incrementally
        pipeline.send(data).unwrap();

        // Send empty chunk to signal EOF and trigger flush
        pipeline.send(Bytes::new()).unwrap();

        let mut out_file = fs::File::create(output_path).unwrap_or_else(|e| {
            panic!("Failed to create {:?}: {}", output_path, e);
        });

        let mut total_bytes = 0usize;
        let mut sum_of_squares = 0.0f64;
        let mut all_samples: Vec<i16> = Vec::new();
        let mut idle_iters = 0u32;
        let max_idle_iters = 200u32; // Generous timeout to allow full flush
        let max_iters = 2000u32;

        for _ in 0..max_iters {
            if let Some(result) = pipeline.try_recv() {
                match result {
                    Ok(audio_data) => {
                        assert_eq!(audio_data.bits_per_sample(), 16);
                        assert_eq!(audio_data.sampling_rate(), 16_000);
                        assert_eq!(audio_data.channel_count(), 1); // Verify mono
                        if !audio_data.data().is_empty() {
                            // Collect samples for waveform and RMS
                            let samples_i16: Vec<i16> = audio_data
                                .data()
                                .chunks_exact(2)
                                .map(|b| i16::from_le_bytes([b[0], b[1]]))
                                .collect();
                            for &sample in &samples_i16 {
                                // Normalize to -1.0..1.0 range
                                let normalized = sample as f64 / 32768.0;
                                sum_of_squares += normalized * normalized;
                            }
                            all_samples.extend_from_slice(&samples_i16);

                            out_file.write_all(audio_data.data()).unwrap();
                            total_bytes += audio_data.data().len();
                        }
                        idle_iters = 0;
                    }
                    Err(e) => return Err(e),
                }
            } else {
                idle_iters += 1;
                if idle_iters >= max_idle_iters {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        }

        if total_bytes == 0 {
            return Err(DecodeError::DecodingFailed(format!(
                "No decoded output for {:?}",
                input_path
            )));
        }

        let sample_count = all_samples.len();
        let rms = if sample_count > 0 {
            (sum_of_squares / sample_count as f64).sqrt()
        } else {
            0.0
        };

        // Compute waveform peaks for visualization
        let waveform = compute_waveform_peaks(&all_samples, WAVEFORM_WIDTH * 2);

        Ok(DecodeResult {
            bytes: total_bytes,
            rms,
            waveform,
        })
    }

    /// Result from decoding
    struct DecodeResult {
        bytes: usize,
        rms: f64,
        waveform: Vec<f32>, // Peak values for waveform display
    }

    const WAVEFORM_WIDTH: usize = 60;
    const WAVEFORM_HEIGHT: usize = 8;

    /// Print ASCII waveform comparison for all formats
    fn print_waveform_chart(results: &[(&str, DecodeResult)]) {
        if results.is_empty() {
            return;
        }

        println!();
        println!("  Decoded Audio Waveforms (16kHz mono s16le)");
        println!("  {}", "═".repeat(70));
        println!();

        for (name, result) in results {
            let duration = result.bytes as f64 / 2.0 / 16_000.0;
            let db = if result.rms > 0.0 {
                20.0 * result.rms.log10()
            } else {
                -96.0
            };

            println!("  {} ({:.2}s, {:.1} dB)", name, duration, db);
            print_waveform(&result.waveform);
            println!();
        }
    }

    /// Print a single ASCII waveform
    fn print_waveform(peaks: &[f32]) {
        if peaks.is_empty() {
            println!("  (no audio data)");
            return;
        }

        // Characters for different amplitude levels (bottom to top)
        let chars = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

        // Resample peaks to fit display width
        let display_peaks: Vec<f32> = if peaks.len() > WAVEFORM_WIDTH {
            (0..WAVEFORM_WIDTH)
                .map(|i| {
                    let start = i * peaks.len() / WAVEFORM_WIDTH;
                    let end = ((i + 1) * peaks.len() / WAVEFORM_WIDTH).min(peaks.len());
                    peaks[start..end]
                        .iter()
                        .map(|x| x.abs())
                        .fold(0.0f32, f32::max)
                })
                .collect()
        } else {
            peaks.iter().map(|x| x.abs()).collect()
        };

        // Find max for normalization
        let max_peak = display_peaks.iter().fold(0.0f32, |a, &b| a.max(b)).max(0.001);

        // Build waveform lines (top half only, mirrored)
        let half_height = WAVEFORM_HEIGHT / 2;

        // Top half (positive)
        for row in (0..half_height).rev() {
            let threshold = (row as f32 + 0.5) / half_height as f32;
            let line: String = display_peaks
                .iter()
                .map(|&p| {
                    let normalized = p / max_peak;
                    if normalized >= threshold {
                        let level = ((normalized - threshold) * half_height as f32 * (chars.len() - 1) as f32) as usize;
                        chars[level.min(chars.len() - 1)]
                    } else {
                        ' '
                    }
                })
                .collect();
            println!("  │{}│", line);
        }

        // Center line
        println!("  ├{}┤", "─".repeat(display_peaks.len()));

        // Bottom half (mirrored)
        for row in 0..half_height {
            let threshold = (row as f32 + 0.5) / half_height as f32;
            let line: String = display_peaks
                .iter()
                .map(|&p| {
                    let normalized = p / max_peak;
                    if normalized >= threshold {
                        let level = ((normalized - threshold) * half_height as f32 * (chars.len() - 1) as f32) as usize;
                        chars[level.min(chars.len() - 1)]
                    } else {
                        ' '
                    }
                })
                .collect();
            println!("  │{}│", line);
        }
    }

    /// Compute waveform peaks from samples for visualization
    fn compute_waveform_peaks(samples: &[i16], num_bins: usize) -> Vec<f32> {
        if samples.is_empty() || num_bins == 0 {
            return Vec::new();
        }

        let bin_size = (samples.len() + num_bins - 1) / num_bins;

        samples
            .chunks(bin_size)
            .map(|chunk| {
                let max_abs = chunk.iter().map(|&s| (s as f32).abs()).fold(0.0f32, f32::max);
                max_abs / 32768.0 // Normalize to 0.0-1.0
            })
            .collect()
    }

    #[test]
    fn test_decode_all_formats_to_s16le_16k_mono() {
        let out_dir = golden_path("");
        fs::create_dir_all(&out_dir).unwrap();

        // Format: (input_dir, output_name)
        let formats = [
            ("flac", "flac"),
            ("opus", "opus"),
            ("ogg_opus", "ogg_opus"),
            ("aac", "aac"),
            ("m4a", "m4a"),
            ("mp3", "mp3"),
            ("webm", "webm"),
            ("wav", "wav"),
        ];

        let mut results: Vec<(&str, DecodeResult)> = Vec::new();

        for (dir_name, output_name) in formats {
            let input_path = select_sample_input(dir_name);
            let output_path = out_dir.join(format!("{}.s16le", output_name));
            match decode_to_s16le_16k_mono(&input_path, &output_path) {
                Ok(result) => {
                    results.push((output_name, result));
                }
                Err(e) => {
                    eprintln!("  {} - decode failed: {}", dir_name, e);
                }
            }
        }

        // Sort by duration for visual comparison
        results.sort_by_key(|(_, r)| r.bytes);

        print_waveform_chart(&results);
    }

    fn benchmark_format_folder(format_name: &str, options: Option<DecodeOptions>) {
        let base_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("testdata")
            .join(format_name);

        if !base_path.exists() {
            return;
        }

        let mut files = Vec::new();
        if let Ok(entries) = fs::read_dir(&base_path) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() {
                    files.push(path);
                }
            }
        }

        if files.is_empty() {
            return;
        }

        let start = std::time::Instant::now();
        let mut total_bytes = 0u64;
        let mut successful_files = 0;

        for file_path in &files {
            let data = match fs::read(file_path) {
                Ok(d) => d,
                Err(_) => continue,
            };

            let data = Bytes::from(data);
            total_bytes += data.len() as u64;

            let pipeline = match options {
                Some(opts) => DecodePipeline::spawn_with_buffers_and_options(256, 256, opts),
                None => DecodePipeline::spawn_with_buffers(256, 256),
            };
            let (mut producer, mut consumer) = pipeline.split();

            // Consumer thread to drain output
            let (done_tx, done_rx) = std::sync::mpsc::channel();
            let consumer_thread = thread::spawn(move || {
                let mut frame_count = 0;
                loop {
                    // Check if we should stop
                    if done_rx.try_recv().is_ok() {
                        // Drain remaining frames
                        while consumer.pop().is_ok() {
                            frame_count += 1;
                        }
                        break;
                    }

                    // Keep draining
                    while consumer.pop().is_ok() {
                        frame_count += 1;
                    }

                    std::thread::sleep(std::time::Duration::from_micros(10));
                }
                frame_count
            });

            // Stream data in chunks
            let chunk_size = 4096;
            for start in (0..data.len()).step_by(chunk_size) {
                let end = (start + chunk_size).min(data.len());
                while producer.push(data.slice(start..end)).is_err() {
                    // Buffer full, yield thread
                    std::thread::yield_now();
                }
            }

            while producer.push(Bytes::new()).is_err() {
                std::thread::yield_now();
            }

            // Wait for processing to complete
            std::thread::sleep(std::time::Duration::from_millis(20));

            // Signal consumer to finish and wait
            let _ = done_tx.send(());
            let _ = consumer_thread.join();

            successful_files += 1;
        }

        let elapsed = start.elapsed();
        let files_per_sec = successful_files as f64 / elapsed.as_secs_f64();
        let mb_per_sec = (total_bytes as f64 / 1_048_576.0) / elapsed.as_secs_f64();

        println!(
            "  {:<10} {:>3} files  {:>5.1}s  {:>5.1} files/s  {:>4.2} MB/s",
            format_name, successful_files, elapsed.as_secs_f64(), files_per_sec, mb_per_sec
        );
    }

    #[test]
    fn verify_resampling_works() {
        // Test with an MP3 file
        let data = Bytes::from(
            fs::read(testdata_path("mp3/A_Tusk_is_used_to_make_costly_gifts.mp3")).unwrap(),
        );
        println!("Loaded {} bytes", data.len());

        // Native decode
        let mut native_pipeline = DecodePipeline::spawn();
        native_pipeline.send(data.clone()).unwrap();
        native_pipeline.send(Bytes::new()).unwrap();

        let mut native_sr = 0u32;
        let mut native_ch = 0u8;
        let mut native_frames = 0;
        loop {
            match native_pipeline.recv() {
                Some(Ok(audio_data)) => {
                    if native_sr == 0 {
                        native_sr = audio_data.sampling_rate();
                        native_ch = audio_data.channel_count();
                    }
                    native_frames += 1;
                }
                Some(Err(e)) => panic!("Native decode error: {:?}", e),
                None => break,
            }
        }
        println!("Native:    {} Hz, {} ch, {} frames", native_sr, native_ch, native_frames);

        // Resampled decode
        let options = DecodeOptions {
            output_bits_per_sample: Some(16),
            output_sample_rate: Some(16_000),
            output_channels: Some(1),
        };
        let mut resample_pipeline = DecodePipeline::spawn_with_options(options);
        resample_pipeline.send(data).unwrap();
        resample_pipeline.send(Bytes::new()).unwrap();

        let mut resample_sr = 0u32;
        let mut resample_ch = 0u8;
        let mut resample_frames = 0;
        loop {
            match resample_pipeline.recv() {
                Some(Ok(audio_data)) => {
                    if resample_sr == 0 {
                        resample_sr = audio_data.sampling_rate();
                        resample_ch = audio_data.channel_count();
                    }
                    resample_frames += 1;
                }
                Some(Err(e)) => panic!("Resample decode error: {:?}", e),
                None => break,
            }
        }
        println!("Resampled: {} Hz, {} ch, {} frames", resample_sr, resample_ch, resample_frames);

        assert!(native_sr > 0, "Native sample rate should be detected, got {}", native_sr);
        assert_eq!(resample_sr, 16_000, "Resampled should be 16kHz");
        assert_eq!(resample_ch, 1, "Resampled should be mono");
    }

    #[test]
    #[ignore = "benchmark-only; run with cargo test -- --ignored"]
    fn bench_all_formats() {
        let formats = ["mp3", "flac", "opus", "ogg_opus", "mac_aac", "webm"];

        println!("\n=== Native (no sample rate/channel conversion) ===");
        for fmt in &formats {
            benchmark_format_folder(fmt, None);
        }

        let resample_opts = DecodeOptions {
            output_bits_per_sample: Some(16),
            output_sample_rate: Some(16_000),
            output_channels: Some(1),
        };

        println!("\n=== 16kHz Mono (resampled + downmixed) ===");
        for fmt in &formats {
            benchmark_format_folder(fmt, Some(resample_opts));
        }
    }
}
