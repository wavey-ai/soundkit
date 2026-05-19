use access_unit::{detect_audio, AudioType};
pub use bytes::Bytes;
use bytes::BytesMut;
use frame_header::{EncodingFlag, Endianness};
use rtrb::{Consumer, Producer, RingBuffer};
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use soundkit::audio_bytes::{interleave_vecs_i16, s32le_to_i32};
use soundkit::audio_packet::Decoder;
use soundkit::audio_pipeline::{deserialize_audio, vec_f32_to_i16, vec_i16_to_f32, vec_i32_to_f32};
use soundkit::audio_types::{AudioData, PcmData};
use soundkit::raw_pcm::RawPcmStreamProcessor;
pub use soundkit::raw_pcm::{RawPcmFormat, RawPcmSampleFormat};
use soundkit::wav::WavStreamProcessor;
use soundkit_aac::{AacDecoder, AacDecoderMp4};
use soundkit_ac3::Ac3Decoder;
use soundkit_aiff::AiffDecoder;
use soundkit_alac::AlacDecoder;
use soundkit_amr::AmrNbDecoder;
use soundkit_flac::FlacDecoderClaxon;
use soundkit_g711::G711Decoder;
pub use soundkit_g711::G711Law;
use soundkit_g722::G722Decoder;
use soundkit_g726::G726Decoder;
pub use soundkit_g726::{G726Packing, G726Rate};
use soundkit_g729::G729Decoder;
use soundkit_gsm::GsmDecoder;
pub use soundkit_gsm::GsmVariant;
use soundkit_mp3::Mp3Decoder;
use soundkit_ogg_opus::OggOpusDecoder;
use soundkit_opus::OpusStreamDecoder;
use soundkit_speex::SpeexDecoder;
use soundkit_vorbis::VorbisDecoder;
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
const RESAMPLE_CHUNK_SIZE: usize = 4096;

/// Error types for decode pipeline
#[derive(Debug, Clone)]
pub enum DecodeError {
    FormatDetectionFailed,
    DecoderInitFailed(String),
    DecodingFailed(String),
    InputBufferFull,
    UnsupportedFormat(AudioType),
    InvalidInputFormat(String),
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
            DecodeError::InvalidInputFormat(msg) => write!(f, "Invalid input format: {}", msg),
        }
    }
}

impl std::error::Error for DecodeError {}

/// Output type for the pipeline
pub type DecodeOutput = Result<AudioData, DecodeError>;

/// Output transformation options for the decoder pipeline
#[derive(Debug, Clone, Copy, Default)]
pub struct DecodeOptions {
    pub output_bits_per_sample: Option<u8>,
    pub output_sample_rate: Option<u32>,
    pub output_channels: Option<u8>,
}

/// Persistent resampler that preserves sinc filter state across decoded frames.
struct StreamingResampler {
    resampler: SincFixedIn<f32>,
    chunk_size: usize,
    channels: usize,
    input_sample_rate: u32,
    output_sample_rate: u32,
    target_bits_per_sample: u8,
    target_channels: u8,
    output_format: EncodingFlag,
    accum: Vec<Vec<f32>>,
}

impl StreamingResampler {
    fn new(
        input_sample_rate: u32,
        output_sample_rate: u32,
        channels: usize,
        target_bits_per_sample: u8,
        target_channels: u8,
        output_format: EncodingFlag,
    ) -> Result<Self, String> {
        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        };

        let resampler = SincFixedIn::<f32>::new(
            output_sample_rate as f64 / input_sample_rate as f64,
            2.0,
            params,
            RESAMPLE_CHUNK_SIZE,
            channels,
        )
        .map_err(|error| format!("Failed to create resampler: {error}"))?;

        Ok(Self {
            resampler,
            chunk_size: RESAMPLE_CHUNK_SIZE,
            channels,
            input_sample_rate,
            output_sample_rate,
            target_bits_per_sample,
            target_channels,
            output_format,
            accum: vec![Vec::new(); channels],
        })
    }

    fn process(&mut self, input: &[Vec<f32>]) -> Result<Vec<Vec<Vec<f32>>>, String> {
        if input.len() != self.channels {
            return Err(format!(
                "Channel count changed mid-stream: expected {}, got {}",
                self.channels,
                input.len()
            ));
        }

        for (channel, samples) in input.iter().enumerate() {
            self.accum[channel].extend_from_slice(samples);
        }

        let mut outputs = Vec::new();
        while self.accum[0].len() >= self.chunk_size {
            let chunk: Vec<Vec<f32>> = self
                .accum
                .iter_mut()
                .map(|channel| channel.drain(..self.chunk_size).collect())
                .collect();

            let resampled = self
                .resampler
                .process(&chunk, None)
                .map_err(|error| format!("Resample failed: {error}"))?;
            if resampled.iter().any(|channel| !channel.is_empty()) {
                outputs.push(resampled);
            }
        }

        Ok(outputs)
    }

    fn flush(&mut self) -> Result<Vec<Vec<Vec<f32>>>, String> {
        let mut outputs = Vec::new();

        if !self.accum[0].is_empty() {
            let remaining = self.accum[0].len();
            let padded_frames = self.chunk_size.saturating_sub(remaining);
            let chunk: Vec<Vec<f32>> = self
                .accum
                .iter_mut()
                .map(|channel| channel.drain(..).collect())
                .collect();
            let mut resampled = self
                .resampler
                .process_partial(Some(&chunk), None)
                .map_err(|error| format!("Resample partial failed: {error}"))?;
            if padded_frames > 0 {
                let trim = ((padded_frames as f64 * self.output_sample_rate as f64)
                    / self.input_sample_rate as f64)
                    .round() as usize;
                for channel in &mut resampled {
                    let keep = channel.len().saturating_sub(trim);
                    channel.truncate(keep);
                }
            }
            if resampled.iter().any(|channel| !channel.is_empty()) {
                outputs.push(resampled);
            }
        } else {
            let flushed = self
                .resampler
                .process_partial(None::<&[Vec<f32>]>, None)
                .map_err(|error| format!("Resample flush failed: {error}"))?;
            if flushed.iter().any(|channel| !channel.is_empty()) {
                outputs.push(flushed);
            }
        }

        Ok(outputs)
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
    Mp3(Box<Mp3Decoder>),
    /// Raw AAC (ADTS) decoder
    Aac(Box<AacDecoder>),
    /// AAC decoder for M4A/MP4 containers
    M4a(Box<AacDecoderMp4>),
    /// FLAC decoder using pure-Rust claxon (avoids libFLAC FFI release-mode bug)
    Flac(Box<FlacDecoderClaxon>),
    /// Raw Opus stream decoder
    Opus(Box<OpusStreamDecoder>),
    /// Ogg-wrapped Opus decoder
    OggOpus(Box<OggOpusDecoder>),
    /// WebM container decoder (Opus or Vorbis audio)
    WebM(Box<WebmDecoder>),
    /// WAV decoder (raw PCM in RIFF container)
    Wav(Box<WavStreamProcessor>),
    /// Headerless raw PCM stream with caller-provided metadata
    RawPcm(Box<RawPcmStreamProcessor>),
    /// AMR-NB file/raw frame stream
    AmrNb(Box<AmrNbDecoder>),
    /// Headerless G.711 stream with caller-provided law/sample metadata
    G711(Box<G711Decoder>),
    /// Headerless G.722 64 kbit/s mono wideband stream
    G722(Box<G722Decoder>),
    /// Headerless G.726 8 kHz mono ADPCM stream
    G726(Box<G726Decoder>),
    /// Headerless G.729 8 kbit/s mono stream
    G729(Box<G729Decoder>),
    /// Headerless GSM 06.10 8 kHz mono stream
    Gsm(Box<GsmDecoder>),
    /// Ogg-wrapped Speex stream
    Speex(Box<SpeexDecoder>),
    /// Ogg-wrapped Vorbis stream
    Vorbis(Box<VorbisDecoder>),
    /// ALAC in M4A/MP4 or CAF containers
    Alac(Box<AlacDecoder>),
    /// AIFF or AIFF-C container decoder
    Aiff(Box<AiffDecoder>),
    /// Raw AC-3 syncframe stream
    Ac3(Box<Ac3Decoder>),
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
fn process_with_add_api<D, F>(
    decoder: &mut D,
    chunk: &[u8],
    add_fn: F,
) -> Result<Vec<AudioData>, String>
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
    while let Some(audio_data) = add_fn(decoder, &[])? {
        results.push(audio_data);
    }

    Ok(results)
}

impl StreamingDecoder for FormatDecoder {
    fn process(&mut self, chunk: &[u8]) -> Result<Vec<AudioData>, String> {
        match self {
            FormatDecoder::Mp3(dec) => {
                decode_i16_with_drain(dec.as_mut(), chunk, |d, samples, output| {
                    let (sample_rate, channels) = (d.sample_rate()?, d.channels()?);
                    Some(create_audio_data_i16(
                        sample_rate,
                        channels,
                        &output[..samples],
                    ))
                })
            }
            FormatDecoder::Aac(dec) => {
                decode_i16_with_drain(dec.as_mut(), chunk, |d, samples, output| {
                    let (sample_rate, channels) = (d.sample_rate()?, d.channels()?);
                    Some(create_audio_data_i16(
                        sample_rate,
                        channels,
                        &output[..samples],
                    ))
                })
            }
            FormatDecoder::M4a(dec) => {
                decode_i16_with_drain(dec.as_mut(), chunk, |d, samples, output| {
                    let (sample_rate, channels) = (d.sample_rate()?, d.channels()?);
                    Some(create_audio_data_i16(
                        sample_rate,
                        channels,
                        &output[..samples],
                    ))
                })
            }
            FormatDecoder::Flac(dec) => {
                decode_with_drain(dec.as_mut(), chunk, |d, samples, output| {
                    let (sample_rate, channels, bits) =
                        (d.sample_rate()?, d.channels()?, d.bits_per_sample()?);
                    Some(create_audio_data_i32_with_bits(
                        sample_rate,
                        channels,
                        bits,
                        &output[..samples],
                    ))
                })
            }
            FormatDecoder::Opus(dec) => {
                process_with_add_api(dec.as_mut(), chunk, |d, data| d.add(data))
            }
            FormatDecoder::OggOpus(dec) => {
                process_with_add_api(dec.as_mut(), chunk, |d, data| d.add(data))
            }
            FormatDecoder::WebM(dec) => {
                process_with_add_api(dec.as_mut(), chunk, |d, data| d.add(data))
            }
            FormatDecoder::Wav(dec) => {
                process_with_add_api(dec.as_mut(), chunk, |d, data| d.add(data))
            }
            FormatDecoder::RawPcm(dec) => {
                process_with_add_api(dec.as_mut(), chunk, |d, data| d.add(data))
            }
            FormatDecoder::AmrNb(dec) => {
                decode_i16_with_drain(dec.as_mut(), chunk, |d, samples, output| {
                    Some(create_audio_data_i16(
                        d.sample_rate(),
                        d.channels(),
                        &output[..samples],
                    ))
                })
            }
            FormatDecoder::G711(dec) => {
                decode_i16_with_drain(dec.as_mut(), chunk, |d, samples, output| {
                    Some(create_audio_data_i16(
                        d.sample_rate(),
                        d.channels(),
                        &output[..samples],
                    ))
                })
            }
            FormatDecoder::G722(dec) => {
                decode_i16_with_drain(dec.as_mut(), chunk, |d, samples, output| {
                    Some(create_audio_data_i16(
                        d.sample_rate(),
                        d.channels(),
                        &output[..samples],
                    ))
                })
            }
            FormatDecoder::G726(dec) => {
                decode_i16_with_drain(dec.as_mut(), chunk, |d, samples, output| {
                    Some(create_audio_data_i16(
                        d.sample_rate(),
                        d.channels(),
                        &output[..samples],
                    ))
                })
            }
            FormatDecoder::G729(dec) => {
                decode_i16_with_drain(dec.as_mut(), chunk, |d, samples, output| {
                    Some(create_audio_data_i16(
                        d.sample_rate(),
                        d.channels(),
                        &output[..samples],
                    ))
                })
            }
            FormatDecoder::Gsm(dec) => {
                decode_i16_with_drain(dec.as_mut(), chunk, |d, samples, output| {
                    Some(create_audio_data_i16(
                        d.sample_rate(),
                        d.channels(),
                        &output[..samples],
                    ))
                })
            }
            FormatDecoder::Speex(dec) => {
                process_with_add_api(dec.as_mut(), chunk, |d, data| d.add(data))
            }
            FormatDecoder::Vorbis(dec) => {
                process_with_add_api(dec.as_mut(), chunk, |d, data| d.add(data))
            }
            FormatDecoder::Alac(dec) => {
                process_with_add_api(dec.as_mut(), chunk, |d, data| d.add(data))
            }
            FormatDecoder::Aiff(dec) => {
                process_with_add_api(dec.as_mut(), chunk, |d, data| d.add(data))
            }
            FormatDecoder::Ac3(dec) => {
                process_with_add_api(dec.as_mut(), chunk, |d, data| d.add(data))
            }
        }
    }

    fn flush(&mut self) -> Result<Vec<AudioData>, String> {
        match self {
            FormatDecoder::RawPcm(dec) => dec.flush().map(|frame| frame.into_iter().collect()),
            FormatDecoder::AmrNb(dec) => {
                dec.flush()?;
                Ok(Vec::new())
            }
            FormatDecoder::G729(dec) => {
                dec.flush()?;
                Ok(Vec::new())
            }
            FormatDecoder::G726(dec) => {
                dec.flush()?;
                Ok(Vec::new())
            }
            FormatDecoder::Gsm(dec) => {
                dec.flush()?;
                Ok(Vec::new())
            }
            _ => self.process(&[]),
        }
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
    pub fn spawn_with_buffers(input_buffer: usize, output_buffer: usize) -> DecodePipelineHandle {
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
        Self::spawn_with_initial_decoder(input_buffer, output_buffer, options, None)
    }

    /// Create a pipeline for headerless raw PCM using caller-provided input metadata.
    pub fn spawn_raw_pcm(format: RawPcmFormat) -> DecodePipelineHandle {
        Self::spawn_raw_pcm_with_options(format, DecodeOptions::default())
    }

    /// Create a raw PCM pipeline with output conversion options.
    pub fn spawn_raw_pcm_with_options(
        format: RawPcmFormat,
        options: DecodeOptions,
    ) -> DecodePipelineHandle {
        let decoder = FormatDecoder::RawPcm(Box::new(RawPcmStreamProcessor::new(format)));
        Self::spawn_with_initial_decoder(
            DEFAULT_INPUT_BUFFER,
            DEFAULT_OUTPUT_BUFFER,
            options,
            Some(decoder),
        )
    }

    /// Create a pipeline for AMR-NB streams.
    ///
    /// Both 3GPP AMR files with `#!AMR\n` magic and raw AMR-NB frame streams
    /// are accepted. AMR-WB is intentionally separate because it has a
    /// different sample rate, frame size, and OpenCORE API.
    pub fn spawn_amr_nb() -> Result<DecodePipelineHandle, DecodeError> {
        Self::spawn_amr_nb_with_options(DecodeOptions::default())
    }

    /// Create an AMR-NB pipeline with output conversion options.
    pub fn spawn_amr_nb_with_options(
        options: DecodeOptions,
    ) -> Result<DecodePipelineHandle, DecodeError> {
        let decoder = FormatDecoder::AmrNb(Box::new(
            AmrNbDecoder::try_new().map_err(DecodeError::DecoderInitFailed)?,
        ));
        Ok(Self::spawn_with_initial_decoder(
            DEFAULT_INPUT_BUFFER,
            DEFAULT_OUTPUT_BUFFER,
            options,
            Some(decoder),
        ))
    }

    /// Create a pipeline for headerless G.711 streams.
    ///
    /// These streams cannot be autodetected reliably, so the law, sample rate,
    /// and channel count must come from the transport layer or integration.
    pub fn spawn_g711(
        law: G711Law,
        sample_rate: u32,
        channels: u8,
    ) -> Result<DecodePipelineHandle, DecodeError> {
        Self::spawn_g711_with_options(law, sample_rate, channels, DecodeOptions::default())
    }

    /// Create a G.711 pipeline with output conversion options.
    pub fn spawn_g711_with_options(
        law: G711Law,
        sample_rate: u32,
        channels: u8,
        options: DecodeOptions,
    ) -> Result<DecodePipelineHandle, DecodeError> {
        if sample_rate == 0 {
            return Err(DecodeError::InvalidInputFormat(
                "G.711 sample rate must be > 0".to_string(),
            ));
        }
        if channels == 0 {
            return Err(DecodeError::InvalidInputFormat(
                "G.711 channel count must be > 0".to_string(),
            ));
        }

        let decoder = FormatDecoder::G711(Box::new(G711Decoder::new_with_law(
            law,
            sample_rate,
            channels,
        )));
        Ok(Self::spawn_with_initial_decoder(
            DEFAULT_INPUT_BUFFER,
            DEFAULT_OUTPUT_BUFFER,
            options,
            Some(decoder),
        ))
    }

    /// Create a pipeline for headerless G.722 64 kbit/s mono wideband streams.
    pub fn spawn_g722() -> DecodePipelineHandle {
        Self::spawn_g722_with_options(DecodeOptions::default())
    }

    /// Create a G.722 pipeline with output conversion options.
    pub fn spawn_g722_with_options(options: DecodeOptions) -> DecodePipelineHandle {
        let decoder = FormatDecoder::G722(Box::new(G722Decoder::new_64k()));
        Self::spawn_with_initial_decoder(
            DEFAULT_INPUT_BUFFER,
            DEFAULT_OUTPUT_BUFFER,
            options,
            Some(decoder),
        )
    }

    /// Create a pipeline for headerless G.726-32 8 kHz mono streams.
    ///
    /// Use `G726Packing::Left` for raw `ffmpeg -f g726` streams and
    /// `G726Packing::Right` for `ffmpeg -f g726le` streams.
    pub fn spawn_g726(packing: G726Packing) -> Result<DecodePipelineHandle, DecodeError> {
        Self::spawn_g726_with_options(packing, DecodeOptions::default())
    }

    /// Create a G.726-32 pipeline with output conversion options.
    pub fn spawn_g726_with_options(
        packing: G726Packing,
        options: DecodeOptions,
    ) -> Result<DecodePipelineHandle, DecodeError> {
        Self::spawn_g726_with_rate_and_options(G726Rate::Rate32000, packing, options)
    }

    /// Create a pipeline for headerless G.726 8 kHz mono streams at the selected bit rate.
    pub fn spawn_g726_with_rate(
        rate: G726Rate,
        packing: G726Packing,
    ) -> Result<DecodePipelineHandle, DecodeError> {
        Self::spawn_g726_with_rate_and_options(rate, packing, DecodeOptions::default())
    }

    /// Create a G.726 pipeline with selected bit rate and output conversion options.
    pub fn spawn_g726_with_rate_and_options(
        rate: G726Rate,
        packing: G726Packing,
        options: DecodeOptions,
    ) -> Result<DecodePipelineHandle, DecodeError> {
        let decoder = FormatDecoder::G726(Box::new(
            G726Decoder::try_new(rate, packing).map_err(DecodeError::DecoderInitFailed)?,
        ));
        Ok(Self::spawn_with_initial_decoder(
            DEFAULT_INPUT_BUFFER,
            DEFAULT_OUTPUT_BUFFER,
            options,
            Some(decoder),
        ))
    }

    /// Create a pipeline for headerless G.729 8 kbit/s mono streams.
    pub fn spawn_g729() -> Result<DecodePipelineHandle, DecodeError> {
        Self::spawn_g729_with_options(DecodeOptions::default())
    }

    /// Create a G.729 pipeline with output conversion options.
    pub fn spawn_g729_with_options(
        options: DecodeOptions,
    ) -> Result<DecodePipelineHandle, DecodeError> {
        let decoder = FormatDecoder::G729(Box::new(
            G729Decoder::try_new().map_err(DecodeError::DecoderInitFailed)?,
        ));
        Ok(Self::spawn_with_initial_decoder(
            DEFAULT_INPUT_BUFFER,
            DEFAULT_OUTPUT_BUFFER,
            options,
            Some(decoder),
        ))
    }

    /// Create a pipeline for headerless GSM 06.10 streams.
    ///
    /// Use `GsmVariant::Standard` for raw `.gsm`/ETSI 33-byte frames and
    /// `GsmVariant::Microsoft` for WAV-49 / `gsm_ms` 65-byte two-frame packets.
    pub fn spawn_gsm(variant: GsmVariant) -> Result<DecodePipelineHandle, DecodeError> {
        Self::spawn_gsm_with_options(variant, DecodeOptions::default())
    }

    /// Create a GSM pipeline with output conversion options.
    pub fn spawn_gsm_with_options(
        variant: GsmVariant,
        options: DecodeOptions,
    ) -> Result<DecodePipelineHandle, DecodeError> {
        let decoder = FormatDecoder::Gsm(Box::new(
            GsmDecoder::try_new(variant).map_err(DecodeError::DecoderInitFailed)?,
        ));
        Ok(Self::spawn_with_initial_decoder(
            DEFAULT_INPUT_BUFFER,
            DEFAULT_OUTPUT_BUFFER,
            options,
            Some(decoder),
        ))
    }

    /// Create a pipeline for Ogg-wrapped Speex streams.
    ///
    /// Speex-in-Ogg is not currently autodetected by `access-unit`, so callers
    /// should choose this explicit path when the transport/container is known.
    pub fn spawn_speex() -> DecodePipelineHandle {
        Self::spawn_speex_with_options(DecodeOptions::default())
    }

    /// Create a Speex pipeline with output conversion options.
    pub fn spawn_speex_with_options(options: DecodeOptions) -> DecodePipelineHandle {
        let mut decoder = SpeexDecoder::new();
        let _ = decoder.init();
        let decoder = FormatDecoder::Speex(Box::new(decoder));
        Self::spawn_with_initial_decoder(
            DEFAULT_INPUT_BUFFER,
            DEFAULT_OUTPUT_BUFFER,
            options,
            Some(decoder),
        )
    }

    /// Create a pipeline for Ogg-wrapped Vorbis streams.
    pub fn spawn_vorbis() -> DecodePipelineHandle {
        Self::spawn_vorbis_with_options(DecodeOptions::default())
    }

    /// Create a Vorbis pipeline with output conversion options.
    pub fn spawn_vorbis_with_options(options: DecodeOptions) -> DecodePipelineHandle {
        let mut decoder = VorbisDecoder::new();
        let _ = decoder.init();
        let decoder = FormatDecoder::Vorbis(Box::new(decoder));
        Self::spawn_with_initial_decoder(
            DEFAULT_INPUT_BUFFER,
            DEFAULT_OUTPUT_BUFFER,
            options,
            Some(decoder),
        )
    }

    /// Create a pipeline for ALAC in M4A/MP4 or CAF containers.
    pub fn spawn_alac() -> DecodePipelineHandle {
        Self::spawn_alac_with_options(DecodeOptions::default())
    }

    /// Create an ALAC pipeline with output conversion options.
    pub fn spawn_alac_with_options(options: DecodeOptions) -> DecodePipelineHandle {
        let mut decoder = AlacDecoder::new();
        let _ = decoder.init();
        let decoder = FormatDecoder::Alac(Box::new(decoder));
        Self::spawn_with_initial_decoder(
            DEFAULT_INPUT_BUFFER,
            DEFAULT_OUTPUT_BUFFER,
            options,
            Some(decoder),
        )
    }

    /// Create a pipeline for AIFF or AIFF-C containers.
    pub fn spawn_aiff() -> DecodePipelineHandle {
        Self::spawn_aiff_with_options(DecodeOptions::default())
    }

    /// Create an AIFF/AIFF-C pipeline with output conversion options.
    pub fn spawn_aiff_with_options(options: DecodeOptions) -> DecodePipelineHandle {
        let mut decoder = AiffDecoder::new();
        let _ = decoder.init();
        let decoder = FormatDecoder::Aiff(Box::new(decoder));
        Self::spawn_with_initial_decoder(
            DEFAULT_INPUT_BUFFER,
            DEFAULT_OUTPUT_BUFFER,
            options,
            Some(decoder),
        )
    }

    /// Create a pipeline for raw AC-3 syncframe streams.
    pub fn spawn_ac3() -> Result<DecodePipelineHandle, DecodeError> {
        Self::spawn_ac3_with_options(DecodeOptions::default())
    }

    /// Create a raw AC-3 pipeline with output conversion options.
    pub fn spawn_ac3_with_options(
        options: DecodeOptions,
    ) -> Result<DecodePipelineHandle, DecodeError> {
        let decoder = FormatDecoder::Ac3(Box::new(
            Ac3Decoder::try_new().map_err(DecodeError::DecoderInitFailed)?,
        ));
        Ok(Self::spawn_with_initial_decoder(
            DEFAULT_INPUT_BUFFER,
            DEFAULT_OUTPUT_BUFFER,
            options,
            Some(decoder),
        ))
    }

    fn spawn_with_initial_decoder(
        input_buffer: usize,
        output_buffer: usize,
        options: DecodeOptions,
        initial_decoder: Option<FormatDecoder>,
    ) -> DecodePipelineHandle {
        let (input_tx, input_rx) = RingBuffer::<Bytes>::new(input_buffer);
        let (output_tx, output_rx) = RingBuffer::<DecodeOutput>::new(output_buffer);

        let worker = thread::spawn(move || {
            pipeline_worker(input_rx, output_tx, options, initial_decoder);
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
    initial_decoder: Option<FormatDecoder>,
) {
    let mut resampler: Option<StreamingResampler> = None;
    let mut state = match initial_decoder {
        Some(decoder) => PipelineState::Decoding { decoder },
        None => PipelineState::Detecting {
            buffer: BytesMut::new(),
            bytes_collected: 0,
        },
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
                                &mut resampler,
                            );
                            flush_decoder(&mut decoder, &mut output_tx, &options, &mut resampler);
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
                                    &mut resampler,
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
                    flush_decoder(&mut decoder, &mut output_tx, &options, &mut resampler);
                    None
                } else {
                    process_with_decoder(
                        &mut decoder,
                        chunk.as_ref(),
                        &mut output_tx,
                        &options,
                        &mut resampler,
                    );
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
            Ok(FormatDecoder::Mp3(Box::new(decoder)))
        }
        AudioType::AAC => {
            // Raw AAC (ADTS format)
            let mut decoder = AacDecoder::new();
            decoder.init().map_err(DecodeError::DecoderInitFailed)?;
            Ok(FormatDecoder::Aac(Box::new(decoder)))
        }
        AudioType::M4A => {
            // AAC in M4A/MP4 container
            let mut decoder = AacDecoderMp4::new();
            decoder.init().map_err(DecodeError::DecoderInitFailed)?;
            Ok(FormatDecoder::M4a(Box::new(decoder)))
        }
        AudioType::FLAC => {
            // Use pure-Rust claxon decoder (avoids libFLAC FFI release-mode bug)
            let mut decoder = FlacDecoderClaxon::new();
            decoder.init().map_err(DecodeError::DecoderInitFailed)?;
            Ok(FormatDecoder::Flac(Box::new(decoder)))
        }
        AudioType::Opus => {
            let mut decoder = OpusStreamDecoder::new();
            decoder.init().map_err(DecodeError::DecoderInitFailed)?;
            Ok(FormatDecoder::Opus(Box::new(decoder)))
        }
        AudioType::OggOpus => {
            let mut decoder = OggOpusDecoder::new();
            decoder.init().map_err(DecodeError::DecoderInitFailed)?;
            Ok(FormatDecoder::OggOpus(Box::new(decoder)))
        }
        AudioType::OggVorbis => {
            let mut decoder = VorbisDecoder::new();
            decoder.init().map_err(DecodeError::DecoderInitFailed)?;
            Ok(FormatDecoder::Vorbis(Box::new(decoder)))
        }
        AudioType::OggSpeex => {
            let mut decoder = SpeexDecoder::new();
            decoder.init().map_err(DecodeError::DecoderInitFailed)?;
            Ok(FormatDecoder::Speex(Box::new(decoder)))
        }
        AudioType::WebM => {
            let mut decoder = WebmDecoder::new();
            decoder.init().map_err(DecodeError::DecoderInitFailed)?;
            Ok(FormatDecoder::WebM(Box::new(decoder)))
        }
        AudioType::Wav => {
            let decoder = WavStreamProcessor::new();
            Ok(FormatDecoder::Wav(Box::new(decoder)))
        }
        AudioType::ALAC => {
            let mut decoder = AlacDecoder::new();
            decoder.init().map_err(DecodeError::DecoderInitFailed)?;
            Ok(FormatDecoder::Alac(Box::new(decoder)))
        }
        AudioType::AIFF => {
            let mut decoder = AiffDecoder::new();
            decoder.init().map_err(DecodeError::DecoderInitFailed)?;
            Ok(FormatDecoder::Aiff(Box::new(decoder)))
        }
        AudioType::AC3 => {
            let decoder = Ac3Decoder::try_new().map_err(DecodeError::DecoderInitFailed)?;
            Ok(FormatDecoder::Ac3(Box::new(decoder)))
        }
        AudioType::Unknown => Err(DecodeError::FormatDetectionFailed),
    }
}

/// Process a chunk through the decoder using the unified StreamingDecoder trait.
/// All codec-specific logic is handled inside the trait implementation.
fn process_with_decoder(
    decoder: &mut FormatDecoder,
    chunk: &[u8],
    output_tx: &mut Producer<DecodeOutput>,
    options: &DecodeOptions,
    resampler: &mut Option<StreamingResampler>,
) {
    match decoder.process(chunk) {
        Ok(audio_frames) => {
            for audio_data in audio_frames {
                push_audio_data(output_tx, audio_data, options, resampler);
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
    resampler: &mut Option<StreamingResampler>,
) {
    match decoder.flush() {
        Ok(audio_frames) => {
            for audio_data in audio_frames {
                push_audio_data(output_tx, audio_data, options, resampler);
            }
        }
        Err(e) => {
            push_output(output_tx, Err(DecodeError::DecodingFailed(e)));
        }
    }

    if let Some(pending) = resampler.take() {
        flush_pending_resampler(output_tx, pending);
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
/// This function converts to the appropriate byte representation.
fn create_audio_data_i32_with_bits(
    sample_rate: u32,
    channels: u8,
    bits_per_sample: u8,
    samples: &[i32],
) -> AudioData {
    let bytes_per_sample = bits_per_sample.div_ceil(8) as usize;
    let mut bytes = Vec::with_capacity(samples.len() * bytes_per_sample);

    match bits_per_sample {
        1..=8 => {
            // 8-bit: samples are in range -128 to 127, convert to unsigned 0-255
            for &sample in samples {
                bytes.push((sample + 128) as u8);
            }
        }
        9..=16 => {
            // 16-bit: samples are in range -32768 to 32767
            for &sample in samples {
                bytes.extend_from_slice(&(sample as i16).to_le_bytes());
            }
        }
        17..=24 => {
            // 24-bit: write 3 bytes per sample
            for &sample in samples {
                let le = sample.to_le_bytes();
                bytes.extend_from_slice(&le[0..3]);
            }
        }
        _ => {
            // 32-bit: write full i32
            for &sample in samples {
                bytes.extend_from_slice(&sample.to_le_bytes());
            }
        }
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
    resampler: &mut Option<StreamingResampler>,
) {
    match apply_output_options(audio_data, options, resampler) {
        Ok(frames) => {
            for frame in frames {
                push_output(output_tx, Ok(frame));
            }
        }
        Err(error) => push_output(output_tx, Err(error)),
    }
}

fn emit_resampled_chunks(
    chunks: Vec<Vec<Vec<f32>>>,
    target_bits_per_sample: u8,
    target_channels: u8,
    target_sample_rate: u32,
    output_format: EncodingFlag,
) -> Result<Vec<AudioData>, DecodeError> {
    let mut out = Vec::with_capacity(chunks.len());
    for mut channels in chunks {
        let output_channel_count = if target_channels < channels.len() as u8 {
            channels = downmix_channels(&channels, target_channels);
            target_channels
        } else {
            channels.len() as u8
        };

        let bytes = f32_channels_to_bytes(&channels, target_bits_per_sample, output_format)
            .map_err(|e| DecodeError::DecodingFailed(format!("Output conversion failed: {e}")))?;

        out.push(AudioData::new(
            target_bits_per_sample,
            output_channel_count,
            target_sample_rate,
            bytes,
            output_format,
            Endianness::LittleEndian,
        ));
    }
    Ok(out)
}

fn flush_resampler_frames(
    mut resampler: StreamingResampler,
) -> Result<Vec<AudioData>, DecodeError> {
    let chunks = resampler
        .flush()
        .map_err(|error| DecodeError::DecodingFailed(format!("Resampler flush failed: {error}")))?;
    emit_resampled_chunks(
        chunks,
        resampler.target_bits_per_sample,
        resampler.target_channels,
        resampler.output_sample_rate,
        resampler.output_format,
    )
}

fn flush_pending_resampler(output_tx: &mut Producer<DecodeOutput>, resampler: StreamingResampler) {
    match flush_resampler_frames(resampler) {
        Ok(frames) => {
            for frame in frames {
                push_output(output_tx, Ok(frame));
            }
        }
        Err(error) => push_output(output_tx, Err(error)),
    }
}

fn apply_output_options(
    audio_data: AudioData,
    options: &DecodeOptions,
    resampler: &mut Option<StreamingResampler>,
) -> Result<Vec<AudioData>, DecodeError> {
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
        return Ok(vec![audio_data]);
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

    let output_format =
        if target_bits_per_sample == 32 && audio_data.audio_format() == EncodingFlag::PCMFloat {
            EncodingFlag::PCMFloat
        } else {
            EncodingFlag::PCMSigned
        };

    // Convert to f32 channels, resampling if needed
    let mut channels = if target_sample_rate != audio_data.sampling_rate() {
        if audio_data.sampling_rate() == 0 {
            return Err(DecodeError::DecodingFailed(
                "Input sample rate must be > 0".to_string(),
            ));
        }

        let input_channels = audio_data_to_f32_channels(&audio_data)
            .map_err(|e| DecodeError::DecodingFailed(format!("Output conversion failed: {e}")))?;

        if let Some(active) = resampler.as_ref() {
            if active.input_sample_rate != audio_data.sampling_rate()
                || active.channels != input_channels.len()
                || active.output_sample_rate != target_sample_rate
            {
                return Err(DecodeError::DecodingFailed(
                    "Resampler configuration changed mid-stream".to_string(),
                ));
            }
        } else {
            *resampler = Some(
                StreamingResampler::new(
                    audio_data.sampling_rate(),
                    target_sample_rate,
                    input_channels.len(),
                    target_bits_per_sample,
                    target_channels,
                    output_format,
                )
                .map_err(DecodeError::DecodingFailed)?,
            );
        }

        let active = resampler
            .as_mut()
            .expect("resampler must exist after initialization");
        let pending = active
            .process(&input_channels)
            .map_err(DecodeError::DecodingFailed)?;

        return emit_resampled_chunks(
            pending,
            target_bits_per_sample,
            target_channels,
            target_sample_rate,
            active.output_format,
        );
    } else {
        audio_data_to_f32_channels(&audio_data)
            .map_err(|e| DecodeError::DecodingFailed(format!("Output conversion failed: {e}")))?
    };

    // Downmix channels if needed
    let output_channel_count = if target_channels < channels.len() as u8 {
        channels = downmix_channels(&channels, target_channels);
        target_channels
    } else {
        channels.len() as u8
    };

    let bytes = f32_channels_to_bytes(&channels, target_bits_per_sample, output_format)
        .map_err(|e| DecodeError::DecodingFailed(format!("Output conversion failed: {e}")))?;

    Ok(vec![AudioData::new(
        target_bits_per_sample,
        output_channel_count,
        target_sample_rate,
        bytes,
        output_format,
        Endianness::LittleEndian,
    )])
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
            let channels_i32: Vec<Vec<i32>> = channels.iter().map(|c| vec_f32_to_s24(c)).collect();
            Ok(interleave_vecs_s24(channels_i32.as_slice()))
        }
        32 => {
            let channels_i32: Vec<Vec<i32>> = channels.iter().map(|c| vec_f32_to_i32(c)).collect();
            Ok(interleave_vecs_i32(channels_i32.as_slice()))
        }
        bits => Err(format!("Unsupported output bits per sample: {}", bits)),
    }
}

fn vec_f32_to_i32(input: &[f32]) -> Vec<i32> {
    let mut output = Vec::with_capacity(input.len());
    for &value in input {
        let clamped = value.clamp(-1.0, 1.0);
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
        let clamped = value.clamp(-1.0, 1.0);
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
    use std::f32::consts::PI;
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

    fn recv_until_done(pipeline: &mut DecodePipelineHandle) -> Vec<AudioData> {
        let mut frames = Vec::new();
        for _ in 0..100 {
            match pipeline.recv() {
                Some(Ok(audio_data)) => frames.push(audio_data),
                Some(Err(error)) => panic!("Decode error: {:?}", error),
                None => break,
            }
        }
        frames
    }

    #[test]
    fn test_decode_explicit_raw_pcm_stream() {
        let format = RawPcmFormat::linear16(8_000, 1).unwrap();
        let mut pipeline = DecodePipeline::spawn_raw_pcm(format);

        pipeline.send(Bytes::from_static(&[0x34])).unwrap();
        pipeline
            .send(Bytes::from_static(&[0x12, 0x78, 0x56]))
            .unwrap();
        pipeline.send(Bytes::new()).unwrap();

        let frames = recv_until_done(&mut pipeline);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].bits_per_sample(), 16);
        assert_eq!(frames[0].channel_count(), 1);
        assert_eq!(frames[0].sampling_rate(), 8_000);
        assert_eq!(frames[0].audio_format(), EncodingFlag::PCMSigned);
        assert_eq!(frames[0].endianness(), Endianness::LittleEndian);
        assert_eq!(frames[0].data(), &vec![0x34, 0x12, 0x78, 0x56]);
    }

    #[test]
    fn test_decode_explicit_g711_mulaw_stream() {
        let samples = [-12000i16, -1024, 0, 1024, 12000];
        let mut encoded = vec![0u8; samples.len()];
        soundkit_g711::encode_i16(G711Law::MuLaw, &samples, &mut encoded).unwrap();

        let mut expected = vec![0i16; samples.len()];
        soundkit_g711::decode_i16(G711Law::MuLaw, &encoded, &mut expected).unwrap();

        let mut pipeline = DecodePipeline::spawn_g711(G711Law::MuLaw, 8_000, 1).unwrap();
        pipeline
            .send(Bytes::copy_from_slice(&encoded[..2]))
            .unwrap();
        pipeline
            .send(Bytes::copy_from_slice(&encoded[2..]))
            .unwrap();
        pipeline.send(Bytes::new()).unwrap();

        let frames = recv_until_done(&mut pipeline);
        assert_eq!(frames.len(), 2);
        assert!(frames.iter().all(|frame| frame.bits_per_sample() == 16));
        assert!(frames.iter().all(|frame| frame.channel_count() == 1));
        assert!(frames.iter().all(|frame| frame.sampling_rate() == 8_000));

        let decoded: Vec<i16> = frames
            .iter()
            .flat_map(|frame| {
                frame
                    .data()
                    .chunks_exact(2)
                    .map(|bytes| i16::from_le_bytes([bytes[0], bytes[1]]))
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(decoded, expected);
    }

    #[test]
    fn test_decode_explicit_g711_rejects_invalid_metadata() {
        assert!(DecodePipeline::spawn_g711(G711Law::MuLaw, 0, 1).is_err());
        assert!(DecodePipeline::spawn_g711(G711Law::MuLaw, 8_000, 0).is_err());
    }

    #[test]
    fn test_decode_explicit_g722_stream() {
        let samples: Vec<i16> = (0..160)
            .map(|index| {
                let phase = index as f32 / 160.0 * PI * 6.0;
                (phase.sin() * 10_000.0) as i16
            })
            .collect();

        let mut encoder = soundkit_g722::G722Encoder::new_64k();
        let mut encoded = Vec::new();
        encoder.encode_to_vec(&samples, &mut encoded).unwrap();

        let mut direct_decoder = soundkit_g722::G722Decoder::new_64k();
        let mut expected = vec![0i16; encoded.len() * 2];
        let expected_len = direct_decoder
            .decode_i16(&encoded, &mut expected, false)
            .unwrap();
        expected.truncate(expected_len);

        let mut pipeline = DecodePipeline::spawn_g722();
        for chunk in encoded.chunks(7) {
            pipeline.send(Bytes::copy_from_slice(chunk)).unwrap();
        }
        pipeline.send(Bytes::new()).unwrap();

        let frames = recv_until_done(&mut pipeline);
        assert!(!frames.is_empty());
        assert!(frames.iter().all(|frame| frame.bits_per_sample() == 16));
        assert!(frames.iter().all(|frame| frame.channel_count() == 1));
        assert!(frames.iter().all(|frame| frame.sampling_rate() == 16_000));

        let decoded: Vec<i16> = frames
            .iter()
            .flat_map(|frame| {
                frame
                    .data()
                    .chunks_exact(2)
                    .map(|bytes| i16::from_le_bytes([bytes[0], bytes[1]]))
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(decoded, expected);
    }

    #[test]
    fn test_decode_explicit_g726_stream() {
        let samples: Vec<i16> = (0..400)
            .map(|index| {
                let phase = index as f32 / 80.0 * PI * 2.0;
                (phase.sin() * 8_000.0) as i16
            })
            .collect();

        for rate in [
            G726Rate::Rate16000,
            G726Rate::Rate24000,
            G726Rate::Rate32000,
            G726Rate::Rate40000,
        ] {
            let mut encoder = soundkit_g726::G726Encoder::try_new(rate, G726Packing::Left).unwrap();
            let mut encoded = Vec::new();
            encoder.encode_to_vec(&samples, &mut encoded).unwrap();
            encoder.flush_to_vec(&mut encoded).unwrap();

            let mut direct_decoder =
                soundkit_g726::G726Decoder::try_new(rate, G726Packing::Left).unwrap();
            let expected_samples = (encoded.len() * 8) / rate.bits_per_sample();
            let mut expected = vec![0i16; expected_samples];
            let expected_len = direct_decoder
                .decode_i16(&encoded, &mut expected, false)
                .unwrap();
            expected.truncate(expected_len);

            let mut pipeline =
                DecodePipeline::spawn_g726_with_rate(rate, G726Packing::Left).unwrap();
            for chunk in encoded.chunks(7) {
                pipeline.send(Bytes::copy_from_slice(chunk)).unwrap();
            }
            pipeline.send(Bytes::new()).unwrap();

            let frames = recv_until_done(&mut pipeline);
            assert!(!frames.is_empty(), "rate {rate:?}");
            assert!(
                frames.iter().all(|frame| frame.bits_per_sample() == 16),
                "rate {rate:?}"
            );
            assert!(
                frames.iter().all(|frame| frame.channel_count() == 1),
                "rate {rate:?}"
            );
            assert!(
                frames.iter().all(|frame| frame.sampling_rate() == 8_000),
                "rate {rate:?}"
            );

            let decoded: Vec<i16> = frames
                .iter()
                .flat_map(|frame| {
                    frame
                        .data()
                        .chunks_exact(2)
                        .map(|bytes| i16::from_le_bytes([bytes[0], bytes[1]]))
                        .collect::<Vec<_>>()
                })
                .collect();
            assert_eq!(decoded, expected, "rate {rate:?}");
        }
    }

    #[test]
    fn test_decode_explicit_g729_stream() {
        let samples: Vec<i16> = (0..240)
            .map(|index| {
                let phase = index as f32 / 80.0 * PI * 2.0;
                (phase.sin() * 8_000.0) as i16
            })
            .collect();

        let mut encoder = soundkit_g729::G729Encoder::new_voice();
        let mut encoded = Vec::new();
        encoder.encode_to_vec(&samples, &mut encoded).unwrap();

        let mut direct_decoder = soundkit_g729::G729Decoder::new_voice();
        let mut expected = vec![0i16; encoded.len() / 10 * 80];
        let expected_len = direct_decoder
            .decode_i16(&encoded, &mut expected, false)
            .unwrap();
        expected.truncate(expected_len);

        let mut pipeline = DecodePipeline::spawn_g729().unwrap();
        for chunk in encoded.chunks(7) {
            pipeline.send(Bytes::copy_from_slice(chunk)).unwrap();
        }
        pipeline.send(Bytes::new()).unwrap();

        let frames = recv_until_done(&mut pipeline);
        assert!(!frames.is_empty());
        assert!(frames.iter().all(|frame| frame.bits_per_sample() == 16));
        assert!(frames.iter().all(|frame| frame.channel_count() == 1));
        assert!(frames.iter().all(|frame| frame.sampling_rate() == 8_000));

        let decoded: Vec<i16> = frames
            .iter()
            .flat_map(|frame| {
                frame
                    .data()
                    .chunks_exact(2)
                    .map(|bytes| i16::from_le_bytes([bytes[0], bytes[1]]))
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(decoded, expected);
    }

    #[test]
    fn test_decode_explicit_amr_nb_stream() {
        let data = Bytes::from(
            fs::read(
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("..")
                    .join("testdata")
                    .join("amr_nb/A_Tusk_is_used_to_make_costly_gifts.amr"),
            )
            .unwrap(),
        );

        let mut pipeline = DecodePipeline::spawn_amr_nb().unwrap();
        for start in (0..data.len()).step_by(997) {
            let end = (start + 997).min(data.len());
            pipeline.send(data.slice(start..end)).unwrap();
        }
        pipeline.send(Bytes::new()).unwrap();

        let frames = recv_until_done(&mut pipeline);
        assert!(!frames.is_empty(), "No AMR-NB frames decoded");
        assert!(frames.iter().all(|frame| frame.bits_per_sample() == 16));
        assert!(frames.iter().all(|frame| frame.channel_count() == 1));
        assert!(frames.iter().all(|frame| frame.sampling_rate() == 8_000));
        assert!(frames
            .iter()
            .flat_map(|frame| frame.data().chunks_exact(2))
            .map(|bytes| i16::from_le_bytes([bytes[0], bytes[1]]))
            .any(|sample| sample != 0));
    }

    #[test]
    fn test_decode_explicit_gsm_stream() {
        let samples: Vec<i16> = (0..480)
            .map(|index| {
                let phase = index as f32 / 80.0 * PI * 2.0;
                (phase.sin() * 8_000.0) as i16
            })
            .collect();

        let mut encoder = soundkit_gsm::GsmEncoder::new_standard();
        let mut encoded = Vec::new();
        encoder.encode_to_vec(&samples, &mut encoded).unwrap();

        let mut direct_decoder = soundkit_gsm::GsmDecoder::new_standard();
        let mut expected = vec![0i16; encoded.len() / 33 * 160];
        let expected_len = direct_decoder
            .decode_i16(&encoded, &mut expected, false)
            .unwrap();
        expected.truncate(expected_len);

        let mut pipeline = DecodePipeline::spawn_gsm(GsmVariant::Standard).unwrap();
        for chunk in encoded.chunks(19) {
            pipeline.send(Bytes::copy_from_slice(chunk)).unwrap();
        }
        pipeline.send(Bytes::new()).unwrap();

        let frames = recv_until_done(&mut pipeline);
        assert!(!frames.is_empty());
        assert!(frames.iter().all(|frame| frame.bits_per_sample() == 16));
        assert!(frames.iter().all(|frame| frame.channel_count() == 1));
        assert!(frames.iter().all(|frame| frame.sampling_rate() == 8_000));

        let decoded: Vec<i16> = frames
            .iter()
            .flat_map(|frame| {
                frame
                    .data()
                    .chunks_exact(2)
                    .map(|bytes| i16::from_le_bytes([bytes[0], bytes[1]]))
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(decoded, expected);
    }

    #[test]
    fn test_decode_explicit_speex_stream() {
        let data = Bytes::from(
            fs::read(
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("..")
                    .join("testdata")
                    .join("speex/A_Tusk_is_used_to_make_costly_gifts.spx"),
            )
            .unwrap(),
        );

        let mut pipeline = DecodePipeline::spawn_speex();
        for start in (0..data.len()).step_by(997) {
            let end = (start + 997).min(data.len());
            pipeline.send(data.slice(start..end)).unwrap();
        }
        pipeline.send(Bytes::new()).unwrap();

        let frames = recv_until_done(&mut pipeline);
        assert!(!frames.is_empty(), "No Speex frames decoded");
        assert!(frames.iter().all(|frame| frame.bits_per_sample() == 16));
        assert!(frames.iter().all(|frame| frame.channel_count() == 1));
        assert!(frames.iter().all(|frame| frame.sampling_rate() == 8_000));
        assert!(frames
            .iter()
            .flat_map(|frame| frame.data().chunks_exact(2))
            .map(|bytes| i16::from_le_bytes([bytes[0], bytes[1]]))
            .any(|sample| sample != 0));
    }

    #[test]
    fn test_decode_explicit_vorbis_stream() {
        let data = Bytes::from(
            fs::read(
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("..")
                    .join("testdata")
                    .join("vorbis/A_Tusk_is_used_to_make_costly_gifts.ogg"),
            )
            .unwrap(),
        );

        let mut pipeline = DecodePipeline::spawn_vorbis();
        for start in (0..data.len()).step_by(997) {
            let end = (start + 997).min(data.len());
            pipeline.send(data.slice(start..end)).unwrap();
        }
        pipeline.send(Bytes::new()).unwrap();

        let frames = recv_until_done(&mut pipeline);
        assert!(!frames.is_empty(), "No Vorbis frames decoded");
        assert!(frames.iter().all(|frame| frame.bits_per_sample() == 16));
        assert!(frames.iter().all(|frame| frame.channel_count() == 1));
        assert!(frames.iter().all(|frame| frame.sampling_rate() == 8_000));
        assert!(frames
            .iter()
            .flat_map(|frame| frame.data().chunks_exact(2))
            .map(|bytes| i16::from_le_bytes([bytes[0], bytes[1]]))
            .any(|sample| sample != 0));
    }

    #[test]
    fn test_decode_ogg_vorbis_autodetect() {
        let data = Bytes::from(
            fs::read(
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("..")
                    .join("testdata")
                    .join("vorbis/A_Tusk_is_used_to_make_costly_gifts.ogg"),
            )
            .unwrap(),
        );

        let mut pipeline = DecodePipeline::spawn();
        pipeline.send(data).unwrap();
        pipeline.send(Bytes::new()).unwrap();

        let frames = recv_until_done(&mut pipeline);
        assert!(!frames.is_empty(), "No autodetected Vorbis frames decoded");
        assert!(frames.iter().all(|frame| frame.channel_count() == 1));
        assert!(frames.iter().all(|frame| frame.sampling_rate() == 8_000));
    }

    #[test]
    fn test_decode_explicit_alac_stream() {
        let data = Bytes::from(
            fs::read(
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("..")
                    .join("testdata")
                    .join("alac/A_Tusk_is_used_to_make_costly_gifts.m4a"),
            )
            .unwrap(),
        );

        let mut pipeline = DecodePipeline::spawn_alac();
        for start in (0..data.len()).step_by(997) {
            let end = (start + 997).min(data.len());
            pipeline.send(data.slice(start..end)).unwrap();
        }
        pipeline.send(Bytes::new()).unwrap();

        let frames = recv_until_done(&mut pipeline);
        assert_eq!(frames.len(), 1, "ALAC should decode once EOF is signalled");
        assert_eq!(frames[0].bits_per_sample(), 16);
        assert_eq!(frames[0].channel_count(), 1);
        assert_eq!(frames[0].sampling_rate(), 8_000);
        assert!(frames[0]
            .data()
            .chunks_exact(2)
            .map(|bytes| i16::from_le_bytes([bytes[0], bytes[1]]))
            .any(|sample| sample != 0));
    }

    #[test]
    fn test_decode_alac_autodetect() {
        let data = Bytes::from(
            fs::read(
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("..")
                    .join("testdata")
                    .join("alac/A_Tusk_is_used_to_make_costly_gifts.m4a"),
            )
            .unwrap(),
        );

        let mut pipeline = DecodePipeline::spawn();
        pipeline.send(data).unwrap();
        pipeline.send(Bytes::new()).unwrap();

        let frames = recv_until_done(&mut pipeline);
        assert_eq!(frames.len(), 1, "No autodetected ALAC frame decoded");
        assert_eq!(frames[0].channel_count(), 1);
        assert_eq!(frames[0].sampling_rate(), 8_000);
    }

    #[test]
    fn test_decode_mp4_he_aac_itag_139_autodetect() {
        // This exercises the default soundkit-decoder AAC-in-MP4 route:
        // pure-Rust container detection/demuxing plus FDK-AAC C bindings.
        let data = Bytes::from(
            fs::read(
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("..")
                    .join("testdata")
                    .join("itag139/yt_itag_139_he_aac.mp4"),
            )
            .unwrap(),
        );

        let mut pipeline = DecodePipeline::spawn();
        pipeline.send(data).unwrap();
        pipeline.send(Bytes::new()).unwrap();

        let frames = recv_until_done(&mut pipeline);
        assert!(
            !frames.is_empty(),
            "No autodetected itag 139 HE-AAC frames decoded"
        );
        assert!(frames.iter().all(|frame| frame.bits_per_sample() == 16));
        assert!(frames.iter().all(|frame| frame.channel_count() == 2));
        assert!(frames.iter().all(|frame| frame.sampling_rate() == 22_050));
        assert!(frames
            .iter()
            .flat_map(|frame| frame.data().chunks_exact(2))
            .map(|bytes| i16::from_le_bytes([bytes[0], bytes[1]]))
            .any(|sample| sample != 0));
    }

    #[test]
    fn test_decode_webm_vorbis_itag_171_autodetect() {
        // This covers legacy YouTube WebM Vorbis audio itags 171/172.
        let data = Bytes::from(
            fs::read(
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("..")
                    .join("testdata")
                    .join("itag171/yt_itag_171_vorbis.webm"),
            )
            .unwrap(),
        );

        let mut pipeline = DecodePipeline::spawn();
        for start in (0..data.len()).step_by(997) {
            let end = (start + 997).min(data.len());
            pipeline.send(data.slice(start..end)).unwrap();
        }
        pipeline.send(Bytes::new()).unwrap();

        let frames = recv_until_done(&mut pipeline);
        assert!(
            !frames.is_empty(),
            "No autodetected itag 171 WebM Vorbis frames decoded"
        );
        assert!(frames.iter().all(|frame| frame.bits_per_sample() == 16));
        assert!(frames.iter().all(|frame| frame.channel_count() == 2));
        assert!(frames.iter().all(|frame| frame.sampling_rate() == 44_100));
        assert!(frames
            .iter()
            .flat_map(|frame| frame.data().chunks_exact(2))
            .map(|bytes| i16::from_le_bytes([bytes[0], bytes[1]]))
            .any(|sample| sample != 0));
    }

    #[test]
    fn test_decode_explicit_aiff_streams() {
        for fixture_name in [
            "aiff/A_Tusk_is_used_to_make_costly_gifts.aiff",
            "aifc/A_Tusk_is_used_to_make_costly_gifts.aifc",
        ] {
            let data = Bytes::from(
                fs::read(
                    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                        .join("..")
                        .join("testdata")
                        .join(fixture_name),
                )
                .unwrap(),
            );

            let mut pipeline = DecodePipeline::spawn_aiff();
            for start in (0..data.len()).step_by(997) {
                let end = (start + 997).min(data.len());
                pipeline.send(data.slice(start..end)).unwrap();
            }
            pipeline.send(Bytes::new()).unwrap();

            let frames = recv_until_done(&mut pipeline);
            assert_eq!(frames.len(), 1, "{fixture_name} should decode at EOF");
            assert_eq!(frames[0].bits_per_sample(), 16);
            assert_eq!(frames[0].channel_count(), 1);
            assert_eq!(frames[0].sampling_rate(), 8_000);
            assert!(frames[0]
                .data()
                .chunks_exact(2)
                .map(|bytes| i16::from_le_bytes([bytes[0], bytes[1]]))
                .any(|sample| sample != 0));
        }
    }

    #[test]
    fn test_decode_aiff_autodetect() {
        for fixture_name in [
            "aiff/A_Tusk_is_used_to_make_costly_gifts.aiff",
            "aifc/A_Tusk_is_used_to_make_costly_gifts.aifc",
        ] {
            let data = Bytes::from(
                fs::read(
                    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                        .join("..")
                        .join("testdata")
                        .join(fixture_name),
                )
                .unwrap(),
            );

            let mut pipeline = DecodePipeline::spawn();
            pipeline.send(data).unwrap();
            pipeline.send(Bytes::new()).unwrap();

            let frames = recv_until_done(&mut pipeline);
            assert_eq!(frames.len(), 1, "No autodetected AIFF frame decoded");
            assert_eq!(frames[0].channel_count(), 1);
            assert_eq!(frames[0].sampling_rate(), 8_000);
        }
    }

    #[test]
    fn test_decode_explicit_ac3_stream() {
        let data = Bytes::from(
            fs::read(
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("..")
                    .join("testdata")
                    .join("ac3/A_Tusk_is_used_to_make_costly_gifts.ac3"),
            )
            .unwrap(),
        );

        let mut pipeline = DecodePipeline::spawn_ac3().unwrap();
        for start in (0..data.len()).step_by(997) {
            let end = (start + 997).min(data.len());
            pipeline.send(data.slice(start..end)).unwrap();
        }
        pipeline.send(Bytes::new()).unwrap();

        let frames = recv_until_done(&mut pipeline);
        assert!(!frames.is_empty(), "No AC-3 frames decoded");
        assert!(frames.iter().all(|frame| frame.bits_per_sample() == 16));
        assert!(frames.iter().all(|frame| frame.channel_count() == 1));
        assert!(frames.iter().all(|frame| frame.sampling_rate() == 48_000));
        assert!(frames
            .iter()
            .flat_map(|frame| frame.data().chunks_exact(2))
            .map(|bytes| i16::from_le_bytes([bytes[0], bytes[1]]))
            .any(|sample| sample != 0));
    }

    #[test]
    fn test_decode_ac3_autodetect() {
        let data = Bytes::from(
            fs::read(
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("..")
                    .join("testdata")
                    .join("ac3/A_Tusk_is_used_to_make_costly_gifts.ac3"),
            )
            .unwrap(),
        );

        let mut pipeline = DecodePipeline::spawn();
        pipeline.send(data).unwrap();
        pipeline.send(Bytes::new()).unwrap();

        let frames = recv_until_done(&mut pipeline);
        assert!(!frames.is_empty(), "No autodetected AC-3 frames decoded");
        assert!(frames.iter().all(|frame| frame.channel_count() == 1));
        assert!(frames.iter().all(|frame| frame.sampling_rate() == 48_000));
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
            fs::read(testdata_path(
                "flac/A_Tusk_is_used_to_make_costly_gifts.flac",
            ))
            .unwrap(),
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
            fs::read(testdata_path(
                "opus/A_Tusk_is_used_to_make_costly_gifts.opus",
            ))
            .unwrap(),
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
            fs::read(testdata_path(
                "ogg_opus/A_Tusk_is_used_to_make_costly_gifts.ogg",
            ))
            .unwrap(),
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
            fs::read(testdata_path(
                "webm/A_Tusk_is_used_to_make_costly_gifts.webm",
            ))
            .unwrap(),
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
        let max_peak = display_peaks
            .iter()
            .fold(0.0f32, |a, &b| a.max(b))
            .max(0.001);

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
                        let level = ((normalized - threshold)
                            * half_height as f32
                            * (chars.len() - 1) as f32)
                            as usize;
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
                        let level = ((normalized - threshold)
                            * half_height as f32
                            * (chars.len() - 1) as f32)
                            as usize;
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

        let bin_size = samples.len().div_ceil(num_bins);

        samples
            .chunks(bin_size)
            .map(|chunk| {
                let max_abs = chunk
                    .iter()
                    .map(|&s| (s as f32).abs())
                    .fold(0.0f32, f32::max);
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
            format_name,
            successful_files,
            elapsed.as_secs_f64(),
            files_per_sec,
            mb_per_sec
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
        println!(
            "Native:    {} Hz, {} ch, {} frames",
            native_sr, native_ch, native_frames
        );

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
        println!(
            "Resampled: {} Hz, {} ch, {} frames",
            resample_sr, resample_ch, resample_frames
        );

        assert!(
            native_sr > 0,
            "Native sample rate should be detected, got {}",
            native_sr
        );
        assert_eq!(resample_sr, 16_000, "Resampled should be 16kHz");
        assert_eq!(resample_ch, 1, "Resampled should be mono");
    }

    fn create_f32_audio(sample_rate: u32, channels: &[Vec<f32>]) -> AudioData {
        AudioData::new(
            32,
            channels.len() as u8,
            sample_rate,
            interleave_vecs_f32(channels),
            EncodingFlag::PCMFloat,
            Endianness::LittleEndian,
        )
    }

    #[test]
    fn streaming_resampler_matches_single_pass_length() {
        let input_rate = 44_100u32;
        let target_rate = 16_000u32;
        let total_input_samples = input_rate as usize * 3 + 137;
        let samples: Vec<f32> = (0..total_input_samples)
            .map(|index| {
                let phase = 2.0 * PI * 440.0 * index as f32 / input_rate as f32;
                0.5 * phase.sin()
            })
            .collect();

        let reference_audio = create_f32_audio(input_rate, &[samples.clone()]);
        let reference =
            soundkit::audio_pipeline::downsample_audio(&reference_audio, target_rate as usize)
                .expect("single-pass downsample should succeed");
        let reference_len = reference.first().map(|channel| channel.len()).unwrap_or(0);

        let options = DecodeOptions {
            output_bits_per_sample: Some(16),
            output_sample_rate: Some(target_rate),
            output_channels: Some(1),
        };
        let mut resampler = None;
        let mut streaming_len = 0usize;

        for chunk in samples.chunks(997) {
            let audio = create_f32_audio(input_rate, &[chunk.to_vec()]);
            let frames = apply_output_options(audio, &options, &mut resampler)
                .expect("streaming resample step should succeed");
            for frame in frames {
                let channels =
                    audio_data_to_f32_channels(&frame).expect("streaming frame should decode");
                streaming_len += channels.first().map(|channel| channel.len()).unwrap_or(0);
            }
        }

        if let Some(resampler) = resampler.take() {
            let frames =
                flush_resampler_frames(resampler).expect("streaming resample flush should succeed");
            for frame in frames {
                let channels =
                    audio_data_to_f32_channels(&frame).expect("flushed frame should decode");
                streaming_len += channels.first().map(|channel| channel.len()).unwrap_or(0);
            }
        }

        assert_eq!(
            streaming_len, reference_len,
            "persistent streaming resampler should preserve full output length"
        );
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

    /// Helper to fully decode MP3 data through the pipeline and return all decoded bytes
    fn decode_mp3_fully(data: Bytes, chunk_size: Option<usize>) -> Vec<u8> {
        let mut pipeline = DecodePipeline::spawn_with_buffers(1024, 4096);

        let mut chunks_sent = 0usize;
        let mut bytes_sent = 0usize;

        if let Some(cs) = chunk_size {
            // Send in chunks
            for start in (0..data.len()).step_by(cs) {
                let end = (start + cs).min(data.len());
                let chunk = data.slice(start..end);
                bytes_sent += chunk.len();
                chunks_sent += 1;
                while pipeline.send(chunk.clone()).is_err() {
                    std::thread::sleep(std::time::Duration::from_micros(100));
                }
                // Brief pause between chunks to simulate network
                std::thread::sleep(std::time::Duration::from_micros(50));
            }
        } else {
            // Send all at once
            bytes_sent = data.len();
            chunks_sent = 1;
            pipeline.send(data).unwrap();
        }

        // Wait for all chunks to be processed before EOF
        std::thread::sleep(std::time::Duration::from_millis(500));

        // Send EOF
        while pipeline.send(Bytes::new()).is_err() {
            std::thread::sleep(std::time::Duration::from_micros(100));
        }

        println!("Sent {} bytes in {} chunks", bytes_sent, chunks_sent);

        // Collect all output
        let mut output = Vec::new();
        let mut idle_iters = 0u32;
        let max_idle = 500u32;
        let mut frame_count = 0usize;
        let mut frame_sizes = Vec::new();

        loop {
            match pipeline.try_recv() {
                Some(Ok(audio_data)) => {
                    frame_count += 1;
                    frame_sizes.push(audio_data.data().len());
                    output.extend_from_slice(audio_data.data());
                    idle_iters = 0;
                }
                Some(Err(e)) => panic!("Decode error: {:?}", e),
                None => {
                    idle_iters += 1;
                    if idle_iters >= max_idle {
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
            }
        }

        println!(
            "Received {} frames, {} bytes total",
            frame_count,
            output.len()
        );
        if frame_count > 0 && frame_count <= 20 {
            println!("Frame sizes: {:?}", frame_sizes);
        }
        output
    }

    /// Test that the pipeline produces identical output regardless of input chunk size
    /// This is critical for HTTP/3 (small chunks) vs HTTP/2 (large chunks) compatibility
    #[test]
    fn test_mp3_pipeline_chunk_invariance() {
        let data = Bytes::from(
            fs::read(testdata_path("mp3/A_Tusk_is_used_to_make_costly_gifts.mp3")).unwrap(),
        );

        // Decode with all data at once (like HTTP/2 with large buffers)
        let large_output = decode_mp3_fully(data.clone(), None);

        // Decode with small chunks (like HTTP/3 with QUIC)
        let small_output = decode_mp3_fully(data.clone(), Some(1200));

        // Decode with very small chunks (extreme case)
        let tiny_output = decode_mp3_fully(data, Some(256));

        println!("All-at-once output: {} bytes", large_output.len());
        println!("1200-byte chunks output: {} bytes", small_output.len());
        println!("256-byte chunks output: {} bytes", tiny_output.len());

        assert_eq!(
            large_output.len(),
            small_output.len(),
            "Pipeline should produce identical output regardless of chunk size! \
             All-at-once: {} bytes, 1200-byte chunks: {} bytes, \
             Difference: {} bytes",
            large_output.len(),
            small_output.len(),
            (large_output.len() as i64 - small_output.len() as i64).abs()
        );

        assert_eq!(
            large_output.len(),
            tiny_output.len(),
            "Pipeline should produce identical output regardless of chunk size! \
             All-at-once: {} bytes, 256-byte chunks: {} bytes, \
             Difference: {} bytes",
            large_output.len(),
            tiny_output.len(),
            (large_output.len() as i64 - tiny_output.len() as i64).abs()
        );
    }
}
