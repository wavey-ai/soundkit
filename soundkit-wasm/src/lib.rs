use frame_header::{EncodingFlag, Endianness};
#[cfg(feature = "aac-lc")]
use js_sys::Float32Array;
use js_sys::{Array, Object, Reflect, Uint8Array};
use sha2::{Digest, Sha256};
use soundkit::audio_content_crypto::{AudioContentCipher, AudioGroupMetadata};
#[cfg(any(
    feature = "aac",
    feature = "m4a",
    feature = "mp3",
    feature = "flac",
    feature = "opus"
))]
use soundkit::audio_packet::Decoder;
#[cfg(any(feature = "flac", feature = "opus"))]
use soundkit::audio_packet::Encoder;
use soundkit::audio_pipeline::{
    audio_to_f32_channels, Stereo48kBlock, StreamingStereo48kNormalizer,
};
use soundkit::audio_types::AudioData;
use soundkit::crypto::ChaCha20Poly1305PacketCipher;
use soundkit::frame_stream::{SoundKitFrame, SoundKitFrameStream, SoundKitFrameStreamOptions};
use soundkit::raw_pcm::{RawPcmFormat, RawPcmStreamProcessor};
use soundkit::wav::{WavSampleFormat, WavStreamEncoder, WavStreamProcessor};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

#[cfg(feature = "detect")]
use access_unit::{detect_audio, AudioType};
#[cfg(feature = "aac")]
use soundkit_aac::AacDecoder;
#[cfg(feature = "m4a")]
use soundkit_aac::AacDecoderMp4;
#[cfg(feature = "aac-debox")]
use soundkit_aac::{AacMp4DemuxEvent, AacMp4Demuxer};
#[cfg(feature = "aac-lc")]
use soundkit_aac_lc::AacLcDecoder;
#[cfg(feature = "ac3")]
use soundkit_ac3::Ac3Decoder;
#[cfg(feature = "aiff")]
use soundkit_aiff::AiffDecoder;
#[cfg(feature = "alac")]
use soundkit_alac::{
    inspect_caf_chunk, validate_caf_file_header, AlacPacketDecoder, CafAlacPacketIndex,
    SEEKABLE_ALAC_REQUIRED,
};
#[cfg(feature = "audio-demux")]
use soundkit_audio_demux::{
    inspect_mp4_top_level_box, AudioCodec, AudioDemuxEvent, AudioTrackConfig, AudioTrackDemuxer,
    CafAudioIndex, MediaSampleIndex, MediaTrackConfig, MediaTrackKind, MediaTrackPacket,
    Mp4MediaDemuxEvent, Mp4MediaDemuxer, Mp4MediaIndex, MxfMediaDemuxEvent, MxfMediaDemuxer,
    PcmEndianness,
};
#[cfg(feature = "flac")]
use soundkit_flac::{FlacDecoderClaxon, FlacEncoder};
#[cfg(feature = "mp3")]
use soundkit_mp3::Mp3Decoder;
#[cfg(feature = "ogg-opus")]
use soundkit_ogg_opus::OggOpusDecoder;
#[cfg(feature = "opus-debox")]
use soundkit_ogg_opus::{OggOpusDemuxEvent, OggOpusDemuxer};
#[cfg(feature = "opus")]
use soundkit_opus::{OpusDecoder, OpusEncoder, OpusStreamDecoder};
#[cfg(feature = "video")]
use soundkit_video::{VideoDecoder, VideoFrame};
#[cfg(feature = "vorbis")]
use soundkit_vorbis::VorbisDecoder;
#[cfg(feature = "webm")]
use soundkit_webm::{WebmDecoder, WebmMediaDemuxEvent, WebmMediaDemuxer, WebmMediaTrackConfig};
#[cfg(feature = "opus-debox")]
use soundkit_webm::{WebmOpusDemuxEvent, WebmOpusDemuxer};

const MIN_DETECTION_BYTES: usize = 8192;
const MAX_DETECTION_BYTES: usize = 65_536;
const MAX_STREAM_INPUT_CHUNK_BYTES: usize = 4 * 1024 * 1024;
const CANONICAL_PCM_BLOCK_FRAMES: usize = 96_000;
#[cfg(feature = "opus")]
const MAX_OPUS_PACKET_FRAMES: usize = 5_760;
#[cfg(any(feature = "aac", feature = "m4a", feature = "mp3", feature = "flac"))]
const DEFAULT_SCRATCH_SAMPLES: usize = 262_144;

/// Return the current WebAssembly linear-memory size in bytes.
#[wasm_bindgen(js_name = wasmMemoryBytes)]
pub fn wasm_memory_bytes() -> usize {
    let memory = wasm_bindgen::memory().unchecked_into::<js_sys::WebAssembly::Memory>();
    Uint8Array::new(&memory.buffer()).byte_length() as usize
}

fn validate_stream_input_chunk(bytes: &[u8]) -> Result<(), String> {
    if bytes.len() > MAX_STREAM_INPUT_CHUNK_BYTES {
        return Err(format!(
            "input chunk exceeds the {MAX_STREAM_INPUT_CHUNK_BYTES} byte streaming budget"
        ));
    }
    Ok(())
}

#[wasm_bindgen]
pub struct WasmAudioContentCipher {
    cipher: AudioContentCipher,
}

#[wasm_bindgen]
impl WasmAudioContentCipher {
    #[wasm_bindgen(constructor)]
    pub fn new(key: &[u8]) -> Result<WasmAudioContentCipher, JsValue> {
        let cipher = AudioContentCipher::new(key).map_err(|error| js_error(error.to_string()))?;
        Ok(Self { cipher })
    }

    pub fn seal(
        &self,
        key_epoch: u32,
        nonce: &[u8],
        plaintext: &[u8],
        authenticated_data: &[u8],
    ) -> Result<Uint8Array, JsValue> {
        let envelope = self
            .cipher
            .seal(key_epoch, nonce, plaintext, authenticated_data)
            .map_err(|error| js_error(error.to_string()))?;
        Ok(Uint8Array::from(envelope.as_slice()))
    }

    pub fn open(
        &self,
        expected_key_epoch: u32,
        envelope: &[u8],
        authenticated_data: &[u8],
    ) -> Result<Uint8Array, JsValue> {
        let plaintext = self
            .cipher
            .open(expected_key_epoch, envelope, authenticated_data)
            .map_err(|error| js_error(error.to_string()))?;
        Ok(Uint8Array::from(plaintext.as_slice()))
    }
}

/// Opens the endpoint-specific envelope that transports an audio content key.
///
/// The wrapping key comes from P-256 ECDH and HKDF-SHA256. The caller supplies
/// the canonical key-exchange context as additional authenticated data.
#[wasm_bindgen]
pub struct WasmAudioContentKeyUnwrapper {
    cipher: ChaCha20Poly1305PacketCipher,
}

#[wasm_bindgen]
impl WasmAudioContentKeyUnwrapper {
    #[wasm_bindgen(constructor)]
    pub fn new(key: &[u8]) -> Result<WasmAudioContentKeyUnwrapper, JsValue> {
        if key.len() != 32 || key.iter().all(|byte| *byte == 0) {
            return Err(js_error("invalid audio content wrapping key".to_owned()));
        }
        let cipher =
            ChaCha20Poly1305PacketCipher::new(key).map_err(|error| js_error(error.to_string()))?;
        Ok(Self { cipher })
    }

    pub fn open(
        &self,
        nonce: &[u8],
        ciphertext: &[u8],
        authenticated_data: &[u8],
    ) -> Result<Uint8Array, JsValue> {
        if nonce.len() != 12 || ciphertext.len() != 48 {
            return Err(js_error("invalid audio content key envelope".to_owned()));
        }
        let mut packet = Vec::with_capacity(nonce.len() + ciphertext.len());
        packet.extend_from_slice(nonce);
        packet.extend_from_slice(ciphertext);
        let plaintext = self
            .cipher
            .decrypt_nonce_prefixed(&packet, authenticated_data)
            .map_err(|error| js_error(error.to_string()))?;
        packet.fill(0);
        if plaintext.len() != 32 || plaintext.iter().all(|byte| *byte == 0) {
            return Err(js_error("invalid audio content key".to_owned()));
        }
        Ok(Uint8Array::from(plaintext.as_slice()))
    }

    pub fn seal(
        &self,
        nonce: &[u8],
        plaintext: &[u8],
        authenticated_data: &[u8],
    ) -> Result<Uint8Array, JsValue> {
        if nonce.len() != 12 || plaintext.len() != 32 || plaintext.iter().all(|byte| *byte == 0) {
            return Err(js_error("invalid audio content key".to_owned()));
        }
        let mut packet = self
            .cipher
            .encrypt_nonce_prefixed(nonce, plaintext, authenticated_data)
            .map_err(|error| js_error(error.to_string()))?;
        let ciphertext = Uint8Array::from(&packet[12..]);
        packet.fill(0);
        Ok(ciphertext)
    }
}

#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = buildAudioGroupAssociatedData)]
pub fn build_audio_group_associated_data(
    session_context: &str,
    transport_session_id: &str,
    config_generation: u32,
    epoch_id: &str,
    pts_samples: &str,
    sample_rate: u32,
    frame_count: u32,
    group_count: u16,
    group_id: u16,
    group_index: u16,
    channel_start: u16,
    channel_count: u16,
    payload_kind: u8,
    sample_format: u8,
    flags: u8,
) -> Result<Uint8Array, JsValue> {
    let transport_session_id = transport_session_id
        .parse::<u64>()
        .map_err(|_| js_error("invalid transport session id".to_owned()))?;
    let epoch_id = epoch_id
        .parse::<u64>()
        .map_err(|_| js_error("invalid audio epoch id".to_owned()))?;
    let pts_samples = pts_samples
        .parse::<u64>()
        .map_err(|_| js_error("invalid audio presentation timestamp".to_owned()))?;
    let aad = AudioGroupMetadata {
        session_context: session_context.as_bytes(),
        transport_session_id,
        config_generation,
        epoch_id,
        pts_samples,
        sample_rate,
        frame_count,
        group_count,
        group_id,
        group_index,
        channel_start,
        channel_count,
        payload_kind,
        sample_format,
        flags,
    }
    .associated_data()
    .map_err(|error| js_error(error.to_string()))?;
    Ok(Uint8Array::from(aad.as_slice()))
}

#[wasm_bindgen]
pub struct WasmMusicDecoder {
    state: DecoderState,
    scratch: DecoderScratch,
}

/// A bounded canonical PCM block for browser and native stream adapters.
#[derive(Debug)]
pub struct CanonicalPcmBlock {
    pub start_frame: u64,
    pub frame_count: u32,
    /// Planar stereo signed 16-bit PCM: all left samples, then all right samples.
    pub pcm_s16_planar: Vec<u8>,
}

/// One bounded result from the canonical 48 kHz stereo decoder.
#[derive(Debug)]
pub struct CanonicalDecodeBatch {
    pub blocks: Vec<CanonicalPcmBlock>,
    pub done: bool,
    pub frame_count: u64,
    pub source_sample_rate: u32,
    pub source_channels: u8,
    pub source_frame_count: u64,
    pub source_identity: Option<String>,
}

/// Format-detecting decode, normalization, and hashing in one bounded session.
#[wasm_bindgen]
pub struct WasmCanonicalPcmDecoder {
    decoder: WasmMusicDecoder,
    normalizer: StreamingStereo48kNormalizer,
    source_digest: Option<Sha256>,
    pending_left: Vec<i16>,
    pending_right: Vec<i16>,
    pending_start: usize,
    emitted_frames: u64,
    finished: bool,
}

/// Incremental RIFF/RF64 PCM writer. The final frame count makes the first
/// emitted header exact, so browser streams never need a complete WAV buffer.
#[wasm_bindgen]
pub struct WasmWavEncoder {
    encoder: WavStreamEncoder,
}

#[cfg(feature = "opus-debox")]
#[wasm_bindgen]
pub struct WasmOpusDeboxer {
    state: OpusDeboxState,
}

#[cfg(feature = "aac-debox")]
#[wasm_bindgen]
pub struct WasmAacDeboxer {
    state: AacDeboxState,
}

#[cfg(feature = "aac-lc")]
#[wasm_bindgen]
pub struct WasmAacLcDecoder {
    decoder: AacLcDecoder,
    interleaved: Vec<f32>,
}

#[cfg(feature = "audio-demux")]
#[wasm_bindgen]
pub struct WasmAudioTrackDemuxer {
    demuxer: AudioTrackDemuxer,
}

/// Bounded ALAC access-unit decoder for seekable MP4 and CAF adapters.
#[cfg(feature = "alac")]
#[wasm_bindgen]
pub struct WasmAlacPacketDecoder {
    decoder: AlacPacketDecoder,
}

/// Seekable, Rust-validated CAF ALAC packet index.
#[cfg(feature = "alac")]
#[wasm_bindgen]
pub struct WasmCafAlacIndex {
    index: CafAlacPacketIndex,
}

/// Seekable, Rust-validated CAF audio sample index.
#[cfg(feature = "audio-demux")]
#[wasm_bindgen]
pub struct WasmCafAudioIndex {
    index: CafAudioIndex,
}

/// Seekable, Rust-validated MOV/MP4 audio-and-video sample index.
#[cfg(feature = "audio-demux")]
#[wasm_bindgen]
pub struct WasmMp4MediaIndex {
    index: Mp4MediaIndex,
}

/// Streaming Rust fragmented-MP4/CMAF audio-and-video demuxer.
#[cfg(feature = "audio-demux")]
#[wasm_bindgen]
pub struct WasmMp4MediaDemuxer {
    demuxer: Mp4MediaDemuxer,
}

/// Streaming Rust MXF KLV demuxer that emits both picture and sound essence.
#[cfg(feature = "audio-demux")]
#[wasm_bindgen]
pub struct WasmMxfMediaDemuxer {
    demuxer: MxfMediaDemuxer,
}

/// Streaming Rust WebM demuxer that emits both video and audio tracks.
#[cfg(feature = "webm")]
#[wasm_bindgen]
pub struct WasmWebmMediaDemuxer {
    demuxer: WebmMediaDemuxer,
}

/// Pure-Rust video access-unit decoder shared by browser and native imports.
#[cfg(feature = "video")]
#[wasm_bindgen]
pub struct WasmVideoDecoder {
    decoder: VideoDecoder,
}

#[cfg(feature = "video")]
#[wasm_bindgen]
impl WasmVideoDecoder {
    #[wasm_bindgen(constructor)]
    pub fn new(codec: &str) -> Result<WasmVideoDecoder, JsValue> {
        let codec = soundkit_video::VideoCodec::parse(codec)
            .ok_or_else(|| js_error(format!("unsupported video codec: {codec}")))?;
        let decoder = VideoDecoder::new(codec).map_err(js_error)?;
        Ok(Self { decoder })
    }

    /// Decode one complete codec access unit. Non-finite timestamps mean
    /// unknown and avoid JavaScript BigInt conversion at this boundary.
    pub fn decode(
        &mut self,
        access_unit: &[u8],
        pts: f64,
        duration: f64,
    ) -> Result<Array, JsValue> {
        let pts = finite_i64(pts);
        let duration = finite_u64(duration);
        export_video_frames(
            self.decoder
                .decode(access_unit, pts, duration)
                .map_err(js_error)?,
        )
    }

    /// Decode a complete Annex-B elementary stream. This is intended for
    /// import validation; normal playback should use access-unit decoding.
    #[wasm_bindgen(js_name = decodeStream)]
    pub fn decode_stream(&mut self, stream: &[u8]) -> Result<Array, JsValue> {
        export_video_frames(self.decoder.decode_stream(stream).map_err(js_error)?)
    }

    pub fn flush(&mut self) -> Result<Array, JsValue> {
        export_video_frames(self.decoder.flush().map_err(js_error)?)
    }
}

#[wasm_bindgen]
pub struct WasmSoundKitFrameDecoder {
    stream: SoundKitFrameStream,
}

/// Bounded incremental SHA-256 for browser streams that are not otherwise
/// passing through a SoundKit import encoder.
#[wasm_bindgen]
pub struct WasmSha256 {
    digest: Option<Sha256>,
}

#[cfg(feature = "flac")]
#[wasm_bindgen]
pub struct WasmFlacEncoder {
    encoder: FlacEncoder,
    channels: u8,
    bits_per_sample: u32,
    frame_size: u32,
}

// Opus encoder backed by soundkit-opus -> libopus-rs (Rust), so both the player
// and the press /cut editor encode Opus through soundkit rather than a separate
// libopus wasm bundle or any C dependency.
#[cfg(feature = "opus")]
#[wasm_bindgen]
pub struct WasmOpusEncoder {
    encoder: OpusEncoder,
    frame_size: u32,
    channels: u8,
    output: Vec<u8>,
}

/// One-pass encoder for the library import fast path.
///
/// A 48 kHz stereo PCM16 WAV is already in the geometry used by the library's
/// Opus cache. Keeping the WAV parser and both encoders together means each
/// bounded input chunk is parsed once and immediately fans out to Opus and,
/// for lossless imports, FLAC. No decoded PCM crosses into JavaScript and no
/// seekable Float32 working copy has to be completed before encoding starts.
#[cfg(all(feature = "wav", feature = "opus", feature = "flac"))]
#[wasm_bindgen]
pub struct WasmPcm16WaveLibraryEncoder {
    decoder: WavStreamProcessor,
    preserve_lossless: bool,
    geometry_checked: bool,
    finished: bool,
    source_digest: Sha256,
    total_frames: u64,
    source_frames: u64,
    opus_frames: u64,
    opus_stream_bytes: u64,
    opus_digest: Sha256,
    opus_index_entries: Vec<(u64, u64)>,
    opus_encoder: Option<OpusEncoder>,
    opus_frame: Vec<i16>,
    flac_encoder: Option<FlacEncoder>,
    flac_frame_size: usize,
    flac_frame: Vec<i32>,
    flac_frames: u64,
    flac_stream_bytes: u64,
    flac_digest: Sha256,
    flac_index_entries: Vec<(u64, u64)>,
    pending_flac_descriptor: Option<(u64, u32)>,
}

/// Bounded, format-detecting library import pipeline.
///
/// Encoded source bytes enter Rust once. SoundKit decodes them incrementally,
/// normalizes each PCM block to the library's 48 kHz stereo geometry, and
/// immediately emits indexed SoundKit-v2 Opus and optional FLAC packets. PCM
/// never crosses the WASM boundary and no complete decoded source is retained.
#[cfg(all(
    feature = "detect",
    feature = "audio-demux",
    feature = "aac-lc",
    feature = "alac",
    feature = "wav",
    feature = "opus",
    feature = "flac"
))]
#[wasm_bindgen]
pub struct WasmStreamingLibraryEncoder {
    decoder: LibrarySourceDecoder,
    alac_decoder: Option<AlacPacketDecoder>,
    aac_lc_decoder: Option<AacLcDecoder>,
    normalizer: StreamingStereo48kNormalizer,
    preserve_lossless: bool,
    finished: bool,
    source_digest: Sha256,
    opus_frames: u64,
    opus_stream_bytes: u64,
    opus_digest: Sha256,
    opus_index_entries: Vec<(u64, u64)>,
    opus_encoder: OpusEncoder,
    opus_frame: Vec<i16>,
    flac_encoder: Option<FlacEncoder>,
    flac_frame_size: usize,
    flac_frame: Vec<i32>,
    flac_sample_rate: u32,
    flac_channels: u8,
    flac_frames: u64,
    flac_stream_bytes: u64,
    flac_digest: Sha256,
    flac_index_entries: Vec<(u64, u64)>,
}

#[cfg(all(feature = "wav", feature = "opus", feature = "flac"))]
pub struct LibraryPacket {
    pub bytes: Vec<u8>,
    pub start_frame: u64,
    pub frame_count: u32,
}

#[cfg(all(feature = "wav", feature = "opus", feature = "flac"))]
type WaveLibraryPacket = LibraryPacket;

/// Rust-native result from the same bounded Library encoder used by WASM.
///
/// Keeping this representation free of `JsValue` lets Apple and browser
/// adapters share byte detection, decode, normalization, framing, hashing,
/// and indexing without moving PCM across either platform boundary.
#[cfg(all(feature = "wav", feature = "opus", feature = "flac"))]
pub struct LibraryEncodeBatch {
    pub opus_packets: Vec<LibraryPacket>,
    pub flac_packets: Vec<LibraryPacket>,
    pub done: bool,
    pub completed_frames: u64,
    pub frame_count: u64,
    pub sample_rate: u32,
    pub channels: u8,
    pub opus_index: Option<Vec<u8>>,
    pub flac_index: Option<Vec<u8>>,
    pub source_identity: Option<String>,
    pub opus_identity: Option<String>,
    pub flac_identity: Option<String>,
}

#[cfg(feature = "opus")]
#[wasm_bindgen]
pub struct WasmOpusDecoder {
    decoder: OpusDecoder,
    output: Vec<i16>,
    decoded_size: usize,
}

#[cfg(feature = "opus")]
#[wasm_bindgen]
pub struct WasmOpusDecodeResult {
    output: Vec<i16>,
    decoded_size: usize,
}

enum DecoderState {
    Detecting { buffer: Vec<u8> },
    Decoding { decoder: FormatDecoder },
    Finished,
}

#[cfg(all(feature = "audio-demux", feature = "aac-lc"))]
enum LibrarySourceDecoder {
    Detecting { buffer: Vec<u8> },
    Music(WasmMusicDecoder),
    MpegTs(ContainerAudioDecoder),
    Mxf(MxfAudioDecoder),
    Finished,
}

#[cfg(all(feature = "audio-demux", feature = "aac-lc"))]
struct ContainerAudioDecoder {
    demuxer: AudioTrackDemuxer,
    decoder: Option<FormatDecoder>,
    config: Option<AudioTrackConfig>,
    scratch: DecoderScratch,
}

#[cfg(all(feature = "audio-demux", feature = "aac-lc"))]
struct MxfAudioDecoder {
    demuxer: MxfMediaDemuxer,
    config: Option<MediaTrackConfig>,
}

enum FormatDecoder {
    #[cfg(feature = "aac")]
    Aac(Box<AacDecoder>),
    #[cfg(feature = "m4a")]
    M4a(Box<AacDecoderMp4>),
    #[cfg(feature = "aiff")]
    Aiff(Box<AiffDecoder>),
    #[cfg(feature = "ac3")]
    Ac3(Box<Ac3Decoder>),
    #[cfg(all(feature = "aac-lc", not(feature = "aac")))]
    AacLcAdts(Box<AacLcAdtsDecoder>),
    #[cfg(all(feature = "aac-lc", feature = "aac-debox", not(feature = "m4a")))]
    AacLcMp4(Box<AacLcMp4Decoder>),
    #[cfg(feature = "flac")]
    Flac(Box<FlacDecoderClaxon>),
    #[cfg(feature = "mp3")]
    Mp3(Box<Mp3Decoder>),
    #[cfg(feature = "ogg-opus")]
    OggOpus(Box<OggOpusDecoder>),
    #[cfg(feature = "opus")]
    Opus(Box<OpusStreamDecoder>),
    RawPcm(Box<RawPcmStreamProcessor>),
    #[cfg(feature = "vorbis")]
    Vorbis(Box<VorbisDecoder>),
    #[cfg(feature = "webm")]
    WebM(Box<WebmDecoder>),
    Wav(Box<WavStreamProcessor>),
}

#[derive(Default)]
struct DecoderScratch {
    i16_samples: Vec<i16>,
    i32_samples: Vec<i32>,
}

impl DecoderScratch {
    #[cfg(any(feature = "aac", feature = "m4a", feature = "mp3"))]
    fn i16_samples(&mut self) -> &mut [i16] {
        if self.i16_samples.len() < DEFAULT_SCRATCH_SAMPLES {
            self.i16_samples.resize(DEFAULT_SCRATCH_SAMPLES, 0);
        }
        &mut self.i16_samples
    }

    #[cfg(feature = "flac")]
    fn i32_samples(&mut self) -> &mut [i32] {
        if self.i32_samples.len() < DEFAULT_SCRATCH_SAMPLES {
            self.i32_samples.resize(DEFAULT_SCRATCH_SAMPLES, 0);
        }
        &mut self.i32_samples
    }
}

#[cfg(all(feature = "aac-lc", not(feature = "aac")))]
struct AacLcAdtsDecoder {
    buffer: Vec<u8>,
    buffer_start: usize,
    decoder: Option<AacLcDecoder>,
    audio_specific_config: Option<[u8; 2]>,
}

#[cfg(all(feature = "aac-lc", not(feature = "aac")))]
impl AacLcAdtsDecoder {
    fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(16 * 1024),
            buffer_start: 0,
            decoder: None,
            audio_specific_config: None,
        }
    }

    fn process(&mut self, bytes: &[u8], finalizing: bool) -> Result<Vec<AudioData>, String> {
        self.compact_buffer(false);
        let buffered_bytes = self.buffer.len().saturating_sub(self.buffer_start);
        if buffered_bytes.saturating_add(bytes.len()) > MAX_STREAM_INPUT_CHUNK_BYTES {
            return Err("AAC-LC ADTS buffer exceeded the streaming budget".to_owned());
        }
        self.buffer.extend_from_slice(bytes);
        let mut frames = Vec::new();
        loop {
            let Some(sync) = self
                .buffer
                .get(self.buffer_start..)
                .unwrap_or_default()
                .windows(2)
                .position(|bytes| bytes[0] == 0xff && (bytes[1] & 0xf6) == 0xf0)
            else {
                let keep = self.buffer.len().saturating_sub(self.buffer_start).min(1);
                self.buffer_start = self.buffer.len().saturating_sub(keep);
                break;
            };
            if sync > 0 {
                self.buffer_start += sync;
            }
            if self.buffer.len().saturating_sub(self.buffer_start) < 7 {
                break;
            }
            let base = self.buffer_start;
            let protection_absent = self.buffer[base + 1] & 1 != 0;
            let header_len = if protection_absent { 7 } else { 9 };
            let frame_len = (((self.buffer[base + 3] & 3) as usize) << 11)
                | ((self.buffer[base + 4] as usize) << 3)
                | ((self.buffer[base + 5] as usize) >> 5);
            if frame_len <= header_len || frame_len > 8191 {
                return Err("AAC-LC ADTS frame has an invalid length".to_owned());
            }
            if self.buffer.len().saturating_sub(base) < frame_len {
                break;
            }
            let object_type = ((self.buffer[base + 2] & 0xc0) >> 6) + 1;
            let sample_rate_index = (self.buffer[base + 2] & 0x3c) >> 2;
            let channels =
                ((self.buffer[base + 2] & 1) << 2) | ((self.buffer[base + 3] & 0xc0) >> 6);
            let config = [
                (object_type << 3) | (sample_rate_index >> 1),
                ((sample_rate_index & 1) << 7) | (channels << 3),
            ];
            if self.audio_specific_config != Some(config) {
                if self.audio_specific_config.is_some() {
                    return Err("AAC-LC format changed during the stream".to_owned());
                }
                self.decoder = Some(
                    AacLcDecoder::from_audio_specific_config(&config)
                        .map_err(|error| error.to_string())?,
                );
                self.audio_specific_config = Some(config);
            }
            let decoder = self
                .decoder
                .as_mut()
                .ok_or_else(|| "AAC-LC decoder was not initialized".to_owned())?;
            let info = decoder.frame_info();
            let mut pcm = Vec::with_capacity(info.frames * info.channels * 2);
            {
                let decoded = decoder
                    .decode_access_unit(&self.buffer[base + header_len..base + frame_len])
                    .map_err(|error| error.to_string())?;
                for frame in 0..decoded.frames() {
                    for channel in decoded.channels() {
                        pcm.extend_from_slice(&library_float_to_i16(channel[frame]).to_le_bytes());
                    }
                }
            }
            frames.push(AudioData::new(
                16,
                u8::try_from(info.channels)
                    .map_err(|_| "AAC-LC channel count exceeds SoundKit".to_owned())?,
                info.sample_rate,
                pcm,
                frame_header::EncodingFlag::PCMSigned,
                frame_header::Endianness::LittleEndian,
            ));
            self.buffer_start += frame_len;
        }
        self.compact_buffer(finalizing);
        if finalizing && !self.buffer.is_empty() {
            return Err("AAC-LC ADTS stream ends with a truncated frame".to_owned());
        }
        Ok(frames)
    }

    fn compact_buffer(&mut self, force: bool) {
        if self.buffer_start == 0 {
            return;
        }
        if self.buffer_start == self.buffer.len() {
            self.buffer.clear();
            self.buffer_start = 0;
            return;
        }
        if force
            || (self.buffer_start >= 16 * 1024
                && self.buffer_start.saturating_mul(2) >= self.buffer.len())
        {
            self.buffer.drain(..self.buffer_start);
            self.buffer_start = 0;
        }
    }
}

#[cfg(all(feature = "aac-lc", feature = "aac-debox", not(feature = "m4a")))]
struct AacLcMp4Decoder {
    demuxer: AacMp4Demuxer,
    decoder: AacLcAdtsDecoder,
}

#[cfg(all(feature = "aac-lc", feature = "aac-debox", not(feature = "m4a")))]
impl AacLcMp4Decoder {
    fn new() -> Result<Self, String> {
        let mut demuxer = AacMp4Demuxer::new();
        demuxer.init()?;
        Ok(Self {
            demuxer,
            decoder: AacLcAdtsDecoder::new(),
        })
    }

    fn process(&mut self, bytes: &[u8], finalizing: bool) -> Result<Vec<AudioData>, String> {
        let events = if finalizing {
            let mut events = self.demuxer.add(bytes)?;
            events.extend(self.demuxer.finish()?);
            events
        } else {
            self.demuxer.add(bytes)?
        };
        let mut frames = Vec::new();
        for event in events {
            if let AacMp4DemuxEvent::Frame(frame) = event {
                frames.extend(self.decoder.process(&frame.adts, false)?);
            }
        }
        if finalizing {
            frames.extend(self.decoder.process(&[], true)?);
        }
        Ok(frames)
    }
}

#[cfg(feature = "opus-debox")]
enum OpusDeboxState {
    Detecting { buffer: Vec<u8> },
    Ogg(OggOpusDemuxer),
    Raw(RawOpusDeboxer),
    WebM(WebmOpusDemuxer),
    Finished,
}

#[cfg(feature = "aac-debox")]
enum AacDeboxState {
    Detecting { buffer: Vec<u8> },
    Mp4(AacMp4Demuxer),
    Finished,
}

#[wasm_bindgen]
impl WasmSha256 {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            digest: Some(Sha256::new()),
        }
    }

    pub fn update(&mut self, bytes: &[u8]) -> Result<(), JsValue> {
        validate_stream_input_chunk(bytes).map_err(js_error)?;
        self.digest
            .as_mut()
            .ok_or_else(|| js_error("SHA-256 digest is already finished".to_owned()))?
            .update(bytes);
        Ok(())
    }

    pub fn finish(&mut self) -> Result<String, JsValue> {
        let digest = self
            .digest
            .take()
            .ok_or_else(|| js_error("SHA-256 digest is already finished".to_owned()))?;
        Ok(format!("sha256:{:x}", digest.finalize()))
    }
}

impl WasmCanonicalPcmDecoder {
    fn from_music_decoder(decoder: WasmMusicDecoder) -> Self {
        Self {
            decoder,
            normalizer: StreamingStereo48kNormalizer::new(),
            source_digest: Some(Sha256::new()),
            pending_left: Vec::with_capacity(CANONICAL_PCM_BLOCK_FRAMES),
            pending_right: Vec::with_capacity(CANONICAL_PCM_BLOCK_FRAMES),
            pending_start: 0,
            emitted_frames: 0,
            finished: false,
        }
    }

    pub fn push_rust(&mut self, bytes: &[u8]) -> Result<CanonicalDecodeBatch, String> {
        if self.finished {
            return Err("canonical PCM decoder is already finished".to_owned());
        }
        validate_stream_input_chunk(bytes)?;
        self.source_digest
            .as_mut()
            .ok_or_else(|| "canonical source identity is already finished".to_owned())?
            .update(bytes);
        let frames = match self.decoder.push_frames(bytes) {
            Ok(frames) => frames,
            Err(error) => {
                self.finished = true;
                return Err(error);
            }
        };
        let blocks = self.normalize_frames(frames)?;
        Ok(self.batch(blocks, false, None))
    }

    pub fn finish_rust(&mut self) -> Result<CanonicalDecodeBatch, String> {
        if self.finished {
            return Err("canonical PCM decoder is already finished".to_owned());
        }
        self.finished = true;
        let frames = self.decoder.flush_frames()?;
        let mut blocks = self.normalize_frames(frames)?;
        if let Some(tail) = self.normalizer.finish()? {
            self.append_normalized_block(tail, &mut blocks)?;
        }
        self.emit_pending_blocks(&mut blocks, true)?;
        let identity = self
            .source_digest
            .take()
            .ok_or_else(|| "canonical source identity is already finished".to_owned())?;
        Ok(self.batch(
            blocks,
            true,
            Some(format!("sha256:{:x}", identity.finalize())),
        ))
    }

    fn normalize_frames(
        &mut self,
        frames: Vec<AudioData>,
    ) -> Result<Vec<CanonicalPcmBlock>, String> {
        let mut blocks = Vec::new();
        for frame in frames {
            if let Some(block) = self.normalizer.push(&frame)? {
                self.append_normalized_block(block, &mut blocks)?;
            }
        }
        Ok(blocks)
    }

    fn append_normalized_block(
        &mut self,
        block: Stereo48kBlock,
        blocks: &mut Vec<CanonicalPcmBlock>,
    ) -> Result<(), String> {
        if block.left.len() != block.right.len() {
            return Err("canonical PCM normalizer returned unequal channels".to_owned());
        }
        self.pending_left
            .extend(block.left.into_iter().map(canonical_float_to_i16));
        self.pending_right
            .extend(block.right.into_iter().map(canonical_float_to_i16));
        self.emit_pending_blocks(blocks, false)
    }

    fn emit_pending_blocks(
        &mut self,
        blocks: &mut Vec<CanonicalPcmBlock>,
        finalizing: bool,
    ) -> Result<(), String> {
        loop {
            let available = self.pending_left.len().saturating_sub(self.pending_start);
            if available == 0 || (!finalizing && available < CANONICAL_PCM_BLOCK_FRAMES) {
                break;
            }
            let frame_count = available.min(CANONICAL_PCM_BLOCK_FRAMES);
            let end = self.pending_start + frame_count;
            let mut pcm = Vec::with_capacity(frame_count * 2 * std::mem::size_of::<i16>());
            for sample in &self.pending_left[self.pending_start..end] {
                pcm.extend_from_slice(&sample.to_le_bytes());
            }
            for sample in &self.pending_right[self.pending_start..end] {
                pcm.extend_from_slice(&sample.to_le_bytes());
            }
            let start_frame = self.emitted_frames;
            self.emitted_frames = self
                .emitted_frames
                .checked_add(frame_count as u64)
                .ok_or_else(|| "canonical PCM frame count overflowed".to_owned())?;
            blocks.push(CanonicalPcmBlock {
                start_frame,
                frame_count: frame_count
                    .try_into()
                    .map_err(|_| "canonical PCM block exceeds u32".to_owned())?,
                pcm_s16_planar: pcm,
            });
            self.pending_start = end;
        }
        if self.pending_start == self.pending_left.len() {
            self.pending_left.clear();
            self.pending_right.clear();
            self.pending_start = 0;
        } else if self.pending_start >= CANONICAL_PCM_BLOCK_FRAMES * 4
            && self.pending_start.saturating_mul(2) >= self.pending_left.len()
        {
            self.pending_left.drain(..self.pending_start);
            self.pending_right.drain(..self.pending_start);
            self.pending_start = 0;
        }
        Ok(())
    }

    fn batch(
        &self,
        blocks: Vec<CanonicalPcmBlock>,
        done: bool,
        source_identity: Option<String>,
    ) -> CanonicalDecodeBatch {
        CanonicalDecodeBatch {
            blocks,
            done,
            frame_count: self.emitted_frames,
            source_sample_rate: self.normalizer.source_sample_rate(),
            source_channels: self.normalizer.source_channels(),
            source_frame_count: self.normalizer.source_frames(),
            source_identity,
        }
    }
}

fn canonical_float_to_i16(sample: f32) -> i16 {
    let sample = if sample.is_finite() {
        sample.clamp(-1.0, 1.0)
    } else {
        0.0
    };
    let scale = if sample < 0.0 { 32_768.0 } else { 32_767.0 };
    ((f64::from(sample) * scale).round() as i32).clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

#[wasm_bindgen]
impl WasmMusicDecoder {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::new_auto()
    }

    #[wasm_bindgen(js_name = newAuto)]
    pub fn new_auto() -> Self {
        Self {
            state: DecoderState::Detecting { buffer: Vec::new() },
            scratch: DecoderScratch::default(),
        }
    }

    #[wasm_bindgen(js_name = newWithFormat)]
    pub fn new_with_format(format: &str) -> Result<WasmMusicDecoder, JsValue> {
        let decoder = decoder_for_format(format).map_err(js_error)?;
        Ok(Self {
            state: DecoderState::Decoding { decoder },
            scratch: DecoderScratch::default(),
        })
    }

    #[wasm_bindgen(js_name = newRawLinear16)]
    pub fn new_raw_linear16(sample_rate: u32, channels: u8) -> Result<WasmMusicDecoder, JsValue> {
        let format = RawPcmFormat::linear16(sample_rate, channels).map_err(js_error)?;
        Ok(Self {
            state: DecoderState::Decoding {
                decoder: FormatDecoder::RawPcm(Box::new(RawPcmStreamProcessor::new(format))),
            },
            scratch: DecoderScratch::default(),
        })
    }

    #[wasm_bindgen(js_name = newRawLinear32)]
    pub fn new_raw_linear32(sample_rate: u32, channels: u8) -> Result<WasmMusicDecoder, JsValue> {
        let format = RawPcmFormat::linear32(sample_rate, channels).map_err(js_error)?;
        Ok(Self {
            state: DecoderState::Decoding {
                decoder: FormatDecoder::RawPcm(Box::new(RawPcmStreamProcessor::new(format))),
            },
            scratch: DecoderScratch::default(),
        })
    }

    /// Push arbitrary encoded bytes and receive all PCM frames currently available.
    ///
    /// This method drains decoder output after each push. Use `flush()` once at EOF
    /// to force final container/codec drain.
    pub fn push(&mut self, bytes: &[u8]) -> Result<Array, JsValue> {
        let frames = self.push_frames(bytes).map_err(js_error)?;
        audio_frames_to_js(frames)
    }

    /// Final EOF/drain call. The decoder should not be reused after this.
    pub fn flush(&mut self) -> Result<Array, JsValue> {
        let frames = self.flush_frames().map_err(js_error)?;
        audio_frames_to_js(frames)
    }
}

#[wasm_bindgen]
impl WasmCanonicalPcmDecoder {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::new_auto()
    }

    #[wasm_bindgen(js_name = newAuto)]
    pub fn new_auto() -> Self {
        Self::from_music_decoder(WasmMusicDecoder::new_auto())
    }

    #[wasm_bindgen(js_name = newWithFormat)]
    pub fn new_with_format(format: &str) -> Result<WasmCanonicalPcmDecoder, JsValue> {
        Ok(Self::from_music_decoder(WasmMusicDecoder::new_with_format(
            format,
        )?))
    }

    #[wasm_bindgen(js_name = newRawLinear16)]
    pub fn new_raw_linear16(
        sample_rate: u32,
        channels: u8,
    ) -> Result<WasmCanonicalPcmDecoder, JsValue> {
        Ok(Self::from_music_decoder(
            WasmMusicDecoder::new_raw_linear16(sample_rate, channels)?,
        ))
    }

    /// Decode one bounded source byte range.
    pub fn push(&mut self, bytes: &[u8]) -> Result<JsValue, JsValue> {
        canonical_decode_batch_to_js(self.push_rust(bytes).map_err(js_error)?)
    }

    /// Drain decoder and normalizer tails and finalize the source identity.
    pub fn finish(&mut self) -> Result<JsValue, JsValue> {
        canonical_decode_batch_to_js(self.finish_rust().map_err(js_error)?)
    }
}

#[cfg(all(feature = "audio-demux", feature = "aac-lc"))]
impl LibrarySourceDecoder {
    fn new() -> Self {
        Self::Detecting { buffer: Vec::new() }
    }

    fn push(&mut self, bytes: &[u8]) -> Result<Vec<AudioData>, String> {
        validate_stream_input_chunk(bytes)?;
        let state = std::mem::replace(self, Self::Finished);
        match state {
            Self::Detecting { mut buffer } => {
                let probe_bytes = (MAX_DETECTION_BYTES - buffer.len()).min(bytes.len());
                buffer.extend_from_slice(&bytes[..probe_bytes]);
                if buffer.len() < MIN_DETECTION_BYTES {
                    *self = Self::Detecting { buffer };
                    return Ok(Vec::new());
                }
                let mut decoder = if looks_like_mxf_source(&buffer) {
                    Self::Mxf(MxfAudioDecoder::new())
                } else if looks_like_mpeg_ts_source(&buffer) {
                    Self::MpegTs(ContainerAudioDecoder::new("mpeg-ts")?)
                } else {
                    Self::Music(WasmMusicDecoder::new_auto())
                };
                let mut frames = decoder.push_selected(&buffer)?;
                if probe_bytes < bytes.len() {
                    frames.extend(decoder.push_selected(&bytes[probe_bytes..])?);
                }
                *self = decoder;
                Ok(frames)
            }
            mut decoder => {
                let frames = decoder.push_selected(bytes)?;
                *self = decoder;
                Ok(frames)
            }
        }
    }

    fn push_selected(&mut self, bytes: &[u8]) -> Result<Vec<AudioData>, String> {
        match self {
            Self::Music(decoder) => decoder.push_frames(bytes),
            Self::MpegTs(decoder) => decoder.process(bytes, false),
            Self::Mxf(decoder) => decoder.process(bytes, false),
            Self::Detecting { .. } => Err("library source decoder is still detecting".to_owned()),
            Self::Finished => Err("library source decoder is finished".to_owned()),
        }
    }

    fn flush(&mut self) -> Result<Vec<AudioData>, String> {
        let state = std::mem::replace(self, Self::Finished);
        match state {
            Self::Detecting { buffer } => {
                let mut decoder = if looks_like_mxf_source(&buffer) {
                    Self::Mxf(MxfAudioDecoder::new())
                } else if looks_like_mpeg_ts_source(&buffer) {
                    Self::MpegTs(ContainerAudioDecoder::new("mpeg-ts")?)
                } else {
                    Self::Music(WasmMusicDecoder::new_auto())
                };
                let mut frames = decoder.push_selected(&buffer)?;
                frames.extend(decoder.flush_selected()?);
                Ok(frames)
            }
            mut decoder => decoder.flush_selected(),
        }
    }

    fn flush_selected(&mut self) -> Result<Vec<AudioData>, String> {
        match self {
            Self::Music(decoder) => decoder.flush_frames(),
            Self::MpegTs(decoder) => decoder.process(&[], true),
            Self::Mxf(decoder) => decoder.process(&[], true),
            Self::Detecting { .. } => Err("library source decoder is still detecting".to_owned()),
            Self::Finished => Ok(Vec::new()),
        }
    }
}

#[cfg(all(feature = "audio-demux", feature = "aac-lc"))]
impl ContainerAudioDecoder {
    fn new(format: &str) -> Result<Self, String> {
        Ok(Self {
            demuxer: AudioTrackDemuxer::new_with_format(format)?,
            decoder: None,
            config: None,
            scratch: DecoderScratch::default(),
        })
    }

    fn process(&mut self, bytes: &[u8], finalizing: bool) -> Result<Vec<AudioData>, String> {
        let events = if finalizing {
            self.demuxer.flush()?
        } else {
            self.demuxer.push(bytes)?
        };
        let mut frames = Vec::new();
        for event in events {
            match event {
                AudioDemuxEvent::Config(config) => self.install_config(config)?,
                AudioDemuxEvent::Packet(packet) => {
                    if let Some(config) = &self.config {
                        if config.track_id.is_some() && config.track_id != packet.track_id {
                            continue;
                        }
                        if config.codec == AudioCodec::Pcm {
                            frames.push(audio_data_from_container_pcm(config, packet.data)?);
                        } else {
                            frames.extend(
                                self.decoder
                                    .as_mut()
                                    .ok_or_else(|| {
                                        "container audio decoder has no codec".to_owned()
                                    })?
                                    .process(&packet.data, &mut self.scratch)?,
                            );
                        }
                    }
                }
            }
        }
        if finalizing {
            if let Some(decoder) = self.decoder.as_mut() {
                frames.extend(decoder.flush(&mut self.scratch)?);
            }
        }
        Ok(frames)
    }

    fn install_config(&mut self, config: AudioTrackConfig) -> Result<(), String> {
        if let Some(active) = &self.config {
            if active.track_id != config.track_id {
                return Ok(());
            }
        }
        self.decoder = match config.codec {
            AudioCodec::Aac => Some(FormatDecoder::AacLcAdts(Box::new(AacLcAdtsDecoder::new()))),
            #[cfg(feature = "ac3")]
            AudioCodec::Ac3 => Some(FormatDecoder::Ac3(Box::new(Ac3Decoder::try_new()?))),
            #[cfg(feature = "mp3")]
            AudioCodec::Mp3 => Some(FormatDecoder::Mp3(Box::new(Mp3Decoder::new()))),
            AudioCodec::Pcm => None,
            ref codec => {
                return Err(format!(
                    "unsupported container audio codec: {}",
                    codec.as_str()
                ))
            }
        };
        self.config = Some(config);
        Ok(())
    }
}

#[cfg(all(feature = "audio-demux", feature = "aac-lc"))]
impl MxfAudioDecoder {
    fn new() -> Self {
        Self {
            demuxer: MxfMediaDemuxer::new(),
            config: None,
        }
    }

    fn process(&mut self, bytes: &[u8], finalizing: bool) -> Result<Vec<AudioData>, String> {
        let events = if finalizing {
            self.demuxer.flush()?
        } else {
            self.demuxer.push(bytes)?
        };
        let mut frames = Vec::new();
        for event in events {
            match event {
                MxfMediaDemuxEvent::Config(config) if config.kind == MediaTrackKind::Audio => {
                    if config.codec != "pcm" {
                        return Err(format!("unsupported MXF audio codec: {}", config.codec));
                    }
                    self.config = Some(config);
                }
                MxfMediaDemuxEvent::Packet(packet) if packet.kind == MediaTrackKind::Audio => {
                    let config = self
                        .config
                        .as_ref()
                        .ok_or_else(|| "MXF audio packet arrived before its config".to_owned())?;
                    if packet.track_id == config.track_id {
                        frames.push(audio_data_from_media_pcm(config, packet.data)?);
                    }
                }
                _ => {}
            }
        }
        Ok(frames)
    }
}

#[cfg(all(feature = "audio-demux", feature = "aac-lc"))]
fn looks_like_mxf_source(bytes: &[u8]) -> bool {
    bytes
        .windows(4)
        .any(|window| window == [0x06, 0x0e, 0x2b, 0x34])
}

#[cfg(all(feature = "audio-demux", feature = "aac-lc"))]
fn looks_like_mpeg_ts_source(bytes: &[u8]) -> bool {
    [0usize, 4].into_iter().any(|offset| {
        [0usize, 188, 376]
            .into_iter()
            .all(|step| bytes.get(offset + step) == Some(&0x47))
    })
}

#[cfg(all(feature = "audio-demux", feature = "aac-lc"))]
fn audio_data_from_container_pcm(
    config: &AudioTrackConfig,
    data: Vec<u8>,
) -> Result<AudioData, String> {
    make_container_pcm_audio(
        config.sample_rate,
        config.channels,
        config.bits_per_sample,
        config.pcm_endianness,
        config.pcm_float,
        data,
    )
}

#[cfg(all(feature = "audio-demux", feature = "aac-lc"))]
fn audio_data_from_media_pcm(
    config: &MediaTrackConfig,
    data: Vec<u8>,
) -> Result<AudioData, String> {
    make_container_pcm_audio(
        config.sample_rate,
        config.channels,
        config.bits_per_sample,
        config.pcm_endianness,
        config.pcm_float,
        data,
    )
}

#[cfg(all(feature = "audio-demux", feature = "aac-lc"))]
fn make_container_pcm_audio(
    sample_rate: Option<u32>,
    channels: Option<u8>,
    bits_per_sample: Option<u8>,
    endianness: Option<PcmEndianness>,
    pcm_float: Option<bool>,
    data: Vec<u8>,
) -> Result<AudioData, String> {
    let sample_rate = sample_rate.ok_or_else(|| "container PCM has no sample rate".to_owned())?;
    let channels = channels.ok_or_else(|| "container PCM has no channel count".to_owned())?;
    let bits = bits_per_sample.ok_or_else(|| "container PCM has no sample depth".to_owned())?;
    if !matches!(bits, 16 | 24 | 32) {
        return Err(format!("unsupported container PCM sample depth: {bits}"));
    }
    Ok(AudioData::new(
        bits,
        channels,
        sample_rate,
        data,
        if pcm_float == Some(true) {
            EncodingFlag::PCMFloat
        } else {
            EncodingFlag::PCMSigned
        },
        if endianness == Some(PcmEndianness::Big) {
            Endianness::BigEndian
        } else {
            Endianness::LittleEndian
        },
    ))
}

#[cfg(all(feature = "audio-demux", feature = "alac"))]
fn decoded_audio_frame_count(audio: &AudioData) -> Result<u32, String> {
    let bytes_per_sample = usize::from(audio.bits_per_sample().div_ceil(8));
    let bytes_per_frame = bytes_per_sample
        .checked_mul(usize::from(audio.channel_count()))
        .ok_or_else(|| "decoded PCM frame size overflow".to_owned())?;
    if bytes_per_frame == 0 || audio.data().len() % bytes_per_frame != 0 {
        return Err("decoder returned misaligned PCM".to_owned());
    }
    u32::try_from(audio.data().len() / bytes_per_frame)
        .map_err(|_| "decoded PCM frame count exceeds u32".to_owned())
}

#[cfg(feature = "alac")]
fn trim_interleaved_audio(
    audio: AudioData,
    source_frame_start: u32,
    frame_count: u32,
) -> Result<Option<AudioData>, String> {
    if frame_count == 0 {
        return Ok(None);
    }
    let bytes_per_sample = usize::from(audio.bits_per_sample().div_ceil(8));
    let bytes_per_frame = bytes_per_sample
        .checked_mul(usize::from(audio.channel_count()))
        .ok_or_else(|| "ALAC PCM frame size overflow".to_owned())?;
    if bytes_per_frame == 0 || audio.data().len() % bytes_per_frame != 0 {
        return Err("ALAC decoder returned misaligned PCM".to_owned());
    }
    let decoded_frames = audio.data().len() / bytes_per_frame;
    let start = usize::try_from(source_frame_start)
        .map_err(|_| "ALAC trim start exceeds this platform".to_owned())?;
    let count = usize::try_from(frame_count)
        .map_err(|_| "ALAC trim length exceeds this platform".to_owned())?;
    let end = start
        .checked_add(count)
        .filter(|end| *end <= decoded_frames)
        .ok_or_else(|| "ALAC trim exceeds the decoded packet".to_owned())?;
    Ok(Some(AudioData::new(
        audio.bits_per_sample(),
        audio.channel_count(),
        audio.sampling_rate(),
        audio.data()[start * bytes_per_frame..end * bytes_per_frame].to_vec(),
        audio.audio_format(),
        audio.endianness(),
    )))
}

#[cfg(feature = "opus-debox")]
#[wasm_bindgen]
impl WasmOpusDeboxer {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::new_auto()
    }

    #[wasm_bindgen(js_name = newAuto)]
    pub fn new_auto() -> Self {
        Self {
            state: OpusDeboxState::Detecting { buffer: Vec::new() },
        }
    }

    #[wasm_bindgen(js_name = newWithFormat)]
    pub fn new_with_format(format: &str) -> Result<WasmOpusDeboxer, JsValue> {
        Ok(Self {
            state: opus_deboxer_for_format(format).map_err(js_error)?,
        })
    }

    /// Push arbitrary container bytes and receive Opus config/packet events.
    ///
    /// Packet events contain encoded Opus packet bytes suitable for a JS Opus
    /// decoder. Config events carry channel/sample-rate/pre-skip metadata.
    pub fn push(&mut self, bytes: &[u8]) -> Result<Array, JsValue> {
        let events = self.push_events(bytes).map_err(js_error)?;
        opus_debox_events_to_js(events)
    }

    /// Final drain call. The deboxer should not be reused after this.
    pub fn flush(&mut self) -> Result<Array, JsValue> {
        let events = self.flush_events().map_err(js_error)?;
        opus_debox_events_to_js(events)
    }
}

#[cfg(feature = "aac-debox")]
#[wasm_bindgen]
impl WasmAacDeboxer {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::new_auto()
    }

    #[wasm_bindgen(js_name = newAuto)]
    pub fn new_auto() -> Self {
        Self {
            state: AacDeboxState::Detecting { buffer: Vec::new() },
        }
    }

    #[wasm_bindgen(js_name = newWithFormat)]
    pub fn new_with_format(format: &str) -> Result<WasmAacDeboxer, JsValue> {
        Ok(Self {
            state: aac_deboxer_for_format(format).map_err(js_error)?,
        })
    }

    /// Push arbitrary MP4/M4A bytes and receive AAC config/packet events.
    ///
    /// Packet events contain ADTS AAC frames in `data` and the original MP4
    /// access unit in `rawData`.
    pub fn push(&mut self, bytes: &[u8]) -> Result<Array, JsValue> {
        let events = self.push_events(bytes).map_err(js_error)?;
        aac_debox_events_to_js(events)
    }

    /// Final drain call. The deboxer should not be reused after this.
    pub fn flush(&mut self) -> Result<Array, JsValue> {
        let events = self.flush_events().map_err(js_error)?;
        aac_debox_events_to_js(events)
    }
}

#[cfg(feature = "aac-lc")]
#[wasm_bindgen]
impl WasmAacLcDecoder {
    #[wasm_bindgen(constructor)]
    pub fn new(audio_specific_config: &[u8]) -> Result<WasmAacLcDecoder, JsValue> {
        Ok(Self {
            decoder: AacLcDecoder::from_audio_specific_config(audio_specific_config)
                .map_err(|error| js_error(error.to_string()))?,
            interleaved: Vec::new(),
        })
    }

    #[wasm_bindgen(getter, js_name = sampleRate)]
    pub fn sample_rate(&self) -> u32 {
        self.decoder.frame_info().sample_rate
    }

    #[wasm_bindgen(getter)]
    pub fn channels(&self) -> usize {
        self.decoder.frame_info().channels
    }

    #[wasm_bindgen(getter, js_name = framesPerAccessUnit)]
    pub fn frames_per_access_unit(&self) -> usize {
        self.decoder.frame_info().frames
    }

    #[wasm_bindgen(js_name = decodeInterleaved)]
    pub fn decode_interleaved(&mut self, access_unit: &[u8]) -> Result<Float32Array, JsValue> {
        let mut interleaved = std::mem::take(&mut self.interleaved);

        {
            let decoded = self
                .decoder
                .decode_access_unit(access_unit)
                .map_err(|error| js_error(error.to_string()))?;
            let channels = decoded.channels();
            let channel_count = channels.len();
            let frames = decoded.frames();

            interleaved.clear();
            interleaved.resize(frames * channel_count, 0.0);

            for frame in 0..frames {
                for (channel_index, channel) in channels.iter().enumerate() {
                    interleaved[frame * channel_count + channel_index] = channel[frame];
                }
            }
        }

        let output = Float32Array::from(interleaved.as_slice());
        self.interleaved = interleaved;
        Ok(output)
    }

    #[wasm_bindgen(js_name = decodeInterleavedInto)]
    pub fn decode_interleaved_into(
        &mut self,
        access_unit: &[u8],
        output: &Float32Array,
    ) -> Result<usize, JsValue> {
        let info = self.decoder.frame_info();
        let required_len = info.frames * info.channels;
        if output.length() < required_len as u32 {
            return Err(js_error(format!(
                "output Float32Array is too small: need {required_len}, got {}",
                output.length()
            )));
        }

        let mut interleaved = std::mem::take(&mut self.interleaved);

        {
            let decoded = self
                .decoder
                .decode_access_unit(access_unit)
                .map_err(|error| js_error(error.to_string()))?;
            let channels = decoded.channels();
            let channel_count = channels.len();
            let frames = decoded.frames();

            interleaved.clear();
            interleaved.resize(frames * channel_count, 0.0);

            for frame in 0..frames {
                for (channel_index, channel) in channels.iter().enumerate() {
                    interleaved[frame * channel_count + channel_index] = channel[frame];
                }
            }
        }

        if output.length() == required_len as u32 {
            output.copy_from(&interleaved);
        } else {
            output
                .subarray(0, required_len as u32)
                .copy_from(&interleaved);
        }
        self.interleaved = interleaved;
        Ok(required_len)
    }

    #[wasm_bindgen(js_name = decodePlanar)]
    pub fn decode_planar(&mut self, access_unit: &[u8]) -> Result<Array, JsValue> {
        let decoded = self
            .decoder
            .decode_access_unit(access_unit)
            .map_err(|error| js_error(error.to_string()))?;
        let array = Array::new();

        for channel in decoded.channels() {
            array.push(&Float32Array::from(channel.as_slice()).into());
        }

        Ok(array)
    }
}

#[cfg(feature = "audio-demux")]
#[wasm_bindgen]
impl WasmAudioTrackDemuxer {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::new_auto()
    }

    #[wasm_bindgen(js_name = newAuto)]
    pub fn new_auto() -> Self {
        Self {
            demuxer: AudioTrackDemuxer::new_auto(),
        }
    }

    #[wasm_bindgen(js_name = newWithFormat)]
    pub fn new_with_format(format: &str) -> Result<WasmAudioTrackDemuxer, JsValue> {
        Ok(Self {
            demuxer: AudioTrackDemuxer::new_with_format(format).map_err(js_error)?,
        })
    }

    /// Push arbitrary container bytes and receive audio-track config/packet events.
    pub fn push(&mut self, bytes: &[u8]) -> Result<Array, JsValue> {
        let events = self.demuxer.push(bytes).map_err(js_error)?;
        audio_demux_events_to_js(events)
    }

    /// Final drain call. The demuxer should not be reused after this.
    pub fn flush(&mut self) -> Result<Array, JsValue> {
        let events = self.demuxer.flush().map_err(js_error)?;
        audio_demux_events_to_js(events)
    }
}

#[cfg(feature = "audio-demux")]
#[wasm_bindgen]
impl WasmCafAudioIndex {
    #[wasm_bindgen(js_name = fromFile)]
    pub fn from_file(bytes: &[u8]) -> Result<WasmCafAudioIndex, JsValue> {
        Ok(Self {
            index: CafAudioIndex::from_file(bytes).map_err(js_error)?,
        })
    }

    pub fn config(&self) -> Result<JsValue, JsValue> {
        audio_demux_event_to_js(AudioDemuxEvent::Config(self.index.config.clone()))
    }

    #[wasm_bindgen(getter, js_name = sampleCount)]
    pub fn sample_count(&self) -> usize {
        self.index.packets.len()
    }

    pub fn sample(&self, index: usize) -> Result<Object, JsValue> {
        let sample = self
            .index
            .packets
            .get(index)
            .ok_or_else(|| js_error(format!("CAF sample index {index} is out of range")))?;
        media_sample_index_to_js(sample)
    }

    pub fn packet(&self, index: usize, source_bytes: &[u8]) -> Result<JsValue, JsValue> {
        let packet = self
            .index
            .packet_from_sample_bytes(index, source_bytes)
            .map_err(js_error)?;
        Ok(media_track_packet_to_js(&packet)?.into())
    }
}

#[cfg(feature = "audio-demux")]
#[wasm_bindgen]
impl WasmMp4MediaIndex {
    /// Construct from the payload bytes inside a `moov` box. This is the
    /// production path for seekable browser files and native file handles.
    #[wasm_bindgen(constructor)]
    pub fn new(moov_payload: &[u8]) -> Result<WasmMp4MediaIndex, JsValue> {
        Ok(Self {
            index: Mp4MediaIndex::from_moov_payload(moov_payload).map_err(js_error)?,
        })
    }

    /// Conformance helper for small complete files. Large browser imports
    /// should locate and read only `moov`, then call the constructor.
    #[wasm_bindgen(js_name = fromFile)]
    pub fn from_file(bytes: &[u8]) -> Result<WasmMp4MediaIndex, JsValue> {
        Ok(Self {
            index: Mp4MediaIndex::from_file(bytes).map_err(js_error)?,
        })
    }

    #[wasm_bindgen(js_name = tracks)]
    pub fn tracks_js(&self) -> Result<Array, JsValue> {
        let output = Array::new();
        for track in &self.index.tracks {
            output.push(&media_track_config_to_js(track)?);
        }
        Ok(output)
    }

    #[wasm_bindgen(getter, js_name = sampleCount)]
    pub fn sample_count(&self) -> usize {
        self.index.samples.len()
    }

    pub fn sample(&self, index: usize) -> Result<Object, JsValue> {
        let sample = self
            .index
            .samples
            .get(index)
            .ok_or_else(|| js_error(format!("MP4 sample index {index} is out of range")))?;
        media_sample_index_to_js(sample)
    }

    /// Validate and normalize exactly one indexed source range.
    pub fn packet(&self, index: usize, source_bytes: &[u8]) -> Result<Object, JsValue> {
        let packet = self
            .index
            .packet_from_sample_bytes(index, source_bytes)
            .map_err(js_error)?;
        media_track_packet_to_js(&packet)
    }

    /// Validate, decode, edit-list trim, and encode one indexed ALAC sample.
    /// JavaScript transports only the requested container byte range; PCM
    /// remains within Rust throughout the operation.
    #[cfg(all(
        feature = "detect",
        feature = "aac-lc",
        feature = "alac",
        feature = "wav",
        feature = "opus",
        feature = "flac"
    ))]
    #[wasm_bindgen(js_name = encodeAlacSample)]
    pub fn encode_alac_sample(
        &self,
        index: usize,
        source_bytes: &[u8],
        encoder: &mut WasmStreamingLibraryEncoder,
    ) -> Result<JsValue, JsValue> {
        let packet = self
            .index
            .packet_from_sample_bytes(index, source_bytes)
            .map_err(js_error)?;
        if packet.kind != MediaTrackKind::Audio || packet.codec != "alac" {
            return Err(js_error(format!("MP4 sample {index} is not ALAC audio")));
        }
        let decoded = encoder.decode_alac_packet(&packet.data).map_err(js_error)?;
        let bytes_per_frame = usize::from(decoded.bits_per_sample().div_ceil(8))
            .checked_mul(usize::from(decoded.channel_count()))
            .ok_or_else(|| js_error("ALAC PCM frame size overflow".to_owned()))?;
        if bytes_per_frame == 0 || decoded.data().len() % bytes_per_frame != 0 {
            return Err(js_error("ALAC decoder returned misaligned PCM".to_owned()));
        }
        let decoded_frames = u32::try_from(decoded.data().len() / bytes_per_frame)
            .map_err(|_| js_error("ALAC packet frame count exceeds u32".to_owned()))?;
        let decoded = match self
            .index
            .pcm_packet_trim(index, decoded_frames)
            .map_err(js_error)?
        {
            Some(trim) => {
                trim_interleaved_audio(decoded, trim.source_frame_start, trim.frame_count)
                    .map_err(js_error)?
            }
            None => None,
        };
        encoder.encode_partial_audio(decoded)
    }

    /// Validate, decode, edit-list trim, and encode one indexed AAC-LC sample.
    #[cfg(all(
        feature = "detect",
        feature = "aac-lc",
        feature = "alac",
        feature = "wav",
        feature = "opus",
        feature = "flac"
    ))]
    #[wasm_bindgen(js_name = encodeAacLcSample)]
    pub fn encode_aac_lc_sample(
        &self,
        index: usize,
        source_bytes: &[u8],
        encoder: &mut WasmStreamingLibraryEncoder,
    ) -> Result<JsValue, JsValue> {
        let packet = self
            .index
            .packet_from_sample_bytes(index, source_bytes)
            .map_err(js_error)?;
        if packet.kind != MediaTrackKind::Audio || packet.codec != "aac" {
            return Err(js_error(format!("MP4 sample {index} is not AAC audio")));
        }
        let decoded = encoder
            .decode_aac_lc_packet(&packet.data)
            .map_err(js_error)?;
        let bytes_per_frame = usize::from(decoded.bits_per_sample().div_ceil(8))
            .checked_mul(usize::from(decoded.channel_count()))
            .ok_or_else(|| js_error("AAC-LC PCM frame size overflow".to_owned()))?;
        let decoded_frames = u32::try_from(decoded.data().len() / bytes_per_frame)
            .map_err(|_| js_error("AAC-LC packet frame count exceeds u32".to_owned()))?;
        let decoded = match self
            .index
            .pcm_packet_trim(index, decoded_frames)
            .map_err(js_error)?
        {
            Some(trim) => {
                trim_interleaved_audio(decoded, trim.source_frame_start, trim.frame_count)
                    .map_err(js_error)?
            }
            None => None,
        };
        encoder.encode_partial_audio(decoded)
    }

    /// Return the Rust-owned slice of decoded PCM that belongs to the edited
    /// programme. `null` means the whole packet is codec preroll or padding.
    #[wasm_bindgen(js_name = pcmTrim)]
    pub fn pcm_trim(&self, index: usize, decoded_frames: u32) -> Result<JsValue, JsValue> {
        let Some(trim) = self
            .index
            .pcm_packet_trim(index, decoded_frames)
            .map_err(js_error)?
        else {
            return Ok(JsValue::NULL);
        };
        pcm_packet_trim_to_js(trim.source_frame_start, trim.frame_count)
    }
}

#[cfg(feature = "alac")]
#[wasm_bindgen]
impl WasmAlacPacketDecoder {
    #[wasm_bindgen(constructor)]
    pub fn new(magic_cookie: &[u8]) -> Result<WasmAlacPacketDecoder, JsValue> {
        Ok(Self {
            decoder: AlacPacketDecoder::new(magic_cookie).map_err(js_error)?,
        })
    }

    /// Decode exactly one container-demuxed ALAC packet.
    pub fn decode(&mut self, packet: &[u8]) -> Result<JsValue, JsValue> {
        let frame = self.decoder.decode_packet(packet).map_err(js_error)?;
        audio_frame_to_js(&frame)
    }

    #[wasm_bindgen(getter, js_name = sampleRate)]
    pub fn sample_rate(&self) -> u32 {
        self.decoder.sample_rate()
    }

    #[wasm_bindgen(getter)]
    pub fn channels(&self) -> u8 {
        self.decoder.channels()
    }

    #[wasm_bindgen(getter, js_name = bitDepth)]
    pub fn bit_depth(&self) -> u8 {
        self.decoder.bit_depth()
    }

    #[wasm_bindgen(getter, js_name = maximumPcmSamples)]
    pub fn maximum_pcm_samples(&self) -> usize {
        self.decoder.maximum_pcm_samples()
    }
}

#[cfg(feature = "alac")]
#[wasm_bindgen]
impl WasmCafAlacIndex {
    #[wasm_bindgen(constructor)]
    pub fn new(
        description: &[u8],
        magic_cookie: &[u8],
        packet_table: &[u8],
        data_payload_offset: f64,
        data_payload_size: f64,
    ) -> Result<WasmCafAlacIndex, JsValue> {
        let data_payload_offset = finite_u64(data_payload_offset).ok_or_else(|| {
            js_error("CAF data offset must be a nonnegative safe integer".to_string())
        })?;
        let data_payload_size = finite_u64(data_payload_size).ok_or_else(|| {
            js_error("CAF data size must be a nonnegative safe integer".to_string())
        })?;
        Ok(Self {
            index: CafAlacPacketIndex::new(
                description,
                magic_cookie,
                packet_table,
                data_payload_offset,
                data_payload_size,
            )
            .map_err(js_error)?,
        })
    }

    #[wasm_bindgen(getter, js_name = magicCookie)]
    pub fn magic_cookie(&self) -> Uint8Array {
        Uint8Array::from(self.index.magic_cookie.as_slice())
    }

    #[wasm_bindgen(getter, js_name = packetCount)]
    pub fn packet_count(&self) -> usize {
        self.index.packets.len()
    }

    #[wasm_bindgen(getter, js_name = sampleRate)]
    pub fn sample_rate(&self) -> u32 {
        self.index.sample_rate
    }

    #[wasm_bindgen(getter)]
    pub fn channels(&self) -> u8 {
        self.index.channels
    }

    #[wasm_bindgen(getter, js_name = bitDepth)]
    pub fn bit_depth(&self) -> u8 {
        self.index.bit_depth
    }

    #[wasm_bindgen(getter, js_name = validFrames)]
    pub fn valid_frames(&self) -> Result<JsValue, JsValue> {
        js_safe_u64(self.index.valid_frames, "CAF valid frame count")
    }

    pub fn sample(&self, index: usize) -> Result<Object, JsValue> {
        let packet = self
            .index
            .packets
            .get(index)
            .ok_or_else(|| js_error(format!("CAF packet index {index} is out of range")))?;
        let object = Object::new();
        Reflect::set(
            &object,
            &"offset".into(),
            &js_safe_u64(packet.offset, "CAF packet offset")?,
        )?;
        Reflect::set(&object, &"size".into(), &packet.size.into())?;
        Ok(object)
    }

    /// Validate exactly one packet range before codec decode.
    pub fn packet(&self, index: usize, source_bytes: &[u8]) -> Result<Uint8Array, JsValue> {
        self.index
            .validate_packet_bytes(index, source_bytes)
            .map_err(js_error)?;
        Ok(Uint8Array::from(source_bytes))
    }

    /// Validate, decode, priming/remainder trim, and encode one CAF packet.
    /// Only the indexed packet bytes cross the WASM boundary.
    #[cfg(all(
        feature = "detect",
        feature = "audio-demux",
        feature = "aac-lc",
        feature = "wav",
        feature = "opus",
        feature = "flac"
    ))]
    #[wasm_bindgen(js_name = encodeAlacSample)]
    pub fn encode_alac_sample(
        &self,
        index: usize,
        source_bytes: &[u8],
        encoder: &mut WasmStreamingLibraryEncoder,
    ) -> Result<JsValue, JsValue> {
        self.index
            .validate_packet_bytes(index, source_bytes)
            .map_err(js_error)?;
        let decoded = encoder.decode_alac_packet(source_bytes).map_err(js_error)?;
        let bytes_per_frame = usize::from(decoded.bits_per_sample().div_ceil(8))
            .checked_mul(usize::from(decoded.channel_count()))
            .ok_or_else(|| js_error("ALAC PCM frame size overflow".to_owned()))?;
        if bytes_per_frame == 0 || decoded.data().len() % bytes_per_frame != 0 {
            return Err(js_error("ALAC decoder returned misaligned PCM".to_owned()));
        }
        let decoded_frames = u64::try_from(decoded.data().len() / bytes_per_frame)
            .map_err(|_| js_error("CAF ALAC packet frame count exceeds u64".to_owned()))?;
        let packet_start = u64::try_from(index)
            .ok()
            .and_then(|index| index.checked_mul(u64::from(self.index.frames_per_packet)))
            .ok_or_else(|| js_error("CAF packet timeline overflow".to_owned()))?;
        let packet_end = packet_start
            .checked_add(decoded_frames)
            .ok_or_else(|| js_error("CAF packet timeline overflow".to_owned()))?;
        let programme_start = u64::from(self.index.priming_frames);
        let programme_end = programme_start
            .checked_add(self.index.valid_frames)
            .ok_or_else(|| js_error("CAF programme timeline overflow".to_owned()))?;
        let start = packet_start.max(programme_start);
        let end = packet_end.min(programme_end);
        let decoded = if end > start {
            trim_interleaved_audio(
                decoded,
                u32::try_from(start - packet_start)
                    .map_err(|_| js_error("CAF ALAC trim start exceeds u32".to_owned()))?,
                u32::try_from(end - start)
                    .map_err(|_| js_error("CAF ALAC trim length exceeds u32".to_owned()))?,
            )
            .map_err(js_error)?
        } else {
            None
        };
        encoder.encode_partial_audio(decoded)
    }
}

/// Validate a CAF file header without reading the source payload.
#[cfg(feature = "alac")]
#[wasm_bindgen(js_name = validateCafFileHeader)]
pub fn validate_caf_file_header_js(header: &[u8], file_size: f64) -> Result<(), JsValue> {
    let file_size = finite_u64(file_size)
        .ok_or_else(|| js_error("CAF file size must be a nonnegative safe integer".to_string()))?;
    validate_caf_file_header(header, file_size).map_err(js_error)
}

/// Inspect one CAF chunk header without reading its payload.
#[cfg(feature = "alac")]
#[wasm_bindgen(js_name = inspectCafChunk)]
pub fn inspect_caf_chunk_js(
    header: &[u8],
    absolute_offset: f64,
    file_size: f64,
) -> Result<Object, JsValue> {
    let absolute_offset = finite_u64(absolute_offset).ok_or_else(|| {
        js_error("CAF chunk offset must be a nonnegative safe integer".to_string())
    })?;
    let file_size = finite_u64(file_size)
        .ok_or_else(|| js_error("CAF file size must be a nonnegative safe integer".to_string()))?;
    let range = inspect_caf_chunk(header, absolute_offset, file_size).map_err(js_error)?;
    let object = Object::new();
    Reflect::set(
        &object,
        &"chunkType".into(),
        &String::from_utf8_lossy(&range.chunk_type).as_ref().into(),
    )?;
    for (field, value) in [
        ("payloadOffset", range.payload_offset),
        ("payloadSize", range.payload_size),
        ("end", range.end),
    ] {
        Reflect::set(&object, &field.into(), &js_safe_u64(value, field)?)?;
    }
    Ok(object)
}

/// Inspect one top-level MOV/MP4 box without reading its payload.
///
/// JavaScript owns only range I/O. Rust owns box sizes, extended sizes, EOF
/// bounds, and the resulting source offsets.
#[cfg(feature = "audio-demux")]
#[wasm_bindgen(js_name = inspectMp4TopLevelBox)]
pub fn inspect_mp4_top_level_box_js(
    header: &[u8],
    absolute_offset: f64,
    file_size: f64,
) -> Result<Object, JsValue> {
    let absolute_offset = finite_u64(absolute_offset).ok_or_else(|| {
        js_error("MOV/MP4 box offset must be a nonnegative safe integer".to_string())
    })?;
    let file_size = finite_u64(file_size).ok_or_else(|| {
        js_error("MOV/MP4 file size must be a nonnegative safe integer".to_string())
    })?;
    let range = inspect_mp4_top_level_box(header, absolute_offset, file_size).map_err(js_error)?;
    let object = Object::new();
    Reflect::set(
        &object,
        &"boxType".into(),
        &String::from_utf8_lossy(&range.box_type).as_ref().into(),
    )?;
    for (field, value) in [
        ("offset", range.offset),
        ("payloadOffset", range.payload_offset),
        ("payloadSize", range.payload_size),
        ("end", range.end),
    ] {
        Reflect::set(&object, &field.into(), &js_safe_u64(value, field)?)?;
    }
    Ok(object)
}

#[cfg(feature = "audio-demux")]
#[wasm_bindgen]
impl WasmMp4MediaDemuxer {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            demuxer: Mp4MediaDemuxer::new(),
        }
    }

    pub fn push(&mut self, bytes: &[u8]) -> Result<Array, JsValue> {
        mp4_media_events_to_js(self.demuxer.push(bytes).map_err(js_error)?)
    }

    pub fn flush(&mut self) -> Result<Array, JsValue> {
        mp4_media_events_to_js(self.demuxer.flush().map_err(js_error)?)
    }

    #[wasm_bindgen(js_name = pcmTrim)]
    pub fn pcm_trim(
        &self,
        track_id: u32,
        presentation_time: f64,
        packet_duration: u32,
        decoded_frames: u32,
    ) -> Result<JsValue, JsValue> {
        let presentation_time = finite_i64(presentation_time).ok_or_else(|| {
            js_error("fragmented MP4 presentation time must be a finite integer".to_string())
        })?;
        let Some(trim) = self
            .demuxer
            .pcm_packet_trim(
                u64::from(track_id),
                presentation_time,
                packet_duration,
                decoded_frames,
            )
            .map_err(js_error)?
        else {
            return Ok(JsValue::NULL);
        };
        pcm_packet_trim_to_js(trim.source_frame_start, trim.frame_count)
    }
}

#[cfg(feature = "audio-demux")]
impl Default for WasmMp4MediaDemuxer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "audio-demux")]
#[wasm_bindgen]
impl WasmMxfMediaDemuxer {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            demuxer: MxfMediaDemuxer::new(),
        }
    }

    pub fn push(&mut self, bytes: &[u8]) -> Result<Array, JsValue> {
        mxf_media_events_to_js(self.demuxer.push(bytes).map_err(js_error)?)
    }

    pub fn flush(&mut self) -> Result<Array, JsValue> {
        mxf_media_events_to_js(self.demuxer.flush().map_err(js_error)?)
    }
}

#[cfg(feature = "audio-demux")]
impl Default for WasmMxfMediaDemuxer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "webm")]
#[wasm_bindgen]
impl WasmWebmMediaDemuxer {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            demuxer: WebmMediaDemuxer::new(),
        }
    }

    pub fn push(&mut self, bytes: &[u8]) -> Result<Array, JsValue> {
        webm_media_events_to_js(self.demuxer.add(bytes).map_err(js_error)?)
    }

    pub fn flush(&mut self) -> Result<Array, JsValue> {
        webm_media_events_to_js(self.demuxer.finish().map_err(js_error)?)
    }
}

#[cfg(feature = "webm")]
impl Default for WasmWebmMediaDemuxer {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl WasmSoundKitFrameDecoder {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::new_unencrypted()
    }

    #[wasm_bindgen(js_name = newUnencrypted)]
    pub fn new_unencrypted() -> Self {
        Self {
            stream: SoundKitFrameStream::default(),
        }
    }

    #[wasm_bindgen(js_name = newWithKeyBytes)]
    pub fn new_with_key_bytes(key: &[u8]) -> Result<WasmSoundKitFrameDecoder, JsValue> {
        let cipher =
            ChaCha20Poly1305PacketCipher::new(key).map_err(|error| js_error(error.to_string()))?;
        Ok(Self::with_cipher(cipher))
    }

    #[wasm_bindgen(js_name = newWithDecimalKey)]
    pub fn new_with_decimal_key(key: &str) -> Result<WasmSoundKitFrameDecoder, JsValue> {
        let cipher = ChaCha20Poly1305PacketCipher::new_from_decimal_key(key)
            .map_err(|error| js_error(error.to_string()))?;
        Ok(Self::with_cipher(cipher))
    }

    #[wasm_bindgen(js_name = setKeyBytes)]
    pub fn set_key_bytes(&mut self, key: &[u8]) -> Result<(), JsValue> {
        let cipher =
            ChaCha20Poly1305PacketCipher::new(key).map_err(|error| js_error(error.to_string()))?;
        self.stream.set_cipher(Some(cipher));
        Ok(())
    }

    #[wasm_bindgen(js_name = setDecimalKey)]
    pub fn set_decimal_key(&mut self, key: &str) -> Result<(), JsValue> {
        let cipher = ChaCha20Poly1305PacketCipher::new_from_decimal_key(key)
            .map_err(|error| js_error(error.to_string()))?;
        self.stream.set_cipher(Some(cipher));
        Ok(())
    }

    #[wasm_bindgen(js_name = clearKey)]
    pub fn clear_key(&mut self) {
        self.stream.set_cipher(None);
    }

    pub fn push(&mut self, bytes: &[u8]) -> Result<Array, JsValue> {
        let frames = self.stream.push(bytes).map_err(js_error)?;
        soundkit_frames_to_js(frames)
    }

    pub fn finish(&self) -> Result<(), JsValue> {
        self.stream.finish().map_err(js_error)
    }

    pub fn reset(&mut self) {
        self.stream.reset();
    }

    #[wasm_bindgen(js_name = bufferedBytes)]
    pub fn buffered_bytes(&self) -> usize {
        self.stream.buffered_bytes()
    }
}

#[wasm_bindgen(js_name = buildSoundKitFrameHeaderV2)]
pub fn build_soundkit_frame_header_v2(
    encoding: u8,
    payload_size: u32,
    sample_size: u32,
    sample_rate: u32,
    channels: u8,
    bits_per_sample: u8,
    pts: f64,
) -> Result<Uint8Array, JsValue> {
    let header = soundkit_frame_header_v2(
        encoding,
        payload_size,
        sample_size,
        sample_rate,
        channels,
        bits_per_sample,
        pts,
    )?;

    let mut output = Vec::with_capacity(header.size());
    header
        .encode(&mut output)
        .map_err(|error| js_error(format!("encode SoundKit v2 header failed: {error}")))?;
    Ok(Uint8Array::from(output.as_slice()))
}

#[wasm_bindgen(js_name = buildSoundKitFrameV2)]
pub fn build_soundkit_frame_v2(
    encoding: u8,
    payload: &[u8],
    sample_size: u32,
    sample_rate: u32,
    channels: u8,
    bits_per_sample: u8,
    pts: f64,
) -> Result<Uint8Array, JsValue> {
    let header = soundkit_frame_header_v2(
        encoding,
        payload.len() as u32,
        sample_size,
        sample_rate,
        channels,
        bits_per_sample,
        pts,
    )?;

    let mut output = Vec::with_capacity(header.size() + payload.len());
    header
        .encode(&mut output)
        .map_err(|error| js_error(format!("encode SoundKit v2 header failed: {error}")))?;
    output.extend_from_slice(payload);
    Ok(Uint8Array::from(output.as_slice()))
}

#[wasm_bindgen]
impl WasmWavEncoder {
    #[wasm_bindgen(constructor)]
    pub fn new(
        sample_rate: u32,
        channels: u16,
        sample_format: &str,
        total_frames: f64,
    ) -> Result<WasmWavEncoder, JsValue> {
        if !total_frames.is_finite()
            || total_frames < 0.0
            || total_frames.fract() != 0.0
            || total_frames > 9_007_199_254_740_991.0
        {
            return Err(js_error(
                "WAV totalFrames must be an exact non-negative JavaScript integer".to_string(),
            ));
        }
        let format = match sample_format.trim().to_ascii_lowercase().as_str() {
            "i16" | "s16" | "pcm16" => WavSampleFormat::I16,
            "i32" | "s32" | "pcm32" => WavSampleFormat::I32,
            "f32" | "float32" => WavSampleFormat::F32,
            other => return Err(js_error(format!("unsupported WAV sample format: {other}"))),
        };
        let encoder =
            WavStreamEncoder::new(format, sample_rate, channels as usize, total_frames as u64)
                .map_err(js_error)?;
        Ok(Self { encoder })
    }

    pub fn header(&self) -> Uint8Array {
        Uint8Array::from(self.encoder.header())
    }

    #[wasm_bindgen(js_name = encodePlanarI16)]
    pub fn encode_planar_i16(
        &mut self,
        planar: &[i16],
        frames_per_channel: u32,
    ) -> Result<Uint8Array, JsValue> {
        validate_wav_encode_chunk(planar.len(), std::mem::size_of::<i16>())?;
        let output = self
            .encoder
            .push_planar_i16(planar, frames_per_channel as usize)
            .map_err(js_error)?;
        Ok(Uint8Array::from(output.as_slice()))
    }

    #[wasm_bindgen(js_name = encodePlanarI32)]
    pub fn encode_planar_i32(
        &mut self,
        planar: &[i32],
        frames_per_channel: u32,
    ) -> Result<Uint8Array, JsValue> {
        validate_wav_encode_chunk(planar.len(), std::mem::size_of::<i32>())?;
        let output = self
            .encoder
            .push_planar_i32(planar, frames_per_channel as usize)
            .map_err(js_error)?;
        Ok(Uint8Array::from(output.as_slice()))
    }

    #[wasm_bindgen(js_name = encodePlanarF32)]
    pub fn encode_planar_f32(
        &mut self,
        planar: &[f32],
        frames_per_channel: u32,
    ) -> Result<Uint8Array, JsValue> {
        validate_wav_encode_chunk(planar.len(), std::mem::size_of::<f32>())?;
        let output = self
            .encoder
            .push_planar_f32(planar, frames_per_channel as usize)
            .map_err(js_error)?;
        Ok(Uint8Array::from(output.as_slice()))
    }

    pub fn finish(&mut self) -> Result<(), JsValue> {
        self.encoder.finish().map_err(js_error)
    }

    #[wasm_bindgen(getter, js_name = framesWritten)]
    pub fn frames_written(&self) -> f64 {
        self.encoder.frames_written() as f64
    }

    #[wasm_bindgen(getter, js_name = totalFrames)]
    pub fn total_frames(&self) -> f64 {
        self.encoder.total_frames() as f64
    }

    #[wasm_bindgen(getter, js_name = isRf64)]
    pub fn is_rf64(&self) -> bool {
        self.encoder.is_rf64()
    }
}

fn validate_wav_encode_chunk(samples: usize, bytes_per_sample: usize) -> Result<(), JsValue> {
    let bytes = samples
        .checked_mul(bytes_per_sample)
        .ok_or_else(|| js_error("WAV encode input size overflows".to_string()))?;
    if bytes > MAX_STREAM_INPUT_CHUNK_BYTES {
        return Err(js_error(format!(
            "WAV encode input chunk exceeds the {MAX_STREAM_INPUT_CHUNK_BYTES} byte streaming budget"
        )));
    }
    Ok(())
}

#[cfg(feature = "flac")]
#[wasm_bindgen]
impl WasmFlacEncoder {
    #[wasm_bindgen(constructor)]
    pub fn new(
        sample_rate: u32,
        channels: u8,
        bits_per_sample: u32,
        frame_size: u32,
        compression_level: u32,
    ) -> Result<WasmFlacEncoder, JsValue> {
        let mut encoder = FlacEncoder::new(
            sample_rate,
            bits_per_sample,
            channels as u32,
            frame_size,
            compression_level,
        );
        encoder.init().map_err(js_error)?;
        Ok(Self {
            encoder,
            channels,
            bits_per_sample,
            frame_size,
        })
    }

    #[wasm_bindgen(js_name = encodePlanarF32)]
    pub fn encode_planar_f32(
        &mut self,
        planar: &[f32],
        frames_per_channel: u32,
    ) -> Result<Uint8Array, JsValue> {
        let channels = self.channels as usize;
        let frames = frames_per_channel as usize;
        let expected = channels
            .checked_mul(frames)
            .ok_or_else(|| js_error("FLAC encode input is too large".to_string()))?;
        if planar.len() < expected {
            return Err(js_error(format!(
                "planar input too short: need {expected} samples, got {}",
                planar.len()
            )));
        }

        let interleaved = planar_f32_to_interleaved_i32(
            &planar[..expected],
            frames,
            channels,
            self.bits_per_sample,
        )?;
        let mut output = vec![0u8; expected.saturating_mul(8).saturating_add(4096)];
        let encoded = self
            .encoder
            .encode_i32(&interleaved, &mut output)
            .map_err(js_error)?;
        output.truncate(encoded);
        Ok(Uint8Array::from(output.as_slice()))
    }

    /// Signal EOF and drain the final FLAC packet.
    /// The encoder can buffer a short final block until this call.
    #[wasm_bindgen(js_name = finish)]
    pub fn finish(&mut self) -> Result<Uint8Array, JsValue> {
        let capacity = (self.frame_size as usize)
            .saturating_mul(self.channels as usize)
            .saturating_mul(8)
            .saturating_add(4096);
        let mut output = vec![0u8; capacity];
        let encoded = self.encoder.finish(&mut output).map_err(js_error)?;
        output.truncate(encoded);
        Ok(Uint8Array::from(output.as_slice()))
    }

    /// Return the current STREAMINFO metadata block. After finish() this
    /// contains the final sample count and PCM MD5.
    #[wasm_bindgen(js_name = streamHeader)]
    pub fn stream_header(&self) -> Uint8Array {
        Uint8Array::from(self.encoder.stream_header())
    }

    pub fn reset(&mut self) -> Result<(), JsValue> {
        self.encoder.reset().map_err(js_error)
    }
}

#[cfg(feature = "opus")]
#[wasm_bindgen]
impl WasmOpusEncoder {
    #[wasm_bindgen(constructor)]
    pub fn new(
        sample_rate: u32,
        channels: u8,
        bitrate: u32,
        frame_size: u32,
    ) -> Result<WasmOpusEncoder, JsValue> {
        // bits_per_sample is unused by the Opus encoder (it operates on i16 PCM).
        let mut encoder = OpusEncoder::new(sample_rate, 16, channels as u32, frame_size, bitrate);
        encoder.init().map_err(js_error)?;
        Ok(Self {
            encoder,
            frame_size,
            channels,
            // Max Opus packet is ~1275 bytes/channel; 4096 covers stereo CBR.
            output: vec![0u8; 4096],
        })
    }

    // Encodes one interleaved-i16 frame of `frame_size * channels` samples (the
    // caller zero-pads the final short frame) and returns the raw Opus packet.
    #[wasm_bindgen(js_name = encodeInterleavedI16)]
    pub fn encode_interleaved_i16(&mut self, interleaved: &[i16]) -> Result<Uint8Array, JsValue> {
        let required = self.frame_size as usize * self.channels as usize;
        if interleaved.len() < required {
            return Err(js_error(format!(
                "opus encode input too short: need {required} samples, got {}",
                interleaved.len()
            )));
        }
        let written = self
            .encoder
            .encode_i16(&interleaved[..required], &mut self.output)
            .map_err(js_error)?;
        Ok(Uint8Array::from(&self.output[..written]))
    }

    pub fn reset(&mut self) -> Result<(), JsValue> {
        self.encoder.reset().map_err(js_error)
    }
}

#[cfg(all(feature = "wav", feature = "opus", feature = "flac"))]
#[wasm_bindgen]
impl WasmPcm16WaveLibraryEncoder {
    #[wasm_bindgen(constructor)]
    pub fn new(preserve_lossless: bool) -> Self {
        Self {
            decoder: WavStreamProcessor::new(),
            preserve_lossless,
            geometry_checked: false,
            finished: false,
            source_digest: Sha256::new(),
            total_frames: 0,
            source_frames: 0,
            opus_frames: 0,
            opus_stream_bytes: 0,
            opus_digest: Sha256::new(),
            opus_index_entries: Vec::new(),
            opus_encoder: None,
            opus_frame: Vec::with_capacity(960 * 2),
            flac_encoder: None,
            flac_frame_size: 0,
            flac_frame: Vec::new(),
            flac_frames: 0,
            flac_stream_bytes: 0,
            flac_digest: Sha256::new(),
            flac_index_entries: Vec::new(),
            pending_flac_descriptor: None,
        }
    }

    /// Parse and encode one bounded WAV byte range.
    pub fn push(&mut self, bytes: &[u8]) -> Result<JsValue, JsValue> {
        if self.finished {
            return Err(js_error(
                "WAV library encoder is already finished".to_owned(),
            ));
        }
        validate_stream_input_chunk(bytes).map_err(js_error)?;
        self.source_digest.update(bytes);
        let decoded = self.decode(bytes).map_err(js_error)?;
        let mut opus_packets = Vec::new();
        let mut flac_packets = Vec::new();
        for audio in decoded {
            self.install_geometry(&audio).map_err(js_error)?;
            self.encode_audio(audio, &mut opus_packets, &mut flac_packets)
                .map_err(js_error)?;
        }
        wave_library_result(
            opus_packets,
            flac_packets,
            false,
            self.source_frames,
            self.total_frames,
            None,
            None,
            None,
            None,
            None,
        )
    }

    /// Drain the last partial Opus/FLAC blocks. No complete PCM is retained.
    pub fn finish(&mut self) -> Result<JsValue, JsValue> {
        if self.finished {
            return Err(js_error(
                "WAV library encoder is already finished".to_owned(),
            ));
        }
        let decoded = self.decode(&[]).map_err(js_error)?;
        let mut opus_packets = Vec::new();
        let mut flac_packets = Vec::new();
        for audio in decoded {
            self.install_geometry(&audio).map_err(js_error)?;
            self.encode_audio(audio, &mut opus_packets, &mut flac_packets)
                .map_err(js_error)?;
        }
        self.finished = true;
        if !self.geometry_checked || self.source_frames == 0 {
            return Err(js_error("WAV contained no PCM16 audio".to_owned()));
        }
        if self.source_frames != self.total_frames {
            return Err(js_error(format!(
                "WAV frame count changed while streaming: expected {}, received {}",
                self.total_frames, self.source_frames
            )));
        }
        if !self.opus_frame.is_empty() {
            self.emit_opus_packet(&mut opus_packets, true)
                .map_err(js_error)?;
        }
        if self.preserve_lossless {
            if !self.flac_frame.is_empty() {
                self.emit_flac_packet(&mut flac_packets, true)
                    .map_err(js_error)?;
            }
            let mut output = vec![
                0u8;
                self.flac_frame_size
                    .saturating_mul(2)
                    .saturating_mul(8)
                    .saturating_add(4096)
            ];
            let written = {
                let encoder = self
                    .flac_encoder
                    .as_mut()
                    .ok_or_else(|| js_error("WAV FLAC encoder is unavailable".to_owned()))?;
                let written = encoder.finish(&mut output).map_err(js_error)?;
                if encoder.stream_header().is_empty() {
                    return Err(js_error("FLAC did not finalize STREAMINFO".to_owned()));
                }
                written
            };
            output.truncate(written);
            match (self.pending_flac_descriptor.take(), output.is_empty()) {
                (Some((start_frame, frame_count)), false) => {
                    let sequence = self.flac_index_entries.len() as u64;
                    output = frame_library_packet(
                        frame_header::EncodingFlag::FLAC,
                        output,
                        frame_count,
                        48_000,
                        2,
                        24,
                        sequence,
                        start_frame,
                    )
                    .map_err(js_error)?;
                    self.flac_index_entries
                        .push((self.flac_stream_bytes, start_frame));
                    self.flac_digest.update(&output);
                    self.flac_stream_bytes = self
                        .flac_stream_bytes
                        .checked_add(output.len() as u64)
                        .ok_or_else(|| js_error("WAV FLAC stream length overflowed".to_owned()))?;
                    flac_packets.push(WaveLibraryPacket {
                        bytes: output,
                        start_frame,
                        frame_count,
                    });
                }
                (Some(_), true) => {
                    return Err(js_error("FLAC did not emit its final WAV block".to_owned()));
                }
                (None, false) => {
                    return Err(js_error("FLAC emitted an unaccounted WAV block".to_owned()));
                }
                (None, true) => {}
            }
        }
        self.opus_encoder = None;
        self.flac_encoder = None;
        let opus_index = soundkit_frame_index(
            48_000,
            self.total_frames,
            self.opus_stream_bytes,
            &self.opus_index_entries,
        )
        .map_err(js_error)?;
        let flac_index = self
            .preserve_lossless
            .then(|| {
                soundkit_frame_index(
                    48_000,
                    self.total_frames,
                    self.flac_stream_bytes,
                    &self.flac_index_entries,
                )
            })
            .transpose()
            .map_err(js_error)?;
        let source_identity = format!("sha256:{:x}", self.source_digest.clone().finalize());
        let opus_identity = format!("sha256:{:x}", self.opus_digest.clone().finalize());
        let flac_identity = self
            .preserve_lossless
            .then(|| format!("sha256:{:x}", self.flac_digest.clone().finalize()));
        wave_library_result(
            opus_packets,
            flac_packets,
            true,
            self.source_frames,
            self.total_frames,
            Some(opus_index),
            flac_index,
            Some(source_identity),
            Some(opus_identity),
            flac_identity,
        )
    }
}

#[cfg(all(feature = "wav", feature = "opus", feature = "flac"))]
impl WasmPcm16WaveLibraryEncoder {
    fn decode(&mut self, bytes: &[u8]) -> Result<Vec<AudioData>, String> {
        let mut decoded = Vec::new();
        if let Some(audio) = self.decoder.add(bytes)? {
            decoded.push(audio);
        }
        while let Some(audio) = self.decoder.add(&[])? {
            decoded.push(audio);
        }
        Ok(decoded)
    }

    fn install_geometry(&mut self, audio: &AudioData) -> Result<(), String> {
        let supported = audio.sampling_rate() == 48_000
            && audio.channel_count() == 2
            && audio.bits_per_sample() == 16
            && audio.audio_format() == frame_header::EncodingFlag::PCMSigned;
        if self.geometry_checked {
            if !supported {
                return Err("WAV PCM geometry changed while streaming".to_owned());
            }
            return Ok(());
        }
        self.geometry_checked = true;
        if !supported {
            return Err(
                "The direct WAV library path requires 48 kHz stereo signed PCM16".to_owned(),
            );
        }
        self.total_frames = self
            .decoder
            .total_frames()
            .ok_or_else(|| "WAV data chunk has no exact frame count".to_owned())?;
        if self.total_frames == 0 {
            return Err("WAV data chunk contains no PCM frames".to_owned());
        }

        let mut opus = OpusEncoder::new(48_000, 16, 2, 960, 192_000);
        opus.init()?;
        self.opus_encoder = Some(opus);

        if self.preserve_lossless {
            self.flac_frame_size = wave_library_flac_frame_size(self.total_frames, 4096)?;
            self.flac_frame = Vec::with_capacity(self.flac_frame_size * 2);
            let mut flac = FlacEncoder::new(48_000, 24, 2, self.flac_frame_size as u32, 5);
            flac.init()?;
            self.flac_encoder = Some(flac);
        }
        Ok(())
    }

    fn encode_audio(
        &mut self,
        audio: AudioData,
        opus_packets: &mut Vec<WaveLibraryPacket>,
        flac_packets: &mut Vec<WaveLibraryPacket>,
    ) -> Result<(), String> {
        if audio.sampling_rate() != 48_000
            || audio.channel_count() != 2
            || audio.bits_per_sample() != 16
            || audio.audio_format() != frame_header::EncodingFlag::PCMSigned
        {
            return Err("WAV PCM geometry changed while streaming".to_owned());
        }
        if !audio.data().len().is_multiple_of(4) {
            return Err("WAV PCM16 stereo block is not frame-aligned".to_owned());
        }
        for frame in audio.data().chunks_exact(4) {
            let left = i16::from_le_bytes([frame[0], frame[1]]);
            let right = i16::from_le_bytes([frame[2], frame[3]]);
            self.opus_frame.push(wave_library_opus_sample(left));
            self.opus_frame.push(wave_library_opus_sample(right));
            if self.opus_frame.len() == 960 * 2 {
                self.emit_opus_packet(opus_packets, false)?;
            }
            if self.preserve_lossless {
                // This is byte-for-byte the old Float32 -> 24-bit mapping for
                // PCM16, without materializing the intermediate Float32 planes.
                self.flac_frame.push(wave_library_flac_sample(left));
                self.flac_frame.push(wave_library_flac_sample(right));
                if self.flac_frame.len() == self.flac_frame_size * 2 {
                    self.emit_flac_packet(flac_packets, false)?;
                }
            }
            self.source_frames += 1;
        }
        if self.source_frames > self.total_frames {
            return Err("WAV emitted more PCM frames than its data chunk declares".to_owned());
        }
        Ok(())
    }

    fn emit_opus_packet(
        &mut self,
        packets: &mut Vec<WaveLibraryPacket>,
        final_packet: bool,
    ) -> Result<(), String> {
        let frame_count = (self.opus_frame.len() / 2) as u32;
        if frame_count == 0 {
            return Ok(());
        }
        if frame_count < 960 {
            if !final_packet {
                return Err("short Opus WAV block appeared before EOF".to_owned());
            }
            self.opus_frame.resize(960 * 2, 0);
        }
        let encoder = self
            .opus_encoder
            .as_mut()
            .ok_or_else(|| "WAV Opus encoder is unavailable".to_owned())?;
        let mut output = vec![0u8; 4096];
        let written = encoder.encode_i16(&self.opus_frame, &mut output)?;
        if written == 0 {
            return Err("Opus emitted an empty WAV packet".to_owned());
        }
        output.truncate(written);
        let start_frame = self.opus_frames;
        let sequence = self.opus_index_entries.len() as u64;
        output = frame_library_packet(
            frame_header::EncodingFlag::Opus,
            output,
            frame_count,
            48_000,
            2,
            16,
            sequence,
            start_frame,
        )?;
        self.opus_index_entries
            .push((self.opus_stream_bytes, start_frame));
        self.opus_digest.update(&output);
        self.opus_stream_bytes = self
            .opus_stream_bytes
            .checked_add(output.len() as u64)
            .ok_or_else(|| "WAV Opus stream length overflowed".to_owned())?;
        packets.push(WaveLibraryPacket {
            bytes: output,
            start_frame,
            frame_count,
        });
        self.opus_frames += u64::from(frame_count);
        self.opus_frame.clear();
        Ok(())
    }

    fn emit_flac_packet(
        &mut self,
        packets: &mut Vec<WaveLibraryPacket>,
        final_packet: bool,
    ) -> Result<(), String> {
        let frame_count = (self.flac_frame.len() / 2) as u32;
        if frame_count == 0 {
            return Ok(());
        }
        let encoder = self
            .flac_encoder
            .as_mut()
            .ok_or_else(|| "WAV FLAC encoder is unavailable".to_owned())?;
        let mut output = vec![0u8; self.flac_frame.len().saturating_mul(8).saturating_add(4096)];
        let written = encoder.encode_i32(&self.flac_frame, &mut output)?;
        output.truncate(written);
        let descriptor = (self.flac_frames, frame_count);
        if output.is_empty() {
            if !final_packet || self.pending_flac_descriptor.is_some() {
                return Err("FLAC buffered an unexpected WAV block".to_owned());
            }
            self.pending_flac_descriptor = Some(descriptor);
        } else {
            let sequence = self.flac_index_entries.len() as u64;
            output = frame_library_packet(
                frame_header::EncodingFlag::FLAC,
                output,
                frame_count,
                48_000,
                2,
                24,
                sequence,
                descriptor.0,
            )?;
            self.flac_index_entries
                .push((self.flac_stream_bytes, descriptor.0));
            self.flac_digest.update(&output);
            self.flac_stream_bytes = self
                .flac_stream_bytes
                .checked_add(output.len() as u64)
                .ok_or_else(|| "WAV FLAC stream length overflowed".to_owned())?;
            packets.push(WaveLibraryPacket {
                bytes: output,
                start_frame: descriptor.0,
                frame_count: descriptor.1,
            });
        }
        self.flac_frames += u64::from(frame_count);
        self.flac_frame.clear();
        Ok(())
    }
}

#[cfg(all(
    feature = "detect",
    feature = "audio-demux",
    feature = "aac-lc",
    feature = "alac",
    feature = "wav",
    feature = "opus",
    feature = "flac"
))]
#[wasm_bindgen]
impl WasmStreamingLibraryEncoder {
    #[wasm_bindgen(constructor)]
    pub fn new(preserve_lossless: bool) -> Result<WasmStreamingLibraryEncoder, JsValue> {
        Self::new_rust(preserve_lossless).map_err(js_error)
    }

    /// Open the same bounded output pipeline for a seekable ALAC container.
    /// The adapter supplies Rust-validated packet ranges; decoded PCM remains
    /// inside this object and feeds the shared Opus/FLAC encoders directly.
    #[wasm_bindgen(js_name = newAlac)]
    pub fn new_alac(
        magic_cookie: &[u8],
        preserve_lossless: bool,
    ) -> Result<WasmStreamingLibraryEncoder, JsValue> {
        Self::new_alac_rust(magic_cookie, preserve_lossless).map_err(js_error)
    }

    /// Open the bounded output pipeline for seekable AAC-LC container samples.
    #[wasm_bindgen(js_name = newAacLc)]
    pub fn new_aac_lc(
        audio_specific_config: &[u8],
        preserve_lossless: bool,
    ) -> Result<WasmStreamingLibraryEncoder, JsValue> {
        Self::new_aac_lc_rust(audio_specific_config, preserve_lossless).map_err(js_error)
    }

    /// Hash a bounded source range without decoding it. Seekable container
    /// adapters use this while scanning metadata and packet ranges once.
    #[wasm_bindgen(js_name = updateSourceBytes)]
    pub fn update_source_bytes(&mut self, bytes: &[u8]) -> Result<(), JsValue> {
        self.update_source_bytes_rust(bytes).map_err(js_error)
    }

    /// Decode one indexed ALAC access unit and encode only its Rust-selected
    /// presentation-frame slice.
    #[wasm_bindgen(js_name = pushAlacPacket)]
    pub fn push_alac_packet(
        &mut self,
        packet: &[u8],
        source_frame_start: u32,
        frame_count: u32,
    ) -> Result<JsValue, JsValue> {
        wave_library_batch_to_js(
            self.push_alac_packet_rust(packet, source_frame_start, frame_count)
                .map_err(js_error)?,
        )
    }

    /// Decode one indexed AAC-LC access unit and encode its selected frames.
    #[wasm_bindgen(js_name = pushAacLcPacket)]
    pub fn push_aac_lc_packet(
        &mut self,
        packet: &[u8],
        source_frame_start: u32,
        frame_count: u32,
    ) -> Result<JsValue, JsValue> {
        wave_library_batch_to_js(
            self.push_aac_lc_packet_rust(packet, source_frame_start, frame_count)
                .map_err(js_error)?,
        )
    }

    /// Decode and encode one bounded source byte range.
    pub fn push(&mut self, bytes: &[u8]) -> Result<JsValue, JsValue> {
        wave_library_batch_to_js(self.push_rust(bytes).map_err(js_error)?)
    }

    /// Drain decoder, resampler, and codec tails without retaining complete
    /// PCM in either Rust or JavaScript.
    pub fn finish(&mut self) -> Result<JsValue, JsValue> {
        wave_library_batch_to_js(self.finish_rust().map_err(js_error)?)
    }
}

#[cfg(all(
    feature = "detect",
    feature = "audio-demux",
    feature = "aac-lc",
    feature = "alac",
    feature = "wav",
    feature = "opus",
    feature = "flac"
))]
impl WasmStreamingLibraryEncoder {
    pub fn new_rust(preserve_lossless: bool) -> Result<Self, String> {
        let mut opus_encoder = OpusEncoder::new(48_000, 16, 2, 960, 192_000);
        opus_encoder.init()?;
        let flac_frame_size = 4096;
        Ok(Self {
            decoder: LibrarySourceDecoder::new(),
            alac_decoder: None,
            aac_lc_decoder: None,
            normalizer: StreamingStereo48kNormalizer::new(),
            preserve_lossless,
            finished: false,
            source_digest: Sha256::new(),
            opus_frames: 0,
            opus_stream_bytes: 0,
            opus_digest: Sha256::new(),
            opus_index_entries: Vec::new(),
            opus_encoder,
            opus_frame: Vec::with_capacity(960 * 2),
            flac_encoder: None,
            flac_frame_size,
            flac_frame: Vec::new(),
            flac_sample_rate: 0,
            flac_channels: 0,
            flac_frames: 0,
            flac_stream_bytes: 0,
            flac_digest: Sha256::new(),
            flac_index_entries: Vec::new(),
        })
    }

    /// Open a Rust-native encoder for indexed ALAC packets.
    pub fn new_alac_rust(magic_cookie: &[u8], preserve_lossless: bool) -> Result<Self, String> {
        let mut encoder = Self::new_rust(preserve_lossless)?;
        encoder.decoder = LibrarySourceDecoder::Finished;
        encoder.alac_decoder = Some(AlacPacketDecoder::new(magic_cookie)?);
        Ok(encoder)
    }

    /// Open a Rust-native encoder for indexed AAC-LC access units.
    pub fn new_aac_lc_rust(
        audio_specific_config: &[u8],
        preserve_lossless: bool,
    ) -> Result<Self, String> {
        let mut encoder = Self::new_rust(preserve_lossless)?;
        encoder.decoder = LibrarySourceDecoder::Finished;
        encoder.aac_lc_decoder = Some(
            AacLcDecoder::from_audio_specific_config(audio_specific_config)
                .map_err(|error| error.to_string())?,
        );
        Ok(encoder)
    }

    /// Open a Rust-native encoder for indexed PCM packets.
    pub fn new_seekable_pcm_rust(preserve_lossless: bool) -> Result<Self, String> {
        let mut encoder = Self::new_rust(preserve_lossless)?;
        encoder.decoder = LibrarySourceDecoder::Finished;
        Ok(encoder)
    }

    /// Add one bounded source range to the source identity without decoding it.
    pub fn update_source_bytes_rust(&mut self, bytes: &[u8]) -> Result<(), String> {
        self.ensure_active()?;
        validate_stream_input_chunk(bytes)?;
        self.source_digest.update(bytes);
        Ok(())
    }

    /// Decode and encode one validated ALAC access unit.
    pub fn push_alac_packet_rust(
        &mut self,
        packet: &[u8],
        source_frame_start: u32,
        frame_count: u32,
    ) -> Result<LibraryEncodeBatch, String> {
        self.ensure_active()?;
        let decoded = self.decode_alac_packet(packet)?;
        let decoded = trim_interleaved_audio(decoded, source_frame_start, frame_count)?;
        self.encode_partial_audio_rust(decoded)
    }

    /// Decode and encode one validated AAC-LC access unit.
    pub fn push_aac_lc_packet_rust(
        &mut self,
        packet: &[u8],
        source_frame_start: u32,
        frame_count: u32,
    ) -> Result<LibraryEncodeBatch, String> {
        self.ensure_active()?;
        let decoded = self.decode_aac_lc_packet(packet)?;
        let decoded = trim_interleaved_audio(decoded, source_frame_start, frame_count)?;
        self.encode_partial_audio_rust(decoded)
    }

    /// Validate, decode, trim, and encode one indexed CAF packet.
    pub fn push_caf_sample_rust(
        &mut self,
        index: &CafAudioIndex,
        sample_index: usize,
        source_bytes: &[u8],
    ) -> Result<LibraryEncodeBatch, String> {
        self.ensure_active()?;
        let packet = index.packet_from_sample_bytes(sample_index, source_bytes)?;
        let decoded = match &index.config.codec {
            AudioCodec::Pcm => audio_data_from_container_pcm(&index.config, packet.data)?,
            AudioCodec::Alac => self.decode_alac_packet(&packet.data)?,
            codec => {
                return Err(format!(
                    "unsupported seekable CAF audio codec: {}",
                    codec.as_str()
                ))
            }
        };
        let decoded_frames = decoded_audio_frame_count(&decoded)?;
        let decoded = match index.pcm_packet_trim(sample_index, decoded_frames)? {
            Some(trim) => {
                trim_interleaved_audio(decoded, trim.source_frame_start, trim.frame_count)?
            }
            None => None,
        };
        self.encode_partial_audio_rust(decoded)
    }

    /// Validate and encode one contiguous run of indexed CAF PCM packets.
    pub fn push_caf_pcm_range_rust(
        &mut self,
        index: &CafAudioIndex,
        sample_start: usize,
        sample_end: usize,
        source_bytes: &[u8],
    ) -> Result<LibraryEncodeBatch, String> {
        self.ensure_active()?;
        if index.config.codec != AudioCodec::Pcm {
            return Err("CAF packet range is not PCM".to_owned());
        }
        let samples = index
            .packets
            .get(sample_start..sample_end)
            .filter(|samples| !samples.is_empty())
            .ok_or_else(|| {
                format!("CAF sample range {sample_start}..{sample_end} is out of range")
            })?;
        let mut expected_bytes = 0usize;
        let mut expected_frames = 0u64;
        let mut expected_offset = samples[0].absolute_offset;
        for sample in samples {
            if sample.absolute_offset != expected_offset {
                return Err("CAF PCM sample range is not contiguous".to_owned());
            }
            expected_bytes = expected_bytes
                .checked_add(sample.size as usize)
                .ok_or_else(|| "CAF PCM sample range length overflows usize".to_owned())?;
            expected_offset = expected_offset
                .checked_add(u64::from(sample.size))
                .ok_or_else(|| "CAF PCM sample range offset overflows u64".to_owned())?;
            expected_frames = expected_frames
                .checked_add(u64::from(sample.duration))
                .ok_or_else(|| "CAF PCM sample range duration overflows u64".to_owned())?;
        }
        if source_bytes.len() != expected_bytes {
            return Err(format!(
                "CAF PCM sample range expected {expected_bytes} bytes, got {}",
                source_bytes.len()
            ));
        }
        let decoded = audio_data_from_container_pcm(&index.config, source_bytes.to_vec())?;
        let decoded_frames = decoded_audio_frame_count(&decoded)?;
        if u64::from(decoded_frames) != expected_frames {
            return Err(format!(
                "CAF PCM sample range expected {expected_frames} frames, decoded {decoded_frames}"
            ));
        }
        let decoded = match index.pcm_packet_trim(sample_start, decoded_frames)? {
            Some(trim) => {
                trim_interleaved_audio(decoded, trim.source_frame_start, trim.frame_count)?
            }
            None => None,
        };
        self.encode_partial_audio_rust(decoded)
    }

    /// Validate, decode, edit-list trim, and encode one indexed MOV/MP4 sample.
    pub fn push_mp4_sample_rust(
        &mut self,
        index: &Mp4MediaIndex,
        sample_index: usize,
        source_bytes: &[u8],
    ) -> Result<LibraryEncodeBatch, String> {
        self.ensure_active()?;
        let packet = index.packet_from_sample_bytes(sample_index, source_bytes)?;
        if packet.kind != MediaTrackKind::Audio {
            return Err(format!("MP4 sample {sample_index} is not audio"));
        }
        let track = index
            .tracks
            .iter()
            .find(|track| track.track_id == packet.track_id)
            .ok_or_else(|| format!("MP4 sample references unknown track {}", packet.track_id))?;
        let decoded = match packet.codec.as_str() {
            "aac" => self.decode_aac_lc_packet(&packet.data)?,
            "alac" => self.decode_alac_packet(&packet.data)?,
            "pcm" => audio_data_from_media_pcm(track, packet.data)?,
            codec => return Err(format!("unsupported seekable MP4 audio codec: {codec}")),
        };
        let decoded_frames = decoded_audio_frame_count(&decoded)?;
        let decoded = match index.pcm_packet_trim(sample_index, decoded_frames)? {
            Some(trim) => {
                trim_interleaved_audio(decoded, trim.source_frame_start, trim.frame_count)?
            }
            None => None,
        };
        self.encode_partial_audio_rust(decoded)
    }

    pub fn push_rust(&mut self, bytes: &[u8]) -> Result<LibraryEncodeBatch, String> {
        self.ensure_active()?;
        validate_stream_input_chunk(bytes)?;
        self.source_digest.update(bytes);
        let decoded = self.decoder.push(bytes)?;
        let mut opus_packets = Vec::new();
        let mut flac_packets = Vec::new();
        self.encode_decoded(decoded, &mut opus_packets, &mut flac_packets)?;
        Ok(wave_library_batch(
            opus_packets,
            flac_packets,
            false,
            self.normalizer.output_frames(),
            0,
            None,
            None,
            None,
            None,
            None,
        ))
    }

    pub fn finish_rust(&mut self) -> Result<LibraryEncodeBatch, String> {
        if self.finished {
            return Err("streaming library encoder is already finished".to_owned());
        }
        let decoded = self.decoder.flush()?;
        let mut opus_packets = Vec::new();
        let mut flac_packets = Vec::new();
        self.encode_decoded(decoded, &mut opus_packets, &mut flac_packets)?;
        if let Some(tail) = self.normalizer.finish()? {
            self.encode_stereo(tail.left, tail.right, &mut opus_packets)?;
        }
        self.finished = true;
        let total_frames = self.normalizer.output_frames();
        if total_frames == 0 {
            return Err("source contained no decoded PCM".to_owned());
        }
        if !self.opus_frame.is_empty() {
            self.emit_opus_packet(&mut opus_packets, true)?;
        }
        if self.preserve_lossless {
            self.finish_flac_packets(&mut flac_packets)?;
            let mut output = vec![
                0u8;
                self.flac_frame_size
                    .saturating_mul(self.flac_channels as usize)
                    .saturating_mul(8)
                    .saturating_add(4096)
            ];
            let written = self
                .flac_encoder
                .as_mut()
                .ok_or_else(|| "streaming FLAC encoder is unavailable".to_owned())?
                .finish(&mut output)?;
            if written != 0 {
                return Err("streaming FLAC encoder emitted an unaccounted final packet".to_owned());
            }
        }
        let opus_index = soundkit_frame_index(
            48_000,
            total_frames,
            self.opus_stream_bytes,
            &self.opus_index_entries,
        )?;
        let flac_index = self
            .preserve_lossless
            .then(|| {
                soundkit_frame_index(
                    self.flac_sample_rate,
                    self.flac_frames,
                    self.flac_stream_bytes,
                    &self.flac_index_entries,
                )
            })
            .transpose()?;
        let source_identity = format!("sha256:{:x}", self.source_digest.clone().finalize());
        let opus_identity = format!("sha256:{:x}", self.opus_digest.clone().finalize());
        let flac_identity = self
            .preserve_lossless
            .then(|| format!("sha256:{:x}", self.flac_digest.clone().finalize()));
        Ok(wave_library_batch(
            opus_packets,
            flac_packets,
            true,
            total_frames,
            total_frames,
            Some(opus_index),
            flac_index,
            Some(source_identity),
            Some(opus_identity),
            flac_identity,
        ))
    }
}

#[cfg(all(
    feature = "detect",
    feature = "audio-demux",
    feature = "aac-lc",
    feature = "alac",
    feature = "wav",
    feature = "opus",
    feature = "flac"
))]
impl WasmStreamingLibraryEncoder {
    fn ensure_active(&self) -> Result<(), String> {
        if self.finished {
            Err("streaming library encoder is already finished".to_owned())
        } else {
            Ok(())
        }
    }

    fn decode_alac_packet(&mut self, packet: &[u8]) -> Result<AudioData, String> {
        validate_stream_input_chunk(packet)?;
        self.alac_decoder
            .as_mut()
            .ok_or_else(|| "ALAC library decoder is not open".to_owned())?
            .decode_packet(packet)
    }

    fn decode_aac_lc_packet(&mut self, packet: &[u8]) -> Result<AudioData, String> {
        validate_stream_input_chunk(packet)?;
        let decoder = self
            .aac_lc_decoder
            .as_mut()
            .ok_or_else(|| "AAC-LC library decoder is not open".to_owned())?;
        let info = decoder.frame_info();
        let decoded = decoder
            .decode_access_unit(packet)
            .map_err(|error| error.to_string())?;
        let mut interleaved = Vec::with_capacity(decoded.frames() * info.channels);
        for frame in 0..decoded.frames() {
            for channel in decoded.channels() {
                interleaved.push(library_float_to_i16(channel[frame]));
            }
        }
        Ok(audio_data_i16(
            info.sample_rate,
            u8::try_from(info.channels)
                .map_err(|_| "AAC-LC channel count exceeds SoundKit".to_owned())?,
            &interleaved,
        ))
    }

    fn encode_partial_audio(&mut self, decoded: Option<AudioData>) -> Result<JsValue, JsValue> {
        wave_library_batch_to_js(self.encode_partial_audio_rust(decoded).map_err(js_error)?)
    }

    fn encode_partial_audio_rust(
        &mut self,
        decoded: Option<AudioData>,
    ) -> Result<LibraryEncodeBatch, String> {
        self.ensure_active()?;
        let mut opus_packets = Vec::new();
        let mut flac_packets = Vec::new();
        if let Some(decoded) = decoded {
            self.encode_decoded(vec![decoded], &mut opus_packets, &mut flac_packets)?;
        }
        Ok(wave_library_batch(
            opus_packets,
            flac_packets,
            false,
            self.normalizer.output_frames(),
            0,
            None,
            None,
            None,
            None,
            None,
        ))
    }

    fn encode_decoded(
        &mut self,
        decoded: Vec<AudioData>,
        opus_packets: &mut Vec<WaveLibraryPacket>,
        flac_packets: &mut Vec<WaveLibraryPacket>,
    ) -> Result<(), String> {
        for audio in decoded {
            if self.preserve_lossless {
                self.encode_preservation_audio(&audio, flac_packets)?;
            }
            if let Some(block) = self.normalizer.push(&audio)? {
                self.encode_stereo(block.left, block.right, opus_packets)?;
            }
        }
        Ok(())
    }

    fn encode_stereo(
        &mut self,
        left: Vec<f32>,
        right: Vec<f32>,
        opus_packets: &mut Vec<WaveLibraryPacket>,
    ) -> Result<(), String> {
        if left.is_empty() || left.len() != right.len() {
            return Err("streaming normalizer returned invalid stereo PCM".to_owned());
        }
        for (left, right) in left.into_iter().zip(right) {
            self.opus_frame.push(library_float_to_i16(left));
            self.opus_frame.push(library_float_to_i16(right));
            if self.opus_frame.len() == 960 * 2 {
                self.emit_opus_packet(opus_packets, false)?;
            }
        }
        Ok(())
    }

    fn encode_preservation_audio(
        &mut self,
        audio: &AudioData,
        packets: &mut Vec<WaveLibraryPacket>,
    ) -> Result<(), String> {
        let sample_rate = audio.sampling_rate();
        let channels = audio.channel_count();
        if sample_rate == 0 || channels == 0 || channels > 8 {
            return Err(format!(
                "lossless preservation has unsupported PCM geometry {sample_rate} Hz/{channels} ch"
            ));
        }
        if self.flac_sample_rate == 0 {
            let mut encoder = FlacEncoder::new(
                sample_rate,
                24,
                u32::from(channels),
                self.flac_frame_size as u32,
                5,
            );
            encoder.init()?;
            self.flac_encoder = Some(encoder);
            self.flac_sample_rate = sample_rate;
            self.flac_channels = channels;
            self.flac_frame =
                Vec::with_capacity((self.flac_frame_size + 31).saturating_mul(channels as usize));
        } else if self.flac_sample_rate != sample_rate || self.flac_channels != channels {
            return Err(format!(
                "decoded PCM geometry changed from {} Hz/{} ch to {sample_rate} Hz/{channels} ch",
                self.flac_sample_rate, self.flac_channels
            ));
        }

        let planar = audio_to_f32_channels(audio)?;
        let frames = planar
            .first()
            .map(Vec::len)
            .ok_or_else(|| "decoded audio contained no channels".to_owned())?;
        if frames == 0 || planar.len() != channels as usize {
            return Err("decoded audio contained an invalid preservation block".to_owned());
        }
        if planar.iter().any(|channel| channel.len() != frames) {
            return Err("decoded audio channels have mismatched frame counts".to_owned());
        }
        for frame in 0..frames {
            for channel in &planar {
                self.flac_frame.push(library_float_to_s24(channel[frame]));
            }
        }
        while self.flac_frame.len() / channels as usize >= self.flac_frame_size + 32 {
            self.emit_flac_packet(packets, self.flac_frame_size)?;
        }
        Ok(())
    }

    fn emit_opus_packet(
        &mut self,
        packets: &mut Vec<WaveLibraryPacket>,
        final_packet: bool,
    ) -> Result<(), String> {
        let frame_count = (self.opus_frame.len() / 2) as u32;
        if frame_count == 0 {
            return Ok(());
        }
        if frame_count < 960 {
            if !final_packet {
                return Err("short Opus block appeared before EOF".to_owned());
            }
            self.opus_frame.resize(960 * 2, 0);
        }
        let mut output = vec![0u8; 4096];
        let written = self
            .opus_encoder
            .encode_i16(&self.opus_frame, &mut output)?;
        if written == 0 {
            return Err("Opus emitted an empty streaming packet".to_owned());
        }
        output.truncate(written);
        let start_frame = self.opus_frames;
        output = frame_library_packet(
            frame_header::EncodingFlag::Opus,
            output,
            frame_count,
            48_000,
            2,
            16,
            self.opus_index_entries.len() as u64,
            start_frame,
        )?;
        self.opus_index_entries
            .push((self.opus_stream_bytes, start_frame));
        self.opus_digest.update(&output);
        self.opus_stream_bytes = self
            .opus_stream_bytes
            .checked_add(output.len() as u64)
            .ok_or_else(|| "streaming Opus length overflowed".to_owned())?;
        packets.push(WaveLibraryPacket {
            bytes: output,
            start_frame,
            frame_count,
        });
        self.opus_frames += u64::from(frame_count);
        self.opus_frame.clear();
        Ok(())
    }

    fn finish_flac_packets(&mut self, packets: &mut Vec<WaveLibraryPacket>) -> Result<(), String> {
        let channels = self.flac_channels as usize;
        if channels == 0 {
            return Err("streaming FLAC encoder received no PCM geometry".to_owned());
        }
        let mut remaining = self.flac_frame.len() / channels;
        if remaining == 0 {
            return Ok(());
        }
        if self.flac_frames == 0 && remaining < 32 {
            return Err("streaming FLAC requires at least 32 PCM frames".to_owned());
        }
        while remaining > self.flac_frame_size {
            let after_full_block = remaining - self.flac_frame_size;
            let count = if after_full_block < 32 {
                remaining - 32
            } else {
                self.flac_frame_size
            };
            self.emit_flac_packet(packets, count)?;
            remaining = self.flac_frame.len() / channels;
        }
        if remaining > 0 {
            self.emit_flac_packet(packets, remaining)?;
        }
        Ok(())
    }

    fn emit_flac_packet(
        &mut self,
        packets: &mut Vec<WaveLibraryPacket>,
        frame_count: usize,
    ) -> Result<(), String> {
        if !(32..=self.flac_frame_size).contains(&frame_count) {
            return Err(format!(
                "streaming FLAC block has {frame_count} frames; expected 32..={}",
                self.flac_frame_size
            ));
        }
        let sample_count = frame_count.saturating_mul(self.flac_channels as usize);
        if self.flac_frame.len() < sample_count {
            return Err("streaming FLAC block is incomplete".to_owned());
        }
        let samples: Vec<i32> = self.flac_frame.drain(..sample_count).collect();
        let mut output = vec![0u8; samples.len().saturating_mul(8).saturating_add(4096)];
        let written = self
            .flac_encoder
            .as_mut()
            .ok_or_else(|| "streaming FLAC encoder is unavailable".to_owned())?
            .encode_i32(&samples, &mut output)?;
        if written == 0 {
            return Err("FLAC emitted an empty streaming packet".to_owned());
        }
        output.truncate(written);
        let start_frame = self.flac_frames;
        output = frame_library_packet(
            frame_header::EncodingFlag::FLAC,
            output,
            frame_count as u32,
            self.flac_sample_rate,
            self.flac_channels,
            24,
            self.flac_index_entries.len() as u64,
            start_frame,
        )?;
        self.flac_index_entries
            .push((self.flac_stream_bytes, start_frame));
        self.flac_digest.update(&output);
        self.flac_stream_bytes = self
            .flac_stream_bytes
            .checked_add(output.len() as u64)
            .ok_or_else(|| "streaming FLAC length overflowed".to_owned())?;
        packets.push(WaveLibraryPacket {
            bytes: output,
            start_frame,
            frame_count: frame_count as u32,
        });
        self.flac_frames += frame_count as u64;
        Ok(())
    }
}

#[cfg(all(feature = "opus", feature = "flac"))]
fn library_float_to_i16(sample: f32) -> i16 {
    let sample = if sample.is_finite() {
        sample.clamp(-1.0, 1.0)
    } else {
        0.0
    };
    let scaled = if sample < 0.0 {
        f64::from(sample) * 32_768.0
    } else {
        f64::from(sample) * 32_767.0
    };
    (scaled.round() as i32).clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

#[cfg(all(feature = "opus", feature = "flac"))]
fn library_float_to_s24(sample: f32) -> i32 {
    let sample = if sample.is_finite() {
        sample.clamp(-1.0, 1.0)
    } else {
        0.0
    };
    ((f64::from(sample) * 8_388_608.0).round() as i64).clamp(-8_388_608, 8_388_607) as i32
}

#[cfg(all(feature = "wav", feature = "opus", feature = "flac"))]
fn wave_library_opus_sample(sample: i16) -> i16 {
    if sample <= 0 {
        return sample;
    }
    // Match the existing canonical Float32 -> i16 conversion exactly. The
    // positive side uses 32767 while the negative side uses 32768.
    ((f64::from(sample) * 32767.0 / 32768.0).round() as i32).clamp(i16::MIN as i32, i16::MAX as i32)
        as i16
}

#[cfg(all(feature = "wav", feature = "opus", feature = "flac"))]
fn wave_library_flac_sample(sample: i16) -> i32 {
    (f64::from(sample) * 8_388_607.0 / 32_768.0).round() as i32
}

#[cfg(all(feature = "wav", feature = "opus", feature = "flac"))]
fn frame_library_packet(
    encoding: frame_header::EncodingFlag,
    payload: Vec<u8>,
    frame_count: u32,
    sample_rate: u32,
    channels: u8,
    bits_per_sample: u8,
    packet_sequence: u64,
    pts: u64,
) -> Result<Vec<u8>, String> {
    if payload.is_empty() || frame_count == 0 {
        return Err("A SoundKit library packet is empty".to_owned());
    }
    let header = frame_header::FrameHeaderV2::new(
        encoding,
        payload.len() as u32,
        frame_count,
        sample_rate,
        channels,
        bits_per_sample,
        frame_header::Endianness::LittleEndian,
        Some(packet_sequence),
        Some(pts),
        None,
    )
    .map_err(|error| error.to_string())?
    .with_packet_crc32(&payload)
    .map_err(|error| error.to_string())?;
    let mut framed = Vec::with_capacity(header.size() + payload.len());
    header
        .encode(&mut framed)
        .map_err(|error| error.to_string())?;
    framed.extend_from_slice(&payload);
    Ok(framed)
}

#[cfg(all(feature = "wav", feature = "opus", feature = "flac"))]
fn soundkit_frame_index(
    sample_rate: u32,
    duration_frames: u64,
    stream_byte_length: u64,
    entries: &[(u64, u64)],
) -> Result<Vec<u8>, String> {
    if sample_rate == 0 || duration_frames == 0 || stream_byte_length == 0 || entries.is_empty() {
        return Err("A SoundKit frame index cannot describe an empty stream".to_owned());
    }
    if entries[0] != (0, 0) {
        return Err("A SoundKit frame index must start at byte and frame zero".to_owned());
    }
    let mut previous = entries[0];
    for entry in entries.iter().copied().skip(1) {
        if entry.0 <= previous.0
            || entry.0 >= stream_byte_length
            || entry.1 <= previous.1
            || entry.1 >= duration_frames
        {
            return Err("SoundKit frame index entries are not ordered".to_owned());
        }
        previous = entry;
    }
    let mut index = Vec::with_capacity(32 + entries.len() * 16);
    index.extend_from_slice(b"SKIDX2\0\0");
    index.extend_from_slice(&1u16.to_le_bytes());
    index.extend_from_slice(&16u16.to_le_bytes());
    index.extend_from_slice(&sample_rate.to_le_bytes());
    index.extend_from_slice(&(entries.len() as u64).to_le_bytes());
    index.extend_from_slice(&duration_frames.to_le_bytes());
    for (byte_offset, start_frame) in entries {
        index.extend_from_slice(&byte_offset.to_le_bytes());
        index.extend_from_slice(&start_frame.to_le_bytes());
    }
    Ok(index)
}

#[cfg(all(feature = "wav", feature = "opus", feature = "flac"))]
fn wave_library_flac_frame_size(total_frames: u64, requested: usize) -> Result<usize, String> {
    let total = usize::try_from(total_frames)
        .map_err(|_| "WAV frame count exceeds this browser's address space".to_owned())?;
    let maximum = 32_767usize.min(total).min(requested);
    if maximum < 32 {
        return Err("WAV FLAC stream requires at least 32 PCM frames".to_owned());
    }
    (32..=maximum)
        .rev()
        .find(|candidate| {
            let final_block = total % candidate;
            final_block == 0 || final_block >= 32
        })
        .ok_or_else(|| "WAV FLAC stream could not select a valid frame size".to_owned())
}

#[cfg(all(feature = "wav", feature = "opus", feature = "flac"))]
fn wave_library_packets_to_js(packets: Vec<WaveLibraryPacket>) -> Result<Array, JsValue> {
    let output = Array::new();
    for packet in packets {
        let object = Object::new();
        Reflect::set(
            &object,
            &JsValue::from_str("data"),
            &Uint8Array::from(packet.bytes.as_slice()),
        )?;
        Reflect::set(
            &object,
            &JsValue::from_str("startFrame"),
            &JsValue::from_f64(packet.start_frame as f64),
        )?;
        Reflect::set(
            &object,
            &JsValue::from_str("frameCount"),
            &JsValue::from_f64(f64::from(packet.frame_count)),
        )?;
        output.push(&object);
    }
    Ok(output)
}

#[cfg(all(feature = "wav", feature = "opus", feature = "flac"))]
fn wave_library_result(
    opus_packets: Vec<WaveLibraryPacket>,
    flac_packets: Vec<WaveLibraryPacket>,
    done: bool,
    completed_frames: u64,
    total_frames: u64,
    opus_index: Option<Vec<u8>>,
    flac_index: Option<Vec<u8>>,
    source_identity: Option<String>,
    opus_identity: Option<String>,
    flac_identity: Option<String>,
) -> Result<JsValue, JsValue> {
    wave_library_batch_to_js(wave_library_batch(
        opus_packets,
        flac_packets,
        done,
        completed_frames,
        total_frames,
        opus_index,
        flac_index,
        source_identity,
        opus_identity,
        flac_identity,
    ))
}

#[cfg(all(feature = "wav", feature = "opus", feature = "flac"))]
fn wave_library_batch(
    opus_packets: Vec<WaveLibraryPacket>,
    flac_packets: Vec<WaveLibraryPacket>,
    done: bool,
    completed_frames: u64,
    frame_count: u64,
    opus_index: Option<Vec<u8>>,
    flac_index: Option<Vec<u8>>,
    source_identity: Option<String>,
    opus_identity: Option<String>,
    flac_identity: Option<String>,
) -> LibraryEncodeBatch {
    LibraryEncodeBatch {
        opus_packets,
        flac_packets,
        done,
        completed_frames,
        frame_count,
        sample_rate: 48_000,
        channels: 2,
        opus_index,
        flac_index,
        source_identity,
        opus_identity,
        flac_identity,
    }
}

#[cfg(all(feature = "wav", feature = "opus", feature = "flac"))]
fn wave_library_batch_to_js(batch: LibraryEncodeBatch) -> Result<JsValue, JsValue> {
    let object = Object::new();
    let opus_packets: JsValue = wave_library_packets_to_js(batch.opus_packets)?.into();
    let flac_packets: JsValue = wave_library_packets_to_js(batch.flac_packets)?.into();
    Reflect::set(&object, &JsValue::from_str("opusPackets"), &opus_packets)?;
    Reflect::set(&object, &JsValue::from_str("flacPackets"), &flac_packets)?;
    Reflect::set(
        &object,
        &JsValue::from_str("done"),
        &JsValue::from_bool(batch.done),
    )?;
    Reflect::set(
        &object,
        &JsValue::from_str("completedFrames"),
        &JsValue::from_f64(batch.completed_frames as f64),
    )?;
    Reflect::set(
        &object,
        &JsValue::from_str("frameCount"),
        &JsValue::from_f64(batch.frame_count as f64),
    )?;
    Reflect::set(
        &object,
        &JsValue::from_str("sampleRate"),
        &JsValue::from_f64(f64::from(batch.sample_rate)),
    )?;
    Reflect::set(
        &object,
        &JsValue::from_str("channels"),
        &JsValue::from_f64(f64::from(batch.channels)),
    )?;
    if let Some(index) = batch.opus_index {
        Reflect::set(
            &object,
            &JsValue::from_str("opusIndexBytes"),
            &Uint8Array::from(index.as_slice()),
        )?;
    }
    if let Some(index) = batch.flac_index {
        Reflect::set(
            &object,
            &JsValue::from_str("flacIndexBytes"),
            &Uint8Array::from(index.as_slice()),
        )?;
    }
    if let Some(identity) = batch.source_identity {
        Reflect::set(
            &object,
            &JsValue::from_str("sourceSha256"),
            &JsValue::from_str(&identity),
        )?;
    }
    if let Some(identity) = batch.opus_identity {
        Reflect::set(
            &object,
            &JsValue::from_str("opusSha256"),
            &JsValue::from_str(&identity),
        )?;
    }
    if let Some(identity) = batch.flac_identity {
        Reflect::set(
            &object,
            &JsValue::from_str("flacSha256"),
            &JsValue::from_str(&identity),
        )?;
    }
    Ok(object.into())
}

#[cfg(feature = "opus")]
#[wasm_bindgen]
impl WasmOpusDecoder {
    #[wasm_bindgen(constructor)]
    pub fn new(
        channels: usize,
        sample_rate: i32,
        frame_size: usize,
    ) -> Result<WasmOpusDecoder, JsValue> {
        let mut decoder =
            OpusDecoder::new_full(sample_rate as usize, channels).map_err(js_error)?;
        decoder.init().map_err(js_error)?;
        Self::with_decoder(decoder, channels, frame_size.max(MAX_OPUS_PACKET_FRAMES))
    }

    /// Uses the allocation-light CELT decoder for SoundKit-owned cache
    /// streams. It rejects SILK or hybrid packets.
    #[wasm_bindgen(js_name = forSoundKitStream)]
    pub fn for_soundkit_stream(
        channels: usize,
        sample_rate: i32,
        frame_size: usize,
    ) -> Result<WasmOpusDecoder, JsValue> {
        let mut decoder =
            OpusDecoder::new_celt_only(sample_rate as usize, channels).map_err(js_error)?;
        decoder.init().map_err(js_error)?;
        Self::with_decoder(decoder, channels, frame_size)
    }

    fn with_decoder(
        decoder: OpusDecoder,
        channels: usize,
        frame_size: usize,
    ) -> Result<WasmOpusDecoder, JsValue> {
        let output_len = frame_size.saturating_mul(channels).max(channels.max(1));
        Ok(Self {
            decoder,
            output: vec![0; output_len],
            decoded_size: 0,
        })
    }

    #[wasm_bindgen(js_name = dec_frame)]
    pub fn dec_frame(&mut self, packet: &[u8]) -> Result<WasmOpusDecodeResult, JsValue> {
        self.decode_reuse(packet)?;
        Ok(WasmOpusDecodeResult {
            output: self.output.clone(),
            decoded_size: self.decoded_size,
        })
    }

    #[wasm_bindgen(js_name = dec_frame_reuse)]
    pub fn dec_frame_reuse(&mut self, packet: &[u8]) -> Result<usize, JsValue> {
        self.decode_reuse(packet)
    }

    #[wasm_bindgen(getter, js_name = decodedSize)]
    pub fn decoded_size(&self) -> usize {
        self.decoded_size
    }

    #[wasm_bindgen(getter, js_name = outputPtr)]
    pub fn output_ptr(&self) -> usize {
        self.output.as_ptr() as usize
    }

    #[wasm_bindgen(getter, js_name = outputLen)]
    pub fn output_len(&self) -> usize {
        self.output.len()
    }

    pub fn destroy(self) {}

    fn decode_reuse(&mut self, packet: &[u8]) -> Result<usize, JsValue> {
        let samples_per_channel = self
            .decoder
            .decode_i16(packet, &mut self.output, false)
            .map_err(js_error)?;
        self.decoded_size = samples_per_channel;
        Ok(self.decoded_size)
    }
}

#[cfg(feature = "opus")]
#[wasm_bindgen]
impl WasmOpusDecodeResult {
    #[wasm_bindgen(getter, js_name = decodedSize)]
    pub fn decoded_size(&self) -> usize {
        self.decoded_size
    }

    #[wasm_bindgen(getter)]
    pub fn output(&self) -> Vec<i16> {
        self.output.clone()
    }
}

impl WasmSoundKitFrameDecoder {
    fn with_cipher(cipher: ChaCha20Poly1305PacketCipher) -> Self {
        Self {
            stream: SoundKitFrameStream::new(SoundKitFrameStreamOptions {
                cipher: Some(cipher),
                ..SoundKitFrameStreamOptions::default()
            }),
        }
    }
}

impl WasmMusicDecoder {
    fn push_frames(&mut self, bytes: &[u8]) -> Result<Vec<AudioData>, String> {
        validate_stream_input_chunk(bytes)?;
        let state = std::mem::replace(&mut self.state, DecoderState::Finished);
        match state {
            DecoderState::Detecting { mut buffer } => {
                let probe_bytes = (MAX_DETECTION_BYTES - buffer.len()).min(bytes.len());
                buffer.extend_from_slice(&bytes[..probe_bytes]);
                let new_bytes_collected = buffer.len();

                if new_bytes_collected < MIN_DETECTION_BYTES {
                    self.state = DecoderState::Detecting { buffer };
                    return Ok(Vec::new());
                }

                match detect_and_init_decoder(&buffer) {
                    Ok(mut decoder) => {
                        let mut frames = decoder.process(&buffer, &mut self.scratch)?;
                        if probe_bytes < bytes.len() {
                            frames
                                .extend(decoder.process(&bytes[probe_bytes..], &mut self.scratch)?);
                        }
                        self.state = DecoderState::Decoding { decoder };
                        Ok(frames)
                    }
                    Err(error) if new_bytes_collected < MAX_DETECTION_BYTES => {
                        self.state = DecoderState::Detecting { buffer };
                        if bytes.is_empty() {
                            self.state = DecoderState::Finished;
                            Err(error)
                        } else {
                            Ok(Vec::new())
                        }
                    }
                    Err(error) => {
                        self.state = DecoderState::Finished;
                        Err(error)
                    }
                }
            }
            DecoderState::Decoding { mut decoder } => {
                let frames = decoder.process(bytes, &mut self.scratch)?;
                self.state = DecoderState::Decoding { decoder };
                Ok(frames)
            }
            DecoderState::Finished => Err("decoder is already finished".to_string()),
        }
    }

    fn flush_frames(&mut self) -> Result<Vec<AudioData>, String> {
        let state = std::mem::replace(&mut self.state, DecoderState::Finished);
        match state {
            DecoderState::Detecting { buffer } => {
                let mut decoder = detect_and_init_decoder(&buffer)?;
                let mut frames = decoder.process(&buffer, &mut self.scratch)?;
                frames.extend(decoder.flush(&mut self.scratch)?);
                Ok(frames)
            }
            DecoderState::Decoding { mut decoder } => decoder.flush(&mut self.scratch),
            DecoderState::Finished => Ok(Vec::new()),
        }
    }
}

#[cfg(feature = "opus-debox")]
impl WasmOpusDeboxer {
    fn push_events(&mut self, bytes: &[u8]) -> Result<Vec<OpusDeboxEvent>, String> {
        validate_stream_input_chunk(bytes)?;
        let state = std::mem::replace(&mut self.state, OpusDeboxState::Finished);
        match state {
            OpusDeboxState::Detecting { mut buffer } => {
                let probe_bytes = (MAX_DETECTION_BYTES - buffer.len()).min(bytes.len());
                buffer.extend_from_slice(&bytes[..probe_bytes]);
                let new_bytes_collected = buffer.len();

                if new_bytes_collected < MIN_DETECTION_BYTES {
                    self.state = OpusDeboxState::Detecting { buffer };
                    return Ok(Vec::new());
                }

                match detect_and_init_opus_deboxer(&buffer) {
                    Ok(mut deboxer) => {
                        let mut events = process_opus_debox_state(&mut deboxer, &buffer)?;
                        if probe_bytes < bytes.len() {
                            events.extend(process_opus_debox_state(
                                &mut deboxer,
                                &bytes[probe_bytes..],
                            )?);
                        }
                        self.state = deboxer;
                        Ok(events)
                    }
                    Err(error) if new_bytes_collected < MAX_DETECTION_BYTES => {
                        self.state = OpusDeboxState::Detecting { buffer };
                        if bytes.is_empty() {
                            self.state = OpusDeboxState::Finished;
                            Err(error)
                        } else {
                            Ok(Vec::new())
                        }
                    }
                    Err(error) => {
                        self.state = OpusDeboxState::Finished;
                        Err(error)
                    }
                }
            }
            mut state @ (OpusDeboxState::Ogg(_)
            | OpusDeboxState::Raw(_)
            | OpusDeboxState::WebM(_)) => {
                let events = process_opus_debox_state(&mut state, bytes)?;
                self.state = state;
                Ok(events)
            }
            OpusDeboxState::Finished => Err("deboxer is already finished".to_string()),
        }
    }

    fn flush_events(&mut self) -> Result<Vec<OpusDeboxEvent>, String> {
        let state = std::mem::replace(&mut self.state, OpusDeboxState::Finished);
        match state {
            OpusDeboxState::Detecting { buffer } => {
                let mut deboxer = detect_and_init_opus_deboxer(&buffer)?;
                process_opus_debox_state(&mut deboxer, &buffer)
            }
            mut state @ (OpusDeboxState::Ogg(_)
            | OpusDeboxState::Raw(_)
            | OpusDeboxState::WebM(_)) => process_opus_debox_state(&mut state, &[]),
            OpusDeboxState::Finished => Ok(Vec::new()),
        }
    }
}

#[cfg(feature = "aac-debox")]
impl WasmAacDeboxer {
    fn push_events(&mut self, bytes: &[u8]) -> Result<Vec<AacDeboxEvent>, String> {
        validate_stream_input_chunk(bytes)?;
        let state = std::mem::replace(&mut self.state, AacDeboxState::Finished);
        match state {
            AacDeboxState::Detecting { mut buffer } => {
                let probe_bytes = (MAX_DETECTION_BYTES - buffer.len()).min(bytes.len());
                buffer.extend_from_slice(&bytes[..probe_bytes]);
                let new_bytes_collected = buffer.len();

                if new_bytes_collected < MIN_DETECTION_BYTES {
                    self.state = AacDeboxState::Detecting { buffer };
                    return Ok(Vec::new());
                }

                match detect_and_init_aac_deboxer(&buffer) {
                    Ok(mut deboxer) => {
                        let mut events = process_aac_debox_state(&mut deboxer, &buffer, false)?;
                        if probe_bytes < bytes.len() {
                            events.extend(process_aac_debox_state(
                                &mut deboxer,
                                &bytes[probe_bytes..],
                                false,
                            )?);
                        }
                        self.state = deboxer;
                        Ok(events)
                    }
                    Err(error) if new_bytes_collected < MAX_DETECTION_BYTES => {
                        self.state = AacDeboxState::Detecting { buffer };
                        if bytes.is_empty() {
                            self.state = AacDeboxState::Finished;
                            Err(error)
                        } else {
                            Ok(Vec::new())
                        }
                    }
                    Err(error) => {
                        self.state = AacDeboxState::Finished;
                        Err(error)
                    }
                }
            }
            mut state @ AacDeboxState::Mp4(_) => {
                let events = process_aac_debox_state(&mut state, bytes, false)?;
                self.state = state;
                Ok(events)
            }
            AacDeboxState::Finished => Err("deboxer is already finished".to_string()),
        }
    }

    fn flush_events(&mut self) -> Result<Vec<AacDeboxEvent>, String> {
        let state = std::mem::replace(&mut self.state, AacDeboxState::Finished);
        match state {
            AacDeboxState::Detecting { buffer } => {
                let mut deboxer = detect_and_init_aac_deboxer(&buffer)?;
                process_aac_debox_state(&mut deboxer, &buffer, true)
            }
            mut state @ AacDeboxState::Mp4(_) => process_aac_debox_state(&mut state, &[], true),
            AacDeboxState::Finished => Ok(Vec::new()),
        }
    }
}

impl FormatDecoder {
    fn process(
        &mut self,
        bytes: &[u8],
        scratch: &mut DecoderScratch,
    ) -> Result<Vec<AudioData>, String> {
        match self {
            #[cfg(feature = "aac")]
            FormatDecoder::Aac(decoder) => decode_i16_with_drain(
                decoder.as_mut(),
                bytes,
                scratch.i16_samples(),
                |decoder, samples, output| {
                    let (sample_rate, channels) = (decoder.sample_rate()?, decoder.channels()?);
                    Some(audio_data_i16(sample_rate, channels, &output[..samples]))
                },
            ),
            #[cfg(feature = "m4a")]
            FormatDecoder::M4a(decoder) => decode_i16_with_drain(
                decoder.as_mut(),
                bytes,
                scratch.i16_samples(),
                |decoder, samples, output| {
                    let (sample_rate, channels) = (decoder.sample_rate()?, decoder.channels()?);
                    Some(audio_data_i16(sample_rate, channels, &output[..samples]))
                },
            ),
            #[cfg(feature = "aiff")]
            FormatDecoder::Aiff(decoder) => {
                process_single_add_api(decoder.as_mut(), bytes, |d, data| d.add(data))
            }
            #[cfg(feature = "ac3")]
            FormatDecoder::Ac3(decoder) => {
                process_add_api(decoder.as_mut(), bytes, |d, data| d.add(data))
            }
            #[cfg(all(feature = "aac-lc", not(feature = "aac")))]
            FormatDecoder::AacLcAdts(decoder) => decoder.process(bytes, false),
            #[cfg(all(feature = "aac-lc", feature = "aac-debox", not(feature = "m4a")))]
            FormatDecoder::AacLcMp4(decoder) => decoder.process(bytes, false),
            #[cfg(feature = "flac")]
            FormatDecoder::Flac(decoder) => decode_i32_with_drain(
                decoder.as_mut(),
                bytes,
                scratch.i32_samples(),
                |decoder, samples, output| {
                    let (sample_rate, channels, bits) = (
                        decoder.sample_rate()?,
                        decoder.channels()?,
                        decoder.bits_per_sample()?,
                    );
                    Some(audio_data_i32(
                        sample_rate,
                        channels,
                        bits,
                        &output[..samples],
                    ))
                },
            ),
            #[cfg(feature = "mp3")]
            FormatDecoder::Mp3(decoder) => decode_i16_with_drain(
                decoder.as_mut(),
                bytes,
                scratch.i16_samples(),
                |decoder, samples, output| {
                    let (sample_rate, channels) = (decoder.sample_rate()?, decoder.channels()?);
                    Some(audio_data_i16(sample_rate, channels, &output[..samples]))
                },
            ),
            #[cfg(feature = "ogg-opus")]
            FormatDecoder::OggOpus(decoder) => {
                process_add_api(decoder.as_mut(), bytes, |d, data| d.add(data))
            }
            #[cfg(feature = "opus")]
            FormatDecoder::Opus(decoder) => {
                process_add_api(decoder.as_mut(), bytes, |d, data| d.add(data))
            }
            FormatDecoder::RawPcm(decoder) => {
                let mut frames = Vec::new();
                if let Some(frame) = decoder.add(bytes)? {
                    frames.push(frame);
                }
                Ok(frames)
            }
            #[cfg(feature = "vorbis")]
            FormatDecoder::Vorbis(decoder) => {
                process_add_api(decoder.as_mut(), bytes, |d, data| d.add(data))
            }
            #[cfg(feature = "webm")]
            FormatDecoder::WebM(decoder) => {
                process_add_api(decoder.as_mut(), bytes, |d, data| d.add(data))
            }
            FormatDecoder::Wav(decoder) => {
                process_add_api(decoder.as_mut(), bytes, |d, data| d.add(data))
            }
        }
    }

    fn flush(&mut self, scratch: &mut DecoderScratch) -> Result<Vec<AudioData>, String> {
        match self {
            #[cfg(all(feature = "aac-lc", not(feature = "aac")))]
            FormatDecoder::AacLcAdts(decoder) => decoder.process(&[], true),
            #[cfg(all(feature = "aac-lc", feature = "aac-debox", not(feature = "m4a")))]
            FormatDecoder::AacLcMp4(decoder) => decoder.process(&[], true),
            #[cfg(feature = "aiff")]
            FormatDecoder::Aiff(decoder) => {
                process_add_api(decoder.as_mut(), &[], |d, data| d.add(data))
            }
            FormatDecoder::RawPcm(decoder) => {
                decoder.flush().map(|frame| frame.into_iter().collect())
            }
            _ => self.process(&[], scratch),
        }
    }
}

fn process_add_api<D, F>(
    decoder: &mut D,
    bytes: &[u8],
    mut add: F,
) -> Result<Vec<AudioData>, String>
where
    F: FnMut(&mut D, &[u8]) -> Result<Option<AudioData>, String>,
{
    let mut frames = Vec::new();
    if let Some(frame) = add(decoder, bytes)? {
        frames.push(frame);
    }

    while let Some(frame) = add(decoder, &[])? {
        frames.push(frame);
    }

    Ok(frames)
}

#[cfg(feature = "aiff")]
fn process_single_add_api<D, F>(
    decoder: &mut D,
    bytes: &[u8],
    mut add: F,
) -> Result<Vec<AudioData>, String>
where
    F: FnMut(&mut D, &[u8]) -> Result<Option<AudioData>, String>,
{
    let mut frames = Vec::new();
    if let Some(frame) = add(decoder, bytes)? {
        frames.push(frame);
    }
    Ok(frames)
}

#[cfg(any(feature = "aac", feature = "m4a", feature = "mp3"))]
fn decode_i16_with_drain<D, F>(
    decoder: &mut D,
    bytes: &[u8],
    output: &mut [i16],
    frame: F,
) -> Result<Vec<AudioData>, String>
where
    D: Decoder,
    F: Fn(&D, usize, &[i16]) -> Option<AudioData>,
{
    let mut frames = Vec::new();
    let samples = decoder.decode_i16(bytes, output, false)?;
    if samples > 0 {
        if let Some(audio) = frame(decoder, samples, &output) {
            frames.push(audio);
        }
    }

    loop {
        let samples = decoder.decode_i16(&[], output, false)?;
        if samples == 0 {
            break;
        }
        if let Some(audio) = frame(decoder, samples, &output) {
            frames.push(audio);
        }
    }

    Ok(frames)
}

#[cfg(feature = "flac")]
fn decode_i32_with_drain<D, F>(
    decoder: &mut D,
    bytes: &[u8],
    output: &mut [i32],
    frame: F,
) -> Result<Vec<AudioData>, String>
where
    D: Decoder,
    F: Fn(&D, usize, &[i32]) -> Option<AudioData>,
{
    let mut frames = Vec::new();
    let samples = decoder.decode_i32(bytes, output, false)?;
    if samples > 0 {
        if let Some(audio) = frame(decoder, samples, &output) {
            frames.push(audio);
        }
    }

    loop {
        let samples = decoder.decode_i32(&[], output, false)?;
        if samples == 0 {
            break;
        }
        if let Some(audio) = frame(decoder, samples, &output) {
            frames.push(audio);
        }
    }

    Ok(frames)
}

#[cfg(any(feature = "aac", feature = "m4a", feature = "mp3"))]
fn audio_data_i16(sample_rate: u32, channels: u8, samples: &[i16]) -> AudioData {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for sample in samples {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }

    AudioData::new(
        16,
        channels,
        sample_rate,
        bytes,
        frame_header::EncodingFlag::PCMSigned,
        frame_header::Endianness::LittleEndian,
    )
}

#[cfg(feature = "flac")]
fn audio_data_i32(sample_rate: u32, channels: u8, bits: u8, samples: &[i32]) -> AudioData {
    let bytes_per_sample = bits.div_ceil(8) as usize;
    let mut bytes = Vec::with_capacity(samples.len() * bytes_per_sample);

    match bits {
        1..=8 => {
            for sample in samples {
                bytes.push((*sample + 128) as u8);
            }
        }
        9..=16 => {
            for sample in samples {
                bytes.extend_from_slice(&(*sample as i16).to_le_bytes());
            }
        }
        17..=24 => {
            for sample in samples {
                let le = sample.to_le_bytes();
                bytes.extend_from_slice(&le[..3]);
            }
        }
        _ => {
            for sample in samples {
                bytes.extend_from_slice(&sample.to_le_bytes());
            }
        }
    }

    AudioData::new(
        bits,
        channels,
        sample_rate,
        bytes,
        frame_header::EncodingFlag::PCMSigned,
        frame_header::Endianness::LittleEndian,
    )
}

#[cfg(feature = "opus-debox")]
#[derive(Clone, Debug)]
enum OpusDeboxEvent {
    Config {
        container: &'static str,
        sample_rate: u32,
        channels: u8,
        pre_skip: u16,
        output_gain: i16,
        mapping_family: u8,
        codec_private: Vec<u8>,
    },
    Tags {
        container: &'static str,
        data: Vec<u8>,
    },
    Packet {
        container: &'static str,
        data: Vec<u8>,
        timecode: Option<i16>,
    },
}

#[cfg(feature = "aac-debox")]
#[derive(Clone, Debug)]
enum AacDeboxEvent {
    Config {
        container: &'static str,
        sample_rate: u32,
        channels: u8,
        track_id: u32,
        sample_count: u32,
    },
    Packet {
        container: &'static str,
        data: Vec<u8>,
        raw_data: Vec<u8>,
        sample_id: u32,
        start_time: u64,
        duration: u32,
        rendering_offset: i32,
        is_sync: bool,
    },
}

#[cfg(feature = "opus-debox")]
struct RawOpusDeboxer {
    buffer: Vec<u8>,
    header_parsed: bool,
}

#[cfg(feature = "opus-debox")]
impl RawOpusDeboxer {
    fn new() -> Self {
        Self {
            buffer: Vec::new(),
            header_parsed: false,
        }
    }

    fn add(&mut self, data: &[u8]) -> Result<Vec<OpusDeboxEvent>, String> {
        self.buffer.extend_from_slice(data);
        let mut events = Vec::new();

        if !self.header_parsed {
            if self.buffer.len() < 19 {
                return Ok(events);
            }
            if !self.buffer.starts_with(b"OpusHead") {
                return Err("Invalid raw Opus stream: missing OpusHead".to_string());
            }

            let head = self.buffer[..19].to_vec();
            events.push(opus_config_event("raw", &head, None, None)?);
            self.buffer.drain(..19);
            self.header_parsed = true;
        }

        while self.buffer.len() >= 2 {
            let packet_len = u16::from_le_bytes([self.buffer[0], self.buffer[1]]) as usize;
            if packet_len == 0 || self.buffer.len() < 2 + packet_len {
                break;
            }

            let packet = self.buffer[2..2 + packet_len].to_vec();
            self.buffer.drain(..2 + packet_len);
            events.push(OpusDeboxEvent::Packet {
                container: "raw",
                data: packet,
                timecode: None,
            });
        }

        Ok(events)
    }
}

#[cfg(feature = "aac-debox")]
fn process_aac_debox_state(
    state: &mut AacDeboxState,
    bytes: &[u8],
    finalizing: bool,
) -> Result<Vec<AacDeboxEvent>, String> {
    match state {
        AacDeboxState::Mp4(demuxer) => {
            let demux_events = if finalizing {
                if !bytes.is_empty() {
                    let mut events = demuxer.add(bytes)?;
                    events.extend(demuxer.finish()?);
                    events
                } else {
                    demuxer.finish()?
                }
            } else {
                demuxer.add(bytes)?
            };

            let mut events = Vec::new();
            for event in demux_events {
                match event {
                    AacMp4DemuxEvent::Config(config) => {
                        events.push(AacDeboxEvent::Config {
                            container: "mp4",
                            sample_rate: config.sample_rate,
                            channels: config.channels,
                            track_id: config.track_id,
                            sample_count: config.sample_count,
                        });
                    }
                    AacMp4DemuxEvent::Frame(frame) => {
                        events.push(AacDeboxEvent::Packet {
                            container: "mp4",
                            data: frame.adts,
                            raw_data: frame.raw,
                            sample_id: frame.sample_id,
                            start_time: frame.start_time,
                            duration: frame.duration,
                            rendering_offset: frame.rendering_offset,
                            is_sync: frame.is_sync,
                        });
                    }
                }
            }
            Ok(events)
        }
        AacDeboxState::Detecting { .. } => Ok(Vec::new()),
        AacDeboxState::Finished => Err("deboxer is already finished".to_string()),
    }
}

#[cfg(feature = "aac-debox")]
fn aac_deboxer_for_format(format: &str) -> Result<AacDeboxState, String> {
    match normalize_format(format).as_str() {
        "m4a" | "mp4" | "aac-mp4" | "mp4-aac" => {
            let mut demuxer = AacMp4Demuxer::new();
            demuxer.init()?;
            Ok(AacDeboxState::Mp4(demuxer))
        }
        other => Err(format!("unsupported AAC debox format: {other}")),
    }
}

#[cfg(all(feature = "aac-debox", feature = "detect"))]
fn detect_and_init_aac_deboxer(bytes: &[u8]) -> Result<AacDeboxState, String> {
    match detect_audio(bytes) {
        AudioType::M4A => aac_deboxer_for_format("m4a"),
        detected => Err(format!(
            "unsupported or disabled detected AAC container: {detected:?}"
        )),
    }
}

#[cfg(all(feature = "aac-debox", not(feature = "detect")))]
fn detect_and_init_aac_deboxer(_bytes: &[u8]) -> Result<AacDeboxState, String> {
    Err("automatic detection is disabled".to_string())
}

#[cfg(feature = "opus-debox")]
fn process_opus_debox_state(
    state: &mut OpusDeboxState,
    bytes: &[u8],
) -> Result<Vec<OpusDeboxEvent>, String> {
    match state {
        OpusDeboxState::Ogg(demuxer) => {
            let mut events = Vec::new();
            for event in demuxer.add(bytes)? {
                match event {
                    OggOpusDemuxEvent::Config(config) => {
                        events.push(OpusDeboxEvent::Config {
                            container: "ogg",
                            sample_rate: config.sample_rate,
                            channels: config.channels,
                            pre_skip: config.pre_skip,
                            output_gain: config.output_gain,
                            mapping_family: config.mapping_family,
                            codec_private: config.head,
                        });
                    }
                    OggOpusDemuxEvent::Tags(data) => {
                        events.push(OpusDeboxEvent::Tags {
                            container: "ogg",
                            data,
                        });
                    }
                    OggOpusDemuxEvent::Packet(data) => {
                        events.push(OpusDeboxEvent::Packet {
                            container: "ogg",
                            data,
                            timecode: None,
                        });
                    }
                }
            }
            Ok(events)
        }
        OpusDeboxState::Raw(deboxer) => deboxer.add(bytes),
        OpusDeboxState::WebM(demuxer) => {
            let mut events = Vec::new();
            for event in demuxer.add(bytes)? {
                match event {
                    WebmOpusDemuxEvent::Config(config) => {
                        events.push(OpusDeboxEvent::Config {
                            container: "webm",
                            sample_rate: config.sample_rate,
                            channels: config.channels,
                            pre_skip: config.pre_skip,
                            output_gain: config.output_gain,
                            mapping_family: config.mapping_family,
                            codec_private: config.codec_private,
                        });
                    }
                    WebmOpusDemuxEvent::Packet { data, timecode } => {
                        events.push(OpusDeboxEvent::Packet {
                            container: "webm",
                            data,
                            timecode: Some(timecode),
                        });
                    }
                }
            }
            Ok(events)
        }
        OpusDeboxState::Detecting { .. } => Ok(Vec::new()),
        OpusDeboxState::Finished => Err("deboxer is already finished".to_string()),
    }
}

#[cfg(feature = "opus-debox")]
fn opus_deboxer_for_format(format: &str) -> Result<OpusDeboxState, String> {
    match normalize_format(format).as_str() {
        "ogg" | "ogg-opus" | "opus-ogg" => {
            let mut demuxer = OggOpusDemuxer::new();
            demuxer.init()?;
            Ok(OpusDeboxState::Ogg(demuxer))
        }
        "opus" | "raw-opus" => Ok(OpusDeboxState::Raw(RawOpusDeboxer::new())),
        "webm" | "webm-opus" => {
            let mut demuxer = WebmOpusDemuxer::new();
            demuxer.init()?;
            Ok(OpusDeboxState::WebM(demuxer))
        }
        other => Err(format!("unsupported Opus debox format: {other}")),
    }
}

#[cfg(all(feature = "opus-debox", feature = "detect"))]
fn detect_and_init_opus_deboxer(bytes: &[u8]) -> Result<OpusDeboxState, String> {
    match detect_audio(bytes) {
        AudioType::OggOpus => opus_deboxer_for_format("ogg-opus"),
        AudioType::Opus => opus_deboxer_for_format("opus"),
        AudioType::WebM => opus_deboxer_for_format("webm"),
        detected => Err(format!(
            "unsupported or disabled detected Opus container: {detected:?}"
        )),
    }
}

#[cfg(all(feature = "opus-debox", not(feature = "detect")))]
fn detect_and_init_opus_deboxer(_bytes: &[u8]) -> Result<OpusDeboxState, String> {
    Err("automatic detection is disabled".to_string())
}

#[cfg(feature = "opus-debox")]
fn opus_config_event(
    container: &'static str,
    opus_head: &[u8],
    sample_rate_override: Option<u32>,
    channels_override: Option<u8>,
) -> Result<OpusDeboxEvent, String> {
    if opus_head.len() < 19 || !opus_head.starts_with(b"OpusHead") {
        return Err("Invalid OpusHead data".to_string());
    }

    let mut sample_rate =
        u32::from_le_bytes([opus_head[12], opus_head[13], opus_head[14], opus_head[15]]);
    if sample_rate == 0 {
        sample_rate = 48_000;
    }

    Ok(OpusDeboxEvent::Config {
        container,
        sample_rate: sample_rate_override.unwrap_or(sample_rate),
        channels: channels_override.unwrap_or(opus_head[9]),
        pre_skip: u16::from_le_bytes([opus_head[10], opus_head[11]]),
        output_gain: i16::from_le_bytes([opus_head[16], opus_head[17]]),
        mapping_family: opus_head[18],
        codec_private: opus_head.to_vec(),
    })
}

fn decoder_for_format(format: &str) -> Result<FormatDecoder, String> {
    match normalize_format(format).as_str() {
        #[cfg(feature = "aac")]
        "aac" | "adts" => {
            let mut decoder = AacDecoder::new();
            decoder.init()?;
            Ok(FormatDecoder::Aac(Box::new(decoder)))
        }
        #[cfg(all(feature = "aac-lc", not(feature = "aac")))]
        "aac" | "adts" | "aac-lc" => {
            Ok(FormatDecoder::AacLcAdts(Box::new(AacLcAdtsDecoder::new())))
        }
        #[cfg(feature = "m4a")]
        "m4a" | "mp4" | "aac-mp4" => {
            let mut decoder = AacDecoderMp4::new();
            decoder.init()?;
            Ok(FormatDecoder::M4a(Box::new(decoder)))
        }
        #[cfg(all(feature = "aac-lc", feature = "aac-debox", not(feature = "m4a")))]
        "m4a" | "mp4" | "aac-mp4" => Ok(FormatDecoder::AacLcMp4(Box::new(AacLcMp4Decoder::new()?))),
        #[cfg(feature = "aiff")]
        "aiff" | "aifc" => {
            let mut decoder = AiffDecoder::new();
            decoder.init()?;
            Ok(FormatDecoder::Aiff(Box::new(decoder)))
        }
        #[cfg(feature = "ac3")]
        "ac3" | "ac-3" => Ok(FormatDecoder::Ac3(Box::new(Ac3Decoder::try_new()?))),
        #[cfg(feature = "alac")]
        "alac" | "caf-alac" => Err(SEEKABLE_ALAC_REQUIRED.to_string()),
        #[cfg(feature = "flac")]
        "flac" => {
            let mut decoder = FlacDecoderClaxon::new();
            decoder.init()?;
            Ok(FormatDecoder::Flac(Box::new(decoder)))
        }
        #[cfg(feature = "mp3")]
        "mp3" => Ok(FormatDecoder::Mp3(Box::new(Mp3Decoder::new()))),
        #[cfg(feature = "ogg-opus")]
        "ogg-opus" | "opus-ogg" => {
            let mut decoder = OggOpusDecoder::new();
            decoder.init()?;
            Ok(FormatDecoder::OggOpus(Box::new(decoder)))
        }
        #[cfg(feature = "opus")]
        "opus" => {
            let mut decoder = OpusStreamDecoder::new();
            decoder.init()?;
            Ok(FormatDecoder::Opus(Box::new(decoder)))
        }
        #[cfg(feature = "vorbis")]
        "ogg" | "ogg-vorbis" | "vorbis" => {
            let mut decoder = VorbisDecoder::new();
            decoder.init()?;
            Ok(FormatDecoder::Vorbis(Box::new(decoder)))
        }
        #[cfg(feature = "webm")]
        "webm" => {
            let mut decoder = WebmDecoder::new();
            decoder.init()?;
            Ok(FormatDecoder::WebM(Box::new(decoder)))
        }
        "wav" | "wave" => Ok(FormatDecoder::Wav(Box::new(WavStreamProcessor::new()))),
        other => Err(format!("unsupported or disabled format: {other}")),
    }
}

#[cfg(feature = "detect")]
fn detect_and_init_decoder(bytes: &[u8]) -> Result<FormatDecoder, String> {
    match detect_audio(bytes) {
        #[cfg(feature = "mp3")]
        AudioType::MP3 => decoder_for_format("mp3"),
        #[cfg(feature = "aac")]
        AudioType::AAC => decoder_for_format("aac"),
        #[cfg(all(feature = "aac-lc", not(feature = "aac")))]
        AudioType::AAC => decoder_for_format("aac"),
        #[cfg(feature = "m4a")]
        AudioType::M4A => decoder_for_format("m4a"),
        #[cfg(all(feature = "aac-lc", feature = "aac-debox", not(feature = "m4a")))]
        AudioType::M4A => decoder_for_format("m4a"),
        #[cfg(feature = "flac")]
        AudioType::FLAC => decoder_for_format("flac"),
        #[cfg(feature = "opus")]
        AudioType::Opus => decoder_for_format("opus"),
        #[cfg(feature = "ogg-opus")]
        AudioType::OggOpus => decoder_for_format("ogg-opus"),
        #[cfg(feature = "vorbis")]
        AudioType::OggVorbis => decoder_for_format("ogg-vorbis"),
        #[cfg(feature = "webm")]
        AudioType::WebM => decoder_for_format("webm"),
        AudioType::Wav => decoder_for_format("wav"),
        #[cfg(feature = "alac")]
        AudioType::ALAC => decoder_for_format("alac"),
        #[cfg(feature = "aiff")]
        AudioType::AIFF => decoder_for_format("aiff"),
        #[cfg(feature = "ac3")]
        AudioType::AC3 => decoder_for_format("ac3"),
        detected => Err(format!(
            "unsupported or disabled detected format: {detected:?}"
        )),
    }
}

#[cfg(not(feature = "detect"))]
fn detect_and_init_decoder(_bytes: &[u8]) -> Result<FormatDecoder, String> {
    Err("automatic detection is disabled".to_string())
}

fn normalize_format(format: &str) -> String {
    format.trim().to_ascii_lowercase().replace('_', "-")
}

fn audio_frames_to_js(frames: Vec<AudioData>) -> Result<Array, JsValue> {
    let array = Array::new();
    for frame in frames {
        array.push(&audio_frame_to_js(&frame)?);
    }
    Ok(array)
}

fn canonical_decode_batch_to_js(batch: CanonicalDecodeBatch) -> Result<JsValue, JsValue> {
    let object = Object::new();
    let blocks = Array::new();
    for block in batch.blocks {
        let item = Object::new();
        Reflect::set(
            &item,
            &JsValue::from_str("startFrame"),
            &JsValue::from_f64(block.start_frame as f64),
        )?;
        Reflect::set(
            &item,
            &JsValue::from_str("frameCount"),
            &JsValue::from_f64(f64::from(block.frame_count)),
        )?;
        Reflect::set(
            &item,
            &JsValue::from_str("pcmS16Planar"),
            &Uint8Array::from(block.pcm_s16_planar.as_slice()),
        )?;
        blocks.push(&item);
    }
    Reflect::set(&object, &JsValue::from_str("blocks"), &blocks)?;
    Reflect::set(
        &object,
        &JsValue::from_str("done"),
        &JsValue::from_bool(batch.done),
    )?;
    Reflect::set(
        &object,
        &JsValue::from_str("frameCount"),
        &JsValue::from_f64(batch.frame_count as f64),
    )?;
    Reflect::set(
        &object,
        &JsValue::from_str("sampleRate"),
        &JsValue::from_f64(48_000.0),
    )?;
    Reflect::set(
        &object,
        &JsValue::from_str("channels"),
        &JsValue::from_f64(2.0),
    )?;
    Reflect::set(
        &object,
        &JsValue::from_str("sourceSampleRate"),
        &JsValue::from_f64(f64::from(batch.source_sample_rate)),
    )?;
    Reflect::set(
        &object,
        &JsValue::from_str("sourceChannels"),
        &JsValue::from_f64(f64::from(batch.source_channels)),
    )?;
    Reflect::set(
        &object,
        &JsValue::from_str("sourceFrameCount"),
        &JsValue::from_f64(batch.source_frame_count as f64),
    )?;
    if let Some(identity) = batch.source_identity {
        Reflect::set(
            &object,
            &JsValue::from_str("sourceSha256"),
            &JsValue::from_str(&identity),
        )?;
    }
    Ok(object.into())
}

#[cfg(any(feature = "video", feature = "audio-demux"))]
fn finite_i64(value: f64) -> Option<i64> {
    value.is_finite().then(|| value.round() as i64)
}

fn finite_u64(value: f64) -> Option<u64> {
    (value.is_finite() && value >= 0.0).then(|| value.round() as u64)
}

#[cfg(feature = "video")]
fn export_video_frames(frames: Vec<VideoFrame>) -> Result<Array, JsValue> {
    let output = Array::new();
    for frame in frames {
        let object = Object::new();
        Reflect::set(&object, &"width".into(), &(frame.width as f64).into())?;
        Reflect::set(&object, &"height".into(), &(frame.height as f64).into())?;
        Reflect::set(
            &object,
            &"bitDepth".into(),
            &(frame.bit_depth as f64).into(),
        )?;
        Reflect::set(
            &object,
            &"chromaSampling".into(),
            &frame.chroma_sampling.as_str().into(),
        )?;
        Reflect::set(
            &object,
            &"colorModel".into(),
            &frame.color_model.as_str().into(),
        )?;
        Reflect::set(
            &object,
            &"hasAlpha".into(),
            &JsValue::from_bool(frame.has_alpha),
        )?;
        Reflect::set(
            &object,
            &"pts".into(),
            &frame
                .pts
                .map(|value| JsValue::from_f64(value as f64))
                .unwrap_or(JsValue::NULL),
        )?;
        Reflect::set(
            &object,
            &"duration".into(),
            &frame
                .duration
                .map(|value| JsValue::from_f64(value as f64))
                .unwrap_or(JsValue::NULL),
        )?;
        let planes = Array::new();
        for plane in frame.planes {
            let plane_object = Object::new();
            Reflect::set(&plane_object, &"width".into(), &(plane.width as f64).into())?;
            Reflect::set(
                &plane_object,
                &"height".into(),
                &(plane.height as f64).into(),
            )?;
            Reflect::set(
                &plane_object,
                &"stride".into(),
                &(plane.stride as f64).into(),
            )?;
            Reflect::set(
                &plane_object,
                &"data".into(),
                &Uint8Array::from(plane.data.as_slice()).into(),
            )?;
            planes.push(&plane_object);
        }
        Reflect::set(&object, &"planes".into(), &planes)?;
        output.push(&object);
    }
    Ok(output)
}

fn audio_frame_to_js(frame: &AudioData) -> Result<JsValue, JsValue> {
    let object = Object::new();
    Reflect::set(
        &object,
        &JsValue::from_str("sampleRate"),
        &JsValue::from_f64(frame.sampling_rate() as f64),
    )?;
    Reflect::set(
        &object,
        &JsValue::from_str("channels"),
        &JsValue::from_f64(frame.channel_count() as f64),
    )?;
    Reflect::set(
        &object,
        &JsValue::from_str("bitsPerSample"),
        &JsValue::from_f64(frame.bits_per_sample() as f64),
    )?;
    Reflect::set(
        &object,
        &JsValue::from_str("data"),
        &Uint8Array::from(frame.data().as_slice()).into(),
    )?;
    Ok(object.into())
}

fn soundkit_frames_to_js(frames: Vec<SoundKitFrame>) -> Result<Array, JsValue> {
    let array = Array::new();
    for frame in frames {
        array.push(&soundkit_frame_to_js(&frame)?);
    }
    Ok(array)
}

fn soundkit_frame_to_js(frame: &SoundKitFrame) -> Result<JsValue, JsValue> {
    let object = Object::new();
    let header = &frame.header;

    let header_object = Object::new();
    Reflect::set(
        &header_object,
        &JsValue::from_str("encoding"),
        &JsValue::from_str(soundkit_encoding_name(header.encoding())),
    )?;
    Reflect::set(
        &header_object,
        &JsValue::from_str("encodingCode"),
        &JsValue::from_f64(soundkit_encoding_code(header.encoding()) as f64),
    )?;
    Reflect::set(
        &header_object,
        &JsValue::from_str("channels"),
        &JsValue::from_f64(header.channels() as f64),
    )?;
    Reflect::set(
        &header_object,
        &JsValue::from_str("sampleRate"),
        &JsValue::from_f64(header.sample_rate() as f64),
    )?;
    Reflect::set(
        &header_object,
        &JsValue::from_str("frameCount"),
        &JsValue::from_f64(header.frame_count() as f64),
    )?;
    Reflect::set(
        &header_object,
        &JsValue::from_str("bitsPerSample"),
        &JsValue::from_f64(header.bits_per_sample() as f64),
    )?;
    Reflect::set(
        &header_object,
        &JsValue::from_str("endianness"),
        &JsValue::from_str(soundkit_endianness_name(header.endianness())),
    )?;
    Reflect::set(
        &header_object,
        &JsValue::from_str("packetFlags"),
        &JsValue::from_f64(header.packet_flags() as f64),
    )?;
    Reflect::set(
        &header_object,
        &JsValue::from_str("payloadSize"),
        &JsValue::from_f64(frame.payload.len() as f64),
    )?;
    Reflect::set(
        &header_object,
        &JsValue::from_str("encryptedPayloadSize"),
        &JsValue::from_f64(frame.encrypted_payload_size as f64),
    )?;
    Reflect::set(
        &header_object,
        &JsValue::from_str("headerBytes"),
        &JsValue::from_f64(frame.encoded_header_bytes.len() as f64),
    )?;
    if let Some(packet_crc32) = header.packet_crc32_value() {
        Reflect::set(
            &header_object,
            &JsValue::from_str("packetCrc32"),
            &JsValue::from_f64(packet_crc32 as f64),
        )?;
    }
    set_optional_u64_string(&header_object, "id", header.id())?;
    set_optional_u64_string(&header_object, "pts", header.pts())?;

    let header_value: JsValue = header_object.into();
    Reflect::set(&object, &JsValue::from_str("header"), &header_value)?;
    Reflect::set(
        &object,
        &JsValue::from_str("data"),
        &Uint8Array::from(frame.payload.as_slice()).into(),
    )?;
    Reflect::set(
        &object,
        &JsValue::from_str("encrypted"),
        &JsValue::from_bool(frame.encrypted),
    )?;
    Reflect::set(
        &object,
        &JsValue::from_str("payloadSize"),
        &JsValue::from_f64(frame.payload.len() as f64),
    )?;
    Reflect::set(
        &object,
        &JsValue::from_str("encryptedPayloadSize"),
        &JsValue::from_f64(frame.encrypted_payload_size as f64),
    )?;
    set_optional_u64_string(&object, "trackId", header.id())?;
    set_optional_u64_string(&object, "id", header.id())?;
    set_optional_u64_string(&object, "pts", header.pts())?;

    Ok(object.into())
}

fn soundkit_encoding_from_code(code: u8) -> Result<frame_header::EncodingFlag, JsValue> {
    match code {
        0 => Ok(frame_header::EncodingFlag::PCMSigned),
        1 => Ok(frame_header::EncodingFlag::PCMFloat),
        2 => Ok(frame_header::EncodingFlag::Opus),
        3 => Ok(frame_header::EncodingFlag::FLAC),
        4 => Ok(frame_header::EncodingFlag::AAC),
        5 => Ok(frame_header::EncodingFlag::H264),
        _ => Err(js_error(format!(
            "Unsupported SoundKit v2 encoding code: {code}"
        ))),
    }
}

fn soundkit_frame_header_v2(
    encoding: u8,
    payload_size: u32,
    sample_size: u32,
    sample_rate: u32,
    channels: u8,
    bits_per_sample: u8,
    pts: f64,
) -> Result<frame_header::FrameHeaderV2, JsValue> {
    if payload_size == 0 {
        return Err(js_error(
            "SoundKit v2 frame requires payload_size > 0.".to_string(),
        ));
    }
    if sample_size == 0 {
        return Err(js_error(
            "SoundKit v2 frame requires sample_size > 0.".to_string(),
        ));
    }
    let pts_value = if pts.is_finite() && pts >= 0.0 {
        Some(pts.round() as u64)
    } else {
        None
    };

    frame_header::FrameHeaderV2::new(
        soundkit_encoding_from_code(encoding)?,
        payload_size,
        sample_size,
        sample_rate,
        channels,
        bits_per_sample,
        frame_header::Endianness::LittleEndian,
        None,
        pts_value,
        None,
    )
    .map_err(|error| js_error(format!("build SoundKit v2 header failed: {error}")))
}

fn soundkit_encoding_code(encoding: &frame_header::EncodingFlag) -> u8 {
    match encoding {
        frame_header::EncodingFlag::PCMSigned => 0,
        frame_header::EncodingFlag::PCMFloat => 1,
        frame_header::EncodingFlag::Opus => 2,
        frame_header::EncodingFlag::FLAC => 3,
        frame_header::EncodingFlag::AAC => 4,
        frame_header::EncodingFlag::H264 => 5,
    }
}

fn soundkit_encoding_name(encoding: &frame_header::EncodingFlag) -> &'static str {
    match encoding {
        frame_header::EncodingFlag::PCMSigned => "PCMSigned",
        frame_header::EncodingFlag::PCMFloat => "PCMFloat",
        frame_header::EncodingFlag::Opus => "Opus",
        frame_header::EncodingFlag::FLAC => "FLAC",
        frame_header::EncodingFlag::AAC => "AAC",
        frame_header::EncodingFlag::H264 => "H264",
    }
}

fn soundkit_endianness_name(endianness: &frame_header::Endianness) -> &'static str {
    match endianness {
        frame_header::Endianness::LittleEndian => "LittleEndian",
        frame_header::Endianness::BigEndian => "BigEndian",
    }
}

fn set_optional_u64_string(object: &Object, key: &str, value: Option<u64>) -> Result<(), JsValue> {
    if let Some(value) = value {
        Reflect::set(
            object,
            &JsValue::from_str(key),
            &JsValue::from_str(&value.to_string()),
        )?;
    }
    Ok(())
}

#[cfg(feature = "audio-demux")]
fn audio_demux_events_to_js(events: Vec<AudioDemuxEvent>) -> Result<Array, JsValue> {
    let array = Array::new();
    for event in events {
        array.push(&audio_demux_event_to_js(event)?);
    }
    Ok(array)
}

#[cfg(any(feature = "audio-demux", feature = "webm"))]
const JS_MAX_SAFE_INTEGER: u64 = 9_007_199_254_740_991;

#[cfg(any(feature = "audio-demux", feature = "webm"))]
fn js_safe_u64(value: u64, field: &str) -> Result<JsValue, JsValue> {
    if value > JS_MAX_SAFE_INTEGER {
        return Err(js_error(format!(
            "{field} {value} exceeds JavaScript's exact integer range"
        )));
    }
    Ok(JsValue::from_f64(value as f64))
}

#[cfg(any(feature = "audio-demux", feature = "webm"))]
fn js_safe_i64(value: i64, field: &str) -> Result<JsValue, JsValue> {
    if value.unsigned_abs() > JS_MAX_SAFE_INTEGER {
        return Err(js_error(format!(
            "{field} {value} exceeds JavaScript's exact integer range"
        )));
    }
    Ok(JsValue::from_f64(value as f64))
}

#[cfg(feature = "audio-demux")]
fn media_track_config_to_js(track: &MediaTrackConfig) -> Result<JsValue, JsValue> {
    let object = Object::new();
    Reflect::set(
        &object,
        &"container".into(),
        &track.container.as_str().into(),
    )?;
    Reflect::set(&object, &"kind".into(), &track.kind.as_str().into())?;
    Reflect::set(
        &object,
        &"trackId".into(),
        &js_safe_u64(track.track_id, "trackId")?,
    )?;
    Reflect::set(&object, &"codec".into(), &track.codec.as_str().into())?;
    Reflect::set(&object, &"codecId".into(), &track.codec_id.as_str().into())?;
    Reflect::set(
        &object,
        &"timescale".into(),
        &JsValue::from_f64(track.timescale as f64),
    )?;
    Reflect::set(
        &object,
        &"sampleCount".into(),
        &JsValue::from_f64(track.sample_count as f64),
    )?;
    let timeline = match track.timeline {
        Some(timeline) => {
            let value = Object::new();
            Reflect::set(
                &value,
                &"presentationStart".into(),
                &js_safe_u64(timeline.presentation_start, "timeline.presentationStart")?,
            )?;
            Reflect::set(
                &value,
                &"mediaStart".into(),
                &js_safe_u64(timeline.media_start, "timeline.mediaStart")?,
            )?;
            Reflect::set(
                &value,
                &"duration".into(),
                &js_safe_u64(timeline.duration, "timeline.duration")?,
            )?;
            value.into()
        }
        None => JsValue::NULL,
    };
    Reflect::set(&object, &"timeline".into(), &timeline)?;
    set_optional_u32(&object, "width", track.width)?;
    set_optional_u32(&object, "height", track.height)?;
    set_optional_u32(&object, "sampleRate", track.sample_rate)?;
    set_optional_u8(&object, "channels", track.channels)?;
    set_optional_u8(&object, "bitsPerSample", track.bits_per_sample)?;
    Reflect::set(
        &object,
        &"pcmEndianness".into(),
        &track
            .pcm_endianness
            .map(|value| JsValue::from_str(value.as_str()))
            .unwrap_or(JsValue::NULL),
    )?;
    Reflect::set(
        &object,
        &"pcmFloat".into(),
        &track
            .pcm_float
            .map(JsValue::from_bool)
            .unwrap_or(JsValue::NULL),
    )?;
    Reflect::set(
        &object,
        &"codecPrivate".into(),
        &Uint8Array::from(track.codec_private.as_slice()).into(),
    )?;
    Reflect::set(
        &object,
        &"decoderConfiguration".into(),
        &Uint8Array::from(track.decoder_configuration.as_slice()).into(),
    )?;
    set_optional_u8(&object, "nalLengthSize", track.nal_length_size)?;
    Ok(object.into())
}

#[cfg(feature = "audio-demux")]
fn pcm_packet_trim_to_js(source_frame_start: u32, frame_count: u32) -> Result<JsValue, JsValue> {
    let object = Object::new();
    Reflect::set(
        &object,
        &"sourceFrameStart".into(),
        &JsValue::from_f64(source_frame_start as f64),
    )?;
    Reflect::set(
        &object,
        &"frameCount".into(),
        &JsValue::from_f64(frame_count as f64),
    )?;
    Ok(object.into())
}

#[cfg(feature = "audio-demux")]
fn mp4_media_events_to_js(events: Vec<Mp4MediaDemuxEvent>) -> Result<Array, JsValue> {
    let output = Array::new();
    for event in events {
        let value = match event {
            Mp4MediaDemuxEvent::Config(track) => {
                let value = media_track_config_to_js(&track)?;
                Reflect::set(&value, &"type".into(), &"config".into())?;
                value
            }
            Mp4MediaDemuxEvent::Packet(packet) => {
                let value: JsValue = media_track_packet_to_js(&packet)?.into();
                Reflect::set(&value, &"type".into(), &"packet".into())?;
                value
            }
        };
        output.push(&value);
    }
    Ok(output)
}

#[cfg(feature = "audio-demux")]
fn mxf_media_events_to_js(events: Vec<MxfMediaDemuxEvent>) -> Result<Array, JsValue> {
    let output = Array::new();
    for event in events {
        let value = match event {
            MxfMediaDemuxEvent::Config(track) => {
                let value = media_track_config_to_js(&track)?;
                Reflect::set(&value, &"type".into(), &"config".into())?;
                value
            }
            MxfMediaDemuxEvent::Packet(packet) => {
                let value: JsValue = media_track_packet_to_js(&packet)?.into();
                Reflect::set(&value, &"type".into(), &"packet".into())?;
                value
            }
        };
        output.push(&value);
    }
    Ok(output)
}

#[cfg(feature = "audio-demux")]
fn media_sample_index_to_js(sample: &MediaSampleIndex) -> Result<Object, JsValue> {
    let object = Object::new();
    Reflect::set(
        &object,
        &"trackId".into(),
        &js_safe_u64(sample.track_id, "trackId")?,
    )?;
    Reflect::set(&object, &"kind".into(), &sample.kind.as_str().into())?;
    Reflect::set(&object, &"codec".into(), &sample.codec.as_str().into())?;
    Reflect::set(
        &object,
        &"sampleId".into(),
        &JsValue::from_f64(sample.sample_id as f64),
    )?;
    Reflect::set(
        &object,
        &"offset".into(),
        &js_safe_u64(sample.absolute_offset, "offset")?,
    )?;
    Reflect::set(
        &object,
        &"size".into(),
        &JsValue::from_f64(sample.size as f64),
    )?;
    Reflect::set(
        &object,
        &"decodeTime".into(),
        &js_safe_u64(sample.decode_time, "decodeTime")?,
    )?;
    Reflect::set(
        &object,
        &"presentationTime".into(),
        &js_safe_i64(sample.presentation_time, "presentationTime")?,
    )?;
    Reflect::set(
        &object,
        &"duration".into(),
        &JsValue::from_f64(sample.duration as f64),
    )?;
    Reflect::set(
        &object,
        &"isSync".into(),
        &JsValue::from_bool(sample.is_sync),
    )?;
    Ok(object)
}

#[cfg(feature = "audio-demux")]
fn media_track_packet_to_js(packet: &MediaTrackPacket) -> Result<Object, JsValue> {
    let object = Object::new();
    Reflect::set(
        &object,
        &"trackId".into(),
        &js_safe_u64(packet.track_id, "trackId")?,
    )?;
    Reflect::set(&object, &"kind".into(), &packet.kind.as_str().into())?;
    Reflect::set(&object, &"codec".into(), &packet.codec.as_str().into())?;
    Reflect::set(
        &object,
        &"sampleId".into(),
        &JsValue::from_f64(packet.sample_id as f64),
    )?;
    Reflect::set(
        &object,
        &"data".into(),
        &Uint8Array::from(packet.data.as_slice()).into(),
    )?;
    Reflect::set(
        &object,
        &"decodeTime".into(),
        &js_safe_u64(packet.decode_time, "decodeTime")?,
    )?;
    Reflect::set(
        &object,
        &"presentationTime".into(),
        &js_safe_i64(packet.presentation_time, "presentationTime")?,
    )?;
    Reflect::set(
        &object,
        &"duration".into(),
        &JsValue::from_f64(packet.duration as f64),
    )?;
    Reflect::set(
        &object,
        &"isSync".into(),
        &JsValue::from_bool(packet.is_sync),
    )?;
    Ok(object)
}

#[cfg(feature = "webm")]
fn webm_media_events_to_js(events: Vec<WebmMediaDemuxEvent>) -> Result<Array, JsValue> {
    let output = Array::new();
    for event in events {
        let object = Object::new();
        match event {
            WebmMediaDemuxEvent::Config {
                timecode_scale_ns,
                track,
            } => {
                Reflect::set(&object, &"type".into(), &"config".into())?;
                Reflect::set(
                    &object,
                    &"timecodeScaleNs".into(),
                    &js_safe_u64(timecode_scale_ns, "timecodeScaleNs")?,
                )?;
                set_webm_media_track(&object, &track)?;
            }
            WebmMediaDemuxEvent::Packet {
                track_number,
                kind,
                codec_id,
                data,
                timestamp_ns,
                duration_ns,
                discard_padding_ns,
                is_keyframe,
            } => {
                Reflect::set(&object, &"type".into(), &"packet".into())?;
                Reflect::set(
                    &object,
                    &"trackId".into(),
                    &js_safe_u64(track_number, "trackId")?,
                )?;
                Reflect::set(&object, &"kind".into(), &kind.as_str().into())?;
                Reflect::set(&object, &"codecId".into(), &codec_id.into())?;
                Reflect::set(
                    &object,
                    &"data".into(),
                    &Uint8Array::from(data.as_slice()).into(),
                )?;
                Reflect::set(
                    &object,
                    &"timestampNs".into(),
                    &js_safe_i64(timestamp_ns, "timestampNs")?,
                )?;
                Reflect::set(
                    &object,
                    &"durationNs".into(),
                    &duration_ns
                        .map(|value| js_safe_u64(value, "durationNs"))
                        .transpose()?
                        .unwrap_or(JsValue::NULL),
                )?;
                Reflect::set(
                    &object,
                    &"discardPaddingNs".into(),
                    &discard_padding_ns
                        .map(|value| js_safe_i64(value, "discardPaddingNs"))
                        .transpose()?
                        .unwrap_or(JsValue::NULL),
                )?;
                Reflect::set(
                    &object,
                    &"isKeyframe".into(),
                    &JsValue::from_bool(is_keyframe),
                )?;
            }
        }
        output.push(&object);
    }
    Ok(output)
}

#[cfg(feature = "webm")]
fn set_webm_media_track(object: &Object, track: &WebmMediaTrackConfig) -> Result<(), JsValue> {
    Reflect::set(
        object,
        &"trackId".into(),
        &js_safe_u64(track.track_number, "trackId")?,
    )?;
    Reflect::set(object, &"kind".into(), &track.kind.as_str().into())?;
    Reflect::set(object, &"codecId".into(), &track.codec_id.as_str().into())?;
    Reflect::set(
        object,
        &"codecPrivate".into(),
        &Uint8Array::from(track.codec_private.as_slice()).into(),
    )?;
    Reflect::set(
        object,
        &"decoderConfiguration".into(),
        &Uint8Array::from(track.decoder_configuration.as_slice()).into(),
    )?;
    Reflect::set(
        object,
        &"nalLengthSize".into(),
        &track
            .nal_length_size
            .map(|value| JsValue::from_f64(value as f64))
            .unwrap_or(JsValue::NULL),
    )?;
    for (key, value) in [
        ("width", track.width),
        ("height", track.height),
        ("sampleRate", track.sample_rate),
    ] {
        Reflect::set(
            object,
            &key.into(),
            &value
                .map(|value| JsValue::from_f64(value as f64))
                .unwrap_or(JsValue::NULL),
        )?;
    }
    Reflect::set(
        object,
        &"channels".into(),
        &track
            .channels
            .map(|value| JsValue::from_f64(value as f64))
            .unwrap_or(JsValue::NULL),
    )?;
    Reflect::set(
        object,
        &"defaultDurationNs".into(),
        &track
            .default_duration_ns
            .map(|value| js_safe_u64(value, "defaultDurationNs"))
            .transpose()?
            .unwrap_or(JsValue::NULL),
    )?;
    Ok(())
}

#[cfg(feature = "audio-demux")]
fn audio_demux_event_to_js(event: AudioDemuxEvent) -> Result<JsValue, JsValue> {
    let object = Object::new();

    match event {
        AudioDemuxEvent::Config(config) => {
            Reflect::set(
                &object,
                &JsValue::from_str("type"),
                &JsValue::from_str("config"),
            )?;
            Reflect::set(
                &object,
                &JsValue::from_str("container"),
                &JsValue::from_str(config.container.as_str()),
            )?;
            Reflect::set(
                &object,
                &JsValue::from_str("codec"),
                &JsValue::from_str(config.codec.as_str()),
            )?;
            if let Some(format) = config.packet_format {
                Reflect::set(
                    &object,
                    &JsValue::from_str("format"),
                    &JsValue::from_str(format.as_str()),
                )?;
            }
            if let Some(codec_id) = config.codec_id {
                Reflect::set(
                    &object,
                    &JsValue::from_str("codecId"),
                    &JsValue::from_str(&codec_id),
                )?;
            }
            set_optional_u64(&object, "trackId", config.track_id)?;
            set_optional_u16(&object, "pid", config.pid)?;
            set_optional_u8(&object, "streamType", config.stream_type)?;
            set_optional_u32(&object, "sampleRate", config.sample_rate)?;
            set_optional_u8(&object, "channels", config.channels)?;
            set_optional_u8(&object, "bitsPerSample", config.bits_per_sample)?;
            if let Some(endianness) = config.pcm_endianness {
                Reflect::set(
                    &object,
                    &JsValue::from_str("pcmEndianness"),
                    &JsValue::from_str(endianness.as_str()),
                )?;
            }
            if let Some(float) = config.pcm_float {
                Reflect::set(
                    &object,
                    &JsValue::from_str("pcmFloat"),
                    &JsValue::from_bool(float),
                )?;
            }
            set_optional_u32(&object, "sampleCount", config.sample_count)?;
            Reflect::set(
                &object,
                &JsValue::from_str("codecPrivate"),
                &Uint8Array::from(config.codec_private.as_slice()).into(),
            )?;
            set_optional_u16(&object, "preSkip", config.pre_skip)?;
            set_optional_i16(&object, "outputGain", config.output_gain)?;
            set_optional_u8(&object, "mappingFamily", config.mapping_family)?;
        }
        AudioDemuxEvent::Packet(packet) => {
            Reflect::set(
                &object,
                &JsValue::from_str("type"),
                &JsValue::from_str("packet"),
            )?;
            Reflect::set(
                &object,
                &JsValue::from_str("container"),
                &JsValue::from_str(packet.container.as_str()),
            )?;
            Reflect::set(
                &object,
                &JsValue::from_str("codec"),
                &JsValue::from_str(packet.codec.as_str()),
            )?;
            Reflect::set(
                &object,
                &JsValue::from_str("format"),
                &JsValue::from_str(packet.format.as_str()),
            )?;
            Reflect::set(
                &object,
                &JsValue::from_str("data"),
                &Uint8Array::from(packet.data.as_slice()).into(),
            )?;
            if let Some(raw_data) = packet.raw_data {
                Reflect::set(
                    &object,
                    &JsValue::from_str("rawData"),
                    &Uint8Array::from(raw_data.as_slice()).into(),
                )?;
            }
            set_optional_u64(&object, "trackId", packet.track_id)?;
            set_optional_u16(&object, "pid", packet.pid)?;
            set_optional_u8(&object, "streamType", packet.stream_type)?;
            set_optional_u32(&object, "sampleId", packet.sample_id)?;
            set_optional_u64(&object, "startTime", packet.start_time)?;
            set_optional_u32(&object, "duration", packet.duration)?;
            set_optional_i32(&object, "renderingOffset", packet.rendering_offset)?;
            if let Some(is_sync) = packet.is_sync {
                Reflect::set(
                    &object,
                    &JsValue::from_str("isSync"),
                    &JsValue::from_bool(is_sync),
                )?;
            }
            set_optional_i64(&object, "timecode", packet.timecode)?;
        }
    }

    Ok(object.into())
}

#[cfg(feature = "audio-demux")]
fn set_optional_u8(object: &Object, key: &str, value: Option<u8>) -> Result<(), JsValue> {
    if let Some(value) = value {
        Reflect::set(
            object,
            &JsValue::from_str(key),
            &JsValue::from_f64(value as f64),
        )?;
    }
    Ok(())
}

#[cfg(feature = "audio-demux")]
fn set_optional_u16(object: &Object, key: &str, value: Option<u16>) -> Result<(), JsValue> {
    if let Some(value) = value {
        Reflect::set(
            object,
            &JsValue::from_str(key),
            &JsValue::from_f64(value as f64),
        )?;
    }
    Ok(())
}

#[cfg(feature = "audio-demux")]
fn set_optional_u32(object: &Object, key: &str, value: Option<u32>) -> Result<(), JsValue> {
    if let Some(value) = value {
        Reflect::set(
            object,
            &JsValue::from_str(key),
            &JsValue::from_f64(value as f64),
        )?;
    }
    Ok(())
}

#[cfg(feature = "audio-demux")]
fn set_optional_u64(object: &Object, key: &str, value: Option<u64>) -> Result<(), JsValue> {
    if let Some(value) = value {
        Reflect::set(
            object,
            &JsValue::from_str(key),
            &JsValue::from_f64(value as f64),
        )?;
    }
    Ok(())
}

#[cfg(feature = "audio-demux")]
fn set_optional_i16(object: &Object, key: &str, value: Option<i16>) -> Result<(), JsValue> {
    if let Some(value) = value {
        Reflect::set(
            object,
            &JsValue::from_str(key),
            &JsValue::from_f64(value as f64),
        )?;
    }
    Ok(())
}

#[cfg(feature = "audio-demux")]
fn set_optional_i32(object: &Object, key: &str, value: Option<i32>) -> Result<(), JsValue> {
    if let Some(value) = value {
        Reflect::set(
            object,
            &JsValue::from_str(key),
            &JsValue::from_f64(value as f64),
        )?;
    }
    Ok(())
}

#[cfg(feature = "audio-demux")]
fn set_optional_i64(object: &Object, key: &str, value: Option<i64>) -> Result<(), JsValue> {
    if let Some(value) = value {
        Reflect::set(
            object,
            &JsValue::from_str(key),
            &JsValue::from_f64(value as f64),
        )?;
    }
    Ok(())
}

#[cfg(feature = "opus-debox")]
fn opus_debox_events_to_js(events: Vec<OpusDeboxEvent>) -> Result<Array, JsValue> {
    let array = Array::new();
    for event in events {
        array.push(&opus_debox_event_to_js(event)?);
    }
    Ok(array)
}

#[cfg(feature = "opus-debox")]
fn opus_debox_event_to_js(event: OpusDeboxEvent) -> Result<JsValue, JsValue> {
    let object = Object::new();

    match event {
        OpusDeboxEvent::Config {
            container,
            sample_rate,
            channels,
            pre_skip,
            output_gain,
            mapping_family,
            codec_private,
        } => {
            Reflect::set(
                &object,
                &JsValue::from_str("type"),
                &JsValue::from_str("config"),
            )?;
            Reflect::set(
                &object,
                &JsValue::from_str("container"),
                &JsValue::from_str(container),
            )?;
            Reflect::set(
                &object,
                &JsValue::from_str("codec"),
                &JsValue::from_str("opus"),
            )?;
            Reflect::set(
                &object,
                &JsValue::from_str("sampleRate"),
                &JsValue::from_f64(sample_rate as f64),
            )?;
            Reflect::set(
                &object,
                &JsValue::from_str("channels"),
                &JsValue::from_f64(channels as f64),
            )?;
            Reflect::set(
                &object,
                &JsValue::from_str("preSkip"),
                &JsValue::from_f64(pre_skip as f64),
            )?;
            Reflect::set(
                &object,
                &JsValue::from_str("outputGain"),
                &JsValue::from_f64(output_gain as f64),
            )?;
            Reflect::set(
                &object,
                &JsValue::from_str("mappingFamily"),
                &JsValue::from_f64(mapping_family as f64),
            )?;
            Reflect::set(
                &object,
                &JsValue::from_str("codecPrivate"),
                &Uint8Array::from(codec_private.as_slice()).into(),
            )?;
        }
        OpusDeboxEvent::Tags { container, data } => {
            Reflect::set(
                &object,
                &JsValue::from_str("type"),
                &JsValue::from_str("tags"),
            )?;
            Reflect::set(
                &object,
                &JsValue::from_str("container"),
                &JsValue::from_str(container),
            )?;
            Reflect::set(
                &object,
                &JsValue::from_str("codec"),
                &JsValue::from_str("opus"),
            )?;
            Reflect::set(
                &object,
                &JsValue::from_str("data"),
                &Uint8Array::from(data.as_slice()).into(),
            )?;
        }
        OpusDeboxEvent::Packet {
            container,
            data,
            timecode,
        } => {
            Reflect::set(
                &object,
                &JsValue::from_str("type"),
                &JsValue::from_str("packet"),
            )?;
            Reflect::set(
                &object,
                &JsValue::from_str("container"),
                &JsValue::from_str(container),
            )?;
            Reflect::set(
                &object,
                &JsValue::from_str("codec"),
                &JsValue::from_str("opus"),
            )?;
            Reflect::set(
                &object,
                &JsValue::from_str("data"),
                &Uint8Array::from(data.as_slice()).into(),
            )?;
            if let Some(timecode) = timecode {
                Reflect::set(
                    &object,
                    &JsValue::from_str("timecode"),
                    &JsValue::from_f64(timecode as f64),
                )?;
            }
        }
    }

    Ok(object.into())
}

#[cfg(feature = "aac-debox")]
fn aac_debox_events_to_js(events: Vec<AacDeboxEvent>) -> Result<Array, JsValue> {
    let array = Array::new();
    for event in events {
        array.push(&aac_debox_event_to_js(event)?);
    }
    Ok(array)
}

#[cfg(feature = "aac-debox")]
fn aac_debox_event_to_js(event: AacDeboxEvent) -> Result<JsValue, JsValue> {
    let object = Object::new();

    match event {
        AacDeboxEvent::Config {
            container,
            sample_rate,
            channels,
            track_id,
            sample_count,
        } => {
            Reflect::set(
                &object,
                &JsValue::from_str("type"),
                &JsValue::from_str("config"),
            )?;
            Reflect::set(
                &object,
                &JsValue::from_str("container"),
                &JsValue::from_str(container),
            )?;
            Reflect::set(
                &object,
                &JsValue::from_str("codec"),
                &JsValue::from_str("aac"),
            )?;
            Reflect::set(
                &object,
                &JsValue::from_str("sampleRate"),
                &JsValue::from_f64(sample_rate as f64),
            )?;
            Reflect::set(
                &object,
                &JsValue::from_str("channels"),
                &JsValue::from_f64(channels as f64),
            )?;
            Reflect::set(
                &object,
                &JsValue::from_str("trackId"),
                &JsValue::from_f64(track_id as f64),
            )?;
            Reflect::set(
                &object,
                &JsValue::from_str("sampleCount"),
                &JsValue::from_f64(sample_count as f64),
            )?;
        }
        AacDeboxEvent::Packet {
            container,
            data,
            raw_data,
            sample_id,
            start_time,
            duration,
            rendering_offset,
            is_sync,
        } => {
            Reflect::set(
                &object,
                &JsValue::from_str("type"),
                &JsValue::from_str("packet"),
            )?;
            Reflect::set(
                &object,
                &JsValue::from_str("container"),
                &JsValue::from_str(container),
            )?;
            Reflect::set(
                &object,
                &JsValue::from_str("codec"),
                &JsValue::from_str("aac"),
            )?;
            Reflect::set(
                &object,
                &JsValue::from_str("format"),
                &JsValue::from_str("adts"),
            )?;
            Reflect::set(
                &object,
                &JsValue::from_str("data"),
                &Uint8Array::from(data.as_slice()).into(),
            )?;
            Reflect::set(
                &object,
                &JsValue::from_str("rawData"),
                &Uint8Array::from(raw_data.as_slice()).into(),
            )?;
            Reflect::set(
                &object,
                &JsValue::from_str("sampleId"),
                &JsValue::from_f64(sample_id as f64),
            )?;
            Reflect::set(
                &object,
                &JsValue::from_str("startTime"),
                &JsValue::from_f64(start_time as f64),
            )?;
            Reflect::set(
                &object,
                &JsValue::from_str("duration"),
                &JsValue::from_f64(duration as f64),
            )?;
            Reflect::set(
                &object,
                &JsValue::from_str("renderingOffset"),
                &JsValue::from_f64(rendering_offset as f64),
            )?;
            Reflect::set(
                &object,
                &JsValue::from_str("isSync"),
                &JsValue::from_bool(is_sync),
            )?;
        }
    }

    Ok(object.into())
}

fn js_error(error: String) -> JsValue {
    JsValue::from_str(&error)
}

#[cfg(feature = "flac")]
fn planar_f32_to_interleaved_i32(
    planar: &[f32],
    frames: usize,
    channels: usize,
    bits_per_sample: u32,
) -> Result<Vec<i32>, JsValue> {
    let scale = match bits_per_sample {
        1..=16 => 32768.0f64,
        17..=24 => 8_388_608.0f64,
        25..=32 => 2_147_483_648.0f64,
        _ => {
            return Err(js_error(format!(
                "Unsupported FLAC bits-per-sample for wasm encoder: {bits_per_sample}"
            )));
        }
    };
    let max_sample = match bits_per_sample {
        1..=16 => i16::MAX as i32,
        17..=24 => 8_388_607i32,
        _ => i32::MAX,
    };
    let min_sample = match bits_per_sample {
        1..=16 => i16::MIN as i32,
        17..=24 => -8_388_608i32,
        _ => i32::MIN,
    };

    let mut interleaved = Vec::with_capacity(frames.saturating_mul(channels));
    for frame in 0..frames {
        for channel in 0..channels {
            let sample = planar[(channel * frames) + frame].clamp(-1.0, 1.0) as f64;
            let scaled = (sample * scale).round();
            interleaved.push((scaled as i64).clamp(min_sample as i64, max_sample as i64) as i32);
        }
    }
    Ok(interleaved)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    fn fixture(path: &str) -> Vec<u8> {
        fs::read(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("testdata")
                .join(path),
        )
        .unwrap()
    }

    fn decode_all(format: &str, data: &[u8], chunk_size: usize) -> Vec<AudioData> {
        let mut decoder = WasmMusicDecoder::new_with_format(format).unwrap();
        let mut frames = Vec::new();
        for chunk in data.chunks(chunk_size) {
            frames.extend(decoder.push_frames(chunk).unwrap());
        }
        frames.extend(decoder.flush_frames().unwrap());
        frames
    }

    fn decode_canonical_raw_mono(
        data: &[u8],
        chunk_size: usize,
    ) -> (Vec<i16>, Vec<i16>, u64, String, Vec<u32>) {
        let mut decoder = WasmCanonicalPcmDecoder::new_raw_linear16(44_100, 1).unwrap();
        let mut left = Vec::new();
        let mut right = Vec::new();
        let mut block_sizes = Vec::new();
        let mut expected_start = 0_u64;
        let mut identity = None;

        let mut collect = |batch: CanonicalDecodeBatch| {
            for block in batch.blocks {
                assert_eq!(block.start_frame, expected_start);
                assert!(block.frame_count as usize <= CANONICAL_PCM_BLOCK_FRAMES);
                block_sizes.push(block.frame_count);
                let channel_bytes = block.frame_count as usize * std::mem::size_of::<i16>();
                assert_eq!(block.pcm_s16_planar.len(), channel_bytes * 2);
                left.extend(
                    block.pcm_s16_planar[..channel_bytes]
                        .chunks_exact(2)
                        .map(|bytes| i16::from_le_bytes([bytes[0], bytes[1]])),
                );
                right.extend(
                    block.pcm_s16_planar[channel_bytes..]
                        .chunks_exact(2)
                        .map(|bytes| i16::from_le_bytes([bytes[0], bytes[1]])),
                );
                expected_start += u64::from(block.frame_count);
            }
            if batch.done {
                identity = batch.source_identity;
            } else {
                assert!(batch.source_identity.is_none());
            }
            assert_eq!(batch.frame_count, expected_start);
            if expected_start > 0 {
                assert_eq!(batch.source_sample_rate, 44_100);
                assert_eq!(batch.source_channels, 1);
            }
        };

        for chunk in data.chunks(chunk_size) {
            collect(decoder.push_rust(chunk).unwrap());
        }
        collect(decoder.finish_rust().unwrap());
        (
            left,
            right,
            expected_start,
            identity.expect("finished canonical source identity"),
            block_sizes,
        )
    }

    fn pcm_caf_fixture(frame_count: usize) -> Vec<u8> {
        fn chunk(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
            let mut bytes = Vec::new();
            bytes.extend_from_slice(kind);
            bytes.extend_from_slice(&(payload.len() as i64).to_be_bytes());
            bytes.extend_from_slice(payload);
            bytes
        }

        let mut description = Vec::with_capacity(32);
        description.extend_from_slice(&48_000f64.to_be_bytes());
        description.extend_from_slice(b"lpcm");
        description.extend_from_slice(&(1u32 | (1u32 << 1)).to_be_bytes());
        description.extend_from_slice(&8u32.to_be_bytes());
        description.extend_from_slice(&1u32.to_be_bytes());
        description.extend_from_slice(&2u32.to_be_bytes());
        description.extend_from_slice(&32u32.to_be_bytes());

        let mut data = Vec::with_capacity(4 + frame_count * 8);
        data.extend_from_slice(&1u32.to_be_bytes());
        for frame in 0..frame_count {
            let left = frame as f32 / frame_count.max(1) as f32;
            data.extend_from_slice(&left.to_le_bytes());
            data.extend_from_slice(&(-0.5f32).to_le_bytes());
        }

        let mut file = b"caff\0\x01\0\0".to_vec();
        file.extend_from_slice(&chunk(b"desc", &description));
        file.extend_from_slice(&chunk(b"data", &data));
        file
    }

    #[cfg(feature = "flac")]
    #[test]
    fn flac_planar_conversion_preserves_normalized_24_bit_pcm() {
        let samples = [-8_388_608, -4_194_305, -1, 0, 1, 4_194_305, 8_388_607];
        let planar = samples
            .iter()
            .map(|sample| *sample as f32 / 8_388_608.0)
            .collect::<Vec<_>>();
        let interleaved = planar_f32_to_interleaved_i32(&planar, planar.len(), 1, 24).unwrap();
        assert_eq!(interleaved, samples);
    }

    #[test]
    fn streaming_entry_points_reject_oversized_chunks() {
        let oversized = vec![0_u8; MAX_STREAM_INPUT_CHUNK_BYTES + 1];
        let mut decoder = WasmMusicDecoder::new_auto();
        let error = decoder.push_frames(&oversized).unwrap_err();
        assert!(error.contains("streaming budget"));

        #[cfg(feature = "opus-debox")]
        {
            let mut deboxer = WasmOpusDeboxer::new_auto();
            let error = deboxer.push_events(&oversized).unwrap_err();
            assert!(error.contains("streaming budget"));
        }

        #[cfg(feature = "aac-debox")]
        {
            let mut deboxer = WasmAacDeboxer::new_auto();
            let error = deboxer.push_events(&oversized).unwrap_err();
            assert!(error.contains("streaming budget"));
        }
    }

    #[test]
    fn automatic_detection_never_retains_more_than_64_kib() {
        let undecidable = vec![0_u8; MAX_DETECTION_BYTES];
        let mut decoder = WasmMusicDecoder::new_auto();
        let error = decoder.push_frames(&undecidable).unwrap_err();
        assert!(!error.is_empty());

        match decoder.state {
            DecoderState::Finished => {}
            _ => panic!("the detector must finish after its bounded probe"),
        }
    }

    #[test]
    fn canonical_pcm_is_chunk_invariant_and_bounded() {
        let source_frames = 4_410usize;
        let mut source = Vec::with_capacity(source_frames * 2);
        for frame in 0..source_frames {
            let sample = (((frame * 97) % 65_535) as i32 - 32_767) as i16;
            source.extend_from_slice(&sample.to_le_bytes());
        }

        let reference = decode_canonical_raw_mono(&source, 65_536);
        assert_eq!(reference.0, reference.1);
        assert_eq!(reference.2, 4_800);
        assert!(reference.3.starts_with("sha256:"));
        assert_eq!(reference.4, vec![4_800]);
        for chunk_size in [1, 7, 256, 997, 4_096] {
            assert_eq!(
                decode_canonical_raw_mono(&source, chunk_size),
                reference,
                "canonical output changed for {chunk_size}-byte chunks"
            );
        }
    }

    #[test]
    fn canonical_pcm_block_boundaries_do_not_follow_input_chunks() {
        let source_frames = 110_250usize;
        let mut source = Vec::with_capacity(source_frames * 2);
        for frame in 0..source_frames {
            source.extend_from_slice(&((frame as i32 % 32_000) as i16).to_le_bytes());
        }

        let small_chunks = decode_canonical_raw_mono(&source, 997);
        let large_chunks = decode_canonical_raw_mono(&source, 65_536);
        assert_eq!(small_chunks, large_chunks);
        assert_eq!(small_chunks.2, 120_000);
        assert_eq!(small_chunks.4, vec![96_000, 24_000]);
    }

    #[cfg(feature = "aiff")]
    #[test]
    fn aiff_push_drains_pcm_frames() {
        let data = fixture("aiff/A_Tusk_is_used_to_make_costly_gifts.aiff");
        let mut decoder = WasmMusicDecoder::new_with_format("aiff").unwrap();
        let mut frames = Vec::new();
        let mut first_output_at = None;
        for (index, chunk) in data.chunks(997).enumerate() {
            let output = decoder.push_frames(chunk).unwrap();
            if !output.is_empty() && first_output_at.is_none() {
                first_output_at = Some((index + 1) * 997);
            }
            frames.extend(output);
        }
        frames.extend(decoder.flush_frames().unwrap());
        assert!(!frames.is_empty());
        assert!(first_output_at.unwrap() < data.len());
        assert_eq!(frames[0].channel_count(), 1);
    }

    #[cfg(feature = "alac")]
    #[test]
    fn alac_generic_decoder_requires_seekable_ranges() {
        assert_eq!(
            decoder_for_format("alac").err().unwrap(),
            SEEKABLE_ALAC_REQUIRED
        );
        assert_eq!(
            decoder_for_format("caf-alac").err().unwrap(),
            SEEKABLE_ALAC_REQUIRED
        );
    }

    #[cfg(feature = "flac")]
    #[test]
    fn flac_push_drains_pcm_frames() {
        let data = fixture("flac/A_Tusk_is_used_to_make_costly_gifts.flac");
        let frames = decode_all("flac", &data, 997);
        assert!(!frames.is_empty());
        assert_eq!(frames[0].bits_per_sample(), 16);
        assert_eq!(frames[0].channel_count(), 1);
        assert_eq!(frames[0].sampling_rate(), 16_000);
    }

    #[cfg(feature = "mp3")]
    #[test]
    fn mp3_push_drains_pcm_frames() {
        let data = fixture("mp3/A_Tusk_is_used_to_make_costly_gifts.mp3");
        let frames = decode_all("mp3", &data, 997);
        assert!(!frames.is_empty());
        assert_eq!(frames[0].bits_per_sample(), 16);
        assert_eq!(frames[0].channel_count(), 1);
        assert_eq!(frames[0].sampling_rate(), 16_000);
    }

    #[cfg(feature = "vorbis")]
    #[test]
    fn vorbis_push_drains_pcm_frames() {
        let data = fixture("vorbis/A_Tusk_is_used_to_make_costly_gifts.ogg");
        let frames = decode_all("ogg-vorbis", &data, 641);
        assert!(!frames.is_empty());
        assert_eq!(frames[0].bits_per_sample(), 16);
        assert_eq!(frames[0].channel_count(), 1);
        assert_eq!(frames[0].sampling_rate(), 8_000);
    }

    #[test]
    fn wav_push_drains_pcm_frames() {
        let data = fixture("wav_stereo/A_Tusk_is_used_to_make_costly_gifts.wav");
        let frames = decode_all("wav", &data, 997);
        assert!(!frames.is_empty());
        assert_eq!(frames[0].bits_per_sample(), 16);
        assert_eq!(frames[0].channel_count(), 2);
        assert_eq!(frames[0].sampling_rate(), 16_000);
    }

    #[cfg(all(
        feature = "detect",
        feature = "audio-demux",
        feature = "aac-lc",
        feature = "alac",
        feature = "wav",
        feature = "opus",
        feature = "flac"
    ))]
    #[test]
    fn rust_library_encoder_streams_without_a_javascript_boundary() {
        let data = fixture("wav_stereo/A_Tusk_is_used_to_make_costly_gifts.wav");
        let mut encoder = WasmStreamingLibraryEncoder::new_rust(true).unwrap();
        let mut opus_bytes = 0usize;
        let mut flac_bytes = 0usize;
        let mut preservation_geometry = None;
        for chunk in data.chunks(997) {
            let batch = encoder.push_rust(chunk).unwrap();
            assert!(!batch.done);
            opus_bytes += batch
                .opus_packets
                .iter()
                .map(|packet| packet.bytes.len())
                .sum::<usize>();
            flac_bytes += batch
                .flac_packets
                .iter()
                .map(|packet| packet.bytes.len())
                .sum::<usize>();
            if preservation_geometry.is_none() {
                preservation_geometry = batch.flac_packets.first().map(|packet| {
                    let header = frame_header::FrameHeaderV2::decode(&mut &packet.bytes[..])
                        .expect("SoundKit-v2 FLAC header");
                    (header.sample_rate(), header.channels())
                });
            }
        }
        let result = encoder.finish_rust().unwrap();
        opus_bytes += result
            .opus_packets
            .iter()
            .map(|packet| packet.bytes.len())
            .sum::<usize>();
        flac_bytes += result
            .flac_packets
            .iter()
            .map(|packet| packet.bytes.len())
            .sum::<usize>();
        if preservation_geometry.is_none() {
            preservation_geometry = result.flac_packets.first().map(|packet| {
                let header = frame_header::FrameHeaderV2::decode(&mut &packet.bytes[..])
                    .expect("SoundKit-v2 FLAC header");
                (header.sample_rate(), header.channels())
            });
        }
        assert!(result.done);
        assert!(result.frame_count > 0);
        assert!(opus_bytes > 0);
        assert!(flac_bytes > 0);
        assert!(result.opus_index.is_some());
        assert!(result.flac_index.is_some());
        assert_eq!(preservation_geometry, Some((16_000, 2)));
        assert!(result
            .source_identity
            .as_deref()
            .is_some_and(|value| { value.starts_with("sha256:") && value.len() == 71 }));
    }

    #[cfg(all(
        feature = "detect",
        feature = "audio-demux",
        feature = "aac-lc",
        feature = "alac",
        feature = "wav",
        feature = "opus",
        feature = "flac"
    ))]
    #[test]
    fn rust_seekable_encoder_imports_pcm_caf_packets() {
        let data = pcm_caf_fixture(960);
        let index = CafAudioIndex::from_file(&data).unwrap();
        let mut encoder = WasmStreamingLibraryEncoder::new_seekable_pcm_rust(true).unwrap();
        for chunk in data.chunks(997) {
            encoder.update_source_bytes_rust(chunk).unwrap();
        }
        let mut opus_bytes = 0usize;
        let first = index.packets.first().unwrap();
        let last = index.packets.last().unwrap();
        let start = first.absolute_offset as usize;
        let end = last.absolute_offset as usize + last.size as usize;
        let decoded =
            audio_data_from_container_pcm(&index.config, data[start..end].to_vec()).unwrap();
        let channels = audio_to_f32_channels(&decoded).unwrap();
        assert!((channels[0][480] - 0.5).abs() < 0.000_001);
        assert!((channels[1][480] + 0.5).abs() < 0.000_001);
        let batch = encoder
            .push_caf_pcm_range_rust(&index, 0, index.packets.len(), &data[start..end])
            .unwrap();
        opus_bytes += batch
            .opus_packets
            .iter()
            .map(|packet| packet.bytes.len())
            .sum::<usize>();
        let result = encoder.finish_rust().unwrap();
        opus_bytes += result
            .opus_packets
            .iter()
            .map(|packet| packet.bytes.len())
            .sum::<usize>();
        assert!(result.done);
        assert_eq!(result.frame_count, 960);
        assert!(opus_bytes > 0);
        assert!(result.opus_index.is_some());
        assert!(result.flac_index.is_some());
        assert!(result.source_identity.is_some());
    }

    #[cfg(all(
        feature = "detect",
        feature = "audio-demux",
        feature = "aac-lc",
        feature = "alac",
        feature = "wav",
        feature = "opus",
        feature = "flac"
    ))]
    #[test]
    fn rust_seekable_encoder_imports_mdat_first_aac_mp4() {
        let data = fixture("mac_aac/A_Tusk_is_used_to_make_costly_gifts.m4a");
        let mdat = data.windows(4).position(|value| value == b"mdat").unwrap();
        let moov = data.windows(4).position(|value| value == b"moov").unwrap();
        assert!(mdat < moov, "fixture must keep mdat before moov");

        let index = Mp4MediaIndex::from_file(&data).unwrap();
        let track = index
            .tracks
            .iter()
            .find(|track| track.kind == MediaTrackKind::Audio && track.codec == "aac")
            .unwrap();
        let track_id = track.track_id;
        let mut encoder =
            WasmStreamingLibraryEncoder::new_aac_lc_rust(&track.codec_private, true).unwrap();
        for chunk in data.chunks(997) {
            encoder.update_source_bytes_rust(chunk).unwrap();
        }
        let mut opus_bytes = 0usize;
        for (sample_index, sample) in index.samples.iter().enumerate() {
            if sample.kind != MediaTrackKind::Audio || sample.track_id != track_id {
                continue;
            }
            let start = sample.absolute_offset as usize;
            let end = start + sample.size as usize;
            let batch = encoder
                .push_mp4_sample_rust(&index, sample_index, &data[start..end])
                .unwrap();
            opus_bytes += batch
                .opus_packets
                .iter()
                .map(|packet| packet.bytes.len())
                .sum::<usize>();
        }
        let result = encoder.finish_rust().unwrap();
        opus_bytes += result
            .opus_packets
            .iter()
            .map(|packet| packet.bytes.len())
            .sum::<usize>();
        assert!(result.done);
        assert!(result.frame_count > 0);
        assert!(opus_bytes > 0);
        assert!(result.opus_index.is_some());
        assert!(result.flac_index.is_some());
        assert!(result.source_identity.is_some());
    }

    #[cfg(feature = "webm")]
    #[test]
    fn webm_vorbis_push_drains_pcm_frames() {
        let data = fixture("itag171/yt_itag_171_vorbis.webm");
        let frames = decode_all("webm", &data, 997);
        assert!(!frames.is_empty());
        assert_eq!(frames[0].bits_per_sample(), 16);
        assert_eq!(frames[0].channel_count(), 2);
        assert_eq!(frames[0].sampling_rate(), 44_100);
    }

    #[cfg(feature = "webm-opus")]
    #[test]
    fn webm_opus_push_drains_pcm_frames() {
        let data = fixture("video-compat/never-final/av1-main-opus.webm");
        let frames = decode_all("webm", &data, 997);
        assert!(!frames.is_empty());
        assert_eq!(frames[0].bits_per_sample(), 16);
        assert!(matches!(frames[0].channel_count(), 1 | 2));
        assert_eq!(frames[0].sampling_rate(), 48_000);
    }

    #[cfg(feature = "ogg-opus")]
    #[test]
    fn ogg_opus_push_drains_pcm_frames() {
        let data = fixture("ogg_opus/A_Tusk_is_used_to_make_costly_gifts_48khz.ogg");
        let frames = decode_all("ogg-opus", &data, 641);
        assert!(!frames.is_empty());
        assert_eq!(frames[0].bits_per_sample(), 16);
        assert_eq!(frames[0].channel_count(), 1);
        assert_eq!(frames[0].sampling_rate(), 48_000);
    }

    #[cfg(feature = "ac3")]
    #[test]
    fn ac3_push_drains_pcm_frames() {
        let data = fixture("ac3/A_Tusk_is_used_to_make_costly_gifts.ac3");
        let frames = decode_all("ac3", &data, 997);
        assert!(!frames.is_empty());
        assert_eq!(frames[0].bits_per_sample(), 16);
    }

    #[cfg(all(feature = "aac-lc", not(feature = "aac")))]
    #[test]
    fn aac_lc_adts_push_drains_pcm_frames() {
        let data =
            fixture("../soundkit-decoder/testdata/aac/A_Tusk_is_used_to_make_costly_gifts.aac");
        let frames = decode_all("aac", &data, 997);
        assert!(!frames.is_empty());
        assert_eq!(frames[0].bits_per_sample(), 16);
    }

    #[cfg(all(feature = "aac-lc", feature = "aac-debox", not(feature = "m4a")))]
    #[test]
    fn aac_lc_mp4_push_drains_pcm_frames() {
        let data = fixture("video-compat/never-final/h264-high-aac.mp4");
        let frames = decode_all("m4a", &data, 997);
        assert!(!frames.is_empty());
        assert_eq!(frames[0].bits_per_sample(), 16);
    }

    #[cfg(all(feature = "audio-demux", feature = "aac-lc"))]
    #[test]
    fn mxf_dnx_pcm_library_source_drains_bounded_audio_frames() {
        let data = fixture("video-compat/never-final/dnxhr-hqx-pcm.mxf");
        let mut decoder = LibrarySourceDecoder::new();
        let mut frames = Vec::new();
        for chunk in data.chunks(64 * 1024) {
            frames.extend(decoder.push(chunk).unwrap());
        }
        frames.extend(decoder.flush().unwrap());
        assert!(!frames.is_empty());
        assert_eq!(frames[0].bits_per_sample(), 24);
        assert_eq!(frames[0].channel_count(), 2);
        assert_eq!(frames[0].sampling_rate(), 48_000);
    }

    #[cfg(feature = "opus-debox")]
    #[test]
    fn opus_debox_ogg_emits_config_and_packets() {
        let data = fixture("ogg_opus/A_Tusk_is_used_to_make_costly_gifts.ogg");
        let mut deboxer = WasmOpusDeboxer::new_with_format("ogg-opus").unwrap();
        let mut events = Vec::new();

        for chunk in data.chunks(641) {
            events.extend(deboxer.push_events(chunk).unwrap());
        }
        events.extend(deboxer.flush_events().unwrap());

        assert!(events.iter().any(|event| matches!(
            event,
            OpusDeboxEvent::Config {
                container: "ogg",
                channels: 1,
                ..
            }
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            OpusDeboxEvent::Packet {
                container: "ogg",
                ..
            }
        )));
    }

    #[cfg(feature = "aac-debox")]
    #[test]
    fn aac_debox_m4a_emits_config_and_adts_packets() {
        let data = fixture("itag139/yt_itag_139_he_aac.mp4");
        let mut deboxer = WasmAacDeboxer::new_with_format("m4a").unwrap();
        let mut events = Vec::new();

        for chunk in data.chunks(997) {
            events.extend(deboxer.push_events(chunk).unwrap());
        }
        events.extend(deboxer.flush_events().unwrap());

        assert!(events.iter().any(|event| matches!(
            event,
            AacDeboxEvent::Config {
                container: "mp4",
                sample_rate: 11_025,
                channels: 2,
                ..
            }
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            AacDeboxEvent::Packet {
                container: "mp4",
                data,
                raw_data,
                ..
            } if data.starts_with(&[0xff, 0xf1]) && data.len() > raw_data.len()
        )));
    }
}
