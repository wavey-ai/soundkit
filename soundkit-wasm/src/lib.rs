#[cfg(feature = "aac-lc")]
use js_sys::Float32Array;
use js_sys::{Array, Object, Reflect, Uint8Array};
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
use soundkit::audio_types::AudioData;
use soundkit::crypto::ChaCha20Poly1305PacketCipher;
use soundkit::frame_stream::{SoundKitFrame, SoundKitFrameStream, SoundKitFrameStreamOptions};
use soundkit::raw_pcm::{RawPcmFormat, RawPcmStreamProcessor};
use soundkit::wav::WavStreamProcessor;
use wasm_bindgen::prelude::*;

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
#[cfg(feature = "aiff")]
use soundkit_aiff::AiffDecoder;
#[cfg(feature = "alac")]
use soundkit_alac::{
    inspect_caf_chunk, validate_caf_file_header, AlacDecoder, AlacPacketDecoder, CafAlacPacketIndex,
};
#[cfg(feature = "audio-demux")]
use soundkit_audio_demux::{
    inspect_mp4_top_level_box, AudioDemuxEvent, AudioTrackDemuxer, MediaSampleIndex,
    MediaTrackConfig, MediaTrackPacket, Mp4MediaDemuxEvent, Mp4MediaDemuxer, Mp4MediaIndex,
    MxfMediaDemuxEvent, MxfMediaDemuxer,
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
#[cfg(any(feature = "aac", feature = "m4a", feature = "mp3", feature = "flac"))]
const DEFAULT_SCRATCH_SAMPLES: usize = 262_144;

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
    Detecting {
        buffer: Vec<u8>,
        bytes_collected: usize,
    },
    Decoding {
        decoder: FormatDecoder,
    },
    Finished,
}

enum FormatDecoder {
    #[cfg(feature = "aac")]
    Aac(Box<AacDecoder>),
    #[cfg(feature = "m4a")]
    M4a(Box<AacDecoderMp4>),
    #[cfg(feature = "aiff")]
    Aiff(Box<AiffDecoder>),
    #[cfg(feature = "alac")]
    Alac(Box<AlacDecoder>),
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

#[cfg(feature = "opus-debox")]
enum OpusDeboxState {
    Detecting {
        buffer: Vec<u8>,
        bytes_collected: usize,
    },
    Ogg(OggOpusDemuxer),
    Raw(RawOpusDeboxer),
    WebM(WebmOpusDemuxer),
    Finished,
}

#[cfg(feature = "aac-debox")]
enum AacDeboxState {
    Detecting {
        buffer: Vec<u8>,
        bytes_collected: usize,
    },
    Mp4(AacMp4Demuxer),
    Finished,
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
            state: DecoderState::Detecting {
                buffer: Vec::new(),
                bytes_collected: 0,
            },
        }
    }

    #[wasm_bindgen(js_name = newWithFormat)]
    pub fn new_with_format(format: &str) -> Result<WasmMusicDecoder, JsValue> {
        let decoder = decoder_for_format(format).map_err(js_error)?;
        Ok(Self {
            state: DecoderState::Decoding { decoder },
        })
    }

    #[wasm_bindgen(js_name = newRawLinear16)]
    pub fn new_raw_linear16(sample_rate: u32, channels: u8) -> Result<WasmMusicDecoder, JsValue> {
        let format = RawPcmFormat::linear16(sample_rate, channels).map_err(js_error)?;
        Ok(Self {
            state: DecoderState::Decoding {
                decoder: FormatDecoder::RawPcm(Box::new(RawPcmStreamProcessor::new(format))),
            },
        })
    }

    #[wasm_bindgen(js_name = newRawLinear32)]
    pub fn new_raw_linear32(sample_rate: u32, channels: u8) -> Result<WasmMusicDecoder, JsValue> {
        let format = RawPcmFormat::linear32(sample_rate, channels).map_err(js_error)?;
        Ok(Self {
            state: DecoderState::Decoding {
                decoder: FormatDecoder::RawPcm(Box::new(RawPcmStreamProcessor::new(format))),
            },
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
            state: OpusDeboxState::Detecting {
                buffer: Vec::new(),
                bytes_collected: 0,
            },
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
            state: AacDeboxState::Detecting {
                buffer: Vec::new(),
                bytes_collected: 0,
            },
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

    /// Signal EOF and drain the final FLAC packet. OxideAV deliberately
    /// buffers a short final block until this call.
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

#[cfg(feature = "opus")]
#[wasm_bindgen]
impl WasmOpusDecoder {
    #[wasm_bindgen(constructor)]
    pub fn new(
        channels: usize,
        sample_rate: i32,
        frame_size: usize,
    ) -> Result<WasmOpusDecoder, JsValue> {
        if sample_rate != 48_000 {
            return Err(js_error(
                "soundkit wasm currently supports 48 kHz CELT-only Opus decode".to_string(),
            ));
        }
        let mut decoder = OpusDecoder::new(sample_rate as usize, channels);
        decoder.init().map_err(js_error)?;
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
        let state = std::mem::replace(&mut self.state, DecoderState::Finished);
        match state {
            DecoderState::Detecting {
                mut buffer,
                bytes_collected,
            } => {
                buffer.extend_from_slice(bytes);
                let new_bytes_collected = bytes_collected + bytes.len();

                if new_bytes_collected < MIN_DETECTION_BYTES {
                    self.state = DecoderState::Detecting {
                        buffer,
                        bytes_collected: new_bytes_collected,
                    };
                    return Ok(Vec::new());
                }

                match detect_and_init_decoder(&buffer) {
                    Ok(mut decoder) => {
                        let frames = decoder.process(&buffer)?;
                        self.state = DecoderState::Decoding { decoder };
                        Ok(frames)
                    }
                    Err(error) if new_bytes_collected < MAX_DETECTION_BYTES => {
                        self.state = DecoderState::Detecting {
                            buffer,
                            bytes_collected: new_bytes_collected,
                        };
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
                let frames = decoder.process(bytes)?;
                self.state = DecoderState::Decoding { decoder };
                Ok(frames)
            }
            DecoderState::Finished => Err("decoder is already finished".to_string()),
        }
    }

    fn flush_frames(&mut self) -> Result<Vec<AudioData>, String> {
        let state = std::mem::replace(&mut self.state, DecoderState::Finished);
        match state {
            DecoderState::Detecting { buffer, .. } => {
                let mut decoder = detect_and_init_decoder(&buffer)?;
                let mut frames = decoder.process(&buffer)?;
                frames.extend(decoder.flush()?);
                Ok(frames)
            }
            DecoderState::Decoding { mut decoder } => decoder.flush(),
            DecoderState::Finished => Ok(Vec::new()),
        }
    }
}

#[cfg(feature = "opus-debox")]
impl WasmOpusDeboxer {
    fn push_events(&mut self, bytes: &[u8]) -> Result<Vec<OpusDeboxEvent>, String> {
        let state = std::mem::replace(&mut self.state, OpusDeboxState::Finished);
        match state {
            OpusDeboxState::Detecting {
                mut buffer,
                bytes_collected,
            } => {
                buffer.extend_from_slice(bytes);
                let new_bytes_collected = bytes_collected + bytes.len();

                if new_bytes_collected < MIN_DETECTION_BYTES {
                    self.state = OpusDeboxState::Detecting {
                        buffer,
                        bytes_collected: new_bytes_collected,
                    };
                    return Ok(Vec::new());
                }

                match detect_and_init_opus_deboxer(&buffer) {
                    Ok(mut deboxer) => {
                        let events = process_opus_debox_state(&mut deboxer, &buffer)?;
                        self.state = deboxer;
                        Ok(events)
                    }
                    Err(error) if new_bytes_collected < MAX_DETECTION_BYTES => {
                        self.state = OpusDeboxState::Detecting {
                            buffer,
                            bytes_collected: new_bytes_collected,
                        };
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
            OpusDeboxState::Detecting { buffer, .. } => {
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
        let state = std::mem::replace(&mut self.state, AacDeboxState::Finished);
        match state {
            AacDeboxState::Detecting {
                mut buffer,
                bytes_collected,
            } => {
                buffer.extend_from_slice(bytes);
                let new_bytes_collected = bytes_collected + bytes.len();

                if new_bytes_collected < MIN_DETECTION_BYTES {
                    self.state = AacDeboxState::Detecting {
                        buffer,
                        bytes_collected: new_bytes_collected,
                    };
                    return Ok(Vec::new());
                }

                match detect_and_init_aac_deboxer(&buffer) {
                    Ok(mut deboxer) => {
                        let events = process_aac_debox_state(&mut deboxer, &buffer, false)?;
                        self.state = deboxer;
                        Ok(events)
                    }
                    Err(error) if new_bytes_collected < MAX_DETECTION_BYTES => {
                        self.state = AacDeboxState::Detecting {
                            buffer,
                            bytes_collected: new_bytes_collected,
                        };
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
            AacDeboxState::Detecting { buffer, .. } => {
                let mut deboxer = detect_and_init_aac_deboxer(&buffer)?;
                process_aac_debox_state(&mut deboxer, &buffer, true)
            }
            mut state @ AacDeboxState::Mp4(_) => process_aac_debox_state(&mut state, &[], true),
            AacDeboxState::Finished => Ok(Vec::new()),
        }
    }
}

impl FormatDecoder {
    fn process(&mut self, bytes: &[u8]) -> Result<Vec<AudioData>, String> {
        match self {
            #[cfg(feature = "aac")]
            FormatDecoder::Aac(decoder) => {
                decode_i16_with_drain(decoder.as_mut(), bytes, |decoder, samples, output| {
                    let (sample_rate, channels) = (decoder.sample_rate()?, decoder.channels()?);
                    Some(audio_data_i16(sample_rate, channels, &output[..samples]))
                })
            }
            #[cfg(feature = "m4a")]
            FormatDecoder::M4a(decoder) => {
                decode_i16_with_drain(decoder.as_mut(), bytes, |decoder, samples, output| {
                    let (sample_rate, channels) = (decoder.sample_rate()?, decoder.channels()?);
                    Some(audio_data_i16(sample_rate, channels, &output[..samples]))
                })
            }
            #[cfg(feature = "aiff")]
            FormatDecoder::Aiff(decoder) => {
                process_single_add_api(decoder.as_mut(), bytes, |d, data| d.add(data))
            }
            #[cfg(feature = "alac")]
            FormatDecoder::Alac(decoder) => {
                process_single_add_api(decoder.as_mut(), bytes, |d, data| d.add(data))
            }
            #[cfg(feature = "flac")]
            FormatDecoder::Flac(decoder) => {
                decode_i32_with_drain(decoder.as_mut(), bytes, |decoder, samples, output| {
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
                })
            }
            #[cfg(feature = "mp3")]
            FormatDecoder::Mp3(decoder) => {
                decode_i16_with_drain(decoder.as_mut(), bytes, |decoder, samples, output| {
                    let (sample_rate, channels) = (decoder.sample_rate()?, decoder.channels()?);
                    Some(audio_data_i16(sample_rate, channels, &output[..samples]))
                })
            }
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

    fn flush(&mut self) -> Result<Vec<AudioData>, String> {
        match self {
            #[cfg(feature = "aiff")]
            FormatDecoder::Aiff(decoder) => {
                process_add_api(decoder.as_mut(), &[], |d, data| d.add(data))
            }
            #[cfg(feature = "alac")]
            FormatDecoder::Alac(decoder) => {
                process_add_api(decoder.as_mut(), &[], |d, data| d.add(data))
            }
            FormatDecoder::RawPcm(decoder) => {
                decoder.flush().map(|frame| frame.into_iter().collect())
            }
            _ => self.process(&[]),
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

#[cfg(any(feature = "aiff", feature = "alac"))]
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
    frame: F,
) -> Result<Vec<AudioData>, String>
where
    D: Decoder,
    F: Fn(&D, usize, &[i16]) -> Option<AudioData>,
{
    let mut frames = Vec::new();
    let mut output = vec![0i16; DEFAULT_SCRATCH_SAMPLES];

    let samples = decoder.decode_i16(bytes, &mut output, false)?;
    if samples > 0 {
        if let Some(audio) = frame(decoder, samples, &output) {
            frames.push(audio);
        }
    }

    loop {
        let samples = decoder.decode_i16(&[], &mut output, false)?;
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
    frame: F,
) -> Result<Vec<AudioData>, String>
where
    D: Decoder,
    F: Fn(&D, usize, &[i32]) -> Option<AudioData>,
{
    let mut frames = Vec::new();
    let mut output = vec![0i32; DEFAULT_SCRATCH_SAMPLES];

    let samples = decoder.decode_i32(bytes, &mut output, false)?;
    if samples > 0 {
        if let Some(audio) = frame(decoder, samples, &output) {
            frames.push(audio);
        }
    }

    loop {
        let samples = decoder.decode_i32(&[], &mut output, false)?;
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
        #[cfg(feature = "m4a")]
        "m4a" | "mp4" | "aac-mp4" => {
            let mut decoder = AacDecoderMp4::new();
            decoder.init()?;
            Ok(FormatDecoder::M4a(Box::new(decoder)))
        }
        #[cfg(feature = "aiff")]
        "aiff" | "aifc" => {
            let mut decoder = AiffDecoder::new();
            decoder.init()?;
            Ok(FormatDecoder::Aiff(Box::new(decoder)))
        }
        #[cfg(feature = "alac")]
        "alac" | "caf-alac" => {
            let mut decoder = AlacDecoder::new();
            decoder.init()?;
            Ok(FormatDecoder::Alac(Box::new(decoder)))
        }
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
        #[cfg(feature = "m4a")]
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

#[cfg(any(feature = "video", feature = "audio-demux"))]
fn finite_i64(value: f64) -> Option<i64> {
    value.is_finite().then(|| value.round() as i64)
}

#[cfg(feature = "video")]
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
            let scaled = if sample < 0.0 {
                (sample * scale).round()
            } else {
                (sample * (scale - 1.0)).round()
            };
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
    fn alac_flush_decodes_pcm_frames() {
        let data = fixture("alac/A_Tusk_is_used_to_make_costly_gifts.m4a");
        let frames = decode_all("alac", &data, 997);
        assert!(!frames.is_empty());
        assert_eq!(frames[0].bits_per_sample(), 16);
        assert_eq!(frames[0].channel_count(), 1);
        assert_eq!(frames[0].sampling_rate(), 8_000);
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
        let data = fixture("mac_aac/A_Tusk_is_used_to_make_costly_gifts.m4a");
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
                sample_rate: 16_000,
                channels: 1,
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
