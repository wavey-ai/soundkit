use access_unit::{detect_audio, AudioType};
pub use bytes::Bytes;
use bytes::BytesMut;
use frame_header::{EncodingFlag, Endianness};
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use soundkit::audio_packet::Decoder;
use soundkit::audio_types::AudioData;
use soundkit::raw_pcm::RawPcmStreamProcessor;
pub use soundkit::raw_pcm::{RawPcmFormat, RawPcmSampleFormat};
use soundkit::wav::WavStreamProcessor;
use soundkit_aac::{AacDecoder, AacDecoderMp4};
use soundkit_aac_lc::AacLcDecoder;
use soundkit_ac3::Ac3Decoder;
use soundkit_aiff::AiffDecoder;
use soundkit_alac::{AlacDecoder, AlacPacketDecoder};
use soundkit_amr::AmrNbDecoder;
use soundkit_audio_demux::{
    AudioCodec, AudioDemuxEvent, AudioPacketFormat, AudioTrackConfig, AudioTrackDemuxer,
    CafAudioIndex, MediaTrackConfig, MediaTrackKind, Mp4MediaDemuxEvent, Mp4MediaDemuxer,
    Mp4MediaIndex, MxfMediaIndex, PcmEndianness,
};
use soundkit_flac::{FlacDecoder, FlacFrameConfig, FlacFrameDecoder, FlacProfile};
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
use soundkit_webm::{
    WebmDecoder, WebmMediaDemuxEvent, WebmMediaDemuxer, WebmMediaTrackConfig, WebmTrackKind,
};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError, TrySendError};
use std::sync::Arc;
use std::thread;
use symphonia_bundle_mp3::MpaDecoder as SymphoniaMpaDecoder;
use symphonia_codec_aac::AacDecoder as SymphoniaAacDecoder;
use symphonia_core::audio::{
    Channels as SymphoniaChannels, GenericAudioBufferRef as SymphoniaAudioBufferRef,
};
use symphonia_core::codecs::audio::well_known::{
    CODEC_ID_AAC as SYMPHONIA_CODEC_ID_AAC, CODEC_ID_MP2 as SYMPHONIA_CODEC_ID_MP2,
};
use symphonia_core::codecs::audio::{
    AudioCodecParameters as SymphoniaAudioCodecParameters, AudioDecoder as SymphoniaAudioDecoder,
    AudioDecoderOptions as SymphoniaAudioDecoderOptions,
};
use symphonia_core::packet::Packet as SymphoniaPacket;
use symphonia_core::units::{Duration as SymphoniaDuration, Timestamp as SymphoniaTimestamp};

/// Unified streaming decoder trait - all decoders implement this interface.
/// This eliminates the need for codec-specific process_* and flush_* functions.
trait StreamingDecoder {
    /// Process a chunk of input data and return decoded audio frames.
    /// An empty chunk signals EOF but does not trigger flush.
    fn process(
        &mut self,
        chunk: &[u8],
        scratch: &mut DecoderScratch,
    ) -> Result<Vec<AudioData>, String>;

    /// Flush any remaining buffered data after EOF.
    fn flush(&mut self, scratch: &mut DecoderScratch) -> Result<Vec<AudioData>, String>;
}

const MIN_DETECTION_BYTES: usize = 8192; // Increased for M4A/MP4 container detection
const MAX_DETECTION_BYTES: usize = 65_536;
const DEFAULT_INPUT_BUFFER: usize = 128;
const DEFAULT_OUTPUT_BUFFER: usize = 16;
const RESAMPLE_CHUNK_SIZE: usize = 4096;
const MAX_INPUT_CHUNK_BYTES: usize = 4 * 1024 * 1024;
const MAX_QUEUED_INPUT_BYTES: usize = 8 * 1024 * 1024;
const DECODER_SCRATCH_SAMPLES: usize = 262_144;

#[derive(Default)]
struct DecoderScratch {
    i16_samples: Vec<i16>,
    i32_samples: Vec<i32>,
}

impl DecoderScratch {
    fn i16_samples(&mut self) -> &mut [i16] {
        if self.i16_samples.len() < DECODER_SCRATCH_SAMPLES {
            self.i16_samples.resize(DECODER_SCRATCH_SAMPLES, 0);
        }
        &mut self.i16_samples
    }

    fn i32_samples(&mut self) -> &mut [i32] {
        if self.i32_samples.len() < DECODER_SCRATCH_SAMPLES {
            self.i32_samples.resize(DECODER_SCRATCH_SAMPLES, 0);
        }
        &mut self.i32_samples
    }
}

/// Error types for decode pipeline
#[derive(Debug, Clone)]
pub enum DecodeError {
    FormatDetectionFailed,
    DecoderInitFailed(String),
    DecodingFailed(String),
    InputBufferFull,
    PipelineClosed,
    InputChunkTooLarge(usize),
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
            DecodeError::PipelineClosed => write!(f, "Decode pipeline is closed"),
            DecodeError::InputChunkTooLarge(bytes) => write!(
                f,
                "Input chunk is {bytes} bytes; the streaming limit is {MAX_INPUT_CHUNK_BYTES} bytes"
            ),
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

/// PCM and descriptive metadata extracted from one complete media file.
///
/// `selected_track` is populated for seekable containers such as MOV/MP4.
/// Elementary streams and simple audio containers are decoded through the
/// incremental pipeline and therefore do not currently expose a container
/// track record here.
#[derive(Debug)]
pub struct DecodedAudioFile {
    pub metadata: soundkit::media_metadata::MediaMetadata,
    pub selected_track: Option<MediaTrackConfig>,
    pub frames: Vec<AudioData>,
}

/// Decode the first supported audio track from a complete audio or video file.
///
/// Unlike [`DecodePipeline`], this entry point can seek through an in-memory
/// source. That is required for ordinary MOV/MP4 files whose `moov` metadata
/// follows `mdat`, and for packet codecs such as ALAC and FLAC-in-MP4.
pub fn decode_audio_file(
    data: &[u8],
    options: DecodeOptions,
) -> Result<DecodedAudioFile, DecodeError> {
    if data.is_empty() {
        return Err(DecodeError::InvalidInputFormat(
            "media file is empty".to_owned(),
        ));
    }
    let mut metadata = soundkit::media_metadata::extract_metadata(data).unwrap_or_default();
    if looks_like_iso_bmff(data) {
        let (selected_track, tracks, frames) = decode_seekable_mp4_audio(data, &options)?;
        populate_mp4_track_metadata(&mut metadata, &tracks);
        return Ok(DecodedAudioFile {
            metadata,
            selected_track: Some(selected_track),
            frames,
        });
    }
    if data.starts_with(b"caff") {
        let (config, frames) = decode_seekable_caf_audio(data, &options)?;
        populate_caf_track_metadata(&mut metadata, &config, &frames);
        return Ok(DecodedAudioFile {
            metadata,
            selected_track: None,
            frames,
        });
    }
    if looks_like_avi(data) {
        let (track, frames) = decode_avi_audio(data, &options)?;
        metadata.container = Some("avi".to_owned());
        metadata.audio_tracks = vec![track];
        return Ok(DecodedAudioFile {
            metadata,
            selected_track: None,
            frames,
        });
    }
    if looks_like_mpeg_ps(data) {
        let (track, frames) = decode_mpeg_ps_audio(data, &options)?;
        metadata.container = Some("mpeg-ps".to_owned());
        metadata.audio_tracks = vec![track];
        return Ok(DecodedAudioFile {
            metadata,
            selected_track: None,
            frames,
        });
    }
    if looks_like_mpeg_ts(data) {
        let (config, frames) = decode_mpeg_ts_audio(data, &options)?;
        populate_transport_track_metadata(&mut metadata, &config, &frames);
        return Ok(DecodedAudioFile {
            metadata,
            selected_track: None,
            frames,
        });
    }
    if looks_like_mxf(data) {
        let (selected_track, tracks, frames) = decode_seekable_mxf_audio(data, &options)?;
        populate_mp4_track_metadata(&mut metadata, &tracks);
        metadata.container = Some("mxf".to_owned());
        return Ok(DecodedAudioFile {
            metadata,
            selected_track: Some(selected_track),
            frames,
        });
    }
    if looks_like_ebml(data) {
        if let Some((tracks, frames)) = decode_matroska_aac_audio(data, &options)? {
            populate_webm_track_metadata(&mut metadata, &tracks);
            return Ok(DecodedAudioFile {
                metadata,
                selected_track: None,
                frames,
            });
        }
    }

    let frames = decode_complete_stream(data, options)?;
    Ok(DecodedAudioFile {
        metadata,
        selected_track: None,
        frames,
    })
}

fn looks_like_avi(data: &[u8]) -> bool {
    data.starts_with(b"RIFF") && data.get(8..12) == Some(b"AVI ")
}

#[derive(Clone, Debug)]
struct AviAudioFormat {
    stream_index: usize,
    format_tag: u16,
    channels: u8,
    sample_rate: u32,
    bits_per_sample: u8,
}

fn decode_avi_audio(
    data: &[u8],
    options: &DecodeOptions,
) -> Result<(soundkit::media_metadata::AudioTrackMetadata, Vec<AudioData>), DecodeError> {
    let format = parse_avi_audio_format(data)?;
    let mut encoded = Vec::new();
    for_each_avi_riff(data, |kind, payload| {
        if kind == b"movi" {
            collect_avi_audio_chunks(payload, format.stream_index, &mut encoded, 0)?;
        }
        Ok(())
    })?;
    if encoded.is_empty() {
        return Err(DecodeError::InvalidInputFormat(
            "AVI audio track contains no packets".to_owned(),
        ));
    }

    let codec = match format.format_tag {
        0x0001 => "pcm",
        0x0003 => "pcm-float",
        0x0055 => "mp3",
        0x2000 => "ac3",
        0x0160..=0x0163 => "wma",
        other => {
            return Err(DecodeError::InvalidInputFormat(format!(
                "unsupported AVI audio format tag 0x{other:04x}"
            )))
        }
    };
    let frames = match format.format_tag {
        0x0055 | 0x2000 => decode_complete_stream(&encoded, *options)?,
        0x0001 | 0x0003 => decode_avi_pcm(&format, encoded, options)?,
        _ => {
            return Err(DecodeError::InvalidInputFormat(
                "AVI WMA decoding is not implemented".to_owned(),
            ))
        }
    };
    if frames.is_empty() {
        return Err(DecodeError::DecodingFailed(
            "AVI audio track emitted no frames".to_owned(),
        ));
    }
    let duration_micros = frames.first().and_then(|first| {
        let total = frames.iter().try_fold(0u64, |sum, frame| {
            decoded_audio_frame_count(frame)
                .ok()
                .and_then(|count| sum.checked_add(u64::from(count)))
        })?;
        Some(total.saturating_mul(1_000_000) / u64::from(first.sampling_rate()))
    });
    Ok((
        soundkit::media_metadata::AudioTrackMetadata {
            id: Some(format.stream_index as u64),
            codec: Some(codec.to_owned()),
            codec_id: Some(format!("0x{:04x}", format.format_tag)),
            sample_rate: Some(format.sample_rate),
            channels: Some(u16::from(format.channels)),
            bits_per_sample: Some(format.bits_per_sample),
            duration_micros,
            ..soundkit::media_metadata::AudioTrackMetadata::default()
        },
        frames,
    ))
}

fn decode_avi_pcm(
    format: &AviAudioFormat,
    encoded: Vec<u8>,
    options: &DecodeOptions,
) -> Result<Vec<AudioData>, DecodeError> {
    if format.format_tag == 0x0003 && format.bits_per_sample != 32 {
        return Err(DecodeError::InvalidInputFormat(format!(
            "unsupported AVI float PCM depth {}",
            format.bits_per_sample
        )));
    }
    if format.format_tag == 0x0001 && !matches!(format.bits_per_sample, 8 | 16 | 24 | 32) {
        return Err(DecodeError::InvalidInputFormat(format!(
            "unsupported AVI integer PCM depth {}",
            format.bits_per_sample
        )));
    }
    let bytes_per_frame = usize::from(format.channels)
        .checked_mul(usize::from(format.bits_per_sample.div_ceil(8)))
        .ok_or_else(|| DecodeError::InvalidInputFormat("AVI PCM frame size overflow".to_owned()))?;
    if bytes_per_frame == 0 || encoded.len() % bytes_per_frame != 0 {
        return Err(DecodeError::InvalidInputFormat(
            "AVI PCM ends with a partial sample frame".to_owned(),
        ));
    }
    let audio = if format.bits_per_sample == 8 {
        let samples = encoded
            .into_iter()
            .map(|sample| (i16::from(sample) - 128) << 8)
            .collect::<Vec<_>>();
        create_audio_data_i16(format.sample_rate, format.channels, &samples)
    } else {
        AudioData::new(
            format.bits_per_sample,
            format.channels,
            format.sample_rate,
            encoded,
            if format.format_tag == 0x0003 {
                EncodingFlag::PCMFloat
            } else {
                EncodingFlag::PCMSigned
            },
            Endianness::LittleEndian,
        )
    };
    let mut resampler = None;
    let mut frames = apply_output_options(audio, options, &mut resampler)?;
    if let Some(pending) = resampler {
        frames.extend(flush_resampler_frames(pending)?);
    }
    Ok(frames)
}

fn parse_avi_audio_format(data: &[u8]) -> Result<AviAudioFormat, DecodeError> {
    let mut format = None;
    let mut stream_index = 0usize;
    for_each_avi_riff(data, |kind, payload| {
        if kind == b"hdrl" && format.is_none() {
            for_each_riff_chunk(payload, |id, chunk| {
                if id == b"LIST" && chunk.get(..4) == Some(b"strl") {
                    let this_stream = stream_index;
                    stream_index += 1;
                    if format.is_none() {
                        format = parse_avi_stream_list(&chunk[4..], this_stream)?;
                    }
                }
                Ok(())
            })?;
        }
        Ok(())
    })?;
    format.ok_or_else(|| DecodeError::InvalidInputFormat("AVI has no audio stream".to_owned()))
}

fn parse_avi_stream_list(
    bytes: &[u8],
    stream_index: usize,
) -> Result<Option<AviAudioFormat>, String> {
    let mut audio = false;
    let mut wave = None;
    for_each_riff_chunk(bytes, |id, payload| {
        if id == b"strh" {
            audio = payload.get(..4) == Some(b"auds");
        } else if id == b"strf" {
            wave = Some(payload.to_vec());
        }
        Ok(())
    })?;
    if !audio {
        return Ok(None);
    }
    let wave = wave.ok_or_else(|| "AVI audio stream has no strf format".to_owned())?;
    if wave.len() < 16 {
        return Err("AVI WAVEFORMAT is truncated".to_owned());
    }
    let channels = u16::from_le_bytes(wave[2..4].try_into().unwrap());
    let channels = u8::try_from(channels)
        .ok()
        .filter(|channels| *channels != 0)
        .ok_or_else(|| "AVI channel count is invalid".to_owned())?;
    let sample_rate = u32::from_le_bytes(wave[4..8].try_into().unwrap());
    if sample_rate == 0 {
        return Err("AVI sample rate is zero".to_owned());
    }
    Ok(Some(AviAudioFormat {
        stream_index,
        format_tag: u16::from_le_bytes(wave[..2].try_into().unwrap()),
        channels,
        sample_rate,
        bits_per_sample: u16::from_le_bytes(wave[14..16].try_into().unwrap()).min(255) as u8,
    }))
}

fn for_each_avi_riff(
    bytes: &[u8],
    mut visit_list: impl FnMut(&[u8; 4], &[u8]) -> Result<(), String>,
) -> Result<(), DecodeError> {
    let mut offset = 0usize;
    let mut forms = 0usize;
    while offset + 12 <= bytes.len() {
        if bytes.get(offset..offset + 4) != Some(b"RIFF") {
            if forms != 0 {
                break;
            }
            return Err(DecodeError::InvalidInputFormat(
                "AVI contains data outside a RIFF form".to_owned(),
            ));
        }
        forms += 1;
        let length = u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().unwrap()) as usize;
        let declared_end = offset
            .checked_add(8)
            .and_then(|start| start.checked_add(length))
            .ok_or_else(|| {
                DecodeError::InvalidInputFormat("AVI RIFF form size overflows".to_owned())
            })?;
        // FATE contains deliberately prefix-truncated AVI samples. Parse the
        // complete chunks present in such a prefix, but never manufacture the
        // incomplete final packet.
        let end = declared_end.min(bytes.len());
        if !matches!(bytes.get(offset + 8..offset + 12), Some(b"AVI " | b"AVIX")) {
            return Err(DecodeError::InvalidInputFormat(
                "invalid AVI RIFF form type".to_owned(),
            ));
        }
        for_each_riff_chunk(&bytes[offset + 12..end], |id, payload| {
            if id == b"LIST" {
                let kind: &[u8; 4] = payload
                    .get(..4)
                    .ok_or_else(|| "AVI LIST type is truncated".to_owned())?
                    .try_into()
                    .unwrap();
                visit_list(kind, &payload[4..])?;
            }
            Ok(())
        })
        .map_err(DecodeError::InvalidInputFormat)?;
        if declared_end > bytes.len() {
            break;
        }
        offset = end + (length & 1);
    }
    Ok(())
}

fn for_each_riff_chunk(
    bytes: &[u8],
    mut visit: impl FnMut(&[u8; 4], &[u8]) -> Result<(), String>,
) -> Result<(), String> {
    let mut offset = 0usize;
    let mut chunks = 0usize;
    while offset + 8 <= bytes.len() {
        chunks += 1;
        if chunks > 1_000_000 {
            return Err("AVI chunk count exceeds budget".to_owned());
        }
        let id: &[u8; 4] = bytes[offset..offset + 4].try_into().unwrap();
        let length = u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().unwrap()) as usize;
        let start = offset + 8;
        let declared_end = start
            .checked_add(length)
            .ok_or_else(|| "AVI chunk size overflows".to_owned())?;
        if declared_end > bytes.len() {
            // A truncated LIST can still contain many complete media chunks.
            // Expose its available prefix; truncated leaf chunks are skipped.
            if id == b"LIST" && bytes.len().saturating_sub(start) >= 4 {
                visit(id, &bytes[start..])?;
            }
            return Ok(());
        }
        visit(id, &bytes[start..declared_end])?;
        offset = declared_end + (length & 1);
    }
    if offset != bytes.len() && !bytes[offset..].iter().all(|byte| *byte == 0) {
        return Err("AVI chunk table has trailing data".to_owned());
    }
    Ok(())
}

fn collect_avi_audio_chunks(
    bytes: &[u8],
    stream_index: usize,
    output: &mut Vec<u8>,
    depth: usize,
) -> Result<(), String> {
    if depth > 8 {
        return Err("AVI record nesting exceeds budget".to_owned());
    }
    for_each_riff_chunk(bytes, |id, payload| {
        if id == b"LIST" && payload.len() >= 4 {
            collect_avi_audio_chunks(&payload[4..], stream_index, output, depth + 1)?;
        } else if &id[2..] == b"wb" && avi_stream_id(&id[..2]) == Some(stream_index) {
            output
                .len()
                .checked_add(payload.len())
                .filter(|length| *length <= 512 * 1024 * 1024)
                .ok_or_else(|| "AVI audio payload exceeds budget".to_owned())?;
            output.extend_from_slice(payload);
        }
        Ok(())
    })
}

fn avi_stream_id(bytes: &[u8]) -> Option<usize> {
    bytes.iter().try_fold(0usize, |value, byte| {
        let digit = match byte {
            b'0'..=b'9' => usize::from(byte - b'0'),
            b'A'..=b'F' => usize::from(byte - b'A' + 10),
            b'a'..=b'f' => usize::from(byte - b'a' + 10),
            _ => return None,
        };
        Some(value * 16 + digit)
    })
}

fn looks_like_mpeg_ps(data: &[u8]) -> bool {
    data.starts_with(b"\0\0\x01\xba")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DvdLpcmFormat {
    sample_rate: u32,
    channels: u8,
    bits_per_sample: u8,
}

fn decode_mpeg_ps_audio(
    data: &[u8],
    options: &DecodeOptions,
) -> Result<(soundkit::media_metadata::AudioTrackMetadata, Vec<AudioData>), DecodeError> {
    let mut format = None;
    let mut packed = Vec::new();
    let mut offset = 0usize;
    while let Some(start) = find_start_code(data, offset) {
        let code = data[start + 3];
        if code == 0xbd {
            let header = data.get(start + 4..start + 6).ok_or_else(|| {
                DecodeError::InvalidInputFormat("MPEG-PS PES length is truncated".to_owned())
            })?;
            let length = u16::from_be_bytes(header.try_into().unwrap()) as usize;
            let end = (start + 6)
                .checked_add(length)
                .filter(|end| *end <= data.len())
                .ok_or_else(|| {
                    DecodeError::InvalidInputFormat("MPEG-PS PES packet is truncated".to_owned())
                })?;
            if let Some(payload) = parse_mpeg_pes_payload(&data[start + 6..end])? {
                if matches!(payload.first(), Some(0xa0..=0xaf)) && payload.len() >= 7 {
                    // private_stream_1: substream id + 3-byte DVD substream
                    // header, followed by the decoder's 3-byte LPCM header.
                    let packet = &payload[4..];
                    let packet_format = parse_dvd_lpcm_header(&packet[..3])?;
                    if let Some(existing) = format {
                        if existing != packet_format {
                            return Err(DecodeError::InvalidInputFormat(
                                "DVD LPCM format changes mid-stream".to_owned(),
                            ));
                        }
                    } else {
                        format = Some(packet_format);
                    }
                    packed
                        .len()
                        .checked_add(packet.len() - 3)
                        .filter(|length| *length <= 512 * 1024 * 1024)
                        .ok_or_else(|| {
                            DecodeError::InvalidInputFormat(
                                "MPEG-PS audio payload exceeds budget".to_owned(),
                            )
                        })?;
                    packed.extend_from_slice(&packet[3..]);
                }
            }
            offset = end;
        } else {
            offset = start + 4;
        }
    }
    let format = format.ok_or_else(|| {
        DecodeError::InvalidInputFormat("MPEG-PS has no supported DVD LPCM track".to_owned())
    })?;
    let pcm = unpack_dvd_lpcm(&packed, format)?;
    let audio = AudioData::new(
        format.bits_per_sample,
        format.channels,
        format.sample_rate,
        pcm,
        EncodingFlag::PCMSigned,
        Endianness::LittleEndian,
    );
    let mut resampler = None;
    let mut frames = apply_output_options(audio, options, &mut resampler)?;
    if let Some(pending) = resampler {
        frames.extend(flush_resampler_frames(pending)?);
    }
    if frames.is_empty() {
        return Err(DecodeError::DecodingFailed(
            "MPEG-PS DVD LPCM track emitted no frames".to_owned(),
        ));
    }
    let total_frames = frames.iter().try_fold(0u64, |sum, frame| {
        decoded_audio_frame_count(frame)
            .ok()
            .and_then(|count| sum.checked_add(u64::from(count)))
    });
    Ok((
        soundkit::media_metadata::AudioTrackMetadata {
            codec: Some("pcm-dvd".to_owned()),
            codec_id: Some("private_stream_1/lpcm".to_owned()),
            sample_rate: Some(format.sample_rate),
            channels: Some(u16::from(format.channels)),
            bits_per_sample: Some(format.bits_per_sample),
            duration_micros: total_frames
                .map(|frames| frames.saturating_mul(1_000_000) / u64::from(format.sample_rate)),
            ..soundkit::media_metadata::AudioTrackMetadata::default()
        },
        frames,
    ))
}

fn find_start_code(bytes: &[u8], offset: usize) -> Option<usize> {
    bytes
        .get(offset..)?
        .windows(3)
        .position(|window| window == b"\0\0\x01")
        .map(|position| offset + position)
        .filter(|start| start + 3 < bytes.len())
}

fn parse_mpeg_pes_payload(packet: &[u8]) -> Result<Option<&[u8]>, DecodeError> {
    let mut cursor = 0usize;
    while packet.get(cursor) == Some(&0xff) {
        cursor += 1;
    }
    let Some(&first) = packet.get(cursor) else {
        return Ok(None);
    };
    if first & 0xc0 == 0x80 {
        let header = packet.get(cursor..cursor + 3).ok_or_else(|| {
            DecodeError::InvalidInputFormat("MPEG-2 PES header is truncated".to_owned())
        })?;
        cursor = cursor
            .checked_add(3 + usize::from(header[2]))
            .filter(|cursor| *cursor <= packet.len())
            .ok_or_else(|| {
                DecodeError::InvalidInputFormat(
                    "MPEG-2 PES optional header is truncated".to_owned(),
                )
            })?;
    } else {
        if first & 0xc0 == 0x40 {
            cursor = cursor.checked_add(2).ok_or_else(|| {
                DecodeError::InvalidInputFormat("MPEG-1 PES header overflows".to_owned())
            })?;
        }
        let first = *packet.get(cursor).ok_or_else(|| {
            DecodeError::InvalidInputFormat("MPEG-1 PES header is truncated".to_owned())
        })?;
        cursor = match first & 0xf0 {
            0x20 => cursor + 5,
            0x30 => cursor + 10,
            _ if first == 0x0f => cursor + 1,
            _ => return Ok(None),
        };
        if cursor > packet.len() {
            return Err(DecodeError::InvalidInputFormat(
                "MPEG-1 PES timestamp is truncated".to_owned(),
            ));
        }
    }
    Ok(Some(&packet[cursor..]))
}

fn parse_dvd_lpcm_header(header: &[u8]) -> Result<DvdLpcmFormat, DecodeError> {
    if header.len() != 3 {
        return Err(DecodeError::InvalidInputFormat(
            "DVD LPCM header is truncated".to_owned(),
        ));
    }
    let bits_per_sample = 16 + ((header[1] >> 6) & 3) * 4;
    if !matches!(bits_per_sample, 16 | 20 | 24) {
        return Err(DecodeError::InvalidInputFormat(format!(
            "unsupported DVD LPCM sample depth {bits_per_sample}"
        )));
    }
    let sample_rate = [48_000, 96_000, 44_100, 32_000][usize::from((header[1] >> 4) & 3)];
    Ok(DvdLpcmFormat {
        sample_rate,
        channels: (header[1] & 7) + 1,
        bits_per_sample,
    })
}

fn unpack_dvd_lpcm(packed: &[u8], format: DvdLpcmFormat) -> Result<Vec<u8>, DecodeError> {
    let channels = usize::from(format.channels);
    if format.bits_per_sample == 16 {
        let block_size = channels * 2;
        if packed.len() % block_size != 0 {
            return Err(DecodeError::InvalidInputFormat(
                "DVD LPCM ends with a partial sample block".to_owned(),
            ));
        }
        let mut output = Vec::with_capacity(packed.len());
        for sample in packed.chunks_exact(2) {
            output.extend_from_slice(&[sample[1], sample[0]]);
        }
        return Ok(output);
    }
    if format.bits_per_sample == 20 {
        return Err(DecodeError::InvalidInputFormat(
            "20-bit DVD LPCM decoding is not implemented".to_owned(),
        ));
    }

    let (groups_per_block, samples_per_block) = match channels {
        1 | 2 | 4 => (1usize, 4 / channels),
        8 => (2, 1),
        _ => (channels, 4),
    };
    let block_size = if matches!(channels, 1 | 2 | 4) {
        12
    } else if channels == 8 {
        24
    } else {
        channels * 12
    };
    if packed.len() % block_size != 0 {
        return Err(DecodeError::InvalidInputFormat(
            "DVD LPCM ends with a partial 24-bit sample block".to_owned(),
        ));
    }
    let samples_per_group = 4usize;
    let samples_per_output_block = channels * samples_per_block;
    let mut output = Vec::with_capacity(packed.len());
    for block in packed.chunks_exact(block_size) {
        let mut cursor = 0usize;
        for _ in 0..groups_per_block {
            let high = &block[cursor..cursor + samples_per_group * 2];
            let low = &block[cursor + samples_per_group * 2..cursor + samples_per_group * 3];
            for sample in 0..samples_per_group {
                output.extend_from_slice(&[low[sample], high[sample * 2 + 1], high[sample * 2]]);
            }
            cursor += samples_per_group * 3;
        }
        debug_assert_eq!(output.len() % (samples_per_output_block * 3), 0);
    }
    Ok(output)
}

fn looks_like_mxf(data: &[u8]) -> bool {
    data.windows(4)
        .take(65_537)
        .any(|window| window == [0x06, 0x0e, 0x2b, 0x34])
}

fn decode_seekable_mxf_audio(
    data: &[u8],
    options: &DecodeOptions,
) -> Result<(MediaTrackConfig, Vec<MediaTrackConfig>, Vec<AudioData>), DecodeError> {
    let index = MxfMediaIndex::from_file(data).map_err(DecodeError::InvalidInputFormat)?;
    let track = index
        .tracks
        .iter()
        .find(|track| track.kind == MediaTrackKind::Audio && track.codec == "pcm")
        .cloned()
        .ok_or_else(|| {
            DecodeError::InvalidInputFormat("MXF has no supported PCM audio track".to_owned())
        })?;
    let mut frames = Vec::new();
    let mut resampler = None;
    for sample in index
        .samples
        .iter()
        .filter(|sample| sample.kind == MediaTrackKind::Audio && sample.track_id == track.track_id)
    {
        let packet = index
            .sample_data(data, sample)
            .map_err(DecodeError::InvalidInputFormat)?;
        let frame = make_container_pcm_audio(&track, packet)?;
        frames.extend(apply_output_options(frame, options, &mut resampler)?);
    }
    if let Some(pending) = resampler.take() {
        frames.extend(flush_resampler_frames(pending)?);
    }
    if frames.is_empty() {
        return Err(DecodeError::DecodingFailed(
            "MXF PCM track emitted no audio frames".to_owned(),
        ));
    }
    Ok((track, index.tracks, frames))
}

fn looks_like_mpeg_ts(data: &[u8]) -> bool {
    [(188_usize, 0_usize), (192, 4)]
        .into_iter()
        .any(|(stride, prefix)| {
            (0..3).all(|packet| data.get(prefix + packet * stride) == Some(&0x47))
        })
}

enum TransportAudioDecoder {
    Aac(Option<AacLcDecoder>),
    Mp2(WaveyMp2Decoder),
    Pcm,
}

struct WaveyMp2Decoder {
    decoder: SymphoniaMpaDecoder,
}

impl WaveyMp2Decoder {
    fn new(config: &AudioTrackConfig) -> Result<Self, DecodeError> {
        let mut params = SymphoniaAudioCodecParameters::new();
        params.for_codec(SYMPHONIA_CODEC_ID_MP2);
        if let Some(sample_rate) = config.sample_rate {
            params.with_sample_rate(sample_rate);
        }
        if let Some(channels) = config.channels {
            params.with_channels(SymphoniaChannels::Discrete(u16::from(channels)));
        }
        let decoder =
            SymphoniaMpaDecoder::try_new(&params, &SymphoniaAudioDecoderOptions::default())
                .map_err(|error| DecodeError::DecoderInitFailed(error.to_string()))?;
        Ok(Self { decoder })
    }

    fn decode_frame(&mut self, frame: &[u8]) -> Result<AudioData, DecodeError> {
        let packet = SymphoniaPacket::new(
            0,
            SymphoniaTimestamp::ZERO,
            SymphoniaDuration::ZERO,
            frame.to_vec(),
        );
        let decoded = self
            .decoder
            .decode(&packet)
            .map_err(|error| DecodeError::DecodingFailed(error.to_string()))?;
        symphonia_audio_to_i16(decoded)
    }
}

fn decode_mpeg_ts_audio(
    data: &[u8],
    options: &DecodeOptions,
) -> Result<(AudioTrackConfig, Vec<AudioData>), DecodeError> {
    let mut demuxer =
        AudioTrackDemuxer::new_with_format("mpeg-ts").map_err(DecodeError::DecoderInitFailed)?;
    let mut config = None;
    let mut decoder = None;
    let mut frames = Vec::new();
    let mut resampler = None;
    for chunk in data.chunks(64 * 1024) {
        let events = demuxer
            .push(chunk)
            .map_err(DecodeError::InvalidInputFormat)?;
        consume_mpeg_ts_events(
            events,
            options,
            &mut config,
            &mut decoder,
            &mut frames,
            &mut resampler,
        )?;
    }
    let events = demuxer.flush().map_err(DecodeError::InvalidInputFormat)?;
    consume_mpeg_ts_events(
        events,
        options,
        &mut config,
        &mut decoder,
        &mut frames,
        &mut resampler,
    )?;
    if let Some(pending) = resampler.take() {
        frames.extend(flush_resampler_frames(pending)?);
    }
    let config = config.ok_or_else(|| {
        DecodeError::InvalidInputFormat("MPEG-TS has no supported audio track".to_owned())
    })?;
    if frames.is_empty() {
        return Err(DecodeError::DecodingFailed(format!(
            "MPEG-TS {} track emitted no audio frames",
            config.codec.as_str()
        )));
    }
    Ok((config, frames))
}

#[allow(clippy::too_many_arguments)]
fn consume_mpeg_ts_events(
    events: Vec<AudioDemuxEvent>,
    options: &DecodeOptions,
    selected: &mut Option<AudioTrackConfig>,
    decoder: &mut Option<TransportAudioDecoder>,
    frames: &mut Vec<AudioData>,
    resampler: &mut Option<StreamingResampler>,
) -> Result<(), DecodeError> {
    for event in events {
        match event {
            AudioDemuxEvent::Config(config) if selected.is_none() => {
                *decoder = match config.codec {
                    AudioCodec::Aac if config.packet_format == Some(AudioPacketFormat::Adts) => {
                        Some(TransportAudioDecoder::Aac(None))
                    }
                    AudioCodec::Unknown(ref codec) if codec == "mpeg-audio" || codec == "mp2" => {
                        Some(TransportAudioDecoder::Mp2(WaveyMp2Decoder::new(&config)?))
                    }
                    AudioCodec::Pcm => Some(TransportAudioDecoder::Pcm),
                    _ => None,
                };
                if decoder.is_some() {
                    *selected = Some(config);
                } else {
                    return Err(DecodeError::InvalidInputFormat(format!(
                        "unsupported MPEG-TS audio codec {}",
                        config.codec.as_str()
                    )));
                }
            }
            AudioDemuxEvent::Packet(packet) => {
                let Some(config) = selected.as_ref() else {
                    continue;
                };
                let state = decoder.as_mut().ok_or_else(|| {
                    DecodeError::DecoderInitFailed(
                        "MPEG-TS audio decoder is not initialized".to_owned(),
                    )
                })?;
                let frame = match state {
                    TransportAudioDecoder::Aac(decoder) => {
                        let (asc, access_unit) = parse_adts_access_unit(&packet.data)?;
                        if decoder.is_none() {
                            *decoder =
                                Some(AacLcDecoder::from_audio_specific_config(&asc).map_err(
                                    |error| DecodeError::DecoderInitFailed(error.to_string()),
                                )?);
                        }
                        decode_aac_access_unit(decoder.as_mut().unwrap(), access_unit)?
                    }
                    TransportAudioDecoder::Mp2(decoder) => decoder.decode_frame(&packet.data)?,
                    TransportAudioDecoder::Pcm => make_audio_track_pcm_audio(config, packet.data)?,
                };
                frames.extend(apply_output_options(frame, options, resampler)?);
            }
            _ => {}
        }
    }
    Ok(())
}

fn parse_adts_access_unit(data: &[u8]) -> Result<([u8; 2], &[u8]), DecodeError> {
    if data.len() < 7 || data[0] != 0xff || data[1] & 0xf6 != 0xf0 {
        return Err(DecodeError::InvalidInputFormat(
            "invalid ADTS access unit".to_owned(),
        ));
    }
    let audio_object_type = ((data[2] >> 6) & 0x03) + 1;
    let sample_rate_index = (data[2] >> 2) & 0x0f;
    let channels = ((data[2] & 1) << 2) | (data[3] >> 6);
    let header_bytes = if data[1] & 1 != 0 { 7 } else { 9 };
    if data.len() < header_bytes {
        return Err(DecodeError::InvalidInputFormat(
            "truncated ADTS header".to_owned(),
        ));
    }
    let asc = [
        (audio_object_type << 3) | (sample_rate_index >> 1),
        ((sample_rate_index & 1) << 7) | (channels << 3),
    ];
    Ok((asc, &data[header_bytes..]))
}

fn decode_seekable_caf_audio(
    data: &[u8],
    options: &DecodeOptions,
) -> Result<(AudioTrackConfig, Vec<AudioData>), DecodeError> {
    let index = CafAudioIndex::from_file(data).map_err(DecodeError::InvalidInputFormat)?;
    let mut alac = match index.config.codec {
        AudioCodec::Alac => Some(
            AlacPacketDecoder::new(&index.config.codec_private)
                .map_err(DecodeError::DecoderInitFailed)?,
        ),
        AudioCodec::Pcm => None,
        ref codec => {
            return Err(DecodeError::InvalidInputFormat(format!(
                "unsupported CAF audio codec {}",
                codec.as_str()
            )))
        }
    };
    let mut frames = Vec::new();
    let mut resampler = None;
    for (sample_index, sample) in index.packets.iter().enumerate() {
        let start = usize::try_from(sample.absolute_offset).map_err(|_| {
            DecodeError::InvalidInputFormat("CAF packet offset exceeds this platform".to_owned())
        })?;
        let end = start.checked_add(sample.size as usize).ok_or_else(|| {
            DecodeError::InvalidInputFormat("CAF packet byte range overflow".to_owned())
        })?;
        let source = data.get(start..end).ok_or_else(|| {
            DecodeError::InvalidInputFormat(format!(
                "CAF packet {sample_index} extends past the source"
            ))
        })?;
        let packet = index
            .packet_from_sample_bytes(sample_index, source)
            .map_err(DecodeError::InvalidInputFormat)?;
        let frame = match alac.as_mut() {
            Some(decoder) => decoder
                .decode_packet(&packet.data)
                .map_err(DecodeError::DecodingFailed)?,
            None => make_audio_track_pcm_audio(&index.config, packet.data)?,
        };
        let decoded_frames = decoded_audio_frame_count(&frame)?;
        let Some(trim) = index
            .pcm_packet_trim(sample_index, decoded_frames)
            .map_err(DecodeError::InvalidInputFormat)?
        else {
            continue;
        };
        let Some(frame) = trim_interleaved_audio(frame, trim.source_frame_start, trim.frame_count)?
        else {
            continue;
        };
        frames.extend(apply_output_options(frame, options, &mut resampler)?);
    }
    if let Some(pending) = resampler.take() {
        frames.extend(flush_resampler_frames(pending)?);
    }
    if frames.is_empty() {
        return Err(DecodeError::DecodingFailed(
            "CAF track emitted no audio frames".to_owned(),
        ));
    }
    Ok((index.config, frames))
}

fn make_audio_track_pcm_audio(
    config: &AudioTrackConfig,
    data: Vec<u8>,
) -> Result<AudioData, DecodeError> {
    let sample_rate = config.sample_rate.ok_or_else(|| {
        DecodeError::InvalidInputFormat("container PCM has no sample rate".to_owned())
    })?;
    let channels = config.channels.ok_or_else(|| {
        DecodeError::InvalidInputFormat("container PCM has no channel count".to_owned())
    })?;
    let bits = config.bits_per_sample.ok_or_else(|| {
        DecodeError::InvalidInputFormat("container PCM has no sample depth".to_owned())
    })?;
    if !matches!(
        (config.pcm_float, bits),
        (Some(false), 16 | 24 | 32) | (Some(true), 32)
    ) {
        return Err(DecodeError::InvalidInputFormat(format!(
            "unsupported container PCM format: float={} bits={bits}",
            config.pcm_float.unwrap_or(false)
        )));
    }
    Ok(AudioData::new(
        bits,
        channels,
        sample_rate,
        data,
        if config.pcm_float == Some(true) {
            EncodingFlag::PCMFloat
        } else {
            EncodingFlag::PCMSigned
        },
        if config.pcm_endianness == Some(PcmEndianness::Big) {
            Endianness::BigEndian
        } else {
            Endianness::LittleEndian
        },
    ))
}

fn populate_transport_track_metadata(
    metadata: &mut soundkit::media_metadata::MediaMetadata,
    config: &AudioTrackConfig,
    frames: &[AudioData],
) {
    metadata.container = Some(config.container.as_str().to_owned());
    populate_audio_track_metadata(metadata, config, frames);
}

fn populate_caf_track_metadata(
    metadata: &mut soundkit::media_metadata::MediaMetadata,
    config: &AudioTrackConfig,
    frames: &[AudioData],
) {
    metadata.container = Some("caf".to_owned());
    populate_audio_track_metadata(metadata, config, frames);
}

fn populate_audio_track_metadata(
    metadata: &mut soundkit::media_metadata::MediaMetadata,
    config: &AudioTrackConfig,
    frames: &[AudioData],
) {
    let duration_micros = frames.first().and_then(|first| {
        let total_frames = frames.iter().try_fold(0_u64, |total, frame| {
            decoded_audio_frame_count(frame)
                .ok()
                .and_then(|count| total.checked_add(u64::from(count)))
        })?;
        Some(total_frames.saturating_mul(1_000_000) / u64::from(first.sampling_rate()))
    });
    metadata.audio_tracks = vec![soundkit::media_metadata::AudioTrackMetadata {
        id: config.track_id,
        codec: Some(config.codec.as_str().to_owned()),
        codec_id: config.codec_id.clone(),
        sample_rate: config.sample_rate,
        channels: config.channels.map(u16::from),
        bits_per_sample: config.bits_per_sample,
        duration_micros,
        ..soundkit::media_metadata::AudioTrackMetadata::default()
    }];
}

fn looks_like_ebml(data: &[u8]) -> bool {
    data.starts_with(&[0x1a, 0x45, 0xdf, 0xa3])
}

fn decode_matroska_aac_audio(
    data: &[u8],
    options: &DecodeOptions,
) -> Result<Option<(Vec<WebmMediaTrackConfig>, Vec<AudioData>)>, DecodeError> {
    let mut demuxer = WebmMediaDemuxer::new();
    let mut tracks = Vec::new();
    let mut selected_track_number = None;
    let mut decoder = None;
    let mut frames = Vec::new();
    let mut resampler = None;

    for chunk in data.chunks(64 * 1024) {
        let events = demuxer
            .add(chunk)
            .map_err(DecodeError::InvalidInputFormat)?;
        consume_matroska_aac_events(
            events,
            options,
            &mut tracks,
            &mut selected_track_number,
            &mut decoder,
            &mut frames,
            &mut resampler,
        )?;
    }
    let events = demuxer.finish().map_err(DecodeError::InvalidInputFormat)?;
    consume_matroska_aac_events(
        events,
        options,
        &mut tracks,
        &mut selected_track_number,
        &mut decoder,
        &mut frames,
        &mut resampler,
    )?;

    let Some(track_number) = selected_track_number else {
        return Ok(None);
    };
    if let Some(pending) = resampler.take() {
        frames.extend(flush_resampler_frames(pending)?);
    }
    if frames.is_empty() {
        return Err(DecodeError::DecodingFailed(format!(
            "Matroska AAC track {track_number} emitted no audio frames"
        )));
    }
    Ok(Some((tracks, frames)))
}

#[allow(clippy::too_many_arguments)]
fn consume_matroska_aac_events(
    events: Vec<WebmMediaDemuxEvent>,
    options: &DecodeOptions,
    tracks: &mut Vec<WebmMediaTrackConfig>,
    selected_track_number: &mut Option<u64>,
    decoder: &mut Option<WaveyAacDecoder>,
    frames: &mut Vec<AudioData>,
    resampler: &mut Option<StreamingResampler>,
) -> Result<(), DecodeError> {
    for event in events {
        match event {
            WebmMediaDemuxEvent::Config { track, .. } => {
                if selected_track_number.is_none()
                    && track.kind == WebmTrackKind::Audio
                    && track.codec_id == "A_AAC"
                {
                    *decoder = Some(WaveyAacDecoder::new(
                        &track.codec_private,
                        track.sample_rate,
                        track.channels,
                    )?);
                    *selected_track_number = Some(track.track_number);
                }
                if !tracks
                    .iter()
                    .any(|known| known.track_number == track.track_number)
                {
                    tracks.push(track);
                }
            }
            WebmMediaDemuxEvent::Packet {
                track_number,
                kind: WebmTrackKind::Audio,
                codec_id,
                data,
                ..
            } if selected_track_number == &Some(track_number) && codec_id == "A_AAC" => {
                let decoder = decoder.as_mut().ok_or_else(|| {
                    DecodeError::DecoderInitFailed(
                        "Matroska AAC decoder is not initialized".to_owned(),
                    )
                })?;
                let frame = decoder.decode_access_unit(&data)?;
                frames.extend(apply_output_options(frame, options, resampler)?);
            }
            _ => {}
        }
    }
    Ok(())
}

fn looks_like_iso_bmff(data: &[u8]) -> bool {
    let mut position = 0usize;
    let limit = data.len().min(64 * 1024);
    for _ in 0..16 {
        let Some(header) = data.get(position..position.saturating_add(8)) else {
            return false;
        };
        let size = u32::from_be_bytes(header[..4].try_into().unwrap()) as usize;
        let box_type = &header[4..8];
        match box_type {
            b"ftyp" | b"moov" | b"mdat" => return true,
            b"free" | b"wide" | b"skip" => {}
            _ => return false,
        }
        if size < 8 || position.saturating_add(size) > limit {
            return false;
        }
        position += size;
    }
    false
}

fn decode_complete_stream(
    data: &[u8],
    options: DecodeOptions,
) -> Result<Vec<AudioData>, DecodeError> {
    let mut pipeline = DecodePipeline::spawn_with_buffers_and_options(256, 4_096, options);
    let mut frames = Vec::new();
    for chunk in data.chunks(64 * 1024) {
        let bytes = Bytes::copy_from_slice(chunk);
        loop {
            match pipeline.send(bytes.clone()) {
                Ok(()) => break,
                Err(DecodeError::InputBufferFull) => {
                    while let Some(frame) = pipeline.try_recv() {
                        frames.push(frame?);
                    }
                    thread::yield_now();
                }
                Err(error) => return Err(error),
            }
        }
        while let Some(frame) = pipeline.try_recv() {
            frames.push(frame?);
        }
    }
    loop {
        match pipeline.finish() {
            Ok(()) => break,
            Err(DecodeError::InputBufferFull) => {
                while let Some(frame) = pipeline.try_recv() {
                    frames.push(frame?);
                }
                thread::yield_now();
            }
            Err(error) => return Err(error),
        }
    }
    while let Some(frame) = pipeline.recv() {
        frames.push(frame?);
    }
    if frames.is_empty() {
        return Err(DecodeError::DecodingFailed(
            "SoundKit emitted no audio frames".to_owned(),
        ));
    }
    Ok(frames)
}

enum SeekableAudioDecoder {
    WaveyAac(WaveyAacDecoder),
    Alac(AlacPacketDecoder),
    Flac(FlacPacketDecoder),
    Pcm,
}

struct FlacPacketDecoder {
    decoder: FlacFrameDecoder,
    scratch: Vec<i32>,
    sample_rate: u32,
    channels: u8,
    bits_per_sample: u8,
}

impl FlacPacketDecoder {
    fn new(track: &MediaTrackConfig) -> Result<Self, DecodeError> {
        let mut metadata = FlacDecoder::new();
        metadata.init().map_err(DecodeError::DecoderInitFailed)?;
        let written = metadata
            .decode_i32(&track.decoder_configuration, &mut [], false)
            .map_err(DecodeError::DecoderInitFailed)?;
        if written != 0 {
            return Err(DecodeError::DecoderInitFailed(
                "MP4 FLAC configuration unexpectedly contained audio frames".to_owned(),
            ));
        }
        let sample_rate = metadata.sample_rate().ok_or_else(|| {
            DecodeError::DecoderInitFailed("MP4 FLAC has no sample rate".to_owned())
        })?;
        let channels = metadata.channels().ok_or_else(|| {
            DecodeError::DecoderInitFailed("MP4 FLAC has no channel count".to_owned())
        })?;
        let bits_per_sample = metadata.bits_per_sample().ok_or_else(|| {
            DecodeError::DecoderInitFailed("MP4 FLAC has no sample depth".to_owned())
        })?;
        let frame_length = metadata.maximum_block_size().ok_or_else(|| {
            DecodeError::DecoderInitFailed("MP4 FLAC has no maximum block size".to_owned())
        })?;
        let config = FlacFrameConfig::new(
            sample_rate,
            u16::from(channels),
            bits_per_sample,
            u32::from(frame_length),
            FlacProfile::Realtime,
        )
        .map_err(|error| DecodeError::DecoderInitFailed(error.to_string()))?;
        let sample_capacity = config
            .sample_count()
            .map_err(|error| DecodeError::DecoderInitFailed(error.to_string()))?;
        Ok(Self {
            decoder: FlacFrameDecoder::new(config)
                .map_err(|error| DecodeError::DecoderInitFailed(error.to_string()))?,
            scratch: vec![0; sample_capacity],
            sample_rate,
            channels,
            bits_per_sample,
        })
    }

    fn decode(&mut self, packet: &[u8]) -> Result<AudioData, DecodeError> {
        let packet = raw_flac_frame_payload(packet).ok_or_else(|| {
            DecodeError::DecodingFailed("MP4 sample contains no raw FLAC frame".to_owned())
        })?;
        let written = self
            .decoder
            .decode_i32_block_into(packet, &mut self.scratch)
            .map_err(|error| DecodeError::DecodingFailed(error.to_string()))?;
        Ok(create_audio_data_i32_with_bits(
            self.sample_rate,
            self.channels,
            self.bits_per_sample,
            &self.scratch[..written],
        ))
    }
}

fn raw_flac_frame_payload(packet: &[u8]) -> Option<&[u8]> {
    packet
        .windows(2)
        .position(|bytes| bytes[0] == 0xff && bytes[1] & 0xfc == 0xf8)
        .map(|offset| &packet[offset..])
}

struct WaveyAacDecoder {
    decoder: SymphoniaAacDecoder,
}

impl WaveyAacDecoder {
    fn new(
        decoder_configuration: &[u8],
        sample_rate: Option<u32>,
        channels: Option<u8>,
    ) -> Result<Self, DecodeError> {
        if decoder_configuration.is_empty() {
            return Err(DecodeError::DecoderInitFailed(
                "container AAC track has no AudioSpecificConfig".to_owned(),
            ));
        }

        let mut params = SymphoniaAudioCodecParameters::new();
        params
            .for_codec(SYMPHONIA_CODEC_ID_AAC)
            .with_extra_data(decoder_configuration.to_vec().into_boxed_slice());
        if let Some(sample_rate) = sample_rate {
            params.with_sample_rate(sample_rate);
        }
        if let Some(channels) = channels {
            params.with_channels(SymphoniaChannels::Discrete(u16::from(channels)));
        }

        let decoder =
            SymphoniaAacDecoder::try_new(&params, &SymphoniaAudioDecoderOptions::default())
                .map_err(|error| DecodeError::DecoderInitFailed(error.to_string()))?;
        Ok(Self { decoder })
    }

    fn decode_access_unit(&mut self, access_unit: &[u8]) -> Result<AudioData, DecodeError> {
        let packet = SymphoniaPacket::new(
            0,
            SymphoniaTimestamp::ZERO,
            SymphoniaDuration::ZERO,
            access_unit.to_vec(),
        );
        let decoded = self
            .decoder
            .decode(&packet)
            .map_err(|error| DecodeError::DecodingFailed(error.to_string()))?;
        symphonia_audio_to_i16(decoded)
    }
}

fn symphonia_audio_to_i16(decoded: SymphoniaAudioBufferRef<'_>) -> Result<AudioData, DecodeError> {
    let sample_rate = decoded.spec().rate();
    let channels = u8::try_from(decoded.num_planes())
        .map_err(|_| DecodeError::DecodingFailed("decoded channel count exceeds u8".to_owned()))?;
    let mut float_samples = Vec::with_capacity(decoded.samples_interleaved());
    decoded.copy_to_vec_interleaved::<f32>(&mut float_samples);
    let interleaved = float_samples
        .into_iter()
        .map(float_sample_to_i16)
        .collect::<Vec<_>>();
    Ok(create_audio_data_i16(sample_rate, channels, &interleaved))
}

fn make_aac_audio_decoder(track: &MediaTrackConfig) -> Result<SeekableAudioDecoder, DecodeError> {
    WaveyAacDecoder::new(
        &track.decoder_configuration,
        track.sample_rate,
        track.channels,
    )
    .map(SeekableAudioDecoder::WaveyAac)
}

fn decode_seekable_mp4_audio(
    data: &[u8],
    options: &DecodeOptions,
) -> Result<(MediaTrackConfig, Vec<MediaTrackConfig>, Vec<AudioData>), DecodeError> {
    let index = Mp4MediaIndex::from_file(data)
        .map_err(|error| DecodeError::InvalidInputFormat(error.to_string()))?;
    let track = index
        .tracks
        .iter()
        .find(|track| {
            track.kind == MediaTrackKind::Audio
                && matches!(track.codec.as_str(), "aac" | "alac" | "flac" | "pcm")
        })
        .cloned()
        .ok_or_else(|| {
            let codecs = index
                .tracks
                .iter()
                .filter(|track| track.kind == MediaTrackKind::Audio)
                .map(|track| track.codec.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            DecodeError::UnsupportedFormat(if codecs.is_empty() {
                AudioType::Unknown
            } else {
                detect_audio(data)
            })
        })?;
    if !index
        .samples
        .iter()
        .any(|sample| sample.kind == MediaTrackKind::Audio && sample.track_id == track.track_id)
    {
        return decode_fragmented_mp4_audio(data, options);
    }
    let mut decoder = match track.codec.as_str() {
        "aac" => make_aac_audio_decoder(&track)?,
        "alac" => SeekableAudioDecoder::Alac(
            AlacPacketDecoder::new(&track.codec_private).map_err(DecodeError::DecoderInitFailed)?,
        ),
        "flac" => SeekableAudioDecoder::Flac(FlacPacketDecoder::new(&track)?),
        "pcm" => SeekableAudioDecoder::Pcm,
        _ => unreachable!("filtered supported codec"),
    };

    let mut frames = Vec::new();
    let mut resampler = None;
    for (sample_index, sample) in index.samples.iter().enumerate() {
        if sample.kind != MediaTrackKind::Audio || sample.track_id != track.track_id {
            continue;
        }
        let start = usize::try_from(sample.absolute_offset).map_err(|_| {
            DecodeError::InvalidInputFormat("MP4 sample offset exceeds this platform".to_owned())
        })?;
        let end = start.checked_add(sample.size as usize).ok_or_else(|| {
            DecodeError::InvalidInputFormat("MP4 sample byte range overflow".to_owned())
        })?;
        let source = data.get(start..end).ok_or_else(|| {
            DecodeError::InvalidInputFormat(format!(
                "MP4 sample {sample_index} extends past the source"
            ))
        })?;
        let packet = index
            .packet_from_sample_bytes(sample_index, source)
            .map_err(DecodeError::InvalidInputFormat)?;
        let decoded = decode_seekable_packet(&mut decoder, &track, &packet.data)?;
        for frame in decoded {
            let decoded_frames = decoded_audio_frame_count(&frame)?;
            let Some(trim) = index
                .pcm_packet_trim_at_sample_rate(sample_index, decoded_frames, frame.sampling_rate())
                .map_err(DecodeError::InvalidInputFormat)?
            else {
                continue;
            };
            let Some(frame) =
                trim_interleaved_audio(frame, trim.source_frame_start, trim.frame_count)?
            else {
                continue;
            };
            frames.extend(apply_output_options(frame, options, &mut resampler)?);
        }
    }
    if let Some(pending) = resampler.take() {
        frames.extend(flush_resampler_frames(pending)?);
    }
    if frames.is_empty() {
        return Err(DecodeError::DecodingFailed(format!(
            "MP4 {} track emitted no audio frames",
            track.codec
        )));
    }
    Ok((track, index.tracks, frames))
}

fn decode_fragmented_mp4_audio(
    data: &[u8],
    options: &DecodeOptions,
) -> Result<(MediaTrackConfig, Vec<MediaTrackConfig>, Vec<AudioData>), DecodeError> {
    let mut demuxer = Mp4MediaDemuxer::new();
    let mut tracks = Vec::new();
    let mut selected = None;
    let mut decoder = None;
    let mut frames = Vec::new();
    let mut resampler = None;
    for chunk in data.chunks(64 * 1024) {
        let events = demuxer
            .push(chunk)
            .map_err(DecodeError::InvalidInputFormat)?;
        consume_fragmented_mp4_events(
            events,
            options,
            &mut tracks,
            &mut selected,
            &mut decoder,
            &mut frames,
            &mut resampler,
        )?;
    }
    let events = demuxer.flush().map_err(DecodeError::InvalidInputFormat)?;
    consume_fragmented_mp4_events(
        events,
        options,
        &mut tracks,
        &mut selected,
        &mut decoder,
        &mut frames,
        &mut resampler,
    )?;
    if let Some(pending) = resampler.take() {
        frames.extend(flush_resampler_frames(pending)?);
    }
    let selected = selected.ok_or_else(|| {
        DecodeError::InvalidInputFormat("fragmented MP4 has no supported audio track".to_owned())
    })?;
    if frames.is_empty() {
        return Err(DecodeError::DecodingFailed(format!(
            "fragmented MP4 {} track emitted no audio frames",
            selected.codec
        )));
    }
    Ok((selected, tracks, frames))
}

#[allow(clippy::too_many_arguments)]
fn consume_fragmented_mp4_events(
    events: Vec<Mp4MediaDemuxEvent>,
    options: &DecodeOptions,
    tracks: &mut Vec<MediaTrackConfig>,
    selected: &mut Option<MediaTrackConfig>,
    decoder: &mut Option<SeekableAudioDecoder>,
    frames: &mut Vec<AudioData>,
    resampler: &mut Option<StreamingResampler>,
) -> Result<(), DecodeError> {
    for event in events {
        match event {
            Mp4MediaDemuxEvent::Config(track) => {
                if selected.is_none()
                    && track.kind == MediaTrackKind::Audio
                    && matches!(track.codec.as_str(), "aac" | "alac" | "flac" | "pcm")
                {
                    *decoder = Some(make_seekable_audio_decoder(&track)?);
                    *selected = Some(track.clone());
                }
                if !tracks.iter().any(|known| known.track_id == track.track_id) {
                    tracks.push(track);
                }
            }
            Mp4MediaDemuxEvent::Packet(packet) => {
                let Some(track) = selected.as_ref() else {
                    continue;
                };
                if packet.kind != MediaTrackKind::Audio || packet.track_id != track.track_id {
                    continue;
                }
                let state = decoder.as_mut().ok_or_else(|| {
                    DecodeError::DecoderInitFailed(
                        "fragmented MP4 audio decoder is not initialized".to_owned(),
                    )
                })?;
                for frame in decode_seekable_packet(state, track, &packet.data)? {
                    frames.extend(apply_output_options(frame, options, resampler)?);
                }
            }
        }
    }
    Ok(())
}

fn make_seekable_audio_decoder(
    track: &MediaTrackConfig,
) -> Result<SeekableAudioDecoder, DecodeError> {
    match track.codec.as_str() {
        "aac" => make_aac_audio_decoder(track),
        "alac" => AlacPacketDecoder::new(&track.codec_private)
            .map(SeekableAudioDecoder::Alac)
            .map_err(DecodeError::DecoderInitFailed),
        "flac" => FlacPacketDecoder::new(track).map(SeekableAudioDecoder::Flac),
        "pcm" => Ok(SeekableAudioDecoder::Pcm),
        codec => Err(DecodeError::InvalidInputFormat(format!(
            "unsupported MP4 audio codec {codec}"
        ))),
    }
}

fn populate_mp4_track_metadata(
    metadata: &mut soundkit::media_metadata::MediaMetadata,
    tracks: &[MediaTrackConfig],
) {
    metadata.audio_tracks.clear();
    metadata.video_tracks.clear();
    for track in tracks {
        let duration_micros = track.timeline.and_then(|timeline| {
            (track.timescale != 0)
                .then(|| timeline.duration.saturating_mul(1_000_000) / u64::from(track.timescale))
        });
        match track.kind {
            MediaTrackKind::Audio => {
                metadata
                    .audio_tracks
                    .push(soundkit::media_metadata::AudioTrackMetadata {
                        id: Some(track.track_id),
                        codec: Some(track.codec.clone()),
                        codec_id: Some(track.codec_id.clone()),
                        sample_rate: track.sample_rate,
                        channels: track.channels.map(u16::from),
                        bits_per_sample: track.bits_per_sample,
                        duration_micros,
                        ..soundkit::media_metadata::AudioTrackMetadata::default()
                    })
            }
            MediaTrackKind::Video => {
                metadata
                    .video_tracks
                    .push(soundkit::media_metadata::VideoTrackMetadata {
                        id: Some(track.track_id),
                        codec: Some(track.codec.clone()),
                        codec_id: Some(track.codec_id.clone()),
                        width: track.width,
                        height: track.height,
                        duration_micros,
                        ..soundkit::media_metadata::VideoTrackMetadata::default()
                    })
            }
        }
    }
}

fn populate_webm_track_metadata(
    metadata: &mut soundkit::media_metadata::MediaMetadata,
    tracks: &[WebmMediaTrackConfig],
) {
    metadata.audio_tracks.clear();
    metadata.video_tracks.clear();
    for track in tracks {
        match track.kind {
            WebmTrackKind::Audio => {
                metadata
                    .audio_tracks
                    .push(soundkit::media_metadata::AudioTrackMetadata {
                        id: Some(track.track_number),
                        codec: Some(track.codec_id.clone()),
                        codec_id: Some(track.codec_id.clone()),
                        sample_rate: track.sample_rate,
                        channels: track.channels.map(u16::from),
                        ..soundkit::media_metadata::AudioTrackMetadata::default()
                    })
            }
            WebmTrackKind::Video => {
                metadata
                    .video_tracks
                    .push(soundkit::media_metadata::VideoTrackMetadata {
                        id: Some(track.track_number),
                        codec: Some(track.codec_id.clone()),
                        codec_id: Some(track.codec_id.clone()),
                        width: track.width,
                        height: track.height,
                        ..soundkit::media_metadata::VideoTrackMetadata::default()
                    })
            }
        }
    }
}

fn decode_seekable_packet(
    decoder: &mut SeekableAudioDecoder,
    track: &MediaTrackConfig,
    packet: &[u8],
) -> Result<Vec<AudioData>, DecodeError> {
    match decoder {
        SeekableAudioDecoder::WaveyAac(decoder) => {
            decoder.decode_access_unit(packet).map(|frame| vec![frame])
        }
        SeekableAudioDecoder::Alac(decoder) => decoder
            .decode_packet(packet)
            .map(|frame| vec![frame])
            .map_err(DecodeError::DecodingFailed),
        SeekableAudioDecoder::Flac(decoder) => decoder.decode(packet).map(|frame| vec![frame]),
        SeekableAudioDecoder::Pcm => {
            make_container_pcm_audio(track, packet.to_vec()).map(|f| vec![f])
        }
    }
}

fn decode_aac_access_unit(
    decoder: &mut AacLcDecoder,
    packet: &[u8],
) -> Result<AudioData, DecodeError> {
    let info = decoder.frame_info();
    let decoded = decoder
        .decode_access_unit(packet)
        .map_err(|error| DecodeError::DecodingFailed(error.to_string()))?;
    let mut interleaved = Vec::with_capacity(decoded.frames() * info.channels);
    for frame in 0..decoded.frames() {
        for channel in decoded.channels() {
            interleaved.push(float_sample_to_i16(channel[frame]));
        }
    }
    Ok(create_audio_data_i16(
        info.sample_rate,
        u8::try_from(info.channels)
            .map_err(|_| DecodeError::DecodingFailed("AAC channel count exceeds u8".to_owned()))?,
        &interleaved,
    ))
}

fn float_sample_to_i16(sample: f32) -> i16 {
    let finite = if sample.is_finite() {
        sample.clamp(-1.0, 1.0)
    } else {
        0.0
    };
    let scaled = if finite < 0.0 {
        f64::from(finite) * 32_768.0
    } else {
        f64::from(finite) * 32_767.0
    };
    (scaled.round() as i32).clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

fn make_container_pcm_audio(
    track: &MediaTrackConfig,
    data: Vec<u8>,
) -> Result<AudioData, DecodeError> {
    let sample_rate = track.sample_rate.ok_or_else(|| {
        DecodeError::InvalidInputFormat("container PCM has no sample rate".to_owned())
    })?;
    let channels = track.channels.ok_or_else(|| {
        DecodeError::InvalidInputFormat("container PCM has no channel count".to_owned())
    })?;
    let bits = track.bits_per_sample.ok_or_else(|| {
        DecodeError::InvalidInputFormat("container PCM has no sample depth".to_owned())
    })?;
    if !matches!(bits, 16 | 24 | 32) {
        return Err(DecodeError::InvalidInputFormat(format!(
            "unsupported container PCM sample depth: {bits}"
        )));
    }
    Ok(AudioData::new(
        bits,
        channels,
        sample_rate,
        data,
        if track.pcm_float == Some(true) {
            EncodingFlag::PCMFloat
        } else {
            EncodingFlag::PCMSigned
        },
        if track.pcm_endianness == Some(PcmEndianness::Big) {
            Endianness::BigEndian
        } else {
            Endianness::LittleEndian
        },
    ))
}

fn decoded_audio_frame_count(audio: &AudioData) -> Result<u32, DecodeError> {
    let bytes_per_sample = usize::from(audio.bits_per_sample().div_ceil(8));
    let bytes_per_frame = bytes_per_sample
        .checked_mul(usize::from(audio.channel_count()))
        .ok_or_else(|| DecodeError::DecodingFailed("PCM frame size overflow".to_owned()))?;
    if bytes_per_frame == 0 || audio.data().len() % bytes_per_frame != 0 {
        return Err(DecodeError::DecodingFailed(
            "decoder returned misaligned PCM".to_owned(),
        ));
    }
    u32::try_from(audio.data().len() / bytes_per_frame)
        .map_err(|_| DecodeError::DecodingFailed("PCM frame count exceeds u32".to_owned()))
}

fn trim_interleaved_audio(
    audio: AudioData,
    source_frame_start: u32,
    frame_count: u32,
) -> Result<Option<AudioData>, DecodeError> {
    if frame_count == 0 {
        return Ok(None);
    }
    let bytes_per_sample = usize::from(audio.bits_per_sample().div_ceil(8));
    let bytes_per_frame = bytes_per_sample
        .checked_mul(usize::from(audio.channel_count()))
        .ok_or_else(|| DecodeError::DecodingFailed("PCM frame size overflow".to_owned()))?;
    if bytes_per_frame == 0 || audio.data().len() % bytes_per_frame != 0 {
        return Err(DecodeError::DecodingFailed(
            "decoder returned misaligned PCM".to_owned(),
        ));
    }
    let start = source_frame_start as usize;
    let count = frame_count as usize;
    let end = start
        .checked_add(count)
        .ok_or_else(|| DecodeError::DecodingFailed("PCM trim range overflow".to_owned()))?;
    if end > audio.data().len() / bytes_per_frame {
        return Err(DecodeError::DecodingFailed(
            "PCM trim exceeds decoded packet".to_owned(),
        ));
    }
    Ok(Some(AudioData::new(
        audio.bits_per_sample(),
        audio.channel_count(),
        audio.sampling_rate(),
        audio.data()[start * bytes_per_frame..end * bytes_per_frame].to_vec(),
        audio.audio_format(),
        audio.endianness(),
    )))
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
    accum_start: usize,
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
            accum_start: 0,
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
            if samples.len() != input[0].len() {
                return Err("Channel sample counts changed mid-stream".to_string());
            }
            self.accum[channel].extend_from_slice(samples);
        }

        let mut outputs = Vec::new();
        while self.accum[0].len().saturating_sub(self.accum_start) >= self.chunk_size {
            let end = self.accum_start + self.chunk_size;
            let chunk: Vec<&[f32]> = self
                .accum
                .iter()
                .map(|channel| &channel[self.accum_start..end])
                .collect();

            let resampled = self
                .resampler
                .process(&chunk, None)
                .map_err(|error| format!("Resample failed: {error}"))?;
            self.accum_start = end;
            if resampled.iter().any(|channel| !channel.is_empty()) {
                outputs.push(resampled);
            }
        }

        if self.accum_start >= self.chunk_size * 8
            && self.accum_start.saturating_mul(2) >= self.accum[0].len()
        {
            for channel in &mut self.accum {
                channel.drain(..self.accum_start);
            }
            self.accum_start = 0;
        }

        Ok(outputs)
    }

    fn flush(&mut self) -> Result<Vec<Vec<Vec<f32>>>, String> {
        let mut outputs = Vec::new();

        let remaining = self.accum[0].len().saturating_sub(self.accum_start);
        if remaining > 0 {
            let padded_frames = self.chunk_size.saturating_sub(remaining);
            let chunk: Vec<&[f32]> = self
                .accum
                .iter()
                .map(|channel| &channel[self.accum_start..])
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
            for channel in &mut self.accum {
                channel.clear();
            }
            self.accum_start = 0;
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
    Detecting { buffer: BytesMut },
    Decoding { decoder: FormatDecoder },
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
    /// FLAC decoder using the unified Wavey pure-Rust codec
    Flac(Box<FlacDecoder>),
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
    /// Sequential ALAC compatibility path. It rejects input because ALAC
    /// containers require the seekable packet API.
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
    output: &mut [i32],
    decode_fn: F,
) -> Result<Vec<AudioData>, String>
where
    D: Decoder,
    F: Fn(&D, usize, &[i32]) -> Option<AudioData>,
{
    let mut results = Vec::new();
    // First call with actual data
    let samples = decoder.decode_i32(chunk, output, false)?;
    if samples > 0 {
        if let Some(audio_data) = decode_fn(decoder, samples, &output) {
            results.push(audio_data);
        }
    }

    // Drain remaining buffered frames
    loop {
        let samples = decoder.decode_i32(&[], output, false)?;
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
    output: &mut [i16],
    decode_fn: F,
) -> Result<Vec<AudioData>, String>
where
    D: Decoder,
    F: Fn(&D, usize, &[i16]) -> Option<AudioData>,
{
    let mut results = Vec::new();
    // First call with actual data
    let samples = decoder.decode_i16(chunk, output, false)?;
    if samples > 0 {
        if let Some(audio_data) = decode_fn(decoder, samples, &output) {
            results.push(audio_data);
        }
    }

    // Drain remaining buffered frames
    loop {
        let samples = decoder.decode_i16(&[], output, false)?;
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

fn process_single_add_api<D, F>(
    decoder: &mut D,
    chunk: &[u8],
    add_fn: F,
) -> Result<Vec<AudioData>, String>
where
    F: Fn(&mut D, &[u8]) -> Result<Option<AudioData>, String>,
{
    Ok(add_fn(decoder, chunk)?.into_iter().collect())
}

impl StreamingDecoder for FormatDecoder {
    fn process(
        &mut self,
        chunk: &[u8],
        scratch: &mut DecoderScratch,
    ) -> Result<Vec<AudioData>, String> {
        match self {
            FormatDecoder::Mp3(dec) => decode_i16_with_drain(
                dec.as_mut(),
                chunk,
                scratch.i16_samples(),
                |d, samples, output| {
                    let (sample_rate, channels) = (d.sample_rate()?, d.channels()?);
                    Some(create_audio_data_i16(
                        sample_rate,
                        channels,
                        &output[..samples],
                    ))
                },
            ),
            FormatDecoder::Aac(dec) => decode_i16_with_drain(
                dec.as_mut(),
                chunk,
                scratch.i16_samples(),
                |d, samples, output| {
                    let (sample_rate, channels) = (d.sample_rate()?, d.channels()?);
                    Some(create_audio_data_i16(
                        sample_rate,
                        channels,
                        &output[..samples],
                    ))
                },
            ),
            FormatDecoder::M4a(dec) => decode_i16_with_drain(
                dec.as_mut(),
                chunk,
                scratch.i16_samples(),
                |d, samples, output| {
                    let (sample_rate, channels) = (d.sample_rate()?, d.channels()?);
                    Some(create_audio_data_i16(
                        sample_rate,
                        channels,
                        &output[..samples],
                    ))
                },
            ),
            FormatDecoder::Flac(dec) => decode_with_drain(
                dec.as_mut(),
                chunk,
                scratch.i32_samples(),
                |d, samples, output| {
                    let (sample_rate, channels, bits) =
                        (d.sample_rate()?, d.channels()?, d.bits_per_sample()?);
                    Some(create_audio_data_i32_with_bits(
                        sample_rate,
                        channels,
                        bits,
                        &output[..samples],
                    ))
                },
            ),
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
            FormatDecoder::AmrNb(dec) => decode_i16_with_drain(
                dec.as_mut(),
                chunk,
                scratch.i16_samples(),
                |d, samples, output| {
                    Some(create_audio_data_i16(
                        d.sample_rate(),
                        d.channels(),
                        &output[..samples],
                    ))
                },
            ),
            FormatDecoder::G711(dec) => decode_i16_with_drain(
                dec.as_mut(),
                chunk,
                scratch.i16_samples(),
                |d, samples, output| {
                    Some(create_audio_data_i16(
                        d.sample_rate(),
                        d.channels(),
                        &output[..samples],
                    ))
                },
            ),
            FormatDecoder::G722(dec) => decode_i16_with_drain(
                dec.as_mut(),
                chunk,
                scratch.i16_samples(),
                |d, samples, output| {
                    Some(create_audio_data_i16(
                        d.sample_rate(),
                        d.channels(),
                        &output[..samples],
                    ))
                },
            ),
            FormatDecoder::G726(dec) => decode_i16_with_drain(
                dec.as_mut(),
                chunk,
                scratch.i16_samples(),
                |d, samples, output| {
                    Some(create_audio_data_i16(
                        d.sample_rate(),
                        d.channels(),
                        &output[..samples],
                    ))
                },
            ),
            FormatDecoder::G729(dec) => decode_i16_with_drain(
                dec.as_mut(),
                chunk,
                scratch.i16_samples(),
                |d, samples, output| {
                    Some(create_audio_data_i16(
                        d.sample_rate(),
                        d.channels(),
                        &output[..samples],
                    ))
                },
            ),
            FormatDecoder::Gsm(dec) => decode_i16_with_drain(
                dec.as_mut(),
                chunk,
                scratch.i16_samples(),
                |d, samples, output| {
                    Some(create_audio_data_i16(
                        d.sample_rate(),
                        d.channels(),
                        &output[..samples],
                    ))
                },
            ),
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
                process_single_add_api(dec.as_mut(), chunk, |d, data| d.add(data))
            }
            FormatDecoder::Ac3(dec) => {
                process_with_add_api(dec.as_mut(), chunk, |d, data| d.add(data))
            }
        }
    }

    fn flush(&mut self, scratch: &mut DecoderScratch) -> Result<Vec<AudioData>, String> {
        match self {
            FormatDecoder::M4a(decoder) => {
                let mut results = Vec::new();
                let output = scratch.i16_samples();
                loop {
                    let samples = decoder.finish_i16(output)?;
                    if samples == 0 {
                        break;
                    }
                    let (sample_rate, channels) = (decoder.sample_rate(), decoder.channels());
                    if let (Some(sample_rate), Some(channels)) = (sample_rate, channels) {
                        results.push(create_audio_data_i16(
                            sample_rate,
                            channels,
                            &output[..samples],
                        ));
                    }
                }
                Ok(results)
            }
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
            _ => self.process(&[], scratch),
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

    /// Create a compatibility pipeline that reports the seekable ALAC requirement.
    pub fn spawn_alac() -> DecodePipelineHandle {
        Self::spawn_alac_with_options(DecodeOptions::default())
    }

    /// Create an ALAC compatibility pipeline with output conversion options.
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
        let (input_tx, input_rx) = mpsc::sync_channel::<Bytes>(input_buffer.max(1));
        let (output_tx, output_rx) = mpsc::sync_channel::<DecodeOutput>(output_buffer.max(1));
        let queued_input_bytes = Arc::new(AtomicUsize::new(0));
        let worker_queued_input_bytes = Arc::clone(&queued_input_bytes);
        let cancelled = Arc::new(AtomicBool::new(false));
        let worker_cancelled = Arc::clone(&cancelled);

        let worker = thread::spawn(move || {
            pipeline_worker(
                input_rx,
                output_tx,
                options,
                initial_decoder,
                worker_queued_input_bytes,
                worker_cancelled,
            );
        });

        DecodePipelineHandle {
            input_tx: Some(input_tx),
            output_rx: Some(output_rx),
            worker: Some(worker),
            queued_input_bytes,
            cancelled,
        }
    }
}

/// Handle for interacting with the pipeline
pub struct DecodePipelineHandle {
    input_tx: Option<SyncSender<Bytes>>,
    output_rx: Option<Receiver<DecodeOutput>>,
    worker: Option<thread::JoinHandle<()>>,
    queued_input_bytes: Arc<AtomicUsize>,
    cancelled: Arc<AtomicBool>,
}

impl DecodePipelineHandle {
    /// Send encoded audio bytes to the pipeline (non-blocking)
    ///
    /// Returns `Err` if the ring buffer is full (backpressure)
    pub fn send(&mut self, data: Bytes) -> Result<(), DecodeError> {
        if data.len() > MAX_INPUT_CHUNK_BYTES {
            return Err(DecodeError::InputChunkTooLarge(data.len()));
        }
        let byte_len = data.len();
        if byte_len > 0
            && self
                .queued_input_bytes
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |queued| {
                    queued
                        .checked_add(byte_len)
                        .filter(|next| *next <= MAX_QUEUED_INPUT_BYTES)
                })
                .is_err()
        {
            return Err(DecodeError::InputBufferFull);
        }

        let Some(input_tx) = self.input_tx.as_ref() else {
            if byte_len > 0 {
                self.queued_input_bytes
                    .fetch_sub(byte_len, Ordering::AcqRel);
            }
            return Err(DecodeError::PipelineClosed);
        };
        match input_tx.try_send(data) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(data)) => {
                if !data.is_empty() {
                    self.queued_input_bytes
                        .fetch_sub(data.len(), Ordering::AcqRel);
                }
                Err(DecodeError::InputBufferFull)
            }
            Err(TrySendError::Disconnected(data)) => {
                if !data.is_empty() {
                    self.queued_input_bytes
                        .fetch_sub(data.len(), Ordering::AcqRel);
                }
                Err(DecodeError::PipelineClosed)
            }
        }
    }

    /// Signal end-of-stream after all encoded bytes have been sent.
    pub fn finish(&mut self) -> Result<(), DecodeError> {
        self.send(Bytes::new())
    }

    /// Try to receive a decoded audio frame without blocking
    ///
    /// Returns `None` if no data is available
    pub fn try_recv(&mut self) -> Option<DecodeOutput> {
        match self.output_rx.as_ref()?.try_recv() {
            Ok(output) => Some(output),
            Err(TryRecvError::Empty | TryRecvError::Disconnected) => None,
        }
    }

    /// Receive a decoded audio frame, blocking until available
    ///
    /// Wait until data is available or the pipeline is closed.
    pub fn recv(&mut self) -> Option<DecodeOutput> {
        self.output_rx.as_ref()?.recv().ok()
    }

    /// Cancel pending work and join the decoder worker.
    pub fn cancel(&mut self) {
        self.shutdown();
    }

    /// Return the encoded bytes that are waiting in the bounded input queue.
    pub fn queued_input_bytes(&self) -> usize {
        self.queued_input_bytes.load(Ordering::Acquire)
    }

    fn shutdown(&mut self) {
        self.cancelled.store(true, Ordering::Release);
        self.output_rx.take();
        self.input_tx.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

impl Drop for DecodePipelineHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Main worker thread function
fn pipeline_worker(
    input_rx: Receiver<Bytes>,
    output_tx: SyncSender<DecodeOutput>,
    options: DecodeOptions,
    initial_decoder: Option<FormatDecoder>,
    queued_input_bytes: Arc<AtomicUsize>,
    cancelled: Arc<AtomicBool>,
) {
    let mut resampler: Option<StreamingResampler> = None;
    let mut decoder_scratch = DecoderScratch::default();
    let mut state = match initial_decoder {
        Some(decoder) => PipelineState::Decoding { decoder },
        None => PipelineState::Detecting {
            buffer: BytesMut::new(),
        },
    };

    loop {
        if cancelled.load(Ordering::Acquire) {
            break;
        }
        let chunk = match input_rx.recv() {
            Ok(chunk) => chunk,
            Err(_) => break,
        };
        if !chunk.is_empty() {
            queued_input_bytes.fetch_sub(chunk.len(), Ordering::AcqRel);
        }
        if cancelled.load(Ordering::Acquire) {
            break;
        }

        // Empty chunk signals end-of-stream, initiate flush
        let is_eof = chunk.is_empty();

        let next_state = match state {
            PipelineState::Detecting { mut buffer } => {
                if is_eof {
                    // EOF during detection - try to decode with whatever we have
                    // Some formats (like Opus) can be detected with very little data
                    match detect_and_init_decoder(buffer.as_ref()) {
                        Ok(mut decoder) => {
                            if process_with_decoder(
                                &mut decoder,
                                buffer.as_ref(),
                                &output_tx,
                                &options,
                                &mut resampler,
                                &mut decoder_scratch,
                            ) {
                                flush_decoder(
                                    &mut decoder,
                                    &output_tx,
                                    &options,
                                    &mut resampler,
                                    &mut decoder_scratch,
                                );
                            }
                        }
                        Err(e) => {
                            push_output(&output_tx, Err(e.clone()));
                        }
                    }
                    None
                } else {
                    let probe_bytes = (MAX_DETECTION_BYTES - buffer.len()).min(chunk.len());
                    buffer.extend_from_slice(&chunk[..probe_bytes]);
                    let new_bytes_collected = buffer.len();

                    // Try early detection for formats with clear magic bytes
                    // This allows smaller files (like short Opus) to be processed
                    // without waiting for MIN_DETECTION_BYTES
                    if new_bytes_collected >= MIN_DETECTION_BYTES {
                        match detect_and_init_decoder(buffer.as_ref()) {
                            Ok(mut decoder) => {
                                // Feed accumulated buffer to decoder
                                let first_ok = process_with_decoder(
                                    &mut decoder,
                                    buffer.as_ref(),
                                    &output_tx,
                                    &options,
                                    &mut resampler,
                                    &mut decoder_scratch,
                                );
                                let remainder_ok = if first_ok && probe_bytes < chunk.len() {
                                    process_with_decoder(
                                        &mut decoder,
                                        &chunk[probe_bytes..],
                                        &output_tx,
                                        &options,
                                        &mut resampler,
                                        &mut decoder_scratch,
                                    )
                                } else {
                                    first_ok
                                };
                                (first_ok && remainder_ok)
                                    .then_some(PipelineState::Decoding { decoder })
                            }
                            Err(_e) if new_bytes_collected < MAX_DETECTION_BYTES => {
                                // Need more data
                                Some(PipelineState::Detecting { buffer })
                            }
                            Err(e) => {
                                // Failed detection
                                push_output(&output_tx, Err(e.clone()));
                                None
                            }
                        }
                    } else {
                        Some(PipelineState::Detecting { buffer })
                    }
                }
            }

            PipelineState::Decoding { mut decoder } => {
                if is_eof {
                    flush_decoder(
                        &mut decoder,
                        &output_tx,
                        &options,
                        &mut resampler,
                        &mut decoder_scratch,
                    );
                    None
                } else {
                    if process_with_decoder(
                        &mut decoder,
                        chunk.as_ref(),
                        &output_tx,
                        &options,
                        &mut resampler,
                        &mut decoder_scratch,
                    ) {
                        Some(PipelineState::Decoding { decoder })
                    } else {
                        None
                    }
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
            let mut decoder = FlacDecoder::new();
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
    output_tx: &SyncSender<DecodeOutput>,
    options: &DecodeOptions,
    resampler: &mut Option<StreamingResampler>,
    scratch: &mut DecoderScratch,
) -> bool {
    match decoder.process(chunk, scratch) {
        Ok(audio_frames) => {
            for audio_data in audio_frames {
                if !push_audio_data(output_tx, audio_data, options, resampler) {
                    return false;
                }
            }
            true
        }
        Err(e) => {
            let _ = push_output(output_tx, Err(DecodeError::DecodingFailed(e)));
            false
        }
    }
}

/// Flush remaining samples from decoder using the unified StreamingDecoder trait.
fn flush_decoder(
    decoder: &mut FormatDecoder,
    output_tx: &SyncSender<DecodeOutput>,
    options: &DecodeOptions,
    resampler: &mut Option<StreamingResampler>,
    scratch: &mut DecoderScratch,
) -> bool {
    let decoded = match decoder.flush(scratch) {
        Ok(audio_frames) => {
            for audio_data in audio_frames {
                if !push_audio_data(output_tx, audio_data, options, resampler) {
                    return false;
                }
            }
            true
        }
        Err(e) => {
            let _ = push_output(output_tx, Err(DecodeError::DecodingFailed(e)));
            false
        }
    };
    if !decoded {
        return false;
    }

    if let Some(pending) = resampler.take() {
        flush_pending_resampler(output_tx, pending)
    } else {
        true
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

fn push_output(output_tx: &SyncSender<DecodeOutput>, output: DecodeOutput) -> bool {
    output_tx.send(output).is_ok()
}

fn push_audio_data(
    output_tx: &SyncSender<DecodeOutput>,
    audio_data: AudioData,
    options: &DecodeOptions,
    resampler: &mut Option<StreamingResampler>,
) -> bool {
    match apply_output_options(audio_data, options, resampler) {
        Ok(frames) => {
            for frame in frames {
                if !push_output(output_tx, Ok(frame)) {
                    return false;
                }
            }
            true
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

fn flush_pending_resampler(
    output_tx: &SyncSender<DecodeOutput>,
    resampler: StreamingResampler,
) -> bool {
    match flush_resampler_frames(resampler) {
        Ok(frames) => {
            for frame in frames {
                if !push_output(output_tx, Ok(frame)) {
                    return false;
                }
            }
            true
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

    // Preserve lossless integer PCM when only reducing sample depth. Going
    // through f32 introduces avoidable one-LSB differences for 24-bit FLAC.
    if target_sample_rate == audio_data.sampling_rate()
        && target_channels == audio_data.channel_count()
        && target_bits_per_sample == 16
        && audio_data.audio_format() == EncodingFlag::PCMSigned
        && matches!(audio_data.bits_per_sample(), 24 | 32)
    {
        return exact_signed_pcm_to_i16(audio_data).map(|frame| vec![frame]);
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

fn exact_signed_pcm_to_i16(audio: AudioData) -> Result<AudioData, DecodeError> {
    let bytes_per_sample = usize::from(audio.bits_per_sample() / 8);
    if audio.data().len() % bytes_per_sample != 0 {
        return Err(DecodeError::DecodingFailed(
            "integer PCM contains a partial sample".to_owned(),
        ));
    }
    let shift = u32::from(audio.bits_per_sample() - 16);
    let mut output = Vec::with_capacity(audio.data().len() / bytes_per_sample * 2);
    for bytes in audio.data().chunks_exact(bytes_per_sample) {
        let sample = match (audio.bits_per_sample(), audio.endianness()) {
            (24, Endianness::LittleEndian) => {
                (i32::from_le_bytes([bytes[0], bytes[1], bytes[2], 0]) << 8) >> 8
            }
            (24, Endianness::BigEndian) => {
                (i32::from_be_bytes([0, bytes[0], bytes[1], bytes[2]]) << 8) >> 8
            }
            (32, Endianness::LittleEndian) => i32::from_le_bytes(bytes.try_into().unwrap()),
            (32, Endianness::BigEndian) => i32::from_be_bytes(bytes.try_into().unwrap()),
            _ => unreachable!("validated signed PCM depth"),
        };
        output.extend_from_slice(&((sample >> shift) as i16).to_le_bytes());
    }
    Ok(AudioData::new(
        16,
        audio.channel_count(),
        audio.sampling_rate(),
        output,
        EncodingFlag::PCMSigned,
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
    let bytes_per_sample = usize::from(audio_data.bits_per_sample().div_ceil(8));
    if !matches!(audio_data.bits_per_sample(), 16 | 24 | 32)
        || audio_data.data().len() % bytes_per_sample != 0
        || audio_data.data().len() / bytes_per_sample % channel_count != 0
    {
        return Err("PCM data is unsupported or contains a partial frame".to_owned());
    }
    if audio_data.audio_format() == EncodingFlag::PCMFloat && audio_data.bits_per_sample() != 32 {
        return Err("floating-point PCM must contain 32-bit samples".to_owned());
    }

    let sample_count = audio_data.data().len() / bytes_per_sample;
    let mut channels = vec![Vec::with_capacity(sample_count / channel_count); channel_count];
    for (sample_index, bytes) in audio_data.data().chunks_exact(bytes_per_sample).enumerate() {
        let sample = match (audio_data.audio_format(), audio_data.bits_per_sample()) {
            (EncodingFlag::PCMFloat, 32) => match audio_data.endianness() {
                Endianness::LittleEndian => f32::from_le_bytes(bytes.try_into().unwrap()),
                Endianness::BigEndian => f32::from_be_bytes(bytes.try_into().unwrap()),
            },
            (_, 16) => {
                let value = match audio_data.endianness() {
                    Endianness::LittleEndian => i16::from_le_bytes(bytes.try_into().unwrap()),
                    Endianness::BigEndian => i16::from_be_bytes(bytes.try_into().unwrap()),
                };
                f32::from(value) / 32_768.0
            }
            (_, 24) => {
                let value = match audio_data.endianness() {
                    Endianness::LittleEndian => {
                        (i32::from_le_bytes([bytes[0], bytes[1], bytes[2], 0]) << 8) >> 8
                    }
                    Endianness::BigEndian => {
                        (i32::from_be_bytes([0, bytes[0], bytes[1], bytes[2]]) << 8) >> 8
                    }
                };
                value as f32 / 8_388_608.0
            }
            (_, 32) => {
                let value = match audio_data.endianness() {
                    Endianness::LittleEndian => i32::from_le_bytes(bytes.try_into().unwrap()),
                    Endianness::BigEndian => i32::from_be_bytes(bytes.try_into().unwrap()),
                };
                value as f32 / 2_147_483_648.0
            }
            _ => unreachable!("validated PCM representation"),
        };
        channels[sample_index % channel_count].push(if sample.is_finite() { sample } else { 0.0 });
    }
    Ok(channels)
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
            let mut output = Vec::with_capacity(sample_count * channels.len() * 2);
            for sample_index in 0..sample_count {
                for channel in channels {
                    let sample = float_sample_to_i16(channel[sample_index]);
                    output.extend_from_slice(&sample.to_le_bytes());
                }
            }
            Ok(output)
        }
        24 => {
            let mut output = Vec::with_capacity(sample_count * channels.len() * 3);
            for sample_index in 0..sample_count {
                for channel in channels {
                    let clamped = channel[sample_index].clamp(-1.0, 1.0);
                    let sample = if clamped >= 0.0 {
                        (clamped * 8_388_607.0) as i32
                    } else {
                        (clamped * 8_388_608.0) as i32
                    };
                    output.extend_from_slice(&sample.to_le_bytes()[..3]);
                }
            }
            Ok(output)
        }
        32 => {
            let mut output = Vec::with_capacity(sample_count * channels.len() * 4);
            for sample_index in 0..sample_count {
                for channel in channels {
                    let clamped = channel[sample_index].clamp(-1.0, 1.0);
                    let sample = if clamped >= 0.0 {
                        (clamped * i32::MAX as f32) as i32
                    } else {
                        (clamped * -(i32::MIN as f32)) as i32
                    };
                    output.extend_from_slice(&sample.to_le_bytes());
                }
            }
            Ok(output)
        }
        bits => Err(format!("Unsupported output bits per sample: {}", bits)),
    }
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
    use soundkit::audio_packet::Encoder;
    use soundkit_opus::OpusEncoder;
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

    #[test]
    fn mp4_flac_packets_decode_without_reconstructing_a_native_stream() {
        use soundkit_audio_demux::AudioContainer;
        use soundkit_flac::stream::Encoder as NativeFlacEncoder;

        let config = FlacFrameConfig::new(48_000, 2, 16, 240, FlacProfile::Balanced).unwrap();
        let samples = (0..config.sample_count().unwrap())
            .map(|index| ((index as i32 * 977) % 65_536) - 32_768)
            .collect::<Vec<_>>();
        let mut native = NativeFlacEncoder::new(config).unwrap();
        let mut legacy_first_packet = Vec::new();
        native
            .encode_i32(&samples, &mut legacy_first_packet)
            .unwrap();
        let stream_header = native.finish().unwrap().to_vec();
        let raw_offset = legacy_first_packet
            .windows(2)
            .position(|bytes| bytes[0] == 0xff && bytes[1] & 0xfc == 0xf8)
            .unwrap();
        let raw_packet = &legacy_first_packet[raw_offset..];
        let mut decoder_configuration = b"fLaC".to_vec();
        decoder_configuration.extend_from_slice(&stream_header);
        let track = MediaTrackConfig {
            container: AudioContainer::Mp4,
            kind: MediaTrackKind::Audio,
            track_id: 1,
            codec: "flac".to_owned(),
            codec_id: "fLaC".to_owned(),
            timescale: 48_000,
            timeline: None,
            edit_timeline: Vec::new(),
            sample_count: 1,
            width: None,
            height: None,
            sample_rate: Some(48_000),
            channels: Some(2),
            bits_per_sample: Some(16),
            pcm_endianness: None,
            pcm_float: None,
            pcm_signed: None,
            pcm_packed: None,
            pcm_aligned_high: None,
            pcm_interleaved: None,
            pcm_bytes_per_frame: None,
            pcm_frames_per_packet: None,
            codec_private: Vec::new(),
            decoder_configuration,
            nal_length_size: None,
        };
        let expected = samples
            .iter()
            .flat_map(|&sample| (sample as i16).to_le_bytes())
            .collect::<Vec<_>>();

        let mut raw_decoder = FlacPacketDecoder::new(&track).unwrap();
        let decoded = raw_decoder.decode(raw_packet).unwrap();
        assert_eq!(decoded.data().as_slice(), expected.as_slice());

        // Older indexed packets prefixed the first raw frame with STREAMINFO.
        // Keep read compatibility while all new writers emit only the frame.
        let mut legacy_decoder = FlacPacketDecoder::new(&track).unwrap();
        let decoded = legacy_decoder.decode(&legacy_first_packet).unwrap();
        assert_eq!(decoded.data().as_slice(), expected.as_slice());
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

    fn supported_raw_opus_stream() -> Bytes {
        const SAMPLE_RATE: u32 = 48_000;
        const CHANNELS: u32 = 1;
        const FRAME_SIZE: u32 = 960;
        const FRAMES: usize = 6;

        let mut stream = Vec::new();
        stream.extend_from_slice(b"OpusHead");
        stream.push(1);
        stream.push(CHANNELS as u8);
        stream.extend_from_slice(&0u16.to_le_bytes());
        stream.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
        stream.extend_from_slice(&0i16.to_le_bytes());
        stream.push(0);

        let mut encoder = OpusEncoder::new(SAMPLE_RATE, 16, CHANNELS, FRAME_SIZE, 64_000);
        encoder
            .init()
            .expect("failed to initialize test opus encoder");

        for frame_index in 0..FRAMES {
            let input = (0..FRAME_SIZE as usize)
                .map(|sample_index| {
                    let t = (frame_index * FRAME_SIZE as usize + sample_index) as f32
                        / SAMPLE_RATE as f32;
                    ((t * 440.0 * std::f32::consts::TAU).sin() * i16::MAX as f32 * 0.2) as i16
                })
                .collect::<Vec<_>>();

            let mut packet = vec![0u8; 4096];
            let encoded_len = encoder
                .encode_i16(&input, &mut packet)
                .expect("failed to encode test opus packet");
            stream.extend_from_slice(&(encoded_len as u16).to_le_bytes());
            stream.extend_from_slice(&packet[..encoded_len]);
        }

        Bytes::from(stream)
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
    fn pipeline_rejects_oversized_input_chunks() {
        let format = RawPcmFormat::linear16(48_000, 2).unwrap();
        let mut pipeline = DecodePipeline::spawn_raw_pcm(format);
        let oversized = Bytes::from(vec![0_u8; MAX_INPUT_CHUNK_BYTES + 1]);
        assert!(matches!(
            pipeline.send(oversized),
            Err(DecodeError::InputChunkTooLarge(bytes))
                if bytes == MAX_INPUT_CHUNK_BYTES + 1
        ));
    }

    #[test]
    fn pipeline_caps_queued_input_bytes() {
        let decoder = FormatDecoder::RawPcm(Box::new(RawPcmStreamProcessor::new(
            RawPcmFormat::linear16(48_000, 2).unwrap(),
        )));
        let mut pipeline = DecodePipeline::spawn_with_initial_decoder(
            64,
            1,
            DecodeOptions::default(),
            Some(decoder),
        );
        let maximum_chunk = Bytes::from(vec![0_u8; MAX_INPUT_CHUNK_BYTES]);
        let mut reached_budget = false;

        for _ in 0..16 {
            match pipeline.send(maximum_chunk.clone()) {
                Ok(()) => {}
                Err(DecodeError::InputBufferFull) => {
                    reached_budget = true;
                    break;
                }
                Err(error) => panic!("unexpected queue error: {error}"),
            }
        }

        assert!(reached_budget, "the byte budget did not apply backpressure");
        assert!(pipeline.queued_input_bytes() <= MAX_QUEUED_INPUT_BYTES);
    }

    #[test]
    fn dropping_pipeline_unblocks_a_worker_with_full_output() {
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        thread::spawn(move || {
            let format = RawPcmFormat::linear16(48_000, 2).unwrap();
            let mut pipeline = DecodePipeline::spawn_raw_pcm(format);
            let chunk = Bytes::from(vec![0_u8; 4096]);
            for _ in 0..64 {
                loop {
                    match pipeline.send(chunk.clone()) {
                        Ok(()) => break,
                        Err(DecodeError::InputBufferFull) => std::thread::yield_now(),
                        Err(error) => panic!("unexpected queue error: {error}"),
                    }
                }
            }
            drop(pipeline);
            done_tx.send(()).unwrap();
        });

        done_rx
            .recv_timeout(std::time::Duration::from_secs(2))
            .expect("dropping the pipeline did not join its blocked worker");
    }

    #[test]
    fn cancel_closes_pipeline_input_and_output() {
        let format = RawPcmFormat::linear16(48_000, 2).unwrap();
        let mut pipeline = DecodePipeline::spawn_raw_pcm(format);
        pipeline.cancel();

        assert!(matches!(
            pipeline.send(Bytes::from_static(&[0, 0, 0, 0])),
            Err(DecodeError::PipelineClosed)
        ));
        assert!(pipeline.recv().is_none());
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
    fn test_explicit_alac_requires_seekable_packet_api() {
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
        pipeline.send(data.slice(..997)).unwrap();
        let error = pipeline
            .recv()
            .expect("ALAC compatibility result")
            .unwrap_err();
        assert!(error.to_string().contains("seekable M4A/MP4 or CAF"));
    }

    #[test]
    fn test_autodetected_alac_requires_seekable_packet_api() {
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
        let error = pipeline
            .recv()
            .expect("ALAC autodetect result")
            .unwrap_err();
        assert!(error.to_string().contains("seekable M4A/MP4 or CAF"));
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
            assert!(!frames.is_empty(), "{fixture_name} should stream PCM");
            assert!(frames.iter().all(|frame| frame.bits_per_sample() == 16));
            assert!(frames.iter().all(|frame| frame.channel_count() == 1));
            assert!(frames.iter().all(|frame| frame.sampling_rate() == 8_000));
            assert!(frames
                .iter()
                .flat_map(|frame| frame.data().chunks_exact(2))
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
        let data = supported_raw_opus_stream();

        let mut pipeline = DecodePipeline::spawn();
        pipeline.send(data).unwrap();
        pipeline.send(Bytes::new()).unwrap(); // EOF signal - required for small files

        let mut frame_count = 0;

        for _ in 0..100 {
            if let Some(result) = pipeline.try_recv() {
                match result {
                    Ok(audio_data) => {
                        assert_eq!(audio_data.bits_per_sample(), 16);
                        assert_eq!(audio_data.sampling_rate(), 48_000);
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
                        assert_eq!(audio_data.sampling_rate(), 48_000);
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

            let mut pipeline = match options {
                Some(opts) => DecodePipeline::spawn_with_buffers_and_options(256, 256, opts),
                None => DecodePipeline::spawn_with_buffers(256, 256),
            };
            let mut frame_count = 0usize;

            // Stream data in chunks
            let chunk_size = 4096;
            for start in (0..data.len()).step_by(chunk_size) {
                let end = (start + chunk_size).min(data.len());
                let chunk = data.slice(start..end);
                loop {
                    match pipeline.send(chunk.clone()) {
                        Ok(()) => break,
                        Err(DecodeError::InputBufferFull) => {
                            while let Some(output) = pipeline.try_recv() {
                                output.expect("benchmark decode failed");
                                frame_count += 1;
                            }
                            std::thread::yield_now();
                        }
                        Err(error) => panic!("benchmark input failed: {error}"),
                    }
                }
            }

            loop {
                match pipeline.finish() {
                    Ok(()) => break,
                    Err(DecodeError::InputBufferFull) => {
                        while let Some(output) = pipeline.try_recv() {
                            output.expect("benchmark decode failed");
                            frame_count += 1;
                        }
                        std::thread::yield_now();
                    }
                    Err(error) => panic!("benchmark finish failed: {error}"),
                }
            }

            while let Some(output) = pipeline.recv() {
                output.expect("benchmark decode failed");
                frame_count += 1;
            }

            if frame_count > 0 {
                successful_files += 1;
            }
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
