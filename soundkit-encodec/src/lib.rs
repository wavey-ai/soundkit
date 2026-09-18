//! EnCodec is a model-backed SoundKit decoder, not a second codec implementation.
//!
//! The caller supplies the existing inference backend. Entropy decoding,
//! fixed-context geometry, overlap-add and s16 rounding remain in encodec-rs.
//! Record/PNG extraction, model downloads and storage are outside this crate.

use anyhow::{bail, Result};
use encodec_rs::ecdc::decode_ecdc_model_windows;
pub use encodec_rs::ecdc::{DecodedEcdcWindowInfo, FrameCodec, LmCodec};
use encodec_rs::seam::{SeamCursor, SeamSegment};
use frame_header::{EncodingFlag, Endianness};
use soundkit::audio_types::AudioData;

#[cfg(feature = "browser")]
pub mod browser;
mod planar;
pub use planar::decode_planar_f32;

/// An owned, final PCM span. Planar storage is retained for zero-reordering
/// writes by browser hosts; native pipeline consumers can request AudioData.
#[derive(Debug)]
pub struct PcmSegment {
    pub chunk_index: usize,
    pub start_frame: usize,
    pub end_frame: usize,
    pub channels: usize,
    pub sample_rate: u32,
    pub planar: Vec<i16>,
}

impl PcmSegment {
    pub fn into_audio_data(self) -> AudioData {
        let frames = self.end_frame - self.start_frame;
        let mut bytes = Vec::with_capacity(self.planar.len() * 2);
        for frame in 0..frames {
            for channel in 0..self.channels {
                bytes.extend_from_slice(&self.planar[channel * frames + frame].to_le_bytes());
            }
        }
        AudioData::new(
            16,
            self.channels as u8,
            self.sample_rate,
            bytes,
            EncodingFlag::PCMSigned,
            Endianness::LittleEndian,
        )
    }
}

/// Streaming PCM half of the handler. Accepts the same model windows, silent
/// windows and validated cache spans as the original player. No model is loaded
/// until the host encounters a cache miss and supplies a decoded window.
pub struct EncodecPcmDecoder {
    cursor: SeamCursor,
    channels: usize,
    sample_rate: u32,
    frame_count: usize,
}

impl EncodecPcmDecoder {
    pub fn new(
        channels: usize,
        sample_rate: u32,
        model_samples: usize,
        owned_samples: usize,
        audio_length: usize,
        frame_count: usize,
        retain_full: bool,
    ) -> Result<Self> {
        if !(1..=2).contains(&channels) || sample_rate == 0 || audio_length == 0 || frame_count == 0
        {
            bail!("invalid EnCodec PCM geometry");
        }
        Ok(Self {
            cursor: SeamCursor::new(
                channels,
                model_samples,
                owned_samples,
                audio_length,
                frame_count,
                retain_full,
            )?,
            channels,
            sample_rate,
            frame_count,
        })
    }

    pub fn add_decoded_frame(&mut self, index: usize, window: &[f32]) -> Result<()> {
        self.check_index(index)?;
        self.cursor.add_decoded_frame(index, window)
    }

    pub fn add_silent_frame(&mut self, index: usize) -> Result<()> {
        self.check_index(index)?;
        self.cursor.add_silent_frame(index)
    }

    pub fn add_cached_range(&mut self, start: usize, end: usize, interleaved: &[i16]) -> bool {
        self.cursor.add_cached_range(start, end, interleaved)
    }

    pub fn emit_after_batch(&mut self, next_index: usize) -> Result<Vec<PcmSegment>> {
        if next_index > self.frame_count {
            bail!("EnCodec batch exceeds the programme");
        }
        let spans = self.cursor.emit_after_batch(next_index);
        Ok(self.package(spans))
    }

    pub fn flush(&mut self) -> Vec<PcmSegment> {
        let spans = self.cursor.flush();
        self.package(spans)
    }

    pub fn result_pcm(&self) -> &[i16] {
        self.cursor.full_channel_data()
    }

    fn check_index(&self, index: usize) -> Result<()> {
        if index >= self.frame_count {
            bail!("EnCodec window exceeds the programme");
        }
        Ok(())
    }

    fn package(&self, spans: Vec<SeamSegment>) -> Vec<PcmSegment> {
        spans
            .into_iter()
            .map(|span| PcmSegment {
                chunk_index: span.chunk_index,
                start_frame: span.start_frame,
                end_frame: span.end_frame,
                channels: self.channels,
                sample_rate: self.sample_rate,
                planar: self.cursor.segment_pcm(&span),
            })
            .collect()
    }
}

/// Decode an extracted ECDC object through its existing model backend and
/// immediately deliver each final PCM span. The sink provides backpressure:
/// the next model window is not decoded until it returns. No full PCM is kept.
pub fn decode_to_sink(
    codec: &mut dyn FrameCodec,
    lm: &mut dyn LmCodec,
    payload: &[u8],
    mut emit: impl FnMut(AudioData) -> Result<()>,
) -> Result<DecodedEcdcWindowInfo> {
    let bundle = codec.metadata().clone();
    let mut pcm = None;
    let info =
        decode_ecdc_model_windows(codec, lm, payload, |info, index, offset, owned, window| {
            let samples = window
                .as_slice()
                .ok_or_else(|| anyhow::anyhow!("noncontiguous EnCodec window"))?;
            if info.context_samples.is_some() {
                if pcm.is_none() {
                    pcm = Some(EncodecPcmDecoder::new(
                        bundle.channels,
                        bundle.sample_rate as u32,
                        info.chunk_layout.samples,
                        info.chunk_layout.stride,
                        info.metadata.audio_length,
                        info.window_count,
                        false,
                    )?);
                }
                let pcm = pcm.as_mut().unwrap();
                pcm.add_decoded_frame(index, samples)?;
                for segment in pcm.emit_after_batch(index + 1)? {
                    emit(segment.into_audio_data())?;
                }
            } else {
                // Non-fixed-context ECDC already returns final owned PCM.
                if samples.len() != bundle.channels * owned {
                    bail!("invalid EnCodec owned window");
                }
                let planar = samples
                    .iter()
                    .map(|&sample| {
                        ((sample as f64).clamp(-1.0, 1.0) * 32767.0 + 0.5).floor() as i16
                    })
                    .collect();
                emit(
                    PcmSegment {
                        chunk_index: index,
                        start_frame: offset,
                        end_frame: offset + owned,
                        channels: bundle.channels,
                        sample_rate: bundle.sample_rate as u32,
                        planar,
                    }
                    .into_audio_data(),
                )?;
            }
            Ok(())
        })?;
    if let Some(pcm) = pcm.as_mut() {
        for segment in pcm.flush() {
            emit(segment.into_audio_data())?;
        }
    }
    Ok(info)
}
