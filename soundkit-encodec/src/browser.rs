//! Thin adapters to the existing encodec-rs browser entropy decoder. Keeping
//! these calls in Rust removes the host's per-symbol PCM/code assembly loop.
use crate::EncodecPcmDecoder;
use anyhow::{bail, Result};
use encodec_rs::binary::read_ecdc_header;
pub use encodec_rs::chunk_decode::lm_ecdc_decode_chunks;
use encodec_rs::chunk_decode::QuantizedLmChunkDecoder;
use encodec_rs::format::EcdcMetadata;
use encodec_rs::metadata::FrameBundleMetadata;
use encodec_rs::stable_hash::stable_hash_hex;
use std::io::Cursor;

pub fn ecdc_metadata(payload: &[u8]) -> Result<EcdcMetadata> {
    read_ecdc_header(&mut Cursor::new(payload))
}

pub struct BrowserDecoder {
    pub pcm: EncodecPcmDecoder,
    bundle_json: String,
    weights: Vec<u8>,
}

impl BrowserDecoder {
    pub fn new(
        bundle_json: &str,
        weights: &[u8],
        expected_hash: &str,
        audio_length: usize,
        frame_count: usize,
        retain_full: bool,
    ) -> Result<Self> {
        if !expected_hash.is_empty() && stable_hash_hex(weights) != expected_hash {
            bail!("EnCodec LM weights do not match the ECDC model identity");
        }
        let bundle: FrameBundleMetadata = serde_json::from_str(bundle_json)?;
        bundle.validate()?;
        let pcm = EncodecPcmDecoder::new(
            bundle.channels,
            bundle.sample_rate as u32,
            bundle.segment_samples,
            bundle.segment_stride,
            audio_length,
            frame_count,
            retain_full,
        )?;
        Ok(Self {
            pcm,
            bundle_json: bundle_json.to_owned(),
            weights: weights.to_vec(),
        })
    }

    pub fn decode_chunk(&self, payload: &[u8], frame_length: usize) -> Result<(f32, Vec<u16>)> {
        let mut decoder = QuantizedLmChunkDecoder::new(&self.bundle_json, &self.weights, payload)?;
        let scale = decoder.scale();
        let codes = decoder.pull_all(frame_length)?;
        Ok((scale, codes))
    }
}
