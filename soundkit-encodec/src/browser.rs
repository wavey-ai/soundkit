//! Thin adapters to the existing encodec-rs browser entropy decoder. Keeping
//! these calls in Rust removes the host's per-symbol PCM/code assembly loop.
use crate::EncodecPcmDecoder;
use anyhow::{bail, Result};
use encodec_rs::binary::read_ecdc_header;
pub use encodec_rs::chunk_decode::lm_ecdc_decode_chunks;
use encodec_rs::chunk_decode::QuantizedLmModel;
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
    /// The LM parsed from `bundle_json` and `weights` by the first chunk and
    /// kept for the rest; the weights are then released.
    model: Option<QuantizedLmModel>,
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
            model: None,
        })
    }

    /// The chunk scale and codes, as a new chunk decoder gives them. The model
    /// is parsed on the first call; a model that fails to parse fails every
    /// call, as before.
    pub fn decode_chunk(&mut self, payload: &[u8], frame_length: usize) -> Result<(f32, Vec<u16>)> {
        if self.model.is_none() {
            self.model = Some(QuantizedLmModel::new(&self.bundle_json, &self.weights)?);
            self.weights = Vec::new();
        }
        let model = self.model.as_mut().expect("the model was parsed above");
        model.decode_chunk(payload, frame_length)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use encodec_rs::chunk_decode::QuantizedLmChunkDecoder;
    use std::path::PathBuf;
    use std::time::Instant;

    /// The q8 LM bundle from a sibling encodec-rs checkout, if present.
    fn bundle() -> Option<(String, Vec<u8>)> {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../encodec-rs/onnx-bundles/encodec_48khz_12kbps_1333ms");
        let json = std::fs::read_to_string(dir.join("bundle.json")).ok()?;
        let weights = std::fs::read(dir.join("lm_weights_q8.bin")).ok()?;
        Some((json, weights))
    }

    /// Decoding chunks with the kept model gives each chunk the scale, codes
    /// and errors of a new chunk decoder per chunk.
    #[test]
    fn kept_model_matches_a_new_decoder_per_chunk() {
        let Some((json, weights)) = bundle() else {
            eprintln!("skipping: the q8 LM bundle is unavailable");
            return;
        };
        let meta: FrameBundleMetadata = serde_json::from_str(&json).unwrap();
        let mut decoder =
            BrowserDecoder::new(&json, &weights, "", meta.segment_stride * 8, 8, false).unwrap();
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let (mut kept, mut fresh) = (std::time::Duration::ZERO, std::time::Duration::ZERO);
        for round in 0..8 {
            let payload: Vec<u8> = (0..400 + round * 97)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    state as u8
                })
                .collect();
            let steps = if round == 5 { 100_000 } else { 8 };
            let start = Instant::now();
            let got = decoder
                .decode_chunk(&payload, steps)
                .map_err(|e| e.to_string());
            kept += start.elapsed();
            let start = Instant::now();
            let want = QuantizedLmChunkDecoder::new(&json, &weights, &payload)
                .and_then(|mut chunk| {
                    let scale = chunk.scale();
                    chunk.pull_all(steps).map(|codes| (scale, codes))
                })
                .map_err(|e| e.to_string());
            fresh += start.elapsed();
            let view = |r: &Result<(f32, Vec<u16>), String>| {
                r.as_ref()
                    .map(|(s, c)| (s.to_bits(), c.clone()))
                    .map_err(Clone::clone)
            };
            assert_eq!(view(&got), view(&want), "round {round}");
        }
        eprintln!("kept model {kept:?}, new decoder per chunk {fresh:?}");
    }
}
