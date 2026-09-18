use anyhow::Result;
use encodec_rs::ecdc::{encode_audio_to_ecdc_with_batch_size, FrameCodec, LmCodec};
use encodec_rs::metadata::FrameBundleMetadata;
use ndarray::{Array2, Array3, Array4};

pub const MODEL: usize = 64960;
pub const OWNED: usize = 64000;

pub fn window(channels: usize, index: usize) -> Vec<f32> {
    (0..channels * MODEL)
        .map(|sample| {
            let channel = sample / MODEL;
            (((sample % MODEL) as f32 * 0.001) + index as f32 * 0.7).sin()
                * if channel == 0 { 0.35 } else { -0.61 }
        })
        .collect()
}

fn metadata() -> FrameBundleMetadata {
    serde_json::from_value(serde_json::json!({
        "schema_version": 1, "model_name": "encodec_48khz_test", "bandwidth_kbps": 6.0,
        "sample_rate": 48000, "channels": 2, "segment_samples": MODEL, "segment_stride": OWNED,
        "normalize": true, "num_codebooks": 2, "frame_length": 203,
        "bits_per_codebook": 2, "codebook_cardinality": 4, "lm_cardinality": 4,
        "encode_model": "host", "decode_model": "host", "opset_version": 17,
        "lm_logit_step": 1.0, "lm_entropy_logit_step": 2.1
    }))
    .unwrap()
}

pub struct Model {
    meta: FrameBundleMetadata,
    pub decoded: usize,
}
impl FrameCodec for Model {
    fn metadata(&self) -> &FrameBundleMetadata {
        &self.meta
    }
    fn encode_frame(&mut self, audio: &Array3<f32>) -> Result<(Array3<i64>, Array2<f32>)> {
        Ok((
            Array3::zeros((audio.shape()[0], 2, 203)),
            Array2::ones((audio.shape()[0], 1)),
        ))
    }
    fn decode_frame(&mut self, _: &Array3<i64>, _: &Array2<f32>) -> Result<Array3<f32>> {
        let samples = window(2, self.decoded);
        self.decoded += 1;
        Ok(Array3::from_shape_vec((1, 2, MODEL), samples)?)
    }
}

pub struct Lm(FrameBundleMetadata);
impl LmCodec for Lm {
    fn metadata(&self) -> &FrameBundleMetadata {
        &self.0
    }
    fn bitstream_lm_hash(&self) -> Option<&str> {
        Some("handler-test-lm")
    }
    fn initial_states(&self, _: usize) -> Result<Vec<Array3<f32>>> {
        Ok(Vec::new())
    }
    fn forward_logits(
        &mut self,
        _: &Array3<i64>,
        offset: i64,
        _: &[Array3<f32>],
    ) -> Result<(Array4<f32>, i64, Vec<Array3<f32>>)> {
        Ok((Array4::zeros((1, 4, 2, 1)), offset + 1, Vec::new()))
    }
}

pub fn fixture() -> (Model, Lm, Vec<u8>, usize) {
    let mut model = Model {
        meta: metadata(),
        decoded: 0,
    };
    let mut lm = Lm(metadata());
    let frames = OWNED * 2 + 17;
    let source = Array3::zeros((1, 2, frames));
    let payload =
        encode_audio_to_ecdc_with_batch_size(&mut model, &mut lm, &source, None, 1).unwrap();
    (model, lm, payload, frames)
}
