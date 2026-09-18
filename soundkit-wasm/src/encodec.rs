//! Model-backed EnCodec handler. JS supplies model-window results using its
//! existing lazy GPU/WASM backend; SoundKit owns entropy, seams and PCM output.
use serde::Serialize;
use soundkit_encodec::{browser, PcmSegment};
use wasm_bindgen::prelude::*;

fn to_js(value: impl Serialize) -> Result<JsValue, JsValue> {
    value
        .serialize(&serde_wasm_bindgen::Serializer::new().serialize_maps_as_objects(true))
        .map_err(|error| super::js_error(error.to_string()))
}

#[wasm_bindgen(js_name = encodecMetadata)]
pub fn metadata(payload: &[u8]) -> Result<JsValue, JsValue> {
    to_js(browser::ecdc_metadata(payload).map_err(|error| super::js_error(error.to_string()))?)
}

#[wasm_bindgen(js_name = encodecDecodeChunks)]
pub fn decode_chunks(bundle_json: &str, payload: &[u8]) -> Result<JsValue, JsValue> {
    to_js(
        browser::lm_ecdc_decode_chunks(bundle_json, payload)
            .map_err(|error| super::js_error(error.to_string()))?,
    )
}

#[wasm_bindgen]
pub struct WasmEncodecCodes {
    scale: f32,
    codes: Vec<u16>,
}

#[wasm_bindgen]
impl WasmEncodecCodes {
    pub fn scale(&self) -> f32 {
        self.scale
    }
    #[wasm_bindgen(js_name = takeCodes)]
    pub fn take_codes(&mut self) -> Vec<u16> {
        std::mem::take(&mut self.codes)
    }
}

#[wasm_bindgen]
pub struct WasmEncodecPcmBatch {
    spans: Vec<u32>,
    pcm: Vec<i16>,
}

impl From<Vec<PcmSegment>> for WasmEncodecPcmBatch {
    fn from(segments: Vec<PcmSegment>) -> Self {
        let mut result = Self {
            spans: Vec::with_capacity(segments.len() * 3),
            pcm: Vec::new(),
        };
        for segment in segments {
            result.spans.extend([
                segment.chunk_index as u32,
                segment.start_frame as u32,
                segment.end_frame as u32,
            ]);
            result.pcm.extend(segment.planar);
        }
        result
    }
}

#[wasm_bindgen]
impl WasmEncodecPcmBatch {
    #[wasm_bindgen(js_name = takeSegments)]
    pub fn take_segments(&mut self) -> Vec<u32> {
        std::mem::take(&mut self.spans)
    }
    #[wasm_bindgen(js_name = takePcm)]
    pub fn take_pcm(&mut self) -> Vec<i16> {
        std::mem::take(&mut self.pcm)
    }
}

#[wasm_bindgen]
pub struct WasmEncodecDecoder {
    inner: browser::BrowserDecoder,
}

#[wasm_bindgen]
impl WasmEncodecDecoder {
    #[wasm_bindgen(constructor)]
    pub fn new(
        bundle_json: &str,
        weights: &[u8],
        expected_hash: &str,
        audio_length: usize,
        frame_count: usize,
        retain_full: bool,
    ) -> Result<WasmEncodecDecoder, JsValue> {
        Ok(Self {
            inner: browser::BrowserDecoder::new(
                bundle_json,
                weights,
                expected_hash,
                audio_length,
                frame_count,
                retain_full,
            )
            .map_err(|error| super::js_error(error.to_string()))?,
        })
    }

    #[wasm_bindgen(js_name = decodeChunk)]
    pub fn decode_chunk(
        &self,
        payload: &[u8],
        frame_length: usize,
    ) -> Result<WasmEncodecCodes, JsValue> {
        let (scale, codes) = self
            .inner
            .decode_chunk(payload, frame_length)
            .map_err(|error| super::js_error(error.to_string()))?;
        Ok(WasmEncodecCodes { scale, codes })
    }

    #[wasm_bindgen(js_name = addDecodedFrame)]
    pub fn add_decoded_frame(&mut self, index: usize, window: &[f32]) -> Result<(), JsValue> {
        self.inner
            .pcm
            .add_decoded_frame(index, window)
            .map_err(|error| super::js_error(error.to_string()))
    }

    #[wasm_bindgen(js_name = addSilentFrame)]
    pub fn add_silent_frame(&mut self, index: usize) -> Result<(), JsValue> {
        self.inner
            .pcm
            .add_silent_frame(index)
            .map_err(|error| super::js_error(error.to_string()))
    }

    #[wasm_bindgen(js_name = addCachedRange)]
    pub fn add_cached_range(&mut self, start: usize, end: usize, pcm: &[i16]) -> bool {
        self.inner.pcm.add_cached_range(start, end, pcm)
    }

    #[wasm_bindgen(js_name = emitAfterBatch)]
    pub fn emit_after_batch(&mut self, next_index: usize) -> Result<WasmEncodecPcmBatch, JsValue> {
        self.inner
            .pcm
            .emit_after_batch(next_index)
            .map(Into::into)
            .map_err(|error| super::js_error(error.to_string()))
    }

    pub fn flush(&mut self) -> WasmEncodecPcmBatch {
        self.inner.pcm.flush().into()
    }

    #[wasm_bindgen(js_name = resultPcm)]
    pub fn result_pcm(&self) -> Vec<i16> {
        self.inner.pcm.result_pcm().to_vec()
    }
}
