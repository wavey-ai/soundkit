use crate::{Application, Decoder as RustDecoder, Encoder as RustEncoder, Error};
use wasm_bindgen::prelude::*;

fn js_error(error: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&error.to_string())
}

fn opus_error(error: Error) -> JsValue {
    js_error(format!("{error:?}"))
}

#[wasm_bindgen]
pub struct Encoder {
    inner: RustEncoder,
    channels: usize,
    frame_size: usize,
}

#[wasm_bindgen]
impl Encoder {
    #[wasm_bindgen(constructor)]
    pub fn new(
        channels: usize,
        sample_rate: i32,
        bitrate: i32,
        frame_size: usize,
    ) -> Result<Encoder, JsValue> {
        if sample_rate != 48_000 {
            return Err(js_error(
                "libopus-rs wasm currently supports 48 kHz CELT-only Opus",
            ));
        }
        let mut inner = RustEncoder::new(sample_rate, channels, Application::RestrictedLowDelay)
            .map_err(opus_error)?;
        inner.set_bitrate(bitrate).map_err(opus_error)?;
        inner.set_vbr(false).map_err(opus_error)?;
        Ok(Self {
            inner,
            channels,
            frame_size,
        })
    }

    #[wasm_bindgen(js_name = enc_frame)]
    pub fn enc_frame(&mut self, input: &[i16]) -> Result<EncodeResult, JsValue> {
        let required = self.frame_size * self.channels;
        if input.len() < required {
            return Err(js_error(format!(
                "Opus encode input too short: got {}, need {required}",
                input.len()
            )));
        }
        let encoded = self
            .inner
            .encode_i16(&input[..required], self.frame_size)
            .map_err(opus_error)?;
        Ok(EncodeResult { encoded })
    }

    #[wasm_bindgen(js_name = set_vbr)]
    pub fn set_vbr(&mut self, enabled: bool) -> Result<(), JsValue> {
        self.inner.set_vbr(enabled).map_err(opus_error)
    }

    pub fn destroy(self) {}
}

#[wasm_bindgen]
pub struct EncodeResult {
    encoded: Vec<u8>,
}

#[wasm_bindgen]
impl EncodeResult {
    #[wasm_bindgen(getter)]
    pub fn ok(&self) -> bool {
        true
    }

    #[wasm_bindgen(getter, js_name = encodedData)]
    pub fn encoded_data(&self) -> Vec<u8> {
        self.encoded.clone()
    }
}

#[wasm_bindgen]
pub struct Decoder {
    inner: RustDecoder,
    channels: usize,
}

#[wasm_bindgen]
impl Decoder {
    #[wasm_bindgen(constructor)]
    pub fn new(channels: usize, sample_rate: i32, _frame_size: usize) -> Result<Decoder, JsValue> {
        if sample_rate != 48_000 {
            return Err(js_error(
                "libopus-rs wasm currently supports 48 kHz CELT-only Opus",
            ));
        }
        let inner = RustDecoder::new(sample_rate, channels).map_err(opus_error)?;
        Ok(Self { inner, channels })
    }

    #[wasm_bindgen(js_name = dec_frame)]
    pub fn dec_frame(&mut self, packet: &[u8]) -> Result<DecodeResult, JsValue> {
        let output = self.inner.decode_i16(packet, false).map_err(opus_error)?;
        let decoded_size = output.len() / self.channels.max(1);
        Ok(DecodeResult {
            output,
            decoded_size,
        })
    }

    pub fn destroy(self) {}
}

#[wasm_bindgen]
pub struct DecodeResult {
    output: Vec<i16>,
    decoded_size: usize,
}

#[wasm_bindgen]
impl DecodeResult {
    #[wasm_bindgen(getter, js_name = decodedSize)]
    pub fn decoded_size(&self) -> usize {
        self.decoded_size
    }

    #[wasm_bindgen(getter)]
    pub fn output(&self) -> Vec<i16> {
        self.output.clone()
    }
}
