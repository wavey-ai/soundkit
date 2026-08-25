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
    input: Vec<i16>,
    input_i24: Vec<i32>,
    output: Vec<u8>,
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
                "soundkit-opus wasm currently supports 48 kHz CELT-only Opus",
            ));
        }
        let mut inner = RustEncoder::with_application(sample_rate, channels, Application::Audio)
            .map_err(opus_error)?;
        inner.set_bitrate(bitrate).map_err(opus_error)?;
        inner.set_vbr(false).map_err(opus_error)?;
        Ok(Self {
            inner,
            channels,
            frame_size,
            input: vec![0; frame_size * channels],
            input_i24: vec![0; frame_size * channels],
            output: Vec::new(),
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
            .encode_i16_vec(&input[..required], self.frame_size)
            .map_err(opus_error)?;
        Ok(EncodeResult { encoded })
    }

    #[wasm_bindgen(js_name = enc_frame_i24)]
    pub fn enc_frame_i24(&mut self, input: &[i32]) -> Result<EncodeResult, JsValue> {
        let required = self.frame_size * self.channels;
        if input.len() < required {
            return Err(js_error(format!(
                "Opus encode input too short: got {}, need {required}",
                input.len()
            )));
        }
        let encoded = self
            .inner
            .encode_i24_vec(&input[..required], self.frame_size)
            .map_err(opus_error)?;
        Ok(EncodeResult { encoded })
    }

    /// Encodes the samples in this encoder's input storage into its output
    /// storage and returns the packet size.
    ///
    /// JavaScript callers fill `inputPtr..inputLen`, then read the packet through
    /// `outputPtr..outputLen`. Views must be refreshed after Wasm memory grows.
    #[wasm_bindgen(js_name = enc_frame_reuse)]
    pub fn enc_frame_reuse(&mut self) -> Result<usize, JsValue> {
        self.inner
            .encode_i16_into(&self.input, self.frame_size, &mut self.output)
            .map_err(opus_error)
    }

    /// Encodes signed 24-bit samples from this encoder's staged `i32` input.
    #[wasm_bindgen(js_name = enc_frame_i24_reuse)]
    pub fn enc_frame_i24_reuse(&mut self) -> Result<usize, JsValue> {
        self.inner
            .encode_i24_into(&self.input_i24, self.frame_size, &mut self.output)
            .map_err(opus_error)
    }

    #[wasm_bindgen(getter, js_name = inputPtr)]
    pub fn input_ptr(&self) -> usize {
        self.input.as_ptr() as usize
    }

    #[wasm_bindgen(getter, js_name = inputLen)]
    pub fn input_len(&self) -> usize {
        self.input.len()
    }

    #[wasm_bindgen(getter, js_name = inputI24Ptr)]
    pub fn input_i24_ptr(&self) -> usize {
        self.input_i24.as_ptr() as usize
    }

    #[wasm_bindgen(getter, js_name = inputI24Len)]
    pub fn input_i24_len(&self) -> usize {
        self.input_i24.len()
    }

    #[wasm_bindgen(getter, js_name = outputPtr)]
    pub fn output_ptr(&self) -> usize {
        self.output.as_ptr() as usize
    }

    #[wasm_bindgen(getter, js_name = outputLen)]
    pub fn output_len(&self) -> usize {
        self.output.len()
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
    output: Vec<i16>,
    output_i24: Vec<i32>,
    decoded_size: usize,
}

#[wasm_bindgen]
impl Decoder {
    #[wasm_bindgen(constructor)]
    pub fn new(channels: usize, sample_rate: i32, _frame_size: usize) -> Result<Decoder, JsValue> {
        if sample_rate != 48_000 {
            return Err(js_error(
                "soundkit-opus wasm currently supports 48 kHz CELT-only Opus",
            ));
        }
        let inner = RustDecoder::new(sample_rate, channels).map_err(opus_error)?;
        Ok(Self {
            inner,
            output: Vec::new(),
            output_i24: Vec::new(),
            decoded_size: 0,
        })
    }

    #[wasm_bindgen(js_name = dec_frame)]
    pub fn dec_frame(&mut self, packet: &[u8]) -> Result<DecodeResult, JsValue> {
        self.decode_reuse(packet)?;
        Ok(DecodeResult {
            output: self.output.clone(),
            decoded_size: self.decoded_size,
        })
    }

    #[wasm_bindgen(js_name = dec_frame_reuse)]
    pub fn dec_frame_reuse(&mut self, packet: &[u8]) -> Result<usize, JsValue> {
        self.decode_reuse(packet)
    }

    #[wasm_bindgen(js_name = dec_frame_i24)]
    pub fn dec_frame_i24(&mut self, packet: &[u8]) -> Result<DecodeResultI24, JsValue> {
        self.decode_i24_reuse(packet)?;
        Ok(DecodeResultI24 {
            output: self.output_i24.clone(),
            decoded_size: self.decoded_size,
        })
    }

    #[wasm_bindgen(js_name = dec_frame_i24_reuse)]
    pub fn dec_frame_i24_reuse(&mut self, packet: &[u8]) -> Result<usize, JsValue> {
        self.decode_i24_reuse(packet)
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

    #[wasm_bindgen(getter, js_name = outputI24Ptr)]
    pub fn output_i24_ptr(&self) -> usize {
        self.output_i24.as_ptr() as usize
    }

    #[wasm_bindgen(getter, js_name = outputI24Len)]
    pub fn output_i24_len(&self) -> usize {
        self.output_i24.len()
    }

    pub fn destroy(self) {}

    fn decode_reuse(&mut self, packet: &[u8]) -> Result<usize, JsValue> {
        self.decoded_size = self
            .inner
            .decode_i16_into(packet, false, &mut self.output)
            .map_err(opus_error)?;
        Ok(self.decoded_size)
    }

    fn decode_i24_reuse(&mut self, packet: &[u8]) -> Result<usize, JsValue> {
        self.decoded_size = self
            .inner
            .decode_i24_into(packet, false, &mut self.output_i24)
            .map_err(opus_error)?;
        Ok(self.decoded_size)
    }
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

#[wasm_bindgen]
pub struct DecodeResultI24 {
    output: Vec<i32>,
    decoded_size: usize,
}

#[wasm_bindgen]
impl DecodeResultI24 {
    #[wasm_bindgen(getter, js_name = decodedSize)]
    pub fn decoded_size(&self) -> usize {
        self.decoded_size
    }

    #[wasm_bindgen(getter)]
    pub fn output(&self) -> Vec<i32> {
        self.output.clone()
    }
}
