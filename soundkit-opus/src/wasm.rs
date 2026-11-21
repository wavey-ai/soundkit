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

    /// Creates an Opus Custom encoder at 48 kHz for `frame_size`-sample
    /// frames: 48, 64 and 72 give 1, 1.33 and 1.5 ms frames. Packets are
    /// the `opus_custom_encode` format (one signalling byte, no TOC).
    /// Defaults follow libopus: the maximum bitrate (every packet fills the
    /// maximum packet size) and CBR; call `set_bitrate` for a target rate.
    #[wasm_bindgen(js_name = newCustom)]
    pub fn new_custom(channels: usize, frame_size: usize) -> Result<Encoder, JsValue> {
        let inner = RustEncoder::new_custom(48_000, channels, frame_size).map_err(opus_error)?;
        Ok(Self {
            inner,
            channels,
            frame_size,
            input: vec![0; frame_size * channels],
            input_i24: vec![0; frame_size * channels],
            output: Vec::new(),
        })
    }

    /// Sets the bitrate in bits per second; -1 selects the maximum (Opus
    /// Custom only).
    #[wasm_bindgen(js_name = set_bitrate)]
    pub fn set_bitrate(&mut self, bitrate: i32) -> Result<(), JsValue> {
        self.inner.set_bitrate(bitrate).map_err(opus_error)
    }

    /// Opus Custom: writes (`true`, the default) or omits the signalling byte.
    #[wasm_bindgen(js_name = set_custom_signalling)]
    pub fn set_custom_signalling(&mut self, enabled: bool) -> Result<(), JsValue> {
        self.inner
            .set_custom_signalling(enabled)
            .map_err(opus_error)
    }

    /// Opus Custom: the largest packet, signalling byte included (2-1276).
    #[wasm_bindgen(js_name = set_max_packet_bytes)]
    pub fn set_max_packet_bytes(&mut self, bytes: usize) -> Result<(), JsValue> {
        self.inner.set_max_packet_bytes(bytes).map_err(opus_error)
    }

    /// Selects libopus's encoder decision constants (`true`) or SoundKit's.
    #[wasm_bindgen(js_name = set_libopus_decisions)]
    pub fn set_libopus_decisions(&mut self, enabled: bool) {
        self.inner.set_libopus_decisions(enabled);
    }

    #[wasm_bindgen(getter, js_name = isCustom)]
    pub fn is_custom(&self) -> bool {
        self.inner.is_custom()
    }

    #[wasm_bindgen(js_name = enc_frame_f32)]
    pub fn enc_frame_f32(&mut self, input: &[f32]) -> Result<EncodeResult, JsValue> {
        let required = self.frame_size * self.channels;
        if input.len() < required {
            return Err(js_error(format!(
                "Opus encode input too short: got {}, need {required}",
                input.len()
            )));
        }
        let encoded = self
            .inner
            .encode_f32_vec(&input[..required], self.frame_size)
            .map_err(opus_error)?;
        Ok(EncodeResult { encoded })
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

    /// Selects constrained VBR (`true`, the default) or unconstrained VBR.
    #[wasm_bindgen(js_name = set_vbr_constraint)]
    pub fn set_vbr_constraint(&mut self, constrained: bool) -> Result<(), JsValue> {
        self.inner
            .set_vbr_constraint(constrained)
            .map_err(opus_error)
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

    /// Creates an Opus Custom decoder at 48 kHz for `frame_size`-sample
    /// frames (48, 64, 72 for 1, 1.33, 1.5 ms). An empty packet conceals a
    /// lost frame.
    #[wasm_bindgen(js_name = newCustom)]
    pub fn new_custom(channels: usize, frame_size: usize) -> Result<Decoder, JsValue> {
        let inner = RustDecoder::new_custom(48_000, channels, frame_size).map_err(opus_error)?;
        Ok(Self {
            inner,
            output: Vec::new(),
            output_i24: Vec::new(),
            decoded_size: 0,
        })
    }

    /// Opus Custom: expects (`true`, the default) or omits the signalling
    /// byte.
    #[wasm_bindgen(js_name = set_custom_signalling)]
    pub fn set_custom_signalling(&mut self, enabled: bool) -> Result<(), JsValue> {
        self.inner
            .set_custom_signalling(enabled)
            .map_err(opus_error)
    }

    #[wasm_bindgen(getter, js_name = isCustom)]
    pub fn is_custom(&self) -> bool {
        self.inner.is_custom()
    }

    /// Conceals one lost frame (packet-loss concealment) into the i16
    /// output.
    #[wasm_bindgen(js_name = dec_lost)]
    pub fn dec_lost(&mut self) -> Result<DecodeResult, JsValue> {
        self.dec_frame(&[])
    }

    /// Decodes to interleaved floating-point samples.
    #[wasm_bindgen(js_name = dec_frame_f32)]
    pub fn dec_frame_f32(&mut self, packet: &[u8]) -> Result<Vec<f32>, JsValue> {
        let mut output = Vec::new();
        self.decoded_size = self
            .inner
            .decode_f32_into(packet, false, &mut output)
            .map_err(opus_error)?;
        Ok(output)
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
