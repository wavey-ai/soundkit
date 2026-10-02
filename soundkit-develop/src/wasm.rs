//! The browser's engine: a frame is loaded once into WebAssembly memory and
//! developed as many times as the recipe changes.

use crate::colour::IDENTITY;
use crate::develop::{self, Camera, Frame, Options, Pixels, Samples};
use crate::recipe::Recipe;
use serde_json::Value;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Engine { frame: Frame }

#[wasm_bindgen]
pub struct Developed { inner: develop::Developed }

fn array<const N: usize>(value: &Value, key: &str, fallback: [f64; N]) -> [f64; N] {
    match value.get(key).and_then(Value::as_array) {
        Some(list) if list.len() >= N => { let mut out = fallback; for i in 0..N { out[i] = list[i].as_f64().unwrap_or(fallback[i]); } out }
        _ => fallback,
    }
}
/// The frame's camera facts from JSON: scale, wb, matrix, daylight and the
/// source width, as the decoder reports them.
fn frame_from(width: usize, height: usize, data: Samples, alpha: Option<Vec<u8>>, meta: &str) -> Frame {
    let v: Value = serde_json::from_str(meta).unwrap_or(Value::Null);
    let matrix = array::<9>(&v, "matrix", IDENTITY);
    let matrix = if matrix.iter().any(|x| *x != 0.0) { matrix } else { IDENTITY };
    let wb = array::<3>(&v, "wb", [1.0; 3]);
    let daylight = v.get("daylight").and_then(Value::as_array).filter(|d| d.len() >= 3).map(|_| array::<3>(&v, "daylight", [1.0; 3]));
    Frame { width, height, data, alpha,
        scale: v.get("scale").and_then(Value::as_f64).unwrap_or(1.0) as f32,
        wb: [wb[0] as f32, wb[1] as f32, wb[2] as f32], matrix,
        camera: Camera { matrix, wb, daylight },
        source_width: v.get("sourceWidth").and_then(Value::as_u64).map(|w| w as usize).unwrap_or(width) }
}

#[wasm_bindgen]
impl Engine {
    /// An 8-bit sRGB photograph, straight from a canvas.
    #[wasm_bindgen(js_name = fromRGBA)]
    pub fn from_rgba(width: usize, height: usize, rgba: &[u8]) -> Engine { Engine { frame: develop::from_rgba(width, height, rgba) } }
    /// A RAW frame's 16-bit sensor samples with its camera facts.
    #[wasm_bindgen(js_name = fromU16)]
    pub fn from_u16(width: usize, height: usize, data: Vec<u16>, meta: &str) -> Engine { Engine { frame: frame_from(width, height, Samples::U16(data), None, meta) } }
    /// A linear float frame with its camera facts.
    #[wasm_bindgen(js_name = fromF32)]
    pub fn from_f32(width: usize, height: usize, data: Vec<f32>, alpha: Option<Vec<u8>>, meta: &str) -> Engine { Engine { frame: frame_from(width, height, Samples::F32(data), alpha, meta) } }
    /// A small working copy for interactive edits.
    pub fn preview(&self, edge: usize) -> Engine { Engine { frame: self.frame.linear_preview(edge) } }
    #[wasm_bindgen(getter)] pub fn width(&self) -> usize { self.frame.width }
    #[wasm_bindgen(getter)] pub fn height(&self) -> usize { self.frame.height }
    #[wasm_bindgen(js_name = hasDaylightReference)]
    pub fn has_daylight_reference(&self) -> bool { self.frame.has_daylight_reference() }
    /// Develops the frame. `options` is JSON: edge, bitDepth, before, clipping.
    pub fn develop(&self, recipe: &str, options: &str) -> Developed {
        let o: Value = serde_json::from_str(options).unwrap_or(Value::Null);
        let options = Options {
            edge: o.get("edge").and_then(Value::as_f64).unwrap_or(0.0).max(0.0) as usize,
            bit_depth: o.get("bitDepth").and_then(Value::as_u64).unwrap_or(8) as u8,
            before: o.get("before").and_then(Value::as_bool).unwrap_or(false),
            clipping: o.get("clipping").and_then(Value::as_bool).unwrap_or(false),
            measure: false,
            // x, y, width, height within the full output, for a 100% view.
            region: o.get("region").and_then(Value::as_array).filter(|r| r.len() == 4)
                .map(|r| std::array::from_fn(|i| r[i].as_f64().unwrap_or(0.0).max(0.0) as usize)),
        };
        Developed { inner: develop::develop(&self.frame, &Recipe::from_json(recipe), &options) }
    }
    /// Auto tone, as JSON: exposure, contrast, highlights, shadows, whites, blacks.
    #[wasm_bindgen(js_name = autoTone)]
    pub fn auto_tone(&self, recipe: &str) -> String {
        let [exposure, contrast, highlights, shadows, whites, blacks] = develop::auto_tone(&self.frame, &Recipe::from_json(recipe));
        format!(r#"{{"exposure":{exposure},"contrast":{contrast},"highlights":{highlights},"shadows":{shadows},"whites":{whites},"blacks":{blacks}}}"#)
    }
}

#[wasm_bindgen]
impl Developed {
    #[wasm_bindgen(getter)] pub fn width(&self) -> usize { self.inner.width }
    #[wasm_bindgen(getter)] pub fn height(&self) -> usize { self.inner.height }
    #[wasm_bindgen(getter, js_name = bitDepth)] pub fn bit_depth(&self) -> u8 { self.inner.bit_depth }
    #[wasm_bindgen(getter, js_name = clippedHigh)] pub fn clipped_high(&self) -> u32 { self.inner.clipped_high }
    #[wasm_bindgen(getter, js_name = clippedLow)] pub fn clipped_low(&self) -> u32 { self.inner.clipped_low }
    // The pixel getters copy straight into a new typed array, so a read does
    // not first clone the image inside WASM memory.
    /// 8-bit output, RGBA.
    #[wasm_bindgen(js_name = dataU8)]
    pub fn data_u8(&self) -> js_sys::Uint8Array { match &self.inner.data { Pixels::U8(v) => js_sys::Uint8Array::from(v.as_slice()), Pixels::U16(_) => js_sys::Uint8Array::new_with_length(0) } }
    /// 10- or 12-bit output, RGBA.
    #[wasm_bindgen(js_name = dataU16)]
    pub fn data_u16(&self) -> js_sys::Uint16Array { match &self.inner.data { Pixels::U16(v) => js_sys::Uint16Array::from(v.as_slice()), Pixels::U8(_) => js_sys::Uint16Array::new_with_length(0) } }
    pub fn histogram(&self) -> Vec<u32> { self.inner.histogram.clone() }
    #[wasm_bindgen(js_name = histogramRGB)]
    pub fn histogram_rgb(&self) -> Vec<u32> { self.inner.histogram_rgb.clone() }
}
