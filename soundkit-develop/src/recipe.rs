//! The development recipe, read from JSON as loosely as the web reads it:
//! a missing or unreadable value takes its default, and every value is held
//! to its range.

use serde_json::Value;

pub const COLOUR_BANDS: [&str; 8] = ["red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta"];
pub const GRADING_ZONES: [&str; 4] = ["shadows", "midtones", "highlights", "global"];

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Profile { Standard, Vivid, Neutral, Monochrome }

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WhiteBalance { Shot, Auto, Daylight, Cloudy, Shade, Tungsten, Fluorescent, Flash }

impl WhiteBalance {
    /// A preset's light: correlated colour temperature in kelvin and its
    /// distance from the black-body locus (Duv, positive towards green).
    pub fn light(self) -> Option<(f64, f64)> {
        match self {
            WhiteBalance::Daylight => Some((5500.0, 0.0)),
            WhiteBalance::Cloudy => Some((6500.0, 0.0)),
            WhiteBalance::Shade => Some((7500.0, 0.0)),
            WhiteBalance::Tungsten => Some((3200.0, 0.0)),
            WhiteBalance::Fluorescent => Some((4000.0, 0.006)),
            WhiteBalance::Flash => Some((5600.0, 0.0)),
            WhiteBalance::Shot | WhiteBalance::Auto => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Band { pub hue: f32, pub saturation: f32, pub luminance: f32 }

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Zone { pub hue: f32, pub saturation: f32, pub luminance: f32 }

#[derive(Clone, Debug, PartialEq)]
pub struct Recipe {
    pub profile: Profile,
    pub white_balance: WhiteBalance,
    pub exposure: f32,
    pub temperature: f32,
    pub tint: f32,
    pub highlights: f32,
    pub shadows: f32,
    pub whites: f32,
    pub blacks: f32,
    pub contrast: f32,
    pub texture: f32,
    pub clarity: f32,
    pub dehaze: f32,
    pub vibrance: f32,
    pub saturation: f32,
    /// Highlights, lights, darks, shadows.
    pub curve: [f32; 4],
    pub mixer: [Band; 8],
    /// Shadows, midtones, highlights, global.
    pub zones: [Zone; 4],
    pub blending: f32,
    pub balance: f32,
    pub sharpen_amount: f32,
    pub sharpen_radius: f32,
    pub sharpen_masking: f32,
    pub noise_luminance: f32,
    pub noise_colour: f32,
    pub vignette_amount: f32,
    pub vignette_midpoint: f32,
    pub vignette_feather: f32,
    pub grain_amount: f32,
    pub grain_size: f32,
}

impl Default for Recipe {
    fn default() -> Self {
        Recipe {
            profile: Profile::Standard, white_balance: WhiteBalance::Shot,
            exposure: 0.0, temperature: 0.0, tint: 0.0, highlights: 0.0, shadows: 0.0, whites: 0.0, blacks: 0.0, contrast: 0.0,
            texture: 0.0, clarity: 0.0, dehaze: 0.0, vibrance: 0.0, saturation: 0.0,
            curve: [0.0; 4], mixer: [Band::default(); 8], zones: [Zone::default(); 4], blending: 50.0, balance: 0.0,
            sharpen_amount: 0.0, sharpen_radius: 1.0, sharpen_masking: 0.0, noise_luminance: 0.0, noise_colour: 0.0,
            vignette_amount: 0.0, vignette_midpoint: 50.0, vignette_feather: 50.0, grain_amount: 0.0, grain_size: 25.0,
        }
    }
}

/// A value as JavaScript's `Number()` reads it: absent is the fallback,
/// null and empty text are zero, true is one.
fn number(value: &Value, path: &[&str], fallback: f64) -> f64 {
    let mut at = value;
    for key in path {
        match at.get(*key) { Some(next) => at = next, None => return fallback }
    }
    let read = match at {
        Value::Null => Some(0.0),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Value::Number(n) => n.as_f64(),
        Value::String(s) => { let t = s.trim(); if t.is_empty() { Some(0.0) } else { t.parse::<f64>().ok() } }
        _ => None,
    };
    match read { Some(x) if x.is_finite() => x, _ => fallback }
}
fn clamped(value: &Value, path: &[&str], fallback: f64, lo: f64, hi: f64) -> f32 { number(value, path, fallback).clamp(lo, hi) as f32 }
fn signed(value: &Value, path: &[&str]) -> f32 { clamped(value, path, 0.0, -100.0, 100.0) }
fn unit(value: &Value, path: &[&str], fallback: f64) -> f32 { clamped(value, path, fallback, 0.0, 100.0) }
fn text<'a>(value: &'a Value, key: &str) -> Option<&'a str> { value.get(key).and_then(Value::as_str) }

impl Recipe {
    pub fn from_json(json: &str) -> Recipe {
        Recipe::from_value(&serde_json::from_str(json).unwrap_or(Value::Null))
    }
    pub fn from_value(v: &Value) -> Recipe {
        let mut r = Recipe::default();
        r.profile = match text(v, "profile") { Some("vivid") => Profile::Vivid, Some("neutral") => Profile::Neutral, Some("monochrome") => Profile::Monochrome, _ => Profile::Standard };
        r.white_balance = match text(v, "whiteBalance") {
            Some("auto") => WhiteBalance::Auto, Some("daylight") => WhiteBalance::Daylight, Some("cloudy") => WhiteBalance::Cloudy,
            Some("shade") => WhiteBalance::Shade, Some("tungsten") => WhiteBalance::Tungsten, Some("fluorescent") => WhiteBalance::Fluorescent,
            Some("flash") => WhiteBalance::Flash, _ => WhiteBalance::Shot,
        };
        r.exposure = clamped(v, &["exposure"], 0.0, -5.0, 5.0);
        r.temperature = signed(v, &["temperature"]); r.tint = signed(v, &["tint"]);
        r.highlights = signed(v, &["highlights"]); r.shadows = signed(v, &["shadows"]);
        r.whites = signed(v, &["whites"]); r.blacks = signed(v, &["blacks"]); r.contrast = signed(v, &["contrast"]);
        r.texture = signed(v, &["texture"]); r.clarity = signed(v, &["clarity"]); r.dehaze = signed(v, &["dehaze"]);
        r.vibrance = signed(v, &["vibrance"]); r.saturation = signed(v, &["saturation"]);
        for (i, key) in ["highlights", "lights", "darks", "shadows"].iter().enumerate() { r.curve[i] = signed(v, &["curve", key]); }
        for (i, band) in COLOUR_BANDS.iter().enumerate() {
            r.mixer[i] = Band { hue: signed(v, &["mixer", band, "hue"]), saturation: signed(v, &["mixer", band, "saturation"]), luminance: signed(v, &["mixer", band, "luminance"]) };
        }
        for (i, zone) in GRADING_ZONES.iter().enumerate() {
            let hue = number(v, &["grading", zone, "hue"], 0.0).rem_euclid(360.0) as f32;
            r.zones[i] = Zone { hue, saturation: unit(v, &["grading", zone, "saturation"], 0.0), luminance: signed(v, &["grading", zone, "luminance"]) };
        }
        r.blending = unit(v, &["grading", "blending"], 50.0);
        r.balance = signed(v, &["grading", "balance"]);
        r.sharpen_amount = clamped(v, &["sharpening", "amount"], 0.0, 0.0, 150.0);
        r.sharpen_radius = clamped(v, &["sharpening", "radius"], 1.0, 0.5, 3.0);
        r.sharpen_masking = unit(v, &["sharpening", "masking"], 0.0);
        r.noise_luminance = unit(v, &["noise", "luminance"], 0.0);
        r.noise_colour = unit(v, &["noise", "colour"], 0.0);
        r.vignette_amount = signed(v, &["vignette", "amount"]);
        r.vignette_midpoint = unit(v, &["vignette", "midpoint"], 50.0);
        r.vignette_feather = unit(v, &["vignette", "feather"], 50.0);
        r.grain_amount = unit(v, &["grain", "amount"], 0.0);
        r.grain_size = unit(v, &["grain", "size"], 25.0);
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reads_like_the_web() {
        let r = Recipe::from_json(r#"{"exposure":9,"contrast":"20","clarity":null,"curve":{"darks":-300},"grading":{"shadows":{"hue":-30}},"mixer":{"orange":{"saturation":12}}}"#);
        assert_eq!(r.exposure, 5.0);
        assert_eq!(r.contrast, 20.0);
        assert_eq!(r.clarity, 0.0);
        assert_eq!(r.curve[2], -100.0);
        assert_eq!(r.zones[0].hue, 330.0);
        assert_eq!(r.mixer[1].saturation, 12.0);
        assert_eq!(r.blending, 50.0);
        assert_eq!(Recipe::from_json("not json"), Recipe::default());
    }
}
