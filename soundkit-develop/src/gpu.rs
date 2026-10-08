//! A plan as the GPU path reads it: one block of parameters, one buffer of
//! tables and grids, and the list of window operations to run before the
//! per-pixel pass. `gpu/develop.wgsl` declares the same layout. The browser
//! host in `gpu/index.mjs` runs the operations in order and knows nothing of
//! what they mean, so every decision stays in this crate.

use crate::curve::CURVE_TABLE;
use crate::develop::{tables, Frame, Plan, Samples, Tone, SCENE_TABLE, ZONE_STEPS};
use crate::filter;

/// Planes of the window, in the planes buffer.
pub mod plane {
    pub const LUM: u32 = 0;
    pub const CHROMA_R: u32 = 1;
    pub const CHROMA_B: u32 = 2;
    pub const TEXTURE: u32 = 3;
    pub const SHARP: u32 = 4;
    pub const NOISE_A: u32 = 5;
    pub const NOISE_B: u32 = 6;
    pub const T0: u32 = 7;
    pub const T1: u32 = 8;
    pub const T2: u32 = 9;
    pub const T3: u32 = 10;
    pub const COUNT: u32 = 11;
}

/// Window operations. Each is five words: the operation, then its arguments.
pub mod op {
    /// Samples the window into LUM, and into CHROMA_R and CHROMA_B when asked.
    pub const PREPARE: u32 = 1;
    /// src, dst, radius: a box blur across rows.
    pub const BOX_ROWS: u32 = 2;
    /// src, dst, radius: a box blur down columns.
    pub const BOX_COLUMNS: u32 = 3;
    /// src, dst: the square of each value.
    pub const SQUARE: u32 = 4;
    /// mean, correlation, epsilon bits; writes the guided filter's a and b into T2 and T3.
    pub const GUIDED: u32 = 5;
    /// src, dst: a copy.
    pub const COPY: u32 = 6;
}

/// Parameter words. `gpu/develop.wgsl` reads the same indices.
#[allow(missing_docs)]
pub mod p {
    pub const WIDTH: usize = 0; pub const HEIGHT: usize = 1;
    pub const RX: usize = 2; pub const RY: usize = 3; pub const RW: usize = 4; pub const RH: usize = 5;
    pub const WX0: usize = 6; pub const WY0: usize = 7; pub const WX1: usize = 8; pub const WY1: usize = 9;
    pub const FW: usize = 10; pub const FH: usize = 11; pub const FORMAT: usize = 12; pub const HAS_ALPHA: usize = 13; pub const EXACT: usize = 14;
    pub const XS: usize = 15; pub const YS: usize = 16; pub const M: usize = 17; pub const EXPOSURE: usize = 26;
    pub const UNTOUCHED_ON: usize = 27; pub const UNTOUCHED: usize = 28;
    pub const BIT_DEPTH: usize = 37; pub const MAX: usize = 38; pub const CLIPPING: usize = 39;
    pub const HAZE_ON: usize = 40; pub const HAZE_AMOUNT: usize = 41; pub const AIR: usize = 42; pub const HAZE_GW: usize = 45; pub const HAZE_GH: usize = 46; pub const HAZE_STEP: usize = 47;
    pub const BASE_ON: usize = 48; pub const BASE_GW: usize = 49; pub const BASE_GH: usize = 50; pub const BASE_STEP: usize = 51;
    pub const CLAR_ON: usize = 52; pub const CLAR_GW: usize = 53; pub const CLAR_GH: usize = 54; pub const CLAR_STEP: usize = 55;
    pub const SHADOWS: usize = 56; pub const HIGHLIGHTS: usize = 57; pub const CLARITY: usize = 58; pub const TEXTURE: usize = 59; pub const SHARPEN: usize = 60; pub const SHARPEN_MASKING: usize = 61;
    pub const NOISE_ON: usize = 62; pub const NOISE_K: usize = 63; pub const CHROMA_ON: usize = 64; pub const CHROMA_AMOUNT: usize = 65;
    pub const TONE_SCENE: usize = 66; pub const TONE_SCALE: usize = 67; pub const CLIP_AT: usize = 68; pub const BLACK: usize = 69; pub const WHITE: usize = 70;
    pub const CURVES_ON: usize = 71;
    pub const COLOUR_ON: usize = 72; pub const MONO: usize = 73; pub const MIXER_ACTIVE: usize = 74; pub const SATURATION: usize = 75; pub const VIBRANCE: usize = 76; pub const GRADING: usize = 77; pub const BALANCE: usize = 78;
    pub const MIXER: usize = 79; pub const TINT: usize = 103; pub const LIFT: usize = 111;
    pub const VIG_AMOUNT: usize = 115; pub const VIG_MID: usize = 116; pub const VIG_FEATHER: usize = 117; pub const GRAIN_AMOUNT: usize = 118; pub const GRAIN_CELL: usize = 119;
    pub const BAND_HUES: usize = 120;
    pub const TEXTURE_ON: usize = 128; pub const SHARP_ON: usize = 129;
    pub const OFF_DECODE: usize = 130; pub const OFF_ENCODE: usize = 131; pub const OFF_TONE: usize = 132; pub const OFF_CURVES: usize = 133;
    pub const OFF_SHADOW_W: usize = 134; pub const OFF_HIGHLIGHT_W: usize = 135; pub const OFF_HAZE: usize = 136;
    pub const OFF_BASE_A: usize = 137; pub const OFF_BASE_B: usize = 138; pub const OFF_CLAR_A: usize = 139; pub const OFF_CLAR_B: usize = 140;
    pub const SRC_TABLE: usize = 141; pub const SRC_ALPHA: usize = 142; pub const PLANE_SIZE: usize = 143; pub const NEED_CHROMA: usize = 144;
    pub const TURN_ON: usize = 145; pub const TURN_COS: usize = 146; pub const TURN_SIN: usize = 147;
    pub const COUNT: usize = 160;
}

/// Sample formats of the source buffer.
pub mod format {
    pub const F32: u32 = 0;
    pub const U16: u32 = 1;
    pub const TABLE8: u32 = 2;
    pub const TABLE16: u32 = 3;
}

pub struct GpuPlan { pub params: Vec<u32>, pub tables: Vec<f32>, pub ops: Vec<u32>, pub planes: u32 }

/// The source buffer's layout, for a frame: its format, the word where the
/// transfer table starts and the word where the opacity starts.
pub fn source_layout(frame: &Frame) -> (u32, u32, u32) {
    let pixels = frame.width * frame.height;
    let samples_words = match &frame.data {
        Samples::F32(_) => pixels * 3,
        Samples::U16(_) | Samples::Srgb16(_) => (pixels * 3).div_ceil(2),
        Samples::Srgb8(_) => (pixels * 3).div_ceil(4),
    };
    let table_words = match &frame.data { Samples::Srgb8(_) => 256, Samples::Srgb16(_) => 65536, _ => 0 };
    let format = match &frame.data { Samples::F32(_) => format::F32, Samples::U16(_) => format::U16, Samples::Srgb8(_) => format::TABLE8, Samples::Srgb16(_) => format::TABLE16 };
    (format, samples_words as u32, (samples_words + table_words) as u32)
}

/// The frame as the source buffer holds it: samples, then the transfer table
/// of an encoded photograph, then opacity packed four to a word.
pub fn source_words(frame: &Frame) -> Vec<u32> {
    let (_, table_at, alpha_at) = source_layout(frame);
    let mut words = vec![0u32; alpha_at as usize + frame.alpha.as_ref().map_or(0, |a| a.len().div_ceil(4))];
    match &frame.data {
        Samples::F32(v) => for (i, x) in v.iter().enumerate() { words[i] = x.to_bits(); },
        Samples::U16(v) => for (i, x) in v.iter().enumerate() { words[i / 2] |= (*x as u32) << (16 * (i % 2)); },
        Samples::Srgb16(v) => for (i, x) in v.iter().enumerate() { words[i / 2] |= (x.0 as u32) << (16 * (i % 2)); },
        Samples::Srgb8(v) => for (i, x) in v.iter().enumerate() { words[i / 4] |= (*x as u32) << (8 * (i % 4)); },
    }
    match &frame.data {
        Samples::Srgb8(_) => {
            let table: &[f32] = match &frame.curve { Some(crate::develop::Curve::Eight(t)) => &t[..], _ => &crate::develop::srgb8()[..] };
            for (i, x) in table.iter().enumerate() { words[table_at as usize + i] = x.to_bits(); }
        }
        Samples::Srgb16(_) => {
            let table: &[f32] = match &frame.curve { Some(crate::develop::Curve::Sixteen(t)) => &t[..], _ => &crate::develop::srgb16()[..] };
            for (i, x) in table.iter().enumerate() { words[table_at as usize + i] = x.to_bits(); }
        }
        _ => {}
    }
    if let Some(a) = &frame.alpha { for (i, x) in a.iter().enumerate() { words[alpha_at as usize + i / 4] |= (*x as u32) << (8 * (i % 4)); } }
    words
}

fn gaussian_ops(ops: &mut Vec<u32>, src: u32, dst: u32, radii: [usize; 3]) {
    let mut from = src;
    let mut any = false;
    for r in radii {
        if r == 0 { continue; }
        ops.extend([op::BOX_ROWS, from, plane::T0, r as u32, 0]);
        ops.extend([op::BOX_COLUMNS, plane::T0, dst, r as u32, 0]);
        from = dst;
        any = true;
    }
    if !any { ops.extend([op::COPY, src, dst, 0, 0]); }
}

/// The plan, laid out for the GPU path.
pub fn gpu_plan(frame: &Frame, p: &Plan) -> GpuPlan {
    let t = tables();
    let mut w = vec![0u32; p::COUNT];
    let f = |v: f32| v.to_bits();
    let mut tables_out: Vec<f32> = Vec::new();
    let mut push = |values: &[f32]| -> u32 { let at = tables_out.len() as u32; tables_out.extend_from_slice(values); at };

    w[p::WIDTH] = p.width as u32; w[p::HEIGHT] = p.height as u32;
    for i in 0..4 { w[p::RX + i] = p.region[i] as u32; w[p::WX0 + i] = p.window[i] as u32; }
    let (format, table_at, alpha_at) = source_layout(frame);
    w[p::FW] = frame.width as u32; w[p::FH] = frame.height as u32; w[p::FORMAT] = format;
    w[p::HAS_ALPHA] = frame.alpha.is_some() as u32;
    w[p::EXACT] = (frame.width == p.width && frame.height == p.height) as u32;
    w[p::XS] = f(frame.width as f32 / p.width as f32); w[p::YS] = f(frame.height as f32 / p.height as f32);
    w[p::SRC_TABLE] = table_at; w[p::SRC_ALPHA] = alpha_at;
    for i in 0..9 { w[p::M + i] = f(p.matrix[i]); }
    w[p::EXPOSURE] = f(p.exposure);
    if let Some(u) = p.untouched { w[p::UNTOUCHED_ON] = 1; for i in 0..9 { w[p::UNTOUCHED + i] = f(u[i]); } }
    w[p::BIT_DEPTH] = p.bit_depth as u32; w[p::MAX] = f(p.max); w[p::CLIPPING] = p.clipping as u32;

    w[p::OFF_DECODE] = push(&t.decode);
    w[p::OFF_ENCODE] = push(&t.encode);
    if let Some(h) = &p.haze {
        w[p::HAZE_ON] = 1; w[p::HAZE_AMOUNT] = f(h.amount);
        for i in 0..3 { w[p::AIR + i] = f(h.air[i]); }
        w[p::HAZE_GW] = h.grid.gw as u32; w[p::HAZE_GH] = h.grid.gh as u32; w[p::HAZE_STEP] = f(h.grid.step);
        w[p::OFF_HAZE] = push(&h.transmission);
    }
    if let Some(b) = &p.base {
        w[p::BASE_ON] = 1; w[p::BASE_GW] = b.grid.gw as u32; w[p::BASE_GH] = b.grid.gh as u32; w[p::BASE_STEP] = f(b.grid.step);
        w[p::OFF_BASE_A] = push(&b.a); w[p::OFF_BASE_B] = push(&b.b);
    }
    if let Some(b) = &p.clarity_base {
        w[p::CLAR_ON] = 1; w[p::CLAR_GW] = b.grid.gw as u32; w[p::CLAR_GH] = b.grid.gh as u32; w[p::CLAR_STEP] = f(b.grid.step);
        w[p::OFF_CLAR_A] = push(&b.a); w[p::OFF_CLAR_B] = push(&b.b);
    }
    w[p::SHADOWS] = f(p.shadows); w[p::HIGHLIGHTS] = f(p.highlights); w[p::CLARITY] = f(p.clarity);
    w[p::TEXTURE] = f(p.texture); w[p::SHARPEN] = f(p.sharpen); w[p::SHARPEN_MASKING] = f(p.sharpen_masking);
    w[p::TEXTURE_ON] = (p.texture != 0.0) as u32; w[p::SHARP_ON] = (p.sharpen != 0.0) as u32;
    if let Some([_, _, k]) = p.noise { w[p::NOISE_ON] = 1; w[p::NOISE_K] = f(k); }
    if let Some((_, amount)) = p.chroma { w[p::CHROMA_ON] = 1; w[p::CHROMA_AMOUNT] = f(amount); }
    match &p.tone {
        Tone::Scene { table } => { w[p::TONE_SCENE] = 1; debug_assert_eq!(table.len(), SCENE_TABLE + 1); w[p::OFF_TONE] = push(table); }
        Tone::Display { table, scale, clip_at, black, white } => {
            w[p::TONE_SCALE] = f(*scale); w[p::CLIP_AT] = f(*clip_at); w[p::BLACK] = f(*black); w[p::WHITE] = f(*white);
            w[p::OFF_TONE] = push(table);
        }
    }
    if let Some(curves) = &p.curves {
        w[p::CURVES_ON] = 1;
        w[p::OFF_CURVES] = push(&curves[0]);
        push(&curves[1]); push(&curves[2]);
        debug_assert_eq!(curves[0].len(), CURVE_TABLE + 1);
    }
    if let Some(c) = &p.colour {
        w[p::COLOUR_ON] = 1; w[p::MONO] = c.mono as u32; w[p::MIXER_ACTIVE] = c.mixer_active as u32;
        if let Some([cos, sin]) = c.turn { w[p::TURN_ON] = 1; w[p::TURN_COS] = f(cos); w[p::TURN_SIN] = f(sin); }
        w[p::SATURATION] = f(c.saturation); w[p::VIBRANCE] = f(c.vibrance); w[p::GRADING] = c.grading as u32; w[p::BALANCE] = f(c.balance);
        for k in 0..8 { for j in 0..3 { w[p::MIXER + k * 3 + j] = f(c.mixer[k][j]); } }
        for i in 0..4 { w[p::TINT + i * 2] = f(c.tint[i][0]); w[p::TINT + i * 2 + 1] = f(c.tint[i][1]); w[p::LIFT + i] = f(c.lift[i]); }
        debug_assert_eq!(c.shadow_weight.len(), ZONE_STEPS + 1);
        w[p::OFF_SHADOW_W] = push(&c.shadow_weight); w[p::OFF_HIGHLIGHT_W] = push(&c.highlight_weight);
    }
    for i in 0..8 { w[p::BAND_HUES + i] = f(t.band_hues[i]); }
    w[p::VIG_AMOUNT] = f(p.vignette[0]); w[p::VIG_MID] = f(p.vignette[1]); w[p::VIG_FEATHER] = f(p.vignette[2]);
    w[p::GRAIN_AMOUNT] = f(p.grain[0]); w[p::GRAIN_CELL] = f(p.grain[1]);

    let ww = p.window[2] - p.window[0];
    let wh = p.window[3] - p.window[1];
    w[p::PLANE_SIZE] = (ww * wh) as u32;
    w[p::NEED_CHROMA] = p.chroma.is_some() as u32;
    let mut ops = Vec::new();
    if p.uses_window() {
        ops.extend([op::PREPARE, 0, 0, 0, 0]);
        if p.texture != 0.0 { gaussian_ops(&mut ops, plane::LUM, plane::TEXTURE, filter::gaussian_radii(p.texture_sigma)); }
        if p.sharpen != 0.0 { let r = p.sharpen_radius; gaussian_ops(&mut ops, plane::LUM, plane::SHARP, [r, r, r]); }
        if let Some([sigma, eps, _]) = p.noise {
            let radii = filter::gaussian_radii(sigma);
            ops.extend([op::SQUARE, plane::LUM, plane::T1, 0, 0]);
            gaussian_ops(&mut ops, plane::T1, plane::NOISE_B, radii);
            gaussian_ops(&mut ops, plane::LUM, plane::NOISE_A, radii);
            ops.extend([op::GUIDED, plane::NOISE_A, plane::NOISE_B, eps.to_bits(), 0]);
            gaussian_ops(&mut ops, plane::T2, plane::NOISE_A, radii);
            gaussian_ops(&mut ops, plane::T3, plane::NOISE_B, radii);
        }
        if let Some((radius, _)) = p.chroma {
            for c in [plane::CHROMA_R, plane::CHROMA_B] {
                ops.extend([op::COPY, c, plane::T1, 0, 0]);
                gaussian_ops(&mut ops, plane::T1, c, [radius, radius, radius]);
            }
        }
    }
    GpuPlan { params: w, tables: tables_out, ops, planes: if p.uses_window() { plane::COUNT } else { 0 } }
}
