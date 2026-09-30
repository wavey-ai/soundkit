//! Development: a linear frame and a recipe in, display pixels out.

use crate::colour::{self, Matrix, IDENTITY};
use crate::fast;
use std::cell::RefCell;
use std::rc::Rc;
use crate::recipe::{Profile, Recipe, WhiteBalance};
use std::sync::OnceLock;

/// The sensor or photograph samples a frame holds: RAW stays 16-bit, a
/// photograph is linear float.
pub enum Samples { F32(Vec<f32>), U16(Vec<u16>) }

impl Samples {
    #[inline(always)]
    fn at(&self, i: usize) -> f32 { match self { Samples::F32(v) => v[i], Samples::U16(v) => v[i] as f32 } }
}

/// A stored sample, read as a float.
pub trait Texel: Copy { fn value(self) -> f32; }
impl Texel for f32 { #[inline(always)] fn value(self) -> f32 { self } }
impl Texel for u16 { #[inline(always)] fn value(self) -> f32 { self as f32 } }

/// The camera a frame came from. A proxy has its camera's white balance
/// baked in and keeps these facts to change it later.
#[derive(Clone, Copy, Debug)]
pub struct Camera { pub matrix: Matrix, pub wb: [f64; 3], pub daylight: Option<[f64; 3]> }

pub struct Frame {
    pub width: usize,
    pub height: usize,
    pub data: Samples,
    pub alpha: Option<Vec<u8>>,
    pub scale: f32,
    pub wb: [f32; 3],
    pub matrix: Matrix,
    pub camera: Camera,
    /// The full-resolution width this frame stands for: radii given in
    /// source pixels shrink with a preview.
    pub source_width: usize,
}

fn to_linear(v: f32) -> f32 { if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) } }
fn to_srgb(v: f32) -> f32 { if v <= 0.0031308 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 } }
#[inline(always)]
fn smooth(a: f32, b: f32, x: f32) -> f32 { let d = b - a; let t = ((x - a) / if d == 0.0 { 1e-6 } else { d }).clamp(0.0, 1.0); t * t * (3.0 - 2.0 * t) }
#[inline(always)]
fn luma(r: f32, g: f32, b: f32) -> f32 { 0.2126 * r + 0.7152 * g + 0.0722 * b }

const TABLE: usize = 4096;
struct Tables { soft_power: Vec<f32>, decode: Vec<f32>, encode: Vec<f32>, band_hues: [f32; 8], band_gap: [f32; 8] }
fn tables() -> &'static Tables {
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(|| {
        let band_hues = colour::band_hues();
        let mut band_gap = [0.0; 8];
        for i in 0..8 {
            let next = (band_hues[(i + 1) % 8] - band_hues[i] + 360.0) % 360.0;
            let previous = (band_hues[i] - band_hues[(i + 7) % 8] + 360.0) % 360.0;
            band_gap[i] = next.min(previous) / 2.0;
        }
        Tables {
            // Luminance to the 0.3 power over 0-8, indexed by the square root
            // so dark values keep their precision.
            soft_power: (0..=TABLE).map(|i| (((i as f32 / TABLE as f32).powi(2) * 8.0) + 1e-4).powf(0.3)).collect(),
            decode: (0..=TABLE).map(|i| to_linear(i as f32 / TABLE as f32)).collect(),
            // Encoding is indexed by the square root of linear light.
            encode: (0..=TABLE).map(|i| to_srgb((i as f32 / TABLE as f32).powi(2))).collect(),
            band_hues, band_gap,
        }
    })
}
#[inline(always)]
fn lookup(table: &[f32], p: f32) -> f32 { let k = (p as usize).min(TABLE - 1); table[k] + (table[k + 1] - table[k]) * (p - k as f32) }

/// Linear frame from 8-bit sRGB RGBA.
pub fn from_rgba(width: usize, height: usize, rgba: &[u8]) -> Frame {
    let ramp: Vec<f32> = (0..256).map(|i| to_linear(i as f32 / 255.0)).collect();
    let pixels = width * height;
    let mut data = Vec::with_capacity(pixels * 3);
    let mut alpha = Vec::with_capacity(pixels);
    for i in 0..pixels {
        data.push(ramp[rgba[i * 4] as usize]); data.push(ramp[rgba[i * 4 + 1] as usize]); data.push(ramp[rgba[i * 4 + 2] as usize]);
        alpha.push(rgba[i * 4 + 3]);
    }
    Frame { width, height, data: Samples::F32(data), alpha: Some(alpha), scale: 1.0, wb: [1.0; 3], matrix: IDENTITY,
        camera: Camera { matrix: IDENTITY, wb: [1.0; 3], daylight: None }, source_width: width }
}

impl Frame {
    pub fn has_daylight_reference(&self) -> bool { self.camera.daylight.is_some() }

    /// Box-filters the frame once into a small working image every edit reuses.
    pub fn linear_preview(&self, edge: usize) -> Frame {
        let factor = (edge as f64 / self.width.max(self.height) as f64).min(1.0);
        let width = ((self.width as f64 * factor).round() as usize).max(1);
        let height = ((self.height as f64 * factor).round() as usize).max(1);
        let mut data = vec![0.0f32; width * height * 3];
        let mut alpha = self.alpha.as_ref().map(|_| vec![0u8; width * height]);
        let m = self.matrix;
        let (sr, sg, sb) = (self.scale * self.wb[0], self.scale * self.wb[1], self.scale * self.wb[2]);
        for y in 0..height { for x in 0..width {
            let left = x * self.width / width; let right = ((x + 1) * self.width / width).max(left + 1);
            let top = y * self.height / height; let bottom = ((y + 1) * self.height / height).max(top + 1);
            let (mut acc, mut count, mut opacity) = ([0.0f64; 3], 0u32, 0u32);
            for sy in top..bottom { for sx in left..right {
                let p = sy * self.width + sx;
                let (r, g, b) = (self.data.at(p * 3) * sr, self.data.at(p * 3 + 1) * sg, self.data.at(p * 3 + 2) * sb);
                let (r, g, b) = (r as f64, g as f64, b as f64);
                acc[0] += m[0] * r + m[1] * g + m[2] * b; acc[1] += m[3] * r + m[4] * g + m[5] * b; acc[2] += m[6] * r + m[7] * g + m[8] * b;
                if let Some(a) = &self.alpha { opacity += a[p] as u32; }
                count += 1;
            } }
            let d = (y * width + x) * 3;
            for c in 0..3 { data[d + c] = (acc[c] / count as f64) as f32; }
            if let Some(a) = alpha.as_mut() { a[y * width + x] = ((opacity as f64 / count as f64).round()) as u8; }
        } }
        Frame { width, height, data: Samples::F32(data), alpha, scale: 1.0, wb: [1.0; 3], matrix: IDENTITY,
            camera: self.camera, source_width: self.source_width }
    }

    /// The transform that changes the as-shot white balance to a RAW preset.
    fn white_balance_transform(&self, choice: WhiteBalance) -> Option<Matrix> {
        let (kelvin, duv) = choice.light()?;
        let daylight = self.camera.daylight?;
        let inverse = colour::invert(&self.camera.matrix)?;
        let to_daylight = [daylight[0] / nonzero(self.camera.wb[0]), daylight[1] / nonzero(self.camera.wb[1]), daylight[2] / nonzero(self.camera.wb[2])];
        // The camera's own daylight balance, then an adaptation from the lamp to daylight.
        let correction = colour::adaptation(colour::white_of(kelvin, duv), colour::white_of(5500.0, 0.0));
        Some(colour::multiply(&correction, &colour::multiply(&self.camera.matrix, &colour::multiply(&colour::diagonal(to_daylight), &inverse))))
    }
}
fn nonzero(v: f64) -> f64 { if v == 0.0 { 1.0 } else { v } }

/// Reads an output pixel's working-space linear colour after white balance
/// and exposure, with its opacity. Generic over the stored sample, so the
/// read compiles into the loop.
struct Sampler<'a, T: Texel> {
    data: &'a [T], alpha: Option<&'a [u8]>, fw: usize, fh: usize,
    m: [f32; 9], n: [f32; 9], balanced: bool, k: f32, s: [f32; 3], xs: f32, ys: f32, exact: bool,
}
impl<'a, T: Texel> Sampler<'a, T> {
    fn new(frame: &'a Frame, data: &'a [T], width: usize, height: usize, balance: Option<Matrix>, exposure: f32) -> Self {
        Sampler { data, alpha: frame.alpha.as_deref(), fw: frame.width, fh: frame.height,
            m: frame.matrix.map(|v| v as f32), n: balance.unwrap_or(IDENTITY).map(|v| v as f32), balanced: balance.is_some(), k: exposure,
            s: [frame.scale * frame.wb[0], frame.scale * frame.wb[1], frame.scale * frame.wb[2]],
            xs: frame.width as f32 / width as f32, ys: frame.height as f32 / height as f32,
            // Output and frame the same size: one sample per pixel, no blend.
            exact: frame.width == width && frame.height == height }
    }
    #[inline(always)]
    fn sample(&self, x: usize, y: usize) -> [f32; 4] {
        let (fw, fh, d) = (self.fw, self.fh, self.data);
        let (cr, cg, cb, alpha);
        if self.exact {
            let p = y * fw + x;
            cr = d[p * 3].value(); cg = d[p * 3 + 1].value(); cb = d[p * 3 + 2].value();
            alpha = match self.alpha { Some(a) => a[p] as f32 / 255.0, None => 1.0 };
        } else {
            let sx = ((x as f32 + 0.5) * self.xs - 0.5).clamp(0.0, (fw - 1) as f32);
            let sy = ((y as f32 + 0.5) * self.ys - 0.5).clamp(0.0, (fh - 1) as f32);
            let (x0, y0) = (sx as usize, sy as usize);
            let (x1, y1) = ((x0 + 1).min(fw - 1), (y0 + 1).min(fh - 1));
            let (dx, dy) = (sx - x0 as f32, sy - y0 as f32);
            let (p00, p10, p01, p11) = ((y0 * fw + x0) * 3, (y0 * fw + x1) * 3, (y1 * fw + x0) * 3, (y1 * fw + x1) * 3);
            let (w00, w10, w01, w11) = ((1.0 - dx) * (1.0 - dy), dx * (1.0 - dy), (1.0 - dx) * dy, dx * dy);
            cr = d[p00].value() * w00 + d[p10].value() * w10 + d[p01].value() * w01 + d[p11].value() * w11;
            cg = d[p00 + 1].value() * w00 + d[p10 + 1].value() * w10 + d[p01 + 1].value() * w01 + d[p11 + 1].value() * w11;
            cb = d[p00 + 2].value() * w00 + d[p10 + 2].value() * w10 + d[p01 + 2].value() * w01 + d[p11 + 2].value() * w11;
            alpha = match self.alpha {
                Some(a) => (a[p00 / 3] as f32 * w00 + a[p10 / 3] as f32 * w10 + a[p01 / 3] as f32 * w01 + a[p11 / 3] as f32 * w11) / 255.0,
                None => 1.0,
            };
        }
        let (cr, cg, cb) = (cr * self.s[0], cg * self.s[1], cb * self.s[2]);
        let m = &self.m;
        let (mut r, mut g, mut b) = (m[0] * cr + m[1] * cg + m[2] * cb, m[3] * cr + m[4] * cg + m[5] * cb, m[6] * cr + m[7] * cg + m[8] * cb);
        if self.balanced {
            let n = &self.n;
            let (r1, g1, b1) = (n[0] * r + n[1] * g + n[2] * b, n[3] * r + n[4] * g + n[5] * b, n[6] * r + n[7] * g + n[8] * b);
            r = r1; g = g1; b = b1;
        }
        [(r * self.k).max(0.0), (g * self.k).max(0.0), (b * self.k).max(0.0), alpha]
    }
}

/// Three box passes approximate a gaussian; `radius` is in pixels. The
/// vertical pass walks rows in order with one running sum per column, so it
/// reads memory the way it is laid out.
fn blur(source: &[f32], width: usize, height: usize, radius: f32) -> Vec<f32> {
    let r = radius.round().max(0.0) as usize;
    let mut a = source.to_vec();
    if r == 0 || width == 0 || height == 0 { return a; }
    let mut b = vec![0.0f32; source.len()];
    let size = (2 * r + 1) as f32;
    let (w, h) = (width, height);
    let mut sums = vec![0.0f32; w];
    for _ in 0..3 {
        for y in 0..h {
            let row = &a[y * w..(y + 1) * w];
            let out = &mut b[y * w..(y + 1) * w];
            let mut sum = 0.0;
            for i in 0..=2 * r { sum += row[i.saturating_sub(r).min(w - 1)]; }
            for x in 0..w {
                out[x] = sum / size;
                sum += row[(x + r + 1).min(w - 1)] - row[x.saturating_sub(r)];
            }
        }
        sums.iter_mut().for_each(|v| *v = 0.0);
        for i in 0..=2 * r {
            let yy = i.saturating_sub(r).min(h - 1);
            for (s, v) in sums.iter_mut().zip(&b[yy * w..(yy + 1) * w]) { *s += *v; }
        }
        for y in 0..h {
            let (add, sub) = ((y + r + 1).min(h - 1), y.saturating_sub(r));
            let out = &mut a[y * w..(y + 1) * w];
            for x in 0..w { out[x] = sums[x] / size; }
            for x in 0..w { sums[x] += b[add * w + x] - b[sub * w + x]; }
        }
    }
    a
}

/// A small grid of the picture, sampled bilinearly: wide blurs and the haze
/// estimate read it instead of the full image.
struct Grid { gw: usize, gh: usize, width: f32, height: f32 }
impl Grid {
    fn new(width: usize, height: usize) -> Grid {
        let cells = 160.0f64;
        let gw = (if width >= height { cells } else { cells * width as f64 / height as f64 }).round().max(2.0) as usize;
        let gh = (if height > width { cells } else { cells * height as f64 / width as f64 }).round().max(2.0) as usize;
        Grid { gw, gh, width: width as f32, height: height as f32 }
    }
    #[inline(always)]
    fn at(&self, values: &[f32], x: usize, y: usize) -> f32 {
        let gx = ((x as f32 + 0.5) / self.width * self.gw as f32 - 0.5).clamp(0.0, (self.gw - 1) as f32);
        let gy = ((y as f32 + 0.5) / self.height * self.gh as f32 - 0.5).clamp(0.0, (self.gh - 1) as f32);
        let (x0, y0) = (gx as usize, gy as usize);
        let (x1, y1) = ((x0 + 1).min(self.gw - 1), (y0 + 1).min(self.gh - 1));
        let (dx, dy) = (gx - x0 as f32, gy - y0 as f32);
        let g = self.gw;
        (values[y0 * g + x0] * (1.0 - dx) + values[y0 * g + x1] * dx) * (1.0 - dy) + (values[y1 * g + x0] * (1.0 - dx) + values[y1 * g + x1] * dx) * dy
    }
    /// The output pixel at sub-sample (i, j) of 3x3 in a cell.
    fn probe(&self, gx: usize, gy: usize, i: usize, j: usize, width: usize, height: usize) -> (usize, usize) {
        ((((gx as f64 + (i as f64 + 0.5) / 3.0) * width as f64 / self.gw as f64) as usize).min(width - 1),
            (((gy as f64 + (j as f64 + 0.5) / 3.0) * height as f64 / self.gh as f64) as usize).min(height - 1))
    }
}

// Hash noise: the same grain for the same pixel on every render.
#[inline(always)]
fn hash(x: i64, y: i64) -> f32 {
    let mut h = (x.wrapping_mul(374761393).wrapping_add(y.wrapping_mul(668265263))) as u32;
    h = (h ^ (h >> 13)).wrapping_mul(1274126177);
    ((h ^ (h >> 16)) as f64 / 4294967295.0 * 2.0 - 1.0) as f32
}
#[inline(always)]
fn grain_at(x: usize, y: usize, cell: f32) -> f32 {
    let (gx, gy) = (x as f32 / cell, y as f32 / cell);
    let (x0, y0) = (gx.floor(), gy.floor());
    let (dx, dy) = (gx - x0, gy - y0);
    let (sx, sy) = (dx * dx * (3.0 - 2.0 * dx), dy * dy * (3.0 - 2.0 * dy));
    let (x0, y0) = (x0 as i64, y0 as i64);
    let top = hash(x0, y0) * (1.0 - sx) + hash(x0 + 1, y0) * sx;
    let bottom = hash(x0, y0 + 1) * (1.0 - sx) + hash(x0 + 1, y0 + 1) * sx;
    top * (1.0 - sy) + bottom * sy
}

pub struct Options { pub edge: usize, pub bit_depth: u8, pub before: bool, pub clipping: bool, pub measure: bool }
impl Default for Options { fn default() -> Self { Options { edge: 0, bit_depth: 8, before: false, clipping: false, measure: false } } }

pub enum Pixels { U8(Vec<u8>), U16(Vec<u16>) }
pub struct Developed {
    pub width: usize, pub height: usize, pub bit_depth: u8, pub data: Pixels,
    pub histogram: Vec<u32>, pub histogram_rgb: Vec<u32>, pub clipped_high: u32, pub clipped_low: u32,
    pub luminance: Option<Vec<f32>>,
}

pub fn develop(frame: &Frame, recipe: &Recipe, options: &Options) -> Developed {
    match &frame.data {
        Samples::F32(data) => develop_on(frame, data.as_slice(), recipe, options),
        Samples::U16(data) => develop_on(frame, data.as_slice(), recipe, options),
    }
}

thread_local! {
    // The display and tone curves only change with Contrast, Shadows and
    // Highlights, so a drag of any other slider reuses them.
    static DISPLAY: RefCell<Option<(u32, Rc<Vec<f32>>)>> = const { RefCell::new(None) };
    static TONES: RefCell<Option<((u32, u32), Rc<Vec<f32>>)>> = const { RefCell::new(None) };
}
fn display_curve(contrast: f64, display_scale: f32) -> Rc<Vec<f32>> {
    let key = (contrast as f32).to_bits();
    DISPLAY.with(|cache| {
        if let Some((k, v)) = cache.borrow().as_ref() { if *k == key { return v.clone(); } }
        let v = Rc::new((0..65536).map(|i| (to_srgb((0.18 * ((i as f64 / display_scale as f64 / 0.18).powf(contrast))) as f32)).clamp(0.0, 1.0)).collect::<Vec<f32>>());
        *cache.borrow_mut() = Some((key, v.clone()));
        v
    })
}
fn tone_curve(shadows: f32, highlights: f32) -> Rc<Vec<f32>> {
    let key = (shadows.to_bits(), highlights.to_bits());
    TONES.with(|cache| {
        if let Some((k, v)) = cache.borrow().as_ref() { if *k == key { return v.clone(); } }
        let v = Rc::new((0..16385).map(|i| {
            let l = i as f32 / 2048.0;
            2f32.powf((shadows * (-l * 6.0).exp() + highlights * (1.0 - (-l * 1.5).exp())) / 100.0)
        }).collect::<Vec<f32>>());
        *cache.borrow_mut() = Some((key, v.clone()));
        v
    })
}

fn develop_on<T: Texel>(frame: &Frame, data: &[T], recipe: &Recipe, options: &Options) -> Developed {
    let default = Recipe::default();
    let s = if options.before { &default } else { recipe };
    let t = tables();
    let factor = if options.edge > 0 { (options.edge as f64 / frame.width.max(frame.height) as f64).min(1.0) } else { 1.0 };
    let width = ((frame.width as f64 * factor).round() as usize).max(1);
    let height = ((frame.height as f64 * factor).round() as usize).max(1);
    let pixels = width * height;
    let bit_depth = if [8, 10, 12].contains(&options.bit_depth) { options.bit_depth } else { 8 };
    let max = ((1u32 << bit_depth) - 1) as f32;
    let mut out8 = if bit_depth == 8 { vec![0u8; pixels * 4] } else { Vec::new() };
    let mut out16 = if bit_depth != 8 { vec![0u16; pixels * 4] } else { Vec::new() };
    let mut histogram = vec![0u32; 256];
    let mut histogram_rgb = vec![0u32; 768];
    let detail_scale = width as f32 / frame.source_width as f32;
    let long_edge = width.max(height) as f32;
    let exposure = 2f32.powf(s.exposure);
    let contrast = 2f64.powf(s.contrast as f64 / 150.0);
    let black = s.blacks / 1000.0;
    let white = 2f32.powf(s.whites / 200.0);
    // The display curve has 16-bit linear resolution before quantising.
    let clip_at = (0.18 * (1.0 / 0.18f64).powf(1.0 / contrast)) as f32;
    let display_scale = 65535.0 / clip_at;
    let display = display_curve(contrast, display_scale);
    let tones = tone_curve(s.shadows, s.highlights);

    let mut balance = match s.white_balance { WhiteBalance::Shot | WhiteBalance::Auto => None, preset => frame.white_balance_transform(preset) };
    if s.white_balance == WhiteBalance::Auto {
        // The mid-tones' average colour is taken as the light and adapted to neutral.
        let probe = Sampler::new(frame, data, width, height, None, 1.0);
        let step = ((long_edge / 200.0).round() as usize).max(1);
        let mut sum = [0.0f64; 3];
        for y in (0..height).step_by(step) { for x in (0..width).step_by(step) {
            let p = probe.sample(x, y);
            let l = luma(p[0], p[1], p[2]);
            if l < 0.02 || l > 0.9 { continue; }
            sum[0] += p[0] as f64; sum[1] += p[1] as f64; sum[2] += p[2] as f64;
        } }
        if sum[0] > 0.0 && sum[1] > 0.0 && sum[2] > 0.0 {
            let light = colour::apply(&colour::RGB_TO_XYZ, sum);
            balance = Some(colour::adaptation([light[0] / light[1], 1.0, light[2] / light[1]], colour::apply(&colour::RGB_TO_XYZ, [1.0; 3])));
        }
    }
    if let Some(shift) = colour::temperature_tint(s.temperature as f64, s.tint as f64) {
        balance = Some(match balance { Some(b) => colour::multiply(&shift, &b), None => shift });
    }
    let sampler = Sampler::new(frame, data, width, height, balance, exposure);
    // Clipping is reported for what the edit did.
    let untouched = if options.before { None } else { Some(Sampler::new(frame, data, width, height, None, 1.0)) };
    let clip_at0 = 1.0 - 1e-6;

    // ------------------------------------------------ what this edit needs
    let tones_active = s.highlights != 0.0 || s.shadows != 0.0;
    let clarity = s.clarity / 100.0; let texture = s.texture / 100.0; let dehaze = s.dehaze / 100.0;
    let sharpen = s.sharpen_amount / 100.0; let nr_luminance = s.noise_luminance / 100.0; let nr_colour = s.noise_colour / 100.0;
    let local = tones_active || clarity != 0.0 || texture != 0.0 || dehaze != 0.0 || sharpen != 0.0 || nr_luminance != 0.0 || nr_colour != 0.0;
    let g = Grid::new(width, height);
    let cells = g.gw * g.gh;
    // Haze: the dark channel on the grid; the atmosphere is the brightest haze.
    let haze: Option<([f32; 3], Vec<f32>)> = if dehaze != 0.0 {
        let mut dark = vec![0.0f32; cells];
        let mut colours = vec![0.0f32; cells * 3];
        for gy in 0..g.gh { for gx in 0..g.gw {
            let cell = gy * g.gw + gx;
            let mut lowest = f32::INFINITY;
            for j in 0..3 { for i in 0..3 {
                let (px, py) = g.probe(gx, gy, i, j, width, height);
                let rgb = sampler.sample(px, py);
                lowest = lowest.min(rgb[0]).min(rgb[1]).min(rgb[2]);
                for c in 0..3 { colours[cell * 3 + c] += rgb[c] / 9.0; }
            } }
            dark[cell] = lowest;
        } }
        let mut order: Vec<usize> = (0..cells).collect();
        order.sort_by(|a, b| dark[*b].partial_cmp(&dark[*a]).unwrap_or(std::cmp::Ordering::Equal));
        order.truncate(((cells as f32 * 0.01).round() as usize).max(1));
        let mut air = [0.0f32; 3];
        for cell in &order { for c in 0..3 { air[c] += colours[cell * 3 + c] / order.len() as f32; } }
        let air_max = air[0].max(air[1]).max(air[2]).max(1e-4);
        let transmission: Vec<f32> = dark.iter().map(|v| 1.0 - 0.95 * (v / air_max).min(1.0)).collect();
        Some((air, blur(&transmission, g.gw, g.gh, 2.0)))
    } else { None };
    let apply_haze = |rgb: &mut [f32; 4], x: usize, y: usize| {
        if let Some((air, transmission)) = &haze {
            if dehaze > 0.0 {
                let t = (1.0 - (1.0 - g.at(transmission, x, y)) * dehaze).max(0.1);
                for c in 0..3 { rgb[c] = ((rgb[c] - air[c]) / t + air[c]).max(0.0); }
            } else {
                let k = -dehaze * 0.6;
                for c in 0..3 { rgb[c] = rgb[c] * (1.0 - k) + air[c] * k; }
            }
        }
    };
    let mut tone_grid: Option<Vec<f32>> = None;
    let mut clarity_grid: Option<Vec<f32>> = None;
    if tones_active || clarity != 0.0 {
        // The wide blurs only need the picture's broad light.
        let mut small = vec![0.0f32; cells];
        for gy in 0..g.gh { for gx in 0..g.gw {
            let mut sum = 0.0;
            for j in 0..3 { for i in 0..3 {
                let (px, py) = g.probe(gx, gy, i, j, width, height);
                let mut rgb = sampler.sample(px, py);
                apply_haze(&mut rgb, px, py);
                sum += fast::log2(luma(rgb[0], rgb[1], rgb[2]) + 1e-4);
            } }
            small[gy * g.gw + gx] = sum / 9.0;
        } }
        // Stored as a linear factor, so the loop multiplies.
        if tones_active { tone_grid = Some(blur(&small, g.gw, g.gh, 5.0).iter().map(|v| 2f32.powf(v * 0.7)).collect()); }
        if clarity != 0.0 { clarity_grid = Some(blur(&small, g.gw, g.gh, 2.0)); }
    }
    let (mut texture_blur, mut fine_blur, mut sharp_blur, mut chroma) = (None, None, None, None);
    if texture != 0.0 || nr_luminance != 0.0 || sharpen != 0.0 || nr_colour != 0.0 {
        let mut lum = vec![0.0f32; pixels];
        let (mut cr, mut cb) = if nr_colour != 0.0 { (vec![0.0f32; pixels], vec![0.0f32; pixels]) } else { (Vec::new(), Vec::new()) };
        for y in 0..height { for x in 0..width {
            let mut rgb = sampler.sample(x, y);
            apply_haze(&mut rgb, x, y);
            let i = y * width + x;
            let l = luma(rgb[0], rgb[1], rgb[2]);
            lum[i] = fast::log2(l + 1e-4);
            if nr_colour != 0.0 { cr[i] = rgb[0] / (l + 1e-4); cb[i] = rgb[2] / (l + 1e-4); }
        } }
        if texture != 0.0 { texture_blur = Some(blur(&lum, width, height, (long_edge * 0.004).max(1.0))); }
        if nr_luminance != 0.0 { fine_blur = Some(blur(&lum, width, height, (1.5 * detail_scale).max(1.0))); }
        if sharpen != 0.0 { sharp_blur = Some(blur(&lum, width, height, (s.sharpen_radius * detail_scale).max(1.0))); }
        if nr_colour != 0.0 {
            let radius = (4.0 * detail_scale * (0.5 + nr_colour)).max(1.0);
            chroma = Some((blur(&cr, width, height, radius), blur(&cb, width, height, radius)));
        }
    }

    // ------------------------------------------------- display-side looks
    let curve = s.curve;
    let curve_active = curve.iter().any(|v| *v != 0.0) || s.profile == Profile::Vivid || s.profile == Profile::Neutral;
    let curve_lut: Option<Vec<f32>> = if curve_active {
        let bend = match s.profile { Profile::Vivid => 0.08, Profile::Neutral => -0.06, _ => 0.0 };
        Some((0..=4096).map(|i| {
            let v = i as f32 / 4096.0;
            let bump = |centre: f32| (1.0 - (v - centre).abs() / 0.25).max(0.0);
            let mut out = v + (curve[3] * bump(0.125) + curve[2] * bump(0.375) + curve[1] * bump(0.625) + curve[0] * bump(0.875)) / 100.0 * 0.25;
            out += bend * ((v - 0.5) * std::f32::consts::PI).sin() * 0.5 * (1.0 - (2.0 * v - 1.0).abs()) * 2.0;
            out.clamp(0.0, 1.0)
        }).collect())
    } else { None };
    let mixer = s.mixer;
    let mono = s.profile == Profile::Monochrome;
    let mixer_active = mixer.iter().any(|b| b.hue != 0.0 || b.saturation != 0.0 || b.luminance != 0.0);
    let vibrance = s.vibrance / 100.0;
    let saturation = s.saturation / 100.0 + match s.profile { Profile::Vivid => 0.15, Profile::Neutral => -0.08, _ => 0.0 };
    let zones = s.zones;
    let grading_active = zones.iter().any(|z| z.saturation != 0.0 || z.luminance != 0.0);
    // Grading hues are OKLCh hues: a zone's colour is pushed along its own hue.
    let tint: Vec<[f32; 2]> = zones.iter().map(|z| { let h = z.hue.to_radians(); [h.cos() * z.saturation / 100.0 * 0.09, h.sin() * z.saturation / 100.0 * 0.09] }).collect();
    let lift: Vec<f32> = zones.iter().map(|z| z.luminance / 100.0 * 0.12).collect();
    let exponent = 1.0 + (100.0 - s.blending) / 100.0 * 3.0;
    const ZONE_STEPS: usize = 1024;
    let shadow_weight: Vec<f32> = (0..=ZONE_STEPS).map(|i| (1.0 - i as f32 / ZONE_STEPS as f32).powf(exponent)).collect();
    let highlight_weight: Vec<f32> = (0..=ZONE_STEPS).map(|i| (i as f32 / ZONE_STEPS as f32).powf(exponent)).collect();
    let colour_active = mono || mixer_active || vibrance != 0.0 || saturation != 0.0 || grading_active;
    let needs_hue = mixer_active || mono || vibrance != 0.0;
    let vignette_amount = s.vignette_amount / 100.0;
    let vignette_mid = 0.25 + s.vignette_midpoint / 100.0 * 0.9;
    let vignette_feather = 0.05 + s.vignette_feather / 100.0 * 0.75;
    let grain_amount = s.grain_amount / 100.0 * 0.12;
    let grain_cell = ((0.6 + s.grain_size / 100.0 * 3.0) * long_edge / 2000.0).max(0.5);
    let needs_base = fine_blur.is_some() || texture_blur.is_some() || clarity_grid.is_some() || sharp_blur.is_some();
    let to_histogram = 255.0 / max;
    let (mut clipped_high, mut clipped_low) = (0u32, 0u32);
    let mut luminance = if options.measure { Some(vec![0.0f32; pixels]) } else { None };
    let mut weights = [0.0f32; 8];

    for y in 0..height { for x in 0..width {
        let i = y * width + x;
        let mut rgb = sampler.sample(x, y);
        apply_haze(&mut rgb, x, y);
        let mut l = luma(rgb[0], rgb[1], rgb[2]);
        if let Some(lm) = luminance.as_mut() { lm[i] = l; }
        let tone;
        if local {
            // Local luminance edits act on log luminance and scale the colour with it.
            let base = if needs_base { fast::log2(l + 1e-4) } else { 0.0 };
            let mut edited = base;
            if let Some(fb) = &fine_blur {
                let edge_weight = fast::exp(-((base - fb[i]).powi(2)) / (0.02 + nr_luminance * 0.3));
                edited += (fb[i] - base) * nr_luminance * edge_weight;
            }
            if let Some(tb) = &texture_blur { edited += (edited - tb[i]) * texture * 0.6; }
            if let Some(cg) = &clarity_grid {
                let mid = fast::exp(-((base + 2.5).powi(2)) / 4.0);
                edited += (edited - g.at(cg, x, y)) * clarity * 0.5 * mid;
            }
            if let Some(sb) = &sharp_blur {
                let detail = base - sb[i];
                let mask = if s.sharpen_masking != 0.0 { smooth(0.0, s.sharpen_masking / 100.0 * 0.15, detail.abs()) } else { 1.0 };
                edited += detail * sharpen * mask;
            }
            if edited != base {
                let gain = fast::exp2(edited - base);
                rgb[0] *= gain; rgb[1] *= gain; rgb[2] *= gain; l *= gain;
            }
            if let Some((cr, cb)) = &chroma {
                let (r1, b1) = (cr[i] * l, cb[i] * l);
                rgb[0] += (r1 - rgb[0]) * nr_colour; rgb[2] += (b1 - rgb[2]) * nr_colour;
                rgb[1] = ((l - 0.2126 * rgb[0] - 0.0722 * rgb[2]) / 0.7152).max(0.0);
            }
            let tone_luminance = match &tone_grid { Some(tg) => g.at(tg, x, y) * lookup(&t.soft_power, (l.clamp(0.0, 8.0) / 8.0).sqrt() * TABLE as f32), None => l };
            tone = tones[fast::round(tone_luminance * 2048.0).min(16384)];
        } else { tone = tones[fast::round(l * 2048.0).min(16384)]; }
        let mut v = [((rgb[0] * tone + black) * white).max(0.0), ((rgb[1] * tone + black) * white).max(0.0), ((rgb[2] * tone + black) * white).max(0.0)];
        let peak = v[0].max(v[1]).max(v[2]);
        let mut hi = peak > clip_at;
        let mut lo = v[0] <= 0.0 && v[1] <= 0.0 && v[2] <= 0.0;
        if hi {
            // Past white a colour keeps its hue and runs to white, in OKLab.
            let k = ((peak - clip_at) / peak).clamp(0.0, 1.0);
            let lab = colour::to_oklab(v[0] / peak, v[1] / peak, v[2] / peak);
            let c = colour::fit_gamut(lab[0] + (1.0 - lab[0]) * k, lab[1] * (1.0 - k), lab[2] * (1.0 - k));
            v = [c[0] * clip_at, c[1] * clip_at, c[2] * clip_at];
        }
        let mut out = [0.0f32; 3];
        for c in 0..3 { out[c] = display[fast::round(v[c] * display_scale).min(65535)]; }
        if let Some(lut) = &curve_lut { for c in 0..3 { let p = out[c] * 4096.0; let k = (p as usize).min(4095); out[c] = lut[k] + (lut[k + 1] - lut[k]) * (p - k as f32); } }
        if colour_active {
            let lab = colour::to_oklab(lookup(&t.decode, out[0].clamp(0.0, 1.0) * TABLE as f32), lookup(&t.decode, out[1].clamp(0.0, 1.0) * TABLE as f32), lookup(&t.decode, out[2].clamp(0.0, 1.0) * TABLE as f32));
            let (mut ll, mut a, mut b) = (lab[0], lab[1], lab[2]);
            let chroma_now = (a * a + b * b).sqrt();
            let hue = if needs_hue { colour::hue_of(a, b) } else { 0.0 };
            // Colour edits reach a colour in proportion to how coloured it is.
            let colourful = smooth(0.0, 0.06, chroma_now);
            let (mut hue_shift, mut sat_shift, mut lum_shift) = (0.0f32, 0.0f32, 0.0f32);
            if mixer_active || mono {
                bands_of(hue, &t.band_hues, &mut weights);
                for k in 0..8 {
                    if weights[k] == 0.0 { continue; }
                    hue_shift += weights[k] * mixer[k].hue / 100.0 * t.band_gap[k];
                    sat_shift += weights[k] * mixer[k].saturation / 100.0;
                    lum_shift += weights[k] * mixer[k].luminance / 100.0;
                }
            }
            if mono {
                // The B&W mix lightens or darkens each colour's grey.
                ll *= fast::exp2(lum_shift * colourful * 0.8);
                a = 0.0; b = 0.0;
            } else {
                let mut factor = (1.0 + saturation).max(0.0) * (1.0 + sat_shift).max(0.0);
                if vibrance != 0.0 {
                    // Vibrance lifts quiet colours more than vivid ones and leaves skin, at OKLCh hue 40-80, mostly be.
                    let skin = smooth(30.0, 45.0, hue) * (1.0 - smooth(75.0, 90.0, hue)) * smooth(0.02, 0.05, chroma_now) * (1.0 - smooth(0.18, 0.26, chroma_now));
                    factor *= if vibrance > 0.0 { 1.0 + vibrance * (1.0 - smooth(0.0, 0.25, chroma_now)) * (1.0 - 0.8 * skin) } else { 1.0 + vibrance };
                }
                if lum_shift != 0.0 { ll *= fast::exp2(lum_shift * colourful * 0.6); }
                if hue_shift != 0.0 {
                    let turn = (hue_shift * colourful).to_radians();
                    let (sin, cos) = turn.sin_cos();
                    let turned = a * cos - b * sin; b = a * sin + b * cos; a = turned;
                }
                a *= factor; b *= factor;
            }
            if grading_active {
                let k = fast::round((ll - s.balance / 250.0).clamp(0.0, 1.0) * ZONE_STEPS as f32);
                let (ws, wh) = (shadow_weight[k], highlight_weight[k]);
                let wm = (1.0 - ws - wh).max(0.0);
                a += tint[0][0] * ws + tint[1][0] * wm + tint[2][0] * wh + tint[3][0];
                b += tint[0][1] * ws + tint[1][1] * wm + tint[2][1] * wh + tint[3][1];
                ll += lift[0] * ws + lift[1] * wm + lift[2] * wh + lift[3];
            }
            let linear = colour::fit_gamut(ll, a, b);
            for c in 0..3 { out[c] = lookup(&t.encode, linear[c].clamp(0.0, 1.0).sqrt() * TABLE as f32); }
        }
        if vignette_amount != 0.0 {
            let dx = (x as f32 + 0.5) / width as f32 * 2.0 - 1.0; let dy = (y as f32 + 0.5) / height as f32 * 2.0 - 1.0;
            let f = smooth(vignette_mid - vignette_feather, vignette_mid + vignette_feather, (dx * dx + dy * dy).sqrt() / std::f32::consts::SQRT_2 * 1.4);
            for c in 0..3 { out[c] = if vignette_amount < 0.0 { out[c] * (1.0 + vignette_amount * f) } else { out[c] + (1.0 - out[c]) * vignette_amount * f }; }
        }
        if grain_amount != 0.0 {
            let n = grain_at(x, y, grain_cell) * grain_amount;
            let lm = luma(out[0], out[1], out[2]);
            let k = n * (0.35 + 2.6 * lm * (1.0 - lm));
            for c in 0..3 { out[c] = (out[c] + k).clamp(0.0, 1.0); }
        }
        let q = [fast::round(out[0] * max) as f32, fast::round(out[1] * max) as f32, fast::round(out[2] * max) as f32];
        histogram_rgb[fast::round(q[0] * to_histogram).min(255)] += 1;
        histogram_rgb[256 + fast::round(q[1] * to_histogram).min(255)] += 1;
        histogram_rgb[512 + fast::round(q[2] * to_histogram).min(255)] += 1;
        histogram[fast::round((0.2126 * q[0] + 0.7152 * q[1] + 0.0722 * q[2]) * to_histogram).min(255)] += 1;
        if hi || lo {
            match &untouched {
                None => { hi = false; lo = false; }
                Some(u) => {
                    let o = u.sample(x, y);
                    if hi && o[0].max(o[1]).max(o[2]) >= clip_at0 { hi = false; }
                    if lo && o[0] <= 0.0 && o[1] <= 0.0 && o[2] <= 0.0 { lo = false; }
                }
            }
        }
        if hi { clipped_high += 1; }
        if lo { clipped_low += 1; }
        let mut px = q;
        if options.clipping && hi { px = [max, 0.0, 0.0]; } else if options.clipping && lo { px = [0.0, 0.0, max]; }
        let alpha = fast::round(rgb[3] * max) as f32;
        let d = i * 4;
        if bit_depth == 8 { out8[d] = px[0] as u8; out8[d + 1] = px[1] as u8; out8[d + 2] = px[2] as u8; out8[d + 3] = alpha as u8; }
        else { out16[d] = px[0] as u16; out16[d + 1] = px[1] as u16; out16[d + 2] = px[2] as u16; out16[d + 3] = alpha as u16; }
    } }
    Developed { width, height, bit_depth, data: if bit_depth == 8 { Pixels::U8(out8) } else { Pixels::U16(out16) },
        histogram, histogram_rgb, clipped_high, clipped_low, luminance }
}

/// The two neighbouring bands of an OKLCh hue and how much of each it takes.
#[inline(always)]
fn bands_of(hue: f32, band_hues: &[f32; 8], weights: &mut [f32; 8]) {
    *weights = [0.0; 8];
    let h = if hue < band_hues[0] { hue + 360.0 } else { hue };
    for i in 0..8 {
        let from = band_hues[i];
        let to = if i + 1 < 8 { band_hues[i + 1] } else { band_hues[0] + 360.0 };
        if h >= from && h < to { let t = (h - from) / (to - from); weights[i] = 1.0 - t; weights[(i + 1) % 8] = t; return; }
    }
}

/// Auto tone: exposure, contrast and the four tone sliders from the picture's
/// own luminance, with its white balance and look left alone.
pub fn auto_tone(frame: &Frame, recipe: &Recipe) -> [f32; 6] {
    let mut r = recipe.clone();
    r.exposure = 0.0; r.contrast = 0.0; r.highlights = 0.0; r.shadows = 0.0; r.whites = 0.0; r.blacks = 0.0;
    let sampled = develop(frame, &r, &Options { edge: 320, measure: true, ..Options::default() });
    let mut values = sampled.luminance.unwrap_or_default();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let at = |p: f32| { let v = values[((p * values.len() as f32) as usize).min(values.len() - 1)]; if v == 0.0 { 1e-4 } else { v } };
    let median = at(0.5).max(1e-4);
    // Mid-grey for the middle of the picture, but never so bright that the
    // highlights go past what Highlights can bring back.
    let exposure = (0.14 / median).log2().min((2.0 / at(0.99).max(1e-4)).log2()).clamp(-2.0, 2.0);
    let lift = 2f32.powf(exposure);
    let (top, floor, deep) = (at(0.995) * lift, at(0.02) * lift, at(0.005) * lift);
    let highlights = if top > 1.0 { -(top.log2() * 45.0).clamp(0.0, 80.0) } else { 0.0 };
    let shadows = if floor < 0.03 { ((0.03 / floor.max(1e-4)).log2() * 14.0).clamp(0.0, 55.0) } else { 0.0 };
    let whites = if top < 0.8 { ((0.95 / top.max(1e-4)).log2() * 60.0).clamp(0.0, 40.0) } else { 0.0 };
    let blacks = -(deep * 1000.0).clamp(0.0, 25.0);
    let range = (at(0.95).max(1e-4) / at(0.05).max(1e-4)).log2();
    let contrast = ((6.0 - range) * 6.0).clamp(-15.0, 25.0);
    [(exposure * 20.0).round() / 20.0, contrast.round(), highlights.round(), shadows.round(), whites.round(), blacks.round()]
}
