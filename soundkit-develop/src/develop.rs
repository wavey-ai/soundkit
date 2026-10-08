//! Development: a linear frame and a recipe in, display pixels out.
//!
//! A recipe is first resolved into a [`Plan`]: every matrix, table and
//! low-resolution plane the picture needs. The plan is then run once per
//! pixel by [`shade`]. The browser's GPU path runs the same plan with a
//! shader that mirrors `shade`, so the two produce the same picture.

use crate::colour::{self, Matrix, IDENTITY};
use crate::curve;
use crate::fast;
use crate::filter;
use crate::recipe::{Profile, Recipe, WhiteBalance};
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

/// The samples a frame holds: a RAW stays 16-bit sensor data, a photograph
/// stays 8-bit or 16-bit encoded and is made linear as it is read, and a
/// working proxy is linear float. An 8-bit photograph takes a quarter of the
/// memory of float, and a 16-bit photograph takes half.
pub enum Samples { F32(Vec<f32>), U16(Vec<u16>), Srgb8(Vec<u8>), Srgb16(Vec<Encoded16>) }

/// A 16-bit sample of an encoded photograph. `u16` alone is linear sensor data.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct Encoded16(pub u16);

/// The transfer curve of an encoded frame, as a table from each stored value
/// to linear light. A frame without a curve is sRGB.
pub enum Curve { Eight(Box<[f32; 256]>), Sixteen(Box<[f32; 65536]>) }

/// 8-bit sRGB to linear light.
pub(crate) fn srgb8() -> &'static [f32; 256] {
    static T: OnceLock<[f32; 256]> = OnceLock::new();
    T.get_or_init(|| std::array::from_fn(|i| to_linear(i as f32 / 255.0)))
}
/// 16-bit sRGB to linear light.
pub(crate) fn srgb16() -> &'static [f32; 65536] {
    static T: OnceLock<Box<[f32; 65536]>> = OnceLock::new();
    T.get_or_init(|| curve_table(to_linear))
}
/// A table of `N` entries over the encoded range of zero to one.
pub(crate) fn curve_table<const N: usize>(decode: impl Fn(f32) -> f32) -> Box<[f32; N]> {
    let values: Vec<f32> = (0..N).map(|i| decode(i as f32 / (N - 1) as f32)).collect();
    values.into_boxed_slice().try_into().unwrap()
}

/// A stored sample, read as linear light through the table of its frame.
pub trait Texel: Copy {
    type Table: ?Sized;
    fn table(frame: &Frame) -> &Self::Table;
    fn value(self, table: &Self::Table) -> f32;
}
impl Texel for f32 {
    type Table = ();
    fn table(_: &Frame) -> &() { &() }
    #[inline(always)] fn value(self, _: &()) -> f32 { self }
}
impl Texel for u16 {
    type Table = ();
    fn table(_: &Frame) -> &() { &() }
    #[inline(always)] fn value(self, _: &()) -> f32 { self as f32 }
}
impl Texel for u8 {
    type Table = [f32; 256];
    fn table(frame: &Frame) -> &[f32; 256] { match &frame.curve { Some(Curve::Eight(table)) => table, _ => srgb8() } }
    #[inline(always)] fn value(self, table: &[f32; 256]) -> f32 { table[self as usize] }
}
impl Texel for Encoded16 {
    type Table = [f32; 65536];
    fn table(frame: &Frame) -> &[f32; 65536] { match &frame.curve { Some(Curve::Sixteen(table)) => table, _ => srgb16() } }
    #[inline(always)] fn value(self, table: &[f32; 65536]) -> f32 { table[self.0 as usize] }
}

/// The camera a frame came from. A proxy has its camera's white balance
/// baked in and keeps these facts to change it later. `matrix` takes white
/// balanced camera RGB to linear sRGB.
#[derive(Clone, Copy, Debug)]
pub struct Camera { pub matrix: Matrix, pub wb: [f64; 3], pub daylight: Option<[f64; 3]> }

pub struct Frame {
    pub width: usize,
    pub height: usize,
    pub data: Samples,
    pub alpha: Option<Vec<u8>>,
    pub scale: f32,
    pub wb: [f32; 3],
    /// Takes the white-balanced samples to linear sRGB, or to the working
    /// space when `working` is set.
    pub matrix: Matrix,
    pub camera: Camera,
    /// The full-resolution width this frame stands for: radii given in
    /// source pixels shrink with a preview.
    pub source_width: usize,
    /// The transfer curve of an encoded photograph that is not sRGB.
    pub curve: Option<Curve>,
    /// The samples are already in the working space: a proxy.
    pub working: bool,
    /// Scene light from a camera sensor, which the tone map brings to the
    /// display, rather than a photograph already made for one.
    pub scene: bool,
    /// Identifies the frame's pixels, for the planes cached between edits.
    pub id: u64,
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);
pub fn next_id() -> u64 { NEXT_ID.fetch_add(1, Ordering::Relaxed) }

pub(crate) fn to_linear(v: f32) -> f32 { if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) } }
pub(crate) fn to_srgb(v: f32) -> f32 { if v <= 0.0031308 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 } }
#[inline(always)]
pub(crate) fn smooth(a: f32, b: f32, x: f32) -> f32 { let d = b - a; let t = ((x - a) / if d == 0.0 { 1e-6 } else { d }).clamp(0.0, 1.0); t * t * (3.0 - 2.0 * t) }
/// Luminance of display sRGB, for the histogram and the grain.
#[inline(always)]
fn luma(r: f32, g: f32, b: f32) -> f32 { 0.2126 * r + 0.7152 * g + 0.0722 * b }
/// Middle grey: log luminance is measured in stops from it.
pub const GREY: f32 = 0.18;

pub const TABLE: usize = 4096;
pub struct Tables { pub decode: Vec<f32>, pub encode: Vec<f32>, pub band_hues: [f32; 8], pub band_gap: [f32; 8] }
pub fn tables() -> &'static Tables {
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
            decode: (0..=TABLE).map(|i| to_linear(i as f32 / TABLE as f32)).collect(),
            // Encoding is indexed by the square root of linear light, so the
            // shadows keep their precision.
            encode: (0..=TABLE).map(|i| to_srgb((i as f32 / TABLE as f32).powi(2))).collect(),
            band_hues, band_gap,
        }
    })
}
#[inline(always)]
fn lookup(table: &[f32], p: f32) -> f32 { let k = (p as usize).min(TABLE - 1); table[k] + (table[k + 1] - table[k]) * (p - k as f32) }
#[inline(always)]
fn decode(t: &Tables, v: f32) -> f32 { lookup(&t.decode, v.clamp(0.0, 1.0) * TABLE as f32) }
#[inline(always)]
fn encode(t: &Tables, v: f32) -> f32 { lookup(&t.encode, v.clamp(0.0, 1.0).sqrt() * TABLE as f32) }

/// A frame from 8-bit sRGB RGBA, kept as 8-bit. Opacity is kept only when
/// some pixel is not opaque.
pub fn from_rgba(width: usize, height: usize, rgba: &[u8]) -> Frame {
    let pixels = width * height;
    let mut data = Vec::with_capacity(pixels * 3);
    let mut opaque = true;
    for i in 0..pixels {
        data.extend_from_slice(&rgba[i * 4..i * 4 + 3]);
        opaque &= rgba[i * 4 + 3] == 255;
    }
    let alpha = if opaque { None } else { Some((0..pixels).map(|i| rgba[i * 4 + 3]).collect()) };
    encoded(width, height, Samples::Srgb8(data), alpha)
}
/// A frame from encoded RGBA of more than 8 bits, kept as 16-bit. The values
/// run from zero to the largest value of `bit_depth`. Opacity is kept at 8
/// bits, and only when some pixel is not opaque.
pub fn from_rgba16(width: usize, height: usize, rgba: &[u16], bit_depth: u8) -> Frame {
    let pixels = width * height;
    let max = (1u32 << bit_depth.clamp(9, 16)) - 1;
    let widen = |v: u16| if max == 65535 { v } else { (((v as u32).min(max) * 65535 + max / 2) / max) as u16 };
    let mut data = Vec::with_capacity(pixels * 3);
    let mut opaque = true;
    for i in 0..pixels {
        data.extend(rgba[i * 4..i * 4 + 3].iter().map(|v| Encoded16(widen(*v))));
        opaque &= rgba[i * 4 + 3] as u32 >= max;
    }
    let alpha = if opaque { None } else { Some((0..pixels).map(|i| (((rgba[i * 4 + 3] as u32).min(max) * 255 + max / 2) / max) as u8).collect()) };
    encoded(width, height, Samples::Srgb16(data), alpha)
}
/// A frame of encoded sRGB samples.
pub(crate) fn encoded(width: usize, height: usize, data: Samples, alpha: Option<Vec<u8>>) -> Frame {
    Frame { width, height, data, alpha, scale: 1.0, wb: [1.0; 3], matrix: IDENTITY,
        camera: Camera { matrix: IDENTITY, wb: [1.0; 3], daylight: None }, source_width: width, curve: None,
        working: false, scene: false, id: next_id() }
}
/// A RAW frame: 16-bit camera samples that reach one at `scale`, the as-shot
/// white balance and the camera matrix. Clipped highlights are rebuilt.
pub fn from_sensor(width: usize, height: usize, mut data: Vec<u16>, scale: f32, camera: Camera) -> Frame {
    let wb = camera.wb.map(|v| v as f32);
    let (_, scale) = crate::highlight::reconstruct(&mut data, width, height, scale, wb);
    Frame { width, height, data: Samples::U16(data), alpha: None, scale, wb, matrix: camera.matrix, camera,
        source_width: width, curve: None, working: false, scene: true, id: next_id() }
}

impl Frame {
    pub fn has_daylight_reference(&self) -> bool { self.camera.daylight.is_some() }

    /// The number of bits in each stored sample.
    pub fn bit_depth(&self) -> u8 { match self.data { Samples::Srgb8(_) => 8, Samples::U16(_) | Samples::Srgb16(_) => 16, Samples::F32(_) => 32 } }

    /// Takes the white-balanced samples to the working space.
    pub fn to_working(&self) -> Matrix { if self.working { self.matrix } else { colour::multiply(&colour::SRGB_TO_WORK, &self.matrix) } }

    /// Reads the frame's colours through an ICC profile. The frame must hold
    /// encoded samples. An sRGB profile leaves the frame as it is. Returns
    /// false, and changes nothing, for a profile that is not an RGB matrix
    /// profile with one curve.
    pub fn set_profile(&mut self, icc: &[u8]) -> bool {
        let Some(profile) = crate::icc::parse(icc) else { return false };
        if profile.is_srgb() { return true; }
        self.curve = match self.data {
            Samples::Srgb8(_) => Some(Curve::Eight(curve_table(|v| profile.to_linear(v)))),
            Samples::Srgb16(_) => Some(Curve::Sixteen(curve_table(|v| profile.to_linear(v)))),
            _ => return false,
        };
        self.matrix = profile.matrix;
        self.id = next_id();
        true
    }

    /// Box-filters the frame once into a small working image every edit reuses.
    pub fn linear_preview(&self, edge: usize) -> Frame {
        match &self.data {
            Samples::F32(data) => self.preview_on(data.as_slice(), edge),
            Samples::U16(data) => self.preview_on(data.as_slice(), edge),
            Samples::Srgb8(data) => self.preview_on(data.as_slice(), edge),
            Samples::Srgb16(data) => self.preview_on(data.as_slice(), edge),
        }
    }
    fn preview_on<T: Texel>(&self, source: &[T], edge: usize) -> Frame {
        let table = T::table(self);
        let factor = (edge as f64 / self.width.max(self.height) as f64).min(1.0);
        let width = ((self.width as f64 * factor).round() as usize).max(1);
        let height = ((self.height as f64 * factor).round() as usize).max(1);
        let mut data = vec![0.0f32; width * height * 3];
        let mut alpha = self.alpha.as_ref().map(|_| vec![0u8; width * height]);
        let m = self.to_working();
        let (sr, sg, sb) = (self.scale * self.wb[0], self.scale * self.wb[1], self.scale * self.wb[2]);
        for y in 0..height { for x in 0..width {
            let left = x * self.width / width; let right = ((x + 1) * self.width / width).max(left + 1);
            let top = y * self.height / height; let bottom = ((y + 1) * self.height / height).max(top + 1);
            let (mut acc, mut count, mut opacity) = ([0.0f64; 3], 0u32, 0u32);
            for sy in top..bottom { for sx in left..right {
                let p = sy * self.width + sx;
                let (r, g, b) = (source[p * 3].value(table) * sr, source[p * 3 + 1].value(table) * sg, source[p * 3 + 2].value(table) * sb);
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
            camera: self.camera, source_width: self.source_width, curve: None, working: true, scene: self.scene, id: next_id() }
    }

    /// The linear-sRGB transform that changes the as-shot white balance to a RAW preset.
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

/// Reads a pixel of the output as working-space linear light: stored
/// sample, transfer table, `matrix` (which holds the scale, the white balance,
/// the camera and any white-balance change), then `gain`. Generic over the
/// stored sample, so the read compiles into the loop.
pub(crate) struct Sampler<'a, T: Texel> {
    data: &'a [T], alpha: Option<&'a [u8]>, fw: usize, fh: usize, table: &'a T::Table,
    m: [f32; 9], gain: f32, xs: f32, ys: f32, exact: bool,
}
impl<'a, T: Texel> Sampler<'a, T> {
    pub(crate) fn new(frame: &'a Frame, data: &'a [T], width: usize, height: usize, matrix: [f32; 9], gain: f32) -> Self {
        Sampler { data, alpha: frame.alpha.as_deref(), fw: frame.width, fh: frame.height, table: T::table(frame), m: matrix, gain,
            xs: frame.width as f32 / width as f32, ys: frame.height as f32 / height as f32,
            // Output and frame the same size: one sample per pixel, no blend.
            exact: frame.width == width && frame.height == height }
    }
    #[inline(always)]
    pub(crate) fn sample(&self, x: usize, y: usize) -> [f32; 4] {
        let (fw, fh, d, t) = (self.fw, self.fh, self.data, self.table);
        let (cr, cg, cb, alpha);
        if self.exact {
            let p = y * fw + x;
            cr = d[p * 3].value(t); cg = d[p * 3 + 1].value(t); cb = d[p * 3 + 2].value(t);
            alpha = match self.alpha { Some(a) => a[p] as f32 / 255.0, None => 1.0 };
        } else {
            let sx = ((x as f32 + 0.5) * self.xs - 0.5).clamp(0.0, (fw - 1) as f32);
            let sy = ((y as f32 + 0.5) * self.ys - 0.5).clamp(0.0, (fh - 1) as f32);
            let (x0, y0) = (sx as usize, sy as usize);
            let (x1, y1) = ((x0 + 1).min(fw - 1), (y0 + 1).min(fh - 1));
            let (dx, dy) = (sx - x0 as f32, sy - y0 as f32);
            let (p00, p10, p01, p11) = ((y0 * fw + x0) * 3, (y0 * fw + x1) * 3, (y1 * fw + x0) * 3, (y1 * fw + x1) * 3);
            let (w00, w10, w01, w11) = ((1.0 - dx) * (1.0 - dy), dx * (1.0 - dy), (1.0 - dx) * dy, dx * dy);
            cr = d[p00].value(t) * w00 + d[p10].value(t) * w10 + d[p01].value(t) * w01 + d[p11].value(t) * w11;
            cg = d[p00 + 1].value(t) * w00 + d[p10 + 1].value(t) * w10 + d[p01 + 1].value(t) * w01 + d[p11 + 1].value(t) * w11;
            cb = d[p00 + 2].value(t) * w00 + d[p10 + 2].value(t) * w10 + d[p01 + 2].value(t) * w01 + d[p11 + 2].value(t) * w11;
            alpha = match self.alpha {
                Some(a) => (a[p00 / 3] as f32 * w00 + a[p10 / 3] as f32 * w10 + a[p01 / 3] as f32 * w01 + a[p11 / 3] as f32 * w11) / 255.0,
                None => 1.0,
            };
        }
        let m = &self.m;
        let k = self.gain;
        [((m[0] * cr + m[1] * cg + m[2] * cb) * k).max(0.0), ((m[3] * cr + m[4] * cg + m[5] * cb) * k).max(0.0), ((m[6] * cr + m[7] * cg + m[8] * cb) * k).max(0.0), alpha]
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

/// `region` is x, y, width and height within the full output: only that
/// part is developed, as for a 100% view.
#[derive(Clone)]
pub struct Options { pub edge: usize, pub bit_depth: u8, pub before: bool, pub clipping: bool, pub measure: bool, pub region: Option<[usize; 4]> }
impl Default for Options { fn default() -> Self { Options { edge: 0, bit_depth: 8, before: false, clipping: false, measure: false, region: None } } }

pub enum Pixels { U8(Vec<u8>), U16(Vec<u16>) }
pub struct Developed {
    pub width: usize, pub height: usize, pub bit_depth: u8, pub data: Pixels,
    pub histogram: Vec<u32>, pub histogram_rgb: Vec<u32>, pub clipped_high: u32, pub clipped_low: u32,
    pub luminance: Option<Vec<f32>>,
}

// ------------------------------------------------------------------ the plan

/// A low-resolution grid over the output: cell (i, j) covers `step` pixels
/// square. Values are read bilinearly at pixel centres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grid { pub gw: usize, pub gh: usize, pub step: f32 }
impl Grid {
    pub fn over(width: usize, height: usize, step: usize) -> Grid {
        let step = step.max(1);
        Grid { gw: width.div_ceil(step), gh: height.div_ceil(step), step: step as f32 }
    }
    /// A grid of about `cells` cells on the long side.
    pub fn of_cells(width: usize, height: usize, cells: usize) -> Grid {
        Grid::over(width, height, (width.max(height) as f32 / cells as f32).ceil().max(1.0) as usize)
    }
    #[inline(always)]
    pub fn at(&self, values: &[f32], x: usize, y: usize) -> f32 {
        let gx = ((x as f32 + 0.5) / self.step - 0.5).clamp(0.0, (self.gw - 1) as f32);
        let gy = ((y as f32 + 0.5) / self.step - 0.5).clamp(0.0, (self.gh - 1) as f32);
        let (x0, y0) = (gx as usize, gy as usize);
        let (x1, y1) = ((x0 + 1).min(self.gw - 1), (y0 + 1).min(self.gh - 1));
        let (dx, dy) = (gx - x0 as f32, gy - y0 as f32);
        let g = self.gw;
        (values[y0 * g + x0] * (1.0 - dx) + values[y0 * g + x1] * dx) * (1.0 - dy) + (values[y1 * g + x0] * (1.0 - dx) + values[y1 * g + x1] * dx) * dy
    }
    /// The output pixel of probe (i, j) of `n × n` in cell (gx, gy).
    fn probe(&self, gx: usize, gy: usize, i: usize, j: usize, n: usize, width: usize, height: usize) -> (usize, usize) {
        let s = self.step as f64;
        ((((gx as f64 + (i as f64 + 0.5) / n as f64) * s) as usize).min(width - 1),
            (((gy as f64 + (j as f64 + 0.5) / n as f64) * s) as usize).min(height - 1))
    }
}

/// Haze: the atmosphere's colour and how much of each grid cell it veils.
#[derive(Clone, Debug)]
pub struct Haze { pub air: [f32; 3], pub amount: f32, pub grid: Grid, pub transmission: Vec<f32> }

/// A guided filter of log luminance, solved on a grid: the filtered value at
/// a pixel is `a * L + b`, with `a` and `b` read from the grid.
#[derive(Clone, Debug)]
pub struct GuidedGrid { pub grid: Grid, pub a: Vec<f32>, pub b: Vec<f32> }
impl GuidedGrid {
    #[inline(always)]
    pub fn at(&self, l: f32, x: usize, y: usize) -> f32 { self.grid.at(&self.a, x, y) * l + self.grid.at(&self.b, x, y) }
}

/// The colour controls, resolved.
#[derive(Clone, Debug)]
pub struct ColourPlan {
    pub mono: bool,
    pub mixer_active: bool,
    /// Per band: hue turn in degrees, saturation and luminance factors.
    pub mixer: [[f32; 3]; 8],
    pub saturation: f32,
    pub vibrance: f32,
    /// The whole picture's turn round the OKLCh circle, as cosine and sine.
    pub turn: Option<[f32; 2]>,
    pub grading: bool,
    pub tint: [[f32; 2]; 4],
    pub lift: [f32; 4],
    pub balance: f32,
    pub shadow_weight: Vec<f32>,
    pub highlight_weight: Vec<f32>,
}
pub const ZONE_STEPS: usize = 1024;

/// How the light reaches the display.
#[derive(Clone, Debug)]
pub enum Tone {
    /// Scene light from a sensor: a tone curve on luminance, indexed by stops
    /// from middle grey over `SCENE_STOPS`. The table holds display luminance.
    Scene { table: Vec<f32> },
    /// A photograph: Contrast as a curve about middle grey per channel, with
    /// Whites as a gain and Blacks as an offset. The table holds encoded
    /// values at 16-bit linear resolution up to `clip_at`.
    Display { table: Rc<Vec<f32>>, scale: f32, clip_at: f32, black: f32, white: f32 },
}
/// The range of the scene tone table, in stops from middle grey.
pub const SCENE_STOPS: [f32; 2] = [-16.0, 8.0];
pub const SCENE_TABLE: usize = 4096;

/// Everything a recipe asks of the picture, resolved once per develop.
pub struct Plan {
    pub width: usize,
    pub height: usize,
    /// The developed part: x, y, width, height.
    pub region: [usize; 4],
    /// The part the window filters cover: x0, y0, x1, y1.
    pub window: [usize; 4],
    pub bit_depth: u8,
    pub max: f32,
    pub clipping: bool,
    pub measure: bool,
    /// Stored samples to working space, without exposure.
    pub matrix: [f32; 9],
    pub exposure: f32,
    /// Stored samples to linear sRGB as shot, for the clipping report; none
    /// for a "before" view.
    pub untouched: Option<[f32; 9]>,
    pub haze: Option<Haze>,
    pub base: Option<Rc<GuidedGrid>>,
    pub clarity_base: Option<Rc<GuidedGrid>>,
    pub shadows: f32,
    pub highlights: f32,
    pub clarity: f32,
    pub texture: f32,
    pub texture_sigma: f32,
    pub sharpen: f32,
    pub sharpen_radius: usize,
    pub sharpen_masking: f32,
    /// Luminance noise reduction: sigma, epsilon and blend.
    pub noise: Option<[f32; 3]>,
    /// Colour noise reduction: blur radius and amount.
    pub chroma: Option<(usize, f32)>,
    pub tone: Tone,
    pub curves: Option<[Vec<f32>; 3]>,
    pub colour: Option<ColourPlan>,
    pub vignette: [f32; 3],
    pub grain: [f32; 2],
}

impl Plan {
    pub fn uses_window(&self) -> bool { self.texture != 0.0 || self.sharpen != 0.0 || self.noise.is_some() || self.chroma.is_some() }
}

thread_local! {
    // The display curve only changes with Contrast, so a drag of any other
    // slider reuses it.
    static DISPLAY: RefCell<Option<(u32, Rc<Vec<f32>>)>> = const { RefCell::new(None) };
    // The broad planes depend on the frame, its white balance and Dehaze,
    // not on the tone sliders: a drag of those reuses them.
    static PLANES: RefCell<Vec<(PlaneKey, Rc<GuidedGrid>)>> = const { RefCell::new(Vec::new()) };
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

#[derive(Clone, Copy, PartialEq)]
struct PlaneKey { frame: u64, width: usize, height: usize, matrix: [u32; 9], dehaze: u32, sigma: u32, eps: u32 }

/// The broad planes are measured on the picture before exposure; exposure
/// shifts log luminance by the same number of stops everywhere, which moves
/// a guided filter's output by the same amount. So `b` takes `(1 - a) * ev`
/// and the plane needs no new solve when Exposure moves.
fn shifted(plane: &GuidedGrid, ev: f32) -> GuidedGrid {
    if ev == 0.0 { return plane.clone(); }
    GuidedGrid { grid: plane.grid, a: plane.a.clone(), b: plane.a.iter().zip(&plane.b).map(|(a, b)| b + (1.0 - a) * ev).collect() }
}

/// Middle-grey-relative log luminance of a working-space colour.
#[inline(always)]
pub fn log_luminance(c: [f32; 3]) -> f32 { fast::log2(colour::work_luma(c).max(1e-6) / GREY) }

/// The scene tone curve: display luminance for a scene luminance `stops`
/// from middle grey. Contrast steepens the curve about grey; the shoulder
/// rolls the highlights smoothly into white instead of clipping them; Whites
/// brighten or darken the upper tones; Blacks deepen or lift the toe.
pub fn scene_tone(stops: f32, contrast: f32, whites: f32, blacks: f32) -> f32 {
    // A raw file is flatter than a camera's own picture: a slightly steeper
    // curve and a fifth of a stop of lift give the familiar default.
    let slope = 1.12 * 2f32.powf(contrast / 150.0);
    let mut x = GREY * 1.15 * 2f32.powf(stops * slope);
    x *= 2f32.powf(0.8 * whites * smooth(0.1, 1.2, x));
    // Shoulder: unchanged up to the knee, then an exponential approach to white.
    const KNEE: f32 = 0.55;
    let mut y = if x <= KNEE { x } else { KNEE + (1.0 - KNEE) * (1.0 - (-(x - KNEE) / (1.0 - KNEE)).exp()) };
    if blacks < 0.0 {
        // A rational toe that pulls the darkest tones down and leaves white at white.
        let t = -blacks * 0.025;
        y = y * y / (y + t) * (1.0 + t);
    } else if blacks > 0.0 {
        let t = blacks * 0.03;
        y = t + (1.0 - t) * y;
    }
    y.clamp(0.0, 1.0)
}

/// Resolves a recipe for a frame.
pub fn plan(frame: &Frame, recipe: &Recipe, options: &Options) -> Plan {
    match &frame.data {
        Samples::F32(data) => plan_on(frame, data.as_slice(), recipe, options),
        Samples::U16(data) => plan_on(frame, data.as_slice(), recipe, options),
        Samples::Srgb8(data) => plan_on(frame, data.as_slice(), recipe, options),
        Samples::Srgb16(data) => plan_on(frame, data.as_slice(), recipe, options),
    }
}

fn plan_on<T: Texel>(frame: &Frame, data: &[T], recipe: &Recipe, options: &Options) -> Plan {
    let default = Recipe::default();
    let s = if options.before { &default } else { recipe };
    let factor = if options.edge > 0 { (options.edge as f64 / frame.width.max(frame.height) as f64).min(1.0) } else { 1.0 };
    let width = ((frame.width as f64 * factor).round() as usize).max(1);
    let height = ((frame.height as f64 * factor).round() as usize).max(1);
    let [rx, ry, rw, rh] = match options.region {
        Some([x, y, w, h]) => { let x = x.min(width - 1); let y = y.min(height - 1); [x, y, w.clamp(1, width - x), h.clamp(1, height - y)] }
        None => [0, 0, width, height],
    };
    let bit_depth = if [8, 10, 12].contains(&options.bit_depth) { options.bit_depth } else { 8 };
    let detail_scale = width as f32 / frame.source_width as f32;
    let long_edge = width.max(height) as f32;

    // ---------------------------------------------------- white balance
    let to_working = frame.to_working();
    let samples = colour::multiply(&to_working, &colour::diagonal([(frame.scale * frame.wb[0]) as f64, (frame.scale * frame.wb[1]) as f64, (frame.scale * frame.wb[2]) as f64]));
    let mut balance = match s.white_balance { WhiteBalance::Shot | WhiteBalance::Auto => None, preset => frame.white_balance_transform(preset).map(|m| colour::to_work(&m)) };
    if s.white_balance == WhiteBalance::Auto {
        // The mid-tones' average colour is taken as the light and adapted to neutral.
        let probe = Sampler::new(frame, data, width, height, samples.map(|v| v as f32), 1.0);
        let step = ((long_edge / 200.0).round() as usize).max(1);
        let mut sum = [0.0f64; 3];
        for y in (0..height).step_by(step) { for x in (0..width).step_by(step) {
            let p = probe.sample(x, y);
            let l = colour::work_luma([p[0], p[1], p[2]]);
            if !(0.02..=0.9).contains(&l) { continue; }
            sum[0] += p[0] as f64; sum[1] += p[1] as f64; sum[2] += p[2] as f64;
        } }
        if sum[0] > 0.0 && sum[1] > 0.0 && sum[2] > 0.0 {
            let light = colour::apply(&colour::RGB_TO_XYZ, colour::apply(&colour::WORK_TO_SRGB, sum));
            let neutral = colour::apply(&colour::RGB_TO_XYZ, [1.0; 3]);
            balance = Some(colour::to_work(&colour::adaptation([light[0] / light[1], 1.0, light[2] / light[1]], neutral)));
        }
    }
    if let Some(shift) = colour::temperature_tint(s.temperature as f64, s.tint as f64) {
        let shift = colour::to_work(&shift);
        balance = Some(match balance { Some(b) => colour::multiply(&shift, &b), None => shift });
    }
    let matrix = match balance { Some(b) => colour::multiply(&b, &samples), None => samples };
    let matrix32 = matrix.map(|v| v as f32);
    let exposure = 2f32.powf(s.exposure);
    let untouched = (!options.before).then(|| colour::multiply(&colour::WORK_TO_SRGB, &samples).map(|v| v as f32));

    // ------------------------------------------------------------- haze
    let dehaze = s.dehaze / 100.0;
    let unexposed = Sampler::new(frame, data, width, height, matrix32, 1.0);
    let haze = (dehaze != 0.0).then(|| {
        let g = Grid::of_cells(width, height, 160);
        let cells = g.gw * g.gh;
        let mut dark = vec![0.0f32; cells];
        let mut colours = vec![0.0f32; cells * 3];
        for gy in 0..g.gh { for gx in 0..g.gw {
            let cell = gy * g.gw + gx;
            let mut lowest = f32::INFINITY;
            for j in 0..3 { for i in 0..3 {
                let (px, py) = g.probe(gx, gy, i, j, 3, width, height);
                let rgb = unexposed.sample(px, py);
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
        Haze { air, amount: dehaze, grid: g, transmission: filter::triple_box(&transmission, g.gw, g.gh, 2) }
    });

    // ------------------------------------------------------ broad planes
    let shadows = s.shadows / 100.0;
    let highlights = s.highlights / 100.0;
    let clarity = s.clarity / 100.0;
    let ev = s.exposure;
    let broad = |sigma_fraction: f32, eps: f32| -> Rc<GuidedGrid> {
        let sigma = (sigma_fraction * long_edge).max(1.0);
        let key = PlaneKey { frame: frame.id, width, height, matrix: matrix32.map(f32::to_bits), dehaze: dehaze.to_bits(), sigma: sigma.to_bits(), eps: eps.to_bits() };
        let solved = PLANES.with(|cache| cache.borrow().iter().find(|(k, _)| *k == key).map(|(_, v)| v.clone()));
        let solved = solved.unwrap_or_else(|| {
            let v = Rc::new(broad_plane(&unexposed, haze.as_ref(), width, height, sigma, eps));
            PLANES.with(|cache| { let mut c = cache.borrow_mut(); c.insert(0, (key, v.clone())); c.truncate(6); });
            v
        });
        Rc::new(shifted(&solved, ev))
    };
    let base = (shadows != 0.0 || highlights != 0.0).then(|| broad(0.02, 0.25));
    let clarity_base = (clarity != 0.0).then(|| broad(0.01, 0.6));

    // ---------------------------------------------------- window filters
    let texture = s.texture / 100.0;
    let texture_sigma = (0.0025 * long_edge).max(0.7);
    let sharpen = s.sharpen_amount / 100.0;
    let sharpen_radius = (s.sharpen_radius * detail_scale).max(1.0).round() as usize;
    let nr = s.noise_luminance / 100.0;
    let noise = (nr != 0.0).then(|| [(0.8 + 2.0 * nr) * detail_scale.max(0.35), 0.003 + 0.2 * nr * nr, nr.powf(0.75)]);
    let nr_colour = s.noise_colour / 100.0;
    let chroma = (nr_colour != 0.0).then(|| ((4.0 * detail_scale * (0.5 + nr_colour)).max(1.0).round() as usize, nr_colour));
    let mut reach = 0usize;
    if texture != 0.0 { reach = reach.max(filter::gaussian_reach(texture_sigma)); }
    if sharpen != 0.0 { reach = reach.max(3 * sharpen_radius); }
    if let Some([sigma, _, _]) = noise { reach = reach.max(filter::guided_reach(sigma)); }
    if let Some((radius, _)) = chroma { reach = reach.max(3 * radius); }
    let window = if reach > 0 {
        let pad = reach + 2;
        [rx.saturating_sub(pad), ry.saturating_sub(pad), (rx + rw + pad).min(width), (ry + rh + pad).min(height)]
    } else { [rx, ry, rx + rw, ry + rh] };

    // -------------------------------------------------------------- tone
    let tone = if frame.scene {
        let (c, w, b) = (s.contrast, s.whites / 100.0, s.blacks / 100.0);
        Tone::Scene { table: (0..=SCENE_TABLE).map(|i| {
            let stops = SCENE_STOPS[0] + (SCENE_STOPS[1] - SCENE_STOPS[0]) * i as f32 / SCENE_TABLE as f32;
            scene_tone(stops, c, w, b)
        }).collect() }
    } else {
        let contrast = 2f64.powf(s.contrast as f64 / 150.0);
        // The display curve has 16-bit linear resolution before quantising.
        let clip_at = (0.18 * (1.0 / 0.18f64).powf(1.0 / contrast)) as f32;
        let scale = 65535.0 / clip_at;
        Tone::Display { table: display_curve(contrast, scale), scale, clip_at, black: s.blacks / 1000.0, white: 2f32.powf(s.whites / 200.0) }
    };

    // ------------------------------------------------------------ curves
    let parametric = s.curve.iter().any(|v| *v != 0.0) || s.profile == Profile::Vivid || s.profile == Profile::Neutral;
    let points: Vec<Option<Vec<f32>>> = s.points.iter().map(|p| curve::table(p)).collect();
    let curves = (parametric || points.iter().any(Option::is_some)).then(|| {
        let bend = match s.profile { Profile::Vivid => 0.08, Profile::Neutral => -0.06, _ => 0.0 };
        let c = s.curve;
        std::array::from_fn(|channel| (0..=curve::CURVE_TABLE).map(|i| {
            let v = i as f32 / curve::CURVE_TABLE as f32;
            // Each region lifts or lowers its own quarter of the range.
            let bump = |centre: f32| (1.0 - (v - centre).abs() / 0.25).max(0.0);
            let mut out = v + (c[3] * bump(0.125) + c[2] * bump(0.375) + c[1] * bump(0.625) + c[0] * bump(0.875)) / 100.0 * 0.25;
            out += bend * ((v - 0.5) * std::f32::consts::PI).sin() * 0.5 * (1.0 - (2.0 * v - 1.0).abs()) * 2.0;
            out = out.clamp(0.0, 1.0);
            if let Some(master) = &points[0] { out = curve::read(master, out); }
            if let Some(own) = &points[1 + channel] { out = curve::read(own, out); }
            out
        }).collect())
    });

    // ------------------------------------------------------------ colour
    let mixer_active = s.mixer.iter().any(|b| b.hue != 0.0 || b.saturation != 0.0 || b.luminance != 0.0);
    let mono = s.profile == Profile::Monochrome;
    let vibrance = s.vibrance / 100.0;
    let saturation = s.saturation / 100.0 + match s.profile { Profile::Vivid => 0.15, Profile::Neutral => -0.08, _ => 0.0 };
    let grading = s.zones.iter().any(|z| z.saturation != 0.0 || z.luminance != 0.0);
    let colour = (mono || mixer_active || vibrance != 0.0 || saturation != 0.0 || grading || s.hue != 0.0).then(|| {
        let t = tables();
        let exponent = 1.0 + (100.0 - s.blending) / 100.0 * 3.0;
        ColourPlan {
            mono, mixer_active,
            mixer: std::array::from_fn(|k| [s.mixer[k].hue / 100.0 * t.band_gap[k], s.mixer[k].saturation / 100.0, s.mixer[k].luminance / 100.0]),
            saturation, vibrance, grading,
            turn: (s.hue != 0.0).then(|| { let (sin, cos) = s.hue.to_radians().sin_cos(); [cos, sin] }),
            // Grading hues are OKLCh hues: a zone's colour is pushed along its own hue.
            tint: std::array::from_fn(|i| { let z = s.zones[i]; let h = z.hue.to_radians(); [h.cos() * z.saturation / 100.0 * 0.09, h.sin() * z.saturation / 100.0 * 0.09] }),
            lift: std::array::from_fn(|i| s.zones[i].luminance / 100.0 * 0.12),
            balance: s.balance / 250.0,
            shadow_weight: (0..=ZONE_STEPS).map(|i| (1.0 - i as f32 / ZONE_STEPS as f32).powf(exponent)).collect(),
            highlight_weight: (0..=ZONE_STEPS).map(|i| (i as f32 / ZONE_STEPS as f32).powf(exponent)).collect(),
        }
    });

    Plan {
        width, height, region: [rx, ry, rw, rh], window, bit_depth, max: ((1u32 << bit_depth) - 1) as f32,
        clipping: options.clipping, measure: options.measure, matrix: matrix32, exposure, untouched, haze,
        base, clarity_base, shadows, highlights, clarity, texture, texture_sigma, sharpen, sharpen_radius,
        sharpen_masking: s.sharpen_masking / 100.0, noise, chroma, tone, curves, colour,
        vignette: [s.vignette_amount / 100.0, 0.25 + s.vignette_midpoint / 100.0 * 0.9, 0.05 + s.vignette_feather / 100.0 * 0.75],
        grain: [s.grain_amount / 100.0 * 0.12, ((0.6 + s.grain_size / 100.0 * 3.0) * long_edge / 2000.0).max(0.5)],
    }
}

/// Dehaze on a working-space colour, before exposure has been applied or
/// after: the atmosphere scales with the light.
#[inline(always)]
fn apply_haze(haze: &Haze, rgb: &mut [f32; 4], x: usize, y: usize, light: f32) {
    let air = [haze.air[0] * light, haze.air[1] * light, haze.air[2] * light];
    if haze.amount > 0.0 {
        let t = (1.0 - (1.0 - haze.grid.at(&haze.transmission, x, y)) * haze.amount).max(0.1);
        for c in 0..3 { rgb[c] = ((rgb[c] - air[c]) / t + air[c]).max(0.0); }
    } else {
        let k = -haze.amount * 0.6;
        for c in 0..3 { rgb[c] = rgb[c] * (1.0 - k) + air[c] * k; }
    }
}

/// A guided filter of the picture's log luminance at `sigma` output pixels,
/// solved on a grid of cells a quarter of sigma wide (at most 16 pixels),
/// each measured with up to 4×4 probes. The grid covers the whole output, so
/// a region and the whole picture read the same plane.
fn broad_plane<T: Texel>(sampler: &Sampler<T>, haze: Option<&Haze>, width: usize, height: usize, sigma: f32, eps: f32) -> GuidedGrid {
    let step = ((sigma / 4.0).floor() as usize).clamp(1, 16);
    let g = Grid::over(width, height, step);
    let n = step.min(4);
    let mut low = vec![0.0f32; g.gw * g.gh];
    for gy in 0..g.gh { for gx in 0..g.gw {
        let mut sum = 0.0;
        for j in 0..n { for i in 0..n {
            let (px, py) = g.probe(gx, gy, i, j, n, width, height);
            let mut rgb = sampler.sample(px, py);
            if let Some(h) = haze { apply_haze(h, &mut rgb, px, py, 1.0); }
            sum += log_luminance([rgb[0], rgb[1], rgb[2]]);
        } }
        low[gy * g.gw + gx] = sum / (n * n) as f32;
    } }
    let (a, b) = filter::guided_coefficients(&low, g.gw, g.gh, sigma / step as f32, eps);
    GuidedGrid { grid: g, a, b }
}

/// The window planes the per-pixel pass reads at each window pixel.
pub struct WindowPlanes {
    pub texture: Option<Vec<f32>>,
    pub sharp: Option<Vec<f32>>,
    /// The luminance noise filter's `a` and `b`.
    pub noise: Option<(Vec<f32>, Vec<f32>)>,
    pub chroma: Option<(Vec<f32>, Vec<f32>)>,
}

/// One output pixel, from its working-space colour (exposure applied) and
/// the window planes at the same place. Returns display sRGB, encoded, and
/// whether the pixel clipped high and low before the roll-off.
#[inline(always)]
pub fn shade(p: &Plan, t: &Tables, rgb: [f32; 3], x: usize, y: usize, wp: &WindowPlanes, w: usize) -> ([f32; 3], bool, bool) {
    let mut c = rgb;
    // ------------------------------------------------- local tone, in stops
    let l = log_luminance(c);
    let mut delta = 0.0f32;
    if let Some(base) = &p.base {
        let b = base.at(l, x, y);
        // Shadows reach the tones below grey, Highlights those above it; a
        // tone's place is read from the edge-aware base, so a dark object in
        // a bright scene moves with the darks around it rather than alone.
        let ws = (1.0 - smooth(-5.0, 0.0, b)) * smooth(-12.0, -8.0, b);
        let wh = smooth(-0.5, 3.0, b);
        delta += p.shadows * 1.5 * ws + p.highlights * 1.5 * wh;
    }
    if let Some(cb) = &p.clarity_base {
        let b = cb.at(l, x, y);
        let detail = (l - b).clamp(-2.0, 2.0);
        // Most in the mid-tones, less at the ends, where it halos.
        delta += p.clarity * 0.7 * detail * (0.3 + 0.7 * fast::exp(-(b / 3.0) * (b / 3.0)));
    }
    if let Some(tb) = &wp.texture {
        let detail = l - tb[w];
        delta += p.texture * detail.clamp(-1.0, 1.0) * (1.0 - 0.5 * smooth(0.5, 1.5, detail.abs()));
    }
    if let Some(sb) = &wp.sharp {
        let detail = l - sb[w];
        let mask = if p.sharpen_masking != 0.0 { smooth(0.0, p.sharpen_masking * 0.15, detail.abs()) } else { 1.0 };
        delta += detail * p.sharpen * mask;
    }
    if let (Some((a, b)), Some([_, _, k])) = (&wp.noise, p.noise) {
        delta += (a[w] * l + b[w] - l) * k;
    }
    if delta != 0.0 {
        let gain = fast::exp2(delta);
        c = [c[0] * gain, c[1] * gain, c[2] * gain];
    }
    if let (Some((cr, cb)), Some((_, amount))) = (&wp.chroma, p.chroma) {
        let y0 = colour::work_luma(c);
        let (r1, b1) = (cr[w] * y0, cb[w] * y0);
        c[0] += (r1 - c[0]) * amount; c[2] += (b1 - c[2]) * amount;
        let lw = colour::WORK_LUMA;
        c[1] = ((y0 - lw[0] * c[0] - lw[2] * c[2]) / lw[1]).max(0.0);
    }

    // --------------------------------------------------------------- tone
    let mut out = [0.0f32; 3];
    let (hi, lo);
    match &p.tone {
        Tone::Scene { table } => {
            let y = colour::work_luma(c);
            let stops = fast::log2(y.max(1e-9) / GREY);
            let pos = ((stops - SCENE_STOPS[0]) / (SCENE_STOPS[1] - SCENE_STOPS[0])).clamp(0.0, 1.0) * SCENE_TABLE as f32;
            let k = (pos as usize).min(SCENE_TABLE - 1);
            let o = table[k] + (table[k + 1] - table[k]) * (pos - k as f32);
            let ratio = if y > 1e-9 { o / y } else { 0.0 };
            let s = colour::work_to_srgb([c[0] * ratio, c[1] * ratio, c[2] * ratio]);
            let peak = s[0].max(s[1]).max(s[2]);
            hi = peak > 1.0;
            lo = o <= 0.0;
            let v = if hi || outside(s) { roll_off(s, 1.0) } else { s };
            for ch in 0..3 { out[ch] = encode(t, v[ch]); }
        }
        Tone::Display { table, scale, clip_at, black, white } => {
            let s = colour::work_to_srgb(c);
            let mut v = [((s[0] + black) * white).max(0.0), ((s[1] + black) * white).max(0.0), ((s[2] + black) * white).max(0.0)];
            let peak = v[0].max(v[1]).max(v[2]);
            hi = peak > *clip_at;
            lo = v[0] <= 0.0 && v[1] <= 0.0 && v[2] <= 0.0;
            // A colour outside the screen's gamut, from the wide working
            // space, is brought in at its own hue.
            if hi || outside(s) { v = roll_off([(s[0] + black) * white, (s[1] + black) * white, (s[2] + black) * white], *clip_at); }
            for ch in 0..3 { out[ch] = table[fast::round(v[ch] * scale).min(65535)]; }
        }
    }
    if let Some(curves) = &p.curves { for ch in 0..3 { out[ch] = curve::read(&curves[ch], out[ch]); } }

    // ------------------------------------------------------------- colour
    if let Some(cp) = &p.colour { out = colour_stage(cp, t, out); }

    // ---------------------------------------------------------- finishing
    let [amount, mid, feather] = p.vignette;
    if amount != 0.0 {
        let dx = (x as f32 + 0.5) / p.width as f32 * 2.0 - 1.0; let dy = (y as f32 + 0.5) / p.height as f32 * 2.0 - 1.0;
        let f = smooth(mid - feather, mid + feather, (dx * dx + dy * dy).sqrt() / std::f32::consts::SQRT_2 * 1.4);
        for ch in 0..3 { out[ch] = if amount < 0.0 { out[ch] * (1.0 + amount * f) } else { out[ch] + (1.0 - out[ch]) * amount * f }; }
    }
    let [grain, cell] = p.grain;
    if grain != 0.0 {
        let n = grain_at(x, y, cell) * grain;
        let lm = luma(out[0], out[1], out[2]);
        let k = n * (0.35 + 2.6 * lm * (1.0 - lm));
        for ch in 0..3 { out[ch] = (out[ch] + k).clamp(0.0, 1.0); }
    }
    (out, hi, lo)
}

/// Whether linear sRGB has a channel below zero by more than rounding.
#[inline(always)]
fn outside(s: [f32; 3]) -> bool { s[0] < -1e-5 || s[1] < -1e-5 || s[2] < -1e-5 }

/// Brings linear sRGB above `white`, or outside the gamut, onto the screen.
/// Past white a colour keeps its hue and runs to white, as film does; a
/// colour too vivid for the screen loses chroma at constant lightness and hue.
#[inline(always)]
fn roll_off(s: [f32; 3], white: f32) -> [f32; 3] {
    let peak = s[0].max(s[1]).max(s[2]);
    if peak > white {
        let k = ((peak - white) / peak).clamp(0.0, 1.0);
        let lab = colour::to_oklab(s[0] / peak, s[1] / peak, s[2] / peak);
        let c = colour::fit_gamut(lab[0] + (1.0 - lab[0]) * k, lab[1] * (1.0 - k), lab[2] * (1.0 - k));
        [c[0] * white, c[1] * white, c[2] * white]
    } else {
        let lab = colour::to_oklab(s[0] / white, s[1] / white, s[2] / white);
        let c = colour::fit_gamut(lab[0], lab[1], lab[2]);
        [c[0] * white, c[1] * white, c[2] * white]
    }
}

/// The colour controls, in OKLab of the encoded output.
#[inline(always)]
fn colour_stage(cp: &ColourPlan, t: &Tables, out: [f32; 3]) -> [f32; 3] {
    let lab = colour::to_oklab(decode(t, out[0]), decode(t, out[1]), decode(t, out[2]));
    let (mut ll, mut a, mut b) = (lab[0], lab[1], lab[2]);
    let chroma = (a * a + b * b).sqrt();
    let needs_hue = cp.mixer_active || cp.mono || cp.vibrance != 0.0;
    let hue = if needs_hue { colour::hue_of(a, b) } else { 0.0 };
    // Colour edits reach a colour in proportion to how coloured it is.
    let colourful = smooth(0.0, 0.06, chroma);
    let (mut hue_shift, mut sat_shift, mut lum_shift) = (0.0f32, 0.0f32, 0.0f32);
    if cp.mixer_active || cp.mono {
        let weights = band_weights(hue, &t.band_hues);
        for k in 0..8 {
            if weights[k] == 0.0 { continue; }
            hue_shift += weights[k] * cp.mixer[k][0];
            sat_shift += weights[k] * cp.mixer[k][1];
            lum_shift += weights[k] * cp.mixer[k][2];
        }
    }
    if cp.mono {
        // The B&W mix lightens or darkens each colour's grey.
        ll *= fast::exp2(lum_shift * colourful * 0.8);
        a = 0.0; b = 0.0;
    } else {
        let mut factor = (1.0 + cp.saturation).max(0.0) * (1.0 + sat_shift).max(0.0);
        if cp.vibrance != 0.0 {
            // Vibrance lifts quiet colours more than vivid ones and leaves skin, at OKLCh hue 40-80, mostly be.
            let skin = smooth(30.0, 45.0, hue) * (1.0 - smooth(75.0, 90.0, hue)) * smooth(0.02, 0.05, chroma) * (1.0 - smooth(0.18, 0.26, chroma));
            factor *= if cp.vibrance > 0.0 { 1.0 + cp.vibrance * (1.0 - smooth(0.0, 0.25, chroma)) * (1.0 - 0.8 * skin) } else { 1.0 + cp.vibrance };
        }
        if lum_shift != 0.0 { ll *= fast::exp2(lum_shift * colourful * 0.6); }
        if hue_shift != 0.0 {
            let turn = (hue_shift * colourful).to_radians();
            let (sin, cos) = turn.sin_cos();
            let turned = a * cos - b * sin; b = a * sin + b * cos; a = turned;
        }
        a *= factor; b *= factor;
    }
    if let Some([cos, sin]) = cp.turn { let turned = a * cos - b * sin; b = a * sin + b * cos; a = turned; }
    if cp.grading {
        let k = fast::round((ll - cp.balance).clamp(0.0, 1.0) * ZONE_STEPS as f32);
        let (ws, wh) = (cp.shadow_weight[k], cp.highlight_weight[k]);
        let wm = (1.0 - ws - wh).max(0.0);
        a += cp.tint[0][0] * ws + cp.tint[1][0] * wm + cp.tint[2][0] * wh + cp.tint[3][0];
        b += cp.tint[0][1] * ws + cp.tint[1][1] * wm + cp.tint[2][1] * wh + cp.tint[3][1];
        ll += cp.lift[0] * ws + cp.lift[1] * wm + cp.lift[2] * wh + cp.lift[3];
    }
    let linear = colour::fit_gamut(ll, a, b);
    [encode(t, linear[0]), encode(t, linear[1]), encode(t, linear[2])]
}

/// How much of each colour band an OKLCh hue takes: its two neighbouring
/// bands share it along a raised cosine, so a hue between two bands moves
/// smoothly from one to the other and the weights always sum to one.
#[inline(always)]
pub fn band_weights(hue: f32, band_hues: &[f32; 8]) -> [f32; 8] {
    let mut weights = [0.0f32; 8];
    let h = if hue < band_hues[0] { hue + 360.0 } else { hue };
    for i in 0..8 {
        let from = band_hues[i];
        let to = if i + 1 < 8 { band_hues[i + 1] } else { band_hues[0] + 360.0 };
        if h >= from && h < to {
            let t = (h - from) / (to - from);
            let near = 0.5 + 0.5 * (std::f32::consts::PI * t).cos();
            weights[i] = near; weights[(i + 1) % 8] = 1.0 - near;
            break;
        }
    }
    weights
}

/// Builds the window planes for a plan.
fn window_planes<T: Texel>(p: &Plan, sampler: &Sampler<T>) -> WindowPlanes {
    if !p.uses_window() { return WindowPlanes { texture: None, sharp: None, noise: None, chroma: None }; }
    let [x0, y0, x1, y1] = p.window;
    let (ww, wh) = (x1 - x0, y1 - y0);
    let mut lum = vec![0.0f32; ww * wh];
    let (mut cr, mut cb) = if p.chroma.is_some() { (vec![0.0f32; ww * wh], vec![0.0f32; ww * wh]) } else { (Vec::new(), Vec::new()) };
    for y in y0..y1 { for x in x0..x1 {
        let mut rgb = sampler.sample(x, y);
        if let Some(h) = &p.haze { apply_haze(h, &mut rgb, x, y, p.exposure); }
        let i = (y - y0) * ww + (x - x0);
        let c = [rgb[0], rgb[1], rgb[2]];
        lum[i] = log_luminance(c);
        if p.chroma.is_some() { let l = colour::work_luma(c) + 1e-6; cr[i] = c[0] / l; cb[i] = c[2] / l; }
    } }
    WindowPlanes {
        texture: (p.texture != 0.0).then(|| filter::gaussian(&lum, ww, wh, p.texture_sigma)),
        sharp: (p.sharpen != 0.0).then(|| filter::triple_box(&lum, ww, wh, p.sharpen_radius)),
        noise: p.noise.map(|[sigma, eps, _]| filter::guided_coefficients(&lum, ww, wh, sigma, eps)),
        chroma: p.chroma.map(|(radius, _)| (filter::triple_box(&cr, ww, wh, radius), filter::triple_box(&cb, ww, wh, radius))),
    }
}

pub fn develop(frame: &Frame, recipe: &Recipe, options: &Options) -> Developed {
    let p = plan(frame, recipe, options);
    run(frame, &p)
}

/// Runs a plan on the CPU.
pub fn run(frame: &Frame, p: &Plan) -> Developed {
    match &frame.data {
        Samples::F32(data) => run_on(frame, data.as_slice(), p),
        Samples::U16(data) => run_on(frame, data.as_slice(), p),
        Samples::Srgb8(data) => run_on(frame, data.as_slice(), p),
        Samples::Srgb16(data) => run_on(frame, data.as_slice(), p),
    }
}

fn run_on<T: Texel>(frame: &Frame, data: &[T], p: &Plan) -> Developed {
    let t = tables();
    let [rx, ry, rw, rh] = p.region;
    let pixels = rw * rh;
    let bit_depth = p.bit_depth;
    let max = p.max;
    let mut out8 = if bit_depth == 8 { vec![0u8; pixels * 4] } else { Vec::new() };
    let mut out16 = if bit_depth != 8 { vec![0u16; pixels * 4] } else { Vec::new() };
    let mut histogram = vec![0u32; 256];
    let mut histogram_rgb = vec![0u32; 768];
    let sampler = Sampler::new(frame, data, p.width, p.height, p.matrix, p.exposure);
    let untouched = p.untouched.map(|m| Sampler::new(frame, data, p.width, p.height, m, 1.0));
    let wp = window_planes(p, &sampler);
    let ww = p.window[2] - p.window[0];
    let to_histogram = 255.0 / max;
    let (mut clipped_high, mut clipped_low) = (0u32, 0u32);
    let mut luminance = if p.measure { Some(vec![0.0f32; pixels]) } else { None };
    let clip_at0 = 1.0 - 1e-6;

    for y in ry..ry + rh { for x in rx..rx + rw {
        // `i` is the output pixel; `w` the same pixel in the window.
        let i = (y - ry) * rw + (x - rx);
        let w = (y - p.window[1]) * ww + (x - p.window[0]);
        let mut rgb = sampler.sample(x, y);
        if let Some(h) = &p.haze { apply_haze(h, &mut rgb, x, y, p.exposure); }
        if let Some(lm) = luminance.as_mut() { lm[i] = colour::work_luma([rgb[0], rgb[1], rgb[2]]); }
        let (out, mut hi, mut lo) = shade(p, t, [rgb[0], rgb[1], rgb[2]], x, y, &wp, w);
        let q = [fast::round(out[0] * max) as f32, fast::round(out[1] * max) as f32, fast::round(out[2] * max) as f32];
        histogram_rgb[fast::round(q[0] * to_histogram).min(255)] += 1;
        histogram_rgb[256 + fast::round(q[1] * to_histogram).min(255)] += 1;
        histogram_rgb[512 + fast::round(q[2] * to_histogram).min(255)] += 1;
        histogram[fast::round(luma(q[0], q[1], q[2]) * to_histogram).min(255)] += 1;
        if hi || lo {
            // Clipping is reported for what the edit did.
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
        if p.clipping && hi { px = [max, 0.0, 0.0]; } else if p.clipping && lo { px = [0.0, 0.0, max]; }
        let alpha = fast::round(rgb[3] * max) as f32;
        let d = i * 4;
        if bit_depth == 8 { out8[d] = px[0] as u8; out8[d + 1] = px[1] as u8; out8[d + 2] = px[2] as u8; out8[d + 3] = alpha as u8; }
        else { out16[d] = px[0] as u16; out16[d + 1] = px[1] as u16; out16[d + 2] = px[2] as u16; out16[d + 3] = alpha as u16; }
    } }
    Developed { width: rw, height: rh, bit_depth, data: if bit_depth == 8 { Pixels::U8(out8) } else { Pixels::U16(out16) },
        histogram, histogram_rgb, clipped_high, clipped_low, luminance }
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

#[cfg(test)]
mod tests {
    use super::*;
    fn card(w: usize, h: usize) -> Vec<u8> {
        let mut rgba = vec![0u8; w * h * 4];
        for y in 0..h { for x in 0..w {
            let i = (y * w + x) * 4;
            rgba[i] = (x * 255 / w) as u8; rgba[i + 1] = (y * 255 / h) as u8; rgba[i + 2] = ((x + y) * 127 / (w + h)) as u8; rgba[i + 3] = 255;
        } }
        rgba
    }
    fn bytes(d: &Developed) -> &[u8] { match &d.data { Pixels::U8(v) => v, Pixels::U16(_) => panic!() } }
    #[test]
    fn a_region_matches_the_same_pixels_of_the_whole() {
        let frame = from_rgba(240, 160, &card(240, 160));
        let recipe = Recipe::from_json(r#"{"highlights":-40,"shadows":30,"texture":40,"clarity":30,"dehaze":20,"sharpening":{"amount":80},"noise":{"luminance":40,"colour":40},"vibrance":20,"vignette":{"amount":-30},"curve":{"points":[[0,0],[100,80],[255,255]]}}"#);
        let whole = develop(&frame, &recipe, &Options::default());
        let part = develop(&frame, &recipe, &Options { region: Some([100, 50, 60, 40]), ..Options::default() });
        let (a, b) = (bytes(&whole), bytes(&part));
        assert_eq!((part.width, part.height), (60, 40));
        for y in 0..40 { for x in 0..60 { for c in 0..4 {
            assert_eq!(a[((50 + y) * 240 + 100 + x) * 4 + c], b[(y * 60 + x) * 4 + c], "pixel {x},{y}");
        } } }
    }
    #[test]
    fn the_default_recipe_returns_a_photograph() {
        let rgba = card(64, 48);
        let developed = develop(&from_rgba(64, 48, &rgba), &Recipe::default(), &Options::default());
        assert_eq!(bytes(&developed), rgba.as_slice());
    }
    #[test]
    fn a_deep_photograph_is_kept_as_sixteen_bit() {
        // 10-bit values, as a HEIC decoder returns them: the largest value is 1023.
        let rgba: Vec<u16> = (0..16).flat_map(|i| [i * 68, 512, 1023, 1023]).collect();
        let frame = from_rgba16(4, 4, &rgba, 10);
        let Samples::Srgb16(data) = &frame.data else { panic!("not a 16-bit frame") };
        assert_eq!((frame.bit_depth(), data[1], data[2]), (16, Encoded16(32800), Encoded16(65535)));
        assert!(frame.alpha.is_none());
        // Neighbouring 10-bit values stay apart in a 10-bit result, and the default recipe returns the picture.
        let developed = develop(&frame, &Recipe::default(), &Options { bit_depth: 10, ..Options::default() });
        let Pixels::U16(out) = &developed.data else { panic!() };
        for (i, pixel) in out.chunks(4).enumerate() { assert!((pixel[0] as i32 - i as i32 * 68).abs() <= 1 && (pixel[1] as i32 - 512).abs() <= 1 && pixel[2] == 1023); }
        // Opacity is kept at 8 bits when a pixel is not opaque.
        let mut clear = rgba.clone(); clear[3] = 512;
        assert_eq!(from_rgba16(4, 4, &clear, 10).alpha.unwrap()[..2], [128, 255]);
        // An ICC profile changes how the same values are read.
        let mut managed = from_rgba16(4, 4, &rgba, 10);
        assert!(managed.set_profile(&crate::icc::tests::display_p3()) && managed.curve.is_some() && managed.matrix[0] > 1.2);
        let mut plain = from_rgba(4, 4, &card(4, 4));
        assert!(!plain.set_profile(b"not a profile") && plain.curve.is_none());
    }
    #[test]
    fn an_opaque_photograph_is_kept_as_eight_bit() {
        let frame = from_rgba(4, 4, &card(4, 4));
        assert!(matches!(frame.data, Samples::Srgb8(_)) && frame.alpha.is_none());
    }
    #[test]
    fn a_proxy_develops_like_its_source() {
        let rgba = card(120, 80);
        let frame = from_rgba(120, 80, &rgba);
        let proxy = frame.linear_preview(120);
        let recipe = Recipe::from_json(r#"{"exposure":0.4,"shadows":40,"clarity":20,"vibrance":30}"#);
        let (a, b) = (develop(&frame, &recipe, &Options::default()), develop(&proxy, &recipe, &Options::default()));
        let worst = bytes(&a).iter().zip(bytes(&b)).map(|(x, y)| (*x as i32 - *y as i32).abs()).max().unwrap();
        assert!(worst <= 1, "{worst}");
    }
    #[test]
    fn shadows_lift_the_darks_and_highlights_lower_the_lights() {
        let rgba = card(160, 120);
        let frame = from_rgba(160, 120, &rgba);
        let plain = develop(&frame, &Recipe::default(), &Options::default());
        let lifted = develop(&frame, &Recipe::from_json(r#"{"shadows":80}"#), &Options::default());
        let lowered = develop(&frame, &Recipe::from_json(r#"{"highlights":-80}"#), &Options::default());
        let mean = |d: &Developed, dark: bool| {
            let (mut s, mut n) = (0.0, 0.0);
            for (k, px) in bytes(&plain).chunks(4).enumerate() {
                let l = luma(px[0] as f32, px[1] as f32, px[2] as f32);
                if (dark && l < 50.0) || (!dark && l > 180.0) { let q = &bytes(d)[k * 4..k * 4 + 3]; s += luma(q[0] as f32, q[1] as f32, q[2] as f32); n += 1.0; }
            }
            s / n
        };
        assert!(mean(&lifted, true) > mean(&plain, true) + 4.0);
        assert!(mean(&lowered, false) < mean(&plain, false) - 4.0);
        // Each leaves the other end mostly where it was.
        assert!((mean(&lifted, false) - mean(&plain, false)).abs() < 3.0);
    }
    #[test]
    fn exposure_reuses_the_broad_planes() {
        // No pure black: log luminance holds still below its floor.
        let frame = from_rgba(160, 120, &card(160, 120).iter().map(|v| v.saturating_add(20)).collect::<Vec<u8>>());
        let a = plan(&frame, &Recipe::from_json(r#"{"highlights":-50}"#), &Options::default());
        let b = plan(&frame, &Recipe::from_json(r#"{"highlights":-50,"exposure":1}"#), &Options::default());
        let (pa, pb) = (a.base.unwrap(), b.base.unwrap());
        assert_eq!(pa.a, pb.a);
        // The shifted plane is the plane the exposed picture would give.
        PLANES.with(|c| c.borrow_mut().clear());
        let sampler_frame = &frame;
        let Samples::Srgb8(data) = &sampler_frame.data else { panic!() };
        let exposed = Sampler::new(sampler_frame, data, 160, 120, b.matrix, 2.0);
        let direct = broad_plane(&exposed, None, 160, 120, (0.02f32 * 160.0).max(1.0), 0.25);
        for (x, y) in pb.b.iter().zip(&direct.b) { assert!((x - y).abs() < 2e-3, "{x} {y}"); }
    }
    #[test]
    fn the_scene_tone_keeps_grey_and_rolls_into_white() {
        let grey = scene_tone(0.0, 0.0, 0.0, 0.0);
        assert!(grey > 0.18 && grey < 0.24, "{grey}");
        assert!(scene_tone(3.0, 0.0, 0.0, 0.0) < 1.0 && scene_tone(3.0, 0.0, 0.0, 0.0) > 0.95);
        let mut last = 0.0;
        for i in -60..40 { let v = scene_tone(i as f32 / 5.0, 0.0, 0.0, 0.0); assert!(v >= last); last = v; }
        assert!(scene_tone(-2.0, 0.0, 0.0, -1.0) < scene_tone(-2.0, 0.0, 0.0, 0.0));
        assert!(scene_tone(1.0, 0.0, 1.0, 0.0) > scene_tone(1.0, 0.0, 0.0, 0.0));
        assert!(scene_tone(1.0, 50.0, 0.0, 0.0) > scene_tone(1.0, 0.0, 0.0, 0.0) && scene_tone(-1.0, 50.0, 0.0, 0.0) < scene_tone(-1.0, 0.0, 0.0, 0.0));
    }
    #[test]
    fn hue_turns_every_colour_and_keeps_greys() {
        let rgba = card(64, 48);
        let frame = from_rgba(64, 48, &rgba);
        let turned = develop(&frame, &Recipe::from_json(r#"{"hue":120}"#), &Options::default());
        let px = bytes(&turned);
        // A red turns towards green; a grey stays grey.
        let red = from_rgba(1, 1, &[200, 30, 30, 255]);
        let out = develop(&red, &Recipe::from_json(r#"{"hue":120}"#), &Options::default());
        let q = bytes(&out);
        assert!(q[1] > q[0], "{q:?}");
        let grey = from_rgba(1, 1, &[128, 128, 128, 255]);
        assert_eq!(&bytes(&develop(&grey, &Recipe::from_json(r#"{"hue":120}"#), &Options::default()))[..3], &[128, 128, 128]);
        assert_ne!(px, rgba.as_slice());
    }
    #[test]
    fn band_weights_sum_to_one_and_peak_on_their_band() {
        let hues = tables().band_hues;
        for i in 0..720 {
            let w = band_weights(i as f32 * 0.5, &hues);
            assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        }
        for (k, h) in hues.iter().enumerate() { assert!((band_weights(*h, &hues)[k] - 1.0).abs() < 1e-5); }
    }
    #[test]
    fn a_saturated_camera_colour_keeps_its_hue() {
        // A deep blue just outside sRGB, as a camera records a blue light:
        // the working space keeps it, and the gamut map keeps its hue.
        let deep = colour::apply(&colour::SRGB_TO_WORK, [-0.05, 0.02, 0.9]);
        let frame = Frame { width: 1, height: 1, data: Samples::F32(deep.map(|v| v as f32).to_vec()), alpha: None, scale: 1.0, wb: [1.0; 3],
            matrix: IDENTITY, camera: Camera { matrix: IDENTITY, wb: [1.0; 3], daylight: None }, source_width: 1, curve: None, working: true, scene: false, id: next_id() };
        let out = develop(&frame, &Recipe::default(), &Options::default());
        let px = bytes(&out);
        let lin = [px[0], px[1], px[2]].map(|v| to_linear(v as f32 / 255.0));
        let (a, b) = (colour::to_oklab(-0.05, 0.02, 0.9), colour::to_oklab(lin[0], lin[1], lin[2]));
        let turn = (colour::hue_of(a[1], a[2]) - colour::hue_of(b[1], b[2])).abs();
        assert!(turn < 3.0 && (a[0] - b[0]).abs() < 0.02, "{px:?} turned {turn}");
    }
}
