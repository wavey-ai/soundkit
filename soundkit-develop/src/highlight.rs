//! Highlight reconstruction on demosaiced camera RGB, before white balance.
//!
//! A sensor channel that reached the white level holds no more information:
//! a bright sky or a lit cheek then turns magenta or cyan, because the other
//! channels kept rising. Reconstruction gives the clipped channel the value
//! that the colour of the surrounding highlight asks for.
//!
//! The colour reference is the white-balanced log ratio of red and blue to
//! green, measured on bright unclipped pixels and averaged on a coarse grid,
//! with the brightest pixels counting most. A push-pull pyramid fills the
//! grid cells inside large clipped areas from their neighbours. A clipped
//! channel then takes the value that the reference colour gives at the
//! brightness of the unclipped channels. Near the clip level the rebuilt
//! value fades in, so no edge shows where clipping starts. A pixel with every
//! channel clipped becomes neutral at the brightest level the sensor records.

/// The fraction of the white level at which a channel counts as clipped.
pub const CLIP: f32 = 0.985;
/// The rebuilt value fades in over this distance below the clip level.
const FADE: f32 = 0.05;
/// Grid cell size, in pixels.
const CELL: usize = 8;
/// Pixels darker than this fraction of white do not set the reference colour.
const REFERENCE_FLOOR: f32 = 0.2;

#[inline(always)]
fn smoothstep(a: f32, b: f32, x: f32) -> f32 { let t = ((x - a) / (b - a)).clamp(0.0, 1.0); t * t * (3.0 - 2.0 * t) }

/// A grid of log colour ratios with a weight per cell.
struct Field { w: usize, h: usize, value: Vec<[f32; 2]>, weight: Vec<f32> }

impl Field {
    /// The next coarser level: each 2×2 block, weighted.
    fn coarser(&self) -> Field {
        let (w, h) = (self.w.div_ceil(2), self.h.div_ceil(2));
        let mut out = Field { w, h, value: vec![[0.0; 2]; w * h], weight: vec![0.0; w * h] };
        for y in 0..self.h { for x in 0..self.w {
            let (i, o) = (y * self.w + x, (y / 2) * w + x / 2);
            let k = self.weight[i];
            if k > 0.0 { out.value[o][0] += self.value[i][0] * k; out.value[o][1] += self.value[i][1] * k; out.weight[o] += k; }
        } }
        for o in 0..w * h { if out.weight[o] > 0.0 { out.value[o][0] /= out.weight[o]; out.value[o][1] /= out.weight[o]; } }
        out
    }
    fn bilinear(&self, fx: f32, fy: f32) -> [f32; 2] {
        let fx = fx.clamp(0.0, (self.w - 1) as f32);
        let fy = fy.clamp(0.0, (self.h - 1) as f32);
        let (x0, y0) = (fx as usize, fy as usize);
        let (x1, y1) = ((x0 + 1).min(self.w - 1), (y0 + 1).min(self.h - 1));
        let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
        let v = |x: usize, y: usize| self.value[y * self.w + x];
        std::array::from_fn(|c| {
            let top = v(x0, y0)[c] * (1.0 - tx) + v(x1, y0)[c] * tx;
            let bottom = v(x0, y1)[c] * (1.0 - tx) + v(x1, y1)[c] * tx;
            top * (1.0 - ty) + bottom * ty
        })
    }
    /// Push-pull: every empty cell takes the value of the next coarser level
    /// at its position, filled the same way first.
    fn fill(&mut self) {
        if self.weight.iter().all(|k| *k > 0.0) || !self.weight.iter().any(|k| *k > 0.0) || self.w * self.h <= 1 { return; }
        let mut coarse = self.coarser();
        coarse.fill();
        for y in 0..self.h { for x in 0..self.w {
            let i = y * self.w + x;
            if self.weight[i] > 0.0 { continue; }
            self.value[i] = coarse.bilinear((x as f32 + 0.5) / 2.0 - 0.5, (y as f32 + 0.5) / 2.0 - 0.5);
            self.weight[i] = f32::MIN_POSITIVE;
        } }
    }
}

/// Rebuilds clipped channels of 16-bit camera RGB in place. A sample times
/// `scale` is one at the sensor's white level; `wb` holds the white balance
/// multipliers applied afterwards. Rebuilt values can pass the white level,
/// so the samples are halved as often as they need to fit in 16 bits, and the
/// returned scale reads them. Returns the number of pixels with a clipped
/// channel and the new scale.
pub fn reconstruct(data: &mut [u16], width: usize, height: usize, scale: f32, wb: [f32; 3]) -> (usize, f32) {
    let (w, h) = (width, height);
    if w == 0 || h == 0 || data.len() < w * h * 3 { return (0, scale); }
    let read = |data: &[u16], i: usize| [data[i * 3] as f32 * scale * wb[0], data[i * 3 + 1] as f32 * scale * wb[1], data[i * 3 + 2] as f32 * scale * wb[2]];
    // In white-balanced units each channel clips at its own level.
    let limit = [CLIP * wb[0], CLIP * wb[1], CLIP * wb[2]];
    let near = |q: [f32; 3], c: usize| q[c] >= limit[c] * (1.0 - FADE / CLIP);
    let clipped = |q: [f32; 3]| (0..3).any(|c| q[c] >= limit[c]);
    let count = (0..w * h).filter(|&i| clipped(read(data, i))).count();
    if count == 0 { return (0, scale); }

    // The reference colour, cell by cell.
    let (gw, gh) = (w.div_ceil(CELL), h.div_ceil(CELL));
    let mut field = Field { w: gw, h: gh, value: vec![[0.0; 2]; gw * gh], weight: vec![0.0; gw * gh] };
    for y in 0..h { for x in 0..w {
        let q = read(data, y * w + x);
        if (0..3).any(|c| near(q, c)) { continue; }
        let peak = (q[0] / wb[0]).max(q[1] / wb[1]).max(q[2] / wb[2]);
        if peak < REFERENCE_FLOOR || q[0] <= 0.0 || q[1] <= 0.0 || q[2] <= 0.0 { continue; }
        let k = peak * peak;
        let cell = (y / CELL) * gw + x / CELL;
        field.value[cell][0] += (q[0] / q[1]).ln() * k;
        field.value[cell][1] += (q[2] / q[1]).ln() * k;
        field.weight[cell] += k;
    } }
    let referenced = field.weight.iter().any(|k| *k > 0.0);
    for i in 0..gw * gh { if field.weight[i] > 0.0 { field.value[i][0] /= field.weight[i]; field.value[i][1] /= field.weight[i]; } }
    field.fill();

    let brightest = limit[0].max(limit[1]).max(limit[2]);
    let mut rebuilt = Vec::with_capacity(count);
    let mut peak = 0.0f32;
    for y in 0..h { for x in 0..w {
        let i = y * w + x;
        let q = read(data, i);
        if !(0..3).any(|c| near(q, c)) { continue; }
        let fade: [f32; 3] = std::array::from_fn(|c| smoothstep(limit[c] * (1.0 - FADE / CLIP), limit[c], q[c]));
        let out = if (0..3).all(|c| q[c] >= limit[c]) {
            [q[0].max(q[1]).max(q[2]).max(brightest); 3]
        } else {
            let [lr, lb] = if referenced { field.bilinear((x as f32 + 0.5) / CELL as f32 - 0.5, (y as f32 + 0.5) / CELL as f32 - 0.5) } else { [0.0, 0.0] };
            let ratio = [lr.exp(), 1.0, lb.exp()];
            // The green level the reference colour implies, from the channels
            // that are surely unclipped; the brightest estimate wins, as a
            // clipped channel can only have been brighter than it reads.
            let mut green = 0.0f32;
            for c in 0..3 { if fade[c] < 1.0 { green = green.max(q[c] / ratio[c]); } }
            std::array::from_fn(|c| q[c] + fade[c] * (green * ratio[c] - q[c]).max(0.0))
        };
        let out = [out[0] / wb[0], out[1] / wb[1], out[2] / wb[2]];
        peak = peak.max(out[0]).max(out[1]).max(out[2]);
        rebuilt.push((i, out));
    } }

    // Halve the samples until the brightest rebuilt value fits.
    let mut shift = 0u32;
    while peak / scale / (1u32 << shift) as f32 > 65535.0 && shift < 8 { shift += 1; }
    if shift > 0 {
        let half = 1u32 << (shift - 1);
        for v in data.iter_mut() { *v = ((*v as u32 + half) >> shift) as u16; }
    }
    let scale = scale * (1u32 << shift) as f32;
    for (i, out) in rebuilt {
        for c in 0..3 { data[i * 3 + c] = (out[c] / scale).round().clamp(0.0, 65535.0) as u16; }
    }
    (count, scale)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn warm_patch(white: f32, bright: f32) -> Vec<u16> {
        let (w, h) = (32, 32);
        let mut data = vec![0u16; w * h * 3];
        for y in 0..h { for x in 0..w {
            let i = (y * w + x) * 3;
            let level = if (10..22).contains(&x) && (10..22).contains(&y) { bright } else { 0.6 };
            for (c, ratio) in [1.0f32, 0.6, 0.3].iter().enumerate() { data[i + c] = (white * (level * ratio).min(1.0)) as u16; }
        } }
        data
    }
    #[test]
    fn a_clipped_channel_takes_the_colour_around_it() {
        let white = 4000.0;
        let mut data = warm_patch(white, 2.0);
        let (count, scale) = reconstruct(&mut data, 32, 32, 1.0 / white, [1.0; 3]);
        assert_eq!(count, 144);
        let i = (16 * 32 + 16) * 3;
        let p = [data[i] as f32 * scale, data[i + 1] as f32 * scale, data[i + 2] as f32 * scale];
        assert!(p[0] > 1.5 && (p[0] / p[1] - 1.0 / 0.6).abs() < 0.05, "{p:?}");
        // The surroundings are untouched.
        assert_eq!(data[0], (white * 0.6) as u16);
    }
    #[test]
    fn every_channel_clipped_turns_neutral() {
        let white = 4000.0;
        let mut data = warm_patch(white, 5.0);
        let wb = [2.0, 1.0, 1.6];
        let (_, scale) = reconstruct(&mut data, 32, 32, 1.0 / white, wb);
        let i = (16 * 32 + 16) * 3;
        let q = [data[i] as f32 * scale * wb[0], data[i + 1] as f32 * scale * wb[1], data[i + 2] as f32 * scale * wb[2]];
        assert!((q[0] - q[1]).abs() < 0.01 && (q[2] - q[1]).abs() < 0.01, "{q:?}");
    }
    #[test]
    fn rebuilt_values_past_sixteen_bits_rescale_the_samples() {
        let (w, h) = (8, 8);
        let mut data = vec![0u16; w * h * 3];
        // Red and green clip in one pixel whose blue says it was six times
        // brighter than the white level in red.
        for i in 0..w * h { data[i * 3] = if i == 27 { 65535 } else { 30000 }; data[i * 3 + 1] = if i == 27 { 65535 } else { 6000 }; data[i * 3 + 2] = if i == 27 { 40000 } else { 3000 }; }
        let (count, scale) = reconstruct(&mut data, w, h, 1.0 / 65535.0, [1.0; 3]);
        assert_eq!(count, 1);
        assert!(scale >= 8.0 / 65535.0, "{}", scale * 65535.0);
        assert!((data[27 * 3] as f32 * scale - 6.1).abs() < 0.2, "{}", data[27 * 3] as f32 * scale);
        assert!((data[0] as f32 * scale - 30000.0 / 65535.0).abs() < 1e-4);
    }
    #[test]
    fn nothing_clipped_changes_nothing() {
        let mut data = vec![1000u16; 4 * 4 * 3];
        let before = data.clone();
        assert_eq!(reconstruct(&mut data, 4, 4, 1.0 / 4095.0, [2.0, 1.0, 1.5]), (0, 1.0 / 4095.0));
        assert_eq!(data, before);
    }
}
