//! Colour science: chromatic adaptation for white balance, and OKLab, the
//! perceptual space the colour controls work in.

pub type Matrix = [f64; 9];
pub const IDENTITY: Matrix = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

pub fn multiply(a: &Matrix, b: &Matrix) -> Matrix {
    let mut out = [0.0; 9];
    for row in 0..3 { for col in 0..3 {
        out[row * 3 + col] = a[row * 3] * b[col] + a[row * 3 + 1] * b[3 + col] + a[row * 3 + 2] * b[6 + col];
    } }
    out
}
pub fn invert(m: &Matrix) -> Option<Matrix> {
    let [a, b, c, d, e, f, g, h, i] = *m;
    let (aa, bb, cc) = (e * i - f * h, -(d * i - f * g), d * h - e * g);
    let det = a * aa + b * bb + c * cc;
    if det.abs() < 1e-9 { return None; }
    Some([aa / det, -(b * i - c * h) / det, (b * f - c * e) / det,
        bb / det, (a * i - c * g) / det, -(a * f - c * d) / det,
        cc / det, -(a * h - b * g) / det, (a * e - b * d) / det])
}
pub fn diagonal([r, g, b]: [f64; 3]) -> Matrix { [r, 0.0, 0.0, 0.0, g, 0.0, 0.0, 0.0, b] }
pub fn apply(m: &Matrix, [x, y, z]: [f64; 3]) -> [f64; 3] {
    [m[0] * x + m[1] * y + m[2] * z, m[3] * x + m[4] * y + m[5] * z, m[6] * x + m[7] * y + m[8] * z]
}

// Linear sRGB (D65) and CIE XYZ.
pub const RGB_TO_XYZ: Matrix = [0.4124564, 0.3575761, 0.1804375, 0.2126729, 0.7151522, 0.0721750, 0.0193339, 0.1191920, 0.9503041];

// The working space is linear Rec.2020 with the sRGB white, so a camera's
// saturated colours keep their hue through every edit until the final gamut
// map. Each row sums to one: a neutral stays exactly neutral.
pub const SRGB_TO_WORK: Matrix = [0.627452331038, 0.329248518088, 0.0432991508737, 0.0691092776042, 0.919530987598, 0.0113597347977, 0.0163975636156, 0.0880301151684, 0.895572321216];
pub const WORK_TO_SRGB: Matrix = [1.66036293778, -0.587540226938, -0.0728227108432, -0.124563751098, 1.1329115641, -0.00834781299972, -0.0181565968008, -0.100601710304, 1.1187583071];
/// Luminance weights of the working space.
pub const WORK_LUMA: [f32; 3] = [0.2627216817, 0.6779893173, 0.0592890010];
/// A linear-sRGB transform, as the same transform in the working space.
pub fn to_work(m: &Matrix) -> Matrix { multiply(&SRGB_TO_WORK, &multiply(m, &WORK_TO_SRGB)) }
#[inline(always)]
pub fn work_to_srgb(c: [f32; 3]) -> [f32; 3] {
    const M: [f32; 9] = [1.66036293778, -0.587540226938, -0.0728227108432, -0.124563751098, 1.1329115641, -0.00834781299972, -0.0181565968008, -0.100601710304, 1.1187583071];
    [M[0] * c[0] + M[1] * c[1] + M[2] * c[2], M[3] * c[0] + M[4] * c[1] + M[5] * c[2], M[6] * c[0] + M[7] * c[1] + M[8] * c[2]]
}
#[inline(always)]
pub fn work_luma(c: [f32; 3]) -> f32 { WORK_LUMA[0] * c[0] + WORK_LUMA[1] * c[1] + WORK_LUMA[2] * c[2] }
// Bradford cone response, for chromatic adaptation.
const BRADFORD: Matrix = [0.8951, 0.2664, -0.1614, -0.7502, 1.7135, 0.0367, 0.0389, -0.0685, 1.0296];

/// The XYZ transform that adapts colours seen under `from` (a white in XYZ)
/// to how they look under `to`.
pub fn adaptation_xyz(from: [f64; 3], to: [f64; 3]) -> Matrix {
    let bradford_inverse = invert(&BRADFORD).unwrap();
    let source = apply(&BRADFORD, from);
    let target = apply(&BRADFORD, to);
    multiply(&bradford_inverse, &multiply(&diagonal([target[0] / source[0], target[1] / source[1], target[2] / source[2]]), &BRADFORD))
}
/// The same adaptation as a linear-sRGB transform.
pub fn adaptation(from: [f64; 3], to: [f64; 3]) -> Matrix {
    let xyz_to_rgb = invert(&RGB_TO_XYZ).unwrap();
    multiply(&xyz_to_rgb, &multiply(&adaptation_xyz(from, to), &RGB_TO_XYZ))
}

/// CIE 1960 uv of the black body at `kelvin` (Krystek's approximation).
fn planckian(kelvin: f64) -> (f64, f64) {
    let t = kelvin.clamp(1000.0, 25000.0);
    ((0.860117757 + 1.54118254e-4 * t + 1.28641212e-7 * t * t) / (1.0 + 8.42420235e-4 * t + 7.08145163e-7 * t * t),
        (0.317398726 + 4.22806245e-5 * t + 4.20481691e-8 * t * t) / (1.0 - 2.89741816e-5 * t + 1.61456053e-7 * t * t))
}
/// The white of a light at `kelvin`, moved `duv` off the black-body locus
/// (positive towards green), as XYZ with Y of one.
pub fn white_of(kelvin: f64, duv: f64) -> [f64; 3] {
    let (u, v) = planckian(kelvin);
    let (u2, v2) = planckian(kelvin + 1.0);
    let (mut nu, mut nv) = (-(v2 - v), u2 - u);
    let length = nu.hypot(nv);
    let length = if length > 0.0 { length } else { 1.0 };
    nu /= length; nv /= length;
    if nv < 0.0 { nu = -nu; nv = -nv; }
    let (uu, vv) = (u + nu * duv, v + nv * duv);
    let d = 2.0 * uu - 8.0 * vv + 4.0;
    let (x, y) = (3.0 * uu / d, 2.0 * vv / d);
    [x / y, 1.0, (1.0 - x - y) / y]
}
const NEUTRAL_KELVIN: f64 = 6504.0;
/// Temperature and Tint as an adaptation: the picture is corrected as if it
/// had been lit by a light that far from neutral. Positive temperature warms
/// the picture and positive tint moves it towards magenta.
pub fn temperature_tint(temperature: f64, tint: f64) -> Option<Matrix> {
    if temperature == 0.0 && tint == 0.0 { return None; }
    let mired = 1e6 / NEUTRAL_KELVIN * 2f64.powf(-temperature / 100.0);
    Some(adaptation(white_of(1e6 / mired, tint / 100.0 * 0.02), white_of(NEUTRAL_KELVIN, 0.0)))
}

// OKLab: lightness, chroma and hue move independently of each other.
#[inline(always)]
pub fn to_oklab(r: f32, g: f32, b: f32) -> [f32; 3] {
    let l = crate::fast::cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
    let m = crate::fast::cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
    let s = crate::fast::cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
    [0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s]
}
#[inline(always)]
pub fn from_oklab(l: f32, a: f32, b: f32) -> [f32; 3] {
    let l0 = l + 0.3963377774 * a + 0.2158037573 * b;
    let m0 = l - 0.1055613458 * a - 0.0638541728 * b;
    let s0 = l - 0.0894841775 * a - 1.2914855480 * b;
    let (l3, m3, s3) = (l0 * l0 * l0, m0 * m0 * m0, s0 * s0 * s0);
    [4.0767416621 * l3 - 3.3077115913 * m3 + 0.2309699292 * s3,
        -1.2684380046 * l3 + 2.6097574011 * m3 - 0.3413193965 * s3,
        -0.0041960863 * l3 - 0.7034186147 * m3 + 1.7076147010 * s3]
}
#[inline(always)]
fn in_gamut(c: &[f32; 3]) -> bool {
    c[0] >= -1e-6 && c[1] >= -1e-6 && c[2] >= -1e-6 && c[0] <= 1.0 + 1e-6 && c[1] <= 1.0 + 1e-6 && c[2] <= 1.0 + 1e-6
}
/// Brings an OKLab colour into linear sRGB by lowering its chroma, keeping
/// its lightness and hue: a colour too vivid for the screen stays the same
/// colour, a little less vivid, instead of changing hue at a clipped channel.
pub fn fit_gamut(l: f32, a: f32, b: f32) -> [f32; 3] {
    let l = l.clamp(0.0, 1.0);
    let first = from_oklab(l, a, b);
    if in_gamut(&first) { return first; }
    // Twelve halvings place the chroma within 0.03 percent: near black, where
    // sRGB is steepest, that is less than a level of 10-bit output.
    let (mut lo, mut hi) = (0.0f32, 1.0f32);
    for _ in 0..12 {
        let k = (lo + hi) * 0.5;
        if in_gamut(&from_oklab(l, a * k, b * k)) { lo = k; } else { hi = k; }
    }
    let c = from_oklab(l, a * lo, b * lo);
    [c[0].clamp(0.0, 1.0), c[1].clamp(0.0, 1.0), c[2].clamp(0.0, 1.0)]
}
#[inline(always)]
pub fn hue_of(a: f32, b: f32) -> f32 { crate::fast::hue(a, b) }

/// The OKLCh hue of each colour band. The bands sit where Lightroom's colour
/// mixer puts them: the sRGB colours at HSV hues 0, 30, 60, 120, 180, 225,
/// 270 and 315 degrees, made linear.
pub fn band_hues() -> [f32; 8] {
    let named: [[f32; 3]; 8] = [[1.0, 0.0, 0.0], [1.0, 0.5, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.5, 1.0], [0.5, 0.0, 1.0], [1.0, 0.0, 0.5]];
    let mut hues = named.map(|c| {
        let [r, g, b] = c.map(|v| if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) });
        let lab = to_oklab(r, g, b);
        hue_of(lab[1], lab[2])
    });
    // Magenta lies just past 360 degrees: the bands run once round the circle
    // from red, so each hue is larger than the one before.
    for i in 1..8 { while hues[i] <= hues[i - 1] { hues[i] += 360.0; } }
    hues
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn white_point_lands_on_d65() {
        let [x, _, z] = white_of(6504.0, 0.0032);
        let (cx, cy) = (x / (x + 1.0 + z), 1.0 / (x + 1.0 + z));
        assert!((cx - 0.3127).abs() < 0.002 && (cy - 0.3290).abs() < 0.002, "{cx} {cy}");
    }
    #[test]
    fn adapting_a_white_to_itself_changes_nothing() {
        let w = white_of(5000.0, 0.0);
        let m = adaptation(w, w);
        for (i, v) in m.iter().enumerate() { assert!((v - IDENTITY[i]).abs() < 1e-9); }
    }
    #[test]
    fn oklab_round_trips() {
        let lab = to_oklab(0.2, 0.5, 0.8);
        let back = from_oklab(lab[0], lab[1], lab[2]);
        for (c, v) in [0.2, 0.5, 0.8].iter().enumerate() { assert!((back[c] - v).abs() < 1e-4); }
    }
    #[test]
    fn band_hues_run_round_the_circle() {
        let h = band_hues();
        for i in 1..8 { assert!(h[i] > h[i - 1]); }
    }
}
