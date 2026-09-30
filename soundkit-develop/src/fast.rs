//! Fast forms of the functions the per-pixel loop calls. WebAssembly has no
//! instruction for these, so the standard forms are software library calls;
//! these stay inline. Each is accurate to about one part in a million or
//! better, well inside what 8- to 12-bit output can show.

/// Rounds a non-negative value to the nearest integer, halves up, as
/// JavaScript's `Math.round` does.
#[inline(always)]
pub fn round(x: f32) -> usize { (x + 0.5) as usize }

/// 2 to the power `x`.
#[inline(always)]
pub fn exp2(x: f32) -> f32 {
    let x = x.clamp(-126.0, 126.0);
    // The integer part goes to the exponent bits; the rest is within half
    // of zero, where the series for 2^f is accurate to about 1e-7.
    let i = (x + 0.5).floor();
    let f = x - i;
    let p = 1.0 + f * (0.693_147_2 + f * (0.240_226_5 + f * (0.055_504_11 + f * (0.009_618_129 + f * (0.001_333_355 + f * 0.000_154_035)))));
    p * f32::from_bits(((i as i32 + 127) as u32) << 23)
}

/// Base-2 logarithm of a positive value.
#[inline(always)]
pub fn log2(x: f32) -> f32 {
    let bits = x.to_bits();
    let mut exponent = ((bits >> 23) & 0xff) as i32 - 127;
    let mut m = f32::from_bits((bits & 0x007f_ffff) | 0x3f80_0000); // [1, 2)
    // Centred on one, the series converges within a few terms.
    if m > std::f32::consts::SQRT_2 { m *= 0.5; exponent += 1; }
    // log2(m) through the arctanh series in t = (m - 1) / (m + 1).
    let t = (m - 1.0) / (m + 1.0);
    let t2 = t * t;
    let series = t * (2.885_390_1 + t2 * (0.961_796_7 + t2 * (0.577_078_0 + t2 * (0.412_198_6 + t2 * 0.320_598_9))));
    exponent as f32 + series
}

/// e to the power `x`.
#[inline(always)]
pub fn exp(x: f32) -> f32 { exp2(x * std::f32::consts::LOG2_E) }

/// `a` to the power `b`, for positive `a`.
#[inline(always)]
pub fn powf(a: f32, b: f32) -> f32 { exp2(b * log2(a)) }

/// Cube root, of any sign.
#[inline(always)]
pub fn cbrt(x: f32) -> f32 {
    if x == 0.0 { return 0.0; }
    let a = x.abs();
    // A first guess from the exponent bits, then two Newton steps.
    let mut y = f32::from_bits(a.to_bits() / 3 + 0x2a51_7d3c);
    y = y - (y * y * y - a) / (3.0 * y * y);
    y = y - (y * y * y - a) / (3.0 * y * y);
    y = y - (y * y * y - a) / (3.0 * y * y);
    if x < 0.0 { -y } else { y }
}

/// The angle of (x, y) in degrees, 0 to 360.
#[inline(always)]
pub fn hue(a: f32, b: f32) -> f32 {
    let (ax, ay) = (a.abs(), b.abs());
    if ax == 0.0 && ay == 0.0 { return 0.0; }
    let (small, large) = if ax > ay { (ay, ax) } else { (ax, ay) };
    let t = small / large;
    let t2 = t * t;
    // atan on [0, 1], error under 1e-5 radians.
    let mut r = t * (0.999_866_0 + t2 * (-0.330_299_5 + t2 * (0.180_141_0 + t2 * (-0.085_133_0 + t2 * 0.020_835_1))));
    if ay > ax { r = std::f32::consts::FRAC_PI_2 - r; }
    if a < 0.0 { r = std::f32::consts::PI - r; }
    if b < 0.0 { r = -r; }
    let degrees = r.to_degrees();
    if degrees < 0.0 { degrees + 360.0 } else { degrees }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fast_forms_track_the_standard_ones() {
        for i in -400..400 { let x = i as f32 / 37.0; assert!((exp2(x) / x.exp2() - 1.0).abs() < 2e-6, "exp2 {x}"); }
        for i in 1..4000 { let x = i as f32 / 97.0; assert!((log2(x) - x.log2()).abs() < 2e-6, "log2 {x}"); }
        for i in -2000..2000 { let x = i as f32 / 113.0; assert!((cbrt(x) - x.cbrt()).abs() <= 1e-6 * x.abs().max(1.0), "cbrt {x}"); }
        for i in 0..720 {
            let t = (i as f32 * 0.5).to_radians();
            let (a, b) = (t.cos() * 0.1, t.sin() * 0.1);
            let exact = (b.atan2(a).to_degrees() + 360.0) % 360.0;
            let d = (hue(a, b) - exact + 540.0) % 360.0 - 180.0;
            assert!(d.abs() < 1e-3, "hue {i}");
        }
        for i in 0..100 { let x = i as f32 + 0.49; assert_eq!(round(x), (x as f64).round() as usize); }
    }
}
