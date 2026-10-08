//! Spatial filters on one plane of floats: box blurs, a Gaussian made of three
//! box blurs, and the guided filter (He, Sun and Tang) that smooths a plane
//! while it keeps the plane's own edges.
//!
//! Every filter holds the edge value past the border, so a filter over a
//! window of the picture matches the filter over the whole picture wherever
//! the window extends past the filter's reach. The running sums are kept in
//! f64 for the same reason: the order of the sums does not change the result.

/// One box blur of `radius` pixels, across rows and then down columns.
pub fn box_blur(source: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
    let mut a = source.to_vec();
    if radius == 0 || width == 0 || height == 0 { return a; }
    let (w, h, r) = (width, height, radius);
    let size = (2 * r + 1) as f64;
    let mut b = vec![0.0f32; source.len()];
    for y in 0..h {
        let row = &a[y * w..(y + 1) * w];
        let out = &mut b[y * w..(y + 1) * w];
        let mut sum = 0.0f64;
        for i in 0..=2 * r { sum += row[i.saturating_sub(r).min(w - 1)] as f64; }
        for x in 0..w {
            out[x] = (sum / size) as f32;
            sum += row[(x + r + 1).min(w - 1)] as f64 - row[x.saturating_sub(r)] as f64;
        }
    }
    let mut sums = vec![0.0f64; w];
    for i in 0..=2 * r {
        let yy = i.saturating_sub(r).min(h - 1);
        for (s, v) in sums.iter_mut().zip(&b[yy * w..(yy + 1) * w]) { *s += *v as f64; }
    }
    for y in 0..h {
        let (add, sub) = ((y + r + 1).min(h - 1), y.saturating_sub(r));
        let out = &mut a[y * w..(y + 1) * w];
        for x in 0..w { out[x] = (sums[x] / size) as f32; }
        for x in 0..w { sums[x] += b[add * w + x] as f64 - b[sub * w + x] as f64; }
    }
    a
}

/// Three box blurs of `radius`: the blur the sharpening radius has always used.
pub fn triple_box(source: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
    let mut a = source.to_vec();
    for _ in 0..3 { a = box_blur(&a, width, height, radius); }
    a
}

/// The radii of three box blurs whose sum approximates a Gaussian of `sigma`
/// (Kovesi's widths). A sigma below a third of a pixel is no blur.
pub fn gaussian_radii(sigma: f32) -> [usize; 3] {
    if sigma <= 0.33 { return [0; 3]; }
    let s2 = 12.0 * (sigma as f64) * (sigma as f64);
    let ideal = (s2 / 3.0 + 1.0).sqrt();
    let mut lower = ideal.floor() as i64;
    if lower % 2 == 0 { lower -= 1; }
    let lower = lower.max(1);
    let upper = lower + 2;
    let m = ((s2 - 3.0 * (lower * lower) as f64 - 12.0 * lower as f64 - 9.0) / (-4.0 * lower as f64 - 4.0)).round() as i64;
    std::array::from_fn(|i| (if (i as i64) < m { lower } else { upper } as usize - 1) / 2)
}

/// A Gaussian of `sigma` pixels, as three box blurs.
pub fn gaussian(source: &[f32], width: usize, height: usize, sigma: f32) -> Vec<f32> {
    let mut a = source.to_vec();
    for r in gaussian_radii(sigma) { a = box_blur(&a, width, height, r); }
    a
}

/// How far a Gaussian of `sigma` reaches, in pixels.
pub fn gaussian_reach(sigma: f32) -> usize { gaussian_radii(sigma).iter().sum() }

/// The coefficients of the self-guided filter of `p`: the output at a pixel
/// is `a * p + b`, with `a` and `b` the returned planes. Gaussian windows of
/// `sigma` pixels; `eps` is the variance below which detail is smoothed.
pub fn guided_coefficients(p: &[f32], width: usize, height: usize, sigma: f32, eps: f32) -> (Vec<f32>, Vec<f32>) {
    let mean = gaussian(p, width, height, sigma);
    let squares: Vec<f32> = p.iter().map(|v| v * v).collect();
    let correlation = gaussian(&squares, width, height, sigma);
    let mut a = vec![0.0f32; p.len()];
    let mut b = vec![0.0f32; p.len()];
    for i in 0..p.len() {
        let variance = (correlation[i] - mean[i] * mean[i]).max(0.0);
        a[i] = variance / (variance + eps);
        b[i] = mean[i] - a[i] * mean[i];
    }
    (gaussian(&a, width, height, sigma), gaussian(&b, width, height, sigma))
}

/// How far the guided filter of `sigma` reaches: two Gaussians.
pub fn guided_reach(sigma: f32) -> usize { 2 * gaussian_reach(sigma) }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gaussian_radii_match_the_variance() {
        for sigma in [0.5f32, 1.0, 2.5, 7.0, 21.0, 90.0] {
            let variance: f64 = gaussian_radii(sigma).iter().map(|&r| { let w = (2 * r + 1) as f64; (w * w - 1.0) / 12.0 }).sum();
            assert!((variance.sqrt() - sigma as f64).abs() < 0.5 + sigma as f64 * 0.05, "{sigma}: {}", variance.sqrt());
        }
    }
    #[test]
    fn a_window_matches_the_whole() {
        let (w, h) = (90, 60);
        let plane: Vec<f32> = (0..w * h).map(|i| (((i % w) as f32 * 0.37).sin() + (i / w) as f32 * 0.05).abs()).collect();
        let whole = guided_coefficients(&plane, w, h, 3.0, 0.1).0;
        // A window that reaches well past the filter, against the image's own left and top edges.
        let (x0, y0, x1, y1) = (0, 0, 70, 45);
        let ww = x1 - x0;
        let part: Vec<f32> = (y0..y1).flat_map(|y| (x0..x1).map(move |x| (x, y))).map(|(x, y)| plane[y * w + x]).collect();
        let windowed = guided_coefficients(&part, ww, y1 - y0, 3.0, 0.1).0;
        let reach = guided_reach(3.0);
        for y in 0..y1 - reach { for x in 0..x1 - reach {
            assert!((windowed[y * ww + x] - whole[y * w + x]).abs() < 1e-6, "{x},{y}");
        } }
    }
    #[test]
    fn the_guided_filter_keeps_a_step_and_flattens_texture() {
        let (w, h) = (64, 8);
        let plane: Vec<f32> = (0..w * h).map(|i| { let x = i % w; (if x < 32 { -2.0 } else { 2.0 }) + if x % 2 == 0 { 0.05 } else { -0.05 } }).collect();
        let (a, b) = guided_coefficients(&plane, w, h, 3.0, 0.35);
        let q: Vec<f32> = (0..w * h).map(|i| a[i] * plane[i] + b[i]).collect();
        // Texture far from the edge is gone; the step is kept.
        assert!((q[4 * w + 10] - q[4 * w + 11]).abs() < 0.02);
        assert!(q[4 * w + 40] - q[4 * w + 20] > 3.5);
    }
}
