//! Point curves: a monotone cubic through the points a person places, so the
//! curve never overshoots between two points and never turns back.

/// Entries in a curve table, over the encoded range of zero to one.
pub const CURVE_TABLE: usize = 4096;

/// The curve through `points` (x, y in 0 to 1, x increasing) as a table of
/// `CURVE_TABLE + 1` entries. Fewer than two points is the identity. Outside
/// the first and last points the curve holds their values.
pub fn table(points: &[[f32; 2]]) -> Option<Vec<f32>> {
    if points.len() < 2 || is_identity(points) { return None; }
    let n = points.len();
    let (x, y): (Vec<f32>, Vec<f32>) = points.iter().map(|p| (p[0], p[1])).unzip();
    // Secant slopes, then Fritsch-Carlson tangents.
    let d: Vec<f32> = (0..n - 1).map(|i| (y[i + 1] - y[i]) / (x[i + 1] - x[i]).max(1e-6)).collect();
    let mut m = vec![0.0f32; n];
    m[0] = d[0];
    m[n - 1] = d[n - 2];
    for i in 1..n - 1 { m[i] = if d[i - 1] * d[i] <= 0.0 { 0.0 } else { (d[i - 1] + d[i]) / 2.0 }; }
    for i in 0..n - 1 {
        if d[i] == 0.0 { m[i] = 0.0; m[i + 1] = 0.0; continue; }
        let (a, b) = (m[i] / d[i], m[i + 1] / d[i]);
        let s = a * a + b * b;
        if s > 9.0 { let t = 3.0 / s.sqrt(); m[i] = t * a * d[i]; m[i + 1] = t * b * d[i]; }
    }
    Some((0..=CURVE_TABLE).map(|k| {
        let v = k as f32 / CURVE_TABLE as f32;
        if v <= x[0] { return y[0].clamp(0.0, 1.0); }
        if v >= x[n - 1] { return y[n - 1].clamp(0.0, 1.0); }
        let i = x.windows(2).position(|w| v <= w[1]).unwrap_or(n - 2);
        let h = (x[i + 1] - x[i]).max(1e-6);
        let t = (v - x[i]) / h;
        let (t2, t3) = (t * t, t * t * t);
        let out = (2.0 * t3 - 3.0 * t2 + 1.0) * y[i] + (t3 - 2.0 * t2 + t) * h * m[i]
            + (-2.0 * t3 + 3.0 * t2) * y[i + 1] + (t3 - t2) * h * m[i + 1];
        out.clamp(0.0, 1.0)
    }).collect())
}

/// Points on the diagonal, from corner to corner, change nothing.
fn is_identity(points: &[[f32; 2]]) -> bool {
    points.iter().all(|p| (p[0] - p[1]).abs() < 1e-6)
        && points.first().is_some_and(|p| p[0] <= 1e-6) && points.last().is_some_and(|p| p[0] >= 1.0 - 1e-6)
}

/// Reads a table at an encoded value, interpolating between entries.
#[inline(always)]
pub fn read(table: &[f32], v: f32) -> f32 {
    let p = v.clamp(0.0, 1.0) * CURVE_TABLE as f32;
    let k = (p as usize).min(CURVE_TABLE - 1);
    table[k] + (table[k + 1] - table[k]) * (p - k as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn a_curve_passes_through_its_points_and_stays_monotone() {
        let points = [[0.0, 0.0], [0.25, 0.15], [0.5, 0.55], [0.75, 0.9], [1.0, 1.0]];
        let t = table(&points).unwrap();
        for p in points { assert!((read(&t, p[0]) - p[1]).abs() < 1e-4, "{p:?}"); }
        for w in t.windows(2) { assert!(w[1] >= w[0] - 1e-6); }
    }
    #[test]
    fn a_steep_step_does_not_overshoot() {
        let t = table(&[[0.0, 0.0], [0.45, 0.02], [0.55, 0.98], [1.0, 1.0]]).unwrap();
        assert!(t.iter().all(|v| (0.0..=1.0).contains(v)));
        for w in t.windows(2) { assert!(w[1] >= w[0] - 1e-6); }
    }
    #[test]
    fn the_diagonal_is_no_curve() {
        assert!(table(&[[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]).is_none());
        assert!(table(&[[0.0, 0.0]]).is_none());
        // Moved endpoints are a curve: they set the black and white points.
        let t = table(&[[0.0, 0.1], [1.0, 0.9]]).unwrap();
        assert!((read(&t, 0.0) - 0.1).abs() < 1e-6 && (read(&t, 1.0) - 0.9).abs() < 1e-6);
    }
}
