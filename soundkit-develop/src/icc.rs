//! ICC profiles of the matrix kind: three colorants and one tone curve.
//! sRGB, Display P3, Adobe RGB and ProPhoto RGB profiles are of this kind.

use crate::colour::{self, Matrix, IDENTITY};

/// A tone curve from an encoded value to linear light.
enum Tone {
    Gamma(f32),
    /// The parametric curve: `(a * x + b) ^ g + e` from `d` up, and `c * x + f` below it.
    Parametric { g: f32, a: f32, b: f32, c: f32, d: f32, e: f32, f: f32 },
    Table(Vec<f32>),
}

pub struct Profile {
    /// From the profile's linear RGB to linear sRGB.
    pub matrix: Matrix,
    tone: Tone,
}

fn u32_at(bytes: &[u8], at: usize) -> Option<u32> { Some(u32::from_be_bytes(bytes.get(at..at + 4)?.try_into().ok()?)) }
fn fixed_at(bytes: &[u8], at: usize) -> Option<f64> { Some(u32_at(bytes, at)? as i32 as f64 / 65536.0) }
/// The data of the tag with this signature.
fn tag<'a>(icc: &'a [u8], signature: &[u8; 4]) -> Option<&'a [u8]> {
    let count = u32_at(icc, 128)? as usize;
    (0..count.min(256)).find_map(|i| {
        let entry = 132 + i * 12;
        if icc.get(entry..entry + 4)? != signature { return None; }
        let offset = u32_at(icc, entry + 4)? as usize;
        icc.get(offset..offset.checked_add(u32_at(icc, entry + 8)? as usize)?)
    })
}
fn colorant(icc: &[u8], signature: &[u8; 4]) -> Option<[f64; 3]> {
    let data = tag(icc, signature)?;
    if data.get(..4)? != b"XYZ " { return None; }
    Some([fixed_at(data, 8)?, fixed_at(data, 12)?, fixed_at(data, 16)?])
}
fn tone(data: &[u8]) -> Option<Tone> {
    match data.get(..4)? {
        b"curv" => {
            let count = u32_at(data, 8)? as usize;
            let entry = |i: usize| Some(u16::from_be_bytes(data.get(12 + i * 2..14 + i * 2)?.try_into().ok()?));
            match count {
                0 => Some(Tone::Gamma(1.0)),
                1 => Some(Tone::Gamma(entry(0)? as f32 / 256.0)),
                _ => Some(Tone::Table((0..count.min(65536)).map(|i| entry(i).map(|v| v as f32 / 65535.0)).collect::<Option<Vec<f32>>>()?)),
            }
        }
        b"para" => {
            let kind = u16::from_be_bytes(data.get(8..10)?.try_into().ok()?);
            let p = |i: usize| fixed_at(data, 12 + i * 4).map(|v| v as f32);
            let g = p(0)?;
            Some(match kind {
                0 => Tone::Gamma(g),
                1 => { let (a, b) = (p(1)?, p(2)?); Tone::Parametric { g, a, b, c: 0.0, d: if a != 0.0 { -b / a } else { 0.0 }, e: 0.0, f: 0.0 } }
                2 => { let (a, b, c) = (p(1)?, p(2)?, p(3)?); Tone::Parametric { g, a, b, c: 0.0, d: if a != 0.0 { -b / a } else { 0.0 }, e: c, f: c } }
                3 => Tone::Parametric { g, a: p(1)?, b: p(2)?, c: p(3)?, d: p(4)?, e: 0.0, f: 0.0 },
                4 => Tone::Parametric { g, a: p(1)?, b: p(2)?, c: p(3)?, d: p(4)?, e: p(5)?, f: p(6)? },
                _ => return None,
            })
        }
        _ => None,
    }
}

/// Reads an RGB matrix profile. Returns nothing for a profile of another
/// kind: one with lookup tables, one for another colour model, or one with a
/// different curve for each channel.
pub fn parse(icc: &[u8]) -> Option<Profile> {
    if icc.get(36..40)? != b"acsp" || icc.get(16..20)? != b"RGB " || icc.get(20..24)? != b"XYZ " { return None; }
    let (r, g, b) = (colorant(icc, b"rXYZ")?, colorant(icc, b"gXYZ")?, colorant(icc, b"bXYZ")?);
    let curve = tag(icc, b"rTRC")?;
    if tag(icc, b"gTRC")? != curve || tag(icc, b"bTRC")? != curve { return None; }
    let tone = tone(curve)?;
    // The colorants are relative to the profile's white, D50. Their sum is
    // that white, and the adaptation takes it to the white of sRGB.
    let to_xyz: Matrix = [r[0], g[0], b[0], r[1], g[1], b[1], r[2], g[2], b[2]];
    let white = colour::apply(&to_xyz, [1.0; 3]);
    if white[1] <= 0.0 { return None; }
    let adapt = colour::adaptation_xyz(white, colour::apply(&colour::RGB_TO_XYZ, [1.0; 3]));
    let matrix = colour::multiply(&colour::invert(&colour::RGB_TO_XYZ)?, &colour::multiply(&adapt, &to_xyz));
    Some(Profile { matrix, tone })
}

impl Profile {
    /// An encoded value from zero to one, as linear light.
    pub fn to_linear(&self, v: f32) -> f32 {
        match &self.tone {
            Tone::Gamma(g) => v.max(0.0).powf(*g),
            Tone::Parametric { g, a, b, c, d, e, f } => if v >= *d { (a * v + b).max(0.0).powf(*g) + e } else { c * v + f },
            Tone::Table(values) => {
                let position = v.clamp(0.0, 1.0) * (values.len() - 1) as f32;
                let k = (position as usize).min(values.len() - 2);
                values[k] + (values[k + 1] - values[k]) * (position - k as f32)
            }
        }
    }
    /// True when the profile's colours and curve are those of sRGB, within
    /// the rounding of a profile's 16-bit numbers.
    pub fn is_srgb(&self) -> bool {
        let srgb = |v: f32| if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) };
        self.matrix.iter().zip(IDENTITY).all(|(a, b)| (a - b).abs() < 2e-3)
            && (0..=64).all(|i| { let v = i as f32 / 64.0; (self.to_linear(v) - srgb(v)).abs() < 2e-3 })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A matrix profile from xy chromaticities and a curve tag, as a camera or an editor writes one.
    pub(crate) fn profile(primaries: [[f64; 2]; 3], white: [f64; 2], curve: &[u8]) -> Vec<u8> {
        let xyz = |[x, y]: [f64; 2]| [x / y, 1.0, (1.0 - x - y) / y];
        let (r, g, b, w) = (xyz(primaries[0]), xyz(primaries[1]), xyz(primaries[2]), xyz(white));
        let basis: Matrix = [r[0], g[0], b[0], r[1], g[1], b[1], r[2], g[2], b[2]];
        let scale = colour::apply(&colour::invert(&basis).unwrap(), w);
        // A profile stores its colorants adapted to D50.
        let adapt = colour::adaptation_xyz(w, [0.9642, 1.0, 0.8249]);
        let mut tags: Vec<(&[u8; 4], Vec<u8>)> = Vec::new();
        for (i, name) in [b"rXYZ", b"gXYZ", b"bXYZ"].into_iter().enumerate() {
            let c = colour::apply(&adapt, [basis[i] * scale[i], basis[3 + i] * scale[i], basis[6 + i] * scale[i]]);
            let mut data = b"XYZ \0\0\0\0".to_vec();
            for v in c { data.extend(((v * 65536.0).round() as i32).to_be_bytes()); }
            tags.push((name, data));
        }
        for name in [b"rTRC", b"gTRC", b"bTRC"] { tags.push((name, curve.to_vec())); }
        let mut icc = vec![0u8; 132 + tags.len() * 12];
        icc[16..20].copy_from_slice(b"RGB "); icc[20..24].copy_from_slice(b"XYZ "); icc[36..40].copy_from_slice(b"acsp");
        icc[128..132].copy_from_slice(&(tags.len() as u32).to_be_bytes());
        for (i, (name, data)) in tags.iter().enumerate() {
            let offset = icc.len() as u32;
            icc[132 + i * 12..136 + i * 12].copy_from_slice(*name);
            icc[136 + i * 12..140 + i * 12].copy_from_slice(&offset.to_be_bytes());
            icc[140 + i * 12..144 + i * 12].copy_from_slice(&(data.len() as u32).to_be_bytes());
            icc.extend(data);
        }
        let size = icc.len() as u32;
        icc[0..4].copy_from_slice(&size.to_be_bytes());
        icc
    }
    const D65: [f64; 2] = [0.3127, 0.3290];
    const SRGB: [[f64; 2]; 3] = [[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]];
    const P3: [[f64; 2]; 3] = [[0.680, 0.320], [0.265, 0.690], [0.150, 0.060]];
    const ADOBE: [[f64; 2]; 3] = [[0.64, 0.33], [0.21, 0.71], [0.15, 0.06]];
    /// The sRGB curve as a parametric tag of type 3.
    pub(crate) fn srgb_curve() -> Vec<u8> {
        let mut data = b"para\0\0\0\0\0\x03\0\0".to_vec();
        for v in [2.4, 1.0 / 1.055, 0.055 / 1.055, 1.0 / 12.92, 0.04045] { data.extend(((v * 65536.0f64).round() as i32).to_be_bytes()); }
        data
    }
    /// A gamma curve as a `curv` tag with one entry.
    pub(crate) fn gamma_curve(gamma: f64) -> Vec<u8> {
        let mut data = b"curv\0\0\0\0\0\0\0\x01".to_vec();
        data.extend(((gamma * 256.0).round() as u16).to_be_bytes());
        data
    }
    pub(crate) fn display_p3() -> Vec<u8> { profile(P3, D65, &srgb_curve()) }
    pub(crate) fn adobe_rgb() -> Vec<u8> { profile(ADOBE, D65, &gamma_curve(2.19921875)) }

    #[test]
    fn an_srgb_profile_is_recognized() {
        let parsed = parse(&profile(SRGB, D65, &srgb_curve())).unwrap();
        assert!(parsed.is_srgb(), "{:?}", parsed.matrix);
        // The same curve as a table, as the common sRGB profile stores it.
        let mut table = b"curv\0\0\0\0".to_vec();
        table.extend(1024u32.to_be_bytes());
        for i in 0..1024 { let v = i as f32 / 1023.0; let l = if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }; table.extend(((l * 65535.0).round() as u16).to_be_bytes()); }
        assert!(parse(&profile(SRGB, D65, &table)).unwrap().is_srgb());
    }
    #[test]
    fn display_p3_and_adobe_rgb_map_to_linear_srgb() {
        let p3 = parse(&display_p3()).unwrap();
        assert!(!p3.is_srgb());
        // The published Display P3 to sRGB matrix.
        let expected = [1.2249, -0.2247, 0.0, -0.0420, 1.0419, 0.0, -0.0197, -0.0786, 1.0979];
        for (a, b) in p3.matrix.iter().zip(expected) { assert!((a - b).abs() < 2e-3, "{:?}", p3.matrix); }
        // White stays white.
        for row in 0..3 { assert!((p3.matrix[row * 3] + p3.matrix[row * 3 + 1] + p3.matrix[row * 3 + 2] - 1.0).abs() < 1e-6); }
        let adobe = parse(&adobe_rgb()).unwrap();
        assert!((adobe.to_linear(0.5) - 0.5f32.powf(2.19921875)).abs() < 1e-6);
        // Adobe RGB green is outside sRGB: its red is negative there.
        assert!(adobe.matrix[1] < -0.3 && (adobe.matrix[0] - 1.3984).abs() < 2e-3, "{:?}", adobe.matrix);
    }
    #[test]
    fn other_profiles_are_refused() {
        assert!(parse(b"not a profile").is_none());
        let mut cmyk = profile(SRGB, D65, &srgb_curve());
        cmyk[16..20].copy_from_slice(b"CMYK");
        assert!(parse(&cmyk).is_none());
        // A different curve for one channel.
        let mut mixed = profile(SRGB, D65, &srgb_curve());
        let blue = u32::from_be_bytes(mixed[132 + 5 * 12 + 4..132 + 5 * 12 + 8].try_into().unwrap()) as usize;
        mixed[blue + 13] ^= 1;
        assert!(parse(&mixed).is_none());
        let truncated = profile(SRGB, D65, &srgb_curve());
        assert!(parse(&truncated[..150]).is_none());
    }
}
