use soundkit_develop::{develop, from_rgba, Options, Recipe};
use std::time::Instant;
fn main() {
    let (w, h) = (4032usize, 3024usize);
    let mut rgba = vec![0u8; w * h * 4];
    for y in 0..h { for x in 0..w {
        let i = (y * w + x) * 4; let v = x as f32 / w as f32; let t = y as f32 / h as f32;
        rgba[i] = (255.0 * v * (0.55 + 0.45 * (t * 6.28).sin())) as u8; rgba[i + 1] = (255.0 * v * (0.55 + 0.45 * (t * 6.28 + 2.1).sin())) as u8;
        rgba[i + 2] = (255.0 * v * (0.55 + 0.45 * (t * 6.28 + 4.2).sin())) as u8; rgba[i + 3] = 255;
    } }
    let frame = from_rgba(w, h, &rgba);
    let only = std::env::args().nth(1);
    for (name, json) in [("untouched", "{}"), ("portrait look", r#"{"highlights":-30,"shadows":25,"texture":-15,"clarity":10,"vibrance":12,"mixer":{"orange":{"luminance":14}},"grading":{"highlights":{"hue":70,"saturation":12}}}"#)] {
        if only.as_deref().is_some_and(|o| !name.starts_with(o)) { continue; }
        let recipe = Recipe::from_json(json);
        for _ in 0..3 {
        let t = Instant::now(); develop(&frame, &recipe, &Options::default());
        println!("native 12MP {name:14} {:>6} ms", t.elapsed().as_millis());
        }
    }
}
