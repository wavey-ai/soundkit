//! Encodes an s32le stereo source and prints the transient counters.
//! cargo run --release --features trace,tuning --example trace_encode -- <src> <bitrate> [knob=value,...]
use soundkit_opus::{trace, tuning, Application, Encoder};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let raw = std::fs::read(&a[1]).unwrap();
    let pcm: Vec<i32> = raw.chunks_exact(4).map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect();
    for pair in a.get(3).map(String::as_str).unwrap_or("").split(',').filter(|s| !s.is_empty()) {
        let (k, v) = pair.split_once('=').unwrap();
        tuning::set(k.parse().unwrap(), v.parse().unwrap());
    }
    let mut enc = Encoder::with_application(48_000, 2, Application::Audio).unwrap();
    enc.set_bitrate(a[2].parse().unwrap()).unwrap();
    let mut packet = Vec::new();
    for frame in pcm.chunks_exact(480) {
        enc.encode_i24_into(frame, 240, &mut packet).unwrap();
    }
    let c = trace::take_counters();
    println!("frames {} analysis transients {} ({:.1}%) patched {} ({:.1}%)", c[0], c[1], 100.0 * c[1] as f64 / c[0] as f64, c[2], 100.0 * c[2] as f64 / c[0] as f64);
}
