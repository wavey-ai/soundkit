//! Decodes a packet file (little-endian u16 length, then the packet) with
//! the `trace` feature and prints the mean per-frame side information.
//!
//! cargo run --release --features trace --example trace_packets -- <packets.bin>...
use soundkit_opus::{trace, Decoder};

fn main() {
    for path in std::env::args().skip(1) {
        let data = std::fs::read(&path).expect("read packets");
        let mut decoder = Decoder::new(48_000, 2).expect("decoder");
        let mut pcm = vec![0i32; 2 * 960];
        let mut pos = 0usize;
        while pos + 2 <= data.len() {
            let len = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
            pos += 2;
            decoder.decode_i24_into_slice(&data[pos..pos + len], false, &mut pcm).expect("decode");
            pos += len;
        }
        let frames = trace::take();
        if std::env::var_os("TRACE_FRAMES").is_some() {
            for (i, f) in frames.iter().enumerate() {
                if f.transient {
                    eprintln!("T {i}");
                }
            }
        }
        let n = frames.len() as f64;
        let mean = |f: &dyn Fn(&trace::FrameTrace) -> f64| frames.iter().map(f).sum::<f64>() / n;
        let mut trims = [0usize; 11];
        let mut spreads = [0usize; 4];
        for f in &frames {
            trims[f.alloc_trim as usize] += 1;
            spreads[f.spread as usize] += 1;
        }
        println!("{path}: {} frames", frames.len());
        println!(
            "  prefilter on {:.1}% mean qgain {:.2} | transient {:.1}% | tf changes {:.2} | dual {:.1}% | intensity {:.2} | coded {:.2}",
            100.0 * mean(&|f| f64::from(u8::from(f.prefilter_gain >= 0))),
            mean(&|f| f.prefilter_gain.max(0) as f64),
            100.0 * mean(&|f| f64::from(u8::from(f.transient))),
            mean(&|f| f.tf_changes as f64),
            100.0 * mean(&|f| f64::from(u8::from(f.dual_stereo))),
            mean(&|f| f.intensity as f64),
            mean(&|f| f.coded_bands as f64),
        );
        println!(
            "  boost {:.1} | fine bits {:.2} | pulse bits/8 {:.1} | trim {:?} | spread {:?}",
            mean(&|f| f.total_boost as f64 / 8.0),
            mean(&|f| f.fine_bits as f64),
            mean(&|f| f.pulse_bits as f64 / 8.0),
            trims,
            spreads
        );
    }
}
