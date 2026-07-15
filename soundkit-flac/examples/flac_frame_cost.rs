use soundkit_flac::{FlacFrameConfig, FlacFrameEncoder, FlacProfile};
use std::hint::black_box;
use std::time::{Duration, Instant};

const SAMPLE_RATE: u32 = 48_000;
const FRAME_LENGTH: u32 = 240;
const CHANNELS: u16 = 2;
const ITERATIONS: usize = 200;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("pure-Rust FLAC component cost; 48 kHz stereo S24, 240-frame blocks");
    for track_count in [1usize, 2, 8, 32] {
        let config = FlacFrameConfig::new(
            SAMPLE_RATE,
            CHANNELS,
            24,
            FRAME_LENGTH,
            FlacProfile::Realtime,
        )?;
        let mut encoders = (0..track_count)
            .map(|_| FlacFrameEncoder::new(config))
            .collect::<Result<Vec<_>, _>>()?;
        let samples = signal(config.sample_count()?);
        let started = Instant::now();
        let mut encoded_bytes = 0usize;
        for _ in 0..ITERATIONS {
            for encoder in &mut encoders {
                let frame = encoder.encode_i32(black_box(&samples))?;
                encoded_bytes = encoded_bytes.saturating_add(black_box(frame.payload.len()));
            }
        }
        let elapsed = started.elapsed();
        print_result(track_count, elapsed, encoded_bytes, config.raw_pcm_bytes()?);
    }
    Ok(())
}

fn signal(sample_count: usize) -> Vec<i32> {
    (0..sample_count)
        .map(|index| {
            let phase = index as f64 * 440.0 * std::f64::consts::TAU / SAMPLE_RATE as f64;
            (phase.sin() * 2_000_000.0) as i32
        })
        .collect()
}

fn print_result(track_count: usize, elapsed: Duration, encoded_bytes: usize, pcm_bytes: usize) {
    let frames = track_count * ITERATIONS;
    let micros_per_frame = elapsed.as_secs_f64() * 1_000_000.0 / frames as f64;
    let ratio = encoded_bytes as f64 / (frames * pcm_bytes) as f64;
    println!(
        "tracks={track_count:>2} frames={frames:>5} total_ms={:>9.3} us_per_track_frame={micros_per_frame:>9.3} encoded/pcm={ratio:.3}",
        elapsed.as_secs_f64() * 1_000.0
    );
}
