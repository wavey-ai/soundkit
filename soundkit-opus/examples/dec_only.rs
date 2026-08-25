use soundkit_opus::{Application, Decoder, Encoder};
use std::hint::black_box;

const SAMPLE_RATE: usize = 48000;
const CHANNELS: usize = 2;

fn fixture(seconds: usize) -> Vec<f32> {
    let n = seconds * SAMPLE_RATE * CHANNELS;
    (0..n)
        .map(|i| {
            let t = (i / CHANNELS) as f32 / SAMPLE_RATE as f32;
            0.4 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
                + 0.25 * (2.0 * std::f32::consts::PI * 660.0 * t).sin()
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let frame_size: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(960);
    let bitrate: i32 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(512000);
    let repeats: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);

    let pcm = fixture(4);
    let frame_ms = frame_size * 1000 / SAMPLE_RATE;
    let _bytes_per_frame = bitrate as usize / 8 * frame_ms / 1000;

    let mut enc = Encoder::with_application(SAMPLE_RATE as i32, CHANNELS, Application::Audio)?;
    enc.set_bitrate(bitrate)?;
    let mut packets = Vec::new();
    for chunk in pcm.chunks(frame_size * CHANNELS) {
        if chunk.len() == frame_size * CHANNELS {
            packets.push(enc.encode_f32_vec(black_box(chunk), frame_size)?);
        }
    }

    {
        let mut dec = Decoder::new(SAMPLE_RATE as i32, CHANNELS)?;
        let mut out = vec![0f32; frame_size * CHANNELS];
        for p in packets.iter().take(50) {
            dec.decode_f32_into(p, false, &mut out)?;
        }
    }

    let mut checksum = 0f64;
    let start = std::time::Instant::now();
    for _ in 0..repeats {
        let mut dec = Decoder::new(SAMPLE_RATE as i32, CHANNELS)?;
        let mut out = vec![0f32; frame_size * CHANNELS];
        for p in &packets {
            dec.decode_f32_into(black_box(p), false, &mut out)?;
            checksum += out[0] as f64 + out[out.len() / 2] as f64;
        }
    }
    let elapsed = start.elapsed();
    let audio = packets.len() as f64 * frame_size as f64 / SAMPLE_RATE as f64 * repeats as f64;
    println!(
        "frame_ms={frame_ms} bitrate={bitrate} decode_realtime={:.2}x checksum={checksum:.3}",
        audio / elapsed.as_secs_f64()
    );
    Ok(())
}
