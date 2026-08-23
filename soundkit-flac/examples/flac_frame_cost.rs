//! Packet-level FLAC cost for SoundKit's 5 ms streaming geometry.

use soundkit_flac::{FlacFrameConfig, FlacFrameDecoder, FlacFrameEncoder, FlacProfile};
use std::env;
use std::fs;
use std::hint::black_box;
use std::path::Path;
use std::time::{Duration, Instant};

const CHANNELS: u16 = 2;
const BITS_PER_SAMPLE: u8 = 24;
const DEFAULT_ITERATIONS: usize = 20_000;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("pure-Rust packet FLAC; stereo S24, 5 ms frames");
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    if !arguments.is_empty() {
        if !(3..=5).contains(&arguments.len()) {
            return Err(
                "usage: flac_frame_cost [48000|96000 realtime|balanced ITERATIONS [PCM_S32LE [PACKET]]]"
                    .into(),
            );
        }
        let sample_rate = arguments[0].parse()?;
        let profile = match arguments[1].as_str() {
            "realtime" => FlacProfile::Realtime,
            "balanced" => FlacProfile::Balanced,
            _ => return Err("profile must be realtime or balanced".into()),
        };
        let iterations = arguments[2].parse()?;
        if !matches!(sample_rate, 48_000 | 96_000) || iterations == 0 {
            return Err("sample rate must be 48000 or 96000 and iterations must be nonzero".into());
        }
        return run_case(
            sample_rate,
            profile,
            iterations,
            arguments.get(3).map(Path::new),
            arguments.get(4).map(Path::new),
        );
    }
    for sample_rate in [48_000_u32, 96_000] {
        for profile in [FlacProfile::Realtime, FlacProfile::Balanced] {
            run_case(sample_rate, profile, DEFAULT_ITERATIONS, None, None)?;
        }
    }
    Ok(())
}

fn run_case(
    sample_rate: u32,
    profile: FlacProfile,
    iterations: usize,
    pcm_path: Option<&Path>,
    decode_packet_path: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let frame_length = sample_rate / 200;
    let config = FlacFrameConfig::new(
        sample_rate,
        CHANNELS,
        BITS_PER_SAMPLE,
        frame_length,
        profile,
    )?;
    let samples = match pcm_path {
        Some(path) => read_s32le(path, config.sample_count()?)?,
        None => signal(sample_rate, config.sample_count()?),
    };
    let mut encoder = FlacFrameEncoder::new(config)?;
    let mut packet = Vec::with_capacity(config.raw_pcm_bytes()? + 64);

    for _ in 0..256 {
        black_box(encoder.encode_i32_into(black_box(&samples), &mut packet)?);
    }

    let started = Instant::now();
    let mut encoded_bytes = 0usize;
    for _ in 0..iterations {
        let written = encoder.encode_i32_into(black_box(&samples), &mut packet)?;
        encoded_bytes = encoded_bytes.saturating_add(black_box(written));
    }
    let encode_elapsed = started.elapsed();

    if let Some(path) = decode_packet_path {
        packet = fs::read(path)?;
    }

    let mut decoder = FlacFrameDecoder::new(config)?;
    let mut decoded = vec![0_i32; config.sample_count()?];
    for _ in 0..256 {
        black_box(decoder.decode_into(black_box(&packet), &mut decoded)?);
    }
    if decoded != samples {
        return Err("packet decoder produced different PCM".into());
    }
    let started = Instant::now();
    let mut decoded_samples = 0usize;
    for _ in 0..iterations {
        decoded_samples = decoded_samples.saturating_add(black_box(
            decoder.decode_into(black_box(&packet), &mut decoded)?,
        ));
    }
    let decode_elapsed = started.elapsed();
    print_result(
        sample_rate,
        frame_length,
        profile,
        encode_elapsed,
        decode_elapsed,
        encoded_bytes,
        decoded_samples,
        config.raw_pcm_bytes()?,
        iterations,
    );
    Ok(())
}

fn read_s32le(path: &Path, sample_count: usize) -> Result<Vec<i32>, Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    if bytes.len() != sample_count * 4 {
        return Err(format!(
            "{} contains {} bytes, expected {}",
            path.display(),
            bytes.len(),
            sample_count * 4
        )
        .into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|sample| i32::from_le_bytes(sample.try_into().unwrap()))
        .collect())
}

fn signal(sample_rate: u32, sample_count: usize) -> Vec<i32> {
    (0..sample_count)
        .map(|index| {
            let phase = index as f64 * 440.0 * std::f64::consts::TAU / sample_rate as f64;
            (phase.sin() * 2_000_000.0) as i32
        })
        .collect()
}

fn print_result(
    sample_rate: u32,
    frame_length: u32,
    profile: FlacProfile,
    encode_elapsed: Duration,
    decode_elapsed: Duration,
    encoded_bytes: usize,
    decoded_samples: usize,
    pcm_bytes: usize,
    iterations: usize,
) {
    let encode_micros = encode_elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64;
    let decode_micros = decode_elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64;
    let ratio = encoded_bytes as f64 / (iterations * pcm_bytes) as f64;
    println!(
        "rate={sample_rate} frame={frame_length} profile={profile:?} encode_us={encode_micros:.3} decode_us={decode_micros:.3} encoded/pcm={ratio:.3} decoded_samples={decoded_samples} frames={iterations}"
    );
}
