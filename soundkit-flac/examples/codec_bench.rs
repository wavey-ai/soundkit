//! In-process codec timing for decode and encode paths.
//!
//! Usage:
//!
//! ```text
//! cargo run --release --example codec_bench -- decode FILE.flac [RUNS]
//! cargo run --release --example codec_bench -- encode FILE.wav balanced|max [RUNS]
//! ```

#![allow(clippy::manual_is_multiple_of)] // Keep the crate's Rust 1.65 MSRV.

use hound::{SampleFormat, WavReader};
use std::env;
use std::error::Error;
use std::fs;
use std::time::{Duration, Instant};
use soundkit_flac::frame::{FlacFrameConfig, FlacProfile};
use soundkit_flac::stream::{Decoder, Encoder};

const INPUT_CHUNK_BYTES: usize = 1024 * 1024;
const OUTPUT_CHUNK_SAMPLES: usize = 256 * 1024;
const BLOCK_SIZE: u32 = 4_096;

type AnyError = Box<dyn Error>;

fn median(samples: &[Duration]) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let middle = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[middle - 1].as_secs_f64() + sorted[middle].as_secs_f64()) / 2.0
    } else {
        sorted[middle].as_secs_f64()
    }
}

fn report(label: &str, runs: usize, warmup: Duration, timings: &[Duration]) {
    eprintln!(
        "{label}: median {:.4} s, min {:.4} s over {runs} runs (warm-up {:.4} s)",
        median(timings),
        timings.iter().min().unwrap().as_secs_f64(),
        warmup.as_secs_f64(),
    );
}

fn run_decode(flac: &[u8], sink: &mut u64) -> Result<Duration, AnyError> {
    let mut decoder = Decoder::new();
    if std::env::var_os("WAVEY_FLAC_NO_CRC").is_some() {
        decoder.set_verify_checksums(false);
    }
    let mut output = vec![0_i32; OUTPUT_CHUNK_SAMPLES];
    let started = Instant::now();
    let mut checksum = 0_u64;
    for chunk in flac.chunks(INPUT_CHUNK_BYTES) {
        let written = decoder.decode_i32(chunk, &mut output)?;
        for &sample in &output[..written] {
            checksum ^= sample as u64;
        }
        loop {
            let written = decoder.decode_i32(&[], &mut output)?;
            if written == 0 {
                break;
            }
            for &sample in &output[..written] {
                checksum ^= sample as u64;
            }
        }
    }
    loop {
        let written = decoder.decode_i32(&[], &mut output)?;
        if written == 0 {
            break;
        }
        for &sample in &output[..written] {
            checksum ^= sample as u64;
        }
    }
    decoder.finish()?;
    *sink = checksum.wrapping_add(*sink);
    Ok(started.elapsed())
}

fn load_wav(path: &str) -> Result<(Vec<i32>, u32, u16, u16, u32), AnyError> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();
    if spec.sample_format != SampleFormat::Int {
        return Err("the source WAV must contain integer PCM".into());
    }
    let bits = spec.bits_per_sample;
    let mut pcm = Vec::new();
    for sample in reader.samples::<i32>() {
        // Normalize everything to signed samples shifted to the LSB.
        match bits {
            16 => pcm.push(sample?),
            24 => pcm.push(sample?),
            _ => return Err(format!("unsupported bit depth {bits}").into()),
        }
    }
    Ok((
        pcm,
        spec.sample_rate,
        spec.channels,
        bits,
        reader.duration(),
    ))
}

fn profile_of(label: &str) -> FlacProfile {
    match label {
        "balanced" => FlacProfile::Balanced,
        "max" | "maximum" => FlacProfile::Maximum,
        other => panic!("unknown profile {other}, expected balanced or max"),
    }
}

fn run_encode(
    pcm: &[i32],
    channels: u16,
    bits: u16,
    sample_rate: u32,
    profile: FlacProfile,
    sink: &mut u64,
) -> Result<Duration, AnyError> {
    let config = FlacFrameConfig::new(sample_rate, channels, bits as u8, BLOCK_SIZE, profile)?;
    let mut encoder = Encoder::new(config)?;
    let samples_per_block = config.sample_count()?;
    let mut packet = Vec::with_capacity(1 << 20);
    let mut checksum = 0_u64;
    let started = Instant::now();
    for block in pcm.chunks(samples_per_block) {
        encoder.encode_i32(block, &mut packet)?;
        checksum = checksum.wrapping_add(packet.len() as u64);
        packet.clear();
    }
    let header = encoder.finish()?;
    checksum ^= header.len() as u64;
    *sink = checksum.wrapping_sub(*sink);
    Ok(started.elapsed())
}

fn main() -> Result<(), AnyError> {
    let mut args = env::args_os()
        .skip(1)
        .map(|v| v.to_string_lossy().into_owned());
    let mode = args.next().ok_or("usage: codec_bench decode|encode ...")?;
    let path = args.next().ok_or("missing input path")?;
    let qualifier = args.next();
    let runs: usize = args.next().map(|v| v.parse()).transpose()?.unwrap_or(5);

    let mut compressed: Vec<u8> = Vec::new();
    let mut pcm: Vec<i32> = Vec::new();
    let (mut channels, mut bits, mut sample_rate) = (0_u16, 0_u16, 0_u32);
    match mode.as_str() {
        "decode" => compressed = fs::read(&path)?,
        "encode" => {
            let loaded = load_wav(&path)?;
            pcm = loaded.0;
            sample_rate = loaded.1;
            channels = loaded.2;
            bits = loaded.3;
        }
        other => return Err(format!("unknown mode {other}").into()),
    }

    let mut sink = 0_u64;
    match mode.as_str() {
        "decode" => {
            let warmup = run_decode(&compressed, &mut sink)?;
            let mut timings = Vec::with_capacity(runs);
            for _ in 0..runs {
                timings.push(run_decode(&compressed, &mut sink)?);
            }
            report("decode", runs, warmup, &timings);
        }
        "encode" => {
            let profile = profile_of(qualifier.as_deref().ok_or("missing profile")?);
            let warmup = run_encode(&pcm, channels, bits, sample_rate, profile, &mut sink)?;
            let mut timings = Vec::with_capacity(runs);
            for _ in 0..runs {
                timings.push(run_encode(
                    &pcm,
                    channels,
                    bits,
                    sample_rate,
                    profile,
                    &mut sink,
                )?);
            }
            report(&format!("encode-{profile:?}"), runs, warmup, &timings);
        }
        _ => unreachable!(),
    }
    eprintln!("checksum sink: {sink}");
    Ok(())
}
