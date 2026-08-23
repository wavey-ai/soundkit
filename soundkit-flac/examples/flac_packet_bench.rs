//! Per-call latency benchmark over a sequence of real PCM frames.
//!
//! Packet bundles use a deliberately tiny interchange format: each raw FLAC
//! frame is prefixed by its byte length as a little-endian `u32`.

use soundkit_flac::{FlacFrameConfig, FlacFrameDecoder, FlacFrameEncoder, FlacProfile};
use std::env;
use std::fs;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Instant;

const WARMUP_CALLS: usize = 1_024;
const MAX_PACKET_BYTES: usize = 16 * 1024 * 1024;

type AnyError = Box<dyn std::error::Error>;

struct Arguments {
    sample_rate: u32,
    channels: u16,
    bits_per_sample: u8,
    profile: FlacProfile,
    iterations: usize,
    pcm_path: PathBuf,
    decode_bundle_path: Option<PathBuf>,
    output_bundle_path: Option<PathBuf>,
}

fn usage() -> &'static str {
    "usage: flac_packet_bench RATE CHANNELS 16|24 realtime|balanced ITERATIONS PCM_S32LE [DECODE_BUNDLE|-] [OUTPUT_BUNDLE]"
}

fn parse_arguments() -> Result<Arguments, AnyError> {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    if !(6..=8).contains(&arguments.len()) {
        return Err(usage().into());
    }
    let sample_rate = arguments[0].parse()?;
    let channels = arguments[1].parse()?;
    let bits_per_sample = arguments[2].parse()?;
    let profile = match arguments[3].as_str() {
        "realtime" => FlacProfile::Realtime,
        "balanced" => FlacProfile::Balanced,
        _ => return Err(usage().into()),
    };
    let iterations = arguments[4].parse()?;
    if !matches!(sample_rate, 48_000 | 96_000)
        || !(1..=8).contains(&channels)
        || !matches!(bits_per_sample, 16 | 24)
        || iterations == 0
    {
        return Err(usage().into());
    }
    Ok(Arguments {
        sample_rate,
        channels,
        bits_per_sample,
        profile,
        iterations,
        pcm_path: arguments[5].as_str().into(),
        decode_bundle_path: arguments
            .get(6)
            .filter(|path| path.as_str() != "-")
            .map(PathBuf::from),
        output_bundle_path: arguments.get(7).map(PathBuf::from),
    })
}

fn read_s32le(path: &Path) -> Result<Vec<i32>, AnyError> {
    let bytes = fs::read(path)?;
    if bytes.is_empty() || !bytes.len().is_multiple_of(4) {
        return Err(format!(
            "{} must contain a non-empty whole number of S32LE samples",
            path.display()
        )
        .into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|sample| i32::from_le_bytes(sample.try_into().unwrap()))
        .collect())
}

fn read_bundle(path: &Path) -> Result<Vec<Vec<u8>>, AnyError> {
    let bytes = fs::read(path)?;
    let mut packets = Vec::new();
    let mut offset = 0usize;
    while offset < bytes.len() {
        let end = offset
            .checked_add(4)
            .ok_or("packet bundle offset overflow")?;
        let length = bytes
            .get(offset..end)
            .ok_or("truncated packet bundle length")?;
        let length = u32::from_le_bytes(length.try_into().unwrap()) as usize;
        if length == 0 || length > MAX_PACKET_BYTES {
            return Err(format!("invalid packet bundle frame length {length}").into());
        }
        offset = end;
        let end = offset
            .checked_add(length)
            .ok_or("packet bundle frame offset overflow")?;
        packets.push(
            bytes
                .get(offset..end)
                .ok_or("truncated packet bundle frame")?
                .to_vec(),
        );
        offset = end;
    }
    if packets.is_empty() {
        return Err("packet bundle contains no frames".into());
    }
    Ok(packets)
}

fn write_bundle(path: &Path, packets: &[Vec<u8>]) -> Result<(), AnyError> {
    let capacity = packets
        .iter()
        .try_fold(0usize, |total, packet| total.checked_add(4 + packet.len()));
    let mut bytes = Vec::with_capacity(capacity.ok_or("packet bundle size overflow")?);
    for packet in packets {
        let length = u32::try_from(packet.len()).map_err(|_| "packet exceeds u32")?;
        bytes.extend_from_slice(&length.to_le_bytes());
        bytes.extend_from_slice(packet);
    }
    fs::write(path, bytes)?;
    Ok(())
}

fn percentile(sorted_nanos: &[u128], percentile: usize) -> f64 {
    let rank = (sorted_nanos.len() * percentile).div_ceil(100);
    sorted_nanos[rank.saturating_sub(1).min(sorted_nanos.len() - 1)] as f64 / 1_000.0
}

fn report(label: &str, mut nanos: Vec<u128>, calls: usize, bytes: usize, pcm_bytes: usize) {
    nanos.sort_unstable();
    println!(
        "{label} p50_us={:.3} p95_us={:.3} p99_us={:.3} min_us={:.3} encoded/pcm={:.4} calls={calls}",
        percentile(&nanos, 50),
        percentile(&nanos, 95),
        percentile(&nanos, 99),
        nanos[0] as f64 / 1_000.0,
        bytes as f64 / pcm_bytes as f64,
    );
}

fn main() -> Result<(), AnyError> {
    let arguments = parse_arguments()?;
    let frame_length = arguments.sample_rate / 200;
    let config = FlacFrameConfig::new(
        arguments.sample_rate,
        arguments.channels,
        arguments.bits_per_sample,
        frame_length,
        arguments.profile,
    )?;
    let samples_per_frame = config.sample_count()?;
    let pcm = read_s32le(&arguments.pcm_path)?;
    if !pcm.len().is_multiple_of(samples_per_frame) {
        return Err(format!(
            "{} has {} samples; expected a multiple of {samples_per_frame}",
            arguments.pcm_path.display(),
            pcm.len()
        )
        .into());
    }
    let pcm_frames = pcm.chunks_exact(samples_per_frame).collect::<Vec<_>>();

    let mut fixture_encoder = FlacFrameEncoder::new(config)?;
    let mut own_packets = Vec::with_capacity(pcm_frames.len());
    for frame in &pcm_frames {
        let mut packet = Vec::with_capacity(config.raw_pcm_bytes()? + 64);
        fixture_encoder.encode_i32_into(frame, &mut packet)?;
        own_packets.push(packet);
    }
    if let Some(path) = &arguments.output_bundle_path {
        write_bundle(path, &own_packets)?;
    }
    let decode_packets = if let Some(path) = &arguments.decode_bundle_path {
        read_bundle(path)?
    } else {
        own_packets.clone()
    };
    if decode_packets.len() != pcm_frames.len() {
        return Err(format!(
            "packet bundle has {} frames but PCM corpus has {}",
            decode_packets.len(),
            pcm_frames.len()
        )
        .into());
    }

    let mut decoder = FlacFrameDecoder::new(config)?;
    let mut decoded = vec![0_i32; samples_per_frame];
    for (index, (packet, expected)) in decode_packets.iter().zip(&pcm_frames).enumerate() {
        let written = decoder.decode_into(packet, &mut decoded)?;
        if written != expected.len() || decoded[..written] != **expected {
            return Err(format!("decoded PCM mismatch in corpus frame {index}").into());
        }
    }

    let mut encoder = FlacFrameEncoder::new(config)?;
    let mut packet = Vec::with_capacity(config.raw_pcm_bytes()? + 64);
    for iteration in 0..WARMUP_CALLS {
        let frame = pcm_frames[iteration % pcm_frames.len()];
        black_box(encoder.encode_i32_into(black_box(frame), &mut packet)?);
    }
    let mut encode_nanos = Vec::with_capacity(arguments.iterations);
    let mut encoded_bytes = 0usize;
    for iteration in 0..arguments.iterations {
        let frame = pcm_frames[iteration % pcm_frames.len()];
        let started = Instant::now();
        let written = encoder.encode_i32_into(black_box(frame), &mut packet)?;
        encode_nanos.push(started.elapsed().as_nanos());
        encoded_bytes = encoded_bytes.saturating_add(black_box(written));
    }

    for iteration in 0..WARMUP_CALLS {
        let packet = &decode_packets[iteration % decode_packets.len()];
        black_box(decoder.decode_into(black_box(packet), &mut decoded)?);
    }
    let mut decode_nanos = Vec::with_capacity(arguments.iterations);
    let mut decoded_samples = 0usize;
    let mut decoded_packet_bytes = 0usize;
    for iteration in 0..arguments.iterations {
        let packet = &decode_packets[iteration % decode_packets.len()];
        let started = Instant::now();
        let written = decoder.decode_into(black_box(packet), &mut decoded)?;
        decode_nanos.push(started.elapsed().as_nanos());
        decoded_samples = decoded_samples.saturating_add(black_box(written));
        decoded_packet_bytes = decoded_packet_bytes.saturating_add(packet.len());
    }

    let raw_bytes_per_frame = config.raw_pcm_bytes()?;
    println!(
        "soundkit corpus rate={} frame={} channels={} bits={} profile={:?} corpus_frames={} corpus_ms={:.1}",
        config.sample_rate,
        config.frame_length,
        config.channels,
        config.bits_per_sample,
        config.profile,
        pcm_frames.len(),
        pcm_frames.len() as f64 * 5.0,
    );
    report(
        "soundkit encode",
        encode_nanos,
        arguments.iterations,
        encoded_bytes,
        arguments.iterations * raw_bytes_per_frame,
    );
    report(
        "soundkit decode",
        decode_nanos,
        arguments.iterations,
        decoded_packet_bytes,
        decoded_samples * usize::from(config.bits_per_sample / 8),
    );
    Ok(())
}
