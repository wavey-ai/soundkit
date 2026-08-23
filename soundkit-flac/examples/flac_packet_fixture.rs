//! Writes one deterministic raw FLAC frame and its expected interleaved PCM.

use soundkit_flac::{FlacFrameConfig, FlacFrameEncoder, FlacProfile};
use std::env;
use std::fs;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = env::args_os().skip(1);
    let sample_rate: u32 = arguments
        .next()
        .ok_or("missing sample rate")?
        .to_string_lossy()
        .parse()?;
    let profile = match arguments
        .next()
        .ok_or("missing profile")?
        .to_string_lossy()
        .as_ref()
    {
        "realtime" => FlacProfile::Realtime,
        "balanced" => FlacProfile::Balanced,
        _ => return Err("profile must be realtime or balanced".into()),
    };
    let packet_path = PathBuf::from(arguments.next().ok_or("missing packet output path")?);
    let pcm_path = PathBuf::from(arguments.next().ok_or("missing PCM output path")?);
    if arguments.next().is_some() || !matches!(sample_rate, 48_000 | 96_000) {
        return Err(
            "usage: flac_packet_fixture 48000|96000 realtime|balanced PACKET PCM_S32LE".into(),
        );
    }

    let config = FlacFrameConfig::new(sample_rate, 2, 24, sample_rate / 200, profile)?;
    let samples = signal(sample_rate, config.sample_count()?);
    let mut packet = Vec::with_capacity(config.raw_pcm_bytes()? + 64);
    FlacFrameEncoder::new(config)?.encode_i32_into(&samples, &mut packet)?;

    let mut pcm = Vec::with_capacity(samples.len() * 4);
    for sample in &samples {
        pcm.extend_from_slice(&sample.to_le_bytes());
    }
    fs::write(&packet_path, &packet)?;
    fs::write(&pcm_path, pcm)?;
    println!(
        "rate={sample_rate} frame={} profile={profile:?} packet_bytes={} pcm_samples={}",
        config.frame_length,
        packet.len(),
        samples.len()
    );
    Ok(())
}

fn signal(sample_rate: u32, sample_count: usize) -> Vec<i32> {
    (0..sample_count)
        .map(|index| {
            let phase = index as f64 * 440.0 * std::f64::consts::TAU / sample_rate as f64;
            (phase.sin() * 2_000_000.0) as i32
        })
        .collect()
}
