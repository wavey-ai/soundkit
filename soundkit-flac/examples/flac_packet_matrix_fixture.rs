//! Generate varied deterministic PCM and a matching SoundKit raw-FLAC bundle.

use soundkit_flac::{FlacFrameConfig, FlacFrameEncoder, FlacProfile};
use std::env;
use std::fs;
use std::path::PathBuf;

fn usage() -> &'static str {
    "usage: flac_packet_matrix_fixture RATE CHANNELS 16|24 realtime|balanced FRAMES PCM_S32LE BUNDLE"
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    if arguments.len() != 7 {
        return Err(usage().into());
    }
    let sample_rate: u32 = arguments[0].parse()?;
    let channels: u16 = arguments[1].parse()?;
    let bits_per_sample: u8 = arguments[2].parse()?;
    let profile = match arguments[3].as_str() {
        "realtime" => FlacProfile::Realtime,
        "balanced" => FlacProfile::Balanced,
        _ => return Err(usage().into()),
    };
    let frame_count: usize = arguments[4].parse()?;
    if !matches!(sample_rate, 48_000 | 96_000)
        || !(1..=8).contains(&channels)
        || !matches!(bits_per_sample, 16 | 24)
        || frame_count == 0
    {
        return Err(usage().into());
    }
    let pcm_path = PathBuf::from(&arguments[5]);
    let bundle_path = PathBuf::from(&arguments[6]);
    let config = FlacFrameConfig::new(
        sample_rate,
        channels,
        bits_per_sample,
        sample_rate / 200,
        profile,
    )?;
    let samples = varied_samples(config, frame_count);
    let mut encoder = FlacFrameEncoder::new(config)?;
    let mut pcm = Vec::with_capacity(samples.len() * 4);
    let mut bundle = Vec::new();
    for frame in samples.chunks_exact(config.sample_count()?) {
        for sample in frame {
            pcm.extend_from_slice(&sample.to_le_bytes());
        }
        let mut packet = Vec::with_capacity(config.raw_pcm_bytes()? + 64);
        encoder.encode_i32_into(frame, &mut packet)?;
        bundle.extend_from_slice(&u32::try_from(packet.len())?.to_le_bytes());
        bundle.extend_from_slice(&packet);
    }
    fs::write(pcm_path, pcm)?;
    fs::write(bundle_path, bundle)?;
    println!(
        "soundkit fixture rate={sample_rate} channels={channels} bits={bits_per_sample} profile={profile:?} frames={frame_count}"
    );
    Ok(())
}

fn varied_samples(config: FlacFrameConfig, packet_count: usize) -> Vec<i32> {
    let channels = usize::from(config.channels);
    let frames = config.frame_length as usize * packet_count;
    let limit = if config.bits_per_sample == 16 {
        30_000_i32
    } else {
        7_500_000_i32
    };
    let mut output = Vec::with_capacity(frames * channels);
    let mut state = 0x9e37_79b9_u32;
    let mut smooth = vec![0_i32; channels];
    for frame in 0..frames {
        let packet = frame / config.frame_length as usize;
        for channel in 0..channels {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let random = ((state >> 8) as i32 & 0xffff) - 32_768;
            let tone = ((frame as f64 * (137.0 + channel as f64 * 53.0) * std::f64::consts::TAU
                / config.sample_rate as f64)
                .sin()
                * f64::from(limit)
                * 0.72) as i32;
            let sample = match packet % 8 {
                0 => 0,
                1 => tone,
                2 => tone.saturating_add(random * (limit / 262_144)),
                3 => {
                    let step = random * (limit / 1_048_576).max(1);
                    smooth[channel] = smooth[channel].saturating_add(step).clamp(-limit, limit);
                    smooth[channel]
                }
                4 => {
                    if frame % (config.frame_length as usize / 3).max(1) == 0 {
                        if (frame + channel) & 1 == 0 {
                            limit
                        } else {
                            -limit
                        }
                    } else {
                        0
                    }
                }
                5 => ((frame % 257) as i32 - 128) * (limit / 160).max(1),
                6 => random * (limit / 32_768),
                _ => (channel as i32 * 2 - channels as i32) * (limit / 16),
            };
            output.push(sample.clamp(-limit, limit));
        }
    }
    output
}
