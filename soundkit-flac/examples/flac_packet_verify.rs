//! Verify a length-prefixed raw FLAC packet sequence with SoundKit.

use soundkit_flac::{FlacFrameConfig, FlacFrameDecoder, FlacProfile};
use std::env;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    if arguments.len() != 5 {
        return Err("usage: flac_packet_verify RATE CHANNELS 16|24 PCM_S32LE BUNDLE".into());
    }
    let sample_rate: u32 = arguments[0].parse()?;
    let channels: u16 = arguments[1].parse()?;
    let bits_per_sample: u8 = arguments[2].parse()?;
    let config = FlacFrameConfig::new(
        sample_rate,
        channels,
        bits_per_sample,
        sample_rate / 200,
        FlacProfile::Realtime,
    )?;
    let pcm_bytes = fs::read(&arguments[3])?;
    if !pcm_bytes.len().is_multiple_of(4) {
        return Err("PCM input is not S32LE-aligned".into());
    }
    let pcm = pcm_bytes
        .chunks_exact(4)
        .map(|sample| i32::from_le_bytes(sample.try_into().unwrap()))
        .collect::<Vec<_>>();
    let bundle = fs::read(&arguments[4])?;
    let samples_per_frame = config.sample_count()?;
    if pcm.is_empty() || !pcm.len().is_multiple_of(samples_per_frame) {
        return Err("PCM input does not contain complete configured frames".into());
    }

    let mut decoder = FlacFrameDecoder::new(config)?;
    decoder.set_verify_checksums(true);
    let mut output = vec![0_i32; samples_per_frame];
    let mut offset = 0usize;
    let mut frame = 0usize;
    while offset < bundle.len() {
        let length_end = offset.checked_add(4).ok_or("bundle offset overflow")?;
        let length = bundle
            .get(offset..length_end)
            .ok_or("truncated bundle length")?;
        let length = u32::from_le_bytes(length.try_into().unwrap()) as usize;
        offset = length_end;
        let packet_end = offset.checked_add(length).ok_or("bundle offset overflow")?;
        let packet = bundle
            .get(offset..packet_end)
            .ok_or("truncated bundle packet")?;
        let expected = pcm
            .get(frame * samples_per_frame..(frame + 1) * samples_per_frame)
            .ok_or("bundle contains more frames than PCM")?;
        let written = decoder.decode_into(packet, &mut output)?;
        if written != expected.len() || output[..written] != *expected {
            return Err(format!("PCM mismatch in packet {frame}").into());
        }
        frame += 1;
        offset = packet_end;
    }
    if frame * samples_per_frame != pcm.len() {
        return Err("bundle contains fewer frames than PCM".into());
    }
    println!(
        "soundkit verified rate={sample_rate} channels={channels} bits={bits_per_sample} frames={frame}"
    );
    Ok(())
}
