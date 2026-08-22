use soundkit_decoder::{decode_audio_file, DecodeOptions};
use std::env;
use std::fs;
use std::path::PathBuf;

fn main() -> Result<(), String> {
    let mut arguments = env::args_os().skip(1);
    let input = PathBuf::from(
        arguments
            .next()
            .ok_or_else(|| "usage: decode-file INPUT OUTPUT.s16le".to_owned())?,
    );
    let output = PathBuf::from(
        arguments
            .next()
            .ok_or_else(|| "usage: decode-file INPUT OUTPUT.s16le".to_owned())?,
    );
    if arguments.next().is_some() {
        return Err("usage: decode-file INPUT OUTPUT.s16le".to_owned());
    }
    let source = fs::read(&input).map_err(|error| format!("read {}: {error}", input.display()))?;
    let decoded = decode_audio_file(
        &source,
        DecodeOptions {
            output_bits_per_sample: Some(16),
            output_sample_rate: None,
            output_channels: None,
        },
    )
    .map_err(|error| error.to_string())?;
    let first = decoded
        .frames
        .first()
        .ok_or_else(|| "decoder emitted no PCM".to_owned())?;
    let sample_rate = first.sampling_rate();
    let channels = first.channel_count();
    let mut pcm = Vec::new();
    for frame in decoded.frames {
        if frame.bits_per_sample() != 16
            || frame.sampling_rate() != sample_rate
            || frame.channel_count() != channels
        {
            return Err("PCM format changed during decode".to_owned());
        }
        pcm.extend_from_slice(frame.data());
    }
    fs::write(&output, &pcm).map_err(|error| format!("write {}: {error}", output.display()))?;
    eprintln!(
        "decoded {} bytes: {} Hz, {} channels; title={:?}, artist={:?}, album={:?}",
        pcm.len(),
        sample_rate,
        channels,
        decoded.metadata.title,
        decoded.metadata.artists,
        decoded.metadata.album,
    );
    Ok(())
}
