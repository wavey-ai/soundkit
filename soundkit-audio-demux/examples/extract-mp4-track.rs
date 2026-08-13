use soundkit_audio_demux::{demux_mp4_media_file, MediaTrackKind};
use std::{env, fs, path::PathBuf};

fn main() -> Result<(), String> {
    let mut args = env::args_os().skip(1);
    let source = args
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| usage().to_string())?;
    let kind = args
        .next()
        .and_then(|value| value.into_string().ok())
        .and_then(|value| match value.as_str() {
            "audio" => Some(MediaTrackKind::Audio),
            "video" => Some(MediaTrackKind::Video),
            _ => None,
        })
        .ok_or_else(|| usage().to_string())?;
    let destination = args
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| usage().to_string())?;
    if args.next().is_some() {
        return Err(usage().to_string());
    }

    let bytes = fs::read(&source).map_err(|error| format!("read {}: {error}", source.display()))?;
    let media = demux_mp4_media_file(&bytes)?;
    let track = media
        .tracks
        .iter()
        .find(|track| track.kind == kind)
        .ok_or_else(|| {
            format!(
                "{} has no supported {} track",
                source.display(),
                kind.as_str()
            )
        })?;
    let mut output = track.decoder_configuration.clone();
    for packet in media
        .packets
        .iter()
        .filter(|packet| packet.track_id == track.track_id)
    {
        output.extend_from_slice(&packet.data);
    }
    fs::write(&destination, &output)
        .map_err(|error| format!("write {}: {error}", destination.display()))?;
    println!(
        "track={} kind={} codec={} bytes={}",
        track.track_id,
        track.kind.as_str(),
        track.codec,
        output.len()
    );
    Ok(())
}

fn usage() -> &'static str {
    "usage: extract-mp4-track <source.mov|source.mp4> <audio|video> <output>"
}
