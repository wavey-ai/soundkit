use soundkit_audio_demux::{demux_mp4_media_file, MediaTrackKind};
use std::{env, fs, path::PathBuf};

fn main() -> Result<(), String> {
    let path = env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| "usage: inspect-mp4-media <file.mov|file.mp4>".to_string())?;
    let bytes = fs::read(&path).map_err(|error| format!("read {}: {error}", path.display()))?;
    let media = demux_mp4_media_file(&bytes)?;
    for track in &media.tracks {
        println!(
            "track={} kind={} codec={} id={} samples={} timescale={} dimensions={}x{} audio={}Hz/{}ch private={} decoderConfig={}",
            track.track_id,
            track.kind.as_str(),
            track.codec,
            track.codec_id,
            track.sample_count,
            track.timescale,
            track.width.unwrap_or_default(),
            track.height.unwrap_or_default(),
            track.sample_rate.unwrap_or_default(),
            track.channels.unwrap_or_default(),
            track.codec_private.len(),
            track.decoder_configuration.len(),
        );
    }
    let video_packets = media
        .packets
        .iter()
        .filter(|packet| packet.kind == MediaTrackKind::Video)
        .count();
    let audio_packets = media.packets.len() - video_packets;
    println!(
        "packets={} video={} audio={}",
        media.packets.len(),
        video_packets,
        audio_packets
    );
    Ok(())
}
