use soundkit_webm::{WebmMediaDemuxEvent, WebmMediaDemuxer};
use std::{env, fs, path::PathBuf};

fn main() -> Result<(), String> {
    let path = env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| "usage: inspect-webm-media <file.webm>".to_string())?;
    let bytes = fs::read(&path).map_err(|error| format!("read {}: {error}", path.display()))?;
    let mut demuxer = WebmMediaDemuxer::new();
    let mut events = Vec::new();
    for chunk in bytes.chunks(64 * 1024) {
        events.extend(demuxer.add(chunk)?);
    }
    events.extend(demuxer.finish()?);

    let mut video_packets = 0usize;
    let mut audio_packets = 0usize;
    for event in events {
        match event {
            WebmMediaDemuxEvent::Config {
                timecode_scale_ns,
                track,
            } => println!(
                "track={} kind={} codec={} scale={} dimensions={}x{} audio={}Hz/{}ch private={}",
                track.track_number,
                track.kind.as_str(),
                track.codec_id,
                timecode_scale_ns,
                track.width.unwrap_or_default(),
                track.height.unwrap_or_default(),
                track.sample_rate.unwrap_or_default(),
                track.channels.unwrap_or_default(),
                track.codec_private.len(),
            ),
            WebmMediaDemuxEvent::Packet { kind, .. } if kind.as_str() == "video" => {
                video_packets += 1
            }
            WebmMediaDemuxEvent::Packet { .. } => audio_packets += 1,
        }
    }
    println!("videoPackets={video_packets} audioPackets={audio_packets}");
    Ok(())
}
