use std::{env, fs};

use soundkit_audio_demux::{MxfMediaDemuxEvent, MxfMediaDemuxer};

fn main() -> Result<(), String> {
    let path = env::args()
        .nth(1)
        .ok_or_else(|| "usage: inspect-mxf-media <file.mxf>".to_string())?;
    let bytes = fs::read(&path).map_err(|error| format!("failed to read {path}: {error}"))?;
    let mut demuxer = MxfMediaDemuxer::new();
    let mut events = Vec::new();
    for chunk in bytes.chunks(32 * 1024) {
        events.extend(demuxer.push(chunk)?);
    }
    events.extend(demuxer.flush()?);

    for event in events {
        match event {
            MxfMediaDemuxEvent::Config(config) => println!(
                "track={} kind={} codec={} codec_id={} timescale={} samples={} {}x{} rate={} channels={} bits={}",
                config.track_id,
                config.kind.as_str(),
                config.codec,
                config.codec_id,
                config.timescale,
                config.sample_count,
                config.width.unwrap_or_default(),
                config.height.unwrap_or_default(),
                config.sample_rate.unwrap_or_default(),
                config.channels.unwrap_or_default(),
                config.bits_per_sample.unwrap_or_default(),
            ),
            MxfMediaDemuxEvent::Packet(packet) => println!(
                "packet track={} kind={} sample={} bytes={} dts={} duration={}",
                packet.track_id,
                packet.kind.as_str(),
                packet.sample_id,
                packet.data.len(),
                packet.decode_time,
                packet.duration,
            ),
        }
    }
    Ok(())
}
