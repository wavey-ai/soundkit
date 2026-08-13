use soundkit_audio_demux::{AudioDemuxEvent, AudioTrackDemuxer};
use std::{env, fs, process};

fn main() {
    let path = env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: inspect <media-file>");
        process::exit(2);
    });
    let bytes = fs::read(&path).unwrap_or_else(|error| {
        eprintln!("could not read {path}: {error}");
        process::exit(2);
    });
    let mut demuxer = AudioTrackDemuxer::new_auto();
    let mut events = Vec::new();
    for chunk in bytes.chunks(64 * 1024) {
        events.extend(demuxer.push(chunk).unwrap_or_else(|error| {
            eprintln!("demux failed: {error}");
            process::exit(2);
        }));
    }
    events.extend(demuxer.flush().unwrap_or_else(|error| {
        eprintln!("demux flush failed: {error}");
        process::exit(2);
    }));
    let mut packets = 0usize;
    let mut bytes = 0usize;
    for event in events {
        match event {
            AudioDemuxEvent::Config(config) => println!(
                "container={} codec={} codecId={} rate={} channels={} bits={} endian={} float={}",
                config.container.as_str(),
                config.codec.as_str(),
                config.codec_id.as_deref().unwrap_or("unknown"),
                config.sample_rate.unwrap_or(0),
                config.channels.unwrap_or(0),
                config.bits_per_sample.unwrap_or(0),
                config
                    .pcm_endianness
                    .map(|value| value.as_str())
                    .unwrap_or("n/a"),
                config.pcm_float.unwrap_or(false),
            ),
            AudioDemuxEvent::Packet(packet) => {
                packets += 1;
                bytes += packet.data.len();
            }
        }
    }
    println!("packets={packets} bytes={bytes}");
}
