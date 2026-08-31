use soundkit_audio_demux::{decode_mp4_keyframes_from_file, Mp4KeyframeOptions};
use std::path::PathBuf;

fn main() -> Result<(), String> {
    let mut args = std::env::args_os().skip(1);
    let source = args
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| usage().to_string())?;
    let mut options = Mp4KeyframeOptions::default();
    if let Some(value) = args.next() {
        options.max_keyframes = value
            .into_string()
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .ok_or_else(|| usage().to_string())?;
    }
    if let Some(value) = args.next() {
        options.stride = value
            .into_string()
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .ok_or_else(|| usage().to_string())?;
    }
    if args.next().is_some() {
        return Err(usage().to_string());
    }

    let timeline = decode_mp4_keyframes_from_file(&source, &options)?;
    println!(
        "track={} codec={} id={} timescale={} size={}x{} keyframes={} timeline={} decoded={}",
        timeline.track_id,
        timeline.codec,
        timeline.codec_id,
        timeline.timescale,
        timeline.width,
        timeline.height,
        timeline.total_keyframes,
        timeline.keyframes.len(),
        timeline.decoded_keyframes,
    );
    for keyframe in &timeline.keyframes {
        match &keyframe.frame {
            Some(frame) => {
                let first = &frame.planes[0];
                let checksum: u64 = first
                    .data
                    .iter()
                    .take(first.data.len().min(1024))
                    .fold(0xcbf29ce484222325_u64, |hash, byte| {
                        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
                    });
                println!(
                    "  sample={} t={:.3}s dts={} dur={} frame={}x{} bitDepth={} chroma={} checksum={:x}",
                    keyframe.sample_id,
                    keyframe.presentation_seconds,
                    keyframe.decode_time,
                    keyframe.duration,
                    frame.width,
                    frame.height,
                    frame.bit_depth,
                    frame.chroma_sampling.as_str(),
                    checksum,
                );
            }
            None => println!(
                "  sample={} t={:.3}s dts={} dur={} frame=-",
                keyframe.sample_id,
                keyframe.presentation_seconds,
                keyframe.decode_time,
                keyframe.duration,
            ),
        }
    }
    Ok(())
}

fn usage() -> &'static str {
    "usage: decode-mp4-keyframes <file.mov|file.mp4> [max-keyframes] [stride]"
}