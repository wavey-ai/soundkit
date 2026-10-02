//! Decode the keyframes of each MOV/MP4 file given and print a digest of
//! every plane of every decoded frame, with the time.
//!
//! Usage: video_digest FILE...
use soundkit_audio_demux::{decode_mp4_keyframes_from_file, Mp4KeyframeOptions};
use std::path::Path;
use std::time::Instant;

fn main() {
    for path in std::env::args().skip(1) {
        let options = Mp4KeyframeOptions {
            max_keyframes: 100_000,
            ..Mp4KeyframeOptions::default()
        };
        let start = Instant::now();
        let line = match decode_mp4_keyframes_from_file(Path::new(&path), &options) {
            Ok(timeline) => {
                let mut digest = 0xcbf2_9ce4_8422_2325u64;
                let mut update = |bytes: &[u8]| {
                    for &byte in bytes {
                        digest ^= u64::from(byte);
                        digest = digest.wrapping_mul(0x0100_0000_01b3);
                    }
                };
                let mut frames = 0;
                for keyframe in &timeline.keyframes {
                    let Some(frame) = &keyframe.frame else {
                        update(b"none");
                        continue;
                    };
                    frames += 1;
                    update(&[frame.bit_depth, frame.has_alpha as u8]);
                    update(
                        format!(
                            "{:?}{:?}{:?}",
                            frame.color_model, frame.chroma_sampling, frame.pts
                        )
                        .as_bytes(),
                    );
                    for plane in &frame.planes {
                        update(&plane.width.to_le_bytes());
                        update(&plane.height.to_le_bytes());
                        update(&plane.stride.to_le_bytes());
                        update(&plane.data);
                    }
                }
                format!(
                    "{digest:016x} frames {frames} of {}",
                    timeline.keyframes.len()
                )
            }
            Err(error) => format!("error {error}"),
        };
        println!("{path}\t{line}\t{:.3}", start.elapsed().as_secs_f64());
    }
}
