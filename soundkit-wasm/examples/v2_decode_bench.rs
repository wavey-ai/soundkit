//! Decode a SoundKit v2 file with the cloud mastering call pattern and print
//! the decode time and SHA-256 digests of the frames and the float PCM.
//!
//! Usage: v2_decode_bench FILE.sk2 [CHUNK_BYTES] [RUNS]
//!
//! Each chunk goes to `SoundKitFrameStream::push` and then to
//! `SoundKitV2Decoder::push_float`. With RUNS above one, each run decodes the
//! file again and the times are the totals of all runs. With the environment
//! variable V2_BENCH_NO_DIGEST set, no digest is computed, for profiling.
use sha2::{Digest, Sha256};
use soundkit::frame_stream::SoundKitFrameStream;
use soundkit_wasm::v2::SoundKitV2Decoder;
use std::time::{Duration, Instant};

const USAGE: &str = "Usage: v2_decode_bench FILE.sk2 [CHUNK_BYTES] [RUNS]";

#[derive(Default)]
struct Times {
    frame_stream: Duration,
    push_float: Duration,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args.next().ok_or(USAGE)?;
    let chunk_bytes: usize = match args.next() {
        Some(value) => value.parse()?,
        None => 256 * 1024,
    };
    let runs: usize = match args.next() {
        Some(value) => value.parse()?,
        None => 1,
    };
    let bytes = std::fs::read(path)?;
    let mut times = Times::default();
    for _ in 1..runs {
        decode(&bytes, chunk_bytes, &mut times)?;
    }
    let report = decode(&bytes, chunk_bytes, &mut times)?;
    println!(
        "frame_stream_seconds {:.3}",
        times.frame_stream.as_secs_f64()
    );
    println!("push_float_seconds {:.3}", times.push_float.as_secs_f64());
    println!(
        "total_seconds {:.3}",
        (times.frame_stream + times.push_float).as_secs_f64()
    );
    print!("{report}");
    Ok(())
}

fn decode(bytes: &[u8], chunk_bytes: usize, times: &mut Times) -> Result<String, String> {
    let mut frames = SoundKitFrameStream::default();
    let mut decoder = SoundKitV2Decoder::new();
    let mut frame_digest = Sha256::new();
    let mut pcm_digest = Sha256::new();
    let mut frame_count = 0usize;
    let mut pcm_bytes = 0usize;
    let digest = std::env::var_os("V2_BENCH_NO_DIGEST").is_none();
    for chunk in bytes.chunks(chunk_bytes.max(1)) {
        let start = Instant::now();
        let read = frames.push(chunk)?;
        times.frame_stream += start.elapsed();
        for frame in read {
            frame_count += 1;
            if digest {
                frame_digest.update(&frame.encoded_header_bytes);
                frame_digest.update(&frame.payload);
            }
        }
        let start = Instant::now();
        let batch = decoder.push_float(chunk)?;
        times.push_float += start.elapsed();
        for block in batch.frames {
            pcm_bytes += block.data().len();
            if digest {
                pcm_digest.update(block.data());
            }
        }
    }
    Ok(format!(
        "frames {frame_count}\npcm_bytes {pcm_bytes}\nbuffered {} {}\nframes_sha256 {:x}\npcm_sha256 {:x}\n",
        frames.buffered_bytes(),
        decoder.buffered_bytes(),
        frame_digest.finalize(),
        pcm_digest.finalize()
    ))
}
