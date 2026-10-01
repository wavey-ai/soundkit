//! Time the lossless SoundKit stream encoder for a stereo 48 kHz programme
//! pushed in pieces, and print a digest of the stream.
//!
//! Usage: flac_stream_bench SECONDS PIECE_FRAMES
//!
//! A PIECE_FRAMES of 0 pushes the whole programme at once.
use soundkit_stream::{PcmOpusStreamOptions, StreamEncoder};
use std::time::Instant;

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let usage = "Usage: flac_stream_bench SECONDS PIECE_FRAMES";
    let seconds: usize = args.next().ok_or(usage)?.parse().map_err(|_| usage)?;
    let piece: usize = args.next().ok_or(usage)?.parse().map_err(|_| usage)?;
    let frames = seconds * 48_000;
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let pcm: Vec<i32> = (0..frames * 2)
        .map(|n| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let phase = ((n / 2) % 480) as i32;
            (240 - phase).abs() * 20_000 - 2_400_000 + (state % 4_001) as i32 - 2_000
        })
        .collect();
    let frame_size = (32..=32_767usize.min(frames).min(240))
        .rev()
        .find(|c| frames % c == 0 || frames % c >= 32)
        .ok_or("No valid FLAC block size")?;
    let start = Instant::now();
    let mut encoder = StreamEncoder::new(true, &PcmOpusStreamOptions::default())?;
    encoder.ensure_flac_geometry(48_000, 2, Some(frame_size))?;
    let piece = if piece == 0 { frames } else { piece };
    for chunk in pcm.chunks(piece * 2) {
        encoder.push_flac_i32(chunk)?;
    }
    encoder.finish_flac()?;
    let elapsed = start.elapsed();
    let mut digest = 0xcbf2_9ce4_8422_2325u64;
    for &byte in encoder.flac_stream() {
        digest ^= u64::from(byte);
        digest = digest.wrapping_mul(0x0100_0000_01b3);
    }
    println!(
        "seconds {seconds} piece {piece} encode_seconds {:.3} bytes {} digest {digest:016x}",
        elapsed.as_secs_f64(),
        encoder.flac_stream().len()
    );
    Ok(())
}
