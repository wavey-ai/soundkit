//! Push a file through `Mp3Decoder` in fixed chunks and print the time,
//! the samples, a digest of them and the first error.
//!
//! Usage: mp3_push_probe FILE CHUNK_BYTES [MAX_BYTES]
use soundkit::audio_packet::Decoder;
use soundkit_mp3::Mp3Decoder;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let data = std::fs::read(&args[1]).unwrap();
    let chunk: usize = args[2].parse().unwrap();
    let limit: usize = args.get(3).map_or(data.len(), |v| v.parse().unwrap());
    let data = &data[..limit.min(data.len())];
    let mut decoder = Mp3Decoder::new();
    let mut out = vec![0i16; 1 << 20];
    let mut digest = 0xcbf2_9ce4_8422_2325u64;
    let mut samples = 0usize;
    let mut dump = Vec::new();
    let mut calls = Vec::new();
    let mut error = None;
    let start = Instant::now();
    let ended = std::env::var_os("MP3_PROBE_NO_END").is_none();
    for piece in data.chunks(chunk).chain(std::iter::once(&[][..])) {
        if piece.is_empty() {
            if !ended {
                break;
            }
            decoder.end_input();
        }
        let mut input = piece;
        loop {
            match decoder.decode_i16(input, &mut out, false) {
                Ok(written) => {
                    samples += written;
                    calls.push(written);
                    dump.extend_from_slice(&out[..written]);
                    for sample in &out[..written] {
                        for byte in sample.to_le_bytes() {
                            digest ^= u64::from(byte);
                            digest = digest.wrapping_mul(0x0100_0000_01b3);
                        }
                    }
                    if written == 0 {
                        break;
                    }
                    input = &[];
                }
                Err(e) => {
                    error = Some(e);
                    break;
                }
            }
        }
        if error.is_some() {
            break;
        }
    }
    if let Some(path) = std::env::var_os("MP3_PROBE_DUMP") {
        let bytes: Vec<u8> = dump.iter().flat_map(|s| s.to_le_bytes()).collect();
        std::fs::write(path, bytes).unwrap();
    }
    let _ = calls;
    println!(
        "seconds {:.3} samples {samples} digest {digest:016x} buffered {} error {}",
        start.elapsed().as_secs_f64(),
        decoder.buffer_len(),
        error.unwrap_or_else(|| "-".into())
    );
}
