use soundkit::audio_packet::Decoder;
use soundkit_g722::G722Decoder;
use std::env;
use std::fs;
use std::hint::black_box;
use std::time::Instant;

fn decode_once(input: &[u8], output: &mut [i16]) -> usize {
    let mut decoder = G722Decoder::new_64k();
    decoder
        .decode_i16(black_box(input), black_box(output), false)
        .expect("decode G.722")
}

fn main() {
    let mut args = env::args().skip(1);
    let path = args.next().expect("usage: decode_bench <input.g722> [iterations]");
    let iterations = args
        .next()
        .map(|value| value.parse::<usize>().expect("integer iterations"))
        .unwrap_or(100);
    let input = fs::read(&path).expect("read G.722 input");
    let mut output = vec![0i16; input.len() * 2];

    for _ in 0..5 {
        black_box(decode_once(&input, &mut output));
    }

    let started = Instant::now();
    let mut samples = 0usize;
    for _ in 0..iterations {
        samples = samples.wrapping_add(black_box(decode_once(&input, &mut output)));
    }
    let elapsed = started.elapsed();
    let checksum = output
        .iter()
        .fold(0i64, |sum, &sample| sum.wrapping_add(i64::from(sample)));

    println!(
        "implementation=soundkit codec=g722 operation=decode input_bytes={} iterations={} samples={} elapsed_ns={} checksum={}",
        input.len(),
        iterations,
        samples,
        elapsed.as_nanos(),
        checksum
    );
}
