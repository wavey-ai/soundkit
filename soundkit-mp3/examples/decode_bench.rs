use soundkit::audio_packet::Decoder;
use soundkit_mp3::Mp3Decoder;
use std::env;
use std::fs;
use std::hint::black_box;
use std::time::Instant;

fn decode_once(input: &[u8], output: &mut [f32]) -> (usize, u32, u8) {
    let mut decoder = Mp3Decoder::new();
    let samples = decoder
        .decode_f32(black_box(input), black_box(output), false)
        .expect("decode MP3");
    (
        samples,
        decoder.sample_rate().expect("MP3 sample rate"),
        decoder.channels().expect("MP3 channels"),
    )
}

fn main() {
    let mut args = env::args().skip(1);
    let path = args
        .next()
        .expect("usage: decode_bench <input.mp3> [iterations]");
    let iterations = args
        .next()
        .map(|value| value.parse::<usize>().expect("integer iterations"))
        .unwrap_or(50);
    let input = fs::read(&path).expect("read MP3 input");
    // Even 8 kbit/s MPEG-2 Layer III remains below this compressed-to-PCM
    // ratio for the supported mono/stereo output formats.
    let mut output = vec![0.0f32; input.len().saturating_mul(64).max(2304)];

    for _ in 0..3 {
        black_box(decode_once(&input, &mut output));
    }

    let started = Instant::now();
    let mut total_samples = 0usize;
    let mut sample_rate = 0;
    let mut channels = 0;
    for _ in 0..iterations {
        let decoded = black_box(decode_once(&input, &mut output));
        total_samples = total_samples.wrapping_add(decoded.0);
        sample_rate = decoded.1;
        channels = decoded.2;
    }
    let elapsed = started.elapsed();
    let checksum = output
        .iter()
        .fold(0.0f64, |sum, &sample| sum + f64::from(sample));

    println!(
        "implementation=soundkit codec=mp3 operation=decode input_bytes={} iterations={} samples={} sample_rate={} channels={} elapsed_ns={} checksum={:.9}",
        input.len(),
        iterations,
        total_samples,
        sample_rate,
        channels,
        elapsed.as_nanos(),
        checksum
    );
}
