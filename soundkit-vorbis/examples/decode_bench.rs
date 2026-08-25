use soundkit_vorbis::VorbisDecoder;
use std::env;
use std::fs;
use std::hint::black_box;
use std::time::Instant;

fn consume_audio(
    audio: Option<soundkit::audio_types::AudioData>,
    samples: &mut usize,
    checksum: &mut i64,
    calculate_checksum: bool,
) {
    let Some(audio) = audio else { return };
    let pcm = black_box(audio.data());
    *samples += pcm.len() / 2;
    if calculate_checksum {
        *checksum += pcm
            .chunks_exact(2)
            .map(|sample| i64::from(i16::from_le_bytes([sample[0], sample[1]])))
            .sum::<i64>();
    }
}

fn decode_once(input: &[u8], calculate_checksum: bool) -> (usize, u32, u8, i64) {
    let mut decoder = VorbisDecoder::new();
    let mut samples = 0usize;
    let mut checksum = 0i64;
    consume_audio(
        decoder.add(black_box(input)).expect("decode Ogg Vorbis"),
        &mut samples,
        &mut checksum,
        calculate_checksum,
    );
    consume_audio(
        decoder.finish().expect("finish Ogg Vorbis"),
        &mut samples,
        &mut checksum,
        calculate_checksum,
    );
    (
        samples,
        decoder.sample_rate().expect("Vorbis sample rate"),
        decoder.channels().expect("Vorbis channels"),
        checksum,
    )
}

fn main() {
    let mut args = env::args().skip(1);
    let path = args
        .next()
        .expect("usage: decode_bench <input.ogg> [iterations]");
    let iterations = args
        .next()
        .map(|value| value.parse::<usize>().expect("integer iterations"))
        .unwrap_or(50);
    let input = fs::read(&path).expect("read Ogg Vorbis input");

    for _ in 0..3 {
        black_box(decode_once(&input, false));
    }

    let started = Instant::now();
    let mut total_samples = 0usize;
    let mut sample_rate = 0u32;
    let mut channels = 0u8;
    for _ in 0..iterations {
        let decoded = black_box(decode_once(&input, false));
        total_samples = total_samples.wrapping_add(decoded.0);
        sample_rate = decoded.1;
        channels = decoded.2;
    }
    let elapsed = started.elapsed();
    let checksum = decode_once(&input, true).3;

    println!(
        "implementation=soundkit-rust codec=vorbis operation=decode input_bytes={} iterations={} samples={} sample_rate={} channels={} elapsed_ns={} checksum={}",
        input.len(),
        iterations,
        total_samples,
        sample_rate,
        channels,
        elapsed.as_nanos(),
        checksum
    );
}
