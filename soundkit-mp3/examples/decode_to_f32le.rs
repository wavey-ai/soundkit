use soundkit::audio_packet::Decoder;
use soundkit_mp3::Mp3Decoder;
use std::env;
use std::fs;

fn main() {
    let mut args = env::args().skip(1);
    let input_path = args
        .next()
        .expect("usage: decode_to_f32le <input.mp3> <output.f32le>");
    let output_path = args
        .next()
        .expect("usage: decode_to_f32le <input.mp3> <output.f32le>");
    let input = fs::read(&input_path).expect("read MP3 input");
    let mut pcm = vec![0.0f32; input.len().saturating_mul(64).max(2304)];
    let mut decoder = Mp3Decoder::new();
    let samples = decoder
        .decode_f32(&input, &mut pcm, false)
        .expect("decode MP3");
    pcm.truncate(samples);

    let mut bytes = Vec::with_capacity(pcm.len() * 4);
    for sample in pcm {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    fs::write(output_path, bytes).expect("write PCM output");
    eprintln!(
        "samples={} sample_rate={} channels={}",
        samples,
        decoder.sample_rate().expect("sample rate"),
        decoder.channels().expect("channels")
    );
}
