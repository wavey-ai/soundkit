use soundkit_vorbis::VorbisDecoder;
use std::env;
use std::fs;

fn main() {
    let mut args = env::args().skip(1);
    let input_path = args
        .next()
        .expect("usage: decode_to_s16le <input.ogg> <output.s16le>");
    let output_path = args
        .next()
        .expect("usage: decode_to_s16le <input.ogg> <output.s16le>");
    let input = fs::read(input_path).expect("read Ogg Vorbis input");
    let mut decoder = VorbisDecoder::new();
    let mut pcm = Vec::new();
    if let Some(audio) = decoder.add(&input).expect("decode Ogg Vorbis") {
        pcm.extend_from_slice(audio.data());
    }
    if let Some(audio) = decoder.finish().expect("finish Ogg Vorbis") {
        pcm.extend_from_slice(audio.data());
    }
    fs::write(output_path, &pcm).expect("write PCM output");
    eprintln!(
        "samples={} sample_rate={} channels={}",
        pcm.len() / 2,
        decoder.sample_rate().expect("sample rate"),
        decoder.channels().expect("channels")
    );
}
