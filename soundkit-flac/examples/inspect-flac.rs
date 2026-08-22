use soundkit::audio_packet::Decoder;
use soundkit_flac::FlacDecoder;
use std::{env, fs, path::PathBuf};

fn main() -> Result<(), String> {
    let path = env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| "usage: inspect-flac <file.flac>".to_string())?;
    let bytes = fs::read(&path).map_err(|error| format!("read {}: {error}", path.display()))?;
    let mut decoder = FlacDecoder::new();
    decoder.init()?;
    let mut scratch = vec![0; 262_144];
    let mut samples = decoder.decode_i32(&bytes, &mut scratch, false)?;
    loop {
        let decoded = decoder.decode_i32(&[], &mut scratch, false)?;
        if decoded == 0 {
            break;
        }
        samples += decoded;
    }
    println!(
        "samples={} sampleRate={} channels={} bitsPerSample={}",
        samples,
        decoder.sample_rate().unwrap_or_default(),
        decoder.channels().unwrap_or_default(),
        decoder.bits_per_sample().unwrap_or_default(),
    );
    decoder.finish()?;
    Ok(())
}
