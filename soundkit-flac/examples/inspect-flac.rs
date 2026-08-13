use soundkit::audio_packet::Decoder;
use soundkit_flac::FlacDecoderClaxon;
use std::{env, fs, io::Cursor, path::PathBuf};

fn main() -> Result<(), String> {
    let path = env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| "usage: inspect-flac <file.flac>".to_string())?;
    let bytes = fs::read(&path).map_err(|error| format!("read {}: {error}", path.display()))?;
    let mut decoder = FlacDecoderClaxon::new();
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
    let mut reader = claxon::FlacReader::new(Cursor::new(&bytes))
        .map_err(|error| format!("Claxon header: {error}"))?;
    let mut direct_samples = 0usize;
    for sample in reader.samples() {
        sample.map_err(|error| format!("Claxon frame: {error}"))?;
        direct_samples += 1;
    }
    println!("claxonDirectSamples={direct_samples}");
    Ok(())
}
