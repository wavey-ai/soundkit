//! Encodes a WAVE file to a FLAC file.
//!
//! usage: encode_file IN.wav OUT.flac

use hound::{SampleFormat, WavReader};
use soundkit_flac::stream::Encoder;
use std::fs::File;
use std::io::BufWriter;
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let input_path = match args.next() {
        Some(path) => path,
        None => {
            eprintln!("usage: encode_file IN.wav OUT.flac");
            return ExitCode::from(2);
        }
    };
    let output_path = match args.next() {
        Some(path) => path,
        None => {
            eprintln!("usage: encode_file IN.wav OUT.flac");
            return ExitCode::from(2);
        }
    };

    if let Err(error) = run(&input_path, &output_path) {
        eprintln!("error: {error}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

fn run(input_path: &str, output_path: &str) -> Result<(), String> {
    let mut reader = WavReader::open(input_path).map_err(|error| error.to_string())?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    let sample_rate = spec.sample_rate;
    let bits = spec.bits_per_sample;
    if spec.sample_format != SampleFormat::Int || bits > 32 {
        return Err("only integer PCM input is supported".to_string());
    }

    let total_samples = reader.duration() as u64 * channels as u64;
    let config = soundkit_flac::frame::FlacFrameConfig::new(
        sample_rate,
        channels as u16,
        bits as u8,
        4096,
        soundkit_flac::frame::FlacProfile::Balanced,
    )
    .map_err(|error| error.to_string())?;
    let mut encoder = Encoder::new(config).map_err(|error| error.to_string())?;

    let mut output = Vec::new();
    let block_len = 4096 * channels;
    let mut carry: Vec<i32> = Vec::new();
    let mut frames_written = 0_u64;

    let samples: Result<Vec<i32>, _> = reader
        .samples::<i32>()
        .map(|sample| sample.map_err(|error| error.to_string()))
        .collect();
    let samples = samples?;

    let shift = 32 - bits;
    for chunk in samples.chunks(block_len) {
        let scaled: Vec<i32> = if shift == 0 {
            chunk.to_vec()
        } else {
            chunk.iter().map(|&x| x << shift).collect()
        };
        carry.extend_from_slice(&scaled);
        while carry.len() >= block_len {
            let block: Vec<i32> = carry.drain(..block_len).collect();
            encoder
                .encode_i32(&block, &mut output)
                .map_err(|error| error.to_string())?;
            frames_written += 1;
        }
    }
    if !carry.is_empty() {
        encoder
            .encode_i32(&carry, &mut output)
            .map_err(|error| error.to_string())?;
        frames_written += 1;
    }

    let final_header = encoder.stream_header().to_vec();
    let mut file = BufWriter::new(File::create(output_path).map_err(|error| error.to_string())?);
    use std::io::Write;
    file.write_all(b"fLaC").map_err(|error| error.to_string())?;
    file.write_all(&final_header)
        .map_err(|error| error.to_string())?;
    file.write_all(&output[final_header.len()..])
        .map_err(|error| error.to_string())?;
    file.flush().map_err(|error| error.to_string())?;

    eprintln!(
        "encoded {frames_written} frames, {} bytes",
        output.len() + 4
    );
    Ok(())
}
