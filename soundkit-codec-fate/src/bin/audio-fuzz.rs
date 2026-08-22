use soundkit::media_metadata::extract_metadata;
use soundkit_decoder::{decode_audio_file, DecodeOptions};
use std::env;
use std::fs;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};

const MAX_SOURCE_BYTES: usize = 64 * 1024 * 1024;

fn manifest_paths(path: &Path) -> Result<Vec<PathBuf>, String> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("read manifest {}: {error}", path.display()))?;
    let mut paths = Vec::new();
    for (line_index, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields = line.split_whitespace().collect::<Vec<_>>();
        if fields.len() != 7 {
            return Err(format!(
                "{}:{} does not contain seven fields",
                path.display(),
                line_index + 1
            ));
        }
        paths.push(PathBuf::from(fields[5]));
    }
    Ok(paths)
}

fn mutations(source: &[u8]) -> Vec<Vec<u8>> {
    let mut cases = vec![Vec::new()];
    for length in [1, source.len() / 2, source.len().saturating_sub(1)] {
        if length < source.len() {
            cases.push(source[..length].to_vec());
        }
    }
    if !source.is_empty() {
        let mut flipped = source.to_vec();
        for numerator in 1..=8 {
            let position = source.len().saturating_mul(numerator) / 9;
            if let Some(byte) = flipped.get_mut(position.min(source.len() - 1)) {
                *byte ^= 0xa5;
            }
        }
        cases.push(flipped);

        let mut overwritten = source.to_vec();
        for numerator in 1..=4 {
            let position = source.len().saturating_mul(numerator) / 5;
            let end = position.saturating_add(4).min(overwritten.len());
            overwritten[position..end].fill(0xff);
        }
        cases.push(overwritten);
    }
    cases
}

fn exercise(data: &[u8]) {
    let _ = extract_metadata(data);
    let _ = decode_audio_file(
        data,
        DecodeOptions {
            output_bits_per_sample: Some(16),
            output_sample_rate: None,
            output_channels: None,
        },
    );
}

fn main() {
    let mut arguments = env::args().skip(1);
    let root = PathBuf::from(arguments.next().unwrap_or_else(|| "testdata".to_owned()));
    let manifest = PathBuf::from(
        arguments
            .next()
            .unwrap_or_else(|| "scripts/media-pcm-fixture-manifest.tsv".to_owned()),
    );
    if arguments.next().is_some() {
        eprintln!("usage: audio-fuzz [fixture-root] [manifest]");
        std::process::exit(2);
    }

    let paths = manifest_paths(&manifest).unwrap_or_else(|error| {
        eprintln!("{error}");
        std::process::exit(2);
    });
    let mut cases = 0usize;
    for relative in paths {
        let path = root.join(&relative);
        let source = fs::read(&path).unwrap_or_else(|error| {
            eprintln!("read {}: {error}", path.display());
            std::process::exit(2);
        });
        if source.len() > MAX_SOURCE_BYTES {
            eprintln!(
                "{} exceeds the {} byte fuzz-source budget",
                path.display(),
                MAX_SOURCE_BYTES
            );
            std::process::exit(2);
        }
        for (mutation, data) in mutations(&source).into_iter().enumerate() {
            cases += 1;
            if catch_unwind(AssertUnwindSafe(|| exercise(&data))).is_err() {
                eprintln!("panic: {} mutation {mutation}", relative.display());
                std::process::exit(1);
            }
        }
    }
    println!("audio/metadata mutation sweep passed: {cases} cases");
}
