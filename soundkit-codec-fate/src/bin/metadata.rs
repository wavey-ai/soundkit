use sha2::{Digest, Sha256};
use soundkit::media_metadata::{extract_metadata, MediaMetadata};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const MAX_SWEEP_FILE_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Debug)]
struct Case {
    name: String,
    path: PathBuf,
    sha256: String,
    expected: Vec<(String, String)>,
}

fn parse_manifest(path: &Path) -> Result<Vec<Case>, String> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("read manifest {}: {error}", path.display()))?;
    let mut cases = Vec::new();
    for (line_index, raw) in text.lines().enumerate() {
        if raw.trim().is_empty() || raw.trim_start().starts_with('#') {
            continue;
        }
        let fields = raw.split('\t').collect::<Vec<_>>();
        if fields.len() < 4 {
            return Err(format!(
                "{}:{} must contain tab-separated name, path, SHA-256, and expectations",
                path.display(),
                line_index + 1
            ));
        }
        let sha256 = fields[2].to_ascii_lowercase();
        if sha256.len() != 64 || !sha256.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(format!(
                "{}:{} has invalid SHA-256",
                path.display(),
                line_index + 1
            ));
        }
        let expected = fields[3..]
            .iter()
            .map(|field| {
                field
                    .split_once('=')
                    .map(|(key, value)| (key.to_owned(), value.to_owned()))
                    .ok_or_else(|| {
                        format!(
                            "{}:{} expectation {field:?} is not key=value",
                            path.display(),
                            line_index + 1
                        )
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        cases.push(Case {
            name: fields[0].to_owned(),
            path: fields[1].into(),
            sha256,
            expected,
        });
    }
    if cases.is_empty() {
        return Err(format!("manifest {} is empty", path.display()));
    }
    Ok(cases)
}

fn actual_value(metadata: &MediaMetadata, key: &str) -> Result<String, String> {
    let optional = |value: &Option<String>| value.clone().unwrap_or_default();
    Ok(match key {
        "title" => optional(&metadata.title),
        "album" => optional(&metadata.album),
        "artists" => metadata.artists.join("|"),
        "album_artists" => metadata.album_artists.join("|"),
        "composers" => metadata.composers.join("|"),
        "genres" => metadata.genres.join("|"),
        "date" => optional(&metadata.date),
        "track_number" => metadata
            .track_number
            .map(|value| value.to_string())
            .unwrap_or_default(),
        "track_total" => metadata
            .track_total
            .map(|value| value.to_string())
            .unwrap_or_default(),
        "disc_number" => metadata
            .disc_number
            .map(|value| value.to_string())
            .unwrap_or_default(),
        "disc_total" => metadata
            .disc_total
            .map(|value| value.to_string())
            .unwrap_or_default(),
        "comment" => optional(&metadata.comment),
        "lyrics" => optional(&metadata.lyrics),
        "copyright" => optional(&metadata.copyright),
        "encoder" => optional(&metadata.encoder),
        "container" => optional(&metadata.container),
        "audio_codec" => metadata
            .audio_tracks
            .first()
            .and_then(|track| track.codec.clone())
            .unwrap_or_default(),
        "sample_rate" => metadata
            .audio_tracks
            .first()
            .and_then(|track| track.sample_rate)
            .map(|value| value.to_string())
            .unwrap_or_default(),
        "channels" => metadata
            .audio_tracks
            .first()
            .and_then(|track| track.channels)
            .map(|value| value.to_string())
            .unwrap_or_default(),
        "artwork_count" => metadata.artwork.len().to_string(),
        "artwork_mime" => metadata
            .artwork
            .first()
            .and_then(|artwork| artwork.mime_type.clone())
            .unwrap_or_default(),
        "artwork_type" => metadata
            .artwork
            .first()
            .and_then(|artwork| artwork.picture_type)
            .map(|value| value.to_string())
            .unwrap_or_default(),
        "artwork_bytes" => metadata
            .artwork
            .first()
            .map(|artwork| artwork.data.len().to_string())
            .unwrap_or_default(),
        other => return Err(format!("unknown metadata expectation key {other}")),
    })
}

fn check(root: &Path, manifest: &Path) -> Result<(), String> {
    let cases = parse_manifest(manifest)?;
    let mut failures = Vec::new();
    for case in &cases {
        let path = root.join(&case.path);
        let bytes = fs::read(&path).map_err(|error| format!("read {}: {error}", path.display()))?;
        let digest = format!("{:x}", Sha256::digest(&bytes));
        if digest != case.sha256 {
            failures.push(format!("{}: SHA-256 mismatch", case.name));
            continue;
        }
        let metadata = match extract_metadata(&bytes) {
            Ok(metadata) => metadata,
            Err(error) => {
                failures.push(format!("{}: {error}", case.name));
                continue;
            }
        };
        let mut differences = Vec::new();
        for (key, expected) in &case.expected {
            let actual = actual_value(&metadata, key)?;
            if &actual != expected {
                differences.push(format!("{key}={actual:?}, expected {expected:?}"));
            }
        }
        if differences.is_empty() {
            println!("accept {:<28} {} fields", case.name, case.expected.len());
        } else {
            failures.push(format!("{}: {}", case.name, differences.join("; ")));
        }
    }
    if failures.is_empty() {
        println!("metadata conformance passed: {} cases", cases.len());
        Ok(())
    } else {
        Err(format!(
            "metadata conformance failed:\n{}",
            failures.join("\n")
        ))
    }
}

fn media_extension(path: &Path) -> bool {
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    matches!(
        extension.as_str(),
        "aac"
            | "ac3"
            | "ape"
            | "asf"
            | "aif"
            | "aiff"
            | "alac"
            | "avi"
            | "caf"
            | "flac"
            | "m4a"
            | "mka"
            | "mkv"
            | "mov"
            | "mp3"
            | "mp4"
            | "oga"
            | "ogg"
            | "opus"
            | "wav"
            | "webm"
            | "wma"
            | "wmv"
            | "wv"
    )
}

fn collect_files(path: &Path, files: &mut Vec<PathBuf>) -> Result<(), String> {
    if path.is_file() {
        if media_extension(path) {
            files.push(path.to_owned());
        }
        return Ok(());
    }
    for entry in fs::read_dir(path).map_err(|error| format!("read {}: {error}", path.display()))? {
        let entry = entry.map_err(|error| format!("read {} entry: {error}", path.display()))?;
        collect_files(&entry.path(), files)?;
    }
    Ok(())
}

fn sweep(root: &Path) -> Result<(), String> {
    let mut files = Vec::new();
    collect_files(root, &mut files)?;
    files.sort();
    let mut accepted = 0usize;
    let mut rejected = 0usize;
    let mut skipped = 0usize;
    let mut panics = Vec::new();
    for path in &files {
        let length = fs::metadata(path)
            .map_err(|error| format!("stat {}: {error}", path.display()))?
            .len();
        if length > MAX_SWEEP_FILE_BYTES {
            skipped += 1;
            continue;
        }
        let bytes = fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
        match std::panic::catch_unwind(|| extract_metadata(&bytes)) {
            Ok(Ok(_)) => accepted += 1,
            Ok(Err(_)) => rejected += 1,
            Err(_) => panics.push(path.display().to_string()),
        }
    }
    println!(
        "metadata sweep: {} accepted, {} cleanly rejected, {} over size budget, {} panics",
        accepted,
        rejected,
        skipped,
        panics.len()
    );
    if panics.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "metadata parser panicked on:\n{}",
            panics.join("\n")
        ))
    }
}

fn usage() -> ! {
    eprintln!("usage: metadata <check ROOT MANIFEST|sweep CORPUS_ROOT>");
    std::process::exit(2)
}

fn main() {
    let mut arguments = env::args().skip(1);
    let mode = arguments.next().unwrap_or_else(|| usage());
    let result = match mode.as_str() {
        "check" => {
            let root = PathBuf::from(arguments.next().unwrap_or_else(|| usage()));
            let manifest = PathBuf::from(arguments.next().unwrap_or_else(|| usage()));
            if arguments.next().is_some() {
                usage();
            }
            check(&root, &manifest)
        }
        "sweep" => {
            let root = PathBuf::from(arguments.next().unwrap_or_else(|| usage()));
            if arguments.next().is_some() {
                usage();
            }
            sweep(&root)
        }
        _ => usage(),
    };
    if let Err(error) = result {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
