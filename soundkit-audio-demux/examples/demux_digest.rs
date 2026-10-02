//! Push every file under the given directories through the streaming
//! demuxers in several chunk sizes and print a digest of the events and the
//! first error, with the time.
//!
//! Usage: demux_digest DIR...
use soundkit_audio_demux::{AudioTrackDemuxer, Mp4MediaDemuxer, MxfMediaDemuxer};
use std::fmt::{Debug, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// FNV-1a over formatted text, so events are digested without a copy.
struct Fnv(u64);

impl Write for Fnv {
    fn write_str(&mut self, text: &str) -> std::fmt::Result {
        for &byte in text.as_bytes() {
            self.0 ^= u64::from(byte);
            self.0 = self.0.wrapping_mul(0x0100_0000_01b3);
        }
        Ok(())
    }
}

fn files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<_> = entries.flatten().map(|entry| entry.path()).collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            files(&path, out);
        } else {
            out.push(path);
        }
    }
}

fn run<E: Debug>(
    data: &[u8],
    chunk: usize,
    mut push: impl FnMut(&[u8]) -> Result<Vec<E>, String>,
) -> String {
    let mut digest = Fnv(0xcbf2_9ce4_8422_2325);
    let mut events = 0usize;
    let mut error = None;
    for piece in data.chunks(chunk) {
        match push(piece) {
            Ok(batch) => {
                for event in batch {
                    events += 1;
                    let _ = write!(digest, "{event:?}");
                }
            }
            Err(e) => {
                error = Some(e);
                break;
            }
        }
    }
    format!(
        "{:016x} events {events} error {}",
        digest.0,
        error.unwrap_or_else(|| "-".into())
    )
}

fn main() {
    let mut paths = Vec::new();
    for dir in std::env::args().skip(1) {
        files(Path::new(&dir), &mut paths);
    }
    for path in paths {
        let Ok(data) = std::fs::read(&path) else {
            continue;
        };
        let mut chunks = vec![4_093usize, 65_536, 4 * 1024 * 1024];
        if data.len() <= 512 * 1024 {
            chunks.insert(0, 1);
        }
        for chunk in chunks {
            let name = path.display();
            let start = Instant::now();
            let mut demuxer = Mp4MediaDemuxer::new();
            let mut line = run(&data, chunk, |b| demuxer.push(b));
            line.push_str(&format!(" flush {}", digest_events(demuxer.flush())));
            println!(
                "{name}\tfmp4/{chunk}\t{line}\t{:.3}",
                start.elapsed().as_secs_f64()
            );

            let start = Instant::now();
            let mut demuxer = MxfMediaDemuxer::new();
            let mut line = run(&data, chunk, |b| demuxer.push(b));
            line.push_str(&format!(" flush {}", digest_events(demuxer.flush())));
            println!(
                "{name}\tmxf/{chunk}\t{line}\t{:.3}",
                start.elapsed().as_secs_f64()
            );

            let start = Instant::now();
            let mut demuxer = AudioTrackDemuxer::new_auto();
            let mut line = run(&data, chunk, |b| demuxer.push(b));
            line.push_str(&format!(" flush {}", digest_events(demuxer.flush())));
            println!(
                "{name}\tauto/{chunk}\t{line}\t{:.3}",
                start.elapsed().as_secs_f64()
            );
        }
    }
}

fn digest_events<E: Debug>(result: Result<Vec<E>, String>) -> String {
    match result {
        Ok(events) => {
            let mut digest = Fnv(0xcbf2_9ce4_8422_2325);
            for event in &events {
                let _ = write!(digest, "{event:?}");
            }
            format!("{:016x} events {}", digest.0, events.len())
        }
        Err(error) => format!("error {error}"),
    }
}
