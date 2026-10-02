//! Decode every file under the given directories and print one digest line
//! per file and decode path, with the decode time.
//!
//! Usage: corpus_digest [--stream-only] PATH...
//!
//! A PATH is a file or a directory, which is read recursively.
//!
//! Each file is decoded by `decode_audio_file` and by `DecodePipeline` in
//! chunks of 4 KiB and 64 KiB. A digest covers the format and bytes of every
//! frame, then the first error. Compare the output of two builds with `diff`
//! after removing the time column.
use soundkit::audio_types::AudioData;
use soundkit_decoder::{decode_audio_file, Bytes, DecodeError, DecodeOptions, DecodePipeline};
use std::path::{Path, PathBuf};
use std::time::Instant;

const MAX_FILE_BYTES: u64 = 256 * 1024 * 1024;

struct Digest {
    state: u64,
    /// The concatenated sample bytes alone, whatever the frame boundaries.
    pcm: u64,
    frames: usize,
    bytes: usize,
}

impl Digest {
    fn new() -> Self {
        Self {
            state: 0xcbf2_9ce4_8422_2325,
            pcm: 0xcbf2_9ce4_8422_2325,
            frames: 0,
            bytes: 0,
        }
    }

    fn update(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.state ^= u64::from(byte);
            self.state = self.state.wrapping_mul(0x0100_0000_01b3);
        }
    }

    fn frame(&mut self, audio: &AudioData) {
        self.frames += 1;
        self.bytes += audio.data().len();
        self.update(&[audio.bits_per_sample(), audio.channel_count()]);
        self.update(&audio.sampling_rate().to_le_bytes());
        self.update(format!("{:?}{:?}", audio.audio_format(), audio.endianness()).as_bytes());
        self.update(&(audio.data().len() as u64).to_le_bytes());
        self.update(audio.data());
        for &byte in audio.data() {
            self.pcm ^= u64::from(byte);
            self.pcm = self.pcm.wrapping_mul(0x0100_0000_01b3);
        }
    }

    fn line(&self, error: Option<String>) -> String {
        format!(
            "{:016x} pcm {:016x} frames {} bytes {} error {}",
            self.state,
            self.pcm,
            self.frames,
            self.bytes,
            error.unwrap_or_else(|| "-".to_owned())
        )
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
        } else if path
            .metadata()
            .is_ok_and(|meta| meta.len() <= MAX_FILE_BYTES)
        {
            out.push(path);
        }
    }
}

fn whole(data: &[u8]) -> String {
    let mut digest = Digest::new();
    match decode_audio_file(data, DecodeOptions::default()) {
        Ok(decoded) => {
            for frame in &decoded.frames {
                digest.frame(frame);
            }
            digest.line(None)
        }
        Err(error) => digest.line(Some(error.to_string())),
    }
}

fn streamed(data: &[u8], chunk: usize) -> String {
    let mut digest = Digest::new();
    let mut pipeline = DecodePipeline::spawn();
    // The first error the decoder sends. A refused send is recorded only as
    // "send stopped": when the decoder has stopped, which refusal the handle
    // gives depends on timing.
    let mut error = None;
    let mut send_stopped = false;
    let mut take = |output: Result<AudioData, DecodeError>, error: &mut Option<String>| match output
    {
        Ok(frame) => {
            if error.is_none() {
                digest.frame(&frame);
            }
        }
        Err(e) => {
            error.get_or_insert(e.to_string());
        }
    };
    let pieces = data
        .chunks(chunk)
        .map(Bytes::copy_from_slice)
        .chain([Bytes::new()]);
    'send: for piece in pieces {
        let mut stalled_since = None;
        loop {
            match pipeline.send(piece.clone()) {
                Ok(()) => break,
                Err(DecodeError::InputBufferFull) => {
                    let queued = pipeline.queued_input_bytes();
                    let mut received = false;
                    while let Some(output) = pipeline.try_recv() {
                        received = true;
                        take(output, &mut error);
                    }
                    // A decoder that has stopped leaves its queue full.
                    let now = Instant::now();
                    match stalled_since {
                        Some((since, before)) if before == queued && !received => {
                            if now.duration_since(since).as_secs_f64() > 1.0 {
                                send_stopped = true;
                                break 'send;
                            }
                        }
                        _ => stalled_since = Some((now, queued)),
                    }
                    std::thread::yield_now();
                }
                Err(_) => {
                    send_stopped = true;
                    break 'send;
                }
            }
        }
    }
    if send_stopped {
        // Collect what the decoder sent before it stopped.
        let deadline = Instant::now();
        while deadline.elapsed().as_secs_f64() < 0.5 {
            match pipeline.try_recv() {
                Some(output) => take(output, &mut error),
                None => std::thread::yield_now(),
            }
        }
        pipeline.cancel();
    } else {
        while let Some(output) = pipeline.recv() {
            take(output, &mut error);
        }
    }
    if error.is_none() && send_stopped {
        error = Some("send stopped".to_owned());
    }
    digest.line(error)
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let stream_only = args.first().is_some_and(|arg| arg == "--stream-only");
    if stream_only {
        args.remove(0);
    }
    let mut paths = Vec::new();
    for arg in &args {
        let path = Path::new(arg);
        if path.is_file() {
            paths.push(path.to_path_buf());
        } else {
            files(path, &mut paths);
        }
    }
    for path in paths {
        let Ok(data) = std::fs::read(&path) else {
            continue;
        };
        let name = path.display();
        if !stream_only {
            let start = Instant::now();
            let line = whole(&data);
            println!(
                "{name}\twhole\t{line}\t{:.3}",
                start.elapsed().as_secs_f64()
            );
        }
        for chunk in [4_096usize, 65_536] {
            let start = Instant::now();
            let line = streamed(&data, chunk);
            println!(
                "{name}\tstream{chunk}\t{line}\t{:.3}",
                start.elapsed().as_secs_f64()
            );
        }
    }
}
