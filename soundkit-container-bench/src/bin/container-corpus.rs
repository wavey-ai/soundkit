use soundkit_audio_demux::{AudioDemuxEvent, AudioTrackDemuxer};
use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::Hasher;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

const CHUNK_SIZES: [usize; 3] = [4 * 1024, 64 * 1024, 4 * 1024 * 1024];
const MAX_PARSER_PEAK_BYTES: usize = 128 * 1024 * 1024;

struct MeasuringAllocator {
    current: AtomicUsize,
    peak: AtomicUsize,
}

impl MeasuringAllocator {
    const fn new() -> Self {
        Self {
            current: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
        }
    }

    fn add_live_bytes(&self, bytes: usize) {
        let current = self.current.fetch_add(bytes, Ordering::Relaxed) + bytes;
        let mut peak = self.peak.load(Ordering::Relaxed);
        while current > peak {
            match self.peak.compare_exchange_weak(
                peak,
                current,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(next) => peak = next,
            }
        }
    }

    fn begin(&self) -> usize {
        let baseline = self.current.load(Ordering::Relaxed);
        self.peak.store(baseline, Ordering::Relaxed);
        baseline
    }

    fn peak_since(&self, baseline: usize) -> usize {
        self.peak.load(Ordering::Relaxed).saturating_sub(baseline)
    }
}

unsafe impl GlobalAlloc for MeasuringAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = System.alloc(layout);
        if !pointer.is_null() {
            self.add_live_bytes(layout.size());
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = System.alloc_zeroed(layout);
        if !pointer.is_null() {
            self.add_live_bytes(layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        System.dealloc(pointer, layout);
        self.current.fetch_sub(layout.size(), Ordering::Relaxed);
    }

    unsafe fn realloc(&self, pointer: *mut u8, old: Layout, new_size: usize) -> *mut u8 {
        let next = System.realloc(pointer, old, new_size);
        if !next.is_null() {
            if new_size >= old.size() {
                self.add_live_bytes(new_size - old.size());
            } else {
                self.current
                    .fetch_sub(old.size() - new_size, Ordering::Relaxed);
            }
        }
        next
    }
}

#[global_allocator]
static ALLOCATOR: MeasuringAllocator = MeasuringAllocator::new();

#[derive(Clone, Debug, PartialEq, Eq)]
struct DemuxSummary {
    configs: usize,
    packets: usize,
    payload_bytes: usize,
    fingerprint: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExpectedResult {
    Accept,
    Reject,
}

#[derive(Debug)]
struct CorpusCase {
    relative_path: PathBuf,
    format: String,
    expected: ExpectedResult,
}

fn hash_event(event: &AudioDemuxEvent, hasher: &mut DefaultHasher) {
    match event {
        AudioDemuxEvent::Config(config) => {
            hasher.write_u8(0);
            hasher.write(format!("{config:?}").as_bytes());
        }
        AudioDemuxEvent::Packet(packet) => {
            hasher.write_u8(1);
            hasher.write(packet.container.as_str().as_bytes());
            hasher.write(packet.codec.as_str().as_bytes());
            hasher.write(packet.format.as_str().as_bytes());
            hasher.write(
                format!(
                    "{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}",
                    packet.track_id,
                    packet.pid,
                    packet.stream_type,
                    packet.timescale,
                    packet.continuity_counter,
                    packet.discontinuity,
                    packet.decode_time,
                    packet.sample_id,
                    packet.start_time,
                    packet.duration,
                    packet.rendering_offset,
                    packet.is_sync,
                    packet.timecode,
                )
                .as_bytes(),
            );
            hasher.write(&packet.data);
            if let Some(raw_data) = &packet.raw_data {
                hasher.write(raw_data);
            }
        }
    }
}

fn demux(data: &[u8], format: &str, chunk_size: usize) -> Result<DemuxSummary, String> {
    let mut demuxer = AudioTrackDemuxer::new_with_format(format)?;
    let mut configs = 0usize;
    let mut packets = 0usize;
    let mut payload_bytes = 0usize;
    let mut fingerprint = DefaultHasher::new();
    for chunk in data.chunks(chunk_size) {
        for event in demuxer.push(chunk)? {
            match &event {
                AudioDemuxEvent::Config(_) => configs += 1,
                AudioDemuxEvent::Packet(packet) => {
                    packets += 1;
                    payload_bytes = payload_bytes.saturating_add(packet.data.len());
                }
            }
            hash_event(&event, &mut fingerprint);
        }
    }
    for event in demuxer.flush()? {
        match &event {
            AudioDemuxEvent::Config(_) => configs += 1,
            AudioDemuxEvent::Packet(packet) => {
                packets += 1;
                payload_bytes = payload_bytes.saturating_add(packet.data.len());
            }
        }
        hash_event(&event, &mut fingerprint);
    }
    Ok(DemuxSummary {
        configs,
        packets,
        payload_bytes,
        fingerprint: fingerprint.finish(),
    })
}

fn parse_manifest(path: &Path) -> Result<Vec<CorpusCase>, String> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("read corpus manifest {}: {error}", path.display()))?;
    let mut cases = Vec::new();
    for (line_index, raw_line) in text.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields = line.split_whitespace().collect::<Vec<_>>();
        if fields.len() != 3 {
            return Err(format!(
                "{}:{} must contain path, format, and accept/reject",
                path.display(),
                line_index + 1
            ));
        }
        let expected = match fields[2] {
            "accept" => ExpectedResult::Accept,
            "reject" => ExpectedResult::Reject,
            value => {
                return Err(format!(
                    "{}:{} has unknown expected result {value}",
                    path.display(),
                    line_index + 1
                ))
            }
        };
        cases.push(CorpusCase {
            relative_path: fields[0].into(),
            format: fields[1].to_owned(),
            expected,
        });
    }
    if cases.is_empty() {
        return Err(format!("corpus manifest {} is empty", path.display()));
    }
    Ok(cases)
}

fn run_case(root: &Path, case: &CorpusCase) -> Result<(), String> {
    let source_path = root.join(&case.relative_path);
    let data = fs::read(&source_path)
        .map_err(|error| format!("read corpus file {}: {error}", source_path.display()))?;
    let mut accepted = Vec::new();
    let mut rejected = Vec::new();
    let mut largest_peak = 0usize;
    for chunk_size in CHUNK_SIZES {
        let baseline = ALLOCATOR.begin();
        let result = demux(&data, &case.format, chunk_size);
        let peak = ALLOCATOR.peak_since(baseline);
        largest_peak = largest_peak.max(peak);
        if peak > MAX_PARSER_PEAK_BYTES {
            return Err(format!(
                "{} at {chunk_size}-byte pushes retained {peak} bytes above its input baseline",
                case.relative_path.display()
            ));
        }
        match result {
            Ok(summary) => accepted.push((chunk_size, summary)),
            Err(error) => rejected.push((chunk_size, error)),
        }
    }

    match case.expected {
        ExpectedResult::Accept => {
            if !rejected.is_empty() {
                return Err(format!(
                    "{} unexpectedly rejected: {rejected:?}",
                    case.relative_path.display()
                ));
            }
            let baseline = &accepted[0].1;
            if baseline.configs == 0 || baseline.packets == 0 || baseline.payload_bytes == 0 {
                return Err(format!(
                    "{} produced no usable audio events: {baseline:?}",
                    case.relative_path.display()
                ));
            }
            for (chunk_size, summary) in &accepted[1..] {
                if summary != baseline {
                    return Err(format!(
                        "{} changed at {chunk_size}-byte pushes: {baseline:?} != {summary:?}",
                        case.relative_path.display()
                    ));
                }
            }
            println!(
                "accept {:<32} configs={} packets={} payload={} peak={}",
                case.relative_path.display(),
                baseline.configs,
                baseline.packets,
                baseline.payload_bytes,
                largest_peak,
            );
        }
        ExpectedResult::Reject => {
            if !accepted.is_empty() {
                return Err(format!(
                    "{} unexpectedly accepted: {accepted:?}",
                    case.relative_path.display()
                ));
            }
            println!(
                "reject {:<32} pushes={} peak={}",
                case.relative_path.display(),
                rejected.len(),
                largest_peak,
            );
        }
    }
    Ok(())
}

fn main() -> Result<(), String> {
    let mut arguments = std::env::args_os().skip(1);
    let root = arguments
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| "usage: container-corpus CORPUS_ROOT MANIFEST".to_owned())?;
    let manifest = arguments
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| "usage: container-corpus CORPUS_ROOT MANIFEST".to_owned())?;
    if arguments.next().is_some() {
        return Err("usage: container-corpus CORPUS_ROOT MANIFEST".to_owned());
    }
    for case in parse_manifest(&manifest)? {
        run_case(&root, &case)?;
    }
    Ok(())
}
