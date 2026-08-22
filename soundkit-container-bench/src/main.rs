use soundkit_audio_demux::{AudioTrackDemuxer, CafAudioIndex, Mp4MediaIndex};
use soundkit_ogg_opus::OggOpusDemuxer;
use std::alloc::{GlobalAlloc, Layout, System};
use std::fs;
use std::hint::black_box;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

const CHUNK_SIZES: [usize; 5] = [1, 188, 4 * 1024, 64 * 1024, 4 * 1024 * 1024];
const LARGE_PUSH_MINIMUM_RATIO: f64 = 0.1;

struct MeasuringAllocator {
    current: AtomicUsize,
    peak: AtomicUsize,
    allocations: AtomicUsize,
}

impl MeasuringAllocator {
    const fn new() -> Self {
        Self {
            current: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            allocations: AtomicUsize::new(0),
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

    fn begin_measurement(&self) -> usize {
        let baseline = self.current.load(Ordering::Relaxed);
        self.peak.store(baseline, Ordering::Relaxed);
        self.allocations.store(0, Ordering::Relaxed);
        baseline
    }
}

unsafe impl GlobalAlloc for MeasuringAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = System.alloc(layout);
        if !pointer.is_null() {
            self.allocations.fetch_add(1, Ordering::Relaxed);
            self.add_live_bytes(layout.size());
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = System.alloc_zeroed(layout);
        if !pointer.is_null() {
            self.allocations.fetch_add(1, Ordering::Relaxed);
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
            self.allocations.fetch_add(1, Ordering::Relaxed);
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

#[derive(Clone, Debug)]
struct Measurement {
    container: &'static str,
    chunk_size: Option<usize>,
    bytes: usize,
    events: usize,
    iterations: usize,
    elapsed: Duration,
    allocations: usize,
    peak_live_bytes: usize,
}

impl Measurement {
    fn throughput_mib_per_second(&self) -> f64 {
        let bytes = self.bytes.saturating_mul(self.iterations) as f64;
        bytes / self.elapsed.as_secs_f64() / (1024.0 * 1024.0)
    }
}

fn fixture(relative: &str) -> Result<Vec<u8>, String> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("testdata")
        .join(relative);
    fs::read(&path).map_err(|error| format!("read {}: {error}", path.display()))
}

fn demux_audio(data: &[u8], format: &str, chunk_size: usize) -> Result<usize, String> {
    let mut demuxer = AudioTrackDemuxer::new_with_format(format)?;
    let mut events = 0usize;
    for chunk in data.chunks(chunk_size) {
        let output = demuxer.push(chunk)?;
        events = events.saturating_add(output.len());
        black_box(output);
    }
    let output = demuxer.flush()?;
    events = events.saturating_add(output.len());
    black_box(output);
    Ok(events)
}

fn demux_ogg(data: &[u8], chunk_size: usize) -> Result<usize, String> {
    let mut demuxer = OggOpusDemuxer::new();
    let mut events = 0usize;
    for chunk in data.chunks(chunk_size) {
        let output = demuxer.add_timed(chunk)?;
        events = events.saturating_add(output.len());
        black_box(output);
    }
    let output = demuxer.finish_timed()?;
    events = events.saturating_add(output.len());
    black_box(output);
    Ok(events)
}

fn index_mp4(data: &[u8]) -> Result<usize, String> {
    let index = Mp4MediaIndex::from_file(data)?;
    let records = index.tracks.len().saturating_add(index.samples.len());
    black_box(index);
    Ok(records)
}

fn index_caf(data: &[u8]) -> Result<usize, String> {
    let index = CafAudioIndex::from_file(data)?;
    let records = 1usize.saturating_add(index.packets.len());
    black_box(index);
    Ok(records)
}

fn measure<F>(
    container: &'static str,
    data: &[u8],
    chunk_size: Option<usize>,
    iterations: usize,
    mut operation: F,
) -> Result<Measurement, String>
where
    F: FnMut() -> Result<usize, String>,
{
    let baseline = ALLOCATOR.begin_measurement();
    let events = operation()?;
    let allocations = ALLOCATOR.allocations.load(Ordering::Relaxed);
    let peak_live_bytes = ALLOCATOR
        .peak
        .load(Ordering::Relaxed)
        .saturating_sub(baseline);

    operation()?;
    let started = Instant::now();
    for _ in 0..iterations {
        black_box(operation()?);
    }
    let elapsed = started.elapsed();
    Ok(Measurement {
        container,
        chunk_size,
        bytes: data.len(),
        events,
        iterations,
        elapsed,
        allocations,
        peak_live_bytes,
    })
}

fn stream_iterations(chunk_size: usize) -> usize {
    match chunk_size {
        1 => 1,
        188 => 3,
        4096 => 10,
        _ => 20,
    }
}

fn chunk_label(chunk_size: Option<usize>) -> String {
    match chunk_size {
        None => "seekable".to_string(),
        Some(1) => "1 B".to_string(),
        Some(188) => "188 B".to_string(),
        Some(4096) => "4 KiB".to_string(),
        Some(65_536) => "64 KiB".to_string(),
        Some(4_194_304) => "4 MiB".to_string(),
        Some(value) => format!("{value} B"),
    }
}

fn print_measurements(measurements: &[Measurement]) {
    println!(
        "{:<10} {:>9} {:>11} {:>11} {:>12} {:>9}",
        "container", "push", "MiB/s", "allocations", "peak bytes", "events"
    );
    for item in measurements {
        println!(
            "{:<10} {:>9} {:>11.2} {:>11} {:>12} {:>9}",
            item.container,
            chunk_label(item.chunk_size),
            item.throughput_mib_per_second(),
            item.allocations,
            item.peak_live_bytes,
            item.events,
        );
    }
}

fn verify_large_pushes(measurements: &[Measurement]) -> Result<(), String> {
    for container in ["fMP4", "WebM", "Ogg", "TS", "M2TS"] {
        let throughput = |chunk_size| {
            measurements
                .iter()
                .find(|item| item.container == container && item.chunk_size == Some(chunk_size))
                .map(Measurement::throughput_mib_per_second)
        };
        let packet = throughput(64 * 1024)
            .ok_or_else(|| format!("missing 64 KiB {container} measurement"))?;
        let large = throughput(4 * 1024 * 1024)
            .ok_or_else(|| format!("missing 4 MiB {container} measurement"))?;
        if large < packet * LARGE_PUSH_MINIMUM_RATIO {
            return Err(format!(
                "{container} 4 MiB throughput {large:.2} MiB/s is below 10% of its 64 KiB throughput {packet:.2} MiB/s"
            ));
        }
    }
    Ok(())
}

fn require_release_build() -> Result<(), String> {
    if cfg!(debug_assertions) {
        return Err(
            "run this benchmark with `cargo run --release -p soundkit-container-bench`".to_string(),
        );
    }
    Ok(())
}

fn main() -> Result<(), String> {
    require_release_build()?;
    let cases = [
        (
            "fMP4",
            fixture("video-compat/never-final/h264-aac-fragmented.mp4")?,
            "fmp4",
        ),
        (
            "WebM",
            fixture("video-compat/never-final/vp9-profile0-opus.webm")?,
            "webm",
        ),
        ("TS", fixture("mpeg-ts/aac-stereo-48k.ts")?, "mpeg-ts"),
        ("M2TS", fixture("mpeg-ts/aac-stereo-48k.m2ts")?, "m2ts"),
    ];
    let ogg = fixture("ogg_opus/A_Tusk_is_used_to_make_costly_gifts_48khz.ogg")?;
    let mp4 = fixture("video-compat/never-final/h264-high-aac.mp4")?;
    let caf = fixture("alac/A_Tusk_is_used_to_make_costly_gifts.caf")?;
    let mut measurements = Vec::new();

    for (container, data, format) in &cases {
        for chunk_size in CHUNK_SIZES {
            measurements.push(measure(
                container,
                data,
                Some(chunk_size),
                stream_iterations(chunk_size),
                || demux_audio(data, format, chunk_size),
            )?);
        }
    }
    for chunk_size in CHUNK_SIZES {
        measurements.push(measure(
            "Ogg",
            &ogg,
            Some(chunk_size),
            stream_iterations(chunk_size),
            || demux_ogg(&ogg, chunk_size),
        )?);
    }
    measurements.push(measure("MP4", &mp4, None, 20, || index_mp4(&mp4))?);
    measurements.push(measure("CAF", &caf, None, 20, || index_caf(&caf))?);

    print_measurements(&measurements);
    verify_large_pushes(&measurements)?;
    Ok(())
}
