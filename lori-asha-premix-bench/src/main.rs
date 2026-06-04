use bytes::Bytes;
use libopus_rust::{
    decoder::Decoder as PureDecoder,
    encoder::{Application as PureApplication, Encoder as PureEncoder, OPUS_SET_BITRATE_REQUEST},
};
use opus_sys;
use soundkit::audio_bytes::i16le_to_i16;
use soundkit::audio_packet::{Decoder as SoundkitDecoderTrait, Encoder as SoundkitEncoderTrait};
use soundkit_decoder::{DecodeOptions, DecodePipeline};
use soundkit_opus::{OpusDecoder as SoundkitDecoder, OpusEncoder as SoundkitEncoder};
use std::collections::{HashSet, VecDeque};
use std::env;
use std::fs;
use std::mem::MaybeUninit;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const TARGET_SAMPLE_RATE: u32 = 48_000;
const TARGET_CHANNELS: u8 = 1;
const TARGET_BITS: u8 = 16;
const DEFAULT_FRAME_SIZE: usize = 960;
const DEFAULT_BITRATE: u32 = 128_000;

const AUDIO_EXTS: [&str; 19] = [
    "aac", "ac3", "aif", "aiff", "alac", "amr", "flac", "gsm", "m4a", "m4v", "mp3", "ogg", "oga",
    "opus", "opusfile", "raw", "wav", "webm", "wma",
];

#[derive(Debug)]
struct TrackData {
    path: PathBuf,
    samples: Vec<i16>,
    duration_seconds: f64,
}

#[derive(Debug, Clone, Copy)]
struct QualityMetrics {
    snr_db: f64,
    rmse: f64,
    mae: f64,
    max_abs_error: f64,
    compared_samples: usize,
    length_delta_samples: isize,
}

#[derive(Debug)]
struct BackendResult {
    encode_time: Duration,
    decode_time: Duration,
    encoded_bytes: usize,
    decoded_bytes: usize,
    metrics: QualityMetrics,
}

#[derive(Debug, Default)]
struct BackendAggregate {
    encoded_bytes: usize,
    decoded_bytes: usize,
    encode_time: Duration,
    decode_time: Duration,
    tracks: usize,
    snr_db: f64,
    rmse: f64,
    mae: f64,
    max_abs_error: f64,
    length_delta_samples: isize,
    total_duration_seconds: f64,
}

impl BackendAggregate {
    fn add(&mut self, result: &BackendResult, duration_seconds: f64) {
        self.encoded_bytes += result.encoded_bytes;
        self.decoded_bytes += result.decoded_bytes;
        self.encode_time += result.encode_time;
        self.decode_time += result.decode_time;
        self.tracks += 1;
        self.total_duration_seconds += duration_seconds;

        if self.tracks == 1 {
            self.snr_db = result.metrics.snr_db;
            self.rmse = result.metrics.rmse;
            self.mae = result.metrics.mae;
            self.max_abs_error = result.metrics.max_abs_error;
        } else {
            self.snr_db += result.metrics.snr_db;
            self.rmse += result.metrics.rmse;
            self.mae += result.metrics.mae;
            self.max_abs_error = self.max_abs_error.max(result.metrics.max_abs_error);
        }

        self.length_delta_samples += result.metrics.length_delta_samples;
    }

    fn mean_snr(&self) -> f64 {
        if self.tracks == 0 {
            0.0
        } else {
            self.snr_db / self.tracks as f64
        }
    }

    fn mean_rmse(&self) -> f64 {
        if self.tracks == 0 {
            0.0
        } else {
            self.rmse / self.tracks as f64
        }
    }

    fn mean_mae(&self) -> f64 {
        if self.tracks == 0 {
            0.0
        } else {
            self.mae / self.tracks as f64
        }
    }

    fn bitrate_kbps(&self) -> f64 {
        if self.total_duration_seconds == 0.0 {
            0.0
        } else {
            self.encoded_bytes as f64 * 8.0 / (self.total_duration_seconds * 1000.0)
        }
    }

    fn decode_bitrate_kbps(&self) -> f64 {
        if self.total_duration_seconds == 0.0 {
            0.0
        } else {
            self.decoded_bytes as f64 * 8.0 / (self.total_duration_seconds * 1000.0)
        }
    }

    fn encode_rtf(&self) -> f64 {
        if self.total_duration_seconds == 0.0 {
            0.0
        } else {
            self.encode_time.as_secs_f64() / self.total_duration_seconds
        }
    }

    fn decode_rtf(&self) -> f64 {
        if self.total_duration_seconds == 0.0 {
            0.0
        } else {
            self.decode_time.as_secs_f64() / self.total_duration_seconds
        }
    }
}

#[derive(Debug)]
struct Config {
    roots: Vec<PathBuf>,
    query_terms: Vec<String>,
    bitrate: u32,
    frame_size: usize,
    show_help: bool,
}

fn main() -> Result<(), String> {
    let config = parse_args()?;

    if config.show_help {
        return Ok(());
    }

    let track_paths = discover_tracks(&config.roots, &config.query_terms)?;
    if track_paths.is_empty() {
        println!(
            "No matching tracks found for {:?} under {:?} and {:?}.",
            config.query_terms,
            config.roots.first().map(|p| p.to_string_lossy()),
            config.roots.get(1).map(|p| p.to_string_lossy())
        );
        return Ok(());
    }

    println!("Found {} candidate tracks", track_paths.len());

    let mut agg_soundkit = BackendAggregate::default();
    let mut agg_pure = BackendAggregate::default();
    let mut agg_c = BackendAggregate::default();

    println!(
        "\n{:<68} {:>7} {:>8} {:>8} {:>9} {:>8} {:>8} {:>8} {:>10} {:>8} | {:>8} {:>8} {:>9} {:>8} {:>8} {:>8} {:>10} {:>10} | {:>8} {:>8} {:>9} {:>8} {:>8} {:>8} {:>10} {:>10}",
        "Track",
        "duration",
        "sk_enc",
        "sk_dec",
        "sk_kbps",
        "sk_decKB",
        "sk_snr",
        "sk_rmse",
        "sk_mae",
        "dlen",
        "pr_enc",
        "pr_dec",
        "pr_kbps",
        "pr_decKB",
        "pr_snr",
        "pr_rmse",
        "pr_mae",
        "dlen",
        "cb_enc",
        "cb_dec",
        "cb_kbps",
        "cb_decKB",
        "cb_snr",
        "cb_rmse",
        "cb_mae",
        "dlen",
    );

    for path in track_paths {
        let track = match decode_for_benchmark(&path) {
            Ok(track) => track,
            Err(err) => {
                println!("{:<70} decode-failed: {}", path.display(), err);
                continue;
            }
        };

        let sk = run_soundkit_benchmark(&track, config.frame_size, config.bitrate);
        let pure = run_pure_libopus_benchmark(&track, config.frame_size, config.bitrate);
        let cb = run_c_libopus_benchmark(&track, config.frame_size, config.bitrate);

        let sk_kbps = kbps_from_stats(
            sk.as_ref().map(|r| r.encoded_bytes).unwrap_or(0),
            track.duration_seconds,
        );
        let sk_dec_kbps = kbps_from_sample_bytes(
            sk.as_ref().map(|r| r.decoded_bytes).unwrap_or(0),
            track.duration_seconds,
        );
        let pr_kbps = kbps_from_stats(
            pure.as_ref().map(|r| r.encoded_bytes).unwrap_or(0),
            track.duration_seconds,
        );
        let pr_dec_kbps = kbps_from_sample_bytes(
            pure.as_ref().map(|r| r.decoded_bytes).unwrap_or(0),
            track.duration_seconds,
        );
        let cb_kbps = kbps_from_stats(
            cb.as_ref().map(|r| r.encoded_bytes).unwrap_or(0),
            track.duration_seconds,
        );
        let cb_dec_kbps = kbps_from_sample_bytes(
            cb.as_ref().map(|r| r.decoded_bytes).unwrap_or(0),
            track.duration_seconds,
        );

        let sk_row = match &sk {
            Ok(result) => format!(
                "{:>8.3} {:>8.3} {:>9.2} {:>8.2} {:>8.2} {:>7.2} {:>10} {:>10}",
                result.encode_time.as_secs_f64() / track.duration_seconds,
                result.decode_time.as_secs_f64() / track.duration_seconds,
                sk_kbps,
                sk_dec_kbps,
                result.metrics.snr_db,
                result.metrics.rmse,
                result.metrics.mae,
                result.metrics.length_delta_samples,
            ),
            Err(_) => format!(
                "{:>8} {:>8} {:>9} {:>8} {:>8} {:>7} {:>10} {:>10}",
                "--", "--", "--", "--", "--", "--", "--", "--"
            ),
        };

        let pure_row = match &pure {
            Ok(result) => format!(
                "{:>8.3} {:>8.3} {:>9.2} {:>8.2} {:>8.2} {:>7.2} {:>10} {:>10}",
                result.encode_time.as_secs_f64() / track.duration_seconds,
                result.decode_time.as_secs_f64() / track.duration_seconds,
                pr_kbps,
                pr_dec_kbps,
                result.metrics.snr_db,
                result.metrics.rmse,
                result.metrics.mae,
                result.metrics.length_delta_samples,
            ),
            Err(_) => {
                format!(
                    "{:>8} {:>8} {:>9} {:>8} {:>8} {:>7} {:>10} {:>10}",
                    "--", "--", "--", "--", "--", "--", "--", "--"
                )
            }
        };

        let cb_row = match &cb {
            Ok(result) => format!(
                "{:>8.3} {:>8.3} {:>9.2} {:>8.2} {:>8.2} {:>7.2} {:>10} {:>10}",
                result.encode_time.as_secs_f64() / track.duration_seconds,
                result.decode_time.as_secs_f64() / track.duration_seconds,
                cb_kbps,
                cb_dec_kbps,
                result.metrics.snr_db,
                result.metrics.rmse,
                result.metrics.mae,
                result.metrics.length_delta_samples,
            ),
            Err(_) => {
                format!(
                    "{:>8} {:>8} {:>9} {:>8} {:>8} {:>7} {:>10} {:>10}",
                    "--", "--", "--", "--", "--", "--", "--", "--"
                )
            }
        };

        println!(
            "{:<70} {:>8.2} {} {} {}",
            path.display(),
            track.duration_seconds,
            sk_row,
            pure_row,
            cb_row
        );

        if let Ok(result) = sk {
            agg_soundkit.add(&result, track.duration_seconds);
        }
        if let Ok(result) = pure {
            agg_pure.add(&result, track.duration_seconds);
        }
        if let Ok(result) = cb {
            agg_c.add(&result, track.duration_seconds);
        }
    }

    println!("\nAggregate");
    println!("Soundkit-opus:");
    print_aggregate(&agg_soundkit);
    println!("libopus-rs:");
    print_aggregate(&agg_pure);
    println!("libopus C:");
    print_aggregate(&agg_c);

    Ok(())
}

fn parse_args() -> Result<Config, String> {
    let mut roots: Vec<PathBuf> = Vec::new();
    let mut bitrate = DEFAULT_BITRATE;
    let mut frame_ms = 20.0f64;
    let mut query = String::from("lori asha premix");
    let mut show_help = false;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                println!("Usage: lori-asha-premix-bench [--dir <path>] [--bitrate <bps>] [--frame-ms <milliseconds>] [--query <text>]");
                println!("  --dir   Add a directory to scan (repeatable). Defaults to ~/Downloads and ~/Documents.");
                println!(
                    "  --bitrate    Opus target bitrate in bps. Default {}.",
                    DEFAULT_BITRATE
                );
                println!(
                    "  --frame-ms   Opus frame size in ms. Allowed: 2.5, 5, 10, 20. Default 20."
                );
                println!(
                    "  --query      Case-insensitive match terms for file path. Default: {}",
                    query
                );
                show_help = true;
            }
            "--dir" | "--root" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("missing value for {arg}. usage: --dir <directory>"))?;
                roots.push(PathBuf::from(value));
            }
            "--bitrate" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("missing value for {arg}. usage: --bitrate <bps>"))?;
                bitrate = value
                    .parse::<u32>()
                    .map_err(|e| format!("invalid bitrate '{}': {}", value, e))?;
            }
            "--frame-ms" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("missing value for {arg}. usage: --frame-ms <ms>"))?;
                frame_ms = value
                    .parse::<f64>()
                    .map_err(|e| format!("invalid frame-ms '{}': {}", value, e))?;
            }
            "--query" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("missing value for {arg}. usage: --query <text>"))?;
                query = value;
            }
            _ => {
                return Err(format!("unknown argument: {arg}"));
            }
        }
    }

    let frame_size = ((TARGET_SAMPLE_RATE as f64 * frame_ms) / 1000.0).round() as usize;
    if frame_size == 0 {
        return Err("frame-ms too small for 48kHz output".to_string());
    }

    if ![120, 240, 480, 960].contains(&frame_size) {
        return Err(format!(
            "frame-size for 48kHz must be one of [2.5, 5, 10, 20]ms => [120, 240, 480, 960] samples; got {}",
            frame_size
        ));
    }

    if bitrate == 0 {
        return Err("bitrate must be greater than 0".to_string());
    }

    if roots.is_empty() {
        let home = env::var("HOME").map_err(|_| "HOME not set".to_string())?;
        roots.push(PathBuf::from(&home).join("Downloads"));
        roots.push(PathBuf::from(&home).join("Documents"));
    }

    let query_terms = query
        .split_whitespace()
        .map(|item| item.to_ascii_lowercase())
        .collect();

    Ok(Config {
        roots,
        query_terms,
        bitrate,
        frame_size,
        show_help,
    })
}

fn discover_tracks(roots: &[PathBuf], query_terms: &[String]) -> Result<Vec<PathBuf>, String> {
    let mut todo = VecDeque::from(roots.to_vec());
    let mut out = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    while let Some(path) = todo.pop_front() {
        if !path.exists() {
            continue;
        }

        let entries = fs::read_dir(&path)
            .map_err(|e| format!("failed reading directory {}: {}", path.to_string_lossy(), e))?;

        for entry in entries {
            let entry = entry.map_err(|e| e.to_string())?;
            let child = entry.path();

            if child.is_dir() {
                todo.push_back(child);
                continue;
            }

            if !child.is_file() {
                continue;
            }

            if !is_audio_like(&child) {
                continue;
            }

            if !path_matches_query(&child, query_terms) {
                continue;
            }

            let key = child.to_string_lossy().to_ascii_lowercase();
            if seen.insert(key) {
                out.push(child);
            }
        }
    }

    out.sort();
    Ok(out)
}

fn is_audio_like(path: &Path) -> bool {
    let Some(ext) = path.extension().and_then(|ext| ext.to_str()) else {
        return false;
    };

    AUDIO_EXTS
        .iter()
        .any(|candidate| ext.eq_ignore_ascii_case(candidate))
}

fn path_matches_query(path: &Path, query_terms: &[String]) -> bool {
    let lowered = path.to_string_lossy().to_ascii_lowercase();
    query_terms.iter().all(|term| lowered.contains(term))
}

fn decode_for_benchmark(path: &Path) -> Result<TrackData, String> {
    let data = fs::read(path).map_err(|e| format!("read failed {}: {}", path.display(), e))?;

    let options = DecodeOptions {
        output_bits_per_sample: Some(TARGET_BITS),
        output_sample_rate: Some(TARGET_SAMPLE_RATE),
        output_channels: Some(TARGET_CHANNELS),
    };

    let mut pipeline = DecodePipeline::spawn_with_options(options);
    pipeline
        .send(Bytes::from(data))
        .map_err(|e| format!("decode send failed {}: {}", path.display(), e))?;
    pipeline
        .send(Bytes::new())
        .map_err(|e| format!("decode eof send failed {}: {}", path.display(), e))?;

    let mut samples = Vec::new();
    loop {
        match pipeline.recv() {
            Some(Ok(audio_data)) => {
                let mut frame = i16le_to_i16(audio_data.data());

                if audio_data.channel_count() > 1 {
                    let channels = audio_data.channel_count() as usize;
                    let mut mono = Vec::with_capacity(frame.len() / channels);

                    for frame_idx in 0..(frame.len() / channels) {
                        let mut sum = 0i32;
                        for channel in 0..channels {
                            sum += frame[frame_idx * channels + channel] as i32;
                        }
                        let sample = (sum as f32 / channels as f32).round() as i32;
                        mono.push(sample.clamp(i16::MIN as i32, i16::MAX as i32) as i16);
                    }

                    frame = mono;
                }

                samples.extend_from_slice(&frame);
            }
            Some(Err(e)) => return Err(format!("decode error {}: {}", path.display(), e)),
            None => break,
        }
    }

    if samples.is_empty() {
        return Err(format!("no decoded output for {}", path.display()));
    }

    let sample_rate = TARGET_SAMPLE_RATE as f64;
    let duration_seconds = samples.len() as f64 / sample_rate;

    Ok(TrackData {
        path: path.to_path_buf(),
        samples,
        duration_seconds,
    })
}

fn run_soundkit_benchmark(
    track: &TrackData,
    frame_size: usize,
    bitrate: u32,
) -> Result<BackendResult, String> {
    let packets = encode_with_soundkit(track, frame_size, bitrate)?;

    let decode_result = decode_with_soundkit(track, &packets.packets)?;
    let metrics = compute_quality(&track.samples, &decode_result.samples);
    Ok(BackendResult {
        encode_time: packets.encode_time,
        decode_time: decode_result.decode_time,
        encoded_bytes: packets.encoded_bytes,
        decoded_bytes: decode_result.decoded_bytes,
        metrics,
    })
}

fn run_pure_libopus_benchmark(
    track: &TrackData,
    frame_size: usize,
    bitrate: u32,
) -> Result<BackendResult, String> {
    let packets = encode_with_pure_libopus(track, frame_size, bitrate)?;

    let decode_result = decode_with_pure_libopus(track, &packets.packets)?;
    let metrics = compute_quality(&track.samples, &decode_result.samples);

    Ok(BackendResult {
        encode_time: packets.encode_time,
        decode_time: decode_result.decode_time,
        encoded_bytes: packets.encoded_bytes,
        decoded_bytes: decode_result.decoded_bytes,
        metrics,
    })
}

fn run_c_libopus_benchmark(
    track: &TrackData,
    frame_size: usize,
    bitrate: u32,
) -> Result<BackendResult, String> {
    let packets = encode_with_c_libopus(track, frame_size, bitrate)?;

    let decode_result = decode_with_c_libopus(track, &packets.packets)?;
    let metrics = compute_quality(&track.samples, &decode_result.samples);
    Ok(BackendResult {
        encode_time: packets.encode_time,
        decode_time: decode_result.decode_time,
        encoded_bytes: packets.encoded_bytes,
        decoded_bytes: decode_result.decoded_bytes,
        metrics,
    })
}

struct PacketEncodeResult {
    packets: Vec<Vec<u8>>,
    encode_time: Duration,
    encoded_bytes: usize,
}

struct PacketDecodeResult {
    samples: Vec<i16>,
    decoded_bytes: usize,
    decode_time: Duration,
}

fn encode_with_soundkit(
    track: &TrackData,
    frame_size: usize,
    bitrate: u32,
) -> Result<PacketEncodeResult, String> {
    let mut encoder = SoundkitEncoder::new(
        TARGET_SAMPLE_RATE,
        TARGET_BITS as u32,
        TARGET_CHANNELS as u32,
        frame_size as u32,
        bitrate,
    );
    encoder
        .init()
        .map_err(|e| format!("soundkit-opus encoder init failed: {}", e))?;

    let channels = TARGET_CHANNELS as usize;
    let frame_samples = frame_size * channels;
    let mut packets = Vec::new();
    let mut encode_time = Duration::ZERO;
    let mut encoded_bytes = 0usize;
    let mut scratch_output = vec![0u8; 4096];

    let mut offset = 0usize;
    while offset < track.samples.len() {
        if offset + frame_samples <= track.samples.len() {
            let chunk = &track.samples[offset..offset + frame_samples];
            let start = Instant::now();
            let written = encoder
                .encode_i16(chunk, &mut scratch_output)
                .map_err(|e| format!("soundkit-opus encode failed: {}", e))?;
            encode_time += start.elapsed();

            if written > 0 {
                packets.push(scratch_output[..written].to_vec());
                encoded_bytes += written;
            }
        } else {
            let mut padded = vec![0i16; frame_samples];
            let tail = &track.samples[offset..track.samples.len()];
            if !tail.is_empty() {
                padded[..tail.len()].copy_from_slice(tail);
                let start = Instant::now();
                let written = encoder
                    .encode_i16(&padded, &mut scratch_output)
                    .map_err(|e| format!("soundkit-opus encode failed: {}", e))?;
                encode_time += start.elapsed();

                if written > 0 {
                    packets.push(scratch_output[..written].to_vec());
                    encoded_bytes += written;
                }
            }
        }

        offset += frame_samples;
    }

    if packets.is_empty() {
        return Err("soundkit-opus produced no packets".to_string());
    }

    Ok(PacketEncodeResult {
        packets,
        encode_time,
        encoded_bytes,
    })
}

fn decode_with_soundkit(
    _track: &TrackData,
    packets: &[Vec<u8>],
) -> Result<PacketDecodeResult, String> {
    let mut decoder = SoundkitDecoder::new(TARGET_SAMPLE_RATE as usize, TARGET_CHANNELS as usize);
    decoder
        .init()
        .map_err(|e| format!("soundkit-opus decoder init failed: {}", e))?;

    let mut decoded = Vec::new();
    let mut decode_time = Duration::ZERO;
    let mut scratch = vec![0i16; 6_144];

    for packet in packets {
        let start = Instant::now();
        let samples_written = decoder
            .decode_i16(packet, &mut scratch, false)
            .map_err(|e| format!("soundkit-opus decode failed: {}", e))?;
        decode_time += start.elapsed();

        if samples_written > 0 {
            let count = samples_written * TARGET_CHANNELS as usize;
            decoded.extend_from_slice(&scratch[..count]);
        }
    }

    if decoded.is_empty() {
        return Err("soundkit-opus produced no decoded samples".to_string());
    }

    let decoded_bytes = decoded.len() * std::mem::size_of::<i16>();
    Ok(PacketDecodeResult {
        samples: decoded,
        decoded_bytes,
        decode_time,
    })
}

fn encode_with_pure_libopus(
    track: &TrackData,
    frame_size: usize,
    bitrate: u32,
) -> Result<PacketEncodeResult, String> {
    let mut encoder = PureEncoder::create(
        TARGET_SAMPLE_RATE as usize,
        TARGET_CHANNELS as usize,
        1,
        0,
        &[0u8, 1u8],
        PureApplication::Audio,
    )
    .map_err(|e| format!("libopus-rs encoder init failed: {}", e))?;
    encoder
        .set_option(OPUS_SET_BITRATE_REQUEST, bitrate)
        .map_err(|e| format!("libopus-rs set bitrate failed: {}", e))?;

    let mut packets = Vec::new();
    let mut scratch_output = vec![0u8; 6_144];
    let mut encode_time = Duration::ZERO;
    let mut encoded_bytes = 0usize;

    let channels = TARGET_CHANNELS as usize;
    let frame_samples = frame_size * channels;
    let mut offset = 0usize;
    while offset < track.samples.len() {
        if offset + frame_samples <= track.samples.len() {
            let chunk = &track.samples[offset..offset + frame_samples];
            let start = Instant::now();
            let packet_len = encoder
                .encode(chunk, &mut scratch_output)
                .map_err(|e| format!("libopus-rs encode failed: {}", e))?;
            encode_time += start.elapsed();

            if packet_len > 0 {
                encoded_bytes += packet_len;
                packets.push(scratch_output[..packet_len].to_vec());
            }
        } else {
            let mut padded = vec![0i16; frame_samples];
            let tail = &track.samples[offset..];
            if !tail.is_empty() {
                padded[..tail.len()].copy_from_slice(tail);
                let start = Instant::now();
                let packet_len = encoder
                    .encode(&padded[..], &mut scratch_output)
                    .map_err(|e| format!("libopus-rs encode failed: {}", e))?;
                encode_time += start.elapsed();

                if packet_len > 0 {
                    encoded_bytes += packet_len;
                    packets.push(scratch_output[..packet_len].to_vec());
                }
            }
        }

        offset += frame_samples;
    }

    if packets.is_empty() {
        return Err("libopus-rs produced no packets".to_string());
    }

    Ok(PacketEncodeResult {
        packets,
        encode_time,
        encoded_bytes,
    })
}

fn decode_with_pure_libopus(
    _track: &TrackData,
    packets: &[Vec<u8>],
) -> Result<PacketDecodeResult, String> {
    let mut decoder = PureDecoder::create(
        TARGET_SAMPLE_RATE as usize,
        TARGET_CHANNELS as usize,
        1,
        1,
        &[0u8, 1u8],
    )
    .map_err(|e| format!("libopus-rs decoder init failed: {}", e))?;

    let mut decoded = Vec::new();
    let mut scratch = vec![0i16; 6_144];
    let mut decode_time = Duration::ZERO;

    for packet in packets {
        let start = Instant::now();
        let samples_written = decoder
            .decode(&packet[..], &mut scratch[..], false)
            .map_err(|e| format!("libopus-rs decode failed: {}", e))?;
        decode_time += start.elapsed();
        if samples_written > 0 {
            let count = samples_written * TARGET_CHANNELS as usize;
            decoded.extend_from_slice(&scratch[..count]);
        }
    }

    if decoded.is_empty() {
        return Err("libopus-rs produced no decoded samples".to_string());
    }

    let decoded_bytes = decoded.len() * std::mem::size_of::<i16>();
    Ok(PacketDecodeResult {
        samples: decoded,
        decoded_bytes,
        decode_time,
    })
}

fn encode_with_c_libopus(
    track: &TrackData,
    frame_size: usize,
    bitrate: u32,
) -> Result<PacketEncodeResult, String> {
    let mut err = MaybeUninit::uninit();
    let mut encoder = unsafe {
        opus_sys::opus_encoder_create(
            TARGET_SAMPLE_RATE as i32,
            TARGET_CHANNELS as i32,
            opus_sys::OPUS_APPLICATION_AUDIO as i32,
            err.as_mut_ptr(),
        )
    };

    let init_err = unsafe { err.assume_init() };
    if init_err != opus_sys::OPUS_OK as i32 || encoder.is_null() {
        return Err(format!("libopus C encoder init failed: {}", init_err));
    }

    let bitrate_result = unsafe {
        opus_sys::opus_encoder_ctl(
            encoder,
            opus_sys::OPUS_SET_BITRATE_REQUEST as i32,
            bitrate as i32,
        )
    };
    if bitrate_result != opus_sys::OPUS_OK as i32 {
        unsafe { opus_sys::opus_encoder_destroy(encoder) };
        return Err(format!("libopus C set bitrate failed: {}", bitrate_result));
    }

    let result = (|| -> Result<PacketEncodeResult, String> {
        let mut packets = Vec::new();
        let mut scratch_output = vec![0u8; 6_144];
        let mut encode_time = Duration::ZERO;
        let mut encoded_bytes = 0usize;

        let channels = TARGET_CHANNELS as usize;
        let frame_samples = frame_size * channels;
        let mut offset = 0usize;
        while offset < track.samples.len() {
            if offset + frame_samples <= track.samples.len() {
                let chunk = &track.samples[offset..offset + frame_samples];
                let start = Instant::now();
                let packet_len = unsafe {
                    opus_sys::opus_encode(
                        encoder,
                        chunk.as_ptr(),
                        frame_size as i32,
                        scratch_output.as_mut_ptr(),
                        scratch_output.len() as i32,
                    )
                };
                encode_time += start.elapsed();

                if packet_len < 0 {
                    return Err(format!("libopus C encode failed: {}", packet_len));
                }
                if packet_len > 0 {
                    let packet_len = packet_len as usize;
                    encoded_bytes += packet_len;
                    packets.push(scratch_output[..packet_len].to_vec());
                }
            } else {
                let mut padded = vec![0i16; frame_samples];
                let tail = &track.samples[offset..];
                if !tail.is_empty() {
                    padded[..tail.len()].copy_from_slice(tail);
                    let start = Instant::now();
                    let packet_len = unsafe {
                        opus_sys::opus_encode(
                            encoder,
                            padded.as_ptr(),
                            frame_size as i32,
                            scratch_output.as_mut_ptr(),
                            scratch_output.len() as i32,
                        )
                    };
                    encode_time += start.elapsed();

                    if packet_len < 0 {
                        return Err(format!("libopus C encode failed: {}", packet_len));
                    }
                    if packet_len > 0 {
                        let packet_len = packet_len as usize;
                        encoded_bytes += packet_len;
                        packets.push(scratch_output[..packet_len].to_vec());
                    }
                }
            }

            offset += frame_samples;
        }

        if packets.is_empty() {
            return Err("libopus C produced no packets".to_string());
        }

        Ok(PacketEncodeResult {
            packets,
            encode_time,
            encoded_bytes,
        })
    })();

    unsafe { opus_sys::opus_encoder_destroy(encoder) };

    result
}

fn decode_with_c_libopus(
    _track: &TrackData,
    packets: &[Vec<u8>],
) -> Result<PacketDecodeResult, String> {
    let mut err = MaybeUninit::uninit();
    let decoder = unsafe {
        opus_sys::opus_decoder_create(
            TARGET_SAMPLE_RATE as i32,
            TARGET_CHANNELS as i32,
            err.as_mut_ptr(),
        )
    };
    let init_err = unsafe { err.assume_init() };
    if init_err != opus_sys::OPUS_OK as i32 || decoder.is_null() {
        return Err(format!("libopus C decoder init failed: {}", init_err));
    }

    let result = (|| -> Result<PacketDecodeResult, String> {
        let mut decoded = Vec::new();
        let mut scratch = vec![0i16; 6_144];
        let mut decode_time = Duration::ZERO;
        let frame_capacity = (scratch.len() / TARGET_CHANNELS as usize) as i32;

        for packet in packets {
            let start = Instant::now();
            let samples_written = unsafe {
                opus_sys::opus_decode(
                    decoder,
                    packet.as_ptr(),
                    packet.len() as i32,
                    scratch.as_mut_ptr(),
                    frame_capacity,
                    0,
                )
            };
            decode_time += start.elapsed();
            if samples_written < 0 {
                return Err(format!("libopus C decode failed: {}", samples_written));
            }
            if samples_written > 0 {
                let count = samples_written as usize * TARGET_CHANNELS as usize;
                decoded.extend_from_slice(&scratch[..count]);
            }
        }

        if decoded.is_empty() {
            return Err("libopus C produced no decoded samples".to_string());
        }

        let decoded_bytes = decoded.len() * std::mem::size_of::<i16>();
        Ok(PacketDecodeResult {
            samples: decoded,
            decoded_bytes,
            decode_time,
        })
    })();

    unsafe { opus_sys::opus_decoder_destroy(decoder) };

    result
}

fn compute_quality(reference: &[i16], candidate: &[i16]) -> QualityMetrics {
    let compared = reference.len().min(candidate.len());

    if compared == 0 {
        return QualityMetrics {
            snr_db: 0.0,
            rmse: 0.0,
            mae: 0.0,
            max_abs_error: 0.0,
            compared_samples: 0,
            length_delta_samples: candidate.len() as isize - reference.len() as isize,
        };
    }

    let mut noise_energy = 0.0f64;
    let mut ref_energy = 0.0f64;
    let mut abs_error = 0.0f64;
    let mut max_error = 0.0f64;

    for i in 0..compared {
        let reference_sample = reference[i] as f64;
        let candidate_sample = candidate[i] as f64;
        let error = reference_sample - candidate_sample;

        noise_energy += error * error;
        ref_energy += reference_sample * reference_sample;
        abs_error += error.abs();
        max_error = max_error.max(error.abs());
    }

    let snr_db = if ref_energy > 0.0 {
        if noise_energy > 0.0 {
            10.0 * (ref_energy / noise_energy).log10()
        } else {
            f64::INFINITY
        }
    } else {
        0.0
    };

    let n = compared as f64;
    let rmse = (noise_energy / n).sqrt();
    let mae = abs_error / n;

    QualityMetrics {
        snr_db,
        rmse,
        mae,
        max_abs_error: max_error,
        compared_samples: compared,
        length_delta_samples: candidate.len() as isize - reference.len() as isize,
    }
}

fn kbps_from_stats(encoded_bytes: usize, duration_seconds: f64) -> f64 {
    if duration_seconds == 0.0 {
        0.0
    } else {
        encoded_bytes as f64 * 8.0 / duration_seconds / 1000.0
    }
}

fn kbps_from_sample_bytes(sample_bytes: usize, duration_seconds: f64) -> f64 {
    if duration_seconds == 0.0 {
        0.0
    } else {
        sample_bytes as f64 * 8.0 / duration_seconds / 1000.0
    }
}

fn print_aggregate(agg: &BackendAggregate) {
    if agg.tracks == 0 {
        println!("  no successful tracks\n");
        return;
    }

    println!("  Tracks: {}", agg.tracks);
    println!(
        "  Encode RTF: {:>7.3}x ({:.3}s total)",
        agg.encode_rtf(),
        agg.encode_time.as_secs_f64(),
    );
    println!(
        "  Decode RTF: {:>7.3}x ({:.3}s total)",
        agg.decode_rtf(),
        agg.decode_time.as_secs_f64(),
    );
    println!(
        "  Encoded output: {:>9} bytes ({:.2} kbps)",
        agg.encoded_bytes,
        agg.bitrate_kbps()
    );
    println!(
        "  Decoded output: {:>9} bytes ({:.2} kbps)",
        agg.decoded_bytes,
        agg.decode_bitrate_kbps()
    );
    println!(
        "  Avg SNR(dB): {:.2} | Avg RMSE: {:.3} | Avg MAE: {:.3} | Max abs err: {:.3} | Avg length delta samples: {:.2}",
        agg.mean_snr(),
        agg.mean_rmse(),
        agg.mean_mae(),
        agg.max_abs_error,
        agg.length_delta_samples as f64 / agg.tracks as f64,
    );
}
