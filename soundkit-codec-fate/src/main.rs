use frame_header::Endianness;
use sha2::{Digest, Sha256};
use soundkit::audio_types::AudioData;
use soundkit_decoder::{Bytes, DecodeError, DecodeOptions, DecodePipeline, DecodePipelineHandle};
use soundkit_video::{ChromaSampling, VideoCodec, VideoColorModel, VideoDecoder, VideoFrame};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{Duration, Instant};

const AUDIO_CHUNK_BYTES: usize = 64 * 1024;
const MAX_ALIGNMENT_FRAMES: usize = 4_096;
const ALIGNMENT_PROBE_FRAMES: usize = 2_048;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MediaKind {
    Audio,
    Video,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExpectedResult {
    Accept,
    Mismatch,
    Reject,
}

#[derive(Debug)]
struct Case {
    name: String,
    kind: MediaKind,
    codec: String,
    expected: ExpectedResult,
    minimum_quality_db: f64,
    relative_path: PathBuf,
    sha256: String,
}

#[derive(Debug)]
struct AudioPcm {
    samples: Vec<i16>,
    sample_rate: u32,
    channels: u8,
}

#[derive(Debug)]
struct VideoPixels {
    bytes: Vec<u8>,
    frames: usize,
    width: u32,
    height: u32,
    bit_depth: u8,
    pixel_format: String,
}

#[derive(Debug)]
struct Quality {
    db: f64,
    aligned_frames: isize,
    compared_samples: usize,
}

fn parse_manifest(path: &Path) -> Result<Vec<Case>, String> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("read manifest {}: {error}", path.display()))?;
    let mut cases = Vec::new();
    for (line_index, raw_line) in text.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields = line.split_whitespace().collect::<Vec<_>>();
        if fields.len() != 7 {
            return Err(format!(
                "{}:{} must contain name, kind, codec, result, quality, path, and SHA-256",
                path.display(),
                line_index + 1
            ));
        }
        let kind = match fields[1] {
            "audio" => MediaKind::Audio,
            "video" => MediaKind::Video,
            other => {
                return Err(format!(
                    "{}:{} has unknown media kind {other}",
                    path.display(),
                    line_index + 1
                ))
            }
        };
        let expected = match fields[3] {
            "accept" => ExpectedResult::Accept,
            "mismatch" => ExpectedResult::Mismatch,
            "reject" => ExpectedResult::Reject,
            other => {
                return Err(format!(
                    "{}:{} has unknown result {other}",
                    path.display(),
                    line_index + 1
                ))
            }
        };
        let minimum_quality_db = fields[4].parse::<f64>().map_err(|error| {
            format!(
                "{}:{} has invalid quality threshold: {error}",
                path.display(),
                line_index + 1
            )
        })?;
        let sha256 = fields[6].to_ascii_lowercase();
        if sha256.len() != 64 || !sha256.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(format!(
                "{}:{} has invalid SHA-256",
                path.display(),
                line_index + 1
            ));
        }
        cases.push(Case {
            name: fields[0].to_owned(),
            kind,
            codec: fields[2].to_owned(),
            expected,
            minimum_quality_db,
            relative_path: fields[5].into(),
            sha256,
        });
    }
    if cases.is_empty() {
        return Err(format!("manifest {} is empty", path.display()));
    }
    Ok(cases)
}

fn verify_source(path: &Path, expected: &str) -> Result<Vec<u8>, String> {
    let bytes = fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    let actual = format!("{:x}", Sha256::digest(&bytes));
    if actual != expected {
        return Err(format!(
            "{} SHA-256 mismatch: expected {expected}, got {actual}",
            path.display()
        ));
    }
    Ok(bytes)
}

fn drain_audio(
    pipeline: &mut DecodePipelineHandle,
    frames: &mut Vec<AudioData>,
) -> Result<(), String> {
    while let Some(output) = pipeline.try_recv() {
        frames.push(output.map_err(|error| error.to_string())?);
    }
    Ok(())
}

fn send_audio(
    pipeline: &mut DecodePipelineHandle,
    bytes: Bytes,
    frames: &mut Vec<AudioData>,
) -> Result<(), String> {
    loop {
        match pipeline.send(bytes.clone()) {
            Ok(()) => return Ok(()),
            Err(DecodeError::InputBufferFull) => {
                drain_audio(pipeline, frames)?;
                std::thread::yield_now();
            }
            Err(error) => return Err(error.to_string()),
        }
    }
}

fn decode_audio_soundkit(data: &[u8], codec: &str) -> Result<AudioPcm, String> {
    let options = DecodeOptions {
        output_bits_per_sample: Some(16),
        output_sample_rate: None,
        output_channels: None,
    };
    let mut pipeline = match codec {
        "auto" => DecodePipeline::spawn_with_buffers_and_options(256, 4_096, options),
        "amr-nb" => {
            DecodePipeline::spawn_amr_nb_with_options(options).map_err(|error| error.to_string())?
        }
        other => return Err(format!("unknown audio decoder {other}")),
    };
    let mut frames = Vec::new();
    for chunk in data.chunks(AUDIO_CHUNK_BYTES) {
        send_audio(&mut pipeline, Bytes::copy_from_slice(chunk), &mut frames)?;
    }
    send_audio(&mut pipeline, Bytes::new(), &mut frames)?;
    while let Some(output) = pipeline.recv() {
        frames.push(output.map_err(|error| error.to_string())?);
    }
    if frames.is_empty() {
        return Err("SoundKit emitted no audio frames".to_owned());
    }

    let sample_rate = frames[0].sampling_rate();
    let channels = frames[0].channel_count();
    let mut samples = Vec::new();
    for frame in frames {
        if frame.bits_per_sample() != 16
            || frame.sampling_rate() != sample_rate
            || frame.channel_count() != channels
        {
            return Err("SoundKit audio format changed during decode".to_owned());
        }
        if frame.data().len() % 2 != 0 {
            return Err("SoundKit emitted an odd-sized 16-bit PCM buffer".to_owned());
        }
        for bytes in frame.data().chunks_exact(2) {
            let pair = [bytes[0], bytes[1]];
            samples.push(match frame.endianness() {
                Endianness::LittleEndian => i16::from_le_bytes(pair),
                Endianness::BigEndian => i16::from_be_bytes(pair),
            });
        }
    }
    if samples.is_empty() || samples.len() % usize::from(channels) != 0 {
        return Err("SoundKit emitted invalid interleaved PCM".to_owned());
    }
    Ok(AudioPcm {
        samples,
        sample_rate,
        channels,
    })
}

fn command_output(mut command: Command, description: &str) -> Result<Output, String> {
    let output = command
        .output()
        .map_err(|error| format!("run {description}: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "{description} failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(output)
}

fn prepare_soundkit_input(case: &Case, path: &Path, source: &[u8]) -> Result<Vec<u8>, String> {
    let (stream, format) = match case.codec.as_str() {
        "aac-adts" => ("0:a:0", "adts"),
        "opus-ogg" => ("0:a:0", "opus"),
        "vp9-webm" => ("0:v:0", "ivf"),
        _ => return Ok(source.to_vec()),
    };
    let mut command = Command::new("ffmpeg");
    command
        .args(["-nostdin", "-v", "error", "-threads", "1", "-i"])
        .arg(path)
        .args(["-map", stream, "-c", "copy", "-f", format, "-"]);
    let output = command_output(command, "FFmpeg elementary-stream extraction")?;
    if output.stdout.is_empty() {
        return Err("FFmpeg emitted an empty elementary stream".to_owned());
    }
    Ok(output.stdout)
}

fn soundkit_audio_codec(codec: &str) -> &str {
    match codec {
        "aac-adts" | "opus-ogg" => "auto",
        other => other,
    }
}

fn soundkit_video_codec(codec: &str) -> &str {
    match codec {
        "vp9-webm" => "vp9",
        other => other,
    }
}

fn ffmpeg_audio(path: &Path) -> Result<Vec<i16>, String> {
    let mut command = Command::new("ffmpeg");
    command
        .args(["-nostdin", "-v", "error", "-threads", "1", "-i"])
        .arg(path)
        .args(["-map", "0:a:0", "-acodec", "pcm_s16le", "-f", "s16le", "-"]);
    let output = command_output(command, "FFmpeg audio reference decode")?;
    if output.stdout.is_empty() || output.stdout.len() % 2 != 0 {
        return Err("FFmpeg emitted invalid 16-bit PCM".to_owned());
    }
    Ok(output
        .stdout
        .chunks_exact(2)
        .map(|bytes| i16::from_le_bytes([bytes[0], bytes[1]]))
        .collect())
}

fn ffprobe_audio(path: &Path) -> Result<(u32, u8), String> {
    let mut command = Command::new("ffprobe");
    command
        .args([
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels",
            "-of",
            "csv=p=0",
        ])
        .arg(path);
    let output = command_output(command, "FFprobe audio metadata")?;
    let text = String::from_utf8_lossy(&output.stdout);
    let fields = text.trim().split(',').collect::<Vec<_>>();
    if fields.len() != 2 {
        return Err(format!(
            "unexpected FFprobe audio metadata: {}",
            text.trim()
        ));
    }
    let sample_rate = fields[0]
        .parse::<u32>()
        .map_err(|error| format!("invalid FFprobe sample rate: {error}"))?;
    let channels = fields[1]
        .parse::<u8>()
        .map_err(|error| format!("invalid FFprobe channel count: {error}"))?;
    Ok((sample_rate, channels))
}

fn best_audio_quality(soundkit: &[i16], reference: &[i16], channels: usize) -> Quality {
    let soundkit_frames = soundkit.len() / channels;
    let reference_frames = reference.len() / channels;
    let maximum_shift = MAX_ALIGNMENT_FRAMES
        .min(soundkit_frames.saturating_sub(1))
        .min(reference_frames.saturating_sub(1));
    let mut best_shift = 0isize;
    let mut best_error = f64::INFINITY;

    for shift in -(maximum_shift as isize)..=(maximum_shift as isize) {
        let (soundkit_start, reference_start) = if shift >= 0 {
            (shift as usize * channels, 0)
        } else {
            (0, shift.unsigned_abs() * channels)
        };
        let count = soundkit
            .len()
            .saturating_sub(soundkit_start)
            .min(reference.len().saturating_sub(reference_start))
            .min(ALIGNMENT_PROBE_FRAMES * channels);
        if count < channels * 32 {
            continue;
        }
        let error = soundkit[soundkit_start..soundkit_start + count]
            .iter()
            .zip(&reference[reference_start..reference_start + count])
            .map(|(left, right)| {
                let difference = f64::from(*left) - f64::from(*right);
                difference * difference
            })
            .sum::<f64>()
            / count as f64;
        if error < best_error {
            best_error = error;
            best_shift = shift;
        }
    }

    let (soundkit_start, reference_start) = if best_shift >= 0 {
        (best_shift as usize * channels, 0)
    } else {
        (0, best_shift.unsigned_abs() * channels)
    };
    let count = soundkit
        .len()
        .saturating_sub(soundkit_start)
        .min(reference.len().saturating_sub(reference_start));
    let (signal, noise) = soundkit[soundkit_start..soundkit_start + count]
        .iter()
        .zip(&reference[reference_start..reference_start + count])
        .fold((0.0, 0.0), |(signal, noise), (left, right)| {
            let reference = f64::from(*right);
            let difference = f64::from(*left) - reference;
            (
                signal + reference * reference,
                noise + difference * difference,
            )
        });
    let db = if noise == 0.0 {
        f64::INFINITY
    } else if signal == 0.0 {
        f64::NEG_INFINITY
    } else {
        10.0 * (signal / noise).log10()
    };
    Quality {
        db,
        aligned_frames: best_shift,
        compared_samples: count,
    }
}

fn decode_ivf(decoder: &mut VideoDecoder, data: &[u8]) -> Result<Vec<VideoFrame>, String> {
    if data.len() < 32 || &data[..4] != b"DKIF" {
        return Err("invalid IVF stream header".to_owned());
    }
    let mut cursor = usize::from(u16::from_le_bytes([data[6], data[7]]));
    let mut frames = Vec::new();
    while cursor < data.len() {
        if data.len() - cursor < 12 {
            return Err("truncated IVF frame header".to_owned());
        }
        let size = u32::from_le_bytes(
            data[cursor..cursor + 4]
                .try_into()
                .map_err(|_| "invalid IVF frame size")?,
        ) as usize;
        let timestamp = u64::from_le_bytes(
            data[cursor + 4..cursor + 12]
                .try_into()
                .map_err(|_| "invalid IVF timestamp")?,
        );
        cursor += 12;
        let end = cursor
            .checked_add(size)
            .filter(|end| *end <= data.len())
            .ok_or_else(|| "truncated IVF frame payload".to_owned())?;
        frames.extend(decoder.decode(&data[cursor..end], i64::try_from(timestamp).ok(), None)?);
        cursor = end;
    }
    frames.extend(decoder.flush()?);
    Ok(frames)
}

fn decode_video_soundkit(data: &[u8], codec_name: &str) -> Result<VideoPixels, String> {
    let codec = VideoCodec::parse(codec_name)
        .ok_or_else(|| format!("unknown SoundKit video codec {codec_name}"))?;
    let mut decoder = VideoDecoder::new(codec)?;
    let mut frames = match codec {
        VideoCodec::H264 => decoder.decode_stream(data)?,
        VideoCodec::Hevc => decoder.decode(data, None, None)?,
        VideoCodec::Vp9 | VideoCodec::Av1 => decode_ivf(&mut decoder, data)?,
        _ => {
            return Err(format!(
                "FATE runner does not handle {} yet",
                codec.as_str()
            ))
        }
    };
    if codec != VideoCodec::Vp9 && codec != VideoCodec::Av1 {
        frames.extend(decoder.flush()?);
    }
    let first = frames
        .first()
        .ok_or_else(|| "SoundKit emitted no video frames".to_owned())?;
    let output_pixel_format = pixel_format(first)?;
    let width = first.width;
    let height = first.height;
    let bit_depth = first.bit_depth;
    let mut bytes = Vec::new();
    for frame in &frames {
        if pixel_format(frame)? != output_pixel_format {
            return Err("SoundKit video pixel format changed during decode".to_owned());
        }
        let bytes_per_sample = if frame.bit_depth <= 8 { 1 } else { 2 };
        for plane in &frame.planes {
            let row_bytes = plane.width as usize * bytes_per_sample;
            let stride_bytes = plane.stride as usize * bytes_per_sample;
            for row in 0..plane.height as usize {
                let start = row * stride_bytes;
                bytes.extend_from_slice(&plane.data[start..start + row_bytes]);
            }
        }
    }
    Ok(VideoPixels {
        bytes,
        frames: frames.len(),
        width,
        height,
        bit_depth,
        pixel_format: output_pixel_format,
    })
}

fn pixel_format(frame: &VideoFrame) -> Result<String, String> {
    let depth = if frame.bit_depth <= 8 {
        String::new()
    } else {
        format!("{}le", frame.bit_depth)
    };
    let format = match (frame.color_model, frame.chroma_sampling, frame.has_alpha) {
        (VideoColorModel::Ycbcr, ChromaSampling::Monochrome, false) => {
            format!("gray{depth}")
        }
        (VideoColorModel::Ycbcr, ChromaSampling::Cs420, false) => format!("yuv420p{depth}"),
        (VideoColorModel::Ycbcr, ChromaSampling::Cs422, false) => format!("yuv422p{depth}"),
        (VideoColorModel::Ycbcr, ChromaSampling::Cs444, false) => format!("yuv444p{depth}"),
        (VideoColorModel::Ycbcr, ChromaSampling::Cs420, true) => format!("yuva420p{depth}"),
        (VideoColorModel::Ycbcr, ChromaSampling::Cs422, true) => format!("yuva422p{depth}"),
        (VideoColorModel::Ycbcr, ChromaSampling::Cs444, true) => format!("yuva444p{depth}"),
        (VideoColorModel::Gbr, ChromaSampling::Cs444, false) => format!("gbrp{depth}"),
        (VideoColorModel::Gbr, ChromaSampling::Cs444, true) => format!("gbrap{depth}"),
        combination => return Err(format!("unsupported output surface {combination:?}")),
    };
    Ok(format)
}

fn ffmpeg_video(path: &Path, pixel_format: &str) -> Result<Vec<u8>, String> {
    let mut command = Command::new("ffmpeg");
    command
        .args(["-nostdin", "-v", "error", "-threads", "1", "-i"])
        .arg(path)
        .args([
            "-map",
            "0:v:0",
            "-pix_fmt",
            pixel_format,
            "-f",
            "rawvideo",
            "-",
        ]);
    let output = command_output(command, "FFmpeg video reference decode")?;
    if output.stdout.is_empty() {
        return Err("FFmpeg emitted no video pixels".to_owned());
    }
    Ok(output.stdout)
}

fn video_quality(soundkit: &VideoPixels, reference: &[u8]) -> Result<f64, String> {
    if soundkit.bytes.len() != reference.len() {
        return Err(format!(
            "video byte count differs: SoundKit={}, FFmpeg={}",
            soundkit.bytes.len(),
            reference.len()
        ));
    }
    let bytes_per_sample = if soundkit.bit_depth <= 8 { 1 } else { 2 };
    let mut error = 0.0;
    let samples = soundkit.bytes.len() / bytes_per_sample;
    if bytes_per_sample == 1 {
        for (left, right) in soundkit.bytes.iter().zip(reference) {
            let difference = f64::from(*left) - f64::from(*right);
            error += difference * difference;
        }
    } else {
        for (left, right) in soundkit
            .bytes
            .chunks_exact(2)
            .zip(reference.chunks_exact(2))
        {
            let left = f64::from(u16::from_le_bytes([left[0], left[1]]));
            let right = f64::from(u16::from_le_bytes([right[0], right[1]]));
            let difference = left - right;
            error += difference * difference;
        }
    }
    if error == 0.0 {
        return Ok(f64::INFINITY);
    }
    let mse = error / samples as f64;
    let peak = ((1u32 << soundkit.bit_depth) - 1) as f64;
    Ok(10.0 * (peak * peak / mse).log10())
}

fn ffmpeg_accepts(path: &Path, kind: MediaKind) -> Result<(), String> {
    let stream = match kind {
        MediaKind::Audio => "0:a:0",
        MediaKind::Video => "0:v:0",
    };
    let mut command = Command::new("ffmpeg");
    command
        .args(["-nostdin", "-v", "error", "-threads", "1", "-i"])
        .arg(path)
        .args(["-map", stream, "-f", "null", "-"]);
    command_output(command, "FFmpeg gap-sample validation").map(|_| ())
}

fn check_case(root: &Path, case: &Case) -> Result<(), String> {
    let path = root.join(&case.relative_path);
    let source = verify_source(&path, &case.sha256)?;
    let data = prepare_soundkit_input(case, &path, &source)?;
    match (case.kind, case.expected) {
        (MediaKind::Audio, ExpectedResult::Accept) => {
            let soundkit = decode_audio_soundkit(&data, soundkit_audio_codec(&case.codec))?;
            let (reference_rate, reference_channels) = ffprobe_audio(&path)?;
            if (soundkit.sample_rate, soundkit.channels) != (reference_rate, reference_channels) {
                return Err(format!(
                    "audio format differs: SoundKit={}/{} FFmpeg={}/{}",
                    soundkit.sample_rate, soundkit.channels, reference_rate, reference_channels
                ));
            }
            let reference = ffmpeg_audio(&path)?;
            let quality = best_audio_quality(
                &soundkit.samples,
                &reference,
                usize::from(soundkit.channels),
            );
            if quality.db < case.minimum_quality_db {
                return Err(format!(
                    "audio SNR {:.2} dB is below {:.2} dB",
                    quality.db, case.minimum_quality_db
                ));
            }
            println!(
                "accept {:<25} audio {:>8} Hz {} ch SNR={:>7.2} dB align={:+} frames samples={}",
                case.name,
                soundkit.sample_rate,
                soundkit.channels,
                quality.db,
                quality.aligned_frames,
                quality.compared_samples
            );
        }
        (MediaKind::Video, ExpectedResult::Accept) => {
            let soundkit = decode_video_soundkit(&data, soundkit_video_codec(&case.codec))?;
            let reference = ffmpeg_video(&path, &soundkit.pixel_format)?;
            let quality = video_quality(&soundkit, &reference)?;
            if quality < case.minimum_quality_db {
                return Err(format!(
                    "video PSNR {quality:.2} dB is below {:.2} dB",
                    case.minimum_quality_db
                ));
            }
            println!(
                "accept {:<25} video {} {}x{} {} frames PSNR={:>7.2} dB",
                case.name,
                soundkit.pixel_format,
                soundkit.width,
                soundkit.height,
                soundkit.frames,
                quality
            );
        }
        (MediaKind::Audio, ExpectedResult::Mismatch) => {
            let soundkit = decode_audio_soundkit(&data, soundkit_audio_codec(&case.codec))?;
            let (reference_rate, reference_channels) = ffprobe_audio(&path)?;
            if (soundkit.sample_rate, soundkit.channels) != (reference_rate, reference_channels) {
                return Err(format!(
                    "expected sample divergence, but audio format differs: SoundKit={}/{} FFmpeg={}/{}",
                    soundkit.sample_rate, soundkit.channels, reference_rate, reference_channels
                ));
            }
            let reference = ffmpeg_audio(&path)?;
            let quality = best_audio_quality(
                &soundkit.samples,
                &reference,
                usize::from(soundkit.channels),
            );
            if quality.db >= case.minimum_quality_db {
                return Err(format!(
                    "known gap now passes at {:.2} dB; change manifest result to accept",
                    quality.db
                ));
            }
            println!(
                "mismatch {:<23} audio SNR={:>7.2} dB (< {:.2} dB) align={:+} frames",
                case.name, quality.db, case.minimum_quality_db, quality.aligned_frames
            );
        }
        (MediaKind::Video, ExpectedResult::Mismatch) => {
            let soundkit = decode_video_soundkit(&data, soundkit_video_codec(&case.codec))?;
            let reference = ffmpeg_video(&path, &soundkit.pixel_format)?;
            match video_quality(&soundkit, &reference) {
                Ok(quality) if quality >= case.minimum_quality_db => {
                    return Err(format!(
                        "known gap now passes at {quality:.2} dB; change manifest result to accept"
                    ));
                }
                Ok(quality) => println!(
                    "mismatch {:<23} video PSNR={:>7.2} dB (< {:.2} dB)",
                    case.name, quality, case.minimum_quality_db
                ),
                Err(error) => println!("mismatch {:<23} {error}", case.name),
            }
        }
        (kind, ExpectedResult::Reject) => {
            ffmpeg_accepts(&path, kind)?;
            let result = match kind {
                MediaKind::Audio => {
                    decode_audio_soundkit(&data, soundkit_audio_codec(&case.codec)).map(|_| ())
                }
                MediaKind::Video => {
                    decode_video_soundkit(&data, soundkit_video_codec(&case.codec)).map(|_| ())
                }
            };
            match result {
                Ok(()) => return Err("SoundKit unexpectedly accepted the gap sample".to_owned()),
                Err(error) => println!("reject {:<25} {error}", case.name),
            }
        }
    }
    Ok(())
}

fn run_checks(root: &Path, cases: &[Case]) -> Result<(), String> {
    let mut failures = Vec::new();
    for case in cases {
        if let Err(error) = check_case(root, case) {
            eprintln!("FAIL {:<25} {error}", case.name);
            failures.push(case.name.as_str());
        }
    }
    if failures.is_empty() {
        println!("FATE codec integration suite passed: {} cases", cases.len());
        Ok(())
    } else {
        Err(format!(
            "FATE codec integration suite failed: {}",
            failures.join(", ")
        ))
    }
}

fn ffmpeg_benchmark(path: &Path, kind: MediaKind) -> Result<Duration, String> {
    let stream = match kind {
        MediaKind::Audio => "0:a:0",
        MediaKind::Video => "0:v:0",
    };
    let mut command = Command::new("ffmpeg");
    command
        .args([
            "-nostdin",
            "-v",
            "info",
            "-benchmark",
            "-threads",
            "1",
            "-i",
        ])
        .arg(path)
        .args(["-map", stream, "-f", "null", "-"]);
    let output = command_output(command, "FFmpeg benchmark decode")?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    let seconds = stderr
        .lines()
        .rev()
        .find(|line| line.starts_with("bench: utime="))
        .and_then(|line| {
            line.split_whitespace()
                .find_map(|field| field.strip_prefix("rtime="))
        })
        .and_then(|field| field.strip_suffix('s'))
        .and_then(|field| field.parse::<f64>().ok())
        .ok_or_else(|| "FFmpeg did not report benchmark rtime".to_owned())?;
    Ok(Duration::from_secs_f64(seconds))
}

fn soundkit_benchmark(data: &[u8], case: &Case) -> Result<Duration, String> {
    let started = Instant::now();
    match case.kind {
        MediaKind::Audio => {
            let _ = decode_audio_soundkit(data, soundkit_audio_codec(&case.codec))?;
        }
        MediaKind::Video => {
            let _ = decode_video_soundkit(data, soundkit_video_codec(&case.codec))?;
        }
    }
    Ok(started.elapsed())
}

fn run_benchmarks(root: &Path, cases: &[Case], iterations: usize) -> Result<(), String> {
    let version = command_output(
        {
            let mut command = Command::new("ffmpeg");
            command.arg("-version");
            command
        },
        "FFmpeg version",
    )?;
    let version = String::from_utf8_lossy(&version.stdout);
    println!("{}", version.lines().next().unwrap_or("ffmpeg"));
    println!(
        "{:<25} {:<6} {:>12} {:>17} {:>9}",
        "case", "kind", "SoundKit ms", "FFmpeg rtime ms", "SK/FF"
    );
    for case in cases
        .iter()
        .filter(|case| case.expected == ExpectedResult::Accept)
    {
        let path = root.join(&case.relative_path);
        let source = verify_source(&path, &case.sha256)?;
        let data = prepare_soundkit_input(case, &path, &source)?;
        let mut soundkit_elapsed = Duration::ZERO;
        let mut ffmpeg_elapsed = Duration::ZERO;
        for _ in 0..iterations {
            soundkit_elapsed += soundkit_benchmark(&data, case)?;
            ffmpeg_elapsed += ffmpeg_benchmark(&path, case.kind)?;
        }
        let soundkit_ms = soundkit_elapsed.as_secs_f64() * 1_000.0 / iterations as f64;
        let ffmpeg_ms = ffmpeg_elapsed.as_secs_f64() * 1_000.0 / iterations as f64;
        println!(
            "{:<25} {:<6} {:>12.3} {:>17.3} {:>9.3}",
            case.name,
            match case.kind {
                MediaKind::Audio => "audio",
                MediaKind::Video => "video",
            },
            soundkit_ms,
            ffmpeg_ms,
            soundkit_ms / ffmpeg_ms
        );
    }
    println!("Each value is the mean of {iterations} local decodes; lower is faster.");
    println!(
        "SoundKit includes in-process decoder setup; FFmpeg uses its reported single-thread rtime, excluding CLI launch."
    );
    Ok(())
}

fn usage() -> ! {
    eprintln!("usage: soundkit-codec-fate <check|bench> <corpus-root> <manifest> [iterations]");
    std::process::exit(2)
}

fn main() {
    let mut arguments = env::args().skip(1);
    let mode = arguments.next().unwrap_or_else(|| usage());
    let root = PathBuf::from(arguments.next().unwrap_or_else(|| usage()));
    let manifest = PathBuf::from(arguments.next().unwrap_or_else(|| usage()));
    let iterations = arguments
        .next()
        .map(|value| value.parse::<usize>())
        .transpose()
        .unwrap_or_else(|error| {
            eprintln!("invalid iteration count: {error}");
            std::process::exit(2);
        })
        .unwrap_or(3);
    if iterations == 0 || arguments.next().is_some() {
        usage();
    }
    let cases = parse_manifest(&manifest).unwrap_or_else(|error| {
        eprintln!("{error}");
        std::process::exit(2);
    });
    let result = match mode.as_str() {
        "check" => run_checks(&root, &cases),
        "bench" => {
            run_checks(&root, &cases).and_then(|_| run_benchmarks(&root, &cases, iterations))
        }
        _ => usage(),
    };
    if let Err(error) = result {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
