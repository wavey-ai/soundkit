use soundkit::audio_packet::Decoder;
use soundkit_aac::{AacDecoder, AacDecoderBackend};
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

const INPUT_CHUNK_BYTES: usize = 1024 * 1024;
const OUTPUT_SAMPLES: usize = 128 * 1024;
const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let fixture = args.next().ok_or_else(|| {
        "usage: bench_adts_decode FIXTURE [ITERATIONS] [auto|soundkit|fdk]".to_string()
    })?;
    let iterations = args
        .next()
        .map(|value| value.parse::<usize>())
        .transpose()
        .map_err(|error| format!("invalid iteration count: {error}"))?
        .unwrap_or(3)
        .max(1);
    let requested_backend = args.next().unwrap_or_else(|| "auto".to_string());
    if args.next().is_some() {
        return Err("too many arguments".to_string());
    }

    let input = std::fs::read(&fixture)
        .map_err(|error| format!("read {}: {error}", Path::new(&fixture).display()))?;
    let quality = decode_once(&input, &requested_backend, true)?;
    let started = Instant::now();
    let mut total_samples = 0u64;
    let mut total_frames = 0u64;
    for _ in 0..iterations {
        let decoded = decode_once(&input, &requested_backend, false)?;
        if decoded.backend != quality.backend
            || decoded.sample_rate != quality.sample_rate
            || decoded.channels != quality.channels
            || decoded.samples != quality.samples
        {
            return Err("AAC production decoder output changed between iterations".to_string());
        }
        total_samples += decoded.samples;
        total_frames += decoded.frames;
        black_box(decoded.samples);
    }
    let elapsed = started.elapsed();
    let audio_seconds =
        total_samples as f64 / f64::from(quality.channels) / f64::from(quality.sample_rate);
    let elapsed_seconds = elapsed.as_secs_f64();

    println!(
        "soundkit-aac-production backend={:?} fixture={} iterations={} decoded_frames={} samples={} sample_rate={} channels={} elapsed_ms={:.3} rtf={:.6} x_realtime={:.1} frames_per_sec={:.1} checksum={:016x}",
        quality.backend,
        Path::new(&fixture)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(&fixture),
        iterations,
        total_frames,
        total_samples,
        quality.sample_rate,
        quality.channels,
        elapsed_seconds * 1000.0,
        elapsed_seconds / audio_seconds,
        audio_seconds / elapsed_seconds,
        total_frames as f64 / elapsed_seconds,
        quality.checksum,
    );
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct DecodeReport {
    backend: AacDecoderBackend,
    sample_rate: u32,
    channels: u8,
    frames: u64,
    samples: u64,
    checksum: u64,
}

fn decode_once(
    input: &[u8],
    requested_backend: &str,
    checksum_output: bool,
) -> Result<DecodeReport, String> {
    let mut decoder = match requested_backend {
        "auto" => AacDecoder::new(),
        "soundkit" => {
            #[cfg(feature = "owned-lc")]
            {
                AacDecoder::new_soundkit_aac_lc()
            }
            #[cfg(not(feature = "owned-lc"))]
            {
                return Err("soundkit backend requires --features owned-lc".to_string());
            }
        }
        "fdk" => {
            #[cfg(feature = "fdk")]
            {
                AacDecoder::new_fdk()
            }
            #[cfg(not(feature = "fdk"))]
            {
                return Err("FDK backend requires --features fdk".to_string());
            }
        }
        value => return Err(format!("unknown backend {value:?}")),
    };
    decoder.init()?;

    let mut output = vec![0i16; OUTPUT_SAMPLES];
    let mut samples = 0u64;
    let mut checksum = FNV_OFFSET;
    for chunk in input.chunks(INPUT_CHUNK_BYTES) {
        let written = decoder.decode_i16(chunk, &mut output, false)?;
        consume_output(
            &output[..written],
            checksum_output,
            &mut checksum,
            &mut samples,
        );
        loop {
            let written = decoder.decode_i16(&[], &mut output, false)?;
            if written == 0 {
                break;
            }
            consume_output(
                &output[..written],
                checksum_output,
                &mut checksum,
                &mut samples,
            );
        }
    }
    loop {
        let written = decoder.decode_i16(&[], &mut output, false)?;
        if written == 0 {
            break;
        }
        consume_output(
            &output[..written],
            checksum_output,
            &mut checksum,
            &mut samples,
        );
    }

    let sample_rate = decoder
        .sample_rate()
        .ok_or_else(|| "AAC decoder did not report a sample rate".to_string())?;
    let channels = decoder
        .channels()
        .ok_or_else(|| "AAC decoder did not report a channel count".to_string())?;
    let samples_per_channel = samples / u64::from(channels);
    if samples % u64::from(channels) != 0 || samples_per_channel % 1024 != 0 {
        return Err(format!(
            "AAC output length {samples} is not whole 1024-sample frames for {channels} channels"
        ));
    }

    Ok(DecodeReport {
        backend: decoder.backend(),
        sample_rate,
        channels,
        frames: samples_per_channel / 1024,
        samples,
        checksum,
    })
}

fn consume_output(output: &[i16], checksum_output: bool, checksum: &mut u64, samples: &mut u64) {
    *samples += output.len() as u64;
    if checksum_output {
        for sample in output {
            for byte in sample.to_le_bytes() {
                *checksum ^= u64::from(byte);
                *checksum = checksum.wrapping_mul(FNV_PRIME);
            }
        }
    } else {
        black_box(output);
    }
}
