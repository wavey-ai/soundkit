use std::env;
use std::path::PathBuf;

use aac_wasm_bench::{
    format_bench_result, format_quality_measurement, parse_adts_frames, DEFAULT_ITERATIONS, FIXTURE,
};
use aac_wasm_bench::{
    DEFAULT_MAX_ABS_TOLERANCE, DEFAULT_MEAN_ABS_TOLERANCE, DEFAULT_MIN_SNR_DB,
    DEFAULT_RMSE_TOLERANCE, FIXTURE_NAME,
};

fn main() {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if args.first().map(String::as_str) == Some("quality-hotspots") {
        let limit = args
            .get(1)
            .and_then(|arg| arg.parse::<usize>().ok())
            .filter(|count| *count > 0)
            .unwrap_or(12);
        print_quality_hotspots(limit);
        return;
    }
    if args.first().map(String::as_str) == Some("frame-features") {
        print_frame_features(&args[1..]);
        return;
    }
    if args.first().map(String::as_str) == Some("frame-errors") {
        print_frame_errors(&args[1..]);
        return;
    }
    if args.first().map(String::as_str) == Some("export-soundkit-wav") {
        #[cfg(feature = "soundkit-lc")]
        {
            let output_path = args
                .get(1)
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("soundkit-aac-lc.wav"));

            let decoded = aac_wasm_bench::decode_soundkit_lc_fixture_pcm()
                .expect("decode SoundKit AAC-LC fixture");
            let channels = decoded.channels as usize;
            let frames = decoded.samples_per_channel as usize;
            let mut planar = vec![vec![0.0f32; frames]; channels];
            for frame in 0..frames {
                for channel in 0..channels {
                    planar[channel][frame] = decoded.pcm[frame * channels + channel];
                }
            }

            let wav = soundkit::wav::generate_wav_buffer(
                &soundkit::audio_types::PcmData::F32(planar),
                decoded.sample_rate,
            )
            .expect("generate decoded WAV");
            std::fs::write(&output_path, wav).expect("write decoded WAV");
            println!(
                "wrote={} frames={} samples/ch={} sr={} ch={}",
                output_path.display(),
                decoded.decoded_frames,
                decoded.samples_per_channel,
                decoded.sample_rate,
                decoded.channels
            );
        }

        #[cfg(not(feature = "soundkit-lc"))]
        panic!("export-soundkit-wav requires --features soundkit-lc");

        return;
    }

    let iterations = env::args()
        .nth(1)
        .and_then(|arg| arg.parse::<usize>().ok())
        .filter(|count| *count > 0)
        .unwrap_or(DEFAULT_ITERATIONS);

    let frames = parse_adts_frames(FIXTURE).expect("parse ADTS fixture");
    let first = frames.first().expect("fixture has frames");
    let fixture_audio_seconds = frames.len() as f64 * 1024.0 / first.sample_rate as f64;

    println!(
        "fixture={} bytes={} adts_frames={} sr={} ch={} audio_seconds={:.3} iterations={}",
        FIXTURE_NAME,
        FIXTURE.len(),
        frames.len(),
        first.sample_rate,
        first.channels,
        fixture_audio_seconds,
        iterations,
    );
    println!(
        "quality_thresholds rmse<={:.6} mean_abs_error<={:.6} max_abs_error<={:.6} snr_db>={:.3}",
        DEFAULT_RMSE_TOLERANCE,
        DEFAULT_MEAN_ABS_TOLERANCE,
        DEFAULT_MAX_ABS_TOLERANCE,
        DEFAULT_MIN_SNR_DB,
    );

    #[cfg(feature = "fdk")]
    match aac_wasm_bench::bench_fdk_fixture(iterations) {
        Ok(result) => println!("{}", format_bench_result(&result)),
        Err(error) => println!("fdk-aac-sys       error={error}"),
    }

    #[cfg(feature = "soundkit-lc")]
    match aac_wasm_bench::bench_soundkit_lc_fixture(iterations) {
        Ok(result) => println!("{}", format_bench_result(&result)),
        Err(error) => println!("soundkit-aac-lc   error={error}"),
    }

    #[cfg(feature = "soundkit-lc")]
    match aac_wasm_bench::bench_soundkit_lc_fixture_reused(iterations) {
        Ok(result) => println!("{}", format_bench_result(&result)),
        Err(error) => println!("soundkit-lc-reuse error={error}"),
    }

    #[cfg(feature = "symphonia")]
    match aac_wasm_bench::bench_symphonia_fixture(iterations) {
        Ok(result) => println!("{}", format_bench_result(&result)),
        Err(error) => println!("symphonia-aac     error={error}"),
    }

    #[cfg(all(feature = "fdk", feature = "symphonia"))]
    match aac_wasm_bench::compare_symphonia_to_fdk() {
        Ok(comparison) => println!(
            "{}",
            aac_wasm_bench::format_quality_comparison("fdk-vs-symphonia", &comparison)
        ),
        Err(error) => println!("fdk-vs-symphonia  error={error}"),
    }

    #[cfg(all(feature = "fdk", feature = "soundkit-lc"))]
    match aac_wasm_bench::compare_soundkit_lc_to_fdk() {
        Ok(comparison) => println!(
            "{}",
            aac_wasm_bench::format_quality_comparison("fdk-vs-soundkit", &comparison)
        ),
        Err(error) => println!("fdk-vs-soundkit   error={error}"),
    }

    #[cfg(not(any(target_arch = "wasm32", target_arch = "wasm64")))]
    println!("source_wav={}", aac_wasm_bench::source_wav_path().display());

    #[cfg(all(
        feature = "fdk",
        not(any(target_arch = "wasm32", target_arch = "wasm64"))
    ))]
    match aac_wasm_bench::compare_fdk_to_source_wav() {
        Ok(comparison) => println!(
            "{}",
            format_quality_measurement("source-vs-fdk", &comparison)
        ),
        Err(error) => println!("source-vs-fdk     error={error}"),
    }

    #[cfg(all(
        feature = "soundkit-lc",
        not(any(target_arch = "wasm32", target_arch = "wasm64"))
    ))]
    match aac_wasm_bench::compare_soundkit_lc_to_source_wav() {
        Ok(comparison) => println!(
            "{}",
            format_quality_measurement("source-vs-soundkit", &comparison)
        ),
        Err(error) => println!("source-vs-soundkit error={error}"),
    }

    #[cfg(all(
        feature = "symphonia",
        not(any(target_arch = "wasm32", target_arch = "wasm64"))
    ))]
    match aac_wasm_bench::compare_symphonia_to_source_wav() {
        Ok(comparison) => println!(
            "{}",
            format_quality_measurement("source-vs-symphonia", &comparison)
        ),
        Err(error) => println!("source-vs-symphonia error={error}"),
    }

    #[cfg(feature = "oxideav")]
    {
        let _ = oxideav_aac::register;
        println!("oxideav-aac       status=not-benchmarked reason=no wired PCM decoder API");
    }

    #[cfg(not(feature = "oxideav"))]
    println!("oxideav-aac       status=not-benchmarked reason=no wired PCM decoder API");
}

fn print_quality_hotspots(limit: usize) {
    println!("fixture={FIXTURE_NAME} mode=quality-hotspots limit={limit}");

    #[cfg(all(
        feature = "soundkit-lc",
        not(any(target_arch = "wasm32", target_arch = "wasm64"))
    ))]
    match aac_wasm_bench::source_soundkit_lc_frame_hotspots(limit) {
        Ok(hotspots) => {
            for hotspot in hotspots {
                print_hotspot("source-vs-soundkit", &hotspot);
            }
        }
        Err(error) => println!("source-vs-soundkit error={error}"),
    }

    #[cfg(all(
        feature = "fdk",
        not(any(target_arch = "wasm32", target_arch = "wasm64"))
    ))]
    match aac_wasm_bench::source_fdk_frame_hotspots(limit) {
        Ok(hotspots) => {
            for hotspot in hotspots {
                println!(
                    "source-vs-fdk      {}",
                    aac_wasm_bench::format_frame_hotspot(&hotspot)
                );
            }
        }
        Err(error) => println!("source-vs-fdk      error={error}"),
    }

    #[cfg(all(
        feature = "symphonia",
        not(any(target_arch = "wasm32", target_arch = "wasm64"))
    ))]
    match aac_wasm_bench::source_symphonia_frame_hotspots(limit) {
        Ok(hotspots) => {
            for hotspot in hotspots {
                println!(
                    "source-vs-symphonia {}",
                    aac_wasm_bench::format_frame_hotspot(&hotspot)
                );
            }
        }
        Err(error) => println!("source-vs-symphonia error={error}"),
    }

    #[cfg(all(
        feature = "fdk",
        feature = "soundkit-lc",
        not(any(target_arch = "wasm32", target_arch = "wasm64"))
    ))]
    match aac_wasm_bench::soundkit_lc_fdk_frame_hotspots(limit) {
        Ok(hotspots) => {
            for hotspot in hotspots {
                print_hotspot("fdk-vs-soundkit   ", &hotspot);
            }
        }
        Err(error) => println!("fdk-vs-soundkit    error={error}"),
    }

    #[cfg(not(all(
        feature = "soundkit-lc",
        not(any(target_arch = "wasm32", target_arch = "wasm64"))
    )))]
    println!("source-vs-soundkit status=unavailable reason=requires native soundkit-lc feature");
}

#[cfg(feature = "soundkit-lc")]
fn print_hotspot(label: &str, hotspot: &aac_wasm_bench::FrameQualityHotspot) {
    println!("{label} {}", aac_wasm_bench::format_frame_hotspot(hotspot));
    match aac_wasm_bench::soundkit_lc_frame_features(hotspot.frame_index) {
        Ok(features) => println!(
            "frame-features     {}",
            aac_wasm_bench::format_frame_features(&features)
        ),
        Err(error) => println!(
            "frame-features     frame={} error={error}",
            hotspot.frame_index
        ),
    }
}

#[cfg(feature = "soundkit-lc")]
fn print_frame_features(args: &[String]) {
    for arg in args {
        match arg.parse::<usize>() {
            Ok(frame_index) => match aac_wasm_bench::soundkit_lc_frame_features(frame_index) {
                Ok(features) => println!("{}", aac_wasm_bench::format_frame_features(&features)),
                Err(error) => println!("frame={frame_index} error={error}"),
            },
            Err(error) => println!("arg={arg} error={error}"),
        }
    }
}

#[cfg(not(feature = "soundkit-lc"))]
fn print_frame_features(_args: &[String]) {
    println!("frame-features status=unavailable reason=requires soundkit-lc feature");
}

#[cfg(all(
    feature = "fdk",
    feature = "soundkit-lc",
    not(any(target_arch = "wasm32", target_arch = "wasm64"))
))]
fn print_frame_errors(args: &[String]) {
    for arg in args {
        match arg.parse::<usize>() {
            Ok(frame_index) => {
                match aac_wasm_bench::soundkit_lc_fdk_frame_region_errors(frame_index) {
                    Ok(errors) => {
                        for error in errors {
                            println!(
                                "fdk-vs-soundkit {}",
                                aac_wasm_bench::format_frame_region_error(&error)
                            );
                        }
                    }
                    Err(error) => println!("frame={frame_index} error={error}"),
                }
            }
            Err(error) => println!("arg={arg} error={error}"),
        }
    }
}

#[cfg(not(all(
    feature = "fdk",
    feature = "soundkit-lc",
    not(any(target_arch = "wasm32", target_arch = "wasm64"))
)))]
fn print_frame_errors(_args: &[String]) {
    println!("frame-errors status=unavailable reason=requires native fdk and soundkit-lc features");
}
