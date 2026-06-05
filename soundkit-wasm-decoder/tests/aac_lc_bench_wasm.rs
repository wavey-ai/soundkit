#![cfg(all(target_arch = "wasm32", feature = "aac-lc-bench"))]

use aac_wasm_bench::{
    decode_soundkit_lc_fixture_pcm_for, decode_wav_pcm_bytes, format_quality_measurement,
    AacFixture, QualityComparison,
};
use js_sys::Float32Array;
use soundkit_aac_lc::AacLcDecoder;
use soundkit_wasm_decoder::WasmAacLcDecoder;
use wasm_bindgen::prelude::*;
use wasm_bindgen_test::{console_log, wasm_bindgen_test};

const FIXTURE_NAME: &str = "WESTSIDE_MIX_4_CONFIRMATION_130323_256k.aac";
const FIXTURE: &[u8] =
    include_bytes!("../../golden/aac/WESTSIDE_MIX_4_CONFIRMATION_130323_256k.aac");
const SOURCE_WAV: &[u8] = include_bytes!(
    "../../../bitneedle/apps/press/testdata/audio-regression/WESTSIDE_MIX 4 CONFIRMATION_130323.wav"
);
const ITERATIONS: usize = 5;
const WARMUP_ITERATIONS: usize = 1;
const EXPECTED_CORE_CHECKSUM: u64 = 0x24eb18ebd8fc27e4;
const EXPECTED_INTERLEAVED_CHECKSUM: u64 = 0x8c2d7264b07abe0a;
const EXPECTED_INTO_CHECKSUM: u64 = 0x0d4a1cec5595b60a;
const MAX_SOURCE_RMSE: f64 = 0.0071;
const MAX_SOURCE_MEAN_ABS: f64 = 0.0048;
const MAX_SOURCE_ABS: f64 = 0.40;
const MAX_SOURCE_P99_ABS: f64 = 0.0245;
const MAX_SOURCE_P999_ABS: f64 = 0.0405;
const MIN_SOURCE_SNR_DB: f64 = 27.3;
const MAX_RMSE_GAP_VS_SAVED_FDK: f64 = 0.00025;
const MAX_MEAN_ABS_GAP_VS_SAVED_FDK: f64 = 0.00010;
const MAX_SNR_GAP_DB_VS_SAVED_FDK: f64 = 0.25;
const SAVED_SOURCE_VS_FDK: SavedQualityBaseline = SavedQualityBaseline {
    name: "saved-source-vs-fdk",
    rmse: 0.006896303,
    mean_abs_error: 0.004594121,
    max_abs_error: 0.334225774,
    p99_abs_error: 0.023480892,
    p999_abs_error: 0.038754225,
    snr_db: 27.510,
};
const SAVED_SOURCE_VS_SYMPHONIA: SavedQualityBaseline = SavedQualityBaseline {
    name: "saved-source-vs-symphonia",
    rmse: 0.006894503,
    mean_abs_error: 0.004592889,
    max_abs_error: 0.363226533,
    p99_abs_error: 0.023478046,
    p999_abs_error: 0.038671419,
    snr_db: 27.511,
};

#[derive(Clone, Copy)]
struct SavedQualityBaseline {
    name: &'static str,
    rmse: f64,
    mean_abs_error: f64,
    max_abs_error: f64,
    p99_abs_error: f64,
    p999_abs_error: f64,
    snr_db: f64,
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = performance)]
    fn now() -> f64;
}

#[wasm_bindgen_test]
fn bench_aac_lc_raw_access_units() {
    let frames = parse_adts_frames(FIXTURE).expect("parse ADTS fixture");
    let first = frames.first().expect("fixture has frames");
    let asc = first.audio_specific_config();
    let fixture_seconds = frames.len() as f64 * 1024.0 / first.sample_rate as f64;

    let core = bench_core_decode(&frames, &asc);
    let api = bench_interleaved_api_decode(&frames, &asc);
    let into = bench_interleaved_into_decode(&frames, &asc);

    console_log!(
        "fixture={} bytes={} adts_frames={} sr={} ch={} audio_seconds={:.3} iterations={}",
        FIXTURE_NAME,
        FIXTURE.len(),
        frames.len(),
        first.sample_rate,
        first.channels,
        fixture_seconds,
        ITERATIONS,
    );
    console_log!("{}", core.format());
    console_log!("{}", api.format());
    console_log!("{}", into.format());

    assert_eq!(core.decoded_frames, (frames.len() * ITERATIONS) as u64);
    assert_eq!(api.decoded_frames, (frames.len() * ITERATIONS) as u64);
    assert_eq!(into.decoded_frames, (frames.len() * ITERATIONS) as u64);
    assert!(core.elapsed_ms > 0.0);
    assert!(api.elapsed_ms > 0.0);
    assert!(into.elapsed_ms > 0.0);
    assert_eq!(core.checksum, EXPECTED_CORE_CHECKSUM);
    assert_eq!(api.checksum, EXPECTED_INTERLEAVED_CHECKSUM);
    assert_eq!(into.checksum, EXPECTED_INTO_CHECKSUM);
}

#[wasm_bindgen_test]
fn quality_aac_lc_against_source_wav() {
    let source = decode_wav_pcm_bytes(SOURCE_WAV).expect("decode source WAV PCM");
    let decoded = decode_soundkit_lc_fixture_pcm_for(AacFixture {
        name: FIXTURE_NAME,
        data: FIXTURE,
    })
    .expect("decode SoundKit AAC-LC fixture in wasm");

    assert_eq!(decoded.decoded_frames, 9171);
    assert_eq!(decoded.samples_per_channel, 9171 * 1024);
    assert_eq!(decoded.sample_rate, source.sample_rate);
    assert_eq!(decoded.channels, source.channels);

    let quality = QualityComparison::compare_aligned(
        &source.pcm,
        &decoded.pcm,
        source.channels as usize,
        2048,
    );

    console_log!("{}", SAVED_SOURCE_VS_FDK.format());
    console_log!("{}", SAVED_SOURCE_VS_SYMPHONIA.format());
    console_log!(
        "{}",
        format_quality_measurement("wasm-source-vs-sk", &quality)
    );

    assert!(quality.compared_samples > 18_000_000);
    assert!(
        quality.rmse <= MAX_SOURCE_RMSE,
        "RMSE {} exceeded {}",
        quality.rmse,
        MAX_SOURCE_RMSE
    );
    assert!(
        quality.mean_abs_error <= MAX_SOURCE_MEAN_ABS,
        "mean abs error {} exceeded {}",
        quality.mean_abs_error,
        MAX_SOURCE_MEAN_ABS
    );
    assert!(
        quality.max_abs_error <= MAX_SOURCE_ABS,
        "max abs error {} exceeded {}",
        quality.max_abs_error,
        MAX_SOURCE_ABS
    );
    assert!(
        quality.p99_abs_error <= MAX_SOURCE_P99_ABS,
        "p99 abs error {} exceeded {}",
        quality.p99_abs_error,
        MAX_SOURCE_P99_ABS
    );
    assert!(
        quality.p999_abs_error <= MAX_SOURCE_P999_ABS,
        "p999 abs error {} exceeded {}",
        quality.p999_abs_error,
        MAX_SOURCE_P999_ABS
    );
    assert!(
        quality.snr_db >= MIN_SOURCE_SNR_DB,
        "SNR {} below {}",
        quality.snr_db,
        MIN_SOURCE_SNR_DB
    );
    assert!(
        quality.rmse <= SAVED_SOURCE_VS_FDK.rmse + MAX_RMSE_GAP_VS_SAVED_FDK,
        "RMSE {} is too far above saved FDK baseline {}",
        quality.rmse,
        SAVED_SOURCE_VS_FDK.rmse
    );
    assert!(
        quality.mean_abs_error
            <= SAVED_SOURCE_VS_FDK.mean_abs_error + MAX_MEAN_ABS_GAP_VS_SAVED_FDK,
        "mean abs error {} is too far above saved FDK baseline {}",
        quality.mean_abs_error,
        SAVED_SOURCE_VS_FDK.mean_abs_error
    );
    assert!(
        quality.snr_db >= SAVED_SOURCE_VS_FDK.snr_db - MAX_SNR_GAP_DB_VS_SAVED_FDK,
        "SNR {} is too far below saved FDK baseline {}",
        quality.snr_db,
        SAVED_SOURCE_VS_FDK.snr_db
    );
}

impl SavedQualityBaseline {
    fn format(self) -> String {
        format!(
            "{:<18} rmse={:.9} mean_abs_error={:.9} max_abs_error={:.9} p99_abs_error={:.9} p999_abs_error={:.9} snr_db={:.3}",
            self.name,
            self.rmse,
            self.mean_abs_error,
            self.max_abs_error,
            self.p99_abs_error,
            self.p999_abs_error,
            self.snr_db,
        )
    }
}

fn bench_core_decode(frames: &[AdtsFrame<'_>], asc: &[u8]) -> WasmBenchResult {
    let first = frames.first().expect("fixture has frames");
    let mut decoder = AacLcDecoder::from_audio_specific_config(asc).expect("create decoder");
    for _ in 0..WARMUP_ITERATIONS {
        for frame in frames {
            let _ = decoder
                .decode_access_unit(frame.raw)
                .expect("warm decode frame");
        }
    }

    let started = now();
    let mut decoded_frames = 0u64;
    let mut samples_per_channel = 0u64;
    let mut checksum = 0xcbf29ce484222325u64;

    for _ in 0..ITERATIONS {
        for frame in frames {
            let decoded = decoder.decode_access_unit(frame.raw).expect("decode frame");
            decoded_frames += 1;
            samples_per_channel += decoded.frames() as u64;
            checksum = mix_planar_checksum(checksum, decoded.channels(), decoded.frames());
        }
    }

    WasmBenchResult {
        name: "wasm-core-raw",
        decoded_frames,
        samples_per_channel,
        sample_rate: first.sample_rate,
        channels: first.channels,
        elapsed_ms: now() - started,
        checksum,
    }
}

fn bench_interleaved_api_decode(frames: &[AdtsFrame<'_>], asc: &[u8]) -> WasmBenchResult {
    let first = frames.first().expect("fixture has frames");
    let mut decoder = WasmAacLcDecoder::new(asc).expect("create wasm decoder");
    for _ in 0..WARMUP_ITERATIONS {
        for frame in frames {
            let _ = decoder
                .decode_interleaved(frame.raw)
                .expect("warm interleaved decode");
        }
    }

    let started = now();
    let mut decoded_frames = 0u64;
    let mut samples_per_channel = 0u64;
    let mut checksum = 0xcbf29ce484222325u64;

    for _ in 0..ITERATIONS {
        for frame in frames {
            let interleaved = decoder
                .decode_interleaved(frame.raw)
                .expect("decode interleaved");
            decoded_frames += 1;
            samples_per_channel += decoder.frames_per_access_unit() as u64;
            checksum = mix_interleaved_checksum(checksum, &interleaved);
        }
    }

    WasmBenchResult {
        name: "wasm-js-interleaved",
        decoded_frames,
        samples_per_channel,
        sample_rate: first.sample_rate,
        channels: first.channels,
        elapsed_ms: now() - started,
        checksum,
    }
}

fn bench_interleaved_into_decode(frames: &[AdtsFrame<'_>], asc: &[u8]) -> WasmBenchResult {
    let first = frames.first().expect("fixture has frames");
    let mut decoder = WasmAacLcDecoder::new(asc).expect("create wasm decoder");
    let output = Float32Array::new_with_length(
        decoder.frames_per_access_unit() as u32 * u32::from(first.channels),
    );
    for _ in 0..WARMUP_ITERATIONS {
        for frame in frames {
            let _ = decoder
                .decode_interleaved_into(frame.raw, &output)
                .expect("warm interleaved into decode");
        }
    }

    let started = now();
    let mut decoded_frames = 0u64;
    let mut samples_per_channel = 0u64;
    let mut checksum = 0xcbf29ce484222325u64;

    for _ in 0..ITERATIONS {
        for frame in frames {
            let written = decoder
                .decode_interleaved_into(frame.raw, &output)
                .expect("decode interleaved into");
            decoded_frames += 1;
            samples_per_channel += decoder.frames_per_access_unit() as u64;
            checksum ^= written as u64;
            checksum = mix_interleaved_checksum(checksum, &output);
        }
    }

    WasmBenchResult {
        name: "wasm-js-into",
        decoded_frames,
        samples_per_channel,
        sample_rate: first.sample_rate,
        channels: first.channels,
        elapsed_ms: now() - started,
        checksum,
    }
}

fn mix_planar_checksum(mut checksum: u64, channels: &[Vec<f32>], frames: usize) -> u64 {
    if frames == 0 {
        return checksum;
    }

    let sample_points = [0, frames / 2, frames - 1];
    for channel in channels {
        for index in sample_points {
            checksum ^= channel[index].to_bits() as u64;
            checksum = checksum.wrapping_mul(0x100000001b3);
        }
    }
    checksum
}

fn mix_interleaved_checksum(mut checksum: u64, interleaved: &Float32Array) -> u64 {
    let len = interleaved.length();
    if len == 0 {
        return checksum;
    }

    let sample_points = [0, len / 2, len - 1];
    for index in sample_points {
        checksum ^= interleaved.get_index(index).to_bits() as u64;
        checksum = checksum.wrapping_mul(0x100000001b3);
    }
    checksum
}

#[derive(Clone, Copy, Debug)]
struct WasmBenchResult {
    name: &'static str,
    decoded_frames: u64,
    samples_per_channel: u64,
    sample_rate: u32,
    channels: u8,
    elapsed_ms: f64,
    checksum: u64,
}

impl WasmBenchResult {
    fn audio_seconds(self) -> f64 {
        self.samples_per_channel as f64 / self.sample_rate as f64
    }

    fn real_time_factor(self) -> f64 {
        (self.elapsed_ms / 1000.0) / self.audio_seconds()
    }

    fn frames_per_second(self) -> f64 {
        self.decoded_frames as f64 / (self.elapsed_ms / 1000.0)
    }

    fn format(self) -> String {
        format!(
            "{:<20} frames={} decoded={} samples/ch={} sr={} ch={} elapsed_ms={:.3} rtf={:.6} frames_per_sec={:.1} checksum={:016x}",
            self.name,
            self.decoded_frames,
            self.decoded_frames,
            self.samples_per_channel,
            self.sample_rate,
            self.channels,
            self.elapsed_ms,
            self.real_time_factor(),
            self.frames_per_second(),
            self.checksum,
        )
    }
}

#[derive(Clone, Copy, Debug)]
struct AdtsFrame<'a> {
    raw: &'a [u8],
    audio_object_type: u8,
    sample_rate_index: u8,
    sample_rate: u32,
    channels: u8,
}

impl AdtsFrame<'_> {
    fn audio_specific_config(self) -> [u8; 2] {
        [
            (self.audio_object_type << 3) | (self.sample_rate_index >> 1),
            ((self.sample_rate_index & 1) << 7) | (self.channels << 3),
        ]
    }
}

fn parse_adts_frames(data: &[u8]) -> Result<Vec<AdtsFrame<'_>>, String> {
    let mut frames = Vec::new();
    let mut offset = 0usize;

    while offset + 7 <= data.len() {
        while offset + 7 <= data.len()
            && !(data[offset] == 0xff && (data[offset + 1] & 0xf0) == 0xf0)
        {
            offset += 1;
        }
        if offset + 7 > data.len() {
            break;
        }

        let protection_absent = (data[offset + 1] & 0x01) != 0;
        let header_len = if protection_absent { 7 } else { 9 };
        let audio_object_type = ((data[offset + 2] & 0xc0) >> 6) + 1;
        let sample_rate_index = (data[offset + 2] & 0x3c) >> 2;
        let sample_rate = adts_sample_rate(sample_rate_index)
            .ok_or_else(|| format!("unsupported ADTS sample-rate index {sample_rate_index}"))?;
        let channels = ((data[offset + 2] & 0x01) << 2) | ((data[offset + 3] & 0xc0) >> 6);
        let frame_len = (((data[offset + 3] & 0x03) as usize) << 11)
            | ((data[offset + 4] as usize) << 3)
            | (((data[offset + 5] & 0xe0) as usize) >> 5);

        if frame_len <= header_len {
            return Err("invalid ADTS frame length".into());
        }
        if offset + frame_len > data.len() {
            return Err("truncated ADTS frame".into());
        }

        frames.push(AdtsFrame {
            raw: &data[offset + header_len..offset + frame_len],
            audio_object_type,
            sample_rate_index,
            sample_rate,
            channels,
        });
        offset += frame_len;
    }

    Ok(frames)
}

fn adts_sample_rate(index: u8) -> Option<u32> {
    match index {
        0 => Some(96_000),
        1 => Some(88_200),
        2 => Some(64_000),
        3 => Some(48_000),
        4 => Some(44_100),
        5 => Some(32_000),
        6 => Some(24_000),
        7 => Some(22_050),
        8 => Some(16_000),
        9 => Some(12_000),
        10 => Some(11_025),
        11 => Some(8_000),
        12 => Some(7_350),
        _ => None,
    }
}
