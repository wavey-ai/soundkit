#![cfg(all(target_arch = "wasm32", feature = "aac-lc-bench"))]

use aac_wasm_bench::{decode_soundkit_lc_fixture_pcm_for, AacFixture};
use js_sys::Float32Array;
use soundkit_aac::AacLcAccessUnitDecoder;
use soundkit_wasm::WasmAacLcDecoder;
use wasm_bindgen::prelude::*;
use wasm_bindgen_test::{console_log, wasm_bindgen_test};

const FIXTURE_NAME: &str = "WESTSIDE_MIX_4_CONFIRMATION_130323_256k.aac";
const FIXTURE: &[u8] =
    include_bytes!("../../golden/aac/WESTSIDE_MIX_4_CONFIRMATION_130323_256k.aac");
const ITERATIONS: usize = 5;
const WARMUP_ITERATIONS: usize = 1;
const EXPECTED_CORE_CHECKSUM: u64 = 0x5fc5accb1cc4818d;
const EXPECTED_INTERLEAVED_CHECKSUM: u64 = 0xff51a7e5db594499;
const EXPECTED_INTO_CHECKSUM: u64 = 0x83f5d150e63a4c99;
const EXPECTED_WASM_PCM_CHECKSUM: u64 = 0x39efaeb0d96395e6;
const EXPECTED_PCM_RMS: f64 = 0.162843870;
const EXPECTED_PCM_PEAK: f64 = 0.918334067;

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
fn quality_aac_lc_matches_native_fdk_checked_output() {
    let decoded = decode_soundkit_lc_fixture_pcm_for(AacFixture {
        name: FIXTURE_NAME,
        data: FIXTURE,
    })
    .expect("decode SoundKit AAC-LC fixture in wasm");

    assert_eq!(decoded.decoded_frames, 9171);
    assert_eq!(decoded.samples_per_channel, 9171 * 1024);
    assert_eq!(decoded.sample_rate, 48_000);
    assert_eq!(decoded.channels, 2);

    let stats = decoded.stats();
    console_log!(
        "wasm-native-parity samples={} rms={:.9} peak={:.9} checksum={:016x}",
        stats.sample_count,
        stats.rms,
        stats.peak_abs,
        stats.checksum,
    );

    assert_eq!(stats.sample_count, 9171 * 1024 * 2);
    // The native gate compares this decoder with FDK. This target-specific
    // checksum makes sure the complete WASM output remains unchanged.
    assert_eq!(stats.checksum, EXPECTED_WASM_PCM_CHECKSUM);
    assert!((stats.rms - EXPECTED_PCM_RMS).abs() <= 1.0e-8);
    assert!((stats.peak_abs - EXPECTED_PCM_PEAK).abs() <= 1.0e-8);
}

fn bench_core_decode(frames: &[AdtsFrame<'_>], asc: &[u8]) -> WasmBenchResult {
    let first = frames.first().expect("fixture has frames");
    let mut decoder =
        AacLcAccessUnitDecoder::from_audio_specific_config(asc).expect("create decoder");
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
