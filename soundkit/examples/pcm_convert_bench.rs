//! Time the PCM conversions of the mastering read and write paths over a
//! 26-minute stereo programme, and print a digest of every output.
//!
//! Usage: pcm_convert_bench [MINUTES]
use frame_header::{EncodingFlag, Endianness};
use soundkit::audio_bytes::{deinterleave_vecs_f32, deinterleave_vecs_i16, deinterleave_vecs_s24};
use soundkit::audio_pipeline::{audio_to_f32_channels, f32s_to_le_bytes};
use soundkit::audio_types::AudioData;
use soundkit::wav::{WavSampleFormat, WavStreamEncoder};
use std::time::{Duration, Instant};

const RATE: usize = 48_000;
/// The engine writes WAV in blocks of this many frames.
const WRITE_FRAMES: usize = 8_192;
/// The engine reads WAV in chunks of this many bytes.
const READ_BYTES: usize = 65_536;

/// FNV-1a, for output digests.
struct Fnv(u64);

impl Fnv {
    fn update(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 ^= u64::from(byte);
            self.0 = self.0.wrapping_mul(0x0100_0000_01b3);
        }
    }

    fn floats(&mut self, values: &[f32]) {
        for value in values {
            self.update(&value.to_bits().to_le_bytes());
        }
    }
}

fn report(name: &str, time: Duration, frames: usize, digest: &Fnv) {
    println!(
        "{name:<28} {:>8.3} s {:>7.2} ns/frame digest {:016x}",
        time.as_secs_f64(),
        time.as_secs_f64() * 1e9 / frames as f64,
        digest.0
    );
}

fn main() -> Result<(), String> {
    let minutes: usize = match std::env::args().nth(1) {
        Some(value) => value.parse().map_err(|_| "MINUTES is a number")?,
        None => 26,
    };
    let total = minutes * 60 * RATE;
    // One minute of stereo PCM24 and float, repeated for the whole programme.
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let source: Vec<[i32; 2]> = (0..60 * RATE)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let left = (state & 0xff_ffff) as i32 - 0x80_0000;
            let right = ((state >> 24) & 0xff_ffff) as i32 - 0x80_0000;
            [left, right]
        })
        .collect();
    let floats: Vec<[f32; 2]> = source
        .iter()
        .map(|frame| frame.map(|sample| sample as f32 / 8_388_608.0 * 1.2))
        .collect();
    let blocks = |size: usize| {
        (0..total)
            .step_by(size)
            .map(move |start| (start, size.min(total - start)))
    };

    // push_planar_f32
    let mut encoder = WavStreamEncoder::new(WavSampleFormat::F32, RATE as u32, 2, total as u64)?;
    let (mut digest, mut time, mut planar) =
        (Fnv(0xcbf2_9ce4_8422_2325), Duration::ZERO, Vec::new());
    for (start, count) in blocks(WRITE_FRAMES) {
        planar.clear();
        for channel in [0, 1] {
            planar.extend((start..start + count).map(|i| floats[i % floats.len()][channel]));
        }
        let begin = Instant::now();
        let bytes = encoder.push_planar_f32(&planar, count)?;
        time += begin.elapsed();
        digest.update(&bytes);
    }
    encoder.finish()?;
    report("push_planar_f32", time, total, &digest);

    // push_planar_i24
    let mut encoder = WavStreamEncoder::new(WavSampleFormat::I24, RATE as u32, 2, total as u64)?;
    let (mut digest, mut time, mut planar) =
        (Fnv(0xcbf2_9ce4_8422_2325), Duration::ZERO, Vec::new());
    for (start, count) in blocks(WRITE_FRAMES) {
        planar.clear();
        for channel in [0, 1] {
            planar.extend((start..start + count).map(|i| source[i % source.len()][channel]));
        }
        let begin = Instant::now();
        let bytes = encoder.push_planar_i24(&planar, count)?;
        time += begin.elapsed();
        digest.update(&bytes);
    }
    encoder.finish()?;
    report("push_planar_i24", time, total, &digest);

    // push_planar_i16
    let mut encoder = WavStreamEncoder::new(WavSampleFormat::I16, RATE as u32, 2, total as u64)?;
    let (mut digest, mut time, mut planar) =
        (Fnv(0xcbf2_9ce4_8422_2325), Duration::ZERO, Vec::new());
    for (start, count) in blocks(WRITE_FRAMES) {
        planar.clear();
        for channel in [0, 1] {
            planar.extend(
                (start..start + count).map(|i| (source[i % source.len()][channel] >> 8) as i16),
            );
        }
        let begin = Instant::now();
        let bytes = encoder.push_planar_i16(&planar, count)?;
        time += begin.elapsed();
        digest.update(&bytes);
    }
    encoder.finish()?;
    report("push_planar_i16", time, total, &digest);

    // f32s_to_le_bytes, as the engine's scratch writes interleaved blocks.
    let (mut digest, mut time, mut flat) = (Fnv(0xcbf2_9ce4_8422_2325), Duration::ZERO, Vec::new());
    for (start, count) in blocks(16_384) {
        flat.clear();
        flat.extend((start..start + count).flat_map(|i| floats[i % floats.len()]));
        let begin = Instant::now();
        let bytes = f32s_to_le_bytes(&flat);
        time += begin.elapsed();
        digest.update(&bytes);
    }
    report("f32s_to_le_bytes", time, total, &digest);

    // The read path: interleaved bytes in 64 KiB pieces to planar samples.
    let s24: Vec<u8> = source
        .iter()
        .flatten()
        .flat_map(|s| s.to_le_bytes()[..3].to_vec())
        .collect();
    let f32le: Vec<u8> = floats
        .iter()
        .flatten()
        .flat_map(|s| s.to_le_bytes())
        .collect();
    let s16: Vec<u8> = source
        .iter()
        .flatten()
        .flat_map(|s| ((s >> 8) as i16).to_le_bytes())
        .collect();
    for (name, bytes, bits, format) in [
        (
            "audio_to_f32_channels s24",
            &s24,
            24u8,
            EncodingFlag::PCMSigned,
        ),
        (
            "audio_to_f32_channels f32",
            &f32le,
            32,
            EncodingFlag::PCMFloat,
        ),
        (
            "audio_to_f32_channels s16",
            &s16,
            16,
            EncodingFlag::PCMSigned,
        ),
    ] {
        let frame_bytes = usize::from(bits / 8) * 2;
        let piece = READ_BYTES / frame_bytes * frame_bytes;
        let (mut digest, mut time, mut frames) = (Fnv(0xcbf2_9ce4_8422_2325), Duration::ZERO, 0);
        while frames < total {
            for chunk in bytes.chunks(piece) {
                let take = (total - frames).min(chunk.len() / frame_bytes);
                if take == 0 {
                    break;
                }
                let audio = AudioData::new(
                    bits,
                    2,
                    RATE as u32,
                    chunk[..take * frame_bytes].to_vec(),
                    format,
                    Endianness::LittleEndian,
                );
                let begin = Instant::now();
                let channels = audio_to_f32_channels(&audio)?;
                time += begin.elapsed();
                for channel in &channels {
                    digest.floats(channel);
                }
                frames += take;
            }
        }
        report(name, time, total, &digest);
    }

    // The deinterleavers on their own.
    for (name, bytes, width) in [
        ("deinterleave_vecs_s24", &s24, 3usize),
        ("deinterleave_vecs_f32", &f32le, 4),
        ("deinterleave_vecs_i16", &s16, 2),
    ] {
        let piece = READ_BYTES / (width * 2) * (width * 2);
        let (mut digest, mut time, mut frames) = (Fnv(0xcbf2_9ce4_8422_2325), Duration::ZERO, 0);
        while frames < total {
            for chunk in bytes.chunks(piece) {
                let begin = Instant::now();
                match width {
                    3 => {
                        let planes = deinterleave_vecs_s24(chunk, 2);
                        time += begin.elapsed();
                        planes
                            .iter()
                            .flatten()
                            .for_each(|s| digest.update(&s.to_le_bytes()));
                    }
                    4 => {
                        let planes = deinterleave_vecs_f32(chunk, 2);
                        time += begin.elapsed();
                        planes.iter().for_each(|p| digest.floats(p));
                    }
                    _ => {
                        let planes = deinterleave_vecs_i16(chunk, 2);
                        time += begin.elapsed();
                        planes
                            .iter()
                            .flatten()
                            .for_each(|s| digest.update(&s.to_le_bytes()));
                    }
                }
                frames += chunk.len() / (width * 2);
            }
        }
        report(name, time, frames, &digest);
    }
    Ok(())
}
