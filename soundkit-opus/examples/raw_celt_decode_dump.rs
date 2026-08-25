use soundkit::audio_packet::Decoder as SoundkitDecoder;
use soundkit_opus::Decoder;
use std::collections::BTreeMap;
use std::env;
use std::hint::black_box;
use std::io::{self, BufRead};
use std::time::Instant;

const SAMPLE_RATE: usize = 48_000;
const CHANNELS: usize = 2;

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct CaseKey {
    implementation: String,
    mode: String,
    frame_size: usize,
    bitrate: i32,
}

#[derive(Clone, Copy)]
struct DecodeQuality {
    lag_frames: isize,
    snr_db: f64,
}

#[derive(Default)]
struct Options {
    seconds: usize,
    repeats: Option<usize>,
    pcm_bits: u8,
    implementation: Option<String>,
    mode: Option<String>,
    frame_size: Option<usize>,
    bitrate: Option<i32>,
}

fn usage() -> ! {
    eprintln!(
        "usage: raw_celt_decode_dump [--seconds n] [--repeats n --pcm-bits 16|24] [--impl c|rust] [--mode cbr|vbr] [--frame-size n] [--bitrate bps] < packet-dump.tsv"
    );
    std::process::exit(2);
}

fn parse_options() -> Options {
    let mut options = Options {
        seconds: 1,
        pcm_bits: 24,
        ..Options::default()
    };
    let args = env::args().collect::<Vec<_>>();
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--seconds" => {
                i += 1;
                options.seconds = args
                    .get(i)
                    .and_then(|value| value.parse().ok())
                    .unwrap_or_else(|| usage());
            }
            "--repeats" => {
                i += 1;
                options.repeats = Some(
                    args.get(i)
                        .and_then(|value| value.parse().ok())
                        .unwrap_or_else(|| usage()),
                );
            }
            "--pcm-bits" => {
                i += 1;
                options.pcm_bits = match args.get(i).map(String::as_str) {
                    Some("16") => 16,
                    Some("24") => 24,
                    _ => usage(),
                };
            }
            "--impl" => {
                i += 1;
                let value = args.get(i).cloned().unwrap_or_else(|| usage());
                if value != "c" && value != "rust" {
                    usage();
                }
                options.implementation = Some(value);
            }
            "--mode" => {
                i += 1;
                let value = args.get(i).cloned().unwrap_or_else(|| usage());
                if value != "cbr" && value != "vbr" {
                    usage();
                }
                options.mode = Some(value);
            }
            "--frame-size" => {
                i += 1;
                options.frame_size = Some(
                    args.get(i)
                        .and_then(|value| value.parse().ok())
                        .unwrap_or_else(|| usage()),
                );
            }
            "--bitrate" => {
                i += 1;
                options.bitrate = Some(
                    args.get(i)
                        .and_then(|value| value.parse().ok())
                        .unwrap_or_else(|| usage()),
                );
            }
            _ => usage(),
        }
        i += 1;
    }
    if options.seconds == 0 || options.repeats == Some(0) || options.frame_size == Some(0) {
        usage();
    }
    options
}

fn generate_fixture(seconds: usize) -> Vec<f32> {
    let frames = SAMPLE_RATE * seconds;
    let mut pcm = Vec::with_capacity(frames * CHANNELS);
    let mut noise = 0x1234_5678u32;
    for i in 0..frames {
        noise = noise.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let tri_a = triangle_wave((i as u32).wrapping_mul(713));
        let tri_b = triangle_wave((i as u32).wrapping_mul(1451).wrapping_add(0x4000));
        let tri_c = triangle_wave((i as u32).wrapping_mul(977).wrapping_add(0x2000));
        let tri_d = triangle_wave((i as u32).wrapping_mul(3511).wrapping_add(0x6000));
        let n = centered_u16(noise) * (1.0 / 4096.0);
        let pulse = (i as u32) & 8191;
        let transient = if pulse < 64 {
            (64 - pulse) as f32 * (1.0 / 512.0)
        } else {
            0.0
        };
        let left = 0.25 * tri_a + 0.125 * tri_b + n + transient;
        let right = 0.21875 * tri_c - 0.09375 * tri_d - n - 0.5 * transient;
        pcm.push(left.clamp(-1.0, 1.0));
        pcm.push(right.clamp(-1.0, 1.0));
    }
    pcm
}

fn centered_u16(value: u32) -> f32 {
    ((value & 0xffff) as i32 - 32_768) as f32 * (1.0 / 32_768.0)
}

fn triangle_wave(phase: u32) -> f32 {
    let p = (phase & 0xffff) as i32;
    let v = if p < 32_768 { p - 16_384 } else { 49_152 - p };
    v as f32 * (1.0 / 16_384.0)
}

fn aligned_quality(reference: &[f32], decoded: &[f32]) -> DecodeQuality {
    let total_frames = (reference.len().min(decoded.len())) / CHANNELS;
    let max_lag = (SAMPLE_RATE / 50).min(total_frames.saturating_sub(16));
    let compare_frames = total_frames.saturating_sub(max_lag);
    let mut best = DecodeQuality {
        lag_frames: 0,
        snr_db: f64::NEG_INFINITY,
    };

    for lag in -(max_lag as isize)..=(max_lag as isize) {
        let (reference_start, decoded_start) = if lag >= 0 {
            (lag as usize, 0)
        } else {
            (0, (-lag) as usize)
        };
        let mut signal = 0.0f64;
        let mut error = 0.0f64;
        for frame in 0..compare_frames {
            let ref_base = (reference_start + frame) * CHANNELS;
            let dec_base = (decoded_start + frame) * CHANNELS;
            for channel in 0..CHANNELS {
                let expected = f64::from(reference[ref_base + channel]);
                let actual = f64::from(decoded[dec_base + channel]);
                let diff = expected - actual;
                signal += expected * expected;
                error += diff * diff;
            }
        }
        let snr_db = if error <= f64::EPSILON {
            f64::INFINITY
        } else if signal <= f64::EPSILON {
            f64::NEG_INFINITY
        } else {
            10.0 * (signal / error).log10()
        };
        if snr_db > best.snr_db {
            best = DecodeQuality {
                lag_frames: lag,
                snr_db,
            };
        }
    }

    best
}

fn decode_hex(hex: &str) -> Option<Vec<u8>> {
    if hex.len() % 2 != 0 {
        return None;
    }
    let mut packet = Vec::with_capacity(hex.len() / 2);
    let bytes = hex.as_bytes();
    for i in (0..bytes.len()).step_by(2) {
        let hi = (bytes[i] as char).to_digit(16)?;
        let lo = (bytes[i + 1] as char).to_digit(16)?;
        packet.push(((hi << 4) | lo) as u8);
    }
    Some(packet)
}

fn parse_dump(options: &Options) -> Result<BTreeMap<CaseKey, Vec<Vec<u8>>>, String> {
    let mut cases = BTreeMap::<CaseKey, Vec<Vec<u8>>>::new();
    for line in io::stdin().lock().lines() {
        let line = line.map_err(|err| err.to_string())?;
        if line.starts_with("impl\t") || line.trim().is_empty() {
            continue;
        }
        let cols = line.split('\t').collect::<Vec<_>>();
        if cols.len() != 8 {
            return Err(format!(
                "expected 8 TSV columns, got {}: {line}",
                cols.len()
            ));
        }
        let implementation = cols[0].to_string();
        let mode = cols[1].to_string();
        let frame_size = cols[2]
            .parse::<usize>()
            .map_err(|_| format!("invalid frame_size: {}", cols[2]))?;
        let bitrate = cols[4]
            .parse::<i32>()
            .map_err(|_| format!("invalid bitrate: {}", cols[4]))?;

        if options
            .implementation
            .as_ref()
            .is_some_and(|value| value != &implementation)
            || options.mode.as_ref().is_some_and(|value| value != &mode)
            || options.frame_size.is_some_and(|value| value != frame_size)
            || options.bitrate.is_some_and(|value| value != bitrate)
        {
            continue;
        }

        let packet =
            decode_hex(cols[7]).ok_or_else(|| format!("invalid hex packet: {}", cols[7]))?;
        let expected_len = cols[6]
            .parse::<usize>()
            .map_err(|_| format!("invalid packet length: {}", cols[6]))?;
        if packet.len() != expected_len {
            return Err(format!(
                "packet length mismatch: header={}, hex={}",
                expected_len,
                packet.len()
            ));
        }

        let key = CaseKey {
            implementation,
            mode,
            frame_size,
            bitrate,
        };
        cases.entry(key).or_default().push(packet);
    }
    Ok(cases)
}

fn decode_case(
    packets: &[Vec<u8>],
    frame_size: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut decoder = Decoder::new(SAMPLE_RATE as i32, CHANNELS)?;
    let mut decoded = Vec::with_capacity(packets.len() * frame_size * CHANNELS);
    let mut samples = Vec::new();
    for (frame, packet) in packets.iter().enumerate() {
        let decoded_frames = decoder.decode_f32_into(packet, false, &mut samples)?;
        if decoded_frames != frame_size || samples.len() != frame_size * CHANNELS {
            return Err(format!(
                "unexpected decoded frame size at frame {frame}: {}",
                samples.len()
            )
            .into());
        }
        decoded.extend_from_slice(&samples);
    }
    Ok(decoded)
}

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(|left, right| left.total_cmp(right));
    samples[samples.len() / 2]
}

fn time_decode_case(
    packets: &[Vec<u8>],
    frame_size: usize,
    repeats: usize,
    pcm_bits: u8,
) -> Result<(f64, u64), Box<dyn std::error::Error>> {
    let mut times = Vec::with_capacity(repeats);
    let mut checksum = 0_u64;
    let mut output_i16 = vec![0_i16; frame_size * CHANNELS];
    let mut output_i24 = vec![0_i32; frame_size * CHANNELS];

    for _ in 0..repeats {
        let mut decoder = Decoder::new(SAMPLE_RATE as i32, CHANNELS)?;
        let mut repeat_checksum = 0_i64;
        let start = Instant::now();
        for packet in packets {
            let decoded_frames = if pcm_bits == 16 {
                decoder.decode_i16(black_box(packet), &mut output_i16, false)?
            } else {
                decoder.decode_i32(black_box(packet), &mut output_i24, false)?
            };
            if decoded_frames != frame_size {
                return Err(format!("unexpected decoded frame size: {decoded_frames}").into());
            }
            if pcm_bits == 16 {
                repeat_checksum = repeat_checksum
                    .wrapping_add(i64::from(output_i16[0]))
                    .wrapping_add(i64::from(output_i16[output_i16.len() / 2]))
                    .wrapping_add(i64::from(output_i16[output_i16.len() - 1]));
            } else {
                repeat_checksum = repeat_checksum
                    .wrapping_add(i64::from(output_i24[0]))
                    .wrapping_add(i64::from(output_i24[output_i24.len() / 2]))
                    .wrapping_add(i64::from(output_i24[output_i24.len() - 1]));
            }
        }
        times.push(start.elapsed().as_secs_f64() * 1000.0);
        checksum = black_box(repeat_checksum as u64);
    }

    Ok((median(&mut times), checksum))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = parse_options();
    let cases = parse_dump(&options)?;
    if cases.is_empty() {
        return Err("no matching packets found".into());
    }

    if let Some(repeats) = options.repeats {
        println!(
            "packet_impl\tdecoder_impl\tmode\tframe_size\tframe_ms\tbitrate\tpcm_bits\tpackets\tbytes\tdecode_ms\tchecksum"
        );
        for (key, packets) in cases {
            let packet_bytes = packets.iter().map(Vec::len).sum::<usize>();
            let (decode_ms, checksum) =
                time_decode_case(&packets, key.frame_size, repeats, options.pcm_bits)?;
            println!(
                "{}\trust\t{}\t{}\t{:.1}\t{}\t{}\t{}\t{}\t{:.4}\t{}",
                key.implementation,
                key.mode,
                key.frame_size,
                key.frame_size as f64 * 1000.0 / SAMPLE_RATE as f64,
                key.bitrate,
                options.pcm_bits,
                packets.len(),
                packet_bytes,
                decode_ms,
                checksum
            );
        }
    } else {
        let reference = generate_fixture(options.seconds);
        println!("impl\tmode\tframe_size\tframe_ms\tbitrate\tframes\tquality_lag\tquality_snr_db");
        for (key, packets) in cases {
            let decoded = decode_case(&packets, key.frame_size)?;
            let quality = aligned_quality(&reference, &decoded);
            println!(
                "{}\t{}\t{}\t{:.1}\t{}\t{}\t{}\t{:.2}",
                key.implementation,
                key.mode,
                key.frame_size,
                key.frame_size as f64 * 1000.0 / SAMPLE_RATE as f64,
                key.bitrate,
                packets.len(),
                quality.lag_frames,
                quality.snr_db
            );
        }
    }

    Ok(())
}
