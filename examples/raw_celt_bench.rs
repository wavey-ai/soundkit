use libopus_rs::{Application, Decoder, Encoder, CELT_FRAME_SIZES_48K};
use std::env;
use std::hint::black_box;
use std::time::Instant;

const SAMPLE_RATE: usize = 48_000;
const CHANNELS: usize = 2;
const BITRATES: [i32; 3] = [48_000, 96_000, 128_000];

#[derive(Clone, Copy)]
struct Options {
    repeats: usize,
    seconds: usize,
}

fn usage() -> ! {
    eprintln!("usage: raw_celt_bench [--repeats n] [--seconds n]");
    std::process::exit(2);
}

fn parse_options() -> Options {
    let mut options = Options {
        repeats: 21,
        seconds: 4,
    };
    let args = env::args().collect::<Vec<_>>();
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--repeats" => {
                i += 1;
                options.repeats = args
                    .get(i)
                    .and_then(|value| value.parse().ok())
                    .unwrap_or_else(|| usage());
            }
            "--seconds" => {
                i += 1;
                options.seconds = args
                    .get(i)
                    .and_then(|value| value.parse().ok())
                    .unwrap_or_else(|| usage());
            }
            _ => usage(),
        }
        i += 1;
    }
    if options.repeats == 0 || options.seconds == 0 {
        usage();
    }
    options
}

fn generate_fixture(seconds: usize) -> Vec<f32> {
    let frames = SAMPLE_RATE * seconds;
    let mut pcm = Vec::with_capacity(frames * CHANNELS);
    let mut noise = 0x1234_5678u32;
    for i in 0..frames {
        let t = i as f32 / SAMPLE_RATE as f32;
        noise = noise.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let n = ((noise >> 9) as f32 / ((1u32 << 23) as f32)) - 1.0;
        let transient = 0.35 * (-900.0 * (t - 1.37) * (t - 1.37)).exp();
        let left = 0.29 * (2.0 * std::f32::consts::PI * 261.63 * t).sin()
            + 0.17 * (2.0 * std::f32::consts::PI * 659.25 * t + 0.2).sin()
            + 0.05 * (2.0 * std::f32::consts::PI * 4210.0 * t).sin()
            + 0.015 * n
            + transient;
        let right = 0.25 * (2.0 * std::f32::consts::PI * 329.63 * t + 0.4).sin()
            - 0.13 * (2.0 * std::f32::consts::PI * 880.0 * t).sin()
            + 0.05 * (2.0 * std::f32::consts::PI * 3910.0 * t + 0.7).sin()
            - 0.012 * n
            - 0.8 * transient;
        pcm.push(left.clamp(-1.0, 1.0));
        pcm.push(right.clamp(-1.0, 1.0));
    }
    pcm
}

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(|a, b| a.total_cmp(b));
    samples[samples.len() / 2]
}

fn packet_checksum(packet: &[u8]) -> u64 {
    let first = packet.first().copied().unwrap_or(0) as u64;
    let last = packet.last().copied().unwrap_or(0) as u64;
    ((packet.len() as u64) << 16) ^ (first << 8) ^ last
}

fn decoded_checksum(decoded: &[f32]) -> f32 {
    let first = decoded.first().copied().unwrap_or(0.0);
    let middle = decoded.get(decoded.len() / 2).copied().unwrap_or(0.0);
    let last = decoded.last().copied().unwrap_or(0.0);
    first + middle + last
}

fn encode_with_encoder(
    encoder: &mut Encoder,
    pcm: &[f32],
    frame_size: usize,
) -> Result<(Vec<Vec<u8>>, usize, u64), Box<dyn std::error::Error>> {
    let frames = pcm.len() / (frame_size * CHANNELS);
    let mut packets = Vec::with_capacity(frames);
    let mut bytes = 0usize;
    let mut checksum = 0u64;
    for frame in 0..frames {
        let start = frame * frame_size * CHANNELS;
        let end = start + frame_size * CHANNELS;
        let packet = encoder.encode_f32(black_box(&pcm[start..end]), frame_size)?;
        bytes += packet.len();
        checksum = checksum.wrapping_add(packet_checksum(&packet));
        packets.push(packet);
    }
    black_box(checksum);
    Ok((packets, bytes, checksum))
}

fn encode_packets(
    pcm: &[f32],
    frame_size: usize,
    bitrate: i32,
) -> Result<(Vec<Vec<u8>>, usize, u64), Box<dyn std::error::Error>> {
    let mut encoder = Encoder::new(
        SAMPLE_RATE as i32,
        CHANNELS,
        Application::RestrictedLowDelay,
    )?;
    encoder.set_bitrate(bitrate)?;
    encode_with_encoder(&mut encoder, pcm, frame_size)
}

fn time_encode(
    pcm: &[f32],
    frame_size: usize,
    bitrate: i32,
    repeats: usize,
) -> Result<(f64, usize, u64), Box<dyn std::error::Error>> {
    let mut times = Vec::with_capacity(repeats);
    let mut last_bytes = 0usize;
    let mut last_checksum = 0u64;
    for _ in 0..repeats {
        let mut encoder = Encoder::new(
            SAMPLE_RATE as i32,
            CHANNELS,
            Application::RestrictedLowDelay,
        )?;
        encoder.set_bitrate(bitrate)?;
        let start = Instant::now();
        let (packets, bytes, checksum) = encode_with_encoder(&mut encoder, pcm, frame_size)?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        black_box(&packets);
        times.push(elapsed);
        last_bytes = bytes;
        last_checksum = checksum;
    }
    Ok((median(&mut times), last_bytes, last_checksum))
}

fn time_decode(
    packets: &[Vec<u8>],
    frame_size: usize,
    repeats: usize,
) -> Result<(f64, f32), Box<dyn std::error::Error>> {
    let mut times = Vec::with_capacity(repeats);
    let mut last_checksum = 0.0f32;
    for _ in 0..repeats {
        let mut decoder = Decoder::new(SAMPLE_RATE as i32, CHANNELS)?;
        let start = Instant::now();
        let mut checksum = 0.0f32;
        for packet in packets {
            let decoded = decoder.decode_f32(black_box(packet), false)?;
            if decoded.len() != frame_size * CHANNELS {
                return Err("unexpected decoded frame size".into());
            }
            checksum += decoded_checksum(&decoded);
        }
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        black_box(checksum);
        times.push(elapsed);
        last_checksum = checksum;
    }
    Ok((median(&mut times), last_checksum))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = parse_options();
    let pcm = generate_fixture(options.seconds);
    println!("impl\tframe_size\tframe_ms\tbitrate\tencode_ms\tdecode_ms\tbytes\tchecksum");
    for &frame_size in &CELT_FRAME_SIZES_48K {
        for &bitrate in &BITRATES {
            let (encode_ms, bytes, encode_checksum) =
                time_encode(&pcm, frame_size, bitrate, options.repeats)?;
            let (packets, _, _) = encode_packets(&pcm, frame_size, bitrate)?;
            let (decode_ms, decode_checksum) = time_decode(&packets, frame_size, options.repeats)?;
            let checksum = encode_checksum ^ u64::from(decode_checksum.to_bits());
            println!(
                "rust\t{}\t{:.1}\t{}\t{:.4}\t{:.4}\t{}\t{}",
                frame_size,
                frame_size as f64 * 1000.0 / SAMPLE_RATE as f64,
                bitrate,
                encode_ms,
                decode_ms,
                bytes,
                checksum
            );
        }
    }
    Ok(())
}
