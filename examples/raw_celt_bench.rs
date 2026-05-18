use libopus_rs::{Application, Decoder, Encoder, CELT_FRAME_SIZES_48K};
use std::env;
use std::hint::black_box;
use std::time::Instant;

const SAMPLE_RATE: usize = 48_000;
const CHANNELS: usize = 2;
const BITRATES: [i32; 9] = [
    48_000, 96_000, 128_000, 160_000, 192_000, 256_000, 320_000, 384_000, 512_000,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BenchMode {
    Cbr,
    Vbr,
}

impl BenchMode {
    const fn label(self) -> &'static str {
        match self {
            Self::Cbr => "cbr",
            Self::Vbr => "vbr",
        }
    }
}

#[derive(Clone, Copy)]
struct Options {
    repeats: usize,
    seconds: usize,
    mode: Option<BenchMode>,
    dump_packets: Option<usize>,
}

struct EncodeResult {
    packets: Vec<Vec<u8>>,
    bytes: usize,
    checksum: u64,
    min_packet: usize,
    max_packet: usize,
}

fn usage() -> ! {
    eprintln!(
        "usage: raw_celt_bench [--repeats n] [--seconds n] [--mode cbr|vbr|both] [--dump-packets n]"
    );
    std::process::exit(2);
}

fn parse_options() -> Options {
    let mut options = Options {
        repeats: 21,
        seconds: 4,
        mode: None,
        dump_packets: None,
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
            "--mode" => {
                i += 1;
                options.mode = match args.get(i).map(String::as_str) {
                    Some("cbr") => Some(BenchMode::Cbr),
                    Some("vbr") => Some(BenchMode::Vbr),
                    Some("both") => None,
                    _ => usage(),
                };
            }
            "--dump-packets" => {
                i += 1;
                options.dump_packets = Some(
                    args.get(i)
                        .and_then(|value| value.parse().ok())
                        .unwrap_or_else(|| usage()),
                );
            }
            _ => usage(),
        }
        i += 1;
    }
    if options.repeats == 0 || options.seconds == 0 || options.dump_packets == Some(0) {
        usage();
    }
    options
}

fn modes(options: &Options) -> &'static [BenchMode] {
    match options.mode {
        Some(BenchMode::Cbr) => &[BenchMode::Cbr],
        Some(BenchMode::Vbr) => &[BenchMode::Vbr],
        None => &[BenchMode::Cbr, BenchMode::Vbr],
    }
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
) -> Result<EncodeResult, Box<dyn std::error::Error>> {
    let frames = pcm.len() / (frame_size * CHANNELS);
    let mut packets = Vec::with_capacity(frames);
    let mut bytes = 0usize;
    let mut checksum = 0u64;
    let mut min_packet = usize::MAX;
    let mut max_packet = 0usize;
    for frame in 0..frames {
        let start = frame * frame_size * CHANNELS;
        let end = start + frame_size * CHANNELS;
        let packet = encoder.encode_f32(black_box(&pcm[start..end]), frame_size)?;
        bytes += packet.len();
        min_packet = min_packet.min(packet.len());
        max_packet = max_packet.max(packet.len());
        checksum = checksum.wrapping_add(packet_checksum(&packet));
        packets.push(packet);
    }
    black_box(checksum);
    Ok(EncodeResult {
        packets,
        bytes,
        checksum,
        min_packet,
        max_packet,
    })
}

fn encode_packets(
    pcm: &[f32],
    frame_size: usize,
    bitrate: i32,
    mode: BenchMode,
) -> Result<EncodeResult, Box<dyn std::error::Error>> {
    let mut encoder = Encoder::new(
        SAMPLE_RATE as i32,
        CHANNELS,
        Application::RestrictedLowDelay,
    )?;
    encoder.set_bitrate(bitrate)?;
    encoder.set_vbr(mode == BenchMode::Vbr)?;
    encode_with_encoder(&mut encoder, pcm, frame_size)
}

fn time_encode(
    pcm: &[f32],
    frame_size: usize,
    bitrate: i32,
    mode: BenchMode,
    repeats: usize,
) -> Result<(f64, usize, usize, usize, u64), Box<dyn std::error::Error>> {
    let mut times = Vec::with_capacity(repeats);
    let mut last_bytes = 0usize;
    let mut last_min_packet = 0usize;
    let mut last_max_packet = 0usize;
    let mut last_checksum = 0u64;
    for _ in 0..repeats {
        let mut encoder = Encoder::new(
            SAMPLE_RATE as i32,
            CHANNELS,
            Application::RestrictedLowDelay,
        )?;
        encoder.set_bitrate(bitrate)?;
        encoder.set_vbr(mode == BenchMode::Vbr)?;
        let start = Instant::now();
        let encoded = encode_with_encoder(&mut encoder, pcm, frame_size)?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        black_box(&encoded.packets);
        times.push(elapsed);
        last_bytes = encoded.bytes;
        last_min_packet = encoded.min_packet;
        last_max_packet = encoded.max_packet;
        last_checksum = encoded.checksum;
    }
    Ok((
        median(&mut times),
        last_bytes,
        last_min_packet,
        last_max_packet,
        last_checksum,
    ))
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

fn print_packet_hex(packet: &[u8]) {
    for byte in packet {
        print!("{byte:02x}");
    }
    println!();
}

fn dump_packets(
    pcm: &[f32],
    options: &Options,
    limit: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("impl\tmode\tframe_size\tframe_ms\tbitrate\tframe\tlen\thex");
    for &mode in modes(options) {
        for &frame_size in &CELT_FRAME_SIZES_48K {
            for &bitrate in &BITRATES {
                let encoded = encode_packets(pcm, frame_size, bitrate, mode)?;
                for (frame, packet) in encoded.packets.iter().take(limit).enumerate() {
                    print!(
                        "rust\t{}\t{}\t{:.1}\t{}\t{}\t{}\t",
                        mode.label(),
                        frame_size,
                        frame_size as f64 * 1000.0 / SAMPLE_RATE as f64,
                        bitrate,
                        frame,
                        packet.len()
                    );
                    print_packet_hex(packet);
                }
            }
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = parse_options();
    let pcm = generate_fixture(options.seconds);
    if let Some(limit) = options.dump_packets {
        return dump_packets(&pcm, &options, limit);
    }

    println!("impl\tmode\tframe_size\tframe_ms\tbitrate\tencode_ms\tdecode_ms\tbytes\tmin_packet\tmax_packet\tchecksum");
    for &mode in modes(&options) {
        for &frame_size in &CELT_FRAME_SIZES_48K {
            for &bitrate in &BITRATES {
                let (encode_ms, bytes, min_packet, max_packet, encode_checksum) =
                    time_encode(&pcm, frame_size, bitrate, mode, options.repeats)?;
                let encoded = encode_packets(&pcm, frame_size, bitrate, mode)?;
                let (decode_ms, decode_checksum) =
                    time_decode(&encoded.packets, frame_size, options.repeats)?;
                let checksum = encode_checksum ^ u64::from(decode_checksum.to_bits());
                println!(
                    "rust\t{}\t{}\t{:.1}\t{}\t{:.4}\t{:.4}\t{}\t{}\t{}\t{}",
                    mode.label(),
                    frame_size,
                    frame_size as f64 * 1000.0 / SAMPLE_RATE as f64,
                    bitrate,
                    encode_ms,
                    decode_ms,
                    bytes,
                    min_packet,
                    max_packet,
                    checksum
                );
            }
        }
    }
    Ok(())
}
