use libopus_rs::{Application, Decoder, Encoder};
use std::env;
use std::fs;
use std::io;
use std::path::Path;

const CHANNELS: usize = 2;
const SAMPLE_RATE: i32 = 48_000;
const VALID_FRAME_SIZES: [usize; 4] = [120, 240, 480, 960];

fn usage() -> ! {
    eprintln!(
        "usage: raw_pcm_roundtrip <frame-size> <bitrate> <cbr|vbr> <input.f32le> <output.f32le>"
    );
    std::process::exit(2);
}

fn read_f32le(path: &Path) -> io::Result<Vec<f32>> {
    let data = fs::read(path)?;
    if data.len() % 4 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "input has a partial f32 sample",
        ));
    }
    let samples = data
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect::<Vec<_>>();
    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "input contains a non-finite sample",
        ));
    }
    Ok(samples)
}

fn write_f32le(path: &Path, samples: &[f32]) -> io::Result<()> {
    let mut data = Vec::with_capacity(samples.len() * 4);
    for sample in samples {
        data.extend_from_slice(&sample.to_le_bytes());
    }
    fs::write(path, data)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = env::args().collect::<Vec<_>>();
    if args.len() != 6 {
        usage();
    }
    let frame_size = args[1].parse::<usize>().unwrap_or_else(|_| usage());
    if !VALID_FRAME_SIZES.contains(&frame_size) {
        usage();
    }
    let bitrate = args[2].parse::<i32>().unwrap_or_else(|_| usage());
    let vbr = match args[3].as_str() {
        "cbr" => false,
        "vbr" => true,
        _ => usage(),
    };
    let input_path = Path::new(&args[4]);
    let output_path = Path::new(&args[5]);
    let input = read_f32le(input_path)?;
    if input.len() % CHANNELS != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "input has a partial stereo frame",
        )
        .into());
    }

    let total_samples = input.len();
    let frame_samples = frame_size * CHANNELS;
    let mut encoder = Encoder::new(SAMPLE_RATE, CHANNELS, Application::Audio)?;
    encoder.set_bitrate(bitrate)?;
    encoder.set_vbr(vbr)?;
    let mut decoder = Decoder::new(SAMPLE_RATE, CHANNELS)?;
    let mut output = Vec::with_capacity(total_samples + frame_samples);
    let mut packet_bytes = 0usize;
    let mut packet_count = 0usize;
    let mut packet_min = usize::MAX;
    let mut packet_max = 0usize;

    for frame in input.chunks(frame_samples) {
        let mut padded = vec![0.0f32; frame_samples];
        padded[..frame.len()].copy_from_slice(frame);
        let packet = encoder.encode_f32(&padded, frame_size)?;
        packet_bytes += packet.len();
        packet_count += 1;
        packet_min = packet_min.min(packet.len());
        packet_max = packet_max.max(packet.len());
        output.extend(decoder.decode_f32(&packet, false)?);
    }
    output.truncate(total_samples);
    write_f32le(output_path, &output)?;

    let mode = if vbr { "vbr" } else { "cbr" };
    println!(
        "{{\"codec\":\"libopus-rs\",\"sample_rate\":{SAMPLE_RATE},\"channels\":{CHANNELS},\"frame_size\":{frame_size},\"bitrate\":{bitrate},\"mode\":\"{mode}\",\"packets\":{packet_count},\"packet_bytes\":{packet_bytes},\"packet_min\":{packet_min},\"packet_max\":{packet_max}}}"
    );
    Ok(())
}
