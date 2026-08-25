use soundkit_opus::{Application, Decoder, Encoder};
use std::env;
use std::fs;
use std::io;
use std::path::Path;

const CHANNELS: usize = 2;
const SAMPLE_RATE: i32 = 48_000;
const VALID_FRAME_SIZES: [usize; 4] = [120, 240, 480, 960];

fn usage() -> ! {
    eprintln!(
        "usage: raw_pcm_roundtrip [--pcm-bits 16|24] <frame-size> <bitrate> \
         <cbr|vbr> <input.raw> <output.f32le>"
    );
    std::process::exit(2);
}

enum PcmInput {
    F32(Vec<f32>),
    I16(Vec<i16>),
    I24(Vec<i32>),
}

impl PcmInput {
    fn len(&self) -> usize {
        match self {
            Self::F32(samples) => samples.len(),
            Self::I16(samples) => samples.len(),
            Self::I24(samples) => samples.len(),
        }
    }

    fn label(&self) -> &'static str {
        match self {
            Self::F32(_) => "float",
            Self::I16(_) => "16",
            Self::I24(_) => "24",
        }
    }
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

fn read_i16le(path: &Path) -> io::Result<Vec<i16>> {
    let data = fs::read(path)?;
    if data.len() % 2 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "input has a partial i16 sample",
        ));
    }
    Ok(data
        .chunks_exact(2)
        .map(|bytes| i16::from_le_bytes([bytes[0], bytes[1]]))
        .collect())
}

fn read_i24_s32le(path: &Path) -> io::Result<Vec<i32>> {
    let data = fs::read(path)?;
    if data.len() % 4 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "input has a partial i32 sample",
        ));
    }
    let samples = data
        .chunks_exact(4)
        .map(|bytes| i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect::<Vec<_>>();
    if samples
        .iter()
        .any(|sample| !(-8_388_608..=8_388_607).contains(sample))
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "input contains a sample outside signed 24-bit range",
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
    let (pcm_bits, offset) = if args.get(1).map(String::as_str) == Some("--pcm-bits") {
        let bits = args
            .get(2)
            .and_then(|value| value.parse::<u8>().ok())
            .filter(|value| matches!(value, 16 | 24))
            .unwrap_or_else(|| usage());
        (Some(bits), 3)
    } else {
        (None, 1)
    };
    if args.len() != offset + 5 {
        usage();
    }
    let frame_size = args[offset].parse::<usize>().unwrap_or_else(|_| usage());
    if !VALID_FRAME_SIZES.contains(&frame_size) {
        usage();
    }
    let bitrate = args[offset + 1].parse::<i32>().unwrap_or_else(|_| usage());
    let vbr = match args[offset + 2].as_str() {
        "cbr" => false,
        "vbr" => true,
        _ => usage(),
    };
    let input_path = Path::new(&args[offset + 3]);
    let output_path = Path::new(&args[offset + 4]);
    let input = match pcm_bits {
        Some(16) => PcmInput::I16(read_i16le(input_path)?),
        Some(24) => PcmInput::I24(read_i24_s32le(input_path)?),
        _ => PcmInput::F32(read_f32le(input_path)?),
    };
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

    for offset in (0..total_samples).step_by(frame_samples) {
        let copied = (total_samples - offset).min(frame_samples);
        let packet = match &input {
            PcmInput::F32(samples) => {
                let mut padded = vec![0.0f32; frame_samples];
                padded[..copied].copy_from_slice(&samples[offset..offset + copied]);
                encoder.encode_f32(&padded, frame_size)?
            }
            PcmInput::I16(samples) => {
                let mut padded = vec![0i16; frame_samples];
                padded[..copied].copy_from_slice(&samples[offset..offset + copied]);
                encoder.encode_i16(&padded, frame_size)?
            }
            PcmInput::I24(samples) => {
                let mut padded = vec![0i32; frame_samples];
                padded[..copied].copy_from_slice(&samples[offset..offset + copied]);
                encoder.encode_i24(&padded, frame_size)?
            }
        };
        packet_bytes += packet.len();
        packet_count += 1;
        packet_min = packet_min.min(packet.len());
        packet_max = packet_max.max(packet.len());
        match &input {
            PcmInput::F32(_) => output.extend(decoder.decode_f32(&packet, false)?),
            PcmInput::I16(_) => output.extend(
                decoder
                    .decode_i16(&packet, false)?
                    .into_iter()
                    .map(|sample| f32::from(sample) * (1.0 / 32_768.0)),
            ),
            PcmInput::I24(_) => output.extend(
                decoder
                    .decode_i24(&packet, false)?
                    .into_iter()
                    .map(|sample| sample as f32 * (1.0 / 8_388_608.0)),
            ),
        }
    }
    output.truncate(total_samples);
    write_f32le(output_path, &output)?;

    let mode = if vbr { "vbr" } else { "cbr" };
    println!(
        "{{\"codec\":\"soundkit-opus\",\"sample_rate\":{SAMPLE_RATE},\"channels\":{CHANNELS},\"pcm_bits\":\"{}\",\"frame_size\":{frame_size},\"bitrate\":{bitrate},\"mode\":\"{mode}\",\"packets\":{packet_count},\"packet_bytes\":{packet_bytes},\"packet_min\":{packet_min},\"packet_max\":{packet_max}}}",
        input.label()
    );
    Ok(())
}
