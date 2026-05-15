use libopus_rs::{Application, Decoder, Encoder};
use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"LORSCELT";
const FRAME_SIZE: usize = 960;

#[derive(Clone, Debug)]
struct WavPcm {
    sample_rate: i32,
    channels: usize,
    samples: Vec<i16>,
}

#[derive(Clone, Debug)]
struct PacketStream {
    sample_rate: i32,
    channels: usize,
    frame_size: usize,
    total_samples: usize,
    packets: Vec<Vec<u8>>,
}

fn read_u16(data: &[u8], offset: usize) -> io::Result<u16> {
    let bytes = data
        .get(offset..offset + 2)
        .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "short u16"))?;
    Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
}

fn read_u32(data: &[u8], offset: usize) -> io::Result<u32> {
    let bytes = data
        .get(offset..offset + 4)
        .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "short u32"))?;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn read_wav(path: &Path) -> io::Result<WavPcm> {
    let data = fs::read(path)?;
    if data.get(0..4) != Some(b"RIFF") || data.get(8..12) != Some(b"WAVE") {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not a RIFF/WAVE file",
        ));
    }

    let mut offset = 12usize;
    let mut fmt = None;
    let mut pcm_data = None;
    while offset + 8 <= data.len() {
        let id = &data[offset..offset + 4];
        let len = read_u32(&data, offset + 4)? as usize;
        let chunk_start = offset + 8;
        let chunk_end = chunk_start
            .checked_add(len)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "chunk overflow"))?;
        if chunk_end > data.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "short chunk"));
        }
        if id == b"fmt " {
            fmt = Some((chunk_start, len));
        } else if id == b"data" {
            pcm_data = Some((chunk_start, len));
        }
        offset = chunk_end + (len & 1);
    }

    let (fmt_start, fmt_len) =
        fmt.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing fmt chunk"))?;
    if fmt_len < 16 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "short fmt chunk",
        ));
    }
    let audio_format = read_u16(&data, fmt_start)?;
    let channels = read_u16(&data, fmt_start + 2)? as usize;
    let sample_rate = read_u32(&data, fmt_start + 4)? as i32;
    let bits_per_sample = read_u16(&data, fmt_start + 14)?;
    if audio_format != 1
        || !(1..=2).contains(&channels)
        || sample_rate != 48_000
        || bits_per_sample != 16
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "expected 48 kHz mono/stereo PCM16 WAV",
        ));
    }

    let (data_start, data_len) =
        pcm_data.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing data chunk"))?;
    if data_len % 2 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "odd PCM byte count",
        ));
    }
    let samples = data[data_start..data_start + data_len]
        .chunks_exact(2)
        .map(|bytes| i16::from_le_bytes([bytes[0], bytes[1]]))
        .collect::<Vec<_>>();

    Ok(WavPcm {
        sample_rate,
        channels,
        samples,
    })
}

fn write_wav(path: &Path, wav: &WavPcm) -> io::Result<()> {
    let data_bytes = wav
        .samples
        .len()
        .checked_mul(2)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "WAV too large"))?;
    let riff_size = 36usize
        .checked_add(data_bytes)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "WAV too large"))?;
    let byte_rate = wav.sample_rate as u32 * wav.channels as u32 * 2;
    let block_align = wav.channels as u16 * 2;

    let mut out = Vec::with_capacity(44 + data_bytes);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(riff_size as u32).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&(wav.channels as u16).to_le_bytes());
    out.extend_from_slice(&(wav.sample_rate as u32).to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data_bytes as u32).to_le_bytes());
    for sample in &wav.samples {
        out.extend_from_slice(&sample.to_le_bytes());
    }
    fs::write(path, out)
}

fn write_stream(path: &Path, stream: &PacketStream) -> io::Result<()> {
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(stream.sample_rate as u32).to_le_bytes());
    out.extend_from_slice(&(stream.channels as u16).to_le_bytes());
    out.extend_from_slice(&(stream.frame_size as u16).to_le_bytes());
    out.extend_from_slice(&(stream.total_samples as u32).to_le_bytes());
    out.extend_from_slice(&(stream.packets.len() as u32).to_le_bytes());
    for packet in &stream.packets {
        if packet.len() > u16::MAX as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "packet too large",
            ));
        }
        out.extend_from_slice(&(packet.len() as u16).to_le_bytes());
        out.extend_from_slice(packet);
    }
    fs::write(path, out)
}

fn read_stream(path: &Path) -> io::Result<PacketStream> {
    let data = fs::read(path)?;
    if data.get(0..8) != Some(MAGIC) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "bad packet stream magic",
        ));
    }
    let sample_rate = read_u32(&data, 8)? as i32;
    let channels = read_u16(&data, 12)? as usize;
    let frame_size = read_u16(&data, 14)? as usize;
    let total_samples = read_u32(&data, 16)? as usize;
    let packet_count = read_u32(&data, 20)? as usize;
    let mut offset = 24usize;
    let mut packets = Vec::with_capacity(packet_count);
    for _ in 0..packet_count {
        let len = read_u16(&data, offset)? as usize;
        offset += 2;
        let packet = data
            .get(offset..offset + len)
            .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "short packet"))?
            .to_vec();
        offset += len;
        packets.push(packet);
    }
    Ok(PacketStream {
        sample_rate,
        channels,
        frame_size,
        total_samples,
        packets,
    })
}

fn encode_wav(input: &Path, output: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let wav = read_wav(input)?;
    let total_samples = wav.samples.len() / wav.channels;
    let mut encoder = Encoder::new(wav.sample_rate, wav.channels, Application::Audio)?;
    let mut packets = Vec::new();
    for frame in wav.samples.chunks(FRAME_SIZE * wav.channels) {
        let mut padded = vec![0i16; FRAME_SIZE * wav.channels];
        padded[..frame.len()].copy_from_slice(frame);
        packets.push(encoder.encode_i16(&padded, FRAME_SIZE)?);
    }
    write_stream(
        output,
        &PacketStream {
            sample_rate: wav.sample_rate,
            channels: wav.channels,
            frame_size: FRAME_SIZE,
            total_samples,
            packets,
        },
    )?;
    Ok(())
}

fn decode_stream(input: &Path, output: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let stream = read_stream(input)?;
    let mut decoder = Decoder::new(stream.sample_rate, stream.channels)?;
    let mut samples = Vec::new();
    for packet in &stream.packets {
        samples.extend(decoder.decode_i16(packet, false)?);
    }
    samples.truncate(stream.total_samples * stream.channels);
    write_wav(
        output,
        &WavPcm {
            sample_rate: stream.sample_rate,
            channels: stream.channels,
            samples,
        },
    )?;
    Ok(())
}

fn usage() -> ! {
    let _ = writeln!(
        io::stderr(),
        "usage:\n  wav_celt encode <input.wav> <output.lors>\n  wav_celt decode <input.lors> <output.wav>\n  wav_celt roundtrip <input.wav> <output.lors> <output.wav>"
    );
    std::process::exit(2);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = env::args().collect::<Vec<_>>();
    match args.get(1).map(String::as_str) {
        Some("encode") if args.len() == 4 => encode_wav(Path::new(&args[2]), Path::new(&args[3])),
        Some("decode") if args.len() == 4 => {
            decode_stream(Path::new(&args[2]), Path::new(&args[3]))
        }
        Some("roundtrip") if args.len() == 5 => {
            encode_wav(Path::new(&args[2]), Path::new(&args[3]))?;
            decode_stream(Path::new(&args[3]), Path::new(&args[4]))
        }
        _ => usage(),
    }
}
