use soundkit_video::{VideoCodec, VideoDecoder, VideoFrame};
use std::{env, fs, process};

fn fail(message: impl AsRef<str>) -> ! {
    eprintln!("{}", message.as_ref());
    process::exit(2);
}

fn decode_ivf(decoder: &mut VideoDecoder, data: &[u8]) -> Result<Vec<VideoFrame>, String> {
    if data.len() < 32 || &data[..4] != b"DKIF" {
        return Err("invalid IVF stream header".to_string());
    }
    let mut cursor = usize::from(u16::from_le_bytes([data[6], data[7]]));
    let mut output = Vec::new();
    while cursor < data.len() {
        if data.len() - cursor < 12 {
            return Err("truncated IVF frame header".to_string());
        }
        let size = u32::from_le_bytes(data[cursor..cursor + 4].try_into().unwrap()) as usize;
        let timestamp = u64::from_le_bytes(data[cursor + 4..cursor + 12].try_into().unwrap());
        cursor += 12;
        let end = cursor
            .checked_add(size)
            .filter(|end| *end <= data.len())
            .ok_or_else(|| "truncated IVF frame payload".to_string())?;
        output.extend(decoder.decode(&data[cursor..end], i64::try_from(timestamp).ok(), None)?);
        cursor = end;
    }
    output.extend(decoder.flush()?);
    Ok(output)
}

fn decode_prores(decoder: &mut VideoDecoder, data: &[u8]) -> Result<Vec<VideoFrame>, String> {
    let mut cursor = 0usize;
    let mut output = Vec::new();
    while cursor < data.len() {
        if data.len() - cursor < 8 {
            return Err("truncated ProRes frame prefix".to_string());
        }
        let size = u32::from_be_bytes(data[cursor..cursor + 4].try_into().unwrap()) as usize;
        let end = cursor
            .checked_add(size)
            .filter(|end| size >= 8 && *end <= data.len())
            .ok_or_else(|| "invalid ProRes frame size".to_string())?;
        output.extend(decoder.decode(&data[cursor..end], None, None)?);
        cursor = end;
    }
    Ok(output)
}

fn checksum(frames: &[VideoFrame]) -> u64 {
    frames
        .iter()
        .flat_map(|frame| frame.planes.iter())
        .flat_map(|plane| plane.data.iter().copied())
        .fold(0xcbf29ce484222325_u64, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
        })
}

fn main() {
    let mut args = env::args().skip(1);
    let codec_name = args.next().unwrap_or_else(|| fail("missing codec"));
    let path = args
        .next()
        .unwrap_or_else(|| fail("missing elementary stream path"));
    if args.next().is_some() {
        fail("usage: conformance <h264|hevc|vp9|av1|prores|dnxhr> <stream>");
    }
    let codec = VideoCodec::parse(&codec_name).unwrap_or_else(|| fail("unsupported codec"));
    let data =
        fs::read(&path).unwrap_or_else(|error| fail(format!("could not read {path}: {error}")));
    let mut decoder = VideoDecoder::new(codec).unwrap_or_else(|error| fail(error));
    let result = match codec {
        VideoCodec::H264 | VideoCodec::Hevc => decoder.decode_stream(&data),
        VideoCodec::Vp9 | VideoCodec::Av1 => decode_ivf(&mut decoder, &data),
        VideoCodec::ProRes => decode_prores(&mut decoder, &data),
        VideoCodec::DnxHd => decoder.decode(&data, None, None),
    };
    let frames = result.unwrap_or_else(|error| fail(error));
    let first = frames
        .first()
        .unwrap_or_else(|| fail("decoder emitted no frames"));
    println!(
        "codec={} frames={} width={} height={} bitDepth={} colorModel={} chroma={} checksum={:016x}",
        codec.as_str(),
        frames.len(),
        first.width,
        first.height,
        first.bit_depth,
        first.color_model.as_str(),
        first.chroma_sampling.as_str(),
        checksum(&frames),
    );
}
