use soundkit_flac::{FlacFrameConfig, FlacFrameEncoder, FlacProfile};
use std::env;
use std::fs;
use std::io;
use std::path::PathBuf;

const SAMPLE_RATE: u32 = 48_000;
const CHANNELS: u16 = 2;
const BITS_PER_SAMPLE: u8 = 24;
const FRAME_LENGTH: u32 = 240;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = env::args_os().skip(1);
    let input = PathBuf::from(arguments.next().ok_or("missing S24LE input path")?);
    let output = PathBuf::from(arguments.next().ok_or("missing FLAC output path")?);
    if arguments.next().is_some() {
        return Err("usage: flac_frame_file INPUT.s24le OUTPUT.flac".into());
    }

    let config = FlacFrameConfig::new(
        SAMPLE_RATE,
        CHANNELS,
        BITS_PER_SAMPLE,
        FRAME_LENGTH,
        FlacProfile::Realtime,
    )?;
    let frame_bytes = config.raw_pcm_bytes()?;
    let pcm = fs::read(&input)?;
    if pcm.is_empty() || pcm.len() % frame_bytes != 0 {
        return Err(format!(
            "{} has {} bytes; expected a nonzero multiple of {frame_bytes}",
            input.display(),
            pcm.len()
        )
        .into());
    }

    let mut encoder = FlacFrameEncoder::new(config)?;
    let mut frames = Vec::with_capacity(pcm.len());
    let mut minimum_frame_bytes = usize::MAX;
    let mut maximum_frame_bytes = 0usize;
    let mut expanded_frames = 0usize;
    for block in pcm.chunks_exact(frame_bytes) {
        let encoded = encoder.encode_s24le(block)?;
        minimum_frame_bytes = minimum_frame_bytes.min(encoded.payload.len());
        maximum_frame_bytes = maximum_frame_bytes.max(encoded.payload.len());
        expanded_frames += usize::from(encoded.payload.len() > frame_bytes + 32);
        frames.extend_from_slice(&encoded.payload);
    }

    let frame_count = pcm.len() / frame_bytes;
    let total_samples = u64::try_from(frame_count)?
        .checked_mul(u64::from(FRAME_LENGTH))
        .ok_or("FLAC sample count overflow")?;
    let streaminfo = streaminfo(
        total_samples,
        u32::try_from(minimum_frame_bytes)?,
        u32::try_from(maximum_frame_bytes)?,
    )?;
    let mut file = Vec::with_capacity(4 + 4 + streaminfo.len() + frames.len());
    file.extend_from_slice(b"fLaC");
    file.push(0x80);
    file.extend_from_slice(&[0, 0, 34]);
    file.extend_from_slice(&streaminfo);
    file.extend_from_slice(&frames);
    fs::write(&output, file)?;

    println!(
        "frames={frame_count} min_bytes={minimum_frame_bytes} \
         max_bytes={maximum_frame_bytes} expanded_frames={expanded_frames}"
    );
    Ok(())
}

fn streaminfo(
    total_samples: u64,
    minimum_frame_bytes: u32,
    maximum_frame_bytes: u32,
) -> Result<[u8; 34], Box<dyn std::error::Error>> {
    if total_samples >= (1_u64 << 36) {
        return Err("FLAC sample count exceeds 36 bits".into());
    }
    let mut output = [0_u8; 34];
    output[0..2].copy_from_slice(&(FRAME_LENGTH as u16).to_be_bytes());
    output[2..4].copy_from_slice(&(FRAME_LENGTH as u16).to_be_bytes());
    write_u24(&mut output[4..7], minimum_frame_bytes)?;
    write_u24(&mut output[7..10], maximum_frame_bytes)?;
    let packed = (u64::from(SAMPLE_RATE) << 44)
        | (u64::from(CHANNELS - 1) << 41)
        | (u64::from(BITS_PER_SAMPLE - 1) << 36)
        | total_samples;
    output[10..18].copy_from_slice(&packed.to_be_bytes());
    Ok(output)
}

fn write_u24(output: &mut [u8], value: u32) -> io::Result<()> {
    if output.len() != 3 || value >= (1 << 24) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "value does not fit in 24 bits",
        ));
    }
    let bytes = value.to_be_bytes();
    output.copy_from_slice(&bytes[1..]);
    Ok(())
}
