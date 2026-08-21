use frame_header::{EncodingFlag, Endianness, FrameHeaderV2};
use hound::{SampleFormat, WavReader};
use soundkit::audio_packet::Encoder;
use soundkit_flac::{FlacFrameConfig, FlacFrameEncoder, FlacProfile};
use soundkit_opus::OpusEncoder;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const SAMPLE_RATE: u32 = 48_000;
const CHANNELS: u8 = 2;
const FRAME_COUNT: usize = 960;
const OPUS_BITRATE: u32 = 192_000;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args_os().skip(1);
    let wav_path = PathBuf::from(
        args.next()
            .ok_or("usage: generate_fixture <wav> <output-dir>")?,
    );
    let output_dir = PathBuf::from(
        args.next()
            .ok_or("usage: generate_fixture <wav> <output-dir>")?,
    );
    if args.next().is_some() {
        return Err("usage: generate_fixture <wav> <output-dir>".into());
    }

    let mut reader = WavReader::open(&wav_path)?;
    let spec = reader.spec();
    if spec.sample_rate != SAMPLE_RATE
        || spec.channels != u16::from(CHANNELS)
        || spec.bits_per_sample != 16
        || spec.sample_format != SampleFormat::Int
    {
        return Err(format!("fixture WAV must be 48 kHz stereo S16 PCM, got {spec:?}").into());
    }
    let samples = reader.samples::<i16>().collect::<Result<Vec<_>, _>>()?;
    let samples_per_packet = FRAME_COUNT * usize::from(CHANNELS);
    if samples.is_empty() || samples.len() % samples_per_packet != 0 {
        return Err("fixture WAV must contain complete 20 ms packets".into());
    }

    let mut opus_encoder = OpusEncoder::new(
        SAMPLE_RATE,
        16,
        u32::from(CHANNELS),
        FRAME_COUNT as u32,
        OPUS_BITRATE,
    );
    opus_encoder.init()?;
    let flac_config = FlacFrameConfig::new(
        SAMPLE_RATE,
        u16::from(CHANNELS),
        16,
        FRAME_COUNT as u32,
        FlacProfile::Realtime,
    )?;
    let mut flac_encoder = FlacFrameEncoder::new(flac_config)?;
    let mut opus_stream = Vec::new();
    let mut flac_stream = Vec::new();
    let mut opus_scratch = vec![0_u8; 16 * 1024];

    for (index, packet_samples) in samples.chunks_exact(samples_per_packet).enumerate() {
        let pts = (index * FRAME_COUNT) as u64;
        let opus_len = opus_encoder.encode_i16(packet_samples, &mut opus_scratch)?;
        append_v2_packet(
            &mut opus_stream,
            EncodingFlag::Opus,
            &opus_scratch[..opus_len],
            index as u64,
            pts,
        )?;
        let flac = flac_encoder.encode_i16(packet_samples)?;
        append_v2_packet(
            &mut flac_stream,
            EncodingFlag::FLAC,
            &flac.payload,
            index as u64,
            pts,
        )?;
    }

    fs::create_dir_all(&output_dir)?;
    let opus_path = output_dir.join("westside-4s-opus.skv2");
    let flac_path = output_dir.join("westside-4s-flac.skv2");
    fs::write(&opus_path, &opus_stream)?;
    fs::write(&flac_path, &flac_stream)?;
    println!(
        "generated {} packets: {} bytes Opus, {} bytes FLAC in {}",
        samples.len() / samples_per_packet,
        opus_stream.len(),
        flac_stream.len(),
        output_dir.display()
    );
    Ok(())
}

fn append_v2_packet(
    output: &mut Vec<u8>,
    encoding: EncodingFlag,
    payload: &[u8],
    id: u64,
    pts: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    let header = FrameHeaderV2::new(
        encoding,
        u32::try_from(payload.len())?,
        FRAME_COUNT as u32,
        SAMPLE_RATE,
        CHANNELS,
        16,
        Endianness::LittleEndian,
        Some(id),
        Some(pts),
        None,
    )?
    .with_packet_crc32(payload)?;
    header.encode(output)?;
    output.extend_from_slice(payload);
    Ok(())
}

#[allow(dead_code)]
fn _is_file(path: &Path) -> bool {
    path.is_file()
}
