use soundkit::audio_packet::Decoder as SoundkitDecoderTrait;
use soundkit_opus::{
    Application, Decoder, Encoder, OpusDecoder, CELT_FRAME_SIZES_48K, CELT_MAX_FRAME_BYTES,
};
use std::env;
use std::fs;
use std::hint::black_box;
use std::time::Instant;

const SAMPLE_RATE: usize = 48_000;
const CHANNELS: usize = 2;
const BITRATES: [i32; 9] = [
    48_000, 96_000, 128_000, 160_000, 192_000, 256_000, 320_000, 384_000, 512_000,
];
const TARGET_BITRATES: [i32; 3] = [192_000, 256_000, 320_000];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BenchMode {
    Cbr,
    Vbr,
}

#[derive(Clone, Copy)]
enum BenchFixture {
    Mixed,
    Tone,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PcmFormat {
    F32,
    I16,
    I24,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DecodeApi {
    Core,
    Adapter,
}

#[derive(Clone, Copy)]
enum PcmInput<'a> {
    F32(&'a [f32]),
    I16(&'a [i16]),
    I24(&'a [i32]),
}

impl PcmInput<'_> {
    const fn len(self) -> usize {
        match self {
            Self::F32(pcm) => pcm.len(),
            Self::I16(pcm) => pcm.len(),
            Self::I24(pcm) => pcm.len(),
        }
    }

    const fn format(self) -> PcmFormat {
        match self {
            Self::F32(_) => PcmFormat::F32,
            Self::I16(_) => PcmFormat::I16,
            Self::I24(_) => PcmFormat::I24,
        }
    }
}

impl BenchMode {
    const fn label(self) -> &'static str {
        match self {
            Self::Cbr => "cbr",
            Self::Vbr => "vbr",
        }
    }
}

#[derive(Clone)]
struct Options {
    repeats: usize,
    seconds: usize,
    mode: Option<BenchMode>,
    dump_packets: Option<usize>,
    frame_size: Option<usize>,
    bitrate: Option<i32>,
    fixture: BenchFixture,
    input_s32le: Option<String>,
    pcm_bits: Option<u8>,
    skip_quality: bool,
    quality_lag: Option<isize>,
    application: Application,
    direct_cubic: bool,
    decode_api: DecodeApi,
}

struct EncodeResult {
    packets: Vec<Vec<u8>>,
}

fn usage() -> ! {
    eprintln!(
        "usage: raw_celt_bench [--repeats n] [--seconds n] [--mode cbr|vbr|both] [--application audio|restricted-lowdelay] [--decode-api core|adapter] [--direct-cubic] [--frame-size n] [--bitrate n] [--fixture mixed|tone] [--input-s32le path] [--pcm-bits 16|24] [--quality-lag frames] [--skip-quality] [--dump-packets n]"
    );
    std::process::exit(2);
}

fn parse_options() -> Options {
    let mut options = Options {
        repeats: 21,
        seconds: 4,
        mode: None,
        dump_packets: None,
        frame_size: None,
        bitrate: None,
        fixture: BenchFixture::Mixed,
        input_s32le: None,
        pcm_bits: None,
        skip_quality: false,
        quality_lag: None,
        application: Application::Audio,
        direct_cubic: false,
        decode_api: DecodeApi::Core,
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
            "--application" => {
                i += 1;
                options.application = match args.get(i).map(String::as_str) {
                    Some("audio") => Application::Audio,
                    Some("restricted-lowdelay") => Application::RestrictedLowDelay,
                    _ => usage(),
                };
            }
            "--direct-cubic" => options.direct_cubic = true,
            "--decode-api" => {
                i += 1;
                options.decode_api = match args.get(i).map(String::as_str) {
                    Some("core") => DecodeApi::Core,
                    Some("adapter") => DecodeApi::Adapter,
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
            "--frame-size" => {
                i += 1;
                let frame_size = args
                    .get(i)
                    .and_then(|value| value.parse().ok())
                    .unwrap_or_else(|| usage());
                if !CELT_FRAME_SIZES_48K.contains(&frame_size) {
                    usage();
                }
                options.frame_size = Some(frame_size);
            }
            "--bitrate" => {
                i += 1;
                let bitrate = args
                    .get(i)
                    .and_then(|value| value.parse().ok())
                    .unwrap_or_else(|| usage());
                if !BITRATES.contains(&bitrate) {
                    usage();
                }
                options.bitrate = Some(bitrate);
            }
            "--fixture" => {
                i += 1;
                options.fixture = match args.get(i).map(String::as_str) {
                    Some("mixed") => BenchFixture::Mixed,
                    Some("tone") => BenchFixture::Tone,
                    _ => usage(),
                };
            }
            "--input-s32le" => {
                i += 1;
                options.input_s32le = Some(args.get(i).cloned().unwrap_or_else(|| usage()));
            }
            "--pcm-bits" => {
                i += 1;
                options.pcm_bits = match args.get(i).map(String::as_str) {
                    Some("16") => Some(16),
                    Some("24") => Some(24),
                    _ => usage(),
                };
            }
            "--quality-lag" => {
                i += 1;
                options.quality_lag = Some(
                    args.get(i)
                        .and_then(|value| value.parse().ok())
                        .unwrap_or_else(|| usage()),
                );
            }
            "--skip-quality" => options.skip_quality = true,
            _ => usage(),
        }
        i += 1;
    }
    if options.repeats == 0 || options.seconds == 0 || options.dump_packets == Some(0) {
        usage();
    }
    if options.decode_api == DecodeApi::Adapter && options.direct_cubic {
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

fn bitrate_enabled(options: &Options, bitrate: i32) -> bool {
    options
        .bitrate
        .map_or(TARGET_BITRATES.contains(&bitrate), |selected| {
            selected == bitrate
        })
}

fn generate_mixed_fixture(seconds: usize) -> Vec<f32> {
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

fn generate_tone_fixture(seconds: usize) -> Vec<f32> {
    let frames = SAMPLE_RATE * seconds;
    let mut pcm = Vec::with_capacity(frames * CHANNELS);
    for i in 0..frames {
        let phase = 6.283_185_307_179_586 * i as f64 / 48.0;
        pcm.push((0.25 * phase.sin()) as f32);
        pcm.push((0.22 * (phase + 0.2).sin()) as f32);
    }
    pcm
}

fn generate_fixture(seconds: usize, fixture: BenchFixture) -> Vec<f32> {
    match fixture {
        BenchFixture::Mixed => generate_mixed_fixture(seconds),
        BenchFixture::Tone => generate_tone_fixture(seconds),
    }
}

fn load_i24_s32le(
    path: &str,
    seconds: usize,
) -> Result<(Vec<f32>, Vec<i32>), Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    let sample_count = SAMPLE_RATE
        .checked_mul(seconds)
        .and_then(|frames| frames.checked_mul(CHANNELS))
        .ok_or("requested input duration is too large")?;
    let byte_count = sample_count
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or("requested input duration is too large")?;
    if bytes.len() < byte_count {
        return Err(format!(
            "{path} contains {} bytes, but {seconds} seconds of stereo 48 kHz S32LE needs {byte_count}",
            bytes.len()
        )
        .into());
    }

    let mut pcm_i24 = Vec::with_capacity(sample_count);
    let mut pcm_f32 = Vec::with_capacity(sample_count);
    for bytes in bytes[..byte_count].chunks_exact(4) {
        let sample = i32::from_le_bytes(bytes.try_into().expect("four-byte sample"));
        if !(-8_388_608..=8_388_607).contains(&sample) {
            return Err(format!("{path} contains a sample outside signed 24-bit range").into());
        }
        pcm_i24.push(sample);
        pcm_f32.push(sample as f32 / 8_388_608.0);
    }
    Ok((pcm_f32, pcm_i24))
}

fn quantize_fixture_i16(pcm: &mut [f32]) -> Vec<i16> {
    let quantized = pcm
        .iter()
        .map(|&sample| {
            (sample * 32_768.0)
                .round()
                .clamp(i16::MIN as f32, i16::MAX as f32) as i16
        })
        .collect::<Vec<_>>();
    for (reference, &sample) in pcm.iter_mut().zip(&quantized) {
        *reference = sample as f32 / 32_768.0;
    }
    quantized
}

fn quantize_fixture_i24(pcm: &mut [f32]) -> Vec<i32> {
    let quantized = pcm
        .iter()
        .map(|&sample| {
            (sample * 8_388_608.0)
                .round()
                .clamp(-8_388_608.0, 8_388_607.0) as i32
        })
        .collect::<Vec<_>>();
    for (reference, &sample) in pcm.iter_mut().zip(&quantized) {
        *reference = sample as f32 / 8_388_608.0;
    }
    quantized
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

#[derive(Clone, Copy)]
struct DecodeQuality {
    lag_frames: isize,
    snr_db: f64,
}

fn decode_packets(
    packets: &[Vec<u8>],
    frame_size: usize,
    format: PcmFormat,
    direct_cubic: bool,
    decode_api: DecodeApi,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    match decode_api {
        DecodeApi::Core => decode_packets_core(packets, frame_size, format, direct_cubic),
        DecodeApi::Adapter => decode_packets_adapter(packets, frame_size, format),
    }
}

fn decode_packets_core(
    packets: &[Vec<u8>],
    frame_size: usize,
    format: PcmFormat,
    direct_cubic: bool,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut decoder = Decoder::new(SAMPLE_RATE as i32, CHANNELS)?;
    decoder.set_experimental_direct_cubic(direct_cubic);
    let mut decoded = Vec::with_capacity(packets.len() * frame_size * CHANNELS);
    match format {
        PcmFormat::I16 => {
            let mut frame = Vec::new();
            for packet in packets {
                let decoded_frames =
                    decoder.decode_i16_into(black_box(packet), false, &mut frame)?;
                if decoded_frames != frame_size || frame.len() != frame_size * CHANNELS {
                    return Err("unexpected decoded frame size".into());
                }
                decoded.extend(frame.iter().map(|&sample| sample as f32 / 32_768.0));
            }
        }
        PcmFormat::I24 => {
            let mut frame = Vec::new();
            for packet in packets {
                let decoded_frames =
                    decoder.decode_i24_into(black_box(packet), false, &mut frame)?;
                if decoded_frames != frame_size || frame.len() != frame_size * CHANNELS {
                    return Err("unexpected decoded frame size".into());
                }
                decoded.extend(frame.iter().map(|&sample| sample as f32 / 8_388_608.0));
            }
        }
        PcmFormat::F32 => {
            let mut frame = Vec::new();
            for packet in packets {
                let decoded_frames =
                    decoder.decode_f32_into(black_box(packet), false, &mut frame)?;
                if decoded_frames != frame_size || frame.len() != frame_size * CHANNELS {
                    return Err("unexpected decoded frame size".into());
                }
                decoded.extend_from_slice(&frame);
            }
        }
    }
    Ok(decoded)
}

fn decode_packets_adapter(
    packets: &[Vec<u8>],
    frame_size: usize,
    format: PcmFormat,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut decoder = OpusDecoder::new_celt_only(SAMPLE_RATE, CHANNELS)?;
    let mut decoded = Vec::with_capacity(packets.len() * frame_size * CHANNELS);
    match format {
        PcmFormat::I16 => {
            let mut frame = vec![0_i16; frame_size * CHANNELS];
            for packet in packets {
                let decoded_frames = decoder.decode_i16(black_box(packet), &mut frame, false)?;
                if decoded_frames != frame_size {
                    return Err("unexpected decoded frame size".into());
                }
                decoded.extend(frame.iter().map(|&sample| sample as f32 / 32_768.0));
            }
        }
        PcmFormat::I24 => {
            let mut frame = vec![0_i32; frame_size * CHANNELS];
            for packet in packets {
                let decoded_frames = decoder.decode_i32(black_box(packet), &mut frame, false)?;
                if decoded_frames != frame_size {
                    return Err("unexpected decoded frame size".into());
                }
                decoded.extend(frame.iter().map(|&sample| sample as f32 / 8_388_608.0));
            }
        }
        PcmFormat::F32 => {
            let mut frame = vec![0.0_f32; frame_size * CHANNELS];
            for packet in packets {
                let decoded_frames = decoder.decode_f32(black_box(packet), &mut frame, false)?;
                if decoded_frames != frame_size {
                    return Err("unexpected decoded frame size".into());
                }
                decoded.extend_from_slice(&frame);
            }
        }
    }
    Ok(decoded)
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
        let candidate = quality_at_lag(reference, decoded, lag, compare_frames);
        if candidate.snr_db > best.snr_db {
            best = candidate;
        }
    }

    best
}

fn quality_at_lag(
    reference: &[f32],
    decoded: &[f32],
    lag: isize,
    requested_frames: usize,
) -> DecodeQuality {
    let total_frames = (reference.len().min(decoded.len())) / CHANNELS;
    let (reference_start, decoded_start) = if lag >= 0 {
        (lag as usize, 0)
    } else {
        (0, (-lag) as usize)
    };
    let compare_frames = requested_frames.min(
        total_frames
            .saturating_sub(reference_start)
            .min(total_frames.saturating_sub(decoded_start)),
    );
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
    DecodeQuality {
        lag_frames: lag,
        snr_db,
    }
}

fn encode_with_encoder(
    encoder: &mut Encoder,
    pcm: PcmInput<'_>,
    frame_size: usize,
) -> Result<EncodeResult, Box<dyn std::error::Error>> {
    let frames = pcm.len() / (frame_size * CHANNELS);
    let mut packets = Vec::with_capacity(frames);
    packets.resize_with(frames, Vec::new);
    encode_with_encoder_into(encoder, pcm, frame_size, &mut packets)?;
    Ok(EncodeResult { packets })
}

fn encode_with_encoder_into(
    encoder: &mut Encoder,
    pcm: PcmInput<'_>,
    frame_size: usize,
    packets: &mut Vec<Vec<u8>>,
) -> Result<(usize, usize, usize, u64), Box<dyn std::error::Error>> {
    let frames = pcm.len() / (frame_size * CHANNELS);
    packets.resize_with(frames, Vec::new);
    packets.truncate(frames);
    let mut bytes = 0usize;
    let mut checksum = 0u64;
    let mut min_packet = usize::MAX;
    let mut max_packet = 0usize;
    for frame in 0..frames {
        let start = frame * frame_size * CHANNELS;
        let end = start + frame_size * CHANNELS;
        let packet = &mut packets[frame];
        match pcm {
            PcmInput::F32(pcm) => {
                encoder.encode_f32_into(black_box(&pcm[start..end]), frame_size, packet)?;
            }
            PcmInput::I16(pcm) => {
                encoder.encode_i16_into(black_box(&pcm[start..end]), frame_size, packet)?;
            }
            PcmInput::I24(pcm) => {
                encoder.encode_i24_into(black_box(&pcm[start..end]), frame_size, packet)?;
            }
        }
        bytes += packet.len();
        min_packet = min_packet.min(packet.len());
        max_packet = max_packet.max(packet.len());
        checksum = checksum.wrapping_add(packet_checksum(packet));
    }
    black_box(checksum);
    Ok((bytes, min_packet, max_packet, checksum))
}

fn encode_packets(
    pcm: PcmInput<'_>,
    frame_size: usize,
    bitrate: i32,
    mode: BenchMode,
    application: Application,
    direct_cubic: bool,
) -> Result<EncodeResult, Box<dyn std::error::Error>> {
    let mut encoder = Encoder::new(SAMPLE_RATE as i32, CHANNELS, application)?;
    encoder.set_experimental_direct_cubic(direct_cubic);
    encoder.set_bitrate(bitrate)?;
    encoder.set_vbr(mode == BenchMode::Vbr)?;
    encode_with_encoder(&mut encoder, pcm, frame_size)
}

fn time_encode(
    pcm: PcmInput<'_>,
    frame_size: usize,
    bitrate: i32,
    mode: BenchMode,
    application: Application,
    direct_cubic: bool,
    repeats: usize,
) -> Result<(f64, usize, usize, usize, u64), Box<dyn std::error::Error>> {
    let mut times = Vec::with_capacity(repeats);
    let mut last_bytes = 0usize;
    let mut last_min_packet = 0usize;
    let mut last_max_packet = 0usize;
    let mut last_checksum = 0u64;
    let frames = pcm.len() / (frame_size * CHANNELS);
    let mut packets = (0..frames)
        .map(|_| Vec::with_capacity(CELT_MAX_FRAME_BYTES + 1))
        .collect::<Vec<_>>();
    for _ in 0..repeats {
        let mut encoder = Encoder::new(SAMPLE_RATE as i32, CHANNELS, application)?;
        encoder.set_experimental_direct_cubic(direct_cubic);
        encoder.set_bitrate(bitrate)?;
        encoder.set_vbr(mode == BenchMode::Vbr)?;
        let start = Instant::now();
        let (bytes, min_packet, max_packet, checksum) =
            encode_with_encoder_into(&mut encoder, pcm, frame_size, &mut packets)?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        black_box(&packets);
        times.push(elapsed);
        last_bytes = bytes;
        last_min_packet = min_packet;
        last_max_packet = max_packet;
        last_checksum = checksum;
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
    format: PcmFormat,
    direct_cubic: bool,
    decode_api: DecodeApi,
) -> Result<(f64, u64), Box<dyn std::error::Error>> {
    match decode_api {
        DecodeApi::Core => time_decode_core(packets, frame_size, repeats, format, direct_cubic),
        DecodeApi::Adapter => time_decode_adapter(packets, frame_size, repeats, format),
    }
}

fn time_decode_core(
    packets: &[Vec<u8>],
    frame_size: usize,
    repeats: usize,
    format: PcmFormat,
    direct_cubic: bool,
) -> Result<(f64, u64), Box<dyn std::error::Error>> {
    let mut times = Vec::with_capacity(repeats);
    let mut last_checksum = 0u64;
    for _ in 0..repeats {
        let mut decoder = Decoder::new(SAMPLE_RATE as i32, CHANNELS)?;
        decoder.set_experimental_direct_cubic(direct_cubic);
        let mut decoded_i16 = Vec::with_capacity(frame_size * CHANNELS);
        let mut decoded_i24 = Vec::with_capacity(frame_size * CHANNELS);
        let mut decoded_f32 = Vec::with_capacity(frame_size * CHANNELS);
        let start = Instant::now();
        last_checksum = match format {
            PcmFormat::I16 => {
                let mut checksum = 0i64;
                for packet in packets {
                    let decoded_frames =
                        decoder.decode_i16_into(black_box(packet), false, &mut decoded_i16)?;
                    if decoded_frames != frame_size || decoded_i16.len() != frame_size * CHANNELS {
                        return Err("unexpected decoded frame size".into());
                    }
                    let first = decoded_i16.first().copied().unwrap_or(0) as i64;
                    let middle =
                        decoded_i16.get(decoded_i16.len() / 2).copied().unwrap_or(0) as i64;
                    let last = decoded_i16.last().copied().unwrap_or(0) as i64;
                    checksum = checksum.wrapping_add(first + middle + last);
                }
                black_box(checksum);
                checksum as u64
            }
            PcmFormat::I24 => {
                let mut checksum = 0i64;
                for packet in packets {
                    let decoded_frames =
                        decoder.decode_i24_into(black_box(packet), false, &mut decoded_i24)?;
                    if decoded_frames != frame_size || decoded_i24.len() != frame_size * CHANNELS {
                        return Err("unexpected decoded frame size".into());
                    }
                    let first = decoded_i24.first().copied().unwrap_or(0) as i64;
                    let middle =
                        decoded_i24.get(decoded_i24.len() / 2).copied().unwrap_or(0) as i64;
                    let last = decoded_i24.last().copied().unwrap_or(0) as i64;
                    checksum = checksum.wrapping_add(first + middle + last);
                }
                black_box(checksum);
                checksum as u64
            }
            PcmFormat::F32 => {
                let mut checksum = 0.0f32;
                for packet in packets {
                    let decoded_frames =
                        decoder.decode_f32_into(black_box(packet), false, &mut decoded_f32)?;
                    if decoded_frames != frame_size || decoded_f32.len() != frame_size * CHANNELS {
                        return Err("unexpected decoded frame size".into());
                    }
                    checksum += decoded_checksum(&decoded_f32);
                }
                black_box(checksum);
                u64::from(checksum.to_bits())
            }
        };
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    Ok((median(&mut times), last_checksum))
}

fn time_decode_adapter(
    packets: &[Vec<u8>],
    frame_size: usize,
    repeats: usize,
    format: PcmFormat,
) -> Result<(f64, u64), Box<dyn std::error::Error>> {
    let mut times = Vec::with_capacity(repeats);
    let mut last_checksum = 0_u64;
    for _ in 0..repeats {
        let mut decoder = OpusDecoder::new_celt_only(SAMPLE_RATE, CHANNELS)?;
        let mut decoded_i16 = vec![0_i16; frame_size * CHANNELS];
        let mut decoded_i24 = vec![0_i32; frame_size * CHANNELS];
        let mut decoded_f32 = vec![0.0_f32; frame_size * CHANNELS];
        let start = Instant::now();
        last_checksum = match format {
            PcmFormat::I16 => {
                let mut checksum = 0_i64;
                for packet in packets {
                    let decoded_frames =
                        decoder.decode_i16(black_box(packet), &mut decoded_i16, false)?;
                    if decoded_frames != frame_size {
                        return Err("unexpected decoded frame size".into());
                    }
                    let first = decoded_i16.first().copied().unwrap_or(0) as i64;
                    let middle =
                        decoded_i16.get(decoded_i16.len() / 2).copied().unwrap_or(0) as i64;
                    let last = decoded_i16.last().copied().unwrap_or(0) as i64;
                    checksum = checksum.wrapping_add(first + middle + last);
                }
                black_box(checksum);
                checksum as u64
            }
            PcmFormat::I24 => {
                let mut checksum = 0_i64;
                for packet in packets {
                    let decoded_frames =
                        decoder.decode_i32(black_box(packet), &mut decoded_i24, false)?;
                    if decoded_frames != frame_size {
                        return Err("unexpected decoded frame size".into());
                    }
                    let first = decoded_i24.first().copied().unwrap_or(0) as i64;
                    let middle =
                        decoded_i24.get(decoded_i24.len() / 2).copied().unwrap_or(0) as i64;
                    let last = decoded_i24.last().copied().unwrap_or(0) as i64;
                    checksum = checksum.wrapping_add(first + middle + last);
                }
                black_box(checksum);
                checksum as u64
            }
            PcmFormat::F32 => {
                let mut checksum = 0.0_f32;
                for packet in packets {
                    let decoded_frames =
                        decoder.decode_f32(black_box(packet), &mut decoded_f32, false)?;
                    if decoded_frames != frame_size {
                        return Err("unexpected decoded frame size".into());
                    }
                    checksum += decoded_checksum(&decoded_f32);
                }
                black_box(checksum);
                u64::from(checksum.to_bits())
            }
        };
        times.push(start.elapsed().as_secs_f64() * 1000.0);
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
    pcm: PcmInput<'_>,
    options: &Options,
    limit: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("impl\tmode\tframe_size\tframe_ms\tbitrate\tframe\tlen\thex");
    for &mode in modes(options) {
        for &frame_size in &CELT_FRAME_SIZES_48K {
            if let Some(selected) = options.frame_size {
                if selected != frame_size {
                    continue;
                }
            }
            for &bitrate in &BITRATES {
                if !bitrate_enabled(options, bitrate) {
                    continue;
                }
                let encoded = encode_packets(
                    pcm,
                    frame_size,
                    bitrate,
                    mode,
                    options.application,
                    options.direct_cubic,
                )?;
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
    let (pcm, pcm_i16, pcm_i24) = if let Some(path) = options.input_s32le.as_deref() {
        let (mut pcm, source_i24) = load_i24_s32le(path, options.seconds)?;
        if options.pcm_bits.unwrap_or(24) == 16 {
            let pcm_i16 = source_i24
                .iter()
                .map(|&sample| (sample >> 8) as i16)
                .collect::<Vec<_>>();
            for (reference, &sample) in pcm.iter_mut().zip(&pcm_i16) {
                *reference = sample as f32 / 32_768.0;
            }
            (pcm, Some(pcm_i16), None)
        } else {
            (pcm, None, Some(source_i24))
        }
    } else {
        let mut pcm = generate_fixture(options.seconds, options.fixture);
        match options.pcm_bits {
            Some(16) => {
                let pcm_i16 = quantize_fixture_i16(&mut pcm);
                (pcm, Some(pcm_i16), None)
            }
            Some(24) => {
                let pcm_i24 = quantize_fixture_i24(&mut pcm);
                (pcm, None, Some(pcm_i24))
            }
            _ => (pcm, None, None),
        }
    };
    let input = if let Some(pcm_i16) = pcm_i16.as_deref() {
        PcmInput::I16(pcm_i16)
    } else if let Some(pcm_i24) = pcm_i24.as_deref() {
        PcmInput::I24(pcm_i24)
    } else {
        PcmInput::F32(&pcm)
    };
    if let Some(limit) = options.dump_packets {
        return dump_packets(input, &options, limit);
    }

    println!("impl\tmode\tframe_size\tframe_ms\tbitrate\tencode_ms\tdecode_ms\tbytes\tmin_packet\tmax_packet\tchecksum\tquality_lag\tquality_snr_db");
    for &mode in modes(&options) {
        for &frame_size in &CELT_FRAME_SIZES_48K {
            if let Some(selected) = options.frame_size {
                if selected != frame_size {
                    continue;
                }
            }
            for &bitrate in &BITRATES {
                if !bitrate_enabled(&options, bitrate) {
                    continue;
                }
                let (encode_ms, bytes, min_packet, max_packet, encode_checksum) = time_encode(
                    input,
                    frame_size,
                    bitrate,
                    mode,
                    options.application,
                    options.direct_cubic,
                    options.repeats,
                )?;
                let encoded = encode_packets(
                    input,
                    frame_size,
                    bitrate,
                    mode,
                    options.application,
                    options.direct_cubic,
                )?;
                let quality = if options.skip_quality {
                    DecodeQuality {
                        lag_frames: 0,
                        snr_db: f64::NAN,
                    }
                } else {
                    let decoded = decode_packets(
                        &encoded.packets,
                        frame_size,
                        input.format(),
                        options.direct_cubic,
                        options.decode_api,
                    )?;
                    if let Some(lag) = options.quality_lag {
                        let total_frames = pcm.len().min(decoded.len()) / CHANNELS;
                        let trim = (SAMPLE_RATE / 50).min(total_frames.saturating_sub(16));
                        quality_at_lag(&pcm, &decoded, lag, total_frames.saturating_sub(trim))
                    } else {
                        aligned_quality(&pcm, &decoded)
                    }
                };
                let (decode_ms, decode_checksum) = time_decode(
                    &encoded.packets,
                    frame_size,
                    options.repeats,
                    input.format(),
                    options.direct_cubic,
                    options.decode_api,
                )?;
                let checksum = encode_checksum ^ decode_checksum;
                println!(
                    "rust\t{}\t{}\t{:.1}\t{}\t{:.4}\t{:.4}\t{}\t{}\t{}\t{}\t{}\t{:.2}",
                    mode.label(),
                    frame_size,
                    frame_size as f64 * 1000.0 / SAMPLE_RATE as f64,
                    bitrate,
                    encode_ms,
                    decode_ms,
                    bytes,
                    min_packet,
                    max_packet,
                    checksum,
                    quality.lag_frames,
                    quality.snr_db
                );
            }
        }
    }
    Ok(())
}
