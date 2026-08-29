use std::time::Instant;

const WESTSIDE: &[u8] =
    include_bytes!("../../golden/aac/WESTSIDE_MIX_4_CONFIRMATION_130323_256k.aac");
const STEREO_MUSIC: &[u8] =
    include_bytes!("../../golden/aac/stereo-music-44100-192k.aac");

const WARMUP: usize = 1;
const MEASURED: usize = 3;

fn main() {
    let fixtures = [
        ("WESTSIDE 256k stereo 48k", WESTSIDE),
        ("stereo-music 192k 44.1k", STEREO_MUSIC),
    ];

    for (name, data) in fixtures {
        println!("=== {name} ({:.1} KiB) ===", data.len() as f64 / 1024.0);

        let adts_frames = parse_adts_frames(data).expect("parse ADTS");
        let first = adts_frames.first().expect("no frames");
        let audio_seconds =
            adts_frames.len() as f64 * 1024.0 / first.sample_rate as f64;

        println!(
            "  frames={} sample_rate={} channels={} audio_seconds={:.2}",
            adts_frames.len(),
            first.sample_rate,
            first.channels,
            audio_seconds
        );

        // ── rusty_aac ──
        let rusty_out = bench_rusty(&adts_frames);
        println!(
            "  rusty_aac:     {:.3} ms  RTF={:.6}  samples={}  checksum={:016x}",
            rusty_out.elapsed_ms, rusty_out.rtf, rusty_out.samples, rusty_out.checksum,
        );

        // ── soundkit-aac-lc ──
        let sk_out = bench_soundkit(&adts_frames);
        println!(
            "  soundkit-aac-lc: {:.3} ms  RTF={:.6}  samples={}  checksum={:016x}",
            sk_out.elapsed_ms, sk_out.rtf, sk_out.samples, sk_out.checksum,
        );

        let speedup = rusty_out.elapsed_ms / sk_out.elapsed_ms;
        println!("  speedup: {speedup:.2}x\n");
    }
}

struct BenchResult {
    elapsed_ms: f64,
    rtf: f64,
    samples: usize,
    checksum: u64,
}

fn bench_rusty(frames: &[AdtsFrame<'_>]) -> BenchResult {
    use rusty_aac::AacDecoder;

    let mut decoder = AacDecoder::new();

    // warmup
    for _ in 0..WARMUP {
        for frame in frames {
            let _ = decoder.decode(frame.full, None);
        }
    }

    let started = Instant::now();
    let mut total_samples = 0usize;
    let mut checksum = 0xcbf29ce484222325u64;
    let mut frames_decoded = 0usize;

    for _ in 0..MEASURED {
        for frame in frames {
            match decoder.decode(frame.full, None) {
                Ok(decoded) => {
                    frames_decoded += 1;
                    total_samples += decoded.samples.len();
                    for &s in &decoded.samples {
                        checksum ^= s.to_bits() as u64;
                        checksum = checksum.wrapping_mul(0x100000001b3);
                    }
                }
                Err(_) => {}
            }
        }
    }

    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let audio_seconds =
        frames_decoded as f64 * 1024.0 / frames[0].sample_rate as f64 / MEASURED as f64;
    BenchResult {
        elapsed_ms,
        rtf: (elapsed_ms / 1000.0) / audio_seconds,
        samples: total_samples / MEASURED,
        checksum,
    }
}

fn bench_soundkit(frames: &[AdtsFrame<'_>]) -> BenchResult {
    use soundkit_aac_lc::AacLcDecoder;

    let asc = frames[0].audio_specific_config();
    let mut decoder = AacLcDecoder::from_audio_specific_config(&asc)
        .expect("AacLcDecoder::from_audio_specific_config");

    // warmup
    for _ in 0..WARMUP {
        for frame in frames {
            let _ = decoder.decode_access_unit(frame.raw);
        }
    }

    let started = Instant::now();
    let mut total_samples = 0usize;
    let mut checksum = 0xcbf29ce484222325u64;
    let mut frames_decoded = 0usize;

    for _ in 0..MEASURED {
        for frame in frames {
            match decoder.decode_access_unit(frame.raw) {
                Ok(pcm) => {
                    frames_decoded += 1;
                    let n_frames = pcm.frames();
                    total_samples += pcm.channels().len() * n_frames;
                    for ch in pcm.channels() {
                        for &s in ch.iter().take(n_frames) {
                            checksum ^= s.to_bits() as u64;
                            checksum = checksum.wrapping_mul(0x100000001b3);
                        }
                    }
                }
                Err(_) => {}
            }
        }
    }

    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let audio_seconds =
        frames_decoded as f64 * 1024.0 / frames[0].sample_rate as f64 / MEASURED as f64;
    BenchResult {
        elapsed_ms,
        rtf: (elapsed_ms / 1000.0) / audio_seconds,
        samples: total_samples / MEASURED,
        checksum,
    }
}

// ── ADTS parser ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
struct AdtsFrame<'a> {
    full: &'a [u8],
    raw: &'a [u8],
    sample_rate: u32,
    channels: u8,
}

impl AdtsFrame<'_> {
    fn audio_specific_config(self) -> [u8; 2] {
        let profile = ((self.full[2] & 0xc0) >> 6) + 1;
        let sample_rate_index = (self.full[2] & 0x3c) >> 2;
        [
            (profile << 3) | (sample_rate_index >> 1),
            ((sample_rate_index & 1) << 7) | (self.channels << 3),
        ]
    }
}

fn adts_sample_rate(index: u8) -> Option<u32> {
    const RATES: [u32; 13] = [
        96_000, 88_200, 64_000, 48_000, 44_100, 32_000, 24_000, 22_050, 16_000, 12_000, 11_025,
        8_000, 7_350,
    ];
    RATES.get(index as usize).copied()
}

fn parse_adts_frames(data: &[u8]) -> Result<Vec<AdtsFrame<'_>>, &'static str> {
    let mut frames = Vec::new();
    let mut offset = 0usize;

    while offset + 7 <= data.len() {
        while offset + 7 <= data.len()
            && !(data[offset] == 0xff && (data[offset + 1] & 0xf0) == 0xf0)
        {
            offset += 1;
        }
        if offset + 7 > data.len() {
            break;
        }

        let protection_absent = (data[offset + 1] & 0x01) != 0;
        let header_len = if protection_absent { 7 } else { 9 };
        let sample_rate_index = (data[offset + 2] & 0x3c) >> 2;
        let sample_rate =
            adts_sample_rate(sample_rate_index).ok_or("unsupported sample-rate index")?;
        let channels = ((data[offset + 2] & 0x01) << 2) | ((data[offset + 3] & 0xc0) >> 6);
        let frame_len = (((data[offset + 3] & 0x03) as usize) << 11)
            | ((data[offset + 4] as usize) << 3)
            | (((data[offset + 5] & 0xe0) as usize) >> 5);

        if frame_len <= header_len {
            return Err("invalid ADTS frame length");
        }
        if offset + frame_len > data.len() {
            return Err("truncated ADTS frame");
        }

        frames.push(AdtsFrame {
            full: &data[offset..offset + frame_len],
            raw: &data[offset + header_len..offset + frame_len],
            sample_rate,
            channels,
        });
        offset += frame_len;
    }

    if frames.is_empty() {
        Err("no ADTS frames found")
    } else {
        Ok(frames)
    }
}
