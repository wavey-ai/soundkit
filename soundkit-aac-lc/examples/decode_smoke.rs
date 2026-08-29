use std::time::Instant;

const WESTSIDE: &[u8] =
    include_bytes!("../../golden/aac/WESTSIDE_MIX_4_CONFIRMATION_130323_256k.aac");
const STEREO_MUSIC: &[u8] =
    include_bytes!("../../golden/aac/stereo-music-44100-192k.aac");

fn main() {
    let adts_frames = parse_adts_frames(WESTSIDE).expect("parse ADTS");
    let first = adts_frames.first().expect("no frames");
    println!(
        "frames={} sample_rate={} channels={}",
        adts_frames.len(),
        first.sample_rate,
        first.channels
    );

    // rusty_aac - single pass
    {
        use rusty_aac::AacDecoder;
        let mut decoder = AacDecoder::new();
        let started = Instant::now();
        let mut ok = 0u64;
        let mut err = 0u64;
        for frame in &adts_frames {
            match decoder.decode(frame.full, None) {
                Ok(_) => ok += 1,
                Err(e) => {
                    err += 1;
                    if err <= 3 {
                        eprintln!("rusty_aac error: {e:?}");
                    }
                }
            }
        }
        println!(
            "rusty_aac: ok={ok} err={err} elapsed={:.1}ms",
            started.elapsed().as_secs_f64() * 1000.0
        );
    }

    // soundkit-aac-lc - single pass
    {
        use soundkit_aac_lc::AacLcDecoder;
        let asc = first.audio_specific_config();
        let mut decoder =
            AacLcDecoder::from_audio_specific_config(&asc).expect("create decoder");
        let started = Instant::now();
        let mut ok = 0u64;
        let mut err = 0u64;
        for frame in &adts_frames {
            match decoder.decode_access_unit(frame.raw) {
                Ok(_) => ok += 1,
                Err(e) => {
                    err += 1;
                    if err <= 3 {
                        eprintln!("soundkit error: {e}");
                    }
                }
            }
        }
        println!(
            "soundkit: ok={ok} err={err} elapsed={:.1}ms",
            started.elapsed().as_secs_f64() * 1000.0
        );
    }
}

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
