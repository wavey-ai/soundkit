use soundkit_aac_lc::{AacLcDecoder, Result};

const FIXTURE: &[u8] =
    include_bytes!("../../golden/aac/WESTSIDE_MIX_4_CONFIRMATION_130323_256k.aac");

fn main() {
    match run() {
        Ok(report) => println!("{report}"),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}

fn run() -> Result<String> {
    let frames = parse_adts_frames(FIXTURE)?;
    let first = frames
        .first()
        .ok_or(soundkit_aac_lc::AacLcError::InvalidBitstream(
            "fixture has no ADTS frames",
        ))?;
    let asc = first.audio_specific_config();
    let mut decoder = AacLcDecoder::from_audio_specific_config(&asc)?;
    let mut decoded = 0usize;
    let mut nonzero_frames = 0usize;

    for (index, frame) in frames.iter().enumerate() {
        match decoder.decode_access_unit(frame.raw) {
            Ok(pcm) => {
                decoded += 1;
                if pcm
                    .channels()
                    .iter()
                    .flat_map(|channel| channel.iter())
                    .any(|sample| *sample != 0.0)
                {
                    nonzero_frames += 1;
                }
            }
            Err(error) => {
                return Ok(format!(
                    "decoded={decoded} nonzero={nonzero_frames} total={} first_error_frame={index} error={error}",
                    frames.len()
                ));
            }
        }
    }

    Ok(format!(
        "decoded={decoded} nonzero={nonzero_frames} total={} status=complete",
        frames.len()
    ))
}

#[derive(Clone, Copy, Debug)]
struct AdtsFrame<'a> {
    raw: &'a [u8],
    profile: u8,
    sample_rate_index: u8,
    channels: u8,
}

impl AdtsFrame<'_> {
    fn audio_specific_config(self) -> [u8; 2] {
        [
            (self.profile << 3) | (self.sample_rate_index >> 1),
            ((self.sample_rate_index & 1) << 7) | (self.channels << 3),
        ]
    }
}

fn parse_adts_frames(data: &[u8]) -> Result<Vec<AdtsFrame<'_>>> {
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
        let profile = ((data[offset + 2] & 0xc0) >> 6) + 1;
        let sample_rate_index = (data[offset + 2] & 0x3c) >> 2;
        let channels = ((data[offset + 2] & 0x01) << 2) | ((data[offset + 3] & 0xc0) >> 6);
        let frame_len = (((data[offset + 3] & 0x03) as usize) << 11)
            | ((data[offset + 4] as usize) << 3)
            | (((data[offset + 5] & 0xe0) as usize) >> 5);

        if frame_len <= header_len {
            return Err(soundkit_aac_lc::AacLcError::InvalidBitstream(
                "invalid ADTS frame length",
            ));
        }
        if offset + frame_len > data.len() {
            return Err(soundkit_aac_lc::AacLcError::UnexpectedEof {
                requested_bits: 8,
                remaining_bits: (data.len() - offset) * 8,
            });
        }

        frames.push(AdtsFrame {
            raw: &data[offset + header_len..offset + frame_len],
            profile,
            sample_rate_index,
            channels,
        });
        offset += frame_len;
    }

    Ok(frames)
}
