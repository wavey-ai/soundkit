//! Decoding a real file, when one is sitting beside the repo.
//!
//! The synthetic bitstreams in the unit tests cover syntax the encoder here
//! can produce. They cannot cover what other encoders emit — a program
//! config element, a data stream element, a low-frequency channel beside a
//! stereo pair — and those are exactly the things this decoder used to
//! refuse. So when a real file is present, it is decoded.
//!
//! The fixture is a symlink in `testdata/local/`, which is not in the repo.
//! Without it the test says so and passes: a machine that has no file to
//! read has not found a fault.

use std::path::Path;

#[test]
fn decodes_the_local_fixtures() {
    let mut found = 0;
    for name in ["never-final.mov", "lori_4k_no_grain.mp4"] {
        if decode_one(name) {
            found += 1;
        }
    }
    eprintln!("{found} local fixture(s) decoded");
}

/// Returns false when the file simply is not on this machine.
fn decode_one(name: &str) -> bool {
    let owned = format!("../testdata/local/{name}");
    let path = Path::new(&owned);
    if !path.exists() {
        eprintln!("no local fixture at {}; nothing to decode", path.display());
        return false;
    }
    eprintln!("\n── {name}");
    let bytes = std::fs::read(path).expect("the fixture reads");
    eprintln!("fixture: {} bytes", bytes.len());

    let index = soundkit_audio_demux::Mp4MediaIndex::from_file(&bytes)
        .expect("the container indexes");
    let track = index
        .tracks
        .iter()
        .find(|track| track.kind == soundkit_audio_demux::MediaTrackKind::Audio)
        .expect("the file has an audio track");
    eprintln!(
        "audio track: codec {}, {} samples, {:?} Hz, {:?} ch, config {} bytes",
        track.codec,
        track.sample_count,
        track.sample_rate,
        track.channels,
        track.codec_private.len()
    );

    let mut decoder = soundkit_aac_lc::AacLcDecoder::from_audio_specific_config(&track.codec_private)
        .expect("the decoder configures from the track's own config");

    let samples: Vec<_> = index
        .samples
        .iter()
        .filter(|sample| sample.track_id == track.track_id)
        .take(400)
        .collect();
    eprintln!("decoding {} of this track's access units", samples.len());

    let mut decoded = 0usize;
    let mut peak = 0.0f32;
    for (position, sample) in samples.iter().enumerate() {
        let from = sample.absolute_offset as usize;
        let packet = &bytes[from..from + sample.size as usize];
        let frame = decoder
            .decode_access_unit(packet)
            .unwrap_or_else(|error| panic!("access unit {position} failed: {error}"));
        for channel in frame.channels() {
            for sample in channel {
                let magnitude = sample.abs();
                if magnitude > peak {
                    peak = magnitude;
                }
            }
        }
        decoded += 1;
    }
    eprintln!("decoded {decoded} access units, peak {peak:.4}");
    assert!(decoded > 0, "{name}: no access unit decoded");
    assert!(peak > 0.001, "{name}: the decode produced silence");
    true
}
