//! Standard Opus decoders keep SoundKit's established results for lost,
//! short and damaged packets unless libopus's handling is asked for.

use soundkit_opus::{Decoder, Encoder, Error};

fn tone(frame: usize, channels: usize, frames: usize) -> Vec<i16> {
    (0..frame * frames)
        .flat_map(|i| {
            let t = i as f32 / 48_000.0;
            let v = (9_000.0 * (2.0 * std::f32::consts::PI * 330.0 * t).sin()
                + 3_000.0 * (2.0 * std::f32::consts::PI * 1_210.0 * t).sin())
                as i16;
            std::iter::repeat_n(v, channels)
        })
        .collect()
}

fn packets(frame: usize, channels: usize, frames: usize) -> Vec<Vec<u8>> {
    let mut encoder = Encoder::try_new(48_000, 16, channels as u32, frame as u32, 128_000).unwrap();
    tone(frame, channels, frames)
        .chunks_exact(frame * channels)
        .map(|chunk| encoder.encode_i16_vec(chunk, frame).unwrap())
        .collect()
}

#[test]
fn standard_decoders_reject_lost_and_short_packets_by_default() {
    for frame in [120, 240, 480, 960] {
        for channels in [1, 2] {
            let stream = packets(frame, channels, 4);
            let mut decoder = Decoder::new(48_000, channels).unwrap();
            assert!(!decoder.conceal_errors());
            decoder.decode_i16_vec(&stream[0], false).unwrap();
            let toc = stream[0][0];
            assert_eq!(
                decoder.decode_i16_vec(&[], false),
                Err(Error::InvalidPacket),
                "{frame} {channels}"
            );
            assert_eq!(
                decoder.decode_i16_vec(&[toc], false),
                Err(Error::BadArg),
                "{frame} {channels}"
            );
            assert_eq!(
                decoder.decode_i16_vec(&[toc, 0x55], false),
                Err(Error::BadArg),
                "{frame} {channels}"
            );
            assert_eq!(
                decoder.decode_i16_vec(&stream[1], false).unwrap().len(),
                frame * channels
            );
        }
    }
}

#[test]
fn standard_mono_decoders_reject_stereo_packets_by_default() {
    let stereo = packets(240, 2, 2);
    let mut decoder = Decoder::new(48_000, 1).unwrap();
    assert_eq!(
        decoder.decode_i16_vec(&stereo[0], false),
        Err(Error::Unimplemented)
    );
    decoder.set_conceal_errors(true).unwrap();
    assert_eq!(
        decoder.decode_i16_vec(&stereo[1], false).unwrap().len(),
        240
    );
}

#[test]
fn conceal_errors_opt_in_conceals_lost_and_short_packets() {
    for frame in [120, 240, 480, 960] {
        let stream = packets(frame, 2, 3);
        let mut decoder = Decoder::new(48_000, 2).unwrap();
        decoder.set_conceal_errors(true).unwrap();
        assert!(decoder.conceal_errors());
        decoder.decode_i16_vec(&stream[0], false).unwrap();
        assert_eq!(decoder.decode_i16_vec(&[], false).unwrap().len(), 2 * frame);
        let toc = stream[0][0];
        assert_eq!(
            decoder.decode_i16_vec(&[toc], false).unwrap().len(),
            2 * frame
        );
        assert_eq!(
            decoder.decode_i16_vec(&[toc, 0x55], false).unwrap().len(),
            2 * frame
        );
        assert_eq!(
            decoder.decode_i16_vec(&stream[1], false).unwrap().len(),
            2 * frame
        );
        // Turning it off restores the default results.
        decoder.set_conceal_errors(false).unwrap();
        assert!(decoder.decode_i16_vec(&[], false).is_err());
        // reset() keeps the choice.
        decoder.set_conceal_errors(true).unwrap();
        decoder.reset().unwrap();
        assert!(decoder.conceal_errors());
    }
}

#[test]
fn decode_lost_conceals_without_the_opt_in() {
    let stream = packets(480, 2, 6);
    let mut decoder = Decoder::new(48_000, 2).unwrap();
    let mut pcm = Vec::new();
    // Before any packet a standard decoder conceals 2.5 ms.
    assert_eq!(decoder.decode_lost_i16_into(&mut pcm).unwrap(), 120);
    assert_eq!(pcm.len(), 240);
    for packet in &stream[..4] {
        decoder.decode_i16_vec(packet, false).unwrap();
    }
    assert_eq!(decoder.decode_lost_i16_into(&mut pcm).unwrap(), 480);
    let mut pcm24 = Vec::new();
    assert_eq!(decoder.decode_lost_i24_into(&mut pcm24).unwrap(), 480);
    let mut pcm32 = Vec::new();
    assert_eq!(decoder.decode_lost_f32_into(&mut pcm32).unwrap(), 480);
    assert_eq!((pcm24.len(), pcm32.len()), (960, 960));
    // The concealment decays from the decoded audio, not from silence.
    assert!(pcm.iter().any(|&sample| sample != 0));
    assert_eq!(
        decoder.decode_i16_vec(&stream[4], false).unwrap().len(),
        960
    );
    assert!(!decoder.conceal_errors());
    assert!(decoder.decode_i16_vec(&[], false).is_err());

    // Concealment that a libopus-style decoder produces for an empty packet.
    let mut a = Decoder::new(48_000, 1).unwrap();
    let mut b = Decoder::new(48_000, 1).unwrap();
    b.set_conceal_errors(true).unwrap();
    let mono = packets(240, 1, 8);
    for packet in &mono[..6] {
        assert_eq!(
            a.decode_i16_vec(packet, false),
            b.decode_i16_vec(packet, false)
        );
    }
    for _ in 0..3 {
        let mut lost = Vec::new();
        a.decode_lost_i16_into(&mut lost).unwrap();
        assert_eq!(Ok(lost), b.decode_i16_vec(&[], false));
    }
    assert_eq!(
        a.decode_i16_vec(&mono[6], false),
        b.decode_i16_vec(&mono[6], false)
    );
}

#[test]
fn damaged_frames_are_errors_by_default_and_decoded_with_the_opt_in() {
    // Find a corrupted frame whose range decoding fails.
    let stream = packets(240, 2, 40);
    let mut found = 0;
    for (k, packet) in stream.iter().enumerate().skip(1) {
        for byte in 1..packet.len().min(24) {
            let mut damaged = packet.clone();
            damaged[byte] ^= 0xa5;
            damaged.truncate(packet.len() - (k % 7));
            let mut strict = Decoder::new(48_000, 2).unwrap();
            let mut libopus = Decoder::new(48_000, 2).unwrap();
            libopus.set_conceal_errors(true).unwrap();
            strict.decode_i16_vec(&stream[k - 1], false).unwrap();
            libopus.decode_i16_vec(&stream[k - 1], false).unwrap();
            if strict.decode_i16_vec(&damaged, false) == Err(Error::InvalidPacket) {
                assert_eq!(libopus.decode_i16_vec(&damaged, false).unwrap().len(), 480);
                found += 1;
            }
        }
    }
    assert!(found > 0, "no damaged frame failed range decoding");
}
