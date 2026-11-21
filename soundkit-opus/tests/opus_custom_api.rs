//! Opus Custom encoder and decoder API behaviour.

use soundkit_opus::{Decoder, Encoder, Error, BITRATE_MAX};

fn tone(frame: usize, channels: usize, frames: usize) -> Vec<i16> {
    (0..frame * frames)
        .flat_map(|i| {
            let v = ((i % 109) as i32 * 600 - 32_000) as i16;
            std::iter::repeat_n(v, channels)
        })
        .collect()
}

#[test]
fn custom_frame_sizes_follow_libopus() {
    for frame in [48, 64, 72, 96, 120, 240, 480, 960] {
        assert!(Encoder::new_custom(48_000, 2, frame).is_ok(), "{frame}");
        assert!(Decoder::new_custom(48_000, 2, frame).is_ok(), "{frame}");
    }
    for frame in [0, 40, 46, 52, 98, 1000] {
        assert!(Encoder::new_custom(48_000, 2, frame).is_err(), "{frame}");
        assert!(Decoder::new_custom(48_000, 2, frame).is_err(), "{frame}");
    }
    assert!(Encoder::new_custom(44_100, 2, 48).is_err());
    assert!(Encoder::new_custom(48_000, 3, 48).is_err());
}

#[test]
fn cbr_packets_have_the_libopus_size_and_signalling_byte() {
    for (frame, channels, bitrate) in [(48, 2, 256_000), (64, 1, 192_000), (72, 2, 320_000)] {
        let mut encoder = Encoder::new_custom(48_000, channels, frame).unwrap();
        encoder.set_bitrate(bitrate).unwrap();
        let pcm = tone(frame, channels, 4);
        // opus_custom_encode: (bitrate * N + 4 * Fs) / (8 * Fs) - signalling,
        // plus the signalling byte itself.
        let payload = (bitrate as usize * frame + 4 * 48_000) / (8 * 48_000) - 1;
        for chunk in pcm.chunks_exact(frame * channels) {
            let packet = encoder.encode_i16_vec(chunk, frame).unwrap();
            assert_eq!(packet.len(), payload + 1, "frame {frame}");
            assert_eq!(packet[0], u8::from(channels == 2) << 2, "frame {frame}");
        }
        encoder.set_custom_signalling(false).unwrap();
        let packet = encoder
            .encode_i16_vec(&pcm[..frame * channels], frame)
            .unwrap();
        assert_eq!(
            packet.len(),
            payload + 1,
            "frame {frame} without signalling"
        );
    }
}

#[test]
fn maximum_bitrate_fills_the_packet_limit() {
    let mut encoder = Encoder::new_custom(48_000, 1, 48).unwrap();
    assert_eq!(encoder.bitrate(), BITRATE_MAX);
    encoder.set_max_packet_bytes(100).unwrap();
    let packet = encoder.encode_i16_vec(&tone(48, 1, 1), 48).unwrap();
    assert_eq!(packet.len(), 100);
    assert!(encoder.set_max_packet_bytes(1).is_err());
    assert!(encoder.set_bitrate(400).is_err());
}

#[test]
fn standard_encoders_reject_custom_controls() {
    let mut encoder = Encoder::try_new(48_000, 16, 2, 240, 128_000).unwrap();
    assert_eq!(encoder.set_custom_signalling(false), Err(Error::BadArg));
    assert!(!encoder.is_custom());
    assert!(!encoder.libopus_decisions());
    assert!(!encoder.libopus_arithmetic());
    let mut decoder = Decoder::new(48_000, 2).unwrap();
    assert_eq!(decoder.set_custom_signalling(false), Err(Error::BadArg));
}

#[test]
fn custom_round_trip_with_and_without_signalling() {
    for signalling in [true, false] {
        let mut encoder = Encoder::new_custom(48_000, 2, 64).unwrap();
        encoder.set_bitrate(256_000).unwrap();
        encoder.set_custom_signalling(signalling).unwrap();
        let mut decoder = Decoder::new_custom(48_000, 2, 64).unwrap();
        decoder.set_custom_signalling(signalling).unwrap();
        let pcm = tone(64, 2, 50);
        let mut decoded = Vec::new();
        for chunk in pcm.chunks_exact(128) {
            let packet = encoder.encode_i16_vec(chunk, 64).unwrap();
            decoded.extend(decoder.decode_i16_vec(&packet, false).unwrap());
        }
        assert_eq!(decoded.len(), pcm.len());
        // The codec delay is the 64-sample overlap.
        let error: f64 = pcm[..pcm.len() - 128]
            .iter()
            .zip(&decoded[128..])
            .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
            .sum();
        let energy: f64 = pcm.iter().map(|&a| (a as f64).powi(2)).sum();
        assert!(
            10.0 * (energy / error).log10() > 15.0,
            "signalling {signalling}"
        );
    }
}

#[test]
fn empty_packets_conceal_a_frame() {
    let mut encoder = Encoder::new_custom(48_000, 1, 48).unwrap();
    encoder.set_bitrate(128_000).unwrap();
    let mut decoder = Decoder::new_custom(48_000, 1, 48).unwrap();
    let pcm = tone(48, 1, 40);
    for chunk in pcm.chunks_exact(48) {
        let packet = encoder.encode_i16_vec(chunk, 48).unwrap();
        assert_eq!(decoder.decode_i16_vec(&packet, false).unwrap().len(), 48);
    }
    for _ in 0..60 {
        let concealed = decoder.decode_i16_vec(&[], false).unwrap();
        assert_eq!(concealed.len(), 48);
    }
    // A one-byte packet holds no CELT data after the signalling byte.
    assert_eq!(decoder.decode_i16_vec(&[0], false).unwrap().len(), 48);

    // Standard Opus decoders conceal frames of the last decoded size.
    let mut encoder = Encoder::try_new(48_000, 16, 2, 240, 128_000).unwrap();
    let mut decoder = Decoder::new(48_000, 2).unwrap();
    let pcm = tone(240, 2, 2);
    let packet = encoder.encode_i16_vec(&pcm[..480], 240).unwrap();
    decoder.decode_i16_vec(&packet, false).unwrap();
    assert_eq!(decoder.decode_i16_vec(&[], false).unwrap().len(), 480);
}

#[test]
fn custom_decoder_rejects_frames_longer_than_configured() {
    let mut encoder = Encoder::new_custom(48_000, 1, 240).unwrap();
    encoder.set_bitrate(64_000).unwrap();
    let packet = encoder.encode_i16_vec(&tone(240, 1, 1), 240).unwrap();
    let mut decoder = Decoder::new_custom(48_000, 1, 120).unwrap();
    assert_eq!(
        decoder.decode_i16_vec(&packet, false),
        Err(Error::BufferTooSmall)
    );
    let mut decoder = Decoder::new_custom(48_000, 1, 240).unwrap();
    assert_eq!(decoder.decode_i16_vec(&packet, false).unwrap().len(), 240);
}

#[test]
fn settings_survive_reset() {
    let mut encoder = Encoder::new_custom(48_000, 2, 72).unwrap();
    encoder.set_bitrate(192_000).unwrap();
    encoder.set_vbr(true).unwrap();
    encoder.set_custom_signalling(false).unwrap();
    encoder.set_portable_pvq(true).unwrap();
    let pcm = tone(72, 2, 3);
    let first: Vec<_> = pcm
        .chunks_exact(144)
        .map(|chunk| encoder.encode_i16_vec(chunk, 72).unwrap())
        .collect();
    encoder.reset().unwrap();
    assert!(encoder.is_custom() && !encoder.custom_signalling() && encoder.vbr());
    assert!(encoder.portable_pvq() && encoder.libopus_arithmetic());
    let second: Vec<_> = pcm
        .chunks_exact(144)
        .map(|chunk| encoder.encode_i16_vec(chunk, 72).unwrap())
        .collect();
    assert_eq!(first, second);

    let mut decoder = Decoder::new_custom(48_000, 2, 72).unwrap();
    decoder.set_custom_signalling(false).unwrap();
    let a = decoder.decode_i16_vec(&first[0], false).unwrap();
    decoder.reset().unwrap();
    assert!(decoder.is_custom() && !decoder.custom_signalling());
    assert_eq!(decoder.decode_i16_vec(&first[0], false).unwrap(), a);
}
