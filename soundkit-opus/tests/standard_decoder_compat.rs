//! Standard Opus decoders keep SoundKit's established results for lost,
//! short and damaged packets unless libopus's handling is asked for, and the
//! frame-count helpers choose between standard and Opus Custom codecs.

use soundkit::audio_packet::Encoder as SoundkitEncoder;
use soundkit_opus::celt::modes::CeltMode;
use soundkit_opus::{custom_frame_size, Decoder, Encoder, Error, OpusEncoder, OpusStreamEncoder};

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

#[test]
fn custom_frame_size_names_custom_only_counts() {
    for frame in [48, 64, 72, 96, 100, 144, 160, 200] {
        assert_eq!(custom_frame_size(48_000, frame), Some(frame), "{frame}");
    }
    for frame in [
        0, 1, 40, 46, 47, 98, 120, 240, 360, 480, 600, 720, 960, 1000, 1920, 5760,
    ] {
        assert_eq!(custom_frame_size(48_000, frame), None, "{frame}");
    }
    for rate in [8_000, 16_000, 24_000, 44_100, 96_000] {
        assert_eq!(custom_frame_size(rate, 48), None, "{rate}");
    }
}

#[test]
fn custom_frame_size_agrees_with_mode_construction() {
    // The helper validates without building modes; check it against
    // building one for every candidate size (odd and out-of-range sizes
    // are rejected before any table is built).
    for frame in 0..=1100 {
        let builds = CeltMode::new(48_000, frame).is_ok();
        assert_eq!(
            CeltMode::valid_48k_frame_size(frame),
            builds,
            "frame size {frame}"
        );
        assert_eq!(
            custom_frame_size(48_000, frame).is_some(),
            builds && frame % 120 != 0,
            "frame size {frame}"
        );
    }
}

#[test]
fn soundkit_constructors_build_custom_encoders_for_custom_frame_sizes() {
    for (frame, channels) in [(48usize, 2u32), (64, 1), (72, 2)] {
        // daw-nexus: OpusEncoder::new(48000, 16, ch, frame, 96000) then init().
        let mut encoder = OpusEncoder::new(48_000, 16, channels, frame as u32, 96_000);
        encoder.init().unwrap();
        assert!(encoder.is_custom());
        assert_eq!(encoder.custom_frame_size(), Some(frame));
        assert!(encoder.custom_signalling() && !encoder.vbr());
        assert_eq!(encoder.bitrate(), 96_000);

        let mut reference = Encoder::new_custom(48_000, channels as usize, frame).unwrap();
        reference.set_bitrate(96_000).unwrap();
        reference.set_vbr(false).unwrap();

        let mut decoder = Decoder::new_for_frame_count(48_000, channels as usize, frame).unwrap();
        assert_eq!(decoder.custom_frame_size(), Some(frame));
        let mut reference_decoder = Decoder::new_custom(48_000, channels as usize, frame).unwrap();

        let pcm = tone(frame, channels as usize, 50);
        let mut output = vec![0u8; 1500];
        for chunk in pcm.chunks_exact(frame * channels as usize) {
            // The SoundKit encoder trait path.
            let len = SoundkitEncoder::encode_i16(&mut encoder, chunk, &mut output).unwrap();
            let packet = &output[..len];
            assert_eq!(packet, reference.encode_i16_vec(chunk, frame).unwrap());
            // libopus CBR size: (96000 * N + 4 * 48000) / (8 * 48000) bytes,
            // the signalling byte included.
            assert_eq!(len, (96_000 * frame + 4 * 48_000) / (8 * 48_000));
            assert_eq!(
                decoder.decode_i16_vec(packet, false),
                reference_decoder.decode_i16_vec(packet, false)
            );
        }
    }
}

#[test]
fn standard_frame_sizes_keep_standard_codecs() {
    for frame in [120u32, 240, 480, 960] {
        let encoder = Encoder::try_new(48_000, 16, 2, frame, 96_000).unwrap();
        assert!(!encoder.is_custom());
        let decoder = Decoder::new_for_frame_count(48_000, 2, frame as usize).unwrap();
        assert!(!decoder.is_custom() && !decoder.conceal_errors());
    }
    for frame in [0usize, 47, 5760] {
        let decoder = Decoder::new_for_frame_count(48_000, 2, frame).unwrap();
        assert!(!decoder.is_custom(), "{frame}");
    }
    // Sizes neither standard nor custom stay errors, with the old message.
    let error = Encoder::try_new(48_000, 16, 2, 47, 96_000).unwrap_err();
    assert!(
        error.contains("unsupported 48 kHz CELT frame size 47"),
        "{error}"
    );
    assert!(Encoder::try_new(48_000, 8, 2, 48, 96_000).is_err());
    // The raw OpusHead stream is standard Opus only.
    assert!(OpusStreamEncoder::new(48_000, 16, 2, 48, 96_000).is_err());
    assert!(OpusStreamEncoder::new(48_000, 16, 2, 240, 96_000).is_ok());
}

#[test]
fn integer_and_float_input_encode_alike() {
    // Integer input skips the input-clip scan; packets must not change.
    let pcm = tone(240, 2, 30);
    let pcm_f32: Vec<f32> = pcm.iter().map(|&v| v as f32 / 32768.0).collect();
    for custom in [false, true] {
        let make = || {
            if custom {
                let mut encoder = Encoder::new_custom(48_000, 2, 64).unwrap();
                encoder.set_bitrate(256_000).unwrap();
                encoder
            } else {
                Encoder::try_new(48_000, 16, 2, 240, 256_000).unwrap()
            }
        };
        let frame = if custom { 64 } else { 240 };
        let (mut a, mut b) = (make(), make());
        for (chunk, chunk_f32) in pcm
            .chunks_exact(2 * frame)
            .zip(pcm_f32.chunks_exact(2 * frame))
        {
            assert_eq!(
                a.encode_i16_vec(chunk, frame).unwrap(),
                b.encode_f32_vec(chunk_f32, frame).unwrap()
            );
        }
    }
}
