use libopus_rs::{
    channels as packet_channels, Application, Decoder, Encoder, CELT_FRAME_SIZES_48K,
    CELT_MAX_FRAME_BYTES, CELT_MIN_FRAME_BYTES,
};

fn tone(frame_size: usize, channels: usize, frame_index: usize) -> Vec<f32> {
    let mut pcm = vec![0.0f32; frame_size * channels];
    for i in 0..frame_size {
        let t = (frame_index * frame_size + i) as f32;
        pcm[i * channels] = 0.18 * (0.017 * t).sin() + 0.04 * (0.071 * t).cos();
        if channels == 2 {
            pcm[i * channels + 1] = 0.16 * (0.019 * t + 0.4).sin() - 0.03 * (0.059 * t).cos();
        }
    }
    pcm
}

#[test]
fn encode_and_decode_48k_celt_only_smoke_path() {
    let mut encoder = Encoder::new(48_000, 2, Application::Audio).unwrap();
    let pcm = (0..960)
        .flat_map(|i| {
            let left = (i as f32 * 0.011).sin() * 0.2;
            let right = (i as f32 * 0.017).cos() * 0.2;
            [left, right]
        })
        .collect::<Vec<_>>();
    let packet = encoder.encode_f32(&pcm, 960).unwrap();
    assert!(!packet.is_empty());

    let mut decoder = Decoder::new(48_000, 2).unwrap();
    let decoded = decoder.decode_f32(&packet, false).unwrap();
    assert_eq!(decoded.len(), pcm.len());
    assert!(decoded.iter().all(|sample| sample.is_finite()));
    assert!(decoded.iter().any(|sample| sample.abs() > 1e-5));
}

#[test]
fn celt_raw_frames_cover_supported_sizes_and_payload_budgets() {
    for channels in [1usize, 2] {
        for &frame_size in &CELT_FRAME_SIZES_48K {
            let mut budgets = vec![
                CELT_MIN_FRAME_BYTES,
                8,
                24,
                (frame_size / 4).max(CELT_MIN_FRAME_BYTES),
                CELT_MAX_FRAME_BYTES,
            ];
            budgets.sort_unstable();
            budgets.dedup();

            for frame_bytes in budgets {
                let mut encoder = Encoder::new(48_000, channels, Application::Audio).unwrap();
                let mut decoder = Decoder::new(48_000, channels).unwrap();
                let packet = encoder
                    .encode_f32_with_frame_bytes(
                        &tone(frame_size, channels, 0),
                        frame_size,
                        frame_bytes,
                    )
                    .unwrap();
                assert_eq!(packet.len(), frame_bytes + 1);
                assert_eq!(decoder.validate_packet(&packet).unwrap(), frame_size);

                let decoded = decoder.decode_f32(&packet, false).unwrap();
                assert_eq!(decoded.len(), frame_size * channels);
                assert!(decoded.iter().all(|sample| sample.is_finite()));
            }
        }
    }
}

#[test]
fn celt_bitrate_control_scales_raw_frame_size() {
    for &frame_size in &CELT_FRAME_SIZES_48K {
        let mut encoder = Encoder::new(48_000, 2, Application::Audio).unwrap();
        encoder.set_bitrate(128_000).unwrap();
        let expected = ((128_000 * frame_size as i32 + 48_000 * 4) / (48_000 * 8)) as usize;
        let packet = encoder
            .encode_f32(&tone(frame_size, 2, 0), frame_size)
            .unwrap();
        assert_eq!(packet.len(), expected);
    }

    let mut encoder = Encoder::new(48_000, 2, Application::Audio).unwrap();
    assert!(encoder.set_bitrate(499).is_err());
    assert!(encoder.set_bitrate(512_001).is_err());
}

#[test]
fn celt_cbr_packet_size_is_capped_to_opus_packet_limit() {
    let mut encoder = Encoder::new(48_000, 2, Application::RestrictedLowDelay).unwrap();
    encoder.set_bitrate(512_000).unwrap();
    let packet = encoder.encode_f32(&tone(960, 2, 0), 960).unwrap();

    assert_eq!(packet.len(), CELT_MAX_FRAME_BYTES + 1);
    assert_eq!(packet.len(), 1275);
}

#[test]
fn low_rate_stereo_celt_can_emit_mono_packets() {
    let mut encoder = Encoder::new(48_000, 2, Application::Audio).unwrap();
    encoder.set_bitrate(48_000).unwrap();
    let packet = encoder.encode_f32(&tone(120, 2, 0), 120).unwrap();
    assert_eq!(packet.len(), 15);
    assert_eq!(packet_channels(&packet).unwrap(), 1);

    let mut decoder = Decoder::new(48_000, 2).unwrap();
    let decoded = decoder.decode_f32(&packet, false).unwrap();
    assert_eq!(decoded.len(), 120 * 2);
    assert!(decoded.iter().all(|sample| sample.is_finite()));
}

#[test]
fn vbr_packet_budget_varies_with_signal_shape() {
    let mut encoder = Encoder::new(48_000, 2, Application::RestrictedLowDelay).unwrap();
    encoder.set_bitrate(96_000).unwrap();
    encoder.set_vbr(true).unwrap();

    let quiet = (0..120)
        .flat_map(|i| {
            let t = i as f32;
            [0.01 * (0.04 * t).sin(), 0.01 * (0.043 * t).cos()]
        })
        .collect::<Vec<_>>();
    let transient = (0..120)
        .flat_map(|i| {
            let t = i as f32;
            let hit = if i < 16 { 0.55 - i as f32 * 0.02 } else { 0.0 };
            [
                0.18 * (0.71 * t).sin() + hit,
                0.16 * (0.67 * t + 0.3).cos() - hit * 0.7,
            ]
        })
        .collect::<Vec<_>>();

    let quiet_packet = encoder.encode_f32(&quiet, 120).unwrap();
    let transient_packet = encoder.encode_f32(&transient, 120).unwrap();
    assert_ne!(quiet_packet.len(), transient_packet.len());

    let mut decoder = Decoder::new(48_000, 2).unwrap();
    assert_eq!(decoder.decode_f32(&quiet_packet, false).unwrap().len(), 240);
    assert_eq!(
        decoder.decode_f32(&transient_packet, false).unwrap().len(),
        240
    );
}
