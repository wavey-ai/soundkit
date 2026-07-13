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

fn centered_u16(value: u32) -> f32 {
    ((value & 0xffff) as i32 - 32_768) as f32 * (1.0 / 32_768.0)
}

fn triangle_wave(phase: u32) -> f32 {
    let p = (phase & 0xffff) as i32;
    let v = if p < 32_768 { p - 16_384 } else { 49_152 - p };
    v as f32 * (1.0 / 16_384.0)
}

fn raw_celt_bench_frame(frame_size: usize, frame_index: usize) -> Vec<f32> {
    let mut pcm = Vec::with_capacity(frame_size * 2);
    let start = frame_index * frame_size;
    let mut noise = 0x1234_5678u32;
    for _ in 0..start {
        noise = noise.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    }
    for i in start..start + frame_size {
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
        pcm.push((0.25 * tri_a + 0.125 * tri_b + n + transient).clamp(-1.0, 1.0));
        pcm.push((0.21875 * tri_c - 0.09375 * tri_d - n - 0.5 * transient).clamp(-1.0, 1.0));
    }
    pcm
}

fn vbr_stats(frame_size: usize, bitrate: i32, frames: usize) -> (usize, usize, usize) {
    let mut encoder = Encoder::new(48_000, 2, Application::RestrictedLowDelay).unwrap();
    encoder.set_bitrate(bitrate).unwrap();
    encoder.set_vbr(true).unwrap();

    let mut bytes = 0usize;
    let mut min_packet = usize::MAX;
    let mut max_packet = 0usize;
    for frame in 0..frames {
        let packet = encoder
            .encode_f32(&raw_celt_bench_frame(frame_size, frame), frame_size)
            .unwrap();
        bytes += packet.len();
        min_packet = min_packet.min(packet.len());
        max_packet = max_packet.max(packet.len());
    }
    (bytes, min_packet, max_packet)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
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

    let mut decoder = Decoder::new(48_000, 2).unwrap();
    let mut decoded_f32_into = Vec::new();
    let decoded_f32_samples = decoder
        .decode_f32_into(&packet, false, &mut decoded_f32_into)
        .unwrap();
    assert_eq!(decoded_f32_samples, 960);
    assert_eq!(decoded_f32_into, decoded);

    let mut decoder = Decoder::new(48_000, 2).unwrap();
    let decoded_i16 = decoder.decode_i16(&packet, false).unwrap();
    let mut decoder = Decoder::new(48_000, 2).unwrap();
    let mut decoded_into = Vec::new();
    let decoded_samples = decoder
        .decode_i16_into(&packet, false, &mut decoded_into)
        .unwrap();
    assert_eq!(decoded_samples, 960);
    assert_eq!(decoded_into, decoded_i16);
}

#[test]
fn celt_encoder_carries_final_range_rng_between_frames() {
    let mut encoder = Encoder::new(48_000, 2, Application::RestrictedLowDelay).unwrap();
    encoder.set_bitrate(128_000).unwrap();
    encoder.set_vbr(false).unwrap();

    let mut packet = Vec::new();
    for frame in 0..=91 {
        packet = encoder
            .encode_f32(&raw_celt_bench_frame(120, frame), 120)
            .unwrap();
    }

    assert_eq!(
        hex(&packet),
        "e4be0dd79fb8ecc723b754a861007abfb47dfb6c6d44417e7cbe7dae671022b8e681556640de34de"
    );
}

#[test]
fn celt_encoder_counts_decay_limited_coarse_energy_badness_like_c() {
    let mut encoder = Encoder::new(48_000, 2, Application::RestrictedLowDelay).unwrap();
    encoder.set_bitrate(128_000).unwrap();
    encoder.set_vbr(false).unwrap();

    let mut packet = Vec::new();
    for frame in 0..=227 {
        packet = encoder
            .encode_f32(&raw_celt_bench_frame(120, frame), 120)
            .unwrap();
    }

    assert_eq!(
        hex(&packet),
        "e4e927f194cee4aa8f6c33e989b902b9c491db3a10a5daca8197018a6fe2aac58518294d3a2c1351"
    );
}

#[test]
fn encode_i16_matches_equivalent_f32_input() {
    let pcm_i16 = (0..960)
        .flat_map(|i| {
            let t = i as f32;
            [
                (0.18 * (0.017 * t).sin() * 32767.0).round() as i16,
                (0.16 * (0.019 * t + 0.4).sin() * 32767.0).round() as i16,
            ]
        })
        .collect::<Vec<_>>();
    let pcm_f32 = pcm_i16
        .iter()
        .map(|sample| *sample as f32 / 32768.0)
        .collect::<Vec<_>>();

    let mut i16_encoder = Encoder::new(48_000, 2, Application::Audio).unwrap();
    let mut f32_encoder = Encoder::new(48_000, 2, Application::Audio).unwrap();
    i16_encoder.set_bitrate(128_000).unwrap();
    f32_encoder.set_bitrate(128_000).unwrap();

    assert_eq!(
        i16_encoder.encode_i16(&pcm_i16, 960).unwrap(),
        f32_encoder.encode_f32(&pcm_f32, 960).unwrap()
    );
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

#[test]
fn celt_vbr_tracks_constrained_reservoir_over_raw_fixture() {
    assert_eq!(vbr_stats(120, 128_000, 400), (16_439, 34, 59));
    assert_eq!(vbr_stats(960, 128_000, 50), (16_370, 320, 500));
}
