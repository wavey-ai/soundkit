use soundkit_opus::celt::bands::{SPREAD_AGGRESSIVE, SPREAD_NORMAL};
use soundkit_opus::celt::codec::{
    decode_alloc_trim, decode_dynalloc_offsets, decode_spectral_frame, decode_spread_decision,
    decode_transient_flag, encode_alloc_trim, encode_dynalloc_offsets, encode_spectral_frame,
    encode_spread_decision, encode_transient_flag, init_caps, tf_decode, tf_encode,
    CeltFrameConfig,
};
use soundkit_opus::celt::entropy::{RangeDecoder, RangeEncoder};
use soundkit_opus::celt::modes::CeltMode;
use soundkit_opus::celt::vq::renormalise_vector;

#[test]
fn official_init_caps_matches_celt_formula() {
    let mode = CeltMode::standard_48k();
    let lm = 3;
    let channels = 2;
    let cap = init_caps(&mode, lm, channels);

    for band in [0usize, 5, 12, mode.nb_ebands - 1] {
        let n = (mode.ebands[band + 1] as i32 - mode.ebands[band] as i32) << lm;
        let idx = mode.nb_ebands * (2 * lm + channels - 1) + band;
        let expected = ((mode.cache.caps[idx] as i32 + 64) * channels as i32 * n) >> 2;
        assert_eq!(cap[band], expected);
    }
    assert!(cap.iter().all(|value| *value > 0));
}

#[test]
fn official_tf_encode_decode_round_trips_mapped_resolutions() {
    let mode = CeltMode::standard_48k();
    for lm in 0..=mode.max_lm {
        for is_transient in [false, true] {
            let start = 0;
            let end = mode.nb_ebands;
            let mut encoded_tf = (0..mode.nb_ebands)
                .map(|i| i32::from((i + lm + usize::from(is_transient)) % 3 == 0))
                .collect::<Vec<_>>();

            let mut enc = RangeEncoder::new(64);
            tf_encode(start, end, is_transient, &mut encoded_tf, lm, 1, &mut enc);
            enc.finish();
            assert_eq!(enc.error(), 0);

            let mut decoded_tf = vec![0i32; mode.nb_ebands];
            let mut dec = RangeDecoder::new(enc.range_data());
            tf_decode(start, end, is_transient, &mut decoded_tf, lm, &mut dec);

            assert_eq!(decoded_tf, encoded_tf, "lm={lm}, transient={is_transient}");
        }
    }
}

#[test]
fn official_celt_control_symbols_round_trip() {
    let mode = CeltMode::standard_48k();
    let lm = 3;
    let channels = 2;
    let packet_bytes = 96;
    let total_bits = (packet_bytes * 8) as i32;
    let total_bits_frac = total_bits << 3;
    let start = 0;
    let end = mode.nb_ebands;
    let cap = init_caps(&mode, lm, channels);

    let mut enc = RangeEncoder::new(packet_bytes);
    let encoded_transient = encode_transient_flag(lm, total_bits, true, &mut enc);
    let mut encoded_tf = (0..mode.nb_ebands)
        .map(|i| i32::from(i % 4 == 1))
        .collect::<Vec<_>>();
    tf_encode(
        start,
        end,
        encoded_transient,
        &mut encoded_tf,
        lm,
        1,
        &mut enc,
    );
    let encoded_spread = encode_spread_decision(SPREAD_AGGRESSIVE, total_bits, &mut enc);

    let mut encoded_offsets = vec![0i32; mode.nb_ebands];
    encoded_offsets[3] = 1;
    encoded_offsets[8] = 2;
    encoded_offsets[14] = 1;
    let encoded_boost = encode_dynalloc_offsets(
        &mode,
        start,
        end,
        &mut encoded_offsets,
        &cap,
        total_bits_frac,
        channels,
        lm,
        &mut enc,
    );
    let encoded_trim = encode_alloc_trim(7, total_bits_frac, encoded_boost, &mut enc);
    enc.finish();
    assert_eq!(enc.error(), 0);

    let mut dec = RangeDecoder::new(enc.range_data());
    let decoded_transient = decode_transient_flag(lm, total_bits, &mut dec);
    let mut decoded_tf = vec![0i32; mode.nb_ebands];
    tf_decode(start, end, decoded_transient, &mut decoded_tf, lm, &mut dec);
    let decoded_spread = decode_spread_decision(total_bits, &mut dec);
    let mut decoded_offsets = vec![0i32; mode.nb_ebands];
    let decoded_boost = decode_dynalloc_offsets(
        &mode,
        start,
        end,
        &mut decoded_offsets,
        &cap,
        total_bits_frac,
        channels,
        lm,
        &mut dec,
    );
    let decoded_trim = decode_alloc_trim(total_bits_frac - decoded_boost, &mut dec);

    assert!(encoded_transient);
    assert_eq!(decoded_transient, encoded_transient);
    assert_eq!(decoded_tf, encoded_tf);
    assert_eq!(decoded_spread, encoded_spread);
    assert_ne!(decoded_spread, SPREAD_NORMAL);
    assert_eq!(decoded_offsets, encoded_offsets);
    assert_eq!(decoded_boost, encoded_boost);
    assert_eq!(decoded_trim, encoded_trim);
}

#[test]
fn official_celt_spectral_frame_round_trips_stereo_bands() {
    let mode = CeltMode::standard_48k();
    let lm = 3;
    let m = 1 << lm;
    let n = mode.short_mdct_size << lm;
    let mut config = CeltFrameConfig::new(&mode, lm, 2, 144).unwrap();
    config.spread = SPREAD_NORMAL;
    config.alloc_trim = 5;

    let mut x_enc = (0..n)
        .map(|i| (i as f32 * 0.019).sin() + 0.16 * (i as f32 * 0.071).cos())
        .collect::<Vec<_>>();
    let mut y_enc = (0..n)
        .map(|i| 0.8 * (i as f32 * 0.027).cos() - 0.11 * (i as f32 * 0.053).sin())
        .collect::<Vec<_>>();
    for band in config.start..config.end {
        let band_start = m * mode.ebands[band] as usize;
        let band_end = m * mode.ebands[band + 1] as usize;
        renormalise_vector(&mut x_enc[band_start..band_end], band_end - band_start, 1.0);
        renormalise_vector(&mut y_enc[band_start..band_end], band_end - band_start, 1.0);
    }

    let mut band_e = vec![1.0f32; 2 * mode.nb_ebands];
    for band in 0..mode.nb_ebands {
        band_e[band] = 0.75 + 0.02 * band as f32;
        band_e[mode.nb_ebands + band] = 0.92 + 0.015 * band as f32;
    }

    let mut old_enc = vec![0.0f32; 2 * mode.nb_ebands];
    let mut energy_error = vec![0.0f32; 2 * mode.nb_ebands];
    let mut delayed_intra = 1.0f32;
    let mut seed_enc = 0x7654_3210;
    let encoded = encode_spectral_frame(
        &mode,
        &config,
        &mut x_enc,
        Some(&mut y_enc),
        &band_e,
        &mut old_enc,
        &mut energy_error,
        &mut delayed_intra,
        &mut seed_enc,
    )
    .unwrap();
    assert_eq!(encoded.data.len(), config.packet_bytes);
    assert_eq!(encoded.spread, config.spread);
    assert!(encoded.allocation.coded_bands > 0);

    let mut old_dec = vec![0.0f32; 2 * mode.nb_ebands];
    let mut seed_dec = 0x7654_3210;
    let decoded =
        decode_spectral_frame(&mode, &config, &encoded.data, &mut old_dec, &mut seed_dec).unwrap();

    assert_eq!(decoded.allocation, encoded.allocation);
    assert_eq!(decoded.tf_res, encoded.tf_res);
    let coded_mask_len = encoded.allocation.coded_bands * config.channels;
    assert_eq!(
        &decoded.collapse_masks[..coded_mask_len],
        &encoded.collapse_masks[..coded_mask_len]
    );
    assert_eq!(decoded.silence, encoded.silence);
    assert_eq!(decoded.is_transient, encoded.is_transient);
    assert_eq!(decoded.spread, encoded.spread);
    assert_eq!(decoded.alloc_trim, encoded.alloc_trim);

    for (i, (decoded, encoded)) in old_dec.iter().zip(old_enc.iter()).enumerate() {
        assert!(
            (decoded - encoded).abs() < 1e-6,
            "energy band={i}, decoded={decoded}, encoded={encoded}"
        );
    }

    let y_dec = decoded.y.as_ref().expect("stereo decode");
    let coded_bound = m * mode.ebands[config.end] as usize;
    let decoded_energy = decoded.x[..coded_bound]
        .iter()
        .chain(y_dec[..coded_bound].iter())
        .map(|v| v * v)
        .sum::<f32>();
    assert!(decoded_energy.is_finite());
    assert!(decoded_energy > 0.0);
}
