use libopus_rs::celt::bands::{SPREAD_AGGRESSIVE, SPREAD_NORMAL};
use libopus_rs::celt::codec::{
    decode_alloc_trim, decode_dynalloc_offsets, decode_spread_decision, decode_transient_flag,
    encode_alloc_trim, encode_dynalloc_offsets, encode_spread_decision, encode_transient_flag,
    init_caps, tf_decode, tf_encode,
};
use libopus_rs::celt::entropy::{RangeDecoder, RangeEncoder};
use libopus_rs::celt::modes::CeltMode;

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
