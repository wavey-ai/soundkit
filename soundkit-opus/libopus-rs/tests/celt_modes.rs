use libopus_rs::celt::modes::{bits2pulses, pulses2bits, CeltMode};

#[test]
fn official_48k_960_mode_shape_matches_static_mode() {
    let mode = CeltMode::standard_48k();
    assert_eq!(mode.fs, 48_000);
    assert_eq!(mode.overlap, 120);
    assert_eq!(mode.nb_ebands, 21);
    assert_eq!(mode.eff_ebands, 21);
    assert_eq!(mode.max_lm, 3);
    assert_eq!(mode.nb_short_mdcts, 8);
    assert_eq!(mode.short_mdct_size, 120);
    assert_eq!(
        mode.ebands,
        vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100]
    );
    assert_eq!(
        mode.log_n,
        vec![0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 16, 16, 16, 21, 21, 24, 29, 34, 36]
    );
    assert_eq!(mode.cache.size, 392);
    assert_eq!(mode.cache.index.len(), 105);
    assert_eq!(mode.cache.bits.len(), 392);
    assert_eq!(mode.cache.caps.len(), 168);
    assert!((mode.window[0] - 6.7286966e-05).abs() < 1e-10);
    assert!((mode.window[119] - 1.0).abs() < 1e-7);
}

#[test]
fn official_mode_pulse_cache_matches_static_prefixes() {
    let mode = CeltMode::standard_48k();
    assert_eq!(
        &mode.cache.index[..21],
        &[-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 41, 41, 41, 82, 82, 123, 164, 200, 222]
    );
    assert_eq!(
        &mode.cache.bits[..15],
        &[40, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
    );
    assert_eq!(
        &mode.cache.caps[..21],
        &[
            224, 224, 224, 224, 224, 224, 224, 224, 160, 160, 160, 160, 185, 185, 185, 178, 178,
            168, 134, 61, 37
        ]
    );
}

#[test]
fn official_rate_cache_bits_and_pulses_round_trip() {
    let mode = CeltMode::standard_48k();
    for band in 8..mode.nb_ebands {
        for lm in 0..=mode.max_lm {
            let cache_offset = mode.cache.index[(lm + 1) * mode.nb_ebands + band] as usize;
            let max_pseudo = mode.cache.bits[cache_offset] as usize;
            for pseudo in 1..=max_pseudo.min(8) {
                let bits = pulses2bits(&mode, band, lm, pseudo);
                let got = bits2pulses(&mode, band, lm, bits);
                assert_eq!(
                    got, pseudo,
                    "band={band}, LM={lm}, pseudo={pseudo}, bits={bits}"
                );
            }
        }
    }
}
