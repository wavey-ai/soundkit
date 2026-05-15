use libopus_rs::celt::bands::SPREAD_NORMAL;
use libopus_rs::celt::codec::{decode_spectral_frame, encode_spectral_frame, CeltFrameConfig};
use libopus_rs::celt::modes::CeltMode;
use libopus_rs::celt::synthesis::{celt_synthesis, deemphasis_interleaved};
use libopus_rs::celt::vq::renormalise_vector;

#[test]
fn official_celt_synthesis_outputs_interleaved_pcm() {
    let mode = CeltMode::standard_48k();
    let lm = 3;
    let m = 1 << lm;
    let n = mode.short_mdct_size << lm;
    let mut config = CeltFrameConfig::new(&mode, lm, 2, 144).unwrap();
    config.spread = SPREAD_NORMAL;

    let mut x = (0..n)
        .map(|i| (i as f32 * 0.023).sin() + 0.12 * (i as f32 * 0.067).cos())
        .collect::<Vec<_>>();
    let mut y = (0..n)
        .map(|i| 0.75 * (i as f32 * 0.031).cos() - 0.10 * (i as f32 * 0.043).sin())
        .collect::<Vec<_>>();
    for band in config.start..config.end {
        let band_start = m * mode.ebands[band] as usize;
        let band_end = m * mode.ebands[band + 1] as usize;
        renormalise_vector(&mut x[band_start..band_end], band_end - band_start, 1.0);
        renormalise_vector(&mut y[band_start..band_end], band_end - band_start, 1.0);
    }

    let mut band_e = vec![1.0f32; 2 * mode.nb_ebands];
    for band in 0..mode.nb_ebands {
        band_e[band] = 0.8 + 0.018 * band as f32;
        band_e[mode.nb_ebands + band] = 0.9 + 0.014 * band as f32;
    }

    let mut old_enc = vec![0.0f32; 2 * mode.nb_ebands];
    let mut delayed_intra = 1.0f32;
    let mut seed_enc = 0x1234_5678;
    let encoded = encode_spectral_frame(
        &mode,
        &config,
        &mut x,
        Some(&mut y),
        &band_e,
        &mut old_enc,
        &mut delayed_intra,
        &mut seed_enc,
    )
    .unwrap();

    let mut band_log_e = vec![0.0f32; 2 * mode.nb_ebands];
    let mut seed_dec = 0x1234_5678;
    let decoded = decode_spectral_frame(
        &mode,
        &config,
        &encoded.data,
        &mut band_log_e,
        &mut seed_dec,
    )
    .unwrap();
    let channels = celt_synthesis(
        &mode,
        &decoded.x,
        decoded.y.as_deref(),
        &band_log_e,
        config.start,
        config.end.min(mode.eff_ebands),
        config.channels,
        decoded.is_transient,
        config.lm,
        1,
        decoded.silence,
    )
    .unwrap();
    let mut preemph_mem = vec![0.0f32; config.channels];
    let pcm = deemphasis_interleaved(&mode, &channels, &mut preemph_mem).unwrap();

    assert_eq!(pcm.len(), n * config.channels);
    assert!(pcm.iter().all(|sample| sample.is_finite()));
    assert!(pcm.iter().any(|sample| sample.abs() > 1e-5));
    assert!(preemph_mem.iter().any(|sample| sample.abs() > 1e-5));
}
