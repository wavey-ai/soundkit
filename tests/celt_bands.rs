use libopus_rs::celt::bands::*;
use libopus_rs::celt::entropy::{RangeDecoder, RangeEncoder};
use libopus_rs::celt::modes::CeltMode;
use libopus_rs::celt::quant_bands::{amp2_log2, E_MEANS};
use libopus_rs::celt::rate::clt_compute_allocation;
use libopus_rs::celt::vq::renormalise_vector;

#[test]
fn official_hysteresis_decision_edges() {
    let thresholds = [10.0, 20.0, 30.0];
    let hysteresis = [1.0, 2.0, 3.0];
    assert_eq!(hysteresis_decision(9.0, &thresholds, &hysteresis, 3, 0), 0);
    assert_eq!(hysteresis_decision(21.0, &thresholds, &hysteresis, 3, 1), 1);
    assert_eq!(hysteresis_decision(23.0, &thresholds, &hysteresis, 3, 1), 2);
    assert_eq!(hysteresis_decision(17.0, &thresholds, &hysteresis, 3, 2), 1);
}

#[test]
fn official_lcg_sequence_matches_bands_c() {
    let mut seed = 1u32;
    let expected = [1_015_568_748, 1_586_005_467, 2_165_703_038, 3_027_450_565];
    for value in expected {
        seed = celt_lcg_rand(seed);
        assert_eq!(seed, value);
    }
}

#[test]
fn official_hadamard_interleave_round_trips() {
    for stride in [2, 4, 8, 16] {
        for hadamard in [false, true] {
            let n0 = 5;
            let mut x = (0..n0 * stride)
                .map(|i| i as f32 - 17.0)
                .collect::<Vec<_>>();
            let original = x.clone();
            deinterleave_hadamard(&mut x, n0, stride, hadamard);
            interleave_hadamard(&mut x, n0, stride, hadamard);
            assert_eq!(x, original, "stride={stride}, hadamard={hadamard}");
        }
    }
}

#[test]
fn official_haar1_is_self_inverse_with_float_tolerance() {
    let mut x = (0..64).map(|i| i as f32 * 0.25 - 3.0).collect::<Vec<_>>();
    let original = x.clone();
    haar1(&mut x, 16, 4);
    haar1(&mut x, 16, 4);
    for (a, b) in x.iter().zip(original.iter()) {
        assert!((*a - *b).abs() < 1e-5);
    }
}

#[test]
fn official_stereo_split_preserves_energy() {
    let mut x = [0.2, -0.4, 0.6, -0.8, 1.0];
    let mut y = [-0.3, 0.5, -0.7, 0.9, -1.1];
    let before = x.iter().chain(y.iter()).map(|v| v * v).sum::<f32>();
    stereo_split(&mut x, &mut y, 5);
    let after = x.iter().chain(y.iter()).map(|v| v * v).sum::<f32>();
    assert!((before - after).abs() < 1e-6);
}

#[test]
fn official_theta_helpers_return_valid_ranges() {
    let x = [0.4, -0.2, 0.1, 0.9];
    let y = [-0.1, 0.3, 0.8, -0.5];
    let itheta = stereo_itheta(&x, &y, true, 4);
    assert!((0..=16_384).contains(&itheta));

    let (qn, delta) = theta_metrics(4, 80, 18, false, 8192);
    assert!((1..=256).contains(&qn));
    assert!(delta.abs() < 20_000);
}

#[test]
fn official_band_energy_normalise_and_denormalise_round_trip() {
    let mode = CeltMode::standard_48k();
    let channels = 1;
    let lm = 3;
    let m = 1 << lm;
    let n = mode.short_mdct_size << lm;
    let end = mode.nb_ebands;
    let freq = (0..n)
        .map(|i| ((i as f32 * 0.071).sin() + 0.35 * (i as f32 * 0.013).cos()) * 0.75)
        .collect::<Vec<_>>();

    let mut band_e = vec![0.0f32; channels * mode.nb_ebands];
    compute_band_energies(&mode, &freq, &mut band_e, end, channels, lm);

    let mut norm = vec![0.0f32; freq.len()];
    normalise_bands(&mode, &freq, &mut norm, &band_e, end, channels, m);
    for i in 0..end {
        let start = m * mode.ebands[i] as usize;
        let band_end = m * mode.ebands[i + 1] as usize;
        let energy = norm[start..band_end].iter().map(|v| v * v).sum::<f32>();
        assert!((energy - 1.0).abs() < 2e-5, "band={i}, energy={energy}");
    }

    let mut band_log_e = vec![0.0f32; mode.nb_ebands];
    amp2_log2(&mode, end, end, &band_e, &mut band_log_e, channels);
    let mut denorm = vec![0.0f32; n];
    denormalise_bands(&mode, &norm, &mut denorm, &band_log_e, 0, end, m, 1, false);

    let coded_bound = m * mode.ebands[end] as usize;
    for (i, (got, expected)) in denorm[..coded_bound].iter().zip(freq.iter()).enumerate() {
        assert!((got - expected).abs() < 2e-5, "bin={i}");
    }
    assert!(denorm[coded_bound..].iter().all(|v| *v == 0.0));
}

#[test]
fn official_denormalise_respects_downsample_and_silence_bounds() {
    let mode = CeltMode::standard_48k();
    let m = 1;
    let n = mode.short_mdct_size;
    let x = vec![1.0f32; n];
    let band_log_e = (0..mode.nb_ebands)
        .map(|i| 1.0 - E_MEANS[i])
        .collect::<Vec<_>>();
    let mut freq = vec![99.0f32; n];
    denormalise_bands(
        &mode,
        &x,
        &mut freq,
        &band_log_e,
        0,
        mode.nb_ebands,
        m,
        2,
        false,
    );
    assert!(freq[..n / 2].iter().any(|v| *v != 0.0));
    assert!(freq[n / 2..].iter().all(|v| *v == 0.0));

    denormalise_bands(
        &mode,
        &x,
        &mut freq,
        &band_log_e,
        0,
        mode.nb_ebands,
        m,
        1,
        true,
    );
    assert!(freq.iter().all(|v| *v == 0.0));
}

#[test]
fn official_spreading_decision_updates_hf_state() {
    let mode = CeltMode::standard_48k();
    let m = 8;
    let n = mode.short_mdct_size << 3;
    let x = (0..n)
        .map(|i| {
            if i % 7 == 0 {
                0.75
            } else {
                0.015 * (i % 5) as f32
            }
        })
        .collect::<Vec<_>>();
    let weights = vec![1; mode.nb_ebands];
    let mut average = 0;
    let mut hf_average = 0;
    let mut tapset_decision = 1;
    let decision = spreading_decision(
        &mode,
        &x,
        &mut average,
        SPREAD_NORMAL,
        &mut hf_average,
        &mut tapset_decision,
        true,
        mode.nb_ebands,
        1,
        m,
        &weights,
    );
    assert!((SPREAD_NONE..=SPREAD_AGGRESSIVE).contains(&decision));
    assert!((0..=2).contains(&tapset_decision));
}

#[test]
fn official_anti_collapse_fills_missing_short_blocks() {
    let mode = CeltMode::standard_48k();
    let lm = 3;
    let size = mode.short_mdct_size << lm;
    let channels = 1;
    let mut x = vec![0.0f32; size];
    let collapse_masks = vec![0u8; mode.nb_ebands * channels];
    let log_e = vec![2.0f32; channels * mode.nb_ebands];
    let prev1 = vec![-8.0f32; channels * mode.nb_ebands];
    let prev2 = vec![-9.0f32; channels * mode.nb_ebands];
    let pulses = vec![0i32; mode.nb_ebands];

    let seed = anti_collapse(
        &mode,
        &mut x,
        &collapse_masks,
        lm,
        channels,
        size,
        8,
        12,
        &log_e,
        &prev1,
        &prev2,
        &pulses,
        1,
        false,
    );

    let start = (mode.ebands[8] as usize) << lm;
    let end = (mode.ebands[12] as usize) << lm;
    assert_ne!(seed, 1);
    assert!(x[start..end].iter().any(|v| *v != 0.0));
}

#[test]
fn official_mono_quant_all_bands_range_round_trip() {
    let mode = CeltMode::standard_48k();
    let start = 0;
    let end = mode.nb_ebands;
    let lm = 0;
    let m = 1 << lm;
    let n = mode.short_mdct_size << lm;
    let total_bits = 3600;
    let channels = 1;
    let offsets = vec![0; mode.nb_ebands];
    let cap = (0..mode.nb_ebands)
        .map(|band| {
            mode.cache.caps[lm * 2 * mode.nb_ebands + (channels - 1) * mode.nb_ebands + band] as i32
        })
        .collect::<Vec<_>>();
    let allocation = clt_compute_allocation(
        &mode,
        start,
        end,
        &offsets,
        &cap,
        5,
        0,
        false,
        total_bits,
        channels,
        lm,
        None,
        mode.nb_ebands,
        mode.nb_ebands - 1,
    );

    let mut x_enc = (0..n)
        .map(|i| ((i as f32 * 0.117).sin() + 0.25 * (i as f32 * 0.031).cos()).max(-0.95))
        .collect::<Vec<_>>();
    for band in start..end {
        let band_start = m * mode.ebands[band] as usize;
        let band_end = m * mode.ebands[band + 1] as usize;
        renormalise_vector(&mut x_enc[band_start..band_end], band_end - band_start, 1.0);
    }
    let mut x_dec = vec![0.0f32; n];
    let band_e = vec![1.0f32; mode.nb_ebands];
    let tf_res = vec![0i32; mode.nb_ebands];

    let mut enc = RangeEncoder::new(512);
    let mut enc_coder = BandCoder::Encode(&mut enc);
    let mut enc_masks = vec![0u8; mode.nb_ebands];
    let mut seed_enc = 12_345u32;
    quant_all_bands_mono(
        &mode,
        start,
        end,
        &mut x_enc,
        &mut enc_masks,
        &band_e,
        &allocation.pulses,
        false,
        SPREAD_NORMAL,
        mode.nb_ebands,
        &tf_res,
        total_bits,
        allocation.balance,
        &mut enc_coder,
        lm,
        allocation.coded_bands,
        &mut seed_enc,
        0,
        true,
    );
    enc.shrink(((enc.tell() + 7) / 8) as usize);
    enc.finish();
    assert_eq!(enc.error(), 0);

    let mut dec = RangeDecoder::new(enc.range_data());
    let mut dec_coder = BandCoder::Decode(&mut dec);
    let mut dec_masks = vec![0u8; mode.nb_ebands];
    let mut seed_dec = 12_345u32;
    quant_all_bands_mono(
        &mode,
        start,
        end,
        &mut x_dec,
        &mut dec_masks,
        &band_e,
        &allocation.pulses,
        false,
        SPREAD_NORMAL,
        mode.nb_ebands,
        &tf_res,
        total_bits,
        allocation.balance,
        &mut dec_coder,
        lm,
        allocation.coded_bands,
        &mut seed_dec,
        0,
        false,
    );

    assert_eq!(dec_masks, enc_masks);
    assert_eq!(seed_dec, seed_enc);
    let coded_bound = m * mode.ebands[end] as usize;
    for (i, (decoded, encoded)) in x_dec[..coded_bound].iter().zip(x_enc.iter()).enumerate() {
        assert!(
            (decoded - encoded).abs() < 2e-5,
            "bin={i}, decoded={decoded}, encoded={encoded}"
        );
    }
}
