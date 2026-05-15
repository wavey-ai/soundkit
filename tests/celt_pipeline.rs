use libopus_rs::celt::bands::{compute_band_energies, normalise_bands};
use libopus_rs::celt::codec::{decode_spectral_frame, encode_spectral_frame, CeltFrameConfig};
use libopus_rs::celt::mdct::clt_mdct_forward;
use libopus_rs::celt::modes::CeltMode;
use libopus_rs::celt::quant_bands::amp2_log2;
use libopus_rs::celt::synthesis::{celt_synthesis_with_overlap, deemphasis_interleaved};
use libopus_rs::{Application, Decoder, Encoder};

fn correlation(a: &[f32], b: &[f32]) -> f32 {
    let mut aa = 0.0f64;
    let mut bb = 0.0f64;
    let mut ab = 0.0f64;
    for (&x, &y) in a.iter().zip(b) {
        aa += f64::from(x) * f64::from(x);
        bb += f64::from(y) * f64::from(y);
        ab += f64::from(x) * f64::from(y);
    }
    (ab / (aa * bb).sqrt()) as f32
}

fn best_lag_correlation(a: &[f32], b: &[f32], max_lag: isize) -> (isize, f32) {
    let mut best = (0, 0.0f32);
    for lag in -max_lag..=max_lag {
        let (a_start, b_start, len) = if lag >= 0 {
            (lag as usize, 0, a.len().saturating_sub(lag as usize))
        } else {
            (0, (-lag) as usize, b.len().saturating_sub((-lag) as usize))
        };
        if len < 16 {
            continue;
        }
        let corr = correlation(&a[a_start..a_start + len], &b[b_start..b_start + len]);
        if corr.abs() > best.1.abs() {
            best = (lag, corr);
        }
    }
    best
}

#[test]
fn mdct_band_round_trip_preserves_audio_shape() {
    let mode = CeltMode::standard_48k();
    let channels = 2;
    let lm = 3;
    let m = 1usize << lm;
    let n = mode.short_mdct_size << lm;
    let overlap = mode.overlap;
    let shift = mode.max_lm - lm;
    let frames = 8;

    let mut analysis_overlap = vec![vec![0.0f32; overlap]; channels];
    let mut analysis_preemph = vec![0.0f32; channels];
    let mut synthesis_overlap = vec![vec![0.0f32; overlap]; channels];
    let mut synthesis_preemph = vec![0.0f32; channels];
    let mut original = Vec::with_capacity(frames * n * channels);
    let mut decoded = Vec::with_capacity(frames * n * channels);

    for frame in 0..frames {
        let mut pcm = vec![0.0f32; n * channels];
        for i in 0..n {
            let t = (frame * n + i) as f32;
            pcm[i * channels] = 0.18 * (0.017 * t).sin() + 0.04 * (0.071 * t).cos();
            pcm[i * channels + 1] = 0.16 * (0.019 * t + 0.4).sin() - 0.03 * (0.059 * t).cos();
        }
        original.extend_from_slice(&pcm);

        let mut inputs = Vec::with_capacity(channels);
        for c in 0..channels {
            let mut input = vec![0.0f32; 2 * n];
            input[..overlap].copy_from_slice(&analysis_overlap[c]);
            for i in 0..n {
                let sample = pcm[i * channels + c];
                input[overlap + i] = sample - analysis_preemph[c];
                analysis_preemph[c] = mode.preemph[0] * sample;
            }
            analysis_overlap[c].copy_from_slice(&input[n..n + overlap]);
            inputs.push(input);
        }

        let mut freq = vec![0.0f32; channels * n];
        for c in 0..channels {
            clt_mdct_forward(
                &mode.mdct,
                &inputs[c],
                &mut freq[c * n..(c + 1) * n],
                &mode.window,
                overlap,
                shift,
                1,
            );
        }

        let mut band_e = vec![0.0f32; channels * mode.nb_ebands];
        compute_band_energies(&mode, &freq, &mut band_e, mode.eff_ebands, channels, lm);
        let mut norm = vec![0.0f32; channels * n];
        normalise_bands(
            &mode,
            &freq,
            &mut norm,
            &band_e,
            mode.eff_ebands,
            channels,
            m,
        );
        let mut band_log_e = vec![0.0f32; channels * mode.nb_ebands];
        amp2_log2(
            &mode,
            mode.eff_ebands,
            mode.nb_ebands,
            &band_e,
            &mut band_log_e,
            channels,
        );

        let (left, right) = norm.split_at(n);
        let channels_out = celt_synthesis_with_overlap(
            &mode,
            left,
            Some(right),
            &band_log_e,
            0,
            mode.eff_ebands,
            channels,
            false,
            lm,
            1,
            false,
            &mut synthesis_overlap,
        )
        .unwrap();
        decoded.extend_from_slice(
            &deemphasis_interleaved(&mode, &channels_out, &mut synthesis_preemph).unwrap(),
        );
    }

    let skip = 2 * n * channels;
    let delay = overlap * channels;
    let corr = correlation(
        &original[skip..original.len() - delay],
        &decoded[skip + delay..],
    );
    let (lag, _) = best_lag_correlation(&original[skip..], &decoded[skip..], 512);
    assert_eq!(lag, -(delay as isize));
    assert!(corr > 0.98);
}

#[test]
fn public_celt_round_trip_preserves_tone_shape() {
    let channels = 2;
    let frame_size = 960;
    let frames = 8;
    let mut encoder = Encoder::new(48_000, channels, Application::Audio).unwrap();
    let mut decoder = Decoder::new(48_000, channels).unwrap();
    let mut original = Vec::with_capacity(frames * frame_size * channels);
    let mut decoded = Vec::with_capacity(frames * frame_size * channels);

    for frame in 0..frames {
        let mut pcm = vec![0.0f32; frame_size * channels];
        for i in 0..frame_size {
            let t = (frame * frame_size + i) as f32;
            pcm[i * channels] = 0.18 * (0.017 * t).sin() + 0.04 * (0.071 * t).cos();
            pcm[i * channels + 1] = 0.16 * (0.019 * t + 0.4).sin() - 0.03 * (0.059 * t).cos();
        }
        let packet = encoder.encode_f32(&pcm, frame_size).unwrap();
        let out = decoder.decode_f32(&packet, false).unwrap();
        original.extend_from_slice(&pcm);
        decoded.extend_from_slice(&out);
    }

    let skip = 2 * frame_size * channels;
    let delay = CeltMode::standard_48k().overlap * channels;
    let corr = correlation(
        &original[skip..original.len() - delay],
        &decoded[skip + delay..],
    );
    let (lag, _) = best_lag_correlation(&original[skip..], &decoded[skip..], 512);
    assert_eq!(lag, -(delay as isize));
    assert!(corr > 0.80);
}

#[test]
fn spectral_quantizer_preserves_tone_frame_shape() {
    let mode = CeltMode::standard_48k();
    let channels = 2;
    let lm = 3;
    let m = 1usize << lm;
    let n = mode.short_mdct_size << lm;
    let overlap = mode.overlap;
    let shift = mode.max_lm - lm;
    let mut pcm = vec![0.0f32; n * channels];
    for i in 0..n {
        let t = i as f32;
        pcm[i * channels] = 0.18 * (0.017 * t).sin() + 0.04 * (0.071 * t).cos();
        pcm[i * channels + 1] = 0.16 * (0.019 * t + 0.4).sin() - 0.03 * (0.059 * t).cos();
    }

    let mut freq = vec![0.0f32; channels * n];
    for c in 0..channels {
        let mut input = vec![0.0f32; 2 * n];
        for i in 0..n {
            let sample = pcm[i * channels + c];
            let prev = if i == 0 {
                0.0
            } else {
                mode.preemph[0] * pcm[(i - 1) * channels + c]
            };
            input[overlap + i] = sample - prev;
        }
        clt_mdct_forward(
            &mode.mdct,
            &input,
            &mut freq[c * n..(c + 1) * n],
            &mode.window,
            overlap,
            shift,
            1,
        );
    }

    let mut band_e = vec![0.0f32; channels * mode.nb_ebands];
    compute_band_energies(&mode, &freq, &mut band_e, mode.eff_ebands, channels, lm);
    let mut norm = vec![0.0f32; channels * n];
    normalise_bands(
        &mode,
        &freq,
        &mut norm,
        &band_e,
        mode.eff_ebands,
        channels,
        m,
    );
    let original_norm = norm.clone();

    let mut config = CeltFrameConfig::new(&mode, lm, channels, 240).unwrap();
    config.alloc_trim = 5;
    let mut old_enc = vec![0.0f32; channels * mode.nb_ebands];
    let mut seed_enc = 0;
    let (left, right) = norm.split_at_mut(n);
    let encoded = encode_spectral_frame(
        &mode,
        &config,
        left,
        Some(right),
        &band_e,
        &mut old_enc,
        &mut seed_enc,
    )
    .unwrap();

    let mut old_dec = vec![0.0f32; channels * mode.nb_ebands];
    let mut seed_dec = 0;
    let decoded =
        decode_spectral_frame(&mode, &config, &encoded.data, &mut old_dec, &mut seed_dec).unwrap();
    let decoded_y = decoded.y.as_ref().unwrap();
    let quant_corr = correlation(&original_norm[..n], &decoded.x);
    let quant_corr_r = correlation(&original_norm[n..], decoded_y);
    assert!(encoded.allocation.pulses.iter().sum::<i32>() > 8_000);
    assert!(quant_corr > 0.80 && quant_corr_r > 0.80);
}
