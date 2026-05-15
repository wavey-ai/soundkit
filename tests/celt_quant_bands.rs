use libopus_rs::celt::entropy::{RangeDecoder, RangeEncoder};
use libopus_rs::celt::modes::CeltMode;
use libopus_rs::celt::quant_bands::{
    amp2_log2, quant_coarse_energy, quant_energy_finalise, quant_fine_energy,
    unquant_coarse_energy, unquant_energy_finalise, unquant_fine_energy, E_MEANS,
};

fn finish_encoder(mut enc: RangeEncoder) -> Vec<u8> {
    enc.shrink(((enc.tell() + 7) / 8) as usize);
    enc.finish();
    assert_eq!(enc.error(), 0);
    enc.range_data().to_vec()
}

#[test]
fn official_amp2log2_matches_float_path_expectations() {
    let mode = CeltMode::standard_48k();
    let channels = 2;
    let mut band_e = vec![0.0f32; channels * mode.nb_ebands];
    for c in 0..channels {
        for i in 0..mode.nb_ebands {
            band_e[i + c * mode.nb_ebands] = 2.0f32.powf(E_MEANS[i] + i as f32 * 0.125 + c as f32);
        }
    }

    let mut band_log_e = vec![0.0f32; band_e.len()];
    amp2_log2(
        &mode,
        18,
        mode.nb_ebands,
        &band_e,
        &mut band_log_e,
        channels,
    );

    for c in 0..channels {
        for i in 0..18 {
            let expected = i as f32 * 0.125 + c as f32;
            assert!(
                (band_log_e[i + c * mode.nb_ebands] - expected).abs() < 2e-6,
                "channel={c}, band={i}"
            );
        }
        for i in 18..mode.nb_ebands {
            assert_eq!(band_log_e[i + c * mode.nb_ebands], -14.0);
        }
    }
}

#[test]
fn official_coarse_energy_round_trips_intra_symbols() {
    let mode = CeltMode::standard_48k();
    let channels = 2;
    let start = 0;
    let end = mode.nb_ebands;
    let lm = 3;
    let len = channels * mode.nb_ebands;
    let mut e_bands = vec![0.0f32; len];
    for c in 0..channels {
        for i in start..end {
            e_bands[i + c * mode.nb_ebands] = -6.0 + i as f32 * 0.45 + c as f32 * 0.75;
        }
    }
    let initial_old = vec![-28.0f32; len];
    let mut enc_old = initial_old.clone();
    let mut error = vec![0.0f32; len];
    let mut delayed_intra = 0.0;
    let packet_bytes = 96;
    let budget = (packet_bytes * 8) as u32;

    let mut enc = RangeEncoder::new(packet_bytes);
    quant_coarse_energy(
        &mode,
        start,
        end,
        end,
        &e_bands,
        &mut enc_old,
        budget,
        &mut error,
        &mut enc,
        channels,
        lm,
        80,
        true,
        &mut delayed_intra,
        false,
        0,
        false,
    );
    enc.finish();
    assert_eq!(enc.error(), 0);
    let data = enc.range_data().to_vec();

    let mut dec = RangeDecoder::new(&data);
    let intra = dec.tell() + 3 <= budget as i32 && dec.decode_bit_logp(3);
    assert!(intra);
    let mut dec_old = initial_old;
    unquant_coarse_energy(
        &mode,
        start,
        end,
        &mut dec_old,
        intra,
        &mut dec,
        channels,
        lm,
    );

    for (i, (decoded, encoded)) in dec_old.iter().zip(enc_old.iter()).enumerate() {
        assert!((decoded - encoded).abs() < 1e-6, "energy {i}");
    }
}

#[test]
fn official_fine_energy_and_finalise_round_trip() {
    let mode = CeltMode::standard_48k();
    let channels = 2;
    let start = 0;
    let end = 12;
    let len = channels * mode.nb_ebands;
    let initial_old = vec![-2.25f32; len];
    let mut enc_old = initial_old.clone();
    let mut error = vec![0.0f32; len];
    for c in 0..channels {
        for i in start..end {
            error[i + c * mode.nb_ebands] = -0.45 + 0.073 * i as f32 + 0.11 * c as f32;
        }
    }
    let fine_quant = (0..mode.nb_ebands)
        .map(|i| if i < end { (i % 4 + 1) as i32 } else { 0 })
        .collect::<Vec<_>>();
    let fine_priority = (0..mode.nb_ebands)
        .map(|i| if i % 3 == 0 { 0 } else { 1 })
        .collect::<Vec<_>>();
    let bits_left = 16;

    let mut enc = RangeEncoder::new(128);
    quant_fine_energy(
        &mode,
        start,
        end,
        &mut enc_old,
        &mut error,
        &fine_quant,
        &mut enc,
        channels,
    );
    quant_energy_finalise(
        &mode,
        start,
        end,
        &mut enc_old,
        &mut error,
        &fine_quant,
        &fine_priority,
        bits_left,
        &mut enc,
        channels,
    );
    let data = finish_encoder(enc);

    let mut dec = RangeDecoder::new(&data);
    let mut dec_old = initial_old;
    unquant_fine_energy(
        &mode,
        start,
        end,
        &mut dec_old,
        &fine_quant,
        &mut dec,
        channels,
    );
    unquant_energy_finalise(
        &mode,
        start,
        end,
        &mut dec_old,
        &fine_quant,
        &fine_priority,
        bits_left,
        &mut dec,
        channels,
    );

    for (i, (decoded, encoded)) in dec_old.iter().zip(enc_old.iter()).enumerate() {
        assert!((decoded - encoded).abs() < 1e-6, "fine energy {i}");
    }
}
