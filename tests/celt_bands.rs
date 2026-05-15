use libopus_rs::celt::bands::*;

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
