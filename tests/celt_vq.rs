use libopus_rs::celt::entropy::{RangeDecoder, RangeEncoder};
use libopus_rs::celt::vq::{
    alg_quant, alg_unquant, cubic_quant, cubic_unquant, op_pvq_search, renormalise_vector,
    SPREAD_NORMAL,
};

#[derive(Clone)]
struct CRand(u32);

impl CRand {
    fn new(seed: u32) -> Self {
        Self(seed)
    }

    fn rand(&mut self) -> i32 {
        self.0 = self.0.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        ((self.0 >> 16) & 0x7fff) as i32
    }
}

#[test]
fn official_pvq_search_allocates_exact_pulse_count() {
    let mut rng = CRand::new(0x7057);
    for n in [4, 8, 15, 32, 50] {
        for k in [1, 3, 5, 9, 17] {
            if 2 * k >= n {
                continue;
            }
            let mut x = vec![0.0f32; n];
            for sample in &mut x {
                *sample = (rng.rand() % 32767 - 16_384) as f32 / 32768.0;
            }
            renormalise_vector(&mut x, n, 1.0);

            let mut iy = vec![0i32; n + 3];
            let yy = op_pvq_search(&mut x, &mut iy, k, n);
            let pulse_sum: usize = iy.iter().take(n).map(|v| v.unsigned_abs() as usize).sum();
            assert_eq!(pulse_sum, k, "N={n}, K={k}");
            assert_eq!(
                yy,
                iy.iter().take(n).map(|v| (v * v) as f32).sum::<f32>(),
                "N={n}, K={k}"
            );
        }
    }
}

#[test]
fn official_alg_quant_unquant_round_trip_resynth() {
    let mut rng = CRand::new(0x1234_5678);
    for (n, k, b) in [(8, 3, 1), (15, 3, 1), (32, 5, 2), (50, 3, 5), (80, 5, 5)] {
        let mut x = vec![0.0f32; n];
        for sample in &mut x {
            *sample = (rng.rand() % 32767 - 16_384) as f32 / 32768.0;
        }
        renormalise_vector(&mut x, n, 1.0);

        let mut enc = RangeEncoder::new(1024);
        let mut encoded_resynth = x.clone();
        let cm_enc = alg_quant(
            &mut encoded_resynth,
            n,
            k,
            SPREAD_NORMAL,
            b,
            &mut enc,
            1.0,
            true,
        );
        enc.shrink(((enc.tell() + 7) / 8) as usize);
        enc.finish();
        assert_eq!(enc.error(), 0);

        let mut dec = RangeDecoder::new(enc.range_data());
        let mut decoded = vec![0.0f32; n];
        let cm_dec = alg_unquant(&mut decoded, n, k, SPREAD_NORMAL, b, &mut dec, 1.0);
        assert_eq!(cm_dec, cm_enc, "N={n}, K={k}");

        for i in 0..n {
            assert!(
                (decoded[i] - encoded_resynth[i]).abs() < 2e-6,
                "N={n}, K={k}, i={i}, decoded={}, encoded={}",
                decoded[i],
                encoded_resynth[i]
            );
        }
    }
}

#[test]
fn qext_cubic_quant_unquant_matches_encoder_resynthesis() {
    let mut rng = CRand::new(0xc0b1_c123);
    for (n, resolution, blocks, gain) in [
        (2, 2, 1, 1.0),
        (8, 4, 1, 1.0),
        (15, 6, 2, 0.75),
        (32, 10, 2, 1.0),
        (50, 14, 4, 0.5),
    ] {
        let mut input = vec![0.0f32; n];
        for value in &mut input {
            *value = (rng.rand() - 16_384) as f32 / 32_768.0;
        }
        renormalise_vector(&mut input, n, 1.0);

        let mut enc = RangeEncoder::new(2048);
        let mut encoded_resynthesis = input.clone();
        let encode_mask = cubic_quant(
            &mut encoded_resynthesis,
            n,
            resolution,
            blocks,
            &mut enc,
            gain,
            true,
        );
        enc.shrink(((enc.tell() + 7) / 8) as usize);
        enc.finish();
        assert_eq!(enc.error(), 0);

        let mut dec = RangeDecoder::new(enc.range_data());
        let mut decoded = vec![0.0f32; n];
        let decode_mask = cubic_unquant(&mut decoded, n, resolution, blocks, &mut dec, gain);
        assert_eq!(decode_mask, encode_mask);
        assert_eq!(dec.error(), 0);

        for i in 0..n {
            assert!(
                (decoded[i] - encoded_resynthesis[i]).abs() < 2e-6,
                "N={n}, resolution={resolution}, i={i}, decoded={}, encoded={}",
                decoded[i],
                encoded_resynthesis[i]
            );
        }
        let energy = decoded.iter().map(|value| value * value).sum::<f32>();
        assert!((energy - gain * gain).abs() < 2e-5);
    }
}

#[test]
fn qext_cubic_zero_resolution_codes_no_shape() {
    let mut input = [0.25f32, -0.5, 0.75, -1.0];
    let mut enc = RangeEncoder::new(8);
    let mask = cubic_quant(&mut input, 4, 0, 1, &mut enc, 1.0, true);
    assert_eq!(mask, 0);
    assert_eq!(input, [0.0; 4]);
}

#[test]
fn qext_cubic_matches_trunk_float_golden_vector() {
    let mut input = [0.12, -0.27, 0.43, -0.91, 0.35, 0.08, -0.52, 0.61];
    let expected = [
        0.123_403_504,
        -0.205_672_52,
        0.287_941_52,
        -0.658_152_04,
        0.287_941_52,
        0.041_134_503,
        -0.370_210_53,
        0.452_479_54,
    ];
    let mut enc = RangeEncoder::new(64);
    assert_eq!(cubic_quant(&mut input, 8, 4, 1, &mut enc, 1.0, true), 1);
    assert_eq!(enc.tell(), 33);
    enc.shrink(5);
    enc.finish();

    assert_eq!(&enc.range_data()[..5], &[0x60, 0x1a, 0x71, 0x76, 0xb3]);
    for (actual, expected) in input.into_iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-7);
    }
}
