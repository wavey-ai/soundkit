use libopus_rs::celt::entropy::{RangeDecoder, RangeEncoder};
use libopus_rs::celt::vq::{
    alg_quant, alg_unquant, op_pvq_search, renormalise_vector, SPREAD_NORMAL,
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
