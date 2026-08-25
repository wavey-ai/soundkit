use soundkit_opus::celt::vq::{exp_rotation, SPREAD_NORMAL};

const MAX_SIZE: usize = 100;

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

fn test_rotation(n: usize, k: usize) {
    let mut rng = CRand::new((n as u32) << 8 | k as u32);
    let mut x0 = vec![0.0f32; MAX_SIZE];
    let mut x1 = vec![0.0f32; MAX_SIZE];

    for i in 0..n {
        x0[i] = (rng.rand() % 32767 - 16_384) as f32;
        x1[i] = x0[i];
    }

    exp_rotation(&mut x1, n, 1, 1, k, SPREAD_NORMAL);
    let mut err = 0.0;
    let mut ener = 0.0;
    for i in 0..n {
        err += (x0[i] as f64 - x1[i] as f64) * (x0[i] as f64 - x1[i] as f64);
        ener += x0[i] as f64 * x0[i] as f64;
    }
    let snr0 = 20.0 * (ener / err).log10();

    exp_rotation(&mut x1, n, -1, 1, k, SPREAD_NORMAL);
    err = 0.0;
    ener = 0.0;
    for i in 0..n {
        err += (x0[i] as f64 - x1[i] as f64) * (x0[i] as f64 - x1[i] as f64);
        ener += x0[i] as f64 * x0[i] as f64;
    }
    let snr = 20.0 * (ener / err).log10();

    assert!(snr >= 60.0, "N={n}, K={k}, inverse SNR={snr}");
    assert!(snr0 <= 20.0, "N={n}, K={k}, forward-only SNR={snr0}");
}

#[test]
fn official_rotation_unit_cases_pass() {
    test_rotation(15, 3);
    test_rotation(23, 5);
    test_rotation(50, 3);
    test_rotation(80, 1);
}
