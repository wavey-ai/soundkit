use libopus_rs::celt::mdct::{clt_mdct_backward, clt_mdct_forward, MdctLookup};

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

fn check_forward(input: &[f32], output: &[f32], nfft: usize) -> f64 {
    let mut errpow = 0.0;
    let mut sigpow = 0.0;

    for bin in 0..nfft / 2 {
        let mut ansr = 0.0;
        for (k, sample) in input.iter().enumerate().take(nfft) {
            let phase = 2.0
                * core::f64::consts::PI
                * (k as f64 + 0.5 + 0.25 * nfft as f64)
                * (bin as f64 + 0.5)
                / nfft as f64;
            ansr += *sample as f64 * phase.cos() / (nfft as f64 / 4.0);
        }
        let difr = ansr - output[bin] as f64;
        errpow += difr * difr;
        sigpow += ansr * ansr;
    }

    10.0 * (sigpow / errpow).log10()
}

fn check_backward(input: &[f32], output: &[f32], nfft: usize) -> f64 {
    let mut errpow = 0.0;
    let mut sigpow = 0.0;

    for bin in 0..nfft {
        let mut ansr = 0.0;
        for (k, sample) in input.iter().enumerate().take(nfft / 2) {
            let phase = 2.0
                * core::f64::consts::PI
                * (bin as f64 + 0.5 + 0.25 * nfft as f64)
                * (k as f64 + 0.5)
                / nfft as f64;
            ansr += *sample as f64 * phase.cos();
        }
        let difr = ansr - output[bin] as f64;
        errpow += difr * difr;
        sigpow += ansr * ansr;
    }

    10.0 * (sigpow / errpow).log10()
}

fn test_1d(nfft: usize, inverse: bool) {
    let cfg = MdctLookup::new(nfft, 0).expect("supported MDCT size");
    let mut rng = CRand::new(nfft as u32 ^ if inverse { 0xa5a5 } else { 0x5a5a });
    let mut input = vec![0.0f32; nfft];
    let input_len = if inverse { nfft / 2 } else { nfft };

    for sample in input.iter_mut().take(input_len) {
        *sample = ((rng.rand() % 32768) - 16_384) as f32;
        *sample *= 32768.0;
    }

    if inverse {
        for sample in input.iter_mut().take(input_len) {
            *sample /= nfft as f32;
        }
    }

    let window = vec![1.0f32; nfft / 2];
    let mut output = vec![0.0f32; nfft];

    if inverse {
        clt_mdct_backward(&cfg, &input, &mut output, &window, nfft / 2, 0, 1);
        for k in 0..nfft / 4 {
            output[nfft - k - 1] = output[nfft / 2 + k];
        }
        let snr = check_backward(&input, &output, nfft);
        assert!(snr > 60.0, "nfft={nfft}, inverse=true, snr={snr}");
    } else {
        clt_mdct_forward(&cfg, &input, &mut output, &window, nfft / 2, 0, 1);
        let snr = check_forward(&input, &output, nfft);
        assert!(snr > 60.0, "nfft={nfft}, inverse=false, snr={snr}");
    }
}

#[test]
fn official_mdct_unit_sizes_pass_forward_and_inverse() {
    for nfft in [
        32, 256, 512, 1024, 2048, 36, 40, 60, 120, 240, 480, 960, 1920,
    ] {
        test_1d(nfft, false);
        test_1d(nfft, true);
    }
}
