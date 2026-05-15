use libopus_rs::celt::kiss_fft::{opus_fft, opus_ifft, KissFftCpx, KissFftState};

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

fn check(in_data: &[KissFftCpx], out: &[KissFftCpx], nfft: usize, inverse: bool) -> f64 {
    let mut errpow = 0.0;
    let mut sigpow = 0.0;

    for bin in 0..nfft {
        let mut ansr = 0.0;
        let mut ansi = 0.0;
        for (k, sample) in in_data.iter().enumerate().take(nfft) {
            let phase = -2.0 * core::f64::consts::PI * bin as f64 * k as f64 / nfft as f64;
            let re = phase.cos();
            let mut im = phase.sin();
            if inverse {
                im = -im;
            }

            let (re, im) = if inverse {
                (re, im)
            } else {
                (re / nfft as f64, im / nfft as f64)
            };

            ansr += sample.r as f64 * re - sample.i as f64 * im;
            ansi += sample.r as f64 * im + sample.i as f64 * re;
        }
        let difr = ansr - out[bin].r as f64;
        let difi = ansi - out[bin].i as f64;
        errpow += difr * difr + difi * difi;
        sigpow += ansr * ansr + ansi * ansi;
    }

    10.0 * (sigpow / errpow).log10()
}

fn test_1d(nfft: usize, inverse: bool) {
    let cfg = KissFftState::new(nfft).expect("supported FFT size");
    let mut rng = CRand::new(nfft as u32 ^ if inverse { 0x7777 } else { 0x3333 });
    let mut input = vec![KissFftCpx::default(); nfft];
    let mut output = vec![KissFftCpx::default(); nfft];

    for sample in input.iter_mut().take(nfft) {
        sample.r = ((rng.rand() % 32767) - 16_384) as f32;
        sample.i = ((rng.rand() % 32767) - 16_384) as f32;
        sample.r *= 32768.0;
        sample.i *= 32768.0;
    }

    if inverse {
        for sample in input.iter_mut().take(nfft) {
            sample.r /= nfft as f32;
            sample.i /= nfft as f32;
        }
        opus_ifft(&cfg, &input, &mut output);
    } else {
        opus_fft(&cfg, &input, &mut output);
    }

    let snr = check(&input, &output, nfft, inverse);
    assert!(snr > 60.0, "nfft={nfft}, inverse={inverse}, snr={snr}");
}

#[test]
fn official_dft_unit_sizes_pass_forward_and_inverse() {
    for nfft in [32, 128, 256, 36, 50, 60, 120, 240, 480] {
        test_1d(nfft, false);
        test_1d(nfft, true);
    }
}
