use libopus_rs::celt::cwrs::get_pulses;
use libopus_rs::celt::entropy::RangeEncoder;
use libopus_rs::celt::modes::{bits2pulses, pulses2bits, CeltMode};
use libopus_rs::celt::vq::{
    alg_quant_with_scratch, cubic_quant_with_scratch, renormalise_vector, SPREAD_NORMAL,
};
use std::env;
use std::hint::black_box;
use std::time::Instant;

const SOURCE_VECTORS: usize = 256;
const CONFIGS: [(usize, usize, u32); 6] = [
    (4, 13, 8),
    (6, 15, 5),
    (8, 17, 4),
    (12, 18, 2),
    (18, 19, 1),
    (22, 20, 1),
];

#[derive(Clone, Copy)]
struct Lcg(u32);

impl Lcg {
    fn sample(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (self.0 as i32 as f32) * (1.0 / i32::MAX as f32)
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn source_vectors(n: usize) -> Vec<Vec<f32>> {
    let mut rng = Lcg(0x51c0_6eed ^ n as u32);
    let mut vectors = Vec::with_capacity(SOURCE_VECTORS);
    for _ in 0..SOURCE_VECTORS {
        let mut vector = (0..n).map(|_| rng.sample()).collect::<Vec<_>>();
        renormalise_vector(&mut vector, n, 1.0);
        vectors.push(vector);
    }
    vectors
}

fn storage_bytes(vectors: usize, target_bits: usize) -> usize {
    ((vectors * (target_bits + 32) + 7) / 8).max(1024)
}

fn bench_cubic(
    sources: &[Vec<f32>],
    n: usize,
    resolution: u32,
    target_bits: usize,
    vectors: usize,
) -> f64 {
    let mut enc = RangeEncoder::new(storage_bytes(vectors, target_bits));
    let mut scratch = Vec::new();
    let mut x = vec![0.0; n];
    let start = Instant::now();
    for i in 0..vectors {
        x.copy_from_slice(&sources[i % sources.len()]);
        cubic_quant_with_scratch(&mut x, n, resolution, 1, &mut enc, 1.0, true, &mut scratch);
    }
    let elapsed = start.elapsed().as_secs_f64();
    assert_eq!(enc.error(), 0);
    black_box((enc.tell(), x));
    elapsed * 1e9 / vectors as f64
}

fn bench_pvq(sources: &[Vec<f32>], n: usize, k: usize, target_bits: usize, vectors: usize) -> f64 {
    let mut enc = RangeEncoder::new(storage_bytes(vectors, target_bits));
    let mut pulse_y = Vec::new();
    let mut pvq_y = Vec::new();
    let mut signs = Vec::new();
    let mut x = vec![0.0; n];
    let start = Instant::now();
    for i in 0..vectors {
        x.copy_from_slice(&sources[i % sources.len()]);
        alg_quant_with_scratch(
            &mut x,
            n,
            k,
            SPREAD_NORMAL,
            1,
            &mut enc,
            1.0,
            true,
            &mut pulse_y,
            &mut pvq_y,
            &mut signs,
        );
    }
    let elapsed = start.elapsed().as_secs_f64();
    assert_eq!(enc.error(), 0);
    black_box((enc.tell(), x));
    elapsed * 1e9 / vectors as f64
}

fn angular_snr(
    sources: &[Vec<f32>],
    n: usize,
    resolution: u32,
    k: usize,
    target_bits: usize,
) -> (f64, f64) {
    let mut cubic_enc = RangeEncoder::new(storage_bytes(sources.len(), target_bits));
    let mut pvq_enc = RangeEncoder::new(storage_bytes(sources.len(), target_bits));
    let mut cubic_scratch = Vec::new();
    let mut pulse_y = Vec::new();
    let mut pvq_y = Vec::new();
    let mut signs = Vec::new();
    let mut cubic_error = 0.0f64;
    let mut pvq_error = 0.0f64;
    for source in sources {
        let mut cubic = source.clone();
        cubic_quant_with_scratch(
            &mut cubic,
            n,
            resolution,
            1,
            &mut cubic_enc,
            1.0,
            true,
            &mut cubic_scratch,
        );
        let mut pvq = source.clone();
        alg_quant_with_scratch(
            &mut pvq,
            n,
            k,
            SPREAD_NORMAL,
            1,
            &mut pvq_enc,
            1.0,
            true,
            &mut pulse_y,
            &mut pvq_y,
            &mut signs,
        );
        for i in 0..n {
            cubic_error += f64::from((source[i] - cubic[i]).powi(2));
            pvq_error += f64::from((source[i] - pvq[i]).powi(2));
        }
    }
    let signal = sources.len() as f64;
    (
        10.0 * (signal / cubic_error).log10(),
        10.0 * (signal / pvq_error).log10(),
    )
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn main() {
    let vectors = env_usize("VQ_BENCH_VECTORS", 65_536);
    let repeats = env_usize("VQ_BENCH_REPEATS", 7);
    let mode = CeltMode::standard_48k_shared();
    println!(
        "n\tresolution\tcubic_bits\tpvq_bits\tpvq_k\tcubic_ns\tpvq_ns\tcubic_vs_pvq\tcubic_snr_db\tpvq_snr_db"
    );

    for (n, band, resolution) in CONFIGS {
        assert_eq!((mode.ebands[band + 1] - mode.ebands[band]) as usize, n);
        let cubic_bits_frac =
            mode.log_n[band] as i32 + ((1 + (n - 1) * resolution as usize) as i32) * 8;
        let q = bits2pulses(mode, band, 0, cubic_bits_frac);
        let k = get_pulses(q);
        let pvq_bits_frac = pulses2bits(mode, band, 0, q);
        let target_bits = ((cubic_bits_frac + 7) / 8) as usize;
        let sources = source_vectors(n);
        let mut cubic_times = Vec::with_capacity(repeats);
        let mut pvq_times = Vec::with_capacity(repeats);
        for repeat in 0..repeats {
            if repeat & 1 == 0 {
                cubic_times.push(bench_cubic(&sources, n, resolution, target_bits, vectors));
                pvq_times.push(bench_pvq(&sources, n, k, target_bits, vectors));
            } else {
                pvq_times.push(bench_pvq(&sources, n, k, target_bits, vectors));
                cubic_times.push(bench_cubic(&sources, n, resolution, target_bits, vectors));
            }
        }
        let cubic_ns = median(cubic_times);
        let pvq_ns = median(pvq_times);
        let (cubic_snr, pvq_snr) = angular_snr(&sources, n, resolution, k, target_bits);
        println!(
            "{n}\t{resolution}\t{:.3}\t{:.3}\t{k}\t{cubic_ns:.2}\t{pvq_ns:.2}\t{:+.1}%\t{cubic_snr:.2}\t{pvq_snr:.2}",
            cubic_bits_frac as f64 / 8.0,
            pvq_bits_frac as f64 / 8.0,
            100.0 * (cubic_ns - pvq_ns) / pvq_ns,
        );
    }
}
