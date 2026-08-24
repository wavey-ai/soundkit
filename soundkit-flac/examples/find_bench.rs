use soundkit_flac::rice::find_partitioned_rice_parameter;
use std::time::Instant;

fn main() {
    let mut state = 0x12345678u32;
    let mut block = vec![0i32; 4096];
    for x in block.iter_mut() {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        *x = ((state >> 16) as i32) - 32768;
    }
    let rounds = 200_000usize;
    let t = Instant::now();
    let mut sink = 0usize;
    for _ in 0..rounds {
        let p = find_partitioned_rice_parameter(&block, 12, 30);
        sink += p.code_bits;
    }
    println!(
        "sink {sink}, median-per-call {:.2} us",
        t.elapsed().as_secs_f64() * 1e6 / rounds as f64
    );
}
