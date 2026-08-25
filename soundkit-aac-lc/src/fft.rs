//! Fixed-size forward complex FFT tuned for AAC synthesis block sizes.
//!
//! The decoder only needs power-of-two forward transforms between 64 and 512
//! points. Runtime planners such as rustfft carry dispatch overhead that is
//! significant on WebAssembly where SIMD backends are unavailable, so this
//! module implements a compact breadth-first radix-2 Cooley-Tukey transform
//! with precomputed twiddles and an unrolled butterfly inner loop.

use std::f32::consts::PI;

#[cfg(not(any(target_arch = "wasm32", target_arch = "wasm64")))]
use std::sync::Arc;

#[cfg(not(any(target_arch = "wasm32", target_arch = "wasm64")))]
use rustfft::{num_complex::Complex32, Fft, FftPlanner};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Complex {
    pub re: f32,
    pub im: f32,
}

impl Complex {
    #[inline]
    pub const fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    #[inline]
    pub fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }

    #[inline]
    pub fn scale(self, factor: f32) -> Self {
        Self::new(self.re * factor, self.im * factor)
    }
}

#[inline(always)]
fn mul(a: Complex, b: Complex) -> Complex {
    Complex::new(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)
}

impl std::ops::Mul for Complex {
    type Output = Complex;

    #[inline]
    fn mul(self, rhs: Complex) -> Complex {
        mul(self, rhs)
    }
}

/// In-order-input, in-order-output forward FFT for power-of-two lengths.
#[derive(Clone)]
pub struct ForwardFft {
    n: usize,
    #[cfg(not(any(target_arch = "wasm32", target_arch = "wasm64")))]
    native: Arc<dyn Fft<f32>>,
    /// Binary digit-reversal permutation applied before the butterfly stages.
    #[cfg(any(target_arch = "wasm32", target_arch = "wasm64"))]
    permute: Vec<u16>,
    /// Twiddle rows indexed by stage: stage s combines half-runs of length
    /// 2^s and needs 2^s roots of unity of order 2^(s+1).
    #[cfg(any(target_arch = "wasm32", target_arch = "wasm64"))]
    stage_twiddles: Vec<Vec<Complex>>,
}

impl std::fmt::Debug for ForwardFft {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ForwardFft")
            .field("n", &self.n)
            .finish_non_exhaustive()
    }
}

impl ForwardFft {
    pub fn new(n: usize) -> Self {
        assert!(
            n != 0 && n.is_power_of_two(),
            "FFT length must be a power of two"
        );

        #[cfg(not(any(target_arch = "wasm32", target_arch = "wasm64")))]
        {
            let mut planner = FftPlanner::<f32>::new();
            return Self {
                n,
                native: planner.plan_fft_forward(n),
            };
        }

        #[cfg(any(target_arch = "wasm32", target_arch = "wasm64"))]
        {
            let shift = u16::try_from(n)
                .expect("FFT length must fit in u16")
                .leading_zeros()
                .saturating_add(1)
                .min(15);
            let permute = (0..n as u16)
                .map(|index| index.reverse_bits() >> shift)
                .collect();

            let mut stage_twiddles = Vec::with_capacity(n.trailing_zeros() as usize);
            let mut width = 2usize;
            while width <= n {
                let row: Vec<Complex> = (0..width / 2)
                    .map(|k| {
                        let angle = -2.0 * PI * k as f32 / width as f32;
                        Complex::new(angle.cos(), angle.sin())
                    })
                    .collect();
                stage_twiddles.push(row);
                width *= 2;
            }

            Self {
                n,
                permute,
                stage_twiddles,
            }
        }
    }

    pub const fn len(&self) -> usize {
        self.n
    }

    pub fn process_inplace(&self, data: &mut [Complex]) {
        debug_assert_eq!(data.len(), self.n);
        #[cfg(not(any(target_arch = "wasm32", target_arch = "wasm64")))]
        {
            self.native.process(native_complex_slice_mut(data));
            return;
        }

        #[cfg(any(target_arch = "wasm32", target_arch = "wasm64"))]
        self.process_custom_inplace(data);
    }

    pub fn process_inplace_with_scratch(&self, data: &mut [Complex], scratch: &mut [Complex]) {
        debug_assert_eq!(data.len(), self.n);
        debug_assert!(scratch.len() >= self.scratch_len());
        #[cfg(not(any(target_arch = "wasm32", target_arch = "wasm64")))]
        {
            self.native.process_with_scratch(
                native_complex_slice_mut(data),
                native_complex_slice_mut(scratch),
            );
        }

        #[cfg(any(target_arch = "wasm32", target_arch = "wasm64"))]
        {
            let _ = scratch;
            self.process_custom_inplace(data);
        }
    }

    pub fn scratch_len(&self) -> usize {
        #[cfg(not(any(target_arch = "wasm32", target_arch = "wasm64")))]
        {
            self.native.get_inplace_scratch_len()
        }

        #[cfg(any(target_arch = "wasm32", target_arch = "wasm64"))]
        {
            0
        }
    }

    #[cfg(any(target_arch = "wasm32", target_arch = "wasm64"))]
    fn process_custom_inplace(&self, data: &mut [Complex]) {
        if self.n == 1 {
            return;
        }

        for (index, target) in self.permute.iter().copied().enumerate() {
            let target = target as usize;
            if index < target {
                data.swap(index, target);
            }
        }

        if self.n == 2 {
            let (lo, hi) = data.split_at_mut(1);
            apply_butterfly(lo, hi, 0, self.stage_twiddles[0][0]);
            return;
        }

        process_fft4_blocks(data, self.stage_twiddles[0][0], &self.stage_twiddles[1]);

        let mut half = 4usize;
        while half < self.n {
            let width = half * 2;
            let twiddles = &self.stage_twiddles[half.trailing_zeros() as usize];
            for chunk in data.chunks_exact_mut(width) {
                let (lo, hi) = chunk.split_at_mut(half);
                merge_halves(lo, hi, twiddles);
            }
            half *= 2;
        }
    }
}

#[cfg(not(any(target_arch = "wasm32", target_arch = "wasm64")))]
#[inline]
fn native_complex_slice_mut(values: &mut [Complex]) -> &mut [Complex32] {
    debug_assert_eq!(
        std::mem::size_of::<Complex>(),
        std::mem::size_of::<Complex32>()
    );
    debug_assert_eq!(
        std::mem::align_of::<Complex>(),
        std::mem::align_of::<Complex32>()
    );
    unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast(), values.len()) }
}

#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
#[inline]
fn process_fft4_blocks(data: &mut [Complex], base_twiddle: Complex, twiddles: &[Complex]) {
    debug_assert_eq!(twiddles.len(), 2);
    for chunk in data.chunks_exact_mut(4) {
        let a = chunk[0];
        let b = chunk[1];
        let c = chunk[2];
        let d = chunk[3];

        let qb = mul(b, base_twiddle);
        let qd = mul(d, base_twiddle);
        let u0 = Complex::new(a.re + qb.re, a.im + qb.im);
        let u1 = Complex::new(a.re - qb.re, a.im - qb.im);
        let v0 = Complex::new(c.re + qd.re, c.im + qd.im);
        let v1 = Complex::new(c.re - qd.re, c.im - qd.im);
        let q0 = mul(v0, twiddles[0]);
        let q1 = mul(v1, twiddles[1]);

        chunk[0] = Complex::new(u0.re + q0.re, u0.im + q0.im);
        chunk[1] = Complex::new(u1.re + q1.re, u1.im + q1.im);
        chunk[2] = Complex::new(u0.re - q0.re, u0.im - q0.im);
        chunk[3] = Complex::new(u1.re - q1.re, u1.im - q1.im);
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
fn process_fft4_blocks(data: &mut [Complex], base_twiddle: Complex, twiddles: &[Complex]) {
    use core::arch::wasm32::{
        f32x4, f32x4_add, f32x4_mul, f32x4_sub, i32x4, i32x4_shuffle, v128, v128_load, v128_store,
        v128_xor,
    };

    debug_assert_eq!(twiddles.len(), 2);
    let twiddle = unsafe { v128_load(twiddles.as_ptr().cast::<v128>()) };
    let twiddle_im_re = i32x4_shuffle::<1, 0, 3, 2>(twiddle, twiddle);
    let sign_mask = i32x4(i32::MIN, 0, i32::MIN, 0);
    let base = f32x4(
        base_twiddle.re,
        base_twiddle.im,
        base_twiddle.re,
        base_twiddle.im,
    );
    let base_im_re = i32x4_shuffle::<1, 0, 3, 2>(base, base);

    for chunk in data.chunks_exact_mut(4) {
        unsafe {
            let ab = v128_load(chunk.as_ptr().cast::<v128>());
            let cd = v128_load(chunk.as_ptr().add(2).cast::<v128>());
            let even = i32x4_shuffle::<0, 1, 4, 5>(ab, cd);
            let odd = i32x4_shuffle::<2, 3, 6, 7>(ab, cd);
            let odd_re = i32x4_shuffle::<0, 0, 2, 2>(odd, odd);
            let odd_im = i32x4_shuffle::<1, 1, 3, 3>(odd, odd);
            let base_signed_im = v128_xor(f32x4_mul(odd_im, base_im_re), sign_mask);
            let base_product = f32x4_add(f32x4_mul(odd_re, base), base_signed_im);
            let stage1_sum = f32x4_add(even, base_product);
            let stage1_diff = f32x4_sub(even, base_product);
            let u = i32x4_shuffle::<0, 1, 4, 5>(stage1_sum, stage1_diff);
            let v = i32x4_shuffle::<2, 3, 6, 7>(stage1_sum, stage1_diff);

            let v_re = i32x4_shuffle::<0, 0, 2, 2>(v, v);
            let v_im = i32x4_shuffle::<1, 1, 3, 3>(v, v);
            let signed_im = v128_xor(f32x4_mul(v_im, twiddle_im_re), sign_mask);
            let product = f32x4_add(f32x4_mul(v_re, twiddle), signed_im);

            v128_store(chunk.as_mut_ptr().cast::<v128>(), f32x4_add(u, product));
            v128_store(
                chunk.as_mut_ptr().add(2).cast::<v128>(),
                f32x4_sub(u, product),
            );
        }
    }
}

#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
#[inline]
fn merge_halves(lo: &mut [Complex], hi: &mut [Complex], twiddles: &[Complex]) {
    let half = lo.len();
    let mut index = 0;
    while index + 2 <= half {
        apply_butterfly_pair(lo, hi, index, twiddles[index], twiddles[index + 1]);
        index += 2;
    }
    if index < half {
        apply_butterfly(lo, hi, index, twiddles[index]);
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
fn merge_halves(lo: &mut [Complex], hi: &mut [Complex], twiddles: &[Complex]) {
    use core::arch::wasm32::{
        f32x4_add, f32x4_mul, f32x4_sub, i32x4, i32x4_shuffle, v128, v128_load, v128_store,
        v128_xor,
    };

    debug_assert_eq!(lo.len(), hi.len());
    debug_assert!(twiddles.len() >= lo.len());
    debug_assert_eq!(
        std::mem::size_of::<Complex>(),
        2 * std::mem::size_of::<f32>()
    );

    let mut index = 0;
    while index + 2 <= lo.len() {
        // Two adjacent complex butterflies occupy one v128. Complex is repr(C)
        // with exactly two f32 fields, and Wasm permits unaligned vector loads.
        unsafe {
            let even = v128_load(lo.as_ptr().add(index).cast::<v128>());
            let odd = v128_load(hi.as_ptr().add(index).cast::<v128>());
            let twiddle = v128_load(twiddles.as_ptr().add(index).cast::<v128>());

            let odd_re = i32x4_shuffle::<0, 0, 2, 2>(odd, odd);
            let odd_im = i32x4_shuffle::<1, 1, 3, 3>(odd, odd);
            let twiddle_im_re = i32x4_shuffle::<1, 0, 3, 2>(twiddle, twiddle);
            let signed_im = v128_xor(
                f32x4_mul(odd_im, twiddle_im_re),
                i32x4(i32::MIN, 0, i32::MIN, 0),
            );
            let product = f32x4_add(f32x4_mul(odd_re, twiddle), signed_im);

            v128_store(
                lo.as_mut_ptr().add(index).cast::<v128>(),
                f32x4_add(even, product),
            );
            v128_store(
                hi.as_mut_ptr().add(index).cast::<v128>(),
                f32x4_sub(even, product),
            );
        }
        index += 2;
    }

    if index < lo.len() {
        apply_butterfly(lo, hi, index, twiddles[index]);
    }
}

#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
#[inline(always)]
fn apply_butterfly_pair(
    lo: &mut [Complex],
    hi: &mut [Complex],
    index: usize,
    w0: Complex,
    w1: Complex,
) {
    let p0 = lo[index];
    let q0 = mul(hi[index], w0);
    lo[index] = Complex::new(p0.re + q0.re, p0.im + q0.im);
    hi[index] = Complex::new(p0.re - q0.re, p0.im - q0.im);

    let p1 = lo[index + 1];
    let q1 = mul(hi[index + 1], w1);
    lo[index + 1] = Complex::new(p1.re + q1.re, p1.im + q1.im);
    hi[index + 1] = Complex::new(p1.re - q1.re, p1.im - q1.im);
}

#[inline(always)]
fn apply_butterfly(lo: &mut [Complex], hi: &mut [Complex], index: usize, w: Complex) {
    let p = lo[index];
    let q = mul(hi[index], w);
    lo[index] = Complex::new(p.re + q.re, p.im + q.im);
    hi[index] = Complex::new(p.re - q.re, p.im - q.im);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_dft(input: &[Complex]) -> Vec<Complex> {
        let n = input.len();
        (0..n)
            .map(|k| {
                let mut sum = Complex::default();
                for (t, value) in input.iter().enumerate() {
                    let angle = -2.0 * std::f64::consts::PI * k as f64 * t as f64 / n as f64;
                    sum.re += value.re * angle.cos() as f32 - value.im * angle.sin() as f32;
                    sum.im += value.re * angle.sin() as f32 + value.im * angle.cos() as f32;
                }
                sum
            })
            .collect()
    }

    fn pseudo_random(seed: u64, index: usize) -> f32 {
        let value = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(index as u64)
            .wrapping_mul(1442695040888963407);
        ((value >> 33) as i64 as f32 / (i64::MAX as f32)) * 2.0
    }

    #[test]
    fn matches_naive_dft_across_sizes() {
        for exponent in 1..=10u32 {
            let n = 1usize << exponent;
            let input: Vec<Complex> = (0..n)
                .map(|index| {
                    Complex::new(
                        pseudo_random(0x1234_5678, index),
                        pseudo_random(0xDEAD_BEEF, index),
                    )
                })
                .collect();

            let expected = naive_dft(&input);
            let mut actual = input.clone();
            let fft = ForwardFft::new(n);
            fft.process_inplace(&mut actual);

            let scale = n as f32;
            for (got, want) in actual.iter().zip(expected.iter()) {
                assert!(
                    (got.re - want.re).abs() / scale < 1e-3,
                    "n={n} real mismatch: {got:?} vs {want:?}"
                );
                assert!(
                    (got.im - want.im).abs() / scale < 1e-3,
                    "n={n} imag mismatch: {got:?} vs {want:?}"
                );
            }
        }
    }

    #[test]
    fn handles_single_point_transform() {
        let mut data = [Complex::new(3.5, -1.25)];
        let fft = ForwardFft::new(1);
        fft.process_inplace(&mut data[..]);
        assert_eq!(data[0], Complex::new(3.5, -1.25));
    }

    #[test]
    fn known_two_point_output() {
        let mut data = [Complex::new(1.0, 0.0), Complex::new(0.0, 1.0)];
        let fft = ForwardFft::new(2);
        fft.process_inplace(&mut data);
        assert_eq!(data[0], Complex::new(1.0, 1.0));
        assert_eq!(data[1], Complex::new(1.0, -1.0));
    }
}
