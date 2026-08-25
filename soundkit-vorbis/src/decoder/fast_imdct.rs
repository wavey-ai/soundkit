//! Allocation-free FFT-based Vorbis IMDCT.

use super::header_cached::CachedBlocksizeDerived;

const MAX_FFT_LEN: usize = 2048;

#[derive(Clone, Copy, Default)]
#[repr(C)]
struct Complex {
    re: f32,
    im: f32,
}

impl Complex {
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

#[inline(never)]
pub(super) fn inverse_mdct(tables: &CachedBlocksizeDerived, output: &mut [f32]) {
    let block_len = output.len();
    debug_assert!(block_len.is_power_of_two());
    let spectral_len = block_len >> 1;
    let fft_len = spectral_len >> 1;
    let quarter_fft_len = fft_len >> 1;
    debug_assert!(fft_len <= MAX_FFT_LEN);
    debug_assert_eq!(tables.fast_imdct_twiddle.len(), fft_len);
    debug_assert_eq!(tables.fast_fft_bitrev.len(), fft_len);

    let mut storage = [Complex::default(); MAX_FFT_LEN];
    let scratch = &mut storage[..fft_len];

    for index in 0..fft_len {
        let even = output[index << 1];
        let odd = -output[spectral_len - 1 - (index << 1)];
        let [wr, wi] = tables.fast_imdct_twiddle[index];
        scratch[index] = Complex {
            re: odd * wi - even * wr,
            im: odd * wr + even * wi,
        };
    }

    fft_in_place(scratch, &tables.fast_fft_bitrev, &tables.fast_fft_stages);

    let (vector0, rest) = output.split_at_mut(fft_len);
    let (vector1, rest) = rest.split_at_mut(fft_len);
    let (vector2, vector3) = rest.split_at_mut(fft_len);

    for index in 0..quarter_fft_len {
        let value = post_twiddle(scratch[index], tables.fast_imdct_twiddle[index]);
        let forward = index << 1;
        let reverse = fft_len - 1 - forward;
        vector0[reverse] = -value.im;
        vector1[forward] = value.im;
        vector2[reverse] = value.re;
        vector3[forward] = value.re;
    }

    for index in 0..quarter_fft_len {
        let source_index = quarter_fft_len + index;
        let value = post_twiddle(
            scratch[source_index],
            tables.fast_imdct_twiddle[source_index],
        );
        let forward = index << 1;
        let reverse = fft_len - 1 - forward;
        vector0[forward] = -value.re;
        vector1[reverse] = value.re;
        vector2[forward] = value.im;
        vector3[reverse] = value.im;
    }
}

#[inline(always)]
fn post_twiddle(value: Complex, twiddle: [f32; 2]) -> Complex {
    let [wr, wi] = twiddle;
    Complex {
        re: wr * value.re + wi * value.im,
        im: wi * value.re - wr * value.im,
    }
}

#[inline(never)]
fn fft_in_place(values: &mut [Complex], bitrev: &[u16], stages: &[Vec<[f32; 2]>]) {
    let len = values.len();
    debug_assert_eq!(bitrev.len(), len);
    for (index, &reverse) in bitrev.iter().enumerate() {
        let reverse = usize::from(reverse);
        if index < reverse {
            values.swap(index, reverse);
        }
    }

    for values in values.chunks_exact_mut(4) {
        let even0 = values[0].add(values[1]);
        let even1 = values[0].sub(values[1]);
        let odd0 = values[2].add(values[3]);
        let odd1 = values[2].sub(values[3]);
        let rotated = Complex {
            re: odd1.im,
            im: -odd1.re,
        };
        values[0] = even0.add(odd0);
        values[1] = even1.add(rotated);
        values[2] = even0.sub(odd0);
        values[3] = even1.sub(rotated);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: Runtime feature detection guarantees AVX2 support. Complex
        // and `[f32; 2]` are contiguous pairs of `f32`, and the kernel uses
        // unaligned loads bounded by the four-element loop condition.
        unsafe { merge_stages_avx2(values, stages) };
        return;
    }

    merge_stages_scalar(values, stages);
}

fn merge_stages_scalar(values: &mut [Complex], stages: &[Vec<[f32; 2]>]) {
    let mut span = 8usize;
    for twiddles in stages {
        let half = span >> 1;
        for chunk in values.chunks_exact_mut(span) {
            let (even, odd) = chunk.split_at_mut(half);
            for index in 0..half {
                let [wr, wi] = twiddles[index];
                let source = odd[index];
                let product = Complex {
                    re: source.re * wr - source.im * wi,
                    im: source.re * wi + source.im * wr,
                };
                let base = even[index];
                even[index] = base.add(product);
                odd[index] = base.sub(product);
            }
        }
        span <<= 1;
    }
}

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn merge_stages_avx2(values: &mut [Complex], stages: &[Vec<[f32; 2]>]) {
    let mut span = 8usize;
    for twiddles in stages {
        let half = span >> 1;
        for chunk in values.chunks_exact_mut(span) {
            let (even, odd) = chunk.split_at_mut(half);
            let mut index = 0usize;
            while index + 4 <= half {
                // SAFETY: `index..index + 4` is in bounds for every slice;
                // each element contains exactly two contiguous `f32` values.
                unsafe {
                    let base = _mm256_loadu_ps(even.as_ptr().add(index).cast::<f32>());
                    let source = _mm256_loadu_ps(odd.as_ptr().add(index).cast::<f32>());
                    let twiddle = _mm256_loadu_ps(twiddles.as_ptr().add(index).cast::<f32>());
                    let source_real = _mm256_moveldup_ps(source);
                    let source_imaginary = _mm256_movehdup_ps(source);
                    let twiddle_swapped = _mm256_permute_ps::<0b1011_0001>(twiddle);
                    let product = _mm256_addsub_ps(
                        _mm256_mul_ps(source_real, twiddle),
                        _mm256_mul_ps(source_imaginary, twiddle_swapped),
                    );
                    _mm256_storeu_ps(
                        even.as_mut_ptr().add(index).cast::<f32>(),
                        _mm256_add_ps(base, product),
                    );
                    _mm256_storeu_ps(
                        odd.as_mut_ptr().add(index).cast::<f32>(),
                        _mm256_sub_ps(base, product),
                    );
                }
                index += 4;
            }
            while index < half {
                let [wr, wi] = twiddles[index];
                let source = odd[index];
                let product = Complex {
                    re: source.re * wr - source.im * wi,
                    im: source.re * wi + source.im * wr,
                };
                let base = even[index];
                even[index] = base.add(product);
                odd[index] = base.sub(product);
                index += 1;
            }
        }
        span <<= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::imdct_test::{IMDCT_INPUT_TEST_ARR_1, IMDCT_OUTPUT_TEST_ARR_1};

    #[test]
    fn fft_imdct_matches_vorbis_reference_vector() {
        let mut actual = vec![0.0; IMDCT_OUTPUT_TEST_ARR_1.len()];
        actual[..IMDCT_INPUT_TEST_ARR_1.len()].copy_from_slice(&IMDCT_INPUT_TEST_ARR_1);
        let tables = CachedBlocksizeDerived::from_blocksize(8);
        inverse_mdct(&tables, &mut actual);
        for (index, (&actual, &expected)) in actual
            .iter()
            .zip(IMDCT_OUTPUT_TEST_ARR_1.iter())
            .enumerate()
        {
            assert!(
                (actual - expected).abs() < 0.00005,
                "sample {index}: actual={actual}, expected={expected}"
            );
        }
    }
}
