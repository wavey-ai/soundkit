#![deny(unsafe_op_in_unsafe_fn)]

/// A single-precision complex value with real and imaginary components.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Complex32 {
    pub r: f32,
    pub i: f32,
}

impl Complex32 {
    #[inline]
    pub const fn new(r: f32, i: f32) -> Self {
        Self { r, i }
    }
}

/// Indices that were validated against one fixed exclusive upper bound.
#[derive(Clone, Debug)]
pub struct InRangeIndices {
    values: Vec<usize>,
    bound: usize,
}

impl InRangeIndices {
    pub fn new(values: Vec<usize>, bound: usize) -> Option<Self> {
        if values.iter().any(|&value| value >= bound) {
            return None;
        }
        Some(Self { values, bound })
    }

    #[inline]
    pub fn as_slice(&self) -> &[usize] {
        &self.values
    }
}

/// Converts a CELT pulse-cache bit budget to a pseudo-pulse index.
///
/// The first cache byte is the largest valid index. This wrapper checks that
/// the complete indexed range exists before the fixed six-step search uses
/// unchecked reads.
#[inline]
pub fn pulse_cache_bits_to_pulses(cache: &[u8], bits: i32) -> Option<usize> {
    let hi = usize::from(*cache.first()?);
    if cache.len() <= hi {
        return None;
    }

    // SAFETY: The check above proves that indices `0..=hi` exist. The search
    // starts with `lo == 0`, never increases `hi`, and only sets `lo` to a
    // midpoint in `lo..=hi`.
    Some(unsafe { pulse_cache_bits_to_pulses_inner(cache, bits, hi) })
}

#[inline]
unsafe fn pulse_cache_bits_to_pulses_inner(cache: &[u8], bits: i32, mut hi: usize) -> usize {
    let cache = cache.as_ptr();
    let mut lo = 0usize;
    let bits = bits - 1;
    for _ in 0..6 {
        let mid = (lo + hi + 1) >> 1;
        // SAFETY: The checked wrapper proves `mid <= hi < cache.len()`.
        if i32::from(unsafe { *cache.add(mid) }) >= bits {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    let lo_bits = if lo == 0 {
        -1
    } else {
        // SAFETY: `lo` starts at zero and never exceeds the checked `hi`.
        i32::from(unsafe { *cache.add(lo) })
    };
    // SAFETY: `hi` never exceeds the maximum index checked by the wrapper.
    let hi_bits = i32::from(unsafe { *cache.add(hi) });
    if bits - lo_bits <= hi_bits - bits {
        lo
    } else {
        hi
    }
}

/// Applies stereo CELT deemphasis and converts the result to signed i24.
///
/// The wrapper validates all input, state, and output ranges before the inner
/// loop uses unchecked indexing. Each float is made finite and clamped before
/// the unchecked integer conversion.
#[inline]
pub fn deemphasis_stereo_i24(
    input0: &[f32],
    input1: &[f32],
    coef: f32,
    mem: &mut [f32],
    output: &mut [i32],
) -> bool {
    let n = input0.len();
    let Some(output_len) = n.checked_mul(2) else {
        return false;
    };
    if n == 0 || input1.len() != n || mem.len() < 2 || output.len() < output_len {
        return false;
    }

    // SAFETY: The checks above validate both n-element inputs, two state
    // values, and `2*n` output values. The inner loop clamps every float to a
    // finite range strictly inside `i32` before `to_int_unchecked`.
    unsafe { deemphasis_stereo_i24_inner(input0, input1, coef, mem, output) };
    true
}

#[inline]
unsafe fn deemphasis_stereo_i24_inner(
    input0: &[f32],
    input1: &[f32],
    coef: f32,
    mem: &mut [f32],
    output: &mut [i32],
) {
    let input0_ptr = input0.as_ptr();
    let input1_ptr = input1.as_ptr();
    let output_ptr = output.as_mut_ptr();
    let mut mem0 = mem[0];
    let mut mem1 = mem[1];

    for frame in 0..input0.len() {
        // SAFETY: The wrapper validates both input ranges and `2*n` outputs.
        let (sample0, sample1) = unsafe { (*input0_ptr.add(frame), *input1_ptr.add(frame)) };
        let tmp0 = sample0 + 1e-30f32 + mem0;
        let tmp1 = sample1 + 1e-30f32 + mem1;
        mem0 = coef * tmp0;
        mem1 = coef * tmp1;

        let scaled0 = (tmp0 * 256.0).max(-8_388_608.0).min(8_388_607.0) + 0.5;
        let scaled1 = (tmp1 * 256.0).max(-8_388_608.0).min(8_388_607.0) + 0.5;
        // SAFETY: `max` replaces NaN with the finite lower bound, `min`
        // bounds positive infinity, and both results are strictly within the
        // representable `i32` range after adding 0.5.
        let truncated0 = unsafe { scaled0.to_int_unchecked::<i32>() };
        let truncated1 = unsafe { scaled1.to_int_unchecked::<i32>() };
        let converted0 = truncated0 - i32::from((truncated0 as f32) > scaled0);
        let converted1 = truncated1 - i32::from((truncated1 as f32) > scaled1);
        unsafe {
            *output_ptr.add(2 * frame) = converted0;
            *output_ptr.add(2 * frame + 1) = converted1;
        }
    }

    mem[0] = mem0;
    mem[1] = mem1;
}

/// Decodes one CWRS/PVQ index from a validated rectangular `U(n,k)` table.
///
/// The table stores one row per remaining dimension with `row_stride` values
/// per row. The function returns `None` when the supplied ranges cannot cover
/// every probe or when malformed table data would underflow the pulse index.
#[inline]
pub fn cwrs_decode_index_rect(
    rows: &[u32],
    row_stride: usize,
    n: usize,
    k_total: usize,
    mut index: u32,
    output: &mut [i32],
) -> Option<i32> {
    let last_column = k_total.checked_add(1)?;
    if n < 2 || k_total == 0 || output.len() < n || last_column >= row_stride {
        return None;
    }
    let required_rows = n.checked_add(1)?.checked_mul(row_stride)?;
    if rows.len() < required_rows {
        return None;
    }

    // SAFETY: The checks above cover all rows through `n` and every pulse
    // column through `k_total + 1`. The inner loop only decreases `dims` and
    // `k`, guards every decrement that could underflow, and writes exactly
    // `n` output values.
    unsafe { cwrs_decode_index_rect_inner(rows, row_stride, n, k_total, &mut index, output) }
}

#[inline]
unsafe fn cwrs_decode_index_rect_inner(
    rows: &[u32],
    row_stride: usize,
    n: usize,
    k_total: usize,
    index: &mut u32,
    output: &mut [i32],
) -> Option<i32> {
    let rows_ptr = rows.as_ptr();
    let output_ptr = output.as_mut_ptr();
    let mut dims = n;
    let mut k = k_total;
    let mut pos = 0usize;
    let mut yy = 0i32;

    while dims > 2 {
        // SAFETY: The wrapper validates rows `0..=n`, and `dims` only falls.
        let row = unsafe { rows_ptr.add(dims * row_stride) };
        if k >= dims {
            // SAFETY: `k + 1 <= k_total + 1 < row_stride`.
            let mut p = unsafe { *row.add(k + 1) };
            let sign = if *index >= p { -1 } else { 0 };
            if sign != 0 {
                *index = index.wrapping_sub(p);
            }
            let k0 = k;
            // SAFETY: This branch proves `dims <= k <= k_total`.
            if unsafe { *row.add(dims) } > *index {
                k = dims;
                loop {
                    if k == 0 {
                        return None;
                    }
                    k -= 1;
                    // SAFETY: `k < dims <= k_total < row_stride`.
                    p = unsafe { *row.add(k) };
                    if p <= *index {
                        break;
                    }
                }
            } else {
                // SAFETY: `k <= k_total < row_stride`.
                p = unsafe { *row.add(k) };
                while p > *index {
                    if k == 0 {
                        return None;
                    }
                    k -= 1;
                    // SAFETY: `k` remains below `row_stride`.
                    p = unsafe { *row.add(k) };
                }
            }
            *index = index.wrapping_sub(p);
            let magnitude = (k0 - k) as i32;
            let value = (magnitude + sign) ^ sign;
            // SAFETY: `pos < n - 2` during this loop.
            unsafe { *output_ptr.add(pos) = value };
            yy += value * value;
        } else {
            // SAFETY: `k + 1 <= k_total + 1 < row_stride`.
            let mut p = unsafe { *row.add(k) };
            let q = unsafe { *row.add(k + 1) };
            if p <= *index && *index < q {
                *index = index.wrapping_sub(p);
                // SAFETY: `pos < n - 2` during this loop.
                unsafe { *output_ptr.add(pos) = 0 };
            } else {
                let sign = if *index >= q { -1 } else { 0 };
                if sign != 0 {
                    *index = index.wrapping_sub(q);
                }
                let k0 = k;
                loop {
                    if k == 0 {
                        return None;
                    }
                    k -= 1;
                    // SAFETY: `k` remains below `row_stride`.
                    p = unsafe { *row.add(k) };
                    if p <= *index {
                        break;
                    }
                }
                *index = index.wrapping_sub(p);
                let magnitude = (k0 - k) as i32;
                let value = (magnitude + sign) ^ sign;
                // SAFETY: `pos < n - 2` during this loop.
                unsafe { *output_ptr.add(pos) = value };
                yy += value * value;
            }
        }
        dims -= 1;
        pos += 1;
    }

    let mut p = (2 * k + 1) as u32;
    let sign = if *index >= p { -1 } else { 0 };
    if sign != 0 {
        *index = index.wrapping_sub(p);
    }
    let k0 = k;
    k = ((*index + 1) >> 1) as usize;
    if k != 0 {
        p = (2 * k - 1) as u32;
        *index = index.wrapping_sub(p);
    }
    let magnitude = (k0 - k) as i32;
    let value = (magnitude + sign) ^ sign;
    // SAFETY: The loop leaves `pos == n - 2`.
    unsafe { *output_ptr.add(pos) = value };
    yy += value * value;

    let sign = if *index != 0 { -1 } else { 0 };
    let value = (k as i32 + sign) ^ sign;
    // SAFETY: `pos + 1 == n - 1`.
    unsafe { *output_ptr.add(pos + 1) = value };
    yy += value * value;
    Some(yy)
}

/// Applies the forward-MDCT pre-rotation in validated bit-reversed order.
#[inline]
pub fn mdct_forward_pre_rotate(
    folded: &[f32],
    trig: &[f32],
    bitrev: &InRangeIndices,
    scale: f32,
    output: &mut [Complex32],
) {
    let n4 = bitrev.values.len();
    assert!(n4 > 0);
    assert!(output.len() >= bitrev.bound);
    let n2 = n4.checked_mul(2).expect("forward MDCT size overflow");
    assert!(folded.len() >= n2);
    assert!(trig.len() >= n2);

    for i in 0..n4 {
        // SAFETY: `InRangeIndices::new` checked every output index against
        // `bound`, and `output.len() >= bound`. The checks above cover both
        // folded input values and both trig-table halves for every iteration.
        unsafe {
            let t0 = *trig.get_unchecked(i);
            let t1 = *trig.get_unchecked(n4 + i);
            let re = *folded.get_unchecked(2 * i);
            let im = *folded.get_unchecked(2 * i + 1);
            let yr = re * t0 - im * t1;
            let yi = im * t0 + re * t1;
            let rev = *bitrev.values.get_unchecked(i);
            *output.get_unchecked_mut(rev) = Complex32::new(yr * scale, yi * scale);
        }
    }
}

/// Applies the forward-MDCT post-rotation in natural order.
#[inline]
pub fn mdct_forward_post_rotate(
    values: &[Complex32],
    trig: &[f32],
    stride: usize,
    output: &mut [f32],
) {
    let n4 = values.len();
    assert!(n4 > 0);
    let n2 = n4.checked_mul(2).expect("forward MDCT size overflow");
    assert!(trig.len() >= n2);
    let last_output = stride
        .checked_mul(n2 - 1)
        .expect("forward MDCT output stride overflow");
    assert!(output.len() > last_output);
    let step = stride
        .checked_mul(2)
        .expect("forward MDCT output step overflow");

    let mut yp1 = 0usize;
    let mut yp2 = last_output;
    for i in 0..n4 {
        // SAFETY: The checks above cover every input, both trig-table halves,
        // and the complete strided output range. Both output cursors remain
        // within that range for all `n4` iterations.
        unsafe {
            let value = *values.get_unchecked(i);
            let t0 = *trig.get_unchecked(i);
            let t1 = *trig.get_unchecked(n4 + i);
            let yr = value.i * t1 - value.r * t0;
            let yi = value.r * t1 + value.i * t0;
            *output.get_unchecked_mut(yp1) = yr;
            *output.get_unchecked_mut(yp2) = yi;
        }
        yp1 += step;
        if i + 1 < n4 {
            yp2 -= step;
        }
    }
}

/// Applies the inverse-MDCT pre-rotation in validated bit-reversed order.
#[inline]
pub fn mdct_backward_pre_rotate(
    input: &[f32],
    trig: &[f32],
    bitrev: &InRangeIndices,
    stride: usize,
    output: &mut [Complex32],
) {
    let n4 = bitrev.values.len();
    assert!(n4 > 0);
    assert!(output.len() >= bitrev.bound);
    let n2 = n4.checked_mul(2).expect("inverse MDCT size overflow");
    assert!(trig.len() >= n2);
    let last_input = stride
        .checked_mul(n2 - 1)
        .expect("inverse MDCT input stride overflow");
    assert!(input.len() > last_input);
    let step = stride
        .checked_mul(2)
        .expect("inverse MDCT input step overflow");

    let mut xp1 = 0usize;
    let mut xp2 = last_input;
    for i in 0..n4 {
        // SAFETY: `InRangeIndices::new` checked every output index against
        // `bound`, and `output.len() >= bound`. The length and overflow checks
        // above cover the two input traversals and both trig-table halves.
        unsafe {
            let rev = *bitrev.values.get_unchecked(i);
            let x1 = *input.get_unchecked(xp1);
            let x2 = *input.get_unchecked(xp2);
            let t0 = *trig.get_unchecked(i);
            let t1 = *trig.get_unchecked(n4 + i);
            let yr = x2 * t0 + x1 * t1;
            let yi = x1 * t0 - x2 * t1;
            *output.get_unchecked_mut(rev) = Complex32::new(yi, yr);
        }
        if i + 1 < n4 {
            xp1 += step;
            xp2 -= step;
        }
    }
}

/// Applies the inverse-MDCT post-rotation in place.
#[inline]
pub fn mdct_backward_post_rotate(values: &mut [Complex32], trig: &[f32]) {
    let n4 = values.len();
    assert!(n4 > 0);
    let n2 = n4.checked_mul(2).expect("inverse MDCT size overflow");
    assert!(trig.len() >= n2);

    let mut lo = 0usize;
    let mut hi = n4 - 1;
    for i in 0..((n4 + 1) >> 1) {
        // SAFETY: `lo` rises from zero and `hi` falls from `n4 - 1` for only
        // ceil(`n4 / 2`) iterations. All four trig indices stay below `n2`.
        unsafe {
            let re = values.get_unchecked(lo).i;
            let im = values.get_unchecked(lo).r;
            let t0 = *trig.get_unchecked(i);
            let t1 = *trig.get_unchecked(n4 + i);
            let yr = re * t0 + im * t1;
            let yi = re * t1 - im * t0;

            let re2 = values.get_unchecked(hi).i;
            let im2 = values.get_unchecked(hi).r;
            values.get_unchecked_mut(lo).r = yr;
            values.get_unchecked_mut(hi).i = yi;

            let t0 = *trig.get_unchecked(n4 - i - 1);
            let t1 = *trig.get_unchecked(n2 - i - 1);
            let yr = re2 * t0 + im2 * t1;
            let yi = re2 * t1 - im2 * t0;
            values.get_unchecked_mut(hi).r = yr;
            values.get_unchecked_mut(lo).i = yi;
        }

        lo += 1;
        hi = hi.saturating_sub(1);
    }
}

/// Mirrors and windows the inverse-MDCT overlap region in place.
#[inline]
pub fn mdct_backward_mirror(output: &mut [f32], window: &[f32], overlap: usize) {
    assert!(output.len() >= overlap);
    assert!(window.len() >= overlap);
    for i in 0..overlap / 2 {
        let lo = i;
        let hi = overlap - 1 - i;
        // SAFETY: Both slices contain at least `overlap` values. The output
        // indices are distinct for every loop iteration.
        unsafe {
            let x1 = *output.get_unchecked(hi);
            let x2 = *output.get_unchecked(lo);
            let wp1 = *window.get_unchecked(i);
            let wp2 = *window.get_unchecked(overlap - 1 - i);
            *output.get_unchecked_mut(lo) = x2 * wp2 - x1 * wp1;
            *output.get_unchecked_mut(hi) = x2 * wp1 + x1 * wp2;
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use core::arch::aarch64::{
        float32x4_t, float32x4x2_t, vadd_f32, vaddq_f32, vdupq_n_f32, vextq_f32, vfmaq_f32,
        vfmaq_lane_f32, vfmaq_laneq_f32, vget_high_f32, vget_lane_f32, vget_low_f32, vld1_dup_f32,
        vld1q_f32, vld2q_f32, vmulq_f32, vnegq_f32, vpadd_f32, vst1q_f32, vst2q_f32, vsubq_f32,
    };

    use super::Complex32;

    #[derive(Clone, Copy)]
    struct Complex4 {
        r: float32x4_t,
        i: float32x4_t,
    }

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn load_complex4(ptr: *const Complex32) -> Complex4 {
        // SAFETY: Callers provide four valid consecutive `Complex32` values.
        let values = unsafe { vld2q_f32(ptr.cast::<f32>()) };
        Complex4 {
            r: values.0,
            i: values.1,
        }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn store_complex4(ptr: *mut Complex32, value: Complex4) {
        let values = float32x4x2_t(value.r, value.i);
        // SAFETY: Callers provide space for four consecutive `Complex32`
        // values. `Complex32` has the same interleaved two-f32 layout.
        unsafe { vst2q_f32(ptr.cast::<f32>(), values) };
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn complex_add(a: Complex4, b: Complex4) -> Complex4 {
        Complex4 {
            r: vaddq_f32(a.r, b.r),
            i: vaddq_f32(a.i, b.i),
        }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn complex_sub(a: Complex4, b: Complex4) -> Complex4 {
        Complex4 {
            r: vsubq_f32(a.r, b.r),
            i: vsubq_f32(a.i, b.i),
        }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn complex_mul(a: Complex4, b: Complex4) -> Complex4 {
        Complex4 {
            r: vsubq_f32(vmulq_f32(a.r, b.r), vmulq_f32(a.i, b.i)),
            i: vaddq_f32(vmulq_f32(a.r, b.i), vmulq_f32(a.i, b.r)),
        }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn mul_scalar(a: Complex4, scalar: float32x4_t) -> Complex4 {
        Complex4 {
            r: vmulq_f32(a.r, scalar),
            i: vmulq_f32(a.i, scalar),
        }
    }

    /// Applies a contiguous radix-5 FFT stage four butterflies at a time.
    ///
    /// `twiddles` contains four `m`-element blocks for twiddle powers one
    /// through four. The checked wrapper validates all vector loads and
    /// stores before entering the unsafe kernel.
    #[inline]
    pub fn fft_radix5_neon(
        values: &mut [Complex32],
        twiddles: &[Complex32],
        ya: Complex32,
        yb: Complex32,
        m: usize,
    ) -> bool {
        let Some(values_len) = m.checked_mul(5) else {
            return false;
        };
        let Some(twiddles_len) = m.checked_mul(4) else {
            return false;
        };
        if !std::arch::is_aarch64_feature_detected!("neon")
            || m == 0
            || m % 4 != 0
            || values.len() < values_len
            || twiddles.len() < twiddles_len
        {
            return false;
        }
        debug_assert_eq!(
            core::mem::size_of::<Complex32>(),
            2 * core::mem::size_of::<f32>()
        );

        // SAFETY: The checks above cover five `m`-element value blocks and
        // four `m`-element twiddle blocks. `Complex32` is `repr(C)` with two
        // adjacent f32 fields, and runtime detection confirms NEON support.
        unsafe { fft_radix5_neon_inner(values, twiddles, ya, yb, m) };
        true
    }

    #[target_feature(enable = "neon")]
    unsafe fn fft_radix5_neon_inner(
        values: &mut [Complex32],
        twiddles: &[Complex32],
        ya: Complex32,
        yb: Complex32,
        m: usize,
    ) {
        let values_ptr = values.as_mut_ptr();
        let twiddles_ptr = twiddles.as_ptr();
        let ya_r = vdupq_n_f32(ya.r);
        let ya_i = vdupq_n_f32(ya.i);
        let yb_r = vdupq_n_f32(yb.r);
        let yb_i = vdupq_n_f32(yb.i);

        for u in (0..m).step_by(4) {
            // SAFETY: The checked wrapper proves that four elements remain in
            // every value and twiddle block at each `u`.
            let (scratch0, scratch1, scratch2, scratch3, scratch4) = unsafe {
                (
                    load_complex4(values_ptr.add(u)),
                    complex_mul(
                        load_complex4(values_ptr.add(m + u)),
                        load_complex4(twiddles_ptr.add(u)),
                    ),
                    complex_mul(
                        load_complex4(values_ptr.add(2 * m + u)),
                        load_complex4(twiddles_ptr.add(m + u)),
                    ),
                    complex_mul(
                        load_complex4(values_ptr.add(3 * m + u)),
                        load_complex4(twiddles_ptr.add(2 * m + u)),
                    ),
                    complex_mul(
                        load_complex4(values_ptr.add(4 * m + u)),
                        load_complex4(twiddles_ptr.add(3 * m + u)),
                    ),
                )
            };

            let scratch7 = complex_add(scratch1, scratch4);
            let scratch10 = complex_sub(scratch1, scratch4);
            let scratch8 = complex_add(scratch2, scratch3);
            let scratch9 = complex_sub(scratch2, scratch3);
            let output0 = complex_add(complex_add(scratch0, scratch7), scratch8);

            let scratch5 = complex_add(
                complex_add(scratch0, mul_scalar(scratch7, ya_r)),
                mul_scalar(scratch8, yb_r),
            );
            let scratch6 = Complex4 {
                r: vaddq_f32(vmulq_f32(scratch10.i, ya_i), vmulq_f32(scratch9.i, yb_i)),
                i: vnegq_f32(vaddq_f32(
                    vmulq_f32(scratch10.r, ya_i),
                    vmulq_f32(scratch9.r, yb_i),
                )),
            };
            let output1 = complex_sub(scratch5, scratch6);
            let output4 = complex_add(scratch5, scratch6);

            let scratch11 = complex_add(
                complex_add(scratch0, mul_scalar(scratch7, yb_r)),
                mul_scalar(scratch8, ya_r),
            );
            let scratch12 = Complex4 {
                r: vsubq_f32(vmulq_f32(scratch9.i, ya_i), vmulq_f32(scratch10.i, yb_i)),
                i: vsubq_f32(vmulq_f32(scratch10.r, yb_i), vmulq_f32(scratch9.r, ya_i)),
            };
            let output2 = complex_add(scratch11, scratch12);
            let output3 = complex_sub(scratch11, scratch12);

            // SAFETY: The checked wrapper covers all five four-element stores.
            unsafe {
                store_complex4(values_ptr.add(u), output0);
                store_complex4(values_ptr.add(m + u), output1);
                store_complex4(values_ptr.add(2 * m + u), output2);
                store_complex4(values_ptr.add(3 * m + u), output3);
                store_complex4(values_ptr.add(4 * m + u), output4);
            }
        }
    }

    /// Computes two dot products that share one input with NEON.
    #[inline]
    pub fn dual_inner_prod_f32(x: &[f32], y1: &[f32], y2: &[f32], n: usize) -> Option<(f32, f32)> {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return None;
        }
        assert!(x.len() >= n);
        assert!(y1.len() >= n);
        assert!(y2.len() >= n);

        // SAFETY: The assertions cover all three pointer ranges. Runtime
        // feature detection confirms NEON support.
        Some(unsafe { dual_inner_prod_neon(x, y1, y2, n) })
    }

    #[target_feature(enable = "neon")]
    unsafe fn dual_inner_prod_neon(x: &[f32], y1: &[f32], y2: &[f32], n: usize) -> (f32, f32) {
        let mut sum1 = vdupq_n_f32(0.0);
        let mut sum2 = vdupq_n_f32(0.0);
        let mut i = 0usize;
        while i + 8 <= n {
            // SAFETY: Eight elements remain in each validated slice.
            let (x0, y10, y20, x1, y11, y21) = unsafe {
                (
                    vld1q_f32(x.as_ptr().add(i)),
                    vld1q_f32(y1.as_ptr().add(i)),
                    vld1q_f32(y2.as_ptr().add(i)),
                    vld1q_f32(x.as_ptr().add(i + 4)),
                    vld1q_f32(y1.as_ptr().add(i + 4)),
                    vld1q_f32(y2.as_ptr().add(i + 4)),
                )
            };
            sum1 = vfmaq_f32(sum1, x0, y10);
            sum2 = vfmaq_f32(sum2, x0, y20);
            sum1 = vfmaq_f32(sum1, x1, y11);
            sum2 = vfmaq_f32(sum2, x1, y21);
            i += 8;
        }
        if i + 4 <= n {
            // SAFETY: Four elements remain in each validated slice.
            let (x0, y10, y20) = unsafe {
                (
                    vld1q_f32(x.as_ptr().add(i)),
                    vld1q_f32(y1.as_ptr().add(i)),
                    vld1q_f32(y2.as_ptr().add(i)),
                )
            };
            sum1 = vfmaq_f32(sum1, x0, y10);
            sum2 = vfmaq_f32(sum2, x0, y20);
            i += 4;
        }

        let pair1 = vadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
        let pair2 = vadd_f32(vget_low_f32(sum2), vget_high_f32(sum2));
        let mut scalar1 = vget_lane_f32::<0>(vpadd_f32(pair1, pair1));
        let mut scalar2 = vget_lane_f32::<0>(vpadd_f32(pair2, pair2));
        while i < n {
            // SAFETY: The loop condition keeps every scalar read in range.
            unsafe {
                scalar1 = x.get_unchecked(i).mul_add(*y1.get_unchecked(i), scalar1);
                scalar2 = x.get_unchecked(i).mul_add(*y2.get_unchecked(i), scalar2);
            }
            i += 1;
        }
        (scalar1, scalar2)
    }

    /// Computes groups of four adjacent pitch correlations with NEON.
    ///
    /// Returns `false` when NEON is unavailable. `max_pitch` must be a
    /// multiple of four. The checks before the unsafe call cover every vector
    /// load and store.
    #[inline]
    pub fn pitch_xcorr_4(
        x: &[f32],
        y: &[f32],
        len: usize,
        max_pitch: usize,
        xcorr: &mut [f32],
    ) -> bool {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return false;
        }
        assert!(len > 0);
        assert_eq!(max_pitch % 4, 0);
        assert!(x.len() >= len);
        assert!(y.len() >= len + max_pitch.saturating_sub(1));
        assert!(xcorr.len() >= max_pitch);

        // SAFETY: The assertions above keep all x loads within `len`, all y
        // loads within `len + max_pitch - 1`, and all stores within
        // `max_pitch`. Runtime feature detection confirms NEON support.
        unsafe { pitch_xcorr_4_neon(x, y, len, max_pitch, xcorr) };
        true
    }

    #[target_feature(enable = "neon")]
    unsafe fn pitch_xcorr_4_neon(
        x: &[f32],
        y: &[f32],
        len: usize,
        max_pitch: usize,
        xcorr: &mut [f32],
    ) {
        let x_ptr = x.as_ptr();
        let y_ptr = y.as_ptr();
        let out_ptr = xcorr.as_mut_ptr();
        for lag in (0..max_pitch).step_by(4) {
            // SAFETY: `pitch_xcorr_4` validates the full pointer ranges once.
            // The kernel advances at most `len` x elements and `len + 3` y
            // elements.
            unsafe {
                xcorr_kernel_neon(x_ptr, y_ptr.add(lag), out_ptr.add(lag), len);
            }
        }
    }

    #[target_feature(enable = "neon")]
    unsafe fn xcorr_kernel_neon(
        mut x: *const f32,
        mut y: *const f32,
        sum_out: *mut f32,
        mut len: usize,
    ) {
        // SAFETY: The caller validates enough elements for every vector load.
        let mut y0 = unsafe { vld1q_f32(y) };
        let mut sum = vdupq_n_f32(0.0);

        while len > 8 {
            // SAFETY: This consumes eight x values and needs eleven y values.
            let (x0, x1, y1, y2) = unsafe {
                (
                    vld1q_f32(x),
                    vld1q_f32(x.add(4)),
                    vld1q_f32(y.add(4)),
                    vld1q_f32(y.add(8)),
                )
            };
            sum = accumulate_four(sum, y0, y1, x0);
            sum = accumulate_four(sum, y1, y2, x1);
            x = unsafe { x.add(8) };
            y = unsafe { y.add(8) };
            y0 = y2;
            len -= 8;
        }

        if len > 4 {
            // SAFETY: This consumes four x values and needs seven y values.
            let (x0, y1) = unsafe { (vld1q_f32(x), vld1q_f32(y.add(4))) };
            sum = accumulate_four(sum, y0, y1, x0);
            x = unsafe { x.add(4) };
            y = unsafe { y.add(4) };
            y0 = y1;
            len -= 4;
        }

        while len > 1 {
            // SAFETY: One x value and four y values remain here.
            let x0 = unsafe { vld1_dup_f32(x) };
            sum = vfmaq_lane_f32::<0>(sum, y0, x0);
            x = unsafe { x.add(1) };
            y = unsafe { y.add(1) };
            y0 = unsafe { vld1q_f32(y) };
            len -= 1;
        }

        // SAFETY: `len` is one, so the final x and four-lane y loads fit.
        let x0 = unsafe { vld1_dup_f32(x) };
        sum = vfmaq_lane_f32::<0>(sum, y0, x0);
        unsafe { vst1q_f32(sum_out, sum) };
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn accumulate_four(
        mut sum: float32x4_t,
        y0: float32x4_t,
        y1: float32x4_t,
        x: float32x4_t,
    ) -> float32x4_t {
        sum = vfmaq_laneq_f32::<0>(sum, y0, x);
        sum = vfmaq_laneq_f32::<1>(sum, vextq_f32::<1>(y0, y1), x);
        sum = vfmaq_laneq_f32::<2>(sum, vextq_f32::<2>(y0, y1), x);
        vfmaq_laneq_f32::<3>(sum, vextq_f32::<3>(y0, y1), x)
    }
}

#[cfg(target_arch = "aarch64")]
pub use aarch64::{dual_inner_prod_f32, fft_radix5_neon, pitch_xcorr_4};

#[cfg(target_arch = "x86_64")]
mod x86_64 {
    use core::arch::x86_64::{
        __m256, _mm256_add_ps, _mm256_fmadd_ps, _mm256_hadd_ps, _mm256_loadu_ps,
        _mm256_loadu_si256, _mm256_maskload_ps, _mm256_permute2f128_ps, _mm256_setzero_ps,
        _mm256_storeu_ps, _mm_add_epi32, _mm_add_ps, _mm_add_ss, _mm_and_si128, _mm_castps_si128,
        _mm_cmpeq_ps, _mm_cmpgt_ps, _mm_cvtsi128_si32, _mm_cvtss_f32, _mm_loadu_ps, _mm_max_epi16,
        _mm_max_ps, _mm_movehl_ps, _mm_mul_ps, _mm_rsqrt_ps, _mm_set1_ps, _mm_set_epi32,
        _mm_setzero_ps, _mm_setzero_si128, _mm_shuffle_ps, _mm_shufflelo_epi16, _mm_unpackhi_epi64,
    };

    const SHUFFLE_SWAP_PAIRS: i32 = 0b01_00_11_10;
    const SHUFFLE_SWAP_ADJACENT: i32 = 0b10_11_00_01;
    static XCORR_TAIL_MASK: [i32; 15] = [-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0];

    /// Computes groups of eight adjacent pitch correlations with AVX2.
    ///
    /// Returns `false` when AVX2 or FMA is unavailable. `max_pitch` must be a
    /// multiple of eight. The checks before the unsafe call cover every load
    /// and store, including the masked final input block.
    #[inline]
    pub fn pitch_xcorr_8(
        x: &[f32],
        y: &[f32],
        len: usize,
        max_pitch: usize,
        xcorr: &mut [f32],
    ) -> bool {
        if !std::arch::is_x86_feature_detected!("avx2")
            || !std::arch::is_x86_feature_detected!("fma")
        {
            return false;
        }
        assert!(len > 0);
        assert_eq!(max_pitch % 8, 0);
        assert!(x.len() >= len);
        assert!(y.len() >= len + max_pitch.saturating_sub(1));
        assert!(xcorr.len() >= max_pitch);

        // SAFETY: The assertions cover all input and output ranges. Runtime
        // feature detection confirms AVX2 and FMA support. The tail uses
        // masked loads, so it does not read beyond the validated slices.
        unsafe { pitch_xcorr_8_avx2(x, y, len, max_pitch, xcorr) };
        true
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn pitch_xcorr_8_avx2(
        x: &[f32],
        y: &[f32],
        len: usize,
        max_pitch: usize,
        xcorr: &mut [f32],
    ) {
        let x_ptr = x.as_ptr();
        let y_ptr = y.as_ptr();
        let output_ptr = xcorr.as_mut_ptr();
        for lag in (0..max_pitch).step_by(8) {
            // SAFETY: The checked wrapper validates `len` x values, `len + 7`
            // y values from each group start, and eight output values.
            unsafe {
                xcorr_kernel_avx2(x_ptr, y_ptr.add(lag), output_ptr.add(lag), len);
            }
        }
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn xcorr_kernel_avx2(x: *const f32, y: *const f32, output: *mut f32, len: usize) {
        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();
        let mut sum2 = _mm256_setzero_ps();
        let mut sum3 = _mm256_setzero_ps();
        let mut sum4 = _mm256_setzero_ps();
        let mut sum5 = _mm256_setzero_ps();
        let mut sum6 = _mm256_setzero_ps();
        let mut sum7 = _mm256_setzero_ps();
        let mut i = 0usize;
        while i + 8 <= len {
            // SAFETY: Eight x values and fifteen y values remain. Each y load
            // starts at one of the first eight positions.
            let x0 = unsafe { _mm256_loadu_ps(x.add(i)) };
            unsafe {
                sum0 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(y.add(i)), sum0);
                sum1 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(y.add(i + 1)), sum1);
                sum2 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(y.add(i + 2)), sum2);
                sum3 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(y.add(i + 3)), sum3);
                sum4 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(y.add(i + 4)), sum4);
                sum5 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(y.add(i + 5)), sum5);
                sum6 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(y.add(i + 6)), sum6);
                sum7 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(y.add(i + 7)), sum7);
            }
            i += 8;
        }
        if i != len {
            let remaining = len - i;
            // SAFETY: `remaining` is in `1..=7`, so the selected eight-value
            // window is inside the 15-value static mask. Masked input loads
            // access only the `remaining` validated elements.
            let mask =
                unsafe { _mm256_loadu_si256(XCORR_TAIL_MASK.as_ptr().add(7 - remaining).cast()) };
            unsafe {
                let x0 = _mm256_maskload_ps(x.add(i), mask);
                sum0 = _mm256_fmadd_ps(x0, _mm256_maskload_ps(y.add(i), mask), sum0);
                sum1 = _mm256_fmadd_ps(x0, _mm256_maskload_ps(y.add(i + 1), mask), sum1);
                sum2 = _mm256_fmadd_ps(x0, _mm256_maskload_ps(y.add(i + 2), mask), sum2);
                sum3 = _mm256_fmadd_ps(x0, _mm256_maskload_ps(y.add(i + 3), mask), sum3);
                sum4 = _mm256_fmadd_ps(x0, _mm256_maskload_ps(y.add(i + 4), mask), sum4);
                sum5 = _mm256_fmadd_ps(x0, _mm256_maskload_ps(y.add(i + 5), mask), sum5);
                sum6 = _mm256_fmadd_ps(x0, _mm256_maskload_ps(y.add(i + 6), mask), sum6);
                sum7 = _mm256_fmadd_ps(x0, _mm256_maskload_ps(y.add(i + 7), mask), sum7);
            }
        }

        let sum04 = horizontal_pair(sum0, sum4);
        let sum15 = horizontal_pair(sum1, sum5);
        let sum26 = horizontal_pair(sum2, sum6);
        let sum37 = horizontal_pair(sum3, sum7);
        let sum0145 = _mm256_hadd_ps(sum04, sum15);
        let sum2367 = _mm256_hadd_ps(sum26, sum37);
        let result = _mm256_hadd_ps(sum0145, sum2367);
        // SAFETY: The checked wrapper reserves eight outputs for this group.
        unsafe { _mm256_storeu_ps(output, result) };
    }

    #[target_feature(enable = "avx2")]
    fn horizontal_pair(low: __m256, high: __m256) -> __m256 {
        _mm256_add_ps(
            _mm256_permute2f128_ps::<0x20>(low, high),
            _mm256_permute2f128_ps::<0x31>(low, high),
        )
    }

    /// Computes two dot products that share one input with SSE.
    #[inline]
    pub fn dual_inner_prod_f32(x: &[f32], y1: &[f32], y2: &[f32], n: usize) -> Option<(f32, f32)> {
        assert!(x.len() >= n);
        assert!(y1.len() >= n);
        assert!(y2.len() >= n);

        // SAFETY: SSE is part of the x86-64 baseline. The assertions cover
        // all three pointer ranges used by the vector and scalar loops.
        Some(unsafe { dual_inner_prod_sse(x, y1, y2, n) })
    }

    #[target_feature(enable = "sse")]
    unsafe fn dual_inner_prod_sse(x: &[f32], y1: &[f32], y2: &[f32], n: usize) -> (f32, f32) {
        let mut sum1 = _mm_setzero_ps();
        let mut sum2 = _mm_setzero_ps();
        let mut i = 0usize;
        while i + 4 <= n {
            // SAFETY: Four elements remain in each validated slice.
            let (x4, y14, y24) = unsafe {
                (
                    _mm_loadu_ps(x.as_ptr().add(i)),
                    _mm_loadu_ps(y1.as_ptr().add(i)),
                    _mm_loadu_ps(y2.as_ptr().add(i)),
                )
            };
            sum1 = _mm_add_ps(sum1, _mm_mul_ps(x4, y14));
            sum2 = _mm_add_ps(sum2, _mm_mul_ps(x4, y24));
            i += 4;
        }

        sum1 = _mm_add_ps(sum1, _mm_movehl_ps(sum1, sum1));
        sum2 = _mm_add_ps(sum2, _mm_movehl_ps(sum2, sum2));
        let mut scalar1 = _mm_cvtss_f32(_mm_add_ss(sum1, _mm_shuffle_ps::<0x55>(sum1, sum1)));
        let mut scalar2 = _mm_cvtss_f32(_mm_add_ss(sum2, _mm_shuffle_ps::<0x55>(sum2, sum2)));
        while i < n {
            // SAFETY: The loop condition keeps every scalar read in range.
            unsafe {
                scalar1 += *x.get_unchecked(i) * *y1.get_unchecked(i);
                scalar2 += *x.get_unchecked(i) * *y2.get_unchecked(i);
            }
            i += 1;
        }
        (scalar1, scalar2)
    }

    /// Finds the best CELT PVQ pulse candidate with trunk Opus's SSE2 score.
    ///
    /// The x86-64 ABI guarantees SSE2. The checked wrapper validates every
    /// direct vector load. The final partial vector uses padded local arrays.
    #[inline]
    pub fn pvq_best_candidate_sse2(x: &[f32], y: &[f32], xy: f32, yy: f32, n: usize) -> usize {
        assert!(n > 0);
        assert!(n <= i16::MAX as usize);
        assert!(x.len() >= n);
        assert!(y.len() >= n);

        // SAFETY: SSE2 is part of the x86-64 baseline. The function loads
        // directly only when four elements remain. Its tail loads use local
        // four-element arrays. The returned index is checked before use.
        let best = unsafe { pvq_best_candidate_sse2_inner(x, y, xy, yy, n) };
        if best < n {
            best
        } else {
            0
        }
    }

    #[target_feature(enable = "sse2")]
    unsafe fn pvq_best_candidate_sse2_inner(
        x: &[f32],
        y: &[f32],
        xy: f32,
        yy: f32,
        n: usize,
    ) -> usize {
        let xy4 = _mm_set1_ps(xy);
        let yy4 = _mm_set1_ps(yy);
        let mut lane_max = _mm_setzero_ps();
        let mut positions = _mm_setzero_si128();
        let mut indices = _mm_set_epi32(3, 2, 1, 0);
        let fours = _mm_set_epi32(4, 4, 4, 4);
        let mut j = 0usize;

        while j < n {
            let (x4, y4) = if j + 4 <= n {
                // SAFETY: The checked wrapper proves that four elements
                // remain in both slices for this branch.
                unsafe {
                    (
                        _mm_loadu_ps(x.as_ptr().add(j)),
                        _mm_loadu_ps(y.as_ptr().add(j)),
                    )
                }
            } else {
                // Match Opus's three padded values without reading past either
                // slice. These sentinels cannot win for valid CELT PVQ state.
                let mut x_tail = [-100.0f32; 4];
                let mut y_tail = [100.0f32; 4];
                let remaining = n - j;
                x_tail[..remaining].copy_from_slice(&x[j..n]);
                y_tail[..remaining].copy_from_slice(&y[j..n]);
                // SAFETY: Both local arrays contain exactly four elements.
                unsafe { (_mm_loadu_ps(x_tail.as_ptr()), _mm_loadu_ps(y_tail.as_ptr())) }
            };

            let numerator = _mm_add_ps(x4, xy4);
            let denominator = _mm_rsqrt_ps(_mm_add_ps(y4, yy4));
            let score = _mm_mul_ps(numerator, denominator);
            let improved = _mm_cmpgt_ps(score, lane_max);
            positions = _mm_max_epi16(
                positions,
                _mm_and_si128(indices, _mm_castps_si128(improved)),
            );
            lane_max = _mm_max_ps(lane_max, score);
            indices = _mm_add_epi32(indices, fours);
            j += 4;
        }

        let mut global_max = _mm_max_ps(
            lane_max,
            _mm_shuffle_ps::<SHUFFLE_SWAP_PAIRS>(lane_max, lane_max),
        );
        global_max = _mm_max_ps(
            global_max,
            _mm_shuffle_ps::<SHUFFLE_SWAP_ADJACENT>(global_max, global_max),
        );
        positions = _mm_and_si128(
            positions,
            _mm_castps_si128(_mm_cmpeq_ps(lane_max, global_max)),
        );
        positions = _mm_max_epi16(positions, _mm_unpackhi_epi64(positions, positions));
        positions = _mm_max_epi16(
            positions,
            _mm_shufflelo_epi16::<SHUFFLE_SWAP_PAIRS>(positions),
        );
        _mm_cvtsi128_si32(positions) as usize
    }
}

#[cfg(target_arch = "x86_64")]
pub use x86_64::{dual_inner_prod_f32, pitch_xcorr_8, pvq_best_candidate_sse2};

#[cfg(test)]
mod tests {
    use super::{
        cwrs_decode_index_rect, deemphasis_stereo_i24, mdct_backward_mirror,
        mdct_backward_post_rotate, mdct_backward_pre_rotate, mdct_forward_post_rotate,
        mdct_forward_pre_rotate, pulse_cache_bits_to_pulses, Complex32, InRangeIndices,
    };

    #[test]
    fn checked_pulse_cache_search_matches_indexed_reference() {
        let cache = [8u8, 7, 15, 23, 31, 39, 47, 55, 63];
        for input_bits in -16..=96 {
            let mut lo = 0usize;
            let mut hi = usize::from(cache[0]);
            let bits = input_bits - 1;
            for _ in 0..6 {
                let mid = (lo + hi + 1) >> 1;
                if i32::from(cache[mid]) >= bits {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            let lo_bits = if lo == 0 { -1 } else { i32::from(cache[lo]) };
            let expected = if bits - lo_bits <= i32::from(cache[hi]) - bits {
                lo
            } else {
                hi
            };
            assert_eq!(
                pulse_cache_bits_to_pulses(&cache, input_bits),
                Some(expected)
            );
        }

        assert_eq!(pulse_cache_bits_to_pulses(&[], 0), None);
        assert_eq!(pulse_cache_bits_to_pulses(&[2, 7], 0), None);
    }

    #[test]
    fn checked_stereo_i24_deemphasis_matches_scalar_reference() {
        for n in [1usize, 2, 5, 120, 240] {
            let input0 = (0..n)
                .map(|index| 24_000.0 * (index as f32 * 0.071 + 0.2).sin())
                .collect::<Vec<_>>();
            let input1 = (0..n)
                .map(|index| 21_000.0 * (index as f32 * 0.053 - 0.4).cos())
                .collect::<Vec<_>>();
            let coef = 0.850_006_1f32;
            let initial_mem = [17.25f32, -31.5f32];
            let mut expected_mem = initial_mem;
            let mut expected = vec![0i32; 2 * n];
            for frame in 0..n {
                for channel in 0..2 {
                    let sample = if channel == 0 {
                        input0[frame]
                    } else {
                        input1[frame]
                    };
                    let tmp = sample + 1e-30f32 + expected_mem[channel];
                    expected_mem[channel] = coef * tmp;
                    let scaled = (tmp * 256.0).clamp(-8_388_608.0, 8_388_607.0) + 0.5;
                    let truncated = scaled as i32;
                    expected[2 * frame + channel] =
                        truncated.saturating_sub(i32::from((truncated as f32) > scaled));
                }
            }

            let mut actual_mem = initial_mem;
            let mut actual = vec![i32::MIN; 2 * n];
            assert!(deemphasis_stereo_i24(
                &input0,
                &input1,
                coef,
                &mut actual_mem,
                &mut actual,
            ));
            assert_eq!(actual, expected, "n={n}");
            assert_eq!(actual_mem[0].to_bits(), expected_mem[0].to_bits(), "n={n}");
            assert_eq!(actual_mem[1].to_bits(), expected_mem[1].to_bits(), "n={n}");
        }

        let mut mem = [0.0f32; 2];
        assert!(!deemphasis_stereo_i24(&[], &[], 0.85, &mut mem, &mut []));
        assert!(!deemphasis_stereo_i24(
            &[0.0; 2],
            &[0.0; 1],
            0.85,
            &mut mem,
            &mut [0; 4],
        ));
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_radix5_matches_scalar_butterflies_bit_exactly() {
        let add = |a: Complex32, b: Complex32| Complex32::new(a.r + b.r, a.i + b.i);
        let sub = |a: Complex32, b: Complex32| Complex32::new(a.r - b.r, a.i - b.i);
        let mul = |a: Complex32, b: Complex32| {
            Complex32::new(a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r)
        };
        let ya = Complex32::new(0.309_017, -0.951_056_54);
        let yb = Complex32::new(-0.809_017, -0.587_785_24);

        for m in [4usize, 8, 12] {
            let sample = |i: usize| {
                Complex32::new(
                    ((i * 17 + 3) as f32 * 0.071).sin(),
                    ((i * 29 + 5) as f32 * 0.043).cos(),
                )
            };
            let mut actual = (0..5 * m).map(sample).collect::<Vec<_>>();
            let twiddles = (0..4 * m).map(|i| sample(i + 5 * m)).collect::<Vec<_>>();
            let mut expected = actual.clone();

            for u in 0..m {
                let scratch0 = expected[u];
                let scratch1 = mul(expected[m + u], twiddles[u]);
                let scratch2 = mul(expected[2 * m + u], twiddles[m + u]);
                let scratch3 = mul(expected[3 * m + u], twiddles[2 * m + u]);
                let scratch4 = mul(expected[4 * m + u], twiddles[3 * m + u]);
                let scratch7 = add(scratch1, scratch4);
                let scratch10 = sub(scratch1, scratch4);
                let scratch8 = add(scratch2, scratch3);
                let scratch9 = sub(scratch2, scratch3);

                expected[u] = Complex32::new(
                    scratch0.r + scratch7.r + scratch8.r,
                    scratch0.i + scratch7.i + scratch8.i,
                );
                let scratch5 = Complex32::new(
                    scratch0.r + scratch7.r * ya.r + scratch8.r * yb.r,
                    scratch0.i + scratch7.i * ya.r + scratch8.i * yb.r,
                );
                let scratch6 = Complex32::new(
                    scratch10.i * ya.i + scratch9.i * yb.i,
                    -(scratch10.r * ya.i + scratch9.r * yb.i),
                );
                expected[m + u] = sub(scratch5, scratch6);
                expected[4 * m + u] = add(scratch5, scratch6);

                let scratch11 = Complex32::new(
                    scratch0.r + scratch7.r * yb.r + scratch8.r * ya.r,
                    scratch0.i + scratch7.i * yb.r + scratch8.i * ya.r,
                );
                let scratch12 = Complex32::new(
                    scratch9.i * ya.i - scratch10.i * yb.i,
                    scratch10.r * yb.i - scratch9.r * ya.i,
                );
                expected[2 * m + u] = add(scratch11, scratch12);
                expected[3 * m + u] = sub(scratch11, scratch12);
            }

            assert!(super::fft_radix5_neon(&mut actual, &twiddles, ya, yb, m));
            for (index, (got, want)) in actual.iter().zip(&expected).enumerate() {
                assert_eq!(
                    got.r.to_bits(),
                    want.r.to_bits(),
                    "real, m={m}, index={index}"
                );
                assert_eq!(
                    got.i.to_bits(),
                    want.i.to_bits(),
                    "imag, m={m}, index={index}"
                );
            }
        }

        assert!(!super::fft_radix5_neon(
            &mut [Complex32::default(); 20],
            &[Complex32::default(); 16],
            ya,
            yb,
            3
        ));
        assert!(!super::fft_radix5_neon(&mut [], &[], ya, yb, usize::MAX));
    }

    #[test]
    fn cwrs_rect_decoder_checks_ranges_and_decodes_two_dimensions() {
        let stride = 4usize;
        let mut rows = vec![0u32; 4 * stride];
        rows[2 * stride..3 * stride].copy_from_slice(&[0, 1, 3, 5]);
        let expected = [
            ([2, 0], 4),
            ([1, 1], 2),
            ([1, -1], 2),
            ([0, 2], 4),
            ([0, -2], 4),
            ([-2, 0], 4),
            ([-1, 1], 2),
            ([-1, -1], 2),
        ];

        for (index, (expected_vector, expected_energy)) in expected.into_iter().enumerate() {
            let mut actual = [99, 99];
            let energy = cwrs_decode_index_rect(&rows, stride, 2, 2, index as u32, &mut actual);
            assert_eq!(energy, Some(expected_energy), "index={index}");
            assert_eq!(actual, expected_vector, "index={index}");
        }

        assert_eq!(
            cwrs_decode_index_rect(&rows, stride, 2, usize::MAX, 0, &mut [0; 2]),
            None
        );
        assert_eq!(
            cwrs_decode_index_rect(&rows[..8], stride, 2, 2, 0, &mut [0; 2]),
            None
        );
    }

    #[test]
    fn validates_indices_once() {
        assert!(InRangeIndices::new(vec![2, 0, 1], 3).is_some());
        assert!(InRangeIndices::new(vec![0, 3], 3).is_none());
        assert!(InRangeIndices::new(Vec::new(), 0).is_some());
    }

    #[test]
    fn inverse_mdct_kernels_match_checked_reference() {
        let n4 = 15usize;
        let n2 = 2 * n4;
        let stride = 3usize;
        let input = (0..stride * (n2 - 1) + 1)
            .map(|i| (i as f32 * 0.03125).sin())
            .collect::<Vec<_>>();
        let trig = (0..n2)
            .map(|i| (i as f32 * 0.0625).cos())
            .collect::<Vec<_>>();
        let indices = (0..n4).map(|i| (i * 4) % n4).collect::<Vec<_>>();
        let bitrev = InRangeIndices::new(indices.clone(), n4).unwrap();

        let mut expected = vec![Complex32::default(); n4];
        for i in 0..n4 {
            let x1 = input[2 * stride * i];
            let x2 = input[stride * (n2 - 1 - 2 * i)];
            let yr = x2 * trig[i] + x1 * trig[n4 + i];
            let yi = x1 * trig[i] - x2 * trig[n4 + i];
            expected[indices[i]] = Complex32::new(yi, yr);
        }
        let mut actual = vec![Complex32::default(); n4];
        mdct_backward_pre_rotate(&input, &trig, &bitrev, stride, &mut actual);
        assert_eq!(actual, expected);

        let post_reference = |values: &mut [Complex32]| {
            let mut lo = 0usize;
            let mut hi = n4 - 1;
            for i in 0..((n4 + 1) >> 1) {
                let re = values[lo].i;
                let im = values[lo].r;
                let yr = re * trig[i] + im * trig[n4 + i];
                let yi = re * trig[n4 + i] - im * trig[i];
                let re2 = values[hi].i;
                let im2 = values[hi].r;
                values[lo].r = yr;
                values[hi].i = yi;
                let yr = re2 * trig[n4 - i - 1] + im2 * trig[n2 - i - 1];
                let yi = re2 * trig[n2 - i - 1] - im2 * trig[n4 - i - 1];
                values[hi].r = yr;
                values[lo].i = yi;
                lo += 1;
                hi = hi.saturating_sub(1);
            }
        };
        post_reference(&mut expected);
        mdct_backward_post_rotate(&mut actual, &trig);
        assert_eq!(actual, expected);

        let overlap = 12usize;
        let window = (0..overlap)
            .map(|i| (i as f32 + 1.0) / overlap as f32)
            .collect::<Vec<_>>();
        let mut expected = (0..overlap).map(|i| i as f32 - 4.0).collect::<Vec<_>>();
        let mut actual = expected.clone();
        for i in 0..overlap / 2 {
            let lo = i;
            let hi = overlap - 1 - i;
            let x1 = expected[hi];
            let x2 = expected[lo];
            expected[lo] = x2 * window[hi] - x1 * window[lo];
            expected[hi] = x2 * window[lo] + x1 * window[hi];
        }
        mdct_backward_mirror(&mut actual, &window, overlap);
        assert_eq!(actual, expected);
    }

    #[test]
    fn forward_mdct_kernels_match_checked_reference() {
        for (n4, stride) in [(1usize, 1usize), (4, 1), (15, 3), (60, 2)] {
            let n2 = 2 * n4;
            let folded = (0..n2)
                .map(|i| (i as f32 * 0.03125 - 0.4).sin())
                .collect::<Vec<_>>();
            let trig = (0..n2)
                .map(|i| (i as f32 * 0.0625 + 0.2).cos())
                .collect::<Vec<_>>();
            let indices = (0..n4).map(|i| (i * 7) % n4).collect::<Vec<_>>();
            let bitrev = InRangeIndices::new(indices.clone(), n4).unwrap();
            let scale = 1.0 / n4 as f32;

            let mut expected = vec![Complex32::default(); n4];
            for i in 0..n4 {
                let t0 = trig[i];
                let t1 = trig[n4 + i];
                let re = folded[2 * i];
                let im = folded[2 * i + 1];
                let yr = re * t0 - im * t1;
                let yi = im * t0 + re * t1;
                expected[indices[i]] = Complex32::new(yr * scale, yi * scale);
            }
            let mut actual = vec![Complex32::default(); n4];
            mdct_forward_pre_rotate(&folded, &trig, &bitrev, scale, &mut actual);
            assert_eq!(actual, expected, "pre-rotation, n4={n4}, stride={stride}");

            let output_len = stride * (n2 - 1) + 1;
            let mut expected_output = vec![f32::NAN; output_len];
            let mut yp1 = 0usize;
            let mut yp2 = stride * (n2 - 1);
            for i in 0..n4 {
                let t0 = trig[i];
                let t1 = trig[n4 + i];
                let yr = expected[i].i * t1 - expected[i].r * t0;
                let yi = expected[i].r * t1 + expected[i].i * t0;
                expected_output[yp1] = yr;
                expected_output[yp2] = yi;
                yp1 += 2 * stride;
                if i + 1 < n4 {
                    yp2 -= 2 * stride;
                }
            }
            let mut actual_output = vec![f32::NAN; output_len];
            mdct_forward_post_rotate(&actual, &trig, stride, &mut actual_output);
            for (index, (got, want)) in actual_output.iter().zip(&expected_output).enumerate() {
                if want.is_nan() {
                    assert!(got.is_nan(), "untouched output, n4={n4}, index={index}");
                } else {
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "post-rotation, n4={n4}, stride={stride}, index={index}"
                    );
                }
            }
        }
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    fn native_dual_inner_product_matches_its_lane_reduction() {
        let mut seed = 0xD07D_1234u32;
        for n in [0usize, 1, 3, 4, 5, 8, 9, 31, 60, 61] {
            let mut next_sample = || {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (seed as i32 as f32) * (0.75 / i32::MAX as f32)
            };
            let x = (0..n).map(|_| next_sample()).collect::<Vec<_>>();
            let y1 = (0..n).map(|_| next_sample()).collect::<Vec<_>>();
            let y2 = (0..n).map(|_| next_sample()).collect::<Vec<_>>();
            let mut lanes1 = [0.0f32; 4];
            let mut lanes2 = [0.0f32; 4];
            let vector_end = n / 4 * 4;
            for i in 0..vector_end {
                let lane = i % 4;
                #[cfg(target_arch = "aarch64")]
                {
                    lanes1[lane] = x[i].mul_add(y1[i], lanes1[lane]);
                    lanes2[lane] = x[i].mul_add(y2[i], lanes2[lane]);
                }
                #[cfg(target_arch = "x86_64")]
                {
                    lanes1[lane] += x[i] * y1[i];
                    lanes2[lane] += x[i] * y2[i];
                }
            }
            let mut expected1 = (lanes1[0] + lanes1[2]) + (lanes1[1] + lanes1[3]);
            let mut expected2 = (lanes2[0] + lanes2[2]) + (lanes2[1] + lanes2[3]);
            for i in vector_end..n {
                #[cfg(target_arch = "aarch64")]
                {
                    expected1 = x[i].mul_add(y1[i], expected1);
                    expected2 = x[i].mul_add(y2[i], expected2);
                }
                #[cfg(target_arch = "x86_64")]
                {
                    expected1 += x[i] * y1[i];
                    expected2 += x[i] * y2[i];
                }
            }

            let actual = super::dual_inner_prod_f32(&x, &y1, &y2, n).unwrap();
            assert_eq!(actual.0.to_bits(), expected1.to_bits(), "first, n={n}");
            assert_eq!(actual.1.to_bits(), expected2.to_bits(), "second, n={n}");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn sse2_pvq_candidate_stays_within_the_valid_slice() {
        let mut seed = 0xC001_C0DEu32;
        for n in 1..=64 {
            let mut next_value = || {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (seed >> 8) as f32 * (1.0 / (u32::MAX >> 8) as f32)
            };
            let x = (0..n).map(|_| next_value()).collect::<Vec<_>>();
            let y = (0..n).map(|_| 2.0 * next_value() + 1.0).collect::<Vec<_>>();
            let best = super::pvq_best_candidate_sse2(&x, &y, 3.0, 4.0, n);
            assert!(best < n, "n={n}, best={best}");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_pitch_xcorr_matches_scalar_reference() {
        if !std::arch::is_x86_feature_detected!("avx2")
            || !std::arch::is_x86_feature_detected!("fma")
        {
            return;
        }
        let mut seed = 0x5EED_1234u32;
        for len in [1usize, 2, 3, 4, 5, 6, 7, 8, 9, 31, 60, 61] {
            for max_pitch in [8usize, 16, 24, 240] {
                let mut next_sample = || {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (seed as i32 as f32) * (0.75 / i32::MAX as f32)
                };
                let x = (0..len).map(|_| next_sample()).collect::<Vec<_>>();
                let y = (0..len + max_pitch - 1)
                    .map(|_| next_sample())
                    .collect::<Vec<_>>();
                let mut expected = vec![0.0f32; max_pitch];
                for lag in 0..max_pitch {
                    for i in 0..len {
                        expected[lag] = x[i].mul_add(y[lag + i], expected[lag]);
                    }
                }

                let mut actual = vec![f32::NAN; max_pitch];
                assert!(super::pitch_xcorr_8(&x, &y, len, max_pitch, &mut actual));
                for index in 0..max_pitch {
                    let tolerance = 2e-5 * expected[index].abs().max(1.0);
                    assert!(
                        (actual[index] - expected[index]).abs() <= tolerance,
                        "len={len}, max_pitch={max_pitch}, index={index}: {} != {}",
                        actual[index],
                        expected[index]
                    );
                }
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_pitch_xcorr_matches_fused_scalar_reference() {
        let mut seed = 0x5EED_1234u32;
        for len in [1usize, 4, 5, 8, 9, 31, 60, 61] {
            for max_pitch in [4usize, 8, 12, 244] {
                let mut next_sample = || {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (seed as i32 as f32) * (0.75 / i32::MAX as f32)
                };
                let x = (0..len).map(|_| next_sample()).collect::<Vec<_>>();
                let y = (0..len + max_pitch - 1)
                    .map(|_| next_sample())
                    .collect::<Vec<_>>();
                let mut expected = vec![0.0f32; max_pitch];
                for lag in 0..max_pitch {
                    for i in 0..len {
                        expected[lag] = x[i].mul_add(y[lag + i], expected[lag]);
                    }
                }

                let mut actual = vec![f32::NAN; max_pitch];
                assert!(super::pitch_xcorr_4(&x, &y, len, max_pitch, &mut actual));
                for index in 0..max_pitch {
                    assert_eq!(
                        actual[index].to_bits(),
                        expected[index].to_bits(),
                        "len={len}, max_pitch={max_pitch}, index={index}"
                    );
                }
            }
        }
    }
}
