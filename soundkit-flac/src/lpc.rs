// Copyright 2022-2024 Google LLC
// Copyright 2025- flacenc-rs developers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Algorithms for quantized linear-prediction coding (QLPC).

use std::collections::BTreeMap;
use std::rc::Rc;

use num_traits::AsPrimitive;
use num_traits::Float;

use super::arrayutils::find_max_abs;
use super::arrayutils::unaligned_map_and_update;
use super::arrayutils::SimdVec;
use super::component::QuantizedParameters;
use super::config::Window;
use super::constant::panic_msg;
use super::constant::qlpc::MAX_ORDER as MAX_LPC_ORDER;
use super::constant::qlpc::MAX_SHIFT as QLPC_MAX_SHIFT;
use super::constant::qlpc::MIN_SHIFT as QLPC_MIN_SHIFT;
use super::repeat;
use super::repeat::repeat;

import_simd!(as simd);

/// Trait for a type that can be used for storing LPC statistics/ parameters.
///
/// Currently, it is only implemented for f32/ f64.
#[allow(clippy::module_name_repetitions)]
pub trait LpcFloat:
    Float
    + std::ops::AddAssign
    + std::ops::MulAssign
    + std::iter::Sum
    + std::fmt::Debug
    + std::fmt::Display
    + simd::SimdElement
    + simd::SimdCast
    + From<f32>
    + From<i16>
    + AsPrimitive<f32>
    + AsPrimitive<i16>
{
    #[allow(dead_code)]
    type Simd<const N: usize>: SimdFloat<Scalar = Self>
        + StdFloat
        + Copy
        + From<simd::Simd<Self, N>>;

    /// Solves symetric positive-definite linear equation in-place.
    ///
    /// This computes `v = matmul(inverse(mat), v)` where `mat` is assumed to be
    /// symmetric positive-definite (SPD), and if not it returns `false`.
    /// Otherwise, it returns `true` and `v` is overwritten by the solution.
    #[allow(dead_code)]
    #[cfg(feature = "experimental")]
    fn solve_sym_mut(mat: &nalgebra::DMatrix<Self>, v: &mut nalgebra::DVector<Self>) -> bool;
}

macro_rules! def_lpc_float {
    ($ty:ty) => {
        impl self::LpcFloat for $ty {
            type Simd<const N: usize> = simd::Simd<$ty, N>;

            #[cfg(feature = "experimental")]
            #[inline]
            fn solve_sym_mut(
                mat: &nalgebra::DMatrix<Self>,
                v: &mut nalgebra::DVector<Self>,
            ) -> bool {
                mat.clone().cholesky().map_or(false, |decompose| {
                    decompose.solve_mut(v);
                    true
                })
            }
        }
    };
}
def_lpc_float!(f32);
def_lpc_float!(f64);

/// Precomputes window function given the window config `win`.
#[inline]
pub fn window_weights(win: &Window, len: usize) -> Vec<f32> {
    match *win {
        Window::Rectangle => vec![1.0f32; len],
        Window::Tukey { alpha: 0.0 } => {
            vec![1.0f32; len]
        }
        Window::Tukey { alpha } => {
            let max_t = len as f32 - 1.0;
            let alpha_len = alpha * max_t;
            let mut ret = Vec::with_capacity(len);
            for t in 0..len {
                let t = t as f32;
                let w = if t < alpha_len / 2.0 {
                    0.5 * (1.0 - (2.0 * std::f32::consts::PI * t / alpha_len).cos())
                } else if t < max_t - alpha_len / 2.0 {
                    1.0
                } else {
                    0.5 * (1.0 - (2.0 * std::f32::consts::PI * (max_t - t) / alpha_len).cos())
                };
                ret.push(w);
            }
            ret
        }
    }
}

/// Quantizes and fingerprints the window function for caching.
fn fingerprint_window(w: &Window) -> u64 {
    match *w {
        Window::Rectangle => 0x01_00_00_00_00_00_00_00u64,
        Window::Tukey { alpha } => {
            let qalpha = (alpha * 65535.0) as u64;
            assert!(qalpha < 65536, "alpha is larger than 1");
            0x02_00_00_00_00_00_00_00u64 + qalpha
        }
    }
}

/// A struct used for indexing window cache.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct WindowKey {
    /// Size of the window cache.
    size: usize,
    /// A fingerprint computed from window-specific hyper-parameters.
    fingerprint: u64,
}

impl WindowKey {
    /// Constructs `WindowKey` with given window size and parameters.
    fn new(size: usize, params: &Window) -> Self {
        Self {
            size,
            fingerprint: fingerprint_window(params),
        }
    }
}

/// Trait for a weighting function when collecting the second order statistics.
///
/// It is only interesting in "experimental" build, so far, only `NoWeight` is used
/// in non-experimental build.
pub trait Weight {
    /// Apply weight to a sample `x` at time-offset `t`.
    fn apply(&self, t: usize, x: f32) -> f32;
    /// Apply weights to a vector of samples `x` starting at time-offset `t0`.
    #[allow(dead_code)]
    fn apply_simd<const N: usize>(&self, t0: usize, x: simd::Simd<f32, N>) -> simd::Simd<f32, N>;
}

struct NoWeight;
#[cfg(feature = "experimental")]
struct VecWeight(Vec<f32>);
#[cfg(feature = "experimental")]
struct ShiftedWeight<const M: usize, W: Weight>(W);

impl<W: Weight> Weight for &W {
    #[inline]
    fn apply(&self, t: usize, x: f32) -> f32 {
        (*self).apply(t, x)
    }
    #[inline]
    fn apply_simd<const N: usize>(&self, t0: usize, x: simd::Simd<f32, N>) -> simd::Simd<f32, N> {
        (*self).apply_simd(t0, x)
    }
}

impl Weight for NoWeight {
    #[inline]
    fn apply(&self, _t: usize, x: f32) -> f32 {
        x
    }
    #[inline]
    fn apply_simd<const N: usize>(&self, _t0: usize, x: simd::Simd<f32, N>) -> simd::Simd<f32, N> {
        x
    }
}

#[cfg(feature = "experimental")]
impl Weight for VecWeight {
    #[inline]
    fn apply(&self, t: usize, x: f32) -> f32 {
        self.0[t] * x
    }
    #[inline]
    fn apply_simd<const N: usize>(&self, t0: usize, x: simd::Simd<f32, N>) -> simd::Simd<f32, N> {
        x * simd::Simd::<f32, N>::from_slice(&self.0[t0..(t0 + N)])
    }
}

#[cfg(feature = "experimental")]
impl<W: Weight, const M: usize> Weight for ShiftedWeight<M, W> {
    #[inline]
    fn apply(&self, t: usize, x: f32) -> f32 {
        self.0.apply(t + M, x)
    }
    #[inline]
    fn apply_simd<const N: usize>(&self, t0: usize, x: simd::Simd<f32, N>) -> simd::Simd<f32, N> {
        self.0.apply_simd(t0 + M, x)
    }
}

const QLPC_WIN_SIMD_N: usize = 16;
type WindowMap = BTreeMap<WindowKey, Rc<SimdVec<f32, QLPC_WIN_SIMD_N>>>;
reusable!(WINDOW_CACHE: WindowMap);

/// Gets the window function for the given config and size.
fn get_window(window: &Window, size: usize) -> Rc<SimdVec<f32, QLPC_WIN_SIMD_N>> {
    let key = WindowKey::new(size, window);
    reuse!(WINDOW_CACHE, |caches: &mut WindowMap| {
        if caches.get(&key).is_none() {
            let v = window_weights(window, size);
            caches.insert(key.clone(), Rc::from(SimdVec::from_slice(&v)));
        }
        Rc::clone(caches.get(&key).expect(panic_msg::ERROR_NOT_EXPECTED))
    })
}

/// Finds shift parameter for quantizing the given set of coefficients.
fn find_shift<T>(coefs: &[T], precision: usize) -> i8
where
    T: LpcFloat,
{
    assert!(precision <= 15);
    assert!(!coefs.is_empty());
    let max_abs_coef: T = coefs
        .iter()
        .copied()
        .map(Float::abs)
        .reduce(T::max)
        .unwrap();
    // location of MSB in binary representations of absolute values.
    let abs_log2: i16 = Float::max(
        Float::ceil(Float::log2(max_abs_coef)),
        <T as From<i16>>::from(i16::MIN + 16),
    )
    .as_();
    let shift: i16 = (precision as i16 - 1) - abs_log2;
    shift.clamp(i16::from(QLPC_MIN_SHIFT), i16::from(QLPC_MAX_SHIFT)) as i8
}

/// Quantizes LPC parameter with the given shift parameter.
#[inline]
fn quantize_parameter<T>(p: T, shift: i8) -> i16
where
    T: LpcFloat,
{
    let scalefac = Float::powi(<T as From<i16>>::from(2), i32::from(shift));
    let scaled_int = Float::round(p * scalefac);
    num_traits::clamp(
        scaled_int,
        <T as From<i16>>::from(i16::MIN),
        <T as From<i16>>::from(i16::MAX),
    )
    .as_()
}

/// Creates [`QuantizedParameters`] by quantizing the given coefficients.
pub fn quantize_parameters<T>(coefs: &[T], precision: usize) -> QuantizedParameters
where
    T: LpcFloat,
{
    if coefs.is_empty() {
        return QuantizedParameters::from_parts(&[], 0, 0, precision);
    }
    let shift = find_shift(coefs, precision);
    let mut q_coefs = [0i16; MAX_LPC_ORDER];

    for (n, coef) in coefs.iter().enumerate() {
        // This clamp op is mainly for safety, but actually required
        // because the shift-width estimation `find_shift` used here is not
        // perfect, and quantization may yields "2^(p-1)" quantized value
        // for precision "p" configuration, that is larger than a maximum
        // p-bits signed integer "2^(p-1) - 1".
        q_coefs[n] = std::cmp::min(
            std::cmp::max(quantize_parameter(*coef, shift), -(1 << (precision - 1))),
            (1 << (precision - 1)) - 1,
        );
    }

    let tail_zeros = q_coefs
        .rsplitn(2, |&x| x != 0)
        .next()
        .map_or(0, <[i16]>::len);
    let order = std::cmp::max(1, q_coefs.len() - tail_zeros);

    QuantizedParameters::from_parts(&q_coefs[0..order], order, shift, precision)
}

/// Implementation of `compute_error` for each SIMD config.
#[inline]
fn compute_error_impl<T, const N: usize>(qps: &QuantizedParameters, signal: &[T], errors: &mut [T])
where
    T: simd::SimdElement + num_traits::int::PrimInt + From<i8> + From<i16> + std::ops::AddAssign<T>,
    simd::Simd<T, N>: std::ops::Shr<simd::Simd<T, N>, Output = simd::Simd<T, N>>
        + std::ops::Sub<simd::Simd<T, N>, Output = simd::Simd<T, N>>
        + std::ops::Mul<simd::Simd<T, N>, Output = simd::Simd<T, N>>
        + std::ops::AddAssign<simd::Simd<T, N>>,
{
    let block_size = signal.len();
    debug_assert!(errors.len() >= block_size);
    errors.fill(T::zero());

    for order in 0..qps.order() {
        let w = qps.coefs[order].into();
        let wv = simd::Simd::<T, N>::splat(w);
        unaligned_map_and_update(
            &signal[0..block_size - order - 1],
            &mut errors[order + 1..],
            #[inline]
            |px, x| {
                *px += w * x;
            },
            #[inline]
            |pv, v| {
                *pv += wv * v;
            },
        );
    }

    let shift = qps.shift() as usize;
    let shift_v = simd::Simd::<T, N>::splat(qps.shift().into());
    unaligned_map_and_update::<T, N, _, _, _>(
        signal,
        errors,
        #[inline]
        |px, x| {
            *px = x - (*px >> shift);
        },
        #[inline]
        |pv, v| {
            *pv = v - (*pv >> shift_v);
        },
    );
    errors[0..qps.order()].fill(T::zero());
}

/// Compute error signal from `QuantizedParameters`.
///
/// # Panics
///
/// This function panics if `errors.len()` is smaller than `signal.len()`.
#[allow(clippy::collapsible_else_if)]
#[allow(clippy::needless_type_cast, reason = "false alarm")]
pub fn compute_error(qps: &QuantizedParameters, signal: &[i32], errors: &mut [i32]) {
    assert!(errors.len() >= signal.len());
    let maxabs_signal: u64 = find_max_abs::<16>(signal).into();
    // `Simd::reduce_sum` is avoided to mitigate overflow error.
    // NOTE: If we restrict the precision to be 11 bit, 24-additions of 11-bit
    //       ints are 16-bit safe. we assume it's reasonably fast.
    let sumabs_coefs: i64 = {
        let mut acc: i64 = 0i64;
        let abs_coefs = qps.coefs.abs();
        repeat!(lane to 32 => {
            acc += i64::from(abs_coefs.as_array()[lane]);
        });
        acc
    };
    let maxabs = maxabs_signal * sumabs_coefs as u64;
    if maxabs < i32::MAX as u64 {
        // larger lanes here can alleviate inefficiency of unaligned reads.
        compute_error_impl::<i32, 64>(qps, signal, errors);
    } else {
        // This is very inefficient, but should rarely happen in BPS=16bit case.
        let signal64: Vec<i64> = signal.iter().map(|v| (*v).into()).collect();
        let mut errors64 = vec![0i64; signal64.len()];
        compute_error_impl::<i64, 64>(qps, &signal64, &mut errors64);
        for (v, p) in errors64
            .into_iter()
            .map(|v| v as i32)
            .zip(errors.iter_mut())
        {
            *p = v;
        }
    }
}

/// Compute auto-correlation coefficients.
///
/// # Panics
///
/// Panics if the number of samples in `signal` is smaller than `order`.
#[allow(dead_code)]
pub fn auto_correlation<T: LpcFloat>(order: usize, signal: &[f32], dest: &mut [T]) {
    weighted_auto_correlation(order, signal, dest, NoWeight);
}

/// Computes the sum of outer products of lagged vectors.
///
/// # Panics
///
/// Panics if the number of samples in `signal` is smaller than `order`.
#[cfg(feature = "experimental")]
#[allow(dead_code)]
pub fn lagged_outer_prod_sum<T>(order: usize, signal: &[f32], dest: &mut nalgebra::DMatrix<T>)
where
    T: LpcFloat,
{
    weighted_lagged_outer_prod_sum(order, signal, dest, NoWeight);
}

/// Computes sum of `x[t] * y[t] * weight(t_offset + t)`s.
#[inline]
#[cfg(feature = "simd-nightly")]
fn weighted_prod_sum<T, W>(t_offset: usize, x: &[f32], y: &[f32], weight: W) -> T
where
    T: LpcFloat,
    W: Weight,
{
    let mut acc = T::zero();
    for (tau, (x, delayed_x)) in x.iter().copied().zip(y.iter().copied()).enumerate() {
        let wx = Into::<T>::into(weight.apply(t_offset + tau, x));
        acc = Float::mul_add(delayed_x.into(), wx, acc);
    }
    acc
}

/// Internal function that computes the sum of `signal[t] * signal[t-DELAY] * weight(t)`s.
///
/// This function takes arguments as const generics, and this necessitates us to have a
/// redundant parameter `LANES_MINUS_DELAY` which is assumed to be always `LANES - DELAY`.
/// This is due to a current limitation of constant computation in Rust.
#[cfg(feature = "simd-nightly")]
#[inline]
fn weighted_delay_prod_sum_impl<
    T,
    W,
    const LANES: usize,
    const DELAY: usize,
    const LANES_MINUS_DELAY: usize,
>(
    warm_up: usize,
    signal: &[f32],
    weight: W,
) -> T
where
    T: LpcFloat,
    W: Weight,
{
    assert!(DELAY <= LANES);
    assert_eq!(LANES_MINUS_DELAY, LANES - DELAY);
    let mut acc = T::zero();

    let delayed_signal = &signal[warm_up - DELAY..];
    let (head, body, foot) = signal[warm_up..].as_simd();
    let mut t_offset = warm_up;

    acc += weighted_prod_sum(t_offset, head, delayed_signal, &weight);
    t_offset += head.len();

    // this is a bit awkward to use f32 for `indices`, but this can reduce some complexity of
    // implementing `fakesimd::Mask::cast`. this is required for compilation even though this
    // loop is not actually used. We can resort conditional compilation as well, but conditional
    // compilation is also not very clean.
    let indices = simd::Simd::from_array(std::array::from_fn(|n| n as f32));
    let mask = indices.simd_lt(simd::Simd::splat(DELAY as f32));
    // ^ first `DELAY` lanes are true.

    let mut prev_v = simd::Simd::from_array(std::array::from_fn(|n| {
        if warm_up + n < LANES {
            0.0
        } else {
            signal.get(t_offset + n - LANES).copied().unwrap_or(0.0)
        }
    }));
    let mut acc_v: T::Simd<LANES> = simd::Simd::splat(T::zero()).into();
    for v in body.iter().copied::<simd::Simd<f32, LANES>>() {
        prev_v = mask.select(
            prev_v.rotate_elements_left::<LANES_MINUS_DELAY>(),
            v.rotate_elements_right::<DELAY>(),
        );
        let wv = weight.apply_simd(t_offset, v);

        acc_v = T::Simd::mul_add(wv.cast().into(), prev_v.cast().into(), acc_v);
        prev_v = v;
        t_offset += LANES; // this needs to be updated in each iteration since weight refers it.
    }

    acc += weighted_prod_sum(
        t_offset,
        foot,
        &delayed_signal[t_offset - warm_up..],
        &weight,
    );
    acc + acc_v.reduce_sum()
}

/// Compute weighted auto-correlation coefficients.
///
/// # Panics
///
/// Panics if the number of samples in `signal` is smaller than `order`.
#[cfg(feature = "simd-nightly")]
#[inline]
#[allow(clippy::cognitive_complexity)] // so far complexity is hidden by seq macros.
pub fn weighted_auto_correlation_simd<T, W>(order: usize, signal: &[f32], dest: &mut [T], weight: W)
where
    T: LpcFloat,
    W: Weight,
{
    let warmup = order - 1;
    let weight = &weight;
    seq_macro::seq!(DELAY in 0..=32 {
        #[allow(clippy::unnecessary_semicolon)]
        if DELAY < order {
            // `LANES` is starting from 8.
            #[allow(clippy::identity_op)] // delay may be zero.
            #[allow(clippy::eq_op)] // delay may be 0x07.
            const LANES: usize = usize::next_power_of_two((DELAY | 0x07) - 1);
            #[allow(clippy::identity_op)] // delay may be zero.
            const LANES_MINUS_DELAY: usize = LANES - DELAY;
            dest[DELAY] = weighted_delay_prod_sum_impl::<
                T, _, LANES, DELAY, LANES_MINUS_DELAY
            >(warmup, signal, weight);
        }
    });
}

pub fn weighted_auto_correlation_nosimd<T, W>(
    order: usize,
    signal: &[f32],
    dest: &mut [T],
    weight: W,
) where
    T: LpcFloat,
    W: Weight,
{
    for t in (order - 1)..signal.len() {
        let wy: T = weight.apply(t, signal[t]).into();
        repeat!(tau to { MAX_LPC_ORDER + 1 } ; while tau < order => {
            dest[tau] = Float::mul_add(Into::<T>::into(signal[t - tau]), wy, dest[tau]);
        });
    }
}

/// Computes auto-correlation function up to `order`.
pub fn weighted_auto_correlation<T, W>(order: usize, signal: &[f32], dest: &mut [T], weight: W)
where
    T: LpcFloat,
    W: Weight,
{
    assert!(dest.len() >= order);
    for p in &mut *dest {
        *p = T::zero();
    }
    #[cfg(feature = "simd-nightly")]
    weighted_auto_correlation_simd(order, signal, dest, weight);
    #[cfg(not(feature = "simd-nightly"))]
    weighted_auto_correlation_nosimd(order, signal, dest, weight);
}

/// Compute weighted lagged-outer-prod-sum statistics.
///
/// # Panics
///
/// Panics if the number of samples in `signal` is smaller than `order`.
#[cfg(feature = "experimental")]
#[inline]
pub fn weighted_lagged_outer_prod_sum<T, W>(
    order: usize,
    signal: &[f32],
    dest: &mut nalgebra::DMatrix<T>,
    weight: W,
) where
    W: Weight,
    T: LpcFloat,
{
    assert!(dest.ncols() >= order);
    assert!(dest.nrows() >= order);

    dest.fill(T::zero());

    for t in (order - 1)..signal.len() {
        for i in 0..order {
            for j in i..order {
                let wx = Into::<T>::into(weight.apply(t, signal[t - j]));
                dest[(i, j)] = Float::mul_add(signal[t - i].into(), wx, dest[(i, j)]);
            }
        }
    }
    for i in 0..order {
        for j in (i + 1)..order {
            dest[(j, i)] = dest[(i, j)];
        }
    }
}

/// Computes raw errors from unquantized LPC coefficients.
///
/// This function computes "prediction - signal" in floating-point numbers.
#[allow(dead_code)] // Used either in experimental or tests of non-experimental.
fn compute_raw_errors<T>(signal: &[i32], lpc_coefs: &[T], errors: &mut [f32])
where
    T: LpcFloat,
{
    let lpc_order = lpc_coefs.len();
    for t in lpc_order..signal.len() {
        errors[t] = -signal[t] as f32;
        for j in 0..lpc_order {
            let coef: f32 = lpc_coefs[j].as_();
            errors[t] = coef.mul_add(signal[t - 1 - j] as f32, errors[t]);
        }
    }
}

/// Solves "y = T x" where T is a Toeplitz matrix with the given coefficients.
///
/// The (i, j)-th element of the Toeplitz matrix "T" is defined by
/// `coefs[(i - j).abs()]`, and the i-th element of "y" is defined as `ys[i]`.
/// The solution "x" will be stored in `dest`.
///
/// # Panics
///
/// Panics if `dest` or `coefs` is shorter than `ys`. In addition to that,
/// the following preconditions are checked.
/// 1. Signal energy `coefs[0]` is non-negative.
/// 2. If signal-energy is zero, all `coefs` and `ys` must be zero.
#[inline]
pub fn symmetric_levinson_recursion<T, const N: usize>(coefs: &[T], ys: &[T], dest: &mut [T])
where
    T: LpcFloat,
    repeat::Count<N>: repeat::Repeat,
{
    assert!(dest.len() >= ys.len());
    assert!(coefs.len() >= ys.len());

    for p in &mut *dest {
        *p = T::zero();
    }

    // coefs[0] is energy of the signal, so must be non-negative.
    assert!(coefs[0] >= T::zero());
    if coefs[0].is_zero() {
        let allzero = ys
            .iter()
            .chain(coefs.iter())
            .fold(true, |f, &v| f & v.is_zero());
        assert!(
            allzero,
            "If signal is digital silence, all coefficients must be zero."
        );
        return;
    }

    let order = ys.len();
    let mut forward = [T::zero(); N];
    let mut forward_next = [T::zero(); N];
    let mut diagonal_loading = T::zero();

    // this actually should use a go-to statement.
    #[allow(clippy::never_loop)]
    loop {
        forward[0] = Float::recip(coefs[0] + diagonal_loading);
        dest[0] = ys[0] / (coefs[0] + diagonal_loading);

        for n in 1..order {
            let error = {
                let mut acc = T::zero();
                repeat!(d to N ; while d < n => {
                    acc = Float::mul_add(coefs[n - d], forward[d], acc);
                });
                acc
            };
            let denom = Float::mul_add(error, -error, T::one());
            if denom.is_zero() {
                diagonal_loading = T::one().max(diagonal_loading + diagonal_loading);
                continue;
            }
            let alpha = Float::recip(denom);
            let beta = -alpha * error;
            repeat!(d to N ; while d <= n => {
                forward_next[d] = Float::mul_add(alpha, forward[d], beta * forward[n - d]);
            });
            repeat!(d to N ; while d <= n => {
                forward[d] = forward_next[d];
            });

            let delta = {
                let mut acc = T::zero();
                repeat!(d to N ; while d < n => {
                    acc = Float::mul_add(coefs[n - d], dest[d], acc);
                });
                acc
            };
            repeat!(d to N ; while d <= n => {
                dest[d] = Float::mul_add(ys[n] - delta, forward[n - d], dest[d]);
            });
        }
        break;
    }
}

/// Working buffer for (unquantized) LPC estimation.
struct LpcEstimator<T> {
    /// Buffer for storing windowed signal.
    windowed_signal: SimdVec<f32, QLPC_WIN_SIMD_N>,
    /// Buffer for storing auto-correlation coefficients.
    corr_coefs: Vec<T>,
    /// Buffer for delay-sum matrix and it's inverse. (not used in auto-correlation mode.)
    #[cfg(feature = "experimental")]
    lagged_outer_prod_sum: nalgebra::DMatrix<T>,
    /// Weights for IRLS.
    #[cfg(feature = "experimental")]
    weights: Vec<f32>,
}

reusable!(CAST_BUFFER: SimdVec<i32, QLPC_WIN_SIMD_N> = SimdVec::new());

impl<T> LpcEstimator<T>
where
    T: LpcFloat,
{
    pub fn new() -> Self {
        Self {
            windowed_signal: SimdVec::new(),
            corr_coefs: vec![],
            #[cfg(feature = "experimental")]
            lagged_outer_prod_sum: nalgebra::DMatrix::zeros(MAX_LPC_ORDER, MAX_LPC_ORDER),
            #[cfg(feature = "experimental")]
            weights: vec![],
        }
    }

    #[allow(clippy::identity_op)] // false-alarm when OFFSET == 0
    fn fill_windowed_signal(
        &mut self,
        signal: &[i32],
        window: &[simd::Simd<f32, QLPC_WIN_SIMD_N>],
    ) {
        debug_assert!(window.len() * QLPC_WIN_SIMD_N >= signal.len());
        reuse!(CAST_BUFFER, |cast_buf: &mut SimdVec<
            i32,
            QLPC_WIN_SIMD_N,
        >| {
            cast_buf.reset_from_slice(signal);

            self.windowed_signal.reset_from_iter_simd(
                signal.len(),
                cast_buf.iter_simd().zip(window).map(|(s, w)| s.cast() * *w),
            );
        });
    }

    /// Performs weighted LPC via auto-correlation coefficients.
    #[allow(clippy::range_plus_one)]
    pub fn weighted_lpc_from_auto_corr<W>(
        &mut self,
        signal: &[i32],
        window: &Window,
        lpc_order: usize,
        weight: W,
    ) -> heapless::Vec<T, MAX_LPC_ORDER>
    where
        W: Weight,
    {
        let mut ret = heapless::Vec::new();
        if lpc_order == 0 {
            return ret;
        }
        ret.resize(lpc_order, T::zero())
            .expect("INTERNAL ERROR: lpc_order specified exceeded max.");
        self.corr_coefs.resize(lpc_order + 1, T::zero());
        self.corr_coefs.fill(T::zero());
        self.fill_windowed_signal(signal, get_window(window, signal.len()).as_ref_simd());

        weighted_auto_correlation(
            lpc_order + 1,
            self.windowed_signal.as_ref(),
            &mut self.corr_coefs,
            weight,
        );
        for &v in &self.corr_coefs {
            assert!(
                !(v.is_nan() || v.is_infinite()),
                "corr_coefs[_] = {v} must be normal or zero."
            );
        }
        symmetric_levinson_recursion::<T, MAX_LPC_ORDER>(
            &self.corr_coefs[0..lpc_order],
            &self.corr_coefs[1..lpc_order + 1],
            &mut ret,
        );
        for &v in &ret {
            assert!(!(v.is_nan() || v.is_infinite()));
        }
        ret
    }

    pub fn lpc_from_auto_corr(
        &mut self,
        signal: &[i32],
        window: &Window,
        lpc_order: usize,
    ) -> heapless::Vec<T, MAX_LPC_ORDER> {
        self.weighted_lpc_from_auto_corr(signal, window, lpc_order, NoWeight)
    }

    /// Optimizes LPC with Mean-Absolute-Error criterion.
    #[cfg(feature = "experimental")]
    pub fn lpc_with_irls_mae(
        &mut self,
        signal: &[i32],
        window: &Window,
        lpc_order: usize,
        steps: usize,
    ) -> heapless::Vec<T, MAX_LPC_ORDER> {
        self.weights.clear();
        self.weights.resize(signal.len(), 1.0f32);
        let mut raw_errors = vec![0.0f32; signal.len()];
        let mut best_coefs = None;
        let mut best_error = f32::MAX;

        let normalizer = signal.iter().map(|x| x.abs()).max().unwrap() as f32;
        let weight_fn = |err: f32| (err.abs().max(1.0) / normalizer).max(0.01).powf(-1.2);

        for _t in 0..=steps {
            let coefs = self.weighted_lpc_with_direct_mse(
                signal,
                window,
                lpc_order,
                VecWeight(self.weights.clone()),
            );
            compute_raw_errors(signal, &coefs, &mut raw_errors);

            let sum_abs_err: f32 = raw_errors.iter().copied().map(f32::abs).sum::<f32>();
            if sum_abs_err < best_error {
                best_error = sum_abs_err;
                best_coefs = Some(coefs);
            }

            for (p, &err) in self.weights.iter_mut().zip(&raw_errors).skip(lpc_order) {
                *p = weight_fn(err);
            }
        }
        best_coefs.unwrap()
    }

    #[cfg(feature = "experimental")]
    fn weighted_lpc_with_direct_mse<W>(
        &mut self,
        signal: &[i32],
        window: &Window,
        lpc_order: usize,
        weight: W,
    ) -> heapless::Vec<T, MAX_LPC_ORDER>
    where
        W: Weight,
    {
        self.corr_coefs.resize(lpc_order + 1, T::zero());
        self.corr_coefs.fill(T::zero());

        self.fill_windowed_signal(signal, get_window(window, signal.len()).as_ref_simd());

        self.lagged_outer_prod_sum.fill(T::zero());
        self.lagged_outer_prod_sum
            .resize_mut(lpc_order, lpc_order, T::zero());

        weighted_auto_correlation_nosimd(
            lpc_order + 1,
            self.windowed_signal.as_ref(),
            &mut self.corr_coefs,
            &weight,
        );
        weighted_lagged_outer_prod_sum(
            lpc_order,
            &self.windowed_signal.as_ref()[0..self.windowed_signal.len() - 1],
            &mut self.lagged_outer_prod_sum,
            ShiftedWeight::<1, _>(weight),
        );

        let mut xy = nalgebra::DVector::<T>::from(self.corr_coefs[1..].to_vec());

        let mut regularizer = T::zero();
        while !T::solve_sym_mut(&self.lagged_outer_prod_sum, &mut xy) {
            let old_regularizer = regularizer;
            regularizer = T::one().max(regularizer + regularizer);
            for i in 0..lpc_order {
                self.lagged_outer_prod_sum[(i, i)] += regularizer - old_regularizer;
            }
        }

        let mut ret = heapless::Vec::new();
        ret.resize(lpc_order, T::zero())
            .expect("INTERNAL ERROR: lpc_order specified exceeded max.");
        for i in 0..lpc_order {
            ret[i] = xy[i];
        }
        ret
    }

    #[cfg(feature = "experimental")]
    fn lpc_with_direct_mse(
        &mut self,
        signal: &[i32],
        window: &Window,
        lpc_order: usize,
    ) -> heapless::Vec<T, MAX_LPC_ORDER> {
        self.weighted_lpc_with_direct_mse(signal, window, lpc_order, NoWeight)
    }
}

reusable!(LPC_ESTIMATOR: LpcEstimator<f64> = LpcEstimator::new());

/// Estimates LPC coefficients with auto-correlation method.
#[allow(clippy::module_name_repetitions)]
pub fn lpc_from_autocorr(
    signal: &[i32],
    window: &Window,
    lpc_order: usize,
) -> heapless::Vec<f64, MAX_LPC_ORDER> {
    LPC_ESTIMATOR.with(|estimator| {
        estimator
            .borrow_mut()
            .lpc_from_auto_corr(signal, window, lpc_order)
    })
}

/// Estimates LPC coefficients with direct MSE method.
#[allow(clippy::module_name_repetitions)]
#[cfg(feature = "experimental")]
pub fn lpc_with_direct_mse(
    signal: &[i32],
    window: &Window,
    lpc_order: usize,
) -> heapless::Vec<f64, MAX_LPC_ORDER> {
    LPC_ESTIMATOR.with(|estimator| {
        estimator
            .borrow_mut()
            .lpc_with_direct_mse(signal, window, lpc_order)
    })
}

#[allow(clippy::module_name_repetitions)]
#[cfg(not(feature = "experimental"))]
pub fn lpc_with_direct_mse(
    _signal: &[i32],
    _window: &Window,
    _lpc_order: usize,
) -> heapless::Vec<f64, MAX_LPC_ORDER> {
    unimplemented!("not built with \"experimental\" feature flag.")
}

/// Estimates LPC coefficients with IRLS-MAE method.
#[allow(clippy::module_name_repetitions)]
#[cfg(feature = "experimental")]
pub fn lpc_with_irls_mae(
    signal: &[i32],
    window: &Window,
    lpc_order: usize,
    steps: usize,
) -> heapless::Vec<f64, MAX_LPC_ORDER> {
    LPC_ESTIMATOR.with(|estimator| {
        estimator
            .borrow_mut()
            .lpc_with_irls_mae(signal, window, lpc_order, steps)
    })
}

#[allow(clippy::module_name_repetitions)]
#[cfg(not(feature = "experimental"))]
pub fn lpc_with_irls_mae(
    _signal: &[i32],
    _window: &Window,
    _lpc_order: usize,
    _steps: usize,
) -> heapless::Vec<f64, MAX_LPC_ORDER> {
    unimplemented!("not built with \"experimental\" feature flag.")
}
