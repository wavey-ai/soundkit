//! CELT pulse-vector CWRS coding, ported from the official Opus `celt/cwrs.c`.

use crate::celt::entropy::{RangeDecoder, RangeEncoder};
use crate::celt::mathops::ec_ilog;

pub const MAX_PSEUDO: usize = 40;
pub const LOG_MAX_PSEUDO: usize = 6;
pub const CELT_MAX_PULSES: usize = 128;

#[inline]
fn abs_i32_to_usize(x: i32) -> usize {
    x.unsigned_abs() as usize
}

#[inline]
fn unext(u: &mut [u32], len: usize, mut ui0: u32) {
    debug_assert!(len >= 2);
    let mut j = 1;
    loop {
        let ui1 = u[j].wrapping_add(u[j - 1]).wrapping_add(ui0);
        u[j - 1] = ui0;
        ui0 = ui1;
        j += 1;
        if j >= len {
            break;
        }
    }
    u[j - 1] = ui0;
}

#[inline]
fn uprev(u: &mut [u32], len: usize, mut ui0: u32) {
    debug_assert!(len >= 2);
    let mut j = 1;
    loop {
        let ui1 = u[j].wrapping_sub(u[j - 1]).wrapping_sub(ui0);
        u[j - 1] = ui0;
        ui0 = ui1;
        j += 1;
        if j >= len {
            break;
        }
    }
    u[j - 1] = ui0;
}

/// Official `get_pulses()` helper from `celt/rate.h`.
#[inline]
pub fn get_pulses(i: usize) -> usize {
    if i < 8 {
        i
    } else {
        (8 + (i & 7)) << ((i >> 3) - 1)
    }
}

/// Guaranteed-overestimate fractional `log2`, ported from the custom-mode
/// path in `celt/cwrs.c`.
pub fn log2_frac(mut val: u32, mut frac: i32) -> i32 {
    debug_assert!(val > 0);
    let mut l = ec_ilog(val);
    if val & val.wrapping_sub(1) != 0 {
        if l > 16 {
            val = ((val - 1) >> ((l - 16) as u32)) + 1;
        } else {
            val <<= (16 - l) as u32;
        }
        l = (l - 1) << frac;
        loop {
            let b = (val >> 16) as i32;
            l += b << frac;
            val = (val + b as u32) >> (b as u32);
            val = (val.wrapping_mul(val).wrapping_add(0x7fff)) >> 15;
            let old_frac = frac;
            frac -= 1;
            if old_frac <= 0 {
                break;
            }
        }
        l + i32::from(val > 0x8000)
    } else {
        (l - 1) << frac
    }
}

/// Computes `V(n,k)`, the number of PVQ/CWRS codewords for a band.
pub fn pvq_v(n: usize, k: usize) -> u32 {
    if k == 0 {
        return 1;
    }
    if n == 0 {
        return 0;
    }
    if n == 1 {
        return 2;
    }
    let mut u = vec![0u32; k + 2];
    ncwrs_urow(n, k, &mut u)
}

/// Fill the required-bit table used by custom modes.
pub fn get_required_bits(bits: &mut [i16], n: usize, maxk: usize, frac: i32) {
    assert!(maxk > 0);
    assert!(bits.len() > maxk);
    bits[0] = 0;
    if n == 1 {
        for bit in bits.iter_mut().take(maxk + 1).skip(1) {
            *bit = (1 << frac) as i16;
        }
        return;
    }

    let mut u = vec![0u32; maxk + 2];
    ncwrs_urow(n, maxk, &mut u);
    for k in 1..=maxk {
        bits[k] = log2_frac(u[k].wrapping_add(u[k + 1]), frac) as i16;
    }
}

/// Compute `V(n,k)` and row `U(n,0..=k+1)`.
///
/// On return `u[i]` contains `U(n,i)` for `i` in `0..=k+1`.
pub fn ncwrs_urow(n: usize, k: usize, u: &mut [u32]) -> u32 {
    assert!(n >= 2);
    assert!(k > 0);
    assert!(u.len() >= k + 2);

    u[0] = 0;
    u[1] = 1;
    for kk in 2..k + 2 {
        u[kk] = (kk as u32) * 2 - 1;
    }
    for _ in 2..n {
        unext(&mut u[1..], k + 1, 1);
    }
    u[k].wrapping_add(u[k + 1])
}

/// Return the pulse vector at codebook index `i`.
pub fn decode_index(n: usize, k: usize, mut i: u32, y: &mut [i32], u: &mut [u32]) -> i32 {
    assert!(n > 0);
    assert!(y.len() >= n);
    assert!(u.len() >= k + 2);

    if k == 0 {
        y[..n].fill(0);
        return 0;
    }
    if n == 1 {
        y[0] = if i != 0 { -(k as i32) } else { k as i32 };
        return (k * k) as i32;
    }

    let mut k = k;
    let mut yy = 0i32;
    for yj_out in y.iter_mut().take(n) {
        let mut p = u[k + 1];
        let s = if i >= p { -1 } else { 0 };
        if s != 0 {
            i = i.wrapping_sub(p);
        }

        let mut yj = k as i32;
        p = u[k];
        while p > i {
            k -= 1;
            p = u[k];
        }
        i = i.wrapping_sub(p);
        yj -= k as i32;

        let val = (yj + s) ^ s;
        *yj_out = val;
        yy += val * val;
        uprev(u, k + 2, 0);
    }
    yy
}

/// Return the codebook index and codeword count for a pulse vector.
pub fn encode_index(n: usize, k_total: usize, y: &[i32], u: &mut [u32]) -> (u32, u32) {
    assert!(n > 0);
    assert!(y.len() >= n);
    assert!(u.len() >= k_total + 2);

    if k_total == 0 {
        return (0, 1);
    }
    if n == 1 {
        return (u32::from(y[0] < 0), 2);
    }

    u[0] = 0;
    for k in 1..=k_total + 1 {
        u[k] = (k as u32) * 2 - 1;
    }

    let mut k = abs_i32_to_usize(y[n - 1]);
    let mut i = u32::from(y[n - 1] < 0);
    let mut j = n - 2;
    i = i.wrapping_add(u[k]);
    k += abs_i32_to_usize(y[j]);
    if y[j] < 0 {
        i = i.wrapping_add(u[k + 1]);
    }

    while j > 0 {
        j -= 1;
        unext(u, k_total + 2, 0);
        i = i.wrapping_add(u[k]);
        k += abs_i32_to_usize(y[j]);
        if y[j] < 0 {
            i = i.wrapping_add(u[k + 1]);
        }
    }

    (i, u[k].wrapping_add(u[k + 1]))
}

/// Encode a pulse vector with the CELT range encoder.
pub fn encode_pulses(y: &[i32], n: usize, k: usize, enc: &mut RangeEncoder) {
    debug_assert!(y.len() >= n);
    debug_assert_eq!(
        y.iter()
            .take(n)
            .map(|v| abs_i32_to_usize(*v))
            .sum::<usize>(),
        k
    );
    if k == 0 {
        return;
    }
    let mut u = vec![0u32; k + 2];
    let (i, nc) = encode_index(n, k, y, &mut u);
    enc.encode_uint(i, nc);
}

/// Decode a pulse vector with the CELT range decoder.
pub fn decode_pulses(y: &mut [i32], n: usize, k: usize, dec: &mut RangeDecoder) -> i32 {
    debug_assert!(y.len() >= n);
    if k == 0 {
        y[..n].fill(0);
        return 0;
    }
    if n == 1 {
        let i = dec.decode_uint(2);
        y[0] = if i != 0 { -(k as i32) } else { k as i32 };
        return (k * k) as i32;
    }
    let mut u = vec![0u32; k + 2];
    let nc = ncwrs_urow(n, k, &mut u);
    let i = dec.decode_uint(nc);
    decode_index(n, k, i, y, &mut u)
}
