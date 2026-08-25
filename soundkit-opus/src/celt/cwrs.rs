//! CELT pulse-vector CWRS coding, ported from the official Opus `celt/cwrs.c`.
//!
//! Indexing and unranking use a compile-time table of `U(n, k)` values, the
//! same approach as FFmpeg's `ff_celt_pvq_u_row`. The table removes the
//! O(n * k) row walks (`unext`/`uprev`) and the per-call row buffers from the
//! hot encode/decode paths. Inputs outside the table range fall back to the
//! original walk-based implementation.

use crate::celt::entropy::{RangeDecoder, RangeEncoder};
use crate::celt::mathops::ec_ilog;

pub const MAX_PSEUDO: usize = 40;
pub const LOG_MAX_PSEUDO: usize = 6;
pub const CELT_MAX_PULSES: usize = 128;
const CWRS_ROW_CACHE_WAYS: usize = 4;
const CWRS_ROW_CACHE_SETS: usize = 32;
const CWRS_ROW_CACHE_LIMIT: usize = CWRS_ROW_CACHE_WAYS * CWRS_ROW_CACHE_SETS;

/// Largest band dimension in standard 48 kHz modes (22 samples * 8 at LM=3).
const TABLE_N_MAX: usize = 176;
/// Largest pseudo-pulse count plus one slot for `U(n, k + 1)` lookups.
const TABLE_K_MAX: usize = 129;
const TABLE_K_LEN: usize = TABLE_K_MAX + 1;
const TABLE_ROWS_LEN: usize = (TABLE_N_MAX + 1) * TABLE_K_LEN;

struct PvqRowTable {
    /// `rows[n * TABLE_K_LEN + k]` holds `U(n, k)` with `U(n, 0) == 0` and
    /// `U(1, k) == 1` for `k >= 1`. Row `n > 2` follows the `ncwrs_urow`
    /// recurrence `U(n, j) = U(n-1, j) + U(n-1, j-1) + U(n, j-1)`.
    ///
    /// Values past the point where `U(n, k)` exceeds `u32::MAX` wrap around
    /// and stop increasing. `strict_len[n]` is the length of the leading
    /// strictly-increasing prefix.
    rows: [u32; TABLE_ROWS_LEN],
    strict_len: [usize; TABLE_N_MAX + 1],
}

const fn build_pvq_rows() -> PvqRowTable {
    let mut rows = [0u32; TABLE_ROWS_LEN];
    let mut strict_len = [0usize; TABLE_N_MAX + 1];
    let mut k = 1usize;
    while k < TABLE_K_LEN {
        rows[TABLE_K_LEN + k] = 1;
        rows[2 * TABLE_K_LEN + k] = 2 * k as u32 - 1;
        k += 1;
    }
    strict_len[1] = 2;
    strict_len[2] = TABLE_K_LEN;
    let mut n = 3usize;
    while n <= TABLE_N_MAX {
        let base = n * TABLE_K_LEN;
        let prev_base = base - TABLE_K_LEN;
        let mut carry = 1u32;
        rows[base] = 0;
        rows[base + 1] = 1;
        let mut j = 2usize;
        while j < TABLE_K_LEN {
            carry = carry
                .wrapping_add(rows[prev_base + j])
                .wrapping_add(rows[prev_base + j - 1]);
            rows[base + j] = carry;
            j += 1;
        }
        let mut len = 2usize;
        while len < TABLE_K_LEN && rows[base + len] > rows[base + len - 1] {
            len += 1;
        }
        strict_len[n] = len;
        n += 1;
    }
    PvqRowTable { rows, strict_len }
}

static PVQ_ROWS: PvqRowTable = build_pvq_rows();

#[inline]
fn pvq_rows() -> &'static PvqRowTable {
    &PVQ_ROWS
}

/// Borrow the raw table row for `dims` remaining dimensions.
///
/// Callers doing repeated lookups for one band should grab this once instead
/// of going through [`table_u`] per probe.
#[inline]
fn pvq_row_for_dims(table: &PvqRowTable, dims: usize) -> &[u32] {
    debug_assert!(dims <= TABLE_N_MAX);
    &table.rows[dims * TABLE_K_LEN..(dims + 1) * TABLE_K_LEN]
}

#[inline]
fn table_u(table: &PvqRowTable, a: usize, b: usize) -> u32 {
    debug_assert!(a <= TABLE_N_MAX);
    debug_assert!(b <= TABLE_K_MAX);
    let (dims, pulses) = if a >= b { (a, b) } else { (b, a) };
    table.rows[dims * TABLE_K_LEN + pulses]
}

#[inline]
fn table_in_range(n: usize, k: usize) -> bool {
    n >= 2 && n <= TABLE_N_MAX && k >= 1 && k + 1 <= TABLE_K_MAX
}

#[inline]
fn table_v(table: &PvqRowTable, n: usize, k: usize) -> u32 {
    debug_assert!(table_in_range(n, k));
    table_u(table, n, k).wrapping_add(table_u(table, n, k + 1))
}

#[derive(Clone, Debug, Default)]
pub struct CwrsDecodeCache {
    entries: Vec<CwrsRowCacheEntry>,
    replace_mask: u64,
}

#[derive(Clone, Debug, Default)]
struct CwrsRowCacheEntry {
    n: usize,
    k: usize,
    nc: u32,
    row: Vec<u32>,
}

impl CwrsDecodeCache {
    fn fill_row(&mut self, n: usize, k: usize, u: &mut Vec<u32>) -> u32 {
        debug_assert!(n >= 2);
        debug_assert!(k > 0);
        if u.len() < k + 2 {
            u.resize(k + 2, 0);
        }
        if self.entries.is_empty() {
            self.entries
                .resize_with(CWRS_ROW_CACHE_LIMIT, CwrsRowCacheEntry::default);
        }

        let set = cwrs_row_cache_set(n, k);
        let first = set * CWRS_ROW_CACHE_WAYS;
        for way in 0..CWRS_ROW_CACHE_WAYS {
            let entry = &self.entries[first + way];
            if entry.n == n && entry.k == k {
                u[..k + 2].copy_from_slice(&entry.row);
                return entry.nc;
            }
        }

        let nc = ncwrs_urow(n, k, &mut u[..k + 2]);
        let shift = set * 2;
        let way = ((self.replace_mask >> shift) & 3) as usize;
        let next_way = (way + 1) & (CWRS_ROW_CACHE_WAYS - 1);
        self.replace_mask = (self.replace_mask & !(3u64 << shift)) | ((next_way as u64) << shift);
        let entry = &mut self.entries[first + way];
        entry.n = n;
        entry.k = k;
        entry.nc = nc;
        entry.row.resize(k + 2, 0);
        entry.row.copy_from_slice(&u[..k + 2]);
        nc
    }
}

#[inline]
fn cwrs_row_cache_set(n: usize, k: usize) -> usize {
    let hash = n.wrapping_mul(31) ^ k.wrapping_mul(17);
    hash & (CWRS_ROW_CACHE_SETS - 1)
}

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
    if table_in_range(n, k) {
        return table_v(pvq_rows(), n, k);
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
pub fn decode_index(n: usize, k: usize, i: u32, y: &mut [i32], u: &mut [u32]) -> i32 {
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

    if let Some(yy) = fast_decode_index(pvq_rows(), n, k, i, y) {
        return yy;
    }
    walk_decode_index(n, k, i, y, u)
}

/// Original walk-based `decode_index`: rebuilds the shrinking `U(n, k)` row in
/// place with `uprev`. Kept as the fallback for inputs outside the table range
/// and as the differential reference for the table path.
fn walk_decode_index(n: usize, mut k: usize, mut i: u32, y: &mut [i32], u: &mut [u32]) -> i32 {
    let mut yy = 0i32;
    for pos in 0..n {
        let yj_out = &mut y[pos];
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
        if pos + 1 == n {
            break;
        }
        if k == 0 {
            y[pos + 1..n].fill(0);
            break;
        }
        uprev(u, k + 2, 0);
    }
    yy
}

/// Table-driven `decode_index` for inputs within the precomputed row range.
///
/// Mirrors the walk-based loop but reads `U(dims, pulses)` straight from the
/// static table instead of mutating a row with `uprev`.
fn fast_decode_index(
    table: &PvqRowTable,
    n: usize,
    k_total: usize,
    i: u32,
    y: &mut [i32],
) -> Option<i32> {
    debug_assert!(y.len() >= n);
    if !table_in_range(n, k_total) {
        return None;
    }

    if k_total + 1 >= table.strict_len[n] {
        return Some(table_walk_decode_index(n, k_total, i, y, &table.rows));
    }

    #[cfg(target_arch = "aarch64")]
    return crate::kernels::cwrs_decode_index_rect(&table.rows, TABLE_K_LEN, n, k_total, i, y);

    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut i = i;
        let mut dims = n;
        let mut k = k_total;
        let mut pos = 0usize;
        let mut yy = 0i32;
        let rows = &table.rows;

        // Match the non-small-footprint reference decoder's two regimes. In the
        // common K < N case a zero coefficient needs only two adjacent table
        // loads, and the final two dimensions have closed-form solutions. This
        // avoids a binary search for every zero in wide, low-pulse bands.
        while dims > 2 {
            let base = dims * TABLE_K_LEN;
            let row = &rows[base..base + TABLE_K_LEN];
            if k >= dims {
                let mut p = row[k + 1];
                let s = if i >= p { -1 } else { 0 };
                if s != 0 {
                    i = i.wrapping_sub(p);
                }
                let k0 = k;
                if row[dims] > i {
                    k = dims;
                    loop {
                        k -= 1;
                        p = row[k];
                        if p <= i {
                            break;
                        }
                    }
                } else {
                    p = row[k];
                    while p > i {
                        k -= 1;
                        p = row[k];
                    }
                }
                i = i.wrapping_sub(p);
                let magnitude = (k0 - k) as i32;
                let value = (magnitude + s) ^ s;
                y[pos] = value;
                yy += value * value;
            } else {
                let mut p = row[k];
                let q = row[k + 1];
                if p <= i && i < q {
                    i = i.wrapping_sub(p);
                    y[pos] = 0;
                } else {
                    let s = if i >= q { -1 } else { 0 };
                    if s != 0 {
                        i = i.wrapping_sub(q);
                    }
                    let k0 = k;
                    loop {
                        k -= 1;
                        p = row[k];
                        if p <= i {
                            break;
                        }
                    }
                    i = i.wrapping_sub(p);
                    let magnitude = (k0 - k) as i32;
                    let value = (magnitude + s) ^ s;
                    y[pos] = value;
                    yy += value * value;
                }
            }
            dims -= 1;
            pos += 1;
        }

        // Two-dimensional U() rows are linear: U(2, K) = 2*K - 1.
        let mut p = (2 * k + 1) as u32;
        let s = if i >= p { -1 } else { 0 };
        if s != 0 {
            i = i.wrapping_sub(p);
        }
        let k0 = k;
        k = ((i + 1) >> 1) as usize;
        if k != 0 {
            p = (2 * k - 1) as u32;
            i = i.wrapping_sub(p);
        }
        let magnitude = (k0 - k) as i32;
        let value = (magnitude + s) ^ s;
        y[pos] = value;
        yy += value * value;

        // The residual index is the sign of the final one-dimensional value.
        let s = if i != 0 { -1 } else { 0 };
        let value = (k as i32 + s) ^ s;
        y[pos + 1] = value;
        yy += value * value;

        Some(yy)
    }
}

/// Exact table analogue of [`walk_decode_index`] for wrapped U rows. Valid
/// standard-mode allocations use the faster monotonic-row path above.
fn table_walk_decode_index(
    n: usize,
    k_total: usize,
    mut i: u32,
    y: &mut [i32],
    rows: &[u32],
) -> i32 {
    let mut k = k_total;
    let mut yy = 0i32;
    for pos in 0..n {
        let dims = n - pos;
        let row = &rows[dims * TABLE_K_LEN..(dims + 1) * TABLE_K_LEN];
        let mut p = row[k + 1];
        let s = if i >= p { -1 } else { 0 };
        if s != 0 {
            i = i.wrapping_sub(p);
        }

        let mut magnitude = k as i32;
        p = row[k];
        while p > i {
            k -= 1;
            p = row[k];
        }
        i = i.wrapping_sub(p);
        magnitude -= k as i32;

        let value = (magnitude + s) ^ s;
        y[pos] = value;
        yy += value * value;
        if k == 0 {
            y[pos + 1..n].fill(0);
            break;
        }
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

    if let Some(result) = fast_encode_index(n, k_total, y) {
        return result;
    }
    walk_encode_index(n, k_total, y, u)
}

/// Original walk-based `encode_index`: grows the `U(n, k)` row in place with
/// `unext`. Fallback for inputs outside the table range and the differential
/// reference for the table path.
fn walk_encode_index(n: usize, k_total: usize, y: &[i32], u: &mut [u32]) -> (u32, u32) {
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

/// Table-driven `encode_index` for inputs within the precomputed row range.
///
/// Walks the pulse vector from the last coefficient forward, adding
/// `U(remaining dims, running sum)` plus a sign term per negative entry.
fn fast_encode_index(n: usize, k_total: usize, y: &[i32]) -> Option<(u32, u32)> {
    debug_assert!(y.len() >= n);
    if !table_in_range(n, k_total) {
        return None;
    }

    let table = pvq_rows();
    let mut idx: u32 = 0;
    let mut sum: usize = 0;
    for pos in (0..n).rev() {
        let dims = n - pos;
        let row = pvq_row_for_dims(table, dims);
        let ay = abs_i32_to_usize(y[pos]);
        idx = idx.wrapping_add(row[sum]);
        if y[pos] < 0 {
            idx = idx.wrapping_add(row[sum + ay + 1]);
        }
        sum += ay;
    }
    Some((idx, table_v(table, n, k_total)))
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
    let (i, nc) = match fast_encode_index(n, k, y) {
        Some(result) => result,
        None => {
            let mut u = vec![0u32; k + 2];
            encode_index(n, k, y, &mut u)
        }
    };
    enc.encode_uint(i, nc);
}

/// Decode a pulse vector with the CELT range decoder.
pub fn decode_pulses(y: &mut [i32], n: usize, k: usize, dec: &mut RangeDecoder<'_>) -> i32 {
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
    if table_in_range(n, k) {
        let table = pvq_rows();
        let nc = table_v(table, n, k);
        let i = dec.decode_uint(nc);
        if let Some(yy) = fast_decode_index(table, n, k, i, y) {
            return yy;
        }
        debug_assert!(false, "table path must cover its own inputs");
    }
    let mut u_stack = [0u32; CELT_MAX_PULSES + 2];
    let mut u_heap = Vec::new();
    let u = if k <= CELT_MAX_PULSES {
        &mut u_stack[..k + 2]
    } else {
        u_heap.resize(k + 2, 0);
        &mut u_heap[..]
    };
    let nc = ncwrs_urow(n, k, u);
    let i = dec.decode_uint(nc);
    decode_index(n, k, i, y, u)
}

/// Decode a pulse vector using caller-owned row scratch and a small `U(n,k)` row cache.
pub fn decode_pulses_with_cache(
    y: &mut [i32],
    n: usize,
    k: usize,
    dec: &mut RangeDecoder<'_>,
    u_scratch: &mut Vec<u32>,
    row_cache: &mut CwrsDecodeCache,
) -> i32 {
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
    if table_in_range(n, k) {
        let table = pvq_rows();
        let nc = table_v(table, n, k);
        let i = dec.decode_uint(nc);
        if let Some(yy) = fast_decode_index(table, n, k, i, y) {
            return yy;
        }
        debug_assert!(false, "table path must cover its own inputs");
    }
    let nc = row_cache.fill_row(n, k, u_scratch);
    let i = dec.decode_uint(nc);
    decode_index(n, k, i, y, &mut u_scratch[..k + 2])
}

#[cfg(test)]
mod pvq_table_tests {
    use super::*;

    fn sample_dims() -> Vec<usize> {
        vec![
            2, 3, 4, 6, 8, 9, 11, 12, 16, 18, 22, 24, 32, 36, 44, 48, 64, 72, 88, 96, 144, 176,
        ]
    }

    fn sample_ks() -> Vec<usize> {
        vec![1, 2, 3, 4, 6, 8, 12, 16, 26, 36, 88, 128]
    }

    #[test]
    fn table_rows_match_walk_generated_rows() {
        let table = pvq_rows();
        for n in sample_dims() {
            for k in sample_ks() {
                let mut u = vec![0u32; k + 2];
                ncwrs_urow(n, k, &mut u);
                for (j, &u_val) in u.iter().enumerate().take(k + 2) {
                    assert_eq!(
                        table.rows[n * TABLE_K_LEN + j],
                        u_val,
                        "U({n},{j}) diverges at K={k}"
                    );
                }
            }
        }
    }

    #[test]
    fn table_counts_match_small_reference() {
        const SMALL_V: [[u32; 10]; 10] = [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [1, 4, 8, 12, 16, 20, 24, 28, 32, 36],
            [1, 6, 18, 38, 66, 102, 146, 198, 258, 326],
            [1, 8, 32, 88, 192, 360, 608, 952, 1408, 1992],
            [1, 10, 50, 170, 450, 1002, 1970, 3530, 5890, 9290],
            [1, 12, 72, 292, 912, 2364, 5336, 10836, 20256, 35436],
            [1, 14, 98, 462, 1666, 4942, 12642, 28814, 59906, 115598],
            [1, 16, 128, 688, 2816, 9424, 27008, 68464, 157184, 332688],
            [1, 18, 162, 978, 4482, 16722, 53154, 148626, 374274, 864146],
        ];
        for (n, row) in SMALL_V.iter().enumerate() {
            for (k, expected) in row.iter().copied().enumerate() {
                assert_eq!(pvq_v(n, k), expected, "V({n},{k})");
            }
        }
    }

    #[test]
    fn fast_and_walk_agree_across_official_grid() {
        for n in sample_dims() {
            for k in sample_ks() {
                if k > get_pulses(MAX_PSEUDO) || !table_in_range(n, k) {
                    continue;
                }
                let mut u = vec![0u32; k + 2];
                let nc = ncwrs_urow(n, k, &mut u);
                let inc = (nc / 97).max(1);
                let mut i = 0u32;
                loop {
                    let mut y_fast = vec![0i32; n];
                    let yy_fast =
                        fast_decode_index(pvq_rows(), n, k, i, &mut y_fast).expect("in range");

                    let mut y_walk = vec![0i32; n];
                    let mut u_decode = u.clone();
                    let yy_walk = walk_decode_index(n, k, i, &mut y_walk, &mut u_decode);

                    assert_eq!(y_fast, y_walk, "decode y diverges N={n} K={k} i={i}");
                    assert_eq!(yy_fast, yy_walk, "decode norm diverges N={n} K={k} i={i}");

                    let (idx_fast, nc_fast) = fast_encode_index(n, k, &y_fast).expect("in range");
                    let mut u_encode = vec![0u32; k + 2];
                    let (idx_walk, nc_walk) = walk_encode_index(n, k, &y_fast, &mut u_encode);
                    assert_eq!(idx_fast, idx_walk, "encode idx diverges N={n} K={k}");
                    assert_eq!(idx_fast, i, "round trip diverges N={n} K={k} i={i}");
                    assert_eq!(nc_fast, nc_walk);
                    assert_eq!(nc_fast, nc);

                    if nc - i <= inc {
                        break;
                    }
                    i += inc;
                }
            }
        }
    }

    #[test]
    fn fast_paths_reject_out_of_range_inputs() {
        assert!(!table_in_range(1, 4));
        assert!(!table_in_range(TABLE_N_MAX + 1, 4));
        assert!(!table_in_range(16, TABLE_K_MAX));
        assert!(fast_encode_index(8, 8, &[8, 0, 0, 0, 0, 0, 0, 0]).is_some());
        assert!(fast_encode_index(TABLE_N_MAX + 1, 2, &vec![-1i32; TABLE_N_MAX + 1]).is_none());
    }
}
