// Vorbis decoder written in Rust
//
// Copyright (c) 2016 est31 <MTest31@outlook.com>
// and contributors. All rights reserved.
// Licensed under MIT license, or Apache 2 license,
// at your option. Please see the LICENSE file
// attached to this source distribution for details.
// Modified by SoundKit in 2026 for cached FFT transform data.

/*!
Cached header info

This mod contains logic to generate and deal with
data derived from header information
that's used later in the decode process.

The caching is done to speed up decoding.
*/

#[derive(Clone)]
pub struct CachedBlocksizeDerived {
    pub window_slope: Vec<f32>,
    pub fast_imdct_twiddle: Vec<[f32; 2]>,
    pub fast_fft_stages: Vec<Vec<[f32; 2]>>,
    pub fast_fft_bitrev: Vec<u16>,
}

impl CachedBlocksizeDerived {
    pub fn from_blocksize(bs: u8) -> Self {
        CachedBlocksizeDerived {
            window_slope: generate_window((1 << (bs as u16)) >> 1),
            fast_imdct_twiddle: compute_fast_imdct_twiddle(bs),
            fast_fft_stages: compute_fast_fft_stages(bs),
            fast_fft_bitrev: compute_fast_fft_bitrev(bs),
        }
    }
}

fn compute_fast_imdct_twiddle(blocksize: u8) -> Vec<[f32; 2]> {
    let spectral_len = (1usize << blocksize) >> 1;
    let fft_len = spectral_len >> 1;
    let angle_scale = std::f64::consts::PI / spectral_len as f64;
    (0..fft_len)
        .map(|index| {
            let angle = angle_scale * (0.125 + index as f64);
            [angle.cos() as f32, angle.sin() as f32]
        })
        .collect()
}

fn compute_fast_fft_stages(blocksize: u8) -> Vec<Vec<[f32; 2]>> {
    let fft_len = (1usize << blocksize) >> 2;
    let mut stages = Vec::new();
    let mut span = 8usize;
    while span <= fft_len {
        let angle_scale = 2.0 * std::f64::consts::PI / span as f64;
        stages.push(
            (0..span / 2)
                .map(|index| {
                    let angle = angle_scale * index as f64;
                    [angle.cos() as f32, -angle.sin() as f32]
                })
                .collect(),
        );
        span <<= 1;
    }
    stages
}

fn compute_fast_fft_bitrev(blocksize: u8) -> Vec<u16> {
    let fft_len = (1usize << blocksize) >> 2;
    let shift = fft_len.trailing_zeros();
    (0..fft_len)
        .map(|index| (index.reverse_bits() >> (usize::BITS - shift)) as u16)
        .collect()
}

fn win_slope(x: u16, n: u16) -> f32 {
    // please note that there might be a MISTAKE
    // in how the spec specifies the right window slope
    // function. See "4.3.1. packet type, mode and window decode"
    // step 7 where it adds an "extra" pi/2.
    // The left slope doesn't have it, only the right one.
    // as stb_vorbis shares the window slope generation function,
    // The *other* possible reason is that we don't need the right
    // window for anything. TODO investigate this more.
    let v = (0.5 * std::f32::consts::PI * (x as f32 + 0.5) / n as f32).sin();
    return (0.5 * std::f32::consts::PI * v * v).sin();
}

fn generate_window(n: u16) -> Vec<f32> {
    let mut window = Vec::with_capacity(n as usize);
    for i in 0..n {
        window.push(win_slope(i, n));
    }
    return window;
}

#[inline]
fn bark(x: f32) -> f32 {
    13.1 * (0.00074 * x).atan() + 2.24 * (0.0000000185 * x * x).atan() + 0.0001 * x
}

/// Precomputes bark map values used by floor type 0 packets
///
/// Precomputes the cos(omega) values for use by floor type 0 computation.
///
/// Note that there is one small difference to the spec: the output
/// vec is n elements long, not n+1. The last element (at index n)
/// is -1 in the spec, we lack it. Users of the result of this function
/// implementation should use it "virtually".
pub fn compute_bark_map_cos_omega(n: u16, floor0_rate: u16, floor0_bark_map_size: u16) -> Vec<f32> {
    let mut res = Vec::with_capacity(n as usize);
    let hfl = floor0_rate as f32 / 2.0;
    let hfl_dn = hfl / n as f32;
    let foobar_const_part = floor0_bark_map_size as f32 / bark(hfl);
    // Bark map size minus 1:
    let bms_m1 = floor0_bark_map_size as f32 - 1.0;
    let omega_factor = std::f32::consts::PI / floor0_bark_map_size as f32;
    for i in 0..n {
        let foobar = (bark(i as f32 * hfl_dn) * foobar_const_part).floor();
        let map_elem = foobar.min(bms_m1);
        let cos_omega = (map_elem * omega_factor).cos();
        res.push(cos_omega);
    }
    return res;
}
