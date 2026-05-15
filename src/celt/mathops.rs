pub const PI: f32 = 3.141592653_f32;

#[inline]
pub fn frac_mul16(a: i32, b: i32) -> i32 {
    (16384 + (a as i16 as i32) * (b as i16 as i32)) >> 15
}

#[inline]
pub fn ec_ilog(v: u32) -> i32 {
    if v == 0 {
        0
    } else {
        (u32::BITS - v.leading_zeros()) as i32
    }
}

#[inline]
pub fn celt_ilog2(v: i32) -> i32 {
    ec_ilog(v as u32) - 1
}

pub fn isqrt32(mut val: u32) -> u32 {
    debug_assert!(val > 0);
    let mut g = 0u32;
    let mut bshift = (ec_ilog(val) - 1) >> 1;
    let mut b = 1u32 << bshift;
    loop {
        let t = ((g << 1) + b) << bshift;
        if t <= val {
            g += b;
            val -= t;
        }
        b >>= 1;
        bshift -= 1;
        if bshift < 0 {
            break;
        }
    }
    g
}

pub fn fast_atan2f(y: f32, x: f32) -> f32 {
    const CA: f32 = 0.43157974;
    const CB: f32 = 0.67848403;
    const CC: f32 = 0.08595542;
    const CE: f32 = PI / 2.0;

    let x2 = x * x;
    let y2 = y * y;
    if x2 + y2 < 1e-18 {
        return 0.0;
    }
    if x2 < y2 {
        let den = (y2 + CB * x2) * (y2 + CC * x2);
        -x * y * (y2 + CA * x2) / den + if y < 0.0 { -CE } else { CE }
    } else {
        let den = (x2 + CB * y2) * (x2 + CC * y2);
        x * y * (x2 + CA * y2) / den + if y < 0.0 { -CE } else { CE }
            - if x * y < 0.0 { -CE } else { CE }
    }
}

pub fn celt_maxabs16(x: &[f32]) -> f32 {
    x.iter().fold(0.0f32, |acc, value| acc.max(value.abs()))
}

pub fn celt_sqrt(x: f32) -> f32 {
    x.sqrt()
}

pub fn celt_rsqrt(x: f32) -> f32 {
    1.0 / celt_sqrt(x)
}

pub fn celt_rsqrt_norm(x: f32) -> f32 {
    celt_rsqrt(x)
}

pub fn celt_cos_norm(x: f32) -> f32 {
    (0.5 * PI * x).cos()
}

pub fn celt_rcp(x: f32) -> f32 {
    1.0 / x
}

pub fn celt_div(a: f32, b: f32) -> f32 {
    a / b
}

pub fn frac_div32(a: f32, b: f32) -> f32 {
    a / b
}

pub fn frac_div32_q29(a: f32, b: f32) -> f32 {
    frac_div32(a, b)
}

pub fn celt_log2(x: f32) -> f32 {
    x.log2()
}

pub fn celt_exp2(x: f32) -> f32 {
    x.exp2()
}

pub fn celt_exp2_db(x: f32) -> f32 {
    celt_exp2(x)
}

pub fn celt_log2_db(x: f32) -> f32 {
    celt_log2(x)
}

pub fn bitexact_cos(x: i16) -> i16 {
    let tmp = (4096 + (x as i32) * (x as i32)) >> 13;
    debug_assert!(tmp <= 32767);
    let x2 = tmp;
    let inner = 8277 + frac_mul16(-626, x2);
    let poly = -7651 + frac_mul16(x2, inner);
    let x2 = (32767 - x2) + frac_mul16(x2, poly);
    debug_assert!(x2 <= 32766);
    (1 + x2) as i16
}

pub fn bitexact_log2tan(isin: i32, icos: i32) -> i32 {
    let lc = ec_ilog(icos as u32);
    let ls = ec_ilog(isin as u32);
    let icos = icos << (15 - lc);
    let isin = isin << (15 - ls);
    (ls - lc) * (1 << 11) + frac_mul16(isin, frac_mul16(isin, -2597) + 7932)
        - frac_mul16(icos, frac_mul16(icos, -2597) + 7932)
}

pub fn float_to_i16(x: f32) -> i16 {
    let x = (x * 32768.0).clamp(-32768.0, 32767.0);
    (x + 0.5).floor() as i16
}

pub fn celt_float2int16(input: &[f32], output: &mut [i16]) {
    assert!(output.len() >= input.len());
    for (out, value) in output.iter_mut().zip(input.iter()) {
        *out = float_to_i16(*value);
    }
}

pub fn opus_limit2_checkwithin1(samples: &mut [f32]) -> bool {
    if samples.is_empty() {
        return true;
    }

    let mut within = true;
    for sample in samples {
        if !(-1.0..=1.0).contains(sample) {
            within = false;
        }
        *sample = sample.clamp(-2.0, 2.0);
    }
    within
}
