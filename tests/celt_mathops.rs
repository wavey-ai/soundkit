use libopus_rs::celt::mathops::*;

#[test]
fn official_isqrt32_matches_integer_floor_sqrt() {
    let mut i = 1u32;
    while i <= 1_000_000_000 {
        let got = isqrt32(i);
        assert!(got * got <= i, "isqrt32({i}) too high: {got}");
        assert!(
            got == u32::MAX || (got + 1).saturating_mul(got + 1) > i,
            "isqrt32({i}) too low: {got}"
        );
        i += 1 + (i >> 10);
    }
}

#[test]
fn official_float_div_and_sqrt_accuracy() {
    for i in 1..=327_670 {
        let val = celt_rcp(i as f32);
        let prod = val * i as f32;
        assert!((prod - 1.0).abs() <= 0.00025, "rcp failed at {i}");
    }

    let mut i = 1u32;
    while i <= 1_000_000_000 {
        let expected = (i as f32).sqrt();
        let val = celt_sqrt(i as f32);
        let ratio = val / expected;
        assert!(
            (ratio - 1.0).abs() <= 0.0005 || (val - expected).abs() <= 2.0,
            "sqrt failed at {i}: {val} / {expected}"
        );
        i += 1 + (i >> 10);
    }
}

#[test]
fn official_bitexact_cos_checksums_match() {
    let mut chk = 0i32;
    let mut max_d = 0i32;
    let mut last = 32767i32;
    let mut min_d = 32767i32;

    for i in 64..=16320 {
        let q = bitexact_cos(i as i16) as i32;
        chk ^= q * i;
        let d = last - q;
        max_d = max_d.max(d);
        min_d = min_d.min(d);
        last = q;
    }

    assert_eq!(chk, 89408644);
    assert_eq!(max_d, 5);
    assert_eq!(min_d, 0);
    assert_eq!(bitexact_cos(64), 32767);
    assert_eq!(bitexact_cos(16320), 200);
    assert_eq!(bitexact_cos(8192), 23171);
}

#[test]
fn official_bitexact_log2tan_checksums_match() {
    let mut fail = false;
    let mut chk = 0i32;
    let mut max_d = 0i32;
    let mut last = 15059i32;
    let mut min_d = 15059i32;

    for i in 64..8193 {
        let mid = bitexact_cos(i as i16) as i32;
        let side = bitexact_cos((16384 - i) as i16) as i32;
        let q = bitexact_log2tan(mid, side);
        chk ^= q * i;
        let d = last - q;
        if q != -bitexact_log2tan(side, mid) {
            fail = true;
        }
        max_d = max_d.max(d);
        min_d = min_d.min(d);
        last = q;
    }

    assert_eq!(chk, 15821257);
    assert_eq!(max_d, 61);
    assert_eq!(min_d, -2);
    assert!(!fail);
    assert_eq!(bitexact_log2tan(32767, 200), 15059);
    assert_eq!(bitexact_log2tan(30274, 12540), 2611);
    assert_eq!(bitexact_log2tan(23171, 23171), 0);
}

#[test]
fn official_float_log2_exp2_accuracy() {
    let mut x = 0.001f32;
    while x < 1_677_700.0 {
        let error = (1.4426950408889634_f64 * (x as f64).ln() - celt_log2(x) as f64).abs();
        assert!(error <= 2.2e-6, "celt_log2 failed at {x}: {error}");
        x += x / 8.0;
    }

    let mut x = -11.0f32;
    while x < 24.0 {
        let exp = celt_exp2(x);
        let error = (x as f64 - 1.4426950408889634_f64 * (exp as f64).ln()).abs();
        assert!(error <= 2.3e-7, "celt_exp2 failed at {x}: {error}");
        let roundtrip = (x - celt_log2(exp)).abs();
        assert!(
            roundtrip <= 2.0e-6,
            "celt_log2(celt_exp2({x})) failed: {roundtrip}"
        );
        x += 0.0007;
    }
}

#[test]
fn official_float_to_i16_and_limit2_behaviour() {
    let mut input = Vec::new();
    let scale = 1.0 / 32768.0;
    input.extend([
        77777.0 * scale,
        33000.0 * scale,
        32768.0 * scale,
        32767.4 * scale,
        32766.6 * scale,
        0.501 * scale,
        0.499 * scale,
        0.0,
        -0.499 * scale,
        -0.501 * scale,
        -32767.6 * scale,
        -32768.4 * scale,
        -32769.0 * scale,
        -33000.0 * scale,
        -77777.0 * scale,
    ]);

    let mut out = [42i16; 32];
    celt_float2int16(&input, &mut out);
    for (i, value) in input.iter().enumerate() {
        assert_eq!(out[i], float_to_i16(*value), "index {i}");
    }
    assert!(out[input.len()..].iter().all(|&sample| sample == 42));

    let mut pattern = [0.0f32; 37];
    for (i, sample) in pattern.iter_mut().enumerate() {
        *sample = if i % 2 == 0 { 1.0 } else { -1.0 };
    }
    let mut buffer = pattern;
    assert!(opus_limit2_checkwithin1(&mut buffer));
    assert_eq!(buffer, pattern);

    for i in 0..buffer.len() {
        buffer = pattern;
        buffer[i] *= 1.001;
        let replace_value = buffer[i];
        assert!(!opus_limit2_checkwithin1(&mut buffer));
        assert_eq!(buffer[i], replace_value);

        buffer = pattern;
        buffer[i] *= 2.1;
        assert!(!opus_limit2_checkwithin1(&mut buffer));
        assert_eq!(buffer[i], if pattern[i] > 0.0 { 2.0 } else { -2.0 });
    }
}
