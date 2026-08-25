use soundkit_opus::celt::entropy::*;

const DATA_SIZE: usize = 10_000_000;
const DATA_SIZE2: usize = 10_000;

#[derive(Clone)]
struct Lcg(u32);

impl Lcg {
    fn new(seed: u32) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        (self.0 >> 1) & 0x7fff_ffff
    }

    fn c_rand(&mut self) -> u32 {
        self.next() & 0x7fff
    }
}

#[test]
fn entropy_uint_and_raw_bits_round_trip() {
    let mut enc = RangeEncoder::new(DATA_SIZE);
    for ft in 2..1024 {
        for i in 0..ft {
            enc.encode_uint(i, ft);
        }
    }
    for ftb in 1..16 {
        for i in 0..(1u32 << ftb) {
            let nbits = enc.tell();
            enc.encode_bits(i, ftb);
            let nbits2 = enc.tell();
            assert_eq!(nbits2 - nbits, ftb as i32);
        }
    }
    let nbits = enc.tell_frac();
    enc.finish();
    assert_eq!(enc.error(), 0);

    let mut dec = RangeDecoder::new(enc.buffer());
    for ft in 2..1024 {
        for i in 0..ft {
            assert_eq!(dec.decode_uint(ft), i);
        }
    }
    for ftb in 1..16 {
        for i in 0..(1u32 << ftb) {
            assert_eq!(dec.decode_bits(ftb), i);
        }
    }
    assert_eq!(dec.tell_frac(), nbits);
    assert_eq!(dec.error(), 0);
}

#[test]
fn entropy_prefers_range_data_when_encoder_busts() {
    let mut enc = RangeEncoder::new(2);
    enc.encode_bits(0x55, 7);
    enc.encode_uint(1, 2);
    enc.encode_uint(1, 3);
    enc.encode_uint(1, 4);
    enc.encode_uint(1, 5);
    enc.encode_uint(2, 6);
    enc.encode_uint(6, 7);
    enc.finish();

    let mut dec = RangeDecoder::new(enc.buffer());
    assert_ne!(enc.error(), 0);
    assert_eq!(dec.decode_bits(7), 0x05);
    assert_eq!(dec.decode_uint(2), 1);
    assert_eq!(dec.decode_uint(3), 1);
    assert_eq!(dec.decode_uint(4), 1);
    assert_eq!(dec.decode_uint(5), 1);
    assert_eq!(dec.decode_uint(6), 2);
    assert_eq!(dec.decode_uint(7), 6);
}

#[test]
fn entropy_random_uint_streams_round_trip() {
    let mut rng = Lcg::new(0x5152_1f5u32);

    for case_idx in 0..4096 {
        let shift = rng.c_rand() % 11;
        let ft = rng.c_rand() / ((0x7fff >> shift) + 1) + 10;
        let shift = rng.c_rand() % 9;
        let sz = (rng.c_rand() / ((0x7fff >> shift) + 1)) as usize;
        let zeros = rng.c_rand() % 13 == 0;
        let mut values = Vec::with_capacity(sz);
        let mut tell = Vec::with_capacity(sz + 1);

        let mut enc = RangeEncoder::new(DATA_SIZE2);
        tell.push(enc.tell_frac());
        for _ in 0..sz {
            let value = if zeros { 0 } else { rng.c_rand() % ft };
            values.push(value);
            enc.encode_uint(value, ft);
            tell.push(enc.tell_frac());
        }
        if rng.c_rand() % 2 == 0 {
            while enc.tell() % 8 != 0 {
                enc.encode_uint(rng.c_rand() % 2, 2);
            }
        }
        let tell_bits = enc.tell();
        enc.finish();
        assert_eq!(tell_bits, enc.tell(), "case {case_idx}");
        assert!(
            ((tell_bits + 7) / 8) as u32 >= enc.range_bytes(),
            "case {case_idx}"
        );

        let mut dec = RangeDecoder::new(enc.buffer());
        assert_eq!(dec.tell_frac(), tell[0], "case {case_idx}");
        for (j, value) in values.iter().enumerate() {
            assert_eq!(dec.decode_uint(ft), *value, "case {case_idx} symbol {j}");
            assert_eq!(dec.tell_frac(), tell[j + 1], "case {case_idx} symbol {j}");
        }
    }
}

#[test]
fn entropy_mixed_boolean_coders_are_compatible() {
    let mut rng = Lcg::new(0x0e1d_f00d);

    for case_idx in 0..4096 {
        let shift = rng.c_rand() % 9;
        let sz = (rng.c_rand() / ((0x7fff >> shift) + 1)) as usize;
        let mut logp1 = Vec::with_capacity(sz);
        let mut values = Vec::with_capacity(sz);
        let mut tell = Vec::with_capacity(sz + 1);
        let mut methods = Vec::with_capacity(sz);

        let mut enc = RangeEncoder::new(DATA_SIZE2);
        tell.push(enc.tell_frac());
        for _ in 0..sz {
            let value = rng.c_rand() / ((0x7fff >> 1) + 1);
            let logp = (rng.c_rand() % 15) + 1;
            let method = rng.c_rand() / ((0x7fff >> 2) + 1);
            values.push(value);
            logp1.push(logp);
            methods.push(method);
            match method {
                0 => enc.encode(
                    if value != 0 { (1 << logp) - 1 } else { 0 },
                    (1 << logp) - if value != 0 { 0 } else { 1 },
                    1 << logp,
                ),
                1 => enc.encode_bin(
                    if value != 0 { (1 << logp) - 1 } else { 0 },
                    (1 << logp) - if value != 0 { 0 } else { 1 },
                    logp,
                ),
                2 => enc.encode_bit_logp(value != 0, logp),
                3 => enc.encode_icdf(value as usize, &[1u8, 0u8], logp),
                _ => unreachable!(),
            }
            tell.push(enc.tell_frac());
        }
        enc.finish();
        assert!(
            ((enc.tell() + 7) / 8) as u32 >= enc.range_bytes(),
            "case {case_idx}"
        );

        let mut dec = RangeDecoder::new(enc.buffer());
        assert_eq!(dec.tell_frac(), tell[0], "case {case_idx}");
        for j in 0..sz {
            let logp = logp1[j];
            let method = rng.c_rand() / ((0x7fff >> 2) + 1);
            let sym = match method {
                0 => {
                    let fs = dec.decode(1 << logp);
                    let sym = u32::from(fs >= (1 << logp) - 1);
                    dec.update(
                        if sym != 0 { (1 << logp) - 1 } else { 0 },
                        (1 << logp) - if sym != 0 { 0 } else { 1 },
                        1 << logp,
                    );
                    sym
                }
                1 => {
                    let fs = dec.decode_bin(logp);
                    let sym = u32::from(fs >= (1 << logp) - 1);
                    dec.update(
                        if sym != 0 { (1 << logp) - 1 } else { 0 },
                        (1 << logp) - if sym != 0 { 0 } else { 1 },
                        1 << logp,
                    );
                    sym
                }
                2 => u32::from(dec.decode_bit_logp(logp)),
                3 => dec.decode_icdf(&[1u8, 0u8], logp) as u32,
                _ => unreachable!(),
            };
            assert_eq!(
                sym, values[j],
                "case {case_idx} symbol {j}, enc method {}",
                methods[j]
            );
            assert_eq!(dec.tell_frac(), tell[j + 1], "case {case_idx} symbol {j}");
        }
    }
}

#[test]
fn entropy_patch_initial_bits_and_overfill_regressions() {
    let mut enc = RangeEncoder::new(DATA_SIZE2);
    enc.encode_bit_logp(false, 1);
    enc.encode_bit_logp(false, 1);
    enc.encode_bit_logp(false, 1);
    enc.encode_bit_logp(false, 1);
    enc.encode_bit_logp(false, 2);
    enc.patch_initial_bits(3, 2);
    assert_eq!(enc.error(), 0);
    enc.patch_initial_bits(0, 5);
    assert_ne!(enc.error(), 0);
    enc.finish();
    assert_eq!(enc.range_bytes(), 1);
    assert_eq!(enc.buffer()[0], 192);

    let mut enc = RangeEncoder::new(DATA_SIZE2);
    enc.encode_bit_logp(false, 1);
    enc.encode_bit_logp(false, 1);
    enc.encode_bit_logp(true, 6);
    enc.encode_bit_logp(false, 2);
    enc.patch_initial_bits(0, 2);
    assert_eq!(enc.error(), 0);
    enc.finish();
    assert_eq!(enc.range_bytes(), 2);
    assert_eq!(enc.buffer()[0], 63);

    let mut enc = RangeEncoder::new(2);
    enc.encode_bit_logp(false, 2);
    for _ in 0..48 {
        enc.encode_bits(0, 1);
    }
    enc.finish();
    assert_ne!(enc.error(), 0);

    let mut enc = RangeEncoder::new(2);
    for _ in 0..17 {
        enc.encode_bits(0, 1);
    }
    enc.finish();
    assert_ne!(enc.error(), 0);
}
