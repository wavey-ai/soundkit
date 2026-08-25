use libopus_rs::celt::entropy::{RangeDecoder, RangeEncoder};
use libopus_rs::celt::laplace::*;

const DATA_SIZE: usize = 40_000;

#[derive(Clone)]
struct CRand(u32);

impl CRand {
    fn new(seed: u32) -> Self {
        Self(seed)
    }

    fn rand(&mut self) -> i32 {
        self.0 = self.0.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        ((self.0 >> 16) & 0x7fff) as i32
    }
}

#[test]
fn official_laplace_round_trip_matches_unit_test_shape() {
    let mut rng = CRand::new(1);
    let mut vals = vec![0i32; 10_000];
    let mut decay = vec![0i32; 10_000];
    vals[0] = 3;
    decay[0] = 6000;
    vals[1] = 0;
    decay[1] = 5800;
    vals[2] = -1;
    decay[2] = 5600;

    for i in 3..10_000 {
        vals[i] = rng.rand() % 15 - 7;
        decay[i] = rng.rand() % 11_000 + 5000;
    }

    let mut enc = RangeEncoder::new(DATA_SIZE);
    for i in 0..vals.len() {
        vals[i] = encode_laplace(&mut enc, vals[i], get_start_freq(decay[i]), decay[i]);
    }
    enc.shrink(((enc.tell() + 7) / 8) as usize);
    enc.finish();
    assert_eq!(enc.error(), 0);

    let mut dec = RangeDecoder::new(enc.range_data());
    for i in 0..vals.len() {
        let decoded = decode_laplace(&mut dec, get_start_freq(decay[i]), decay[i]);
        assert_eq!(decoded, vals[i], "symbol {i}");
    }
}

#[test]
fn official_laplace_p0_round_trip() {
    let values = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, -1, -7, -16];
    let p0 = 16_000;
    let decay = 16_000;

    let mut enc = RangeEncoder::new(DATA_SIZE);
    for value in values {
        encode_laplace_p0(&mut enc, value, p0, decay);
    }
    enc.shrink(((enc.tell() + 7) / 8) as usize);
    enc.finish();
    assert_eq!(enc.error(), 0);

    let mut dec = RangeDecoder::new(enc.range_data());
    for expected in values {
        assert_eq!(decode_laplace_p0(&mut dec, p0, decay), expected);
    }
}
