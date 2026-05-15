use libopus_rs::celt::cwrs::*;
use libopus_rs::celt::entropy::{RangeDecoder, RangeEncoder};

const PN: [usize; 22] = [
    2, 3, 4, 6, 8, 9, 11, 12, 16, 18, 22, 24, 32, 36, 44, 48, 64, 72, 88, 96, 144, 176,
];
const PKMAX: [usize; 22] = [
    128, 128, 128, 88, 36, 26, 18, 16, 12, 11, 9, 9, 7, 7, 6, 6, 5, 5, 5, 5, 4, 4,
];

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

#[test]
fn official_small_pvq_counts_match_cwrs_comment_table() {
    for (n, row) in SMALL_V.iter().enumerate() {
        for (k, expected) in row.iter().copied().enumerate() {
            assert_eq!(pvq_v(n, k), expected, "V({n},{k})");
        }
    }
}

#[test]
fn official_cwrs_index_round_trip_matches_unit_test_shape() {
    for (dim_idx, &n) in PN.iter().enumerate() {
        for pseudo in 1..MAX_PSEUDO + 1 {
            let k = get_pulses(pseudo);
            if k > PKMAX[dim_idx] {
                break;
            }

            let mut u = vec![0u32; k + 2];
            let nc = ncwrs_urow(n, k, &mut u);
            let inc = (nc / 4096).max(1);
            let mut i = 0u32;
            loop {
                let mut y = vec![0i32; n];
                let mut u_decode = u.clone();
                decode_index(n, k, i, &mut y, &mut u_decode);

                let pulse_sum: usize = y.iter().map(|v| v.unsigned_abs() as usize).sum();
                assert_eq!(pulse_sum, k, "N={n}, K={k}, i={i}");

                let mut u_encode = vec![0u32; k + 2];
                let (ii, v) = encode_index(n, k, &y, &mut u_encode);
                assert_eq!(ii, i, "N={n}, K={k}");
                assert_eq!(v, nc, "N={n}, K={k}");

                if nc - i <= inc {
                    break;
                }
                i += inc;
            }
        }
    }
}

#[test]
fn official_cwrs_range_encoder_round_trip() {
    let vectors: [(&[i32], usize); 6] = [
        (&[3, 0, -1, 0], 4),
        (&[0, -2, 1, 0, 1, 0], 4),
        (&[5, -1], 6),
        (&[0, 0, 0, -8, 3, 1, 0, 0], 12),
        (&[1, -1, 1, -1, 1, -1, 1, -1], 8),
        (&[0, 0, 0, 4, 0, -4, 0, 0, 0], 8),
    ];

    let mut enc = RangeEncoder::new(1024);
    for (y, k) in vectors {
        encode_pulses(y, y.len(), k, &mut enc);
    }
    enc.shrink(((enc.tell() + 7) / 8) as usize);
    enc.finish();
    assert_eq!(enc.error(), 0);

    let mut dec = RangeDecoder::new(enc.range_data());
    for (expected, k) in vectors {
        let mut decoded = vec![0i32; expected.len()];
        let yy = decode_pulses(&mut decoded, expected.len(), k, &mut dec);
        assert_eq!(decoded, expected);
        assert_eq!(yy, expected.iter().map(|v| v * v).sum::<i32>());
    }
}

#[derive(Clone)]
struct CRand(u32);

impl CRand {
    fn new(seed: u32) -> Self {
        Self(seed)
    }

    fn rand(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        (self.0 >> 16) & 0x7fff
    }
}

#[test]
fn official_cwrs_vector_index_vector_round_trips() {
    let mut rng = CRand::new(0xc012_5eed);
    for (n, k) in [(3, 2), (6, 4), (15, 3), (32, 5), (80, 5)] {
        for case_idx in 0..512 {
            let mut counts = vec![0i32; n];
            for _ in 0..k {
                let pos = (rng.rand() as usize) % n;
                counts[pos] += 1;
            }
            let mut y = vec![0i32; n];
            for pos in 0..n {
                y[pos] = if rng.rand() & 1 != 0 {
                    counts[pos]
                } else {
                    -counts[pos]
                };
            }

            let mut u_encode = vec![0u32; k + 2];
            let (i, nc) = encode_index(n, k, &y, &mut u_encode);
            assert!(
                i < nc,
                "N={n}, K={k}, case={case_idx}, i={i}, nc={nc}, y={y:?}"
            );

            let mut u_decode = vec![0u32; k + 2];
            assert_eq!(ncwrs_urow(n, k, &mut u_decode), nc);
            let mut decoded = vec![0i32; n];
            decode_index(n, k, i, &mut decoded, &mut u_decode);
            assert_eq!(decoded, y, "N={n}, K={k}, case={case_idx}, i={i}");
        }
    }
}

#[test]
fn official_cwrs_range_round_trips_valid_vectors() {
    let mut rng = CRand::new(0x1234_abcd);
    for (n, k) in [(15, 3), (32, 5), (80, 5)] {
        for case_idx in 0..128 {
            let mut counts = vec![0i32; n];
            for _ in 0..k {
                counts[(rng.rand() as usize) % n] += 1;
            }
            let mut y = vec![0i32; n];
            for pos in 0..n {
                y[pos] = if rng.rand() & 1 != 0 {
                    counts[pos]
                } else {
                    -counts[pos]
                };
            }

            let mut enc = RangeEncoder::new(64);
            encode_pulses(&y, n, k, &mut enc);
            enc.shrink(((enc.tell() + 7) / 8) as usize);
            enc.finish();
            assert_eq!(enc.error(), 0);

            let mut dec = RangeDecoder::new(enc.range_data());
            let mut decoded = vec![0i32; n];
            decode_pulses(&mut decoded, n, k, &mut dec);
            assert_eq!(
                decoded,
                y,
                "N={n}, K={k}, case={case_idx}, bytes={:?}",
                enc.range_data()
            );
        }
    }
}

#[test]
fn official_log2_frac_is_a_conservative_estimate() {
    for val in [1, 2, 3, 4, 5, 31, 32, 33, 255, 256, 257, 65_535, 1_000_000] {
        let estimate = log2_frac(val, 4);
        let actual = (val as f64).log2() * 16.0;
        assert!((estimate as f64) >= actual);
        assert!(
            (estimate as f64) - actual < 2.0,
            "val={val}, estimate={estimate}, actual={actual}"
        );
    }
}
