//! Opus Custom mode tables checked against libopus.
//!
//! Each `tests/data/opus_custom/mode_<frame>.txt` file is the output of
//! `tools/opus_custom_interop.c modeinfo --frame <frame>` built with
//! `OPUS_CUSTOM_INTERNALS` against libopus with custom modes enabled. It lists
//! the mode's band layout, allocation table, `logN`, window (as `f32` bit
//! patterns) and pulse cache, as `opus_custom_mode_create(48000, frame)`
//! builds them.

use soundkit_opus::celt::modes::CeltMode;
use std::collections::BTreeMap;

const FRAMES: [usize; 19] = [
    48, 50, 64, 72, 80, 96, 100, 128, 144, 160, 200, 256, 320, 400, 512, 640, 1024, 120, 240,
];

fn fixture(frame: usize) -> BTreeMap<String, Vec<String>> {
    let path = format!(
        "{}/tests/data/opus_custom/mode_{frame}.txt",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("{path}: {error}"));
    text.lines()
        .filter_map(|line| {
            let mut parts = line.split_whitespace();
            let key = parts.next()?.to_string();
            Some((key, parts.map(str::to_string).collect()))
        })
        .collect()
}

fn ints(table: &BTreeMap<String, Vec<String>>, key: &str) -> Vec<i64> {
    table[key]
        .iter()
        .map(|value| value.parse().expect("integer"))
        .collect()
}

fn int(table: &BTreeMap<String, Vec<String>>, key: &str) -> i64 {
    ints(table, key)[0]
}

#[test]
fn custom_mode_tables_match_libopus() {
    for frame in FRAMES {
        let table = fixture(frame);
        let mode = CeltMode::custom_48k_shared(frame).expect("libopus accepts this frame size");
        let context = format!("frame {frame}");
        assert_eq!(mode.fs as i64, int(&table, "fs"), "{context}");
        assert_eq!(mode.max_lm as i64, int(&table, "max_lm"), "{context}");
        assert_eq!(
            mode.short_mdct_size as i64,
            int(&table, "short_mdct_size"),
            "{context}"
        );
        assert_eq!(
            mode.nb_short_mdcts as i64,
            int(&table, "nb_short_mdcts"),
            "{context}"
        );
        assert_eq!(mode.overlap as i64, int(&table, "overlap"), "{context}");
        assert_eq!(mode.nb_ebands as i64, int(&table, "nb_ebands"), "{context}");
        assert_eq!(
            mode.eff_ebands as i64,
            int(&table, "eff_ebands"),
            "{context}"
        );
        assert_eq!(
            mode.nb_alloc_vectors as i64,
            int(&table, "nb_alloc_vectors"),
            "{context}"
        );
        let ebands: Vec<i64> = mode.ebands.iter().map(|&v| v as i64).collect();
        assert_eq!(ebands, ints(&table, "ebands"), "{context}");
        let alloc: Vec<i64> = mode.alloc_vectors.iter().map(|&v| v as i64).collect();
        assert_eq!(alloc, ints(&table, "alloc_vectors"), "{context}");
        let log_n: Vec<i64> = mode.log_n.iter().map(|&v| v as i64).collect();
        assert_eq!(log_n, ints(&table, "log_n"), "{context}");
        let window: Vec<String> = mode
            .window
            .iter()
            .map(|value| format!("{:08x}", value.to_bits()))
            .collect();
        assert_eq!(window, table["window_bits"], "{context}");
        assert_eq!(
            mode.cache.size as i64,
            int(&table, "cache_size"),
            "{context}"
        );
        let index: Vec<i64> = mode.cache.index.iter().map(|&v| v as i64).collect();
        assert_eq!(index, ints(&table, "cache_index"), "{context}");
        let bits: Vec<i64> = mode.cache.bits.iter().map(|&v| v as i64).collect();
        assert_eq!(bits, ints(&table, "cache_bits"), "{context}");
        let caps: Vec<i64> = mode.cache.caps.iter().map(|&v| v as i64).collect();
        assert_eq!(caps, ints(&table, "cache_caps"), "{context}");
        // The codec look-ahead (OPUS_GET_LOOKAHEAD) is the MDCT overlap.
        assert_eq!(mode.overlap as i64, int(&table, "lookahead"), "{context}");
    }
}

#[test]
fn power_of_two_fractions_of_960_use_the_standard_mode() {
    let standard = CeltMode::standard_48k_exact_shared();
    for frame in [120, 240, 480, 960] {
        let mode = CeltMode::custom_48k_shared(frame).expect("standard frame size");
        assert!(std::ptr::eq(mode, standard), "frame {frame}");
        assert!(mode.exact_math && mode.static_tables);
        let approx = CeltMode::custom_48k_shared_with_math(frame, false).expect("standard");
        assert!(std::ptr::eq(approx, CeltMode::standard_48k_shared()));
        assert_eq!(
            mode.lm_for_frame_size(frame),
            Some(frame.trailing_zeros() as usize - 3)
        );
    }
}

#[test]
fn custom_modes_are_shared_per_frame_size() {
    let a = CeltMode::custom_48k_shared(48).expect("1 ms mode");
    let b = CeltMode::custom_48k_shared(48).expect("1 ms mode");
    assert!(std::ptr::eq(a, b));
    assert!(!a.static_tables);
    assert_eq!(a.frame_size(), 48);
    assert_eq!(a.lm_for_frame_size(48), Some(0));
    assert_eq!(a.lm_for_frame_size(96), None);
}

#[test]
fn frame_sizes_libopus_rejects_are_rejected() {
    // 46: under 1 ms. 1000 and 1022: short blocks over 3.3 ms. 162: odd
    // short-block limit. 52, 58, 98: the N/4-point FFT needs a radix above 5.
    // 47: odd. 1026: over 1024.
    for frame in [0, 46, 47, 52, 58, 98, 162, 1000, 1022, 1026] {
        assert!(
            CeltMode::custom_48k_shared(frame).is_err(),
            "frame {frame} should be rejected"
        );
    }
}
