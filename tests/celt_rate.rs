use libopus_rs::celt::entropy::{RangeDecoder, RangeEncoder};
use libopus_rs::celt::modes::CeltMode;
use libopus_rs::celt::rate::{clt_compute_allocation, AllocationCoder};

#[test]
fn official_allocation_encode_decode_paths_match() {
    let mode = CeltMode::standard_48k();
    let start = 0;
    let end = mode.nb_ebands;
    let lm = 3;
    let channels = 2;
    let offsets = vec![0; mode.nb_ebands];
    let cap = (0..mode.nb_ebands)
        .map(|band| {
            mode.cache.caps[lm * 2 * mode.nb_ebands + (channels - 1) * mode.nb_ebands + band] as i32
        })
        .collect::<Vec<_>>();

    let mut enc = RangeEncoder::new(16);
    let mut enc_coder = AllocationCoder::Encode(&mut enc);
    let encoded = clt_compute_allocation(
        &mode,
        start,
        end,
        &offsets,
        &cap,
        5,
        17,
        false,
        3600,
        channels,
        lm,
        Some(&mut enc_coder),
        21,
        20,
    );
    enc.shrink(((enc.tell() + 7) / 8) as usize);
    enc.finish();
    assert_eq!(enc.error(), 0);

    let mut dec = RangeDecoder::new(enc.range_data());
    let mut dec_coder = AllocationCoder::Decode(&mut dec);
    let decoded = clt_compute_allocation(
        &mode,
        start,
        end,
        &offsets,
        &cap,
        5,
        17,
        false,
        3600,
        channels,
        lm,
        Some(&mut dec_coder),
        21,
        20,
    );

    assert_eq!(decoded, encoded);
    assert!(encoded.coded_bands > 0);
    assert!(encoded.pulses.iter().all(|p| *p >= 0));
    assert!(encoded.ebits.iter().all(|p| *p >= 0));
}

#[test]
fn official_allocation_without_coder_has_sane_invariants() {
    let mode = CeltMode::standard_48k();
    let offsets = vec![0; mode.nb_ebands];
    let lm = 0;
    let channels = 1;
    let cap = (0..mode.nb_ebands)
        .map(|band| {
            mode.cache.caps[lm * 2 * mode.nb_ebands + (channels - 1) * mode.nb_ebands + band] as i32
        })
        .collect::<Vec<_>>();

    let allocation = clt_compute_allocation(
        &mode,
        0,
        mode.nb_ebands,
        &offsets,
        &cap,
        5,
        0,
        false,
        900,
        channels,
        lm,
        None,
        mode.nb_ebands,
        mode.nb_ebands - 1,
    );

    assert!(allocation.coded_bands > 0);
    assert!(allocation.coded_bands <= mode.nb_ebands);
    assert_eq!(allocation.intensity, 0);
    assert!(!allocation.dual_stereo);
    assert!(allocation.pulses[..allocation.coded_bands]
        .iter()
        .all(|p| *p >= 0));
}
