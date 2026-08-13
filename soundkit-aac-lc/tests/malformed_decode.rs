use std::panic::{catch_unwind, AssertUnwindSafe};

use soundkit_aac_lc::AacLcDecoder;

#[test]
fn malformed_stereo_access_units_never_panic() {
    let mut state = 0x8f6d_7a51_c392_4b0du64;

    for case in 0..2_048usize {
        let length = case % 513;
        let mut data = vec![0u8; length];
        for byte in &mut data {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *byte = state as u8;
        }

        for config in [[0x12, 0x10], [0x11, 0x90]] {
            let mut decoder = AacLcDecoder::from_audio_specific_config(&config).unwrap();
            let outcome = catch_unwind(AssertUnwindSafe(|| {
                if let Ok(pcm) = decoder.decode_access_unit(&data) {
                    assert_eq!(pcm.channels().len(), 2);
                    assert_eq!(pcm.frames(), 1024);
                    assert!(pcm
                        .channels()
                        .iter()
                        .flatten()
                        .all(|sample| sample.is_finite()));
                }
            }));
            outcome.unwrap_or_else(|_| {
                panic!(
                    "AAC-LC decoder panicked for malformed case {case}, length {length}, config {config:02x?}"
                )
            });
        }
    }
}
