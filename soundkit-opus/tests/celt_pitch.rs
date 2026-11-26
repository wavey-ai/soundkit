use soundkit_opus::celt::modes::CeltMode;
use soundkit_opus::celt::pitch::{comb_filter, comb_filter_in_place, comb_filter_in_place_mode};

fn rms_diff(a: &[f32], b: &[f32]) -> f32 {
    let err = a
        .iter()
        .zip(b)
        .map(|(left, right)| {
            let diff = left - right;
            diff * diff
        })
        .sum::<f32>();
    (err / a.len() as f32).sqrt()
}

#[test]
fn official_decoder_comb_filter_uses_in_place_feedback() {
    let base = 64;
    let n = 96;
    let mut in_place = vec![0.0f32; base + n + 4];
    for (i, sample) in in_place.iter_mut().enumerate() {
        let phase = i as f32;
        *sample = 0.21 * (0.13 * phase).sin() + 0.08 * (0.37 * phase).cos();
    }

    let source = in_place.clone();
    let mut out_of_place = in_place.clone();
    let window = vec![0.5f32; 120];

    comb_filter_in_place(
        &mut in_place,
        base,
        15,
        17,
        n,
        0.25,
        0.5,
        0,
        2,
        Some(&window),
        24,
    );
    comb_filter(
        &mut out_of_place,
        base,
        &source,
        base,
        15,
        17,
        n,
        0.25,
        0.5,
        0,
        2,
        Some(&window),
        24,
    );

    let diff = rms_diff(&in_place[base..base + n], &out_of_place[base..base + n]);
    assert!(
        diff > 0.001,
        "decoder postfilter must retain libopus' in-place feedback behavior"
    );
}

#[test]
fn exact_1ms_comb_filter_matches_scalar_with_in_place_feedback() {
    const BASE: usize = 1024;
    const FRAME: usize = 48;
    let mode = CeltMode::custom_48k_shared(FRAME).unwrap();
    let mut seed = 17403_u32;
    for (first, next) in [(15, 16), (17, 15), (17, 49), (49, 18), (23, 23), (1022, 15)] {
        for first_tap in 0..3 {
            for next_tap in 0..3 {
                for (old_gain, new_gain) in [(0.0, 0.25), (0.75, 0.0), (0.125, 0.6)] {
                    let input: Vec<_> = (0..BASE + FRAME)
                        .map(|index| {
                            seed ^= seed << 13;
                            seed ^= seed >> 17;
                            seed ^= seed << 5;
                            if index % 15 == 0 {
                                -0.0
                            } else {
                                (seed as i32 as f32) / 65_536.0
                            }
                        })
                        .collect();
                    let outputs: Vec<Vec<u32>> = [false, true]
                        .into_iter()
                        .map(|exact| {
                            let mut output = input.clone();
                            comb_filter_in_place_mode(
                                exact,
                                &mut output,
                                BASE,
                                first,
                                next,
                                FRAME,
                                old_gain,
                                new_gain,
                                first_tap,
                                next_tap,
                                Some(&mode.window),
                                FRAME,
                            );
                            output.into_iter().map(f32::to_bits).collect()
                        })
                        .collect();
                    assert_eq!(outputs[0], outputs[1]);
                }
            }
        }
    }
}
