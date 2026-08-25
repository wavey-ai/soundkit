use libopus_rs::celt::pitch::{comb_filter, comb_filter_in_place};

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
