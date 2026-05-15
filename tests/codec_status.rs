use libopus_rs::{Application, Decoder, Encoder};

#[test]
fn encode_and_decode_48k_celt_only_smoke_path() {
    let mut encoder = Encoder::new(48_000, 2, Application::Audio).unwrap();
    let pcm = (0..960)
        .flat_map(|i| {
            let left = (i as f32 * 0.011).sin() * 0.2;
            let right = (i as f32 * 0.017).cos() * 0.2;
            [left, right]
        })
        .collect::<Vec<_>>();
    let packet = encoder.encode_f32(&pcm, 960).unwrap();
    assert!(!packet.is_empty());

    let mut decoder = Decoder::new(48_000, 2).unwrap();
    let decoded = decoder.decode_f32(&packet, false).unwrap();
    assert_eq!(decoded.len(), pcm.len());
    assert!(decoded.iter().all(|sample| sample.is_finite()));
    assert!(decoded.iter().any(|sample| sample.abs() > 1e-5));
}
