#![cfg(feature = "encodec")]
#[path = "../../soundkit-encodec/tests/support/mod.rs"]
mod support;
use soundkit_decoder::{
    decode_encodec_to_sink, Bytes, DecodeError, DecodeOptions, DecodePipeline, RawPcmFormat,
};

#[test]
fn encodec_uses_the_common_pcm_output_conversion() {
    let (mut model, mut lm, payload, frames) = support::fixture();
    let mut received = 0;
    let mut raw_chunks = Vec::new();
    decode_encodec_to_sink(
        &mut model,
        &mut lm,
        &payload,
        DecodeOptions::default(),
        |audio| {
            assert_eq!(audio.sampling_rate(), 48000);
            assert_eq!(audio.channel_count(), 2);
            assert_eq!(audio.bits_per_sample(), 16);
            received += audio.data().len() / 4;
            raw_chunks.push(audio.data().clone());
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(received, frames);

    let (mut model, mut lm, payload, _) = support::fixture();
    let mut converted = Vec::new();
    let options = DecodeOptions {
        output_sample_rate: Some(16000),
        output_channels: Some(1),
        output_bits_per_sample: Some(32),
    };
    decode_encodec_to_sink(&mut model, &mut lm, &payload, options, |audio| {
        assert_eq!(audio.sampling_rate(), 16000);
        assert_eq!(audio.channel_count(), 1);
        assert_eq!(audio.bits_per_sample(), 32);
        converted.extend_from_slice(audio.data());
        Ok(())
    })
    .unwrap();
    // Compare against the established pipeline, including its existing filter
    // delay/tail behavior. The handler must not introduce its own resampler.
    let mut raw_pipeline = DecodePipeline::spawn_raw_pcm_with_options(
        RawPcmFormat::linear16(48000, 2).unwrap(),
        options,
    );
    for chunk in raw_chunks {
        raw_pipeline.send(Bytes::from(chunk)).unwrap();
    }
    raw_pipeline.finish().unwrap();
    let mut expected = Vec::new();
    while let Some(frame) = raw_pipeline.recv() {
        expected.extend_from_slice(frame.unwrap().data());
    }
    assert!(!converted.is_empty());
    assert_eq!(converted, expected);
}

#[test]
fn encodec_propagates_sink_cancellation_without_decoding_the_remaining_windows() {
    let (mut model, mut lm, payload, _) = support::fixture();
    let result = decode_encodec_to_sink(
        &mut model,
        &mut lm,
        &payload,
        DecodeOptions::default(),
        |_| Err(DecodeError::PipelineClosed),
    );
    assert!(matches!(result, Err(DecodeError::PipelineClosed)));
    assert!(model.decoded < 3);
}
