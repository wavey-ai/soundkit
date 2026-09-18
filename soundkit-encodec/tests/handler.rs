use anyhow::bail;
use encodec_rs::seam::{triangle_overlap_add_planar_frames, SeamCursor};
use soundkit_encodec::{decode_planar_f32, decode_to_sink, EncodecPcmDecoder};

mod support;
use support::{fixture, window, MODEL, OWNED};

#[test]
fn streamed_cached_silent_and_retained_pcm_are_bit_exact_with_original_cursor() {
    for channels in [1, 2] {
        for retain in [false, true] {
            let length = OWNED * 3 + 17;
            let mut old = SeamCursor::new(channels, MODEL, OWNED, length, 4, retain).unwrap();
            let mut handler =
                EncodecPcmDecoder::new(channels, 48000, MODEL, OWNED, length, 4, retain).unwrap();
            let cached = (0..OWNED * channels)
                .map(|i| (i as i16).wrapping_mul(7))
                .collect::<Vec<_>>();
            assert!(old.add_cached_range(OWNED, OWNED * 2, &cached));
            assert!(handler.add_cached_range(OWNED, OWNED * 2, &cached));
            let mut frames = 0;
            for index in 0..4 {
                if index == 2 {
                    old.add_silent_frame(index).unwrap();
                    handler.add_silent_frame(index).unwrap();
                } else {
                    let samples = window(channels, index);
                    old.add_decoded_frame(index, &samples).unwrap();
                    handler.add_decoded_frame(index, &samples).unwrap();
                }
                let expected = old.emit_after_batch(index + 1);
                let actual = handler.emit_after_batch(index + 1).unwrap();
                assert_eq!(actual.len(), expected.len());
                for (actual, expected) in actual.into_iter().zip(expected) {
                    assert_eq!(
                        (actual.chunk_index, actual.start_frame, actual.end_frame),
                        (
                            expected.chunk_index,
                            expected.start_frame,
                            expected.end_frame
                        )
                    );
                    assert_eq!(actual.planar, old.segment_pcm(&expected));
                    assert_eq!(actual.start_frame, frames);
                    frames = actual.end_frame;
                    let expected_samples = actual.planar.clone();
                    let count = actual.end_frame - actual.start_frame;
                    let audio = actual.into_audio_data();
                    assert_eq!(audio.sampling_rate(), 48000);
                    assert_eq!(audio.channel_count(), channels as u8);
                    for (i, bytes) in audio.data().chunks_exact(2).enumerate() {
                        assert_eq!(
                            i16::from_le_bytes([bytes[0], bytes[1]]),
                            expected_samples[(i % channels) * count + i / channels]
                        );
                    }
                }
            }
            let expected = old.flush();
            let actual = handler.flush();
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.into_iter().zip(expected) {
                assert_eq!(actual.planar, old.segment_pcm(&expected));
                assert_eq!(actual.start_frame, frames);
                frames = actual.end_frame;
            }
            assert_eq!(frames, length);
            assert_eq!(handler.result_pcm(), old.full_channel_data());
            assert!(handler.flush().is_empty());
        }
    }
}

#[test]
fn extracted_ecdc_streams_through_the_handler_without_changing_model_output() {
    let (mut model, mut lm, payload, frames) = fixture();
    let mut bytes = Vec::new();
    let mut blocks = 0;
    let info = decode_to_sink(&mut model, &mut lm, &payload, |audio| {
        assert!(audio.data().len() <= OWNED * 2 * 2);
        blocks += 1;
        bytes.extend_from_slice(audio.data());
        Ok(())
    })
    .unwrap();
    assert_eq!(info.metadata.audio_length, frames);
    assert_eq!(bytes.len(), frames * 2 * 2);
    assert_eq!(blocks, 3);
    let mut reference = SeamCursor::new(2, MODEL, OWNED, frames, 3, true).unwrap();
    for i in 0..3 {
        reference.add_decoded_frame(i, &window(2, i)).unwrap();
        reference.emit_after_batch(i + 1);
    }
    reference.flush();
    for (i, bytes) in bytes.chunks_exact(2).enumerate() {
        assert_eq!(
            i16::from_le_bytes([bytes[0], bytes[1]]),
            reference.full_channel_data()[(i % 2) * frames + i / 2]
        );
    }
}

#[test]
fn native_float_compatibility_is_bit_exact_with_the_previous_assembly() {
    let (mut model, mut lm, payload, frames) = fixture();
    let actual = decode_planar_f32(&mut model, &mut lm, &payload).unwrap();
    let windows = (0..3).flat_map(|i| window(2, i)).collect::<Vec<_>>();
    let old = triangle_overlap_add_planar_frames(&windows, 3, 2, MODEL, OWNED).unwrap();
    let old_frames = OWNED * 2 + MODEL;
    for channel in 0..2 {
        for i in 0..frames {
            assert_eq!(
                actual[[0, channel, i]].to_bits(),
                old[channel * old_frames + 480 + i].to_bits()
            );
        }
    }
}

#[test]
fn sink_failure_stops_model_work_and_bad_packets_fail() {
    let (mut model, mut lm, mut payload, _) = fixture();
    let error = decode_to_sink(&mut model, &mut lm, &payload, |_| {
        bail!("cancelled by sink")
    })
    .unwrap_err();
    assert!(error.to_string().contains("cancelled by sink"));
    assert!(model.decoded < 3);
    let last = payload.len() - 1;
    payload[last] ^= 0x80;
    assert!(decode_to_sink(&mut model, &mut lm, &payload, |_| Ok(())).is_err());
}

#[test]
fn malformed_windows_and_out_of_range_indices_fail_before_output() {
    let mut handler = EncodecPcmDecoder::new(2, 48000, MODEL, OWNED, OWNED, 1, false).unwrap();
    assert!(handler.add_decoded_frame(0, &[0.0; 2]).is_err());
    assert!(handler.add_silent_frame(1).is_err());
    assert!(handler.emit_after_batch(2).is_err());
    assert!(handler.result_pcm().is_empty());
}
