// Native compatibility output: this is the established float assembly path,
// moved unchanged from bitneedle-native-core. Do not quantize or change its
// triangle accumulation while introducing the SoundKit handler.
use anyhow::{bail, Result};
use encodec_rs::ecdc::{decode_ecdc_model_windows, FrameCodec, LmCodec};
use encodec_rs::seam::triangle_overlap_add_planar_frames;
use ndarray::Array3;

pub fn decode_planar_f32(
    codec: &mut dyn FrameCodec,
    lm_codec: &mut dyn LmCodec,
    payload: &[u8],
) -> Result<Array3<f32>> {
    let bundle = codec.metadata().clone();
    let mut decoded_windows = Vec::new();
    let mut direct_audio = Vec::new();
    let info = decode_ecdc_model_windows(
        codec,
        lm_codec,
        payload,
        |info, _window_index, offset, owned_samples, window| {
            let window = window
                .as_slice()
                .ok_or_else(|| anyhow::anyhow!("decoded model window is not contiguous"))?;
            if info.context_samples.is_some() {
                if window.len() != bundle.channels * info.chunk_layout.samples {
                    bail!(
                        "decoded model window has {} values; expected {} channels by {} samples",
                        window.len(),
                        bundle.channels,
                        info.chunk_layout.samples,
                    );
                }
                if decoded_windows.is_empty() {
                    let values = info
                        .window_count
                        .checked_mul(bundle.channels)
                        .and_then(|value| value.checked_mul(info.chunk_layout.samples))
                        .ok_or_else(|| {
                            anyhow::anyhow!("decoded model window buffer overflows usize")
                        })?;
                    decoded_windows.try_reserve_exact(values)?;
                }
                decoded_windows.extend_from_slice(window);
                return Ok(());
            }

            if window.len() != bundle.channels * owned_samples {
                bail!(
                    "decoded owned window has {} values; expected {} channels by {} samples",
                    window.len(),
                    bundle.channels,
                    owned_samples,
                );
            }
            if direct_audio.is_empty() {
                direct_audio.resize(bundle.channels * info.metadata.audio_length, 0.0);
            }
            let destination_end = offset
                .checked_add(owned_samples)
                .ok_or_else(|| anyhow::anyhow!("decoded owned window offset overflows usize"))?;
            if destination_end > info.metadata.audio_length {
                bail!("decoded owned window exceeds the programme length");
            }
            for channel in 0..bundle.channels {
                let source_start = channel * owned_samples;
                let destination_start = channel * info.metadata.audio_length + offset;
                direct_audio[destination_start..destination_start + owned_samples]
                    .copy_from_slice(&window[source_start..source_start + owned_samples]);
            }
            Ok(())
        },
    )?;

    if info.context_samples.is_none() {
        if direct_audio.is_empty() {
            bail!("decoded ECDC programme has no model windows");
        }
        return Ok(Array3::from_shape_vec(
            (1, bundle.channels, info.metadata.audio_length),
            direct_audio,
        )?);
    }

    let mut output = triangle_overlap_add_planar_frames(
        &decoded_windows,
        info.window_count,
        bundle.channels,
        info.chunk_layout.samples,
        info.chunk_layout.stride,
    )?;
    drop(decoded_windows);

    let output_frames = info
        .chunk_layout
        .stride
        .checked_mul(info.window_count.saturating_sub(1))
        .and_then(|value| value.checked_add(info.chunk_layout.samples))
        .ok_or_else(|| anyhow::anyhow!("overlap-add output length overflows usize"))?;
    let audio_length = info.metadata.audio_length;
    let context_samples = info.context_samples.unwrap_or(0);
    let crop_end = context_samples
        .checked_add(audio_length)
        .ok_or_else(|| anyhow::anyhow!("decoded audio crop overflows usize"))?;
    if crop_end > output_frames {
        bail!("decoded overlap-add output has {output_frames} samples, but crop needs {crop_end}");
    }

    for channel in 0..bundle.channels {
        let source_start = channel * output_frames + context_samples;
        let source_end = source_start + audio_length;
        let destination_start = channel * audio_length;
        output.copy_within(source_start..source_end, destination_start);
    }
    output.truncate(bundle.channels * audio_length);
    Ok(Array3::from_shape_vec(
        (1, bundle.channels, audio_length),
        output,
    )?)
}
