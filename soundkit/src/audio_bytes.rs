use byteorder::{BigEndian, ByteOrder, LittleEndian};

pub fn i16le_to_f32(bytes: &[u8]) -> Vec<f32> {
    assert!(
        bytes.len().is_multiple_of(2),
        "Bytes length must be a multiple of 2"
    );
    bytes
        .chunks(2)
        .map(|chunk| {
            let i16_sample = i16::from_le_bytes(chunk.try_into().unwrap());
            i16_sample as f32 / 32768.0
        })
        .collect()
}

pub fn i16_to_i16le(data: &[i16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 2); // Each i16 is 2 bytes
    for value in data {
        bytes.extend(&value.to_le_bytes());
    }
    bytes
}

pub fn i16le_to_i16(bytes: &[u8]) -> Vec<i16> {
    assert!(
        bytes.len().is_multiple_of(2),
        "Bytes length must be a multiple of 2"
    );
    bytes
        .chunks(2)
        .map(|chunk| i16::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

pub fn s24le_to_i32(data: &[u8]) -> Vec<i32> {
    let sample_count = data.len() / 3;
    let mut result = Vec::with_capacity(sample_count);
    data.chunks_exact(3).for_each(|chunk| {
        let unsigned_sample = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], 0]);
        let signed_sample = if unsigned_sample & 0x800000 != 0 {
            (unsigned_sample | 0xFF000000) as i32
        } else {
            unsigned_sample as i32
        };
        result.push(signed_sample);
    });
    result
}

pub fn s24le_to_i16(data: &[u8]) -> Vec<i16> {
    let sample_count = data.len() / 3;
    let mut result = Vec::with_capacity(sample_count);
    data.chunks_exact(3).for_each(|chunk| {
        let unsigned_sample = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], 0]);
        let signed_sample = if unsigned_sample & 0x800000 != 0 {
            (unsigned_sample | 0xFF000000) as i32
        } else {
            unsigned_sample as i32
        };
        result.push((signed_sample >> 8) as i16);
    });
    result
}

pub fn s24be_to_i16(data: &[u8]) -> Vec<i16> {
    let sample_count = data.len() / 3;
    let mut result = Vec::with_capacity(sample_count);
    data.chunks_exact(3).for_each(|chunk| {
        let unsigned_sample = u32::from_be_bytes([0, chunk[0], chunk[1], chunk[2]]);
        let signed_sample = if unsigned_sample & 0x800000 != 0 {
            (unsigned_sample | 0xFF000000) as i32
        } else {
            unsigned_sample as i32
        };
        result.push((signed_sample >> 8) as i16);
    });
    result
}

pub fn s32le_to_i32(data: &[u8]) -> Vec<i32> {
    let sample_count = data.len() / 4;
    let mut result = Vec::with_capacity(sample_count);
    data.chunks_exact(4).for_each(|chunk| {
        let s32_sample = i32::from_le_bytes(chunk.try_into().unwrap());
        result.push(s32_sample);
    });
    result
}

pub fn s32be_to_i32(data: &[u8]) -> Vec<i32> {
    let sample_count: usize = data.len() / 4;
    let mut result: Vec<i32> = Vec::with_capacity(sample_count);
    data.chunks_exact(4).for_each(|chunk| {
        let s32_sample: i32 = i32::from_be_bytes(chunk.try_into().unwrap());
        result.push(s32_sample);
    });
    result
}

pub fn s32le_to_s24(data: &[u8]) -> Vec<i32> {
    let sample_count = data.len() / 4;
    let mut result = Vec::with_capacity(sample_count);
    data.chunks_exact(4).for_each(|chunk| {
        let s32_sample = i32::from_le_bytes(chunk.try_into().unwrap());
        let s24_sample = s32_sample & 0x00FFFFFF;
        result.push(s24_sample);
    });
    result
}

pub fn s32be_to_s24(data: &[u8]) -> Vec<i32> {
    let sample_count: usize = data.len() / 4;
    let mut result: Vec<i32> = Vec::with_capacity(sample_count);
    data.chunks_exact(4).for_each(|chunk| {
        let s32_sample: i32 = i32::from_be_bytes(chunk.try_into().unwrap());
        let s24_sample: i32 = s32_sample & 0x00FFFFFF;
        result.push(s24_sample);
    });
    result
}

pub fn s32le_to_f32(data: &[u8]) -> Vec<f32> {
    let sample_count = data.len() / 4;
    let mut result = Vec::with_capacity(sample_count);
    data.chunks_exact(4).for_each(|chunk| {
        let s32_sample = i32::from_le_bytes(chunk.try_into().unwrap());
        let f32_sample = (s32_sample as f32) / (2.0f32.powi(31) - 1.0);
        result.push(f32_sample);
    });
    result
}

pub fn s32be_to_f32(data: &[u8]) -> Vec<f32> {
    let sample_count = data.len() / 4;
    let mut result = Vec::with_capacity(sample_count);
    data.chunks_exact(4).for_each(|chunk| {
        let s32_sample = i32::from_be_bytes(chunk.try_into().unwrap());
        let f32_sample = (s32_sample as f32) / (2.0f32.powi(31) - 1.0);
        result.push(f32_sample);
    });
    result
}

pub fn s32le_to_i16(data: &[u8]) -> Vec<i16> {
    let sample_count: usize = data.len() / 4;
    let mut result: Vec<i16> = Vec::with_capacity(sample_count);
    data.chunks_exact(4).for_each(|chunk| {
        let s32_sample: i32 = i32::from_le_bytes(chunk.try_into().unwrap());
        let i16_sample: i16 = (s32_sample >> 16) as i16;
        result.push(i16_sample);
    });
    result
}

pub fn s32be_to_i16(data: &[u8]) -> Vec<i16> {
    let sample_count: usize = data.len() / 4;
    let mut result: Vec<i16> = Vec::with_capacity(sample_count);
    data.chunks_exact(4).for_each(|chunk| {
        let s32_sample: i32 = i32::from_be_bytes(chunk.try_into().unwrap());
        let i16_sample: i16 = (s32_sample >> 16) as i16;
        result.push(i16_sample);
    });
    result
}

pub fn f32le_to_i16(data: &[u8]) -> Vec<i16> {
    let sample_count = data.len() / 4;
    let mut result = Vec::with_capacity(sample_count);
    data.chunks_exact(4).for_each(|chunk| {
        let f32_sample = LittleEndian::read_f32(chunk);
        let i16_sample = (f32_sample.clamp(-1.0, 1.0) * 32767.0) as i16;
        result.push(i16_sample);
    });
    result
}

pub fn f32be_to_i16(data: &[u8]) -> Vec<i16> {
    let sample_count = data.len() / 4;
    let mut result = Vec::with_capacity(sample_count);
    data.chunks_exact(4).for_each(|chunk| {
        let f32_sample = BigEndian::read_f32(chunk);
        let i16_sample = (f32_sample.clamp(-1.0, 1.0) * 32767.0) as i16;
        result.push(i16_sample);
    });
    result
}

pub fn f32le_to_i32(data: &[u8]) -> Vec<i32> {
    let sample_count = data.len() / 4;
    let mut result = Vec::with_capacity(sample_count);
    data.chunks_exact(4).for_each(|chunk| {
        let f32_sample = LittleEndian::read_f32(chunk);
        let clamped = f32_sample.clamp(-1.0, 1.0);
        let sample = if clamped >= 0.0 {
            (clamped * i32::MAX as f32) as i32
        } else {
            (clamped * -(i32::MIN as f32)) as i32
        };
        result.push(sample);
    });
    result
}

pub fn f32le_to_s24(data: &[u8]) -> Vec<i32> {
    let sample_count = data.len() / 4;
    let mut result = Vec::with_capacity(sample_count);
    data.chunks_exact(4).for_each(|chunk| {
        let f32_sample = f32::from_le_bytes(chunk.try_into().unwrap());
        let clamped = f32_sample.clamp(-1.0, 1.0);
        let s24_max = 8388607; // 2^23 - 1
        let sample = if clamped >= 0.0 {
            (clamped * s24_max as f32) as i32
        } else {
            (clamped * (s24_max + 1) as f32) as i32
        };
        result.push(sample);
    });
    result
}

pub fn s16be_to_i16(data: &[u8]) -> Vec<i16> {
    let sample_count = data.len() / 2;
    let mut result = Vec::with_capacity(sample_count);
    data.chunks_exact(2).for_each(|chunk| {
        result.push(BigEndian::read_i16(chunk));
    });
    result
}

pub fn s16le_to_i16(data: &[u8]) -> Vec<i16> {
    let sample_count = data.len() / 2;
    let mut result = Vec::with_capacity(sample_count);
    data.chunks_exact(2).for_each(|chunk| {
        result.push(LittleEndian::read_i16(chunk));
    });
    result
}

pub fn s16le_to_i32(data: &[u8]) -> Vec<i32> {
    let sample_count = data.len() / 2;
    let mut result = Vec::with_capacity(sample_count);
    data.chunks_exact(2).for_each(|chunk| {
        let sample = LittleEndian::read_i16(chunk) as i32;
        result.push(sample);
    });
    result
}

pub fn interleave_vecs_i16(channels: &[Vec<i16>]) -> Vec<u8> {
    let channel_count = channels.len();
    let sample_count = channels[0].len();
    let mut result = Vec::with_capacity(channel_count * sample_count * 2);

    for i in 0..sample_count {
        for channel in channels {
            result.extend_from_slice(&channel[i].to_le_bytes());
        }
    }

    result
}

pub fn deinterleave_vecs_i16(input: &[u8], channel_count: usize) -> Vec<Vec<i16>> {
    deinterleave_vecs(input, channel_count, i16::from_le_bytes)
}

pub fn deinterleave_vecs_s24(input: &[u8], channel_count: usize) -> Vec<Vec<i32>> {
    deinterleave_vecs(input, channel_count, s24le_to_i32_sample)
}

pub fn deinterleave_vecs_f32(input: &[u8], channel_count: usize) -> Vec<Vec<f32>> {
    deinterleave_vecs(input, channel_count, f32::from_le_bytes)
}

/// One vector per channel of the whole interleaved frames in `input`, each
/// sample `W` bytes. A channel count of zero panics.
///
/// Each channel is collected in its own pass over the frames: an exact-size
/// collect writes the samples without a capacity check per sample.
fn deinterleave_vecs<T, const W: usize>(
    input: &[u8],
    channel_count: usize,
    sample: impl Fn([u8; W]) -> T,
) -> Vec<Vec<T>> {
    let frame_bytes = channel_count * W;
    // A channel count of zero divides by zero here.
    let _frames = input.len() / frame_bytes;
    (0..channel_count)
        .map(|channel| {
            let offset = channel * W;
            input
                .chunks_exact(frame_bytes)
                .map(|frame| {
                    let mut bytes = [0u8; W];
                    bytes.copy_from_slice(&frame[offset..offset + W]);
                    sample(bytes)
                })
                .collect()
        })
        .collect()
}

/// Little-endian bytes of `samples`, written into a buffer of the final size.
pub fn i16s_to_le_bytes(samples: &[i16]) -> Vec<u8> {
    let mut bytes = vec![0u8; samples.len() * 2];
    for (target, sample) in bytes.chunks_exact_mut(2).zip(samples) {
        target.copy_from_slice(&sample.to_le_bytes());
    }
    bytes
}

/// PCM bytes of samples held in i32 at `bits_per_sample`: unsigned 8-bit up
/// to 8 bits, then 16-bit, 24-bit and 32-bit little-endian.
pub fn i32s_to_pcm_bytes(samples: &[i32], bits_per_sample: u8) -> Vec<u8> {
    fn write<const W: usize>(samples: &[i32], bytes: impl Fn(i32) -> [u8; W]) -> Vec<u8> {
        let mut out = vec![0u8; samples.len() * W];
        for (target, &sample) in out.chunks_exact_mut(W).zip(samples) {
            target.copy_from_slice(&bytes(sample));
        }
        out
    }
    match bits_per_sample {
        1..=8 => samples.iter().map(|&sample| (sample + 128) as u8).collect(),
        9..=16 => write(samples, |sample| (sample as i16).to_le_bytes()),
        17..=24 => write(samples, |sample| {
            let [a, b, c, _] = sample.to_le_bytes();
            [a, b, c]
        }),
        _ => write(samples, i32::to_le_bytes),
    }
}

pub fn s24le_to_i32_sample(sample_bytes: [u8; 3]) -> i32 {
    let sample = i32::from_le_bytes([sample_bytes[0], sample_bytes[1], sample_bytes[2], 0]);
    (sample << 8) >> 8 // sign extend
}

pub fn stereo_to_mono_take_left(input: &[i16]) -> Vec<i16> {
    assert!(
        input.len().is_multiple_of(2),
        "Stereo buffer must contain an even number of samples"
    );

    let frames = input.len() / 2;
    let mut out = Vec::with_capacity(frames);
    for i in 0..frames {
        out.push(input[2 * i]);
    }
    out
}

pub fn stereo_to_mono_inplace_take_left(samples: &mut [i16]) -> &mut [i16] {
    assert!(
        samples.len().is_multiple_of(2),
        "Stereo buffer must contain an even number of samples"
    );

    let frames = samples.len() / 2;
    for i in 0..frames {
        samples[i] = samples[2 * i];
    }
    &mut samples[..frames]
}

pub fn stereo_to_mono_avg(input: &[i16]) -> Vec<i16> {
    assert!(
        input.len().is_multiple_of(2),
        "Stereo buffer must contain an even number of samples"
    );

    let frames = input.len() / 2;
    let mut out = Vec::with_capacity(frames);
    for i in 0..frames {
        let l = input[2 * i] as i32;
        let r = input[2 * i + 1] as i32;
        out.push(((l + r) / 2) as i16);
    }
    out
}

pub fn stereo_to_mono_inplace_avg(samples: &mut [i16]) -> &mut [i16] {
    assert!(
        samples.len().is_multiple_of(2),
        "Stereo buffer must contain an even number of samples"
    );

    let frames = samples.len() / 2;
    for i in 0..frames {
        let l = samples[2 * i] as i32;
        let r = samples[2 * i + 1] as i32;
        samples[i] = ((l + r) / 2) as i16;
    }
    &mut samples[..frames]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deinterleave_vecs_i16() {
        let input = vec![1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0]; // Little Endian u16 values [1, 2, 3, 4, 5, 6]
        let result = deinterleave_vecs_i16(&input, 2);
        assert_eq!(result, vec![vec![1, 3, 5], vec![2, 4, 6]]);
    }

    #[test]
    fn test_interleave_vecs_i16() {
        let input = vec![vec![1, 3, 5], vec![2, 4, 6]];
        let result = interleave_vecs_i16(&input);
        assert_eq!(result, vec![1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0]);
    }

    #[test]
    fn test_deinterleave_vecs_s24() {
        let input = vec![1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0, 5, 0, 0, 6, 0, 0]; // Little Endian u24 values [1, 2, 3, 4, 5, 6]
        let result = deinterleave_vecs_s24(&input, 2);
        assert_eq!(result, vec![vec![1, 3, 5], vec![2, 4, 6]]);
    }

    #[test]
    fn test_deinterleave_vecs_f32() {
        let input = vec![
            0, 0, 128, 63, 0, 0, 0, 64, // f32: 1.0, 2.0
            0, 0, 64, 64, 0, 0, 128, 64, // f32: 3.0, 4.0
            0, 0, 160, 64, 0, 0, 192, 64, // f32: 5.0, 6.0
        ];
        let result = deinterleave_vecs_f32(&input, 2);
        assert_eq!(
            result,
            vec![
                vec![1.0, 3.0, 5.0], // channel 1
                vec![2.0, 4.0, 6.0], // channel 2
            ]
        );
    }

    #[test]
    fn test_i16le_to_f32() {
        // Little-endian bytes for i16 values: 0, 16384, 32767, -16384, -32768
        let input = vec![
            0, 0, // 0
            0, 64, // 16384
            255, 127, // 32767
            0, 192, // -16384
            0, 128, // -32768
        ];
        let expected = [0.0, 0.5, 0.9999694, -0.5, -1.0];
        let result = i16le_to_f32(&input);

        assert_eq!(result.len(), expected.len());
        for (i, (&expected, &actual)) in expected.iter().zip(result.iter()).enumerate() {
            assert!(
                (expected - actual).abs() < 0.0001,
                "Sample {} mismatch: expected {}, got {}",
                i,
                expected,
                actual
            );
        }
    }

    #[test]
    fn test_stereo_to_mono_take_left() {
        let input = vec![10, 20, -30, -40, 50, 60];
        let result = stereo_to_mono_take_left(&input);
        assert_eq!(result, vec![10, -30, 50]);
    }

    #[test]
    fn test_stereo_to_mono_inplace_take_left() {
        let mut samples = vec![10, 20, -30, -40, 50, 60];
        let mono = stereo_to_mono_inplace_take_left(&mut samples);
        assert_eq!(mono, &[10, -30, 50]);
    }

    #[test]
    fn test_stereo_to_mono_avg() {
        let input = vec![100, -100, 50, 150, -200, 200];
        let result = stereo_to_mono_avg(&input);
        assert_eq!(result, vec![0, 100, 0]);
    }

    #[test]
    fn test_stereo_to_mono_inplace_avg() {
        let mut samples = vec![100, -100, 50, 150, -200, 200];
        let mono = stereo_to_mono_inplace_avg(&mut samples);
        assert_eq!(mono, &[0, 100, 0]);
    }

    /// The deinterleavers as they were: one sample pushed at a time.
    fn reference_deinterleave<T: Clone, const W: usize>(
        input: &[u8],
        channel_count: usize,
        sample: impl Fn([u8; W]) -> T,
    ) -> Vec<Vec<T>> {
        let sample_count = input.len() / (channel_count * W);
        let mut result = vec![Vec::with_capacity(sample_count); channel_count];
        input.chunks_exact(channel_count * W).for_each(|chunk| {
            chunk
                .chunks_exact(W)
                .enumerate()
                .for_each(|(channel, bytes)| {
                    result[channel].push(sample(bytes.try_into().unwrap()));
                });
        });
        result
    }

    #[test]
    fn deinterleavers_match_sample_at_a_time_deinterleavers() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let input: Vec<u8> = (0..4_099)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as u8
            })
            .collect();
        for channels in 1..=8 {
            for length in [0usize, 1, 5, 23, 24, 25, 1_000, 4_099] {
                let bytes = &input[..length];
                assert_eq!(
                    deinterleave_vecs_i16(bytes, channels),
                    reference_deinterleave(bytes, channels, i16::from_le_bytes)
                );
                assert_eq!(
                    deinterleave_vecs_s24(bytes, channels),
                    reference_deinterleave(bytes, channels, s24le_to_i32_sample)
                );
                let bits = |planes: Vec<Vec<f32>>| -> Vec<Vec<u32>> {
                    planes
                        .into_iter()
                        .map(|plane| plane.into_iter().map(f32::to_bits).collect())
                        .collect()
                };
                assert_eq!(
                    bits(deinterleave_vecs_f32(bytes, channels)),
                    bits(reference_deinterleave(bytes, channels, f32::from_le_bytes))
                );
            }
        }
        assert!(std::panic::catch_unwind(|| deinterleave_vecs_s24(&input, 0)).is_err());
    }

    #[test]
    fn pcm_byte_writers_match_sample_at_a_time_writers() {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let samples: Vec<i32> = (0..3_001)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state as i32) >> (state >> 59)
            })
            .collect();
        let shorts: Vec<i16> = samples.iter().map(|&sample| sample as i16).collect();
        let mut want = Vec::new();
        for sample in &shorts {
            want.extend_from_slice(&sample.to_le_bytes());
        }
        assert_eq!(i16s_to_le_bytes(&shorts), want);
        for bits in [1u8, 8, 9, 16, 17, 24, 25, 32, 0] {
            let input: Vec<i32> = if bits <= 8 && bits > 0 {
                samples.iter().map(|&sample| sample >> 24).collect()
            } else {
                samples.clone()
            };
            let mut want = Vec::new();
            match bits {
                1..=8 => input
                    .iter()
                    .for_each(|&sample| want.push((sample + 128) as u8)),
                9..=16 => input
                    .iter()
                    .for_each(|&sample| want.extend_from_slice(&(sample as i16).to_le_bytes())),
                17..=24 => input
                    .iter()
                    .for_each(|&sample| want.extend_from_slice(&sample.to_le_bytes()[0..3])),
                _ => input
                    .iter()
                    .for_each(|&sample| want.extend_from_slice(&sample.to_le_bytes())),
            }
            assert_eq!(i32s_to_pcm_bytes(&input, bits), want, "{bits} bits");
        }
    }
}
