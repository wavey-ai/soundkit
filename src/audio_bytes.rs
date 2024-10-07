use byteorder::{BigEndian, ByteOrder, LittleEndian};

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
        let i16_sample = (f32_sample.max(-1.0).min(1.0) * 32767.0) as i16;
        result.push(i16_sample);
    });
    result
}

pub fn f32be_to_i16(data: &[u8]) -> Vec<i16> {
    let sample_count = data.len() / 4;
    let mut result = Vec::with_capacity(sample_count);
    data.chunks_exact(4).for_each(|chunk| {
        let f32_sample = BigEndian::read_f32(chunk);
        let i16_sample = (f32_sample.max(-1.0).min(1.0) * 32767.0) as i16;
        result.push(i16_sample);
    });
    result
}

pub fn f32le_to_i32(data: &[u8]) -> Vec<i32> {
    let sample_count = data.len() / 4;
    let mut result = Vec::with_capacity(sample_count);
    data.chunks_exact(4).for_each(|chunk| {
        let f32_sample = LittleEndian::read_f32(chunk);
        let clamped = f32_sample.max(-1.0).min(1.0);
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
        let clamped = f32_sample.max(-1.0).min(1.0);
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
    let sample_count = input.len() / (channel_count * 2);
    let mut result = vec![Vec::with_capacity(sample_count); channel_count];

    input.chunks_exact(channel_count * 2).for_each(|chunk| {
        chunk
            .chunks_exact(2)
            .enumerate()
            .for_each(|(channel, bytes)| {
                result[channel].push(i16::from_le_bytes([bytes[0], bytes[1]]));
            });
    });

    result
}

pub fn deinterleave_vecs_s24(input: &[u8], channel_count: usize) -> Vec<Vec<i32>> {
    let sample_count = input.len() / (channel_count * 3);
    let mut result = vec![Vec::with_capacity(sample_count); channel_count];

    input.chunks_exact(channel_count * 3).for_each(|chunk| {
        chunk
            .chunks_exact(3)
            .enumerate()
            .for_each(|(channel, bytes)| {
                result[channel].push(s24le_to_i32_sample([bytes[0], bytes[1], bytes[2]]));
            });
    });

    result
}

pub fn deinterleave_vecs_f32(input: &[u8], channel_count: usize) -> Vec<Vec<f32>> {
    let sample_count = input.len() / (channel_count * 4);
    let mut result = vec![Vec::with_capacity(sample_count); channel_count];

    input.chunks_exact(channel_count * 4).for_each(|chunk| {
        chunk
            .chunks_exact(4)
            .enumerate()
            .for_each(|(channel, bytes)| {
                result[channel].push(f32::from_le_bytes(bytes.try_into().unwrap()));
            });
    });

    result
}

pub fn s24le_to_i32_sample(sample_bytes: [u8; 3]) -> i32 {
    let sample = i32::from_le_bytes([sample_bytes[0], sample_bytes[1], sample_bytes[2], 0]);
    (sample << 8) >> 8 // sign extend
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
}
