use byteorder::{BigEndian, ByteOrder, LittleEndian};

pub fn s24le_to_i16(data: &[u8]) -> Vec<i16> {
    // Converts a signed 24-bit little-endian PCM audio stream to a vector of i16 samples.
    let mut result = Vec::new();
    let mut i = 0;

    while i < data.len() {
        let sample = LittleEndian::read_i24(&data[i..i + 3]);
        let i16_sample = (sample >> 8) as i16; // Convert to 16-bit
        result.push(i16_sample);
        i += 3;
    }

    result
}

pub fn s24be_to_i16(data: &[u8]) -> Vec<i16> {
    // Converts a signed 24-bit big-endian PCM audio stream to a vector of i16 samples.
    let mut result = Vec::new();
    let mut i = 0;

    while i < data.len() {
        let sample = BigEndian::read_i24(&data[i..i + 3]);
        let i16_sample = (sample >> 8) as i16; // Convert to 16-bit
        result.push(i16_sample);
        i += 3;
    }

    result
}

pub fn i32le_to_i16(data: &[u8]) -> Vec<i16> {
    // Converts a 32-bit little-endian integer PCM audio stream to a vector of i16 samples.
    let mut result = Vec::new();
    let mut i = 0;

    while i < data.len() {
        let sample = LittleEndian::read_i32(&data[i..i + 4]);
        let i16_sample = (sample >> 16) as i16; // Convert to 16-bit
        result.push(i16_sample);
        i += 4;
    }

    result
}

pub fn i32be_to_i16(data: &[u8]) -> Vec<i16> {
    // Converts a 32-bit big-endian integer PCM audio stream to a vector of i16 samples.
    let mut result = Vec::new();
    let mut i = 0;

    while i < data.len() {
        let sample = BigEndian::read_i32(&data[i..i + 4]);
        let i16_sample = (sample >> 16) as i16; // Convert to 16-bit
        result.push(i16_sample);
        i += 4;
    }

    result
}

pub fn f32le_to_i16(data: &[u8]) -> Vec<i16> {
    // Converts a 32-bit little-endian float PCM audio stream to a vector of i16 samples.
    let mut result = Vec::new();
    let mut i = 0;

    while i < data.len() {
        let f32_sample = LittleEndian::read_f32(&data[i..i + 4]);
        let i16_sample = (f32_sample * 32767.0) as i16; // Convert to 16-bit integer range
        result.push(i16_sample);
        i += 4;
    }

    result
}

pub fn f32be_to_i16(data: &[u8]) -> Vec<i16> {
    // Converts a 32-bit big-endian float PCM audio stream to a vector of i16 samples.
    let mut result = Vec::new();
    let mut i = 0;

    while i < data.len() {
        let f32_sample = BigEndian::read_f32(&data[i..i + 4]);
        let i16_sample = (f32_sample * 32767.0) as i16; // Convert to 16-bit integer range
        result.push(i16_sample);
        i += 4;
    }

    result
}

pub fn s16be_to_i16(data: &[u8]) -> Vec<i16> {
    // Converts a 16-bit big-endian PCM audio stream to a vector of i16 samples.
    let mut result = Vec::new();
    let mut i = 0;

    while i < data.len() {
        // Read the sample from two bytes in big-endian order.
        let sample = BigEndian::read_i16(&data[i..i + 2]);

        result.push(sample);
        i += 2;
    }

    result
}

pub fn s16le_to_i16(data: &[u8]) -> Vec<i16> {
    // Converts a 16-bit little-endian PCM audio stream to a vector of i16 samples.
    let mut result = Vec::new();
    let mut i = 0;

    while i < data.len() {
        // Read the sample from two bytes in little-endian order.
        let sample = LittleEndian::read_i16(&data[i..i + 2]);

        result.push(sample);
        i += 2;
    }

    result
}

pub fn interleave_vecs_i16(channels: &[Vec<i16>]) -> Vec<u8> {
    let channel_count = channels.len();
    let sample_size = channels[0].len();
    let mut result = vec![0; channel_count * sample_size * 2];

    for i in 0..sample_size {
        for channel in 0..channel_count {
            let value = channels[channel][i];
            let bytes = value.to_le_bytes();
            let start = (i * channel_count + channel) * 2;
            result[start] = bytes[0];
            result[start + 1] = bytes[1];
        }
    }

    result
}

pub fn deinterleave_vecs_i16(input: &[u8], channel_count: usize) -> Vec<Vec<i16>> {
    let sample_size = input.len() / (channel_count * 2);
    let mut result = vec![vec![0; sample_size]; channel_count];

    for i in 0..sample_size {
        for channel in 0..channel_count {
            let start = (i * channel_count + channel) * 2;
            let value = i16::from_le_bytes([input[start], input[start + 1]]);
            result[channel][i] = value;
        }
    }

    result
}

pub fn deinterleave_vecs_24bit(input: &[u8], channel_count: usize) -> Vec<Vec<i32>> {
    let sample_size = input.len() / (channel_count * 3);
    let mut result = vec![vec![0; sample_size]; channel_count];

    for i in 0..sample_size {
        for channel in 0..channel_count {
            let start = (i * channel_count + channel) * 3;
            let value = i32::from_le_bytes([input[start], input[start + 1], input[start + 2], 0]);
            result[channel][i] = value;
        }
    }

    result
}

pub fn deinterleave_vecs_f32(input: &[u8], channel_count: usize) -> Vec<Vec<f32>> {
    let sample_size = input.len() / (channel_count * 4);
    let mut result = vec![vec![0.0; sample_size]; channel_count];

    for i in 0..sample_size {
        for channel in 0..channel_count {
            let start = (i * channel_count + channel) * 4;
            let value = f32::from_le_bytes(input[start..start + 4].try_into().unwrap());
            result[channel][i] = value;
        }
    }

    result
}

pub fn s24le_to_i32(sample_bytes: [u8; 3]) -> i32 {
    let sample = i32::from(sample_bytes[0])
        | (i32::from(sample_bytes[1]) << 8)
        | (i32::from(sample_bytes[2]) << 16);

    // If the most significant bit of the 24-bit sample is set (negative value),
    // sign-extend it to 32 bits
    if sample & 0x800000 != 0 {
        sample | (0xFF000000u32 as i32)
    } else {
        sample
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Read;

    fn read_test_file(filename: &str) -> Vec<u8> {
        let mut file = File::open(filename).unwrap();
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();
        buffer
    }

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
    fn test_deinterleave_vecs_24bit() {
        let input = vec![1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0, 5, 0, 0, 6, 0, 0]; // Little Endian u24 values [1, 2, 3, 4, 5, 6]
        let result = deinterleave_vecs_24bit(&input, 2);
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
