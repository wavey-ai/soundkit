pub fn f32_to_i16(input: Vec<f32>) -> Vec<i16> {
    let mut output: Vec<i16> = Vec::with_capacity(input.len());

    for value in input {
        let clamped_value = value.max(-1.0).min(1.0);
        let scaled_value = (clamped_value * 32767.0) as i16;
        output.push(scaled_value);
    }

    output
}

pub fn i16_to_f32(input: Vec<i16>) -> Vec<f32> {
    let mut output: Vec<f32> = Vec::with_capacity(input.len());

    for value in input {
        let scaled_value = value as f32 / 32768.0; // Division by 32768 instead of 32767 for better centering around 0
        output.push(scaled_value);
    }

    output
}
