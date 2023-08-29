pub fn f32_to_i16(input: Vec<f32>) -> Vec<i16> {
    let mut output: Vec<i16> = Vec::with_capacity(input.len());

    for value in input {
        let clamped_value = value.max(-1.0).min(1.0);
        let scaled_value = (clamped_value * 32767.0) as i16;
        output.push(scaled_value);
    }

    output
}
