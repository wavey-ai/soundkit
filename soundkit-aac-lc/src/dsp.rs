use crate::error::{AacLcError, Result};
use crate::ics::{WindowSequence, WindowShape};
use rustfft::{num_complex::Complex32, Fft, FftPlanner};
use std::{
    fmt,
    sync::{Arc, OnceLock},
};

pub const LONG_SPECTRUM_LEN: usize = 1024;
pub const LONG_WINDOW_LEN: usize = LONG_SPECTRUM_LEN * 2;
pub const SHORT_SPECTRUM_LEN: usize = 128;
pub const SHORT_WINDOW_LEN: usize = SHORT_SPECTRUM_LEN * 2;
const PCM_F32_SCALE: f32 = 1.0 / 32768.0;
const POW43_TABLE_LEN: usize = 8192;
const SCALE_FACTOR_TABLE_MIN: i16 = -256;
const SCALE_FACTOR_TABLE_MAX: i16 = 511;
const SCALE_FACTOR_TABLE_LEN: usize =
    (SCALE_FACTOR_TABLE_MAX as i32 - SCALE_FACTOR_TABLE_MIN as i32 + 1) as usize;

#[derive(Debug, Clone)]
pub struct AacDsp {
    long_sine: Vec<f32>,
    long_kbd: Vec<f32>,
    short_sine: Vec<f32>,
    short_kbd: Vec<f32>,
    long_imdct: ImdctTransform,
    short_imdct: ImdctTransform,
}

impl AacDsp {
    pub fn new() -> Self {
        warm_dequant_tables();
        Self {
            long_sine: sine_window(LONG_WINDOW_LEN),
            long_kbd: kbd_window(LONG_WINDOW_LEN, 4.0),
            short_sine: sine_window(SHORT_WINDOW_LEN),
            short_kbd: kbd_window(SHORT_WINDOW_LEN, 6.0),
            long_imdct: ImdctTransform::new(LONG_SPECTRUM_LEN),
            short_imdct: ImdctTransform::new(SHORT_SPECTRUM_LEN),
        }
    }

    pub fn long_window(&self, shape: WindowShape) -> &[f32] {
        match shape {
            WindowShape::Sine => &self.long_sine,
            WindowShape::KaiserBesselDerived => &self.long_kbd,
        }
    }

    pub fn short_window(&self, shape: WindowShape) -> &[f32] {
        match shape {
            WindowShape::Sine => &self.short_sine,
            WindowShape::KaiserBesselDerived => &self.short_kbd,
        }
    }

    pub fn long_imdct_transform(&self) -> &ImdctTransform {
        &self.long_imdct
    }

    pub fn short_imdct_transform(&self) -> &ImdctTransform {
        &self.short_imdct
    }
}

impl Default for AacDsp {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone)]
pub struct ImdctTransform {
    input_len: usize,
    output_len: usize,
    output_scale: f32,
    twiddle: Vec<Complex32>,
    fft: Arc<dyn Fft<f32>>,
    fft_scratch_len: usize,
}

impl fmt::Debug for ImdctTransform {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ImdctTransform")
            .field("input_len", &self.input_len)
            .field("output_len", &self.output_len)
            .field("output_scale", &self.output_scale)
            .field("fft_scratch_len", &self.fft_scratch_len)
            .finish_non_exhaustive()
    }
}

impl ImdctTransform {
    pub fn new(input_len: usize) -> Self {
        assert!(input_len != 0 && input_len.is_power_of_two());

        let nf = input_len as f32;
        let output_len = input_len * 2;
        let output_scale = PCM_F32_SCALE / nf;
        let fft_len = input_len / 2;
        let twiddle_scale = output_scale.sqrt();
        let twiddle = (0..fft_len)
            .map(|bin| {
                complex_cis(std::f32::consts::PI / nf * (bin as f32 + 0.125)) * twiddle_scale
            })
            .collect();
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(fft_len);
        let fft_scratch_len = fft.get_inplace_scratch_len();

        Self {
            input_len,
            output_len,
            output_scale,
            twiddle,
            fft,
            fft_scratch_len,
        }
    }

    pub const fn input_len(&self) -> usize {
        self.input_len
    }

    pub const fn output_len(&self) -> usize {
        self.output_len
    }

    pub const fn fft_scratch_len(&self) -> usize {
        self.fft_scratch_len
    }

    pub const fn fft_len(&self) -> usize {
        self.input_len / 2
    }
}

fn complex_cis(angle: f32) -> Complex32 {
    Complex32::new(angle.cos(), angle.sin())
}

#[derive(Debug, Clone)]
pub struct DspChannel {
    delay: Vec<f32>,
    previous_window_shape: WindowShape,
    imdct: Vec<f32>,
    short_imdct: Vec<f32>,
    long_fft: Vec<Complex32>,
    long_fft_scratch: Vec<Complex32>,
    short_fft: Vec<Complex32>,
    short_fft_scratch: Vec<Complex32>,
}

impl DspChannel {
    pub fn new(frame_len: usize) -> Self {
        let long_fft_len = frame_len / 2;
        let long_fft_scratch_len = fft_scratch_len(long_fft_len);
        let short_fft_len = SHORT_SPECTRUM_LEN / 2;
        let short_fft_scratch_len = fft_scratch_len(short_fft_len);

        Self {
            delay: vec![0.0; frame_len],
            previous_window_shape: WindowShape::Sine,
            imdct: vec![0.0; frame_len * 2],
            short_imdct: vec![0.0; SHORT_WINDOW_LEN],
            long_fft: vec![Complex32::default(); long_fft_len],
            long_fft_scratch: vec![Complex32::default(); long_fft_scratch_len],
            short_fft: vec![Complex32::default(); short_fft_len],
            short_fft_scratch: vec![Complex32::default(); short_fft_scratch_len],
        }
    }

    pub fn synthesize_zero(&mut self, output: &mut [f32]) -> Result<()> {
        if output.len() != self.delay.len() {
            return Err(AacLcError::InvalidConfig(
                "DSP output length does not match AAC frame length",
            ));
        }
        output.fill(0.0);
        self.delay.fill(0.0);
        self.imdct.fill(0.0);
        Ok(())
    }

    pub fn synthesize_zero_with_window(
        &mut self,
        window: &[f32],
        output: &mut [f32],
    ) -> Result<()> {
        if window.len() != self.delay.len() * 2 {
            return Err(AacLcError::InvalidConfig(
                "DSP window length does not match AAC frame length",
            ));
        }
        self.synthesize_zero(output)
    }

    pub fn synthesize_long(
        &mut self,
        coeffs: &[f32],
        transform: &ImdctTransform,
        window: &[f32],
        output: &mut [f32],
    ) -> Result<()> {
        let n = self.delay.len();
        if coeffs.len() != n || output.len() != n || window.len() != 2 * n {
            return Err(AacLcError::InvalidConfig(
                "invalid long-block DSP buffer length",
            ));
        }

        imdct_fast(
            coeffs,
            &mut self.imdct,
            transform,
            &mut self.long_fft,
            &mut self.long_fft_scratch,
        )?;

        for i in 0..n {
            let first = self.imdct[i] * window[i];
            let second = self.imdct[i + n] * window[i + n];
            output[i] = first + self.delay[i];
            self.delay[i] = second;
        }

        Ok(())
    }

    pub fn synthesize_long_sequence(
        &mut self,
        coeffs: &[f32],
        sequence: WindowSequence,
        transform: &ImdctTransform,
        previous_long_window: &[f32],
        long_window: &[f32],
        previous_short_window: &[f32],
        short_window: &[f32],
        output: &mut [f32],
    ) -> Result<()> {
        let n = self.delay.len();
        if coeffs.len() != n
            || output.len() != n
            || previous_long_window.len() != 2 * n
            || long_window.len() != 2 * n
            || previous_short_window.len() != SHORT_WINDOW_LEN
            || short_window.len() != SHORT_WINDOW_LEN
        {
            return Err(AacLcError::InvalidConfig(
                "invalid window-sequence DSP buffer length",
            ));
        }
        if sequence == WindowSequence::EightShort {
            return Err(AacLcError::InvalidConfig(
                "long sequence synthesis cannot handle eight short windows",
            ));
        }

        imdct_fast(
            coeffs,
            &mut self.imdct,
            transform,
            &mut self.long_fft,
            &mut self.long_fft_scratch,
        )?;

        for i in 0..n {
            let first = self.imdct[i]
                * long_sequence_first_window(
                    sequence,
                    previous_long_window,
                    previous_short_window,
                    i,
                );
            let second = self.imdct[i + n]
                * long_sequence_second_window(sequence, long_window, short_window, i);
            output[i] = first + self.delay[i];
            self.delay[i] = second;
        }

        Ok(())
    }

    pub fn synthesize_eight_short(
        &mut self,
        coeffs: &[f32],
        transform: &ImdctTransform,
        previous_short_window: &[f32],
        short_window: &[f32],
        output: &mut [f32],
    ) -> Result<()> {
        let n = self.delay.len();
        if coeffs.len() != n
            || output.len() != n
            || previous_short_window.len() != SHORT_WINDOW_LEN
            || short_window.len() != SHORT_WINDOW_LEN
        {
            return Err(AacLcError::InvalidConfig(
                "invalid short-block DSP buffer length",
            ));
        }

        self.imdct.fill(0.0);
        for window in 0..8 {
            let coeff_start = window * SHORT_SPECTRUM_LEN;
            let out_start = 448 + window * SHORT_SPECTRUM_LEN;
            imdct_fast(
                &coeffs[coeff_start..coeff_start + SHORT_SPECTRUM_LEN],
                &mut self.short_imdct,
                transform,
                &mut self.short_fft,
                &mut self.short_fft_scratch,
            )?;

            if window == 0 {
                for sample in 0..SHORT_SPECTRUM_LEN {
                    self.imdct[out_start + sample] +=
                        self.short_imdct[sample] * previous_short_window[sample];
                }
                for sample in SHORT_SPECTRUM_LEN..SHORT_WINDOW_LEN {
                    self.imdct[out_start + sample] +=
                        self.short_imdct[sample] * short_window[sample];
                }
            } else {
                for sample in 0..SHORT_WINDOW_LEN {
                    self.imdct[out_start + sample] +=
                        self.short_imdct[sample] * short_window[sample];
                }
            }
        }

        for i in 0..n {
            output[i] = self.imdct[i] + self.delay[i];
            self.delay[i] = self.imdct[i + n];
        }

        Ok(())
    }

    pub fn delay(&self) -> &[f32] {
        &self.delay
    }

    pub const fn previous_window_shape(&self) -> WindowShape {
        self.previous_window_shape
    }

    pub fn set_previous_window_shape(&mut self, shape: WindowShape) {
        self.previous_window_shape = shape;
    }
}

fn long_sequence_first_window(
    sequence: WindowSequence,
    previous_long_window: &[f32],
    previous_short_window: &[f32],
    index: usize,
) -> f32 {
    match sequence {
        WindowSequence::OnlyLong | WindowSequence::LongStart => previous_long_window[index],
        WindowSequence::LongStop => match index {
            0..448 => 0.0,
            448..576 => previous_short_window[index - 448],
            _ => 1.0,
        },
        WindowSequence::EightShort => unreachable!("short sequence is handled separately"),
    }
}

fn long_sequence_second_window(
    sequence: WindowSequence,
    long_window: &[f32],
    short_window: &[f32],
    index: usize,
) -> f32 {
    match sequence {
        WindowSequence::OnlyLong | WindowSequence::LongStop => {
            long_window[index + LONG_SPECTRUM_LEN]
        }
        WindowSequence::LongStart => match index {
            0..448 => 1.0,
            448..576 => short_window[SHORT_SPECTRUM_LEN + index - 448],
            _ => 0.0,
        },
        WindowSequence::EightShort => unreachable!("short sequence is handled separately"),
    }
}

pub fn dequantize_signed(quantized: i32, scale_factor: i16) -> f32 {
    dequantize_signed_scaled(
        quantized,
        scalefactor_multiplier(scale_factor),
        pow43_table(),
    )
}

pub(crate) fn dequantize_signed_scaled(quantized: i32, scale: f32, pow43_table: &[f32]) -> f32 {
    if quantized == 0 {
        return 0.0;
    }

    let sign = if quantized < 0 { -1.0 } else { 1.0 };
    let magnitude = pow43_with_table(quantized.unsigned_abs() as usize, pow43_table);
    sign * magnitude * scale
}

pub fn scalefactor_multiplier(scale_factor: i16) -> f32 {
    if (SCALE_FACTOR_TABLE_MIN..=SCALE_FACTOR_TABLE_MAX).contains(&scale_factor) {
        return scale_factor_table()[(scale_factor - SCALE_FACTOR_TABLE_MIN) as usize];
    }

    2.0_f32.powf((scale_factor as f32 - 100.0) * 0.25)
}

fn warm_dequant_tables() {
    let _ = pow43_table();
    let _ = scale_factor_table();
}

pub(crate) fn pow43_table() -> &'static [f32] {
    static TABLE: OnceLock<Box<[f32]>> = OnceLock::new();
    TABLE
        .get_or_init(|| {
            (0..POW43_TABLE_LEN)
                .map(|value| (value as f32).powf(4.0 / 3.0))
                .collect()
        })
        .as_ref()
}

fn pow43_with_table(value: usize, table: &[f32]) -> f32 {
    if value < table.len() {
        return table[value];
    }

    (value as f32).powf(4.0 / 3.0)
}

fn scale_factor_table() -> &'static [f32] {
    static TABLE: OnceLock<Box<[f32]>> = OnceLock::new();
    TABLE
        .get_or_init(|| {
            let mut values = Vec::with_capacity(SCALE_FACTOR_TABLE_LEN);
            for scale_factor in SCALE_FACTOR_TABLE_MIN..=SCALE_FACTOR_TABLE_MAX {
                values.push(2.0_f32.powf((scale_factor as f32 - 100.0) * 0.25));
            }
            values.into_boxed_slice()
        })
        .as_ref()
}

#[cfg(test)]
fn imdct(input: &[f32], output: &mut [f32]) -> Result<()> {
    let n = input.len();
    if n == 0 || !n.is_power_of_two() || output.len() != 2 * n {
        return Err(AacLcError::InvalidConfig("invalid IMDCT buffer length"));
    }

    let nf = n as f32;
    let half_n = nf * 0.5;
    let output_scale = PCM_F32_SCALE / nf;
    for (sample, out) in output.iter_mut().enumerate() {
        let sample_phase = sample as f32 + 0.5 + half_n;
        let mut acc = 0.0f32;
        for (bin, coeff) in input.iter().enumerate() {
            let bin_phase = bin as f32 + 0.5;
            let angle = std::f32::consts::PI / nf * sample_phase * bin_phase;
            acc += *coeff * angle.cos();
        }
        *out = acc * output_scale;
    }

    Ok(())
}

fn imdct_fast(
    input: &[f32],
    output: &mut [f32],
    transform: &ImdctTransform,
    fft: &mut [Complex32],
    fft_scratch: &mut [Complex32],
) -> Result<()> {
    if input.len() != transform.input_len
        || output.len() != transform.output_len
        || fft.len() != transform.fft_len()
        || fft_scratch.len() < transform.fft_scratch_len
    {
        return Err(AacLcError::InvalidConfig("invalid IMDCT buffer length"));
    }

    let n = transform.input_len;
    let half = n / 2;
    let quarter = n / 4;

    for idx in 0..half {
        let even = input[idx * 2];
        let odd = -input[n - 1 - idx * 2];
        let twiddle = transform.twiddle[idx];
        fft[idx] = Complex32::new(
            odd * twiddle.im - even * twiddle.re,
            odd * twiddle.re + even * twiddle.im,
        );
    }

    transform.fft.process_with_scratch(fft, fft_scratch);

    let (out0, rest) = output.split_at_mut(half);
    let (out1, rest) = rest.split_at_mut(half);
    let (out2, out3) = rest.split_at_mut(half);

    for idx in 0..quarter {
        let value = transform.twiddle[idx] * fft[idx].conj();
        let forward = idx * 2;
        let reverse = half - 1 - idx * 2;

        out0[reverse] = -value.im;
        out1[forward] = value.im;
        out2[reverse] = value.re;
        out3[forward] = value.re;
    }

    for idx in quarter..half {
        let value = transform.twiddle[idx] * fft[idx].conj();
        let local = idx - quarter;
        let forward = local * 2;
        let reverse = half - 1 - local * 2;

        out0[forward] = -value.re;
        out1[reverse] = value.re;
        out2[forward] = value.im;
        out3[reverse] = value.im;
    }

    Ok(())
}

fn fft_scratch_len(len: usize) -> usize {
    let mut planner = FftPlanner::<f32>::new();
    planner.plan_fft_inverse(len).get_inplace_scratch_len()
}

fn sine_window(len: usize) -> Vec<f32> {
    let scale = std::f32::consts::PI / len as f32;
    (0..len)
        .map(|idx| ((idx as f32 + 0.5) * scale).sin())
        .collect()
}

fn kbd_window(len: usize, alpha: f32) -> Vec<f32> {
    let half = len / 2;
    let mut kernel = vec![0.0f64; half + 1];
    let denom_arg = std::f64::consts::PI * alpha as f64;

    for (idx, value) in kernel.iter_mut().enumerate() {
        let ratio = 2.0 * idx as f64 / half as f64 - 1.0;
        let arg = denom_arg * (1.0 - ratio * ratio).max(0.0).sqrt();
        *value = bessel_i0_f64(arg);
    }

    let total: f64 = kernel.iter().sum();
    let mut cumulative = 0.0f64;
    let mut window = vec![0.0; len];
    for idx in 0..half {
        cumulative += kernel[idx];
        window[idx] = (cumulative / total).sqrt() as f32;
        window[len - 1 - idx] = window[idx];
    }

    window
}

fn bessel_i0_f64(x: f64) -> f64 {
    let half = x * 0.5;
    let mut sum = 1.0f64;
    let mut term = 1.0f64;

    for k in 1..=64 {
        let ratio = half / k as f64;
        term *= ratio * ratio;
        sum += term;
        if term.abs() < 1.0e-14 * sum {
            break;
        }
    }

    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sine_window_has_princen_bradley_property() {
        let window = sine_window(LONG_WINDOW_LEN);

        for idx in 0..LONG_SPECTRUM_LEN {
            let sum = window[idx] * window[idx]
                + window[idx + LONG_SPECTRUM_LEN] * window[idx + LONG_SPECTRUM_LEN];
            assert!((sum - 1.0).abs() < 2.0e-6);
        }
    }

    #[test]
    fn kbd_window_is_symmetric_and_complementary() {
        let window = kbd_window(LONG_WINDOW_LEN, 4.0);

        for idx in 0..LONG_SPECTRUM_LEN {
            assert!((window[idx] - window[LONG_WINDOW_LEN - 1 - idx]).abs() < 1.0e-6);
            let sum = window[idx] * window[idx]
                + window[idx + LONG_SPECTRUM_LEN] * window[idx + LONG_SPECTRUM_LEN];
            assert!((sum - 1.0).abs() < 2.0e-6);
        }
    }

    #[test]
    fn imdct_zero_input_produces_zero_output() {
        let input = [0.0f32; 8];
        let mut output = [1.0f32; 16];

        imdct(&input, &mut output).unwrap();

        assert!(output.iter().all(|sample| *sample == 0.0));
    }

    #[test]
    fn imdct_fast_matches_reference_path_for_small_block() {
        let input = [0.0f32, 1.0, -2.0, 0.5, 3.0, -4.0, 0.25, -0.75];
        let transform = ImdctTransform::new(input.len());
        let mut reference = [0.0f32; 16];
        let mut fast_output = [0.0f32; 16];
        let mut fft = vec![Complex32::default(); transform.fft_len()];
        let mut fft_scratch = vec![Complex32::default(); transform.fft_scratch_len()];

        imdct(&input, &mut reference).unwrap();
        imdct_fast(
            &input,
            &mut fast_output,
            &transform,
            &mut fft,
            &mut fft_scratch,
        )
        .unwrap();

        for (expected, actual) in reference.iter().zip(fast_output.iter()) {
            assert!(
                (*expected - *actual).abs() < 1.0e-10,
                "expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn imdct_fast_matches_reference_path_for_aac_blocks() {
        for len in [SHORT_SPECTRUM_LEN, LONG_SPECTRUM_LEN] {
            let input = (0..len)
                .map(|idx| match idx % 9 {
                    0 => 0.0,
                    1 => 1.0,
                    2 => -2.0,
                    3 => 0.5,
                    4 => -0.25,
                    5 => 4.0,
                    6 => -8.0,
                    7 => 0.125,
                    _ => -0.75,
                })
                .collect::<Vec<_>>();
            let transform = ImdctTransform::new(len);
            let mut reference = vec![0.0f32; len * 2];
            let mut fast_output = vec![0.0f32; len * 2];
            let mut fft = vec![Complex32::default(); transform.fft_len()];
            let mut fft_scratch = vec![Complex32::default(); transform.fft_scratch_len()];

            imdct(&input, &mut reference).unwrap();
            imdct_fast(
                &input,
                &mut fast_output,
                &transform,
                &mut fft,
                &mut fft_scratch,
            )
            .unwrap();

            for (expected, actual) in reference.iter().zip(fast_output.iter()) {
                assert!(
                    (*expected - *actual).abs() < 2.0e-8,
                    "len {len}: expected {expected}, got {actual}"
                );
            }
        }
    }

    #[test]
    fn imdct_fast_matches_reference_path_for_seeded_spectra() {
        for len in [8usize, 16, SHORT_SPECTRUM_LEN, LONG_SPECTRUM_LEN] {
            for seed in [0x1234_5678u32, 0xa5a5_0101, 0xdead_beef] {
                let input = seeded_spectrum(len, seed);
                let transform = ImdctTransform::new(len);
                let mut reference = vec![0.0f32; len * 2];
                let mut fast_output = vec![0.0f32; len * 2];
                let mut fft = vec![Complex32::default(); transform.fft_len()];
                let mut fft_scratch = vec![Complex32::default(); transform.fft_scratch_len()];

                imdct(&input, &mut reference).unwrap();
                imdct_fast(
                    &input,
                    &mut fast_output,
                    &transform,
                    &mut fft,
                    &mut fft_scratch,
                )
                .unwrap();

                for (expected, actual) in reference.iter().zip(fast_output.iter()) {
                    assert!(
                        (*expected - *actual).abs() < 4.0e-8,
                        "len {len}, seed {seed:#x}: expected {expected}, got {actual}"
                    );
                }
            }
        }
    }

    fn seeded_spectrum(len: usize, seed: u32) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|idx| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                if idx % 7 == 0 {
                    0.0
                } else {
                    let centered = ((state >> 8) & 0xffff) as f32 / 32768.0 - 1.0;
                    centered * 12.0
                }
            })
            .collect()
    }

    #[test]
    fn long_synthesis_zero_clears_output_and_delay() {
        let mut channel = DspChannel::new(LONG_SPECTRUM_LEN);
        let mut output = vec![1.0; LONG_SPECTRUM_LEN];

        channel.synthesize_zero(&mut output).unwrap();

        assert!(output.iter().all(|sample| *sample == 0.0));
        assert!(channel.delay().iter().all(|sample| *sample == 0.0));
    }

    #[test]
    fn long_synthesis_zero_coefficients_emit_zero() {
        let dsp = AacDsp::new();
        let mut channel = DspChannel::new(LONG_SPECTRUM_LEN);
        let coeffs = vec![0.0; LONG_SPECTRUM_LEN];
        let mut output = vec![1.0; LONG_SPECTRUM_LEN];

        channel
            .synthesize_long(
                &coeffs,
                dsp.long_imdct_transform(),
                dsp.long_window(WindowShape::Sine),
                &mut output,
            )
            .unwrap();

        assert!(output.iter().all(|sample| sample.abs() < 1.0e-6));
        assert!(channel.delay().iter().all(|sample| sample.abs() < 1.0e-6));
    }

    #[test]
    fn sequence_windows_use_previous_shape_on_left_overlap() {
        let dsp = AacDsp::new();
        let prev_long = dsp.long_window(WindowShape::Sine);
        let curr_long = dsp.long_window(WindowShape::KaiserBesselDerived);
        let prev_short = dsp.short_window(WindowShape::Sine);
        let curr_short = dsp.short_window(WindowShape::KaiserBesselDerived);

        assert_eq!(
            long_sequence_first_window(WindowSequence::OnlyLong, prev_long, prev_short, 100),
            prev_long[100]
        );
        assert_eq!(
            long_sequence_second_window(WindowSequence::OnlyLong, curr_long, curr_short, 100),
            curr_long[LONG_SPECTRUM_LEN + 100]
        );
        assert_eq!(
            long_sequence_first_window(WindowSequence::LongStop, prev_long, prev_short, 448),
            prev_short[0]
        );
        assert_eq!(
            long_sequence_second_window(WindowSequence::LongStart, curr_long, curr_short, 448),
            curr_short[SHORT_SPECTRUM_LEN]
        );
    }

    #[test]
    fn channel_tracks_previous_window_shape() {
        let mut channel = DspChannel::new(LONG_SPECTRUM_LEN);
        assert_eq!(channel.previous_window_shape(), WindowShape::Sine);

        channel.set_previous_window_shape(WindowShape::KaiserBesselDerived);
        assert_eq!(
            channel.previous_window_shape(),
            WindowShape::KaiserBesselDerived
        );
    }

    #[test]
    fn dequantizes_signed_values() {
        assert_eq!(dequantize_signed(0, 100), 0.0);
        assert!((dequantize_signed(1, 100) - 1.0).abs() < 1.0e-6);
        assert!((dequantize_signed(-1, 100) + 1.0).abs() < 1.0e-6);
        assert!((dequantize_signed(8, 100) - 16.0).abs() < 1.0e-5);
    }

    #[test]
    fn scalefactor_multiplier_tracks_quarter_octaves() {
        assert!((scalefactor_multiplier(100) - 1.0).abs() < 1.0e-6);
        assert!((scalefactor_multiplier(104) - 2.0).abs() < 1.0e-6);
        assert!((scalefactor_multiplier(96) - 0.5).abs() < 1.0e-6);
    }
}
