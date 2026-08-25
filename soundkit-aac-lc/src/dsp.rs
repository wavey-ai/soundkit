use crate::error::{AacLcError, Result};
use crate::fft::{Complex, ForwardFft};
use crate::ics::{WindowSequence, WindowShape};
use std::{fmt, sync::OnceLock};

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
    twiddle: Vec<Complex>,
    fft: ForwardFft,
}

impl fmt::Debug for ImdctTransform {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ImdctTransform")
            .field("input_len", &self.input_len)
            .field("output_len", &self.output_len)
            .field("output_scale", &self.output_scale)
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
                complex_cis(std::f32::consts::PI / nf * (bin as f32 + 0.125)).scale(twiddle_scale)
            })
            .collect();

        Self {
            input_len,
            output_len,
            output_scale,
            twiddle,
            fft: ForwardFft::new(fft_len),
        }
    }

    pub const fn input_len(&self) -> usize {
        self.input_len
    }

    pub const fn output_len(&self) -> usize {
        self.output_len
    }

    pub const fn fft_len(&self) -> usize {
        self.input_len / 2
    }
}

fn complex_cis(angle: f32) -> Complex {
    Complex::new(angle.cos(), angle.sin())
}

#[derive(Debug, Clone)]
pub struct DspChannel {
    delay: Vec<f32>,
    previous_window_shape: WindowShape,
    imdct: Vec<f32>,
    short_imdct: Vec<f32>,
    long_fft: Vec<Complex>,
    long_fft_scratch: Vec<Complex>,
    short_fft: Vec<Complex>,
    short_fft_scratch: Vec<Complex>,
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
            long_fft: vec![Complex::default(); long_fft_len],
            long_fft_scratch: vec![Complex::default(); long_fft_scratch_len],
            short_fft: vec![Complex::default(); short_fft_len],
            short_fft_scratch: vec![Complex::default(); short_fft_scratch_len],
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

        let (first, second) = self.imdct.split_at(n);
        overlap_windowed(
            first,
            second,
            &window[..n],
            &window[n..],
            &mut self.delay,
            output,
        );

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

        let (first, second) = self.imdct.split_at(n);
        match sequence {
            WindowSequence::OnlyLong => overlap_windowed(
                first,
                second,
                &previous_long_window[..n],
                &long_window[n..],
                &mut self.delay,
                output,
            ),
            WindowSequence::LongStart => {
                const FLAT: usize = (LONG_SPECTRUM_LEN - SHORT_SPECTRUM_LEN) / 2;
                const SHORT_END: usize = FLAT + SHORT_SPECTRUM_LEN;

                overlap_first_window_second_copy(
                    &first[..FLAT],
                    &second[..FLAT],
                    &previous_long_window[..FLAT],
                    &mut self.delay[..FLAT],
                    &mut output[..FLAT],
                );
                overlap_windowed(
                    &first[FLAT..SHORT_END],
                    &second[FLAT..SHORT_END],
                    &previous_long_window[FLAT..SHORT_END],
                    &short_window[SHORT_SPECTRUM_LEN..SHORT_WINDOW_LEN],
                    &mut self.delay[FLAT..SHORT_END],
                    &mut output[FLAT..SHORT_END],
                );
                overlap_first_window_second_zero(
                    &first[SHORT_END..],
                    &second[SHORT_END..],
                    &previous_long_window[SHORT_END..n],
                    &mut self.delay[SHORT_END..],
                    &mut output[SHORT_END..],
                );
            }
            WindowSequence::LongStop => {
                const FLAT: usize = (LONG_SPECTRUM_LEN - SHORT_SPECTRUM_LEN) / 2;
                const SHORT_END: usize = FLAT + SHORT_SPECTRUM_LEN;

                overlap_first_zero_second_windowed(
                    &first[..FLAT],
                    &second[..FLAT],
                    &long_window[n..n + FLAT],
                    &mut self.delay[..FLAT],
                    &mut output[..FLAT],
                );
                overlap_windowed(
                    &first[FLAT..SHORT_END],
                    &second[FLAT..SHORT_END],
                    &previous_short_window[..SHORT_SPECTRUM_LEN],
                    &long_window[n + FLAT..n + SHORT_END],
                    &mut self.delay[FLAT..SHORT_END],
                    &mut output[FLAT..SHORT_END],
                );
                overlap_first_copy_second_windowed(
                    &first[SHORT_END..],
                    &second[SHORT_END..],
                    &long_window[n + SHORT_END..],
                    &mut self.delay[SHORT_END..],
                    &mut output[SHORT_END..],
                );
            }
            WindowSequence::EightShort => unreachable!("short sequence is handled separately"),
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

#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
fn overlap_windowed(
    first: &[f32],
    second: &[f32],
    first_window: &[f32],
    second_window: &[f32],
    delay: &mut [f32],
    output: &mut [f32],
) {
    debug_assert_eq!(first.len(), second.len());
    debug_assert_eq!(first.len(), first_window.len());
    debug_assert_eq!(first.len(), second_window.len());
    debug_assert_eq!(first.len(), delay.len());
    debug_assert_eq!(first.len(), output.len());

    for index in 0..first.len() {
        output[index] = first[index] * first_window[index] + delay[index];
        delay[index] = second[index] * second_window[index];
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn overlap_windowed(
    first: &[f32],
    second: &[f32],
    first_window: &[f32],
    second_window: &[f32],
    delay: &mut [f32],
    output: &mut [f32],
) {
    use core::arch::wasm32::{f32x4_add, f32x4_mul, v128, v128_load, v128_store};

    debug_assert_eq!(first.len(), second.len());
    debug_assert_eq!(first.len(), first_window.len());
    debug_assert_eq!(first.len(), second_window.len());
    debug_assert_eq!(first.len(), delay.len());
    debug_assert_eq!(first.len(), output.len());

    let mut index = 0;
    while index + 4 <= first.len() {
        unsafe {
            let first_value = f32x4_mul(
                v128_load(first.as_ptr().add(index).cast::<v128>()),
                v128_load(first_window.as_ptr().add(index).cast::<v128>()),
            );
            let second_value = f32x4_mul(
                v128_load(second.as_ptr().add(index).cast::<v128>()),
                v128_load(second_window.as_ptr().add(index).cast::<v128>()),
            );
            let delayed = v128_load(delay.as_ptr().add(index).cast::<v128>());
            v128_store(
                output.as_mut_ptr().add(index).cast::<v128>(),
                f32x4_add(first_value, delayed),
            );
            v128_store(delay.as_mut_ptr().add(index).cast::<v128>(), second_value);
        }
        index += 4;
    }
    while index < first.len() {
        output[index] = first[index] * first_window[index] + delay[index];
        delay[index] = second[index] * second_window[index];
        index += 1;
    }
}

fn overlap_first_window_second_copy(
    first: &[f32],
    second: &[f32],
    first_window: &[f32],
    delay: &mut [f32],
    output: &mut [f32],
) {
    debug_assert_eq!(first.len(), second.len());
    debug_assert_eq!(first.len(), first_window.len());
    debug_assert_eq!(first.len(), delay.len());
    debug_assert_eq!(first.len(), output.len());

    for index in 0..first.len() {
        output[index] = first[index] * first_window[index] + delay[index];
        delay[index] = second[index];
    }
}

fn overlap_first_window_second_zero(
    first: &[f32],
    second: &[f32],
    first_window: &[f32],
    delay: &mut [f32],
    output: &mut [f32],
) {
    debug_assert_eq!(first.len(), second.len());
    debug_assert_eq!(first.len(), first_window.len());
    debug_assert_eq!(first.len(), delay.len());
    debug_assert_eq!(first.len(), output.len());

    for index in 0..first.len() {
        output[index] = first[index] * first_window[index] + delay[index];
        delay[index] = second[index] * 0.0;
    }
}

fn overlap_first_zero_second_windowed(
    first: &[f32],
    second: &[f32],
    second_window: &[f32],
    delay: &mut [f32],
    output: &mut [f32],
) {
    debug_assert_eq!(first.len(), second.len());
    debug_assert_eq!(first.len(), second_window.len());
    debug_assert_eq!(first.len(), delay.len());
    debug_assert_eq!(first.len(), output.len());

    for index in 0..first.len() {
        output[index] = first[index] * 0.0 + delay[index];
        delay[index] = second[index] * second_window[index];
    }
}

fn overlap_first_copy_second_windowed(
    first: &[f32],
    second: &[f32],
    second_window: &[f32],
    delay: &mut [f32],
    output: &mut [f32],
) {
    debug_assert_eq!(first.len(), second.len());
    debug_assert_eq!(first.len(), second_window.len());
    debug_assert_eq!(first.len(), delay.len());
    debug_assert_eq!(first.len(), output.len());

    for index in 0..first.len() {
        output[index] = first[index] + delay[index];
        delay[index] = second[index] * second_window[index];
    }
}

#[cfg(test)]
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

#[cfg(test)]
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
    fft: &mut [Complex],
    fft_scratch: &mut [Complex],
) -> Result<()> {
    if input.len() != transform.input_len
        || output.len() != transform.output_len
        || fft.len() != transform.fft_len()
        || fft_scratch.len() < transform.fft.scratch_len()
    {
        return Err(AacLcError::InvalidConfig("invalid IMDCT buffer length"));
    }

    let n = transform.input_len;
    let half = n / 2;
    let quarter = n / 4;

    prepare_imdct_fft(input, &transform.twiddle, fft);

    transform.fft.process_inplace_with_scratch(fft, fft_scratch);

    let (out0, rest) = output.split_at_mut(half);
    let (out1, rest) = rest.split_at_mut(half);
    let (out2, out3) = rest.split_at_mut(half);

    // Pair the transform's low and mirrored high bins. This writes adjacent
    // output samples instead of four separate stride-two streams, matching
    // the symmetry used by native C IMDCT implementations.
    for idx in 0..quarter {
        let mirror = half - 1 - idx;
        let low = unsafe { *transform.twiddle.get_unchecked(idx) * fft.get_unchecked(idx).conj() };
        let high =
            unsafe { *transform.twiddle.get_unchecked(mirror) * fft.get_unchecked(mirror).conj() };
        let forward = idx * 2;
        let reverse = half - 2 - forward;

        unsafe {
            *out0.get_unchecked_mut(reverse) = -high.re;
            *out0.get_unchecked_mut(reverse + 1) = -low.im;
            *out1.get_unchecked_mut(forward) = low.im;
            *out1.get_unchecked_mut(forward + 1) = high.re;
            *out2.get_unchecked_mut(reverse) = high.im;
            *out2.get_unchecked_mut(reverse + 1) = low.re;
            *out3.get_unchecked_mut(forward) = low.re;
            *out3.get_unchecked_mut(forward + 1) = high.im;
        }
    }

    Ok(())
}

fn fft_scratch_len(len: usize) -> usize {
    ForwardFft::new(len).scratch_len()
}

#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
fn prepare_imdct_fft(input: &[f32], twiddles: &[Complex], fft: &mut [Complex]) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        unsafe { prepare_imdct_fft_avx2(input, twiddles, fft) };
        return;
    }

    let n = input.len();
    for idx in 0..fft.len() {
        let even = input[idx * 2];
        let odd = -input[n - 1 - idx * 2];
        let twiddle = twiddles[idx];
        fft[idx] = Complex::new(
            odd * twiddle.im - even * twiddle.re,
            odd * twiddle.re + even * twiddle.im,
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn prepare_imdct_fft_avx2(input: &[f32], twiddles: &[Complex], fft: &mut [Complex]) {
    use core::arch::x86_64::{
        __m256, _mm256_addsub_ps, _mm256_castps128_ps256, _mm256_insertf128_ps, _mm256_loadu_ps,
        _mm256_mul_ps, _mm256_permute_ps, _mm256_storeu_ps, _mm_loadu_ps, _mm_shuffle_ps,
        _mm_xor_ps,
    };

    debug_assert_eq!(twiddles.len(), fft.len());
    debug_assert_eq!(input.len(), fft.len() * 2);

    let n = input.len();
    let sign = _mm_loadu_ps([-0.0f32; 4].as_ptr());
    let mut idx = 0usize;
    while idx + 4 <= fft.len() {
        let forward0 = _mm_loadu_ps(input.as_ptr().add(idx * 2));
        let forward1 = _mm_loadu_ps(input.as_ptr().add(idx * 2 + 4));
        let even = _mm_shuffle_ps::<0x88>(forward0, forward1);

        let reverse = input.as_ptr().add(n - idx * 2 - 8);
        let reverse0 = _mm_loadu_ps(reverse);
        let reverse1 = _mm_loadu_ps(reverse.add(4));
        let odd = _mm_xor_ps(_mm_shuffle_ps::<0x77>(reverse1, reverse0), sign);

        let even_lo = core::arch::x86_64::_mm_unpacklo_ps(even, even);
        let even_hi = core::arch::x86_64::_mm_unpackhi_ps(even, even);
        let even_pairs: __m256 =
            _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(even_lo), even_hi);
        let odd_lo = core::arch::x86_64::_mm_unpacklo_ps(odd, odd);
        let odd_hi = core::arch::x86_64::_mm_unpackhi_ps(odd, odd);
        let odd_pairs: __m256 = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(odd_lo), odd_hi);

        let twiddle = _mm256_loadu_ps(twiddles.as_ptr().add(idx).cast::<f32>());
        let swapped_twiddle = _mm256_permute_ps::<0xb1>(twiddle);
        let value = _mm256_addsub_ps(
            _mm256_mul_ps(odd_pairs, swapped_twiddle),
            _mm256_mul_ps(even_pairs, twiddle),
        );
        _mm256_storeu_ps(fft.as_mut_ptr().add(idx).cast::<f32>(), value);
        idx += 4;
    }

    while idx < fft.len() {
        let even = input[idx * 2];
        let odd = -input[n - 1 - idx * 2];
        let twiddle = twiddles[idx];
        fft[idx] = Complex::new(
            odd * twiddle.im - even * twiddle.re,
            odd * twiddle.re + even * twiddle.im,
        );
        idx += 1;
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn prepare_imdct_fft(input: &[f32], twiddles: &[Complex], fft: &mut [Complex]) {
    use core::arch::wasm32::{
        f32x4_add, f32x4_mul, i32x4, i32x4_shuffle, v128, v128_load, v128_store, v128_xor,
    };

    debug_assert_eq!(twiddles.len(), fft.len());
    debug_assert_eq!(input.len(), fft.len() * 2);
    debug_assert_eq!(
        std::mem::size_of::<Complex>(),
        2 * std::mem::size_of::<f32>()
    );

    let n = input.len();
    let mut idx = 0;
    while idx + 2 <= fft.len() {
        unsafe {
            let even_source = v128_load(input.as_ptr().add(idx * 2).cast::<v128>());
            let odd_source = v128_load(input.as_ptr().add(n - 4 - idx * 2).cast::<v128>());
            let twiddle = v128_load(twiddles.as_ptr().add(idx).cast::<v128>());

            let even = i32x4_shuffle::<0, 0, 2, 2>(even_source, even_source);
            let odd = v128_xor(
                i32x4_shuffle::<3, 3, 1, 1>(odd_source, odd_source),
                i32x4(i32::MIN, i32::MIN, i32::MIN, i32::MIN),
            );
            let swapped_twiddle = i32x4_shuffle::<1, 0, 3, 2>(twiddle, twiddle);
            let signed_even_product =
                v128_xor(f32x4_mul(even, twiddle), i32x4(i32::MIN, 0, i32::MIN, 0));
            let value = f32x4_add(f32x4_mul(odd, swapped_twiddle), signed_even_product);
            v128_store(fft.as_mut_ptr().add(idx).cast::<v128>(), value);
        }
        idx += 2;
    }

    while idx < fft.len() {
        let even = input[idx * 2];
        let odd = -input[n - 1 - idx * 2];
        let twiddle = twiddles[idx];
        fft[idx] = Complex::new(
            odd * twiddle.im - even * twiddle.re,
            odd * twiddle.re + even * twiddle.im,
        );
        idx += 1;
    }
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
        let mut fft = vec![Complex::default(); transform.fft_len()];
        let mut fft_scratch = vec![Complex::default(); transform.fft.scratch_len()];

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
            let mut fft = vec![Complex::default(); transform.fft_len()];
            let mut fft_scratch = vec![Complex::default(); transform.fft.scratch_len()];

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
                let mut fft = vec![Complex::default(); transform.fft_len()];
                let mut fft_scratch = vec![Complex::default(); transform.fft.scratch_len()];

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
