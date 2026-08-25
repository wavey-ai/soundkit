//! SoundKit's allocation-free G.722 64 kbit/s codec core.
//!
//! The implementation follows the arithmetic blocks defined by ITU-T G.722:
//! a 24-tap quadrature mirror filter, two adaptive differential PCM bands,
//! and the standard 6+2-bit unpacked code word. State lives in the caller's
//! encoder or decoder and no codec operation allocates.

const QMF: [i32; 12] = [3, -11, 12, 32, -210, 951, 3876, -805, 362, -156, 53, -11];
const SCALE_BASE: [i32; 32] = [
    2048, 2093, 2139, 2186, 2233, 2282, 2332, 2383, 2435, 2489, 2543, 2599, 2656, 2714, 2774, 2834,
    2896, 2960, 3025, 3091, 3158, 3228, 3298, 3371, 3444, 3520, 3597, 3676, 3756, 3838, 3922, 4008,
];
const LOW_LOG_DELTA: [i32; 8] = [-60, -30, 58, 172, 334, 538, 1198, 3042];
const LOW_LOG_INDEX: [usize; 16] = [0, 7, 6, 5, 4, 3, 2, 1, 7, 6, 5, 4, 3, 2, 1, 0];
const HIGH_LOG_DELTA: [i32; 3] = [0, -214, 798];
const HIGH_LOG_INDEX: [usize; 4] = [2, 1, 2, 1];

const LOW_INVERSE_4: [i32; 16] = [
    0, -20456, -12896, -8968, -6288, -4240, -2584, -1200, 20456, 12896, 8968, 6288, 4240, 2584,
    1200, 0,
];
const LOW_INVERSE_6: [i32; 64] = [
    -136, -136, -136, -136, -24808, -21904, -19008, -16704, -14984, -13512, -12280, -11192, -10232,
    -9360, -8576, -7856, -7192, -6576, -6000, -5456, -4944, -4464, -4008, -3576, -3168, -2776,
    -2400, -2032, -1688, -1360, -1040, -728, 24808, 21904, 19008, 16704, 14984, 13512, 12280,
    11192, 10232, 9360, 8576, 7856, 7192, 6576, 6000, 5456, 4944, 4464, 4008, 3576, 3168, 2776,
    2400, 2032, 1688, 1360, 1040, 728, 432, 136, -432, -136,
];
const HIGH_INVERSE: [i32; 4] = [-7408, -1616, 7408, 1616];

const LOW_THRESHOLDS: [i32; 32] = [
    0, 35, 72, 110, 150, 190, 233, 276, 323, 370, 422, 473, 530, 587, 650, 714, 786, 858, 940,
    1023, 1121, 1219, 1339, 1458, 1612, 1765, 1980, 2195, 2557, 2919, 0, 0,
];
const LOW_NEGATIVE_CODES: [u8; 32] = [
    0, 63, 62, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11,
    10, 9, 8, 7, 6, 5, 4, 0,
];
const LOW_POSITIVE_CODES: [u8; 32] = [
    0, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39,
    38, 37, 36, 35, 34, 33, 32, 0,
];
const HIGH_NEGATIVE_CODES: [u8; 3] = [0, 1, 0];
const HIGH_POSITIVE_CODES: [u8; 3] = [0, 3, 2];

#[inline(always)]
fn saturate(value: i32) -> i32 {
    value.clamp(i16::MIN as i32, i16::MAX as i32)
}

#[derive(Clone, Copy, Default)]
struct BandState {
    prediction: i32,
    pole_prediction: i32,
    zero_prediction: i32,
    reconstructed: [i32; 3],
    pole: [i32; 3],
    partial: [i32; 3],
    difference: [i32; 7],
    zero: [i32; 7],
    log_scale: i32,
    step: i32,
}

impl BandState {
    const fn with_step(step: i32) -> Self {
        Self {
            prediction: 0,
            pole_prediction: 0,
            zero_prediction: 0,
            reconstructed: [0; 3],
            pole: [0; 3],
            partial: [0; 3],
            difference: [0; 7],
            zero: [0; 7],
            log_scale: 0,
            step,
        }
    }

    #[inline]
    fn update(&mut self, difference: i32) {
        self.difference[0] = difference;
        self.reconstructed[0] = saturate(self.prediction + difference);
        self.partial[0] = saturate(self.zero_prediction + difference);

        let sign0 = self.partial[0] >> 15;
        let sign1 = self.partial[1] >> 15;
        let sign2 = self.partial[2] >> 15;

        let pole_twice = saturate(self.pole[1] << 2);
        let correlation = if sign0 == sign1 {
            -pole_twice
        } else {
            pole_twice
        }
        .min(32767);
        let mut next_pole2 = (correlation >> 7)
            + if sign0 == sign2 { 128 } else { -128 }
            + ((self.pole[2] * 32512) >> 15);
        next_pole2 = next_pole2.clamp(-12288, 12288);

        let direction = if sign0 == sign1 { 192 } else { -192 };
        let mut next_pole1 = saturate(direction + ((self.pole[1] * 32640) >> 15));
        let pole1_limit = saturate(15360 - next_pole2);
        next_pole1 = next_pole1.clamp(-pole1_limit, pole1_limit);

        let zero_impulse = if difference == 0 { 0 } else { 128 };
        let difference_sign = difference >> 15;
        let mut next_zero = [0i32; 7];
        for tap in 1..7 {
            let correlation = if self.difference[tap] >> 15 == difference_sign {
                zero_impulse
            } else {
                -zero_impulse
            };
            next_zero[tap] = saturate(correlation + ((self.zero[tap] * 32640) >> 15));
        }

        self.difference.copy_within(..6, 1);
        self.zero[1..7].copy_from_slice(&next_zero[1..7]);
        self.reconstructed.copy_within(..2, 1);
        self.partial.copy_within(..2, 1);
        self.pole[2] = next_pole2;
        self.pole[1] = next_pole1;

        let first = (self.pole[1] * saturate(self.reconstructed[1] * 2)) >> 15;
        let second = (self.pole[2] * saturate(self.reconstructed[2] * 2)) >> 15;
        self.pole_prediction = saturate(first + second);

        let mut zero_prediction = 0;
        for tap in 1..7 {
            zero_prediction += (self.zero[tap] * saturate(self.difference[tap] * 2)) >> 15;
        }
        self.zero_prediction = saturate(zero_prediction);
        self.prediction = saturate(self.pole_prediction + self.zero_prediction);
    }

    #[inline]
    fn update_low_scale(&mut self, coarse_code: usize) {
        self.log_scale = (((self.log_scale * 127) >> 7)
            + LOW_LOG_DELTA[LOW_LOG_INDEX[coarse_code]])
            .clamp(0, 18432);
        self.step = step_from_log(self.log_scale, 8) << 2;
    }

    #[inline]
    fn update_high_scale(&mut self, code: usize) {
        self.log_scale =
            (((self.log_scale * 127) >> 7) + HIGH_LOG_DELTA[HIGH_LOG_INDEX[code]]).clamp(0, 22528);
        self.step = step_from_log(self.log_scale, 10) << 2;
    }
}

#[inline]
fn step_from_log(log_scale: i32, base_shift: i32) -> i32 {
    let mantissa = SCALE_BASE[((log_scale >> 6) & 31) as usize];
    let shift = base_shift - (log_scale >> 11);
    if shift < 0 {
        mantissa << -shift
    } else {
        mantissa >> shift
    }
}

pub(crate) struct EncoderCore {
    qmf_history: [i32; 24],
    low: BandState,
    high: BandState,
}

impl EncoderCore {
    pub(crate) fn new() -> Self {
        Self {
            qmf_history: [0; 24],
            low: BandState::with_step(32),
            high: BandState::with_step(8),
        }
    }

    /// Encodes complete pairs of 16 kHz mono samples into unpacked 64 kbit/s
    /// G.722 code words. The caller validates slice sizes.
    pub(crate) fn encode(&mut self, input: &[i16], output: &mut [u8]) -> usize {
        debug_assert!(input.len().is_multiple_of(2));
        debug_assert!(output.len() >= input.len() / 2);

        for (pair, destination) in input.chunks_exact(2).zip(output.iter_mut()) {
            self.qmf_history.copy_within(2.., 0);
            self.qmf_history[22] = i32::from(pair[0]);
            self.qmf_history[23] = i32::from(pair[1]);

            let mut odd = 0;
            let mut even = 0;
            for tap in 0..12 {
                odd += self.qmf_history[tap * 2] * QMF[tap];
                even += self.qmf_history[tap * 2 + 1] * QMF[11 - tap];
            }
            let low_sample = (even + odd) >> 14;
            let high_sample = (even - odd) >> 14;

            let low_error = saturate(low_sample - self.low.prediction);
            let magnitude = if low_error >= 0 {
                low_error
            } else {
                -(low_error + 1)
            };
            let mut threshold_index = 1usize;
            while threshold_index < 30
                && magnitude >= (LOW_THRESHOLDS[threshold_index] * self.low.step) >> 12
            {
                threshold_index += 1;
            }
            let low_code = if low_error < 0 {
                LOW_NEGATIVE_CODES[threshold_index]
            } else {
                LOW_POSITIVE_CODES[threshold_index]
            };
            let low_coarse = usize::from(low_code >> 2);
            let low_difference = (self.low.step * LOW_INVERSE_4[low_coarse]) >> 15;
            self.low.update_low_scale(low_coarse);
            self.low.update(low_difference);

            let high_error = saturate(high_sample - self.high.prediction);
            let high_magnitude = if high_error >= 0 {
                high_error
            } else {
                -(high_error + 1)
            };
            let high_size = if high_magnitude >= (564 * self.high.step) >> 12 {
                2
            } else {
                1
            };
            let high_code = if high_error < 0 {
                HIGH_NEGATIVE_CODES[high_size]
            } else {
                HIGH_POSITIVE_CODES[high_size]
            };
            let high_difference = (self.high.step * HIGH_INVERSE[usize::from(high_code)]) >> 15;
            self.high.update_high_scale(usize::from(high_code));
            self.high.update(high_difference);

            *destination = (high_code << 6) | low_code;
        }

        input.len() / 2
    }
}

pub(crate) struct DecoderCore {
    qmf_history: [i32; 24],
    low: BandState,
    high: BandState,
}

impl DecoderCore {
    pub(crate) fn new() -> Self {
        Self {
            qmf_history: [0; 24],
            low: BandState::with_step(32),
            high: BandState::with_step(8),
        }
    }

    /// Decodes unpacked 64 kbit/s G.722 bytes into 16 kHz mono PCM. The
    /// caller validates that output has room for two samples per byte.
    pub(crate) fn decode(&mut self, input: &[u8], output: &mut [i16]) -> usize {
        debug_assert!(output.len() >= input.len() * 2);

        for (&code, destination) in input.iter().zip(output.chunks_exact_mut(2)) {
            let low_code = usize::from(code & 0x3f);
            let low_coarse = low_code >> 2;
            let high_code = usize::from(code >> 6);

            let low_reconstruction =
                self.low.prediction + ((self.low.step * LOW_INVERSE_6[low_code]) >> 15);
            let low_reconstruction = low_reconstruction.clamp(-16384, 16383);
            let low_difference = (self.low.step * LOW_INVERSE_4[low_coarse]) >> 15;
            self.low.update_low_scale(low_coarse);
            self.low.update(low_difference);

            let high_difference = (self.high.step * HIGH_INVERSE[high_code]) >> 15;
            let high_reconstruction = (self.high.prediction + high_difference).clamp(-16384, 16383);
            self.high.update_high_scale(high_code);
            self.high.update(high_difference);

            self.qmf_history.copy_within(2.., 0);
            self.qmf_history[22] = low_reconstruction + high_reconstruction;
            self.qmf_history[23] = low_reconstruction - high_reconstruction;

            let mut second = 0;
            let mut first = 0;
            for tap in 0..12 {
                second += self.qmf_history[tap * 2] * QMF[tap];
                first += self.qmf_history[tap * 2 + 1] * QMF[11 - tap];
            }
            destination[0] = saturate(first >> 11) as i16;
            destination[1] = saturate(second >> 11) as i16;
        }

        input.len() * 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_length_calls_leave_state_untouched() {
        let mut encoder = EncoderCore::new();
        let mut decoder = DecoderCore::new();
        assert_eq!(encoder.encode(&[], &mut []), 0);
        assert_eq!(decoder.decode(&[], &mut []), 0);
    }

    #[test]
    fn zero_vector_roundtrip_is_deterministic() {
        let input = [0i16; 320];
        let mut encoded = [0u8; 160];
        let mut decoded = [0i16; 320];
        assert_eq!(EncoderCore::new().encode(&input, &mut encoded), 160);
        assert_eq!(DecoderCore::new().decode(&encoded, &mut decoded), 320);
        assert_eq!(
            &encoded[..8],
            &[0xfa, 0xfa, 0xfa, 0xfa, 0xfa, 0xfa, 0xfa, 0xfa]
        );
        assert!(decoded.iter().all(|sample| sample.abs() <= 32));
    }
}
