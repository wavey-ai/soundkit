use crate::analysis::{AnalysisInfo, TonalityAnalysisState};
use crate::celt::bands::{
    compute_band_energies, hysteresis_decision, normalise_bands, spreading_decision, SPREAD_NORMAL,
};
use crate::celt::codec::{encode_spectral_frame, CeltFrameConfig};
use crate::celt::mathops::{celt_log2, celt_sqrt};
use crate::celt::mdct::clt_mdct_forward;
use crate::celt::modes::CeltMode;
use crate::celt::pitch::{run_prefilter, COMBFILTER_MAXPERIOD, COMBFILTER_MINPERIOD};
use crate::celt::quant_bands::{amp2_log2, E_MEANS};
use crate::constants::{valid_channels, valid_sample_rate};
use crate::packet;
use crate::{Error, Result};

pub const CELT_FRAME_SIZES_48K: [usize; 4] = [120, 240, 480, 960];
pub const CELT_MIN_BITRATE: i32 = 500;
pub const CELT_MAX_BITRATE: i32 = 512_000;
pub const CELT_MIN_FRAME_BYTES: usize = 2;
pub const CELT_MAX_FRAME_BYTES: usize = packet::MAX_FRAME_BYTES as usize - 1;

const INTENSITY_THRESHOLDS: [f32; 21] = [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 16.0, 24.0, 36.0, 44.0, 50.0, 56.0, 62.0, 67.0, 72.0,
    79.0, 88.0, 106.0, 134.0,
];
const INTENSITY_HYSTERESIS: [f32; 21] = [
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0,
    8.0, 8.0,
];
const CELT_SIG_SCALE: f32 = 32_768.0;
const TRANSIENT_INV_TABLE: [u8; 128] = [
    255, 255, 156, 110, 86, 70, 59, 51, 45, 40, 37, 33, 31, 28, 26, 25, 23, 22, 21, 20, 19, 18, 17,
    16, 16, 15, 15, 14, 13, 13, 12, 12, 12, 12, 11, 11, 11, 10, 10, 10, 9, 9, 9, 9, 9, 9, 8, 8, 8,
    8, 8, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2,
];

#[derive(Clone, Copy, Debug)]
struct TransientAnalysis {
    is_transient: bool,
    tf_estimate: f32,
    tf_chan: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Application {
    Voip,
    Audio,
    RestrictedLowDelay,
}

impl Application {
    pub const fn code(self) -> i32 {
        match self {
            Self::Voip => 2048,
            Self::Audio => 2049,
            Self::RestrictedLowDelay => 2051,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Encoder {
    sample_rate: i32,
    channels: usize,
    application: Application,
    mode: CeltMode,
    bitrate: i32,
    old_band_e: Vec<f32>,
    energy_error: Vec<f32>,
    preemph_mem: Vec<f32>,
    hp_mem: Vec<f32>,
    overlap_mem: Vec<Vec<f32>>,
    prefilter_mem: Vec<Vec<f32>>,
    prefilter_period: usize,
    prefilter_gain: f32,
    prefilter_tapset: usize,
    tonal_average: i32,
    hf_average: i32,
    tapset_decision: i32,
    spread_decision: i32,
    seed: u32,
    intensity: usize,
    delayed_intra: f32,
    stream_channels: usize,
    stereo_saving: f32,
    last_coded_bands: usize,
    vbr: bool,
    vbr_reservoir: f32,
    vbr_prev_energy: f32,
    analysis: TonalityAnalysisState,
    analysis_info: AnalysisInfo,
}

impl Encoder {
    pub fn new(sample_rate: i32, channels: usize, application: Application) -> Result<Self> {
        if !valid_sample_rate(sample_rate) || !valid_channels(channels as i32) {
            return Err(Error::BadArg);
        }
        let mode = CeltMode::standard_48k();
        let bitrate = if channels == 1 { 64_000 } else { 96_000 };
        Ok(Self {
            sample_rate,
            channels,
            application,
            bitrate,
            old_band_e: vec![0.0; channels * mode.nb_ebands],
            energy_error: vec![0.0; channels * mode.nb_ebands],
            preemph_mem: vec![0.0; channels],
            hp_mem: vec![0.0; channels * 2],
            overlap_mem: vec![vec![0.0; mode.overlap]; channels],
            prefilter_mem: vec![vec![0.0; COMBFILTER_MAXPERIOD]; channels],
            prefilter_period: 0,
            prefilter_gain: 0.0,
            prefilter_tapset: 0,
            tonal_average: 256,
            hf_average: 0,
            tapset_decision: 0,
            spread_decision: SPREAD_NORMAL,
            seed: 0,
            intensity: 0,
            delayed_intra: 1.0,
            stream_channels: channels,
            stereo_saving: 0.0,
            last_coded_bands: 0,
            vbr: false,
            vbr_reservoir: 0.0,
            vbr_prev_energy: 0.0,
            analysis: TonalityAnalysisState::new(),
            analysis_info: AnalysisInfo::default(),
            mode,
        })
    }

    pub const fn sample_rate(&self) -> i32 {
        self.sample_rate
    }

    pub const fn channels(&self) -> usize {
        self.channels
    }

    pub const fn application(&self) -> Application {
        self.application
    }

    pub const fn bitrate(&self) -> i32 {
        self.bitrate
    }

    pub const fn vbr(&self) -> bool {
        self.vbr
    }

    pub fn set_bitrate(&mut self, bitrate: i32) -> Result<()> {
        if !(CELT_MIN_BITRATE..=CELT_MAX_BITRATE).contains(&bitrate) {
            return Err(Error::BadArg);
        }
        self.bitrate = bitrate;
        self.vbr_reservoir = 0.0;
        Ok(())
    }

    pub fn set_vbr(&mut self, enabled: bool) -> Result<()> {
        self.vbr = enabled;
        self.vbr_reservoir = 0.0;
        self.vbr_prev_energy = 0.0;
        Ok(())
    }

    fn frame_lm(&self, frame_size: usize) -> Result<usize> {
        for (lm, size) in CELT_FRAME_SIZES_48K.iter().copied().enumerate() {
            if lm <= self.mode.max_lm && size == frame_size {
                return Ok(lm);
            }
        }
        Err(Error::BadArg)
    }

    fn frame_bytes_for_bitrate(&self, frame_size: usize) -> usize {
        let total_packet_bytes = (self.bitrate as i64 * frame_size as i64
            + self.sample_rate as i64 * 4)
            / (self.sample_rate as i64 * 8);
        (total_packet_bytes as usize)
            .saturating_sub(1)
            .clamp(CELT_MIN_FRAME_BYTES, CELT_MAX_FRAME_BYTES)
    }

    fn vbr_frame_bytes(&mut self, pcm: &[f32], frame_size: usize) -> usize {
        let target = self.frame_bytes_for_bitrate(frame_size) as f32 + 1.0;
        let sample_count = frame_size * self.channels;
        let mut energy = 0.0f32;
        let mut derivative = 0.0f32;
        let mut stereo_diff = 0.0f32;
        let mut stereo_sum = 0.0f32;

        for i in 0..frame_size {
            for c in 0..self.channels {
                let sample = pcm[i * self.channels + c];
                energy += sample * sample;
                if i > 0 {
                    let previous = pcm[(i - 1) * self.channels + c];
                    let delta = sample - previous;
                    derivative += delta * delta;
                }
            }
            if self.channels == 2 {
                let left = pcm[i * 2];
                let right = pcm[i * 2 + 1];
                let sum = left + right;
                let diff = left - right;
                stereo_sum += sum * sum;
                stereo_diff += diff * diff;
            }
        }

        let rms = (energy / sample_count as f32).sqrt();
        let derivative = (derivative / sample_count.max(1) as f32).sqrt();
        let hf_score = (derivative / (rms + 1e-5) * 0.35).clamp(0.0, 1.0);
        let energy_score = (rms * 3.0).clamp(0.0, 1.0);
        let transient_score = if self.vbr_prev_energy > 0.0 {
            ((rms / (self.vbr_prev_energy + 1e-5)) - 1.0).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let stereo_score = if self.channels == 2 {
            (stereo_diff / (stereo_sum + stereo_diff + 1e-5)).sqrt()
        } else {
            0.0
        };

        let complexity =
            (0.38 * energy_score + 0.28 * hf_score + 0.22 * transient_score + 0.12 * stereo_score)
                .clamp(0.0, 1.0);
        let min_bytes = (target * 0.45).round().max(CELT_MIN_FRAME_BYTES as f32);
        let max_bytes = (target * 1.75).round().min(CELT_MAX_FRAME_BYTES as f32);
        // Bias toward Opus constrained-VBR behavior: slightly higher base than the legacy
        // scalar plus a mild reservoir feedback term to stabilize long-term bitrate.
        let reservoir_correction = (self.vbr_reservoir / target).clamp(-0.25, 0.25);
        let desired = target * (0.86 + 0.60 * complexity) * (1.0 + 0.2 * reservoir_correction);
        let chosen = desired.round().clamp(min_bytes, max_bytes) as usize;

        self.vbr_reservoir =
            (self.vbr_reservoir + target - chosen as f32).clamp(-target * 50.0, target * 50.0);
        self.vbr_prev_energy = rms;
        chosen
    }

    fn validate_frame_bytes(frame_bytes: usize) -> Result<()> {
        if !(CELT_MIN_FRAME_BYTES..=CELT_MAX_FRAME_BYTES).contains(&frame_bytes) {
            return Err(Error::BadArg);
        }
        Ok(())
    }

    fn equiv_rate(&self, frame_bytes: usize, lm: usize, stream_channels: usize) -> i32 {
        let overhead = (40 * stream_channels as i32 + 20) * ((400 >> lm) - 50);
        let rate = ((frame_bytes as i32 * 8 * 50) << (3 - lm)) - overhead;
        rate.min(self.bitrate - overhead)
    }

    fn analysis_signal_bandwidth(&self, equiv_rate: i32, stream_channels: usize) -> usize {
        if !self.analysis_info.valid {
            return self.mode.nb_ebands - 1;
        }

        let channels = stream_channels as i32;
        let min_bandwidth = if equiv_rate < 32_000 * channels {
            13
        } else if equiv_rate < 48_000 * channels {
            16
        } else if equiv_rate < 60_000 * channels {
            18
        } else if equiv_rate < 80_000 * channels {
            19
        } else {
            20
        };
        self.analysis_info
            .bandwidth
            .max(min_bandwidth)
            .min(self.mode.nb_ebands - 1)
    }

    fn compute_equiv_rate_for_channels(&self, frame_size: usize, stream_channels: usize) -> i32 {
        let frame_rate = self.sample_rate / frame_size as i32;
        let mut equiv = self.bitrate;
        if frame_rate > 50 {
            equiv -= (40 * stream_channels as i32 + 20) * (frame_rate - 50);
        }
        if !self.vbr {
            equiv -= equiv / 12;
        }
        // Equivalent-rate approximation of upstream CELT complexity path.
        equiv * 95 / 100
    }

    fn transient_analysis(inputs: &[Vec<f32>], channels: usize, len: usize) -> TransientAnalysis {
        let len2 = len / 2;
        let mut mask_metric = 0i32;
        let mut tf_chan = 0usize;

        for (c, input) in inputs.iter().take(channels).enumerate() {
            let mut tmp = vec![0.0f32; len];
            let mut mem0 = 0.0f32;
            let mut mem1 = 0.0f32;
            for i in 0..len {
                let x = input[i];
                let y = mem0 + x;
                let mem00 = mem0;
                mem0 = mem0 - x + 0.5 * mem1;
                mem1 = x - mem00;
                tmp[i] = y;
            }
            tmp.iter_mut().take(12).for_each(|sample| *sample = 0.0);

            let mut mean = 0.0f32;
            mem0 = 0.0;
            let forward_decay = 0.0625f32;
            for i in 0..len2 {
                let x2 = tmp[2 * i] * tmp[2 * i] + tmp[2 * i + 1] * tmp[2 * i + 1];
                mean += x2;
                mem0 = x2 + (1.0 - forward_decay) * mem0;
                tmp[i] = forward_decay * mem0;
            }

            mem0 = 0.0;
            let mut max_e = 0.0f32;
            for i in (0..len2).rev() {
                mem0 = tmp[i] + 0.875 * mem0;
                tmp[i] = 0.125 * mem0;
                max_e = max_e.max(tmp[i]);
            }

            mean = celt_sqrt(mean * max_e * 0.5 * len2 as f32);
            let norm = len2 as f32 / (1e-15f32 + mean);
            let mut unmask = 0i32;
            for i in (12..len2.saturating_sub(5)).step_by(4) {
                let id = (64.0 * norm * (tmp[i] + 1e-15f32))
                    .floor()
                    .clamp(0.0, 127.0) as usize;
                unmask += i32::from(TRANSIENT_INV_TABLE[id]);
            }
            unmask = 64 * unmask * 4 / (6 * (len2 as i32 - 17));
            if unmask > mask_metric {
                mask_metric = unmask;
                tf_chan = c;
            }
        }

        let is_transient = mask_metric > 200;
        let tf_max = celt_sqrt(27.0 * mask_metric as f32).max(42.0) - 42.0;
        let tf_estimate = celt_sqrt((0.0069 * tf_max.min(163.0) - 0.139).max(0.0));
        TransientAnalysis {
            is_transient,
            tf_estimate,
            tf_chan,
        }
    }

    fn compute_mdcts(
        mode: &CeltMode,
        inputs: &[Vec<f32>],
        freq: &mut [f32],
        channels: usize,
        stream_channels: usize,
        lm: usize,
        short_blocks: usize,
    ) {
        let frame_size = mode.short_mdct_size << lm;
        if short_blocks > 0 {
            let shift = mode.max_lm;
            for (c, input) in inputs.iter().take(channels).enumerate() {
                for b in 0..short_blocks {
                    let input_offset = b * mode.short_mdct_size;
                    let output_offset = c * frame_size + b;
                    clt_mdct_forward(
                        &mode.mdct,
                        &input[input_offset..],
                        &mut freq[output_offset..],
                        &mode.window,
                        mode.overlap,
                        shift,
                        short_blocks,
                    );
                }
            }
        } else {
            let shift = mode.max_lm - lm;
            for (c, input) in inputs.iter().take(channels).enumerate() {
                clt_mdct_forward(
                    &mode.mdct,
                    input,
                    &mut freq[c * frame_size..(c + 1) * frame_size],
                    &mode.window,
                    mode.overlap,
                    shift,
                    1,
                );
            }
        }

        if channels == 2 && stream_channels == 1 {
            for i in 0..frame_size {
                freq[i] = 0.5 * (freq[i] + freq[frame_size + i]);
            }
        }
    }

    fn patch_transient_decision(
        new_e: &[f32],
        old_e: &[f32],
        nb_ebands: usize,
        start: usize,
        end: usize,
        channels: usize,
    ) -> bool {
        if end <= start + 1
            || new_e.len() < channels * nb_ebands
            || old_e.len() < channels * nb_ebands
        {
            return false;
        }

        let mut spread_old = vec![0.0f32; nb_ebands];
        if channels == 1 {
            spread_old[start] = old_e[start];
            for i in start + 1..end {
                spread_old[i] = (spread_old[i - 1] - 1.0).max(old_e[i]);
            }
        } else {
            spread_old[start] = old_e[start].max(old_e[start + nb_ebands]);
            for i in start + 1..end {
                spread_old[i] = (spread_old[i - 1] - 1.0).max(old_e[i].max(old_e[i + nb_ebands]));
            }
        }
        for i in (start..end - 1).rev() {
            spread_old[i] = spread_old[i].max(spread_old[i + 1] - 1.0);
        }

        let first = 2.max(start);
        if end <= first + 1 {
            return false;
        }

        let mut mean_diff = 0.0f32;
        for c in 0..channels {
            for i in first..end - 1 {
                let x1 = new_e[i + c * nb_ebands].max(0.0);
                let x2 = spread_old[i].max(0.0);
                mean_diff += (x1 - x2).max(0.0);
            }
        }
        mean_diff /= (channels * (end - 1 - first)) as f32;
        mean_diff > 1.0
    }

    fn spread_weights(
        mode: &CeltMode,
        band_log_e: &[f32],
        end: usize,
        channels: usize,
    ) -> Vec<i32> {
        const LSB_DEPTH: i32 = 24;

        let mut weights = vec![32i32; mode.nb_ebands];
        if end == 0 || band_log_e.len() < channels * mode.nb_ebands {
            return weights;
        }

        let mut noise_floor = vec![0.0f32; mode.nb_ebands];
        for i in 0..end {
            noise_floor[i] = 0.0625 * mode.log_n[i] as f32 + 0.5 + (9 - LSB_DEPTH) as f32
                - E_MEANS[i]
                + 0.0062 * (i as f32 + 5.0) * (i as f32 + 5.0);
        }

        let mut max_depth = -31.9f32;
        let mut mask = vec![0.0f32; mode.nb_ebands];
        for i in 0..end {
            let mut value = band_log_e[i] - noise_floor[i];
            for c in 1..channels {
                value = value.max(band_log_e[c * mode.nb_ebands + i] - noise_floor[i]);
            }
            max_depth = max_depth.max(value);
            mask[i] = value;
        }

        let signal = mask.clone();
        for i in 1..end {
            mask[i] = mask[i].max(mask[i - 1] - 2.0);
        }
        for i in (0..end.saturating_sub(1)).rev() {
            mask[i] = mask[i].max(mask[i + 1] - 3.0);
        }

        let depth_floor = 0.0f32.max(max_depth - 12.0);
        for i in 0..end {
            let smr = signal[i] - depth_floor.max(mask[i]);
            let shift = (-(0.5 + smr).floor() as i32).clamp(0, 5);
            weights[i] = 32 >> shift;
        }

        weights
    }

    fn alloc_trim_analysis(
        mode: &CeltMode,
        norm: &[f32],
        band_log_e: &[f32],
        end: usize,
        lm: usize,
        channels: usize,
        n: usize,
        intensity: usize,
        stereo_saving: &mut f32,
        tf_estimate: f32,
        analysis_tonality_slope: Option<f32>,
        equiv_rate: i32,
    ) -> i32 {
        let mut trim = 5.0f32;
        if equiv_rate < 64_000 {
            trim = 4.0;
        } else if equiv_rate < 80_000 {
            let frac = (equiv_rate - 64_000) >> 10;
            trim = 4.0 + (1.0 / 16.0) * frac as f32;
        }

        if channels == 2 {
            let mut sum = 0.0f32;
            for i in 0..8 {
                let band_start = (mode.ebands[i] as usize) << lm;
                let band_end = (mode.ebands[i + 1] as usize) << lm;
                let partial = (band_start..band_end)
                    .map(|j| norm[j] * norm[n + j])
                    .sum::<f32>();
                sum += partial;
            }
            sum = (0.125 * sum).abs().min(1.0);
            let mut min_xc = sum;
            for i in 8..intensity {
                let band_start = (mode.ebands[i] as usize) << lm;
                let band_end = (mode.ebands[i + 1] as usize) << lm;
                let partial = (band_start..band_end)
                    .map(|j| norm[j] * norm[n + j])
                    .sum::<f32>();
                min_xc = min_xc.min(partial.abs());
            }
            min_xc = min_xc.abs().min(1.0);
            let log_xc = celt_log2(1.001 - sum * sum);
            let log_xc2 = (0.5 * log_xc).max(celt_log2(1.001 - min_xc * min_xc));
            trim += (-4.0f32).max(0.75 * log_xc);
            *stereo_saving = (*stereo_saving + 0.25).min(-0.5 * log_xc2);
        }

        let mut diff = 0.0f32;
        for c in 0..channels {
            for i in 0..end.saturating_sub(1) {
                diff += band_log_e[i + c * mode.nb_ebands] * (2 + 2 * i as i32 - end as i32) as f32;
            }
        }
        if end > 1 {
            diff /= (channels * (end - 1)) as f32;
        }

        trim -= ((diff + 1.0) / 6.0).clamp(-2.0, 2.0);
        trim -= 2.0 * tf_estimate;
        if let Some(tonality_slope) = analysis_tonality_slope {
            trim -= (2.0 * (tonality_slope + 0.05)).clamp(-2.0, 2.0);
        }
        (0.5 + trim).floor().clamp(0.0, 10.0) as i32
    }

    fn stereo_analysis(mode: &CeltMode, x: &[f32], y: &[f32], lm: usize) -> bool {
        let mut sum_lr = f32::EPSILON;
        let mut sum_ms = f32::EPSILON;
        for i in 0..13 {
            let band_start = (mode.ebands[i] as usize) << lm;
            let band_end = (mode.ebands[i + 1] as usize) << lm;
            for j in band_start..band_end {
                let left = x[j];
                let right = y[j];
                sum_lr += left.abs() + right.abs();
                sum_ms += (left + right).abs() + (left - right).abs();
            }
        }
        sum_ms *= core::f32::consts::FRAC_1_SQRT_2;
        let mut thetas = 13usize;
        if lm <= 1 {
            thetas -= 8;
        }
        (((mode.ebands[13] as usize) << (lm + 1)) + thetas) as f32 * sum_ms
            > ((mode.ebands[13] as usize) << (lm + 1)) as f32 * sum_lr
    }

    fn dc_reject_frame(&mut self, pcm: &[f32], frame_size: usize) -> Vec<f32> {
        let cutoff_hz = 3.0f32;
        let coef = 6.3 * cutoff_hz / self.sample_rate as f32;
        let coef2 = 1.0 - coef;
        let mut filtered = vec![0.0f32; frame_size * self.channels];

        if self.channels == 2 {
            let mut m0 = self.hp_mem[0];
            let mut m2 = self.hp_mem[2];
            for i in 0..frame_size {
                let left = pcm[2 * i];
                let right = pcm[2 * i + 1];
                filtered[2 * i] = left - m0;
                filtered[2 * i + 1] = right - m2;
                m0 = coef * left + 1e-30f32 + coef2 * m0;
                m2 = coef * right + 1e-30f32 + coef2 * m2;
            }
            self.hp_mem[0] = m0;
            self.hp_mem[2] = m2;
        } else {
            let mut m0 = self.hp_mem[0];
            for i in 0..frame_size {
                let sample = pcm[i];
                filtered[i] = sample - m0;
                m0 = coef * sample + 1e-30f32 + coef2 * m0;
            }
            self.hp_mem[0] = m0;
        }

        filtered
    }

    fn encode_filtered_f32_with_frame_bytes(
        &mut self,
        pcm: &[f32],
        frame_size: usize,
        frame_bytes: usize,
    ) -> Result<Vec<u8>> {
        let lm = self.frame_lm(frame_size)?;
        Self::validate_frame_bytes(frame_bytes)?;
        let stream_channels = self.choose_stream_channels(frame_size);
        let mut config = CeltFrameConfig::new(&self.mode, lm, stream_channels, frame_bytes)?;
        config.spread = SPREAD_NORMAL;
        config.last_coded_bands = self.last_coded_bands;
        config.vbr = self.vbr;
        config.constrained_vbr = self.vbr;
        let analysis_tonality_slope = self
            .analysis_info
            .valid
            .then_some(self.analysis_info.tonality_slope);
        let equiv_rate = self.equiv_rate(frame_bytes, lm, stream_channels);
        config.signal_bandwidth = self.analysis_signal_bandwidth(equiv_rate, stream_channels);
        if self.analysis_info.valid {
            config.analysis_leak_boost = Some(self.analysis_info.leak_boost);
        }

        let n = frame_size;
        let m = 1usize << lm;
        let overlap = self.mode.overlap;

        let mut inputs = Vec::with_capacity(self.channels);
        for c in 0..self.channels {
            let mut input = vec![0.0f32; 2 * n];
            input[..overlap].copy_from_slice(&self.overlap_mem[c]);
            for i in 0..n {
                let sample = pcm[i * self.channels + c] * CELT_SIG_SCALE;
                input[overlap + i] = sample - self.preemph_mem[c];
                self.preemph_mem[c] = self.mode.preemph[0] * sample;
            }
            inputs.push(input);
        }

        let prefilter_enabled =
            frame_bytes > 12 * stream_channels && (frame_bytes * 8) as i32 >= 16;
        let prefilter_tapset = self.tapset_decision as usize;
        let (prefilter, prefilter_gain) = run_prefilter(
            &self.mode,
            &mut inputs,
            &mut self.prefilter_mem,
            self.prefilter_period,
            self.prefilter_gain,
            self.prefilter_tapset,
            prefilter_tapset,
            prefilter_enabled,
            frame_bytes,
            self.channels,
            n,
        );
        config.prefilter = Some(prefilter);
        self.prefilter_period = if prefilter.pitch > 0 {
            prefilter.pitch as usize
        } else {
            COMBFILTER_MINPERIOD
        };
        self.prefilter_gain = prefilter_gain;
        self.prefilter_tapset = prefilter_tapset;
        for c in 0..self.channels {
            self.overlap_mem[c].copy_from_slice(&inputs[c][n..n + overlap]);
        }

        let transient = Self::transient_analysis(&inputs, self.channels, n + overlap);
        config.is_transient = lm > 0 && transient.is_transient && (frame_bytes * 8) as i32 >= 16;
        config.tf_estimate = transient.tf_estimate;
        config.tf_chan = if stream_channels == 1 {
            0
        } else {
            transient.tf_chan.min(stream_channels - 1)
        };

        let mut freq = vec![0.0f32; self.channels * n];
        let short_blocks = if config.is_transient { m } else { 0 };
        Self::compute_mdcts(
            &self.mode,
            &inputs,
            &mut freq,
            self.channels,
            stream_channels,
            lm,
            short_blocks,
        );

        let eff_end = self.mode.eff_ebands;
        let mut band_e = vec![0.0f32; stream_channels * self.mode.nb_ebands];
        compute_band_energies(&self.mode, &freq, &mut band_e, eff_end, stream_channels, lm);
        let mut band_log_e = vec![0.0f32; stream_channels * self.mode.nb_ebands];
        amp2_log2(
            &self.mode,
            eff_end,
            config.end,
            &band_e,
            &mut band_log_e,
            stream_channels,
        );
        let mut band_log_e2 = None;
        if config.is_transient {
            let mut long_freq = vec![0.0f32; self.channels * n];
            Self::compute_mdcts(
                &self.mode,
                &inputs,
                &mut long_freq,
                self.channels,
                stream_channels,
                lm,
                0,
            );
            let mut band_e2 = vec![0.0f32; stream_channels * self.mode.nb_ebands];
            compute_band_energies(
                &self.mode,
                &long_freq,
                &mut band_e2,
                eff_end,
                stream_channels,
                lm,
            );
            let mut long_band_log_e = vec![0.0f32; stream_channels * self.mode.nb_ebands];
            amp2_log2(
                &self.mode,
                eff_end,
                config.end,
                &band_e2,
                &mut long_band_log_e,
                stream_channels,
            );
            for c in 0..stream_channels {
                for i in 0..config.end {
                    long_band_log_e[i + c * self.mode.nb_ebands] += 0.5 * lm as f32;
                }
            }
            band_log_e2 = Some(long_band_log_e);
        } else if lm > 0
            && Self::patch_transient_decision(
                &band_log_e,
                &self.old_band_e,
                self.mode.nb_ebands,
                config.start,
                config.end,
                stream_channels,
            )
        {
            config.is_transient = true;
            config.tf_estimate = 0.2;
            let mut long_band_log_e = band_log_e.clone();
            for c in 0..stream_channels {
                for i in 0..config.end {
                    long_band_log_e[i + c * self.mode.nb_ebands] += 0.5 * lm as f32;
                }
            }
            band_log_e2 = Some(long_band_log_e);
            Self::compute_mdcts(
                &self.mode,
                &inputs,
                &mut freq,
                self.channels,
                stream_channels,
                lm,
                m,
            );
            compute_band_energies(&self.mode, &freq, &mut band_e, eff_end, stream_channels, lm);
            amp2_log2(
                &self.mode,
                eff_end,
                config.end,
                &band_e,
                &mut band_log_e,
                stream_channels,
            );
        }
        config.band_log_e2 = band_log_e2;
        let mut norm = vec![0.0f32; stream_channels * n];
        normalise_bands(
            &self.mode,
            &freq,
            &mut norm,
            &band_e,
            eff_end,
            stream_channels,
            m,
        );
        if !config.is_transient && frame_bytes >= 10 * stream_channels && stream_channels > 0 {
            let spread_weight =
                Self::spread_weights(&self.mode, &band_log_e, eff_end, stream_channels);
            config.spread = spreading_decision(
                &self.mode,
                &norm,
                &mut self.tonal_average,
                self.spread_decision,
                &mut self.hf_average,
                &mut self.tapset_decision,
                prefilter.enabled,
                eff_end,
                stream_channels,
                m,
                &spread_weight,
            );
            self.spread_decision = config.spread;
        } else {
            config.spread = SPREAD_NORMAL;
            self.spread_decision = config.spread;
        }

        if stream_channels == 1 {
            config.alloc_trim = Self::alloc_trim_analysis(
                &self.mode,
                &norm,
                &band_log_e,
                config.end,
                lm,
                stream_channels,
                n,
                config.intensity,
                &mut self.stereo_saving,
                config.tf_estimate,
                analysis_tonality_slope,
                equiv_rate,
            );
        } else {
            {
                let (left, right) = norm.split_at_mut(n);
                self.intensity = hysteresis_decision(
                    (equiv_rate / 1000) as f32,
                    &INTENSITY_THRESHOLDS,
                    &INTENSITY_HYSTERESIS,
                    INTENSITY_THRESHOLDS.len(),
                    self.intensity,
                )
                .clamp(config.start, config.end);
                config.intensity = self.intensity;
                config.dual_stereo = lm != 0 && Self::stereo_analysis(&self.mode, left, right, lm);
            }
            config.alloc_trim = Self::alloc_trim_analysis(
                &self.mode,
                &norm,
                &band_log_e,
                config.end,
                lm,
                stream_channels,
                n,
                config.intensity,
                &mut self.stereo_saving,
                config.tf_estimate,
                analysis_tonality_slope,
                equiv_rate,
            );
        }

        let encoded = if stream_channels == 1 {
            encode_spectral_frame(
                &self.mode,
                &config,
                &mut norm,
                None,
                &band_e,
                &mut self.old_band_e[..self.mode.nb_ebands],
                &mut self.energy_error[..self.mode.nb_ebands],
                &mut self.delayed_intra,
                &mut self.seed,
            )?
        } else {
            let (left, right) = norm.split_at_mut(n);
            encode_spectral_frame(
                &self.mode,
                &config,
                left,
                Some(right),
                &band_e,
                &mut self.old_band_e,
                &mut self.energy_error,
                &mut self.delayed_intra,
                &mut self.seed,
            )?
        };
        if stream_channels == 2 {
            self.intensity = encoded.allocation.intensity;
        }
        self.last_coded_bands = if self.last_coded_bands != 0 {
            (self.last_coded_bands + 1).min(
                self.last_coded_bands
                    .saturating_sub(1)
                    .max(encoded.allocation.coded_bands),
            )
        } else {
            encoded.allocation.coded_bands
        };
        if self.channels == 2 && stream_channels == 1 {
            let (left, right) = self.old_band_e.split_at_mut(self.mode.nb_ebands);
            right[..self.mode.nb_ebands].copy_from_slice(left);
            let (left, right) = self.energy_error.split_at_mut(self.mode.nb_ebands);
            right[..self.mode.nb_ebands].copy_from_slice(left);
        }

        let mut packet = Vec::with_capacity(1 + encoded.data.len());
        packet.push(packet::make_celt_only_fullband_toc(lm, stream_channels)?);
        packet.extend_from_slice(&encoded.data);
        Ok(packet)
    }

    fn opus_equiv_rate_for_packet(&self, frame_size: usize, stream_channels: usize) -> i32 {
        self.compute_equiv_rate_for_channels(frame_size, stream_channels)
    }

    fn choose_stream_channels(&mut self, frame_size: usize) -> usize {
        if self.channels != 2 {
            self.stream_channels = self.channels;
            return self.stream_channels;
        }

        let voice_est = match self.application {
            Application::Voip => 115,
            Application::Audio | Application::RestrictedLowDelay => 48,
        };
        let mut threshold = 17_000 + ((voice_est * voice_est * (19_000 - 17_000)) >> 14);
        if self.stream_channels == 2 {
            threshold -= 1_000;
        } else {
            threshold += 1_000;
        }
        let equiv_rate = self.opus_equiv_rate_for_packet(frame_size, self.channels);
        self.stream_channels = if equiv_rate > threshold { 2 } else { 1 };
        self.stream_channels
    }

    pub fn encode_i16(&mut self, pcm: &[i16], frame_size: usize) -> Result<Vec<u8>> {
        let pcm_f32 = pcm
            .iter()
            .map(|sample| *sample as f32 / 32768.0)
            .collect::<Vec<_>>();
        self.encode_f32(&pcm_f32, frame_size)
    }

    pub fn encode_i16_with_frame_bytes(
        &mut self,
        pcm: &[i16],
        frame_size: usize,
        frame_bytes: usize,
    ) -> Result<Vec<u8>> {
        let pcm_f32 = pcm
            .iter()
            .map(|sample| *sample as f32 / 32768.0)
            .collect::<Vec<_>>();
        self.encode_f32_with_frame_bytes(&pcm_f32, frame_size, frame_bytes)
    }

    pub fn encode_f32(&mut self, pcm: &[f32], frame_size: usize) -> Result<Vec<u8>> {
        if self.sample_rate != 48_000 {
            return Err(Error::Unimplemented);
        }
        if pcm.len() < frame_size * self.channels {
            return Err(Error::BadArg);
        }
        self.analysis_info = self.analysis.run(pcm, frame_size, self.channels);
        let filtered = self.dc_reject_frame(pcm, frame_size);
        let frame_bytes = if self.vbr {
            self.vbr_frame_bytes(&filtered, frame_size)
        } else {
            self.frame_bytes_for_bitrate(frame_size)
        };
        self.encode_filtered_f32_with_frame_bytes(&filtered, frame_size, frame_bytes)
    }

    pub fn encode_f32_with_frame_bytes(
        &mut self,
        pcm: &[f32],
        frame_size: usize,
        frame_bytes: usize,
    ) -> Result<Vec<u8>> {
        if self.sample_rate != 48_000 {
            return Err(Error::Unimplemented);
        }
        if pcm.len() < frame_size * self.channels {
            return Err(Error::BadArg);
        }
        self.analysis_info = self.analysis.run(pcm, frame_size, self.channels);
        let filtered = self.dc_reject_frame(pcm, frame_size);
        self.encode_filtered_f32_with_frame_bytes(&filtered, frame_size, frame_bytes)
    }
}
