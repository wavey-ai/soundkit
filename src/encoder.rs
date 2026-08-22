use crate::analysis::{AnalysisInfo, TonalityAnalysisState};
use crate::celt::bands::{
    compute_band_energies, hysteresis_decision, normalise_bands, spreading_decision, SPREAD_NORMAL,
};
use crate::celt::codec::{
    encode_spectral_frame_with_scratch, CeltFrameConfig, CeltFrameEncodeScratch, CeltVbrConfig,
};
use crate::celt::mathops::{celt_log2, celt_sqrt};
use crate::celt::mdct::{clt_mdct_forward_with_scratch, MdctScratch};
use crate::celt::modes::CeltMode;
use crate::celt::pitch::{
    run_prefilter, tone_detect, PrefilterScratch, ToneAnalysis, COMBFILTER_MAXPERIOD,
    COMBFILTER_MINPERIOD,
};
use crate::celt::quant_bands::{amp2_log2, E_MEANS};
use crate::constants::{valid_channels, valid_sample_rate, PCM_I24_MAX, PCM_I24_MIN};
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
const BITRES: i32 = 3;
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

#[derive(Clone, Debug, Default)]
struct SpreadAnalysisScratch {
    weights: Vec<i32>,
    noise_floor: Vec<f32>,
    mask: Vec<f32>,
    signal: Vec<f32>,
}

#[derive(Clone, Debug, Default)]
struct EncoderFrameScratch {
    inputs: Vec<Vec<f32>>,
    freq: Vec<f32>,
    band_e: Vec<f32>,
    band_log_e: Vec<f32>,
    norm: Vec<f32>,
    transient_old: Vec<f32>,
    spread: SpreadAnalysisScratch,
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
    hybrid_stereo_width_q14: i32,
    stereo_saving: f32,
    last_coded_bands: usize,
    vbr: bool,
    vbr_reservoir: i32,
    vbr_drift: i32,
    vbr_offset: i32,
    vbr_count: i32,
    spec_avg: f32,
    analysis: TonalityAnalysisState,
    analysis_info: AnalysisInfo,
    mdct_scratch: MdctScratch,
    spectral_scratch: CeltFrameEncodeScratch,
    pcm_f32_scratch: Vec<f32>,
    filtered_scratch: Vec<f32>,
    tone_scratch: Vec<f32>,
    transient_scratch: Vec<f32>,
    prefilter_scratch: PrefilterScratch,
    frame_scratch: EncoderFrameScratch,
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
            hybrid_stereo_width_q14: 1 << 14,
            stereo_saving: 0.0,
            last_coded_bands: 0,
            vbr: false,
            vbr_reservoir: 0,
            vbr_drift: 0,
            vbr_offset: 0,
            vbr_count: 0,
            spec_avg: 0.0,
            analysis: TonalityAnalysisState::new(),
            analysis_info: AnalysisInfo::default(),
            mdct_scratch: MdctScratch::default(),
            spectral_scratch: CeltFrameEncodeScratch::default(),
            pcm_f32_scratch: Vec::new(),
            filtered_scratch: Vec::new(),
            tone_scratch: Vec::new(),
            transient_scratch: Vec::new(),
            prefilter_scratch: PrefilterScratch::default(),
            frame_scratch: EncoderFrameScratch::default(),
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
        self.reset_vbr_state();
        Ok(())
    }

    pub fn set_vbr(&mut self, enabled: bool) -> Result<()> {
        self.vbr = enabled;
        self.reset_vbr_state();
        Ok(())
    }

    fn reset_vbr_state(&mut self) {
        self.vbr_reservoir = 0;
        self.vbr_drift = 0;
        self.vbr_offset = 0;
        self.vbr_count = 0;
        self.spec_avg = 0.0;
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

    fn vbr_rate_frac(&self, frame_size: usize) -> i32 {
        let den = self.sample_rate >> BITRES;
        (self.bitrate * frame_size as i32 + (den >> 1)) / den
    }

    fn vbr_initial_frame_bytes(&self, frame_size: usize) -> usize {
        let vbr_rate = self.vbr_rate_frac(frame_size);
        let max_allowed = ((2 * vbr_rate - self.vbr_reservoir) >> (BITRES + 3))
            .clamp(2, CELT_MAX_FRAME_BYTES as i32);
        max_allowed as usize
    }

    fn temporal_vbr(
        &mut self,
        band_log_e: &[f32],
        start: usize,
        end: usize,
        channels: usize,
        lm: usize,
        is_transient: bool,
    ) -> f32 {
        if end <= start || band_log_e.len() < channels * self.mode.nb_ebands {
            return 0.0;
        }
        let mut follow = -10.0f32;
        let mut frame_avg = 0.0f32;
        let offset = if is_transient { 0.5 * lm as f32 } else { 0.0 };
        for i in start..end {
            follow = (follow - 1.0).max(band_log_e[i] - offset);
            if channels == 2 {
                follow = follow.max(band_log_e[self.mode.nb_ebands + i] - offset);
            }
            frame_avg += follow;
        }
        frame_avg /= (end - start) as f32;
        let temporal_vbr = (frame_avg - self.spec_avg).clamp(-1.5, 3.0);
        self.spec_avg += 0.02 * temporal_vbr;
        temporal_vbr
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

    fn target_stereo_width_q14(equiv_rate: i32) -> i32 {
        if equiv_rate > 32_000 {
            1 << 14
        } else if equiv_rate < 16_000 {
            0
        } else {
            16_384 - 2_048 * (32_000 - equiv_rate) / (equiv_rate - 14_000)
        }
    }

    fn apply_stereo_width_fade(
        mode: &CeltMode,
        frame_size: usize,
        previous_width_q14: i32,
        target_width_q14: i32,
        pcm: &mut [f32],
    ) {
        let reduction_start = 1.0 - previous_width_q14 as f32 * (1.0 / 16_384.0);
        let reduction_end = 1.0 - target_width_q14 as f32 * (1.0 / 16_384.0);
        let overlap = mode.overlap.min(frame_size);
        for i in 0..overlap {
            let w = mode.window[i] * mode.window[i];
            let reduction = w * reduction_end + (1.0 - w) * reduction_start;
            let diff = 0.5 * (pcm[2 * i] - pcm[2 * i + 1]) * reduction;
            pcm[2 * i] -= diff;
            pcm[2 * i + 1] += diff;
        }
        for i in overlap..frame_size {
            let diff = 0.5 * (pcm[2 * i] - pcm[2 * i + 1]) * reduction_end;
            pcm[2 * i] -= diff;
            pcm[2 * i + 1] += diff;
        }
    }

    fn transient_analysis(
        inputs: &[Vec<f32>],
        channels: usize,
        len: usize,
        tone: ToneAnalysis,
        scratch: &mut Vec<f32>,
    ) -> TransientAnalysis {
        let len2 = len / 2;
        let mut mask_metric = 0i32;
        let mut tf_chan = 0usize;

        for (c, input) in inputs.iter().take(channels).enumerate() {
            scratch.resize(len, 0.0);
            let tmp = &mut scratch[..len];
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

        let mut is_transient = mask_metric > 200;
        if tone.toneishness > 0.98 && tone.frequency < 0.026 {
            is_transient = false;
            mask_metric = 0;
        }
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
        scratch: &mut MdctScratch,
    ) {
        let frame_size = mode.short_mdct_size << lm;
        if short_blocks > 0 {
            let shift = mode.max_lm;
            for (c, input) in inputs.iter().take(channels).enumerate() {
                for b in 0..short_blocks {
                    let input_offset = b * mode.short_mdct_size;
                    let output_offset = c * frame_size + b;
                    clt_mdct_forward_with_scratch(
                        &mode.mdct,
                        &input[input_offset..],
                        &mut freq[output_offset..],
                        &mode.window,
                        mode.overlap,
                        shift,
                        short_blocks,
                        scratch,
                    );
                }
            }
        } else {
            let shift = mode.max_lm - lm;
            for (c, input) in inputs.iter().take(channels).enumerate() {
                clt_mdct_forward_with_scratch(
                    &mode.mdct,
                    input,
                    &mut freq[c * frame_size..(c + 1) * frame_size],
                    &mode.window,
                    mode.overlap,
                    shift,
                    1,
                    scratch,
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
        spread_old: &mut Vec<f32>,
    ) -> bool {
        if end <= start + 1
            || new_e.len() < channels * nb_ebands
            || old_e.len() < channels * nb_ebands
        {
            return false;
        }

        spread_old.resize(nb_ebands, 0.0);
        spread_old[..nb_ebands].fill(0.0);
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
        scratch: &mut SpreadAnalysisScratch,
    ) {
        const LSB_DEPTH: i32 = 24;

        scratch.weights.resize(mode.nb_ebands, 32);
        scratch.weights[..mode.nb_ebands].fill(32);
        if end == 0 || band_log_e.len() < channels * mode.nb_ebands {
            return;
        }

        scratch.noise_floor.resize(mode.nb_ebands, 0.0);
        scratch.noise_floor[..mode.nb_ebands].fill(0.0);
        for i in 0..end {
            scratch.noise_floor[i] = 0.0625 * mode.log_n[i] as f32 + 0.5 + (9 - LSB_DEPTH) as f32
                - E_MEANS[i]
                + 0.0062 * (i as f32 + 5.0) * (i as f32 + 5.0);
        }

        let mut max_depth = -31.9f32;
        scratch.mask.resize(mode.nb_ebands, 0.0);
        scratch.mask[..mode.nb_ebands].fill(0.0);
        for i in 0..end {
            let mut value = band_log_e[i] - scratch.noise_floor[i];
            for c in 1..channels {
                value = value.max(band_log_e[c * mode.nb_ebands + i] - scratch.noise_floor[i]);
            }
            max_depth = max_depth.max(value);
            scratch.mask[i] = value;
        }

        scratch.signal.resize(mode.nb_ebands, 0.0);
        scratch.signal[..mode.nb_ebands].copy_from_slice(&scratch.mask[..mode.nb_ebands]);
        for i in 1..end {
            scratch.mask[i] = scratch.mask[i].max(scratch.mask[i - 1] - 2.0);
        }
        for i in (0..end.saturating_sub(1)).rev() {
            scratch.mask[i] = scratch.mask[i].max(scratch.mask[i + 1] - 3.0);
        }

        let depth_floor = 0.0f32.max(max_depth - 12.0);
        for i in 0..end {
            let smr = scratch.signal[i] - depth_floor.max(scratch.mask[i]);
            let shift = (-(0.5 + smr).floor() as i32).clamp(0, 5);
            scratch.weights[i] = 32 >> shift;
        }
    }

    fn apply_energy_error_feedback(
        mode: &CeltMode,
        band_log_e: &mut [f32],
        old_band_e: &[f32],
        energy_error: &[f32],
        start: usize,
        end: usize,
        channels: usize,
    ) {
        for c in 0..channels {
            for i in start..end {
                let idx = i + c * mode.nb_ebands;
                if (band_log_e[idx] - old_band_e[idx]).abs() < 2.0 {
                    band_log_e[idx] -= 0.25 * energy_error[idx];
                }
            }
        }
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

        let tilt = ((diff + 1.0) / 6.0).clamp(-2.0, 2.0);
        trim -= tilt;
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

    fn dc_reject_frame_into(&mut self, pcm: &[f32], frame_size: usize, filtered: &mut Vec<f32>) {
        let cutoff_hz = 3.0f32;
        let coef = 6.3 * cutoff_hz / self.sample_rate as f32;
        let coef2 = 1.0 - coef;
        filtered.resize(frame_size * self.channels, 0.0);

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
    }

    fn encode_filtered_f32_with_frame_bytes(
        &mut self,
        pcm: &mut [f32],
        frame_size: usize,
        frame_bytes: usize,
        allow_vbr_shrink: bool,
    ) -> Result<Vec<u8>> {
        let mut scratch = std::mem::take(&mut self.frame_scratch);
        let result = self.encode_filtered_f32_with_frame_bytes_inner(
            pcm,
            frame_size,
            frame_bytes,
            allow_vbr_shrink,
            &mut scratch,
        );
        self.frame_scratch = scratch;
        result
    }

    fn encode_filtered_f32_with_frame_bytes_inner(
        &mut self,
        pcm: &mut [f32],
        frame_size: usize,
        frame_bytes: usize,
        allow_vbr_shrink: bool,
        scratch: &mut EncoderFrameScratch,
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
        let target_stereo_width_q14 = Self::target_stereo_width_q14(equiv_rate);
        let previous_stereo_width_q14 = self.hybrid_stereo_width_q14;
        if self.channels == 2
            && (previous_stereo_width_q14 < (1 << 14) || target_stereo_width_q14 < (1 << 14))
        {
            Self::apply_stereo_width_fade(
                &self.mode,
                frame_size,
                previous_stereo_width_q14,
                target_stereo_width_q14,
                pcm,
            );
        }
        self.hybrid_stereo_width_q14 = target_stereo_width_q14;

        scratch.inputs.resize_with(self.channels, Vec::new);
        for c in 0..self.channels {
            let input = &mut scratch.inputs[c];
            input.resize(2 * n, 0.0);
            input[..2 * n].fill(0.0);
            input[..overlap].copy_from_slice(&self.overlap_mem[c]);
            for i in 0..n {
                let sample = pcm[i * self.channels + c] * CELT_SIG_SCALE;
                input[overlap + i] = sample - self.preemph_mem[c];
                self.preemph_mem[c] = self.mode.preemph[0] * sample;
            }
        }

        let tone = tone_detect(
            &scratch.inputs,
            self.channels,
            n + overlap,
            self.mode.fs as usize,
            &mut self.tone_scratch,
        );
        let transient = Self::transient_analysis(
            &scratch.inputs,
            self.channels,
            n + overlap,
            tone,
            &mut self.transient_scratch,
        );
        let toneishness = tone.toneishness.min(1.0 - transient.tf_estimate);
        config.is_transient = lm > 0 && transient.is_transient && (frame_bytes * 8) as i32 >= 16;
        config.tf_estimate = transient.tf_estimate;
        config.tone_frequency = tone.frequency;
        config.toneishness = toneishness;
        config.tf_chan = if stream_channels == 1 {
            0
        } else {
            transient.tf_chan.min(stream_channels - 1)
        };

        let prefilter_enabled =
            frame_bytes > 12 * stream_channels && (frame_bytes * 8) as i32 >= 16;
        let prefilter_tapset = self.tapset_decision as usize;
        let previous_prefilter_period = self.prefilter_period.max(COMBFILTER_MINPERIOD);
        let previous_prefilter_gain = self.prefilter_gain;
        let (prefilter, prefilter_gain) = run_prefilter(
            &self.mode,
            &mut scratch.inputs,
            &mut self.prefilter_mem,
            self.prefilter_period,
            self.prefilter_gain,
            self.prefilter_tapset,
            prefilter_tapset,
            prefilter_enabled,
            transient.tf_estimate,
            tone.frequency,
            toneishness,
            self.analysis_info
                .valid
                .then_some(self.analysis_info.max_pitch_ratio),
            frame_bytes,
            self.channels,
            n,
            &mut self.prefilter_scratch,
        );
        let pitch_change = (prefilter_gain > 0.4 || previous_prefilter_gain > 0.4)
            && (!self.analysis_info.valid || self.analysis_info.tonality > 0.3)
            && ((prefilter.pitch as f32) > 1.26 * previous_prefilter_period as f32
                || (prefilter.pitch as f32) < 0.79 * previous_prefilter_period as f32);
        config.prefilter = Some(prefilter);
        self.prefilter_period = if prefilter.pitch > 0 {
            prefilter.pitch as usize
        } else {
            COMBFILTER_MINPERIOD
        };
        self.prefilter_gain = prefilter_gain;
        self.prefilter_tapset = prefilter_tapset;
        for c in 0..self.channels {
            self.overlap_mem[c].copy_from_slice(&scratch.inputs[c][n..n + overlap]);
        }

        scratch.freq.resize(self.channels * n, 0.0);
        scratch.freq[..self.channels * n].fill(0.0);
        let short_blocks = if config.is_transient { m } else { 0 };
        Self::compute_mdcts(
            &self.mode,
            &scratch.inputs,
            &mut scratch.freq,
            self.channels,
            stream_channels,
            lm,
            short_blocks,
            &mut self.mdct_scratch,
        );
        let eff_end = self.mode.eff_ebands;
        let band_count = stream_channels * self.mode.nb_ebands;
        scratch.band_e.resize(band_count, 0.0);
        scratch.band_e[..band_count].fill(0.0);
        compute_band_energies(
            &self.mode,
            &scratch.freq,
            &mut scratch.band_e,
            eff_end,
            stream_channels,
            lm,
        );
        scratch.band_log_e.resize(band_count, 0.0);
        scratch.band_log_e[..band_count].fill(0.0);
        amp2_log2(
            &self.mode,
            eff_end,
            config.end,
            &scratch.band_e,
            &mut scratch.band_log_e,
            stream_channels,
        );
        let temporal_vbr = if allow_vbr_shrink && self.vbr {
            self.temporal_vbr(
                &scratch.band_log_e,
                config.start,
                config.end,
                stream_channels,
                lm,
                config.is_transient,
            )
        } else {
            0.0
        };
        let mut band_log_e2 = None;
        if config.is_transient {
            let mut long_freq = vec![0.0f32; self.channels * n];
            Self::compute_mdcts(
                &self.mode,
                &scratch.inputs,
                &mut long_freq,
                self.channels,
                stream_channels,
                lm,
                0,
                &mut self.mdct_scratch,
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
                &scratch.band_log_e,
                &self.old_band_e,
                self.mode.nb_ebands,
                config.start,
                config.end,
                stream_channels,
                &mut scratch.transient_old,
            )
        {
            config.is_transient = true;
            config.tf_estimate = 0.2;
            let mut long_band_log_e = scratch.band_log_e.clone();
            for c in 0..stream_channels {
                for i in 0..config.end {
                    long_band_log_e[i + c * self.mode.nb_ebands] += 0.5 * lm as f32;
                }
            }
            band_log_e2 = Some(long_band_log_e);
            Self::compute_mdcts(
                &self.mode,
                &scratch.inputs,
                &mut scratch.freq,
                self.channels,
                stream_channels,
                lm,
                m,
                &mut self.mdct_scratch,
            );
            compute_band_energies(
                &self.mode,
                &scratch.freq,
                &mut scratch.band_e,
                eff_end,
                stream_channels,
                lm,
            );
            amp2_log2(
                &self.mode,
                eff_end,
                config.end,
                &scratch.band_e,
                &mut scratch.band_log_e,
                stream_channels,
            );
        }
        config.band_log_e2 = band_log_e2;
        scratch.norm.resize(stream_channels * n, 0.0);
        scratch.norm[..stream_channels * n].fill(0.0);
        normalise_bands(
            &self.mode,
            &scratch.freq,
            &mut scratch.norm,
            &scratch.band_e,
            eff_end,
            stream_channels,
            m,
        );
        if !config.is_transient && frame_bytes >= 10 * stream_channels && stream_channels > 0 {
            Self::spread_weights(
                &self.mode,
                &scratch.band_log_e,
                eff_end,
                stream_channels,
                &mut scratch.spread,
            );
            config.spread = spreading_decision(
                &self.mode,
                &scratch.norm,
                &mut self.tonal_average,
                self.spread_decision,
                &mut self.hf_average,
                &mut self.tapset_decision,
                prefilter.enabled,
                eff_end,
                stream_channels,
                m,
                &scratch.spread.weights,
            );
            self.spread_decision = config.spread;
        } else {
            config.spread = SPREAD_NORMAL;
            self.spread_decision = config.spread;
        }

        // libopus applies this feedback to bandLogE before trim analysis. The
        // spectral encoder recomputes its own copy, so mirror it here too.
        Self::apply_energy_error_feedback(
            &self.mode,
            &mut scratch.band_log_e,
            &self.old_band_e,
            &self.energy_error,
            config.start,
            config.end,
            stream_channels,
        );

        if stream_channels == 1 {
            config.alloc_trim = Self::alloc_trim_analysis(
                &self.mode,
                &scratch.norm,
                &scratch.band_log_e,
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
                let (left, right) = scratch.norm.split_at_mut(n);
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
                &scratch.norm,
                &scratch.band_log_e,
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
        if allow_vbr_shrink && self.vbr {
            let vbr_rate = self.vbr_rate_frac(frame_size);
            config.vbr_state = Some(CeltVbrConfig {
                equiv_rate,
                vbr_rate,
                effective_bytes: (vbr_rate >> (BITRES + 3)).max(2) as usize,
                reservoir: self.vbr_reservoir,
                drift: self.vbr_drift,
                offset: self.vbr_offset,
                count: self.vbr_count,
                stereo_saving: self.stereo_saving,
                temporal_vbr,
                analysis_valid: self.analysis_info.valid,
                activity: self.analysis_info.activity,
                tonality: self.analysis_info.tonality,
                pitch_change,
            });
        }

        let encoded = if stream_channels == 1 {
            encode_spectral_frame_with_scratch(
                &self.mode,
                &config,
                &mut scratch.norm,
                None,
                &scratch.band_e,
                &mut self.old_band_e[..self.mode.nb_ebands],
                &mut self.energy_error[..self.mode.nb_ebands],
                &mut self.delayed_intra,
                &mut self.seed,
                &mut self.spectral_scratch,
            )?
        } else {
            let (left, right) = scratch.norm.split_at_mut(n);
            encode_spectral_frame_with_scratch(
                &self.mode,
                &config,
                left,
                Some(right),
                &scratch.band_e,
                &mut self.old_band_e,
                &mut self.energy_error,
                &mut self.delayed_intra,
                &mut self.seed,
                &mut self.spectral_scratch,
            )?
        };
        if stream_channels == 2 {
            self.intensity = encoded.allocation.intensity;
        }
        if let Some(update) = encoded.vbr_update {
            self.vbr_reservoir = update.reservoir;
            self.vbr_drift = update.drift;
            self.vbr_offset = update.offset;
            self.vbr_count = update.count;
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

        let mut packet = encoded.data;
        packet.insert(0, packet::make_celt_only_fullband_toc(lm, stream_channels)?);
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
        let required = frame_size * self.channels;
        if pcm.len() < required {
            return Err(Error::BadArg);
        }
        let mut pcm_f32 = std::mem::take(&mut self.pcm_f32_scratch);
        pcm_f32.resize(required, 0.0);
        for (dst, src) in pcm_f32.iter_mut().zip(pcm.iter().take(required)) {
            *dst = *src as f32 / 32768.0;
        }
        let result = self.encode_f32(&pcm_f32, frame_size);
        self.pcm_f32_scratch = pcm_f32;
        result
    }

    pub fn encode_i16_with_frame_bytes(
        &mut self,
        pcm: &[i16],
        frame_size: usize,
        frame_bytes: usize,
    ) -> Result<Vec<u8>> {
        let required = frame_size * self.channels;
        if pcm.len() < required {
            return Err(Error::BadArg);
        }
        let mut pcm_f32 = std::mem::take(&mut self.pcm_f32_scratch);
        pcm_f32.resize(required, 0.0);
        for (dst, src) in pcm_f32.iter_mut().zip(pcm.iter().take(required)) {
            *dst = *src as f32 / 32768.0;
        }
        let result = self.encode_f32_with_frame_bytes(&pcm_f32, frame_size, frame_bytes);
        self.pcm_f32_scratch = pcm_f32;
        result
    }

    /// Encodes signed 24-bit PCM stored sign-extended in `i32` samples.
    ///
    /// Every sample must be in `PCM_I24_MIN..=PCM_I24_MAX`.
    pub fn encode_i24(&mut self, pcm: &[i32], frame_size: usize) -> Result<Vec<u8>> {
        let required = frame_size * self.channels;
        if pcm.len() < required
            || pcm
                .iter()
                .take(required)
                .any(|&sample| !(PCM_I24_MIN..=PCM_I24_MAX).contains(&sample))
        {
            return Err(Error::BadArg);
        }
        let mut pcm_f32 = std::mem::take(&mut self.pcm_f32_scratch);
        pcm_f32.resize(required, 0.0);
        for (dst, src) in pcm_f32.iter_mut().zip(pcm.iter().take(required)) {
            *dst = *src as f32 / 8_388_608.0;
        }
        let result = self.encode_f32(&pcm_f32, frame_size);
        self.pcm_f32_scratch = pcm_f32;
        result
    }

    /// Encodes signed 24-bit PCM to an exact compressed-frame byte budget.
    ///
    /// Samples are stored sign-extended in `i32` and must be in
    /// `PCM_I24_MIN..=PCM_I24_MAX`.
    pub fn encode_i24_with_frame_bytes(
        &mut self,
        pcm: &[i32],
        frame_size: usize,
        frame_bytes: usize,
    ) -> Result<Vec<u8>> {
        let required = frame_size * self.channels;
        if pcm.len() < required
            || pcm
                .iter()
                .take(required)
                .any(|&sample| !(PCM_I24_MIN..=PCM_I24_MAX).contains(&sample))
        {
            return Err(Error::BadArg);
        }
        let mut pcm_f32 = std::mem::take(&mut self.pcm_f32_scratch);
        pcm_f32.resize(required, 0.0);
        for (dst, src) in pcm_f32.iter_mut().zip(pcm.iter().take(required)) {
            *dst = *src as f32 / 8_388_608.0;
        }
        let result = self.encode_f32_with_frame_bytes(&pcm_f32, frame_size, frame_bytes);
        self.pcm_f32_scratch = pcm_f32;
        result
    }

    pub fn encode_f32(&mut self, pcm: &[f32], frame_size: usize) -> Result<Vec<u8>> {
        if self.sample_rate != 48_000 {
            return Err(Error::Unimplemented);
        }
        if pcm.len() < frame_size * self.channels {
            return Err(Error::BadArg);
        }
        self.analysis_info = self.analysis.run(pcm, frame_size, self.channels);
        let mut filtered = std::mem::take(&mut self.filtered_scratch);
        self.dc_reject_frame_into(pcm, frame_size, &mut filtered);
        let frame_bytes = if self.vbr {
            self.vbr_initial_frame_bytes(frame_size)
        } else {
            self.frame_bytes_for_bitrate(frame_size)
        };
        let result =
            self.encode_filtered_f32_with_frame_bytes(&mut filtered, frame_size, frame_bytes, true);
        self.filtered_scratch = filtered;
        result
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
        let mut filtered = std::mem::take(&mut self.filtered_scratch);
        self.dc_reject_frame_into(pcm, frame_size, &mut filtered);
        let result = self.encode_filtered_f32_with_frame_bytes(
            &mut filtered,
            frame_size,
            frame_bytes,
            false,
        );
        self.filtered_scratch = filtered;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::{Encoder, ToneAnalysis};

    #[test]
    fn low_frequency_tone_clears_the_transient_metric() {
        let mut input = vec![vec![0.0; 1_080]];
        input[0][540] = 1.0;
        let mut scratch = Vec::new();

        let without_tone = Encoder::transient_analysis(
            &input,
            1,
            input[0].len(),
            ToneAnalysis::default(),
            &mut scratch,
        );
        assert!(without_tone.tf_estimate > 0.0);

        let low_tone = Encoder::transient_analysis(
            &input,
            1,
            input[0].len(),
            ToneAnalysis {
                frequency: 0.01,
                toneishness: 0.99,
            },
            &mut scratch,
        );
        assert!(!low_tone.is_transient);
        assert_eq!(low_tone.tf_estimate, 0.0);
    }
}
