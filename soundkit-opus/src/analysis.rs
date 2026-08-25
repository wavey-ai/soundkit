use crate::celt::kiss_fft::{opus_fft, KissFftCpx, KissFftState};
use crate::celt::mathops::fast_atan2f;

const NB_FRAMES: usize = 8;
const NB_TBANDS: usize = 18;
const LEAK_BANDS: usize = 19;
const ANALYSIS_BUF_SIZE: usize = 720;
const DETECT_SIZE: usize = 100;
const CELT_SIG_SCALE: f32 = 32768.0;
const INV_CELT_SIG_SCALE_SQUARED: f32 = 1.0 / (CELT_SIG_SCALE * CELT_SIG_SCALE);
const ANALYSIS_COUNT_MAX: usize = 10_000;
const NB_TONAL_SKIP_BANDS: usize = 9;
const ANALYSIS_LOG2_E: f32 = 1.442_695;
const ANALYSIS_PI: f64 = 3.141_592_653;
const ANALYSIS_INV_2PI: f32 = (0.5 / ANALYSIS_PI) as f32;
const ANALYSIS_PI4: f32 = (ANALYSIS_PI * ANALYSIS_PI * ANALYSIS_PI * ANALYSIS_PI) as f32;
const LEAKAGE_OFFSET: f32 = 2.5;
const LEAKAGE_SLOPE: f32 = 2.0;
const TBANDS: [usize; NB_TBANDS + 1] = [
    4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 136, 160, 192, 240,
];

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct AnalysisInfo {
    pub valid: bool,
    pub tonality: f32,
    pub tonality_slope: f32,
    pub activity: f32,
    pub max_pitch_ratio: f32,
    pub bandwidth: usize,
    pub leak_boost: [u8; LEAK_BANDS],
}

#[derive(Clone, Debug, Default)]
struct TonalityAnalysisScratch {
    downmix_tmp: Vec<f32>,
    resampled: Vec<f32>,
    fft_input: Vec<KissFftCpx>,
    fft_output: Vec<KissFftCpx>,
}

#[derive(Clone, Debug)]
pub(crate) struct TonalityAnalysisState {
    angle: [f32; 240],
    d_angle: [f32; 240],
    d2_angle: [f32; 240],
    inmem: [f32; ANALYSIS_BUF_SIZE],
    mem_fill: usize,
    prev_band_tonality: [f32; NB_TBANDS],
    prev_tonality: f32,
    low_e: [f32; NB_TBANDS],
    high_e: [f32; NB_TBANDS],
    mean_e: [f32; NB_TBANDS + 1],
    prev_bandwidth: usize,
    energy: [[f32; NB_TBANDS]; NB_FRAMES],
    energy_count: usize,
    count: usize,
    analysis_offset: usize,
    write_pos: usize,
    read_pos: usize,
    read_subframe: usize,
    hp_ener_accum: f32,
    initialized: bool,
    downmix_state: [f32; 3],
    info: [AnalysisInfo; DETECT_SIZE],
    fft: KissFftState,
    scratch: TonalityAnalysisScratch,
}

impl TonalityAnalysisState {
    pub(crate) fn new() -> Self {
        Self {
            angle: [0.0; 240],
            d_angle: [0.0; 240],
            d2_angle: [0.0; 240],
            inmem: [0.0; ANALYSIS_BUF_SIZE],
            mem_fill: 0,
            prev_band_tonality: [0.0; NB_TBANDS],
            prev_tonality: 0.0,
            low_e: [0.0; NB_TBANDS],
            high_e: [0.0; NB_TBANDS],
            mean_e: [0.0; NB_TBANDS + 1],
            prev_bandwidth: 0,
            energy: [[0.0; NB_TBANDS]; NB_FRAMES],
            energy_count: 0,
            count: 0,
            analysis_offset: 0,
            write_pos: 0,
            read_pos: 0,
            read_subframe: 0,
            hp_ener_accum: 0.0,
            initialized: false,
            downmix_state: [0.0; 3],
            info: [AnalysisInfo::default(); DETECT_SIZE],
            fft: KissFftState::new(480).expect("480-point analysis FFT is supported"),
            scratch: TonalityAnalysisScratch::default(),
        }
    }

    pub(crate) fn run(&mut self, pcm: &[f32], frame_size: usize, channels: usize) -> AnalysisInfo {
        let analysis_frame_size = frame_size & !1;
        if analysis_frame_size > self.analysis_offset {
            let mut pcm_len = analysis_frame_size - self.analysis_offset;
            let mut offset = self.analysis_offset;
            while pcm_len > 0 {
                let chunk = 960.min(pcm_len);
                self.tonality_analysis(pcm, chunk, offset, channels);
                offset += 960;
                pcm_len = pcm_len.saturating_sub(960);
            }
            self.analysis_offset = analysis_frame_size.saturating_sub(frame_size);
        }

        self.get_info(frame_size)
    }

    fn analysis_window(i: usize) -> f32 {
        let phase = 0.5 * core::f32::consts::PI * (i as f32 + 1.0) / 240.0;
        let s = phase.sin();
        s * s
    }

    fn half_log2_energy(energy: f32) -> f32 {
        0.5 * ANALYSIS_LOG2_E * ((energy + 1e-10_f32) as f64).ln() as f32
    }

    fn float2int(value: f32) -> f32 {
        value.round_ties_even()
    }

    fn downmix_and_resample(
        &mut self,
        pcm: &[f32],
        out: &mut [f32],
        subframe_24k: usize,
        offset_24k: usize,
        channels: usize,
        tmp: &mut Vec<f32>,
    ) -> f32 {
        if subframe_24k == 0 {
            return 0.0;
        }

        let subframe = subframe_24k * 2;
        let offset = offset_24k * 2;
        tmp.resize(subframe, 0.0);
        for j in 0..subframe {
            let frame = offset + j;
            let mut sample = pcm[frame * channels];
            for c in 1..channels {
                sample += pcm[frame * channels + c];
            }
            if channels == 2 {
                sample *= 0.5;
            }
            tmp[j] = sample * CELT_SIG_SCALE;
        }

        let mut hp_ener = 0.0f32;
        for k in 0..subframe_24k {
            let in32 = tmp[2 * k];
            let y = in32 - self.downmix_state[0];
            let x = 0.607_437_1 * y;
            let mut out32 = self.downmix_state[0] + x;
            self.downmix_state[0] = in32 + x;
            let mut out32_hp = out32;

            let in32 = tmp[2 * k + 1];
            let y = in32 - self.downmix_state[1];
            let x = 0.150_63 * y;
            out32 += self.downmix_state[1] + x;
            self.downmix_state[1] = in32 + x;

            let y = -in32 - self.downmix_state[2];
            let x = 0.150_63 * y;
            out32_hp += self.downmix_state[2] + x;
            self.downmix_state[2] = -in32 + x;

            hp_ener += out32_hp * out32_hp;
            out[k] = 0.5 * out32;
        }

        hp_ener * INV_CELT_SIG_SCALE_SQUARED
    }

    fn tonality_analysis(
        &mut self,
        pcm: &[f32],
        len_48k: usize,
        offset_48k: usize,
        channels: usize,
    ) {
        let mut scratch = std::mem::take(&mut self.scratch);
        self.tonality_analysis_with_scratch(pcm, len_48k, offset_48k, channels, &mut scratch);
        self.scratch = scratch;
    }

    fn tonality_analysis_with_scratch(
        &mut self,
        pcm: &[f32],
        len_48k: usize,
        offset_48k: usize,
        channels: usize,
        scratch: &mut TonalityAnalysisScratch,
    ) {
        if !self.initialized {
            self.mem_fill = 240;
            self.initialized = true;
        }

        let len = len_48k / 2;
        let offset = offset_48k / 2;
        let writable = len.min(ANALYSIS_BUF_SIZE - self.mem_fill);
        scratch.resampled.resize(writable, 0.0);
        let hp_ener_first = self.downmix_and_resample(
            pcm,
            &mut scratch.resampled,
            writable,
            offset,
            channels,
            &mut scratch.downmix_tmp,
        );
        self.inmem[self.mem_fill..self.mem_fill + writable].copy_from_slice(&scratch.resampled);
        self.hp_ener_accum += hp_ener_first;
        if self.mem_fill + len < ANALYSIS_BUF_SIZE {
            self.mem_fill += len;
            return;
        }

        let hp_ener = self.hp_ener_accum;
        let write_pos = self.write_pos;
        self.write_pos = (self.write_pos + 1) % DETECT_SIZE;

        if self.inmem.iter().all(|sample| *sample == 0.0) {
            let prev_pos = (self.write_pos + DETECT_SIZE - 2) % DETECT_SIZE;
            self.info[write_pos] = self.info[prev_pos];
            return;
        }

        scratch.fft_input.resize(480, KissFftCpx::default());
        scratch.fft_input[..480].fill(KissFftCpx::default());
        scratch.fft_output.resize(480, KissFftCpx::default());
        scratch.fft_output[..480].fill(KissFftCpx::default());
        for i in 0..240 {
            let w = Self::analysis_window(i);
            scratch.fft_input[i].r = w * self.inmem[i];
            scratch.fft_input[i].i = w * self.inmem[240 + i];
            scratch.fft_input[480 - i - 1].r = w * self.inmem[480 - i - 1];
            scratch.fft_input[480 - i - 1].i = w * self.inmem[720 - i - 1];
        }

        self.inmem
            .copy_within(ANALYSIS_BUF_SIZE - 240..ANALYSIS_BUF_SIZE, 0);
        let remaining = len - (ANALYSIS_BUF_SIZE - self.mem_fill);
        scratch.resampled.resize(remaining, 0.0);
        self.hp_ener_accum = self.downmix_and_resample(
            pcm,
            &mut scratch.resampled,
            remaining,
            offset + writable,
            channels,
            &mut scratch.downmix_tmp,
        );
        self.inmem[240..240 + remaining].copy_from_slice(&scratch.resampled);
        self.mem_fill = 240 + remaining;

        opus_fft(&self.fft, &scratch.fft_input, &mut scratch.fft_output);

        let mut tonality = [0.0f32; 240];
        let mut tonality2 = [0.0f32; 240];
        let mut noisiness = [0.0f32; 240];
        for i in 1..240 {
            let x1r = scratch.fft_output[i].r + scratch.fft_output[480 - i].r;
            let x1i = scratch.fft_output[i].i - scratch.fft_output[480 - i].i;
            let x2r = scratch.fft_output[i].i + scratch.fft_output[480 - i].i;
            let x2i = scratch.fft_output[480 - i].r - scratch.fft_output[i].r;

            let angle = ANALYSIS_INV_2PI * fast_atan2f(x1i, x1r);
            let d_angle = angle - self.angle[i];
            let d2_angle = d_angle - self.d_angle[i];

            let angle2 = ANALYSIS_INV_2PI * fast_atan2f(x2i, x2r);
            let d_angle2 = angle2 - angle;
            let d2_angle2 = d_angle2 - d_angle;

            let mut mod1 = d2_angle - Self::float2int(d2_angle);
            let noisiness1 = mod1.abs();
            mod1 *= mod1;
            mod1 *= mod1;

            let mut mod2 = d2_angle2 - Self::float2int(d2_angle2);
            noisiness[i] = noisiness1 + mod2.abs();
            mod2 *= mod2;
            mod2 *= mod2;

            let avg_mod = 0.25 * (self.d2_angle[i] + mod1 + 2.0 * mod2);
            tonality[i] = 1.0 / (1.0 + 40.0 * 16.0 * ANALYSIS_PI4 * avg_mod) - 0.015;
            tonality2[i] = 1.0 / (1.0 + 40.0 * 16.0 * ANALYSIS_PI4 * mod2) - 0.015;

            self.angle[i] = angle2;
            self.d_angle[i] = d_angle2;
            self.d2_angle[i] = mod2;
        }

        for i in 2..239 {
            let tt = tonality2[i].min(tonality2[i - 1].max(tonality2[i + 1]));
            tonality[i] = 0.9 * tonality[i].max(tt - 0.1);
        }

        let alpha_e2 = if self.count <= 1 {
            1.0
        } else {
            1.0 / (1 + self.count).min(100) as f32
        };
        let mut bandwidth_mask = 0.0f32;
        let mut bandwidth = 0usize;
        let mut max_e = 0.0f32;
        let mut is_masked = [false; NB_TBANDS + 1];
        let mut noise_floor = 5.7e-4f32 / ((1u32 << 16) as f32);
        noise_floor *= noise_floor;
        let mut band_log2 = [0.0f32; NB_TBANDS + 1];
        let mut leakage_from = [0.0f32; NB_TBANDS + 1];
        let mut leakage_to = [0.0f32; NB_TBANDS + 1];
        let mut first_band_energy =
            (2.0 * scratch.fft_output[0].r).powi(2) + (2.0 * scratch.fft_output[0].i).powi(2);
        for i in 1..4 {
            first_band_energy += scratch.fft_output[i].r * scratch.fft_output[i].r
                + scratch.fft_output[480 - i].r * scratch.fft_output[480 - i].r
                + scratch.fft_output[i].i * scratch.fft_output[i].i
                + scratch.fft_output[480 - i].i * scratch.fft_output[480 - i].i;
        }
        band_log2[0] = Self::half_log2_energy(first_band_energy * INV_CELT_SIG_SCALE_SQUARED);

        let mut slope = 0.0f32;
        let mut frame_tonality = 0.0f32;
        let mut frame_noisiness = 0.0f32;
        let mut relative_e = 0.0f32;
        let mut below_max_pitch = 0.0f32;
        let mut above_max_pitch = 0.0f32;
        let mut max_frame_tonality = 0.0f32;
        let mut band_tonality_values = [0.0f32; NB_TBANDS];
        for b in 0..NB_TBANDS {
            let mut energy = 0.0f32;
            let mut tonal_energy = 0.0f32;
            let mut noise_energy = 0.0f32;
            for i in TBANDS[b]..TBANDS[b + 1] {
                let raw_bin_energy = scratch.fft_output[i].r * scratch.fft_output[i].r
                    + scratch.fft_output[480 - i].r * scratch.fft_output[480 - i].r
                    + scratch.fft_output[i].i * scratch.fft_output[i].i
                    + scratch.fft_output[480 - i].i * scratch.fft_output[480 - i].i;
                let bin_energy = raw_bin_energy * INV_CELT_SIG_SCALE_SQUARED;
                energy += bin_energy;
                tonal_energy += bin_energy * tonality[i].max(0.0);
                noise_energy += bin_energy * 2.0 * (0.5 - noisiness[i]);
            }
            let log_energy = (energy + 1e-10).ln();
            band_log2[b + 1] = 0.5 * ANALYSIS_LOG2_E * log_energy;
            if TBANDS[b] < 64 {
                below_max_pitch += energy;
            } else {
                above_max_pitch += energy;
            }

            self.energy[self.energy_count][b] = energy;
            frame_noisiness += noise_energy / (1e-15 + energy);
            if self.count == 0 {
                self.high_e[b] = log_energy;
                self.low_e[b] = log_energy;
            }
            if self.high_e[b] > self.low_e[b] + 7.5 {
                if self.high_e[b] - log_energy > log_energy - self.low_e[b] {
                    self.high_e[b] -= 0.01;
                } else {
                    self.low_e[b] += 0.01;
                }
            }
            if log_energy > self.high_e[b] {
                self.high_e[b] = log_energy;
                self.low_e[b] = self.low_e[b].max(self.high_e[b] - 15.0);
            } else if log_energy < self.low_e[b] {
                self.low_e[b] = log_energy;
                self.high_e[b] = self.high_e[b].min(self.low_e[b] + 15.0);
            }
            relative_e += (log_energy - self.low_e[b]) / (1e-5 + self.high_e[b] - self.low_e[b]);

            let mut l1 = 0.0f32;
            let mut l2 = 0.0f32;
            for frame in 0..NB_FRAMES {
                l1 += self.energy[frame][b].sqrt();
                l2 += self.energy[frame][b];
            }
            let mut stationarity = (l1 / (1e-15 + NB_FRAMES as f32 * l2).sqrt()).min(0.99);
            stationarity *= stationarity;
            stationarity *= stationarity;
            let band_tonality =
                (tonal_energy / (1e-15 + energy)).max(stationarity * self.prev_band_tonality[b]);
            band_tonality_values[b] = band_tonality;
            frame_tonality += band_tonality;
            if b >= NB_TBANDS - NB_TONAL_SKIP_BANDS {
                frame_tonality -= band_tonality_values[b + NB_TONAL_SKIP_BANDS - NB_TBANDS];
            }
            max_frame_tonality = max_frame_tonality
                .max((1.0 + 0.03 * (b as f32 - NB_TBANDS as f32)) * frame_tonality);
            slope += band_tonality * (b as f32 - 8.0);
            self.prev_band_tonality[b] = band_tonality;

            max_e = max_e.max(energy);
            self.mean_e[b] = ((1.0 - alpha_e2) * self.mean_e[b]).max(energy);
            let em = energy.max(self.mean_e[b]);
            let band_width = (TBANDS[b + 1] - TBANDS[b]) as f32;
            if energy * 1e9 > max_e
                && (em > 3.0 * noise_floor * band_width || energy > noise_floor * band_width)
            {
                bandwidth = b + 1;
            }
            let mask_threshold = if self.prev_bandwidth >= b + 1 {
                0.01
            } else {
                0.05
            };
            is_masked[b] = energy < mask_threshold * bandwidth_mask;
            bandwidth_mask = (0.05 * bandwidth_mask).max(energy);
        }

        leakage_from[0] = band_log2[0];
        leakage_to[0] = band_log2[0] - LEAKAGE_OFFSET;
        for b in 1..=NB_TBANDS {
            let leak_slope = LEAKAGE_SLOPE * (TBANDS[b] - TBANDS[b - 1]) as f32 / 4.0;
            leakage_from[b] = (leakage_from[b - 1] + leak_slope).min(band_log2[b]);
            leakage_to[b] = (leakage_to[b - 1] - leak_slope).max(band_log2[b] - LEAKAGE_OFFSET);
        }
        for b in (0..NB_TBANDS - 1).rev() {
            let leak_slope = LEAKAGE_SLOPE * (TBANDS[b + 1] - TBANDS[b]) as f32 / 4.0;
            leakage_from[b] = (leakage_from[b + 1] + leak_slope).min(leakage_from[b]);
            leakage_to[b] = (leakage_to[b + 1] - leak_slope).max(leakage_to[b]);
        }
        let mut leak_boost = [0u8; LEAK_BANDS];
        for b in 0..=NB_TBANDS {
            let boost = (leakage_to[b] - band_log2[b]).max(0.0)
                + (band_log2[b] - (leakage_from[b] + LEAKAGE_OFFSET)).max(0.0);
            leak_boost[b] = (0.5 + 64.0 * boost).floor().min(255.0) as u8;
        }

        let hp_band_energy = hp_ener * (1.0 / (60.0 * 60.0));
        above_max_pitch += hp_band_energy;
        let noise_ratio = if self.prev_bandwidth == 20 {
            10.0
        } else {
            30.0
        };
        self.mean_e[NB_TBANDS] = ((1.0 - alpha_e2) * self.mean_e[NB_TBANDS]).max(hp_band_energy);
        let hp_em = hp_band_energy.max(self.mean_e[NB_TBANDS]);
        if hp_em > 3.0 * noise_ratio * noise_floor * 160.0
            || hp_band_energy > noise_ratio * noise_floor * 160.0
        {
            bandwidth = 20;
        }
        let mask_threshold = if self.prev_bandwidth == 20 {
            0.01
        } else {
            0.05
        };
        is_masked[NB_TBANDS] = hp_band_energy < mask_threshold * bandwidth_mask;

        if bandwidth == 20 && is_masked[NB_TBANDS] {
            bandwidth -= 2;
        } else if bandwidth > 0 && bandwidth <= NB_TBANDS && is_masked[bandwidth - 1] {
            bandwidth -= 1;
        }
        if self.count <= 2 {
            bandwidth = 20;
        }

        slope /= 64.0;
        let max_pitch_ratio = if above_max_pitch > below_max_pitch {
            below_max_pitch / above_max_pitch
        } else {
            1.0
        };
        frame_noisiness /= NB_TBANDS as f32;
        relative_e /= NB_TBANDS as f32;
        if self.count < 10 {
            relative_e = 0.5;
        }
        let activity = frame_noisiness + (1.0 - frame_noisiness) * relative_e;
        let mut frame_tonality = max_frame_tonality / (NB_TBANDS - NB_TONAL_SKIP_BANDS) as f32;
        frame_tonality = frame_tonality.max(self.prev_tonality * 0.8);
        self.prev_tonality = frame_tonality;
        self.energy_count = (self.energy_count + 1) % NB_FRAMES;
        self.count = (self.count + 1).min(ANALYSIS_COUNT_MAX);
        self.info[write_pos] = AnalysisInfo {
            valid: true,
            tonality: frame_tonality,
            tonality_slope: slope,
            activity,
            max_pitch_ratio,
            bandwidth,
            leak_boost,
        };
        self.prev_bandwidth = bandwidth;
    }

    fn get_info(&mut self, frame_size: usize) -> AnalysisInfo {
        let mut pos = self.read_pos;
        self.read_subframe += frame_size / 120;
        while self.read_subframe >= 8 {
            self.read_subframe -= 8;
            self.read_pos += 1;
        }
        if self.read_pos >= DETECT_SIZE {
            self.read_pos -= DETECT_SIZE;
        }

        if frame_size > 960 && pos != self.write_pos {
            pos += 1;
            if pos == DETECT_SIZE {
                pos = 0;
            }
        }
        if pos == self.write_pos {
            pos = if pos == 0 { DETECT_SIZE - 1 } else { pos - 1 };
        }
        let mut info = self.info[pos];
        if !info.valid {
            return info;
        }

        let pos0 = pos;
        let mut tonality_max = info.tonality;
        let mut tonality_avg = info.tonality;
        let mut tonality_count = 1usize;
        let mut bandwidth_span = 6usize;
        for _ in 0..3 {
            pos += 1;
            if pos == DETECT_SIZE {
                pos = 0;
            }
            if pos == self.write_pos {
                break;
            }
            tonality_max = tonality_max.max(self.info[pos].tonality);
            tonality_avg += self.info[pos].tonality;
            tonality_count += 1;
            info.bandwidth = info.bandwidth.max(self.info[pos].bandwidth);
            bandwidth_span -= 1;
        }

        pos = pos0;
        for _ in 0..bandwidth_span {
            pos = if pos == 0 { DETECT_SIZE - 1 } else { pos - 1 };
            if pos == self.write_pos {
                break;
            }
            info.bandwidth = info.bandwidth.max(self.info[pos].bandwidth);
        }
        info.tonality = (tonality_avg / tonality_count as f32).max(tonality_max - 0.2);

        info
    }
}
