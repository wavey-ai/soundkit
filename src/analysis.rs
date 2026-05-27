use crate::celt::kiss_fft::{opus_fft, KissFftCpx, KissFftState};
use crate::celt::mathops::fast_atan2f;

const NB_FRAMES: usize = 8;
const NB_TBANDS: usize = 18;
const ANALYSIS_BUF_SIZE: usize = 720;
const DETECT_SIZE: usize = 100;
const CELT_SIG_SCALE: f32 = 32768.0;
const ANALYSIS_COUNT_MAX: usize = 10_000;
const TBANDS: [usize; NB_TBANDS + 1] = [
    4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 136, 160, 192, 240,
];

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct AnalysisInfo {
    pub valid: bool,
    pub tonality_slope: f32,
    pub bandwidth: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct TonalityAnalysisState {
    angle: [f32; 240],
    d_angle: [f32; 240],
    d2_angle: [f32; 240],
    inmem: [f32; ANALYSIS_BUF_SIZE],
    mem_fill: usize,
    prev_band_tonality: [f32; NB_TBANDS],
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

    fn downmix_and_resample(
        &mut self,
        pcm: &[f32],
        out: &mut [f32],
        subframe_24k: usize,
        offset_24k: usize,
        channels: usize,
    ) -> f32 {
        if subframe_24k == 0 {
            return 0.0;
        }

        let subframe = subframe_24k * 2;
        let offset = offset_24k * 2;
        let mut tmp = vec![0.0f32; subframe];
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

        hp_ener * (1.0 / (CELT_SIG_SCALE * CELT_SIG_SCALE))
    }

    fn tonality_analysis(
        &mut self,
        pcm: &[f32],
        len_48k: usize,
        offset_48k: usize,
        channels: usize,
    ) {
        if !self.initialized {
            self.mem_fill = 240;
            self.initialized = true;
        }

        let len = len_48k / 2;
        let offset = offset_48k / 2;
        let writable = len.min(ANALYSIS_BUF_SIZE - self.mem_fill);
        let mut first = vec![0.0f32; writable];
        let hp_ener_first = self.downmix_and_resample(pcm, &mut first, writable, offset, channels);
        self.inmem[self.mem_fill..self.mem_fill + writable].copy_from_slice(&first);
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

        let mut input = vec![KissFftCpx::default(); 480];
        let mut output = vec![KissFftCpx::default(); 480];
        for i in 0..240 {
            let w = Self::analysis_window(i);
            input[i].r = w * self.inmem[i];
            input[i].i = w * self.inmem[240 + i];
            input[480 - i - 1].r = w * self.inmem[480 - i - 1];
            input[480 - i - 1].i = w * self.inmem[720 - i - 1];
        }

        self.inmem
            .copy_within(ANALYSIS_BUF_SIZE - 240..ANALYSIS_BUF_SIZE, 0);
        let remaining = len - (ANALYSIS_BUF_SIZE - self.mem_fill);
        let mut rest = vec![0.0f32; remaining];
        self.hp_ener_accum =
            self.downmix_and_resample(pcm, &mut rest, remaining, offset + writable, channels);
        self.inmem[240..240 + remaining].copy_from_slice(&rest);
        self.mem_fill = 240 + remaining;

        opus_fft(&self.fft, &input, &mut output);

        let mut tonality = vec![0.0f32; 240];
        let mut tonality2 = vec![0.0f32; 240];
        let pi4 = core::f32::consts::PI.powi(4);
        for i in 1..240 {
            let x1r = output[i].r + output[480 - i].r;
            let x1i = output[i].i - output[480 - i].i;
            let x2r = output[i].i + output[480 - i].i;
            let x2i = output[480 - i].r - output[i].r;

            let angle = (0.5 / core::f32::consts::PI) * fast_atan2f(x1i, x1r);
            let d_angle = angle - self.angle[i];
            let d2_angle = d_angle - self.d_angle[i];

            let angle2 = (0.5 / core::f32::consts::PI) * fast_atan2f(x2i, x2r);
            let d_angle2 = angle2 - angle;
            let d2_angle2 = d_angle2 - d_angle;

            let mut mod1 = d2_angle - d2_angle.round();
            let noisiness1 = mod1.abs();
            mod1 *= mod1;
            mod1 *= mod1;

            let mut mod2 = d2_angle2 - d2_angle2.round();
            let _noisiness = noisiness1 + mod2.abs();
            mod2 *= mod2;
            mod2 *= mod2;

            let avg_mod = 0.25 * (self.d2_angle[i] + mod1 + 2.0 * mod2);
            tonality[i] = 1.0 / (1.0 + 40.0 * 16.0 * pi4 * avg_mod) - 0.015;
            tonality2[i] = 1.0 / (1.0 + 40.0 * 16.0 * pi4 * mod2) - 0.015;

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

        let mut slope = 0.0f32;
        for b in 0..NB_TBANDS {
            let mut energy = 0.0f32;
            let mut tonal_energy = 0.0f32;
            for i in TBANDS[b]..TBANDS[b + 1] {
                let bin_energy = output[i].r * output[i].r
                    + output[480 - i].r * output[480 - i].r
                    + output[i].i * output[i].i
                    + output[480 - i].i * output[480 - i].i;
                let bin_energy = bin_energy * (1.0 / (CELT_SIG_SCALE * CELT_SIG_SCALE));
                energy += bin_energy;
                tonal_energy += bin_energy * tonality[i].max(0.0);
            }

            self.energy[self.energy_count][b] = energy;
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

        let hp_band_energy = hp_ener * (1.0 / (60.0 * 60.0));
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
        self.energy_count = (self.energy_count + 1) % NB_FRAMES;
        self.count = (self.count + 1).min(ANALYSIS_COUNT_MAX);
        self.info[write_pos] = AnalysisInfo {
            valid: true,
            tonality_slope: slope,
            bandwidth,
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
        let mut bandwidth_span = 6usize;
        for _ in 0..3 {
            pos += 1;
            if pos == DETECT_SIZE {
                pos = 0;
            }
            if pos == self.write_pos {
                break;
            }
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

        info
    }
}
