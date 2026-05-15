use crate::celt::bands::{
    compute_band_energies, hysteresis_decision, normalise_bands, SPREAD_NORMAL,
};
use crate::celt::codec::{encode_spectral_frame, CeltFrameConfig};
use crate::celt::mdct::clt_mdct_forward;
use crate::celt::modes::CeltMode;
use crate::constants::{valid_channels, valid_sample_rate};
use crate::packet;
use crate::{Error, Result};

pub const CELT_FRAME_SIZES_48K: [usize; 4] = [120, 240, 480, 960];
pub const CELT_MIN_BITRATE: i32 = 500;
pub const CELT_MAX_BITRATE: i32 = 512_000;
pub const CELT_MIN_FRAME_BYTES: usize = 2;
pub const CELT_MAX_FRAME_BYTES: usize = packet::MAX_FRAME_BYTES as usize;

const INTENSITY_THRESHOLDS: [f32; 21] = [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 16.0, 24.0, 36.0, 44.0, 50.0, 56.0, 62.0, 67.0, 72.0,
    79.0, 88.0, 106.0, 134.0,
];
const INTENSITY_HYSTERESIS: [f32; 21] = [
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0,
    8.0, 8.0,
];

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
    preemph_mem: Vec<f32>,
    overlap_mem: Vec<Vec<f32>>,
    seed: u32,
    intensity: usize,
    delayed_intra: f32,
    stream_channels: usize,
    vbr: bool,
    vbr_reservoir: f32,
    vbr_prev_energy: f32,
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
            preemph_mem: vec![0.0; channels],
            overlap_mem: vec![vec![0.0; mode.overlap]; channels],
            seed: 0,
            intensity: 0,
            delayed_intra: 1.0,
            stream_channels: channels,
            vbr: false,
            vbr_reservoir: 0.0,
            vbr_prev_energy: 0.0,
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
        let target = self.frame_bytes_for_bitrate(frame_size) as f32;
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
        let desired = target * (0.70 + 0.75 * complexity) + self.vbr_reservoir * 0.10;
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

    fn equiv_rate(&self, frame_bytes: usize, lm: usize) -> i32 {
        let overhead = (40 * self.channels as i32 + 20) * ((400 >> lm) - 50);
        let rate = ((frame_bytes as i32 * 8 * 50) << (3 - lm)) - overhead;
        rate.min(self.bitrate - overhead)
    }

    fn alloc_trim_for_rate(equiv_rate: i32) -> i32 {
        if equiv_rate < 64_000 {
            4
        } else {
            5
        }
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

    fn opus_equiv_rate_for_packet(&self, frame_size: usize, frame_bytes: usize) -> i32 {
        let packet_bitrate =
            (((frame_bytes + 1) as i64 * 8 * self.sample_rate as i64) / frame_size as i64) as i32;
        let frame_rate = self.sample_rate / frame_size as i32;
        let mut equiv = packet_bitrate;
        if frame_rate > 50 {
            equiv -= (40 * self.channels as i32 + 20) * (frame_rate - 50);
        }
        equiv -= equiv / 12;
        equiv * 95 / 100
    }

    fn choose_stream_channels(&mut self, frame_size: usize, frame_bytes: usize) -> usize {
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
        self.stream_channels =
            if self.opus_equiv_rate_for_packet(frame_size, frame_bytes) > threshold {
                2
            } else {
                1
            };
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
        if pcm.len() < frame_size * self.channels {
            return Err(Error::BadArg);
        }
        let frame_bytes = if self.vbr {
            self.vbr_frame_bytes(pcm, frame_size)
        } else {
            self.frame_bytes_for_bitrate(frame_size)
        };
        self.encode_f32_with_frame_bytes(pcm, frame_size, frame_bytes)
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
        let lm = self.frame_lm(frame_size)?;
        Self::validate_frame_bytes(frame_bytes)?;
        let stream_channels = self.choose_stream_channels(frame_size, frame_bytes);
        let mut config = CeltFrameConfig::new(&self.mode, lm, stream_channels, frame_bytes)?;
        config.spread = SPREAD_NORMAL;
        let equiv_rate = self.equiv_rate(frame_bytes, lm);
        config.alloc_trim = Self::alloc_trim_for_rate(equiv_rate);

        let n = frame_size;
        let m = 1usize << lm;
        let shift = self.mode.max_lm - lm;
        let overlap = self.mode.overlap;

        let mut inputs = Vec::with_capacity(self.channels);
        for c in 0..self.channels {
            let mut input = vec![0.0f32; 2 * n];
            input[..overlap].copy_from_slice(&self.overlap_mem[c]);
            for i in 0..n {
                let sample = pcm[i * self.channels + c];
                input[overlap + i] = sample - self.preemph_mem[c];
                self.preemph_mem[c] = self.mode.preemph[0] * sample;
            }
            self.overlap_mem[c].copy_from_slice(&input[n..n + overlap]);
            inputs.push(input);
        }

        let mut freq = vec![0.0f32; self.channels * n];
        for (c, input) in inputs.iter().enumerate() {
            clt_mdct_forward(
                &self.mode.mdct,
                input,
                &mut freq[c * n..(c + 1) * n],
                &self.mode.window,
                overlap,
                shift,
                1,
            );
        }
        if self.channels == 2 && stream_channels == 1 {
            for i in 0..n {
                freq[i] = 0.5 * (freq[i] + freq[n + i]);
            }
        }

        let eff_end = self.mode.eff_ebands;
        let mut band_e = vec![0.0f32; stream_channels * self.mode.nb_ebands];
        compute_band_energies(&self.mode, &freq, &mut band_e, eff_end, stream_channels, lm);
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

        let encoded = if stream_channels == 1 {
            encode_spectral_frame(
                &self.mode,
                &config,
                &mut norm,
                None,
                &band_e,
                &mut self.old_band_e[..self.mode.nb_ebands],
                &mut self.delayed_intra,
                &mut self.seed,
            )?
        } else {
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
            encode_spectral_frame(
                &self.mode,
                &config,
                left,
                Some(right),
                &band_e,
                &mut self.old_band_e,
                &mut self.delayed_intra,
                &mut self.seed,
            )?
        };
        if self.channels == 2 && stream_channels == 1 {
            let (left, right) = self.old_band_e.split_at_mut(self.mode.nb_ebands);
            right[..self.mode.nb_ebands].copy_from_slice(left);
        }

        let mut packet = Vec::with_capacity(1 + encoded.data.len());
        packet.push(packet::make_celt_only_fullband_toc(lm, stream_channels)?);
        packet.extend_from_slice(&encoded.data);
        Ok(packet)
    }
}
