use crate::celt::bands::{compute_band_energies, normalise_bands, SPREAD_NORMAL};
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

    pub fn set_bitrate(&mut self, bitrate: i32) -> Result<()> {
        if !(CELT_MIN_BITRATE..=CELT_MAX_BITRATE).contains(&bitrate) {
            return Err(Error::BadArg);
        }
        self.bitrate = bitrate;
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
        let bytes = (self.bitrate as i64 * frame_size as i64 + self.sample_rate as i64 * 4)
            / (self.sample_rate as i64 * 8);
        (bytes as usize).clamp(CELT_MIN_FRAME_BYTES, CELT_MAX_FRAME_BYTES)
    }

    fn validate_frame_bytes(frame_bytes: usize) -> Result<()> {
        if !(CELT_MIN_FRAME_BYTES..=CELT_MAX_FRAME_BYTES).contains(&frame_bytes) {
            return Err(Error::BadArg);
        }
        Ok(())
    }

    pub fn encode_i16(&mut self, pcm: &[i16], frame_size: usize) -> Result<Vec<u8>> {
        let frame_bytes = self.frame_bytes_for_bitrate(frame_size);
        self.encode_i16_with_frame_bytes(pcm, frame_size, frame_bytes)
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
        let frame_bytes = self.frame_bytes_for_bitrate(frame_size);
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
        let mut config = CeltFrameConfig::new(&self.mode, lm, self.channels, frame_bytes)?;
        config.spread = SPREAD_NORMAL;
        config.alloc_trim = 5;

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

        let eff_end = self.mode.eff_ebands;
        let mut band_e = vec![0.0f32; self.channels * self.mode.nb_ebands];
        compute_band_energies(&self.mode, &freq, &mut band_e, eff_end, self.channels, lm);
        let mut norm = vec![0.0f32; self.channels * n];
        normalise_bands(
            &self.mode,
            &freq,
            &mut norm,
            &band_e,
            eff_end,
            self.channels,
            m,
        );

        let encoded = if self.channels == 1 {
            encode_spectral_frame(
                &self.mode,
                &config,
                &mut norm,
                None,
                &band_e,
                &mut self.old_band_e,
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
                &mut self.seed,
            )?
        };

        let mut packet = Vec::with_capacity(1 + encoded.data.len());
        packet.push(packet::make_celt_only_fullband_toc(lm, self.channels)?);
        packet.extend_from_slice(&encoded.data);
        Ok(packet)
    }
}
