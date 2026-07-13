use crate::celt::codec::{
    decode_spectral_frame_into_with_anti_collapse, CeltFrameConfig, CeltFrameDecodeScratch,
};
use crate::celt::modes::CeltMode;
use crate::celt::pitch::{comb_filter_in_place, COMBFILTER_MAXPERIOD, COMBFILTER_MINPERIOD};
use crate::celt::synthesis::{
    celt_synthesis_with_overlap_into, deemphasis_interleaved_i16_into, deemphasis_interleaved_into,
    SynthesisScratch,
};
use crate::constants::{valid_channels, valid_sample_rate, Bandwidth};
use crate::packet;
use crate::{Error, Result};

#[derive(Clone, Debug)]
pub struct Decoder {
    sample_rate: i32,
    channels: usize,
    mode: CeltMode,
    old_band_e: Vec<f32>,
    old_log_e: Vec<f32>,
    old_log_e2: Vec<f32>,
    preemph_mem: Vec<f32>,
    overlap_mem: Vec<Vec<f32>>,
    postfilter_mem: Vec<Vec<f32>>,
    postfilter_work: Vec<Vec<f32>>,
    decode_scratch: CeltFrameDecodeScratch,
    synthesis_channels: Vec<Vec<f32>>,
    synthesis_scratch: SynthesisScratch,
    postfilter_period: usize,
    postfilter_period_old: usize,
    postfilter_gain: f32,
    postfilter_gain_old: f32,
    postfilter_tapset: usize,
    postfilter_tapset_old: usize,
    seed: u32,
}

impl Decoder {
    pub fn new(sample_rate: i32, channels: usize) -> Result<Self> {
        if !valid_sample_rate(sample_rate) || !valid_channels(channels as i32) {
            return Err(Error::BadArg);
        }
        let mode = CeltMode::standard_48k();
        Ok(Self {
            sample_rate,
            channels,
            old_band_e: vec![0.0; channels * mode.nb_ebands],
            old_log_e: vec![-28.0; channels * mode.nb_ebands],
            old_log_e2: vec![-28.0; channels * mode.nb_ebands],
            preemph_mem: vec![0.0; channels],
            overlap_mem: vec![vec![0.0; mode.overlap]; channels],
            postfilter_mem: vec![vec![0.0; COMBFILTER_MAXPERIOD]; channels],
            postfilter_work: vec![Vec::new(); channels],
            decode_scratch: CeltFrameDecodeScratch::default(),
            synthesis_channels: vec![Vec::new(); channels],
            synthesis_scratch: SynthesisScratch::default(),
            postfilter_period: 0,
            postfilter_period_old: 0,
            postfilter_gain: 0.0,
            postfilter_gain_old: 0.0,
            postfilter_tapset: 0,
            postfilter_tapset_old: 0,
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

    pub fn validate_packet(&self, packet: &[u8]) -> Result<usize> {
        let samples = packet::sample_count(packet, self.sample_rate)?;
        Ok(samples as usize)
    }

    pub fn decode_i16(&mut self, packet: &[u8], decode_fec: bool) -> Result<Vec<i16>> {
        let mut pcm = Vec::new();
        self.decode_i16_into(packet, decode_fec, &mut pcm)?;
        Ok(pcm)
    }

    pub fn decode_i16_into(
        &mut self,
        packet: &[u8],
        decode_fec: bool,
        pcm: &mut Vec<i16>,
    ) -> Result<usize> {
        let channels = self.decode_channels(packet, decode_fec)?;
        deemphasis_interleaved_i16_into(
            &self.mode,
            &self.synthesis_channels[..channels],
            &mut self.preemph_mem,
            pcm,
        )?;
        Ok(pcm.len() / channels)
    }

    pub fn decode_f32(&mut self, packet: &[u8], decode_fec: bool) -> Result<Vec<f32>> {
        let mut pcm = Vec::new();
        self.decode_f32_into(packet, decode_fec, &mut pcm)?;
        Ok(pcm)
    }

    pub fn decode_f32_into(
        &mut self,
        packet: &[u8],
        decode_fec: bool,
        pcm: &mut Vec<f32>,
    ) -> Result<usize> {
        let channels = self.decode_channels(packet, decode_fec)?;
        deemphasis_interleaved_into(
            &self.mode,
            &self.synthesis_channels[..channels],
            &mut self.preemph_mem,
            pcm,
        )?;
        Ok(pcm.len() / channels)
    }

    fn decode_channels(&mut self, packet: &[u8], _decode_fec: bool) -> Result<usize> {
        if self.sample_rate != 48_000 {
            return Err(Error::Unimplemented);
        }

        let parsed = packet::parse_packet_slice(packet, false)?;
        let samples_per_frame =
            packet::packet_get_samples_per_frame_byte(parsed.toc, self.sample_rate);
        let packet_samples = samples_per_frame as i64 * parsed.count as i64;
        if packet_samples * 25 > self.sample_rate as i64 * 3 {
            return Err(Error::InvalidPacket);
        }

        let stream_channels = if parsed.toc & 0x4 != 0 { 2 } else { 1 };
        if parsed.count != 1
            || !packet::is_celt_only(parsed.toc)
            || packet::bandwidth(packet)? != Bandwidth::Fullband
            || stream_channels > self.channels
        {
            return Err(Error::Unimplemented);
        }

        let lm = packet::celt_only_lm(parsed.toc)?;
        let expected_frame_size = self.mode.short_mdct_size << lm;
        if samples_per_frame as usize != expected_frame_size {
            return Err(Error::InvalidPacket);
        }

        let frame_offset = parsed.frame_offsets[0];
        let frame_size = parsed.sizes[0] as usize;
        let frame = &packet[frame_offset..frame_offset + frame_size];
        if stream_channels == 1 && self.channels == 2 {
            for i in 0..self.mode.nb_ebands {
                self.old_band_e[i] =
                    self.old_band_e[i].max(self.old_band_e[self.mode.nb_ebands + i]);
            }
        }

        let config = CeltFrameConfig::new(&self.mode, lm, stream_channels, frame.len())?;
        let decoded = decode_spectral_frame_into_with_anti_collapse(
            &self.mode,
            &config,
            frame,
            &mut self.old_band_e,
            &self.old_log_e,
            &self.old_log_e2,
            &mut self.seed,
            &mut self.decode_scratch,
        )?;
        let y = (config.channels == 2).then(|| &self.decode_scratch.y[..decoded.samples]);
        celt_synthesis_with_overlap_into(
            &self.mode,
            &self.decode_scratch.x[..decoded.samples],
            y,
            &self.old_band_e,
            config.start,
            config.end.min(self.mode.eff_ebands),
            config.channels,
            decoded.is_transient,
            config.lm,
            1,
            decoded.silence,
            &mut self.overlap_mem[..stream_channels],
            &mut self.synthesis_channels,
            &mut self.synthesis_scratch,
        )?;
        self.apply_postfilter(stream_channels, config.lm, decoded.prefilter);
        if stream_channels == 1 && self.channels == 2 {
            self.old_band_e
                .copy_within(0..self.mode.nb_ebands, self.mode.nb_ebands);
            {
                let (left, right) = self.overlap_mem.split_at_mut(1);
                right[0].clone_from(&left[0]);
            }
            {
                let (left, right) = self.postfilter_mem.split_at_mut(1);
                right[0].clone_from(&left[0]);
            }
            {
                let (left, right) = self.synthesis_channels.split_at_mut(1);
                right[0].clear();
                right[0].extend_from_slice(&left[0]);
            }
            self.update_energy_history(decoded.is_transient);
            Ok(self.channels)
        } else {
            self.update_energy_history(decoded.is_transient);
            Ok(stream_channels)
        }
    }

    fn update_energy_history(&mut self, is_transient: bool) {
        if is_transient {
            for (history, &energy) in self.old_log_e.iter_mut().zip(&self.old_band_e) {
                *history = history.min(energy);
            }
        } else {
            self.old_log_e2.copy_from_slice(&self.old_log_e);
            self.old_log_e.copy_from_slice(&self.old_band_e);
        }
    }

    fn apply_postfilter(
        &mut self,
        stream_channels: usize,
        lm: usize,
        prefilter: Option<crate::celt::codec::DecodedPrefilter>,
    ) {
        let n = self.mode.short_mdct_size << lm;
        let short = self.mode.short_mdct_size.min(n);
        let postfilter_pitch = prefilter
            .map(|prefilter| prefilter.pitch as usize)
            .unwrap_or(0);
        let postfilter_gain = prefilter
            .map(|prefilter| 0.09375 * (prefilter.qgain + 1) as f32)
            .unwrap_or(0.0);
        let postfilter_tapset = prefilter
            .map(|prefilter| prefilter.tapset as usize)
            .unwrap_or(0);

        self.postfilter_period = self.postfilter_period.max(COMBFILTER_MINPERIOD);
        self.postfilter_period_old = self.postfilter_period_old.max(COMBFILTER_MINPERIOD);
        let filters_active = self.postfilter_gain_old != 0.0
            || self.postfilter_gain != 0.0
            || (lm != 0 && postfilter_gain != 0.0);
        if filters_active {
            for c in 0..stream_channels {
                let needed = COMBFILTER_MAXPERIOD + n;
                if self.postfilter_work[c].len() < needed {
                    self.postfilter_work[c].resize(needed, 0.0);
                }
                let work = &mut self.postfilter_work[c][..needed];
                work[..COMBFILTER_MAXPERIOD].copy_from_slice(&self.postfilter_mem[c]);
                work[COMBFILTER_MAXPERIOD..COMBFILTER_MAXPERIOD + n]
                    .copy_from_slice(&self.synthesis_channels[c]);

                comb_filter_in_place(
                    work,
                    COMBFILTER_MAXPERIOD,
                    self.postfilter_period_old,
                    self.postfilter_period,
                    short,
                    self.postfilter_gain_old,
                    self.postfilter_gain,
                    self.postfilter_tapset_old,
                    self.postfilter_tapset,
                    Some(&self.mode.window),
                    self.mode.overlap,
                );
                if lm != 0 {
                    comb_filter_in_place(
                        work,
                        COMBFILTER_MAXPERIOD + short,
                        self.postfilter_period,
                        postfilter_pitch.max(COMBFILTER_MINPERIOD),
                        n - short,
                        self.postfilter_gain,
                        postfilter_gain,
                        self.postfilter_tapset,
                        postfilter_tapset,
                        Some(&self.mode.window),
                        self.mode.overlap,
                    );
                }

                self.synthesis_channels[c]
                    .copy_from_slice(&work[COMBFILTER_MAXPERIOD..COMBFILTER_MAXPERIOD + n]);
                self.postfilter_mem[c].copy_from_slice(&work[n..n + COMBFILTER_MAXPERIOD]);
            }
        } else {
            for c in 0..stream_channels {
                if n >= COMBFILTER_MAXPERIOD {
                    self.postfilter_mem[c]
                        .copy_from_slice(&self.synthesis_channels[c][n - COMBFILTER_MAXPERIOD..n]);
                } else {
                    self.postfilter_mem[c].copy_within(n..COMBFILTER_MAXPERIOD, 0);
                    self.postfilter_mem[c][COMBFILTER_MAXPERIOD - n..]
                        .copy_from_slice(&self.synthesis_channels[c][..n]);
                }
            }
        }

        self.postfilter_period_old = self.postfilter_period;
        self.postfilter_gain_old = self.postfilter_gain;
        self.postfilter_tapset_old = self.postfilter_tapset;
        self.postfilter_period = postfilter_pitch;
        self.postfilter_gain = postfilter_gain;
        self.postfilter_tapset = postfilter_tapset;
        if lm != 0 {
            self.postfilter_period_old = self.postfilter_period;
            self.postfilter_gain_old = self.postfilter_gain;
            self.postfilter_tapset_old = self.postfilter_tapset;
        }
    }
}
