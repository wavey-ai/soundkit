use crate::celt::codec::{decode_spectral_frame, CeltFrameConfig};
use crate::celt::mathops::celt_float2int16;
use crate::celt::modes::CeltMode;
use crate::celt::pitch::{comb_filter, COMBFILTER_MAXPERIOD, COMBFILTER_MINPERIOD};
use crate::celt::synthesis::{celt_synthesis_with_overlap, deemphasis_interleaved};
use crate::constants::{valid_channels, valid_sample_rate, Bandwidth};
use crate::packet;
use crate::{Error, Result};

#[derive(Clone, Debug)]
pub struct Decoder {
    sample_rate: i32,
    channels: usize,
    mode: CeltMode,
    old_band_e: Vec<f32>,
    preemph_mem: Vec<f32>,
    overlap_mem: Vec<Vec<f32>>,
    postfilter_mem: Vec<Vec<f32>>,
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
            preemph_mem: vec![0.0; channels],
            overlap_mem: vec![vec![0.0; mode.overlap]; channels],
            postfilter_mem: vec![vec![0.0; COMBFILTER_MAXPERIOD]; channels],
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
        let pcm = self.decode_f32(packet, decode_fec)?;
        let mut out = vec![0i16; pcm.len()];
        celt_float2int16(&pcm, &mut out);
        Ok(out)
    }

    pub fn decode_f32(&mut self, packet: &[u8], _decode_fec: bool) -> Result<Vec<f32>> {
        self.validate_packet(packet)?;
        if self.sample_rate != 48_000 {
            return Err(Error::Unimplemented);
        }

        let parsed = packet::parse_packet(packet)?;
        let stream_channels = packet::channels(packet)?;
        if parsed.frame_count() != 1
            || !packet::is_celt_only(parsed.toc)
            || packet::bandwidth(packet)? != Bandwidth::Fullband
            || stream_channels > self.channels
        {
            return Err(Error::Unimplemented);
        }

        let lm = packet::celt_only_lm(parsed.toc)?;
        let expected_frame_size = self.mode.short_mdct_size << lm;
        if packet::samples_per_frame(packet, self.sample_rate)? as usize != expected_frame_size {
            return Err(Error::InvalidPacket);
        }

        let frame = parsed.frames()[0].data;
        if stream_channels == 1 && self.channels == 2 {
            for i in 0..self.mode.nb_ebands {
                self.old_band_e[i] =
                    self.old_band_e[i].max(self.old_band_e[self.mode.nb_ebands + i]);
            }
        }

        let config = CeltFrameConfig::new(&self.mode, lm, stream_channels, frame.len())?;
        let decoded = decode_spectral_frame(
            &self.mode,
            &config,
            frame,
            &mut self.old_band_e,
            &mut self.seed,
        )?;
        let channels = celt_synthesis_with_overlap(
            &self.mode,
            &decoded.x,
            decoded.y.as_deref(),
            &self.old_band_e,
            config.start,
            config.end.min(self.mode.eff_ebands),
            config.channels,
            decoded.is_transient,
            config.lm,
            1,
            decoded.silence,
            &mut self.overlap_mem[..stream_channels],
        )?;
        let mut channels = channels;
        self.apply_postfilter(&mut channels, stream_channels, config.lm, decoded.prefilter);
        let channels = if stream_channels == 1 && self.channels == 2 {
            self.old_band_e
                .copy_within(0..self.mode.nb_ebands, self.mode.nb_ebands);
            self.overlap_mem[1] = self.overlap_mem[0].clone();
            self.postfilter_mem[1] = self.postfilter_mem[0].clone();
            vec![channels[0].clone(), channels[0].clone()]
        } else {
            channels
        };
        deemphasis_interleaved(&self.mode, &channels, &mut self.preemph_mem)
    }

    fn apply_postfilter(
        &mut self,
        channels: &mut [Vec<f32>],
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
        for c in 0..stream_channels {
            let mut work = vec![0.0f32; COMBFILTER_MAXPERIOD + n];
            work[..COMBFILTER_MAXPERIOD].copy_from_slice(&self.postfilter_mem[c]);
            work[COMBFILTER_MAXPERIOD..COMBFILTER_MAXPERIOD + n].copy_from_slice(&channels[c]);

            let source = work.clone();
            comb_filter(
                &mut work,
                COMBFILTER_MAXPERIOD,
                &source,
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
                let source = work.clone();
                comb_filter(
                    &mut work,
                    COMBFILTER_MAXPERIOD + short,
                    &source,
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

            channels[c].copy_from_slice(&work[COMBFILTER_MAXPERIOD..COMBFILTER_MAXPERIOD + n]);
            self.postfilter_mem[c].copy_from_slice(&work[n..n + COMBFILTER_MAXPERIOD]);
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
