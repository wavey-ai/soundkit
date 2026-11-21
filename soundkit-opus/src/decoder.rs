use crate::celt::bands::celt_lcg_rand;
use crate::celt::codec::{
    decode_spectral_frame_into_with_anti_collapse, CeltFrameConfig, CeltFrameDecodeScratch,
};
use crate::celt::mathops::{float_to_i16, float_to_i16_exact, float_to_i24_exact};
use crate::celt::modes::CeltMode;
use crate::celt::pitch::{comb_filter_in_place_mode, COMBFILTER_MAXPERIOD, COMBFILTER_MINPERIOD};
use crate::celt::refkern;
use crate::celt::synthesis::{
    celt_synthesis_channels_into, celt_synthesis_with_overlap_into,
    deemphasis_interleaved_f32_slice, deemphasis_interleaved_i24_into,
    deemphasis_interleaved_i24_slice, deemphasis_interleaved_into, SynthesisScratch,
};
use crate::celt::vq::renormalise_vector_mode;
use crate::constants::{valid_channels, valid_sample_rate, Bandwidth};
use crate::custom;
use crate::packet;
use crate::{Error, Result};
use soundkit::audio_packet::Decoder as SoundkitDecoder;

/// Samples of post-filtered output history the decoder keeps for the pitch
/// post-filter and for packet-loss concealment (`DECODE_BUFFER_SIZE`).
pub const DECODE_BUFFER_SIZE: usize = 2048;
const PLC_PITCH_LAG_MAX: usize = 720;
const PLC_PITCH_LAG_MIN: usize = 100;
const CELT_LPC_ORDER: usize = 24;
const MAX_PERIOD: usize = 1024;
const ENERGY_FLOOR: f32 = -28.0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FrameType {
    None,
    Normal,
    PlcNoise,
    PlcPeriodic,
}

/// How packets are framed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Framing {
    /// Standard Opus packets with a TOC byte (RFC 6716).
    Opus,
    /// Opus Custom packets, as `opus_custom_decode` reads them.
    Custom { signalling: bool, frame_size: usize },
}

#[derive(Clone, Debug, Default)]
struct PlcScratch {
    memory: Vec<Vec<f32>>,
    lp_pitch: Vec<f32>,
    pitch: refkern::PitchScratch,
    sig: Vec<f32>,
    exc: Vec<f32>,
    old_overlap: Vec<f32>,
    etmp: Vec<f32>,
    iir: Vec<f32>,
    window: Vec<f32>,
    noise: Vec<f32>,
    fold: Vec<f32>,
    overlap: Vec<Vec<f32>>,
}

#[derive(Clone, Debug)]
pub struct Decoder {
    sample_rate: i32,
    channels: usize,
    experimental_direct_cubic: bool,
    mode: &'static CeltMode,
    framing: Framing,
    /// Last coded band plus one (`st->end`); a signalled custom packet can
    /// lower it.
    end: usize,
    // Energy state, laid out as libopus does: always two channels of
    // `nb_ebands` values, whatever the channel count.
    old_band_e: Vec<f32>,
    old_log_e: Vec<f32>,
    old_log_e2: Vec<f32>,
    background_log_e: Vec<f32>,
    preemph_mem: Vec<f32>,
    /// The second half of the last inverse MDCT, to overlap with the next
    /// frame (`decode_mem[DECODE_BUFFER_SIZE..]` in libopus).
    overlap_mem: Vec<Vec<f32>>,
    /// The last `DECODE_BUFFER_SIZE` post-filtered samples per channel
    /// (`decode_mem[..DECODE_BUFFER_SIZE]` in libopus). While a frame is
    /// filtered the buffer briefly holds the new frame after the history.
    decode_mem: Vec<Vec<f32>>,
    decode_scratch: CeltFrameDecodeScratch,
    synthesis_channels: Vec<Vec<f32>>,
    synthesis_scratch: SynthesisScratch,
    pcm_f32_scratch: Vec<f32>,
    postfilter_period: usize,
    postfilter_period_old: usize,
    postfilter_gain: f32,
    postfilter_gain_old: f32,
    postfilter_tapset: usize,
    postfilter_tapset_old: usize,
    seed: u32,
    // Packet-loss concealment state.
    last_frame_size: usize,
    last_pitch_index: usize,
    plc_decay: [f32; 2],
    loss_duration: i32,
    plc_duration: i32,
    last_frame_type: FrameType,
    skip_plc: bool,
    prefilter_and_fold: bool,
    lpc: [[f32; CELT_LPC_ORDER]; 2],
    plc_scratch: PlcScratch,
    /// A stereo decoder whose two channels hold identical state: every
    /// frame so far was mono and nothing was concealed. A mono frame can
    /// then be synthesised once and copied, with the same result as
    /// libopus's per-channel synthesis.
    channels_mirrored: bool,
}

impl Decoder {
    pub fn new(sample_rate: i32, channels: usize) -> Result<Self> {
        if sample_rate != 48_000
            || !valid_sample_rate(sample_rate)
            || !valid_channels(channels as i32)
        {
            return Err(Error::BadArg);
        }
        Ok(Self::with_mode(
            sample_rate,
            channels,
            CeltMode::standard_48k_shared(),
            Framing::Opus,
        ))
    }

    /// Creates an Opus Custom decoder for `frame_size`-sample frames at
    /// 48 kHz, as `opus_custom_decoder_create(opus_custom_mode_create(48000,
    /// frame_size), channels)` does.
    ///
    /// The frame size selects the mode (see [`CeltMode::custom_48k_shared`]):
    /// 48, 64 and 72 samples give 1, 1.33 and 1.5 ms custom modes; 120 to 960
    /// samples give the standard mode. Packets carry the libopus signalling
    /// byte unless [`Decoder::set_custom_signalling`] turns it off. An empty
    /// packet, or one with at most one byte of CELT data, conceals a lost
    /// frame. The decoder uses libopus's default float arithmetic (see
    /// [`Decoder::set_libopus_arithmetic`]), so its output matches
    /// `opus_custom_decode` sample for sample on the same packets.
    pub fn new_custom(sample_rate: i32, channels: usize, frame_size: usize) -> Result<Self> {
        if sample_rate != 48_000 || !valid_channels(channels as i32) {
            return Err(Error::BadArg);
        }
        let mode = CeltMode::custom_48k_shared(frame_size)?;
        if mode.lm_for_frame_size(frame_size).is_none() {
            return Err(Error::BadArg);
        }
        Ok(Self::with_mode(
            sample_rate,
            channels,
            mode,
            Framing::Custom {
                signalling: true,
                frame_size,
            },
        ))
    }

    fn with_mode(
        sample_rate: i32,
        channels: usize,
        mode: &'static CeltMode,
        framing: Framing,
    ) -> Self {
        let nb = mode.nb_ebands;
        let last_frame_size = match framing {
            Framing::Opus => sample_rate as usize / 400,
            Framing::Custom { frame_size, .. } => frame_size,
        };
        Self {
            sample_rate,
            channels,
            experimental_direct_cubic: false,
            end: mode.eff_ebands,
            old_band_e: vec![0.0; 2 * nb],
            old_log_e: vec![ENERGY_FLOOR; 2 * nb],
            old_log_e2: vec![ENERGY_FLOOR; 2 * nb],
            background_log_e: vec![0.0; 2 * nb],
            preemph_mem: vec![0.0; channels],
            overlap_mem: vec![vec![0.0; mode.overlap]; channels],
            decode_mem: vec![vec![0.0; DECODE_BUFFER_SIZE]; channels],
            decode_scratch: CeltFrameDecodeScratch::default(),
            synthesis_channels: vec![Vec::new(); channels],
            synthesis_scratch: SynthesisScratch::default(),
            pcm_f32_scratch: Vec::new(),
            postfilter_period: 0,
            postfilter_period_old: 0,
            postfilter_gain: 0.0,
            postfilter_gain_old: 0.0,
            postfilter_tapset: 0,
            postfilter_tapset_old: 0,
            seed: 0,
            last_frame_size,
            last_pitch_index: 0,
            plc_decay: [0.0; 2],
            loss_duration: 0,
            plc_duration: 0,
            last_frame_type: FrameType::None,
            skip_plc: true,
            prefilter_and_fold: false,
            lpc: [[0.0; CELT_LPC_ORDER]; 2],
            plc_scratch: PlcScratch::default(),
            channels_mirrored: true,
            mode,
            framing,
        }
    }

    pub const fn sample_rate(&self) -> i32 {
        self.sample_rate
    }

    pub const fn channels(&self) -> usize {
        self.channels
    }

    /// The CELT mode this decoder uses.
    pub fn mode(&self) -> &'static CeltMode {
        self.mode
    }

    /// True for an Opus Custom decoder.
    pub const fn is_custom(&self) -> bool {
        matches!(self.framing, Framing::Custom { .. })
    }

    /// Frame size of an Opus Custom decoder, or `None` for standard Opus.
    pub const fn custom_frame_size(&self) -> Option<usize> {
        match self.framing {
            Framing::Custom { frame_size, .. } => Some(frame_size),
            Framing::Opus => None,
        }
    }

    /// Reports whether an Opus Custom decoder expects the signalling byte.
    pub const fn custom_signalling(&self) -> bool {
        matches!(
            self.framing,
            Framing::Custom {
                signalling: true,
                ..
            }
        )
    }

    /// Selects whether Opus Custom packets carry the signalling byte
    /// (`CELT_SET_SIGNALLING`). With it off, every packet is one CELT frame of
    /// the configured size with this decoder's channel count. Returns
    /// `BadArg` for a standard Opus decoder.
    pub fn set_custom_signalling(&mut self, enabled: bool) -> Result<()> {
        match &mut self.framing {
            Framing::Custom { signalling, .. } => {
                *signalling = enabled;
                Ok(())
            }
            Framing::Opus => Err(Error::BadArg),
        }
    }

    /// Reports whether the decoder uses libopus's default float arithmetic
    /// (see [`CeltMode::exact_math`]).
    pub fn libopus_arithmetic(&self) -> bool {
        self.mode.exact_math
    }

    /// Selects libopus's default float arithmetic: `log2`/`exp2` in double
    /// precision, libopus's x86-64 summation order, and integer output
    /// rounded half to even. Opus Custom decoders start with it on, and their
    /// output then matches `opus_custom_decode`. Standard Opus decoders start
    /// with it off and keep the output SoundKit has always produced; turning
    /// it on makes them match libopus's CELT decoder too. Takes effect after
    /// a reset of the stream state, so call it before decoding.
    pub fn set_libopus_arithmetic(&mut self, enabled: bool) -> Result<()> {
        let frame_size = self.mode.frame_size();
        self.mode = if enabled == self.mode.exact_math {
            self.mode
        } else {
            CeltMode::custom_48k_shared_with_math(frame_size, enabled)?
        };
        self.reset_state();
        Ok(())
    }

    pub fn init(&mut self) -> Result<()> {
        self.reset()
    }

    /// Resets the packet decoder while retaining the selected stream format.
    pub fn reset(&mut self) -> Result<()> {
        self.reset_state();
        Ok(())
    }

    fn reset_state(&mut self) {
        let experimental_direct_cubic = self.experimental_direct_cubic;
        *self = Self::with_mode(self.sample_rate, self.channels, self.mode, self.framing);
        self.experimental_direct_cubic = experimental_direct_cubic;
    }

    /// Reports whether this decoder accepts the experimental direct-cubic packet syntax.
    pub const fn experimental_direct_cubic(&self) -> bool {
        self.experimental_direct_cubic
    }

    /// Selects the experimental direct-cubic packet syntax.
    ///
    /// Set the same value on the encoder before it creates these packets. This
    /// mode is not compatible with standard Opus packets.
    pub fn set_experimental_direct_cubic(&mut self, enabled: bool) {
        self.experimental_direct_cubic = enabled;
    }

    pub fn validate_packet(&self, packet: &[u8]) -> Result<usize> {
        match self.framing {
            Framing::Opus => {
                let samples = packet::sample_count(packet, self.sample_rate)?;
                Ok(samples as usize)
            }
            Framing::Custom {
                signalling,
                frame_size,
            } => {
                if packet.is_empty() || !signalling {
                    return Ok(frame_size);
                }
                let parsed = custom::parse_signalled(self.mode, packet)?;
                let samples = self.mode.short_mdct_size << parsed.lm;
                if samples > frame_size {
                    return Err(Error::BufferTooSmall);
                }
                Ok(samples)
            }
        }
    }

    pub fn decode_i16_vec(&mut self, packet: &[u8], decode_fec: bool) -> Result<Vec<i16>> {
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
        deemphasis_interleaved_into(
            self.mode,
            &self.synthesis_channels[..channels],
            &mut self.preemph_mem,
            &mut self.pcm_f32_scratch,
        )?;
        pcm.resize(self.pcm_f32_scratch.len(), 0);
        self.convert_i16(pcm);
        Ok(pcm.len() / channels)
    }

    fn convert_i16(&self, pcm: &mut [i16]) {
        let exact = self.mode.exact_math;
        for (output, &sample) in pcm.iter_mut().zip(&self.pcm_f32_scratch) {
            *output = if exact {
                float_to_i16_exact(sample)
            } else {
                float_to_i16(sample)
            };
        }
    }

    /// Decodes directly into caller-owned signed 16-bit PCM storage.
    pub fn decode_i16_into_slice(
        &mut self,
        packet: &[u8],
        decode_fec: bool,
        pcm: &mut [i16],
    ) -> Result<usize> {
        let channels = self.decode_channels(packet, decode_fec)?;
        deemphasis_interleaved_into(
            self.mode,
            &self.synthesis_channels[..channels],
            &mut self.preemph_mem,
            &mut self.pcm_f32_scratch,
        )?;
        let sample_count = self.pcm_f32_scratch.len();
        let pcm = pcm.get_mut(..sample_count).ok_or(Error::BufferTooSmall)?;
        self.convert_i16(pcm);
        Ok(sample_count / channels)
    }

    /// Decodes to signed 24-bit PCM stored sign-extended in `i32` samples.
    pub fn decode_i24_vec(&mut self, packet: &[u8], decode_fec: bool) -> Result<Vec<i32>> {
        let mut pcm = Vec::new();
        self.decode_i24_into(packet, decode_fec, &mut pcm)?;
        Ok(pcm)
    }

    /// Decodes into signed 24-bit PCM stored sign-extended in `i32` samples.
    ///
    /// With libopus arithmetic the samples follow libopus `FLOAT2INT24`,
    /// which clamps to +/-2^24 rather than to the 24-bit range.
    pub fn decode_i24_into(
        &mut self,
        packet: &[u8],
        decode_fec: bool,
        pcm: &mut Vec<i32>,
    ) -> Result<usize> {
        let channels = self.decode_channels(packet, decode_fec)?;
        if self.mode.exact_math {
            deemphasis_interleaved_into(
                self.mode,
                &self.synthesis_channels[..channels],
                &mut self.preemph_mem,
                &mut self.pcm_f32_scratch,
            )?;
            pcm.clear();
            pcm.extend(
                self.pcm_f32_scratch
                    .iter()
                    .map(|&sample| float_to_i24_exact(sample)),
            );
            return Ok(pcm.len() / channels);
        }
        deemphasis_interleaved_i24_into(
            self.mode,
            &self.synthesis_channels[..channels],
            &mut self.preemph_mem,
            pcm,
        )?;
        Ok(pcm.len() / channels)
    }

    /// Decodes directly into caller-owned signed 24-bit PCM storage.
    pub fn decode_i24_into_slice(
        &mut self,
        packet: &[u8],
        decode_fec: bool,
        pcm: &mut [i32],
    ) -> Result<usize> {
        let channels = self.decode_channels(packet, decode_fec)?;
        if self.mode.exact_math {
            deemphasis_interleaved_into(
                self.mode,
                &self.synthesis_channels[..channels],
                &mut self.preemph_mem,
                &mut self.pcm_f32_scratch,
            )?;
            let count = self.pcm_f32_scratch.len();
            let pcm = pcm.get_mut(..count).ok_or(Error::BufferTooSmall)?;
            for (output, &sample) in pcm.iter_mut().zip(&self.pcm_f32_scratch) {
                *output = float_to_i24_exact(sample);
            }
            return Ok(count / channels);
        }
        deemphasis_interleaved_i24_slice(
            self.mode,
            &self.synthesis_channels[..channels],
            &mut self.preemph_mem,
            pcm,
        )
    }

    pub fn decode_f32_vec(&mut self, packet: &[u8], decode_fec: bool) -> Result<Vec<f32>> {
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
            self.mode,
            &self.synthesis_channels[..channels],
            &mut self.preemph_mem,
            pcm,
        )?;
        Ok(pcm.len() / channels)
    }

    /// Decodes directly into caller-owned floating-point PCM storage.
    pub fn decode_f32_into_slice(
        &mut self,
        packet: &[u8],
        decode_fec: bool,
        pcm: &mut [f32],
    ) -> Result<usize> {
        let channels = self.decode_channels(packet, decode_fec)?;
        deemphasis_interleaved_f32_slice(
            self.mode,
            &self.synthesis_channels[..channels],
            &mut self.preemph_mem,
            pcm,
        )
    }

    /// Decodes one packet into `synthesis_channels` and returns the number of
    /// output channels. An empty packet conceals a lost frame.
    fn decode_channels(&mut self, packet: &[u8], decode_fec: bool) -> Result<usize> {
        if decode_fec {
            return Err(Error::Unimplemented);
        }
        if self.sample_rate != 48_000 {
            return Err(Error::Unimplemented);
        }
        match self.framing {
            Framing::Opus => self.decode_opus(packet),
            Framing::Custom {
                signalling,
                frame_size,
            } => self.decode_custom(packet, signalling, frame_size),
        }
    }

    fn decode_opus(&mut self, packet: &[u8]) -> Result<usize> {
        if packet.is_empty() {
            let n = self.last_frame_size;
            let lm = self.mode.lm_for_frame_size(n).ok_or(Error::BadArg)?;
            return self.conceal(n, lm);
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
        if frame.len() <= 1 {
            return self.conceal(expected_frame_size, lm);
        }
        self.decode_frame(frame, lm, stream_channels)
    }

    fn decode_custom(
        &mut self,
        packet: &[u8],
        signalling: bool,
        frame_size: usize,
    ) -> Result<usize> {
        if packet.is_empty() {
            let lm = self
                .mode
                .lm_for_frame_size(frame_size)
                .ok_or(Error::BadArg)?;
            return self.conceal(frame_size, lm);
        }
        let (frame, lm, stream_channels) = if signalling {
            // libopus sets the end band before it validates the rest of the
            // packet, so even a rejected packet changes it.
            if let Some(end) = custom::signalled_end(self.mode, packet[0]) {
                self.end = end;
            }
            let parsed = custom::parse_signalled(self.mode, packet)?;
            if self.mode.short_mdct_size << parsed.lm > frame_size {
                return Err(Error::BufferTooSmall);
            }
            (parsed.frame, parsed.lm, parsed.stream_channels)
        } else {
            let lm = self
                .mode
                .lm_for_frame_size(frame_size)
                .ok_or(Error::BadArg)?;
            (packet, lm, self.channels)
        };
        if frame.len() > 1275 {
            return Err(Error::BadArg);
        }
        if frame.len() <= 1 {
            return self.conceal(self.mode.short_mdct_size << lm, lm);
        }
        self.decode_frame(frame, lm, stream_channels)
    }

    fn decode_frame(&mut self, frame: &[u8], lm: usize, stream_channels: usize) -> Result<usize> {
        let mode = self.mode;
        let nb = mode.nb_ebands;
        let n = mode.short_mdct_size << lm;
        if self.loss_duration == 0 {
            self.skip_plc = false;
        }
        if stream_channels == 1 {
            for i in 0..nb {
                self.old_band_e[i] = self.old_band_e[i].max(self.old_band_e[nb + i]);
            }
        }

        let mut config = CeltFrameConfig::new(mode, lm, stream_channels, frame.len())?;
        config.end = self.end;
        config.experimental_direct_cubic = self.experimental_direct_cubic;
        config.loss_duration = self.loss_duration;
        // libopus disables intensity-stereo phase inversion in mono decoders
        // (`st->disable_inv = channels == 1`) so that a downmixed stereo
        // stream does not cancel.
        config.disable_inv = self.channels == 1;
        let decoded = decode_spectral_frame_into_with_anti_collapse(
            mode,
            &config,
            frame,
            &mut self.old_band_e,
            &self.old_log_e,
            &self.old_log_e2,
            &mut self.seed,
            &mut self.decode_scratch,
        )?;
        if self.prefilter_and_fold {
            self.fold_overlap(n);
        }
        // libopus synthesises and post-filters every output channel from its
        // own state, also when a mono stream feeds a stereo decoder or a
        // stereo stream a mono one. While a stereo decoder's channels are
        // still identical, one channel is computed and copied.
        let mirror = stream_channels == 1 && self.channels == 2 && self.channels_mirrored;
        let output_channels = if mirror { 1 } else { self.channels };
        let y = (config.channels == 2).then(|| &self.decode_scratch.y[..decoded.samples]);
        celt_synthesis_channels_into(
            mode,
            &self.decode_scratch.x[..decoded.samples],
            y,
            &self.old_band_e,
            config.start,
            config.end.min(mode.eff_ebands),
            config.channels,
            output_channels,
            decoded.is_transient,
            config.lm,
            1,
            decoded.silence,
            &mut self.overlap_mem[..output_channels],
            &mut self.synthesis_channels[..output_channels],
            &mut self.synthesis_scratch,
        )?;
        self.apply_postfilter(output_channels, config.lm, decoded.prefilter);
        if mirror {
            {
                let (left, right) = self.overlap_mem.split_at_mut(1);
                right[0].clone_from(&left[0]);
            }
            {
                let (left, right) = self.decode_mem.split_at_mut(1);
                right[0].clone_from(&left[0]);
            }
            {
                let (left, right) = self.synthesis_channels.split_at_mut(1);
                right[0].clear();
                right[0].extend_from_slice(&left[0]);
            }
        }
        self.channels_mirrored = mirror;
        self.update_energy_history(stream_channels, decoded.is_transient, 1 << lm);
        self.loss_duration = 0;
        self.plc_duration = 0;
        self.last_frame_type = FrameType::Normal;
        self.prefilter_and_fold = false;
        self.last_frame_size = n;
        Ok(self.channels)
    }

    /// The energy bookkeeping at the end of `celt_decode_with_ec`.
    fn update_energy_history(&mut self, stream_channels: usize, is_transient: bool, m: i32) {
        let nb = self.mode.nb_ebands;
        if stream_channels == 1 {
            self.old_band_e.copy_within(0..nb, nb);
        }
        if is_transient {
            for (history, &energy) in self.old_log_e.iter_mut().zip(&self.old_band_e) {
                *history = history.min(energy);
            }
        } else {
            self.old_log_e2.copy_from_slice(&self.old_log_e);
            self.old_log_e.copy_from_slice(&self.old_band_e);
        }
        let max_background_increase = (160.min(self.loss_duration + m)) as f32 * 0.001;
        for (background, &energy) in self.background_log_e.iter_mut().zip(&self.old_band_e) {
            *background = (*background + max_background_increase).min(energy);
        }
        for c in 0..2 {
            for i in self.end..nb {
                self.old_band_e[c * nb + i] = 0.0;
                self.old_log_e[c * nb + i] = ENERGY_FLOOR;
                self.old_log_e2[c * nb + i] = ENERGY_FLOOR;
            }
        }
    }

    fn apply_postfilter(
        &mut self,
        channels: usize,
        lm: usize,
        prefilter: Option<crate::celt::codec::DecodedPrefilter>,
    ) {
        let exact = self.mode.exact_math;
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
        for c in 0..channels {
            // The history buffer gains the new frame at its end, is filtered
            // in place, and then drops its oldest `n` samples.
            let memory = &mut self.decode_mem[c];
            memory.truncate(DECODE_BUFFER_SIZE);
            memory.extend_from_slice(&self.synthesis_channels[c][..n]);
            if filters_active {
                comb_filter_in_place_mode(
                    exact,
                    memory,
                    DECODE_BUFFER_SIZE,
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
                    comb_filter_in_place_mode(
                        exact,
                        memory,
                        DECODE_BUFFER_SIZE + short,
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
                self.synthesis_channels[c][..n]
                    .copy_from_slice(&memory[DECODE_BUFFER_SIZE..DECODE_BUFFER_SIZE + n]);
            }
            memory.copy_within(n..DECODE_BUFFER_SIZE + n, 0);
            memory.truncate(DECODE_BUFFER_SIZE);
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

    /// `prefilter_and_fold` on the next frame's MDCT overlap: undo the
    /// post-filter the concealment applied, then simulate TDAC so the
    /// concealed audio blends with the next decoded frame.
    fn fold_overlap(&mut self, _n: usize) {
        let overlap = self.mode.overlap;
        let history = COMBFILTER_MAXPERIOD + 2;
        for c in 0..self.channels {
            let fold = &mut self.plc_scratch.fold;
            fold.clear();
            fold.extend_from_slice(&self.decode_mem[c][DECODE_BUFFER_SIZE - history..]);
            fold.extend_from_slice(&self.overlap_mem[c]);
            let etmp = &mut self.plc_scratch.etmp;
            etmp.resize(overlap, 0.0);
            refkern::comb_filter_no_window(
                fold,
                history,
                etmp,
                self.postfilter_period,
                overlap,
                -self.postfilter_gain,
                self.postfilter_tapset,
            );
            let window = &self.mode.window;
            for i in 0..overlap / 2 {
                self.overlap_mem[c][i] =
                    window[i] * etmp[overlap - 1 - i] + window[overlap - i - 1] * etmp[i];
            }
        }
    }

    /// `celt_decode_lost`: conceals one lost frame of `n` samples into
    /// `synthesis_channels`.
    fn conceal(&mut self, n: usize, lm: usize) -> Result<usize> {
        let mode = self.mode;
        let cc = self.channels;
        let overlap = mode.overlap;
        let dbs = DECODE_BUFFER_SIZE;
        // Lay the state out as libopus's decode_mem: history, then overlap.
        let mut memory = std::mem::take(&mut self.plc_scratch.memory);
        memory.resize_with(cc, Vec::new);
        for c in 0..cc {
            memory[c].clear();
            memory[c].extend_from_slice(&self.decode_mem[c]);
            memory[c].extend_from_slice(&self.overlap_mem[c]);
        }

        let noise = self.plc_duration >= 40 || self.skip_plc;
        let frame_type = if noise {
            self.conceal_noise(&mut memory, n, lm)?;
            FrameType::PlcNoise
        } else {
            self.conceal_periodic(&mut memory, n);
            FrameType::PlcPeriodic
        };

        for c in 0..cc {
            self.decode_mem[c].clear();
            self.decode_mem[c].extend_from_slice(&memory[c][..dbs]);
            self.overlap_mem[c].copy_from_slice(&memory[c][dbs..dbs + overlap]);
            self.synthesis_channels[c].clear();
            self.synthesis_channels[c].extend_from_slice(&memory[c][dbs - n..dbs]);
        }
        self.plc_scratch.memory = memory;
        self.channels_mirrored = false;
        self.loss_duration = 10_000.min(self.loss_duration + (1 << lm));
        self.plc_duration = 10_000.min(self.plc_duration + (1 << lm));
        self.last_frame_type = frame_type;
        self.last_frame_size = n;
        Ok(cc)
    }

    fn conceal_noise(&mut self, memory: &mut [Vec<f32>], n: usize, lm: usize) -> Result<()> {
        let mode = self.mode;
        let cc = self.channels;
        let nb = mode.nb_ebands;
        let overlap = mode.overlap;
        let dbs = DECODE_BUFFER_SIZE;
        let start = 0usize;
        let end = self.end;
        let eff_end = start.max(end.min(mode.eff_ebands));
        for buffer in memory.iter_mut() {
            buffer.copy_within(n..dbs + overlap, 0);
        }
        if self.prefilter_and_fold {
            // The concealed frame's overlap now sits at `dbs - n`.
            for buffer in memory.iter_mut() {
                let etmp = &mut self.plc_scratch.etmp;
                etmp.resize(overlap, 0.0);
                refkern::comb_filter_no_window(
                    buffer,
                    dbs - n,
                    etmp,
                    self.postfilter_period,
                    overlap,
                    -self.postfilter_gain,
                    self.postfilter_tapset,
                );
                for i in 0..overlap / 2 {
                    buffer[dbs - n + i] = mode.window[i] * etmp[overlap - 1 - i]
                        + mode.window[overlap - i - 1] * etmp[i];
                }
            }
        }

        let mut decay = (1i32 << lm) as f32 * 0.0625;
        if self.last_frame_type != FrameType::PlcNoise {
            decay += 1.0f32.max(160.min(self.loss_duration) as f32 * 0.0625);
        }
        for c in 0..cc {
            for i in start..end {
                let idx = c * nb + i;
                self.old_band_e[idx] = self.background_log_e[idx].max(self.old_band_e[idx] - decay);
            }
        }

        let noise = &mut self.plc_scratch.noise;
        noise.clear();
        noise.resize(cc * n, 0.0);
        let mut seed = self.seed;
        for c in 0..cc {
            for i in start..eff_end {
                let offset = n * c + ((mode.ebands[i] as usize) << lm);
                let len = ((mode.ebands[i + 1] - mode.ebands[i]) as usize) << lm;
                for value in &mut noise[offset..offset + len] {
                    seed = celt_lcg_rand(seed);
                    *value = ((seed as i32) >> 20) as f32;
                }
                renormalise_vector_mode(true, &mut noise[offset..offset + len], len, 1.0);
            }
        }
        self.seed = seed;

        // Synthesise into the frame slot, overlapping with the moved tail.
        let mut overlap_mem = std::mem::take(&mut self.plc_scratch.overlap);
        overlap_mem.resize_with(cc, Vec::new);
        for c in 0..cc {
            overlap_mem[c].clear();
            overlap_mem[c].extend_from_slice(&memory[c][dbs - n..dbs - n + overlap]);
        }
        let (x, y) = noise.split_at(n);
        celt_synthesis_with_overlap_into(
            mode,
            x,
            (cc == 2).then_some(y),
            &self.old_band_e,
            start,
            eff_end,
            cc,
            false,
            lm,
            1,
            false,
            &mut overlap_mem,
            &mut self.synthesis_channels,
            &mut self.synthesis_scratch,
        )?;
        for c in 0..cc {
            memory[c][dbs - n..dbs].copy_from_slice(&self.synthesis_channels[c][..n]);
            memory[c][dbs..dbs + overlap].copy_from_slice(&overlap_mem[c]);
        }
        self.plc_scratch.overlap = overlap_mem;

        self.postfilter_period = self.postfilter_period.max(COMBFILTER_MINPERIOD);
        self.postfilter_period_old = self.postfilter_period_old.max(COMBFILTER_MINPERIOD);
        let short = mode.short_mdct_size;
        for buffer in memory.iter_mut() {
            comb_filter_in_place_mode(
                mode.exact_math,
                buffer,
                dbs - n,
                self.postfilter_period_old,
                self.postfilter_period,
                short,
                self.postfilter_gain_old,
                self.postfilter_gain,
                self.postfilter_tapset_old,
                self.postfilter_tapset,
                Some(&mode.window),
                overlap,
            );
            if lm != 0 {
                comb_filter_in_place_mode(
                    mode.exact_math,
                    buffer,
                    dbs - n + short,
                    self.postfilter_period,
                    self.postfilter_period,
                    n - short,
                    self.postfilter_gain,
                    self.postfilter_gain,
                    self.postfilter_tapset,
                    self.postfilter_tapset,
                    Some(&mode.window),
                    overlap,
                );
            }
        }
        self.postfilter_period_old = self.postfilter_period;
        self.postfilter_gain_old = self.postfilter_gain;
        self.postfilter_tapset_old = self.postfilter_tapset;
        self.prefilter_and_fold = false;
        self.skip_plc = true;
        Ok(())
    }

    fn plc_pitch_search(&mut self, memory: &[Vec<f32>]) -> usize {
        let half = DECODE_BUFFER_SIZE >> 1;
        let scratch = &mut self.plc_scratch;
        scratch.lp_pitch.clear();
        scratch.lp_pitch.resize(half, 0.0);
        let channels: [&[f32]; 2] = [&memory[0], memory.get(1).map_or(&[][..], |m| &m[..])];
        refkern::pitch_downsample(
            &channels[..memory.len()],
            &mut scratch.lp_pitch,
            half,
            2,
            &mut scratch.pitch,
        );
        let lp = std::mem::take(&mut scratch.lp_pitch);
        let pitch = refkern::pitch_search(
            &lp[PLC_PITCH_LAG_MAX >> 1..],
            &lp,
            DECODE_BUFFER_SIZE - PLC_PITCH_LAG_MAX,
            PLC_PITCH_LAG_MAX - PLC_PITCH_LAG_MIN,
            &mut scratch.pitch,
        );
        scratch.lp_pitch = lp;
        PLC_PITCH_LAG_MAX - pitch
    }

    fn conceal_periodic(&mut self, memory: &mut [Vec<f32>], n: usize) {
        let mode = self.mode;
        let cc = self.channels;
        let overlap = mode.overlap;
        let dbs = DECODE_BUFFER_SIZE;
        let first_loss = self.last_frame_type != FrameType::PlcPeriodic;
        let mut fade = 1.0f32;
        if first_loss {
            self.last_pitch_index = self.plc_pitch_search(memory);
        } else if self.plc_duration * mode.short_mdct_size as i32 >= self.last_pitch_index as i32 {
            fade = 0.8;
        }
        let exc_length = MAX_PERIOD.max(2 * self.last_pitch_index);
        let window = &mode.window;
        let scratch = &mut self.plc_scratch;
        scratch.sig.resize(exc_length + CELT_LPC_ORDER, 0.0);
        scratch.exc.resize(exc_length, 0.0);
        scratch.old_overlap.resize(overlap >> 1, 0.0);
        scratch.etmp.resize(overlap, 0.0);

        for c in 0..cc {
            let mut pitch_index = self.last_pitch_index;
            let buf = &mut memory[c];
            for i in 0..exc_length + CELT_LPC_ORDER {
                scratch.sig[i] = buf[dbs - exc_length - CELT_LPC_ORDER + i];
            }
            if first_loss {
                let mut ac = [0.0f32; CELT_LPC_ORDER + 1];
                let offset = CELT_LPC_ORDER + exc_length - MAX_PERIOD;
                refkern::autocorr(
                    &scratch.sig[offset..offset + MAX_PERIOD],
                    &mut ac,
                    Some(window),
                    overlap,
                    CELT_LPC_ORDER,
                    MAX_PERIOD,
                    &mut scratch.window,
                );
                ac[0] *= 1.0001;
                for (i, value) in ac.iter_mut().enumerate().skip(1) {
                    *value -= *value * (0.008f32 * 0.008f32) * i as f32 * i as f32;
                }
                refkern::lpc(&mut self.lpc[c], &ac, CELT_LPC_ORDER);
            }
            refkern::fir(
                &scratch.sig,
                &self.lpc[c],
                &mut scratch.exc,
                exc_length,
                CELT_LPC_ORDER,
            );
            let exc = &scratch.exc;
            let decay = if first_loss {
                let mut e1 = 1.0f32;
                let mut e2 = 1.0f32;
                for i in 0..pitch_index {
                    let e = exc[exc_length - pitch_index + i];
                    e1 += e * e;
                    let e = exc[exc_length - 2 * pitch_index + i];
                    e2 += e * e;
                }
                e1 = e1.min(e2);
                let decay = (e1 / e2).sqrt();
                self.plc_decay[c] = decay;
                decay
            } else {
                self.plc_decay[c]
            };
            {
                let mut e0 = 1.0f32;
                let k_max = exc_length / pitch_index;
                for i in 0..pitch_index {
                    let e = exc[exc_length - pitch_index + i];
                    e0 += e * e;
                }
                let mut k = 2usize;
                while k <= k_max {
                    let mut ek = 1.0f32;
                    for i in 0..pitch_index {
                        let e = exc[exc_length - k * pitch_index + i];
                        ek += e * e;
                    }
                    if ek < 0.25 * e0 || e0 < 0.25 * ek {
                        break;
                    }
                    k += 1;
                }
                pitch_index *= k - 1;
            }
            if self.last_frame_type == FrameType::Normal {
                scratch.old_overlap[..overlap >> 1]
                    .copy_from_slice(&buf[dbs..dbs + (overlap >> 1)]);
            }
            buf.copy_within(n..dbs, 0);

            let extrapolation_offset = exc_length - pitch_index;
            let extrapolation_len = n + overlap;
            let mut attenuation = fade * decay;
            let mut s1 = 0.0f32;
            let mut j = 0usize;
            for i in 0..extrapolation_len {
                if j >= pitch_index {
                    j -= pitch_index;
                    attenuation *= decay;
                }
                buf[dbs - n + i] = attenuation * exc[extrapolation_offset + j];
                let tmp = attenuation * scratch.sig[CELT_LPC_ORDER + extrapolation_offset + j];
                s1 += tmp * tmp;
                j += 1;
            }
            let mut lpc_mem = [0.0f32; CELT_LPC_ORDER];
            for (i, value) in lpc_mem.iter_mut().enumerate() {
                *value = buf[dbs - n - 1 - i];
            }
            refkern::iir(
                &mut buf[dbs - n..dbs - n + extrapolation_len],
                &self.lpc[c],
                extrapolation_len,
                CELT_LPC_ORDER,
                &mut lpc_mem,
                &mut scratch.iir,
            );
            let mut s2 = 0.0f32;
            for &value in &buf[dbs - n..dbs - n + extrapolation_len] {
                s2 += value * value;
            }
            // Written as in libopus so that a NaN from the filter also
            // zeroes the frame.
            #[allow(clippy::neg_cmp_op_on_partial_ord)]
            let explosion = !(s1 > 0.2 * s2);
            if explosion {
                buf[dbs - n..dbs - n + extrapolation_len].fill(0.0);
            } else if s1 < s2 {
                let ratio = ((s1 + 1.0) / (s2 + 1.0)).sqrt();
                for i in 0..overlap {
                    let gain = 1.0 - window[i] * (1.0 - ratio);
                    buf[dbs - n + i] *= gain;
                }
                for value in &mut buf[dbs - n + overlap..dbs - n + extrapolation_len] {
                    *value *= ratio;
                }
            }
            if self.last_frame_type == FrameType::Normal {
                refkern::comb_filter_no_window(
                    buf,
                    dbs - n,
                    &mut scratch.etmp,
                    self.postfilter_period_old,
                    overlap,
                    -self.postfilter_gain_old,
                    self.postfilter_tapset_old,
                );
                let etmp = &scratch.etmp;
                for i in 0..overlap / 2 {
                    let w1 = window[i];
                    let w2 = window[overlap - 1 - i];
                    let x2 = scratch.old_overlap[i];
                    let x1 = w2 * etmp[overlap - 1 - i] - w1 * etmp[i];
                    buf[dbs - n + i] = w2 * x2 - w1 * x1;
                    buf[dbs - n + overlap - 1 - i] = w1 * x2 + w2 * x1;
                }
                comb_filter_in_place_mode(
                    true,
                    buf,
                    dbs - n,
                    self.postfilter_period_old,
                    self.postfilter_period_old,
                    overlap,
                    self.postfilter_gain_old,
                    self.postfilter_gain_old,
                    self.postfilter_tapset_old,
                    self.postfilter_tapset_old,
                    None,
                    0,
                );
            }
        }
        self.prefilter_and_fold = true;
    }
}

impl SoundkitDecoder for Decoder {
    #[inline]
    fn decode_i16(
        &mut self,
        input: &[u8],
        output: &mut [i16],
        fec: bool,
    ) -> std::result::Result<usize, String> {
        self.decode_i16_into_slice(input, fec, output)
            .map_err(|error| error.to_string())
    }

    #[inline]
    fn decode_i32(
        &mut self,
        input: &[u8],
        output: &mut [i32],
        fec: bool,
    ) -> std::result::Result<usize, String> {
        self.decode_i24_into_slice(input, fec, output)
            .map_err(|error| error.to_string())
    }

    #[inline]
    fn decode_f32(
        &mut self,
        input: &[u8],
        output: &mut [f32],
        fec: bool,
    ) -> std::result::Result<usize, String> {
        self.decode_f32_into_slice(input, fec, output)
            .map_err(|error| error.to_string())
    }
}
