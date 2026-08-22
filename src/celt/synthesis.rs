//! CELT synthesis and deemphasis helpers ported from the official
//! `celt/celt_decoder.c` floating-point path.

use crate::celt::bands::denormalise_bands;
use crate::celt::mathops::{float_to_i16, float_to_i24};
use crate::celt::mdct::{clt_mdct_backward_with_scratch, MdctScratch};
use crate::celt::modes::CeltMode;
use crate::{Error, Result};

const CELT_SIG_SCALE: f32 = 32_768.0;

#[derive(Clone, Debug, Default)]
pub struct SynthesisScratch {
    freq: Vec<f32>,
    mdct: MdctScratch,
}

pub fn celt_synthesis(
    mode: &CeltMode,
    x: &[f32],
    y: Option<&[f32]>,
    band_log_e: &[f32],
    start: usize,
    eff_end: usize,
    channels: usize,
    is_transient: bool,
    lm: usize,
    downsample: usize,
    silence: bool,
) -> Result<Vec<Vec<f32>>> {
    let mut overlap_mem = vec![vec![0.0f32; mode.overlap]; channels];
    celt_synthesis_with_overlap(
        mode,
        x,
        y,
        band_log_e,
        start,
        eff_end,
        channels,
        is_transient,
        lm,
        downsample,
        silence,
        &mut overlap_mem,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn celt_synthesis_with_overlap(
    mode: &CeltMode,
    x: &[f32],
    y: Option<&[f32]>,
    band_log_e: &[f32],
    start: usize,
    eff_end: usize,
    channels: usize,
    is_transient: bool,
    lm: usize,
    downsample: usize,
    silence: bool,
    overlap_mem: &mut [Vec<f32>],
) -> Result<Vec<Vec<f32>>> {
    let mut out = Vec::with_capacity(channels);
    for _ in 0..channels {
        out.push(Vec::new());
    }
    let mut scratch = SynthesisScratch::default();
    celt_synthesis_with_overlap_into(
        mode,
        x,
        y,
        band_log_e,
        start,
        eff_end,
        channels,
        is_transient,
        lm,
        downsample,
        silence,
        overlap_mem,
        &mut out,
        &mut scratch,
    )?;
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn celt_synthesis_with_overlap_into(
    mode: &CeltMode,
    x: &[f32],
    y: Option<&[f32]>,
    band_log_e: &[f32],
    start: usize,
    eff_end: usize,
    channels: usize,
    is_transient: bool,
    lm: usize,
    downsample: usize,
    silence: bool,
    overlap_mem: &mut [Vec<f32>],
    out: &mut [Vec<f32>],
    scratch: &mut SynthesisScratch,
) -> Result<()> {
    if lm > mode.max_lm
        || start > eff_end
        || eff_end > mode.nb_ebands
        || !(1..=2).contains(&channels)
        || downsample == 0
        || overlap_mem.len() < channels
        || out.len() < channels
    {
        return Err(Error::BadArg);
    }

    let n = mode.short_mdct_size << lm;
    if x.len() < n
        || (channels == 2 && y.map_or(true, |right| right.len() < n))
        || band_log_e.len() < channels * mode.nb_ebands
        || overlap_mem
            .iter()
            .take(channels)
            .any(|memory| memory.len() < mode.overlap)
    {
        return Err(Error::BadArg);
    }

    let m = 1usize << lm;
    let (blocks, block_len, shift) = if is_transient {
        (m, mode.short_mdct_size, mode.max_lm)
    } else {
        (1, n, mode.max_lm - lm)
    };

    if scratch.freq.len() < n {
        scratch.freq.resize(n, 0.0);
    }
    for c in 0..channels {
        let norm = if c == 0 {
            x
        } else {
            y.expect("right channel validated")
        };
        let log_e = &band_log_e[c * mode.nb_ebands..(c + 1) * mode.nb_ebands];
        let freq = &mut scratch.freq[..n];
        denormalise_bands(
            mode, norm, freq, log_e, start, eff_end, m, downsample, silence,
        );

        out[c].resize(n + mode.overlap, 0.0);
        let channel = &mut out[c];
        channel[..mode.overlap].copy_from_slice(&overlap_mem[c][..mode.overlap]);
        for block in 0..blocks {
            clt_mdct_backward_with_scratch(
                &mode.mdct,
                &freq[block..],
                &mut channel[block_len * block..],
                &mode.window,
                mode.overlap,
                shift,
                blocks,
                &mut scratch.mdct,
            );
        }
        overlap_mem[c][..mode.overlap].copy_from_slice(&channel[n..n + mode.overlap]);
        channel.truncate(n);
    }

    Ok(())
}

pub fn deemphasis_interleaved(
    mode: &CeltMode,
    channels: &[Vec<f32>],
    preemph_mem: &mut [f32],
) -> Result<Vec<f32>> {
    if channels.is_empty() || channels.len() > 2 || preemph_mem.len() < channels.len() {
        return Err(Error::BadArg);
    }
    let n = channels[0].len();
    if channels.iter().any(|channel| channel.len() != n) {
        return Err(Error::BadArg);
    }

    let mut pcm = Vec::new();
    deemphasis_interleaved_into(mode, channels, preemph_mem, &mut pcm)?;
    Ok(pcm)
}

pub fn deemphasis_interleaved_into(
    mode: &CeltMode,
    channels: &[Vec<f32>],
    preemph_mem: &mut [f32],
    pcm: &mut Vec<f32>,
) -> Result<()> {
    if channels.is_empty() || channels.len() > 2 || preemph_mem.len() < channels.len() {
        return Err(Error::BadArg);
    }
    let n = channels[0].len();
    if channels.iter().any(|channel| channel.len() != n) {
        return Err(Error::BadArg);
    }

    let c_count = channels.len();
    let coef0 = mode.preemph[0];
    pcm.resize(n * c_count, 0.0);
    for c in 0..c_count {
        let mut mem = preemph_mem[c];
        for j in 0..n {
            let tmp = channels[c][j] + 1e-30f32 + mem;
            mem = coef0 * tmp;
            pcm[j * c_count + c] = tmp / CELT_SIG_SCALE;
        }
        preemph_mem[c] = mem;
    }
    Ok(())
}

pub fn deemphasis_interleaved_i16_into(
    mode: &CeltMode,
    channels: &[Vec<f32>],
    preemph_mem: &mut [f32],
    pcm: &mut Vec<i16>,
) -> Result<()> {
    if channels.is_empty() || channels.len() > 2 || preemph_mem.len() < channels.len() {
        return Err(Error::BadArg);
    }
    let n = channels[0].len();
    if channels.iter().any(|channel| channel.len() != n) {
        return Err(Error::BadArg);
    }

    let c_count = channels.len();
    let coef0 = mode.preemph[0];
    pcm.resize(n * c_count, 0);

    if c_count == 2 {
        let mut mem0 = preemph_mem[0];
        let mut mem1 = preemph_mem[1];
        for (output, (&sample0, &sample1)) in pcm
            .chunks_exact_mut(2)
            .zip(channels[0].iter().zip(&channels[1]))
        {
            let tmp0 = sample0 + 1e-30f32 + mem0;
            let tmp1 = sample1 + 1e-30f32 + mem1;
            mem0 = coef0 * tmp0;
            mem1 = coef0 * tmp1;
            output[0] = float_to_i16(tmp0 / CELT_SIG_SCALE);
            output[1] = float_to_i16(tmp1 / CELT_SIG_SCALE);
        }
        preemph_mem[0] = mem0;
        preemph_mem[1] = mem1;
        return Ok(());
    }

    for c in 0..c_count {
        let mut mem = preemph_mem[c];
        for j in 0..n {
            let tmp = channels[c][j] + 1e-30f32 + mem;
            mem = coef0 * tmp;
            pcm[j * c_count + c] = float_to_i16(tmp / CELT_SIG_SCALE);
        }
        preemph_mem[c] = mem;
    }
    Ok(())
}

pub fn deemphasis_interleaved_i24_into(
    mode: &CeltMode,
    channels: &[Vec<f32>],
    preemph_mem: &mut [f32],
    pcm: &mut Vec<i32>,
) -> Result<()> {
    if channels.is_empty() || channels.len() > 2 || preemph_mem.len() < channels.len() {
        return Err(Error::BadArg);
    }
    let n = channels[0].len();
    if channels.iter().any(|channel| channel.len() != n) {
        return Err(Error::BadArg);
    }

    let c_count = channels.len();
    let coef0 = mode.preemph[0];
    pcm.resize(n * c_count, 0);

    for c in 0..c_count {
        let mut mem = preemph_mem[c];
        for j in 0..n {
            let tmp = channels[c][j] + 1e-30f32 + mem;
            mem = coef0 * tmp;
            pcm[j * c_count + c] = float_to_i24(tmp / CELT_SIG_SCALE);
        }
        preemph_mem[c] = mem;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{deemphasis_interleaved_i16_into, CELT_SIG_SCALE};
    use crate::celt::mathops::float_to_i16;
    use crate::celt::modes::CeltMode;

    #[test]
    fn stereo_i16_deemphasis_matches_channel_major_reference() {
        let mode = CeltMode::standard_48k();
        let channels = vec![
            (0..240)
                .map(|index| 12_000.0 * (index as f32 * 0.071).sin())
                .collect::<Vec<_>>(),
            (0..240)
                .map(|index| 10_000.0 * (index as f32 * 0.053 + 0.4).cos())
                .collect::<Vec<_>>(),
        ];
        let initial_mem = [17.25f32, -31.5f32];
        let mut expected_mem = initial_mem;
        let mut expected_pcm = vec![0i16; channels[0].len() * 2];
        for channel in 0..2 {
            let mut mem = expected_mem[channel];
            for sample in 0..channels[channel].len() {
                let tmp = channels[channel][sample] + 1e-30f32 + mem;
                mem = mode.preemph[0] * tmp;
                expected_pcm[2 * sample + channel] = float_to_i16(tmp / CELT_SIG_SCALE);
            }
            expected_mem[channel] = mem;
        }

        let mut actual_mem = initial_mem;
        let mut actual_pcm = Vec::new();
        deemphasis_interleaved_i16_into(&mode, &channels, &mut actual_mem, &mut actual_pcm)
            .unwrap();

        assert_eq!(actual_pcm, expected_pcm);
        assert_eq!(actual_mem, expected_mem);
    }
}
