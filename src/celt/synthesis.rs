//! CELT synthesis and deemphasis helpers ported from the official
//! `celt/celt_decoder.c` floating-point path.

use crate::celt::bands::denormalise_bands;
use crate::celt::mdct::clt_mdct_backward;
use crate::celt::modes::CeltMode;
use crate::{Error, Result};

const CELT_SIG_SCALE: f32 = 32_768.0;

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
    if lm > mode.max_lm
        || start > eff_end
        || eff_end > mode.nb_ebands
        || !(1..=2).contains(&channels)
        || downsample == 0
        || overlap_mem.len() < channels
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

    let mut out = Vec::with_capacity(channels);
    for c in 0..channels {
        let norm = if c == 0 {
            x
        } else {
            y.expect("right channel validated")
        };
        let log_e = &band_log_e[c * mode.nb_ebands..(c + 1) * mode.nb_ebands];
        let mut freq = vec![0.0f32; n];
        denormalise_bands(
            mode, norm, &mut freq, log_e, start, eff_end, m, downsample, silence,
        );

        let mut channel = vec![0.0f32; n + mode.overlap];
        channel[..mode.overlap].copy_from_slice(&overlap_mem[c][..mode.overlap]);
        for block in 0..blocks {
            clt_mdct_backward(
                &mode.mdct,
                &freq[block..],
                &mut channel[block_len * block..],
                &mode.window,
                mode.overlap,
                shift,
                blocks,
            );
        }
        overlap_mem[c][..mode.overlap].copy_from_slice(&channel[n..n + mode.overlap]);
        channel.truncate(n);
        out.push(channel);
    }

    Ok(out)
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

    let c_count = channels.len();
    let coef0 = mode.preemph[0];
    let mut pcm = vec![0.0f32; n * c_count];
    for c in 0..c_count {
        let mut mem = preemph_mem[c];
        for j in 0..n {
            let tmp = channels[c][j] + 1e-30f32 + mem;
            mem = coef0 * tmp;
            pcm[j * c_count + c] = tmp / CELT_SIG_SCALE;
        }
        preemph_mem[c] = mem;
    }
    Ok(pcm)
}
