use crate::{Error, Result};

pub fn pcm_soft_clip(pcm: &mut [f32], channels: usize, softclip_mem: &mut [f32]) -> Result<()> {
    if channels == 0 || pcm.len() < channels || !pcm.len().is_multiple_of(channels) {
        return Err(Error::BadArg);
    }
    if softclip_mem.len() < channels {
        return Err(Error::BadArg);
    }

    let frame_size = pcm.len() / channels;
    for c in 0..channels {
        let mut a = softclip_mem[c];
        let mut i = 0usize;
        while i < frame_size {
            let idx = i * channels + c;
            let x = pcm[idx];
            if x * a >= 0.0 {
                break;
            }
            pcm[idx] = x + a * x * x;
            i += 1;
        }

        let mut curr = 0usize;
        let first = pcm[c];
        loop {
            while curr < frame_size {
                let x = pcm[curr * channels + c];
                if !(-1.0..=1.0).contains(&x) {
                    break;
                }
                curr += 1;
            }
            if curr == frame_size {
                a = 0.0;
                break;
            }

            let sign = pcm[curr * channels + c].signum();
            let mut start = curr;
            while start > 0 && pcm[(start - 1) * channels + c] * sign >= 0.0 {
                start -= 1;
            }
            let mut end = curr;
            let mut peak = curr;
            let mut maxval = pcm[curr * channels + c].abs().min(2.0);
            while end < frame_size && pcm[end * channels + c] * sign >= 0.0 {
                let idx = end * channels + c;
                pcm[idx] = pcm[idx].clamp(-2.0, 2.0);
                let abs = pcm[idx].abs();
                if abs > maxval {
                    maxval = abs;
                    peak = end;
                }
                end += 1;
            }

            if maxval > 1.0 {
                a = (maxval - 1.0) / (maxval * maxval);
                a += a * 2.4e-7;
                if pcm[peak * channels + c] > 0.0 {
                    a = -a;
                }
                for j in start..end {
                    let idx = j * channels + c;
                    let x = pcm[idx];
                    pcm[idx] = (x + a * x * x).clamp(-1.0, 1.0);
                }
            }

            if start == 0 && peak >= 2 {
                let offset = first - pcm[c];
                let delta = offset / peak as f32;
                let mut ramp = offset;
                for j in 0..peak {
                    ramp -= delta;
                    let idx = j * channels + c;
                    pcm[idx] = (pcm[idx] + ramp).clamp(-1.0, 1.0);
                }
            }
            curr = end;
            if curr == frame_size {
                break;
            }
        }
        softclip_mem[c] = a;
    }

    Ok(())
}
