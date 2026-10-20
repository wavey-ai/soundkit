//! Whole-buffer wasm A/B entry points. `tools/wasm_ab/wasm_ab.c` exports the
//! same ABI for libopus, so `tools/run_wasm_ab.mjs` can time both modules on
//! identical PCM and identical packets.
//!
//! Packets are stored as a little-endian `u16` length followed by the packet.
//! PCM is interleaved stereo, signed 24-bit in sign-extended `i32`. With
//! `pcm_bits == 16`, the encoder input is `sample >> 8` and the decoder output
//! is the 16-bit sample shifted back up by 8 bits. `vbr` is 0 for CBR, 1 for
//! constrained VBR, and 2 for unconstrained VBR.

use soundkit_opus::{Application, Decoder, Encoder};

const CHANNELS: usize = 2;
const MAX_PACKET: usize = 1275;

#[no_mangle]
pub extern "C" fn ab_alloc(bytes: u32) -> *mut u8 {
    let mut buffer = vec![0u8; bytes as usize].into_boxed_slice();
    let ptr = buffer.as_mut_ptr();
    std::mem::forget(buffer);
    ptr
}

/// # Safety
/// `ptr` and `bytes` must come from one `ab_alloc` call.
#[no_mangle]
pub unsafe extern "C" fn ab_free(ptr: *mut u8, bytes: u32) {
    drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
        ptr,
        bytes as usize,
    )));
}

/// # Safety
/// `pcm` must hold `samples * 2` values and `out` must hold `cap` bytes.
#[no_mangle]
pub unsafe extern "C" fn ab_encode(
    pcm: *const i32,
    samples: u32,
    frame_size: u32,
    bitrate: u32,
    vbr: u32,
    pcm_bits: u32,
    out: *mut u8,
    cap: u32,
) -> i32 {
    let frame_size = frame_size as usize;
    let frames = samples as usize / frame_size;
    let pcm = std::slice::from_raw_parts(pcm, frames * frame_size * CHANNELS);
    let out = std::slice::from_raw_parts_mut(out, cap as usize);
    let Ok(mut encoder) = Encoder::with_application(48_000, CHANNELS, Application::Audio) else {
        return -1;
    };
    if encoder.set_bitrate(bitrate as i32).is_err()
        || encoder.set_vbr(vbr != 0).is_err()
        || encoder.set_vbr_constraint(vbr != 2).is_err()
    {
        return -1;
    }
    let mut packet = Vec::with_capacity(MAX_PACKET);
    let mut pcm16 = vec![0i16; frame_size * CHANNELS];
    let mut pos = 0usize;
    for frame in pcm.chunks_exact(frame_size * CHANNELS) {
        let encoded = if pcm_bits == 16 {
            for (dst, &src) in pcm16.iter_mut().zip(frame) {
                *dst = (src >> 8) as i16;
            }
            encoder.encode_i16_into(&pcm16, frame_size, &mut packet)
        } else {
            encoder.encode_i24_into(frame, frame_size, &mut packet)
        };
        let Ok(len) = encoded else {
            return -2;
        };
        if pos + 2 + len > out.len() {
            return -3;
        }
        out[pos..pos + 2].copy_from_slice(&(len as u16).to_le_bytes());
        out[pos + 2..pos + 2 + len].copy_from_slice(&packet[..len]);
        pos += 2 + len;
    }
    pos as i32
}

/// # Safety
/// `packets` must hold `len` bytes and `out` must hold every decoded sample.
#[no_mangle]
pub unsafe extern "C" fn ab_decode(
    packets: *const u8,
    len: u32,
    frame_size: u32,
    pcm_bits: u32,
    out: *mut i32,
    out_samples: u32,
) -> i32 {
    let packets = std::slice::from_raw_parts(packets, len as usize);
    let out = std::slice::from_raw_parts_mut(out, out_samples as usize * CHANNELS);
    let Ok(mut decoder) = Decoder::new(48_000, CHANNELS) else {
        return -1;
    };
    let frame_size = frame_size as usize;
    let mut pcm16 = vec![0i16; frame_size * CHANNELS];
    let mut pos = 0usize;
    let mut total = 0usize;
    while pos + 2 <= packets.len() {
        let n = u16::from_le_bytes([packets[pos], packets[pos + 1]]) as usize;
        pos += 2;
        let Some(packet) = packets.get(pos..pos + n) else {
            return -2;
        };
        pos += n;
        let Some(dst) = out.get_mut(total * CHANNELS..(total + frame_size) * CHANNELS) else {
            return -3;
        };
        let decoded = if pcm_bits == 16 {
            let result = decoder.decode_i16_into_slice(packet, false, &mut pcm16);
            for (d, &s) in dst.iter_mut().zip(&pcm16) {
                *d = (s as i32) << 8;
            }
            result
        } else {
            decoder.decode_i24_into_slice(packet, false, dst)
        };
        let Ok(decoded) = decoded else {
            return -4;
        };
        total += decoded;
    }
    total as i32
}

fn main() {}

/// Overrides one encoder decision knob; see `soundkit_opus::tuning`.
#[cfg(feature = "tuning")]
#[no_mangle]
pub extern "C" fn ab_tune(knob: u32, value: f32) {
    soundkit_opus::tuning::set(knob as usize, value);
}
