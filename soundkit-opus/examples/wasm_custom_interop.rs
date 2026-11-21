//! WebAssembly entry points for the Opus Custom interop harness.
//!
//! Build with `cargo build --release --target wasm32-unknown-unknown
//! --example wasm_custom_interop` (add `-C target-feature=+simd128` to
//! RUSTFLAGS for the SIMD build). `tools/run_wasm_custom_interop.mjs` loads
//! the module in Node.js and takes the arguments of the native
//! `opus_custom_interop` example, so the two produce files that can be
//! compared byte for byte.
//!
//! PCM and packet buffers use the formats of `tools/opus_custom_interop.c`.
//! `format` is 0 for s16, 1 for s24 (in `i32`), 2 for f32.

use soundkit_opus::{Decoder, Encoder};

#[no_mangle]
pub extern "C" fn ci_alloc(bytes: u32) -> *mut u8 {
    let mut buffer = vec![0u8; bytes as usize].into_boxed_slice();
    let ptr = buffer.as_mut_ptr();
    std::mem::forget(buffer);
    ptr
}

/// # Safety
/// `ptr` and `bytes` must come from one `ci_alloc` call.
#[no_mangle]
pub unsafe extern "C" fn ci_free(ptr: *mut u8, bytes: u32) {
    drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
        ptr,
        bytes as usize,
    )));
}

fn sample_bytes(format: u32) -> usize {
    if format == 0 {
        2
    } else {
        4
    }
}

/// Encodes whole frames of `pcm` and writes length-prefixed packets to
/// `out`. Returns the bytes written or a negative error.
///
/// # Safety
/// `pcm` must hold `pcm_bytes` bytes and `out` must hold `cap` bytes.
#[no_mangle]
pub unsafe extern "C" fn ci_encode(
    pcm: *const u8,
    pcm_bytes: u32,
    format: u32,
    frame: u32,
    channels: u32,
    bitrate: i32,
    vbr: u32,
    cvbr: u32,
    signalling: u32,
    max_bytes: u32,
    libopus_decisions: u32,
    libopus_arith: u32,
    portable_pvq: u32,
    out: *mut u8,
    cap: u32,
) -> i32 {
    let pcm = std::slice::from_raw_parts(pcm, pcm_bytes as usize);
    let out = std::slice::from_raw_parts_mut(out, cap as usize);
    let (frame, channels) = (frame as usize, channels as usize);
    let Ok(mut encoder) = Encoder::new_custom(48_000, channels, frame) else {
        return -1;
    };
    if encoder.set_custom_signalling(signalling != 0).is_err()
        || encoder.set_bitrate(bitrate).is_err()
        || encoder.set_vbr(vbr != 0).is_err()
        || encoder.set_vbr_constraint(cvbr != 0).is_err()
        || encoder.set_max_packet_bytes(max_bytes as usize).is_err()
    {
        return -1;
    }
    encoder.set_libopus_decisions(libopus_decisions != 0);
    if encoder.set_libopus_arithmetic(libopus_arith != 0).is_err()
        || encoder.set_portable_pvq(portable_pvq != 0).is_err()
    {
        return -1;
    }
    let frame_bytes = frame * channels * sample_bytes(format);
    let mut packet = Vec::new();
    let mut s16 = vec![0i16; frame * channels];
    let mut s32 = vec![0i32; frame * channels];
    let mut f32s = vec![0f32; frame * channels];
    let mut pos = 0usize;
    for chunk in pcm.chunks_exact(frame_bytes) {
        let result = match format {
            0 => {
                for (d, b) in s16.iter_mut().zip(chunk.chunks_exact(2)) {
                    *d = i16::from_le_bytes([b[0], b[1]]);
                }
                encoder.encode_i16_into(&s16, frame, &mut packet)
            }
            1 => {
                for (d, b) in s32.iter_mut().zip(chunk.chunks_exact(4)) {
                    *d = i32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                }
                encoder.encode_i24_into(&s32, frame, &mut packet)
            }
            _ => {
                for (d, b) in f32s.iter_mut().zip(chunk.chunks_exact(4)) {
                    *d = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                }
                encoder.encode_f32_into(&f32s, frame, &mut packet)
            }
        };
        let Ok(len) = result else {
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

/// Decodes length-prefixed packets (a zero length conceals a lost frame) and
/// writes PCM to `out`. Returns the bytes written or a negative error.
///
/// # Safety
/// `packets` must hold `len` bytes and `out` must hold `cap` bytes.
#[no_mangle]
pub unsafe extern "C" fn ci_decode(
    packets: *const u8,
    len: u32,
    format: u32,
    frame: u32,
    channels: u32,
    signalling: u32,
    libopus_arith: u32,
    out: *mut u8,
    cap: u32,
) -> i32 {
    let packets = std::slice::from_raw_parts(packets, len as usize);
    let out = std::slice::from_raw_parts_mut(out, cap as usize);
    let Ok(mut decoder) = Decoder::new_custom(48_000, channels as usize, frame as usize) else {
        return -1;
    };
    if decoder.set_custom_signalling(signalling != 0).is_err()
        || decoder.set_libopus_arithmetic(libopus_arith != 0).is_err()
    {
        return -1;
    }
    let mut s16 = Vec::new();
    let mut s32 = Vec::new();
    let mut f32s = Vec::new();
    let mut pos = 0usize;
    let mut written = 0usize;
    while pos + 2 <= packets.len() {
        let n = u16::from_le_bytes([packets[pos], packets[pos + 1]]) as usize;
        pos += 2;
        let Some(packet) = packets.get(pos..pos + n) else {
            return -2;
        };
        pos += n;
        let mut emit = |bytes: &[u8]| -> bool {
            let Some(dst) = out.get_mut(written..written + bytes.len()) else {
                return false;
            };
            dst.copy_from_slice(bytes);
            written += bytes.len();
            true
        };
        let ok = match format {
            0 => {
                decoder.decode_i16_into(packet, false, &mut s16).is_ok()
                    && s16.iter().all(|v| emit(&v.to_le_bytes()))
            }
            1 => {
                decoder.decode_i24_into(packet, false, &mut s32).is_ok()
                    && s32.iter().all(|v| emit(&v.to_le_bytes()))
            }
            _ => {
                decoder.decode_f32_into(packet, false, &mut f32s).is_ok()
                    && f32s.iter().all(|v| emit(&v.to_le_bytes()))
            }
        };
        if !ok {
            return -4;
        }
    }
    written as i32
}

fn main() {}
