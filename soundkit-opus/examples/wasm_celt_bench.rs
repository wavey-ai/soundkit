use soundkit_opus::{Application, Encoder};

const SAMPLE_RATE: i32 = 48_000;
const CHANNELS: usize = 2;

fn centered_u16(value: u32) -> f32 {
    ((value & 0xffff) as i32 - 32_768) as f32 * (1.0 / 32_768.0)
}

fn triangle_wave(phase: u32) -> f32 {
    let p = (phase & 0xffff) as i32;
    let v = if p < 32_768 { p - 16_384 } else { 49_152 - p };
    v as f32 * (1.0 / 16_384.0)
}

fn fill_frame(pcm: &mut [f32], frame: usize, frame_size: usize) {
    let mut noise = 0x1234_5678u32.wrapping_add(frame as u32);
    for i in 0..frame_size {
        let t = frame * frame_size + i;
        noise = noise.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let tri_a = triangle_wave((t as u32).wrapping_mul(713));
        let tri_b = triangle_wave((t as u32).wrapping_mul(1451).wrapping_add(0x4000));
        let tri_c = triangle_wave((t as u32).wrapping_mul(977).wrapping_add(0x2000));
        let tri_d = triangle_wave((t as u32).wrapping_mul(3511).wrapping_add(0x6000));
        let n = centered_u16(noise) * (1.0 / 4096.0);
        let pulse = (t as u32) & 8191;
        let transient = if pulse < 64 {
            (64 - pulse) as f32 * (1.0 / 512.0)
        } else {
            0.0
        };
        pcm[2 * i] = (0.25 * tri_a + 0.125 * tri_b + n + transient).clamp(-1.0, 1.0);
        pcm[2 * i + 1] = (0.21875 * tri_c - 0.09375 * tri_d - n - 0.5 * transient).clamp(-1.0, 1.0);
    }
}

#[no_mangle]
pub extern "C" fn raw_celt_encode_bench(frames: u32, frame_size: u32, bitrate: u32) -> u32 {
    let frames = frames.max(1) as usize;
    let frame_size = frame_size.clamp(120, 960) as usize;
    let mut encoder =
        Encoder::new(SAMPLE_RATE, CHANNELS, Application::Audio).expect("encoder init");
    encoder.set_bitrate(bitrate as i32).expect("set bitrate");
    encoder.set_vbr(false).expect("set vbr");

    let mut pcm = vec![0.0f32; frame_size * CHANNELS];
    let mut checksum = 0u32;
    for frame in 0..frames {
        fill_frame(&mut pcm, frame, frame_size);
        let packet = encoder.encode_f32(&pcm, frame_size).expect("encode");
        checksum = checksum
            .wrapping_add(packet.len() as u32)
            .wrapping_add((packet.first().copied().unwrap_or(0) as u32) << 8)
            .wrapping_add(packet.last().copied().unwrap_or(0) as u32);
    }
    checksum
}

fn main() {}
