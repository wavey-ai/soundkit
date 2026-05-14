// G.726-32 is implemented as a Rust port of the Sun Microsystems G.721
// reference code, whose header permits unrestricted use, copying, and
// modification without charge. No SpanDSP/libg726 C dependency is used here.

use soundkit::audio_packet::{Decoder, Encoder};

pub const G726_SAMPLE_RATE: u32 = 8_000;
pub const G726_CHANNELS: u8 = 1;

const G726_BITS_PER_SAMPLE: usize = 4;
const G726_SAMPLES_PER_BYTE: usize = 2;
const G726_32K_BIT_RATE: u32 = 32_000;

const POWER2: [i32; 15] = [
    1 << 0,
    1 << 1,
    1 << 2,
    1 << 3,
    1 << 4,
    1 << 5,
    1 << 6,
    1 << 7,
    1 << 8,
    1 << 9,
    1 << 10,
    1 << 11,
    1 << 12,
    1 << 13,
    1 << 14,
];

const Q_TAB_721: [i32; 7] = [-124, 80, 178, 246, 300, 349, 400];
const DQLN_TAB_721: [i32; 16] = [
    -2048, 4, 135, 213, 273, 323, 373, 425, 425, 373, 323, 273, 213, 135, 4, -2048,
];
const WI_TAB_721: [i32; 16] = [
    -12, 18, 41, 64, 112, 198, 355, 1122, 1122, 355, 198, 112, 64, 41, 18, -12,
];
const FI_TAB_721: [i32; 16] = [
    0, 0, 0, 0x200, 0x200, 0x200, 0x600, 0xE00, 0xE00, 0x600, 0x200, 0x200, 0x200, 0, 0, 0,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum G726Rate {
    /// G.726-32, also known as the 4-bit G.721 ADPCM profile.
    Rate32000,
}

impl G726Rate {
    pub fn from_bitrate(_bit_rate: u32) -> Self {
        Self::Rate32000
    }

    pub fn bits_per_sample(self) -> usize {
        match self {
            Self::Rate32000 => G726_BITS_PER_SAMPLE,
        }
    }

    pub fn bit_rate(self) -> u32 {
        match self {
            Self::Rate32000 => G726_32K_BIT_RATE,
        }
    }

    fn samples_per_byte_group(self) -> usize {
        match self {
            Self::Rate32000 => G726_SAMPLES_PER_BYTE,
        }
    }

    fn bytes_per_group(self) -> usize {
        match self {
            Self::Rate32000 => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum G726Packing {
    /// G.726 left-justified raw packing (`ffmpeg -f g726`).
    Left,
    /// G.726 little-endian/right-justified raw packing (`ffmpeg -f g726le`).
    Right,
}

#[derive(Clone, Debug)]
struct G726Core {
    yl: i32,
    yu: i32,
    dms: i32,
    dml: i32,
    ap: i32,
    a: [i32; 2],
    b: [i32; 6],
    pk: [i32; 2],
    dq: [i32; 6],
    sr: [i32; 2],
    td: i32,
}

impl Default for G726Core {
    fn default() -> Self {
        Self {
            yl: 34_816,
            yu: 544,
            dms: 0,
            dml: 0,
            ap: 0,
            a: [0; 2],
            b: [0; 6],
            pk: [0; 2],
            dq: [32; 6],
            sr: [32; 2],
            td: 0,
        }
    }
}

impl G726Core {
    fn encode_sample(&mut self, sample: i16) -> u8 {
        // G.726-32 is the 4-bit G.721 algorithm. This Rust port follows the
        // unrestricted-use Sun Microsystems reference implementation.
        let sl = i32::from(sample) >> 2;
        let sezi = self.predictor_zero();
        let sez = sezi >> 1;
        let se = (sezi + self.predictor_pole()) >> 1;
        let d = sl - se;
        let y = self.step_size();
        let i = quantize(d, y) as usize;
        let dq = reconstruct((i & 0x08) != 0, DQLN_TAB_721[i], y);
        let sr = if dq < 0 { se - (dq & 0x3FFF) } else { se + dq };
        let dqsez = sr + sez - se;
        self.update(y, WI_TAB_721[i] << 5, FI_TAB_721[i], dq, sr, dqsez);
        (i as u8) & 0x0F
    }

    fn decode_code(&mut self, code: u8) -> i16 {
        let i = usize::from(code & 0x0F);
        let sezi = self.predictor_zero();
        let sez = sezi >> 1;
        let sei = sezi + self.predictor_pole();
        let se = sei >> 1;
        let y = self.step_size();
        let dq = reconstruct((i & 0x08) != 0, DQLN_TAB_721[i], y);
        let sr = if dq < 0 { se - (dq & 0x3FFF) } else { se + dq };
        let dqsez = sr - se + sez;
        self.update(y, WI_TAB_721[i] << 5, FI_TAB_721[i], dq, sr, dqsez);
        (sr << 2).clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
    }

    fn predictor_zero(&self) -> i32 {
        self.b
            .iter()
            .zip(self.dq.iter())
            .map(|(&b, &dq)| fmult(b >> 2, dq))
            .sum()
    }

    fn predictor_pole(&self) -> i32 {
        fmult(self.a[1] >> 2, self.sr[1]) + fmult(self.a[0] >> 2, self.sr[0])
    }

    fn step_size(&self) -> i32 {
        if self.ap >= 256 {
            return self.yu;
        }

        let y = self.yl >> 6;
        let dif = self.yu - y;
        let al = self.ap >> 2;
        if dif > 0 {
            y + ((dif * al) >> 6)
        } else if dif < 0 {
            y + ((dif * al + 0x3F) >> 6)
        } else {
            y
        }
    }

    fn update(&mut self, y: i32, wi: i32, fi: i32, dq: i32, sr: i32, dqsez: i32) {
        let pk0 = i32::from(dqsez < 0);
        let mag = dq & 0x7FFF;

        let ylint = self.yl >> 15;
        let ylfrac = (self.yl >> 10) & 0x1F;
        let thr1 = (32 + ylfrac) << ylint;
        let thr2 = if ylint > 9 { 31 << 10 } else { thr1 };
        let dqthr = (thr2 + (thr2 >> 1)) >> 1;

        let tr = i32::from(self.td != 0 && mag > dqthr);

        self.yu = y + ((wi - y) >> 5);
        self.yu = self.yu.clamp(544, 5120);
        self.yl += self.yu + ((-self.yl) >> 6);

        let a2p = if tr != 0 {
            self.a = [0; 2];
            self.b = [0; 6];
            0
        } else {
            let pks1 = pk0 ^ self.pk[0];
            let mut a2p = self.a[1] - (self.a[1] >> 7);

            if dqsez != 0 {
                let fa1 = if pks1 != 0 { self.a[0] } else { -self.a[0] };
                if fa1 < -8191 {
                    a2p -= 0x100;
                } else if fa1 > 8191 {
                    a2p += 0xFF;
                } else {
                    a2p += fa1 >> 5;
                }

                if (pk0 ^ self.pk[1]) != 0 {
                    if a2p <= -12160 {
                        a2p = -12288;
                    } else if a2p >= 12416 {
                        a2p = 12288;
                    } else {
                        a2p -= 0x80;
                    }
                } else if a2p <= -12416 {
                    a2p = -12288;
                } else if a2p >= 12160 {
                    a2p = 12288;
                } else {
                    a2p += 0x80;
                }
            }

            self.a[1] = a2p;

            self.a[0] -= self.a[0] >> 8;
            if dqsez != 0 {
                if pks1 == 0 {
                    self.a[0] += 192;
                } else {
                    self.a[0] -= 192;
                }
            }

            let a1ul = 15360 - a2p;
            self.a[0] = self.a[0].clamp(-a1ul, a1ul);

            for index in 0..6 {
                self.b[index] -= self.b[index] >> 8;
                if (dq & 0x7FFF) != 0 {
                    if (dq ^ self.dq[index]) >= 0 {
                        self.b[index] += 128;
                    } else {
                        self.b[index] -= 128;
                    }
                }
            }

            a2p
        };

        for index in (1..6).rev() {
            self.dq[index] = self.dq[index - 1];
        }
        self.dq[0] = if mag == 0 {
            if dq >= 0 {
                0x20
            } else {
                -0x3E0
            }
        } else {
            let exp = quan(mag, &POWER2) as i32;
            let mant = (mag << 6) >> exp;
            let value = (exp << 6) + mant;
            if dq >= 0 {
                value
            } else {
                value - 0x400
            }
        };

        self.sr[1] = self.sr[0];
        self.sr[0] = if sr == 0 {
            0x20
        } else if sr > 0 {
            let exp = quan(sr, &POWER2) as i32;
            (exp << 6) + ((sr << 6) >> exp)
        } else if sr > -32768 {
            let mag = -sr;
            let exp = quan(mag, &POWER2) as i32;
            (exp << 6) + ((mag << 6) >> exp) - 0x400
        } else {
            -0x3E0
        };

        self.pk[1] = self.pk[0];
        self.pk[0] = pk0;

        self.td = if tr != 0 { 0 } else { i32::from(a2p < -11776) };

        self.dms += (fi - self.dms) >> 5;
        self.dml += ((fi << 2) - self.dml) >> 7;

        if tr != 0 {
            self.ap = 256;
        } else if y < 1536 || self.td != 0 || ((self.dms << 2) - self.dml).abs() >= (self.dml >> 3)
        {
            self.ap += (0x200 - self.ap) >> 4;
        } else {
            self.ap += -self.ap >> 4;
        }
    }
}

fn quan(value: i32, table: &[i32]) -> usize {
    table
        .iter()
        .position(|&threshold| value < threshold)
        .unwrap_or(table.len())
}

fn fmult(an: i32, srn: i32) -> i32 {
    let anmag = if an > 0 { an } else { (-an) & 0x1FFF };
    let anexp = quan(anmag, &POWER2) as i32 - 6;
    let anmant = if anmag == 0 {
        32
    } else if anexp >= 0 {
        anmag >> anexp
    } else {
        anmag << -anexp
    };
    let wanexp = anexp + ((srn >> 6) & 0x0F) - 13;
    let wanmant = (anmant * (srn & 0x3F) + 0x30) >> 4;
    let retval = if wanexp >= 0 {
        (wanmant << wanexp) & 0x7FFF
    } else {
        wanmant >> -wanexp
    };

    if (an ^ srn) < 0 {
        -retval
    } else {
        retval
    }
}

fn quantize(d: i32, y: i32) -> u8 {
    let dqm = d.abs();
    let exp = quan(dqm >> 1, &POWER2) as i32;
    let mant = ((dqm << 7) >> exp) & 0x7F;
    let dl = (exp << 7) + mant;
    let dln = dl - (y >> 2);
    let i = quan(dln, &Q_TAB_721);

    if d < 0 {
        (15 - i) as u8
    } else if i == 0 {
        15
    } else {
        i as u8
    }
}

fn reconstruct(sign: bool, dqln: i32, y: i32) -> i32 {
    let dql = dqln + (y >> 2);
    if dql < 0 {
        if sign {
            -0x8000
        } else {
            0
        }
    } else {
        let dex = (dql >> 7) & 15;
        let dqt = 128 + (dql & 127);
        let dq = (dqt << 7) >> (14 - dex);
        if sign {
            dq - 0x8000
        } else {
            dq
        }
    }
}

fn pack_codes(first: u8, second: u8, packing: G726Packing) -> u8 {
    match packing {
        G726Packing::Left => ((first & 0x0F) << 4) | (second & 0x0F),
        G726Packing::Right => (first & 0x0F) | ((second & 0x0F) << 4),
    }
}

fn unpack_codes(byte: u8, packing: G726Packing) -> [u8; 2] {
    match packing {
        G726Packing::Left => [byte >> 4, byte & 0x0F],
        G726Packing::Right => [byte & 0x0F, byte >> 4],
    }
}

pub struct G726Encoder {
    core: G726Core,
    rate: G726Rate,
    packing: G726Packing,
    pending_samples: Vec<i16>,
}

impl G726Encoder {
    pub fn try_new(rate: G726Rate, packing: G726Packing) -> Result<Self, String> {
        Ok(Self {
            core: G726Core::default(),
            rate,
            packing,
            pending_samples: Vec::with_capacity(rate.samples_per_byte_group()),
        })
    }

    pub fn new_32k_left() -> Self {
        Self::try_new(G726Rate::Rate32000, G726Packing::Left)
            .expect("failed to initialize G.726 encoder")
    }

    pub fn new_32k_right() -> Self {
        Self::try_new(G726Rate::Rate32000, G726Packing::Right)
            .expect("failed to initialize G.726 encoder")
    }

    pub fn encode_to_vec(&mut self, input: &[i16], output: &mut Vec<u8>) -> Result<usize, String> {
        let start = output.len();
        let required = self.required_encode_bytes(input.len(), false);
        output.resize(start + required, 0);
        let written = self.encode_i16(input, &mut output[start..])?;
        output.truncate(start + written);
        Ok(written)
    }

    pub fn flush_into(&mut self, output: &mut [u8]) -> Result<usize, String> {
        let required = self.required_encode_bytes(0, true);
        if output.len() < required {
            return Err(format!(
                "Output buffer too small for G.726 flush: need {}, have {}",
                required,
                output.len()
            ));
        }

        if self.pending_samples.is_empty() {
            return Ok(0);
        }

        let mut samples = self.pending_samples.clone();
        samples.resize(self.rate.samples_per_byte_group(), 0);
        self.pending_samples.clear();
        Ok(self.encode_complete_samples(&samples, output))
    }

    pub fn flush_to_vec(&mut self, output: &mut Vec<u8>) -> Result<usize, String> {
        let start = output.len();
        let required = self.required_encode_bytes(0, true);
        output.resize(start + required, 0);
        let written = self.flush_into(&mut output[start..])?;
        output.truncate(start + written);
        Ok(written)
    }

    pub fn rate(&self) -> G726Rate {
        self.rate
    }

    pub fn packing(&self) -> G726Packing {
        self.packing
    }

    fn required_encode_bytes(&self, input_samples: usize, flush: bool) -> usize {
        let group_samples = self.rate.samples_per_byte_group();
        let total_samples = self.pending_samples.len() + input_samples;
        let groups = if flush {
            total_samples.div_ceil(group_samples)
        } else {
            total_samples / group_samples
        };
        groups * self.rate.bytes_per_group()
    }

    fn encode_complete_samples(&mut self, samples: &[i16], output: &mut [u8]) -> usize {
        let mut written = 0;
        for pair in samples.chunks_exact(G726_SAMPLES_PER_BYTE) {
            let first = self.core.encode_sample(pair[0]);
            let second = self.core.encode_sample(pair[1]);
            output[written] = pack_codes(first, second, self.packing);
            written += 1;
        }
        written
    }
}

impl Default for G726Encoder {
    fn default() -> Self {
        Self::new_32k_left()
    }
}

impl Encoder for G726Encoder {
    fn new(
        _sample_rate: u32,
        _bits_per_sample: u32,
        _channels: u32,
        _frame_size: u32,
        bitrate: u32,
    ) -> Self {
        Self::try_new(G726Rate::from_bitrate(bitrate), G726Packing::Left)
            .expect("failed to initialize G.726 encoder")
    }

    fn init(&mut self) -> Result<(), String> {
        Ok(())
    }

    fn encode_i16(&mut self, input: &[i16], output: &mut [u8]) -> Result<usize, String> {
        let required = self.required_encode_bytes(input.len(), false);
        if output.len() < required {
            return Err(format!(
                "Output buffer too small for G.726 encode: need {}, have {}",
                required,
                output.len()
            ));
        }

        let group_samples = self.rate.samples_per_byte_group();
        let total_samples = self.pending_samples.len() + input.len();
        let complete_samples = (total_samples / group_samples) * group_samples;

        let mut samples = Vec::with_capacity(total_samples);
        samples.extend_from_slice(&self.pending_samples);
        samples.extend_from_slice(input);

        let written = if complete_samples == 0 {
            0
        } else {
            self.encode_complete_samples(&samples[..complete_samples], output)
        };

        self.pending_samples.clear();
        self.pending_samples
            .extend_from_slice(&samples[complete_samples..]);

        Ok(written)
    }

    fn encode_i32(&mut self, input: &[i32], output: &mut [u8]) -> Result<usize, String> {
        let samples: Vec<i16> = input.iter().map(|sample| (sample >> 16) as i16).collect();
        self.encode_i16(&samples, output)
    }

    fn reset(&mut self) -> Result<(), String> {
        self.core = G726Core::default();
        self.pending_samples.clear();
        Ok(())
    }
}

pub struct G726Decoder {
    core: G726Core,
    rate: G726Rate,
    packing: G726Packing,
}

impl G726Decoder {
    pub fn try_new(rate: G726Rate, packing: G726Packing) -> Result<Self, String> {
        Ok(Self {
            core: G726Core::default(),
            rate,
            packing,
        })
    }

    pub fn new_32k_left() -> Self {
        Self::try_new(G726Rate::Rate32000, G726Packing::Left)
            .expect("failed to initialize G.726 decoder")
    }

    pub fn new_32k_right() -> Self {
        Self::try_new(G726Rate::Rate32000, G726Packing::Right)
            .expect("failed to initialize G.726 decoder")
    }

    pub fn sample_rate(&self) -> u32 {
        G726_SAMPLE_RATE
    }

    pub fn channels(&self) -> u8 {
        G726_CHANNELS
    }

    pub fn rate(&self) -> G726Rate {
        self.rate
    }

    pub fn packing(&self) -> G726Packing {
        self.packing
    }

    pub fn flush(&mut self) -> Result<(), String> {
        Ok(())
    }
}

impl Default for G726Decoder {
    fn default() -> Self {
        Self::new_32k_left()
    }
}

impl Decoder for G726Decoder {
    fn decode_i16(
        &mut self,
        input: &[u8],
        output: &mut [i16],
        _fec: bool,
    ) -> Result<usize, String> {
        let required = input.len() * self.rate.samples_per_byte_group();
        if output.len() < required {
            return Err(format!(
                "Output buffer too small for G.726 decode: need {}, have {}",
                required,
                output.len()
            ));
        }

        let mut written = 0;
        for &byte in input {
            for code in unpack_codes(byte, self.packing) {
                output[written] = self.core.decode_code(code);
                written += 1;
            }
        }
        Ok(written)
    }

    fn decode_i32(
        &mut self,
        input: &[u8],
        output: &mut [i32],
        _fec: bool,
    ) -> Result<usize, String> {
        let required = input.len() * self.rate.samples_per_byte_group();
        if output.len() < required {
            return Err(format!(
                "Output buffer too small for G.726 decode: need {}, have {}",
                required,
                output.len()
            ));
        }

        let mut samples = vec![0i16; required];
        let written = self.decode_i16(input, &mut samples, false)?;
        for (dst, sample) in output.iter_mut().zip(samples.into_iter()).take(written) {
            *dst = i32::from(sample) << 16;
        }
        Ok(written)
    }

    fn decode_f32(
        &mut self,
        input: &[u8],
        output: &mut [f32],
        _fec: bool,
    ) -> Result<usize, String> {
        let required = input.len() * self.rate.samples_per_byte_group();
        if output.len() < required {
            return Err(format!(
                "Output buffer too small for G.726 decode: need {}, have {}",
                required,
                output.len()
            ));
        }

        let mut samples = vec![0i16; required];
        let written = self.decode_i16(input, &mut samples, false)?;
        for (dst, sample) in output.iter_mut().zip(samples.into_iter()).take(written) {
            *dst = f32::from(sample) / 32768.0;
        }
        Ok(written)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::process::Command;

    fn testdata_path(file: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("testdata")
            .join(file)
    }

    fn golden_path(file: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("golden")
            .join(file)
    }

    fn read_s16le_fixture(path: &str) -> Vec<i16> {
        let bytes = fs::read(testdata_path(path)).unwrap();
        assert!(!bytes.is_empty(), "{path} missing or empty");
        bytes
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
            .collect()
    }

    fn samples() -> Vec<i16> {
        (0..397)
            .map(|index| {
                let phase = index as f32 / 80.0 * std::f32::consts::TAU;
                (phase.sin() * 10_000.0) as i16
            })
            .collect()
    }

    fn encode_samples(input: &[i16], packing: G726Packing) -> Vec<u8> {
        let mut encoder = G726Encoder::try_new(G726Rate::Rate32000, packing).unwrap();
        let mut encoded = Vec::new();
        for chunk in input.chunks(37) {
            encoder.encode_to_vec(chunk, &mut encoded).unwrap();
        }
        encoder.flush_to_vec(&mut encoded).unwrap();
        encoded
    }

    #[test]
    fn streaming_encoder_matches_padded_whole_encode() {
        let input = samples();
        let mut padded = input.clone();
        padded.resize(
            input.len().div_ceil(G726_SAMPLES_PER_BYTE) * G726_SAMPLES_PER_BYTE,
            0,
        );

        let mut whole_encoder = G726Encoder::new_32k_left();
        let mut whole = vec![0u8; padded.len() / G726_SAMPLES_PER_BYTE];
        let whole_len = whole_encoder.encode_i16(&padded, &mut whole).unwrap();
        whole.truncate(whole_len);

        let mut stream_encoder = G726Encoder::new_32k_left();
        let mut chunked = Vec::new();
        let mut scratch = [0u8; 64];
        for chunk in input.chunks(37) {
            let written = stream_encoder.encode_i16(chunk, &mut scratch).unwrap();
            chunked.extend_from_slice(&scratch[..written]);
        }
        stream_encoder.flush_to_vec(&mut chunked).unwrap();

        assert_eq!(chunked, whole);
    }

    #[test]
    fn streaming_decoder_matches_whole_decode() {
        let encoded = encode_samples(&samples(), G726Packing::Left);

        let mut whole_decoder = G726Decoder::new_32k_left();
        let mut whole = vec![0i16; encoded.len() * 2];
        let whole_len = whole_decoder
            .decode_i16(&encoded, &mut whole, false)
            .unwrap();
        whole.truncate(whole_len);

        let mut stream_decoder = G726Decoder::new_32k_left();
        let mut chunked = Vec::new();
        let mut scratch = [0i16; 64];
        for chunk in encoded.chunks(19) {
            let written = stream_decoder
                .decode_i16(chunk, &mut scratch, false)
                .unwrap();
            chunked.extend_from_slice(&scratch[..written]);
        }
        stream_decoder.flush().unwrap();

        assert_eq!(chunked, whole);
    }

    #[test]
    fn right_packing_round_trips_with_matching_decoder() {
        let encoded = encode_samples(&samples(), G726Packing::Right);

        let mut right_decoder = G726Decoder::new_32k_right();
        let mut decoded = vec![0i16; encoded.len() * 2];
        let decoded_len = right_decoder
            .decode_i16(&encoded, &mut decoded, false)
            .unwrap();
        decoded.truncate(decoded_len);

        assert_eq!(decoded.len(), encoded.len() * 2);
        assert!(decoded.iter().any(|&sample| sample != 0));
    }

    #[test]
    fn decoder_trait_supports_i16_i32_and_f32_output() {
        let encoded = encode_samples(&samples(), G726Packing::Left);
        let sample_count = encoded.len() * 2;
        let mut decoder_i16 = G726Decoder::new_32k_left();
        let mut decoder_i32 = G726Decoder::new_32k_left();
        let mut decoder_f32 = G726Decoder::new_32k_left();
        let mut i16_out = vec![0i16; sample_count];
        let mut i32_out = vec![0i32; sample_count];
        let mut f32_out = vec![0.0f32; sample_count];

        let i16_len = decoder_i16
            .decode_i16(&encoded, &mut i16_out, false)
            .unwrap();
        let i32_len = decoder_i32
            .decode_i32(&encoded, &mut i32_out, false)
            .unwrap();
        let f32_len = decoder_f32
            .decode_f32(&encoded, &mut f32_out, false)
            .unwrap();

        assert_eq!(i16_len, i32_len);
        assert_eq!(i16_len, f32_len);
        for i in 0..i16_len {
            assert_eq!(i32_out[i], i32::from(i16_out[i]) << 16);
            assert!((f32_out[i] - f32::from(i16_out[i]) / 32768.0).abs() < f32::EPSILON);
        }
    }

    #[test]
    #[ignore = "regenerates the committed G.726 fixture using ffmpeg"]
    fn generate_g726_fixture_with_ffmpeg() {
        let input = testdata_path("linear16_8/A_Tusk_is_used_to_make_costly_gifts.s16le");
        let output = testdata_path("g726/A_Tusk_is_used_to_make_costly_gifts.g726");
        fs::create_dir_all(output.parent().unwrap()).unwrap();
        let status = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "s16le",
                "-ar",
                "8000",
                "-ac",
                "1",
                "-i",
            ])
            .arg(&input)
            .args(["-c:a", "g726", "-code_size", "4", "-f", "g726"])
            .arg(&output)
            .status()
            .unwrap();
        assert!(status.success());
    }

    #[test]
    fn decode_g726_fixture_and_write_golden_wav() {
        let fixture = fs::read(testdata_path(
            "g726/A_Tusk_is_used_to_make_costly_gifts.g726",
        ))
        .unwrap();
        assert!(!fixture.is_empty(), "G.726 fixture missing or empty");

        let mut decoder = G726Decoder::new_32k_left();
        let mut decoded = Vec::new();
        let mut scratch = [0i16; 254];

        for chunk in fixture.chunks(127) {
            let written = decoder.decode_i16(chunk, &mut scratch, false).unwrap();
            decoded.extend_from_slice(&scratch[..written]);
        }
        decoder.flush().unwrap();

        assert_eq!(decoded.len(), fixture.len() * 2);
        assert!(decoded.iter().any(|&sample| sample != 0));

        let wav = soundkit::wav::generate_wav_buffer(
            &soundkit::audio_types::PcmData::I16(vec![decoded]),
            G726_SAMPLE_RATE,
        )
        .unwrap();
        let output_path = golden_path("g726/A_Tusk_is_used_to_make_costly_gifts.decoded.wav");
        fs::create_dir_all(output_path.parent().unwrap()).unwrap();
        fs::write(output_path, wav).unwrap();
    }

    #[test]
    fn ffmpeg_can_decode_g726_fixture() {
        let input = testdata_path("g726/A_Tusk_is_used_to_make_costly_gifts.g726");
        let output = std::env::temp_dir().join("soundkit-g726-fixture.s16le");
        let status = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "g726",
                "-ar",
                "8000",
                "-ac",
                "1",
                "-i",
            ])
            .arg(&input)
            .args(["-f", "s16le", "-acodec", "pcm_s16le"])
            .arg(&output)
            .status()
            .unwrap();
        assert!(status.success());

        let decoded = fs::read(output).unwrap();
        assert!(decoded
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
            .any(|sample| sample != 0));
    }

    #[test]
    fn ffmpeg_can_decode_native_encoder_output() {
        let samples = read_s16le_fixture("linear16_8/A_Tusk_is_used_to_make_costly_gifts.s16le");
        let encoded = encode_samples(&samples, G726Packing::Left);
        let input = std::env::temp_dir().join("soundkit-g726-native.g726");
        let output = std::env::temp_dir().join("soundkit-g726-native.s16le");
        fs::write(&input, encoded).unwrap();

        let status = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "g726",
                "-ar",
                "8000",
                "-ac",
                "1",
                "-i",
            ])
            .arg(&input)
            .args(["-f", "s16le", "-acodec", "pcm_s16le"])
            .arg(&output)
            .status()
            .unwrap();
        assert!(status.success());

        let decoded = fs::read(output).unwrap();
        assert!(decoded
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
            .any(|sample| sample != 0));
    }

    #[test]
    #[ignore = "writes a native encoder comparison fixture"]
    fn generate_g726_fixture_with_native_encoder() {
        let samples = read_s16le_fixture("linear16_8/A_Tusk_is_used_to_make_costly_gifts.s16le");
        let encoded = encode_samples(&samples, G726Packing::Left);

        let output_path = testdata_path("g726/A_Tusk_is_used_to_make_costly_gifts.native.g726");
        fs::create_dir_all(output_path.parent().unwrap()).unwrap();
        fs::write(output_path, encoded).unwrap();
    }
}
