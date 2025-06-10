use mp3lame_encoder::{
    max_required_buffer_size, Bitrate, Builder, FlushNoGap, InterleavedPcm, MonoPcm,
};
use soundkit::audio_packet::{Decoder, Encoder};
use std::vec::Vec;

pub struct Mp3Encoder {
    inner: mp3lame_encoder::Encoder,
    channels: u8,
}

impl Mp3Encoder {
    pub fn flush(&mut self, out: &mut Vec<u8>) -> Result<usize, String> {
        self.inner
            .flush_to_vec::<FlushNoGap>(out)
            .map_err(|e| e.to_string())
    }
}

impl Encoder for Mp3Encoder {
    fn new(
        sample_rate: u32,
        _bits_per_sample: u32,
        channels: u32,
        _frame_size: u32,
        bitrate: u32,
    ) -> Self {
        let mut builder = Builder::new().expect("lame_init");
        builder.set_sample_rate(sample_rate).unwrap();
        builder.set_num_channels(channels as u8).unwrap();
        let kbps = match bitrate {
            b if b >= 320_000 => Bitrate::Kbps320,
            b if b >= 256_000 => Bitrate::Kbps256,
            b if b >= 192_000 => Bitrate::Kbps192,
            b if b >= 128_000 => Bitrate::Kbps128,
            b if b >= 96_000 => Bitrate::Kbps96,
            _ => Bitrate::Kbps128,
        };
        builder.set_brate(kbps).unwrap();
        builder.set_to_write_vbr_tag(true).unwrap();
        let enc = builder.build().expect("lame_init_params");
        Mp3Encoder {
            inner: enc,
            channels: channels as u8,
        }
    }

    fn init(&mut self) -> Result<(), String> {
        Ok(())
    }

    fn encode_i16(&mut self, input: &[i16], out_buf: &mut [u8]) -> Result<usize, String> {
        let mut tmp_out = Vec::with_capacity(max_required_buffer_size(input.len()));

        let written = if self.channels == 1 {
            self.inner
                .encode_to_vec(MonoPcm(input), &mut tmp_out)
                .map_err(|e| e.to_string())?
        } else {
            self.inner
                .encode_to_vec(InterleavedPcm(input), &mut tmp_out)
                .map_err(|e| e.to_string())?
        };

        if written > out_buf.len() {
            return Err("Output buffer is too small".to_string());
        }

        out_buf[..written].copy_from_slice(&tmp_out[..written]);
        Ok(written)
    }

    fn encode_i32(&mut self, _input: &[i32], _out: &mut [u8]) -> Result<usize, String> {
        Err("not implemented".into())
    }

    fn reset(&mut self) -> Result<(), String> {
        Ok(())
    }
}

pub struct Mp3Decoder;
impl Decoder for Mp3Decoder {
    fn decode_i16(&mut self, _input: &[u8], _out: &mut [i16], _fec: bool) -> Result<usize, String> {
        Err("not implemented".into())
    }
    fn decode_i32(&mut self, _input: &[u8], _out: &mut [i32], _fec: bool) -> Result<usize, String> {
        Err("not implemented".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use soundkit::audio_bytes::s16le_to_i16;
    use soundkit::wav::WavStreamProcessor;
    use std::fs;
    use std::path::Path;

    #[test]
    fn test_mp3_encoder_16bit() {
        let data = fs::read("../testdata/s16le.wav").unwrap();
        let mut proc = WavStreamProcessor::new();
        let audio = proc.add(&data).unwrap().unwrap();
        let samples = s16le_to_i16(audio.data());
        let sample_rate = audio.sampling_rate();
        let channels = audio.channel_count() as u32;
        let bits = audio.bits_per_sample() as u32;
        let bitrate = 128_000;

        let mut enc = Mp3Encoder::new(sample_rate, bits, channels, 0, bitrate);
        enc.init().unwrap();

        let mut out = Vec::new();
        let mut encode_buf = vec![0u8; max_required_buffer_size(samples.len())];

        let written = enc.encode_i16(&samples, &mut encode_buf).unwrap();
        out.extend_from_slice(&encode_buf[..written]);

        enc.flush(&mut out).unwrap();

        assert!(!out.is_empty(), "Output buffer should not be empty");
        assert_eq!(out[0], 0xFF, "First byte of an MP3 frame should be 0xFF");

        let output_path = Path::new("../testdata/s16le.wav.mp3");
        fs::write(output_path, &out).expect("Failed to write test MP3 file");
    }
}
