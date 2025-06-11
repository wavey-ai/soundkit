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
    /// Encode i16 samples into a standalone MP3 `Vec<u8>`, automatically sizing the buffer.
    pub fn encode_to_vec(&mut self, samples: &[i16]) -> Result<Vec<u8>, String> {
        // reserve worst-case space for encode + flush
        let mut mp3 = Vec::new();
        mp3.reserve(max_required_buffer_size(samples.len()) + 7200);

        if self.channels == 1 {
            self.inner
                .encode_to_vec(MonoPcm(samples), &mut mp3)
                .map_err(|e| e.to_string())?;
        } else {
            self.inner
                .encode_to_vec(InterleavedPcm(samples), &mut mp3)
                .map_err(|e| e.to_string())?;
        }

        self.inner
            .flush_to_vec::<FlushNoGap>(&mut mp3)
            .map_err(|e| e.to_string())?;

        Ok(mp3)
    }

    /// Low-level flush if you used `encode_i16` directly.
    pub fn flush(&mut self, out: &mut Vec<u8>) -> Result<usize, String> {
        out.reserve(7200);
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
        let mp3 = self.encode_to_vec(input)?;
        let take = mp3.len().min(out_buf.len());
        out_buf[..take].copy_from_slice(&mp3[..take]);
        Ok(take)
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
    use mp3lame_encoder::max_required_buffer_size;
    use soundkit::audio_bytes::s16le_to_i16;
    use soundkit::wav::WavStreamProcessor;
    use std::fs;
    use std::path::Path;

    #[test]
    fn test_mp3_encoder_encode_i16() {
        // load a 16-bit WAV
        let data = fs::read("../testdata/s16le.wav").unwrap();
        let mut proc = WavStreamProcessor::new();
        let audio = proc.add(&data).unwrap().unwrap();
        let samples = s16le_to_i16(audio.data());

        // build the encoder
        let mut enc = Mp3Encoder::new(
            audio.sampling_rate(),
            audio.bits_per_sample() as u32,
            audio.channel_count() as u32,
            0,
            128_000,
        );
        enc.init().unwrap();

        // prepare an output buffer sized for the worst case
        let buf_size = max_required_buffer_size(samples.len()) + 7200;
        let mut out_buf = vec![0u8; buf_size];

        // encode into our buffer
        let written = enc.encode_i16(&samples, &mut out_buf).unwrap();
        assert!(written > 0, "no bytes were written");
        assert_eq!(out_buf[0], 0xFF, "MP3 frames should start with 0xFF");

        // write exactly the written bytes to disk for manual inspection
        let output_path = Path::new("../testdata/s16le.wav.mp3");
        fs::write(output_path, &out_buf[..written]).unwrap();
    }
}
