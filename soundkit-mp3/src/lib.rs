use mp3lame_encoder::{
    max_required_buffer_size, Bitrate, Builder, FlushNoGap, InterleavedPcm, MonoPcm,
};
use nanomp3::{Decoder as NanoDecoder, FrameInfo, MAX_SAMPLES_PER_FRAME};
use soundkit::audio_packet::{Decoder, Encoder};
use std::mem::MaybeUninit;
use std::vec::Vec;
use std::slice;

pub struct Mp3Encoder {
    inner: mp3lame_encoder::Encoder,
    channels: u8,
}

impl Mp3Encoder {
    /// Encode i16 samples into a standalone MP3 `Vec<u8>`, automatically sizing the buffer.
    pub fn encode_to_vec(&mut self, samples: &[i16]) -> Result<Vec<u8>, String> {
        // Offline convenience: encode then flush once.
        let mut mp3 = Vec::new();
        self.encode_chunk_to_vec(samples, &mut mp3)?;
        self.flush(&mut mp3)?;
        Ok(mp3)
    }

    /// Low-level flush if you used `encode_i16` directly.
    /// `out` must have at least ~7200 bytes of tailroom.
    pub fn flush_into(&mut self, out: &mut [u8]) -> Result<usize, String> {
        let out_uninit = unsafe {
            slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<u8>, out.len())
        };
        self.inner
            .flush::<FlushNoGap>(out_uninit)
            .map_err(|e| e.to_string())
    }

    /// Flush into a Vec, extending it by the exact number of bytes produced.
    pub fn flush(&mut self, out: &mut Vec<u8>) -> Result<usize, String> {
        let start = out.len();
        out.resize(start + 7200, 0);
        let written = self.flush_into(&mut out[start..])?;
        out.truncate(start + written);
        Ok(written)
    }

    /// Encode a single PCM chunk without flushing (stream-friendly).
    pub fn encode_chunk_to_vec(&mut self, samples: &[i16], out: &mut Vec<u8>) -> Result<usize, String> {
        let reserve = max_required_buffer_size(samples.len());
        let start = out.len();
        out.resize(start + reserve, 0);
        let written = self.encode_chunk(samples, &mut out[start..])?;
        out.truncate(start + written);
        Ok(written)
    }

    fn encode_chunk(&mut self, samples: &[i16], out: &mut [u8]) -> Result<usize, String> {
        let required = max_required_buffer_size(samples.len());
        if out.len() < required {
            return Err(format!(
                "Output buffer too small for chunk: need {}, have {}",
                required,
                out.len()
            ));
        }

        let out_uninit = unsafe {
            slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<u8>, out.len())
        };

        let written = if self.channels == 1 {
            self.inner.encode(MonoPcm(samples), out_uninit)
        } else {
            self.inner.encode(InterleavedPcm(samples), out_uninit)
        };

        written.map_err(|e| e.to_string())
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
            8_000 => Bitrate::Kbps8,
            16_000 => Bitrate::Kbps16,
            24_000 => Bitrate::Kbps24,
            32_000 => Bitrate::Kbps32,
            40_000 => Bitrate::Kbps40,
            48_000 => Bitrate::Kbps48,
            64_000 => Bitrate::Kbps64,
            80_000 => Bitrate::Kbps80,
            96_000 => Bitrate::Kbps96,
            112_000 => Bitrate::Kbps112,
            128_000 => Bitrate::Kbps128,
            160_000 => Bitrate::Kbps160,
            192_000 => Bitrate::Kbps192,
            224_000 => Bitrate::Kbps224,
            256_000 => Bitrate::Kbps256,
            320_000 => Bitrate::Kbps320,
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
        self.encode_chunk(input, out_buf)
    }

    fn encode_i32(&mut self, _input: &[i32], _out: &mut [u8]) -> Result<usize, String> {
        Err("not implemented".into())
    }

    fn reset(&mut self) -> Result<(), String> {
        Ok(())
    }
}

pub struct Mp3Decoder {
    inner: NanoDecoder,
    buffer: Vec<u8>,
    pcm: [f32; MAX_SAMPLES_PER_FRAME],
    sample_rate: Option<u32>,
    channels: Option<u8>,
}

impl Mp3Decoder {
    pub fn new() -> Self {
        Self {
            inner: NanoDecoder::new(),
            buffer: Vec::with_capacity(16 * 1024),
            pcm: [0.0; MAX_SAMPLES_PER_FRAME],
            sample_rate: None,
            channels: None,
        }
    }

    pub fn sample_rate(&self) -> Option<u32> {
        self.sample_rate
    }

    pub fn channels(&self) -> Option<u8> {
        self.channels
    }

    /// Get current buffer length (for debugging)
    pub fn buffer_len(&self) -> usize {
        self.buffer.len()
    }

    pub fn reset(&mut self) {
        self.inner = NanoDecoder::new();
        self.buffer.clear();
        self.sample_rate = None;
        self.channels = None;
    }

    fn capture_header(&mut self, info: &FrameInfo) {
        if self.sample_rate.is_none() || self.channels.is_none() {
            tracing::debug!(
                sample_rate_hz = info.sample_rate,
                channel_mode = ?info.channels,
                channels = info.channels.num(),
                bitrate_kbps = info.bitrate,
                samples_produced = info.samples_produced,
                "parsed MP3 frame header"
            );
        }

        self.sample_rate.get_or_insert(info.sample_rate);
        self.channels.get_or_insert(info.channels.num());
    }

    fn log_frame_decode(&self, info: &FrameInfo, consumed: usize, frame_samples: usize) {
        tracing::trace!(
            sample_rate_hz = info.sample_rate,
            channel_mode = ?info.channels,
            channels = info.channels.num(),
            bitrate_kbps = info.bitrate,
            samples_produced = info.samples_produced,
            bytes_consumed = consumed,
            pcm_samples_written = frame_samples,
            "decoded MP3 frame"
        );
    }

    fn write_frame_i16(&self, info: &FrameInfo, output: &mut [i16]) -> Result<usize, String> {
        let channels = info.channels.num() as usize;
        let frame_samples = info.samples_produced * channels;

        if frame_samples > output.len() {
            return Err(format!(
                "Output buffer too small for decoded frame (needed {}, had {})",
                frame_samples,
                output.len()
            ));
        }

        for (dst, &sample) in output[..frame_samples]
            .iter_mut()
            .zip(self.pcm[..frame_samples].iter())
        {
            *dst = f32_to_i16(sample);
        }

        Ok(frame_samples)
    }

    fn write_frame_i32(&self, info: &FrameInfo, output: &mut [i32]) -> Result<usize, String> {
        let channels = info.channels.num() as usize;
        let frame_samples = info.samples_produced * channels;

        if frame_samples > output.len() {
            return Err(format!(
                "Output buffer too small for decoded frame (needed {}, had {})",
                frame_samples,
                output.len()
            ));
        }

        for (dst, &sample) in output[..frame_samples]
            .iter_mut()
            .zip(self.pcm[..frame_samples].iter())
        {
            *dst = f32_to_i32(sample);
        }

        Ok(frame_samples)
    }
}

impl Default for Mp3Decoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Decoder for Mp3Decoder {
    fn decode_i16(&mut self, input: &[u8], out: &mut [i16], _fec: bool) -> Result<usize, String> {
        self.buffer.extend_from_slice(input);

        let mut written = 0;
        while !self.buffer.is_empty() {
            let (consumed, frame) = self.inner.decode(&self.buffer, &mut self.pcm);

            if consumed > 0 {
                self.buffer.drain(..consumed);
            }

            let Some(info) = frame else {
                break;
            };

            self.capture_header(&info);
            let frame_written = self.write_frame_i16(&info, &mut out[written..])?;
            self.log_frame_decode(&info, consumed, frame_written);
            written += frame_written;

            if out.len().saturating_sub(written) < MAX_SAMPLES_PER_FRAME {
                break;
            }
        }

        Ok(written)
    }

    fn decode_i32(&mut self, input: &[u8], out: &mut [i32], _fec: bool) -> Result<usize, String> {
        self.buffer.extend_from_slice(input);

        let mut written = 0;
        while !self.buffer.is_empty() {
            let (consumed, frame) = self.inner.decode(&self.buffer, &mut self.pcm);

            if consumed > 0 {
                self.buffer.drain(..consumed);
            }

            let Some(info) = frame else {
                break;
            };

            self.capture_header(&info);
            let frame_written = self.write_frame_i32(&info, &mut out[written..])?;
            self.log_frame_decode(&info, consumed, frame_written);
            written += frame_written;

            if out.len().saturating_sub(written) < MAX_SAMPLES_PER_FRAME {
                break;
            }
        }

        Ok(written)
    }

    fn decode_f32(&mut self, input: &[u8], out: &mut [f32], _fec: bool) -> Result<usize, String> {
        self.buffer.extend_from_slice(input);

        let mut written = 0;
        while !self.buffer.is_empty() {
            let (consumed, frame) = self.inner.decode(&self.buffer, &mut self.pcm);

            if consumed > 0 {
                self.buffer.drain(..consumed);
            }

            let Some(info) = frame else {
                break;
            };

            self.capture_header(&info);

            let channels = info.channels.num() as usize;
            let frame_samples = info.samples_produced * channels;

            if frame_samples > out[written..].len() {
                return Err(format!(
                    "Output buffer too small for decoded frame (needed {}, had {})",
                    frame_samples,
                    out[written..].len()
                ));
            }

            out[written..written + frame_samples].copy_from_slice(&self.pcm[..frame_samples]);
            self.log_frame_decode(&info, consumed, frame_samples);
            written += frame_samples;

            if out.len().saturating_sub(written) < MAX_SAMPLES_PER_FRAME {
                break;
            }
        }

        Ok(written)
    }
}

fn f32_to_i16(sample: f32) -> i16 {
    let scaled = (sample * i16::MAX as f32).round();
    if scaled > i16::MAX as f32 {
        i16::MAX
    } else if scaled < i16::MIN as f32 {
        i16::MIN
    } else {
        scaled as i16
    }
}

fn f32_to_i32(sample: f32) -> i32 {
    let scaled = (sample * i32::MAX as f32).round();
    if scaled > i32::MAX as f32 {
        i32::MAX
    } else if scaled < i32::MIN as f32 {
        i32::MIN
    } else {
        scaled as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mp3lame_encoder::max_required_buffer_size;
    use soundkit::audio_bytes::s16le_to_i16;
    use soundkit::test_utils::{print_waveform_with_header, DecodeResult};
    use soundkit::wav::WavStreamProcessor;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::Once;

    fn init_tracing() {
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            let _ = tracing_subscriber::fmt()
                .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
                .with_test_writer()
                .try_init();
        });
    }

    const TEST_FILE: &str = "A_Tusk_is_used_to_make_costly_gifts";

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

    fn outputs_path(file: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("outputs")
            .join(file)
    }

    #[test]
    fn test_mp3_decode_waveform() {
        let input_path = testdata_path(&format!("mp3/{}.mp3", TEST_FILE));
        let mp3_bytes = fs::read(&input_path).unwrap();
        assert!(!mp3_bytes.is_empty(), "fixture mp3 missing or empty");

        init_tracing();

        let mut decoder = Mp3Decoder::new();
        let mut decoded = Vec::new();
        let mut scratch = vec![0i16; MAX_SAMPLES_PER_FRAME * 2];

        for chunk in mp3_bytes.chunks(4096) {
            let written = decoder.decode_i16(chunk, &mut scratch, false).unwrap();
            decoded.extend_from_slice(&scratch[..written]);
        }

        // Drain remaining
        loop {
            let written = decoder.decode_i16(&[], &mut scratch, false).unwrap();
            if written == 0 {
                break;
            }
            decoded.extend_from_slice(&scratch[..written]);
        }

        assert!(!decoded.is_empty(), "decoder produced no PCM samples");

        let result = DecodeResult::new(
            &decoded,
            decoder.sample_rate().unwrap_or(16000),
            decoder.channels().unwrap_or(1),
        );
        print_waveform_with_header("MP3", &result);
    }

    #[test]
    fn test_mp3_encoder_encode_i16() {
        // load a 16-bit WAV
        let input_path = testdata_path("wav_stereo/A_Tusk_is_used_to_make_costly_gifts.wav");
        let data = fs::read(&input_path).unwrap();
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

        // stream through in chunks without flushing until the end
        let chunk_samples = 1152 * audio.channel_count() as usize; // typical MP3 granule
        let mut chunk_buf = vec![0u8; max_required_buffer_size(chunk_samples)];
        let mut out = Vec::new();
        for chunk in samples.chunks(chunk_samples) {
            let written = enc.encode_i16(chunk, &mut chunk_buf).unwrap();
            if written > 0 {
                out.extend_from_slice(&chunk_buf[..written]);
            }
        }

        // finalize once
        let mut flush_buf = vec![0u8; 8000];
        let flushed = enc.flush_into(&mut flush_buf).unwrap();
        out.extend_from_slice(&flush_buf[..flushed]);

        assert!(!out.is_empty(), "no bytes were written");
        assert_eq!(out[0], 0xFF, "MP3 frames should start with 0xFF");

        // write exactly the written bytes to disk for manual inspection
        let output_path =
            golden_path("mp3/A_Tusk_is_used_to_make_costly_gifts_encoded.mp3");
        fs::create_dir_all(output_path.parent().unwrap()).unwrap();
        fs::write(&output_path, &out[..]).unwrap();
    }

    #[test]
    fn test_mp3_decoder_streaming_decode() {
        // decode the real fixture MP3, not a freshly encoded one
        let input_path = testdata_path("mp3/A_Tusk_is_used_to_make_costly_gifts.mp3");
        let mp3_bytes = fs::read(&input_path).unwrap();
        assert!(!mp3_bytes.is_empty(), "fixture mp3 missing or empty");

        // decode in small chunks to exercise streaming
        init_tracing();
        let mut dec = Mp3Decoder::new();
        let mut decoded = Vec::new();
        let mut scratch = vec![0i16; MAX_SAMPLES_PER_FRAME * 2];

        for chunk in mp3_bytes.chunks(4096) {
            let written = dec.decode_i16(chunk, &mut scratch, false).unwrap();
            decoded.extend_from_slice(&scratch[..written]);
        }

        // final drain if anything buffered
        loop {
            let written = dec.decode_i16(&[], &mut scratch, false).unwrap();
            if written == 0 {
                break;
            }
            decoded.extend_from_slice(&scratch[..written]);
        }

        assert!(
            !decoded.is_empty(),
            "decoder produced no PCM samples"
        );
        assert_eq!(dec.sample_rate(), Some(16_000), "fixture sample rate");
        assert_eq!(dec.channels(), Some(1), "fixture channel count");

        // persist decoded PCM for manual inspection
        let output_path = outputs_path("A_Tusk_is_used_to_make_costly_gifts.s16le");
        fs::create_dir_all(output_path.parent().unwrap()).unwrap();
        let pcm_bytes: Vec<u8> = decoded.iter().flat_map(|s| s.to_le_bytes()).collect();
        fs::write(&output_path, pcm_bytes).unwrap();
    }

    /// Test that simulates the pipeline detection pattern:
    /// First 8192 bytes at once, then small chunks
    #[test]
    fn test_mp3_detection_pattern() {
        let input_path = testdata_path("mp3/A_Tusk_is_used_to_make_costly_gifts.mp3");
        let mp3_bytes = fs::read(&input_path).unwrap();
        assert!(!mp3_bytes.is_empty(), "fixture mp3 missing or empty");

        init_tracing();

        const MIN_DETECTION: usize = 8192;
        const SMALL_CHUNK: usize = 256;

        let mut decoder = Mp3Decoder::new();
        let mut decoded = Vec::new();
        let mut scratch = vec![0i16; MAX_SAMPLES_PER_FRAME * 2];

        // Phase 1: Detection - process first 8192 bytes at once
        let detection_bytes = &mp3_bytes[..MIN_DETECTION.min(mp3_bytes.len())];
        let written = decoder.decode_i16(detection_bytes, &mut scratch, false).unwrap();
        decoded.extend_from_slice(&scratch[..written]);
        println!("Detection phase: {} bytes in, {} samples out", detection_bytes.len(), written);

        // Drain after detection
        loop {
            let w = decoder.decode_i16(&[], &mut scratch, false).unwrap();
            if w == 0 { break; }
            decoded.extend_from_slice(&scratch[..w]);
            println!("Detection drain: {} samples", w);
        }

        let detection_samples = decoded.len();
        println!("After detection: {} samples total", detection_samples);

        println!("Decoder buffer len after detection: {}", decoder.buffer_len());

        // Phase 2: Remaining bytes in small chunks
        let mut chunks_processed = 0;
        let mut post_detection_samples = 0;
        let mut total_bytes_added = 0usize;
        for chunk in mp3_bytes[MIN_DETECTION..].chunks(SMALL_CHUNK) {
            total_bytes_added += chunk.len();
            let buf_before = decoder.buffer_len();
            let written = decoder.decode_i16(chunk, &mut scratch, false).unwrap();
            let buf_after = decoder.buffer_len();
            if written > 0 || chunks_processed < 5 {
                println!("Chunk {}: {} bytes in, buf {}→{}, {} samples out",
                    chunks_processed, chunk.len(), buf_before, buf_after, written);
            }
            if written > 0 {
                decoded.extend_from_slice(&scratch[..written]);
                post_detection_samples += written;
            }
            chunks_processed += 1;

            // Drain after each chunk (like the pipeline does)
            loop {
                let w = decoder.decode_i16(&[], &mut scratch, false).unwrap();
                if w == 0 { break; }
                decoded.extend_from_slice(&scratch[..w]);
                post_detection_samples += w;
                println!("Chunk {} drain: {} samples, buf now {}", chunks_processed, w, decoder.buffer_len());
            }
        }

        println!("Total bytes added post-detection: {}", total_bytes_added);
        println!("Final buffer len: {}", decoder.buffer_len());

        // Final flush
        loop {
            let w = decoder.decode_i16(&[], &mut scratch, false).unwrap();
            if w == 0 { break; }
            decoded.extend_from_slice(&scratch[..w]);
            post_detection_samples += w;
            println!("Final flush: {} samples", w);
        }

        println!("Post-detection: {} chunks processed, {} samples", chunks_processed, post_detection_samples);
        println!("Total: {} samples ({} bytes PCM)", decoded.len(), decoded.len() * 2);
    }

    /// Test that chunk size doesn't affect decoded output
    /// This reproduces the issue where HTTP/3 (small chunks) produces different output than HTTP/2 (large chunks)
    #[test]
    fn test_mp3_chunk_size_invariance() {
        let input_path = testdata_path("mp3/A_Tusk_is_used_to_make_costly_gifts.mp3");
        let mp3_bytes = fs::read(&input_path).unwrap();
        assert!(!mp3_bytes.is_empty(), "fixture mp3 missing or empty");

        init_tracing();

        // Decode with large chunks (simulating HTTP/2)
        let large_chunk_output = {
            let mut decoder = Mp3Decoder::new();
            let mut decoded = Vec::new();
            let mut scratch = vec![0i16; MAX_SAMPLES_PER_FRAME * 2];

            // Send all data in 2 large chunks (like HTTP/2)
            let mid = mp3_bytes.len() / 2;
            for chunk in [&mp3_bytes[..mid], &mp3_bytes[mid..]] {
                let written = decoder.decode_i16(chunk, &mut scratch, false).unwrap();
                decoded.extend_from_slice(&scratch[..written]);
            }

            // Drain remaining
            loop {
                let written = decoder.decode_i16(&[], &mut scratch, false).unwrap();
                if written == 0 {
                    break;
                }
                decoded.extend_from_slice(&scratch[..written]);
            }

            decoded
        };

        // Decode with small chunks (simulating HTTP/3)
        let small_chunk_output = {
            let mut decoder = Mp3Decoder::new();
            let mut decoded = Vec::new();
            let mut scratch = vec![0i16; MAX_SAMPLES_PER_FRAME * 2];

            // Send data in many small chunks (like HTTP/3 with QUIC)
            for chunk in mp3_bytes.chunks(1200) {
                let written = decoder.decode_i16(chunk, &mut scratch, false).unwrap();
                decoded.extend_from_slice(&scratch[..written]);
            }

            // Drain remaining
            loop {
                let written = decoder.decode_i16(&[], &mut scratch, false).unwrap();
                if written == 0 {
                    break;
                }
                decoded.extend_from_slice(&scratch[..written]);
            }

            decoded
        };

        println!("Large chunk output: {} samples ({} bytes PCM)",
            large_chunk_output.len(),
            large_chunk_output.len() * 2);
        println!("Small chunk output: {} samples ({} bytes PCM)",
            small_chunk_output.len(),
            small_chunk_output.len() * 2);

        assert_eq!(
            large_chunk_output.len(),
            small_chunk_output.len(),
            "Chunk size should not affect decoded output length! \
             Large: {} samples, Small: {} samples, \
             Difference: {} samples ({} bytes)",
            large_chunk_output.len(),
            small_chunk_output.len(),
            (large_chunk_output.len() as i64 - small_chunk_output.len() as i64).abs(),
            ((large_chunk_output.len() as i64 - small_chunk_output.len() as i64).abs() * 2)
        );

        assert_eq!(
            large_chunk_output,
            small_chunk_output,
            "Decoded PCM should be identical regardless of input chunk size"
        );
    }
}
