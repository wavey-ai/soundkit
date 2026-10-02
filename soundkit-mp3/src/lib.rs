#[cfg(feature = "encode")]
use mp3lame_encoder::{
    max_required_buffer_size, Bitrate, Builder, FlushNoGap, InterleavedPcm, MonoPcm,
};
use soundkit::audio_packet::Decoder;
#[cfg(feature = "encode")]
use soundkit::audio_packet::Encoder;
#[cfg(feature = "encode")]
use std::mem::MaybeUninit;
#[cfg(feature = "encode")]
use std::slice;
use std::vec::Vec;

mod decoder;

const MAX_SAMPLES_PER_FRAME: usize = 1152 * 2;

struct Mp3FrameDecoder(decoder::Layer3Decoder);

#[derive(Debug, Clone, Copy)]
struct FrameInfo {
    samples_produced: usize,
    channels: Channels,
    sample_rate: u32,
    bitrate: u32,
}

#[derive(Debug, Clone, Copy)]
enum Channels {
    Mono = 1,
    Stereo,
}

impl Channels {
    const fn num(self) -> u8 {
        self as u8
    }
}

impl Mp3FrameDecoder {
    const fn new() -> Self {
        Self(decoder::Layer3Decoder::new())
    }

    /// Decodes the frame at or after the start of `mp3`. Returns the bytes
    /// up to the end of the frame, whether the search found no frame, and
    /// the frame when it produced audio.
    fn decode(&mut self, mp3: &[u8], pcm: &mut [f32]) -> (usize, bool, Option<FrameInfo>) {
        assert!(pcm.len() >= MAX_SAMPLES_PER_FRAME, "PCM buffer too small");
        let mut info = decoder::CoreFrameInfo::default();
        let samples = decoder::decode_frame(&mut self.0, mp3, pcm, &mut info);
        if samples == 0 {
            return (info.frame_bytes, info.not_found, None);
        }
        let channels = match info.channels {
            1 => Channels::Mono,
            2 => Channels::Stereo,
            _ => return (info.frame_bytes, false, None),
        };
        (
            info.frame_bytes,
            false,
            Some(FrameInfo {
                samples_produced: samples,
                channels,
                sample_rate: info.sample_rate,
                bitrate: info.bitrate_kbps,
            }),
        )
    }
}

#[cfg(feature = "encode")]
pub struct Mp3Encoder {
    inner: mp3lame_encoder::Encoder,
    channels: u8,
}

#[cfg(feature = "encode")]
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
    pub fn encode_chunk_to_vec(
        &mut self,
        samples: &[i16],
        out: &mut Vec<u8>,
    ) -> Result<usize, String> {
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

#[cfg(feature = "encode")]
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

/// Streaming MP3 decoder.
///
/// A frame is decoded only when the decision that locates it would be the
/// same with any further input, so the output is the same for every way the
/// stream is cut. The last frame waits for [`Mp3Decoder::end_input`].
pub struct Mp3Decoder {
    inner: Mp3FrameDecoder,
    buffer: Vec<u8>,
    buffer_start: usize,
    /// Bytes after `buffer_start` that a frame search skips whatever data
    /// follows. The search resumes here, at a byte that is not a header.
    search_from: usize,
    input_ended: bool,
    pcm: [f32; MAX_SAMPLES_PER_FRAME],
    sample_rate: Option<u32>,
    channels: Option<u8>,
}

const MAX_MP3_STREAM_BUFFER_BYTES: usize = 4 * 1024 * 1024;

impl Mp3Decoder {
    pub fn new() -> Self {
        Self {
            inner: Mp3FrameDecoder::new(),
            buffer: Vec::with_capacity(16 * 1024),
            buffer_start: 0,
            search_from: 0,
            input_ended: false,
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
        self.buffer.len() - self.buffer_start
    }

    pub fn reset(&mut self) {
        self.inner = Mp3FrameDecoder::new();
        self.buffer.clear();
        self.buffer_start = 0;
        self.search_from = 0;
        self.input_ended = false;
        self.sample_rate = None;
        self.channels = None;
    }

    /// States that no more input follows. Later decode calls decode the
    /// frames that waited for data after them, including the last frame.
    pub fn end_input(&mut self) {
        self.input_ended = true;
    }

    /// Decodes the next frame into `self.pcm`. Frames that produce no audio,
    /// such as a frame whose bit reservoir starts before the stream, are
    /// skipped as minimp3 specifies.
    fn next_frame(&mut self) -> Option<FrameInfo> {
        loop {
            let available = &self.buffer[self.buffer_start..];
            if available.is_empty() {
                return None;
            }
            if !self.input_ended {
                match decoder::next_frame(&self.inner.0, available) {
                    decoder::NextFrame::Continues { .. } => {
                        debug_assert_eq!(self.search_from, 0);
                    }
                    decoder::NextFrame::NeedsData => return None,
                    decoder::NextFrame::Searches => {
                        let searched = &available[self.search_from..];
                        let (found, finality) = decoder::probe_frame(searched);
                        let ready = found.is_some_and(|(offset, _)| {
                            finality.accept_final && finality.first_open == offset
                        });
                        if !ready {
                            // Every position before `first_open` is skipped by
                            // any later search. Resume at the last of them
                            // that is not a header, so the search starts on a
                            // byte it skips.
                            let open = finality.first_open.min(searched.len());
                            if let Some(skip) = (1..open)
                                .rev()
                                .find(|&at| decoder::starts_with_invalid_header(&searched[at..]))
                            {
                                self.search_from += skip;
                            }
                            return None;
                        }
                    }
                }
            }

            let from = self.buffer_start + self.search_from;
            let (frame_bytes, not_found, frame) =
                self.inner.decode(&self.buffer[from..], &mut self.pcm);
            if not_found || frame_bytes == 0 {
                return None;
            }
            self.buffer_start = from + frame_bytes;
            self.search_from = 0;
            if frame.is_some() {
                return frame;
            }
        }
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

    fn log_frame_decode(&self, info: &FrameInfo, frame_samples: usize) {
        tracing::trace!(
            sample_rate_hz = info.sample_rate,
            channel_mode = ?info.channels,
            channels = info.channels.num(),
            bitrate_kbps = info.bitrate,
            samples_produced = info.samples_produced,
            pcm_samples_written = frame_samples,
            "decoded MP3 frame"
        );
    }

    fn append_input(&mut self, input: &[u8]) -> Result<(), String> {
        self.compact_buffer();
        if self.buffer_len().saturating_add(input.len()) > MAX_MP3_STREAM_BUFFER_BYTES {
            return Err(format!(
                "MP3 stream exceeds the {MAX_MP3_STREAM_BUFFER_BYTES} byte buffer budget"
            ));
        }
        self.buffer.extend_from_slice(input);
        Ok(())
    }

    #[inline]
    fn compact_buffer(&mut self) {
        if self.buffer_start == self.buffer.len() {
            self.buffer.clear();
            self.buffer_start = 0;
        } else if self.buffer_start >= 16 * 1024 && self.buffer_start >= self.buffer.len() / 2 {
            self.buffer.copy_within(self.buffer_start.., 0);
            self.buffer.truncate(self.buffer.len() - self.buffer_start);
            self.buffer_start = 0;
        }
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
        self.append_input(input)?;

        let mut written = 0;
        while let Some(info) = self.next_frame() {
            self.capture_header(&info);
            let frame_written = self.write_frame_i16(&info, &mut out[written..])?;
            self.log_frame_decode(&info, frame_written);
            written += frame_written;

            if out.len().saturating_sub(written) < MAX_SAMPLES_PER_FRAME {
                break;
            }
        }

        self.compact_buffer();

        Ok(written)
    }

    fn decode_i32(&mut self, input: &[u8], out: &mut [i32], _fec: bool) -> Result<usize, String> {
        self.append_input(input)?;

        let mut written = 0;
        while let Some(info) = self.next_frame() {
            self.capture_header(&info);
            let frame_written = self.write_frame_i32(&info, &mut out[written..])?;
            self.log_frame_decode(&info, frame_written);
            written += frame_written;

            if out.len().saturating_sub(written) < MAX_SAMPLES_PER_FRAME {
                break;
            }
        }

        self.compact_buffer();

        Ok(written)
    }

    fn decode_f32(&mut self, input: &[u8], out: &mut [f32], _fec: bool) -> Result<usize, String> {
        self.append_input(input)?;

        let mut written = 0;
        while let Some(info) = self.next_frame() {
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
            self.log_frame_decode(&info, frame_samples);
            written += frame_samples;

            if out.len().saturating_sub(written) < MAX_SAMPLES_PER_FRAME {
                break;
            }
        }

        self.compact_buffer();

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
    #[cfg(feature = "encode")]
    use mp3lame_encoder::max_required_buffer_size;
    #[cfg(feature = "encode")]
    use soundkit::audio_bytes::s16le_to_i16;
    use soundkit::test_utils::{print_waveform_with_header, DecodeResult};
    #[cfg(feature = "encode")]
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

    #[cfg(feature = "encode")]
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

    #[cfg(feature = "encode")]
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
        let output_path = golden_path("mp3/A_Tusk_is_used_to_make_costly_gifts_encoded.mp3");
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

        assert!(!decoded.is_empty(), "decoder produced no PCM samples");
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
        let written = decoder
            .decode_i16(detection_bytes, &mut scratch, false)
            .unwrap();
        decoded.extend_from_slice(&scratch[..written]);
        println!(
            "Detection phase: {} bytes in, {} samples out",
            detection_bytes.len(),
            written
        );

        // Drain after detection
        loop {
            let w = decoder.decode_i16(&[], &mut scratch, false).unwrap();
            if w == 0 {
                break;
            }
            decoded.extend_from_slice(&scratch[..w]);
            println!("Detection drain: {} samples", w);
        }

        let detection_samples = decoded.len();
        println!("After detection: {} samples total", detection_samples);

        println!(
            "Decoder buffer len after detection: {}",
            decoder.buffer_len()
        );

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
                println!(
                    "Chunk {}: {} bytes in, buf {}→{}, {} samples out",
                    chunks_processed,
                    chunk.len(),
                    buf_before,
                    buf_after,
                    written
                );
            }
            if written > 0 {
                decoded.extend_from_slice(&scratch[..written]);
                post_detection_samples += written;
            }
            chunks_processed += 1;

            // Drain after each chunk (like the pipeline does)
            loop {
                let w = decoder.decode_i16(&[], &mut scratch, false).unwrap();
                if w == 0 {
                    break;
                }
                decoded.extend_from_slice(&scratch[..w]);
                post_detection_samples += w;
                println!(
                    "Chunk {} drain: {} samples, buf now {}",
                    chunks_processed,
                    w,
                    decoder.buffer_len()
                );
            }
        }

        println!("Total bytes added post-detection: {}", total_bytes_added);
        println!("Final buffer len: {}", decoder.buffer_len());

        // Final flush
        loop {
            let w = decoder.decode_i16(&[], &mut scratch, false).unwrap();
            if w == 0 {
                break;
            }
            decoded.extend_from_slice(&scratch[..w]);
            post_detection_samples += w;
            println!("Final flush: {} samples", w);
        }

        println!(
            "Post-detection: {} chunks processed, {} samples",
            chunks_processed, post_detection_samples
        );
        println!(
            "Total: {} samples ({} bytes PCM)",
            decoded.len(),
            decoded.len() * 2
        );
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

        println!(
            "Large chunk output: {} samples ({} bytes PCM)",
            large_chunk_output.len(),
            large_chunk_output.len() * 2
        );
        println!(
            "Small chunk output: {} samples ({} bytes PCM)",
            small_chunk_output.len(),
            small_chunk_output.len() * 2
        );

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
            large_chunk_output, small_chunk_output,
            "Decoded PCM should be identical regardless of input chunk size"
        );
    }

    /// The decode loop before frames were located chunk-independently: a
    /// frame is consumed only when it produces audio.
    fn reference_whole_decode(data: &[u8]) -> Vec<i16> {
        let mut inner = Mp3FrameDecoder::new();
        let mut pcm = [0.0f32; MAX_SAMPLES_PER_FRAME];
        let mut start = 0;
        let mut out = Vec::new();
        while start < data.len() {
            let (frame_bytes, _, frame) = inner.decode(&data[start..], &mut pcm);
            let Some(info) = frame else {
                break;
            };
            start += frame_bytes;
            let samples = info.samples_produced * info.channels.num() as usize;
            out.extend(pcm[..samples].iter().map(|&sample| f32_to_i16(sample)));
        }
        out
    }

    fn stream_decode(data: &[u8], chunking: &mut dyn FnMut() -> usize) -> Vec<i16> {
        let mut decoder = Mp3Decoder::new();
        let mut out = vec![0i16; MAX_SAMPLES_PER_FRAME * 3];
        let mut decoded = Vec::new();
        let mut rest = data;
        loop {
            let piece = if rest.is_empty() {
                decoder.end_input();
                &[][..]
            } else {
                let take = chunking().clamp(1, rest.len());
                let (piece, tail) = rest.split_at(take);
                rest = tail;
                piece
            };
            let mut input = piece;
            loop {
                let written = decoder.decode_i16(input, &mut out, false).unwrap();
                decoded.extend_from_slice(&out[..written]);
                if written == 0 {
                    break;
                }
                input = &[];
            }
            if piece.is_empty() {
                return decoded;
            }
        }
    }

    struct Xorshift(u64);

    impl Xorshift {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
    }

    /// Every cut of a stream decodes to the same samples, and a clean stream,
    /// with or without a tag and junk before it, decodes to the samples the
    /// whole-buffer loop gave before.
    #[test]
    fn decode_is_the_same_for_every_cut_of_the_stream() {
        let fixture =
            fs::read(testdata_path("mp3/A_Tusk_is_used_to_make_costly_gifts.mp3")).unwrap();
        let mut rng = Xorshift(0x2545_f491_4f6c_dd1d);
        // An ID3v2 tag of random bytes, with 0xff runs, then the stream.
        let mut tagged = b"ID3\x03\x00\x00".to_vec();
        let body: Vec<u8> = (0..40_000)
            .map(|index| {
                if index % 997 < 3 {
                    0xff
                } else {
                    rng.next() as u8
                }
            })
            .collect();
        let size = body.len() as u32;
        tagged.extend([
            (size >> 21) as u8 & 0x7f,
            (size >> 14) as u8 & 0x7f,
            (size >> 7) as u8 & 0x7f,
            size as u8 & 0x7f,
        ]);
        tagged.extend_from_slice(&body);
        tagged.extend_from_slice(&fixture);
        let mut corrupt = fixture.clone();
        for _ in 0..25 {
            let at = 600 + (rng.next() as usize) % (corrupt.len() - 600);
            corrupt[at] = rng.next() as u8;
        }
        let cut = fixture[fixture.len() / 3 + 51..].to_vec();

        for (name, data, clean) in [
            ("fixture", &fixture, true),
            ("tagged", &tagged, true),
            ("corrupt", &corrupt, false),
            ("cut", &cut, false),
        ] {
            let whole = stream_decode(data, &mut || usize::MAX);
            assert!(!whole.is_empty(), "{name}");
            if clean {
                assert_eq!(whole, reference_whole_decode(data), "{name}");
            }
            for size in [1usize, 2, 3, 7, 417, 4_096] {
                assert_eq!(
                    stream_decode(data, &mut || size),
                    whole,
                    "{name} in {size}-byte pieces"
                );
            }
            let mut sizes = Xorshift(data.len() as u64);
            assert_eq!(
                stream_decode(data, &mut || 1 + (sizes.next() % 3_000) as usize),
                whole,
                "{name} in random pieces"
            );
        }
        // A stream that starts inside the bit reservoir decoded to nothing.
        assert!(reference_whole_decode(&cut).is_empty());
    }

    #[test]
    fn inputs_shorter_than_a_header_wait_for_more() {
        for bytes in [
            &[0xffu8][..],
            &[0xff, 0xfb],
            &[0xff, 0xfb, 0x90],
            &[1, 2, 3, 4],
        ] {
            let mut decoder = Mp3Decoder::new();
            let mut out = vec![0i16; MAX_SAMPLES_PER_FRAME];
            assert_eq!(decoder.decode_i16(bytes, &mut out, false), Ok(0));
            decoder.end_input();
            assert_eq!(decoder.decode_i16(&[], &mut out, false), Ok(0));
            assert_eq!(decoder.buffer_len(), bytes.len());
        }
    }
}
