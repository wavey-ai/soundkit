use fdk_aac::dec::{Decoder as AacLibDecoder, DecoderError, Transport as DecoderTransport};
use fdk_aac::enc::EncodeInfo as AacEncodeInfo;
use fdk_aac::enc::{
    AudioObjectType, BitRate, ChannelMode, Encoder as AacLibEncoder, EncoderParams,
    Transport as EncoderTransport,
};
use soundkit::audio_packet::{Decoder, Encoder};
use std::cell::RefCell;
use std::rc::Rc;
use tracing::{debug, error, trace};

pub struct AacEncoder {
    encoder: AacLibEncoder,
    buffer: Rc<RefCell<Vec<u8>>>,
    _channels: u32,
    _sample_rate: u32,
}

impl Encoder for AacEncoder {
    fn new(
        sample_rate: u32,
        _bits_per_sample: u32, // Not used in AAC, can be set to 16 or 24 internally
        channels: u32,
        _frame_length: u32,      // Optional for frame size control
        _compression_level: u32, // Not used in AAC, we can use bitrate modes instead
    ) -> Self {
        let params = EncoderParams {
            bit_rate: BitRate::VbrVeryHigh,
            sample_rate,
            transport: EncoderTransport::Adts, // Transport can be set to Raw or Adts
            channels: if channels == 1 {
                ChannelMode::Mono
            } else {
                ChannelMode::Stereo
            },
            audio_object_type: AudioObjectType::Mpeg4LowComplexity,
        };

        let encoder = AacLibEncoder::new(params).expect("Failed to initialize AAC encoder");

        AacEncoder {
            encoder,
            buffer: Rc::new(RefCell::new(Vec::new())),
            _channels: channels,
            _sample_rate: sample_rate,
        }
    }

    fn init(&mut self) -> Result<(), String> {
        Ok(()) // The encoder is already initialized in the constructor
    }

    fn encode_i16(&mut self, input: &[i16], output: &mut [u8]) -> Result<usize, String> {
        // Clear the internal buffer before encoding
        self.buffer.borrow_mut().clear();

        let encoded_info: AacEncodeInfo = match self.encoder.encode(input, output) {
            Ok(info) => info,
            Err(err) => {
                error!("Encoding failed: {:?}", err);
                return Err(format!("Encoding failed: {}", err));
            }
        };

        if encoded_info.output_size > output.len() {
            return Err(format!(
                "Output buffer too small: {} bytes needed but only {} bytes available",
                encoded_info.output_size,
                output.len(),
            ));
        }

        Ok(encoded_info.output_size)
    }

    fn encode_i32(&mut self, _input: &[i32], _output: &mut [u8]) -> Result<usize, String> {
        Err("Not implemented.".to_string())
    }

    fn reset(&mut self) -> Result<(), String> {
        // No explicit reset required for this AAC encoder
        Ok(())
    }
}

impl Drop for AacEncoder {
    fn drop(&mut self) {
        // Drop the encoder and cleanup
    }
}

pub struct AacDecoder {
    decoder: AacLibDecoder,
    input_buffer: Vec<u8>,
    sample_rate: Option<u32>,
    channels: Option<u8>,
}

impl AacDecoder {
    pub fn new() -> Self {
        let decoder = AacLibDecoder::new(DecoderTransport::Adts);

        AacDecoder {
            decoder,
            input_buffer: Vec::new(),
            sample_rate: None,
            channels: None,
        }
    }

    pub fn init(&mut self) -> Result<(), String> {
        Ok(())
    }

    pub fn sample_rate(&self) -> Option<u32> {
        self.sample_rate
    }

    pub fn channels(&self) -> Option<u8> {
        self.channels
    }
}

impl Decoder for AacDecoder {
    fn decode_i16(
        &mut self,
        input: &[u8],
        output: &mut [i16],
        _fec: bool,
    ) -> Result<usize, String> {
        if !input.is_empty() {
            self.input_buffer.extend_from_slice(input);
        }

        let mut written = 0usize;
        let mut total_consumed = 0usize;

        loop {
            let consumed = if self.input_buffer.is_empty() {
                0
            } else {
                match self.decoder.fill(&self.input_buffer) {
                    Ok(bytes) => bytes,
                    Err(err) => return Err(format!("Error filling decoder: {}", err)),
                }
            };

            if consumed > 0 {
                total_consumed += consumed;
                self.input_buffer.drain(..consumed);
            }

            let remaining = output.len().saturating_sub(written);
            if remaining == 0 {
                break;
            }

            match self.decoder.decode_frame(&mut output[written..]) {
                Ok(()) => {
                    let info = self.decoder.stream_info();
                    let frame_samples =
                        info.numChannels as usize * info.frameSize as usize;

                    if frame_samples == 0 {
                        break;
                    }

                    if remaining < frame_samples {
                        return Err(format!(
                            "Output buffer too small for decoded frame (needed {}, had {})",
                            frame_samples, remaining
                        ));
                    }

                    let first_frame = self.sample_rate.is_none() || self.channels.is_none();
                    self.sample_rate.get_or_insert(info.sampleRate as u32);
                    self.channels.get_or_insert(info.numChannels as u8);
                    written += frame_samples;

                    if first_frame {
                        debug!(
                            sample_rate_hz = info.sampleRate,
                            channels = info.numChannels,
                            frame_samples,
                            bytes_consumed = total_consumed,
                            "decoded AAC frame"
                        );
                    } else {
                        trace!(
                            sample_rate_hz = info.sampleRate,
                            channels = info.numChannels,
                            frame_samples,
                            bytes_consumed = total_consumed,
                            "decoded AAC frame"
                        );
                    }
                }
                Err(err) => {
                    if err == DecoderError::NOT_ENOUGH_BITS {
                        // need more data
                        break;
                    }

                    return Err(format!("Decoding error: {}", err));
                }
            }
        }

        Ok(written)
    }

    fn decode_i32(
        &mut self,
        _input: &[u8],
        _output: &mut [i32],
        _fec: bool,
    ) -> Result<usize, String> {
        Err("Not implemented.".to_string())
    }
}

impl Drop for AacDecoder {
    fn drop(&mut self) {
        // The decoder will automatically handle cleanup in its Drop implementation
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use access_unit::aac::is_aac;
    use soundkit::audio_bytes::s16le_to_i16;
    use soundkit::wav::WavStreamProcessor;
    use std::fs::{self, File};
    use std::io::Read;
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::time::Instant;
    use tracing_subscriber;

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
    fn test_aac_decoder_streaming_decode() {
        // use the real fixture AAC, not one we just encoded
        let input_path = golden_path("aac/A_Tusk_is_used_to_make_costly_gifts_encoded.aac");
        let aac_bytes = fs::read(&input_path).unwrap();
        assert!(!aac_bytes.is_empty(), "fixture aac missing or empty");

        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .try_init();

        let mut decoder = AacDecoder::new();
        decoder.init().expect("Decoder initialization failed");

        let mut decoded = Vec::new();
        let mut scratch = vec![0i16; 4096];

        for chunk in aac_bytes.chunks(2048) {
            let written = decoder.decode_i16(chunk, &mut scratch, false).unwrap();
            decoded.extend_from_slice(&scratch[..written]);
        }

        // final drain if anything buffered
        loop {
            let written = decoder.decode_i16(&[], &mut scratch, false).unwrap();
            if written == 0 {
                break;
            }
            decoded.extend_from_slice(&scratch[..written]);
        }

        assert!(!decoded.is_empty(), "decoder produced no PCM samples");
        assert_eq!(decoder.sample_rate(), Some(16_000), "fixture sample rate");
        assert_eq!(decoder.channels(), Some(2), "fixture channel count");

        let output_path = outputs_path("aac_decoder_streaming_decode.pcm");
        fs::create_dir_all(output_path.parent().unwrap()).unwrap();
        let pcm_bytes: Vec<u8> = decoded.iter().flat_map(|s| s.to_le_bytes()).collect();
        fs::write(&output_path, pcm_bytes).unwrap();
    }

    fn run_aac_encoder_with_wav_file(file_path: &Path, output_path: &Path) {
        let mut decoder = AacDecoder::new();
        decoder.init().expect("Decoder initialization failed");

        let frame_size = 1024;
        let mut file = File::open(file_path).unwrap();
        let mut file_buffer = Vec::new();
        file.read_to_end(&mut file_buffer).unwrap();

        let mut processor = WavStreamProcessor::new();
        let audio_data = processor.add(&file_buffer).unwrap().unwrap();

        dbg!(file_path, audio_data.sampling_rate());

        let mut encoder = AacEncoder::new(
            audio_data.sampling_rate(),
            audio_data.bits_per_sample() as u32,
            audio_data.channel_count() as u32,
            0 as u32,
            5,
        );
        encoder.init().expect("Failed to initialize aac encoder");

        let i16_samples = match audio_data.bits_per_sample() {
            16 => s16le_to_i16(audio_data.data()),
            _ => {
                unreachable!()
            }
        };

        let mut encoded_data = Vec::new();
        let chunk_size = frame_size * audio_data.channel_count() as usize;
        let mut decoded_samples = vec![0i16; chunk_size * 2];

        for (i, chunk) in i16_samples.chunks(chunk_size).enumerate() {
            let start_time = Instant::now();
            let mut output_buffer = vec![0u8; chunk.len() * std::mem::size_of::<i32>() * 2];
            match encoder.encode_i16(chunk, &mut output_buffer) {
                Ok(encoded_len) => {
                    if encoded_len > 0 {
                        let elapsed_time = start_time.elapsed();
                        println!("Encoding took: {:.2?} seconds", elapsed_time);
                        assert!(is_aac(&output_buffer[..encoded_len]));
                        match decoder.decode_i16(
                            &output_buffer[..encoded_len],
                            &mut decoded_samples,
                            false,
                        ) {
                            Ok(samples_read) => {
                                println!(
                                    "Decoded {} samples of {} data successfully.",
                                    samples_read, encoded_len
                                );
                            }
                            Err(e) => panic!("Decoding failed: {}", e),
                        }
                    }
                    encoded_data.extend_from_slice(&output_buffer[..encoded_len]);
                }
                Err(e) => {
                    panic!("Failed to encode chunk {}: {:?}", i, e);
                }
            }
        }

        fs::create_dir_all(output_path.parent().unwrap()).unwrap();
        let mut file = File::create(output_path)
            .expect("Failed to create output file");
        file.write_all(&encoded_data)
            .expect("Failed to write to output file");

        encoder.reset().expect("Failed to reset encoder");
    }

    #[test]
    fn test_aac_encoder_with_wave_16bit() {
        run_aac_encoder_with_wav_file(
            &testdata_path("wav_stereo/A_Tusk_is_used_to_make_costly_gifts.wav"),
            &golden_path("aac/A_Tusk_is_used_to_make_costly_gifts_encoded.aac"),
        );
    }
}
