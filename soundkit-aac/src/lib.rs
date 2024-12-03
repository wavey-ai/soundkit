use fdk_aac::dec::{Decoder as AacLibDecoder, DecoderError, Transport as DecoderTransport};
use fdk_aac::enc::EncodeInfo as AacEncodeInfo;
use fdk_aac::enc::{
    AudioObjectType, BitRate, ChannelMode, Encoder as AacLibEncoder, EncoderParams,
    Transport as EncoderTransport,
};
use soundkit::audio_packet::{Decoder, Encoder};
use std::cell::RefCell;
use std::rc::Rc;
use tracing::{debug, error};

pub struct AacEncoder {
    encoder: AacLibEncoder,
    buffer: Rc<RefCell<Vec<u8>>>,
    channels: u32,
    sample_rate: u32,
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
            channels,
            sample_rate,
        }
    }

    fn init(&mut self) -> Result<(), String> {
        Ok(()) // The encoder is already initialized in the constructor
    }

    fn encode_i16(&mut self, input: &[i16], output: &mut [u8]) -> Result<usize, String> {
        let mut encoded_info: AacEncodeInfo;
        let mut result = Ok(());

        // Clear the internal buffer before encoding
        self.buffer.borrow_mut().clear();

        encoded_info = match self.encoder.encode(input, output) {
            Ok(info) => info,
            Err(err) => {
                error!("Encoding failed: {:?}", err);
                return Err(format!("Encoding failed: {}", err));
            }
        };

        if encoded_info.output_size > output.len() {
            result = Err(format!(
                "Output buffer too small: {} bytes needed but only {} bytes available",
                encoded_info.output_size,
                output.len(),
            ));
        }

        Ok(encoded_info.output_size)
    }

    fn encode_i32(&mut self, input: &[i32], output: &mut [u8]) -> Result<usize, String> {
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
    output_buffer: Vec<i16>, // Stores decoded PCM data
    input_buffer: Vec<u8>,   // Holds the raw AAC data
    input_position: usize,   // Tracks current position in the input buffer
}

impl AacDecoder {
    pub fn new() -> Self {
        let decoder = AacLibDecoder::new(DecoderTransport::Adts);

        AacDecoder {
            decoder,
            output_buffer: Vec::new(),
            input_buffer: Vec::new(),
            input_position: 0,
        }
    }

    pub fn init(&mut self) -> Result<(), String> {
        Ok(())
    }
}

impl Decoder for AacDecoder {
    fn decode_i16(
        &mut self,
        input: &[u8],
        output: &mut [i16],
        _fec: bool,
    ) -> Result<usize, String> {
        self.output_buffer.clear();
        self.input_buffer.clear();
        self.input_position = 0;

        self.input_buffer.extend_from_slice(input);

        let bytes_consumed = match self.decoder.fill(&self.input_buffer[self.input_position..]) {
            Ok(bytes) => bytes,
            Err(err) => return Err(format!("Error filling decoder: {}", err)),
        };

        self.input_position += bytes_consumed;

        match self.decoder.decode_frame(output) {
            Ok(()) => {
                let decoded_frame_size = self.decoder.decoded_frame_size();
                Ok(decoded_frame_size)
            }
            Err(err) => Err(format!("Decoding error: {}", err)),
        }
    }

    fn decode_i32(
        &mut self,
        input: &[u8],
        output: &mut [i32],
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
    use std::fs::File;
    use std::io::Read;
    use std::io::Write;
    use std::time::Instant;

    fn run_acc_encoder_with_wav_file(file_path: &str) {
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
        encoder.init().expect("Failed to initialize acc encoder");

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

        let mut file =
            File::create(file_path.to_owned() + ".acc").expect("Failed to create output file");
        file.write_all(&encoded_data)
            .expect("Failed to write to output file");

        encoder.reset().expect("Failed to reset encoder");
    }

    #[test]
    fn test_acc_encoder_with_wave_16bit() {
        run_acc_encoder_with_wav_file("../testdata/s16le.wav");
    }
}
