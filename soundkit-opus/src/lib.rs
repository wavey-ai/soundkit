use libopus::{decoder, encoder};
use soundkit::audio_packet::{Decoder, Encoder};

pub struct OpusEncoder {
    encoder: encoder::Encoder,
    _sample_rate: u32,
    _channels: u32,
    _bits_per_sample: u32,
    _frame_size: u32,
    bitrate: u32,
}

impl Encoder for OpusEncoder {
    fn new(
        sample_rate: u32,
        bits_per_sample: u32,
        channels: u32,
        frame_size: u32,
        bitrate: u32,
    ) -> Self {
        let encoder = encoder::Encoder::create(
            sample_rate as usize,
            channels as usize,
            1,
            if channels > 1 { 1 } else { 0 },
            &[0u8, 1u8],
            encoder::Application::Audio,
        )
        .unwrap();

        Self {
            encoder,
            _sample_rate: sample_rate,
            _channels: channels,
            _bits_per_sample: bits_per_sample,
            _frame_size: frame_size,
            bitrate,
        }
    }

    fn init(&mut self) -> Result<(), String> {
        self.reset()
    }

    fn encode_i16(&mut self, input: &[i16], output: &mut [u8]) -> Result<usize, String> {
        self.encoder
            .encode(input, output)
            .map_err(|e| e.to_string())
    }

    fn encode_i32(&mut self, _input: &[i32], _output: &mut [u8]) -> Result<usize, String> {
        Err("Not implemented.".to_string())
    }

    fn reset(&mut self) -> Result<(), String> {
        match self
            .encoder
            .set_option(encoder::OPUS_SET_BITRATE_REQUEST, self.bitrate as u32)
        {
            Ok(_) => Ok(()),
            Err(e) => Err(format!("error reseting opus: {}", e)),
        }
    }
}

pub struct OpusDecoder {
    decoder: decoder::Decoder,
}

impl OpusDecoder {
    pub fn new(sample_rate: usize, channels: usize) -> Self {
        let decoder = decoder::Decoder::create(sample_rate, channels, 1, 1, &[0u8, 1u8]).unwrap();

        OpusDecoder { decoder }
    }

    pub fn init(&mut self) -> Result<(), String> {
        Ok(())
    }
}

impl Decoder for OpusDecoder {
    fn decode_i16(&mut self, input: &[u8], output: &mut [i16], fec: bool) -> Result<usize, String> {
        self.decoder
            .decode(input, output, fec)
            .map_err(|e| e.to_string())
    }
    fn decode_i32(&mut self, _input: &[u8], _output: &mut [i32], _fec: bool) -> Result<usize, String> {
        return Err("not implemented.".to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use soundkit::audio_bytes::s16le_to_i16;
    use soundkit::wav::WavStreamProcessor;
    use std::fs::{self, File};
    use std::io::Read;
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::time::Instant;

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

    fn run_opus_encoder_with_wav_file(
        file_path: &Path,
        encoded_output: &Path,
        decoded_output: &Path,
    ) {
        let mut file = File::open(file_path).unwrap();
        let mut file_buffer = Vec::new();
        file.read_to_end(&mut file_buffer).unwrap();

        let mut processor = WavStreamProcessor::new();
        let audio_data = processor.add(&file_buffer).unwrap().unwrap();

        dbg!(audio_data.bits_per_sample());

        let mut decoder = OpusDecoder::new(
            audio_data.sampling_rate() as usize,
            audio_data.channel_count() as usize,
        );
        decoder.init().expect("Decoder initialization failed");

        let frame_size = std::cmp::max(1, (audio_data.sampling_rate() / 50) as usize);

        let mut encoder = OpusEncoder::new(
            audio_data.sampling_rate(),
            audio_data.bits_per_sample() as u32,
            audio_data.channel_count() as u32,
            frame_size as u32,
            128_000,
        );
        encoder.init().expect("Failed to initialize opus encoder");

        let i16_samples = match audio_data.bits_per_sample() {
            16 => s16le_to_i16(audio_data.data()),
            _ => {
                unreachable!()
            }
        };

        let mut encoded_data = Vec::new();
        let chunk_size = frame_size * audio_data.channel_count() as usize;
        let mut decoded_samples = vec![0i16; chunk_size * 2];
        let mut output = Vec::new();
        for (i, chunk) in i16_samples.chunks(chunk_size).enumerate() {
            let start_time = Instant::now();
            let mut output_buffer = vec![0u8; chunk.len() * std::mem::size_of::<i32>() * 2];
            match encoder.encode_i16(chunk, &mut output_buffer) {
                Ok(encoded_len) => {
                    if encoded_len > 0 {
                        let elapsed_time = start_time.elapsed();
                        println!("Encoding took: {:.2?}", elapsed_time);
                        match decoder.decode_i16(
                            &output_buffer[..encoded_len],
                            &mut decoded_samples,
                            false,
                        ) {
                            Ok(samples_read) => {
                                println!(
                                    "Decoded {} samples of {} bytes successfully.",
                                    samples_read, encoded_len
                                );
                                for sample in &decoded_samples
                                    [..samples_read * audio_data.channel_count() as usize]
                                {
                                    output.extend(sample.to_le_bytes());
                                }
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

        fs::create_dir_all(encoded_output.parent().unwrap()).unwrap();
        let mut file = File::create(encoded_output).expect("Failed to create output file");
        file.write_all(&encoded_data)
            .expect("Failed to write to output file");
        let mut file = File::create(decoded_output).expect("Failed to create output file");
        file.write_all(&output[..])
            .expect("Failed to write to output file");

        encoder.reset().expect("Failed to reset encoder");
    }

    #[test]
    fn test_opus_encoder_with_wave_16bit() {
        run_opus_encoder_with_wav_file(
            &testdata_path("wav_stereo/A_Tusk_is_used_to_make_costly_gifts.wav"),
            &golden_path("opus/A_Tusk_is_used_to_make_costly_gifts_encoded.opus"),
            &golden_path("opus/A_Tusk_is_used_to_make_costly_gifts_decoded_from_opus.wav"),
        );
    }
}
