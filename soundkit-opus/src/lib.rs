use libopus::{decoder, encoder};
use soundkit::audio_packet::{Decoder, Encoder};
use soundkit::audio_types::AudioData;
use frame_header::{EncodingFlag, Endianness};
use tracing::{debug, trace};

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
    sample_rate: u32,
    channels: u8,
    first_frame_logged: bool,
}

impl OpusDecoder {
    pub fn new(sample_rate: usize, channels: usize) -> Self {
        let decoder = decoder::Decoder::create(sample_rate, channels, 1, 1, &[0u8, 1u8]).unwrap();

        OpusDecoder {
            decoder,
            sample_rate: sample_rate as u32,
            channels: channels as u8,
            first_frame_logged: false,
        }
    }

    pub fn init(&mut self) -> Result<(), String> {
        Ok(())
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn channels(&self) -> u8 {
        self.channels
    }
}

impl Decoder for OpusDecoder {
    fn decode_i16(&mut self, input: &[u8], output: &mut [i16], fec: bool) -> Result<usize, String> {
        let samples_per_channel = self
            .decoder
            .decode(input, output, fec)
            .map_err(|e| e.to_string())?;

        if !self.first_frame_logged {
            debug!(
                sample_rate_hz = self.sample_rate,
                channels = self.channels,
                packet_len = input.len(),
                pcm_samples_written = samples_per_channel * self.channels as usize,
                "decoded Opus packet"
            );
        } else {
            trace!(
                sample_rate_hz = self.sample_rate,
                channels = self.channels,
                packet_len = input.len(),
                pcm_samples_written = samples_per_channel * self.channels as usize,
                "decoded Opus packet"
            );
        }
        self.first_frame_logged = true;

        Ok(samples_per_channel)
    }
    fn decode_i32(&mut self, _input: &[u8], _output: &mut [i32], _fec: bool) -> Result<usize, String> {
        return Err("not implemented.".to_string());
    }

    fn decode_f32(&mut self, input: &[u8], output: &mut [f32], fec: bool) -> Result<usize, String> {
        // Opus decoder outputs i16, convert to f32
        let mut i16_buf = vec![0i16; output.len()];
        let samples = self.decode_i16(input, &mut i16_buf, fec)?;

        for i in 0..samples {
            output[i] = (i16_buf[i] as f32) / 32768.0;
        }

        Ok(samples)
    }
}

const MAX_OPUS_FRAME_SAMPLES: usize = 5760; // 120 ms @ 48 kHz

/// Streaming decoder for raw Opus format (OpusHead + length-prefixed packets)
pub struct OpusStreamDecoder {
    buffer: Vec<u8>,
    decoder: Option<OpusDecoder>,
    sample_rate: Option<u32>,
    channels: Option<u8>,
    pre_skip_remaining: usize,
    header_parsed: bool,
}

impl OpusStreamDecoder {
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            decoder: None,
            sample_rate: None,
            channels: None,
            pre_skip_remaining: 0,
            header_parsed: false,
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

    /// Add data and return decoded AudioData if a complete packet was decoded
    pub fn add(&mut self, data: &[u8]) -> Result<Option<AudioData>, String> {
        self.buffer.extend_from_slice(data);

        // Parse header if not done yet
        if !self.header_parsed && self.buffer.len() >= 19 {
            if !self.buffer.starts_with(b"OpusHead") {
                return Err("Invalid Opus stream: missing OpusHead".to_string());
            }

            self.sample_rate = Some(u32::from_le_bytes([
                self.buffer[12],
                self.buffer[13],
                self.buffer[14],
                self.buffer[15],
            ]));

            let sample_rate = self.sample_rate.unwrap();
            if sample_rate == 0 {
                self.sample_rate = Some(48_000);
            }

            self.channels = Some(self.buffer[9]);
            let pre_skip = u16::from_le_bytes([self.buffer[10], self.buffer[11]]);
            self.pre_skip_remaining = pre_skip as usize * self.channels.unwrap() as usize;

            let channels = self.channels.unwrap();
            let decoder = OpusDecoder::new(
                self.sample_rate.unwrap() as usize,
                channels as usize,
            );

            self.decoder = Some(decoder);
            self.header_parsed = true;

            debug!(
                sample_rate_hz = self.sample_rate.unwrap(),
                channels = channels,
                pre_skip = pre_skip,
                "initialized Opus stream decoder"
            );

            // Remove header from buffer
            self.buffer.drain(..19);
        }

        // Try to decode a packet
        if self.decoder.is_some() && self.buffer.len() >= 2 {
            let packet_len = u16::from_le_bytes([self.buffer[0], self.buffer[1]]) as usize;

            // Check if we have the complete packet
            if packet_len > 0 && self.buffer.len() >= 2 + packet_len {
                let packet = &self.buffer[2..2 + packet_len];
                let decoder = self.decoder.as_mut().unwrap();
                let channels = self.channels.unwrap();

                let mut scratch = vec![0i16; MAX_OPUS_FRAME_SAMPLES * channels as usize];

                match decoder.decode_i16(packet, &mut scratch, false) {
                    Ok(samples_per_channel) if samples_per_channel > 0 => {
                        let mut frame_samples = samples_per_channel * channels as usize;
                        let mut start = 0;

                        // Handle pre-skip
                        if self.pre_skip_remaining > 0 {
                            let skip = self.pre_skip_remaining.min(frame_samples);
                            self.pre_skip_remaining -= skip;
                            start = skip;
                        }

                        frame_samples = frame_samples.saturating_sub(start);

                        // Remove processed packet from buffer
                        self.buffer.drain(..2 + packet_len);

                        if frame_samples > 0 {
                            // Convert to bytes
                            let mut pcm_bytes = Vec::with_capacity(frame_samples * 2);
                            for &sample in &scratch[start..start + frame_samples] {
                                pcm_bytes.extend_from_slice(&sample.to_le_bytes());
                            }

                            let audio_data = AudioData::new(
                                16,
                                channels,
                                self.sample_rate.unwrap(),
                                pcm_bytes,
                                EncodingFlag::PCMSigned,
                                Endianness::LittleEndian,
                            );

                            return Ok(Some(audio_data));
                        }
                    }
                    Ok(_) => {
                        // No samples decoded, remove packet and continue
                        self.buffer.drain(..2 + packet_len);
                    }
                    Err(e) => {
                        // Decode error, remove packet and return error
                        self.buffer.drain(..2 + packet_len);
                        return Err(e);
                    }
                }
            }
        }

        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use soundkit::audio_bytes::s16le_to_i16;
    use soundkit::test_utils::{print_waveform_with_header, DecodeResult};
    use soundkit::wav::WavStreamProcessor;
    use std::fs::{self, File};
    use std::io::Read;
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::time::Instant;
    use std::sync::Once;
    use tracing::debug;
    use tracing_subscriber;

    const TEST_FILE: &str = "A_Tusk_is_used_to_make_costly_gifts";

    #[derive(Debug)]
    struct RawOpusHeader {
        sample_rate: u32,
        channels: u8,
        pre_skip: u16,
    }

    fn init_tracing() {
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            let _ = tracing_subscriber::fmt()
                .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
                .with_test_writer()
                .try_init();
        });
    }

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

    fn parse_length_prefixed_opus(
        data: &[u8],
    ) -> Result<(RawOpusHeader, Vec<&[u8]>), String> {
        if data.len() < 19 || !data.starts_with(b"OpusHead") {
            return Err("Missing OpusHead".to_string());
        }

        let header = RawOpusHeader {
            sample_rate: u32::from_le_bytes([data[12], data[13], data[14], data[15]]),
            channels: data[9],
            pre_skip: u16::from_le_bytes([data[10], data[11]]),
        };

        let mut packets = Vec::new();
        let mut cursor = &data[19..];
        while cursor.len() >= 2 {
            let len = u16::from_le_bytes([cursor[0], cursor[1]]) as usize;
            cursor = &cursor[2..];
            if len == 0 || cursor.len() < len {
                break;
            }
            let (packet, rest) = cursor.split_at(len);
            packets.push(packet);
            cursor = rest;
        }

        Ok((header, packets))
    }

    #[test]
    fn test_opus_decode_waveform() {
        let input_path = testdata_path(&format!("opus/{}.opus", TEST_FILE));
        let opus_bytes = fs::read(&input_path).unwrap();
        assert!(!opus_bytes.is_empty(), "fixture opus missing or empty");

        init_tracing();

        let (header, packets) =
            parse_length_prefixed_opus(&opus_bytes).expect("failed to parse opus fixture");

        let mut decoder = OpusDecoder::new(header.sample_rate as usize, header.channels as usize);
        decoder.init().expect("Decoder initialization failed");

        let mut decoded = Vec::new();
        let mut scratch = vec![0i16; MAX_OPUS_FRAME_SAMPLES * header.channels as usize];
        let mut pre_skip = header.pre_skip as usize * header.channels as usize;

        for packet in packets {
            let samples_per_channel = decoder.decode_i16(packet, &mut scratch, false).unwrap();
            if samples_per_channel == 0 {
                continue;
            }

            let mut frame_samples = samples_per_channel * header.channels as usize;
            let mut start = 0;
            if pre_skip > 0 {
                let skip = pre_skip.min(frame_samples);
                pre_skip -= skip;
                start = skip;
            }

            frame_samples = frame_samples.saturating_sub(start);
            if frame_samples == 0 {
                continue;
            }

            decoded.extend_from_slice(&scratch[start..start + frame_samples]);
        }

        assert!(!decoded.is_empty(), "decoder produced no PCM samples");

        let result = DecodeResult::new(&decoded, decoder.sample_rate(), decoder.channels());
        print_waveform_with_header("Opus", &result);
    }

    #[test]
    fn test_opus_decoder_streaming_decode() {
        // decode the real fixture opus stream; it is already length-prefixed
        let input_path = testdata_path("opus/A_Tusk_is_used_to_make_costly_gifts.opus");
        let opus_bytes = fs::read(&input_path).unwrap();
        assert!(!opus_bytes.is_empty(), "fixture opus missing or empty");

        init_tracing();

        let (header, packets) =
            parse_length_prefixed_opus(&opus_bytes).expect("failed to parse opus fixture");

        const MAX_OPUS_FRAME_SAMPLES: usize = 5760; // 120 ms @ 48kHz
        let mut decoder =
            OpusDecoder::new(header.sample_rate as usize, header.channels as usize);
        decoder.init().expect("Decoder initialization failed");

        let mut decoded = Vec::new();
        let mut scratch =
            vec![0i16; MAX_OPUS_FRAME_SAMPLES * header.channels as usize];
        let mut pre_skip = header.pre_skip as usize * header.channels as usize;

        for packet in packets {
            let samples_per_channel =
                decoder.decode_i16(packet, &mut scratch, false).unwrap();
            if samples_per_channel == 0 {
                continue;
            }

            let mut frame_samples =
                samples_per_channel * header.channels as usize;
            let mut start = 0;
            if pre_skip > 0 {
                let skip = pre_skip.min(frame_samples);
                pre_skip -= skip;
                start = skip;
            }

            frame_samples = frame_samples.saturating_sub(start);
            if frame_samples == 0 {
                continue;
            }

            decoded.extend_from_slice(&scratch[start..start + frame_samples]);
        }

        assert!(!decoded.is_empty(), "decoder produced no PCM samples");
        assert_eq!(decoder.sample_rate(), 16_000);
        assert_eq!(decoder.channels(), 1);

        let output_path = outputs_path("A_Tusk_is_used_to_make_costly_gifts.s16le");
        fs::create_dir_all(output_path.parent().unwrap()).unwrap();
        let pcm_bytes: Vec<u8> = decoded.iter().flat_map(|s| s.to_le_bytes()).collect();
        fs::write(&output_path, pcm_bytes).unwrap();
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

        init_tracing();
        debug!(
            bits_per_sample = audio_data.bits_per_sample(),
            sample_rate_hz = audio_data.sampling_rate(),
            channels = audio_data.channel_count(),
            "loaded WAV fixture"
        );

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
                        debug!(
                            chunk = i,
                            encoded_len,
                            elapsed_micros = elapsed_time.as_micros() as u64,
                            "encoded chunk"
                        );
                        match decoder.decode_i16(
                            &output_buffer[..encoded_len],
                            &mut decoded_samples,
                            false,
                        ) {
                            Ok(samples_read) => {
                                debug!(
                                    chunk = i,
                                    samples_read,
                                    encoded_len,
                                    "decoded opus chunk"
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
