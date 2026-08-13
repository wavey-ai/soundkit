#[cfg(feature = "fdk")]
use fdk_aac::dec::{Decoder as AacLibDecoder, DecoderError, Transport as DecoderTransport};
#[cfg(feature = "fdk")]
use fdk_aac::enc::EncodeInfo as AacEncodeInfo;
#[cfg(feature = "fdk")]
use fdk_aac::enc::{
    AudioObjectType, BitRate, ChannelMode, Encoder as AacLibEncoder, EncoderParams,
    Transport as EncoderTransport,
};
#[cfg(feature = "fdk")]
use soundkit::audio_packet::{Decoder, Encoder};
#[cfg(feature = "fdk")]
use std::cell::RefCell;
#[cfg(feature = "fdk")]
use std::rc::Rc;
#[cfg(feature = "fdk")]
use tracing::{debug, error, trace};

#[cfg(feature = "fdk")]
const MAX_INPUT_CHUNK_BYTES: usize = 4 * 1024 * 1024;
#[cfg(feature = "fdk")]
const MAX_AAC_BUFFERED_BYTES: usize = 4 * 1024 * 1024;

#[cfg(feature = "fdk")]
pub struct AacEncoder {
    encoder: AacLibEncoder,
    buffer: Rc<RefCell<Vec<u8>>>,
    _channels: u32,
    _sample_rate: u32,
}

#[cfg(feature = "fdk")]
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

#[cfg(feature = "fdk")]
impl Drop for AacEncoder {
    fn drop(&mut self) {
        // Drop the encoder and cleanup
    }
}

#[cfg(feature = "fdk")]
pub struct AacDecoder {
    decoder: AacLibDecoder,
    input_buffer: Vec<u8>,
    sample_rate: Option<u32>,
    channels: Option<u8>,
}

#[cfg(feature = "fdk")]
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

#[cfg(feature = "fdk")]
impl Default for AacDecoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "fdk")]
impl Decoder for AacDecoder {
    fn decode_i16(
        &mut self,
        input: &[u8],
        output: &mut [i16],
        _fec: bool,
    ) -> Result<usize, String> {
        if input.len() > MAX_INPUT_CHUNK_BYTES {
            return Err(format!(
                "AAC input chunk exceeds the {MAX_INPUT_CHUNK_BYTES} byte streaming budget"
            ));
        }
        if !input.is_empty() {
            if self.input_buffer.len().saturating_add(input.len()) > MAX_AAC_BUFFERED_BYTES {
                return Err(format!(
                    "AAC decoder buffer exceeds the {MAX_AAC_BUFFERED_BYTES} byte streaming budget"
                ));
            }
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
                    let frame_samples = info.numChannels as usize * info.frameSize as usize;

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

    fn decode_f32(&mut self, input: &[u8], output: &mut [f32], fec: bool) -> Result<usize, String> {
        // Decode to i16 then convert to f32
        let mut i16_buf = vec![0i16; output.len()];
        let samples = self.decode_i16(input, &mut i16_buf, fec)?;

        for i in 0..samples {
            output[i] = (i16_buf[i] as f32) / 32768.0;
        }

        Ok(samples)
    }
}

#[cfg(feature = "fdk")]
impl Drop for AacDecoder {
    fn drop(&mut self) {
        // The decoder will automatically handle cleanup in its Drop implementation
    }
}

#[cfg(feature = "mp4-demux")]
mod mp4_demux {
    use soundkit_audio_demux::{AudioCodec, AudioDemuxEvent, AudioTrackDemuxer};
    use tracing::debug;

    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct AacMp4Config {
        pub sample_rate: u32,
        pub channels: u8,
        pub track_id: u32,
        pub sample_count: u32,
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct AacMp4Frame {
        pub sample_id: u32,
        pub start_time: u64,
        pub duration: u32,
        pub rendering_offset: i32,
        pub is_sync: bool,
        pub adts: Vec<u8>,
        pub raw: Vec<u8>,
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    pub enum AacMp4DemuxEvent {
        Config(AacMp4Config),
        Frame(AacMp4Frame),
    }

    pub struct AacMp4Demuxer {
        demuxer: AudioTrackDemuxer,
        sample_rate: Option<u32>,
        channels: Option<u8>,
    }

    impl Default for AacMp4Demuxer {
        fn default() -> Self {
            Self::new()
        }
    }

    impl AacMp4Demuxer {
        pub fn new() -> Self {
            Self {
                demuxer: AudioTrackDemuxer::new_with_format("mp4")
                    .expect("MP4 support is enabled for the AAC demuxer"),
                sample_rate: None,
                channels: None,
            }
        }

        pub fn init(&mut self) -> Result<(), String> {
            Ok(())
        }

        pub fn add(&mut self, input: &[u8]) -> Result<Vec<AacMp4DemuxEvent>, String> {
            self.add_inner(input, false)
        }

        pub fn finish(&mut self) -> Result<Vec<AacMp4DemuxEvent>, String> {
            self.add_inner(&[], true)
        }

        pub fn sample_rate(&self) -> Option<u32> {
            self.sample_rate
        }

        pub fn channels(&self) -> Option<u8> {
            self.channels
        }

        fn add_inner(
            &mut self,
            input: &[u8],
            finalizing: bool,
        ) -> Result<Vec<AacMp4DemuxEvent>, String> {
            let events = if finalizing {
                if !input.is_empty() {
                    let mut events = self.demuxer.push(input)?;
                    events.extend(self.demuxer.flush()?);
                    events
                } else {
                    self.demuxer.flush()?
                }
            } else {
                self.demuxer.push(input)?
            };
            self.convert_events(events)
        }

        fn convert_events(
            &mut self,
            events: Vec<AudioDemuxEvent>,
        ) -> Result<Vec<AacMp4DemuxEvent>, String> {
            let mut output = Vec::new();
            for event in events {
                match event {
                    AudioDemuxEvent::Config(config) => {
                        if config.codec != AudioCodec::Aac {
                            return Err("MP4/M4A audio track is not AAC".to_string());
                        }
                        let sample_rate = config
                            .sample_rate
                            .ok_or_else(|| "MP4 AAC track has no sample rate".to_string())?;
                        let channels = config
                            .channels
                            .ok_or_else(|| "MP4 AAC track has no channel count".to_string())?;
                        let track_id = u32::try_from(config.track_id.unwrap_or_default())
                            .map_err(|_| "MP4 AAC track ID exceeds u32".to_string())?;
                        self.sample_rate = Some(sample_rate);
                        self.channels = Some(channels);
                        output.push(AacMp4DemuxEvent::Config(AacMp4Config {
                            sample_rate,
                            channels,
                            track_id,
                            sample_count: config.sample_count.unwrap_or_default(),
                        }));
                    }
                    AudioDemuxEvent::Packet(packet) => {
                        if packet.codec != AudioCodec::Aac {
                            return Err("MP4/M4A packet is not AAC".to_string());
                        }
                        let raw = packet
                            .raw_data
                            .ok_or_else(|| "MP4 AAC packet has no raw access unit".to_string())?;
                        output.push(AacMp4DemuxEvent::Frame(AacMp4Frame {
                            sample_id: packet.sample_id.unwrap_or_default(),
                            start_time: packet.start_time.unwrap_or_default(),
                            duration: packet.duration.unwrap_or_default(),
                            rendering_offset: packet.rendering_offset.unwrap_or_default(),
                            is_sync: packet.is_sync.unwrap_or(true),
                            adts: packet.data,
                            raw,
                        }));
                    }
                }
            }
            debug!(events = output.len(), "streamed MP4 AAC demux events");
            Ok(output)
        }
    }
}

#[cfg(feature = "mp4-decoder")]
mod mp4_decoder {
    use super::{AacDecoder, AacMp4DemuxEvent, AacMp4Demuxer, MAX_AAC_BUFFERED_BYTES};
    use soundkit::audio_packet::Decoder;
    use std::collections::VecDeque;

    pub struct AacDecoderMp4 {
        demuxer: AacMp4Demuxer,
        decoder: AacDecoder,
        pending_adts: VecDeque<Vec<u8>>,
        pending_bytes: usize,
        sample_rate: Option<u32>,
        channels: Option<u8>,
        demux_finished: bool,
    }

    impl Default for AacDecoderMp4 {
        fn default() -> Self {
            Self::new()
        }
    }

    impl AacDecoderMp4 {
        pub fn new() -> Self {
            Self {
                demuxer: AacMp4Demuxer::new(),
                decoder: AacDecoder::new(),
                pending_adts: VecDeque::new(),
                pending_bytes: 0,
                sample_rate: None,
                channels: None,
                demux_finished: false,
            }
        }

        pub fn init(&mut self) -> Result<(), String> {
            self.demuxer.init()?;
            self.decoder.init()
        }

        pub fn sample_rate(&self) -> Option<u32> {
            self.sample_rate.or_else(|| self.decoder.sample_rate())
        }

        pub fn channels(&self) -> Option<u8> {
            self.channels.or_else(|| self.decoder.channels())
        }

        pub fn finish_i16(&mut self, output: &mut [i16]) -> Result<usize, String> {
            if !self.demux_finished {
                let events = self.demuxer.finish()?;
                self.enqueue(events)?;
                self.demux_finished = true;
            }
            self.decode_pending(output)
        }

        fn enqueue(&mut self, events: Vec<AacMp4DemuxEvent>) -> Result<(), String> {
            for event in events {
                match event {
                    AacMp4DemuxEvent::Config(config) => {
                        self.sample_rate = Some(config.sample_rate);
                        self.channels = Some(config.channels);
                    }
                    AacMp4DemuxEvent::Frame(frame) => {
                        let next_bytes = self.pending_bytes.saturating_add(frame.adts.len());
                        if next_bytes > MAX_AAC_BUFFERED_BYTES {
                            return Err(format!(
                                "MP4 AAC packet queue exceeds the {MAX_AAC_BUFFERED_BYTES} byte streaming budget"
                            ));
                        }
                        self.pending_bytes = next_bytes;
                        self.pending_adts.push_back(frame.adts);
                    }
                }
            }
            Ok(())
        }

        fn decode_pending(&mut self, output: &mut [i16]) -> Result<usize, String> {
            let mut written = 0usize;
            while written < output.len() {
                let Some(packet) = self.pending_adts.pop_front() else {
                    break;
                };
                self.pending_bytes = self.pending_bytes.saturating_sub(packet.len());
                let samples = self
                    .decoder
                    .decode_i16(&packet, &mut output[written..], false)?;
                written += samples;
            }
            self.sample_rate = self.decoder.sample_rate().or(self.sample_rate);
            self.channels = self.decoder.channels().or(self.channels);
            Ok(written)
        }
    }

    impl Decoder for AacDecoderMp4 {
        fn decode_i16(
            &mut self,
            input: &[u8],
            output: &mut [i16],
            _fec: bool,
        ) -> Result<usize, String> {
            if self.demux_finished && !input.is_empty() {
                return Err("MP4 AAC decoder is already finished".to_string());
            }
            if !input.is_empty() {
                let events = self.demuxer.add(input)?;
                self.enqueue(events)?;
            }
            self.decode_pending(output)
        }

        fn decode_i32(
            &mut self,
            _input: &[u8],
            _output: &mut [i32],
            _fec: bool,
        ) -> Result<usize, String> {
            Err("Not implemented.".to_string())
        }

        fn decode_f32(
            &mut self,
            input: &[u8],
            output: &mut [f32],
            fec: bool,
        ) -> Result<usize, String> {
            let mut i16_buf = vec![0i16; output.len()];
            let samples = self.decode_i16(input, &mut i16_buf, fec)?;
            for index in 0..samples {
                output[index] = f32::from(i16_buf[index]) / 32768.0;
            }
            Ok(samples)
        }
    }
}
#[cfg(feature = "mp4-decoder")]
pub use mp4_decoder::AacDecoderMp4;
#[cfg(feature = "mp4-demux")]
pub use mp4_demux::{AacMp4Config, AacMp4DemuxEvent, AacMp4Demuxer, AacMp4Frame};

#[cfg(test)]
mod tests {
    #[cfg(feature = "fdk")]
    use super::*;
    #[cfg(feature = "fdk")]
    use access_unit::aac::is_aac;
    #[cfg(feature = "fdk")]
    use soundkit::audio_bytes::s16le_to_i16;
    #[cfg(feature = "fdk")]
    use soundkit::test_utils::{print_waveform_with_header, DecodeResult};
    #[cfg(feature = "fdk")]
    use soundkit::wav::WavStreamProcessor;
    use std::fs;
    #[cfg(feature = "fdk")]
    use std::fs::File;
    #[cfg(feature = "fdk")]
    use std::io::{Read, Write};
    #[cfg(feature = "fdk")]
    use std::path::Path;
    use std::path::PathBuf;
    #[cfg(feature = "fdk")]
    use std::time::Instant;
    #[cfg(any(feature = "fdk", feature = "mp4-decoder"))]
    use tracing::trace;

    #[cfg(any(feature = "fdk", feature = "mp4-decoder"))]
    fn testdata_path(file: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("testdata")
            .join(file)
    }

    #[cfg(feature = "fdk")]
    fn golden_path(file: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("golden")
            .join(file)
    }

    #[cfg(feature = "fdk")]
    fn outputs_path(file: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("outputs")
            .join(file)
    }

    #[cfg(any(feature = "fdk", feature = "mp4-decoder"))]
    fn init_tracing() {
        use std::sync::Once;
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            let _ = tracing_subscriber::fmt()
                .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
                .with_test_writer()
                .try_init();
        });
    }

    #[test]
    #[cfg(feature = "mp4-decoder")]
    fn test_aac_decode_waveform() {
        use crate::AacDecoderMp4;

        let input_path = testdata_path("itag139/yt_itag_139_he_aac.mp4");
        let m4a_bytes = fs::read(&input_path).unwrap();
        assert!(!m4a_bytes.is_empty(), "fixture m4a missing or empty");

        init_tracing();

        let mut decoder = AacDecoderMp4::new();
        decoder.init().expect("Decoder initialization failed");

        let mut decoded = Vec::new();
        let mut scratch = vec![0i16; 16384];

        // Feed all data
        match decoder.decode_i16(&m4a_bytes, &mut scratch, false) {
            Ok(written) => {
                decoded.extend_from_slice(&scratch[..written]);
            }
            Err(e) => panic!("Decode failed: {}", e),
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
            decoder.sample_rate().unwrap_or(44100),
            decoder.channels().unwrap_or(1),
        );
        print_waveform_with_header("AAC (M4A)", &result);
    }

    #[test]
    #[cfg(feature = "fdk")]
    fn test_aac_decoder_streaming_decode() {
        // use the real fixture AAC, not one we just encoded
        let input_path = golden_path("aac/A_Tusk_is_used_to_make_costly_gifts_encoded.aac");
        let aac_bytes = fs::read(&input_path).unwrap();
        assert!(!aac_bytes.is_empty(), "fixture aac missing or empty");

        init_tracing();

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

        let output_path = outputs_path("A_Tusk_is_used_to_make_costly_gifts.s16le");
        fs::create_dir_all(output_path.parent().unwrap()).unwrap();
        let pcm_bytes: Vec<u8> = decoded.iter().flat_map(|s| s.to_le_bytes()).collect();
        fs::write(&output_path, pcm_bytes).unwrap();
    }

    #[cfg(feature = "fdk")]
    fn run_aac_encoder_with_wav_file(file_path: &Path, output_path: &Path) {
        init_tracing();

        let mut decoder = AacDecoder::new();
        decoder.init().expect("Decoder initialization failed");

        let frame_size = 1024;
        let mut file = File::open(file_path).unwrap();
        let mut file_buffer = Vec::new();
        file.read_to_end(&mut file_buffer).unwrap();

        let mut processor = WavStreamProcessor::new();
        let audio_data = processor.add(&file_buffer).unwrap().unwrap();

        trace!(
            file = ?file_path,
            sample_rate_hz = audio_data.sampling_rate(),
            "loaded WAV for AAC encoding"
        );

        let mut encoder = AacEncoder::new(
            audio_data.sampling_rate(),
            audio_data.bits_per_sample() as u32,
            audio_data.channel_count() as u32,
            0_u32,
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
                        trace!(
                            chunk = i,
                            encoded_len,
                            elapsed_micros = elapsed_time.as_micros() as u64,
                            "encoded AAC chunk"
                        );
                        assert!(is_aac(&output_buffer[..encoded_len]));
                        match decoder.decode_i16(
                            &output_buffer[..encoded_len],
                            &mut decoded_samples,
                            false,
                        ) {
                            Ok(samples_read) => {
                                trace!(chunk = i, samples_read, encoded_len, "decoded AAC chunk");
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
        let mut file = File::create(output_path).expect("Failed to create output file");
        file.write_all(&encoded_data)
            .expect("Failed to write to output file");

        encoder.reset().expect("Failed to reset encoder");
    }

    #[test]
    #[cfg(feature = "fdk")]
    fn test_aac_encoder_with_wave_16bit() {
        run_aac_encoder_with_wav_file(
            &testdata_path("wav_stereo/A_Tusk_is_used_to_make_costly_gifts.wav"),
            &golden_path("aac/A_Tusk_is_used_to_make_costly_gifts_encoded.aac"),
        );
    }

    #[test]
    #[cfg(feature = "mp4-decoder")]
    fn test_tail_moov_aac_requires_seekable_ranges() {
        use crate::AacDecoderMp4;

        let input_path = testdata_path("mac_aac/A_Tusk_is_used_to_make_costly_gifts.m4a");
        let m4a_bytes = fs::read(&input_path).unwrap();
        assert!(!m4a_bytes.is_empty(), "fixture m4a missing or empty");

        init_tracing();

        let mut decoder = AacDecoderMp4::new();
        decoder.init().expect("Decoder initialization failed");

        let mut scratch = vec![0i16; 16384];
        let error = decoder
            .decode_i16(&m4a_bytes, &mut scratch, false)
            .unwrap_err();
        assert!(error.contains("seekable MP4 packet API"));
    }

    #[test]
    #[cfg(feature = "mp4-decoder")]
    fn test_mp4_he_aac_itag_139_decoder() {
        use crate::AacDecoderMp4;
        use mp4::{AudioObjectType, ChannelConfig, MediaType, Mp4Reader};
        use std::io::Cursor;

        let input_path = testdata_path("itag139/yt_itag_139_he_aac.mp4");
        let mp4_bytes = fs::read(&input_path).unwrap();
        assert!(!mp4_bytes.is_empty(), "fixture MP4 missing or empty");

        let reader = Mp4Reader::read_header(Cursor::new(mp4_bytes.clone()), mp4_bytes.len() as u64)
            .expect("parse itag 139 MP4 header");
        let (_track_id, track) = reader
            .tracks()
            .iter()
            .find(|(_, track)| track.media_type().ok() == Some(MediaType::AAC))
            .expect("fixture should contain an AAC track");
        // This is the mp4-demux + FDK-AAC C-binding path. The HE-AAC fixture carries an AAC-LC core at
        // 11.025 kHz; SBR doubles the decoded output rate to 22.05 kHz.
        assert_eq!(
            track.audio_profile().unwrap(),
            AudioObjectType::AacLowComplexity
        );
        assert_eq!(track.sample_freq_index().unwrap().freq(), 11_025);
        assert_eq!(track.channel_config().unwrap(), ChannelConfig::Stereo);

        init_tracing();

        let mut decoder = AacDecoderMp4::new();
        decoder.init().expect("Decoder initialization failed");

        let mut decoded = Vec::new();
        let mut scratch = vec![0i16; 16384];

        let written = decoder
            .decode_i16(&mp4_bytes, &mut scratch, false)
            .expect("decode itag 139 HE-AAC MP4");
        decoded.extend_from_slice(&scratch[..written]);

        loop {
            let written = decoder
                .decode_i16(&[], &mut scratch, false)
                .expect("drain itag 139 HE-AAC MP4");
            if written == 0 {
                break;
            }
            decoded.extend_from_slice(&scratch[..written]);
        }

        assert!(!decoded.is_empty(), "decoder produced no PCM samples");
        assert_eq!(decoder.sample_rate(), Some(22_050));
        assert_eq!(decoder.channels(), Some(2));
        assert!(
            decoded.iter().any(|sample| *sample != 0),
            "decoded PCM should contain non-zero samples"
        );
    }

    #[test]
    #[cfg(feature = "mp4-demux")]
    fn test_mp4_aac_demux_to_adts_frames() {
        use crate::{AacMp4DemuxEvent, AacMp4Demuxer};

        let input_path = testdata_path("itag139/yt_itag_139_he_aac.mp4");
        let m4a_bytes = fs::read(&input_path).unwrap();
        assert!(!m4a_bytes.is_empty(), "fixture m4a missing or empty");

        let mut demuxer = AacMp4Demuxer::new();
        demuxer.init().expect("demuxer initialization failed");

        let mut events = Vec::new();
        for chunk in m4a_bytes.chunks(997) {
            events.extend(demuxer.add(chunk).expect("demux m4a chunk"));
        }
        events.extend(demuxer.finish().expect("finish m4a demux"));

        let config = events
            .iter()
            .find_map(|event| match event {
                AacMp4DemuxEvent::Config(config) => Some(config),
                _ => None,
            })
            .expect("config event");
        assert_eq!(config.sample_rate, 11_025);
        assert_eq!(config.channels, 2);

        let frames: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                AacMp4DemuxEvent::Frame(frame) => Some(frame),
                _ => None,
            })
            .collect();
        assert!(!frames.is_empty(), "demuxer produced no AAC frames");
        assert_eq!(frames.len() as u32, config.sample_count);
        assert!(frames[0].adts.starts_with(&[0xff, 0xf1]));
        assert!(frames[0].adts.len() > frames[0].raw.len());
    }
}
