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
    use access_unit::aac::create_adts_header;
    use mp4::{MediaType, Mp4Reader};
    use std::io::Cursor;
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
        input_buffer: Vec<u8>,
        track_id: Option<u32>,
        current_sample_id: u32,
        sample_count: u32,
        sample_rate: Option<u32>,
        channels: Option<u8>,
        initialized: bool,
        emitted_config: bool,
    }

    impl Default for AacMp4Demuxer {
        fn default() -> Self {
            Self::new()
        }
    }

    impl AacMp4Demuxer {
        pub fn new() -> Self {
            Self {
                input_buffer: Vec::new(),
                track_id: None,
                current_sample_id: 1,
                sample_count: 0,
                sample_rate: None,
                channels: None,
                initialized: false,
                emitted_config: false,
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
            if !input.is_empty() {
                self.input_buffer.extend_from_slice(input);
            }

            if !self.try_initialize_mp4(finalizing)? {
                return Ok(Vec::new());
            }

            let mut events = Vec::new();
            if !self.emitted_config {
                events.push(AacMp4DemuxEvent::Config(AacMp4Config {
                    sample_rate: self.sample_rate.unwrap_or(44_100),
                    channels: self.channels.unwrap_or(2),
                    track_id: self.track_id.unwrap_or_default(),
                    sample_count: self.sample_count,
                }));
                self.emitted_config = true;
            }

            events.extend(self.read_available_samples(finalizing)?);
            Ok(events)
        }

        fn try_initialize_mp4(&mut self, finalizing: bool) -> Result<bool, String> {
            if self.initialized {
                return Ok(true);
            }

            if self.input_buffer.len() < 1024 && !finalizing {
                return Ok(false);
            }

            let cursor = Cursor::new(self.input_buffer.clone());
            let size = self.input_buffer.len() as u64;

            let mp4 = match Mp4Reader::read_header(cursor, size) {
                Ok(mp4) => mp4,
                Err(error) if finalizing => {
                    return Err(format!("Failed to parse MP4/M4A header: {error}"));
                }
                Err(_) => return Ok(false),
            };

            let mut audio_track_id = None;
            for (track_id, track) in mp4.tracks() {
                if track.media_type().ok() != Some(MediaType::AAC) {
                    continue;
                }

                audio_track_id = Some(*track_id);
                if let Ok(freq_index) = track.sample_freq_index() {
                    self.sample_rate = Some(freq_index.freq());
                }
                if let Ok(channel_config) = track.channel_config() {
                    self.channels = Some(channel_config as u8);
                }
                break;
            }

            let track_id =
                audio_track_id.ok_or_else(|| "No AAC audio track found in MP4/M4A".to_string())?;
            let sample_count = mp4
                .sample_count(track_id)
                .map_err(|error| format!("Failed to get AAC sample count: {error}"))?;

            self.track_id = Some(track_id);
            self.sample_count = sample_count;
            self.initialized = true;

            debug!(
                track_id,
                sample_count,
                sample_rate = ?self.sample_rate,
                channels = ?self.channels,
                "Initialized MP4 AAC demuxer"
            );

            Ok(true)
        }

        fn read_available_samples(
            &mut self,
            finalizing: bool,
        ) -> Result<Vec<AacMp4DemuxEvent>, String> {
            let track_id = match self.track_id {
                Some(track_id) => track_id,
                None => return Ok(Vec::new()),
            };

            if self.current_sample_id > self.sample_count {
                return Ok(Vec::new());
            }

            let cursor = Cursor::new(self.input_buffer.clone());
            let size = self.input_buffer.len() as u64;
            let mut mp4_reader = match Mp4Reader::read_header(cursor, size) {
                Ok(reader) => reader,
                Err(error) if finalizing => {
                    return Err(format!("Failed to re-read MP4/M4A header: {error}"));
                }
                Err(_) => return Ok(Vec::new()),
            };

            let mut events = Vec::new();
            while self.current_sample_id <= self.sample_count {
                let sample_id = self.current_sample_id;
                let sample = match mp4_reader.read_sample(track_id, sample_id) {
                    Ok(Some(sample)) => sample,
                    Ok(None) => break,
                    Err(error) if !finalizing && is_incomplete_mp4_error(&error.to_string()) => {
                        break;
                    }
                    Err(error) => {
                        return Err(format!("Failed to read AAC sample {sample_id}: {error}"));
                    }
                };

                self.current_sample_id += 1;

                let raw = sample.bytes.to_vec();
                let adts_header = create_adts_header(
                    0x66,
                    self.channels.unwrap_or(2),
                    self.sample_rate.unwrap_or(44_100),
                    raw.len(),
                    false,
                );
                let mut adts = Vec::with_capacity(adts_header.len() + raw.len());
                adts.extend_from_slice(&adts_header);
                adts.extend_from_slice(&raw);

                events.push(AacMp4DemuxEvent::Frame(AacMp4Frame {
                    sample_id,
                    start_time: sample.start_time,
                    duration: sample.duration,
                    rendering_offset: sample.rendering_offset,
                    is_sync: sample.is_sync,
                    adts,
                    raw,
                }));
            }

            Ok(events)
        }
    }

    fn is_incomplete_mp4_error(error: &str) -> bool {
        let error = error.to_ascii_lowercase();
        error.contains("unexpected")
            || error.contains("eof")
            || error.contains("failed to fill whole buffer")
    }
}

#[cfg(feature = "mp4-decoder")]
mod mp4_decoder {
    use access_unit::aac::create_adts_header;
    use fdk_aac::dec::{Decoder as AacLibDecoder, Transport as DecoderTransport};
    use mp4::{MediaType, Mp4Reader};
    use soundkit::audio_packet::Decoder;
    use std::io::Cursor;
    use tracing::{debug, error, trace};

    pub struct AacDecoderMp4 {
        input_buffer: Vec<u8>,
        mp4_reader: Option<Mp4Reader<Cursor<Vec<u8>>>>,
        fdk_decoder: AacLibDecoder,
        track_id: Option<u32>,
        current_sample_id: u32,
        sample_count: u32,
        sample_rate: Option<u32>,
        channels: Option<u8>,
        initialized: bool,
        adts_buffer: Vec<u8>, // Buffer of ADTS frames ready for decoding
    }

    impl Default for AacDecoderMp4 {
        fn default() -> Self {
            Self::new()
        }
    }

    impl AacDecoderMp4 {
        pub fn new() -> Self {
            // Initialize FDK decoder in ADTS mode
            // We'll add ADTS headers to raw MP4 samples
            let decoder = AacLibDecoder::new(DecoderTransport::Adts);

            AacDecoderMp4 {
                input_buffer: Vec::new(),
                mp4_reader: None,
                fdk_decoder: decoder,
                track_id: None,
                current_sample_id: 1,
                sample_count: 0,
                sample_rate: None,
                channels: None,
                initialized: false,
                adts_buffer: Vec::new(),
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

        fn try_initialize_mp4(&mut self) -> Result<bool, String> {
            if self.initialized {
                return Ok(true);
            }

            // Need at least some data to parse MP4 header
            if self.input_buffer.len() < 1024 {
                return Ok(false);
            }

            let cursor = Cursor::new(self.input_buffer.clone());
            let size = self.input_buffer.len() as u64;

            match Mp4Reader::read_header(cursor, size) {
                Ok(mp4) => {
                    // Find the first audio track
                    let mut audio_track_id = None;
                    for (track_id, track) in mp4.tracks() {
                        if let Ok(media_type) = track.media_type() {
                            if media_type == MediaType::AAC {
                                audio_track_id = Some(*track_id);

                                // Extract sample rate and channels from track
                                if let Ok(freq_index) = track.sample_freq_index() {
                                    self.sample_rate = Some(freq_index.freq());
                                }
                                if let Ok(channel_config) = track.channel_config() {
                                    self.channels = Some(channel_config as u8);
                                }

                                break;
                            }
                        }
                    }

                    if let Some(track_id) = audio_track_id {
                        let sample_count = mp4
                            .sample_count(track_id)
                            .map_err(|e| format!("Failed to get sample count: {}", e))?;

                        self.track_id = Some(track_id);
                        self.sample_count = sample_count;
                        self.mp4_reader = Some(mp4);
                        self.initialized = true;

                        debug!(
                            track_id,
                            sample_count,
                            sample_rate = ?self.sample_rate,
                            channels = ?self.channels,
                            "Initialized MP4 AAC decoder"
                        );

                        Ok(true)
                    } else {
                        Err("No AAC audio track found in MP4".to_string())
                    }
                }
                Err(e) => {
                    // Not enough data yet, or invalid MP4
                    if self.input_buffer.len() > 1024 * 1024 {
                        // If we have > 1MB and still can't parse, it's probably not valid MP4
                        Err(format!("Failed to parse MP4 after buffering 1MB: {}", e))
                    } else {
                        // Need more data
                        Ok(false)
                    }
                }
            }
        }
    }

    impl Decoder for AacDecoderMp4 {
        fn decode_i16(
            &mut self,
            input: &[u8],
            output: &mut [i16],
            _fec: bool,
        ) -> Result<usize, String> {
            // Accumulate input data
            if !input.is_empty() {
                self.input_buffer.extend_from_slice(input);
            }

            // Try to initialize MP4 reader if not done yet
            if !self.initialized {
                match self.try_initialize_mp4() {
                    Ok(true) => {
                        // Successfully initialized
                    }
                    Ok(false) => {
                        // Need more data
                        return Ok(0);
                    }
                    Err(e) => {
                        return Err(e);
                    }
                }
            }

            let track_id = match self.track_id {
                Some(id) => id,
                None => return Ok(0),
            };

            // If we've read all samples, we're done
            if self.current_sample_id > self.sample_count {
                return Ok(0);
            }

            //  First, extract all remaining MP4 samples and convert to ADTS frames
            let samples_before = self.current_sample_id;
            while self.current_sample_id <= self.sample_count {
                let mp4_reader = match &mut self.mp4_reader {
                    Some(reader) => reader,
                    None => break,
                };

                // Read the sample from MP4
                let sample = match mp4_reader.read_sample(track_id, self.current_sample_id) {
                    Ok(Some(sample)) => sample,
                    Ok(None) => break,
                    Err(e) => {
                        if e.to_string().contains("UnexpectedEof") {
                            break;
                        }
                        return Err(format!("Failed to read MP4 sample: {}", e));
                    }
                };

                self.current_sample_id += 1;

                // Create ADTS header for this sample using access-unit function
                // codec_id 0x66 = AAC-LC
                let adts_header = create_adts_header(
                    0x66, // AAC-LC
                    self.channels.unwrap_or(2),
                    self.sample_rate.unwrap_or(44100),
                    sample.bytes.len(),
                    false, // no CRC
                );

                // Combine ADTS header + raw AAC data and add to buffer
                self.adts_buffer.extend_from_slice(&adts_header);
                self.adts_buffer.extend_from_slice(&sample.bytes);
            }

            trace!(
                samples_extracted = self.current_sample_id - samples_before,
                adts_buffer_len = self.adts_buffer.len(),
                "extracted MP4 samples to ADTS buffer"
            );

            // Now decode from the ADTS buffer using the same pattern as AacDecoder
            let mut written = 0usize;
            let mut total_consumed = 0usize;

            trace!("starting decode loop");

            loop {
                // Fill FDK with data from ADTS buffer
                let consumed = if self.adts_buffer.is_empty() {
                    trace!("ADTS buffer empty");
                    0
                } else {
                    match self.fdk_decoder.fill(&self.adts_buffer) {
                        Ok(bytes) => {
                            trace!(consumed = bytes, "FDK fill successful");
                            bytes
                        }
                        Err(err) => return Err(format!("Error filling decoder: {}", err)),
                    }
                };

                if consumed > 0 {
                    total_consumed += consumed;
                    self.adts_buffer.drain(..consumed);
                }

                let remaining = output.len().saturating_sub(written);
                if remaining == 0 {
                    trace!("output buffer full");
                    break;
                }

                trace!("attempting decode_frame");
                match self.fdk_decoder.decode_frame(&mut output[written..]) {
                    Ok(()) => {
                        let info = self.fdk_decoder.stream_info();
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

                        let first_frame = written == 0;
                        self.sample_rate = Some(info.sampleRate as u32);
                        self.channels = Some(info.numChannels as u8);
                        written += frame_samples;

                        if first_frame {
                            debug!(
                                sample_rate_hz = info.sampleRate,
                                channels = info.numChannels,
                                frame_samples,
                                bytes_consumed = total_consumed,
                                "decoded MP4 AAC frame"
                            );
                        } else {
                            trace!(
                                sample_rate_hz = info.sampleRate,
                                channels = info.numChannels,
                                frame_samples,
                                bytes_consumed = total_consumed,
                                "decoded MP4 AAC frame"
                            );
                        }
                    }
                    Err(fdk_aac::dec::DecoderError::NOT_ENOUGH_BITS) => {
                        trace!("FDK needs more bits");
                        // need more data
                        break;
                    }
                    Err(err) => {
                        error!("FDK decoding error: {:?}", err);
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

        fn decode_f32(
            &mut self,
            input: &[u8],
            output: &mut [f32],
            fec: bool,
        ) -> Result<usize, String> {
            let mut i16_buf = vec![0i16; output.len()];
            let samples = self.decode_i16(input, &mut i16_buf, fec)?;

            for i in 0..samples {
                output[i] = (i16_buf[i] as f32) / 32768.0;
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
    const TEST_FILE: &str = "A_Tusk_is_used_to_make_costly_gifts";

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

        let input_path = testdata_path(&format!("mac_aac/{}.m4a", TEST_FILE));
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
    fn test_mp4_aac_decoder() {
        use crate::AacDecoderMp4;

        let input_path = testdata_path("mac_aac/A_Tusk_is_used_to_make_costly_gifts.m4a");
        let m4a_bytes = fs::read(&input_path).unwrap();
        assert!(!m4a_bytes.is_empty(), "fixture m4a missing or empty");

        init_tracing();

        let mut decoder = AacDecoderMp4::new();
        decoder.init().expect("Decoder initialization failed");

        let mut decoded = Vec::new();
        let mut scratch = vec![0i16; 16384];

        // Feed all data at once for testing
        match decoder.decode_i16(&m4a_bytes, &mut scratch, false) {
            Ok(written) => {
                trace!(samples_written = written, "first decode pass");
                decoded.extend_from_slice(&scratch[..written]);
            }
            Err(e) => {
                panic!("Decode failed: {}", e);
            }
        }

        // Drain any remaining
        loop {
            let written = decoder.decode_i16(&[], &mut scratch, false).unwrap();
            trace!(samples_written = written, "drain decode pass");
            if written == 0 {
                break;
            }
            decoded.extend_from_slice(&scratch[..written]);
        }

        trace!(
            total_samples = decoded.len(),
            sample_rate = ?decoder.sample_rate(),
            channels = ?decoder.channels(),
            "decoding complete"
        );

        assert!(!decoded.is_empty(), "decoder produced no PCM samples");
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

        let input_path = testdata_path("mac_aac/A_Tusk_is_used_to_make_costly_gifts.m4a");
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
        assert_eq!(config.sample_rate, 16_000);
        assert_eq!(config.channels, 1);

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
