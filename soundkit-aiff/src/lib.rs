use aifc::{AifcReader, Sample, SampleFormat};
use audio_codec_algorithms::{decode_adpcm_ima_ima4, decode_alaw, decode_ulaw, AdpcmImaState};
use frame_header::{EncodingFlag, Endianness};
use soundkit::audio_types::AudioData;
use std::io::Cursor;

const MAX_CHANNELS: u8 = 32;
const MAX_COMM_BYTES: usize = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ContainerKind {
    Aiff,
    Aifc,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StreamSampleFormat {
    Unsigned8,
    SignedBe(u8),
    SignedLe(u8),
    Float32Be,
    Float64Be,
    Ulaw,
    Alaw,
    Ima4,
}

impl StreamSampleFormat {
    fn encoded_group_bytes(self, channels: u8) -> usize {
        let channels = usize::from(channels);
        match self {
            Self::Unsigned8 | Self::Ulaw | Self::Alaw => 1,
            Self::SignedBe(bytes) | Self::SignedLe(bytes) => usize::from(bytes),
            Self::Float32Be => 4,
            Self::Float64Be => 8,
            Self::Ima4 => 34 * channels,
        }
    }

    fn output_contract(self) -> (u8, EncodingFlag) {
        match self {
            Self::Unsigned8
            | Self::SignedBe(1)
            | Self::SignedLe(1)
            | Self::Ulaw
            | Self::Alaw
            | Self::Ima4 => (16, EncodingFlag::PCMSigned),
            Self::SignedBe(bytes) | Self::SignedLe(bytes) => {
                (bytes.saturating_mul(8), EncodingFlag::PCMSigned)
            }
            Self::Float32Be | Self::Float64Be => (32, EncodingFlag::PCMFloat),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct StreamInfo {
    sample_rate: u32,
    channels: u8,
    format: StreamSampleFormat,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ParseState {
    FormHeader,
    ChunkHeader,
    Comm {
        size: usize,
        padded: bool,
    },
    SsndHeader {
        remaining: u64,
        padded: bool,
    },
    SsndOffset {
        skip: u64,
        remaining_audio: u64,
        padded: bool,
    },
    Audio {
        remaining: u64,
        padded: bool,
    },
    Skip {
        remaining: u64,
        padded: bool,
    },
    Padding,
    Done,
}

/// Incremental AIFF and AIFF-C decoder.
///
/// The parser retains only incomplete container headers and one incomplete
/// sample or IMA4 channel group. Unknown chunks are skipped incrementally.
pub struct AiffDecoder {
    buffer: Vec<u8>,
    pending_audio: Vec<u8>,
    state: ParseState,
    container: Option<ContainerKind>,
    stream_info: Option<StreamInfo>,
    form_remaining: u64,
    padding_next: ParseState,
    ima4_state: [AdpcmImaState; 2],
    finished: bool,
}

impl AiffDecoder {
    pub fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(4096),
            pending_audio: Vec::new(),
            state: ParseState::FormHeader,
            container: None,
            stream_info: None,
            form_remaining: 0,
            padding_next: ParseState::ChunkHeader,
            ima4_state: [AdpcmImaState::new(), AdpcmImaState::new()],
            finished: false,
        }
    }

    pub fn init(&mut self) -> Result<(), String> {
        Ok(())
    }

    pub fn add(&mut self, data: &[u8]) -> Result<Option<AudioData>, String> {
        if self.finished {
            return Ok(None);
        }
        let finalizing = data.is_empty();
        self.buffer.extend_from_slice(data);
        let pcm = self.parse_available()?;
        if finalizing {
            if self.state != ParseState::Done {
                return Err(format!("truncated AIFF stream in state {:?}", self.state));
            }
            if !self.pending_audio.is_empty() {
                return Err("AIFF sound data ends inside an encoded sample group".to_string());
            }
            self.finished = true;
        }
        if pcm.is_empty() {
            return Ok(None);
        }
        let info = self
            .stream_info
            .ok_or_else(|| "AIFF PCM arrived before COMM metadata".to_string())?;
        let (bits_per_sample, encoding) = info.format.output_contract();
        Ok(Some(AudioData::new(
            bits_per_sample,
            info.channels,
            info.sample_rate,
            pcm,
            encoding,
            Endianness::LittleEndian,
        )))
    }

    pub fn buffered_bytes(&self) -> usize {
        self.buffer.len() + self.pending_audio.len()
    }

    fn parse_available(&mut self) -> Result<Vec<u8>, String> {
        let mut position = 0usize;
        let mut pcm = Vec::new();
        loop {
            let available = self.buffer.len().saturating_sub(position);
            match self.state {
                ParseState::FormHeader => {
                    if available < 12 {
                        break;
                    }
                    let header = &self.buffer[position..position + 12];
                    if &header[..4] != b"FORM" {
                        return Err("AIFF stream does not start with FORM".to_string());
                    }
                    let form_size = u32::from_be_bytes(header[4..8].try_into().unwrap());
                    if form_size < 4 {
                        return Err("AIFF FORM is shorter than its type field".to_string());
                    }
                    self.container = Some(match &header[8..12] {
                        b"AIFF" => ContainerKind::Aiff,
                        b"AIFC" => ContainerKind::Aifc,
                        kind => {
                            return Err(format!(
                                "unsupported FORM type {}",
                                String::from_utf8_lossy(kind)
                            ))
                        }
                    });
                    self.form_remaining = u64::from(form_size - 4);
                    position += 12;
                    self.state = self.next_chunk_state();
                }
                ParseState::ChunkHeader => {
                    if self.form_remaining == 0 {
                        self.state = ParseState::Done;
                        continue;
                    }
                    if self.form_remaining < 8 {
                        return Err("AIFF FORM ends inside a chunk header".to_string());
                    }
                    if available < 8 {
                        break;
                    }
                    let id: [u8; 4] = self.buffer[position..position + 4].try_into().unwrap();
                    let size = u32::from_be_bytes(
                        self.buffer[position + 4..position + 8].try_into().unwrap(),
                    );
                    self.consume_form(8)?;
                    position += 8;
                    let padded = size & 1 != 0;
                    let required = u64::from(size) + u64::from(padded);
                    if required > self.form_remaining {
                        return Err(format!(
                            "AIFF chunk {} exceeds the FORM boundary",
                            String::from_utf8_lossy(&id)
                        ));
                    }
                    self.state = match &id {
                        b"COMM" => {
                            if size as usize > MAX_COMM_BYTES {
                                return Err(format!(
                                    "AIFF COMM exceeds the {MAX_COMM_BYTES} byte budget"
                                ));
                            }
                            ParseState::Comm {
                                size: size as usize,
                                padded,
                            }
                        }
                        b"SSND" => {
                            if size < 8 {
                                return Err("AIFF SSND is shorter than its header".to_string());
                            }
                            if self.stream_info.is_none() {
                                return Err("AIFF SSND appears before COMM".to_string());
                            }
                            ParseState::SsndHeader {
                                remaining: u64::from(size),
                                padded,
                            }
                        }
                        _ => ParseState::Skip {
                            remaining: u64::from(size),
                            padded,
                        },
                    };
                }
                ParseState::Comm { size, padded } => {
                    if available < size {
                        break;
                    }
                    let payload = &self.buffer[position..position + size];
                    let container = self.container.expect("FORM parsed before chunks");
                    self.stream_info = Some(parse_stream_info(payload, container)?);
                    self.consume_form(size as u64)?;
                    position += size;
                    self.finish_chunk(padded);
                }
                ParseState::SsndHeader { remaining, padded } => {
                    if available < 8 {
                        break;
                    }
                    let offset =
                        u32::from_be_bytes(self.buffer[position..position + 4].try_into().unwrap())
                            as u64;
                    let audio_and_offset = remaining - 8;
                    if offset > audio_and_offset {
                        return Err("AIFF SSND offset exceeds its chunk".to_string());
                    }
                    self.consume_form(8)?;
                    position += 8;
                    self.state = ParseState::SsndOffset {
                        skip: offset,
                        remaining_audio: audio_and_offset - offset,
                        padded,
                    };
                }
                ParseState::SsndOffset {
                    skip,
                    remaining_audio,
                    padded,
                } => {
                    let take = available.min(skip as usize);
                    position += take;
                    self.consume_form(take as u64)?;
                    if take as u64 == skip {
                        self.state = ParseState::Audio {
                            remaining: remaining_audio,
                            padded,
                        };
                    } else {
                        self.state = ParseState::SsndOffset {
                            skip: skip - take as u64,
                            remaining_audio,
                            padded,
                        };
                        break;
                    }
                }
                ParseState::Audio { remaining, padded } => {
                    let take = available.min(remaining as usize);
                    let info = self.stream_info.expect("COMM checked before SSND");
                    decode_stream_bytes(
                        info,
                        &self.buffer[position..position + take],
                        &mut self.pending_audio,
                        &mut self.ima4_state,
                        &mut pcm,
                    )?;
                    position += take;
                    self.consume_form(take as u64)?;
                    if take as u64 == remaining {
                        if !self.pending_audio.is_empty() {
                            return Err("AIFF SSND ends inside an encoded sample group".to_string());
                        }
                        self.finish_chunk(padded);
                    } else {
                        self.state = ParseState::Audio {
                            remaining: remaining - take as u64,
                            padded,
                        };
                        break;
                    }
                }
                ParseState::Skip { remaining, padded } => {
                    let take = available.min(remaining as usize);
                    position += take;
                    self.consume_form(take as u64)?;
                    if take as u64 == remaining {
                        self.finish_chunk(padded);
                    } else {
                        self.state = ParseState::Skip {
                            remaining: remaining - take as u64,
                            padded,
                        };
                        break;
                    }
                }
                ParseState::Padding => {
                    if available == 0 {
                        break;
                    }
                    position += 1;
                    self.consume_form(1)?;
                    self.state = self.padding_next;
                }
                ParseState::Done => {
                    if available != 0 {
                        return Err("AIFF stream has bytes after the FORM boundary".to_string());
                    }
                    break;
                }
            }
        }
        self.buffer.drain(..position);
        Ok(pcm)
    }

    fn consume_form(&mut self, count: u64) -> Result<(), String> {
        self.form_remaining = self
            .form_remaining
            .checked_sub(count)
            .ok_or_else(|| "AIFF parser crossed the FORM boundary".to_string())?;
        Ok(())
    }

    fn next_chunk_state(&self) -> ParseState {
        if self.form_remaining == 0 {
            ParseState::Done
        } else {
            ParseState::ChunkHeader
        }
    }

    fn finish_chunk(&mut self, padded: bool) {
        let next = self.next_chunk_state();
        if padded {
            self.padding_next = next;
            self.state = ParseState::Padding;
        } else {
            self.state = next;
        }
    }
}

impl Default for AiffDecoder {
    fn default() -> Self {
        Self::new()
    }
}

fn parse_stream_info(data: &[u8], container: ContainerKind) -> Result<StreamInfo, String> {
    if data.len() < 18 {
        return Err("AIFF COMM is shorter than 18 bytes".to_string());
    }
    let channels = u16::from_be_bytes([data[0], data[1]]);
    let channels = u8::try_from(channels)
        .ok()
        .filter(|value| (1..=MAX_CHANNELS).contains(value))
        .ok_or_else(|| format!("invalid AIFF channel count: {channels}"))?;
    let sample_size = u16::from_be_bytes([data[6], data[7]]);
    let sample_rate = parse_extended_sample_rate(&data[8..18])?;
    let signed_be = || match sample_size {
        1..=8 => Ok(StreamSampleFormat::SignedBe(1)),
        9..=16 => Ok(StreamSampleFormat::SignedBe(2)),
        17..=24 => Ok(StreamSampleFormat::SignedBe(3)),
        25..=32 => Ok(StreamSampleFormat::SignedBe(4)),
        _ => Err(format!("unsupported AIFF sample size: {sample_size}")),
    };
    let format = match container {
        ContainerKind::Aiff => signed_be()?,
        ContainerKind::Aifc => {
            if data.len() < 22 {
                return Err("AIFF-C COMM has no compression type".to_string());
            }
            match &data[18..22] {
                b"NONE" => signed_be()?,
                b"raw " => StreamSampleFormat::Unsigned8,
                b"twos" => StreamSampleFormat::SignedBe(2),
                b"sowt" => StreamSampleFormat::SignedLe(2),
                b"in24" => StreamSampleFormat::SignedBe(3),
                b"in32" => StreamSampleFormat::SignedBe(4),
                b"23ni" => StreamSampleFormat::SignedLe(4),
                b"FL32" | b"fl32" => StreamSampleFormat::Float32Be,
                b"FL64" | b"fl64" => StreamSampleFormat::Float64Be,
                b"ULAW" | b"ulaw" => StreamSampleFormat::Ulaw,
                b"ALAW" | b"alaw" => StreamSampleFormat::Alaw,
                b"ima4" => StreamSampleFormat::Ima4,
                tag => {
                    return Err(format!(
                        "unsupported AIFF-C compression type: {}",
                        String::from_utf8_lossy(tag)
                    ))
                }
            }
        }
    };
    if format == StreamSampleFormat::Ima4 && channels > 2 {
        return Err("AIFF-C IMA4 supports at most two channels".to_string());
    }
    Ok(StreamInfo {
        sample_rate,
        channels,
        format,
    })
}

fn parse_extended_sample_rate(data: &[u8]) -> Result<u32, String> {
    if data.len() != 10 {
        return Err("AIFF sample rate needs 10 bytes".to_string());
    }
    let exponent_word = u16::from_be_bytes([data[0], data[1]]);
    if exponent_word & 0x8000 != 0 {
        return Err("AIFF sample rate is negative".to_string());
    }
    let exponent = exponent_word & 0x7fff;
    let mantissa = u64::from_be_bytes(data[2..10].try_into().unwrap());
    if exponent == 0 && mantissa == 0 {
        return Err("AIFF sample rate is zero".to_string());
    }
    if exponent == 0x7fff {
        return Err("AIFF sample rate is not finite".to_string());
    }
    let value = (mantissa as f64) * 2f64.powi(i32::from(exponent) - 16_383 - 63);
    parse_sample_rate(value)
}

fn decode_stream_bytes(
    info: StreamInfo,
    input: &[u8],
    pending: &mut Vec<u8>,
    ima4_state: &mut [AdpcmImaState; 2],
    output: &mut Vec<u8>,
) -> Result<(), String> {
    pending.extend_from_slice(input);
    let group_bytes = info.format.encoded_group_bytes(info.channels);
    let complete_bytes = pending.len() / group_bytes * group_bytes;
    let mut complete = std::mem::take(pending);
    *pending = complete.split_off(complete_bytes);
    if pending.len() >= group_bytes {
        return Err("AIFF decoder retained more than one encoded group".to_string());
    }

    match info.format {
        StreamSampleFormat::Unsigned8 => {
            output.reserve(complete.len() * 2);
            for value in complete {
                output.extend_from_slice(&((i16::from(value) - 128) << 8).to_le_bytes());
            }
        }
        StreamSampleFormat::SignedBe(1) | StreamSampleFormat::SignedLe(1) => {
            output.reserve(complete.len() * 2);
            for value in complete {
                output.extend_from_slice(&(i16::from(value as i8) << 8).to_le_bytes());
            }
        }
        StreamSampleFormat::SignedBe(bytes) => {
            let bytes = usize::from(bytes);
            output.reserve(complete.len());
            for sample in complete.chunks_exact(bytes) {
                output.extend(sample.iter().rev());
            }
        }
        StreamSampleFormat::SignedLe(_) => output.extend_from_slice(&complete),
        StreamSampleFormat::Float32Be => {
            output.reserve(complete.len());
            for sample in complete.chunks_exact(4) {
                output.extend_from_slice(
                    &f32::from_be_bytes(sample.try_into().unwrap()).to_le_bytes(),
                );
            }
        }
        StreamSampleFormat::Float64Be => {
            output.reserve(complete.len() / 2);
            for sample in complete.chunks_exact(8) {
                let value = f64::from_be_bytes(sample.try_into().unwrap()) as f32;
                output.extend_from_slice(&value.to_le_bytes());
            }
        }
        StreamSampleFormat::Ulaw => {
            output.reserve(complete.len() * 2);
            for sample in complete {
                output.extend_from_slice(&decode_ulaw(sample).to_le_bytes());
            }
        }
        StreamSampleFormat::Alaw => {
            output.reserve(complete.len() * 2);
            for sample in complete {
                output.extend_from_slice(&decode_alaw(sample).to_le_bytes());
            }
        }
        StreamSampleFormat::Ima4 => {
            let channels = usize::from(info.channels);
            output.reserve(complete.len() * 4);
            for group in complete.chunks_exact(group_bytes) {
                let mut decoded = [[0i16; 64]; 2];
                for channel in 0..channels {
                    let start = channel * 34;
                    let block: &[u8; 34] = group[start..start + 34].try_into().unwrap();
                    decode_adpcm_ima_ima4(block, &mut ima4_state[channel], &mut decoded[channel]);
                }
                for frame in 0..64 {
                    for channel_samples in decoded.iter().take(channels) {
                        output.extend_from_slice(&channel_samples[frame].to_le_bytes());
                    }
                }
            }
        }
    }
    Ok(())
}

pub fn decode_aiff_container(data: &[u8]) -> Result<AudioData, String> {
    if data.is_empty() {
        return Err("AIFF input is empty".to_string());
    }

    let mut reader =
        AifcReader::new(Cursor::new(data.to_vec())).map_err(|error| format!("{error:?}"))?;
    let info = reader.info();
    let sample_rate = parse_sample_rate(info.sample_rate)?;
    let channels = parse_channels(info.channels)?;
    let (bits_per_sample, audio_format) = output_format(info.sample_format)?;

    let mut pcm = Vec::new();
    for sample in reader.samples().map_err(|error| format!("{error:?}"))? {
        append_sample(sample.map_err(|error| format!("{error:?}"))?, &mut pcm);
    }

    Ok(AudioData::new(
        bits_per_sample,
        channels,
        sample_rate,
        pcm,
        audio_format,
        Endianness::LittleEndian,
    ))
}

fn parse_sample_rate(value: f64) -> Result<u32, String> {
    if !value.is_finite() || value <= 0.0 || value > u32::MAX as f64 {
        return Err(format!("Invalid AIFF sample rate: {value}"));
    }
    Ok(value.round() as u32)
}

fn parse_channels(value: i16) -> Result<u8, String> {
    if value <= 0 || value > u8::MAX as i16 {
        return Err(format!("Invalid AIFF channel count: {value}"));
    }
    Ok(value as u8)
}

fn output_format(sample_format: SampleFormat) -> Result<(u8, EncodingFlag), String> {
    match sample_format {
        SampleFormat::U8
        | SampleFormat::I8
        | SampleFormat::I16
        | SampleFormat::I16LE
        | SampleFormat::CompressedUlaw
        | SampleFormat::CompressedAlaw
        | SampleFormat::CompressedIma4 => Ok((16, EncodingFlag::PCMSigned)),
        SampleFormat::I24 => Ok((24, EncodingFlag::PCMSigned)),
        SampleFormat::I32 | SampleFormat::I32LE => Ok((32, EncodingFlag::PCMSigned)),
        SampleFormat::F32 | SampleFormat::F64 => Ok((32, EncodingFlag::PCMFloat)),
        SampleFormat::Custom(tag) => Err(format!(
            "Unsupported AIFF compression type: {}",
            String::from_utf8_lossy(&tag)
        )),
    }
}

fn append_sample(sample: Sample, out: &mut Vec<u8>) {
    match sample {
        Sample::U8(value) => {
            let signed = (i16::from(value) - 128) << 8;
            out.extend_from_slice(&signed.to_le_bytes());
        }
        Sample::I8(value) => {
            let widened = i16::from(value) << 8;
            out.extend_from_slice(&widened.to_le_bytes());
        }
        Sample::I16(value) => out.extend_from_slice(&value.to_le_bytes()),
        Sample::I24(value) => out.extend_from_slice(&value.to_le_bytes()[..3]),
        Sample::I32(value) => out.extend_from_slice(&value.to_le_bytes()),
        Sample::F32(value) => out.extend_from_slice(&value.to_le_bytes()),
        Sample::F64(value) => out.extend_from_slice(&(value as f32).to_le_bytes()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::process::Command;

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

    fn decode_chunks(data: &[u8], chunk_size: usize) -> AudioData {
        let mut decoder = AiffDecoder::new();
        let mut frames = Vec::new();
        for chunk in data.chunks(chunk_size) {
            if let Some(frame) = decoder.add(chunk).unwrap() {
                frames.push(frame);
            }
        }
        if let Some(frame) = decoder.add(&[]).unwrap() {
            frames.push(frame);
        }
        let first = frames.first().expect("AIFF emitted PCM before or at EOF");
        let mut pcm = Vec::new();
        for frame in &frames {
            assert_eq!(frame.bits_per_sample(), first.bits_per_sample());
            assert_eq!(frame.channel_count(), first.channel_count());
            assert_eq!(frame.sampling_rate(), first.sampling_rate());
            pcm.extend_from_slice(frame.data());
        }
        AudioData::new(
            first.bits_per_sample(),
            first.channel_count(),
            first.sampling_rate(),
            pcm,
            first.audio_format(),
            first.endianness(),
        )
    }

    #[test]
    fn emits_pcm_before_eof_with_bounded_parser_state() {
        let fixture = fs::read(testdata_path(
            "aiff/A_Tusk_is_used_to_make_costly_gifts.aiff",
        ))
        .unwrap();
        let mut decoder = AiffDecoder::new();
        let mut first_output_at = None;
        for (index, chunk) in fixture.chunks(257).enumerate() {
            if decoder.add(chunk).unwrap().is_some() && first_output_at.is_none() {
                first_output_at = Some((index + 1) * 257);
            }
            assert!(decoder.buffered_bytes() < MAX_COMM_BYTES + 257);
        }
        decoder.add(&[]).unwrap();
        assert!(first_output_at.unwrap() < fixture.len());
    }

    #[test]
    #[ignore = "regenerates committed AIFF and AIFF-C fixtures using ffmpeg"]
    fn generate_aiff_fixtures_with_ffmpeg() {
        let input = testdata_path("linear16_8/A_Tusk_is_used_to_make_costly_gifts.s16le");
        let aiff_output = testdata_path("aiff/A_Tusk_is_used_to_make_costly_gifts.aiff");
        let aifc_output = testdata_path("aifc/A_Tusk_is_used_to_make_costly_gifts.aifc");
        fs::create_dir_all(aiff_output.parent().unwrap()).unwrap();
        fs::create_dir_all(aifc_output.parent().unwrap()).unwrap();

        let aiff_status = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "s16le",
                "-ar",
                "8000",
                "-ac",
                "1",
                "-i",
            ])
            .arg(&input)
            .args(["-c:a", "pcm_s16be", "-f", "aiff"])
            .arg(&aiff_output)
            .status()
            .unwrap();
        assert!(aiff_status.success());

        let aifc_status = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "s16le",
                "-ar",
                "8000",
                "-ac",
                "1",
                "-i",
            ])
            .arg(&input)
            .args(["-c:a", "pcm_s16le", "-f", "aiff"])
            .arg(&aifc_output)
            .status()
            .unwrap();
        assert!(aifc_status.success());
    }

    #[test]
    fn chunked_decoder_matches_whole_decode() {
        for fixture_name in [
            "aiff/A_Tusk_is_used_to_make_costly_gifts.aiff",
            "aifc/A_Tusk_is_used_to_make_costly_gifts.aifc",
        ] {
            let fixture = fs::read(testdata_path(fixture_name)).unwrap();
            assert!(!fixture.is_empty(), "AIFF fixture missing or empty");

            let whole = decode_chunks(&fixture, fixture.len());
            let chunked = decode_chunks(&fixture, 997);
            assert_eq!(chunked.bits_per_sample(), whole.bits_per_sample());
            assert_eq!(chunked.channel_count(), whole.channel_count());
            assert_eq!(chunked.sampling_rate(), whole.sampling_rate());
            assert_eq!(chunked.data(), whole.data());
        }
    }

    #[test]
    fn streamed_aifc_encodings_match_seekable_reference() {
        for fixture_name in [
            "aifc/stream-alaw.aifc",
            "aifc/stream-f32be.aifc",
            "aifc/stream-f64be.aifc",
            "aifc/stream-ima4.aifc",
            "aifc/stream-s24be.aifc",
            "aifc/stream-s32be.aifc",
            "aifc/stream-ulaw.aifc",
        ] {
            let fixture = fs::read(testdata_path(fixture_name)).unwrap();
            let streamed = decode_chunks(&fixture, 113);
            let reference = decode_aiff_container(&fixture).unwrap();
            assert_eq!(streamed.bits_per_sample(), reference.bits_per_sample());
            assert_eq!(streamed.channel_count(), reference.channel_count());
            assert_eq!(streamed.sampling_rate(), reference.sampling_rate());
            assert_eq!(streamed.audio_format(), reference.audio_format());
            assert_eq!(streamed.data(), reference.data(), "{fixture_name}");
        }
    }

    #[test]
    fn decode_aiff_fixtures_and_write_golden_wavs() {
        for (fixture_name, golden_name) in [
            (
                "aiff/A_Tusk_is_used_to_make_costly_gifts.aiff",
                "aiff/A_Tusk_is_used_to_make_costly_gifts.decoded.wav",
            ),
            (
                "aifc/A_Tusk_is_used_to_make_costly_gifts.aifc",
                "aifc/A_Tusk_is_used_to_make_costly_gifts.decoded.wav",
            ),
        ] {
            let fixture = fs::read(testdata_path(fixture_name)).unwrap();
            let audio = decode_chunks(&fixture, 641);
            assert_eq!(audio.bits_per_sample(), 16);
            assert_eq!(audio.channel_count(), 1);
            assert_eq!(audio.sampling_rate(), 8_000);
            assert!(audio
                .data()
                .chunks_exact(2)
                .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
                .any(|sample| sample != 0));

            let samples: Vec<i16> = audio
                .data()
                .chunks_exact(2)
                .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
                .collect();
            let wav = soundkit::wav::generate_wav_buffer(
                &soundkit::audio_types::PcmData::I16(vec![samples]),
                8_000,
            )
            .unwrap();
            let output_path = golden_path(golden_name);
            fs::create_dir_all(output_path.parent().unwrap()).unwrap();
            fs::write(output_path, wav).unwrap();
        }
    }

    #[test]
    fn native_decode_matches_ffmpeg_pcm() {
        for fixture_name in [
            "aiff/A_Tusk_is_used_to_make_costly_gifts.aiff",
            "aifc/A_Tusk_is_used_to_make_costly_gifts.aifc",
        ] {
            let input = testdata_path(fixture_name);
            let ffmpeg_pcm = std::env::temp_dir().join(format!(
                "soundkit-{}-ffmpeg.s16le",
                fixture_name.replace('/', "-")
            ));
            let status = Command::new("ffmpeg")
                .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
                .arg(&input)
                .args(["-f", "s16le", "-acodec", "pcm_s16le"])
                .arg(&ffmpeg_pcm)
                .status()
                .unwrap();
            assert!(status.success());

            let fixture = fs::read(input).unwrap();
            let audio = decode_chunks(&fixture, 1024);
            assert_eq!(audio.data(), &fs::read(ffmpeg_pcm).unwrap());
        }
    }
}
