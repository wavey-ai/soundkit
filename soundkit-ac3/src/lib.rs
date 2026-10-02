use frame_header::{EncodingFlag, Endianness};
use oxideav_ac3::{bsi, decoder::SAMPLES_PER_FRAME, syncinfo};
use oxideav_core::{
    CodecId, CodecParameters, Decoder as OxideDecoder, Error as OxideError, Frame,
    Packet as OxidePacket, TimeBase,
};
use soundkit::audio_types::AudioData;
use std::collections::VecDeque;

const MAX_AC3_STREAM_BUFFER_BYTES: usize = 4 * 1024 * 1024;

/// Streaming raw AC-3 syncframe decoder.
///
/// This accepts elementary AC-3 streams, not MP4/Matroska containerized AC-3.
/// Complete syncframes are decoded as they arrive and returned as interleaved
/// signed 16-bit little-endian PCM.
pub struct Ac3Decoder {
    decoder: Box<dyn OxideDecoder>,
    buffer: Vec<u8>,
    pending: VecDeque<AudioData>,
    frame_index: i64,
}

pub fn looks_like_ac3(data: &[u8]) -> bool {
    let Some(offset) = syncinfo::find_syncword(data, 0) else {
        return false;
    };
    if offset > 0 || data.len() < offset + 5 {
        return false;
    }
    syncinfo::parse(&data[offset..]).is_ok()
}

impl Ac3Decoder {
    pub fn try_new() -> Result<Self, String> {
        let params = CodecParameters::audio(CodecId::new("ac3"));
        let decoder = oxideav_ac3::decoder::make_decoder(&params).map_err(oxide_error_to_string)?;
        Ok(Self {
            decoder,
            buffer: Vec::new(),
            pending: VecDeque::new(),
            frame_index: 0,
        })
    }

    pub fn init(&mut self) -> Result<(), String> {
        Ok(())
    }

    pub fn add(&mut self, data: &[u8]) -> Result<Option<AudioData>, String> {
        if !data.is_empty() {
            if self.buffer.len().saturating_add(data.len()) > MAX_AC3_STREAM_BUFFER_BYTES {
                return Err(format!(
                    "AC-3 stream exceeds the {MAX_AC3_STREAM_BUFFER_BYTES} byte buffer budget"
                ));
            }
            self.buffer.extend_from_slice(data);
        }

        if let Some(audio) = self.pending.pop_front() {
            return Ok(Some(audio));
        }

        self.decode_available_frames()?;
        Ok(self.pending.pop_front())
    }

    /// Decodes every whole frame in the buffer. Frames are read at an offset
    /// and the consumed bytes are removed once, also when a frame is refused.
    fn decode_available_frames(&mut self) -> Result<(), String> {
        let mut start = 0;
        let result = self.decode_frames_from(&mut start);
        self.buffer.drain(..start);
        result
    }

    fn decode_frames_from(&mut self, start: &mut usize) -> Result<(), String> {
        loop {
            let Some(sync_offset) = syncinfo::find_syncword(&self.buffer[*start..], 0) else {
                *start = self.buffer.len();
                return Ok(());
            };
            *start += sync_offset;
            let available = &self.buffer[*start..];

            if available.len() < 5 {
                return Ok(());
            }

            let sync = syncinfo::parse(available).map_err(oxide_error_to_string)?;
            let frame_len = sync.frame_length as usize;
            if available.len() < frame_len {
                return Ok(());
            }

            let frame = available[..frame_len].to_vec();
            *start += frame_len;
            let stream_info = bsi::parse(&frame[5..]).map_err(oxide_error_to_string)?;
            let channels = stream_info.nchans as u8;
            if channels == 0 {
                return Err("AC-3 stream reports zero channels".to_string());
            }

            let pkt = OxidePacket::new(0, TimeBase::new(1, sync.sample_rate as i64), frame)
                .with_pts(self.frame_index * SAMPLES_PER_FRAME as i64);
            self.frame_index += 1;
            self.decoder
                .send_packet(&pkt)
                .map_err(oxide_error_to_string)?;

            loop {
                match self.decoder.receive_frame() {
                    Ok(Frame::Audio(audio)) => {
                        let Some(bytes) = audio.data.into_iter().next() else {
                            return Err(
                                "AC-3 decoder returned audio frame with no data".to_string()
                            );
                        };
                        self.pending.push_back(AudioData::new(
                            16,
                            channels,
                            sync.sample_rate,
                            bytes,
                            EncodingFlag::PCMSigned,
                            Endianness::LittleEndian,
                        ));
                    }
                    Ok(_) => continue,
                    Err(OxideError::NeedMore) => break,
                    Err(OxideError::Eof) => break,
                    Err(error) => return Err(oxide_error_to_string(error)),
                }
            }
        }
    }
}

fn oxide_error_to_string(error: OxideError) -> String {
    format!("{error:?}")
}

impl Default for Ac3Decoder {
    fn default() -> Self {
        Self::try_new().expect("failed to create AC-3 decoder")
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

    fn decode_chunks(data: &[u8], chunk_size: usize) -> Vec<AudioData> {
        let mut decoder = Ac3Decoder::try_new().unwrap();
        let mut frames = Vec::new();
        for chunk in data.chunks(chunk_size) {
            if let Some(audio) = decoder.add(chunk).unwrap() {
                frames.push(audio);
            }
            while let Some(audio) = decoder.add(&[]).unwrap() {
                frames.push(audio);
            }
        }
        while let Some(audio) = decoder.add(&[]).unwrap() {
            frames.push(audio);
        }
        frames
    }

    #[test]
    #[ignore = "regenerates the committed raw AC-3 fixture using ffmpeg"]
    fn generate_ac3_fixture_with_ffmpeg() {
        let input = testdata_path("linear16_8/A_Tusk_is_used_to_make_costly_gifts.s16le");
        let output = testdata_path("ac3/A_Tusk_is_used_to_make_costly_gifts.ac3");
        fs::create_dir_all(output.parent().unwrap()).unwrap();
        let status = Command::new("ffmpeg")
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
            .args([
                "-ar", "48000", "-ac", "1", "-c:a", "ac3", "-b:a", "96k", "-f", "ac3",
            ])
            .arg(&output)
            .status()
            .unwrap();
        assert!(status.success());
    }

    #[test]
    fn chunked_decoder_matches_whole_decode() {
        let fixture =
            fs::read(testdata_path("ac3/A_Tusk_is_used_to_make_costly_gifts.ac3")).unwrap();
        assert!(!fixture.is_empty(), "AC-3 fixture missing or empty");

        let whole = decode_chunks(&fixture, fixture.len());
        let chunked = decode_chunks(&fixture, 997);
        assert_eq!(chunked.len(), whole.len());
        assert_eq!(
            chunked
                .iter()
                .flat_map(|frame| frame.data())
                .copied()
                .collect::<Vec<_>>(),
            whole
                .iter()
                .flat_map(|frame| frame.data())
                .copied()
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn decode_ac3_fixture_and_write_golden_wav() {
        let fixture =
            fs::read(testdata_path("ac3/A_Tusk_is_used_to_make_costly_gifts.ac3")).unwrap();
        let frames = decode_chunks(&fixture, 641);
        assert!(!frames.is_empty(), "No AC-3 frames decoded");
        assert!(frames.iter().all(|frame| frame.bits_per_sample() == 16));
        assert!(frames.iter().all(|frame| frame.channel_count() == 1));
        assert!(frames.iter().all(|frame| frame.sampling_rate() == 48_000));

        let mut pcm = Vec::new();
        for frame in frames {
            pcm.extend_from_slice(frame.data());
        }
        assert!(pcm
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
            .any(|sample| sample != 0));

        let samples: Vec<i16> = pcm
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
            .collect();
        let wav = soundkit::wav::generate_wav_buffer(
            &soundkit::audio_types::PcmData::I16(vec![samples]),
            48_000,
        )
        .unwrap();
        let output_path = golden_path("ac3/A_Tusk_is_used_to_make_costly_gifts.decoded.wav");
        fs::create_dir_all(output_path.parent().unwrap()).unwrap();
        fs::write(output_path, wav).unwrap();
    }

    #[test]
    fn ffmpeg_can_decode_ac3_fixture() {
        let input = testdata_path("ac3/A_Tusk_is_used_to_make_costly_gifts.ac3");
        let output = std::env::temp_dir().join("soundkit-ac3-fixture.s16le");
        let status = Command::new("ffmpeg")
            .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
            .arg(&input)
            .args(["-f", "s16le", "-acodec", "pcm_s16le"])
            .arg(&output)
            .status()
            .unwrap();
        assert!(status.success());

        let decoded = fs::read(output).unwrap();
        assert!(decoded
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
            .any(|sample| sample != 0));
    }

    /// `decode_available_frames` as it was: each frame and each skipped run
    /// of bytes removed from the front of the buffer.
    fn reference_decode_available(decoder: &mut Ac3Decoder) -> Result<(), String> {
        loop {
            let Some(sync_offset) = syncinfo::find_syncword(&decoder.buffer, 0) else {
                decoder.buffer.clear();
                return Ok(());
            };
            if sync_offset > 0 {
                decoder.buffer.drain(..sync_offset);
            }
            if decoder.buffer.len() < 5 {
                return Ok(());
            }
            let sync = syncinfo::parse(&decoder.buffer).map_err(oxide_error_to_string)?;
            let frame_len = sync.frame_length as usize;
            if decoder.buffer.len() < frame_len {
                return Ok(());
            }
            let frame = decoder.buffer[..frame_len].to_vec();
            decoder.buffer.drain(..frame_len);
            let stream_info = bsi::parse(&frame[5..]).map_err(oxide_error_to_string)?;
            let channels = stream_info.nchans as u8;
            if channels == 0 {
                return Err("AC-3 stream reports zero channels".to_string());
            }
            let pkt = OxidePacket::new(0, TimeBase::new(1, sync.sample_rate as i64), frame)
                .with_pts(decoder.frame_index * SAMPLES_PER_FRAME as i64);
            decoder.frame_index += 1;
            decoder
                .decoder
                .send_packet(&pkt)
                .map_err(oxide_error_to_string)?;
            loop {
                match decoder.decoder.receive_frame() {
                    Ok(Frame::Audio(audio)) => {
                        let Some(bytes) = audio.data.into_iter().next() else {
                            return Err(
                                "AC-3 decoder returned audio frame with no data".to_string()
                            );
                        };
                        decoder.pending.push_back(AudioData::new(
                            16,
                            channels,
                            sync.sample_rate,
                            bytes,
                            EncodingFlag::PCMSigned,
                            Endianness::LittleEndian,
                        ));
                    }
                    Ok(_) => continue,
                    Err(OxideError::NeedMore) => break,
                    Err(OxideError::Eof) => break,
                    Err(error) => return Err(oxide_error_to_string(error)),
                }
            }
        }
    }

    fn reference_add(decoder: &mut Ac3Decoder, data: &[u8]) -> Result<Option<AudioData>, String> {
        if !data.is_empty() {
            if decoder.buffer.len().saturating_add(data.len()) > MAX_AC3_STREAM_BUFFER_BYTES {
                return Err(format!(
                    "AC-3 stream exceeds the {MAX_AC3_STREAM_BUFFER_BYTES} byte buffer budget"
                ));
            }
            decoder.buffer.extend_from_slice(data);
        }
        if let Some(audio) = decoder.pending.pop_front() {
            return Ok(Some(audio));
        }
        reference_decode_available(decoder)?;
        Ok(decoder.pending.pop_front())
    }

    type AudioView = Option<(u8, u8, u32, Vec<u8>)>;

    fn view(result: Result<Option<AudioData>, String>) -> Result<AudioView, String> {
        result.map(|audio| {
            audio.map(|audio| {
                (
                    audio.bits_per_sample(),
                    audio.channel_count(),
                    audio.sampling_rate(),
                    audio.data().clone(),
                )
            })
        })
    }

    /// The offset reader returns the same audio, errors and buffered bytes as
    /// the front-removing reader for the fixture, the fixture behind junk,
    /// corrupt and truncated copies, in pushes of 1 byte to the whole stream.
    #[test]
    fn offset_reader_matches_front_removing_reader() {
        let fixture =
            fs::read(testdata_path("ac3/A_Tusk_is_used_to_make_costly_gifts.ac3")).unwrap();
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut junk_first: Vec<u8> = (0..777).map(|_| next() as u8 & 0x7f).collect();
        junk_first.extend_from_slice(&fixture);
        let mut corrupt = fixture.clone();
        for _ in 0..12 {
            let at = (next() as usize) % corrupt.len();
            corrupt[at] = next() as u8;
        }
        let streams = [
            fixture.clone(),
            junk_first,
            corrupt,
            fixture[..fixture.len() * 2 / 3].to_vec(),
        ];
        let mut frames = 0;
        for stream in &streams {
            for chunk in [1usize, 7, 1_000, 65_536, stream.len()] {
                let mut actual = Ac3Decoder::try_new().unwrap();
                let mut reference = Ac3Decoder::try_new().unwrap();
                let pieces = stream.chunks(chunk).chain(std::iter::repeat_n(&[][..], 4));
                for piece in pieces {
                    loop {
                        let got = view(actual.add(piece));
                        let want = view(reference_add(&mut reference, piece));
                        assert_eq!(got, want);
                        assert_eq!(actual.buffer, reference.buffer);
                        frames += usize::from(matches!(want, Ok(Some(_))));
                        if !matches!(want, Ok(Some(_))) || !piece.is_empty() {
                            break;
                        }
                    }
                }
            }
        }
        assert!(frames > 0);
    }
}
