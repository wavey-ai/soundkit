use frame_header::{EncodingFlag, Endianness};
use oxideav_ac3::{bsi, decoder::SAMPLES_PER_FRAME, syncinfo};
use oxideav_core::{
    CodecId, CodecParameters, Decoder as OxideDecoder, Error as OxideError, Frame,
    Packet as OxidePacket, TimeBase,
};
use soundkit::audio_types::AudioData;
use std::collections::VecDeque;

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
            self.buffer.extend_from_slice(data);
        }

        if let Some(audio) = self.pending.pop_front() {
            return Ok(Some(audio));
        }

        self.decode_available_frames()?;
        Ok(self.pending.pop_front())
    }

    fn decode_available_frames(&mut self) -> Result<(), String> {
        loop {
            let Some(sync_offset) = syncinfo::find_syncword(&self.buffer, 0) else {
                self.buffer.clear();
                return Ok(());
            };
            if sync_offset > 0 {
                self.buffer.drain(..sync_offset);
            }

            if self.buffer.len() < 5 {
                return Ok(());
            }

            let sync = syncinfo::parse(&self.buffer).map_err(oxide_error_to_string)?;
            let frame_len = sync.frame_length as usize;
            if self.buffer.len() < frame_len {
                return Ok(());
            }

            let frame = self.buffer[..frame_len].to_vec();
            self.buffer.drain(..frame_len);
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
}
