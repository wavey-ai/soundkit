use frame_header::{EncodingFlag, Endianness};
use lewton::audio::{read_audio_packet_generic, PreviousWindowRight};
use lewton::header::{
    read_header_comment, read_header_ident, read_header_setup, IdentHeader, SetupHeader,
};
use lewton::samples::{InterleavedSamples, Samples};
use soundkit::audio_types::AudioData;
use soundkit::ogg::{OggPacket as SharedOggPacket, OggPacketParser as SharedOggPacketParser};
#[cfg(test)]
const MAX_OGG_INPUT_CHUNK_BYTES: usize = 4 * 1024 * 1024;

#[derive(Clone, Copy)]
struct VorbisStreamInfo {
    sample_rate: u32,
    channels: u8,
    serial: u32,
}

enum VorbisDecodeState {
    HeaderIdent,
    HeaderComment {
        ident: IdentHeader,
    },
    HeaderSetup {
        ident: IdentHeader,
    },
    Audio {
        ident: IdentHeader,
        setup: SetupHeader,
        pwr: PreviousWindowRight,
        cur_absgp: Option<u64>,
    },
}

#[derive(Clone, Copy)]
struct VorbisPacketStreamInfo {
    sample_rate: u32,
    channels: u8,
}

/// Streaming raw Vorbis packet decoder.
///
/// This accepts the three Vorbis header packets followed by raw Vorbis audio
/// packets, as used by Matroska/WebM `A_VORBIS`. Ogg page parsing stays in
/// `VorbisDecoder`.
pub struct VorbisPacketDecoder {
    state: VorbisDecodeState,
    info: Option<VorbisPacketStreamInfo>,
}

impl VorbisPacketDecoder {
    pub fn new() -> Self {
        Self {
            state: VorbisDecodeState::HeaderIdent,
            info: None,
        }
    }

    pub fn init(&mut self) -> Result<(), String> {
        Ok(())
    }

    pub fn sample_rate(&self) -> Option<u32> {
        self.info.map(|info| info.sample_rate)
    }

    pub fn channels(&self) -> Option<u8> {
        self.info.map(|info| info.channels)
    }

    pub fn decode_packet(&mut self, packet: &[u8]) -> Result<Option<Vec<i16>>, String> {
        let state = std::mem::replace(&mut self.state, VorbisDecodeState::HeaderIdent);
        match state {
            VorbisDecodeState::HeaderIdent => {
                let ident = read_header_ident(packet).map_err(|error| format!("{error:?}"))?;
                self.info = Some(VorbisPacketStreamInfo {
                    sample_rate: ident.audio_sample_rate,
                    channels: ident.audio_channels,
                });
                self.state = VorbisDecodeState::HeaderComment { ident };
                Ok(None)
            }
            VorbisDecodeState::HeaderComment { ident } => {
                read_header_comment(packet).map_err(|error| format!("{error:?}"))?;
                self.state = VorbisDecodeState::HeaderSetup { ident };
                Ok(None)
            }
            VorbisDecodeState::HeaderSetup { ident } => {
                let setup = read_header_setup(
                    packet,
                    ident.audio_channels,
                    (ident.blocksize_0, ident.blocksize_1),
                )
                .map_err(|error| format!("{error:?}"))?;
                self.state = VorbisDecodeState::Audio {
                    ident,
                    setup,
                    pwr: PreviousWindowRight::new(),
                    cur_absgp: None,
                };
                Ok(None)
            }
            VorbisDecodeState::Audio {
                ident,
                setup,
                mut pwr,
                ..
            } => {
                let decoded: InterleavedSamples<i16> =
                    read_audio_packet_generic(&ident, &setup, packet, &mut pwr)
                        .map_err(|error| format!("{error:?}"))?;

                self.state = VorbisDecodeState::Audio {
                    ident,
                    setup,
                    pwr,
                    cur_absgp: None,
                };

                Ok(Some(decoded.samples))
            }
        }
    }
}

impl Default for VorbisPacketDecoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Streaming Ogg Vorbis decoder.
///
/// Vorbis is carried in Ogg for normal `.ogg` files. This decoder parses Ogg
/// pages incrementally, decodes Vorbis packets with the pure Rust `lewton`
/// decoder, and returns interleaved signed 16-bit PCM.
pub struct VorbisDecoder {
    parser: SharedOggPacketParser,
    state: VorbisDecodeState,
    info: Option<VorbisStreamInfo>,
}

impl VorbisDecoder {
    pub fn new() -> Self {
        Self {
            parser: SharedOggPacketParser::new(),
            state: VorbisDecodeState::HeaderIdent,
            info: None,
        }
    }

    pub fn init(&mut self) -> Result<(), String> {
        Ok(())
    }

    pub fn sample_rate(&self) -> Option<u32> {
        self.info.map(|info| info.sample_rate)
    }

    pub fn channels(&self) -> Option<u8> {
        self.info.map(|info| info.channels)
    }

    /// Feed more Ogg Vorbis bytes. Returns decoded interleaved S16LE PCM when
    /// complete packets have produced audio.
    pub fn add(&mut self, data: &[u8]) -> Result<Option<AudioData>, String> {
        let packets = self.parser.add(data)?;
        let mut samples = Vec::new();

        for packet in packets {
            self.process_packet(packet, &mut samples)?;
        }

        self.audio_data_from_samples(samples)
    }

    pub fn finish(&mut self) -> Result<Option<AudioData>, String> {
        let packets = self.parser.finish()?;
        let mut samples = Vec::new();
        for packet in packets {
            self.process_packet(packet, &mut samples)?;
        }
        self.audio_data_from_samples(samples)
    }

    fn audio_data_from_samples(&self, samples: Vec<i16>) -> Result<Option<AudioData>, String> {
        if samples.is_empty() {
            return Ok(None);
        }
        let info = self
            .info
            .ok_or_else(|| "Vorbis stream info unavailable after decode".to_string())?;
        let mut pcm_bytes = Vec::with_capacity(samples.len() * 2);
        for sample in samples {
            pcm_bytes.extend_from_slice(&sample.to_le_bytes());
        }
        Ok(Some(AudioData::new(
            16,
            info.channels,
            info.sample_rate,
            pcm_bytes,
            EncodingFlag::PCMSigned,
            Endianness::LittleEndian,
        )))
    }

    fn process_packet(
        &mut self,
        packet: SharedOggPacket,
        samples: &mut Vec<i16>,
    ) -> Result<(), String> {
        let state = std::mem::replace(&mut self.state, VorbisDecodeState::HeaderIdent);
        match state {
            VorbisDecodeState::HeaderIdent => {
                if !packet.first_in_stream {
                    return Err(
                        "Expected Vorbis identification header as first Ogg packet".to_string()
                    );
                }
                let ident =
                    read_header_ident(&packet.data).map_err(|error| format!("{error:?}"))?;
                self.info = Some(VorbisStreamInfo {
                    sample_rate: ident.audio_sample_rate,
                    channels: ident.audio_channels,
                    serial: packet.serial,
                });
                self.state = VorbisDecodeState::HeaderComment { ident };
            }
            VorbisDecodeState::HeaderComment { ident } => {
                self.ensure_serial(packet.serial)?;
                read_header_comment(&packet.data).map_err(|error| format!("{error:?}"))?;
                self.state = VorbisDecodeState::HeaderSetup { ident };
            }
            VorbisDecodeState::HeaderSetup { ident } => {
                self.ensure_serial(packet.serial)?;
                let setup = read_header_setup(
                    &packet.data,
                    ident.audio_channels,
                    (ident.blocksize_0, ident.blocksize_1),
                )
                .map_err(|error| format!("{error:?}"))?;
                self.state = VorbisDecodeState::Audio {
                    ident,
                    setup,
                    pwr: PreviousWindowRight::new(),
                    cur_absgp: None,
                };
            }
            VorbisDecodeState::Audio {
                ident,
                setup,
                mut pwr,
                mut cur_absgp,
            } => {
                self.ensure_serial(packet.serial)?;
                let mut decoded: InterleavedSamples<i16> =
                    read_audio_packet_generic(&ident, &setup, &packet.data, &mut pwr)
                        .map_err(|error| format!("{error:?}"))?;

                if let (Some(previous_absgp), true, Some(page_absgp)) =
                    (cur_absgp, packet.last_in_stream, packet.granule_position)
                {
                    decoded.truncate(page_absgp.saturating_sub(previous_absgp) as usize);
                }

                if packet.last_in_page {
                    if let Some(page_absgp) = packet.granule_position {
                        cur_absgp = Some(page_absgp);
                    }
                } else if let Some(absgp) = cur_absgp.as_mut() {
                    *absgp += decoded.num_samples() as u64;
                }

                samples.extend(decoded.samples);
                self.state = VorbisDecodeState::Audio {
                    ident,
                    setup,
                    pwr,
                    cur_absgp,
                };
            }
        }
        Ok(())
    }

    fn ensure_serial(&self, serial: u32) -> Result<(), String> {
        let info = self
            .info
            .ok_or_else(|| "Vorbis stream info unavailable".to_string())?;
        if serial == info.serial {
            Ok(())
        } else {
            Err("Unexpected second Ogg logical bitstream".to_string())
        }
    }
}

impl Default for VorbisDecoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::process::Command;

    #[test]
    fn ogg_parser_bounds_garbage_and_input_chunks() {
        let mut parser = SharedOggPacketParser::new();
        assert!(parser.add(&vec![0x55; 65_536]).is_err());

        let mut decoder = VorbisDecoder::new();
        let error = decoder
            .add(&vec![0; MAX_OGG_INPUT_CHUNK_BYTES + 1])
            .unwrap_err();
        assert!(error.contains("streaming budget"));
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

    fn decode_chunks(data: &[u8], chunk_size: usize) -> Vec<u8> {
        let mut decoder = VorbisDecoder::new();
        let mut pcm = Vec::new();
        for chunk in data.chunks(chunk_size) {
            if let Some(audio) = decoder.add(chunk).unwrap() {
                pcm.extend_from_slice(audio.data());
            }
        }
        if let Some(audio) = decoder.finish().unwrap() {
            pcm.extend_from_slice(audio.data());
        }
        pcm
    }

    #[test]
    #[ignore = "regenerates the committed Ogg Vorbis fixture using ffmpeg/libvorbis"]
    fn generate_vorbis_fixture_with_ffmpeg() {
        let input = testdata_path("linear16_8/A_Tusk_is_used_to_make_costly_gifts.s16le");
        let output = testdata_path("vorbis/A_Tusk_is_used_to_make_costly_gifts.ogg");
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
            .args(["-c:a", "libvorbis", "-q:a", "4", "-f", "ogg"])
            .arg(&output)
            .status()
            .unwrap();
        assert!(status.success());
    }

    #[test]
    fn streaming_decoder_matches_whole_decode() {
        let fixture = fs::read(testdata_path(
            "vorbis/A_Tusk_is_used_to_make_costly_gifts.ogg",
        ))
        .unwrap();
        assert!(!fixture.is_empty(), "Vorbis fixture missing or empty");

        let whole = decode_chunks(&fixture, fixture.len());
        let chunked = decode_chunks(&fixture, 997);
        assert_eq!(chunked, whole);
        assert!(whole
            .chunks_exact(2)
            .any(|chunk| { i16::from_le_bytes([chunk[0], chunk[1]]) != 0 }));
    }

    #[test]
    fn decode_vorbis_fixture_and_write_golden_wav() {
        let fixture = fs::read(testdata_path(
            "vorbis/A_Tusk_is_used_to_make_costly_gifts.ogg",
        ))
        .unwrap();
        assert!(!fixture.is_empty(), "Vorbis fixture missing or empty");

        let mut decoder = VorbisDecoder::new();
        let mut decoded = Vec::new();
        for chunk in fixture.chunks(641) {
            if let Some(audio) = decoder.add(chunk).unwrap() {
                assert_eq!(audio.bits_per_sample(), 16);
                assert_eq!(audio.channel_count(), 1);
                assert_eq!(audio.sampling_rate(), 8_000);
                decoded.extend_from_slice(audio.data());
            }
        }
        if let Some(audio) = decoder.finish().unwrap() {
            decoded.extend_from_slice(audio.data());
        }

        assert!(decoded
            .chunks_exact(2)
            .any(|chunk| { i16::from_le_bytes([chunk[0], chunk[1]]) != 0 }));
        let samples: Vec<i16> = decoded
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
            .collect();

        let wav = soundkit::wav::generate_wav_buffer(
            &soundkit::audio_types::PcmData::I16(vec![samples]),
            8_000,
        )
        .unwrap();
        let output_path = golden_path("vorbis/A_Tusk_is_used_to_make_costly_gifts.decoded.wav");
        fs::create_dir_all(output_path.parent().unwrap()).unwrap();
        fs::write(output_path, wav).unwrap();
    }

    #[test]
    fn ffmpeg_can_decode_vorbis_fixture() {
        let input = testdata_path("vorbis/A_Tusk_is_used_to_make_costly_gifts.ogg");
        let output = std::env::temp_dir().join("soundkit-vorbis-fixture.s16le");
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
