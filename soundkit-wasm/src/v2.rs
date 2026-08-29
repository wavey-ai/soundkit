//! Decoding a SoundKit v2 stream, whatever is inside it.
//!
//! A v2 stream is frames: a header describing the audio, then one packet of
//! whatever codec that header names. It is not any codec's own file format —
//! there is no `fLaC` marker, no `OpusHead`, no ADTS — so handing the file to
//! a codec decoder gets you a complaint about a missing header. The frames
//! have to be read off first and each payload given to the right decoder.
//!
//! That dispatch is what this is. It reads the encoding from every frame, so
//! a stream that changes codec partway is decoded rather than refused: the
//! decoder for the old codec is dropped and one for the new codec takes over
//! at that frame.
//!
//! FLAC is the awkward one. Its decoder wants a `STREAMINFO` before any
//! frame, and a v2 stream has none — but it does not need to carry one: the
//! v2 header already states the sample rate, the channel count, the bit depth
//! and the block size, which is everything `STREAMINFO` requires that is not
//! allowed to be unknown. So one is derived from the first FLAC frame and
//! pushed in front of it.

use frame_header::{EncodingFlag, Endianness};
use soundkit::audio_types::AudioData;
use soundkit::frame_stream::{SoundKitFrameStream, SoundKitFrameStreamOptions};

#[cfg(feature = "flac")]
use soundkit::audio_packet::Decoder as PacketDecoder;
#[cfg(feature = "flac")]
use soundkit_flac::FlacDecoder;
#[cfg(feature = "opus")]
use soundkit_opus::Decoder as OpusPacketDecoder;

/// The audio a stream decoded to, in the order it arrived.
#[derive(Debug, Default)]
pub struct SoundKitV2Batch {
    pub frames: Vec<AudioData>,
}

impl SoundKitV2Batch {
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }
}

enum Codec {
    #[cfg(feature = "opus")]
    Opus(Box<OpusPacketDecoder>),
    #[cfg(feature = "flac")]
    Flac(Box<FlacDecoder>),
    Pcm,
}

/// Decodes a v2 stream to PCM, one frame at a time.
pub struct SoundKitV2Decoder {
    frames: SoundKitFrameStream,
    codec: Option<Codec>,
    /// What the live decoder was built for. A frame naming anything else
    /// replaces it.
    encoding: Option<EncodingFlag>,
    sample_rate: u32,
    channels: u8,
}

impl Default for SoundKitV2Decoder {
    fn default() -> Self {
        Self::new()
    }
}

impl SoundKitV2Decoder {
    pub fn new() -> Self {
        Self::with_options(SoundKitFrameStreamOptions::default())
    }

    pub fn with_options(options: SoundKitFrameStreamOptions) -> Self {
        Self {
            frames: SoundKitFrameStream::new(options),
            codec: None,
            encoding: None,
            sample_rate: 0,
            channels: 0,
        }
    }

    /// The rate and channel count the last frame declared.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn channels(&self) -> u8 {
        self.channels
    }

    pub fn buffered_bytes(&self) -> usize {
        self.frames.buffered_bytes()
    }

    pub fn reset(&mut self) {
        self.frames.reset();
        self.codec = None;
        self.encoding = None;
    }

    /// Feeds the next slice of the stream and takes whatever it completes.
    ///
    /// The stream may be cut anywhere: a frame split across two calls is
    /// held until the rest of it arrives.
    pub fn push(&mut self, bytes: &[u8]) -> Result<SoundKitV2Batch, String> {
        let read = self.frames.push(bytes)?;
        let mut batch = SoundKitV2Batch::default();
        for frame in read {
            let encoding = *frame.header.encoding();
            let rate = frame.header.sample_rate();
            let channels = frame.header.channels().max(1);
            let bits = frame.header.bits_per_sample();
            self.sample_rate = rate;
            self.channels = channels;
            // A change of codec mid-stream is a new decoder, not an error.
            if self.encoding != Some(encoding) {
                self.codec = Some(Self::build(encoding, rate, channels, bits, &frame)?);
                self.encoding = Some(encoding);
            }
            let pcm = self.decode(&frame.payload, encoding, channels, bits)?;
            if pcm.is_empty() {
                continue;
            }
            batch.frames.push(AudioData::new(
                16,
                channels,
                rate,
                pcm,
                EncodingFlag::PCMSigned,
                Endianness::LittleEndian,
            ));
        }
        Ok(batch)
    }

    fn build(
        encoding: EncodingFlag,
        rate: u32,
        channels: u8,
        _bits: u8,
        _frame: &soundkit::frame_stream::SoundKitFrame,
    ) -> Result<Codec, String> {
        match encoding {
            EncodingFlag::PCMSigned | EncodingFlag::PCMFloat => Ok(Codec::Pcm),
            #[cfg(feature = "opus")]
            EncodingFlag::Opus => {
                let decoder = OpusPacketDecoder::new(rate as i32, usize::from(channels))
                    .map_err(|error| error.to_string())?;
                Ok(Codec::Opus(Box::new(decoder)))
            }
            #[cfg(feature = "flac")]
            EncodingFlag::FLAC => {
                let mut decoder = FlacDecoder::new();
                decoder.init()?;
                // The decoder wants a STREAMINFO before any frame and the
                // stream carries none, so one is derived from this header
                // and pushed in front of the first packet.
                let header = derived_flac_stream_info(rate, channels, _bits);
                let mut sink = vec![0i16; 1 << 12];
                decoder
                    .decode_i16(&header, &mut sink, false)
                    .map_err(|error| {
                        format!("the derived FLAC STREAMINFO was refused: {error}")
                    })?;
                Ok(Codec::Flac(Box::new(decoder)))
            }
            other => Err(format!(
                "a SoundKit v2 stream carrying {other:?} cannot be decoded here yet"
            )),
        }
    }

    fn decode(
        &mut self,
        payload: &[u8],
        encoding: EncodingFlag,
        channels: u8,
        bits: u8,
    ) -> Result<Vec<u8>, String> {
        match (&mut self.codec, encoding) {
            (Some(Codec::Pcm), EncodingFlag::PCMSigned) => Ok(pcm_signed_to_i16_bytes(payload, bits)),
            (Some(Codec::Pcm), EncodingFlag::PCMFloat) => Ok(pcm_float_to_i16_bytes(payload)),
            #[cfg(feature = "opus")]
            (Some(Codec::Opus(decoder)), EncodingFlag::Opus) => {
                let decoded = decoder
                    .decode_i16_vec(payload, false)
                    .map_err(|error| error.to_string())?;
                Ok(i16s_to_le_bytes(&decoded))
            }
            #[cfg(feature = "flac")]
            (Some(Codec::Flac(decoder)), EncodingFlag::FLAC) => {
                let mut out = vec![0i16; 1 << 16];
                let written = decoder
                    .decode_i16(payload, &mut out, false)
                    .map_err(|error| error.to_string())?;
                Ok(i16s_to_le_bytes(&out[..written]))
            }
            _ => Err(format!(
                "no decoder is standing for {encoding:?} with {channels} channels"
            )),
        }
    }
}

fn i16s_to_le_bytes(samples: &[i16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for sample in samples {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    bytes
}

fn pcm_signed_to_i16_bytes(payload: &[u8], bits: u8) -> Vec<u8> {
    match bits {
        16 => payload.to_vec(),
        24 => {
            let mut out = Vec::with_capacity(payload.len() / 3 * 2);
            for chunk in payload.chunks_exact(3) {
                let value = i32::from(chunk[2] as i8) << 16
                    | i32::from(chunk[1]) << 8
                    | i32::from(chunk[0]);
                out.extend_from_slice(&((value >> 8) as i16).to_le_bytes());
            }
            out
        }
        32 => {
            let mut out = Vec::with_capacity(payload.len() / 2);
            for chunk in payload.chunks_exact(4) {
                let value = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                out.extend_from_slice(&((value >> 16) as i16).to_le_bytes());
            }
            out
        }
        _ => payload.to_vec(),
    }
}

fn pcm_float_to_i16_bytes(payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(payload.len() / 2);
    for chunk in payload.chunks_exact(4) {
        let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let clamped = (value.clamp(-1.0, 1.0) * f32::from(i16::MAX)) as i16;
        out.extend_from_slice(&clamped.to_le_bytes());
    }
    out
}

/// A `STREAMINFO` derived from a v2 frame header.
///
/// The header states the rate, the channels, the depth and the block size.
/// Everything else `STREAMINFO` holds is allowed to be unknown: the frame
/// sizes and the total sample count are zero for "not stated", the MD5 is
/// zero for "not computed", and the block-size bounds are left wide so a
/// stream whose last frame is short is still described truthfully.
#[cfg(feature = "flac")]
pub fn derived_flac_stream_info(sample_rate: u32, channels: u8, bits_per_sample: u8) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + 4 + 34);
    out.extend_from_slice(b"fLaC");
    // Metadata block header: last block, type 0 (STREAMINFO), length 34.
    out.push(0x80);
    out.extend_from_slice(&[0, 0, 34]);

    let mut info = [0u8; 34];
    // Minimum and maximum block size: the legal extremes, meaning "varies".
    info[0..2].copy_from_slice(&16u16.to_be_bytes());
    info[2..4].copy_from_slice(&65_535u16.to_be_bytes());
    // Minimum and maximum frame size stay zero: not stated.
    let rate = sample_rate & 0x000F_FFFF;
    let channels = u32::from(channels.max(1) - 1) & 0x7;
    let depth = u32::from(bits_per_sample.max(4) - 1) & 0x1F;
    // 20 bits of rate, 3 of channels, 5 of depth, then 36 of total samples.
    let packed = (u64::from(rate) << 44) | (u64::from(channels) << 41) | (u64::from(depth) << 36);
    info[10..18].copy_from_slice(&packed.to_be_bytes());
    out.extend_from_slice(&info);
    out
}


#[cfg(test)]
mod tests {
    use super::*;
    use frame_header::FrameHeaderV2;

    /// One v2 frame: the header the stream states, then the packet.
    fn frame(
        encoding: EncodingFlag,
        payload: &[u8],
        frame_count: u32,
        sample_rate: u32,
        channels: u8,
        bits_per_sample: u8,
    ) -> Vec<u8> {
        let header = FrameHeaderV2::new(
            encoding,
            payload.len() as u32,
            frame_count,
            sample_rate,
            channels,
            bits_per_sample,
            Endianness::LittleEndian,
            None,
            None,
            None,
        )
        .expect("a v2 header is describable");
        let mut out = Vec::new();
        header.encode(&mut out).expect("v2 header encodes");
        out.extend_from_slice(payload);
        out
    }

    fn i16_bytes(samples: &[i16]) -> Vec<u8> {
        let mut out = Vec::new();
        for sample in samples {
            out.extend_from_slice(&sample.to_le_bytes());
        }
        out
    }

    #[test]
    fn decodes_pcm_signed_frames() {
        let samples: Vec<i16> = (0..960).map(|n| (n as i16).wrapping_mul(17)).collect();
        let stream = frame(
            EncodingFlag::PCMSigned,
            &i16_bytes(&samples),
            480,
            48_000,
            2,
            16,
        );
        let mut decoder = SoundKitV2Decoder::new();
        let batch = decoder.push(&stream).expect("a PCM v2 frame decodes");
        assert_eq!(batch.frames.len(), 1);
        assert_eq!(decoder.sample_rate(), 48_000);
        assert_eq!(decoder.channels(), 2);
    }

    #[test]
    fn decodes_pcm_float_frames() {
        let mut payload = Vec::new();
        for n in 0..480 {
            let value = (n as f32 / 480.0) * 2.0 - 1.0;
            payload.extend_from_slice(&value.to_le_bytes());
        }
        let stream = frame(EncodingFlag::PCMFloat, &payload, 240, 48_000, 2, 32);
        let mut decoder = SoundKitV2Decoder::new();
        let batch = decoder.push(&stream).expect("a float v2 frame decodes");
        assert_eq!(batch.frames.len(), 1);
    }

    /// The point of reading the encoding off every frame: a stream that
    /// changes codec partway is decoded, not refused.
    #[test]
    fn switches_codec_midstream() {
        let signed: Vec<i16> = (0..480).map(|n| n as i16).collect();
        let mut floats = Vec::new();
        for n in 0..480 {
            floats.extend_from_slice(&((n as f32 / 480.0) - 0.5).to_le_bytes());
        }
        let mut stream = frame(
            EncodingFlag::PCMSigned,
            &i16_bytes(&signed),
            240,
            48_000,
            2,
            16,
        );
        stream.extend_from_slice(&frame(
            EncodingFlag::PCMFloat,
            &floats,
            240,
            48_000,
            2,
            32,
        ));
        let mut decoder = SoundKitV2Decoder::new();
        let batch = decoder.push(&stream).expect("a mixed v2 stream decodes");
        assert_eq!(
            batch.frames.len(),
            2,
            "both frames decode across the codec change"
        );
    }

    /// A stream arrives in whatever slices the transport gives it, and a
    /// frame cut in half must still decode once the rest lands.
    #[test]
    fn decodes_across_split_pushes() {
        let samples: Vec<i16> = (0..480).map(|n| n as i16).collect();
        let stream = frame(
            EncodingFlag::PCMSigned,
            &i16_bytes(&samples),
            240,
            48_000,
            2,
            16,
        );
        let cut = stream.len() / 3;
        let mut decoder = SoundKitV2Decoder::new();
        let first = decoder.push(&stream[..cut]).expect("a part frame is held");
        assert!(first.is_empty(), "an incomplete frame yields nothing yet");
        let rest = decoder.push(&stream[cut..]).expect("the rest completes it");
        assert_eq!(rest.frames.len(), 1);
    }

    #[cfg(feature = "flac")]
    #[test]
    fn derived_stream_info_states_the_format() {
        let info = derived_flac_stream_info(44_100, 2, 16);
        assert_eq!(&info[..4], b"fLaC");
        assert_eq!(info[4], 0x80, "STREAMINFO is the last metadata block");
        assert_eq!(&info[5..8], &[0, 0, 34], "STREAMINFO is 34 bytes");
        let packed = u64::from_be_bytes(info[18..26].try_into().unwrap());
        assert_eq!((packed >> 44) as u32, 44_100, "the rate is stated");
        assert_eq!(((packed >> 41) & 0x7) + 1, 2, "the channel count is stated");
        assert_eq!(((packed >> 36) & 0x1F) + 1, 16, "the depth is stated");
    }

    /// A real Opus packet in a v2 frame: encoded by SoundKit's own encoder,
    /// framed, then read back the way a stored track is.
    #[cfg(feature = "opus")]
    #[test]
    fn decodes_a_real_opus_packet() {
        let frame_size = 960usize;
        let channels = 2usize;
        let pcm: Vec<i16> = (0..frame_size * channels)
            .map(|n| ((n as f32 / 40.0).sin() * 8_000.0) as i16)
            .collect();
        let mut encoder = soundkit_opus::Encoder::new(
            48_000,
            16,
            channels as u32,
            frame_size as u32,
            96_000,
        );
        encoder.init().expect("the Opus encoder starts");
        let packet = encoder
            .encode_i16_vec(&pcm, frame_size)
            .expect("a packet is produced");
        let written = packet.len();
        assert!(written > 0, "the encoder produced a packet");

        let stream = frame(
            EncodingFlag::Opus,
            &packet[..written],
            frame_size as u32,
            48_000,
            channels as u8,
            16,
        );
        let mut decoder = SoundKitV2Decoder::new();
        let batch = decoder.push(&stream).expect("an Opus v2 frame decodes");
        assert_eq!(batch.frames.len(), 1, "the packet came back as audio");
        assert_eq!(decoder.channels(), 2);
        assert_eq!(decoder.sample_rate(), 48_000);
    }

    /// The same for FLAC, which is the one that needs a STREAMINFO invented
    /// for it before its first frame will decode at all.
    #[cfg(feature = "flac")]
    #[test]
    fn decodes_a_real_flac_packet() {
        use soundkit::audio_packet::Encoder as PacketEncoder;
        use soundkit_flac::FlacEncoder;

        let frame_size = 1_024usize;
        let channels = 2usize;
        let pcm: Vec<i32> = (0..frame_size * channels)
            .map(|n| ((n as f32 / 30.0).sin() * 6_000.0) as i32)
            .collect();
        let mut encoder = <FlacEncoder as PacketEncoder>::new(
            48_000,
            16,
            channels as u32,
            frame_size as u32,
            3,
        );
        encoder.init().expect("the FLAC encoder starts");
        let mut packet = vec![0u8; 1 << 16];
        let written = encoder
            .encode_i32(&pcm, &mut packet)
            .expect("a FLAC packet is produced");
        assert!(written > 0, "the encoder produced a packet");

        let stream = frame(
            EncodingFlag::FLAC,
            &packet[..written],
            frame_size as u32,
            48_000,
            channels as u8,
            16,
        );
        let mut decoder = SoundKitV2Decoder::new();
        let batch = decoder
            .push(&stream)
            .expect("a FLAC v2 frame decodes against a derived STREAMINFO");
        assert_eq!(batch.frames.len(), 1, "the packet came back as audio");
    }
}
