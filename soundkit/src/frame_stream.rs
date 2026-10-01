use crate::crypto::ChaCha20Poly1305PacketCipher;
use frame_header::FrameHeaderV2;

const DEFAULT_MAX_BUFFERED_BYTES: usize = 4 * 1024 * 1024;
const DEFAULT_MAX_PAYLOAD_BYTES: usize = 1024 * 1024;

#[derive(Debug, Clone)]
pub struct SoundKitFrame {
    pub header: FrameHeaderV2,
    pub payload: Vec<u8>,
    pub encrypted: bool,
    pub encoded_header_bytes: Vec<u8>,
    pub encrypted_payload_size: usize,
}

#[derive(Clone)]
pub struct SoundKitFrameStreamOptions {
    pub max_buffered_bytes: usize,
    pub max_payload_bytes: usize,
    pub verify_packet_crc32: bool,
    pub cipher: Option<ChaCha20Poly1305PacketCipher>,
}

impl Default for SoundKitFrameStreamOptions {
    fn default() -> Self {
        Self {
            max_buffered_bytes: DEFAULT_MAX_BUFFERED_BYTES,
            max_payload_bytes: DEFAULT_MAX_PAYLOAD_BYTES,
            verify_packet_crc32: true,
            cipher: None,
        }
    }
}

pub struct SoundKitFrameStream {
    buffer: Vec<u8>,
    options: SoundKitFrameStreamOptions,
}

impl Default for SoundKitFrameStream {
    fn default() -> Self {
        Self::new(SoundKitFrameStreamOptions::default())
    }
}

impl SoundKitFrameStream {
    pub fn new(options: SoundKitFrameStreamOptions) -> Self {
        Self {
            buffer: Vec::new(),
            options,
        }
    }

    pub fn set_cipher(&mut self, cipher: Option<ChaCha20Poly1305PacketCipher>) {
        self.options.cipher = cipher;
    }

    pub fn reset(&mut self) {
        self.buffer.clear();
    }

    pub fn buffered_bytes(&self) -> usize {
        self.buffer.len()
    }

    pub fn push(&mut self, chunk: &[u8]) -> Result<Vec<SoundKitFrame>, String> {
        if !chunk.is_empty() {
            self.buffer.extend_from_slice(chunk);
        }
        if self.buffer.len() > self.options.max_buffered_bytes {
            return Err(format!(
                "SoundKit frame buffer exceeded {} bytes",
                self.options.max_buffered_bytes
            ));
        }

        // Frames are read at an offset and the consumed prefix is removed
        // once per push. Removing each frame from the front of the buffer
        // moves the rest of the buffer for every frame.
        let mut frames = Vec::new();
        let mut consumed = 0;
        let result = self.read_frames(&mut frames, &mut consumed);
        self.buffer.drain(..consumed);
        result.map(|()| frames)
    }

    fn read_frames(
        &self,
        frames: &mut Vec<SoundKitFrame>,
        consumed: &mut usize,
    ) -> Result<(), String> {
        loop {
            let available = &self.buffer[*consumed..];
            if available.len() < FrameHeaderV2::BASE_SIZE {
                return Ok(());
            }

            let header_size = FrameHeaderV2::header_size(available)?;
            if available.len() < header_size {
                return Ok(());
            }

            let encoded_header = available[..header_size].to_vec();
            let header = FrameHeaderV2::decode(&mut &encoded_header[..])
                .map_err(|error| format!("decode SoundKit v2 header failed: {error}"))?;
            let payload_size = header.payload_size() as usize;
            if payload_size > self.options.max_payload_bytes {
                return Err(format!(
                    "SoundKit frame payload exceeded {} bytes",
                    self.options.max_payload_bytes
                ));
            }

            let frame_size = header_size
                .checked_add(payload_size)
                .ok_or_else(|| "SoundKit frame size overflow".to_string())?;
            if available.len() < frame_size {
                return Ok(());
            }

            let mut payload = available[header_size..frame_size].to_vec();
            if self.options.verify_packet_crc32
                && header.packet_crc32_value().is_some()
                && !verify_packet_crc32(&header, &encoded_header, &payload)?
            {
                return Err("SoundKit frame CRC32 mismatch".to_string());
            }

            let encrypted = header.is_encrypted();
            if encrypted {
                let cipher = self.options.cipher.as_ref().ok_or_else(|| {
                    "SoundKit frame is encrypted but no cipher is configured".to_string()
                })?;
                payload = cipher
                    .decrypt_nonce_prefixed(&payload, &[])
                    .map_err(|error| error.to_string())?;
            }

            frames.push(SoundKitFrame {
                header,
                payload,
                encrypted,
                encoded_header_bytes: encoded_header,
                encrypted_payload_size: payload_size,
            });

            *consumed += frame_size;
        }
    }

    pub fn finish(&self) -> Result<(), String> {
        if self.buffer.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "SoundKit frame stream ended with {} buffered bytes",
                self.buffer.len()
            ))
        }
    }
}

/// The packet CRC-32 check of `FrameHeaderV2::verify_packet_crc32`, with the
/// same results and errors. `crc32fast` computes the same IEEE CRC-32 with the
/// processor's CRC instructions where they exist.
pub fn verify_packet_crc32(
    header: &FrameHeaderV2,
    encoded_header: &[u8],
    payload: &[u8],
) -> Result<bool, String> {
    let Some(expected) = header.packet_crc32_value() else {
        return Ok(false);
    };
    let header_size = header.size();
    if encoded_header.len() < header_size {
        return Err("Encoded header too small".to_string());
    }
    Ok(packet_crc32(&encoded_header[..header_size - 4], payload) == expected)
}

/// The IEEE CRC-32 of `header_without_crc` followed by `payload`: the value
/// of `frame_header::packet_crc32`.
pub fn packet_crc32(header_without_crc: &[u8], payload: &[u8]) -> u32 {
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(header_without_crc);
    hasher.update(payload);
    hasher.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::{
        chacha20_poly1305_key_from_decimal, ChaCha20Poly1305PacketCipher,
        CHACHA20_POLY1305_NONCE_BYTES,
    };
    use frame_header::{EncodingFlag, Endianness};

    const TEST_KEY_DECIMAL: &str =
        "83843157117408337365446905028299378179116700186920144823595584430653437972238";

    fn encode_frame(payload: &[u8], encrypted: bool) -> Vec<u8> {
        let packet_flags = if encrypted {
            FrameHeaderV2::FLAG_ENCRYPTED
        } else {
            0
        };
        let header = FrameHeaderV2::new(
            EncodingFlag::Opus,
            payload.len() as u32,
            960,
            48000,
            2,
            0,
            Endianness::LittleEndian,
            Some(5),
            Some(20_000),
            None,
        )
        .unwrap()
        .with_packet_flags(packet_flags)
        .unwrap()
        .with_packet_crc32(payload)
        .unwrap();

        let mut output = Vec::with_capacity(header.size() + payload.len());
        header.encode(&mut output).unwrap();
        output.extend_from_slice(payload);
        output
    }

    #[test]
    fn parses_plain_v2_frames() {
        let packet = encode_frame(b"opus", false);
        let mut stream = SoundKitFrameStream::default();

        let frames = stream.push(&packet).unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].payload, b"opus");
        assert!(!frames[0].encrypted);
        assert_eq!(frames[0].header.id(), Some(5));
        stream.finish().unwrap();
    }

    #[test]
    fn decrypts_encrypted_v2_frames_when_flagged() {
        let key = chacha20_poly1305_key_from_decimal(TEST_KEY_DECIMAL).unwrap();
        let cipher = ChaCha20Poly1305PacketCipher::new(&key).unwrap();
        let nonce = [3u8; CHACHA20_POLY1305_NONCE_BYTES];
        let encrypted_payload = cipher.encrypt_nonce_prefixed(&nonce, b"opus", &[]).unwrap();
        let packet = encode_frame(&encrypted_payload, true);
        let mut stream = SoundKitFrameStream::new(SoundKitFrameStreamOptions {
            cipher: Some(cipher),
            ..SoundKitFrameStreamOptions::default()
        });

        let frames = stream.push(&packet).unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].payload, b"opus");
        assert!(frames[0].encrypted);
        assert_eq!(frames[0].encrypted_payload_size, encrypted_payload.len());
    }

    /// The frame stream before frames were read at an offset: each frame is
    /// removed from the front of the buffer, and `frame_header` checks the
    /// packet CRC-32.
    struct ReferenceStream {
        buffer: Vec<u8>,
        options: SoundKitFrameStreamOptions,
    }

    impl ReferenceStream {
        fn push(&mut self, chunk: &[u8]) -> Result<Vec<SoundKitFrame>, String> {
            if !chunk.is_empty() {
                self.buffer.extend_from_slice(chunk);
            }
            if self.buffer.len() > self.options.max_buffered_bytes {
                return Err(format!(
                    "SoundKit frame buffer exceeded {} bytes",
                    self.options.max_buffered_bytes
                ));
            }
            let mut frames = Vec::new();
            loop {
                if self.buffer.len() < FrameHeaderV2::BASE_SIZE {
                    break;
                }
                let header_size = FrameHeaderV2::header_size(&self.buffer)?;
                if self.buffer.len() < header_size {
                    break;
                }
                let encoded_header = self.buffer[..header_size].to_vec();
                let header = FrameHeaderV2::decode(&mut &encoded_header[..])
                    .map_err(|error| format!("decode SoundKit v2 header failed: {error}"))?;
                let payload_size = header.payload_size() as usize;
                if payload_size > self.options.max_payload_bytes {
                    return Err(format!(
                        "SoundKit frame payload exceeded {} bytes",
                        self.options.max_payload_bytes
                    ));
                }
                let frame_size = header_size
                    .checked_add(payload_size)
                    .ok_or_else(|| "SoundKit frame size overflow".to_string())?;
                if self.buffer.len() < frame_size {
                    break;
                }
                let mut payload = self.buffer[header_size..frame_size].to_vec();
                if self.options.verify_packet_crc32
                    && header.packet_crc32_value().is_some()
                    && !header.verify_packet_crc32(&encoded_header, &payload)?
                {
                    return Err("SoundKit frame CRC32 mismatch".to_string());
                }
                let encrypted = header.is_encrypted();
                if encrypted {
                    let cipher = self.options.cipher.as_ref().ok_or_else(|| {
                        "SoundKit frame is encrypted but no cipher is configured".to_string()
                    })?;
                    payload = cipher
                        .decrypt_nonce_prefixed(&payload, &[])
                        .map_err(|error| error.to_string())?;
                }
                frames.push(SoundKitFrame {
                    header,
                    payload,
                    encrypted,
                    encoded_header_bytes: encoded_header,
                    encrypted_payload_size: payload_size,
                });
                self.buffer.drain(..frame_size);
            }
            Ok(frames)
        }
    }

    type FrameView = (FrameHeaderV2, Vec<u8>, bool, Vec<u8>, usize);

    fn view(result: Result<Vec<SoundKitFrame>, String>) -> Result<Vec<FrameView>, String> {
        result.map(|frames| {
            frames
                .into_iter()
                .map(|frame| {
                    (
                        frame.header,
                        frame.payload,
                        frame.encrypted,
                        frame.encoded_header_bytes,
                        frame.encrypted_payload_size,
                    )
                })
                .collect()
        })
    }

    struct Xorshift(u64);

    impl Xorshift {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }

        fn below(&mut self, bound: usize) -> usize {
            (self.next() % bound as u64) as usize
        }
    }

    /// Frames of every header shape the stream reads: with and without an
    /// ID, a PTS and a packet CRC-32, plain and encrypted.
    fn mixed_stream(rng: &mut Xorshift, cipher: &ChaCha20Poly1305PacketCipher) -> Vec<u8> {
        let mut out = Vec::new();
        for index in 0..48u64 {
            let length = match index % 5 {
                0 => 0,
                1 => 1,
                _ => rng.below(700),
            };
            let mut payload: Vec<u8> = (0..length).map(|_| rng.next() as u8).collect();
            let encrypted = index % 7 == 3;
            if encrypted {
                let nonce = [index as u8; CHACHA20_POLY1305_NONCE_BYTES];
                payload = cipher
                    .encrypt_nonce_prefixed(&nonce, &payload, &[])
                    .unwrap();
            }
            let id = match index % 3 {
                0 => None,
                1 => Some(index),
                _ => Some(u64::from(u32::MAX) + index),
            };
            let pts = (index % 2 == 0).then_some(index * 240);
            let mut header = FrameHeaderV2::new(
                if index % 4 == 0 {
                    EncodingFlag::FLAC
                } else {
                    EncodingFlag::Opus
                },
                payload.len() as u32,
                240,
                48_000,
                2,
                24,
                Endianness::LittleEndian,
                id,
                pts,
                None,
            )
            .unwrap();
            if encrypted {
                header = header
                    .with_packet_flags(FrameHeaderV2::FLAG_ENCRYPTED)
                    .unwrap();
            }
            if index % 6 != 5 {
                header = header.with_packet_crc32(&payload).unwrap();
            }
            header.encode(&mut out).unwrap();
            out.extend_from_slice(&payload);
        }
        out
    }

    fn compare(
        stream: &[u8],
        options: SoundKitFrameStreamOptions,
        chunking: &mut dyn FnMut() -> usize,
    ) -> usize {
        let mut actual = SoundKitFrameStream::new(options.clone());
        let mut reference = ReferenceStream {
            buffer: Vec::new(),
            options,
        };
        let mut rest = stream;
        let mut errors = 0;
        while !rest.is_empty() {
            let take = chunking().clamp(1, rest.len());
            let (chunk, tail) = rest.split_at(take);
            rest = tail;
            let got = view(actual.push(chunk));
            let want = view(reference.push(chunk));
            errors += usize::from(want.is_err());
            assert_eq!(got, want);
            assert_eq!(actual.buffered_bytes(), reference.buffer.len());
        }
        assert_eq!(view(actual.push(&[])), view(reference.push(&[])));
        assert_eq!(actual.buffered_bytes(), reference.buffer.len());
        errors
    }

    /// The offset reader returns the same frames, errors and buffered byte
    /// counts as the front-removing reader, for intact, corrupt and truncated
    /// streams cut at many boundaries, and after an error.
    #[test]
    fn offset_reader_matches_front_removing_reader() {
        let key = chacha20_poly1305_key_from_decimal(TEST_KEY_DECIMAL).unwrap();
        let cipher = ChaCha20Poly1305PacketCipher::new(&key).unwrap();
        let mut rng = Xorshift(0x9e37_79b9_7f4a_7c15);
        let intact = mixed_stream(&mut rng, &cipher);
        let mut streams = vec![intact.clone(), intact[..intact.len() - 1].to_vec()];
        for _ in 0..12 {
            let mut corrupt = intact.clone();
            for _ in 0..1 + rng.below(3) {
                let at = rng.below(corrupt.len());
                corrupt[at] ^= 1 << rng.below(8);
            }
            streams.push(corrupt);
            let cut = rng.below(intact.len());
            streams.push(intact[..cut].to_vec());
        }
        let options = [
            SoundKitFrameStreamOptions {
                cipher: Some(cipher.clone()),
                ..SoundKitFrameStreamOptions::default()
            },
            SoundKitFrameStreamOptions::default(),
            SoundKitFrameStreamOptions {
                cipher: Some(cipher.clone()),
                verify_packet_crc32: false,
                ..SoundKitFrameStreamOptions::default()
            },
            SoundKitFrameStreamOptions {
                cipher: Some(cipher.clone()),
                max_buffered_bytes: 4_000,
                max_payload_bytes: 600,
                ..SoundKitFrameStreamOptions::default()
            },
        ];
        let mut errors = 0;
        for stream in &streams {
            for option in &options {
                for size in [1usize, 7, 53, 4_096, 256 * 1024, stream.len().max(1)] {
                    errors += compare(stream, option.clone(), &mut || size);
                }
                let mut sizes = Xorshift(stream.len() as u64 | 1);
                errors += compare(stream, option.clone(), &mut || 1 + sizes.below(3_000));
            }
        }
        // The corrupt streams and the small limits reach the error paths.
        assert!(errors > 100, "{errors} errors");
    }

    #[test]
    fn packet_crc32_matches_frame_header() {
        let mut rng = Xorshift(0x2545_f491_4f6c_dd1d);
        let data: Vec<u8> = (0..9_000).map(|_| rng.next() as u8).collect();
        for split in [0usize, 1, 3, 4, 15, 16, 17, 31, 64, 100] {
            for length in [0usize, 1, 7, 8, 9, 63, 64, 65, 255, 4_095, 8_000] {
                let header = &data[..split];
                let payload = &data[split..split + length];
                assert_eq!(
                    packet_crc32(header, payload),
                    frame_header::packet_crc32(header, payload)
                );
            }
        }
    }
}
