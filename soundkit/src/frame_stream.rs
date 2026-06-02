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
}
