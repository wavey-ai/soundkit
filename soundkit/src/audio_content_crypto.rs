use crate::crypto::{
    ChaCha20Poly1305PacketCipher, CryptoError, CHACHA20_POLY1305_KEY_BYTES,
    CHACHA20_POLY1305_PACKET_OVERHEAD_BYTES,
};
use std::fmt;

pub const AUDIO_CONTENT_ENVELOPE_MAGIC: [u8; 4] = *b"ACE1";
pub const AUDIO_CONTENT_ENVELOPE_HEADER_BYTES: usize = 8;
pub const AUDIO_CONTENT_ENVELOPE_OVERHEAD_BYTES: usize =
    AUDIO_CONTENT_ENVELOPE_HEADER_BYTES + CHACHA20_POLY1305_PACKET_OVERHEAD_BYTES;
pub const MAX_AUDIO_CONTENT_PLAINTEXT_BYTES: usize = 16 * 1024 * 1024;

const AUDIO_CONTENT_AAD_DOMAIN: &[u8] = b"infidelity.audio-content.v1\0";
const AUDIO_GROUP_AAD_MAGIC: &[u8; 4] = b"AEG1";
pub const MAX_AUDIO_CONTENT_SESSION_CONTEXT_BYTES: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioGroupMetadata<'a> {
    pub session_context: &'a [u8],
    pub transport_session_id: u64,
    pub config_generation: u32,
    pub epoch_id: u64,
    pub pts_samples: u64,
    pub sample_rate: u32,
    pub frame_count: u32,
    pub group_count: u16,
    pub group_id: u16,
    pub group_index: u16,
    pub channel_start: u16,
    pub channel_count: u16,
    pub payload_kind: u8,
    pub sample_format: u8,
    pub flags: u8,
}

impl AudioGroupMetadata<'_> {
    pub fn associated_data(&self) -> Result<Vec<u8>, AudioContentCryptoError> {
        if self.session_context.is_empty()
            || self.session_context.len() > MAX_AUDIO_CONTENT_SESSION_CONTEXT_BYTES
        {
            return Err(AudioContentCryptoError::InvalidSessionContext {
                maximum: MAX_AUDIO_CONTENT_SESSION_CONTEXT_BYTES,
                actual: self.session_context.len(),
            });
        }
        let context_length = u8::try_from(self.session_context.len()).map_err(|_| {
            AudioContentCryptoError::InvalidSessionContext {
                maximum: MAX_AUDIO_CONTENT_SESSION_CONTEXT_BYTES,
                actual: self.session_context.len(),
            }
        })?;
        let mut aad = Vec::with_capacity(59 + self.session_context.len());
        aad.extend_from_slice(AUDIO_GROUP_AAD_MAGIC);
        aad.push(context_length);
        aad.extend_from_slice(self.session_context);
        aad.extend_from_slice(&self.transport_session_id.to_le_bytes());
        aad.extend_from_slice(&self.config_generation.to_le_bytes());
        aad.extend_from_slice(&self.epoch_id.to_le_bytes());
        aad.extend_from_slice(&self.pts_samples.to_le_bytes());
        aad.extend_from_slice(&self.sample_rate.to_le_bytes());
        aad.extend_from_slice(&self.frame_count.to_le_bytes());
        aad.extend_from_slice(&self.group_count.to_le_bytes());
        aad.extend_from_slice(&self.group_id.to_le_bytes());
        aad.extend_from_slice(&self.group_index.to_le_bytes());
        aad.extend_from_slice(&self.channel_start.to_le_bytes());
        aad.extend_from_slice(&self.channel_count.to_le_bytes());
        aad.push(self.payload_kind);
        aad.push(self.sample_format);
        aad.push(self.flags);
        Ok(aad)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AudioContentCryptoError {
    InvalidKeyLength { expected: usize, actual: usize },
    ZeroKey,
    InvalidKeyEpoch,
    InvalidSessionContext { maximum: usize, actual: usize },
    PlaintextTooLarge { maximum: usize, actual: usize },
    EnvelopeTooShort { minimum: usize, actual: usize },
    EnvelopeTooLarge { maximum: usize, actual: usize },
    InvalidEnvelopeMagic,
    KeyEpochMismatch { expected: u32, actual: u32 },
    Cipher(CryptoError),
}

impl fmt::Display for AudioContentCryptoError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidKeyLength { expected, actual } => {
                write!(
                    formatter,
                    "invalid content key length {actual}; expected {expected} bytes"
                )
            }
            Self::ZeroKey => write!(formatter, "the content key must not be all zero"),
            Self::InvalidKeyEpoch => write!(formatter, "the content key epoch must be positive"),
            Self::InvalidSessionContext { maximum, actual } => write!(
                formatter,
                "audio session context is {actual} bytes; expected 1 to {maximum} bytes"
            ),
            Self::PlaintextTooLarge { maximum, actual } => write!(
                formatter,
                "audio content is {actual} bytes; the maximum is {maximum} bytes"
            ),
            Self::EnvelopeTooShort { minimum, actual } => write!(
                formatter,
                "audio content envelope is {actual} bytes; the minimum is {minimum} bytes"
            ),
            Self::EnvelopeTooLarge { maximum, actual } => write!(
                formatter,
                "audio content envelope is {actual} bytes; the maximum is {maximum} bytes"
            ),
            Self::InvalidEnvelopeMagic => write!(formatter, "invalid audio content envelope"),
            Self::KeyEpochMismatch { expected, actual } => write!(
                formatter,
                "audio content key epoch {actual} does not match expected epoch {expected}"
            ),
            Self::Cipher(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for AudioContentCryptoError {}

impl From<CryptoError> for AudioContentCryptoError {
    fn from(value: CryptoError) -> Self {
        Self::Cipher(value)
    }
}

/// Encrypts application audio before a transport or relay receives it.
///
/// The envelope contains a version marker, a key epoch, a 96-bit nonce, the
/// ciphertext, and a 128-bit authentication tag. The caller supplies stable
/// protocol metadata as additional authenticated data (AAD).
#[derive(Clone)]
pub struct AudioContentCipher {
    cipher: ChaCha20Poly1305PacketCipher,
}

impl fmt::Debug for AudioContentCipher {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AudioContentCipher")
            .finish_non_exhaustive()
    }
}

impl AudioContentCipher {
    pub fn new(key: &[u8]) -> Result<Self, AudioContentCryptoError> {
        if key.len() != CHACHA20_POLY1305_KEY_BYTES {
            return Err(AudioContentCryptoError::InvalidKeyLength {
                expected: CHACHA20_POLY1305_KEY_BYTES,
                actual: key.len(),
            });
        }
        if key.iter().all(|byte| *byte == 0) {
            return Err(AudioContentCryptoError::ZeroKey);
        }
        Ok(Self {
            cipher: ChaCha20Poly1305PacketCipher::new(key)?,
        })
    }

    pub fn seal(
        &self,
        key_epoch: u32,
        nonce: &[u8],
        plaintext: &[u8],
        aad: &[u8],
    ) -> Result<Vec<u8>, AudioContentCryptoError> {
        validate_key_epoch(key_epoch)?;
        if plaintext.len() > MAX_AUDIO_CONTENT_PLAINTEXT_BYTES {
            return Err(AudioContentCryptoError::PlaintextTooLarge {
                maximum: MAX_AUDIO_CONTENT_PLAINTEXT_BYTES,
                actual: plaintext.len(),
            });
        }
        let authenticated = authenticated_data(key_epoch, aad);
        let encrypted = self
            .cipher
            .encrypt_nonce_prefixed(nonce, plaintext, &authenticated)?;
        let mut envelope =
            Vec::with_capacity(AUDIO_CONTENT_ENVELOPE_HEADER_BYTES + encrypted.len());
        envelope.extend_from_slice(&AUDIO_CONTENT_ENVELOPE_MAGIC);
        envelope.extend_from_slice(&key_epoch.to_le_bytes());
        envelope.extend_from_slice(&encrypted);
        Ok(envelope)
    }

    pub fn open(
        &self,
        expected_key_epoch: u32,
        envelope: &[u8],
        aad: &[u8],
    ) -> Result<Vec<u8>, AudioContentCryptoError> {
        validate_key_epoch(expected_key_epoch)?;
        if envelope.len() < AUDIO_CONTENT_ENVELOPE_OVERHEAD_BYTES {
            return Err(AudioContentCryptoError::EnvelopeTooShort {
                minimum: AUDIO_CONTENT_ENVELOPE_OVERHEAD_BYTES,
                actual: envelope.len(),
            });
        }
        let maximum = MAX_AUDIO_CONTENT_PLAINTEXT_BYTES + AUDIO_CONTENT_ENVELOPE_OVERHEAD_BYTES;
        if envelope.len() > maximum {
            return Err(AudioContentCryptoError::EnvelopeTooLarge {
                maximum,
                actual: envelope.len(),
            });
        }
        if envelope[..4] != AUDIO_CONTENT_ENVELOPE_MAGIC {
            return Err(AudioContentCryptoError::InvalidEnvelopeMagic);
        }
        let actual_key_epoch =
            u32::from_le_bytes(envelope[4..8].try_into().expect("header checked"));
        if actual_key_epoch != expected_key_epoch {
            return Err(AudioContentCryptoError::KeyEpochMismatch {
                expected: expected_key_epoch,
                actual: actual_key_epoch,
            });
        }
        let authenticated = authenticated_data(actual_key_epoch, aad);
        self.cipher
            .decrypt_nonce_prefixed(
                &envelope[AUDIO_CONTENT_ENVELOPE_HEADER_BYTES..],
                &authenticated,
            )
            .map_err(Into::into)
    }
}

fn validate_key_epoch(key_epoch: u32) -> Result<(), AudioContentCryptoError> {
    if key_epoch == 0 {
        Err(AudioContentCryptoError::InvalidKeyEpoch)
    } else {
        Ok(())
    }
}

fn authenticated_data(key_epoch: u32, aad: &[u8]) -> Vec<u8> {
    let mut authenticated = Vec::with_capacity(AUDIO_CONTENT_AAD_DOMAIN.len() + 4 + aad.len());
    authenticated.extend_from_slice(AUDIO_CONTENT_AAD_DOMAIN);
    authenticated.extend_from_slice(&key_epoch.to_le_bytes());
    authenticated.extend_from_slice(aad);
    authenticated
}

#[cfg(test)]
mod tests {
    use super::*;

    const KEY: [u8; 32] = [
        0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x8b, 0x8c, 0x8d, 0x8e,
        0x8f, 0x90, 0x91, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0x9b, 0x9c, 0x9d,
        0x9e, 0x9f,
    ];
    const NONCE: [u8; 12] = [
        0x07, 0, 0, 0, 0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47,
    ];

    #[test]
    fn round_trips_audio_and_authenticates_metadata() {
        let cipher = AudioContentCipher::new(&KEY).unwrap();
        let aad = b"session=ses_one;stream=trk_program;sequence=41";
        let envelope = cipher.seal(7, &NONCE, b"audio frame", aad).unwrap();

        assert_eq!(&envelope[..4], b"ACE1");
        assert_eq!(u32::from_le_bytes(envelope[4..8].try_into().unwrap()), 7);
        assert_eq!(cipher.open(7, &envelope, aad).unwrap(), b"audio frame");
        assert!(cipher
            .open(7, &envelope, b"session=ses_one;stream=other;sequence=41")
            .is_err());
    }

    #[test]
    fn rejects_zero_keys_wrong_epochs_and_modified_ciphertext() {
        assert_eq!(
            AudioContentCipher::new(&[0; 32]).unwrap_err(),
            AudioContentCryptoError::ZeroKey
        );
        let cipher = AudioContentCipher::new(&KEY).unwrap();
        assert_eq!(
            cipher.seal(0, &NONCE, b"audio", b"metadata").unwrap_err(),
            AudioContentCryptoError::InvalidKeyEpoch
        );

        let mut envelope = cipher.seal(7, &NONCE, b"audio", b"metadata").unwrap();
        assert_eq!(
            cipher.open(8, &envelope, b"metadata").unwrap_err(),
            AudioContentCryptoError::KeyEpochMismatch {
                expected: 8,
                actual: 7,
            }
        );
        *envelope.last_mut().unwrap() ^= 0x80;
        assert!(cipher.open(7, &envelope, b"metadata").is_err());
    }

    #[test]
    fn group_metadata_is_canonical_and_changes_with_every_routing_identity() {
        let metadata = AudioGroupMetadata {
            session_context: b"ses_example",
            transport_session_id: 1,
            config_generation: 2,
            epoch_id: 3,
            pts_samples: 4,
            sample_rate: 48_000,
            frame_count: 240,
            group_count: 2,
            group_id: 7,
            group_index: 1,
            channel_start: 16,
            channel_count: 2,
            payload_kind: 3,
            sample_format: 2,
            flags: 0x81,
        };
        let aad = metadata.associated_data().unwrap();
        assert_eq!(&aad[..4], b"AEG1");
        assert_eq!(aad[4], 11);

        let mut changed = metadata;
        changed.group_id += 1;
        assert_ne!(aad, changed.associated_data().unwrap());
        changed = metadata;
        changed.flags ^= 0x80;
        assert_ne!(aad, changed.associated_data().unwrap());
    }
}
