use chacha20poly1305::{
    aead::{Aead, AeadInPlace, KeyInit, Payload},
    ChaCha20Poly1305, Nonce,
};
use std::fmt;

pub const CHACHA20_POLY1305_KEY_BYTES: usize = 32;
pub const CHACHA20_POLY1305_NONCE_BYTES: usize = 12;
pub const CHACHA20_POLY1305_TAG_BYTES: usize = 16;
pub const CHACHA20_POLY1305_PACKET_OVERHEAD_BYTES: usize =
    CHACHA20_POLY1305_NONCE_BYTES + CHACHA20_POLY1305_TAG_BYTES;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CryptoError {
    InvalidKeyLength { expected: usize, actual: usize },
    InvalidNonceLength { expected: usize, actual: usize },
    InvalidDecimalKey,
    DecimalKeyOverflow,
    PacketTooShort { minimum: usize, actual: usize },
    EncryptFailed,
    DecryptFailed,
}

impl fmt::Display for CryptoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CryptoError::InvalidKeyLength { expected, actual } => {
                write!(f, "invalid key length {actual}; expected {expected} bytes")
            }
            CryptoError::InvalidNonceLength { expected, actual } => {
                write!(
                    f,
                    "invalid nonce length {actual}; expected {expected} bytes"
                )
            }
            CryptoError::InvalidDecimalKey => write!(f, "invalid decimal key string"),
            CryptoError::DecimalKeyOverflow => write!(f, "decimal key does not fit in 32 bytes"),
            CryptoError::PacketTooShort { minimum, actual } => {
                write!(
                    f,
                    "encrypted packet too short {actual}; expected at least {minimum} bytes"
                )
            }
            CryptoError::EncryptFailed => write!(f, "ChaCha20-Poly1305 encryption failed"),
            CryptoError::DecryptFailed => write!(f, "ChaCha20-Poly1305 decryption failed"),
        }
    }
}

impl std::error::Error for CryptoError {}

#[derive(Clone)]
pub struct ChaCha20Poly1305PacketCipher {
    cipher: ChaCha20Poly1305,
}

impl ChaCha20Poly1305PacketCipher {
    pub fn new(key: &[u8]) -> Result<Self, CryptoError> {
        if key.len() != CHACHA20_POLY1305_KEY_BYTES {
            return Err(CryptoError::InvalidKeyLength {
                expected: CHACHA20_POLY1305_KEY_BYTES,
                actual: key.len(),
            });
        }
        Ok(Self {
            cipher: ChaCha20Poly1305::new_from_slice(key).map_err(|_| {
                CryptoError::InvalidKeyLength {
                    expected: CHACHA20_POLY1305_KEY_BYTES,
                    actual: key.len(),
                }
            })?,
        })
    }

    pub fn new_from_decimal_key(encoded: &str) -> Result<Self, CryptoError> {
        let key = chacha20_poly1305_key_from_decimal(encoded)?;
        Self::new(&key)
    }

    pub fn encrypt_nonce_prefixed(
        &self,
        nonce: &[u8],
        plaintext: &[u8],
        aad: &[u8],
    ) -> Result<Vec<u8>, CryptoError> {
        if nonce.len() != CHACHA20_POLY1305_NONCE_BYTES {
            return Err(CryptoError::InvalidNonceLength {
                expected: CHACHA20_POLY1305_NONCE_BYTES,
                actual: nonce.len(),
            });
        }

        let ciphertext = self
            .cipher
            .encrypt(
                Nonce::from_slice(nonce),
                Payload {
                    msg: plaintext,
                    aad,
                },
            )
            .map_err(|_| CryptoError::EncryptFailed)?;
        let mut output = Vec::with_capacity(nonce.len() + ciphertext.len());
        output.extend_from_slice(nonce);
        output.extend_from_slice(&ciphertext);
        Ok(output)
    }

    pub fn encrypt_nonce_prefixed_to(
        &self,
        nonce: &[u8],
        plaintext: &[u8],
        aad: &[u8],
        output: &mut Vec<u8>,
    ) -> Result<(), CryptoError> {
        if nonce.len() != CHACHA20_POLY1305_NONCE_BYTES {
            return Err(CryptoError::InvalidNonceLength {
                expected: CHACHA20_POLY1305_NONCE_BYTES,
                actual: nonce.len(),
            });
        }

        output.clear();
        output.reserve(nonce.len() + plaintext.len() + CHACHA20_POLY1305_TAG_BYTES);
        output.extend_from_slice(nonce);
        output.extend_from_slice(plaintext);

        let tag = self
            .cipher
            .encrypt_in_place_detached(
                Nonce::from_slice(nonce),
                aad,
                &mut output[CHACHA20_POLY1305_NONCE_BYTES..],
            )
            .map_err(|_| CryptoError::EncryptFailed)?;
        output.extend_from_slice(&tag);
        Ok(())
    }

    pub fn decrypt_nonce_prefixed(
        &self,
        packet: &[u8],
        aad: &[u8],
    ) -> Result<Vec<u8>, CryptoError> {
        if packet.len() < CHACHA20_POLY1305_PACKET_OVERHEAD_BYTES {
            return Err(CryptoError::PacketTooShort {
                minimum: CHACHA20_POLY1305_PACKET_OVERHEAD_BYTES,
                actual: packet.len(),
            });
        }

        let (nonce, ciphertext) = packet.split_at(CHACHA20_POLY1305_NONCE_BYTES);
        self.cipher
            .decrypt(
                Nonce::from_slice(nonce),
                Payload {
                    msg: ciphertext,
                    aad,
                },
            )
            .map_err(|_| CryptoError::DecryptFailed)
    }
}

pub fn chacha20_poly1305_key_from_decimal(encoded: &str) -> Result<[u8; 32], CryptoError> {
    let encoded = encoded.trim();
    if encoded.is_empty() {
        return Err(CryptoError::InvalidDecimalKey);
    }

    let mut key = [0u8; CHACHA20_POLY1305_KEY_BYTES];
    for byte in encoded.bytes() {
        if !byte.is_ascii_digit() {
            return Err(CryptoError::InvalidDecimalKey);
        }

        let mut carry = u16::from(byte - b'0');
        for value in key.iter_mut().rev() {
            let next = u16::from(*value) * 10 + carry;
            *value = (next & 0xFF) as u8;
            carry = next >> 8;
        }
        if carry != 0 {
            return Err(CryptoError::DecimalKeyOverflow);
        }
    }

    Ok(key)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_KEY_DECIMAL: &str =
        "83843157117408337365446905028299378179116700186920144823595584430653437972238";

    #[test]
    fn decimal_key_parser_returns_32_bytes() {
        let key = chacha20_poly1305_key_from_decimal(TEST_KEY_DECIMAL).unwrap();
        assert_eq!(key.len(), CHACHA20_POLY1305_KEY_BYTES);
        assert_ne!(key, [0u8; CHACHA20_POLY1305_KEY_BYTES]);
    }

    #[test]
    fn nonce_prefixed_chacha20_poly1305_round_trips() {
        let key = chacha20_poly1305_key_from_decimal(TEST_KEY_DECIMAL).unwrap();
        let cipher = ChaCha20Poly1305PacketCipher::new(&key).unwrap();
        let nonce = [7u8; CHACHA20_POLY1305_NONCE_BYTES];
        let aad = b"soundkit-v2-header";
        let plaintext = b"opus-packet";

        let encrypted = cipher
            .encrypt_nonce_prefixed(&nonce, plaintext, aad)
            .unwrap();
        assert_eq!(&encrypted[..CHACHA20_POLY1305_NONCE_BYTES], nonce);
        assert_ne!(&encrypted[CHACHA20_POLY1305_NONCE_BYTES..], plaintext);

        let decrypted = cipher.decrypt_nonce_prefixed(&encrypted, aad).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn nonce_prefixed_in_place_encryption_matches_allocating_path() {
        let key = chacha20_poly1305_key_from_decimal(TEST_KEY_DECIMAL).unwrap();
        let cipher = ChaCha20Poly1305PacketCipher::new(&key).unwrap();
        let nonce = [9u8; CHACHA20_POLY1305_NONCE_BYTES];
        let aad = b"soundkit-v2-header";
        let plaintext = b"opus-packet";

        let allocated = cipher
            .encrypt_nonce_prefixed(&nonce, plaintext, aad)
            .unwrap();
        let mut reused = Vec::new();
        cipher
            .encrypt_nonce_prefixed_to(&nonce, plaintext, aad, &mut reused)
            .unwrap();

        assert_eq!(reused, allocated);
        assert_eq!(
            cipher.decrypt_nonce_prefixed(&reused, aad).unwrap(),
            plaintext
        );
    }
}
