use crate::error::{AacLcError, Result};

#[derive(Debug, Clone)]
pub struct BitReader<'a> {
    data: &'a [u8],
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    pub const fn new(data: &'a [u8]) -> Self {
        Self { data, bit_pos: 0 }
    }

    pub const fn bit_pos(&self) -> usize {
        self.bit_pos
    }

    pub const fn len_bits(&self) -> usize {
        self.data.len() * 8
    }

    pub const fn remaining_bits(&self) -> usize {
        self.len_bits().saturating_sub(self.bit_pos)
    }

    pub const fn is_empty(&self) -> bool {
        self.remaining_bits() == 0
    }

    pub fn read_bool(&mut self) -> Result<bool> {
        Ok(self.read_bits(1)? != 0)
    }

    pub fn read_u8(&mut self, bits: u8) -> Result<u8> {
        if bits > 8 {
            return Err(AacLcError::InvalidConfig("read_u8 accepts at most 8 bits"));
        }
        Ok(self.read_bits(bits)? as u8)
    }

    pub fn read_u16(&mut self, bits: u8) -> Result<u16> {
        if bits > 16 {
            return Err(AacLcError::InvalidConfig(
                "read_u16 accepts at most 16 bits",
            ));
        }
        Ok(self.read_bits(bits)? as u16)
    }

    pub fn read_u32(&mut self, bits: u8) -> Result<u32> {
        self.read_bits(bits)
    }

    pub fn peek_u32(&mut self, bits: u8) -> Result<u32> {
        if bits == 0 {
            return Ok(0);
        }
        if bits > 32 {
            return Err(AacLcError::InvalidConfig(
                "cannot read more than 32 bits at once",
            ));
        }
        if self.remaining_bits() < bits as usize {
            return Err(AacLcError::UnexpectedEof {
                requested_bits: bits,
                remaining_bits: self.remaining_bits(),
            });
        }

        Ok(self.peek_bits_unchecked(bits))
    }

    pub(crate) fn peek_prefix(&mut self, max_bits: u8) -> Result<(u32, u8)> {
        if max_bits == 0 {
            return Ok((0, 0));
        }
        if max_bits > 32 {
            return Err(AacLcError::InvalidConfig(
                "cannot read more than 32 bits at once",
            ));
        }

        let bits = max_bits.min(self.remaining_bits().min(u8::MAX as usize) as u8);
        if bits == 0 {
            return Ok((0, 0));
        }

        Ok((self.peek_bits_unchecked(bits), bits))
    }

    /// Fast fixed-width prefix access for padded interior bytes. AAC VLC
    /// tables need at most 19 bits, so four source bytes cover the prefix at
    /// every bit alignment. Only the final three bytes take the checked tail
    /// path.
    #[inline(always)]
    pub(crate) fn peek_prefix_fast<const MAX_BITS: u8>(&mut self) -> Result<(u32, u8)> {
        debug_assert!(MAX_BITS > 0 && MAX_BITS <= 24);
        let byte_pos = self.bit_pos >> 3;
        if self.data.len() - byte_pos >= 4 {
            let word = unsafe {
                core::ptr::read_unaligned(self.data.as_ptr().add(byte_pos).cast::<u32>())
            };
            let bit_offset = (self.bit_pos & 7) as u32;
            return Ok((
                (u32::from_be(word) << bit_offset) >> (32 - u32::from(MAX_BITS)),
                MAX_BITS,
            ));
        }
        self.peek_prefix(MAX_BITS)
    }

    pub fn read_bits(&mut self, bits: u8) -> Result<u32> {
        if bits == 0 {
            return Ok(0);
        }
        if bits > 32 {
            return Err(AacLcError::InvalidConfig(
                "cannot read more than 32 bits at once",
            ));
        }
        if self.remaining_bits() < bits as usize {
            return Err(AacLcError::UnexpectedEof {
                requested_bits: bits,
                remaining_bits: self.remaining_bits(),
            });
        }

        let value = self.peek_bits_unchecked(bits);
        self.bit_pos += usize::from(bits);
        Ok(value)
    }

    pub fn skip_bits(&mut self, bits: usize) -> Result<()> {
        if self.remaining_bits() < bits {
            return Err(AacLcError::UnexpectedEof {
                requested_bits: bits.min(u8::MAX as usize) as u8,
                remaining_bits: self.remaining_bits(),
            });
        }
        self.bit_pos += bits;
        Ok(())
    }

    pub(crate) fn consume_cached_prefix(&mut self, bits: u8) {
        debug_assert!(usize::from(bits) <= self.remaining_bits());
        self.bit_pos += usize::from(bits);
    }

    /// Reads a single sign bit without the full `read_bits` accounting path.
    #[inline]
    pub(crate) fn read_sign_bit(&mut self) -> Result<bool> {
        if self.remaining_bits() == 0 {
            return Err(AacLcError::UnexpectedEof {
                requested_bits: 1,
                remaining_bits: 0,
            });
        }
        let bit = self.peek_bits_unchecked(1);
        self.bit_pos += 1;
        Ok(bit != 0)
    }

    /// Reads up to four consecutive spectral sign bits. AAC codebooks never
    /// need more than four signs per tuple.
    #[inline(always)]
    pub(crate) fn read_sign_bits(&mut self, bits: u8) -> Result<u32> {
        debug_assert!(bits <= 4);
        if bits == 0 {
            return Ok(0);
        }
        if self.remaining_bits() < usize::from(bits) {
            return Err(AacLcError::UnexpectedEof {
                requested_bits: bits,
                remaining_bits: self.remaining_bits(),
            });
        }

        let value = self.peek_bits_unchecked(bits);
        self.bit_pos += usize::from(bits);
        Ok(value)
    }

    pub fn align_to_byte(&mut self) {
        let padding = (8 - (self.bit_pos & 7)) & 7;
        if padding != 0 {
            self.skip_bits(padding)
                .expect("valid bit reader position can align within the current byte");
        }
    }

    #[inline(always)]
    fn peek_bits_unchecked(&self, bits: u8) -> u32 {
        debug_assert!(bits > 0 && bits <= 32);
        debug_assert!(usize::from(bits) <= self.remaining_bits());

        let byte_pos = self.bit_pos >> 3;
        let bit_offset = (self.bit_pos & 7) as u32;
        if u32::from(bits) + bit_offset <= 32 && self.data.len() - byte_pos >= 4 {
            // AAC stores bits most-significant first. An unaligned native word
            // load plus a byte swap maps directly to FFmpeg's AV_RB32-style
            // bit access on little-endian Wasm and x86 hosts.
            let word = unsafe {
                core::ptr::read_unaligned(self.data.as_ptr().add(byte_pos).cast::<u32>())
            };
            return (u32::from_be(word) << bit_offset) >> (32 - u32::from(bits));
        }

        // Only reads at the end of the input or 25-32-bit unaligned reads take
        // this path. Syntax and Huffman fields use the direct word path above.
        let mut value = 0u32;
        for offset in 0..usize::from(bits) {
            let position = self.bit_pos + offset;
            value =
                (value << 1) | u32::from((self.data[position >> 3] >> (7 - (position & 7))) & 1);
        }
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_msb_first_across_byte_boundary() {
        let mut reader = BitReader::new(&[0b1010_1100, 0b0110_0001]);

        assert_eq!(reader.read_bits(3).unwrap(), 0b101);
        assert_eq!(reader.read_bits(5).unwrap(), 0b0_1100);
        assert_eq!(reader.read_bits(4).unwrap(), 0b0110);
        assert_eq!(reader.read_bits(4).unwrap(), 0b0001);
        assert!(reader.is_empty());
    }

    #[test]
    fn reports_eof_without_advancing() {
        let mut reader = BitReader::new(&[0xff]);

        assert_eq!(reader.read_bits(7).unwrap(), 0x7f);
        let err = reader.read_bits(2).unwrap_err();
        assert_eq!(
            err,
            AacLcError::UnexpectedEof {
                requested_bits: 2,
                remaining_bits: 1,
            }
        );
        assert_eq!(reader.bit_pos(), 7);
        assert_eq!(reader.read_bits(1).unwrap(), 1);
    }

    #[test]
    fn peeks_without_advancing_logical_position() {
        let mut reader = BitReader::new(&[0b1010_1100, 0b0110_0001, 0b1111_0000]);

        assert_eq!(reader.read_bits(3).unwrap(), 0b101);
        assert_eq!(reader.peek_u32(9).unwrap(), 0b0_1100_0110);
        assert_eq!(reader.bit_pos(), 3);
        assert_eq!(reader.read_bits(9).unwrap(), 0b0_1100_0110);
        assert_eq!(reader.bit_pos(), 12);
    }

    #[test]
    fn reads_batched_sign_bits_from_cache() {
        let mut reader = BitReader::new(&[0b1011_0010, 0b1100_0000]);

        assert_eq!(reader.peek_u32(8).unwrap(), 0b1011_0010);
        assert_eq!(reader.read_sign_bits(3).unwrap(), 0b101);
        assert_eq!(reader.read_sign_bits(0).unwrap(), 0);
        assert_eq!(reader.read_sign_bits(4).unwrap(), 0b1001);
        assert_eq!(reader.read_sign_bits(4).unwrap(), 0b0110);
        assert_eq!(reader.bit_pos(), 11);
    }

    #[test]
    fn skips_across_prefetched_bytes() {
        let mut reader = BitReader::new(&[
            0b1111_0000,
            0b1010_0101,
            0b0011_1100,
            0b0101_1010,
            0b0000_1111,
        ]);

        assert_eq!(reader.peek_u32(16).unwrap(), 0b1111_0000_1010_0101);
        reader.skip_bits(20).unwrap();
        assert_eq!(reader.bit_pos(), 20);
        assert_eq!(reader.read_bits(8).unwrap(), 0b1100_0101);
    }

    #[test]
    fn aligns_to_next_byte() {
        let mut reader = BitReader::new(&[0xaa, 0x55]);

        assert_eq!(reader.read_bits(3).unwrap(), 0b101);
        reader.align_to_byte();
        assert_eq!(reader.bit_pos(), 8);
        assert_eq!(reader.read_bits(8).unwrap(), 0x55);
    }
}
