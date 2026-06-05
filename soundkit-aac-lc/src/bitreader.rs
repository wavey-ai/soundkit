use crate::error::{AacLcError, Result};

#[derive(Debug, Clone)]
pub struct BitReader<'a> {
    data: &'a [u8],
    bit_pos: usize,
    byte_pos: usize,
    cache: u64,
    cache_bits: u8,
}

impl<'a> BitReader<'a> {
    pub const fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            bit_pos: 0,
            byte_pos: 0,
            cache: 0,
            cache_bits: 0,
        }
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

        self.fill_until(bits);
        Ok((self.cache >> (u64::BITS - bits as u32)) as u32)
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

        let remaining = self.remaining_bits();
        let bits = max_bits.min(remaining.min(u8::MAX as usize) as u8);
        if bits == 0 {
            return Ok((0, 0));
        }

        self.fill_until(bits);
        Ok(((self.cache >> (u64::BITS - bits as u32)) as u32, bits))
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

        self.fill_until(bits);
        let value = (self.cache >> (u64::BITS - bits as u32)) as u32;
        self.consume_cached(bits);
        Ok(value)
    }

    pub fn skip_bits(&mut self, bits: usize) -> Result<()> {
        if self.remaining_bits() < bits {
            return Err(AacLcError::UnexpectedEof {
                requested_bits: bits.min(u8::MAX as usize) as u8,
                remaining_bits: self.remaining_bits(),
            });
        }

        let cached = bits.min(self.cache_bits as usize);
        self.consume_cached(cached as u8);

        let bits = bits - cached;
        let whole_bytes = bits / 8;
        self.byte_pos += whole_bytes;
        self.bit_pos += whole_bytes * 8;

        let tail_bits = (bits % 8) as u8;
        if tail_bits != 0 {
            self.fill_until(tail_bits);
            self.consume_cached(tail_bits);
        }
        Ok(())
    }

    pub(crate) fn consume_cached_prefix(&mut self, bits: u8) {
        self.consume_cached(bits);
    }

    pub fn align_to_byte(&mut self) {
        let padding = (8 - (self.bit_pos & 7)) & 7;
        if padding != 0 {
            self.skip_bits(padding)
                .expect("valid bit reader position can align within the current byte");
        }
    }

    fn fill_until(&mut self, target_bits: u8) {
        while self.cache_bits < target_bits && self.byte_pos < self.data.len() {
            debug_assert!(self.cache_bits <= 56);
            let shift = u64::BITS - 8 - self.cache_bits as u32;
            self.cache |= u64::from(self.data[self.byte_pos]) << shift;
            self.cache_bits += 8;
            self.byte_pos += 1;
        }
    }

    fn consume_cached(&mut self, bits: u8) {
        debug_assert!(bits <= self.cache_bits);
        if bits == 0 {
            return;
        }

        if bits == u64::BITS as u8 {
            self.cache = 0;
        } else {
            self.cache <<= bits as u32;
        }
        self.cache_bits -= bits;
        self.bit_pos += bits as usize;
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
