// Decoder bootstrap derived from alac 0.5.0.
// Copyright (c) 2016 Edward Barnard.
// Modified and maintained in-tree by the SoundKit authors.

/// A buffered, most-significant-bit-first ALAC bit reader.
///
/// Keeping up to 64 bits in a register is important for Rice decoding: the
/// common path consumes several tiny fields per sample, so touching the input
/// slice and checking its bounds for every individual bit is unnecessarily
/// expensive.
#[derive(Clone)]
pub struct BitCursor<'a> {
    bytes: &'a [u8],
    byte_pos: usize,
    cache: u64,
    cached_bits: u8,
}

#[derive(Debug)]
pub struct NotEnoughData;

impl<'a> BitCursor<'a> {
    #[inline]
    pub fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            byte_pos: 0,
            cache: 0,
            cached_bits: 0,
        }
    }

    #[inline(always)]
    pub fn read_bit(&mut self) -> Result<bool, NotEnoughData> {
        self.refill(1)?;
        let value = self.cache >> 63 != 0;
        self.consume(1);
        Ok(value)
    }

    #[inline(always)]
    pub fn read_u8(&mut self, bits: usize) -> Result<u8, NotEnoughData> {
        assert!(bits <= 8);
        Ok(self.read_u32(bits)? as u8)
    }

    #[inline(always)]
    pub fn read_u16(&mut self, bits: usize) -> Result<u16, NotEnoughData> {
        assert!(bits <= 16);
        Ok(self.read_u32(bits)? as u16)
    }

    #[inline(always)]
    pub fn read_u32(&mut self, bits: usize) -> Result<u32, NotEnoughData> {
        assert!(bits <= 32);
        if bits == 0 {
            return Ok(0);
        }
        self.refill(bits as u8)?;
        let value = (self.cache >> (64 - bits)) as u32;
        self.consume(bits as u8);
        Ok(value)
    }

    /// Reads ALAC's unary Rice quotient: at most nine one bits followed by a
    /// zero. The nine-bit fast path replaces up to ten separately checked bit
    /// reads with one cache inspection.
    #[inline(always)]
    pub fn read_unary_ones_max9(&mut self) -> Result<u32, NotEnoughData> {
        const LIMIT: u32 = 9;
        if self.cached_bits < LIMIT as u8 {
            if self.remaining_bits() < LIMIT as usize {
                let mut ones = 0;
                while ones != LIMIT && self.read_bit()? {
                    ones += 1;
                }
                return Ok(ones);
            }
            self.refill(LIMIT as u8)?;
        }
        let ones = self.cache.leading_ones().min(LIMIT);
        self.consume(if ones == LIMIT {
            LIMIT as u8
        } else {
            (ones + 1) as u8
        });
        Ok(ones)
    }

    #[inline]
    pub fn skip(&mut self, bits: usize) -> Result<(), NotEnoughData> {
        if bits > self.remaining_bits() {
            return Err(NotEnoughData);
        }
        if bits <= usize::from(self.cached_bits) {
            self.consume(bits as u8);
            return Ok(());
        }

        let remaining = bits - usize::from(self.cached_bits);
        self.cache = 0;
        self.cached_bits = 0;
        self.byte_pos += remaining / 8;
        let tail = (remaining & 7) as u8;
        if tail != 0 {
            self.refill(tail)?;
            self.consume(tail);
        }
        Ok(())
    }

    #[inline]
    pub fn skip_to_byte(&mut self) -> Result<(), NotEnoughData> {
        let position_in_byte = (8 - (usize::from(self.cached_bits) & 7)) & 7;
        if position_in_byte == 0 {
            Ok(())
        } else {
            self.skip(8 - position_in_byte)
        }
    }

    #[inline(always)]
    fn remaining_bits(&self) -> usize {
        usize::from(self.cached_bits) + (self.bytes.len() - self.byte_pos) * 8
    }

    #[inline(always)]
    fn consume(&mut self, bits: u8) {
        debug_assert!(bits <= self.cached_bits);
        self.cache = if bits == 64 { 0 } else { self.cache << bits };
        self.cached_bits -= bits;
    }

    #[inline(always)]
    fn refill(&mut self, required: u8) -> Result<(), NotEnoughData> {
        if self.cached_bits >= required {
            return Ok(());
        }
        if self.remaining_bits() < usize::from(required) {
            return Err(NotEnoughData);
        }

        if self.cached_bits == 0 && self.bytes.len() - self.byte_pos >= 8 {
            let end = self.byte_pos + 8;
            self.cache = u64::from_be_bytes(
                self.bytes[self.byte_pos..end]
                    .try_into()
                    .expect("checked eight-byte ALAC cache fill"),
            );
            self.byte_pos = end;
            self.cached_bits = 64;
            return Ok(());
        }

        while self.cached_bits < required {
            let byte = self.bytes[self.byte_pos];
            self.byte_pos += 1;
            self.cache |= u64::from(byte) << (56 - self.cached_bits);
            self.cached_bits += 8;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::BitCursor;

    #[test]
    fn reads_across_cache_boundaries() {
        let data: Vec<u8> = (0..17).map(|value| value * 13).collect();
        let mut reader = BitCursor::new(&data);
        assert_eq!(reader.read_u8(4).unwrap(), 0);
        assert_eq!(reader.read_u16(12).unwrap(), 13);
        assert_eq!(reader.read_u32(32).unwrap(), 0x1a27_3441);
        reader.skip(63).unwrap();
        assert_eq!(reader.read_u8(8).unwrap(), 0xdb);
    }

    #[test]
    fn reads_unary_quotients_without_consuming_remainder() {
        let mut reader = BitCursor::new(&[0b1110_1011, 0xff, 0xaa]);
        assert_eq!(reader.read_unary_ones_max9().unwrap(), 3);
        assert_eq!(reader.read_u8(3).unwrap(), 0b101);
        assert_eq!(reader.read_unary_ones_max9().unwrap(), 9);
        assert_eq!(reader.read_u8(8).unwrap(), 0xaa);
    }

    #[test]
    fn skip_to_byte() {
        let data = &[0xde, 0xad];
        let mut reader = BitCursor::new(data);
        reader.read_u8(5).unwrap();
        reader.skip_to_byte().unwrap();
        assert_eq!(reader.read_u8(8).unwrap(), 0xad);
    }
}
