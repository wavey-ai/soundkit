// Claxon -- A FLAC decoding library in Rust
// Copyright 2014-2018 Ruud van Asseldonk
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// A copy of the License has been included in the root of the repository.

//! Bit-level reading over byte-level readers and contiguous buffers.
//!
//! This module merges the two reader layers of the decoder:
//!
//! * [`ReadBytes`] and [`BufferedReader`] provide buffered byte access with
//!   low overhead, so checksum wrappers can observe every consumed byte.
//! * [`Bitstream`] reads individual bits through any [`ReadBytes`] source.
//! * [`BitReader`] is the fast path for contiguous input: it keeps a 64-bit
//!   left-aligned cache plus one window of look-ahead, so most reads touch
//!   only registers.
//!
//! Decoding code is generic over the [`BitSource`] trait, which both
//! bit-level readers implement, so each source monomorphizes into its own
//! specialized copy.

use std::cmp;
use std::io;

/// Similar to `std::io::BufRead`, but more performant.
///
/// There is no simple way to wrap a standard `BufRead` such that it can compute
/// checksums on consume. This is really something that needs a less restrictive
/// interface. Apart from enabling checksum computations, this buffered reader
/// has some convenience functions.
pub struct BufferedReader<R: io::Read> {
    /// The wrapped reader.
    inner: R,

    /// The buffer that holds data read from the inner reader.
    buf: Box<[u8]>,

    /// The index of the first byte in the buffer which has not been consumed.
    pos: u32,

    /// The number of bytes of the buffer which have meaningful content.
    num_valid: u32,
}

impl<R: io::Read> BufferedReader<R> {
    /// Wrap the reader in a new buffered reader.
    pub fn new(inner: R) -> BufferedReader<R> {
        // Use a large-ish buffer size, such that system call overhead is
        // negligible when replenishing the buffer. However, when fuzzing we
        // want to have small samples, and still trigger the case where we have
        // to replenish the buffer, so use a smaller buffer size there.
        #[cfg(not(fuzzing))]
        const CAPACITY: usize = 2048;

        #[cfg(fuzzing)]
        const CAPACITY: usize = 31;

        let buf = vec![0; CAPACITY].into_boxed_slice();
        BufferedReader {
            inner: inner,
            buf: buf,
            pos: 0,
            num_valid: 0,
        }
    }

    /// Destroys the buffered reader, returning the wrapped reader.
    ///
    /// Anything in the buffer will be lost.
    pub fn into_inner(self) -> R {
        self.inner
    }
}

/// Provides convenience methods to make input less cumbersome.
pub trait ReadBytes {
    /// Reads a single byte, failing on EOF.
    fn read_u8(&mut self) -> io::Result<u8>;

    /// Reads a single byte, not failing on EOF.
    fn read_u8_or_eof(&mut self) -> io::Result<Option<u8>>;

    /// Reads until the provided buffer is full.
    fn read_into(&mut self, buffer: &mut [u8]) -> io::Result<()>;

    /// Skips over the specified number of bytes.
    ///
    /// For a buffered reader, this can help a lot by just bumping a pointer.
    fn skip(&mut self, amount: u32) -> io::Result<()>;

    /// Reads two bytes and interprets them as a big-endian 16-bit unsigned integer.
    fn read_be_u16(&mut self) -> io::Result<u16> {
        let b0 = (self.read_u8())? as u16;
        let b1 = (self.read_u8())? as u16;
        Ok(b0 << 8 | b1)
    }

    /// Reads two bytes and interprets them as a big-endian 16-bit unsigned integer.
    fn read_be_u16_or_eof(&mut self) -> io::Result<Option<u16>> {
        if let Some(b0) = (self.read_u8_or_eof())? {
            if let Some(b1) = (self.read_u8_or_eof())? {
                return Ok(Some((b0 as u16) << 8 | (b1 as u16)));
            }
        }
        Ok(None)
    }

    /// Reads three bytes and interprets them as a big-endian 24-bit unsigned integer.
    fn read_be_u24(&mut self) -> io::Result<u32> {
        let b0 = (self.read_u8())? as u32;
        let b1 = (self.read_u8())? as u32;
        let b2 = (self.read_u8())? as u32;
        Ok(b0 << 16 | b1 << 8 | b2)
    }

    /// Reads four bytes and interprets them as a big-endian 32-bit unsigned integer.
    fn read_be_u32(&mut self) -> io::Result<u32> {
        let b0 = (self.read_u8())? as u32;
        let b1 = (self.read_u8())? as u32;
        let b2 = (self.read_u8())? as u32;
        let b3 = (self.read_u8())? as u32;
        Ok(b0 << 24 | b1 << 16 | b2 << 8 | b3)
    }

    /// Reads four bytes and interprets them as a little-endian 32-bit unsigned integer.
    fn read_le_u32(&mut self) -> io::Result<u32> {
        let b0 = (self.read_u8())? as u32;
        let b1 = (self.read_u8())? as u32;
        let b2 = (self.read_u8())? as u32;
        let b3 = (self.read_u8())? as u32;
        Ok(b3 << 24 | b2 << 16 | b1 << 8 | b0)
    }
}

impl<R: io::Read> ReadBytes for BufferedReader<R> {
    #[inline(always)]
    fn read_u8(&mut self) -> io::Result<u8> {
        if self.pos == self.num_valid {
            // The buffer was depleted, replenish it first.
            self.pos = 0;
            self.num_valid = (self.inner.read(&mut self.buf))? as u32;

            if self.num_valid == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "Expected one more byte.",
                ));
            }
        }

        // At this point there is at least one more byte in the buffer, we
        // checked that above. However, when using regular indexing, the
        // compiler still inserts a bounds check here. It is safe to avoid it.
        let byte = unsafe { *self.buf.get_unchecked(self.pos as usize) };
        self.pos += 1;
        Ok(byte)
    }

    fn read_u8_or_eof(&mut self) -> io::Result<Option<u8>> {
        if self.pos == self.num_valid {
            // The buffer was depleted, try to replenish it first.
            self.pos = 0;
            self.num_valid = (self.inner.read(&mut self.buf))? as u32;

            if self.num_valid == 0 {
                return Ok(None);
            }
        }

        Ok(Some((self.read_u8())?))
    }

    fn read_into(&mut self, buffer: &mut [u8]) -> io::Result<()> {
        let mut bytes_left = buffer.len();

        while bytes_left > 0 {
            let from = buffer.len() - bytes_left;
            let count = cmp::min(bytes_left, (self.num_valid - self.pos) as usize);
            buffer[from..from + count]
                .copy_from_slice(&self.buf[self.pos as usize..self.pos as usize + count]);
            bytes_left -= count;
            self.pos += count as u32;

            if bytes_left > 0 {
                // Replenish the buffer if there is more to be read.
                self.pos = 0;
                self.num_valid = (self.inner.read(&mut self.buf))? as u32;
                if self.num_valid == 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "Expected more bytes.",
                    ));
                }
            }
        }

        Ok(())
    }

    fn skip(&mut self, mut amount: u32) -> io::Result<()> {
        while amount > 0 {
            let num_left = self.num_valid - self.pos;
            let read_now = cmp::min(amount, num_left);
            self.pos += read_now;
            amount -= read_now;

            if amount > 0 {
                // If there is more to skip, refill the buffer first.
                self.pos = 0;
                self.num_valid = (self.inner.read(&mut self.buf))? as u32;

                if self.num_valid == 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "Expected more bytes.",
                    ));
                }
            }
        }
        Ok(())
    }
}

impl<'r, R: ReadBytes> ReadBytes for &'r mut R {
    #[inline(always)]
    fn read_u8(&mut self) -> io::Result<u8> {
        (*self).read_u8()
    }

    fn read_u8_or_eof(&mut self) -> io::Result<Option<u8>> {
        (*self).read_u8_or_eof()
    }

    fn read_into(&mut self, buffer: &mut [u8]) -> io::Result<()> {
        (*self).read_into(buffer)
    }

    fn skip(&mut self, amount: u32) -> io::Result<()> {
        (*self).skip(amount)
    }
}

impl<T: AsRef<[u8]>> ReadBytes for io::Cursor<T> {
    fn read_u8(&mut self) -> io::Result<u8> {
        let pos = self.position();
        if pos < self.get_ref().as_ref().len() as u64 {
            self.set_position(pos + 1);
            Ok(self.get_ref().as_ref()[pos as usize])
        } else {
            Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "unexpected eof",
            ))
        }
    }

    fn read_u8_or_eof(&mut self) -> io::Result<Option<u8>> {
        let pos = self.position();
        if pos < self.get_ref().as_ref().len() as u64 {
            self.set_position(pos + 1);
            Ok(Some(self.get_ref().as_ref()[pos as usize]))
        } else {
            Ok(None)
        }
    }

    fn read_into(&mut self, buffer: &mut [u8]) -> io::Result<()> {
        let pos = self.position();
        if pos + buffer.len() as u64 <= self.get_ref().as_ref().len() as u64 {
            let start = pos as usize;
            let end = pos as usize + buffer.len();
            buffer.copy_from_slice(&self.get_ref().as_ref()[start..end]);
            self.set_position(pos + buffer.len() as u64);
            Ok(())
        } else {
            Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "unexpected eof",
            ))
        }
    }

    fn skip(&mut self, amount: u32) -> io::Result<()> {
        let pos = self.position();
        if pos + amount as u64 <= self.get_ref().as_ref().len() as u64 {
            self.set_position(pos + amount as u64);
            Ok(())
        } else {
            Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "unexpected eof",
            ))
        }
    }
}

#[test]
fn verify_read_into_buffered_reader() {
    let mut reader = BufferedReader::new(io::Cursor::new(vec![2u8, 3, 5, 7, 11, 13, 17, 19, 23]));
    let mut buf1 = [0u8; 3];
    let mut buf2 = [0u8; 5];
    let mut buf3 = [0u8; 2];
    reader.read_into(&mut buf1).ok().unwrap();
    reader.read_into(&mut buf2).ok().unwrap();
    assert!(reader.read_into(&mut buf3).is_err());
    assert_eq!(&buf1[..], &[2u8, 3, 5]);
    assert_eq!(&buf2[..], &[7u8, 11, 13, 17, 19]);
}

#[test]
fn verify_read_into_cursor() {
    let mut cursor = io::Cursor::new(vec![2u8, 3, 5, 7, 11, 13, 17, 19, 23]);
    let mut buf1 = [0u8; 3];
    let mut buf2 = [0u8; 5];
    let mut buf3 = [0u8; 2];
    cursor.read_into(&mut buf1).ok().unwrap();
    cursor.read_into(&mut buf2).ok().unwrap();
    assert!(cursor.read_into(&mut buf3).is_err());
    assert_eq!(&buf1[..], &[2u8, 3, 5]);
    assert_eq!(&buf2[..], &[7u8, 11, 13, 17, 19]);
}

#[test]
fn verify_read_u8_buffered_reader() {
    let mut reader = BufferedReader::new(io::Cursor::new(vec![0u8, 2, 129, 89, 122]));
    assert_eq!(reader.read_u8().unwrap(), 0);
    assert_eq!(reader.read_u8().unwrap(), 2);
    assert_eq!(reader.read_u8().unwrap(), 129);
    assert_eq!(reader.read_u8().unwrap(), 89);
    assert_eq!(reader.read_u8_or_eof().unwrap(), Some(122));
    assert_eq!(reader.read_u8_or_eof().unwrap(), None);
    assert!(reader.read_u8().is_err());
}

#[test]
fn verify_read_u8_cursor() {
    let mut reader = io::Cursor::new(vec![0u8, 2, 129, 89, 122]);
    assert_eq!(reader.read_u8().unwrap(), 0);
    assert_eq!(reader.read_u8().unwrap(), 2);
    assert_eq!(reader.read_u8().unwrap(), 129);
    assert_eq!(reader.read_u8().unwrap(), 89);
    assert_eq!(reader.read_u8_or_eof().unwrap(), Some(122));
    assert_eq!(reader.read_u8_or_eof().unwrap(), None);
    assert!(reader.read_u8().is_err());
}

#[test]
fn verify_read_be_u16_buffered_reader() {
    let mut reader = BufferedReader::new(io::Cursor::new(vec![0u8, 2, 129, 89, 122]));
    assert_eq!(reader.read_be_u16().ok(), Some(2));
    assert_eq!(reader.read_be_u16().ok(), Some(33113));
    assert!(reader.read_be_u16().is_err());
}

#[test]
fn verify_read_be_u16_cursor() {
    let mut cursor = io::Cursor::new(vec![0u8, 2, 129, 89, 122]);
    assert_eq!(cursor.read_be_u16().ok(), Some(2));
    assert_eq!(cursor.read_be_u16().ok(), Some(33113));
    assert!(cursor.read_be_u16().is_err());
}

#[test]
fn verify_read_be_u24_buffered_reader() {
    let mut reader = BufferedReader::new(io::Cursor::new(vec![0u8, 0, 2, 0x8f, 0xff, 0xf3, 122]));
    assert_eq!(reader.read_be_u24().ok(), Some(2));
    assert_eq!(reader.read_be_u24().ok(), Some(9_437_171));
    assert!(reader.read_be_u24().is_err());
}

#[test]
fn verify_read_be_u24_cursor() {
    let mut cursor = io::Cursor::new(vec![0u8, 0, 2, 0x8f, 0xff, 0xf3, 122]);
    assert_eq!(cursor.read_be_u24().ok(), Some(2));
    assert_eq!(cursor.read_be_u24().ok(), Some(9_437_171));
    assert!(cursor.read_be_u24().is_err());
}

#[test]
fn verify_read_be_u32_buffered_reader() {
    let mut reader = BufferedReader::new(io::Cursor::new(vec![
        0u8, 0, 0, 2, 0x80, 0x01, 0xff, 0xe9, 0,
    ]));
    assert_eq!(reader.read_be_u32().ok(), Some(2));
    assert_eq!(reader.read_be_u32().ok(), Some(2_147_614_697));
    assert!(reader.read_be_u32().is_err());
}

#[test]
fn verify_read_be_u32_cursor() {
    let mut cursor = io::Cursor::new(vec![0u8, 0, 0, 2, 0x80, 0x01, 0xff, 0xe9, 0]);
    assert_eq!(cursor.read_be_u32().ok(), Some(2));
    assert_eq!(cursor.read_be_u32().ok(), Some(2_147_614_697));
    assert!(cursor.read_be_u32().is_err());
}

#[test]
fn verify_read_le_u32_buffered_reader() {
    let mut reader = BufferedReader::new(io::Cursor::new(vec![
        2u8, 0, 0, 0, 0xe9, 0xff, 0x01, 0x80, 0,
    ]));
    assert_eq!(reader.read_le_u32().ok(), Some(2));
    assert_eq!(reader.read_le_u32().ok(), Some(2_147_614_697));
    assert!(reader.read_le_u32().is_err());
}

#[test]
fn verify_read_le_u32_cursor() {
    let mut reader = io::Cursor::new(vec![2u8, 0, 0, 0, 0xe9, 0xff, 0x01, 0x80, 0]);
    assert_eq!(reader.read_le_u32().ok(), Some(2));
    assert_eq!(reader.read_le_u32().ok(), Some(2_147_614_697));
    assert!(reader.read_le_u32().is_err());
}

/// Left shift that does not panic when shifting by the integer width.
#[inline(always)]
fn shift_left(x: u8, shift: u32) -> u8 {
    debug_assert!(shift <= 8);

    // We cannot shift a u8 by 8 or more, because Rust panics when shifting by
    // the integer width. But we can definitely shift a u32.
    ((x as u32) << shift) as u8
}

/// Right shift that does not panic when shifting by the integer width.
#[inline(always)]
fn shift_right(x: u8, shift: u32) -> u8 {
    debug_assert!(shift <= 8);

    // We cannot shift a u8 by 8 or more, because Rust panics when shifting by
    // the integer width. But we can definitely shift a u32.
    ((x as u32) >> shift) as u8
}

/// Wraps a `Reader` to facilitate reading that is not byte-aligned.

pub struct Bitstream<R: ReadBytes> {
    /// The source where bits are read from.
    reader: R,
    /// Data read from the reader, but not yet fully consumed.
    data: u8,
    /// The number of bits of `data` that have not been consumed.
    bits_left: u32,
    /// Total number of bytes pulled from the reader so far (diagnostics).
    pulled: u64,
}

impl<R: ReadBytes> Bitstream<R> {
    /// Wraps the reader with a reader that facilitates reading individual bits.
    pub fn new(reader: R) -> Bitstream<R> {
        Bitstream {
            reader: reader,
            data: 0,
            bits_left: 0,
            pulled: 0,
        }
    }

    /// Diagnostics: reports how many bits have been consumed so far.
    pub fn debug_bits_consumed(&self) -> u64 {
        self.pulled * 8 - self.bits_left as u64
    }

    /// Generates a bitmask with 1s in the `bits` most significant bits.
    #[inline(always)]
    fn mask_u8(bits: u32) -> u8 {
        debug_assert!(bits <= 8);

        shift_left(0xff, 8 - bits)
    }

    /// Reads a single bit.
    ///
    /// Reading a single bit can be done more efficiently than reading
    /// more than one bit, because a bit never straddles a byte boundary.
    #[inline(always)]
    pub fn read_bit(&mut self) -> io::Result<bool> {
        // If no bits are left, we will need to read the next byte.
        let result = if self.bits_left == 0 {
            self.pulled += 1;
            let fresh_byte = (self.reader.read_u8())?;

            // What remains later are the 7 least significant bits.
            self.data = fresh_byte << 1;
            self.bits_left = 7;

            // What we report is the most significant bit of the fresh byte.
            fresh_byte & 0b1000_0000
        } else {
            // Consume the most significant bit of the buffer byte.
            let bit = self.data & 0b1000_0000;
            self.data = self.data << 1;
            self.bits_left = self.bits_left - 1;
            bit
        };

        Ok(result != 0)
    }

    /// Reads bits until a 1 is read, and returns the number of zeros read.
    ///
    /// Because the reader buffers a byte internally, reading unary can be done
    /// more efficiently than by just reading bit by bit.
    #[inline(always)]
    pub fn read_unary(&mut self) -> io::Result<u32> {
        // Start initially with the number of zeros that are in the buffer byte
        // already (counting from the most significant bit).
        let mut n = self.data.leading_zeros();

        // If the number of zeros plus the one following it was not more than
        // the bytes left, then there is no need to look further.
        if n < self.bits_left {
            // Note: this shift never shifts by more than 7 places, because
            // bits_left is always at most 7 in between read calls, and the
            // least significant bit of the buffer byte is 0 in that case. So
            // we count either 8 zeros, or less than 7. In the former case we
            // would not have taken this branch, in the latter the shift below
            // is safe.
            self.data = self.data << (n + 1);
            self.bits_left = self.bits_left - (n + 1);
        } else {
            // We inspected more bits than available, so our count is incorrect,
            // and we need to look at the next byte.
            n = self.bits_left;

            // Continue reading bytes until we encounter a one.
            loop {
                self.pulled += 1;
                let fresh_byte = (self.reader.read_u8())?;
                let zeros = fresh_byte.leading_zeros();
                n = n + zeros;
                if zeros < 8 {
                    // We consumed the zeros, plus the one following it.
                    self.bits_left = 8 - (zeros + 1);
                    self.data = shift_left(fresh_byte, zeros + 1);
                    break;
                }
            }
        }

        Ok(n)
    }

    /// Reads at most eight bits.
    #[inline(always)]
    pub fn read_leq_u8(&mut self, bits: u32) -> io::Result<u8> {
        // Of course we can read no more than 8 bits, but we do not want the
        // performance overhead of the assertion, so only do it in debug mode.
        debug_assert!(bits <= 8);

        // If not enough bits left, we will need to read the next byte.
        let result = if self.bits_left < bits {
            // Most significant bits are shifted to the right position already.
            let msb = self.data;

            // Read a single byte.
            self.pulled += 1;
            self.data = (self.reader.read_u8())?;

            // From the next byte, we take the additional bits that we need.
            // Those start at the most significant bit, so we need to shift so
            // that it does not overlap with what we have already.
            let lsb =
                (self.data & Bitstream::<R>::mask_u8(bits - self.bits_left)) >> self.bits_left;

            // Shift out the bits that we have consumed.
            self.data = shift_left(self.data, bits - self.bits_left);
            self.bits_left = 8 - (bits - self.bits_left);

            msb | lsb
        } else {
            let result = self.data & Bitstream::<R>::mask_u8(bits);

            // Shift out the bits that we have consumed.
            self.data = self.data << bits;
            self.bits_left = self.bits_left - bits;

            result
        };

        // If there are more than 8 bits left, we read too far.
        debug_assert!(self.bits_left < 8);

        // The least significant bits should be zero.
        debug_assert_eq!(self.data & !Bitstream::<R>::mask_u8(self.bits_left), 0u8);

        // The resulting data is padded with zeros in the least significant
        // bits, but we want to pad in the most significant bits, so shift.
        Ok(shift_right(result, 8 - bits))
    }

    /// Read n bits, where 8 < n <= 16.
    #[inline(always)]
    pub fn read_gt_u8_leq_u16(&mut self, bits: u32) -> io::Result<u32> {
        debug_assert!((8 < bits) && (bits <= 16));

        // The most significant bits of the current byte are valid. Shift them
        // by 2 so they become the most significant bits of the 10 bit number.
        let mask_msb = 0xffffffff << (bits - self.bits_left);
        let msb = ((self.data as u32) << (bits - 8)) & mask_msb;

        // Continue reading the next bits, because no matter how many bits were
        // still left, there were less than 10.
        let bits_to_read = bits - self.bits_left;
        self.pulled += 1;
        let fresh_byte = (self.reader.read_u8())? as u32;
        let lsb = if bits_to_read >= 8 {
            fresh_byte << (bits_to_read - 8)
        } else {
            fresh_byte >> (8 - bits_to_read)
        };
        let combined = msb | lsb;

        let result = if bits_to_read <= 8 {
            // We have all bits already, update the internal state. If no
            // bits are left we might shift by 8 which is invalid, but in that
            // case the value is not used, so a masked shift is appropriate.
            self.bits_left = 8 - bits_to_read;
            self.data = fresh_byte.wrapping_shl(8 - self.bits_left) as u8;
            combined
        } else {
            // We need to read one more byte to get the final bits.
            self.pulled += 1;
            let fresher_byte = (self.reader.read_u8())? as u32;
            let lsb = fresher_byte >> (16 - bits_to_read);

            // Update the reader state. The wrapping shift is appropriate for
            // the same reason as above.
            self.bits_left = 16 - bits_to_read;
            self.data = fresher_byte.wrapping_shl(8 - self.bits_left) as u8;

            combined | lsb
        };

        Ok(result)
    }

    /// Reads at most 16 bits.
    #[inline(always)]
    pub fn read_leq_u16(&mut self, bits: u32) -> io::Result<u16> {
        // As with read_leq_u8, this only makes sense if we read <= 16 bits.
        debug_assert!(bits <= 16);

        // Note: the following is not the most efficient implementation
        // possible, but it avoids duplicating the complexity of `read_leq_u8`.

        if bits <= 8 {
            let result = (self.read_leq_u8(bits))?;
            Ok(result as u16)
        } else {
            // First read the 8 most significant bits, then read what is left.
            let msb = (self.read_leq_u8(8))? as u16;
            let lsb = (self.read_leq_u8(bits - 8))? as u16;
            Ok((msb << (bits - 8)) | lsb)
        }
    }

    /// Reads at most 32 bits.
    #[inline(always)]
    pub fn read_leq_u32(&mut self, bits: u32) -> io::Result<u32> {
        // As with read_leq_u8, this only makes sense if we read <= 32 bits.
        debug_assert!(bits <= 32);

        // Note: the following is not the most efficient implementation
        // possible, but it avoids duplicating the complexity of `read_leq_u8`.

        if bits <= 16 {
            let result = (self.read_leq_u16(bits))?;
            Ok(result as u32)
        } else {
            // First read the 16 most significant bits, then read what is left.
            let msb = (self.read_leq_u16(16))? as u32;
            let lsb = (self.read_leq_u16(bits - 16))? as u32;
            Ok((msb << (bits - 16)) | lsb)
        }
    }
}

/// The bit-level read operations that FLAC subframe decoding needs.
///
/// [`Bitstream`] implements this for streaming readers, and [`BitReader`] for
/// contiguous in-memory input. Decoding code stays generic over this trait, so
/// each reader monomorphizes into its own specialized copy.
pub trait BitSource {
    /// Reads a single bit.
    fn read_bit(&mut self) -> io::Result<bool>;

    /// Reads bits until a 1 is read, and returns the number of zeros read.
    fn read_unary(&mut self) -> io::Result<u32>;

    /// Reads at most eight bits.
    fn read_leq_u8(&mut self, bits: u32) -> io::Result<u8>;

    /// Reads n bits, where 8 < n <= 16.
    fn read_gt_u8_leq_u16(&mut self, bits: u32) -> io::Result<u32>;

    /// Reads at most 16 bits.
    fn read_leq_u16(&mut self, bits: u32) -> io::Result<u16>;

    /// Reads at most 32 bits.
    fn read_leq_u32(&mut self, bits: u32) -> io::Result<u32>;

    /// Diagnostics hook: reports bits consumed when the implementation can
    /// answer cheaply, otherwise `None`.
    fn debug_bits_consumed(&self) -> Option<u64> {
        None
    }

    /// Reads one Rice-coded unsigned value.
    ///
    /// The value is a unary-encoded quotient (a run of zeros terminated by a
    /// one) followed by a `rice_param`-bit remainder, packed as
    /// `(quotient << rice_param) | remainder`. Implementations with a wide
    /// cache resolve most symbols from a single window; the default reads
    /// both parts separately.
    fn read_rice_unsigned(&mut self, rice_param: u32) -> io::Result<u32> {
        let q = self.read_unary()?;
        let r = if rice_param <= 8 {
            (self.read_leq_u8(rice_param))? as u32
        } else if rice_param <= 16 {
            (self.read_gt_u8_leq_u16(rice_param))?
        } else {
            (self.read_leq_u32(rice_param))?
        };
        Ok((q << rice_param) | r)
    }

    /// Decodes a whole partition of Rice-coded unsigned values.
    ///
    /// Each value uses `param_bits` remainder bits and a unary quotient,
    /// exactly as `read_rice_unsigned` defines, mapped through `map` before
    /// it is stored. Implementations that can amortize work across symbols
    /// override this; the default decodes one symbol at a time through
    /// `read_rice_unsigned`.
    fn decode_rice_partition_into<F>(
        &mut self,
        param_bits: u32,
        buffer: &mut [i32],
        mut map: F,
    ) -> io::Result<()>
    where
        F: FnMut(u32) -> i32,
    {
        for sample in buffer.iter_mut() {
            *sample = map(self.read_rice_unsigned(param_bits)?);
        }
        Ok(())
    }
}

impl<R: ReadBytes> BitSource for Bitstream<R> {
    fn debug_bits_consumed(&self) -> Option<u64> {
        Some(Bitstream::debug_bits_consumed(self))
    }

    #[inline(always)]
    fn read_bit(&mut self) -> io::Result<bool> {
        Bitstream::read_bit(self)
    }

    #[inline(always)]
    fn read_unary(&mut self) -> io::Result<u32> {
        Bitstream::read_unary(self)
    }

    #[inline(always)]
    fn read_leq_u8(&mut self, bits: u32) -> io::Result<u8> {
        Bitstream::read_leq_u8(self, bits)
    }

    #[inline(always)]
    fn read_gt_u8_leq_u16(&mut self, bits: u32) -> io::Result<u32> {
        Bitstream::read_gt_u8_leq_u16(self, bits)
    }

    #[inline(always)]
    fn read_leq_u16(&mut self, bits: u32) -> io::Result<u16> {
        Bitstream::read_leq_u16(self, bits)
    }

    #[inline(always)]
    fn read_leq_u32(&mut self, bits: u32) -> io::Result<u32> {
        Bitstream::read_leq_u32(self, bits)
    }

    // `read_rice_unsigned` is not overridden for the narrow streaming
    // `Bitstream`: it has no wide cache to resolve a symbol from, so the
    // default trait implementation over unary plus remainder reads is what
    // it uses. The fused read lives on `BitReader`, which keeps a 64-bit
    // cache over contiguous input.
}

/// A bit-level reader over contiguous input bytes.
///
/// This is the fast path used by the incremental stream decoder. It keeps a
/// 64-bit cache so that most reads touch only registers, without per-byte
/// calls through a byte-level reader. The next bit to consume is the most
/// significant bit of the cache; bits below the unconsumed region are zero.
pub struct BitReader<'a> {
    /// The bytes to read from.
    bytes: &'a [u8],

    /// One past the index of the last byte loaded into the cache.
    pos: usize,

    /// Unconsumed bits, left-aligned; lower bits are always zero.
    cache: u64,

    /// The number of unconsumed bits in `cache`.
    bits_left: u32,

    /// Bits that follow the unconsumed region of `cache`, also left-aligned.
    ///
    /// Refills first promote these into `cache` and only then touch the
    /// slice, so a refill touches memory once per consumed window instead of
    /// rebuilding its position every time.
    ahead: u64,

    /// The number of valid bits in `ahead`.
    ahead_bits: u32,
}

/// Differential tracing support: records primitive bit-source ops so two
/// reader implementations can be compared op-for-op on identical data.
///
/// Compiled only with the `decode-trace` feature; production builds carry
/// none of this code.
#[cfg(feature = "decode-trace")]
#[doc(hidden)]
pub mod op_trace {
    #[derive(Clone, Copy, Debug)]
    pub struct Op {
        pub tag: u8,
        pub width: u32,
        pub value: u64,
        pub pos_before: u64,
    }

    /// Collects ops when installed via [`install`]; otherwise records nothing.
    use std::cell::RefCell;

    thread_local! {
        static OPS: RefCell<Vec<Op>> = const { RefCell::new(Vec::new()) };
        static ACTIVE: RefCell<bool> = const { RefCell::new(false) };
    }

    pub fn install() {
        OPS.with(|o| o.borrow_mut().clear());
        ACTIVE.with(|a| *a.borrow_mut() = true);
    }

    pub fn uninstall() {
        ACTIVE.with(|a| *a.borrow_mut() = false);
    }

    pub fn record(tag: u8, width: u32, value: u64, pos_before: u64) {
        if !ACTIVE.with(|a| *a.borrow()) {
            return;
        }
        OPS.with(|o| {
            o.borrow_mut().push(Op {
                tag,
                width,
                value,
                pos_before,
            })
        });
    }

    pub fn take_all() -> Vec<Op> {
        OPS.with(|o| std::mem::take(&mut *o.borrow_mut()))
    }
}

/// Wraps any `BitSource`, forwarding reads while optionally logging each op.
#[cfg(feature = "decode-trace")]
pub struct Traced<'s, S: BitSource + ?Sized>(&'s mut S);

#[cfg(feature = "decode-trace")]
impl<S: BitSource + ?Sized> Traced<'_, S> {
    /// Wraps `inner`, recording every forwarded op while tracing is on.
    pub fn new(inner: &mut S) -> Traced<'_, S> {
        Traced(inner)
    }
}

#[cfg(feature = "decode-trace")]
impl<S: BitSource + ?Sized> BitSource for Traced<'_, S> {
    fn read_bit(&mut self) -> io::Result<bool> {
        let pos = self.0.debug_bits_consumed().unwrap_or(0);
        let v = self.0.read_bit()?;
        op_trace::record(b'B', 1, v as u64, pos);
        Ok(v)
    }
    fn read_unary(&mut self) -> io::Result<u32> {
        let pos = self.0.debug_bits_consumed().unwrap_or(0);
        let v = self.0.read_unary()?;
        op_trace::record(b'U', 0, v as u64, pos);
        Ok(v)
    }
    fn read_leq_u8(&mut self, bits: u32) -> io::Result<u8> {
        let pos = self.0.debug_bits_consumed().unwrap_or(0);
        let v = self.0.read_leq_u8(bits)?;
        op_trace::record(b'L', bits, v as u64, pos);
        Ok(v)
    }
    fn read_gt_u8_leq_u16(&mut self, bits: u32) -> io::Result<u32> {
        let pos = self.0.debug_bits_consumed().unwrap_or(0);
        let v = self.0.read_gt_u8_leq_u16(bits)?;
        op_trace::record(b'G', bits, v as u64, pos);
        Ok(v)
    }
    fn read_leq_u16(&mut self, bits: u32) -> io::Result<u16> {
        let pos = self.0.debug_bits_consumed().unwrap_or(0);
        let v = self.0.read_leq_u16(bits)?;
        op_trace::record(b'S', bits, v as u64, pos);
        Ok(v)
    }
    fn read_leq_u32(&mut self, bits: u32) -> io::Result<u32> {
        let pos = self.0.debug_bits_consumed().unwrap_or(0);
        let v = self.0.read_leq_u32(bits)?;
        op_trace::record(b'W', bits, v as u64, pos);
        Ok(v)
    }
    fn debug_bits_consumed(&self) -> Option<u64> {
        self.0.debug_bits_consumed()
    }
}

fn eof_error() -> io::Error {
    io::Error::new(io::ErrorKind::UnexpectedEof, "unexpected end of stream")
}

/// Resolved once per process so the per-symbol fast path never scans the
/// environment.
///
/// Which Rice read is fastest depends on the microarchitecture. The fused
/// single-lookup read wins on the x86-64 hardware we measure (one
/// leading-zero count resolves quotient and remainder together); the plain
/// two-step read wins on Apple silicon under LLVM. Setting
/// `SOUNDKIT_FLAC_RICE_FUSED=1` or `=0` overrides the default either way, so
/// the comparison can be repeated as compilers and CPUs change.
fn fused_rice_selected() -> bool {
    use std::sync::OnceLock;

    #[cfg(target_arch = "x86_64")]
    let arch_default = true;
    #[cfg(not(target_arch = "x86_64"))]
    let arch_default = false;

    static FUSED: OnceLock<bool> = OnceLock::new();
    *FUSED.get_or_init(|| match std::env::var_os("SOUNDKIT_FLAC_RICE_FUSED") {
        Some(value) => value != "0",
        None => arch_default,
    })
}

/// The default Rice read: unary quotient, then the remainder in one go.
fn read_rice_unsigned_default<S: BitSource + ?Sized>(
    source: &mut S,
    rice_param: u32,
) -> io::Result<u32> {
    let q = (source.read_unary())?;
    let r = if rice_param <= 8 {
        ((source.read_leq_u8(rice_param))?) as u32
    } else if rice_param <= 16 {
        (source.read_gt_u8_leq_u16(rice_param))?
    } else {
        (source.read_leq_u32(rice_param))?
    };
    Ok((q << rice_param) | r)
}

impl<'a> BitReader<'a> {
    /// Creates a bit reader over the provided bytes.
    pub fn new(bytes: &'a [u8]) -> BitReader<'a> {
        BitReader {
            bytes,
            pos: 0,
            cache: 0,
            bits_left: 0,
            ahead: 0,
            ahead_bits: 0,
        }
    }

    /// Loads the window of bytes that starts at `pos`.
    ///
    /// Returns the big-endian word with any partial tail zero-padded, the
    /// number of valid bits it holds (at most 64), and how many whole bytes
    /// were taken.
    fn load_window(&self) -> (u64, u32, usize) {
        let available = self.bytes.len() - self.pos;
        if available == 0 {
            return (0, 0, 0);
        }
        let whole = available.min(8);
        let word = if whole == 8 {
            // A single unaligned big-endian load; `whole` proved all eight
            // bytes readable.
            unsafe {
                u64::from_be((self.bytes.as_ptr().add(self.pos) as *const u64).read_unaligned())
            }
        } else {
            let mut tail = [0_u8; 8];
            tail[..whole].copy_from_slice(&self.bytes[self.pos..self.pos + whole]);
            u64::from_be_bytes(tail)
        };
        (word, whole as u32 * 8, whole)
    }

    /// Appends fresh bytes to `ahead` once its bits moved into `cache`.
    #[inline(always)]
    fn refill_ahead(&mut self) {
        if self.ahead_bits > 56 {
            return;
        }
        let (word, valid, taken) = self.load_window();
        if valid == 0 {
            return;
        }
        if self.ahead_bits == 0 {
            self.ahead = word;
        } else {
            let take = (64 - self.ahead_bits).min(valid);
            self.ahead = (self.ahead << take) | (word >> (valid - take));
            self.ahead_bits += take;
        }
        self.pos += taken;
    }

    /// Loads a fresh 64-bit window from the current bit position until the
    /// cache holds more than 56 bits or the input is exhausted.
    ///
    /// Refills prefer promoting the look-ahead window into the cache; the
    /// slice is touched only once per consumed window. One unaligned 8-byte
    /// load replaces per-byte assembly; only the final fragment of the
    /// input, where fewer than eight bytes remain, goes through byte-wise
    /// padding.
    #[inline(always)]
    fn fill(&mut self) {
        if self.bits_left > 56 {
            return;
        }
        if self.ahead_bits > 0 {
            // Append the look-ahead below the unconsumed region of `cache`,
            // keeping one contiguous left-aligned run of unconsumed bits.
            let take = (64 - self.bits_left).min(self.ahead_bits);
            let incoming = if take == 64 {
                self.ahead
            } else {
                self.ahead >> (64 - take)
            };
            self.cache = if self.bits_left == 0 {
                incoming
            } else {
                (self.cache << take) | incoming
            };
            self.bits_left += take;
            self.ahead = if take == 64 { 0 } else { self.ahead << take };
            self.ahead_bits -= take;
        }
        if self.bits_left <= 56 {
            // `ahead` is drained: rebuild the cache from the slice, as when
            // the reader starts. This runs near the end of the input.
            let consumed_bits = self.pos * 8 - self.bits_left as usize;
            let byte = (consumed_bits >> 3) as usize;
            let shift = (consumed_bits & 7) as u32;
            let available = self.bytes.len() - byte;
            if available == 0 {
                return;
            }
            let word = if available >= 8 {
                unsafe {
                    u64::from_be((self.bytes.as_ptr().add(byte) as *const u64).read_unaligned())
                }
            } else {
                let mut tail = [0_u8; 8];
                tail[..available].copy_from_slice(&self.bytes[byte..]);
                u64::from_be_bytes(tail)
            };
            self.cache = word << shift;
            self.pos = byte + available.min(8);
            self.bits_left = (available as u32 * 8 - shift).min(64 - shift);
            return;
        }
        self.refill_ahead();
    }

    /// Consumes `bits` (at most 56) bits and returns them as the low-order
    /// bits of the value, or `None` when the input holds fewer bits.
    #[inline(always)]
    fn try_read(&mut self, bits: u32) -> Option<u64> {
        if bits == 0 {
            return Some(0);
        }
        if bits > self.bits_left {
            self.fill();
            if bits > self.bits_left {
                return None;
            }
        }
        // The unconsumed region starts at the top of the cache, so the top
        // `bits` bits are exactly the requested ones. No mask needed.
        let value = self.cache >> (64 - bits);
        self.cache <<= bits;
        self.bits_left -= bits;
        Some(value)
    }

    /// Consumes trailing padding to the next byte boundary and reports how
    /// many whole bytes the consumed bits span, including that padding.
    pub fn finish_aligned(&mut self) -> io::Result<usize> {
        let consumed_bits = self.pos * 8 - self.bits_left as usize - self.ahead_bits as usize;
        Ok((consumed_bits + 7) / 8)
    }

    /// Reports how many bits have been consumed from the slice so far.
    pub fn bits_consumed(&self) -> usize {
        self.pos * 8 - self.bits_left as usize - self.ahead_bits as usize
    }
}

impl BitSource for BitReader<'_> {
    fn debug_bits_consumed(&self) -> Option<u64> {
        Some(self.bits_consumed() as u64)
    }

    #[inline(always)]
    fn read_bit(&mut self) -> io::Result<bool> {
        match self.try_read(1) {
            Some(value) => Ok(value != 0),
            None => Err(eof_error()),
        }
    }

    fn read_unary(&mut self) -> io::Result<u32> {
        let mut zeros = 0;
        loop {
            if self.bits_left == 0 {
                self.fill();
                if self.bits_left == 0 {
                    return Err(eof_error());
                }
            }
            // The unconsumed region sits at the top of the cache. While it
            // holds a set bit, leading_zeros counts only zeros inside the
            // region; anything below the region is never inspected.
            let leading = self.cache.leading_zeros();
            if leading < self.bits_left {
                // Consume the zero run plus its terminating one bit. When
                // that empties the cache, assign zero outright: a shift by
                // the full width would be masked in release builds and
                // panic in debug builds, either of which resurrects the
                // consumed bit as a ghost below the next refill.
                let take_bits = leading + 1;
                self.bits_left -= take_bits;
                self.cache = if self.bits_left == 0 {
                    0
                } else {
                    self.cache << take_bits
                };
                return Ok(zeros + leading);
            }
            // All remaining buffered bits are zero; account for them and
            // continue with freshly loaded bytes.
            zeros += self.bits_left;
            self.cache = 0;
            self.bits_left = 0;
        }
    }

    #[inline(always)]
    fn read_leq_u8(&mut self, bits: u32) -> io::Result<u8> {
        debug_assert!(bits <= 8);
        match self.try_read(bits) {
            Some(value) => Ok(value as u8),
            None => Err(eof_error()),
        }
    }

    #[inline(always)]
    fn read_gt_u8_leq_u16(&mut self, bits: u32) -> io::Result<u32> {
        debug_assert!((8 < bits) && (bits <= 16));
        match self.try_read(bits) {
            Some(value) => Ok(value as u32),
            None => Err(eof_error()),
        }
    }

    #[inline(always)]
    fn read_leq_u16(&mut self, bits: u32) -> io::Result<u16> {
        debug_assert!(bits <= 16);
        match self.try_read(bits) {
            Some(value) => Ok(value as u16),
            None => Err(eof_error()),
        }
    }

    #[inline(always)]
    fn read_leq_u32(&mut self, bits: u32) -> io::Result<u32> {
        debug_assert!(bits <= 32);
        match self.try_read(bits) {
            Some(value) => Ok(value as u32),
            None => Err(eof_error()),
        }
    }

    #[inline(always)]
    fn read_rice_unsigned(&mut self, rice_param: u32) -> io::Result<u32> {
        debug_assert!(rice_param < 31);

        if !fused_rice_selected() {
            return read_rice_unsigned_default(self, rice_param);
        }

        self.fill();
        // Zeros inside the unconsumed region. Bits below that region are
        // zero as well but were never loaded, so clamp the count to what is
        // actually buffered; padding must not count toward a quotient.
        let zeros = self.cache.leading_zeros().min(self.bits_left);

        // A complete symbol needs `zeros` zero bits, a terminating one, and
        // `rice_param` remainder bits. When all of that sits in the cache,
        // quotient and remainder resolve without touching the slice. This
        // mirrors the fused Golomb read in FFmpeg's FLAC decoder.
        if zeros + 1 + rice_param <= self.bits_left {
            let rest = (self.cache << zeros) << 1;
            let r = if rice_param > 0 {
                (rest >> (64 - rice_param)) as u32
            } else {
                0
            };
            // Consume the whole symbol in one go, assigning zero outright
            // when it empties the cache. A shift by the full width would be
            // masked in release builds and panic in debug builds.
            let symbol_bits = zeros + 1 + rice_param;
            self.bits_left -= symbol_bits;
            self.cache = if self.bits_left == 0 {
                0
            } else {
                self.cache << symbol_bits
            };
            return Ok(((zeros as u32) << rice_param) | r);
        }

        // Slow path: the quotient run is long, or input ends nearby.
        read_rice_unsigned_default(self, rice_param)
    }

    fn decode_rice_partition_into<F>(
        &mut self,
        param_bits: u32,
        buffer: &mut [i32],
        mut map: F,
    ) -> io::Result<()>
    where
        F: FnMut(u32) -> i32,
    {
        debug_assert!(param_bits < 31);

        // Windowed decoding: refill once, then resolve every symbol that
        // fits entirely in the current cache before touching the input
        // again. Inside the window a zero run that terminates before
        // `bits_left` consists of real bits: the zero padding below the
        // unconsumed region can only push `leading_zeros` to or past
        // `bits_left`, and such runs exit to the refill scan below.
        let shift_back = 64_u32.wrapping_sub(param_bits);
        let mut i = 0;
        while i < buffer.len() {
            self.fill();
            if self.bits_left == 0 {
                return Err(eof_error());
            }

            // Resolve symbols while each whole symbol sits in one window:
            // quotient and remainder resolve with shifts alone.
            loop {
                let zeros = self.cache.leading_zeros();
                let symbol_bits = zeros + 1 + param_bits;
                if symbol_bits > self.bits_left {
                    break;
                }
                let rest = (self.cache << zeros) << 1;
                let r = if param_bits > 0 {
                    (rest >> shift_back) as u32
                } else {
                    0
                };
                self.bits_left -= symbol_bits;
                self.cache = if self.bits_left == 0 {
                    0
                } else {
                    self.cache << symbol_bits
                };
                buffer[i] = map(((zeros as u32) << param_bits) | r);
                i += 1;
                if i == buffer.len() {
                    return Ok(());
                }
            }

            // The inner loop exits for two different reasons and they need
            // different handling:
            //
            // * The terminating one-bit sits inside this window and only
            //   the remainder spilled past it. Consume the quotient here,
            //   then read the remainder across the next refill.
            // * The whole zero run reaches past the window. Account for
            //   what is buffered, then scan fresh windows until a set bit
            //   appears. Zeroing the cache is safe because every buffered
            //   bit is part of the quotient.
            let zeros = self.cache.leading_zeros();
            if zeros < self.bits_left {
                let take_bits = zeros + 1;
                self.bits_left -= take_bits;
                self.cache = if self.bits_left == 0 {
                    0
                } else {
                    self.cache << take_bits
                };
                let remainder = if param_bits > 0 {
                    match self.try_read(param_bits) {
                        Some(value) => value as u32,
                        None => return Err(eof_error()),
                    }
                } else {
                    0
                };
                buffer[i] = map(((zeros as u32) << param_bits) | remainder);
                i += 1;
                continue;
            }
            let mut quotient = zeros.min(self.bits_left);
            self.cache = 0;
            self.bits_left = 0;
            loop {
                self.fill();
                if self.bits_left == 0 {
                    return Err(eof_error());
                }
                let leading = self.cache.leading_zeros();
                if leading < self.bits_left {
                    quotient += leading;
                    let take_bits = leading + 1;
                    self.bits_left -= take_bits;
                    self.cache = if self.bits_left == 0 {
                        0
                    } else {
                        self.cache << take_bits
                    };
                    break;
                }
                quotient += self.bits_left;
                self.cache = 0;
                self.bits_left = 0;
            }

            // With the terminating bit consumed, only the remainder remains.
            let remainder = if param_bits > 0 {
                match self.try_read(param_bits) {
                    Some(value) => value as u32,
                    None => return Err(eof_error()),
                }
            } else {
                0
            };
            buffer[i] = map(((quotient as u32) << param_bits) | remainder);
            i += 1;
        }
        Ok(())
    }
}

#[test]
fn verify_read_bit() {
    let data = io::Cursor::new(vec![0b1010_0100, 0b1110_0001]);
    let mut bits = Bitstream::new(BufferedReader::new(data));

    assert_eq!(bits.read_bit().unwrap(), true);
    assert_eq!(bits.read_bit().unwrap(), false);
    assert_eq!(bits.read_bit().unwrap(), true);
    // Mix in reading more bits as well, to ensure that they are compatible.
    assert_eq!(bits.read_leq_u8(1).unwrap(), 0);
    assert_eq!(bits.read_bit().unwrap(), false);
    assert_eq!(bits.read_bit().unwrap(), true);
    assert_eq!(bits.read_bit().unwrap(), false);
    assert_eq!(bits.read_bit().unwrap(), false);

    assert_eq!(bits.read_bit().unwrap(), true);
    assert_eq!(bits.read_bit().unwrap(), true);
    assert_eq!(bits.read_bit().unwrap(), true);
    assert_eq!(bits.read_leq_u8(2).unwrap(), 0);
    assert_eq!(bits.read_bit().unwrap(), false);
    assert_eq!(bits.read_bit().unwrap(), false);
    assert_eq!(bits.read_bit().unwrap(), true);

    assert!(bits.read_bit().is_err());
}

#[test]
fn verify_read_unary() {
    let data = io::Cursor::new(vec![
        0b1010_0100,
        0b1000_0000,
        0b0010_0000,
        0b0000_0000,
        0b0000_1010,
    ]);
    let mut bits = Bitstream::new(BufferedReader::new(data));

    assert_eq!(bits.read_unary().unwrap(), 0);
    assert_eq!(bits.read_unary().unwrap(), 1);
    assert_eq!(bits.read_unary().unwrap(), 2);

    // The ending one is after the first byte boundary.
    assert_eq!(bits.read_unary().unwrap(), 2);

    assert_eq!(bits.read_unary().unwrap(), 9);

    // This one skips a full byte of zeros.
    assert_eq!(bits.read_unary().unwrap(), 17);

    // Verify that the ending position is still correct.
    assert_eq!(bits.read_leq_u8(3).unwrap(), 0b010);
    assert!(bits.read_bit().is_err());
}

#[test]
fn verify_read_leq_u8() {
    let data = io::Cursor::new(vec![
        0b1010_0101,
        0b1110_0001,
        0b1101_0010,
        0b0101_0101,
        0b0111_0011,
        0b0011_1111,
        0b1010_1010,
        0b0000_1100,
    ]);
    let mut bits = Bitstream::new(BufferedReader::new(data));

    assert_eq!(bits.read_leq_u8(0).unwrap(), 0);
    assert_eq!(bits.read_leq_u8(1).unwrap(), 1);
    assert_eq!(bits.read_leq_u8(1).unwrap(), 0);
    assert_eq!(bits.read_leq_u8(2).unwrap(), 0b10);
    assert_eq!(bits.read_leq_u8(2).unwrap(), 0b01);
    assert_eq!(bits.read_leq_u8(3).unwrap(), 0b011);
    assert_eq!(bits.read_leq_u8(3).unwrap(), 0b110);
    assert_eq!(bits.read_leq_u8(4).unwrap(), 0b0001);
    assert_eq!(bits.read_leq_u8(5).unwrap(), 0b11010);
    assert_eq!(bits.read_leq_u8(6).unwrap(), 0b010010);
    assert_eq!(bits.read_leq_u8(7).unwrap(), 0b1010101);
    assert_eq!(bits.read_leq_u8(8).unwrap(), 0b11001100);
    assert_eq!(bits.read_leq_u8(6).unwrap(), 0b111111);
    assert_eq!(bits.read_leq_u8(8).unwrap(), 0b10101010);
    assert_eq!(bits.read_leq_u8(4).unwrap(), 0b0000);
    assert_eq!(bits.read_leq_u8(1).unwrap(), 1);
    assert_eq!(bits.read_leq_u8(1).unwrap(), 1);
    assert_eq!(bits.read_leq_u8(2).unwrap(), 0b00);
}

#[test]
fn verify_read_gt_u8_get_u16() {
    let data = io::Cursor::new(vec![
        0b1010_0101,
        0b1110_0001,
        0b1101_0010,
        0b0101_0101,
        0b1111_0000,
    ]);
    let mut bits = Bitstream::new(BufferedReader::new(data));

    assert_eq!(bits.read_gt_u8_leq_u16(10).unwrap(), 0b1010_0101_11);
    assert_eq!(bits.read_gt_u8_leq_u16(10).unwrap(), 0b10_0001_1101);
    assert_eq!(bits.read_leq_u8(3).unwrap(), 0b001);
    assert_eq!(bits.read_gt_u8_leq_u16(10).unwrap(), 0b0_0101_0101_1);
    assert_eq!(bits.read_leq_u8(7).unwrap(), 0b111_0000);
    assert!(bits.read_gt_u8_leq_u16(10).is_err());
}

#[test]
fn verify_read_leq_u16() {
    let data = io::Cursor::new(vec![0b1010_0101, 0b1110_0001, 0b1101_0010, 0b0101_0101]);
    let mut bits = Bitstream::new(BufferedReader::new(data));

    assert_eq!(bits.read_leq_u16(0).unwrap(), 0);
    assert_eq!(bits.read_leq_u16(1).unwrap(), 1);
    assert_eq!(bits.read_leq_u16(13).unwrap(), 0b010_0101_1110_00);
    assert_eq!(bits.read_leq_u16(9).unwrap(), 0b01_1101_001);
}

#[test]
fn verify_read_leq_u32() {
    let data = io::Cursor::new(vec![0b1010_0101, 0b1110_0001, 0b1101_0010, 0b0101_0101]);
    let mut bits = Bitstream::new(BufferedReader::new(data));

    assert_eq!(bits.read_leq_u32(1).unwrap(), 1);
    assert_eq!(bits.read_leq_u32(17).unwrap(), 0b010_0101_1110_0001_11);
    assert_eq!(bits.read_leq_u32(14).unwrap(), 0b01_0010_0101_0101);
}

#[test]
fn verify_read_mixed() {
    // These test data are warm-up samples from an actual stream.
    let data = io::Cursor::new(vec![
        0x03, 0xc7, 0xbf, 0xe5, 0x9b, 0x74, 0x1e, 0x3a, 0xdd, 0x7d, 0xc5, 0x5e, 0xf6, 0xbf, 0x78,
        0x1b, 0xbd,
    ]);
    let mut bits = Bitstream::new(BufferedReader::new(data));

    assert_eq!(bits.read_leq_u8(6).unwrap(), 0);
    assert_eq!(bits.read_leq_u8(1).unwrap(), 1);
    let minus = 1u32 << 16;
    assert_eq!(
        bits.read_leq_u32(17).unwrap(),
        minus | (-14401_i16 as u16 as u32)
    );
    assert_eq!(
        bits.read_leq_u32(17).unwrap(),
        minus | (-13514_i16 as u16 as u32)
    );
    assert_eq!(
        bits.read_leq_u32(17).unwrap(),
        minus | (-12168_i16 as u16 as u32)
    );
    assert_eq!(
        bits.read_leq_u32(17).unwrap(),
        minus | (-10517_i16 as u16 as u32)
    );
    assert_eq!(
        bits.read_leq_u32(17).unwrap(),
        minus | (-09131_i16 as u16 as u32)
    );
    assert_eq!(
        bits.read_leq_u32(17).unwrap(),
        minus | (-08489_i16 as u16 as u32)
    );
    assert_eq!(
        bits.read_leq_u32(17).unwrap(),
        minus | (-08698_i16 as u16 as u32)
    );
}

#[test]
fn bitreader_matches_bitstream_on_random_op_sequences() {
    use std::io::Cursor;

    // Deterministic xorshift so failures reproduce.
    let mut seed: u64 = 0x1234_5678_9abc_def0;
    let mut rng = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };

    for _case in 0..2000 {
        // Random byte payload; occasionally make long zero runs to exercise
        // multi-byte unary quotients.
        let len = (rng() % 64) as usize + 8;
        let mut data: Vec<u8> = Vec::with_capacity(len);
        let zero_run = rng() % 4 == 0;
        for i in 0..len {
            let byte = if zero_run && i < len / 2 {
                0
            } else {
                rng() as u8
            };
            data.push(byte);
        }

        let mut streaming = Bitstream::new(BufferedReader::new(Cursor::new(data.clone())));
        let mut slice = BitReader::new(&data);

        let ops = (rng() % 40) as u32 + 10;
        for _ in 0..ops {
            let op = rng() % 6;
            let bits = (rng() % 32 + 1) as u32;
            match op {
                0 => {
                    let a = streaming.read_bit().ok().map(|b| b as u32);
                    let b = slice.read_bit().ok().map(|b| b as u32);
                    assert_eq!(a, b, "read_bit mismatch on data {data:?}");
                    if a.is_none() {
                        break;
                    }
                }
                1 => {
                    let w = bits % 9;
                    let a = streaming.read_leq_u8(w).ok().map(|v| v as u32);
                    let b = slice.read_leq_u8(w).ok().map(|v| v as u32);
                    assert_eq!(a, b, "read_leq_u8({w}) mismatch on data {data:?}");
                    if a.is_none() {
                        break;
                    }
                }
                2 => {
                    let w = 9 + (rng() % 8) as u32;
                    let a = streaming.read_gt_u8_leq_u16(w).ok();
                    let b = slice.read_gt_u8_leq_u16(w).ok();
                    assert_eq!(a, b, "read_gt_u8_leq_u16({w}) mismatch on data {data:?}");
                    if a.is_none() {
                        break;
                    }
                }
                3 => {
                    let a = streaming.read_unary().ok();
                    let b = slice.read_unary().ok();
                    assert_eq!(a, b, "read_unary mismatch on data {data:?}");
                    if a.is_none() {
                        break;
                    }
                }
                4 => {
                    let a = streaming.read_leq_u32(bits).ok();
                    let b = slice.read_leq_u32(bits).ok();
                    assert_eq!(a, b, "read_leq_u32({bits}) mismatch on data {data:?}");
                    if a.is_none() {
                        break;
                    }
                }
                _ => {
                    let k = (rng() % 31) as u32;
                    let a = BitSource::read_rice_unsigned(&mut streaming, k).ok();
                    let b = BitSource::read_rice_unsigned(&mut slice, k).ok();
                    assert_eq!(a, b, "read_rice_unsigned({k}) mismatch on data {data:?}");
                    if a.is_none() {
                        break;
                    }
                }
            }
        }
    }
}

#[test]
fn verify_unary_across_full_cache_refill_no_ghost_bits() {
    // Regression pin for the ghost-bit corruption class.
    //
    // Historically, a unary symbol of exactly 64 bits (63 zeros plus its
    // terminator) was consumed with `cache <<= 64`. Release builds mask that
    // shift to a no-op, so the terminator survived below the unconsumed
    // window, and a refill that merged bytes into the cache ORed fresh data
    // on top of the surviving bit. Symbols parsed afterwards read one
    // spurious 1.
    //
    // Two independent changes now prevent that corruption: the refill
    // rebuilds the whole cache instead of merging into it, and every consume
    // that empties the cache assigns zero instead of shifting by the full
    // width (which also avoids the guaranteed panic in debug builds). The
    // layout below pins that behavior: a 64-bit symbol, then a 56-bit symbol
    // whose consume leaves 8 bits buffered, then a final refill of fewer
    // than 8 bytes ahead of an 8-zero symbol.
    let mut bits: Vec<bool> = Vec::new();
    // Symbol 1: quotient 63, spanning the whole first cache window.
    bits.extend(std::iter::repeat(false).take(63));
    bits.push(true);
    // Symbol 2: quotient 55, ending 8 bits short of a second full window.
    bits.extend(std::iter::repeat(false).take(55));
    bits.push(true);
    // Symbol 3: quotient 8, whose zeros straddle the final refill.
    bits.extend(std::iter::repeat(false).take(8));
    bits.push(true);

    let mut encoded = vec![0_u8; (bits.len() + 7) / 8];
    for (index, &bit) in bits.iter().enumerate() {
        if bit {
            encoded[index / 8] |= 1 << (7 - index % 8);
        }
    }

    let mut reader = BitReader::new(&encoded);
    assert_eq!(reader.read_unary().unwrap(), 63);
    assert_eq!(reader.read_unary().unwrap(), 55);
    assert_eq!(reader.read_unary().unwrap(), 8);
}
