use crate::bitreader::BitReader;
use crate::error::{AacLcError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VlcEntry<T> {
    pub code: u32,
    pub bits: u8,
    pub value: T,
}

#[derive(Debug, Clone, Copy)]
pub struct VlcTable<'a, T> {
    entries: &'a [VlcEntry<T>],
    max_bits: u8,
}

impl<'a, T: Copy + Eq> VlcTable<'a, T> {
    pub fn new(entries: &'a [VlcEntry<T>]) -> Result<Self> {
        if entries.is_empty() {
            return Err(AacLcError::InvalidConfig("VLC table has no entries"));
        }

        let mut max_bits = 0u8;
        for entry in entries {
            if entry.bits == 0 || entry.bits > 32 {
                return Err(AacLcError::InvalidConfig("invalid VLC code length"));
            }
            if entry.code >= (1u32 << entry.bits) {
                return Err(AacLcError::InvalidConfig("VLC code exceeds code length"));
            }
            max_bits = max_bits.max(entry.bits);
        }

        Ok(Self { entries, max_bits })
    }

    pub const fn max_bits(&self) -> u8 {
        self.max_bits
    }

    pub fn read(&self, reader: &mut BitReader<'_>) -> Result<T> {
        let mut code = 0u32;
        for bits in 1..=self.max_bits {
            code = (code << 1) | reader.read_u32(1)?;
            for entry in self.entries {
                if entry.bits == bits && entry.code == code {
                    return Ok(entry.value);
                }
            }
        }

        Err(AacLcError::InvalidBitstream("invalid VLC codeword"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TABLE: [VlcEntry<i8>; 3] = [
        VlcEntry {
            code: 0b0,
            bits: 1,
            value: 0,
        },
        VlcEntry {
            code: 0b10,
            bits: 2,
            value: 1,
        },
        VlcEntry {
            code: 0b11,
            bits: 2,
            value: -1,
        },
    ];

    #[test]
    fn decodes_prefix_codes() {
        let table = VlcTable::new(&TABLE).unwrap();
        let mut reader = BitReader::new(&[0b0101_1000]);

        assert_eq!(table.read(&mut reader).unwrap(), 0);
        assert_eq!(table.read(&mut reader).unwrap(), 1);
        assert_eq!(table.read(&mut reader).unwrap(), -1);
        assert_eq!(table.read(&mut reader).unwrap(), 0);
    }

    #[test]
    fn rejects_invalid_table_code() {
        let entries = [VlcEntry {
            code: 0b10,
            bits: 1,
            value: 0u8,
        }];

        assert_eq!(
            VlcTable::new(&entries).unwrap_err(),
            AacLcError::InvalidConfig("VLC code exceeds code length")
        );
    }

    #[test]
    fn reports_invalid_codeword() {
        let entries = [VlcEntry {
            code: 0b0,
            bits: 1,
            value: 0u8,
        }];
        let table = VlcTable::new(&entries).unwrap();
        let mut reader = BitReader::new(&[0b1000_0000]);

        assert_eq!(
            table.read(&mut reader).unwrap_err(),
            AacLcError::InvalidBitstream("invalid VLC codeword")
        );
    }
}
