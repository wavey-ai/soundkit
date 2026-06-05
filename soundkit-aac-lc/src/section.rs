use crate::bitreader::BitReader;
use crate::error::{AacLcError, Result};
use crate::ics::{IcsInfo, MAX_WINDOW_GROUPS};

pub const MAX_SCALE_FACTOR_BANDS: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SectionCodebook {
    Zero,
    Spectral(u8),
    Noise,
    Intensity,
    IntensityNegative,
}

impl SectionCodebook {
    pub fn read(reader: &mut BitReader<'_>) -> Result<Self> {
        Self::from_bits(reader.read_u8(4)?)
    }

    pub fn from_bits(value: u8) -> Result<Self> {
        match value {
            0 => Ok(Self::Zero),
            1..=11 => Ok(Self::Spectral(value)),
            12 => Err(AacLcError::InvalidBitstream(
                "reserved AAC section codebook",
            )),
            13 => Ok(Self::Noise),
            14 => Ok(Self::Intensity),
            15 => Ok(Self::IntensityNegative),
            _ => unreachable!("4-bit codebook id cannot exceed 15"),
        }
    }

    pub const fn bits(self) -> u8 {
        match self {
            SectionCodebook::Zero => 0,
            SectionCodebook::Spectral(value) => value,
            SectionCodebook::Noise => 13,
            SectionCodebook::Intensity => 14,
            SectionCodebook::IntensityNegative => 15,
        }
    }
}

impl Default for SectionCodebook {
    fn default() -> Self {
        Self::Zero
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SectionData {
    max_sfb: u8,
    num_window_groups: u8,
    codebooks: [[SectionCodebook; MAX_SCALE_FACTOR_BANDS]; MAX_WINDOW_GROUPS],
}

impl SectionData {
    pub fn read(reader: &mut BitReader<'_>, info: &IcsInfo) -> Result<Self> {
        if info.max_sfb as usize > MAX_SCALE_FACTOR_BANDS {
            return Err(AacLcError::InvalidBitstream(
                "max_sfb exceeds parser capacity",
            ));
        }

        let section_len_bits = if info.window_sequence.is_short() {
            3
        } else {
            5
        };
        let section_escape = (1u8 << section_len_bits) - 1;
        let max_sfb = info.max_sfb as usize;
        let num_window_groups = info.num_window_groups as usize;
        let mut codebooks = [[SectionCodebook::Zero; MAX_SCALE_FACTOR_BANDS]; MAX_WINDOW_GROUPS];

        for group in 0..num_window_groups {
            let mut sfb = 0usize;
            while sfb < max_sfb {
                let codebook = SectionCodebook::read(reader)?;
                let mut section_len = 0usize;

                loop {
                    let section_len_incr = reader.read_u8(section_len_bits)? as usize;
                    section_len += section_len_incr;
                    if section_len_incr != section_escape as usize {
                        break;
                    }
                }

                if section_len == 0 {
                    return Err(AacLcError::InvalidBitstream("zero-length section"));
                }
                if sfb + section_len > max_sfb {
                    return Err(AacLcError::InvalidBitstream(
                        "section length exceeds max_sfb",
                    ));
                }

                for band in sfb..sfb + section_len {
                    codebooks[group][band] = codebook;
                }
                sfb += section_len;
            }
        }

        Ok(Self {
            max_sfb: info.max_sfb,
            num_window_groups: info.num_window_groups,
            codebooks,
        })
    }

    pub const fn max_sfb(&self) -> u8 {
        self.max_sfb
    }

    pub const fn num_window_groups(&self) -> u8 {
        self.num_window_groups
    }

    pub fn codebook(&self, group: usize, sfb: usize) -> Option<SectionCodebook> {
        if group >= self.num_window_groups as usize || sfb >= self.max_sfb as usize {
            return None;
        }
        Some(self.codebooks[group][sfb])
    }

    pub fn is_all_zero(&self) -> bool {
        for group in 0..self.num_window_groups as usize {
            for sfb in 0..self.max_sfb as usize {
                if self.codebooks[group][sfb] != SectionCodebook::Zero {
                    return false;
                }
            }
        }
        true
    }

    pub fn has_intensity_stereo(&self) -> bool {
        for group in 0..self.num_window_groups as usize {
            for sfb in 0..self.max_sfb as usize {
                if matches!(
                    self.codebooks[group][sfb],
                    SectionCodebook::Intensity | SectionCodebook::IntensityNegative
                ) {
                    return true;
                }
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ics::{WindowSequence, WindowShape};

    #[test]
    fn parses_single_long_window_section() {
        let info = long_info(4);
        let bytes = build_bits(&[
            (1, 4), // spectral codebook 1
            (4, 5), // section length
        ]);
        let mut reader = BitReader::new(&bytes);
        let data = SectionData::read(&mut reader, &info).unwrap();

        assert_eq!(data.max_sfb(), 4);
        assert_eq!(data.num_window_groups(), 1);
        for sfb in 0..4 {
            assert_eq!(data.codebook(0, sfb), Some(SectionCodebook::Spectral(1)));
        }
        assert!(!data.is_all_zero());
        assert_eq!(data.codebook(0, 4), None);
    }

    #[test]
    fn detects_all_zero_sections() {
        let info = long_info(3);
        let bytes = build_bits(&[
            (0, 4), // zero codebook
            (3, 5), // section length
        ]);
        let mut reader = BitReader::new(&bytes);
        let data = SectionData::read(&mut reader, &info).unwrap();

        assert!(data.is_all_zero());
    }

    #[test]
    fn parses_long_window_escaped_section_length() {
        let info = long_info(33);
        let bytes = build_bits(&[
            (2, 4),  // spectral codebook 2
            (31, 5), // escape
            (2, 5),  // total length 33
        ]);
        let mut reader = BitReader::new(&bytes);
        let data = SectionData::read(&mut reader, &info).unwrap();

        assert_eq!(data.max_sfb(), 33);
        assert_eq!(data.codebook(0, 0), Some(SectionCodebook::Spectral(2)));
        assert_eq!(data.codebook(0, 32), Some(SectionCodebook::Spectral(2)));
    }

    #[test]
    fn parses_short_window_sections_per_group() {
        let info = IcsInfo {
            window_sequence: WindowSequence::EightShort,
            window_shape: WindowShape::Sine,
            max_sfb: 3,
            num_windows: 8,
            num_window_groups: 2,
            window_group_len: [3, 5, 0, 0, 0, 0, 0, 0],
        };
        let bytes = build_bits(&[
            (1, 4), // group 0 codebook 1
            (1, 3), // len 1
            (0, 4), // group 0 zero
            (2, 3), // len 2
            (5, 4), // group 1 codebook 5
            (3, 3), // len 3
        ]);
        let mut reader = BitReader::new(&bytes);
        let data = SectionData::read(&mut reader, &info).unwrap();

        assert_eq!(data.codebook(0, 0), Some(SectionCodebook::Spectral(1)));
        assert_eq!(data.codebook(0, 1), Some(SectionCodebook::Zero));
        assert_eq!(data.codebook(0, 2), Some(SectionCodebook::Zero));
        for sfb in 0..3 {
            assert_eq!(data.codebook(1, sfb), Some(SectionCodebook::Spectral(5)));
        }
    }

    #[test]
    fn rejects_reserved_codebook() {
        let info = long_info(1);
        let bytes = build_bits(&[(12, 4)]);
        let mut reader = BitReader::new(&bytes);

        assert_eq!(
            SectionData::read(&mut reader, &info).unwrap_err(),
            AacLcError::InvalidBitstream("reserved AAC section codebook")
        );
    }

    #[test]
    fn rejects_section_length_overflow() {
        let info = long_info(2);
        let bytes = build_bits(&[
            (1, 4), // spectral codebook 1
            (3, 5), // section length exceeds max_sfb
        ]);
        let mut reader = BitReader::new(&bytes);

        assert_eq!(
            SectionData::read(&mut reader, &info).unwrap_err(),
            AacLcError::InvalidBitstream("section length exceeds max_sfb")
        );
    }

    fn long_info(max_sfb: u8) -> IcsInfo {
        IcsInfo {
            window_sequence: WindowSequence::OnlyLong,
            window_shape: WindowShape::Sine,
            max_sfb,
            num_windows: 1,
            num_window_groups: 1,
            window_group_len: [1, 0, 0, 0, 0, 0, 0, 0],
        }
    }

    fn build_bits(fields: &[(u32, u8)]) -> Vec<u8> {
        let bit_count: usize = fields.iter().map(|(_, width)| *width as usize).sum();
        let mut out = vec![0u8; (bit_count + 7) / 8];
        let mut bit_pos = 0usize;

        for &(value, width) in fields {
            for bit in (0..width).rev() {
                if ((value >> bit) & 1) != 0 {
                    out[bit_pos / 8] |= 1 << (7 - (bit_pos % 8));
                }
                bit_pos += 1;
            }
        }

        out
    }
}
