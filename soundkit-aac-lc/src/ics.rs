use crate::bitreader::BitReader;
use crate::error::{AacLcError, Result};

pub const MAX_WINDOW_GROUPS: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowSequence {
    OnlyLong,
    LongStart,
    EightShort,
    LongStop,
}

impl WindowSequence {
    pub const fn from_bits(value: u8) -> Self {
        match value {
            0 => Self::OnlyLong,
            1 => Self::LongStart,
            2 => Self::EightShort,
            3 => Self::LongStop,
            _ => unreachable!(),
        }
    }

    pub const fn is_short(self) -> bool {
        matches!(self, Self::EightShort)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowShape {
    Sine,
    KaiserBesselDerived,
}

impl WindowShape {
    pub const fn from_bit(value: bool) -> Self {
        if value {
            Self::KaiserBesselDerived
        } else {
            Self::Sine
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IcsInfo {
    pub window_sequence: WindowSequence,
    pub window_shape: WindowShape,
    pub max_sfb: u8,
    pub num_windows: u8,
    pub num_window_groups: u8,
    pub window_group_len: [u8; MAX_WINDOW_GROUPS],
}

impl IcsInfo {
    pub fn read(reader: &mut BitReader<'_>) -> Result<Self> {
        let reserved_bit = reader.read_bool()?;
        if reserved_bit {
            return Err(AacLcError::InvalidConfig("ICS reserved bit is set"));
        }

        let window_sequence = WindowSequence::from_bits(reader.read_u8(2)?);
        let window_shape = WindowShape::from_bit(reader.read_bool()?);

        if window_sequence.is_short() {
            let max_sfb = reader.read_u8(4)?;
            let scale_factor_grouping = reader.read_u8(7)?;
            let mut window_group_len = [0u8; MAX_WINDOW_GROUPS];
            let mut group = 0usize;
            window_group_len[0] = 1;

            for bit_index in 0..7 {
                let bit = (scale_factor_grouping >> (6 - bit_index)) & 1;
                if bit == 1 {
                    window_group_len[group] += 1;
                } else {
                    group += 1;
                    window_group_len[group] = 1;
                }
            }

            Ok(Self {
                window_sequence,
                window_shape,
                max_sfb,
                num_windows: 8,
                num_window_groups: (group + 1) as u8,
                window_group_len,
            })
        } else {
            let max_sfb = reader.read_u8(6)?;
            let predictor_data_present = reader.read_bool()?;
            if predictor_data_present {
                return Err(AacLcError::UnsupportedFeature("AAC prediction"));
            }

            let mut window_group_len = [0u8; MAX_WINDOW_GROUPS];
            window_group_len[0] = 1;

            Ok(Self {
                window_sequence,
                window_shape,
                max_sfb,
                num_windows: 1,
                num_window_groups: 1,
                window_group_len,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_only_long_ics_info() {
        let bytes = build_bits(&[
            (0, 1),  // reserved
            (0, 2),  // only long
            (1, 1),  // KBD window
            (42, 6), // max_sfb
            (0, 1),  // predictor_data_present
        ]);
        let mut reader = BitReader::new(&bytes);
        let info = IcsInfo::read(&mut reader).unwrap();

        assert_eq!(info.window_sequence, WindowSequence::OnlyLong);
        assert_eq!(info.window_shape, WindowShape::KaiserBesselDerived);
        assert_eq!(info.max_sfb, 42);
        assert_eq!(info.num_windows, 1);
        assert_eq!(info.num_window_groups, 1);
        assert_eq!(info.window_group_len[0], 1);
    }

    #[test]
    fn parses_eight_short_window_groups() {
        let bytes = build_bits(&[
            (0, 1),          // reserved
            (2, 2),          // eight short
            (0, 1),          // sine window
            (12, 4),         // max_sfb
            (0b1100_100, 7), // grouping: 3,1,2,1,1
        ]);
        let mut reader = BitReader::new(&bytes);
        let info = IcsInfo::read(&mut reader).unwrap();

        assert_eq!(info.window_sequence, WindowSequence::EightShort);
        assert_eq!(info.window_shape, WindowShape::Sine);
        assert_eq!(info.max_sfb, 12);
        assert_eq!(info.num_windows, 8);
        assert_eq!(info.num_window_groups, 5);
        assert_eq!(&info.window_group_len[..5], &[3, 1, 2, 1, 1]);
    }

    #[test]
    fn rejects_reserved_bit() {
        let bytes = build_bits(&[(1, 1)]);
        let mut reader = BitReader::new(&bytes);

        assert_eq!(
            IcsInfo::read(&mut reader).unwrap_err(),
            AacLcError::InvalidConfig("ICS reserved bit is set")
        );
    }

    #[test]
    fn rejects_long_window_prediction_for_aac_lc_path() {
        let bytes = build_bits(&[
            (0, 1), // reserved
            (0, 2), // only long
            (0, 1), // sine window
            (20, 6),
            (1, 1), // predictor_data_present
        ]);
        let mut reader = BitReader::new(&bytes);

        assert_eq!(
            IcsInfo::read(&mut reader).unwrap_err(),
            AacLcError::UnsupportedFeature("AAC prediction")
        );
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
