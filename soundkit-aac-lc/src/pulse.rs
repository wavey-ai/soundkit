use crate::bitreader::BitReader;
use crate::error::Result;

pub const MAX_PULSES: usize = 4;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Pulse {
    pub offset: u8,
    pub amp: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PulseData {
    pub pulse_start_sfb: u8,
    pub count: u8,
    pub pulses: [Pulse; MAX_PULSES],
}

impl PulseData {
    pub fn read(reader: &mut BitReader<'_>) -> Result<Self> {
        let count = reader.read_u8(2)? + 1;
        let pulse_start_sfb = reader.read_u8(6)?;
        let mut pulses = [Pulse::default(); MAX_PULSES];

        for pulse in pulses.iter_mut().take(count as usize) {
            pulse.offset = reader.read_u8(5)?;
            pulse.amp = reader.read_u8(4)?;
        }

        Ok(Self {
            pulse_start_sfb,
            count,
            pulses,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_one_pulse() {
        let bytes = build_bits(&[
            (0, 2),  // count minus one
            (12, 6), // start sfb
            (7, 5),  // offset
            (9, 4),  // amp
        ]);
        let mut reader = BitReader::new(&bytes);
        let pulse = PulseData::read(&mut reader).unwrap();

        assert_eq!(pulse.count, 1);
        assert_eq!(pulse.pulse_start_sfb, 12);
        assert_eq!(pulse.pulses[0], Pulse { offset: 7, amp: 9 });
    }

    #[test]
    fn parses_four_pulses() {
        let bytes = build_bits(&[
            (3, 2),
            (4, 6),
            (1, 5),
            (2, 4),
            (3, 5),
            (4, 4),
            (5, 5),
            (6, 4),
            (7, 5),
            (8, 4),
        ]);
        let mut reader = BitReader::new(&bytes);
        let pulse = PulseData::read(&mut reader).unwrap();

        assert_eq!(pulse.count, 4);
        assert_eq!(pulse.pulses[3], Pulse { offset: 7, amp: 8 });
    }

    fn build_bits(fields: &[(u32, u8)]) -> Vec<u8> {
        let bit_count: usize = fields.iter().map(|(_, width)| *width as usize).sum();
        let mut out = vec![0u8; bit_count.div_ceil(8)];
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
