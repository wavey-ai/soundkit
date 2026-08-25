use crate::bitreader::BitReader;
use crate::channel::IndividualChannelStreamPrefix;
use crate::dsp::scalefactor_multiplier;
use crate::error::{AacLcError, Result};
use crate::ics::MAX_WINDOW_GROUPS;
use crate::section::{SectionCodebook, MAX_SCALE_FACTOR_BANDS, MAX_SCALE_FACTOR_ENTRIES};
use crate::vlc::VlcTable;
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ScaleFactorValue {
    #[default]
    Zero,
    Spectral(i16),
    Noise(i16),
    Intensity(i16),
    IntensityNegative(i16),
}

#[derive(Debug, Clone)]
pub struct ScaleFactorData {
    max_sfb: u8,
    num_window_groups: u8,
    values: [ScaleFactorValue; MAX_SCALE_FACTOR_ENTRIES],
    multipliers: [f32; MAX_SCALE_FACTOR_ENTRIES],
}

impl PartialEq for ScaleFactorData {
    fn eq(&self, other: &Self) -> bool {
        self.max_sfb == other.max_sfb
            && self.num_window_groups == other.num_window_groups
            && self.values == other.values
    }
}

impl Eq for ScaleFactorData {}

pub trait ScaleFactorDecoder {
    fn read_delta(&mut self, reader: &mut BitReader<'_>) -> Result<i16>;
}

#[derive(Debug, Default)]
pub struct NotImplementedScaleFactorDecoder;

impl ScaleFactorDecoder for NotImplementedScaleFactorDecoder {
    fn read_delta(&mut self, _reader: &mut BitReader<'_>) -> Result<i16> {
        Err(AacLcError::NotImplemented(
            "AAC scalefactor Huffman codebook",
        ))
    }
}

#[derive(Debug)]
pub struct VlcScaleFactorDecoder<'a> {
    table: VlcTable<'a, i16>,
}

impl<'a> VlcScaleFactorDecoder<'a> {
    pub const fn new(table: VlcTable<'a, i16>) -> Self {
        Self { table }
    }
}

impl ScaleFactorDecoder for VlcScaleFactorDecoder<'_> {
    fn read_delta(&mut self, reader: &mut BitReader<'_>) -> Result<i16> {
        self.table.read(reader)
    }
}

#[derive(Debug)]
pub struct StandardScaleFactorDecoder {
    lookup: &'static [u32],
}

impl StandardScaleFactorDecoder {
    pub fn new() -> Self {
        Self {
            lookup: standard_scalefactor_lookup(),
        }
    }
}

impl Default for StandardScaleFactorDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl ScaleFactorDecoder for StandardScaleFactorDecoder {
    fn read_delta(&mut self, reader: &mut BitReader<'_>) -> Result<i16> {
        read_standard_scalefactor_delta(reader, self.lookup)
    }
}

impl ScaleFactorData {
    pub fn read<D: ScaleFactorDecoder>(
        reader: &mut BitReader<'_>,
        prefix: &IndividualChannelStreamPrefix,
        decoder: &mut D,
    ) -> Result<Self> {
        let max_sfb = prefix.ics_info.max_sfb as usize;
        let num_window_groups = prefix.ics_info.num_window_groups as usize;
        if max_sfb > MAX_SCALE_FACTOR_BANDS {
            return Err(AacLcError::InvalidBitstream(
                "max_sfb exceeds scalefactor parser capacity",
            ));
        }
        let entry_count = max_sfb * num_window_groups;
        if num_window_groups > MAX_WINDOW_GROUPS || entry_count > MAX_SCALE_FACTOR_ENTRIES {
            return Err(AacLcError::InvalidBitstream(
                "scale-factor layout exceeds parser capacity",
            ));
        }

        let mut values = [ScaleFactorValue::Zero; MAX_SCALE_FACTOR_ENTRIES];
        let mut multipliers = [0.0; MAX_SCALE_FACTOR_ENTRIES];
        let mut spectral = prefix.global_gain as i16;
        let mut noise = prefix.global_gain as i16 - 90;
        let mut intensity = 0i16;
        let mut first_noise = true;

        for group in 0..num_window_groups {
            for sfb in 0..max_sfb {
                let index = group * max_sfb + sfb;
                let value = &mut values[index];
                *value = match prefix.section_data.codebook(group, sfb).ok_or(
                    AacLcError::InvalidBitstream("missing section codebook for scalefactor band"),
                )? {
                    SectionCodebook::Zero => ScaleFactorValue::Zero,
                    SectionCodebook::Spectral(_) => {
                        spectral = spectral.checked_add(decoder.read_delta(reader)?).ok_or(
                            AacLcError::InvalidBitstream("spectral scalefactor overflow"),
                        )?;
                        multipliers[index] = scalefactor_multiplier(spectral);
                        ScaleFactorValue::Spectral(spectral)
                    }
                    SectionCodebook::Noise => {
                        if first_noise {
                            noise = noise.checked_add(read_noise_pcm_delta(reader)?).ok_or(
                                AacLcError::InvalidBitstream("noise scalefactor overflow"),
                            )?;
                            first_noise = false;
                        } else {
                            noise = noise.checked_add(decoder.read_delta(reader)?).ok_or(
                                AacLcError::InvalidBitstream("noise scalefactor overflow"),
                            )?;
                        }
                        multipliers[index] = scalefactor_multiplier(noise);
                        ScaleFactorValue::Noise(noise)
                    }
                    SectionCodebook::Intensity => {
                        intensity = intensity.checked_add(decoder.read_delta(reader)?).ok_or(
                            AacLcError::InvalidBitstream("intensity scalefactor overflow"),
                        )?;
                        multipliers[index] = intensity_multiplier(intensity);
                        ScaleFactorValue::Intensity(intensity)
                    }
                    SectionCodebook::IntensityNegative => {
                        intensity = intensity.checked_add(decoder.read_delta(reader)?).ok_or(
                            AacLcError::InvalidBitstream("intensity scalefactor overflow"),
                        )?;
                        multipliers[index] = intensity_multiplier(intensity);
                        ScaleFactorValue::IntensityNegative(intensity)
                    }
                };
            }
        }

        Ok(Self {
            max_sfb: prefix.ics_info.max_sfb,
            num_window_groups: prefix.ics_info.num_window_groups,
            values,
            multipliers,
        })
    }

    pub const fn max_sfb(&self) -> u8 {
        self.max_sfb
    }

    pub const fn num_window_groups(&self) -> u8 {
        self.num_window_groups
    }

    pub fn value(&self, group: usize, sfb: usize) -> Option<ScaleFactorValue> {
        if group >= self.num_window_groups as usize || sfb >= self.max_sfb as usize {
            return None;
        }
        Some(self.values[group * self.max_sfb as usize + sfb])
    }

    pub fn spectral_multiplier(&self, group: usize, sfb: usize) -> Result<f32> {
        match self.value(group, sfb).ok_or(AacLcError::InvalidBitstream(
            "missing scalefactor value for band",
        ))? {
            ScaleFactorValue::Spectral(_) => {
                Ok(self.multipliers[group * self.max_sfb as usize + sfb])
            }
            _ => Err(AacLcError::InvalidBitstream(
                "spectral band does not have spectral scalefactor",
            )),
        }
    }

    pub fn noise_multiplier(&self, group: usize, sfb: usize) -> Result<f32> {
        match self.value(group, sfb).ok_or(AacLcError::InvalidBitstream(
            "missing noise scalefactor value",
        ))? {
            ScaleFactorValue::Noise(_) => Ok(self.multipliers[group * self.max_sfb as usize + sfb]),
            _ => Err(AacLcError::InvalidBitstream(
                "noise band does not have a noise scalefactor",
            )),
        }
    }

    pub fn intensity_multiplier(&self, group: usize, sfb: usize) -> Result<f32> {
        match self.value(group, sfb).ok_or(AacLcError::InvalidBitstream(
            "missing intensity scalefactor value",
        ))? {
            ScaleFactorValue::Intensity(_) | ScaleFactorValue::IntensityNegative(_) => {
                Ok(self.multipliers[group * self.max_sfb as usize + sfb])
            }
            _ => Err(AacLcError::InvalidBitstream(
                "intensity band does not have an intensity scalefactor",
            )),
        }
    }
}

fn read_noise_pcm_delta(reader: &mut BitReader<'_>) -> Result<i16> {
    Ok(reader.read_u16(9)? as i16 - 256)
}

fn intensity_multiplier(position: i16) -> f32 {
    2.0_f32.powf(-0.25 * position as f32)
}

const STANDARD_SCALE_FACTOR_LAV: i16 = 60;
const STANDARD_SCALE_FACTOR_MAX_BITS: u8 = 19;
const STANDARD_SCALE_FACTOR_CODEBOOK_LEN: usize = 121;
const SCALE_FACTOR_ROOT_BITS: u8 = 7;
const SCALE_FACTOR_BLOCK_BITS: u8 = 6;
const SCALE_FACTOR_ROOT_LEN: usize = 1 << SCALE_FACTOR_ROOT_BITS;
const SCALE_FACTOR_BLOCK_LEN: usize = 1 << SCALE_FACTOR_BLOCK_BITS;
const SCALE_FACTOR_LOOKUP_BITS_MASK: u32 = 0x1f;
const SCALE_FACTOR_LOOKUP_VALUE_SHIFT: u32 = 8;
const SCALE_FACTOR_LOOKUP_JUMP: u32 = 1 << 31;

const STANDARD_SCALE_FACTOR_CODE_LENGTHS: [u8; STANDARD_SCALE_FACTOR_CODEBOOK_LEN] = [
    0x12, 0x12, 0x12, 0x12, 0x13, 0x13, 0x13, 0x13, 0x13, 0x13, 0x13, 0x13, 0x13, 0x13, 0x13, 0x13,
    0x13, 0x13, 0x13, 0x12, 0x13, 0x12, 0x11, 0x11, 0x10, 0x11, 0x10, 0x10, 0x10, 0x10, 0x0f, 0x0f,
    0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0d, 0x0d, 0x0c, 0x0c, 0x0c, 0x0b, 0x0c, 0x0b, 0x0a, 0x0a,
    0x0a, 0x09, 0x09, 0x08, 0x08, 0x08, 0x07, 0x06, 0x06, 0x05, 0x04, 0x03, 0x01, 0x04, 0x04, 0x05,
    0x06, 0x06, 0x07, 0x07, 0x08, 0x08, 0x09, 0x09, 0x0a, 0x0a, 0x0a, 0x0b, 0x0b, 0x0b, 0x0b, 0x0c,
    0x0c, 0x0d, 0x0d, 0x0d, 0x0e, 0x0e, 0x10, 0x0f, 0x10, 0x0f, 0x12, 0x13, 0x13, 0x13, 0x13, 0x13,
    0x13, 0x13, 0x13, 0x13, 0x13, 0x13, 0x13, 0x13, 0x13, 0x13, 0x13, 0x13, 0x13, 0x13, 0x13, 0x13,
    0x13, 0x13, 0x13, 0x13, 0x13, 0x13, 0x13, 0x13, 0x13,
];

const STANDARD_SCALE_FACTOR_CODES: [u32; STANDARD_SCALE_FACTOR_CODEBOOK_LEN] = [
    0x0003ffe8, 0x0003ffe6, 0x0003ffe7, 0x0003ffe5, 0x0007fff5, 0x0007fff1, 0x0007ffed, 0x0007fff6,
    0x0007ffee, 0x0007ffef, 0x0007fff0, 0x0007fffc, 0x0007fffd, 0x0007ffff, 0x0007fffe, 0x0007fff7,
    0x0007fff8, 0x0007fffb, 0x0007fff9, 0x0003ffe4, 0x0007fffa, 0x0003ffe3, 0x0001ffef, 0x0001fff0,
    0x0000fff5, 0x0001ffee, 0x0000fff2, 0x0000fff3, 0x0000fff4, 0x0000fff1, 0x00007ff6, 0x00007ff7,
    0x00003ff9, 0x00003ff5, 0x00003ff7, 0x00003ff3, 0x00003ff6, 0x00003ff2, 0x00001ff7, 0x00001ff5,
    0x00000ff9, 0x00000ff7, 0x00000ff6, 0x000007f9, 0x00000ff4, 0x000007f8, 0x000003f9, 0x000003f7,
    0x000003f5, 0x000001f8, 0x000001f7, 0x000000fa, 0x000000f8, 0x000000f6, 0x00000079, 0x0000003a,
    0x00000038, 0x0000001a, 0x0000000b, 0x00000004, 0x00000000, 0x0000000a, 0x0000000c, 0x0000001b,
    0x00000039, 0x0000003b, 0x00000078, 0x0000007a, 0x000000f7, 0x000000f9, 0x000001f6, 0x000001f9,
    0x000003f4, 0x000003f6, 0x000003f8, 0x000007f5, 0x000007f4, 0x000007f6, 0x000007f7, 0x00000ff5,
    0x00000ff8, 0x00001ff4, 0x00001ff6, 0x00001ff8, 0x00003ff8, 0x00003ff4, 0x0000fff0, 0x00007ff4,
    0x0000fff6, 0x00007ff5, 0x0003ffe2, 0x0007ffd9, 0x0007ffda, 0x0007ffdb, 0x0007ffdc, 0x0007ffdd,
    0x0007ffde, 0x0007ffd8, 0x0007ffd2, 0x0007ffd3, 0x0007ffd4, 0x0007ffd5, 0x0007ffd6, 0x0007fff2,
    0x0007ffdf, 0x0007ffe7, 0x0007ffe8, 0x0007ffe9, 0x0007ffea, 0x0007ffeb, 0x0007ffe6, 0x0007ffe0,
    0x0007ffe1, 0x0007ffe2, 0x0007ffe3, 0x0007ffe4, 0x0007ffe5, 0x0007ffd7, 0x0007ffec, 0x0007fff4,
    0x0007fff3,
];

fn read_standard_scalefactor_delta(reader: &mut BitReader<'_>, table: &[u32]) -> Result<i16> {
    let (lookahead, lookahead_bits) =
        reader.peek_prefix_fast::<STANDARD_SCALE_FACTOR_MAX_BITS>()?;
    let padded =
        (lookahead as usize) << (STANDARD_SCALE_FACTOR_MAX_BITS as usize - lookahead_bits as usize);
    let mut entry = unsafe {
        *table.get_unchecked(padded >> (STANDARD_SCALE_FACTOR_MAX_BITS - SCALE_FACTOR_ROOT_BITS))
    };
    if entry & SCALE_FACTOR_LOOKUP_JUMP != 0 {
        let base = (entry & !SCALE_FACTOR_LOOKUP_JUMP) as usize;
        let index = (padded >> SCALE_FACTOR_BLOCK_BITS) & (SCALE_FACTOR_BLOCK_LEN - 1);
        entry = unsafe { *table.get_unchecked(base + index) };
        if entry & SCALE_FACTOR_LOOKUP_JUMP != 0 {
            let base = (entry & !SCALE_FACTOR_LOOKUP_JUMP) as usize;
            entry = unsafe { *table.get_unchecked(base + (padded & (SCALE_FACTOR_BLOCK_LEN - 1))) };
        }
    }

    let bits = (entry & SCALE_FACTOR_LOOKUP_BITS_MASK) as u8;
    if bits != 0 && bits <= lookahead_bits {
        reader.consume_cached_prefix(bits);
        let value = ((entry >> SCALE_FACTOR_LOOKUP_VALUE_SHIFT) & 0xff) as i16;
        return Ok(value - STANDARD_SCALE_FACTOR_LAV);
    }

    Err(AacLcError::InvalidBitstream(
        "invalid AAC scalefactor codeword",
    ))
}

fn standard_scalefactor_lookup() -> &'static [u32] {
    static LOOKUP: OnceLock<Box<[u32]>> = OnceLock::new();
    LOOKUP
        .get_or_init(build_standard_scalefactor_lookup)
        .as_ref()
}

fn build_standard_scalefactor_lookup() -> Box<[u32]> {
    let mut table = vec![0u32; SCALE_FACTOR_ROOT_LEN];
    let mut jump_bases = vec![usize::MAX; SCALE_FACTOR_ROOT_LEN];

    for index in 0..STANDARD_SCALE_FACTOR_CODEBOOK_LEN {
        let bits = STANDARD_SCALE_FACTOR_CODE_LENGTHS[index];
        if bits == 0 {
            continue;
        }

        let code = STANDARD_SCALE_FACTOR_CODES[index] as usize;
        let entry = u32::from(bits) | ((index as u32) << SCALE_FACTOR_LOOKUP_VALUE_SHIFT);
        if bits <= SCALE_FACTOR_ROOT_BITS {
            let prefix = code << (SCALE_FACTOR_ROOT_BITS - bits);
            let slots = 1usize << (SCALE_FACTOR_ROOT_BITS - bits);
            fill_scalefactor_entries(&mut table, prefix, slots, entry);
            continue;
        }

        let remaining = bits - SCALE_FACTOR_ROOT_BITS;
        let root_index = code >> remaining;
        let second_base = ensure_scalefactor_block(&mut table, &mut jump_bases, root_index);
        if remaining <= SCALE_FACTOR_BLOCK_BITS {
            let suffix_mask = (1usize << remaining) - 1;
            let prefix = (code & suffix_mask) << (SCALE_FACTOR_BLOCK_BITS - remaining);
            let slots = 1usize << (SCALE_FACTOR_BLOCK_BITS - remaining);
            fill_scalefactor_entries(&mut table, second_base + prefix, slots, entry);
            continue;
        }

        let tail_bits = remaining - SCALE_FACTOR_BLOCK_BITS;
        let second_index = (code >> tail_bits) & (SCALE_FACTOR_BLOCK_LEN - 1);
        let third_base =
            ensure_scalefactor_block(&mut table, &mut jump_bases, second_base + second_index);
        let suffix_mask = (1usize << tail_bits) - 1;
        let prefix = (code & suffix_mask) << (SCALE_FACTOR_BLOCK_BITS - tail_bits);
        let slots = 1usize << (SCALE_FACTOR_BLOCK_BITS - tail_bits);
        fill_scalefactor_entries(&mut table, third_base + prefix, slots, entry);
    }

    table.into_boxed_slice()
}

fn ensure_scalefactor_block(
    table: &mut Vec<u32>,
    jump_bases: &mut Vec<usize>,
    jump_index: usize,
) -> usize {
    if jump_bases[jump_index] != usize::MAX {
        return jump_bases[jump_index];
    }

    let base = table.len();
    table.resize(base + SCALE_FACTOR_BLOCK_LEN, 0);
    jump_bases.resize(base + SCALE_FACTOR_BLOCK_LEN, usize::MAX);
    debug_assert_eq!(table[jump_index], 0);
    table[jump_index] = SCALE_FACTOR_LOOKUP_JUMP | base as u32;
    jump_bases[jump_index] = base;
    base
}

fn fill_scalefactor_entries(table: &mut [u32], start: usize, slots: usize, entry: u32) {
    for slot in &mut table[start..start + slots] {
        debug_assert_eq!(*slot, 0);
        *slot = entry;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ics::{IcsInfo, WindowSequence, WindowShape};
    use crate::section::SectionData;
    use crate::vlc::VlcEntry;

    const MINI_SF: [VlcEntry<i16>; 3] = [
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
    fn parses_zero_sections_without_reading_scalefactor_huffman() {
        let prefix = prefix_with_sections(100, 2, &[(0, 4), (2, 5)]);
        let mut reader = BitReader::new(&[]);
        let mut decoder = NotImplementedScaleFactorDecoder;

        let data = ScaleFactorData::read(&mut reader, &prefix, &mut decoder).unwrap();

        assert_eq!(data.value(0, 0), Some(ScaleFactorValue::Zero));
        assert_eq!(data.value(0, 1), Some(ScaleFactorValue::Zero));
    }

    #[test]
    fn parses_spectral_scalefactors_with_vlc_decoder() {
        let prefix = prefix_with_sections(100, 2, &[(1, 4), (2, 5)]);
        let table = VlcTable::new(&MINI_SF).unwrap();
        let mut decoder = VlcScaleFactorDecoder::new(table);
        let mut reader = BitReader::new(&[0b1011_0000]);

        let data = ScaleFactorData::read(&mut reader, &prefix, &mut decoder).unwrap();

        assert_eq!(data.value(0, 0), Some(ScaleFactorValue::Spectral(101)));
        assert_eq!(data.value(0, 1), Some(ScaleFactorValue::Spectral(100)));
    }

    #[test]
    fn parses_first_noise_scalefactor_from_pcm_bits() {
        let prefix = prefix_with_sections(100, 1, &[(13, 4), (1, 5)]);
        let mut decoder = NotImplementedScaleFactorDecoder;
        let bytes = build_bits(&[(5, 9)]);
        let mut reader = BitReader::new(&bytes);

        let data = ScaleFactorData::read(&mut reader, &prefix, &mut decoder).unwrap();

        assert_eq!(data.value(0, 0), Some(ScaleFactorValue::Noise(-241)));
    }

    #[test]
    fn parses_spectral_scalefactors_with_standard_aac_codebook() {
        let prefix = prefix_with_sections(100, 3, &[(1, 4), (3, 5)]);
        let mut decoder = StandardScaleFactorDecoder::new();
        let mut reader = BitReader::new(&[0b1000_1010]);

        let data = ScaleFactorData::read(&mut reader, &prefix, &mut decoder).unwrap();

        assert_eq!(data.value(0, 0), Some(ScaleFactorValue::Spectral(99)));
        assert_eq!(data.value(0, 1), Some(ScaleFactorValue::Spectral(99)));
        assert_eq!(data.value(0, 2), Some(ScaleFactorValue::Spectral(100)));
    }

    #[test]
    fn standard_scalefactor_table_is_prefix_free() {
        for left in 0..STANDARD_SCALE_FACTOR_CODEBOOK_LEN {
            let left_len = STANDARD_SCALE_FACTOR_CODE_LENGTHS[left];
            let left_code = STANDARD_SCALE_FACTOR_CODES[left];
            assert!(left_len > 0 && left_len <= STANDARD_SCALE_FACTOR_MAX_BITS);
            assert!(left_code < (1u32 << left_len));

            for right in (left + 1)..STANDARD_SCALE_FACTOR_CODEBOOK_LEN {
                let right_len = STANDARD_SCALE_FACTOR_CODE_LENGTHS[right];
                let right_code = STANDARD_SCALE_FACTOR_CODES[right];
                assert!(right_len > 0 && right_len <= STANDARD_SCALE_FACTOR_MAX_BITS);
                assert!(right_code < (1u32 << right_len));

                if left_len <= right_len {
                    let shifted = right_code >> (right_len - left_len);
                    assert_ne!(left_code, shifted);
                } else {
                    let shifted = left_code >> (left_len - right_len);
                    assert_ne!(shifted, right_code);
                }
            }
        }
    }

    #[test]
    fn placeholder_scalefactor_decoder_still_rejects_real_scalefactors() {
        let prefix = prefix_with_sections(100, 1, &[(1, 4), (1, 5)]);
        let mut decoder = NotImplementedScaleFactorDecoder;
        let mut reader = BitReader::new(&[]);

        assert_eq!(
            ScaleFactorData::read(&mut reader, &prefix, &mut decoder).unwrap_err(),
            AacLcError::NotImplemented("AAC scalefactor Huffman codebook")
        );
    }

    fn prefix_with_sections(
        global_gain: u8,
        max_sfb: u8,
        section_fields: &[(u32, u8)],
    ) -> IndividualChannelStreamPrefix {
        let ics_info = IcsInfo {
            window_sequence: WindowSequence::OnlyLong,
            window_shape: WindowShape::Sine,
            max_sfb,
            num_windows: 1,
            num_window_groups: 1,
            window_group_len: [1, 0, 0, 0, 0, 0, 0, 0],
        };
        let bytes = build_bits(section_fields);
        let mut reader = BitReader::new(&bytes);
        let section_data = SectionData::read(&mut reader, &ics_info).unwrap();

        IndividualChannelStreamPrefix {
            global_gain,
            ics_info,
            section_data,
        }
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
