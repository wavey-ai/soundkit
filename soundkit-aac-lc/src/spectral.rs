use crate::bitreader::BitReader;
use crate::dsp::{dequantize_signed_scaled, pow43_table};
use crate::error::{AacLcError, Result};
use crate::ics::WindowSequence;
use crate::pulse::PulseData;
use crate::scalefactor::ScaleFactorData;
use crate::section::SectionCodebook;
use crate::vlc::VlcTable;
use crate::IndividualChannelStreamPrefix;
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpectralCodebookKind {
    SignedQuad,
    UnsignedQuad,
    SignedPair,
    UnsignedPair,
    UnsignedPairEscape,
}

impl SpectralCodebookKind {
    pub fn from_codebook_id(codebook: u8) -> Result<Self> {
        match codebook {
            1 | 2 => Ok(Self::SignedQuad),
            3 | 4 => Ok(Self::UnsignedQuad),
            5 | 6 => Ok(Self::SignedPair),
            7..=10 => Ok(Self::UnsignedPair),
            11 => Ok(Self::UnsignedPairEscape),
            0 => Err(AacLcError::InvalidConfig(
                "zero codebook has no spectral Huffman data",
            )),
            12 => Err(AacLcError::InvalidBitstream(
                "reserved AAC spectral codebook",
            )),
            13..=15 => Err(AacLcError::InvalidConfig(
                "non-spectral codebook cannot decode coefficients",
            )),
            _ => Err(AacLcError::InvalidBitstream(
                "invalid AAC spectral codebook id",
            )),
        }
    }

    pub const fn tuple_len(self) -> usize {
        match self {
            Self::SignedQuad | Self::UnsignedQuad => 4,
            Self::SignedPair | Self::UnsignedPair | Self::UnsignedPairEscape => 2,
        }
    }

    pub const fn uses_extra_sign_bits(self) -> bool {
        matches!(
            self,
            Self::UnsignedQuad | Self::UnsignedPair | Self::UnsignedPairEscape
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TupleCodebook<'a, const N: usize> {
    table: VlcTable<'a, [i32; N]>,
}

impl<'a, const N: usize> TupleCodebook<'a, N> {
    pub const fn new(table: VlcTable<'a, [i32; N]>) -> Self {
        Self { table }
    }

    pub fn read(&self, reader: &mut BitReader<'_>) -> Result<[i32; N]> {
        self.table.read(reader)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BandLayout<'a> {
    offsets: &'a [usize],
}

impl<'a> BandLayout<'a> {
    pub const fn new(offsets: &'a [usize]) -> Self {
        Self { offsets }
    }

    pub const fn offsets(&self) -> &'a [usize] {
        self.offsets
    }

    pub fn band_count(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    pub fn band_range(&self, sfb: usize) -> Result<std::ops::Range<usize>> {
        let start = *self.offsets.get(sfb).ok_or(AacLcError::InvalidConfig(
            "missing scale-factor band offset",
        ))?;
        let end = *self.offsets.get(sfb + 1).ok_or(AacLcError::InvalidConfig(
            "missing scale-factor band end offset",
        ))?;
        if end < start {
            return Err(AacLcError::InvalidConfig(
                "scale-factor band offsets are not monotonic",
            ));
        }
        Ok(start..end)
    }
}

pub trait SpectralDecoder {
    fn read_quantized(
        &mut self,
        reader: &mut BitReader<'_>,
        codebook: u8,
        out: &mut [i32],
    ) -> Result<()>;
}

pub fn decode_signed_tuples<const N: usize>(
    reader: &mut BitReader<'_>,
    table: TupleCodebook<'_, N>,
    out: &mut [i32],
) -> Result<()> {
    if out.len() % N != 0 {
        return Err(AacLcError::InvalidConfig(
            "spectral output length is not a multiple of tuple length",
        ));
    }

    for chunk in out.chunks_exact_mut(N) {
        chunk.copy_from_slice(&table.read(reader)?);
    }

    Ok(())
}

pub fn decode_unsigned_tuples<const N: usize>(
    reader: &mut BitReader<'_>,
    table: TupleCodebook<'_, N>,
    out: &mut [i32],
) -> Result<()> {
    if out.len() % N != 0 {
        return Err(AacLcError::InvalidConfig(
            "spectral output length is not a multiple of tuple length",
        ));
    }

    for chunk in out.chunks_exact_mut(N) {
        let mut tuple = table.read(reader)?;
        for value in &mut tuple {
            if *value < 0 {
                return Err(AacLcError::InvalidBitstream(
                    "unsigned spectral codebook produced a negative value",
                ));
            }
            if *value != 0 && reader.read_bool()? {
                *value = -*value;
            }
        }
        chunk.copy_from_slice(&tuple);
    }

    Ok(())
}

pub fn decode_unsigned_escape_pairs(
    reader: &mut BitReader<'_>,
    table: TupleCodebook<'_, 2>,
    out: &mut [i32],
) -> Result<()> {
    if out.len() % 2 != 0 {
        return Err(AacLcError::InvalidConfig(
            "escape pair output length is not even",
        ));
    }

    for chunk in out.chunks_exact_mut(2) {
        let mut tuple = table.read(reader)?;
        for value in &mut tuple {
            if *value < 0 {
                return Err(AacLcError::InvalidBitstream(
                    "escape spectral codebook produced a negative value",
                ));
            }
        }
        finish_unsigned_escape_pair(reader, &mut tuple)?;
        chunk.copy_from_slice(&tuple);
    }

    Ok(())
}

fn finish_unsigned_escape_pair(reader: &mut BitReader<'_>, tuple: &mut [i32; 2]) -> Result<()> {
    let mut signs = [false; 2];
    for (sign, value) in signs.iter_mut().zip(tuple.iter()) {
        if *value != 0 {
            *sign = reader.read_bool()?;
        }
    }

    for value in tuple.iter_mut() {
        if *value == 16 {
            *value = read_escape_magnitude(reader)?;
        }
    }

    for (value, sign) in tuple.iter_mut().zip(signs) {
        if sign {
            *value = -*value;
        }
    }

    Ok(())
}

fn read_escape_magnitude(reader: &mut BitReader<'_>) -> Result<i32> {
    let mut extra_bits = 4u8;
    while reader.read_bool()? {
        extra_bits = extra_bits
            .checked_add(1)
            .ok_or(AacLcError::InvalidBitstream(
                "AAC escape value is too large",
            ))?;
        if extra_bits > 16 {
            return Err(AacLcError::UnsupportedFeature(
                "AAC escape value above 16 extra bits",
            ));
        }
    }

    Ok((1i32 << extra_bits) + reader.read_u32(extra_bits)? as i32)
}

#[derive(Debug, Default)]
pub struct NotImplementedSpectralDecoder;

impl SpectralDecoder for NotImplementedSpectralDecoder {
    fn read_quantized(
        &mut self,
        _reader: &mut BitReader<'_>,
        _codebook: u8,
        _out: &mut [i32],
    ) -> Result<()> {
        Err(AacLcError::NotImplemented("AAC spectral Huffman codebook"))
    }
}

#[derive(Debug, Default)]
pub struct StandardSpectralDecoder;

impl SpectralDecoder for StandardSpectralDecoder {
    fn read_quantized(
        &mut self,
        reader: &mut BitReader<'_>,
        codebook: u8,
        out: &mut [i32],
    ) -> Result<()> {
        match codebook {
            1 => decode_standard_signed_quad(
                reader,
                out,
                standard_quad_lookup(1),
                STANDARD_CODEBOOK_1_MAX_BITS,
            ),
            2 => decode_standard_signed_quad(
                reader,
                out,
                standard_quad_lookup(2),
                STANDARD_CODEBOOK_2_MAX_BITS,
            ),
            3 => decode_standard_unsigned_quad(
                reader,
                out,
                standard_quad_lookup(3),
                STANDARD_CODEBOOK_3_MAX_BITS,
            ),
            4 => decode_standard_unsigned_quad(
                reader,
                out,
                standard_quad_lookup(4),
                STANDARD_CODEBOOK_4_MAX_BITS,
            ),
            5 => decode_standard_signed_pair(
                reader,
                out,
                standard_pair_lookup(5),
                STANDARD_CODEBOOK_5_MAX_BITS,
            ),
            6 => decode_standard_signed_pair(
                reader,
                out,
                standard_pair_lookup(6),
                STANDARD_CODEBOOK_6_MAX_BITS,
            ),
            7 => decode_standard_unsigned_pair(
                reader,
                out,
                standard_pair_lookup(7),
                STANDARD_CODEBOOK_7_MAX_BITS,
            ),
            8 => decode_standard_unsigned_pair(
                reader,
                out,
                standard_pair_lookup(8),
                STANDARD_CODEBOOK_8_MAX_BITS,
            ),
            9 => decode_standard_unsigned_pair(
                reader,
                out,
                standard_pair_lookup(9),
                STANDARD_CODEBOOK_9_MAX_BITS,
            ),
            10 => decode_standard_unsigned_pair(
                reader,
                out,
                standard_pair_lookup(10),
                STANDARD_CODEBOOK_10_MAX_BITS,
            ),
            11 => decode_standard_escape_pair(reader, out),
            _ => {
                SpectralCodebookKind::from_codebook_id(codebook)?;
                Err(AacLcError::NotImplemented("AAC spectral Huffman codebook"))
            }
        }
    }
}

impl StandardSpectralDecoder {
    fn read_scaled(
        &mut self,
        reader: &mut BitReader<'_>,
        codebook: u8,
        scale: f32,
        pow43: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        match codebook {
            1 => decode_standard_signed_quad_scaled(
                reader,
                out,
                standard_quad_lookup(1),
                STANDARD_CODEBOOK_1_MAX_BITS,
                scale,
                pow43,
            ),
            2 => decode_standard_signed_quad_scaled(
                reader,
                out,
                standard_quad_lookup(2),
                STANDARD_CODEBOOK_2_MAX_BITS,
                scale,
                pow43,
            ),
            3 => decode_standard_unsigned_quad_scaled(
                reader,
                out,
                standard_quad_lookup(3),
                STANDARD_CODEBOOK_3_MAX_BITS,
                scale,
                pow43,
            ),
            4 => decode_standard_unsigned_quad_scaled(
                reader,
                out,
                standard_quad_lookup(4),
                STANDARD_CODEBOOK_4_MAX_BITS,
                scale,
                pow43,
            ),
            5 => decode_standard_signed_pair_scaled(
                reader,
                out,
                standard_pair_lookup(5),
                STANDARD_CODEBOOK_5_MAX_BITS,
                scale,
                pow43,
            ),
            6 => decode_standard_signed_pair_scaled(
                reader,
                out,
                standard_pair_lookup(6),
                STANDARD_CODEBOOK_6_MAX_BITS,
                scale,
                pow43,
            ),
            7 => decode_standard_unsigned_pair_scaled(
                reader,
                out,
                standard_pair_lookup(7),
                STANDARD_CODEBOOK_7_MAX_BITS,
                scale,
                pow43,
            ),
            8 => decode_standard_unsigned_pair_scaled(
                reader,
                out,
                standard_pair_lookup(8),
                STANDARD_CODEBOOK_8_MAX_BITS,
                scale,
                pow43,
            ),
            9 => decode_standard_unsigned_pair_scaled(
                reader,
                out,
                standard_pair_lookup(9),
                STANDARD_CODEBOOK_9_MAX_BITS,
                scale,
                pow43,
            ),
            10 => decode_standard_unsigned_pair_scaled(
                reader,
                out,
                standard_pair_lookup(10),
                STANDARD_CODEBOOK_10_MAX_BITS,
                scale,
                pow43,
            ),
            11 => decode_standard_escape_pair_scaled(reader, out, scale, pow43),
            _ => {
                SpectralCodebookKind::from_codebook_id(codebook)?;
                Err(AacLcError::NotImplemented("AAC spectral Huffman codebook"))
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LengthHalf {
    High,
    Low,
}

#[derive(Clone, Copy, Default)]
struct QuadLookupEntry {
    bits: u8,
    value: [i16; 4],
}

#[derive(Clone, Copy, Default)]
struct PairLookupEntry {
    bits: u8,
    value: [i16; 2],
}

pub(crate) fn warm_standard_spectral_tables() {
    for codebook in 1..=4 {
        let _ = standard_quad_lookup(codebook);
    }
    for codebook in 5..=10 {
        let _ = standard_pair_lookup(codebook);
    }
    let _ = standard_escape_pair_lookup();
}

fn decode_standard_signed_quad(
    reader: &mut BitReader<'_>,
    out: &mut [i32],
    lookup: &[QuadLookupEntry],
    max_bits: u8,
) -> Result<()> {
    if out.len() % 4 != 0 {
        return Err(AacLcError::InvalidConfig(
            "spectral output length is not a multiple of tuple length",
        ));
    }

    for chunk in out.chunks_exact_mut(4) {
        chunk.copy_from_slice(&read_standard_quad_tuple(reader, lookup, max_bits)?);
    }

    Ok(())
}

fn decode_standard_unsigned_quad(
    reader: &mut BitReader<'_>,
    out: &mut [i32],
    lookup: &[QuadLookupEntry],
    max_bits: u8,
) -> Result<()> {
    if out.len() % 4 != 0 {
        return Err(AacLcError::InvalidConfig(
            "spectral output length is not a multiple of tuple length",
        ));
    }

    for chunk in out.chunks_exact_mut(4) {
        let mut tuple = read_standard_quad_tuple(reader, lookup, max_bits)?;
        for value in &mut tuple {
            if *value != 0 && reader.read_bool()? {
                *value = -*value;
            }
        }
        chunk.copy_from_slice(&tuple);
    }

    Ok(())
}

fn decode_standard_signed_quad_scaled(
    reader: &mut BitReader<'_>,
    out: &mut [f32],
    lookup: &[QuadLookupEntry],
    max_bits: u8,
    scale: f32,
    pow43: &[f32],
) -> Result<()> {
    if out.len() % 4 != 0 {
        return Err(AacLcError::InvalidConfig(
            "spectral output length is not a multiple of tuple length",
        ));
    }

    for chunk in out.chunks_exact_mut(4) {
        let tuple = read_standard_quad_tuple(reader, lookup, max_bits)?;
        write_scaled_tuple(&tuple, scale, pow43, chunk);
    }

    Ok(())
}

fn decode_standard_unsigned_quad_scaled(
    reader: &mut BitReader<'_>,
    out: &mut [f32],
    lookup: &[QuadLookupEntry],
    max_bits: u8,
    scale: f32,
    pow43: &[f32],
) -> Result<()> {
    if out.len() % 4 != 0 {
        return Err(AacLcError::InvalidConfig(
            "spectral output length is not a multiple of tuple length",
        ));
    }

    for chunk in out.chunks_exact_mut(4) {
        let mut tuple = read_standard_quad_tuple(reader, lookup, max_bits)?;
        apply_unsigned_signs(reader, &mut tuple)?;
        write_scaled_tuple(&tuple, scale, pow43, chunk);
    }

    Ok(())
}

fn read_standard_quad_tuple(
    reader: &mut BitReader<'_>,
    lookup: &[QuadLookupEntry],
    max_bits: u8,
) -> Result<[i32; 4]> {
    let (lookahead, lookahead_bits) = reader.peek_prefix(max_bits)?;
    let index = (lookahead as usize) << (max_bits as usize - lookahead_bits as usize);
    let entry = lookup[index];

    if entry.bits != 0 && entry.bits <= lookahead_bits {
        reader.consume_cached_prefix(entry.bits);
        return Ok([
            entry.value[0] as i32,
            entry.value[1] as i32,
            entry.value[2] as i32,
            entry.value[3] as i32,
        ]);
    }

    Err(AacLcError::InvalidBitstream(
        "invalid AAC spectral Huffman codeword",
    ))
}

fn standard_quad_len(
    lengths: &[[[[u32; 3]; 3]; 3]; 3],
    half: LengthHalf,
    a: usize,
    b: usize,
    c: usize,
    d: usize,
) -> u8 {
    let packed = lengths[a][b][c][d];
    match half {
        LengthHalf::High => (packed >> 16) as u8,
        LengthHalf::Low => (packed & 0xffff) as u8,
    }
}

fn decode_standard_signed_pair(
    reader: &mut BitReader<'_>,
    out: &mut [i32],
    lookup: &[PairLookupEntry],
    max_bits: u8,
) -> Result<()> {
    if out.len() % 2 != 0 {
        return Err(AacLcError::InvalidConfig(
            "spectral output length is not even",
        ));
    }

    for chunk in out.chunks_exact_mut(2) {
        chunk.copy_from_slice(&read_standard_pair_tuple(reader, lookup, max_bits)?);
    }

    Ok(())
}

fn decode_standard_unsigned_pair(
    reader: &mut BitReader<'_>,
    out: &mut [i32],
    lookup: &[PairLookupEntry],
    max_bits: u8,
) -> Result<()> {
    if out.len() % 2 != 0 {
        return Err(AacLcError::InvalidConfig(
            "spectral output length is not even",
        ));
    }

    for chunk in out.chunks_exact_mut(2) {
        let mut tuple = read_standard_pair_tuple(reader, lookup, max_bits)?;
        for value in &mut tuple {
            if *value != 0 && reader.read_bool()? {
                *value = -*value;
            }
        }
        chunk.copy_from_slice(&tuple);
    }

    Ok(())
}

fn decode_standard_signed_pair_scaled(
    reader: &mut BitReader<'_>,
    out: &mut [f32],
    lookup: &[PairLookupEntry],
    max_bits: u8,
    scale: f32,
    pow43: &[f32],
) -> Result<()> {
    if out.len() % 2 != 0 {
        return Err(AacLcError::InvalidConfig(
            "spectral output length is not even",
        ));
    }

    for chunk in out.chunks_exact_mut(2) {
        let tuple = read_standard_pair_tuple(reader, lookup, max_bits)?;
        write_scaled_tuple(&tuple, scale, pow43, chunk);
    }

    Ok(())
}

fn decode_standard_unsigned_pair_scaled(
    reader: &mut BitReader<'_>,
    out: &mut [f32],
    lookup: &[PairLookupEntry],
    max_bits: u8,
    scale: f32,
    pow43: &[f32],
) -> Result<()> {
    if out.len() % 2 != 0 {
        return Err(AacLcError::InvalidConfig(
            "spectral output length is not even",
        ));
    }

    for chunk in out.chunks_exact_mut(2) {
        let mut tuple = read_standard_pair_tuple(reader, lookup, max_bits)?;
        apply_unsigned_signs(reader, &mut tuple)?;
        write_scaled_tuple(&tuple, scale, pow43, chunk);
    }

    Ok(())
}

fn read_standard_pair_tuple(
    reader: &mut BitReader<'_>,
    lookup: &[PairLookupEntry],
    max_bits: u8,
) -> Result<[i32; 2]> {
    let (lookahead, lookahead_bits) = reader.peek_prefix(max_bits)?;
    let index = (lookahead as usize) << (max_bits as usize - lookahead_bits as usize);
    let entry = lookup[index];

    if entry.bits != 0 && entry.bits <= lookahead_bits {
        reader.consume_cached_prefix(entry.bits);
        return Ok([entry.value[0] as i32, entry.value[1] as i32]);
    }

    Err(AacLcError::InvalidBitstream(
        "invalid AAC spectral Huffman codeword",
    ))
}

fn decode_standard_escape_pair_scaled(
    reader: &mut BitReader<'_>,
    out: &mut [f32],
    scale: f32,
    pow43: &[f32],
) -> Result<()> {
    if out.len() % 2 != 0 {
        return Err(AacLcError::InvalidConfig(
            "spectral output length is not even",
        ));
    }

    for chunk in out.chunks_exact_mut(2) {
        let mut tuple = read_standard_pair_tuple(
            reader,
            standard_escape_pair_lookup(),
            STANDARD_CODEBOOK_11_MAX_BITS,
        )?;
        finish_unsigned_escape_pair(reader, &mut tuple)?;
        write_scaled_tuple(&tuple, scale, pow43, chunk);
    }

    Ok(())
}

fn apply_unsigned_signs(reader: &mut BitReader<'_>, tuple: &mut [i32]) -> Result<()> {
    for value in tuple {
        if *value != 0 && reader.read_bool()? {
            *value = -*value;
        }
    }
    Ok(())
}

fn write_scaled_tuple(tuple: &[i32], scale: f32, pow43: &[f32], out: &mut [f32]) {
    for (sample, value) in out.iter_mut().zip(tuple.iter().copied()) {
        *sample = dequantize_signed_scaled(value, scale, pow43);
    }
}

fn standard_pair_len<const N: usize>(
    lengths: &[[u32; N]; N],
    half: LengthHalf,
    a: usize,
    b: usize,
) -> u8 {
    let packed = lengths[a][b];
    match half {
        LengthHalf::High => (packed >> 16) as u8,
        LengthHalf::Low => (packed & 0xffff) as u8,
    }
}

fn standard_quad_lookup(codebook: u8) -> &'static [QuadLookupEntry] {
    static CODEBOOK_1: OnceLock<Box<[QuadLookupEntry]>> = OnceLock::new();
    static CODEBOOK_2: OnceLock<Box<[QuadLookupEntry]>> = OnceLock::new();
    static CODEBOOK_3: OnceLock<Box<[QuadLookupEntry]>> = OnceLock::new();
    static CODEBOOK_4: OnceLock<Box<[QuadLookupEntry]>> = OnceLock::new();

    match codebook {
        1 => CODEBOOK_1
            .get_or_init(|| {
                build_quad_lookup(
                    &STANDARD_CODEBOOK_1_2_LENGTHS,
                    LengthHalf::High,
                    &STANDARD_CODEBOOK_1_CODES,
                    STANDARD_CODEBOOK_1_MAX_BITS,
                    -1,
                )
            })
            .as_ref(),
        2 => CODEBOOK_2
            .get_or_init(|| {
                build_quad_lookup(
                    &STANDARD_CODEBOOK_1_2_LENGTHS,
                    LengthHalf::Low,
                    &STANDARD_CODEBOOK_2_CODES,
                    STANDARD_CODEBOOK_2_MAX_BITS,
                    -1,
                )
            })
            .as_ref(),
        3 => CODEBOOK_3
            .get_or_init(|| {
                build_quad_lookup(
                    &STANDARD_CODEBOOK_3_4_LENGTHS,
                    LengthHalf::High,
                    &STANDARD_CODEBOOK_3_CODES,
                    STANDARD_CODEBOOK_3_MAX_BITS,
                    0,
                )
            })
            .as_ref(),
        4 => CODEBOOK_4
            .get_or_init(|| {
                build_quad_lookup(
                    &STANDARD_CODEBOOK_3_4_LENGTHS,
                    LengthHalf::Low,
                    &STANDARD_CODEBOOK_4_CODES,
                    STANDARD_CODEBOOK_4_MAX_BITS,
                    0,
                )
            })
            .as_ref(),
        _ => unreachable!("standard quad codebook id"),
    }
}

fn standard_pair_lookup(codebook: u8) -> &'static [PairLookupEntry] {
    static CODEBOOK_5: OnceLock<Box<[PairLookupEntry]>> = OnceLock::new();
    static CODEBOOK_6: OnceLock<Box<[PairLookupEntry]>> = OnceLock::new();
    static CODEBOOK_7: OnceLock<Box<[PairLookupEntry]>> = OnceLock::new();
    static CODEBOOK_8: OnceLock<Box<[PairLookupEntry]>> = OnceLock::new();
    static CODEBOOK_9: OnceLock<Box<[PairLookupEntry]>> = OnceLock::new();
    static CODEBOOK_10: OnceLock<Box<[PairLookupEntry]>> = OnceLock::new();

    match codebook {
        5 => CODEBOOK_5
            .get_or_init(|| {
                build_pair_lookup(
                    &STANDARD_CODEBOOK_5_6_LENGTHS,
                    LengthHalf::High,
                    &STANDARD_CODEBOOK_5_CODES,
                    STANDARD_CODEBOOK_5_MAX_BITS,
                    -4,
                )
            })
            .as_ref(),
        6 => CODEBOOK_6
            .get_or_init(|| {
                build_pair_lookup(
                    &STANDARD_CODEBOOK_5_6_LENGTHS,
                    LengthHalf::Low,
                    &STANDARD_CODEBOOK_6_CODES,
                    STANDARD_CODEBOOK_6_MAX_BITS,
                    -4,
                )
            })
            .as_ref(),
        7 => CODEBOOK_7
            .get_or_init(|| {
                build_pair_lookup(
                    &STANDARD_CODEBOOK_7_8_LENGTHS,
                    LengthHalf::High,
                    &STANDARD_CODEBOOK_7_CODES,
                    STANDARD_CODEBOOK_7_MAX_BITS,
                    0,
                )
            })
            .as_ref(),
        8 => CODEBOOK_8
            .get_or_init(|| {
                build_pair_lookup(
                    &STANDARD_CODEBOOK_7_8_LENGTHS,
                    LengthHalf::Low,
                    &STANDARD_CODEBOOK_8_CODES,
                    STANDARD_CODEBOOK_8_MAX_BITS,
                    0,
                )
            })
            .as_ref(),
        9 => CODEBOOK_9
            .get_or_init(|| {
                build_pair_lookup(
                    &STANDARD_CODEBOOK_9_10_LENGTHS,
                    LengthHalf::High,
                    &STANDARD_CODEBOOK_9_CODES,
                    STANDARD_CODEBOOK_9_MAX_BITS,
                    0,
                )
            })
            .as_ref(),
        10 => CODEBOOK_10
            .get_or_init(|| {
                build_pair_lookup(
                    &STANDARD_CODEBOOK_9_10_LENGTHS,
                    LengthHalf::Low,
                    &STANDARD_CODEBOOK_10_CODES,
                    STANDARD_CODEBOOK_10_MAX_BITS,
                    0,
                )
            })
            .as_ref(),
        _ => unreachable!("standard pair codebook id"),
    }
}

fn standard_escape_pair_lookup() -> &'static [PairLookupEntry] {
    static CODEBOOK_11: OnceLock<Box<[PairLookupEntry]>> = OnceLock::new();
    CODEBOOK_11
        .get_or_init(|| build_escape_pair_lookup(STANDARD_CODEBOOK_11_MAX_BITS))
        .as_ref()
}

fn build_quad_lookup(
    lengths: &[[[[u32; 3]; 3]; 3]; 3],
    half: LengthHalf,
    codes: &[[[[u16; 3]; 3]; 3]; 3],
    max_bits: u8,
    value_offset: i16,
) -> Box<[QuadLookupEntry]> {
    let mut table = vec![QuadLookupEntry::default(); 1usize << max_bits];

    for a in 0..3 {
        for b in 0..3 {
            for c in 0..3 {
                for d in 0..3 {
                    let bits = standard_quad_len(lengths, half, a, b, c, d);
                    if bits == 0 {
                        continue;
                    }
                    let value = [
                        a as i16 + value_offset,
                        b as i16 + value_offset,
                        c as i16 + value_offset,
                        d as i16 + value_offset,
                    ];
                    fill_quad_lookup(&mut table, max_bits, codes[a][b][c][d], bits, value);
                }
            }
        }
    }

    table.into_boxed_slice()
}

fn build_pair_lookup<const N: usize>(
    lengths: &[[u32; N]; N],
    half: LengthHalf,
    codes: &[[u16; N]; N],
    max_bits: u8,
    value_offset: i16,
) -> Box<[PairLookupEntry]> {
    let mut table = vec![PairLookupEntry::default(); 1usize << max_bits];

    for a in 0..N {
        for b in 0..N {
            let bits = standard_pair_len(lengths, half, a, b);
            if bits == 0 {
                continue;
            }
            fill_pair_lookup(
                &mut table,
                max_bits,
                codes[a][b],
                bits,
                [a as i16 + value_offset, b as i16 + value_offset],
            );
        }
    }

    table.into_boxed_slice()
}

fn build_escape_pair_lookup(max_bits: u8) -> Box<[PairLookupEntry]> {
    let mut table = vec![PairLookupEntry::default(); 1usize << max_bits];

    for a in 0..17 {
        for b in 0..17 {
            let bits = STANDARD_CODEBOOK_11_LENGTHS[a][b];
            if bits == 0 {
                continue;
            }
            fill_pair_lookup(
                &mut table,
                max_bits,
                STANDARD_CODEBOOK_11_CODES[a][b],
                bits,
                [a as i16, b as i16],
            );
        }
    }

    table.into_boxed_slice()
}

fn fill_quad_lookup(
    table: &mut [QuadLookupEntry],
    max_bits: u8,
    code: u16,
    bits: u8,
    value: [i16; 4],
) {
    let prefix = (code as usize) << (max_bits - bits);
    let slots = 1usize << (max_bits - bits);
    for entry in &mut table[prefix..prefix + slots] {
        debug_assert_eq!(entry.bits, 0);
        *entry = QuadLookupEntry { bits, value };
    }
}

fn fill_pair_lookup(
    table: &mut [PairLookupEntry],
    max_bits: u8,
    code: u16,
    bits: u8,
    value: [i16; 2],
) {
    let prefix = (code as usize) << (max_bits - bits);
    let slots = 1usize << (max_bits - bits);
    for entry in &mut table[prefix..prefix + slots] {
        debug_assert_eq!(entry.bits, 0);
        *entry = PairLookupEntry { bits, value };
    }
}

fn decode_standard_escape_pair(reader: &mut BitReader<'_>, out: &mut [i32]) -> Result<()> {
    if out.len() % 2 != 0 {
        return Err(AacLcError::InvalidConfig(
            "spectral output length is not even",
        ));
    }

    for chunk in out.chunks_exact_mut(2) {
        let mut tuple = read_standard_pair_tuple(
            reader,
            standard_escape_pair_lookup(),
            STANDARD_CODEBOOK_11_MAX_BITS,
        )?;
        finish_unsigned_escape_pair(reader, &mut tuple)?;
        chunk.copy_from_slice(&tuple);
    }

    Ok(())
}

const STANDARD_CODEBOOK_1_MAX_BITS: u8 = 11;
const STANDARD_CODEBOOK_2_MAX_BITS: u8 = 9;
const STANDARD_CODEBOOK_3_MAX_BITS: u8 = 16;
const STANDARD_CODEBOOK_4_MAX_BITS: u8 = 12;
const STANDARD_CODEBOOK_5_MAX_BITS: u8 = 13;
const STANDARD_CODEBOOK_6_MAX_BITS: u8 = 11;
const STANDARD_CODEBOOK_7_MAX_BITS: u8 = 12;
const STANDARD_CODEBOOK_8_MAX_BITS: u8 = 10;
const STANDARD_CODEBOOK_9_MAX_BITS: u8 = 15;
const STANDARD_CODEBOOK_10_MAX_BITS: u8 = 12;
const STANDARD_CODEBOOK_11_MAX_BITS: u8 = 12;

const STANDARD_CODEBOOK_1_2_LENGTHS: [[[[u32; 3]; 3]; 3]; 3] = [
    [
        [
            [0x000b0009, 0x00090007, 0x000b0009],
            [0x000a0008, 0x00070006, 0x000a0008],
            [0x000b0009, 0x00090008, 0x000b0009],
        ],
        [
            [0x000a0008, 0x00070006, 0x000a0007],
            [0x00070006, 0x00050005, 0x00070006],
            [0x00090007, 0x00070006, 0x000a0008],
        ],
        [
            [0x000b0009, 0x00090007, 0x000b0008],
            [0x00090008, 0x00070006, 0x00090008],
            [0x000b0009, 0x00090007, 0x000b0009],
        ],
    ],
    [
        [
            [0x00090008, 0x00070006, 0x00090007],
            [0x00070006, 0x00050005, 0x00070006],
            [0x00090007, 0x00070006, 0x00090008],
        ],
        [
            [0x00070006, 0x00050005, 0x00070006],
            [0x00050005, 0x00010003, 0x00050005],
            [0x00070006, 0x00050005, 0x00070006],
        ],
        [
            [0x00090008, 0x00070006, 0x00090007],
            [0x00070006, 0x00050005, 0x00070006],
            [0x00090008, 0x00070006, 0x00090008],
        ],
    ],
    [
        [
            [0x000b0009, 0x00090007, 0x000b0009],
            [0x00090008, 0x00070006, 0x00090008],
            [0x000b0008, 0x00090007, 0x000b0009],
        ],
        [
            [0x000a0008, 0x00070006, 0x00090007],
            [0x00070006, 0x00050004, 0x00070006],
            [0x00090008, 0x00070006, 0x000a0007],
        ],
        [
            [0x000b0009, 0x00090007, 0x000b0009],
            [0x000a0007, 0x00070006, 0x00090008],
            [0x000b0009, 0x00090007, 0x000b0009],
        ],
    ],
];

const STANDARD_CODEBOOK_1_CODES: [[[[u16; 3]; 3]; 3]; 3] = [
    [
        [
            [0x07f8, 0x01f1, 0x07fd],
            [0x03f5, 0x0068, 0x03f0],
            [0x07f7, 0x01ec, 0x07f5],
        ],
        [
            [0x03f1, 0x0072, 0x03f4],
            [0x0074, 0x0011, 0x0076],
            [0x01eb, 0x006c, 0x03f6],
        ],
        [
            [0x07fc, 0x01e1, 0x07f1],
            [0x01f0, 0x0061, 0x01f6],
            [0x07f2, 0x01ea, 0x07fb],
        ],
    ],
    [
        [
            [0x01f2, 0x0069, 0x01ed],
            [0x0077, 0x0017, 0x006f],
            [0x01e6, 0x0064, 0x01e5],
        ],
        [
            [0x0067, 0x0015, 0x0062],
            [0x0012, 0x0000, 0x0014],
            [0x0065, 0x0016, 0x006d],
        ],
        [
            [0x01e9, 0x0063, 0x01e4],
            [0x006b, 0x0013, 0x0071],
            [0x01e3, 0x0070, 0x01f3],
        ],
    ],
    [
        [
            [0x07fe, 0x01e7, 0x07f3],
            [0x01ef, 0x0060, 0x01ee],
            [0x07f0, 0x01e2, 0x07fa],
        ],
        [
            [0x03f3, 0x006a, 0x01e8],
            [0x0075, 0x0010, 0x0073],
            [0x01f4, 0x006e, 0x03f7],
        ],
        [
            [0x07f6, 0x01e0, 0x07f9],
            [0x03f2, 0x0066, 0x01f5],
            [0x07ff, 0x01f7, 0x07f4],
        ],
    ],
];

const STANDARD_CODEBOOK_2_CODES: [[[[u16; 3]; 3]; 3]; 3] = [
    [
        [
            [0x01f3, 0x006f, 0x01fd],
            [0x00eb, 0x0023, 0x00ea],
            [0x01f7, 0x00e8, 0x01fa],
        ],
        [
            [0x00f2, 0x002d, 0x0070],
            [0x0020, 0x0006, 0x002b],
            [0x006e, 0x0028, 0x00e9],
        ],
        [
            [0x01f9, 0x0066, 0x00f8],
            [0x00e7, 0x001b, 0x00f1],
            [0x01f4, 0x006b, 0x01f5],
        ],
    ],
    [
        [
            [0x00ec, 0x002a, 0x006c],
            [0x002c, 0x000a, 0x0027],
            [0x0067, 0x001a, 0x00f5],
        ],
        [
            [0x0024, 0x0008, 0x001f],
            [0x0009, 0x0000, 0x0007],
            [0x001d, 0x000b, 0x0030],
        ],
        [
            [0x00ef, 0x001c, 0x0064],
            [0x001e, 0x000c, 0x0029],
            [0x00f3, 0x002f, 0x00f0],
        ],
    ],
    [
        [
            [0x01fc, 0x0071, 0x01f2],
            [0x00f4, 0x0021, 0x00e6],
            [0x00f7, 0x0068, 0x01f8],
        ],
        [
            [0x00ee, 0x0022, 0x0065],
            [0x0031, 0x0002, 0x0026],
            [0x00ed, 0x0025, 0x006a],
        ],
        [
            [0x01fb, 0x0072, 0x01fe],
            [0x0069, 0x002e, 0x00f6],
            [0x01ff, 0x006d, 0x01f6],
        ],
    ],
];

const STANDARD_CODEBOOK_3_4_LENGTHS: [[[[u32; 3]; 3]; 3]; 3] = [
    [
        [
            [0x00010004, 0x00040005, 0x00080008],
            [0x00040005, 0x00050004, 0x00080008],
            [0x00090009, 0x00090008, 0x000a000b],
        ],
        [
            [0x00040005, 0x00060005, 0x00090008],
            [0x00060005, 0x00060004, 0x00090008],
            [0x00090008, 0x00090007, 0x000a000a],
        ],
        [
            [0x00090009, 0x000a0008, 0x000d000b],
            [0x00090008, 0x00090008, 0x000b000a],
            [0x000b000b, 0x000a000a, 0x000c000b],
        ],
    ],
    [
        [
            [0x00040004, 0x00060005, 0x000a0008],
            [0x00060004, 0x00070004, 0x000a0008],
            [0x000a0008, 0x000a0008, 0x000c000a],
        ],
        [
            [0x00050004, 0x00070004, 0x000b0008],
            [0x00060004, 0x00070004, 0x000a0007],
            [0x00090008, 0x00090007, 0x000b0009],
        ],
        [
            [0x00090008, 0x000a0008, 0x000d000a],
            [0x00080007, 0x00090007, 0x000c0009],
            [0x000a000a, 0x000b0009, 0x000c000a],
        ],
    ],
    [
        [
            [0x00080008, 0x000a0008, 0x000f000b],
            [0x00090008, 0x000b0007, 0x000f000a],
            [0x000d000b, 0x000e000a, 0x0010000c],
        ],
        [
            [0x00080008, 0x000a0007, 0x000e000a],
            [0x00090007, 0x000a0007, 0x000e0009],
            [0x000c000a, 0x000c0009, 0x000f000b],
        ],
        [
            [0x000b000b, 0x000c000a, 0x0010000c],
            [0x000a000a, 0x000b0009, 0x000f000b],
            [0x000c000b, 0x000c000a, 0x000f000b],
        ],
    ],
];

const STANDARD_CODEBOOK_3_CODES: [[[[u16; 3]; 3]; 3]; 3] = [
    [
        [
            [0x0000, 0x0009, 0x00ef],
            [0x000b, 0x0019, 0x00f0],
            [0x01eb, 0x01e6, 0x03f2],
        ],
        [
            [0x000a, 0x0035, 0x01ef],
            [0x0034, 0x0037, 0x01e9],
            [0x01ed, 0x01e7, 0x03f3],
        ],
        [
            [0x01ee, 0x03ed, 0x1ffa],
            [0x01ec, 0x01f2, 0x07f9],
            [0x07f8, 0x03f8, 0x0ff8],
        ],
    ],
    [
        [
            [0x0008, 0x0038, 0x03f6],
            [0x0036, 0x0075, 0x03f1],
            [0x03eb, 0x03ec, 0x0ff4],
        ],
        [
            [0x0018, 0x0076, 0x07f4],
            [0x0039, 0x0074, 0x03ef],
            [0x01f3, 0x01f4, 0x07f6],
        ],
        [
            [0x01e8, 0x03ea, 0x1ffc],
            [0x00f2, 0x01f1, 0x0ffb],
            [0x03f5, 0x07f3, 0x0ffc],
        ],
    ],
    [
        [
            [0x00ee, 0x03f7, 0x7ffe],
            [0x01f0, 0x07f5, 0x7ffd],
            [0x1ffb, 0x3ffa, 0xffff],
        ],
        [
            [0x00f1, 0x03f0, 0x3ffc],
            [0x01ea, 0x03ee, 0x3ffb],
            [0x0ff6, 0x0ffa, 0x7ffc],
        ],
        [
            [0x07f2, 0x0ff5, 0xfffe],
            [0x03f4, 0x07f7, 0x7ffb],
            [0x0ff7, 0x0ff9, 0x7ffa],
        ],
    ],
];

const STANDARD_CODEBOOK_4_CODES: [[[[u16; 3]; 3]; 3]; 3] = [
    [
        [
            [0x0007, 0x0016, 0x00f6],
            [0x0018, 0x0008, 0x00ef],
            [0x01ef, 0x00f3, 0x07f8],
        ],
        [
            [0x0019, 0x0017, 0x00ed],
            [0x0015, 0x0001, 0x00e2],
            [0x00f0, 0x0070, 0x03f0],
        ],
        [
            [0x01ee, 0x00f1, 0x07fa],
            [0x00ee, 0x00e4, 0x03f2],
            [0x07f6, 0x03ef, 0x07fd],
        ],
    ],
    [
        [
            [0x0005, 0x0014, 0x00f2],
            [0x0009, 0x0004, 0x00e5],
            [0x00f4, 0x00e8, 0x03f4],
        ],
        [
            [0x0006, 0x0002, 0x00e7],
            [0x0003, 0x0000, 0x006b],
            [0x00e3, 0x0069, 0x01f3],
        ],
        [
            [0x00eb, 0x00e6, 0x03f6],
            [0x006e, 0x006a, 0x01f4],
            [0x03ec, 0x01f0, 0x03f9],
        ],
    ],
    [
        [
            [0x00f5, 0x00ec, 0x07fb],
            [0x00ea, 0x006f, 0x03f7],
            [0x07f9, 0x03f3, 0x0fff],
        ],
        [
            [0x00e9, 0x006d, 0x03f8],
            [0x006c, 0x0068, 0x01f5],
            [0x03ee, 0x01f2, 0x07f4],
        ],
        [
            [0x07f7, 0x03f1, 0x0ffe],
            [0x03ed, 0x01f1, 0x07f5],
            [0x07fe, 0x03f5, 0x07fc],
        ],
    ],
];

const STANDARD_CODEBOOK_5_6_LENGTHS: [[u32; 9]; 9] = [
    [
        0x000d000b, 0x000c000a, 0x000b0009, 0x000b0009, 0x000a0009, 0x000b0009, 0x000b0009,
        0x000c000a, 0x000d000b,
    ],
    [
        0x000c000a, 0x000b0009, 0x000a0008, 0x00090007, 0x00080007, 0x00090007, 0x000a0008,
        0x000b0009, 0x000c000a,
    ],
    [
        0x000c0009, 0x000a0008, 0x00090006, 0x00080006, 0x00070006, 0x00080006, 0x00090006,
        0x000a0008, 0x000b0009,
    ],
    [
        0x000b0009, 0x00090007, 0x00080006, 0x00050004, 0x00040004, 0x00050004, 0x00080006,
        0x00090007, 0x000b0009,
    ],
    [
        0x000a0009, 0x00080007, 0x00070006, 0x00040004, 0x00010004, 0x00040004, 0x00070006,
        0x00080007, 0x000b0009,
    ],
    [
        0x000b0009, 0x00090007, 0x00080006, 0x00050004, 0x00040004, 0x00050004, 0x00080006,
        0x00090007, 0x000b0009,
    ],
    [
        0x000b0009, 0x000a0008, 0x00090006, 0x00080006, 0x00070006, 0x00080006, 0x00090006,
        0x000a0008, 0x000b0009,
    ],
    [
        0x000c000a, 0x000b0009, 0x000a0008, 0x00090007, 0x00080007, 0x00090007, 0x000a0007,
        0x000b0008, 0x000c000a,
    ],
    [
        0x000d000b, 0x000c000a, 0x000c0009, 0x000b0009, 0x000a0009, 0x000a0009, 0x000b0009,
        0x000c000a, 0x000d000b,
    ],
];

const STANDARD_CODEBOOK_5_CODES: [[u16; 9]; 9] = [
    [
        0x1fff, 0x0ff7, 0x07f4, 0x07e8, 0x03f1, 0x07ee, 0x07f9, 0x0ff8, 0x1ffd,
    ],
    [
        0x0ffd, 0x07f1, 0x03e8, 0x01e8, 0x00f0, 0x01ec, 0x03ee, 0x07f2, 0x0ffa,
    ],
    [
        0x0ff4, 0x03ef, 0x01f2, 0x00e8, 0x0070, 0x00ec, 0x01f0, 0x03ea, 0x07f3,
    ],
    [
        0x07eb, 0x01eb, 0x00ea, 0x001a, 0x0008, 0x0019, 0x00ee, 0x01ef, 0x07ed,
    ],
    [
        0x03f0, 0x00f2, 0x0073, 0x000b, 0x0000, 0x000a, 0x0071, 0x00f3, 0x07e9,
    ],
    [
        0x07ef, 0x01ee, 0x00ef, 0x0018, 0x0009, 0x001b, 0x00eb, 0x01e9, 0x07ec,
    ],
    [
        0x07f6, 0x03eb, 0x01f3, 0x00ed, 0x0072, 0x00e9, 0x01f1, 0x03ed, 0x07f7,
    ],
    [
        0x0ff6, 0x07f0, 0x03e9, 0x01ed, 0x00f1, 0x01ea, 0x03ec, 0x07f8, 0x0ff9,
    ],
    [
        0x1ffc, 0x0ffc, 0x0ff5, 0x07ea, 0x03f3, 0x03f2, 0x07f5, 0x0ffb, 0x1ffe,
    ],
];

const STANDARD_CODEBOOK_6_CODES: [[u16; 9]; 9] = [
    [
        0x07fe, 0x03fd, 0x01f1, 0x01eb, 0x01f4, 0x01ea, 0x01f0, 0x03fc, 0x07fd,
    ],
    [
        0x03f6, 0x01e5, 0x00ea, 0x006c, 0x0071, 0x0068, 0x00f0, 0x01e6, 0x03f7,
    ],
    [
        0x01f3, 0x00ef, 0x0032, 0x0027, 0x0028, 0x0026, 0x0031, 0x00eb, 0x01f7,
    ],
    [
        0x01e8, 0x006f, 0x002e, 0x0008, 0x0004, 0x0006, 0x0029, 0x006b, 0x01ee,
    ],
    [
        0x01ef, 0x0072, 0x002d, 0x0002, 0x0000, 0x0003, 0x002f, 0x0073, 0x01fa,
    ],
    [
        0x01e7, 0x006e, 0x002b, 0x0007, 0x0001, 0x0005, 0x002c, 0x006d, 0x01ec,
    ],
    [
        0x01f9, 0x00ee, 0x0030, 0x0024, 0x002a, 0x0025, 0x0033, 0x00ec, 0x01f2,
    ],
    [
        0x03f8, 0x01e4, 0x00ed, 0x006a, 0x0070, 0x0069, 0x0074, 0x00f1, 0x03fa,
    ],
    [
        0x07ff, 0x03f9, 0x01f6, 0x01ed, 0x01f8, 0x01e9, 0x01f5, 0x03fb, 0x07fc,
    ],
];

const STANDARD_CODEBOOK_7_8_LENGTHS: [[u32; 8]; 8] = [
    [
        0x00010005, 0x00030004, 0x00060005, 0x00070006, 0x00080007, 0x00090008, 0x000a0009,
        0x000b000a,
    ],
    [
        0x00030004, 0x00040003, 0x00060004, 0x00070005, 0x00080006, 0x00080007, 0x00090007,
        0x00090008,
    ],
    [
        0x00060005, 0x00060004, 0x00070004, 0x00080005, 0x00080006, 0x00090007, 0x00090007,
        0x000a0008,
    ],
    [
        0x00070006, 0x00070005, 0x00080005, 0x00080006, 0x00090006, 0x00090007, 0x000a0008,
        0x000a0008,
    ],
    [
        0x00080007, 0x00080006, 0x00090006, 0x00090006, 0x000a0007, 0x000a0007, 0x000a0008,
        0x000b0009,
    ],
    [
        0x00090008, 0x00080007, 0x00090006, 0x00090007, 0x000a0007, 0x000a0008, 0x000b0008,
        0x000b000a,
    ],
    [
        0x000a0009, 0x00090007, 0x00090007, 0x000a0008, 0x000a0008, 0x000b0008, 0x000c0009,
        0x000c0009,
    ],
    [
        0x000b000a, 0x000a0008, 0x000a0008, 0x000a0008, 0x000b0009, 0x000b0009, 0x000c0009,
        0x000c000a,
    ],
];

const STANDARD_CODEBOOK_7_CODES: [[u16; 8]; 8] = [
    [
        0x0000, 0x0005, 0x0037, 0x0074, 0x00f2, 0x01eb, 0x03ed, 0x07f7,
    ],
    [
        0x0004, 0x000c, 0x0035, 0x0071, 0x00ec, 0x00ee, 0x01ee, 0x01f5,
    ],
    [
        0x0036, 0x0034, 0x0072, 0x00ea, 0x00f1, 0x01e9, 0x01f3, 0x03f5,
    ],
    [
        0x0073, 0x0070, 0x00eb, 0x00f0, 0x01f1, 0x01f0, 0x03ec, 0x03fa,
    ],
    [
        0x00f3, 0x00ed, 0x01e8, 0x01ef, 0x03ef, 0x03f1, 0x03f9, 0x07fb,
    ],
    [
        0x01ed, 0x00ef, 0x01ea, 0x01f2, 0x03f3, 0x03f8, 0x07f9, 0x07fc,
    ],
    [
        0x03ee, 0x01ec, 0x01f4, 0x03f4, 0x03f7, 0x07f8, 0x0ffd, 0x0ffe,
    ],
    [
        0x07f6, 0x03f0, 0x03f2, 0x03f6, 0x07fa, 0x07fd, 0x0ffc, 0x0fff,
    ],
];

const STANDARD_CODEBOOK_8_CODES: [[u16; 8]; 8] = [
    [
        0x000e, 0x0005, 0x0010, 0x0030, 0x006f, 0x00f1, 0x01fa, 0x03fe,
    ],
    [
        0x0003, 0x0000, 0x0004, 0x0012, 0x002c, 0x006a, 0x0075, 0x00f8,
    ],
    [
        0x000f, 0x0002, 0x0006, 0x0014, 0x002e, 0x0069, 0x0072, 0x00f5,
    ],
    [
        0x002f, 0x0011, 0x0013, 0x002a, 0x0032, 0x006c, 0x00ec, 0x00fa,
    ],
    [
        0x0071, 0x002b, 0x002d, 0x0031, 0x006d, 0x0070, 0x00f2, 0x01f9,
    ],
    [
        0x00ef, 0x0068, 0x0033, 0x006b, 0x006e, 0x00ee, 0x00f9, 0x03fc,
    ],
    [
        0x01f8, 0x0074, 0x0073, 0x00ed, 0x00f0, 0x00f6, 0x01f6, 0x01fd,
    ],
    [
        0x03fd, 0x00f3, 0x00f4, 0x00f7, 0x01f7, 0x01fb, 0x01fc, 0x03ff,
    ],
];

const STANDARD_CODEBOOK_9_10_LENGTHS: [[u32; 13]; 13] = [
    [
        0x00010006, 0x00030005, 0x00060006, 0x00080006, 0x00090007, 0x000a0008, 0x000a0009,
        0x000b000a, 0x000b000a, 0x000c000a, 0x000c000b, 0x000d000b, 0x000d000c,
    ],
    [
        0x00030005, 0x00040004, 0x00060004, 0x00070005, 0x00080006, 0x00080007, 0x00090007,
        0x000a0008, 0x000a0008, 0x000a0009, 0x000b000a, 0x000c000a, 0x000c000b,
    ],
    [
        0x00060006, 0x00060004, 0x00070005, 0x00080005, 0x00080006, 0x00090006, 0x000a0007,
        0x000a0008, 0x000a0008, 0x000b0009, 0x000c0009, 0x000c000a, 0x000c000a,
    ],
    [
        0x00080006, 0x00070005, 0x00080005, 0x00090005, 0x00090006, 0x000a0007, 0x000a0007,
        0x000b0008, 0x000b0008, 0x000b0009, 0x000c0009, 0x000c000a, 0x000d000a,
    ],
    [
        0x00090007, 0x00080006, 0x00090006, 0x00090006, 0x000a0006, 0x000a0007, 0x000b0007,
        0x000b0008, 0x000b0008, 0x000c0009, 0x000c0009, 0x000c000a, 0x000d000a,
    ],
    [
        0x000a0008, 0x00090007, 0x00090006, 0x000a0007, 0x000b0007, 0x000b0007, 0x000b0008,
        0x000c0008, 0x000b0008, 0x000c0009, 0x000c000a, 0x000d000a, 0x000d000b,
    ],
    [
        0x000b0009, 0x00090007, 0x000a0007, 0x000b0007, 0x000b0007, 0x000b0008, 0x000c0008,
        0x000c0009, 0x000c0009, 0x000c0009, 0x000d000a, 0x000d000a, 0x000d000b,
    ],
    [
        0x000b0009, 0x000a0008, 0x000a0008, 0x000b0008, 0x000b0008, 0x000c0008, 0x000c0009,
        0x000d0009, 0x000d0009, 0x000d000a, 0x000d000a, 0x000d000b, 0x000d000b,
    ],
    [
        0x000b0009, 0x000a0008, 0x000a0008, 0x000b0008, 0x000b0008, 0x000b0008, 0x000c0009,
        0x000c0009, 0x000d000a, 0x000d000a, 0x000e000a, 0x000d000b, 0x000e000b,
    ],
    [
        0x000b000a, 0x000a0009, 0x000b0009, 0x000b0009, 0x000c0009, 0x000c0009, 0x000c0009,
        0x000c000a, 0x000d000a, 0x000d000a, 0x000e000b, 0x000e000b, 0x000e000c,
    ],
    [
        0x000c000a, 0x000b0009, 0x000b0009, 0x000c0009, 0x000c0009, 0x000c000a, 0x000d000a,
        0x000d000a, 0x000d000a, 0x000e000b, 0x000e000b, 0x000e000b, 0x000f000c,
    ],
    [
        0x000c000b, 0x000b000a, 0x000c0009, 0x000c000a, 0x000c000a, 0x000d000a, 0x000d000a,
        0x000d000a, 0x000d000b, 0x000e000b, 0x000e000b, 0x000f000b, 0x000f000c,
    ],
    [
        0x000d000b, 0x000c000a, 0x000c000a, 0x000c000a, 0x000d000a, 0x000d000a, 0x000d000a,
        0x000d000b, 0x000e000b, 0x000e000c, 0x000e000c, 0x000e000c, 0x000f000c,
    ],
];

const STANDARD_CODEBOOK_9_CODES: [[u16; 13]; 13] = [
    [
        0x0000, 0x0005, 0x0037, 0x00e7, 0x01de, 0x03ce, 0x03d9, 0x07c8, 0x07cd, 0x0fc8, 0x0fdd,
        0x1fe4, 0x1fec,
    ],
    [
        0x0004, 0x000c, 0x0035, 0x0072, 0x00ea, 0x00ed, 0x01e2, 0x03d1, 0x03d3, 0x03e0, 0x07d8,
        0x0fcf, 0x0fd5,
    ],
    [
        0x0036, 0x0034, 0x0071, 0x00e8, 0x00ec, 0x01e1, 0x03cf, 0x03dd, 0x03db, 0x07d0, 0x0fc7,
        0x0fd4, 0x0fe4,
    ],
    [
        0x00e6, 0x0070, 0x00e9, 0x01dd, 0x01e3, 0x03d2, 0x03dc, 0x07cc, 0x07ca, 0x07de, 0x0fd8,
        0x0fea, 0x1fdb,
    ],
    [
        0x01df, 0x00eb, 0x01dc, 0x01e6, 0x03d5, 0x03de, 0x07cb, 0x07dd, 0x07dc, 0x0fcd, 0x0fe2,
        0x0fe7, 0x1fe1,
    ],
    [
        0x03d0, 0x01e0, 0x01e4, 0x03d6, 0x07c5, 0x07d1, 0x07db, 0x0fd2, 0x07e0, 0x0fd9, 0x0feb,
        0x1fe3, 0x1fe9,
    ],
    [
        0x07c4, 0x01e5, 0x03d7, 0x07c6, 0x07cf, 0x07da, 0x0fcb, 0x0fda, 0x0fe3, 0x0fe9, 0x1fe6,
        0x1ff3, 0x1ff7,
    ],
    [
        0x07d3, 0x03d8, 0x03e1, 0x07d4, 0x07d9, 0x0fd3, 0x0fde, 0x1fdd, 0x1fd9, 0x1fe2, 0x1fea,
        0x1ff1, 0x1ff6,
    ],
    [
        0x07d2, 0x03d4, 0x03da, 0x07c7, 0x07d7, 0x07e2, 0x0fce, 0x0fdb, 0x1fd8, 0x1fee, 0x3ff0,
        0x1ff4, 0x3ff2,
    ],
    [
        0x07e1, 0x03df, 0x07c9, 0x07d6, 0x0fca, 0x0fd0, 0x0fe5, 0x0fe6, 0x1feb, 0x1fef, 0x3ff3,
        0x3ff4, 0x3ff5,
    ],
    [
        0x0fe0, 0x07ce, 0x07d5, 0x0fc6, 0x0fd1, 0x0fe1, 0x1fe0, 0x1fe8, 0x1ff0, 0x3ff1, 0x3ff8,
        0x3ff6, 0x7ffc,
    ],
    [
        0x0fe8, 0x07df, 0x0fc9, 0x0fd7, 0x0fdc, 0x1fdc, 0x1fdf, 0x1fed, 0x1ff5, 0x3ff9, 0x3ffb,
        0x7ffd, 0x7ffe,
    ],
    [
        0x1fe7, 0x0fcc, 0x0fd6, 0x0fdf, 0x1fde, 0x1fda, 0x1fe5, 0x1ff2, 0x3ffa, 0x3ff7, 0x3ffc,
        0x3ffd, 0x7fff,
    ],
];

const STANDARD_CODEBOOK_10_CODES: [[u16; 13]; 13] = [
    [
        0x0022, 0x0008, 0x001d, 0x0026, 0x005f, 0x00d3, 0x01cf, 0x03d0, 0x03d7, 0x03ed, 0x07f0,
        0x07f6, 0x0ffd,
    ],
    [
        0x0007, 0x0000, 0x0001, 0x0009, 0x0020, 0x0054, 0x0060, 0x00d5, 0x00dc, 0x01d4, 0x03cd,
        0x03de, 0x07e7,
    ],
    [
        0x001c, 0x0002, 0x0006, 0x000c, 0x001e, 0x0028, 0x005b, 0x00cd, 0x00d9, 0x01ce, 0x01dc,
        0x03d9, 0x03f1,
    ],
    [
        0x0025, 0x000b, 0x000a, 0x000d, 0x0024, 0x0057, 0x0061, 0x00cc, 0x00dd, 0x01cc, 0x01de,
        0x03d3, 0x03e7,
    ],
    [
        0x005d, 0x0021, 0x001f, 0x0023, 0x0027, 0x0059, 0x0064, 0x00d8, 0x00df, 0x01d2, 0x01e2,
        0x03dd, 0x03ee,
    ],
    [
        0x00d1, 0x0055, 0x0029, 0x0056, 0x0058, 0x0062, 0x00ce, 0x00e0, 0x00e2, 0x01da, 0x03d4,
        0x03e3, 0x07eb,
    ],
    [
        0x01c9, 0x005e, 0x005a, 0x005c, 0x0063, 0x00ca, 0x00da, 0x01c7, 0x01ca, 0x01e0, 0x03db,
        0x03e8, 0x07ec,
    ],
    [
        0x01e3, 0x00d2, 0x00cb, 0x00d0, 0x00d7, 0x00db, 0x01c6, 0x01d5, 0x01d8, 0x03ca, 0x03da,
        0x07ea, 0x07f1,
    ],
    [
        0x01e1, 0x00d4, 0x00cf, 0x00d6, 0x00de, 0x00e1, 0x01d0, 0x01d6, 0x03d1, 0x03d5, 0x03f2,
        0x07ee, 0x07fb,
    ],
    [
        0x03e9, 0x01cd, 0x01c8, 0x01cb, 0x01d1, 0x01d7, 0x01df, 0x03cf, 0x03e0, 0x03ef, 0x07e6,
        0x07f8, 0x0ffa,
    ],
    [
        0x03eb, 0x01dd, 0x01d3, 0x01d9, 0x01db, 0x03d2, 0x03cc, 0x03dc, 0x03ea, 0x07ed, 0x07f3,
        0x07f9, 0x0ff9,
    ],
    [
        0x07f2, 0x03ce, 0x01e4, 0x03cb, 0x03d8, 0x03d6, 0x03e2, 0x03e5, 0x07e8, 0x07f4, 0x07f5,
        0x07f7, 0x0ffb,
    ],
    [
        0x07fa, 0x03ec, 0x03df, 0x03e1, 0x03e4, 0x03e6, 0x03f0, 0x07e9, 0x07ef, 0x0ff8, 0x0ffe,
        0x0ffc, 0x0fff,
    ],
];

const STANDARD_CODEBOOK_11_LENGTHS: [[u8; 17]; 17] = [
    [
        0x04, 0x05, 0x06, 0x07, 0x08, 0x08, 0x09, 0x0a, 0x0a, 0x0a, 0x0b, 0x0b, 0x0c, 0x0b, 0x0c,
        0x0c, 0x0a,
    ],
    [
        0x05, 0x04, 0x05, 0x06, 0x07, 0x07, 0x08, 0x08, 0x09, 0x09, 0x09, 0x0a, 0x0a, 0x0a, 0x0a,
        0x0b, 0x08,
    ],
    [
        0x06, 0x05, 0x05, 0x06, 0x07, 0x07, 0x08, 0x08, 0x08, 0x09, 0x09, 0x09, 0x0a, 0x0a, 0x0a,
        0x0a, 0x08,
    ],
    [
        0x07, 0x06, 0x06, 0x06, 0x07, 0x07, 0x08, 0x08, 0x08, 0x09, 0x09, 0x09, 0x0a, 0x0a, 0x0a,
        0x0a, 0x08,
    ],
    [
        0x08, 0x07, 0x07, 0x07, 0x07, 0x08, 0x08, 0x08, 0x08, 0x09, 0x09, 0x09, 0x0a, 0x0a, 0x0a,
        0x0a, 0x08,
    ],
    [
        0x08, 0x07, 0x07, 0x07, 0x07, 0x08, 0x08, 0x08, 0x09, 0x09, 0x09, 0x09, 0x0a, 0x0a, 0x0a,
        0x0a, 0x08,
    ],
    [
        0x09, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x09, 0x09, 0x09, 0x0a, 0x0a, 0x0a, 0x0a,
        0x0a, 0x08,
    ],
    [
        0x09, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x09, 0x09, 0x09, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a,
        0x0a, 0x08,
    ],
    [
        0x0a, 0x09, 0x08, 0x08, 0x09, 0x09, 0x09, 0x09, 0x09, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a,
        0x0b, 0x08,
    ],
    [
        0x0a, 0x09, 0x09, 0x09, 0x09, 0x09, 0x09, 0x09, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0b,
        0x0b, 0x08,
    ],
    [
        0x0b, 0x09, 0x09, 0x09, 0x09, 0x09, 0x09, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0b, 0x0a, 0x0b,
        0x0b, 0x08,
    ],
    [
        0x0b, 0x0a, 0x09, 0x09, 0x0a, 0x09, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0b, 0x0b, 0x0b, 0x0b,
        0x0b, 0x08,
    ],
    [
        0x0b, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0b, 0x0b, 0x0b, 0x0b,
        0x0b, 0x09,
    ],
    [
        0x0b, 0x0a, 0x09, 0x09, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0b, 0x0b, 0x0b, 0x0b, 0x0b,
        0x0b, 0x09,
    ],
    [
        0x0b, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0b, 0x0b, 0x0b, 0x0b, 0x0b,
        0x0b, 0x09,
    ],
    [
        0x0c, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0b, 0x0b, 0x0b, 0x0b, 0x0b, 0x0b, 0x0c,
        0x0c, 0x09,
    ],
    [
        0x09, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08,
        0x09, 0x05,
    ],
];

const STANDARD_CODEBOOK_11_CODES: [[u16; 17]; 17] = [
    [
        0x0000, 0x0006, 0x0019, 0x003d, 0x009c, 0x00c6, 0x01a7, 0x0390, 0x03c2, 0x03df, 0x07e6,
        0x07f3, 0x0ffb, 0x07ec, 0x0ffa, 0x0ffe, 0x038e,
    ],
    [
        0x0005, 0x0001, 0x0008, 0x0014, 0x0037, 0x0042, 0x0092, 0x00af, 0x0191, 0x01a5, 0x01b5,
        0x039e, 0x03c0, 0x03a2, 0x03cd, 0x07d6, 0x00ae,
    ],
    [
        0x0017, 0x0007, 0x0009, 0x0018, 0x0039, 0x0040, 0x008e, 0x00a3, 0x00b8, 0x0199, 0x01ac,
        0x01c1, 0x03b1, 0x0396, 0x03be, 0x03ca, 0x009d,
    ],
    [
        0x003c, 0x0015, 0x0016, 0x001a, 0x003b, 0x0044, 0x0091, 0x00a5, 0x00be, 0x0196, 0x01ae,
        0x01b9, 0x03a1, 0x0391, 0x03a5, 0x03d5, 0x0094,
    ],
    [
        0x009a, 0x0036, 0x0038, 0x003a, 0x0041, 0x008c, 0x009b, 0x00b0, 0x00c3, 0x019e, 0x01ab,
        0x01bc, 0x039f, 0x038f, 0x03a9, 0x03cf, 0x0093,
    ],
    [
        0x00bf, 0x003e, 0x003f, 0x0043, 0x0045, 0x009e, 0x00a7, 0x00b9, 0x0194, 0x01a2, 0x01ba,
        0x01c3, 0x03a6, 0x03a7, 0x03bb, 0x03d4, 0x009f,
    ],
    [
        0x01a0, 0x008f, 0x008d, 0x0090, 0x0098, 0x00a6, 0x00b6, 0x00c4, 0x019f, 0x01af, 0x01bf,
        0x0399, 0x03bf, 0x03b4, 0x03c9, 0x03e7, 0x00a8,
    ],
    [
        0x01b6, 0x00ab, 0x00a4, 0x00aa, 0x00b2, 0x00c2, 0x00c5, 0x0198, 0x01a4, 0x01b8, 0x038c,
        0x03a4, 0x03c4, 0x03c6, 0x03dd, 0x03e8, 0x00ad,
    ],
    [
        0x03af, 0x0192, 0x00bd, 0x00bc, 0x018e, 0x0197, 0x019a, 0x01a3, 0x01b1, 0x038d, 0x0398,
        0x03b7, 0x03d3, 0x03d1, 0x03db, 0x07dd, 0x00b4,
    ],
    [
        0x03de, 0x01a9, 0x019b, 0x019c, 0x01a1, 0x01aa, 0x01ad, 0x01b3, 0x038b, 0x03b2, 0x03b8,
        0x03ce, 0x03e1, 0x03e0, 0x07d2, 0x07e5, 0x00b7,
    ],
    [
        0x07e3, 0x01bb, 0x01a8, 0x01a6, 0x01b0, 0x01b2, 0x01b7, 0x039b, 0x039a, 0x03ba, 0x03b5,
        0x03d6, 0x07d7, 0x03e4, 0x07d8, 0x07ea, 0x00ba,
    ],
    [
        0x07e8, 0x03a0, 0x01bd, 0x01b4, 0x038a, 0x01c4, 0x0392, 0x03aa, 0x03b0, 0x03bc, 0x03d7,
        0x07d4, 0x07dc, 0x07db, 0x07d5, 0x07f0, 0x00c1,
    ],
    [
        0x07fb, 0x03c8, 0x03a3, 0x0395, 0x039d, 0x03ac, 0x03ae, 0x03c5, 0x03d8, 0x03e2, 0x03e6,
        0x07e4, 0x07e7, 0x07e0, 0x07e9, 0x07f7, 0x0190,
    ],
    [
        0x07f2, 0x0393, 0x01be, 0x01c0, 0x0394, 0x0397, 0x03ad, 0x03c3, 0x03c1, 0x03d2, 0x07da,
        0x07d9, 0x07df, 0x07eb, 0x07f4, 0x07fa, 0x0195,
    ],
    [
        0x07f8, 0x03bd, 0x039c, 0x03ab, 0x03a8, 0x03b3, 0x03b9, 0x03d0, 0x03e3, 0x03e5, 0x07e2,
        0x07de, 0x07ed, 0x07f1, 0x07f9, 0x07fc, 0x0193,
    ],
    [
        0x0ffd, 0x03dc, 0x03b6, 0x03c7, 0x03cc, 0x03cb, 0x03d9, 0x03da, 0x07d3, 0x07e1, 0x07ee,
        0x07ef, 0x07f5, 0x07f6, 0x0ffc, 0x0fff, 0x019d,
    ],
    [
        0x01c2, 0x00b5, 0x00a1, 0x0096, 0x0097, 0x0095, 0x0099, 0x00a0, 0x00a2, 0x00ac, 0x00a9,
        0x00b1, 0x00b3, 0x00bb, 0x00c0, 0x018f, 0x0004,
    ],
];

#[derive(Debug, Clone)]
pub struct SpectralCoefficients {
    quantized: Vec<i32>,
    dequantized: Vec<f32>,
}

impl SpectralCoefficients {
    pub fn new(frame_len: usize) -> Self {
        Self {
            quantized: vec![0; frame_len],
            dequantized: vec![0.0; frame_len],
        }
    }

    pub fn decode<D: SpectralDecoder>(
        &mut self,
        reader: &mut BitReader<'_>,
        prefix: &IndividualChannelStreamPrefix,
        scale_factors: &ScaleFactorData,
        layout: BandLayout<'_>,
        decoder: &mut D,
    ) -> Result<&[f32]> {
        match prefix.ics_info.window_sequence {
            WindowSequence::EightShort => {
                self.decode_short(reader, prefix, scale_factors, layout, decoder)
            }
            WindowSequence::OnlyLong | WindowSequence::LongStart | WindowSequence::LongStop => {
                self.decode_long(reader, prefix, scale_factors, layout, decoder)
            }
        }
    }

    pub fn decode_standard(
        &mut self,
        reader: &mut BitReader<'_>,
        prefix: &IndividualChannelStreamPrefix,
        scale_factors: &ScaleFactorData,
        layout: BandLayout<'_>,
    ) -> Result<&[f32]> {
        self.decode_standard_with_pulse(reader, prefix, scale_factors, layout, None)
    }

    pub fn decode_standard_with_pulse(
        &mut self,
        reader: &mut BitReader<'_>,
        prefix: &IndividualChannelStreamPrefix,
        scale_factors: &ScaleFactorData,
        layout: BandLayout<'_>,
        pulse_data: Option<&PulseData>,
    ) -> Result<&[f32]> {
        let mut pns_state = PNS_LCG_SEED;
        self.decode_standard_with_pulse_and_pns(
            reader,
            prefix,
            scale_factors,
            layout,
            pulse_data,
            &mut pns_state,
        )
    }

    pub fn decode_standard_with_pulse_and_pns(
        &mut self,
        reader: &mut BitReader<'_>,
        prefix: &IndividualChannelStreamPrefix,
        scale_factors: &ScaleFactorData,
        layout: BandLayout<'_>,
        pulse_data: Option<&PulseData>,
        pns_state: &mut u32,
    ) -> Result<&[f32]> {
        match prefix.ics_info.window_sequence {
            WindowSequence::EightShort => {
                if pulse_data.is_some() {
                    return Err(AacLcError::InvalidBitstream(
                        "pulse data is not allowed for short windows",
                    ));
                }
                self.decode_short_standard(reader, prefix, scale_factors, layout, pns_state)
            }
            WindowSequence::OnlyLong | WindowSequence::LongStart | WindowSequence::LongStop => {
                match pulse_data {
                    Some(pulse_data) => self.decode_long_standard_with_pulse(
                        reader,
                        prefix,
                        scale_factors,
                        layout,
                        pulse_data,
                        pns_state,
                    ),
                    None => {
                        self.decode_long_standard(reader, prefix, scale_factors, layout, pns_state)
                    }
                }
            }
        }
    }

    fn decode_long_standard(
        &mut self,
        reader: &mut BitReader<'_>,
        prefix: &IndividualChannelStreamPrefix,
        scale_factors: &ScaleFactorData,
        layout: BandLayout<'_>,
        pns_state: &mut u32,
    ) -> Result<&[f32]> {
        if prefix.ics_info.window_sequence == WindowSequence::EightShort {
            return Err(AacLcError::NotImplemented(
                "short spectral coefficient mapping requires short layout",
            ));
        }

        self.dequantized.fill(0.0);
        let pow43 = pow43_table();
        let mut decoder = StandardSpectralDecoder;

        let max_sfb = prefix.ics_info.max_sfb as usize;
        for sfb in 0..max_sfb {
            let range = layout.band_range(sfb)?;
            if range.end > self.dequantized.len() {
                return Err(AacLcError::InvalidConfig(
                    "scale-factor band exceeds coefficient buffer",
                ));
            }

            let codebook =
                prefix
                    .section_data
                    .codebook(0, sfb)
                    .ok_or(AacLcError::InvalidBitstream(
                        "missing spectral codebook for scale-factor band",
                    ))?;

            match codebook {
                SectionCodebook::Zero => {}
                SectionCodebook::Spectral(codebook_id) => {
                    let scale = spectral_scale(scale_factors, 0, sfb)?;
                    decoder.read_scaled(
                        reader,
                        codebook_id,
                        scale,
                        pow43,
                        &mut self.dequantized[range],
                    )?;
                }
                SectionCodebook::Noise => {
                    let scale = noise_scale(scale_factors, 0, sfb)?;
                    synthesize_noise_band(scale, pns_state, &mut self.dequantized[range])?;
                }
                SectionCodebook::Intensity | SectionCodebook::IntensityNegative => {}
            }
        }

        Ok(&self.dequantized)
    }

    fn decode_long_standard_with_pulse(
        &mut self,
        reader: &mut BitReader<'_>,
        prefix: &IndividualChannelStreamPrefix,
        scale_factors: &ScaleFactorData,
        layout: BandLayout<'_>,
        pulse_data: &PulseData,
        pns_state: &mut u32,
    ) -> Result<&[f32]> {
        let mut decoder = StandardSpectralDecoder;
        self.read_long_quantized(reader, prefix, layout, &mut decoder)?;
        self.apply_pulse_data(pulse_data, prefix, layout)?;
        self.dequantize_long(prefix, scale_factors, layout, pns_state)?;
        Ok(&self.dequantized)
    }

    fn decode_short_standard(
        &mut self,
        reader: &mut BitReader<'_>,
        prefix: &IndividualChannelStreamPrefix,
        scale_factors: &ScaleFactorData,
        layout: BandLayout<'_>,
        pns_state: &mut u32,
    ) -> Result<&[f32]> {
        if prefix.ics_info.window_sequence != WindowSequence::EightShort {
            return Err(AacLcError::InvalidConfig(
                "short spectral coefficient mapping requires eight short windows",
            ));
        }
        if self.dequantized.len() < 8 * SHORT_WINDOW_COEFFICIENTS {
            return Err(AacLcError::InvalidConfig(
                "short-window coefficient buffer is too small",
            ));
        }

        self.dequantized.fill(0.0);
        let pow43 = pow43_table();
        let mut decoder = StandardSpectralDecoder;
        let max_sfb = prefix.ics_info.max_sfb as usize;
        let mut window_start = 0usize;

        for group in 0..prefix.ics_info.num_window_groups as usize {
            let group_len = prefix.ics_info.window_group_len[group] as usize;
            if group_len == 0 {
                return Err(AacLcError::InvalidBitstream(
                    "short-window group has zero length",
                ));
            }
            if window_start + group_len > 8 {
                return Err(AacLcError::InvalidBitstream(
                    "short-window groups exceed eight windows",
                ));
            }

            for sfb in 0..max_sfb {
                let range = layout.band_range(sfb)?;
                if range.end > SHORT_WINDOW_COEFFICIENTS {
                    return Err(AacLcError::InvalidConfig(
                        "short scale-factor band exceeds window length",
                    ));
                }

                let codebook = prefix.section_data.codebook(group, sfb).ok_or(
                    AacLcError::InvalidBitstream("missing spectral codebook for scale-factor band"),
                )?;

                match codebook {
                    SectionCodebook::Zero => {}
                    SectionCodebook::Spectral(codebook_id) => {
                        let scale = spectral_scale(scale_factors, group, sfb)?;
                        for window in window_start..window_start + group_len {
                            let start = window * SHORT_WINDOW_COEFFICIENTS + range.start;
                            let end = window * SHORT_WINDOW_COEFFICIENTS + range.end;
                            decoder.read_scaled(
                                reader,
                                codebook_id,
                                scale,
                                pow43,
                                &mut self.dequantized[start..end],
                            )?;
                        }
                    }
                    SectionCodebook::Noise => {
                        let scale = noise_scale(scale_factors, group, sfb)?;
                        for window in window_start..window_start + group_len {
                            let start = window * SHORT_WINDOW_COEFFICIENTS + range.start;
                            let end = window * SHORT_WINDOW_COEFFICIENTS + range.end;
                            synthesize_noise_band(
                                scale,
                                pns_state,
                                &mut self.dequantized[start..end],
                            )?;
                        }
                    }
                    SectionCodebook::Intensity | SectionCodebook::IntensityNegative => {}
                }
            }

            window_start += group_len;
        }

        if window_start != 8 {
            return Err(AacLcError::InvalidBitstream(
                "short-window groups do not cover eight windows",
            ));
        }

        Ok(&self.dequantized)
    }

    pub fn decode_long<D: SpectralDecoder>(
        &mut self,
        reader: &mut BitReader<'_>,
        prefix: &IndividualChannelStreamPrefix,
        scale_factors: &ScaleFactorData,
        layout: BandLayout<'_>,
        decoder: &mut D,
    ) -> Result<&[f32]> {
        if prefix.ics_info.window_sequence == WindowSequence::EightShort {
            return Err(AacLcError::NotImplemented(
                "short spectral coefficient mapping requires short layout",
            ));
        }

        self.read_long_quantized(reader, prefix, layout, decoder)?;
        let mut pns_state = PNS_LCG_SEED;
        self.dequantize_long(prefix, scale_factors, layout, &mut pns_state)?;
        Ok(&self.dequantized)
    }

    pub fn decode_long_with_pulse<D: SpectralDecoder>(
        &mut self,
        reader: &mut BitReader<'_>,
        prefix: &IndividualChannelStreamPrefix,
        scale_factors: &ScaleFactorData,
        layout: BandLayout<'_>,
        decoder: &mut D,
        pulse_data: Option<&PulseData>,
    ) -> Result<&[f32]> {
        if prefix.ics_info.window_sequence == WindowSequence::EightShort {
            return Err(AacLcError::NotImplemented(
                "short spectral coefficient mapping requires short layout",
            ));
        }

        self.read_long_quantized(reader, prefix, layout, decoder)?;
        if let Some(pulse_data) = pulse_data {
            self.apply_pulse_data(pulse_data, prefix, layout)?;
        }
        let mut pns_state = PNS_LCG_SEED;
        self.dequantize_long(prefix, scale_factors, layout, &mut pns_state)?;
        Ok(&self.dequantized)
    }

    fn read_long_quantized<D: SpectralDecoder>(
        &mut self,
        reader: &mut BitReader<'_>,
        prefix: &IndividualChannelStreamPrefix,
        layout: BandLayout<'_>,
        decoder: &mut D,
    ) -> Result<()> {
        self.quantized.fill(0);

        let max_sfb = prefix.ics_info.max_sfb as usize;
        for sfb in 0..max_sfb {
            let range = layout.band_range(sfb)?;
            if range.end > self.quantized.len() {
                return Err(AacLcError::InvalidConfig(
                    "scale-factor band exceeds coefficient buffer",
                ));
            }

            let codebook =
                prefix
                    .section_data
                    .codebook(0, sfb)
                    .ok_or(AacLcError::InvalidBitstream(
                        "missing spectral codebook for scale-factor band",
                    ))?;

            match codebook {
                SectionCodebook::Zero => {}
                SectionCodebook::Spectral(codebook_id) => {
                    decoder.read_quantized(
                        reader,
                        codebook_id,
                        &mut self.quantized[range.clone()],
                    )?;
                }
                SectionCodebook::Noise => {}
                SectionCodebook::Intensity | SectionCodebook::IntensityNegative => {}
            }
        }

        Ok(())
    }

    fn apply_pulse_data(
        &mut self,
        pulse_data: &PulseData,
        prefix: &IndividualChannelStreamPrefix,
        layout: BandLayout<'_>,
    ) -> Result<()> {
        let max_sfb = prefix.ics_info.max_sfb as usize;
        let pulse_start_sfb = pulse_data.pulse_start_sfb as usize;
        if pulse_start_sfb >= max_sfb {
            return Err(AacLcError::InvalidBitstream(
                "pulse start scale-factor band exceeds max_sfb",
            ));
        }

        let mut index = layout.band_range(pulse_start_sfb)?.start;
        for pulse in pulse_data.pulses.iter().take(pulse_data.count as usize) {
            index = index
                .checked_add(pulse.offset as usize)
                .ok_or(AacLcError::InvalidBitstream("pulse offset overflow"))?;
            if index >= self.quantized.len() {
                return Err(AacLcError::InvalidBitstream(
                    "pulse target exceeds spectral coefficient buffer",
                ));
            }

            let sfb = band_for_coefficient(layout, max_sfb, index)?;
            match prefix.section_data.codebook(0, sfb) {
                Some(SectionCodebook::Spectral(_)) => {}
                Some(_) => {
                    return Err(AacLcError::InvalidBitstream(
                        "pulse target is not in a spectral band",
                    ));
                }
                None => {
                    return Err(AacLcError::InvalidBitstream(
                        "missing spectral codebook for pulse target",
                    ));
                }
            }

            let amp = i32::from(pulse.amp);
            if self.quantized[index] > 0 {
                self.quantized[index] += amp;
            } else {
                self.quantized[index] -= amp;
            }
        }

        Ok(())
    }

    fn dequantize_long(
        &mut self,
        prefix: &IndividualChannelStreamPrefix,
        scale_factors: &ScaleFactorData,
        layout: BandLayout<'_>,
        pns_state: &mut u32,
    ) -> Result<()> {
        self.dequantized.fill(0.0);
        let pow43 = pow43_table();

        let max_sfb = prefix.ics_info.max_sfb as usize;
        for sfb in 0..max_sfb {
            let range = layout.band_range(sfb)?;
            if range.end > self.dequantized.len() {
                return Err(AacLcError::InvalidConfig(
                    "scale-factor band exceeds coefficient buffer",
                ));
            }

            let codebook =
                prefix
                    .section_data
                    .codebook(0, sfb)
                    .ok_or(AacLcError::InvalidBitstream(
                        "missing spectral codebook for scale-factor band",
                    ))?;

            match codebook {
                SectionCodebook::Zero => {}
                SectionCodebook::Spectral(_) => {
                    let scale = spectral_scale(scale_factors, 0, sfb)?;
                    for idx in range {
                        self.dequantized[idx] =
                            dequantize_signed_scaled(self.quantized[idx], scale, pow43);
                    }
                }
                SectionCodebook::Noise => {
                    let scale = noise_scale(scale_factors, 0, sfb)?;
                    synthesize_noise_band(scale, pns_state, &mut self.dequantized[range])?;
                }
                SectionCodebook::Intensity | SectionCodebook::IntensityNegative => {}
            }
        }

        Ok(())
    }

    pub fn decode_short<D: SpectralDecoder>(
        &mut self,
        reader: &mut BitReader<'_>,
        prefix: &IndividualChannelStreamPrefix,
        scale_factors: &ScaleFactorData,
        layout: BandLayout<'_>,
        decoder: &mut D,
    ) -> Result<&[f32]> {
        if prefix.ics_info.window_sequence != WindowSequence::EightShort {
            return Err(AacLcError::InvalidConfig(
                "short spectral coefficient mapping requires eight short windows",
            ));
        }
        if self.quantized.len() < 8 * SHORT_WINDOW_COEFFICIENTS {
            return Err(AacLcError::InvalidConfig(
                "short-window coefficient buffer is too small",
            ));
        }

        self.quantized.fill(0);
        self.dequantized.fill(0.0);
        let pow43 = pow43_table();
        let mut pns_state = PNS_LCG_SEED;

        let max_sfb = prefix.ics_info.max_sfb as usize;
        let mut window_start = 0usize;

        for group in 0..prefix.ics_info.num_window_groups as usize {
            let group_len = prefix.ics_info.window_group_len[group] as usize;
            if group_len == 0 {
                return Err(AacLcError::InvalidBitstream(
                    "short-window group has zero length",
                ));
            }
            if window_start + group_len > 8 {
                return Err(AacLcError::InvalidBitstream(
                    "short-window groups exceed eight windows",
                ));
            }

            for sfb in 0..max_sfb {
                let range = layout.band_range(sfb)?;
                if range.end > SHORT_WINDOW_COEFFICIENTS {
                    return Err(AacLcError::InvalidConfig(
                        "short scale-factor band exceeds window length",
                    ));
                }

                let codebook = prefix.section_data.codebook(group, sfb).ok_or(
                    AacLcError::InvalidBitstream("missing spectral codebook for scale-factor band"),
                )?;

                match codebook {
                    SectionCodebook::Zero => {}
                    SectionCodebook::Spectral(codebook_id) => {
                        let scale = spectral_scale(scale_factors, group, sfb)?;

                        for window in window_start..window_start + group_len {
                            let start = window * SHORT_WINDOW_COEFFICIENTS + range.start;
                            let end = window * SHORT_WINDOW_COEFFICIENTS + range.end;
                            decoder.read_quantized(
                                reader,
                                codebook_id,
                                &mut self.quantized[start..end],
                            )?;
                            for idx in start..end {
                                self.dequantized[idx] =
                                    dequantize_signed_scaled(self.quantized[idx], scale, pow43);
                            }
                        }
                    }
                    SectionCodebook::Noise => {
                        let scale = noise_scale(scale_factors, group, sfb)?;
                        for window in window_start..window_start + group_len {
                            let start = window * SHORT_WINDOW_COEFFICIENTS + range.start;
                            let end = window * SHORT_WINDOW_COEFFICIENTS + range.end;
                            synthesize_noise_band(
                                scale,
                                &mut pns_state,
                                &mut self.dequantized[start..end],
                            )?;
                        }
                    }
                    SectionCodebook::Intensity | SectionCodebook::IntensityNegative => {}
                }
            }

            window_start += group_len;
        }

        if window_start != 8 {
            return Err(AacLcError::InvalidBitstream(
                "short-window groups do not cover eight windows",
            ));
        }

        Ok(&self.dequantized)
    }

    pub fn dequantized(&self) -> &[f32] {
        &self.dequantized
    }

    pub fn dequantized_mut(&mut self) -> &mut [f32] {
        &mut self.dequantized
    }

    pub fn quantized(&self) -> &[i32] {
        &self.quantized
    }
}

fn spectral_scale(scale_factors: &ScaleFactorData, group: usize, sfb: usize) -> Result<f32> {
    scale_factors.spectral_multiplier(group, sfb)
}

fn noise_scale(scale_factors: &ScaleFactorData, group: usize, sfb: usize) -> Result<f32> {
    scale_factors.noise_multiplier(group, sfb)
}

fn synthesize_noise_band(scale: f32, pns_state: &mut u32, out: &mut [f32]) -> Result<()> {
    if out.is_empty() {
        return Ok(());
    }

    let mut energy = 0.0f32;
    for sample in out.iter_mut() {
        let noise = next_noise_sample(pns_state);
        *sample = noise;
        energy += noise * noise;
    }

    if energy <= f32::EPSILON {
        return Err(AacLcError::InvalidBitstream(
            "PNS noise band has zero energy",
        ));
    }

    let normalizer = scale / energy.sqrt();
    for sample in out {
        *sample *= normalizer;
    }

    Ok(())
}

fn next_noise_sample(state: &mut u32) -> f32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    ((*state as i32) >> 16) as i16 as f32
}

fn band_for_coefficient(layout: BandLayout<'_>, max_sfb: usize, index: usize) -> Result<usize> {
    for sfb in 0..max_sfb {
        if layout.band_range(sfb)?.contains(&index) {
            return Ok(sfb);
        }
    }

    Err(AacLcError::InvalidBitstream(
        "pulse target exceeds coded scale-factor bands",
    ))
}

pub const PNS_LCG_SEED: u32 = 0x1f2e_3d4c;
const SHORT_WINDOW_COEFFICIENTS: usize = 128;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SamplingFrequency;
    use crate::ics::{IcsInfo, WindowShape};
    use crate::pulse::{Pulse, PulseData};
    use crate::scalefactor::{ScaleFactorValue, VlcScaleFactorDecoder};
    use crate::section::SectionData;
    use crate::sfb::long_window_band_layout;
    use crate::vlc::{VlcEntry, VlcTable};

    const MINI_SF: [VlcEntry<i16>; 1] = [VlcEntry {
        code: 0,
        bits: 1,
        value: 0,
    }];

    const SHORT_SPECTRAL_VALUES: [i32; 32] = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        26, 27, 28, 29, 30, 31, 32,
    ];

    const SIGNED_PAIR_TABLE: [VlcEntry<[i32; 2]>; 2] = [
        VlcEntry {
            code: 0b0,
            bits: 1,
            value: [1, -1],
        },
        VlcEntry {
            code: 0b1,
            bits: 1,
            value: [-2, 0],
        },
    ];

    const UNSIGNED_QUAD_TABLE: [VlcEntry<[i32; 4]>; 1] = [VlcEntry {
        code: 0b0,
        bits: 1,
        value: [1, 0, 2, 3],
    }];

    const ESCAPE_PAIR_TABLE: [VlcEntry<[i32; 2]>; 1] = [VlcEntry {
        code: 0b0,
        bits: 1,
        value: [16, 2],
    }];

    #[derive(Debug)]
    struct MiniSpectralDecoder {
        values: &'static [i32],
        pos: usize,
    }

    impl SpectralDecoder for MiniSpectralDecoder {
        fn read_quantized(
            &mut self,
            _reader: &mut BitReader<'_>,
            codebook: u8,
            out: &mut [i32],
        ) -> Result<()> {
            assert_eq!(codebook, 1);
            let end = self.pos + out.len();
            out.copy_from_slice(&self.values[self.pos..end]);
            self.pos = end;
            Ok(())
        }
    }

    #[test]
    fn maps_codebook_ids_to_tuple_kinds() {
        assert_eq!(
            SpectralCodebookKind::from_codebook_id(1).unwrap(),
            SpectralCodebookKind::SignedQuad
        );
        assert_eq!(
            SpectralCodebookKind::from_codebook_id(2).unwrap(),
            SpectralCodebookKind::SignedQuad
        );
        assert_eq!(
            SpectralCodebookKind::from_codebook_id(3).unwrap(),
            SpectralCodebookKind::UnsignedQuad
        );
        assert_eq!(
            SpectralCodebookKind::from_codebook_id(4).unwrap(),
            SpectralCodebookKind::UnsignedQuad
        );
        assert_eq!(
            SpectralCodebookKind::from_codebook_id(6).unwrap(),
            SpectralCodebookKind::SignedPair
        );
        assert_eq!(
            SpectralCodebookKind::from_codebook_id(7).unwrap(),
            SpectralCodebookKind::UnsignedPair
        );
        assert_eq!(
            SpectralCodebookKind::from_codebook_id(9).unwrap(),
            SpectralCodebookKind::UnsignedPair
        );
        assert_eq!(
            SpectralCodebookKind::from_codebook_id(10).unwrap(),
            SpectralCodebookKind::UnsignedPair
        );
        assert_eq!(
            SpectralCodebookKind::from_codebook_id(11).unwrap(),
            SpectralCodebookKind::UnsignedPairEscape
        );
        assert!(!SpectralCodebookKind::SignedQuad.uses_extra_sign_bits());
        assert!(!SpectralCodebookKind::SignedPair.uses_extra_sign_bits());
        assert!(SpectralCodebookKind::UnsignedQuad.uses_extra_sign_bits());
        assert!(SpectralCodebookKind::UnsignedPair.uses_extra_sign_bits());
    }

    #[test]
    fn standard_quad_tables_are_prefix_free() {
        assert_quad_table_is_prefix_free(
            &STANDARD_CODEBOOK_1_2_LENGTHS,
            LengthHalf::High,
            &STANDARD_CODEBOOK_1_CODES,
            STANDARD_CODEBOOK_1_MAX_BITS,
        );
        assert_quad_table_is_prefix_free(
            &STANDARD_CODEBOOK_1_2_LENGTHS,
            LengthHalf::Low,
            &STANDARD_CODEBOOK_2_CODES,
            STANDARD_CODEBOOK_2_MAX_BITS,
        );
        assert_quad_table_is_prefix_free(
            &STANDARD_CODEBOOK_3_4_LENGTHS,
            LengthHalf::High,
            &STANDARD_CODEBOOK_3_CODES,
            STANDARD_CODEBOOK_3_MAX_BITS,
        );
        assert_quad_table_is_prefix_free(
            &STANDARD_CODEBOOK_3_4_LENGTHS,
            LengthHalf::Low,
            &STANDARD_CODEBOOK_4_CODES,
            STANDARD_CODEBOOK_4_MAX_BITS,
        );
    }

    fn assert_quad_table_is_prefix_free(
        lengths: &[[[[u32; 3]; 3]; 3]; 3],
        half: LengthHalf,
        codes: &[[[[u16; 3]; 3]; 3]; 3],
        max_bits: u8,
    ) {
        let mut entries = Vec::new();

        for a in 0..3 {
            for b in 0..3 {
                for c in 0..3 {
                    for d in 0..3 {
                        let bits = standard_quad_len(lengths, half, a, b, c, d);
                        let code = codes[a][b][c][d] as u32;
                        assert!(bits > 0);
                        assert!(bits <= max_bits);
                        assert!(code < (1u32 << bits));
                        entries.push((code, bits, [a, b, c, d]));
                    }
                }
            }
        }

        for (idx, &(left_code, left_bits, left_value)) in entries.iter().enumerate() {
            for &(right_code, right_bits, right_value) in &entries[idx + 1..] {
                let prefix_match = if left_bits <= right_bits {
                    left_code == (right_code >> (right_bits - left_bits))
                } else {
                    right_code == (left_code >> (left_bits - right_bits))
                };
                assert!(
                    !prefix_match,
                    "prefix collision between {left_value:?} and {right_value:?}"
                );
            }
        }
    }

    #[test]
    fn standard_pair_tables_are_prefix_free() {
        assert_pair_table_is_prefix_free(
            &STANDARD_CODEBOOK_5_6_LENGTHS,
            LengthHalf::High,
            &STANDARD_CODEBOOK_5_CODES,
            STANDARD_CODEBOOK_5_MAX_BITS,
        );
        assert_pair_table_is_prefix_free(
            &STANDARD_CODEBOOK_5_6_LENGTHS,
            LengthHalf::Low,
            &STANDARD_CODEBOOK_6_CODES,
            STANDARD_CODEBOOK_6_MAX_BITS,
        );
        assert_pair_table_is_prefix_free(
            &STANDARD_CODEBOOK_7_8_LENGTHS,
            LengthHalf::High,
            &STANDARD_CODEBOOK_7_CODES,
            STANDARD_CODEBOOK_7_MAX_BITS,
        );
        assert_pair_table_is_prefix_free(
            &STANDARD_CODEBOOK_7_8_LENGTHS,
            LengthHalf::Low,
            &STANDARD_CODEBOOK_8_CODES,
            STANDARD_CODEBOOK_8_MAX_BITS,
        );
        assert_pair_table_is_prefix_free(
            &STANDARD_CODEBOOK_9_10_LENGTHS,
            LengthHalf::High,
            &STANDARD_CODEBOOK_9_CODES,
            STANDARD_CODEBOOK_9_MAX_BITS,
        );
        assert_pair_table_is_prefix_free(
            &STANDARD_CODEBOOK_9_10_LENGTHS,
            LengthHalf::Low,
            &STANDARD_CODEBOOK_10_CODES,
            STANDARD_CODEBOOK_10_MAX_BITS,
        );
    }

    fn assert_pair_table_is_prefix_free<const N: usize>(
        lengths: &[[u32; N]; N],
        half: LengthHalf,
        codes: &[[u16; N]; N],
        max_bits: u8,
    ) {
        let mut entries = Vec::new();

        for a in 0..N {
            for b in 0..N {
                let bits = standard_pair_len(lengths, half, a, b);
                let code = codes[a][b] as u32;
                assert!(bits > 0);
                assert!(bits <= max_bits);
                assert!(code < (1u32 << bits));
                entries.push((code, bits, [a, b]));
            }
        }

        for (idx, &(left_code, left_bits, left_value)) in entries.iter().enumerate() {
            for &(right_code, right_bits, right_value) in &entries[idx + 1..] {
                let prefix_match = if left_bits <= right_bits {
                    left_code == (right_code >> (right_bits - left_bits))
                } else {
                    right_code == (left_code >> (left_bits - right_bits))
                };
                assert!(
                    !prefix_match,
                    "prefix collision between {left_value:?} and {right_value:?}"
                );
            }
        }
    }

    #[test]
    fn standard_escape_table_is_prefix_free() {
        let mut entries = Vec::new();

        for a in 0..17 {
            for b in 0..17 {
                let bits = STANDARD_CODEBOOK_11_LENGTHS[a][b];
                let code = STANDARD_CODEBOOK_11_CODES[a][b] as u32;
                assert!(bits > 0);
                assert!(bits <= STANDARD_CODEBOOK_11_MAX_BITS);
                assert!(code < (1u32 << bits));
                entries.push((code, bits, [a, b]));
            }
        }

        for (idx, &(left_code, left_bits, left_value)) in entries.iter().enumerate() {
            for &(right_code, right_bits, right_value) in &entries[idx + 1..] {
                let prefix_match = if left_bits <= right_bits {
                    left_code == (right_code >> (right_bits - left_bits))
                } else {
                    right_code == (left_code >> (left_bits - right_bits))
                };
                assert!(
                    !prefix_match,
                    "prefix collision between {left_value:?} and {right_value:?}"
                );
            }
        }
    }

    #[test]
    fn standard_spectral_decoder_decodes_signed_quad_codebooks() {
        let bytes = build_bits(&[(0b0, 1), (0b10100, 5), (0b0, 1)]);
        let mut reader = BitReader::new(&bytes);
        let mut decoder = StandardSpectralDecoder;
        let mut out = [0i32; 12];

        decoder.read_quantized(&mut reader, 1, &mut out).unwrap();

        assert_eq!(out, [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]);

        let bytes = build_bits(&[(0b000, 3), (0b00111, 5)]);
        let mut reader = BitReader::new(&bytes);
        let mut out = [0i32; 8];

        decoder.read_quantized(&mut reader, 2, &mut out).unwrap();

        assert_eq!(out, [0, 0, 0, 0, 0, 0, 0, 1]);
    }

    #[test]
    fn standard_spectral_decoder_decodes_unsigned_quad_codebooks_with_sign_bits() {
        let bytes = build_bits(&[(0b0, 1), (0b1000, 4), (1, 1)]);
        let mut reader = BitReader::new(&bytes);
        let mut decoder = StandardSpectralDecoder;
        let mut out = [0i32; 8];

        decoder.read_quantized(&mut reader, 3, &mut out).unwrap();

        assert_eq!(out, [0, 0, 0, 0, -1, 0, 0, 0]);

        let bytes = build_bits(&[(0b0111, 4), (0b0101, 4), (0, 1)]);
        let mut reader = BitReader::new(&bytes);
        let mut out = [0i32; 8];

        decoder.read_quantized(&mut reader, 4, &mut out).unwrap();

        assert_eq!(out, [0, 0, 0, 0, 1, 0, 0, 0]);
    }

    #[test]
    fn standard_spectral_decoder_decodes_signed_pair_codebooks() {
        let bytes = build_bits(&[(0b0, 1), (0b1010, 4)]);
        let mut reader = BitReader::new(&bytes);
        let mut decoder = StandardSpectralDecoder;
        let mut out = [0i32; 4];

        decoder.read_quantized(&mut reader, 5, &mut out).unwrap();

        assert_eq!(out, [0, 0, 0, 1]);

        let bytes = build_bits(&[(0b0000, 4), (0b0001, 4)]);
        let mut reader = BitReader::new(&bytes);
        let mut out = [0i32; 4];

        decoder.read_quantized(&mut reader, 6, &mut out).unwrap();

        assert_eq!(out, [0, 0, 1, 0]);
    }

    #[test]
    fn standard_spectral_decoder_decodes_unsigned_pair_codebooks_with_sign_bits() {
        let bytes = build_bits(&[(0b0, 1), (0b100, 3), (1, 1)]);
        let mut reader = BitReader::new(&bytes);
        let mut decoder = StandardSpectralDecoder;
        let mut out = [0i32; 4];

        decoder.read_quantized(&mut reader, 7, &mut out).unwrap();

        assert_eq!(out, [0, 0, -1, 0]);

        let bytes = build_bits(&[(0b000, 3), (1, 1), (0, 1)]);
        let mut reader = BitReader::new(&bytes);
        let mut out = [0i32; 2];

        decoder.read_quantized(&mut reader, 8, &mut out).unwrap();

        assert_eq!(out, [-1, 1]);

        let bytes = build_bits(&[(0b0, 1), (0b100, 3), (0, 1)]);
        let mut reader = BitReader::new(&bytes);
        let mut out = [0i32; 4];

        decoder.read_quantized(&mut reader, 9, &mut out).unwrap();

        assert_eq!(out, [0, 0, 1, 0]);

        let bytes = build_bits(&[(0b0000, 4), (1, 1), (0, 1)]);
        let mut reader = BitReader::new(&bytes);
        let mut out = [0i32; 2];

        decoder.read_quantized(&mut reader, 10, &mut out).unwrap();

        assert_eq!(out, [-1, 1]);
    }

    #[test]
    fn standard_spectral_decoder_decodes_escape_pair_codebook() {
        let bytes = build_bits(&[(0b0000, 4), (0b0001, 4), (0, 1), (1, 1)]);
        let mut reader = BitReader::new(&bytes);
        let mut decoder = StandardSpectralDecoder;
        let mut out = [0i32; 4];

        decoder.read_quantized(&mut reader, 11, &mut out).unwrap();

        assert_eq!(out, [0, 0, 1, -1]);

        let bytes = build_bits(&[(0b111000010, 9), (1, 1), (0b00101, 5)]);
        let mut reader = BitReader::new(&bytes);
        let mut out = [0i32; 2];

        decoder.read_quantized(&mut reader, 11, &mut out).unwrap();

        assert_eq!(out, [-21, 0]);
    }

    #[test]
    fn standard_spectral_decoder_rejects_reserved_codebook() {
        let mut reader = BitReader::new(&[]);
        let mut decoder = StandardSpectralDecoder;
        let mut out = [0i32; 4];

        assert_eq!(
            decoder
                .read_quantized(&mut reader, 12, &mut out)
                .unwrap_err(),
            AacLcError::InvalidBitstream("reserved AAC spectral codebook")
        );
    }

    #[test]
    fn decodes_signed_spectral_tuples_without_extra_sign_bits() {
        let table = TupleCodebook::new(VlcTable::new(&SIGNED_PAIR_TABLE).unwrap());
        let mut reader = BitReader::new(&[0b0100_0000]);
        let mut out = [0i32; 4];

        decode_signed_tuples(&mut reader, table, &mut out).unwrap();

        assert_eq!(out, [1, -1, -2, 0]);
    }

    #[test]
    fn decodes_unsigned_spectral_tuples_with_sign_bits() {
        let table = TupleCodebook::new(VlcTable::new(&UNSIGNED_QUAD_TABLE).unwrap());
        let mut reader = BitReader::new(&[0b0010_0000]);
        let mut out = [0i32; 4];

        decode_unsigned_tuples(&mut reader, table, &mut out).unwrap();

        assert_eq!(out, [1, 0, -2, 3]);
    }

    #[test]
    fn decodes_unsigned_escape_pairs() {
        let table = TupleCodebook::new(VlcTable::new(&ESCAPE_PAIR_TABLE).unwrap());
        let mut reader = BitReader::new(&[0b0010_0101]);
        let mut out = [0i32; 2];

        decode_unsigned_escape_pairs(&mut reader, table, &mut out).unwrap();

        assert_eq!(out, [21, -2]);
    }

    #[test]
    fn decodes_and_dequantizes_long_spectral_band() {
        let prefix = prefix_with_sections(100, 2, &[(1, 4), (1, 5), (0, 4), (1, 5)]);
        let mut sf_reader = BitReader::new(&[0]);
        let mut sf_decoder = VlcScaleFactorDecoder::new(VlcTable::new(&MINI_SF).unwrap());
        let scale_factors =
            ScaleFactorData::read(&mut sf_reader, &prefix, &mut sf_decoder).unwrap();
        assert_eq!(
            scale_factors.value(0, 0),
            Some(ScaleFactorValue::Spectral(100))
        );

        let mut coeffs = SpectralCoefficients::new(8);
        let mut spectral_decoder = MiniSpectralDecoder {
            values: &[1, -1, 8, 0],
            pos: 0,
        };
        let layout = long_window_band_layout(SamplingFrequency::from_index(4).unwrap()).unwrap();
        let mut spectral_reader = BitReader::new(&[]);

        let out = coeffs
            .decode_long(
                &mut spectral_reader,
                &prefix,
                &scale_factors,
                layout,
                &mut spectral_decoder,
            )
            .unwrap();

        assert!((out[0] - 1.0).abs() < 1.0e-6);
        assert!((out[1] + 1.0).abs() < 1.0e-6);
        assert!((out[2] - 16.0).abs() < 1.0e-5);
        assert_eq!(out[3], 0.0);
        assert_eq!(&out[4..8], &[0.0; 4]);
    }

    #[test]
    fn applies_pulse_data_before_long_dequantization() {
        let prefix = prefix_with_sections(100, 2, &[(1, 4), (2, 5)]);
        let mut sf_reader = BitReader::new(&[0]);
        let mut sf_decoder = VlcScaleFactorDecoder::new(VlcTable::new(&MINI_SF).unwrap());
        let scale_factors =
            ScaleFactorData::read(&mut sf_reader, &prefix, &mut sf_decoder).unwrap();
        let mut coeffs = SpectralCoefficients::new(8);
        let mut spectral_decoder = MiniSpectralDecoder {
            values: &[1, -1, 0, 4, 5, 6, 7, 8],
            pos: 0,
        };
        let pulse_data = PulseData {
            pulse_start_sfb: 0,
            count: 2,
            pulses: [
                Pulse { offset: 0, amp: 3 },
                Pulse { offset: 2, amp: 2 },
                Pulse::default(),
                Pulse::default(),
            ],
        };
        let mut spectral_reader = BitReader::new(&[]);

        coeffs
            .decode_long_with_pulse(
                &mut spectral_reader,
                &prefix,
                &scale_factors,
                BandLayout::new(&[0, 4, 8]),
                &mut spectral_decoder,
                Some(&pulse_data),
            )
            .unwrap();

        assert_eq!(coeffs.quantized(), &[4, -1, -2, 4, 5, 6, 7, 8]);
        assert!((coeffs.dequantized()[0] - 4.0_f32.powf(4.0 / 3.0)).abs() < 1.0e-5);
        assert!((coeffs.dequantized()[2] + 2.0_f32.powf(4.0 / 3.0)).abs() < 1.0e-5);
    }

    #[test]
    fn reconstructs_long_pns_band_from_noise_scalefactor() {
        let prefix = prefix_with_sections(100, 1, &[(13, 4), (1, 5)]);
        let sf_bits = build_bits(&[(346, 9)]);
        let mut sf_reader = BitReader::new(&sf_bits);
        let mut sf_decoder = crate::scalefactor::NotImplementedScaleFactorDecoder;
        let scale_factors =
            ScaleFactorData::read(&mut sf_reader, &prefix, &mut sf_decoder).unwrap();
        assert_eq!(
            scale_factors.value(0, 0),
            Some(ScaleFactorValue::Noise(100))
        );

        let mut coeffs = SpectralCoefficients::new(4);
        let mut spectral_decoder = NotImplementedSpectralDecoder;
        let mut spectral_reader = BitReader::new(&[]);

        coeffs
            .decode_long(
                &mut spectral_reader,
                &prefix,
                &scale_factors,
                BandLayout::new(&[0, 4]),
                &mut spectral_decoder,
            )
            .unwrap();
        let first = coeffs.dequantized().to_vec();

        assert!(first.iter().any(|sample| *sample != 0.0));
        assert!((energy(&first) - 1.0).abs() < 1.0e-6);

        coeffs
            .decode_long(
                &mut BitReader::new(&[]),
                &prefix,
                &scale_factors,
                BandLayout::new(&[0, 4]),
                &mut spectral_decoder,
            )
            .unwrap();
        assert_eq!(coeffs.dequantized(), first.as_slice());
    }

    #[test]
    fn reconstructs_grouped_short_pns_bands() {
        let prefix = prefix_with_short_sections(
            100,
            1,
            &[2, 6],
            &[
                (13, 4), // group 0 noise
                (1, 3),
                (13, 4), // group 1 noise
                (1, 3),
            ],
        );
        let sf_bits = build_bits(&[
            (346, 9), // first noise value: 100 - 90 + (346 - 256)
            (0, 1),   // second noise delta from MINI_SF
        ]);
        let mut sf_reader = BitReader::new(&sf_bits);
        let mut sf_decoder = VlcScaleFactorDecoder::new(VlcTable::new(&MINI_SF).unwrap());
        let scale_factors =
            ScaleFactorData::read(&mut sf_reader, &prefix, &mut sf_decoder).unwrap();
        let mut coeffs = SpectralCoefficients::new(1024);
        let mut spectral_decoder = NotImplementedSpectralDecoder;

        coeffs
            .decode(
                &mut BitReader::new(&[]),
                &prefix,
                &scale_factors,
                BandLayout::new(&[0, 4]),
                &mut spectral_decoder,
            )
            .unwrap();

        assert!((energy(&coeffs.dequantized()[0..4]) - 1.0).abs() < 1.0e-6);
        assert!((energy(&coeffs.dequantized()[128..132]) - 1.0).abs() < 1.0e-6);
        assert!((energy(&coeffs.dequantized()[256..260]) - 1.0).abs() < 1.0e-6);
        assert_eq!(&coeffs.dequantized()[4..128], &[0.0; 124]);
    }

    #[test]
    fn rejects_pulse_target_in_zero_codebook_band() {
        let prefix = prefix_with_sections(100, 2, &[(1, 4), (1, 5), (0, 4), (1, 5)]);
        let mut sf_reader = BitReader::new(&[0]);
        let mut sf_decoder = VlcScaleFactorDecoder::new(VlcTable::new(&MINI_SF).unwrap());
        let scale_factors =
            ScaleFactorData::read(&mut sf_reader, &prefix, &mut sf_decoder).unwrap();
        let mut coeffs = SpectralCoefficients::new(8);
        let mut spectral_decoder = MiniSpectralDecoder {
            values: &[1, -1, 0, 4],
            pos: 0,
        };
        let pulse_data = PulseData {
            pulse_start_sfb: 0,
            count: 1,
            pulses: [
                Pulse { offset: 4, amp: 1 },
                Pulse::default(),
                Pulse::default(),
                Pulse::default(),
            ],
        };
        let mut spectral_reader = BitReader::new(&[]);

        assert_eq!(
            coeffs
                .decode_long_with_pulse(
                    &mut spectral_reader,
                    &prefix,
                    &scale_factors,
                    BandLayout::new(&[0, 4, 8]),
                    &mut spectral_decoder,
                    Some(&pulse_data),
                )
                .unwrap_err(),
            AacLcError::InvalidBitstream("pulse target is not in a spectral band")
        );
    }

    #[test]
    fn decodes_long_spectral_band_with_standard_codebook_1() {
        let prefix = prefix_with_sections(100, 1, &[(1, 4), (1, 5)]);
        let mut sf_reader = BitReader::new(&[0]);
        let mut sf_decoder = VlcScaleFactorDecoder::new(VlcTable::new(&MINI_SF).unwrap());
        let scale_factors =
            ScaleFactorData::read(&mut sf_reader, &prefix, &mut sf_decoder).unwrap();
        let mut coeffs = SpectralCoefficients::new(8);
        let mut spectral_decoder = StandardSpectralDecoder;
        let mut spectral_reader = BitReader::new(&[0]);

        let out = coeffs
            .decode_long(
                &mut spectral_reader,
                &prefix,
                &scale_factors,
                BandLayout::new(&[0, 8]),
                &mut spectral_decoder,
            )
            .unwrap();

        assert_eq!(out, &[0.0; 8]);
        assert_eq!(coeffs.quantized(), &[0; 8]);
    }

    #[test]
    fn decodes_grouped_short_window_spectral_bands() {
        let prefix = prefix_with_short_sections(
            100,
            2,
            &[2, 6],
            &[
                (1, 4), // group 0, sfb 0 spectral
                (1, 3),
                (0, 4), // group 0, sfb 1 zero
                (1, 3),
                (0, 4), // group 1, sfb 0 zero
                (1, 3),
                (1, 4), // group 1, sfb 1 spectral
                (1, 3),
            ],
        );
        let mut sf_reader = BitReader::new(&[0]);
        let mut sf_decoder = VlcScaleFactorDecoder::new(VlcTable::new(&MINI_SF).unwrap());
        let scale_factors =
            ScaleFactorData::read(&mut sf_reader, &prefix, &mut sf_decoder).unwrap();
        let mut coeffs = SpectralCoefficients::new(1024);
        let mut spectral_decoder = MiniSpectralDecoder {
            values: &SHORT_SPECTRAL_VALUES,
            pos: 0,
        };
        let mut spectral_reader = BitReader::new(&[]);

        coeffs
            .decode(
                &mut spectral_reader,
                &prefix,
                &scale_factors,
                BandLayout::new(&[0, 4, 8]),
                &mut spectral_decoder,
            )
            .unwrap();

        let quantized = coeffs.quantized();
        assert_eq!(&quantized[0..8], &[1, 2, 3, 4, 0, 0, 0, 0]);
        assert_eq!(&quantized[128..136], &[5, 6, 7, 8, 0, 0, 0, 0]);
        assert_eq!(&quantized[256..264], &[0, 0, 0, 0, 9, 10, 11, 12]);
        assert_eq!(&quantized[384..392], &[0, 0, 0, 0, 13, 14, 15, 16]);
        assert_eq!(&quantized[512..520], &[0, 0, 0, 0, 17, 18, 19, 20]);
        assert_eq!(&quantized[640..648], &[0, 0, 0, 0, 21, 22, 23, 24]);
        assert_eq!(&quantized[768..776], &[0, 0, 0, 0, 25, 26, 27, 28]);
        assert_eq!(&quantized[896..904], &[0, 0, 0, 0, 29, 30, 31, 32]);
        assert_eq!(spectral_decoder.pos, SHORT_SPECTRAL_VALUES.len());
    }

    #[test]
    fn decode_long_rejects_short_window_mapping() {
        let mut prefix = prefix_with_sections(100, 1, &[(0, 4), (1, 5)]);
        prefix.ics_info.window_sequence = WindowSequence::EightShort;
        let scale_factors = zero_scale_factors(&prefix);
        let mut coeffs = SpectralCoefficients::new(8);
        let mut spectral_decoder = NotImplementedSpectralDecoder;
        let mut reader = BitReader::new(&[]);

        assert_eq!(
            coeffs
                .decode_long(
                    &mut reader,
                    &prefix,
                    &scale_factors,
                    BandLayout::new(&[0, 8]),
                    &mut spectral_decoder,
                )
                .unwrap_err(),
            AacLcError::NotImplemented("short spectral coefficient mapping requires short layout")
        );
    }

    #[test]
    fn not_implemented_decoder_rejects_standard_spectral_tables() {
        let prefix = prefix_with_sections(100, 1, &[(1, 4), (1, 5)]);
        let mut sf_reader = BitReader::new(&[0]);
        let mut sf_decoder = VlcScaleFactorDecoder::new(VlcTable::new(&MINI_SF).unwrap());
        let scale_factors =
            ScaleFactorData::read(&mut sf_reader, &prefix, &mut sf_decoder).unwrap();
        let mut coeffs = SpectralCoefficients::new(8);
        let mut spectral_decoder = NotImplementedSpectralDecoder;
        let mut reader = BitReader::new(&[]);

        assert_eq!(
            coeffs
                .decode_long(
                    &mut reader,
                    &prefix,
                    &scale_factors,
                    BandLayout::new(&[0, 8]),
                    &mut spectral_decoder,
                )
                .unwrap_err(),
            AacLcError::NotImplemented("AAC spectral Huffman codebook")
        );
    }

    fn zero_scale_factors(prefix: &IndividualChannelStreamPrefix) -> ScaleFactorData {
        let mut reader = BitReader::new(&[]);
        let mut decoder = crate::scalefactor::NotImplementedScaleFactorDecoder;
        ScaleFactorData::read(&mut reader, prefix, &mut decoder).unwrap()
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

    fn prefix_with_short_sections(
        global_gain: u8,
        max_sfb: u8,
        group_lens: &[u8],
        section_fields: &[(u32, u8)],
    ) -> IndividualChannelStreamPrefix {
        let mut window_group_len = [0u8; crate::ics::MAX_WINDOW_GROUPS];
        window_group_len[..group_lens.len()].copy_from_slice(group_lens);
        let ics_info = IcsInfo {
            window_sequence: WindowSequence::EightShort,
            window_shape: WindowShape::Sine,
            max_sfb,
            num_windows: 8,
            num_window_groups: group_lens.len() as u8,
            window_group_len,
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

    fn energy(samples: &[f32]) -> f32 {
        samples.iter().map(|sample| sample * sample).sum()
    }
}
