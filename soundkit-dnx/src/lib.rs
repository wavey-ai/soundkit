// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Memory-safe Rust VC-3/DNxHD/DNxHR decoder. The bitstream structure, codec
// tables, and integer transform are derived from FFmpeg's LGPL DNxHD decoder
// at ca821e458aabe2fa211d9e94eac38cd69fe2ea09.

mod tables;

use std::fmt;

const MAX_FRAME_PIXELS: usize = 7680 * 4320;
const MIN_HEADER_BYTES: usize = 0x280;
const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DnxPlane {
    pub width: u32,
    pub height: u32,
    pub stride: usize,
    pub samples: Vec<u16>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DnxColorModel {
    /// Planes are ordered Y, Cb, Cr.
    Ycbcr,
    /// Planes are ordered G, B, R, matching the VC-3 4:4:4 bitstream.
    Gbr,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DnxFrame {
    pub width: u32,
    pub height: u32,
    pub bit_depth: u8,
    pub color_model: DnxColorModel,
    pub planes: [DnxPlane; 3],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DnxError(String);

impl DnxError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for DnxError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for DnxError {}

/// Decode one complete progressive DNxHR coding unit.
///
/// DNxHR 444, HQX, HQ, SQ, and LB are supported. Interlaced legacy VC-3 and
/// 12-bit DNxHR 444 still fail before allocating output planes.
pub fn decode_frame(data: &[u8]) -> Result<DnxFrame, DnxError> {
    let header = Header::parse(data)?;
    let supported = matches!(
        (header.cid, header.bit_depth, header.is_444),
        (1270, 10, true) | (1271, 10, false) | (1272..=1274, 8, false)
    );
    if !supported || header.interlaced {
        return Err(DnxError::new(format!(
            "DNx decoder supports progressive 10-bit DNxHR 444/HQX and 8-bit HQ/SQ/LB; got CID {}, {}-bit, 4:{}, interlaced={}",
            header.cid,
            header.bit_depth,
            if header.is_444 { "4:4" } else { "2:2" },
            header.interlaced,
        )));
    }
    Decoder::new(data, header)?.decode()
}

/// Backwards-compatible name for callers that require HQX specifically.
pub fn decode_hqx_frame(data: &[u8]) -> Result<DnxFrame, DnxError> {
    let frame = decode_frame(data)?;
    if read_u32(data, 0x28)? != 1271 {
        return Err(DnxError::new("DNx coding unit is not the HQX profile"));
    }
    Ok(frame)
}

#[derive(Clone, Copy, Debug)]
struct Header {
    width: usize,
    height: usize,
    mb_width: usize,
    mb_height: usize,
    data_offset: usize,
    cid: u32,
    bit_depth: u8,
    is_444: bool,
    adaptive_color_transform: bool,
    interlaced: bool,
    mbaff: bool,
}

#[derive(Clone, Copy)]
struct CodecTables {
    luma_weight: &'static [u8; 64],
    chroma_weight: &'static [u8; 64],
    dc_codes: &'static [u8],
    dc_bits: &'static [u8],
    ac_codes: &'static [u16],
    ac_bits: &'static [u8],
    ac_info: &'static [u8],
    run_codes: &'static [u16],
    run_bits: &'static [u8],
    run_values: &'static [u8],
    eob_index: u16,
    index_bits: usize,
    level_bias: i32,
    level_shift: u32,
    dc_shift: u32,
}

impl CodecTables {
    fn for_header(header: Header) -> Result<Self, DnxError> {
        match (header.cid, header.bit_depth) {
            (1270, 10) => Ok(Self {
                luma_weight: &tables::DNXHD_1235_LUMA_WEIGHT,
                chroma_weight: &tables::DNXHD_1235_LUMA_WEIGHT,
                dc_codes: &tables::DNXHD_1235_DC_CODES,
                dc_bits: &tables::DNXHD_1235_DC_BITS,
                ac_codes: &tables::DNXHD_1235_AC_CODES,
                ac_bits: &tables::DNXHD_1235_AC_BITS,
                ac_info: &tables::DNXHD_1235_AC_INFO,
                run_codes: &tables::DNXHD_1235_RUN_CODES,
                run_bits: &tables::DNXHD_1235_RUN_BITS,
                run_values: &tables::DNXHD_1235_RUN,
                eob_index: 4,
                index_bits: 6,
                level_bias: 32,
                level_shift: 6,
                dc_shift: 0,
            }),
            (1271, 10) => Ok(Self {
                luma_weight: &tables::DNXHD_1241_LUMA_WEIGHT,
                chroma_weight: &tables::DNXHD_1241_CHROMA_WEIGHT,
                dc_codes: &tables::DNXHD_1235_DC_CODES,
                dc_bits: &tables::DNXHD_1235_DC_BITS,
                ac_codes: &tables::DNXHD_1235_AC_CODES,
                ac_bits: &tables::DNXHD_1235_AC_BITS,
                ac_info: &tables::DNXHD_1235_AC_INFO,
                run_codes: &tables::DNXHD_1235_RUN_CODES,
                run_bits: &tables::DNXHD_1235_RUN_BITS,
                run_values: &tables::DNXHD_1235_RUN,
                eob_index: 4,
                index_bits: 6,
                level_bias: 32,
                level_shift: 6,
                dc_shift: 0,
            }),
            (1272, 8) => Ok(Self {
                luma_weight: &tables::DNXHD_1238_LUMA_WEIGHT,
                chroma_weight: &tables::DNXHD_1238_CHROMA_WEIGHT,
                dc_codes: &tables::DNXHD_1237_DC_CODES,
                dc_bits: &tables::DNXHD_1237_DC_BITS,
                ac_codes: &tables::DNXHD_1238_AC_CODES,
                ac_bits: &tables::DNXHD_1238_AC_BITS,
                ac_info: &tables::DNXHD_1238_AC_INFO,
                run_codes: &tables::DNXHD_1235_RUN_CODES,
                run_bits: &tables::DNXHD_1235_RUN_BITS,
                run_values: &tables::DNXHD_1238_RUN,
                eob_index: 4,
                index_bits: 4,
                level_bias: 32,
                level_shift: 6,
                dc_shift: 0,
            }),
            (1273 | 1274, 8) => Ok(Self {
                luma_weight: &tables::DNXHD_1237_LUMA_WEIGHT,
                chroma_weight: &tables::DNXHD_1237_CHROMA_WEIGHT,
                dc_codes: &tables::DNXHD_1237_DC_CODES,
                dc_bits: &tables::DNXHD_1237_DC_BITS,
                ac_codes: &tables::DNXHD_1237_AC_CODES,
                ac_bits: &tables::DNXHD_1237_AC_BITS,
                ac_info: &tables::DNXHD_1237_AC_INFO,
                run_codes: &tables::DNXHD_1237_RUN_CODES,
                run_bits: &tables::DNXHD_1237_RUN_BITS,
                run_values: &tables::DNXHD_1237_RUN,
                eob_index: 3,
                index_bits: 4,
                level_bias: 32,
                level_shift: 6,
                dc_shift: 0,
            }),
            _ => Err(DnxError::new(format!(
                "unsupported DNx profile/depth combination CID {} {}-bit",
                header.cid, header.bit_depth
            ))),
        }
    }
}

impl Header {
    fn parse(data: &[u8]) -> Result<Self, DnxError> {
        if data.len() < MIN_HEADER_BYTES {
            return Err(DnxError::new(
                "DNx coding unit is shorter than its 640-byte header",
            ));
        }
        let prefix = (u64::from(read_u32(data, 0)?) << 16) | (u64::from(data[4]) << 8);
        let prefix_offset = read_u16(data, 2)? as usize;
        let legacy = prefix == 0x0000_0280_0100 || prefix == 0x0000_0280_0200;
        let hr = (prefix & 0xffff_0000_ffff) == 0x0300
            && (0x280..=0x2170).contains(&prefix_offset)
            && prefix_offset.is_multiple_of(4);
        if !legacy && !hr {
            return Err(DnxError::new(format!(
                "invalid DNx header prefix 0x{prefix:012x}"
            )));
        }
        let width = read_u16(data, 0x1a)? as usize;
        let height = read_u16(data, 0x18)? as usize;
        let pixels = width
            .checked_mul(height)
            .ok_or_else(|| DnxError::new("DNx dimensions overflow"))?;
        if width == 0 || height == 0 || pixels > MAX_FRAME_PIXELS {
            return Err(DnxError::new(format!(
                "DNx frame {width}x{height} exceeds the pixel budget"
            )));
        }
        let bit_depth = match data[0x21] >> 5 {
            1 => 8,
            2 => 10,
            3 => 12,
            value => {
                return Err(DnxError::new(format!(
                    "unsupported DNx bit-depth indicator {value}"
                )))
            }
        };
        let cid = read_u32(data, 0x28)?;
        let mb_height = read_u16(data, 0x16c)? as usize;
        let mb_width = width.div_ceil(16);
        let interlaced = data[5] & 2 != 0;
        let data_offset = if mb_height > 68 && hr {
            0x170usize
                .checked_add(
                    mb_height
                        .checked_mul(4)
                        .ok_or_else(|| DnxError::new("DNx row table overflow"))?,
                )
                .ok_or_else(|| DnxError::new("DNx data offset overflow"))?
        } else {
            if mb_height > 68 {
                return Err(DnxError::new("legacy DNx macroblock height exceeds 68"));
            }
            MIN_HEADER_BYTES
        };
        if data.len() < data_offset {
            return Err(DnxError::new("DNx coding unit ends before macroblock data"));
        }
        let logical_mb_height = mb_height << usize::from(interlaced);
        if logical_mb_height > height.div_ceil(16) {
            return Err(DnxError::new("DNx macroblock rows exceed the coded height"));
        }
        for row in 0..mb_height {
            let offset = read_u32(data, 0x170 + row * 4)? as usize;
            if offset > data.len() - data_offset {
                return Err(DnxError::new(format!(
                    "DNx row {row} offset exceeds its coding unit"
                )));
            }
        }
        Ok(Self {
            width,
            height,
            mb_width,
            mb_height,
            data_offset,
            cid,
            bit_depth,
            is_444: data[0x2c] >> 6 & 1 != 0,
            adaptive_color_transform: data[0x2c] & 1 != 0,
            interlaced,
            mbaff: data[6] >> 5 & 1 != 0,
        })
    }
}

struct Decoder<'a> {
    data: &'a [u8],
    header: Header,
    tables: CodecTables,
    planes: [DnxPlane; 3],
    color_model: Option<DnxColorModel>,
    dc: Vlc,
    ac: Vlc,
    run: Vlc,
}

impl<'a> Decoder<'a> {
    fn new(data: &'a [u8], header: Header) -> Result<Self, DnxError> {
        let tables = CodecTables::for_header(header)?;
        let y_stride = header.width;
        let c_width = if header.is_444 {
            header.width
        } else {
            header.width.div_ceil(2)
        };
        let plane = |width: usize, stride: usize| -> Result<DnxPlane, DnxError> {
            let samples = stride
                .checked_mul(header.height)
                .ok_or_else(|| DnxError::new("DNx output plane size overflow"))?;
            Ok(DnxPlane {
                width: width as u32,
                height: header.height as u32,
                stride,
                samples: vec![0; samples],
            })
        };
        Ok(Self {
            data,
            header,
            tables,
            planes: [
                plane(header.width, y_stride)?,
                plane(c_width, c_width)?,
                plane(c_width, c_width)?,
            ],
            color_model: (!header.is_444).then_some(DnxColorModel::Ycbcr),
            dc: Vlc::new(tables.dc_codes, tables.dc_bits, None)?,
            ac: Vlc::new(tables.ac_codes, tables.ac_bits, None)?,
            run: Vlc::new(tables.run_codes, tables.run_bits, Some(tables.run_values))?,
        })
    }

    fn decode(mut self) -> Result<DnxFrame, DnxError> {
        if self.header.mbaff {
            return Err(DnxError::new(
                "DNxHR HQX macroblock-adaptive interlace is not supported",
            ));
        }
        for y in 0..self.header.mb_height {
            self.decode_row(y)?;
        }
        Ok(DnxFrame {
            width: self.header.width as u32,
            height: self.header.height as u32,
            bit_depth: self.header.bit_depth,
            color_model: self
                .color_model
                .ok_or_else(|| DnxError::new("DNx 4:4:4 frame has no color-model decision"))?,
            planes: self.planes,
        })
    }

    fn decode_row(&mut self, y: usize) -> Result<(), DnxError> {
        let row_offset = read_u32(self.data, 0x170 + y * 4)? as usize;
        let start = self
            .header
            .data_offset
            .checked_add(row_offset)
            .ok_or_else(|| DnxError::new("DNx row offset overflow"))?;
        let end = if y + 1 < self.header.mb_height {
            self.header.data_offset + read_u32(self.data, 0x170 + (y + 1) * 4)? as usize
        } else {
            self.data.len()
        };
        let bytes = self
            .data
            .get(start..end)
            .ok_or_else(|| DnxError::new(format!("DNx row {y} byte range is invalid")))?;
        let mut bits = BitReader::new(bytes);
        let mut last_dc = [1i32 << (self.header.bit_depth + 2); 3];
        let mut scales = ([0i32; 64], [0i32; 64]);
        let mut last_qscale = u16::MAX;
        for x in 0..self.header.mb_width {
            let qscale = bits.read(11)? as u16;
            let act = bits.read(1)? != 0;
            if self.header.is_444 {
                if act && !self.header.adaptive_color_transform {
                    return Err(DnxError::new(
                        "DNx macroblock enables ACT while the frame header disables it",
                    ));
                }
                let color_model = if act {
                    DnxColorModel::Ycbcr
                } else {
                    DnxColorModel::Gbr
                };
                if self
                    .color_model
                    .is_some_and(|current| current != color_model)
                {
                    return Err(DnxError::new(
                        "variable DNx adaptive color transforms are not supported",
                    ));
                }
                self.color_model = Some(color_model);
            }
            if qscale != last_qscale {
                for index in 0..64 {
                    scales.0[index] = i32::from(qscale) * i32::from(self.tables.luma_weight[index]);
                    scales.1[index] =
                        i32::from(qscale) * i32::from(self.tables.chroma_weight[index]);
                }
                last_qscale = qscale;
            }
            let mut blocks = [[0i16; 64]; 12];
            let block_count = if self.header.is_444 { 12 } else { 8 };
            for (n, block) in blocks.iter_mut().take(block_count).enumerate() {
                self.decode_block(&mut bits, n, block, &mut last_dc, &scales)?;
            }
            self.put_macroblock(x, y, &mut blocks);
        }
        Ok(())
    }

    fn decode_block(
        &self,
        bits: &mut BitReader<'_>,
        n: usize,
        block: &mut [i16; 64],
        last_dc: &mut [i32; 3],
        scales: &([i32; 64], [i32; 64]),
    ) -> Result<(), DnxError> {
        let (component, scale, weight) = if self.header.is_444 {
            let component = (n >> 1) % 3;
            if component == 0 {
                (component, &scales.0, self.tables.luma_weight)
            } else {
                (component, &scales.1, self.tables.chroma_weight)
            }
        } else if n & 2 != 0 {
            (1 + (n & 1), &scales.1, self.tables.chroma_weight)
        } else {
            (0, &scales.0, self.tables.luma_weight)
        };
        let dc_len = self.dc.decode(bits)?;
        if dc_len > 0 {
            let value = bits.read(dc_len as usize)? as i32;
            let delta = if value & (1 << (dc_len - 1)) == 0 {
                value - ((1 << dc_len) - 1)
            } else {
                value
            };
            last_dc[component] = last_dc[component]
                .checked_add(delta)
                .ok_or_else(|| DnxError::new("DNx DC predictor overflow"))?;
        }
        block[0] = i16::try_from(last_dc[component] << self.tables.dc_shift)
            .map_err(|_| DnxError::new("DNx DC coefficient overflow"))?;
        let mut i = 0usize;
        loop {
            let index = self.ac.decode(bits)? as usize;
            if index == self.tables.eob_index as usize {
                break;
            }
            let info_offset = index
                .checked_mul(2)
                .ok_or_else(|| DnxError::new("DNx AC index overflow"))?;
            let mut level = i32::from(
                *self
                    .tables
                    .ac_info
                    .get(info_offset)
                    .ok_or_else(|| DnxError::new("DNx AC symbol is out of range"))?,
            );
            let flags = *self
                .tables
                .ac_info
                .get(info_offset + 1)
                .ok_or_else(|| DnxError::new("DNx AC flags are out of range"))?;
            let negative = bits.read(1)? != 0;
            if flags & 1 != 0 {
                level += (bits.read(self.tables.index_bits)? as i32) << 7;
            }
            if flags & 2 != 0 {
                i = i
                    .checked_add(self.run.decode(bits)? as usize)
                    .ok_or_else(|| DnxError::new("DNx AC run overflow"))?;
            }
            i += 1;
            if i > 63 {
                return Err(DnxError::new("DNx AC run exceeds its 8x8 block"));
            }
            level *= scale[i];
            level += scale[i] >> 1;
            if self.tables.level_bias < 32 || i32::from(weight[i]) != self.tables.level_bias {
                level += self.tables.level_bias;
            }
            level >>= self.tables.level_shift;
            if negative {
                level = -level;
            }
            block[ZIGZAG[i]] =
                i16::try_from(level).map_err(|_| DnxError::new("DNx AC coefficient overflow"))?;
        }
        Ok(())
    }

    fn put_macroblock(&mut self, x: usize, y: usize, blocks: &mut [[i16; 64]; 12]) {
        let y_x = x * 16;
        let top = y * 16;
        let depth = self.header.bit_depth;
        if self.header.is_444 {
            for (plane, block_indices) in [[0, 1, 6, 7], [2, 3, 8, 9], [4, 5, 10, 11]]
                .into_iter()
                .enumerate()
            {
                put_block(
                    &mut self.planes[plane],
                    y_x,
                    top,
                    &mut blocks[block_indices[0]],
                    depth,
                );
                put_block(
                    &mut self.planes[plane],
                    y_x + 8,
                    top,
                    &mut blocks[block_indices[1]],
                    depth,
                );
                put_block(
                    &mut self.planes[plane],
                    y_x,
                    top + 8,
                    &mut blocks[block_indices[2]],
                    depth,
                );
                put_block(
                    &mut self.planes[plane],
                    y_x + 8,
                    top + 8,
                    &mut blocks[block_indices[3]],
                    depth,
                );
            }
        } else {
            let c_x = x * 8;
            put_block(&mut self.planes[0], y_x, top, &mut blocks[0], depth);
            put_block(&mut self.planes[0], y_x + 8, top, &mut blocks[1], depth);
            put_block(&mut self.planes[1], c_x, top, &mut blocks[2], depth);
            put_block(&mut self.planes[2], c_x, top, &mut blocks[3], depth);
            put_block(&mut self.planes[0], y_x, top + 8, &mut blocks[4], depth);
            put_block(&mut self.planes[0], y_x + 8, top + 8, &mut blocks[5], depth);
            put_block(&mut self.planes[1], c_x, top + 8, &mut blocks[6], depth);
            put_block(&mut self.planes[2], c_x, top + 8, &mut blocks[7], depth);
        }
    }
}

#[derive(Clone)]
struct Vlc {
    nodes: Vec<VlcNode>,
    max_bits: u8,
}

#[derive(Clone, Default)]
struct VlcNode {
    children: [Option<usize>; 2],
    value: Option<u16>,
}

impl Vlc {
    fn new<C: Copy + Into<u64>>(
        codes: &[C],
        lengths: &[u8],
        values: Option<&[u8]>,
    ) -> Result<Self, DnxError> {
        if codes.len() != lengths.len() || values.is_some_and(|items| items.len() != codes.len()) {
            return Err(DnxError::new("DNx VLC table lengths disagree"));
        }
        let mut nodes = vec![VlcNode::default()];
        let mut max_bits = 0;
        for (index, (&code, &bits)) in codes.iter().zip(lengths).enumerate() {
            if bits == 0 || bits > 16 {
                return Err(DnxError::new("DNx VLC code length is invalid"));
            }
            let code = u16::try_from(code.into())
                .map_err(|_| DnxError::new("DNx VLC code overflows u16"))?;
            if u32::from(code) >= 1u32 << bits {
                return Err(DnxError::new("DNx VLC code exceeds its declared length"));
            }
            let mut node_index = 0usize;
            for shift in (0..bits).rev() {
                if nodes[node_index].value.is_some() {
                    return Err(DnxError::new("DNx VLC table contains a prefix collision"));
                }
                let branch = usize::from((code >> shift) & 1);
                node_index = match nodes[node_index].children[branch] {
                    Some(child) => child,
                    None => {
                        let child = nodes.len();
                        nodes.push(VlcNode::default());
                        nodes[node_index].children[branch] = Some(child);
                        child
                    }
                };
            }
            if nodes[node_index].value.is_some()
                || nodes[node_index].children.iter().any(Option::is_some)
            {
                return Err(DnxError::new("DNx VLC table contains a prefix collision"));
            }
            nodes[node_index].value =
                Some(values.map_or(index as u16, |items| items[index] as u16));
            max_bits = max_bits.max(bits);
        }
        Ok(Self { nodes, max_bits })
    }

    fn decode(&self, bits: &mut BitReader<'_>) -> Result<u16, DnxError> {
        let mut node_index = 0usize;
        for _ in 0..self.max_bits {
            let branch = bits.read(1)? as usize;
            node_index = self.nodes[node_index].children[branch]
                .ok_or_else(|| DnxError::new("invalid DNx VLC code"))?;
            if let Some(value) = self.nodes[node_index].value {
                return Ok(value);
            }
        }
        Err(DnxError::new("invalid DNx VLC code"))
    }
}

struct BitReader<'a> {
    data: &'a [u8],
    bit: usize,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, bit: 0 }
    }

    fn read(&mut self, count: usize) -> Result<u32, DnxError> {
        if count > 32
            || self
                .bit
                .checked_add(count)
                .is_none_or(|end| end > self.data.len() * 8)
        {
            return Err(DnxError::new("DNx row bitstream is truncated"));
        }
        let mut value = 0u32;
        for _ in 0..count {
            let byte = self.data[self.bit / 8];
            value = (value << 1) | u32::from((byte >> (7 - self.bit % 8)) & 1);
            self.bit += 1;
        }
        Ok(value)
    }
}

fn put_block(plane: &mut DnxPlane, x: usize, y: usize, block: &mut [i16; 64], bit_depth: u8) {
    idct(block, bit_depth);
    let maximum = (1i16 << bit_depth) - 1;
    let width = plane.width as usize;
    let height = plane.height as usize;
    for row in 0..8 {
        if y + row >= height {
            break;
        }
        for column in 0..8 {
            if x + column >= width {
                break;
            }
            plane.samples[(y + row) * plane.stride + x + column] =
                block[row * 8 + column].clamp(0, maximum) as u16;
        }
    }
}

// FFmpeg's scalar 10-bit simple IDCT, translated to fixed-width Rust. Keeping
// the integer transform makes native and WASM output byte-identical.
fn idct(block: &mut [i16; 64], bit_depth: u8) {
    let parameters = IdctParameters::for_depth(bit_depth);
    for row in 0..8 {
        idct_row(&mut block[row * 8..row * 8 + 8], parameters);
    }
    for column in 0..8 {
        idct_column(block, column, parameters);
    }
}

#[derive(Clone, Copy)]
struct IdctParameters {
    weights: [i64; 7],
    row_shift: i64,
    column_shift: i64,
    dc_shift: u32,
}

impl IdctParameters {
    fn for_depth(bit_depth: u8) -> Self {
        match bit_depth {
            8 => Self {
                weights: [22725, 21407, 19266, 16383, 12873, 8867, 4520],
                row_shift: 11,
                column_shift: 20,
                dc_shift: 3,
            },
            10 => Self {
                weights: [22725, 21407, 19265, 16384, 12873, 8867, 4520],
                row_shift: 12,
                column_shift: 19,
                dc_shift: 2,
            },
            _ => unreachable!("unsupported DNx IDCT depth"),
        }
    }
}

fn idct_row(row: &mut [i16], parameters: IdctParameters) {
    let [w1, w2, w3, w4, w5, w6, w7] = parameters.weights;
    let shift = parameters.row_shift;
    if row[1..].iter().all(|value| *value == 0) {
        let value = i32::from(row[0]) * (1 << parameters.dc_shift);
        row.fill(value as i16);
        return;
    }
    let mut a0 = w4 * i64::from(row[0]) + (1 << (shift - 1));
    let mut a1 = a0;
    let mut a2 = a0;
    let mut a3 = a0;
    a0 += w2 * i64::from(row[2]);
    a1 += w6 * i64::from(row[2]);
    a2 -= w6 * i64::from(row[2]);
    a3 -= w2 * i64::from(row[2]);
    let mut b0 = w1 * i64::from(row[1]) + w3 * i64::from(row[3]);
    let mut b1 = w3 * i64::from(row[1]) - w7 * i64::from(row[3]);
    let mut b2 = w5 * i64::from(row[1]) - w1 * i64::from(row[3]);
    let mut b3 = w7 * i64::from(row[1]) - w5 * i64::from(row[3]);
    a0 += w4 * i64::from(row[4]) + w6 * i64::from(row[6]);
    a1 -= w4 * i64::from(row[4]) + w2 * i64::from(row[6]);
    a2 += -w4 * i64::from(row[4]) + w2 * i64::from(row[6]);
    a3 += w4 * i64::from(row[4]) - w6 * i64::from(row[6]);
    b0 += w5 * i64::from(row[5]) + w7 * i64::from(row[7]);
    b1 -= w1 * i64::from(row[5]) + w5 * i64::from(row[7]);
    b2 += w7 * i64::from(row[5]) + w3 * i64::from(row[7]);
    b3 += w3 * i64::from(row[5]) - w1 * i64::from(row[7]);
    let values = [
        a0 + b0,
        a1 + b1,
        a2 + b2,
        a3 + b3,
        a3 - b3,
        a2 - b2,
        a1 - b1,
        a0 - b0,
    ];
    for (target, value) in row.iter_mut().zip(values) {
        *target = wrapping_shift(value, shift) as i16;
    }
}

fn idct_column(block: &mut [i16; 64], column: usize, parameters: IdctParameters) {
    let [w1, w2, w3, w4, w5, w6, w7] = parameters.weights;
    let shift = parameters.column_shift;
    let get = |row: usize| i64::from(block[row * 8 + column]);
    let mut a0 = w4 * (get(0) + ((1 << (shift - 1)) / w4));
    let mut a1 = a0;
    let mut a2 = a0;
    let mut a3 = a0;
    a0 += w2 * get(2) + w4 * get(4) + w6 * get(6);
    a1 += w6 * get(2) - w4 * get(4) - w2 * get(6);
    a2 += -w6 * get(2) - w4 * get(4) + w2 * get(6);
    a3 += -w2 * get(2) + w4 * get(4) - w6 * get(6);
    let b0 = w1 * get(1) + w3 * get(3) + w5 * get(5) + w7 * get(7);
    let b1 = w3 * get(1) - w7 * get(3) - w1 * get(5) - w5 * get(7);
    let b2 = w5 * get(1) - w1 * get(3) + w7 * get(5) + w3 * get(7);
    let b3 = w7 * get(1) - w5 * get(3) + w3 * get(5) - w1 * get(7);
    let values = [
        a0 + b0,
        a1 + b1,
        a2 + b2,
        a3 + b3,
        a3 - b3,
        a2 - b2,
        a1 - b1,
        a0 - b0,
    ];
    for (row, value) in values.into_iter().enumerate() {
        block[row * 8 + column] = wrapping_shift(value, shift) as i16;
    }
}

fn wrapping_shift(value: i64, shift: i64) -> i32 {
    (value as u32 as i32) >> shift
}

fn read_u16(data: &[u8], offset: usize) -> Result<u16, DnxError> {
    let bytes = data
        .get(offset..offset + 2)
        .ok_or_else(|| DnxError::new("truncated DNx header"))?;
    Ok(u16::from_be_bytes([bytes[0], bytes[1]]))
}

fn read_u32(data: &[u8], offset: usize) -> Result<u32, DnxError> {
    let bytes = data
        .get(offset..offset + 4)
        .ok_or_else(|| DnxError::new("truncated DNx header"))?;
    Ok(u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    fn reference_coding_unit(file: &str, coding_unit_size: usize) -> Vec<u8> {
        let container = std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../testdata/video-compat/never-final")
                .join(file),
        )
        .unwrap();
        let prefix = [0x00, 0x00, 0x02, 0x80, 0x03];
        let start = container
            .windows(prefix.len())
            .position(|window| window == prefix)
            .expect("DNxHR coding-unit prefix");
        container[start..start + coding_unit_size].to_vec()
    }

    fn frame_hash(frame: &DnxFrame) -> String {
        let mut digest = Sha256::new();
        for plane in &frame.planes {
            for sample in &plane.samples {
                digest.update(sample.to_le_bytes());
            }
        }
        format!("{:x}", digest.finalize())
    }

    #[test]
    fn decodes_reference_dnxhr_hqx_frame() {
        let frame = decode_hqx_frame(&reference_coding_unit("dnxhr-hqx-pcm.mov", 102_400)).unwrap();
        assert_eq!((frame.width, frame.height, frame.bit_depth), (640, 360, 10));
        assert_eq!((frame.planes[0].width, frame.planes[0].height), (640, 360));
        assert_eq!((frame.planes[1].width, frame.planes[1].height), (320, 360));

        assert_eq!(
            frame_hash(&frame),
            "32c0a789031d1f47b9e16c252acda863510fb71b1c1105b9a895da932448a154"
        );
    }

    #[test]
    fn decodes_dnxhr_444_color_models_byte_identically() {
        let profiles = [
            (
                "dnxhr-444-gbr10-pcm.mov",
                DnxColorModel::Gbr,
                "8b3d3bd87f40d4351503b4719e8f1681a8de8c107a6f0f096f81251fd5c9aedb",
            ),
            (
                "dnxhr-444-yuv10-pcm.mov",
                DnxColorModel::Ycbcr,
                "32d6262761eb97cac6650c2ba14f301e0e0d0d103a3057be37a585ab4ff964c2",
            ),
        ];
        for (file, color_model, expected_hash) in profiles {
            let frame = decode_frame(&reference_coding_unit(file, 208_896)).unwrap();
            assert_eq!((frame.width, frame.height, frame.bit_depth), (640, 360, 10));
            assert_eq!(frame.color_model, color_model);
            assert!(frame
                .planes
                .iter()
                .all(|plane| (plane.width, plane.height) == (640, 360)));
            assert_eq!(frame_hash(&frame), expected_hash, "{file}");
        }
    }

    #[test]
    fn rejects_444_act_that_contradicts_the_frame_header() {
        let mut unit = reference_coding_unit("dnxhr-444-gbr10-pcm.mov", 208_896);
        let data_offset = read_u16(&unit, 2).unwrap() as usize;
        unit[data_offset + 1] |= 0x10;
        let error = decode_frame(&unit).unwrap_err();
        assert!(error.to_string().contains("header disables"));
    }

    #[test]
    fn rejects_variable_444_color_models() {
        let mut unit = reference_coding_unit("dnxhr-444-yuv10-pcm.mov", 208_896);
        let data_offset = read_u16(&unit, 2).unwrap() as usize;
        unit[data_offset + 1] &= !0x10;
        let error = decode_frame(&unit).unwrap_err();
        assert!(error.to_string().contains("variable"));
    }

    #[test]
    fn decodes_all_422_dnxhr_profiles_byte_identically() {
        // FFmpeg's scalar C output (`-cpuflags 0`) is the cross-platform
        // reference. Some SIMD IDCTs are intentionally only approximately
        // conformant and can differ by one in a small number of chroma pixels.
        let profiles = [
            (
                "dnxhr-hq-pcm.mov",
                102_400,
                "048892a9aa9159d01a4961fc93315e72ccef855d80ffeb33a7dd63df1ce43295",
            ),
            (
                "dnxhr-sq-pcm.mov",
                69_632,
                "f600249c1fa3f723f7f865098bf91c0fa3a6a95a472cf351af930a08c13e127c",
            ),
            (
                "dnxhr-lb-pcm.mov",
                20_480,
                "5a702a21494bd1507dd528c56aa7a4029fee6d7029212859a369f9126e91092b",
            ),
        ];
        for (file, coding_unit_size, expected_hash) in profiles {
            let frame = decode_frame(&reference_coding_unit(file, coding_unit_size)).unwrap();
            assert_eq!((frame.width, frame.height, frame.bit_depth), (640, 360, 8));
            let mut digest = Sha256::new();
            for plane in &frame.planes {
                digest.update(
                    plane
                        .samples
                        .iter()
                        .map(|sample| *sample as u8)
                        .collect::<Vec<_>>(),
                );
            }
            assert_eq!(format!("{:x}", digest.finalize()), expected_hash, "{file}");
        }
    }

    #[test]
    fn rejects_truncation_before_allocating_pixels() {
        let unit = reference_coding_unit("dnxhr-hqx-pcm.mov", 102_400);
        let error = decode_hqx_frame(&unit[..0x280]).unwrap_err();
        assert!(error.to_string().contains("offset"));
    }
}
