//! SoundKit's in-tree MPEG Layer III frame decoder.
//!
//! Frame parsing, reservoir handling, Huffman decoding, requantization, IMDCT,
//! and scalar synthesis use bounded Rust slices. Native x86-64 builds select an
//! authored AVX2 synthesis kernel at runtime and retain an SSE2 fallback.

mod tables;
use core::iter;

use tables::*;

#[derive(Default)]
pub(super) struct CoreFrameInfo {
    pub frame_bytes: usize,
    pub frame_offset: usize,
    pub channels: u32,
    pub sample_rate: u32,
    pub layer: u8,
    pub bitrate_kbps: u32,
}

pub(super) struct Layer3Decoder {
    mdct_overlap: [[f32; 288]; 2],
    qmf_state: [f32; 960],
    reserv: i32,
    free_format_bytes: usize,
    header: [u8; 4],
    reserv_buf: [u8; 511],
}

impl Layer3Decoder {
    pub const fn new() -> Self {
        Self {
            mdct_overlap: [[0.; 288]; 2],
            qmf_state: [0.; 960],
            reserv: 0,
            free_format_bytes: 0,
            header: [0; 4],
            reserv_buf: [0; 511],
        }
    }
}

struct DecodeScratch {
    grbuf: [[f32; 576]; 2],
    scf: [f32; 40],
    syn: [[f32; 64]; 33],
    ist_pos: [[u8; 39]; 2],
}
#[derive(Copy, Clone)]
struct GranuleInfo {
    sfbtab: &'static [u8],
    part_23_length: u16,
    big_values: u16,
    scalefac_compress: u16,
    global_gain: u8,
    block_type: u8,
    mixed_block_flag: u8,
    n_long_sfb: u8,
    n_short_sfb: u8,
    table_select: [u8; 3],
    region_count: [u8; 3],
    subblock_gain: [u8; 3],
    preflag: u8,
    scalefac_scale: u8,
    count1_table: u8,
    scfsi: u8,
}
struct BitReader<'a> {
    buf: &'a [u8],
    pos: i32,
    limit: i32,
}
fn bit_reader(data: &[u8], bytes: i32) -> BitReader<'_> {
    BitReader {
        buf: data,
        pos: 0,
        limit: bytes * 8,
    }
}

fn get_bits(bs: &mut BitReader, n: i32) -> u32 {
    let mut cache = 0u32;
    let bit_offset = (bs.pos & 7) as u32;
    let mut shift = n + bit_offset as i32;
    let mut byte = (bs.pos >> 3) as usize;
    bs.pos += n;
    if bs.pos > bs.limit {
        return 0;
    }

    let mut next = u32::from(bs.buf[byte]) & (0xff >> bit_offset);
    byte += 1;
    loop {
        shift -= 8;
        if shift <= 0 {
            break;
        }
        cache |= next << shift;
        next = u32::from(bs.buf[byte]);
        byte += 1;
    }
    cache | next >> -shift
}

fn hdr_valid(h: &[u8]) -> bool {
    h[0] == 0xff
        && (h[1] & 0xf0 == 0xf0 || h[1] & 0xfe == 0xe2)
        && h[1] >> 1 & 3 != 0
        && h[2] >> 4 != 15
        && h[2] >> 2 & 3 != 3
}

fn hdr_compare(h1: &[u8], h2: &[u8]) -> bool {
    hdr_valid(h2)
        && (h1[1] ^ h2[1]) & 0xfe == 0
        && (h1[2] ^ h2[2]) & 0xc == 0
        && (h1[2] & 0xf0 == 0) == (h2[2] & 0xf0 == 0)
}

fn hdr_bitrate_kbps(h: &[u8]) -> u32 {
    2 * (HDR_BITRATE_KBPS_HALFRATE[(h[1] & 0x8 != 0) as usize][((h[1] >> 1 & 3) - 1) as usize]
        [(h[2] >> 4) as usize] as u32)
}

fn hdr_sample_rate_hz(h: &[u8]) -> u32 {
    HDR_SAMPLE_RATE_HZ_G_HZ[(h[2] >> 2 & 3) as usize]
        >> (h[1] & 0x8 == 0) as u32
        >> (h[1] & 0x10 == 0) as u32
}

fn hdr_frame_samples(h: &[u8]) -> u32 {
    if h[1] & 6 == 6 {
        384
    } else {
        1152 >> (h[1] & 14 == 2) as u32
    }
}

fn hdr_frame_bytes(h: &[u8], free_format_size: usize) -> usize {
    let mut frame_bytes = (hdr_frame_samples(h))
        .wrapping_mul(hdr_bitrate_kbps(h))
        .wrapping_mul(125)
        .wrapping_div(hdr_sample_rate_hz(h)) as usize;
    if h[1] & 6 == 6 {
        frame_bytes &= !3;
    }
    if frame_bytes != 0 {
        frame_bytes
    } else {
        free_format_size
    }
}

fn hdr_padding(h: &[u8]) -> usize {
    if h[2] & 0x2 != 0 {
        if h[1] & 6 == 6 {
            4
        } else {
            1
        }
    } else {
        0
    }
}

fn read_side_info(bs: &mut BitReader, gr: &mut [GranuleInfo], hdr: &[u8]) -> i32 {
    let mut scfsi = 0u32;
    let mut part_23_sum = 0;
    let mut sr_idx = (i32::from(hdr[2]) >> 2 & 3)
        + ((i32::from(hdr[1]) >> 3 & 1) + (i32::from(hdr[1]) >> 4 & 1)) * 3;
    sr_idx -= i32::from(sr_idx != 0);
    let mut gr_count = if hdr[3] & 0xc0 == 0xc0 {
        1usize
    } else {
        2usize
    };
    let main_data_begin = if hdr[1] & 0x8 != 0 {
        gr_count *= 2;
        let value = get_bits(bs, 9) as i32;
        scfsi = get_bits(bs, 7 + gr_count as i32);
        value
    } else {
        (get_bits(bs, 8 + gr_count as i32) >> gr_count) as i32
    };

    for info in gr.iter_mut().take(gr_count) {
        if hdr[3] & 0xc0 == 0xc0 {
            scfsi <<= 4;
        }
        info.part_23_length = get_bits(bs, 12) as u16;
        part_23_sum += i32::from(info.part_23_length);
        info.big_values = get_bits(bs, 9) as u16;
        if info.big_values > 288 {
            return -1;
        }
        info.global_gain = get_bits(bs, 8) as u8;
        info.scalefac_compress = get_bits(bs, if hdr[1] & 0x8 != 0 { 4 } else { 9 }) as u16;
        info.sfbtab = &L3_READ_SIDE_INFO_G_SCF_LONG[sr_idx as usize];
        info.n_long_sfb = 22;
        info.n_short_sfb = 0;
        let tables = if get_bits(bs, 1) != 0 {
            info.block_type = get_bits(bs, 2) as u8;
            if info.block_type == 0 {
                return -1;
            }
            info.mixed_block_flag = get_bits(bs, 1) as u8;
            info.region_count[0] = 7;
            info.region_count[1] = 255;
            if info.block_type == 2 {
                scfsi &= 0xf0f;
                if info.mixed_block_flag == 0 {
                    info.region_count[0] = 8;
                    info.sfbtab = &L3_READ_SIDE_INFO_G_SCF_SHORT[sr_idx as usize];
                    info.n_long_sfb = 0;
                    info.n_short_sfb = 39;
                } else {
                    info.sfbtab = &L3_READ_SIDE_INFO_G_SCF_MIXED[sr_idx as usize];
                    info.n_long_sfb = if hdr[1] & 0x8 != 0 { 8 } else { 6 };
                    info.n_short_sfb = 30;
                }
            }
            let tables = get_bits(bs, 10) << 5;
            info.subblock_gain[0] = get_bits(bs, 3) as u8;
            info.subblock_gain[1] = get_bits(bs, 3) as u8;
            info.subblock_gain[2] = get_bits(bs, 3) as u8;
            tables
        } else {
            info.block_type = 0;
            info.mixed_block_flag = 0;
            let tables = get_bits(bs, 15);
            info.region_count[0] = get_bits(bs, 4) as u8;
            info.region_count[1] = get_bits(bs, 3) as u8;
            info.region_count[2] = 255;
            tables
        };
        info.table_select[0] = (tables >> 10) as u8;
        info.table_select[1] = (tables >> 5 & 31) as u8;
        info.table_select[2] = (tables & 31) as u8;
        info.preflag = (if hdr[1] & 0x8 != 0 {
            get_bits(bs, 1)
        } else {
            u32::from(info.scalefac_compress >= 500)
        }) as u8;
        info.scalefac_scale = get_bits(bs, 1) as u8;
        info.count1_table = get_bits(bs, 1) as u8;
        info.scfsi = (scfsi >> 12 & 15) as u8;
        scfsi <<= 4;
    }
    if part_23_sum + bs.pos > bs.limit + main_data_begin * 8 {
        return -1;
    }
    main_data_begin
}
fn read_scalefactors(
    scalefactors: &mut [u8; 40],
    intensity_positions: &mut [u8; 39],
    bit_widths: &[u8; 4],
    partition_counts: &[u8],
    bitstream: &mut BitReader,
    mut scfsi: i32,
) {
    let mut offset = 0usize;
    for partition in 0..4 {
        let count = usize::from(partition_counts[partition]);
        if count == 0 {
            break;
        }
        debug_assert!(offset + count <= intensity_positions.len());

        if scfsi & 8 != 0 {
            scalefactors[offset..offset + count]
                .copy_from_slice(&intensity_positions[offset..offset + count]);
        } else {
            let bits = i32::from(bit_widths[partition]);
            if bits == 0 {
                scalefactors[offset..offset + count].fill(0);
                intensity_positions[offset..offset + count].fill(0);
            } else {
                let intensity_escape = if scfsi < 0 { (1 << bits) - 1 } else { -1 };
                for index in offset..offset + count {
                    let value = get_bits(bitstream, bits) as i32;
                    scalefactors[index] = value as u8;
                    intensity_positions[index] = if value == intensity_escape {
                        u8::MAX
                    } else {
                        value as u8
                    };
                }
            }
        }

        offset += count;
        scfsi *= 2;
    }
    scalefactors[offset..offset + 3].fill(0);
}
fn scale_pow2_quarter(mut y: f32, mut exp_q2: i32) -> f32 {
    loop {
        let e = exp_q2.min(120);
        y *= L3_LDEXP_Q2_G_EXPFRAC[(e & 3) as usize] * ((1_i32 << 30) >> (e >> 2)) as f32;
        exp_q2 -= e;
        if exp_q2 <= 0 {
            break;
        }
    }
    y
}
fn decode_scalefactors(
    hdr: &[u8],
    intensity_positions: &mut [u8; 39],
    bs: &mut BitReader,
    gr: &GranuleInfo,
    scales: &mut [f32; 40],
    ch: u32,
) {
    let partition_row = usize::from(gr.n_short_sfb != 0) + usize::from(gr.n_long_sfb == 0);
    let mut partition_offset = 0usize;
    let mut scf_size: [u8; 4] = [0; 4];
    let mut iscf: [u8; 40] = [0; 40];
    let scf_shift = i32::from(gr.scalefac_scale) + 1;
    let mut scfsi = i32::from(gr.scfsi);
    if hdr[1] & 0x8 != 0 {
        let part =
            i32::from(L3_DECODE_SCALEFACTORS_G_SCFC_DECODE[usize::from(gr.scalefac_compress)]);
        scf_size[0] = (part >> 2) as u8;
        scf_size[1] = scf_size[0];
        scf_size[2] = (part & 3) as u8;
        scf_size[3] = scf_size[2];
    } else {
        let intensity_stereo = i32::from(hdr[3] & 0x10 != 0 && ch != 0);
        let mut compressed = i32::from(gr.scalefac_compress) >> intensity_stereo;
        let mut modulus_offset = intensity_stereo as usize * 12;
        while compressed >= 0 {
            let mut product = 1;
            for index in (0..4).rev() {
                let modulus = i32::from(L3_DECODE_SCALEFACTORS_G_MOD[modulus_offset + index]);
                scf_size[index] = (compressed / product % modulus) as u8;
                product *= modulus;
            }
            compressed -= product;
            modulus_offset += 4;
            partition_offset += 4;
        }
        scfsi = -16;
    }
    read_scalefactors(
        &mut iscf,
        intensity_positions,
        &scf_size,
        &L3_DECODE_SCALEFACTORS_G_SCM_PARTITIONS[partition_row][partition_offset..],
        bs,
        scfsi,
    );
    if gr.n_short_sfb != 0 {
        let shift = 3 - scf_shift;
        let start = usize::from(gr.n_long_sfb);
        for band in (0..usize::from(gr.n_short_sfb)).step_by(3) {
            for window in 0..3 {
                iscf[start + band + window] = (i32::from(iscf[start + band + window])
                    + (i32::from(gr.subblock_gain[window]) << shift))
                    as u8;
            }
        }
    } else if gr.preflag != 0 {
        for (value, &preamp) in iscf[11..21]
            .iter_mut()
            .zip(L3_DECODE_SCALEFACTORS_G_PREAMP.iter())
        {
            *value += preamp;
        }
    }
    let gain_exponent = i32::from(gr.global_gain) - 4 - 210 - i32::from(hdr[3] & 0xe0 == 0x60) * 2;
    let gain = scale_pow2_quarter(2048.0, 44 - gain_exponent);
    let band_count = usize::from(gr.n_long_sfb + gr.n_short_sfb);
    for index in 0..band_count {
        scales[index] = scale_pow2_quarter(gain, i32::from(iscf[index]) << scf_shift);
    }
}

fn pow_four_thirds(mut x: i32) -> f32 {
    let mut multiplier = 256;
    if x < 129 {
        return G_POW43[(16 + x) as usize];
    }
    if x < 1024 {
        multiplier = 16;
        x <<= 3;
    }
    let sign = (2 * x) & 64;
    let fraction = ((x & 63) - sign) as f32 / ((x & !63) + sign) as f32;
    G_POW43[(16 + ((x + sign) >> 6)) as usize]
        * (1.0 + fraction * (4.0 / 3.0 + fraction * (2.0 / 9.0)))
        * multiplier as f32
}
struct HuffmanBits<'a> {
    data: &'a [u8],
    cache: u32,
    shift: i32,
    next: usize,
}

impl<'a> HuffmanBits<'a> {
    #[inline(always)]
    fn new(bs: &BitReader<'a>) -> Self {
        let start = (bs.pos / 8) as usize;
        let bytes = &bs.buf[start..start + 4];
        let cache =
            u32::from_be_bytes(bytes.try_into().expect("four-byte Huffman cache")) << (bs.pos & 7);
        Self {
            data: bs.buf,
            cache,
            shift: (bs.pos & 7) - 8,
            next: start + 4,
        }
    }

    #[inline(always)]
    fn consume(&mut self, bits: i32) {
        self.cache <<= bits;
        self.shift += bits;
    }

    #[inline(always)]
    fn refill(&mut self) {
        while self.shift >= 0 {
            self.cache |= u32::from(self.data[self.next]) << self.shift;
            self.next += 1;
            self.shift -= 8;
        }
    }

    #[inline(always)]
    fn decode_leaf(&mut self, codebook: &[i16]) -> i32 {
        let mut width = 5;
        let mut leaf = i32::from(codebook[(self.cache >> (32 - width)) as usize]);
        while leaf < 0 {
            self.consume(width);
            width = leaf & 7;
            let index = (self.cache >> (32 - width)).wrapping_sub((leaf >> 3) as u32);
            leaf = i32::from(codebook[index as usize]);
        }
        self.consume(leaf >> 8);
        leaf
    }

    #[inline(always)]
    fn signed_unit(&self, magnitude: f32) -> f32 {
        if (self.cache as i32) < 0 {
            -magnitude
        } else {
            magnitude
        }
    }

    #[inline(always)]
    fn dequantize_plain(&mut self, magnitude: i32, scale: f32) -> f32 {
        let sign = (self.cache >> 31) as i32;
        let pow_index = 16 + magnitude - 16 * sign;
        let value = G_POW43[pow_index as usize] * scale;
        if magnitude != 0 {
            self.consume(1);
        }
        value
    }

    #[inline(always)]
    fn dequantize_linbits(&mut self, mut magnitude: i32, linbits: i32, scale: f32) -> f32 {
        let value = if magnitude == 15 {
            magnitude += (self.cache >> (32 - linbits)) as i32;
            self.consume(linbits);
            self.refill();
            scale * pow_four_thirds(magnitude) * self.signed_unit(1.0)
        } else {
            let sign = (self.cache >> 31) as i32;
            let pow_index = 16 + magnitude - 16 * sign;
            G_POW43[pow_index as usize] * scale
        };
        if magnitude != 0 {
            self.consume(1);
        }
        value
    }
}

fn decode_huffman(
    dst: &mut [f32; 576],
    bs: &mut BitReader,
    gr_info: &GranuleInfo,
    scf: &[f32; 40],
    layer3gr_limit: i32,
) {
    let mut bits = HuffmanBits::new(bs);
    let mut dst_index = 0usize;
    let mut scale_index = 0usize;
    let mut sfb_index = 0usize;
    let mut one = 0.0f32;
    let mut big_value_pairs = i32::from(gr_info.big_values);

    for region in 0..3 {
        if big_value_pairs <= 0 {
            break;
        }
        let table = usize::from(gr_info.table_select[region]);
        let codebook = &L3_HUFFMAN_TABS[L3_HUFFMAN_TABINDEX[table] as usize..];
        let linbits = i32::from(L3_HUFFMAN_G_LINBITS[table]);
        let mut bands_left = i32::from(gr_info.region_count[region]);

        loop {
            let band_pairs = i32::from(gr_info.sfbtab[sfb_index]) / 2;
            sfb_index += 1;
            let pairs_to_decode = big_value_pairs.min(band_pairs);
            one = scf[scale_index];
            scale_index += 1;

            if linbits == 0 {
                for _ in 0..pairs_to_decode {
                    let leaf = bits.decode_leaf(codebook);
                    dst[dst_index] = bits.dequantize_plain(leaf & 0xf, one);
                    dst[dst_index + 1] = bits.dequantize_plain((leaf >> 4) & 0xf, one);
                    dst_index += 2;
                    bits.refill();
                }
            } else {
                for _ in 0..pairs_to_decode {
                    let leaf = bits.decode_leaf(codebook);
                    dst[dst_index] = bits.dequantize_linbits(leaf & 0xf, linbits, one);
                    dst[dst_index + 1] = bits.dequantize_linbits((leaf >> 4) & 0xf, linbits, one);
                    dst_index += 2;
                    bits.refill();
                }
            }

            big_value_pairs -= band_pairs;
            if big_value_pairs <= 0 {
                break;
            }
            bands_left -= 1;
            if bands_left < 0 {
                break;
            }
        }
    }

    let count1_codebook = if gr_info.count1_table != 0 {
        &L3_HUFFMAN_TAB33[..]
    } else {
        &L3_HUFFMAN_TAB32[..]
    };
    let mut pairs_left_in_band = 1 - big_value_pairs;
    loop {
        let mut leaf = i32::from(count1_codebook[(bits.cache >> 28) as usize]);
        if leaf & 8 == 0 {
            let index = ((leaf >> 3) as u32).wrapping_add(bits.cache << 4 >> (32 - (leaf & 3)));
            leaf = i32::from(count1_codebook[index as usize]);
        }
        bits.consume(leaf & 7);
        if bits.next as i32 * 8 - 24 + bits.shift > layer3gr_limit || dst_index + 3 >= dst.len() {
            break;
        }

        pairs_left_in_band -= 1;
        if pairs_left_in_band == 0 {
            pairs_left_in_band = i32::from(gr_info.sfbtab[sfb_index]) / 2;
            sfb_index += 1;
            if pairs_left_in_band == 0 {
                break;
            }
            one = scf[scale_index];
            scale_index += 1;
        }
        if leaf & 128 != 0 {
            dst[dst_index] = bits.signed_unit(one);
            bits.consume(1);
        }
        if leaf & 64 != 0 {
            dst[dst_index + 1] = bits.signed_unit(one);
            bits.consume(1);
        }

        pairs_left_in_band -= 1;
        if pairs_left_in_band == 0 {
            pairs_left_in_band = i32::from(gr_info.sfbtab[sfb_index]) / 2;
            sfb_index += 1;
            if pairs_left_in_band == 0 {
                break;
            }
            one = scf[scale_index];
            scale_index += 1;
        }
        if leaf & 32 != 0 {
            dst[dst_index + 2] = bits.signed_unit(one);
            bits.consume(1);
        }
        if leaf & 16 != 0 {
            dst[dst_index + 3] = bits.signed_unit(one);
            bits.consume(1);
        }
        bits.refill();
        dst_index += 4;
    }
    bs.pos = layer3gr_limit;
}

fn apply_mid_side_stereo(left: &mut [f32], n: usize) {
    let (left, right) = left.split_at_mut(576);
    for (l, r) in iter::zip(left, right).take(n) {
        let a = *l;
        let b = *r;
        *l = a + b;
        *r = a - b;
    }
}

fn apply_intensity_stereo_band(left: &mut [f32], n: usize, kl: f32, kr: f32) {
    for i in 0..n {
        left[i + 576] = left[i] * kr;
        left[i] *= kl;
    }
}

fn stereo_top_band(right: &[f32], sfb: &[u8], nbands: usize, max_band: &mut [i32; 3]) {
    max_band.fill(-1);
    let mut offset = 0usize;
    for band in 0..nbands {
        let width = usize::from(sfb[band]);
        let values = &right[offset..offset + width];
        let mut index = 0;
        while index < width {
            if values[index] != 0.0 || values[index + 1] != 0.0 {
                max_band[band % 3] = band as i32;
                break;
            }
            index += 2;
        }
        offset += width;
    }
}
fn process_stereo_band(
    left: &mut [f32],
    ist_pos: &[u8; 39],
    sfb: &[u8],
    hdr: &[u8],
    max_band: &[i32; 3],
    mpeg2_sh: i32,
) {
    let max_pos = if hdr[1] & 0x8 != 0 { 7 } else { 64 };
    let mut offset = 0usize;
    for (band, &width) in sfb.iter().enumerate().take_while(|(_, width)| **width != 0) {
        let width = usize::from(width);
        let ipos = u32::from(ist_pos[band]);
        let samples = &mut left[offset..];
        if band as i32 > max_band[band % 3] && ipos < max_pos {
            let stereo_scale = if hdr[3] & 0x20 != 0 {
                core::f32::consts::SQRT_2
            } else {
                1f32
            };
            let (mut left_scale, mut right_scale);
            if hdr[1] & 0x8 != 0 {
                left_scale = L3_STEREO_PROCESS_G_PAN[(2 * ipos) as usize];
                right_scale = L3_STEREO_PROCESS_G_PAN[(2 * ipos + 1) as usize];
            } else {
                left_scale = 1.0;
                right_scale = scale_pow2_quarter(1.0, (((ipos + 1) >> 1) << mpeg2_sh) as i32);
                if ipos & 1 != 0 {
                    core::mem::swap(&mut left_scale, &mut right_scale);
                }
            }
            apply_intensity_stereo_band(
                samples,
                width,
                left_scale * stereo_scale,
                right_scale * stereo_scale,
            );
        } else if hdr[3] & 0x20 != 0 {
            apply_mid_side_stereo(samples, width);
        }
        offset += width;
    }
}
fn apply_intensity_stereo(
    left: &mut [f32],
    ist_pos: &mut [u8; 39],
    gr: &[GranuleInfo],
    hdr: &[u8],
) {
    let mut max_band: [i32; 3] = [0; 3];
    let n_sfb = usize::from(gr[0].n_long_sfb + gr[0].n_short_sfb);
    let max_blocks = if gr[0].n_short_sfb != 0 { 3 } else { 1 };
    stereo_top_band(&left[576..], gr[0].sfbtab, n_sfb, &mut max_band);
    if gr[0].n_long_sfb != 0 {
        let maximum = *max_band.iter().max().unwrap();
        max_band.fill(maximum);
    }
    for (window, &last_nonzero_band) in max_band.iter().enumerate().take(max_blocks) {
        let default_position = if hdr[1] & 0x8 != 0 { 3 } else { 0 };
        let top = n_sfb - max_blocks + window;
        let previous = top - max_blocks;
        ist_pos[top] = if last_nonzero_band >= previous as i32 {
            default_position
        } else {
            ist_pos[previous]
        };
    }
    process_stereo_band(
        left,
        ist_pos,
        gr[0].sfbtab,
        hdr,
        &max_band,
        i32::from(gr[1].scalefac_compress & 1),
    );
}
fn reorder_short_blocks(grbuf: &mut [f32], scratch: &mut [f32], sfb: &[u8]) {
    let mut source = 0usize;
    let mut destination = 0usize;
    let mut band = 0usize;
    loop {
        let width = usize::from(sfb[band]);
        if width == 0 {
            break;
        }
        for index in 0..width {
            scratch[destination] = grbuf[source + index];
            scratch[destination + 1] = grbuf[source + width + index];
            scratch[destination + 2] = grbuf[source + width * 2 + index];
            destination += 3;
        }
        source += width * 3;
        band += 3;
    }
    grbuf[..destination].copy_from_slice(&scratch[..destination]);
}
fn antialias(grbuf: &mut [f32], nbands: i32) {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: every band is an 18-sample window inside the 576-sample
        // granule buffer, and the coefficient tables each contain eight values.
        unsafe { antialias_x86(grbuf, nbands) };
        return;
    }

    #[cfg(not(target_arch = "x86_64"))]
    let mut grbuf = grbuf;
    #[cfg(not(target_arch = "x86_64"))]
    for _ in 0..nbands {
        for i in 0..8 {
            let u = grbuf[18 + i];
            let d = grbuf[17 - i];
            grbuf[18 + i] = u * L3_ANTIALIAS_G_AA[0][i] - d * L3_ANTIALIAS_G_AA[1][i];
            grbuf[17 - i] = u * L3_ANTIALIAS_G_AA[1][i] + d * L3_ANTIALIAS_G_AA[0][i];
        }
        grbuf = &mut grbuf[18..];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn antialias_x86(grbuf: &mut [f32], nbands: i32) {
    use core::arch::x86_64::*;

    let samples = grbuf.as_mut_ptr();
    let aa0 = L3_ANTIALIAS_G_AA[0].as_ptr();
    let aa1 = L3_ANTIALIAS_G_AA[1].as_ptr();
    for band in 0..nbands.max(0) as usize {
        let base = band * 18;
        for index in [0usize, 4] {
            let upper = _mm_loadu_ps(samples.add(base + 18 + index));
            let lower = _mm_shuffle_ps::<0x1b>(
                _mm_loadu_ps(samples.add(base + 14 - index)),
                _mm_loadu_ps(samples.add(base + 14 - index)),
            );
            let coefficient0 = _mm_loadu_ps(aa0.add(index));
            let coefficient1 = _mm_loadu_ps(aa1.add(index));
            _mm_storeu_ps(
                samples.add(base + 18 + index),
                _mm_sub_ps(
                    _mm_mul_ps(upper, coefficient0),
                    _mm_mul_ps(lower, coefficient1),
                ),
            );
            let lower = _mm_add_ps(
                _mm_mul_ps(upper, coefficient1),
                _mm_mul_ps(lower, coefficient0),
            );
            _mm_storeu_ps(
                samples.add(base + 14 - index),
                _mm_shuffle_ps::<0x1b>(lower, lower),
            );
        }
    }
}

// Keep the MPEG transform coefficients in their reference decimal form.
#[allow(clippy::excessive_precision)]
fn dct3_9(y: &mut [f32]) {
    let mut s0 = y[0];
    let mut s2 = y[2];
    let mut s4 = y[4];
    let mut s6 = y[6];
    let mut s8 = y[8];
    let mut t0 = s0 + s6 * 0.5f32;
    s0 -= s6;
    let mut t4 = (s4 + s2) * 0.93969262f32;
    let mut t2 = (s8 + s2) * 0.76604444f32;
    s6 = (s4 - s8) * 0.17364818f32;
    s4 += s8 - s2;
    s2 = s0 - s4 * 0.5f32;
    y[4] = s4 + s0;
    s8 = t0 - t2 + s6;
    s0 = t0 - t4 + t2;
    s4 = t0 + t4 - s6;
    let mut s1 = y[1];
    let mut s3 = y[3];
    let mut s5 = y[5];
    let mut s7 = y[7];
    s3 *= 0.86602540f32;
    t0 = (s5 + s1) * 0.98480775f32;
    t4 = (s5 - s7) * 0.34202014f32;
    t2 = (s1 + s7) * 0.64278761f32;
    s1 = (s1 - s5 - s7) * 0.86602540f32;
    s5 = t0 - s3 - t2;
    s7 = t4 - s3 - t0;
    s3 = t4 + s3 - t2;
    y[0] = s4 - s7;
    y[1] = s2 + s1;
    y[2] = s0 - s3;
    y[3] = s8 + s5;
    y[5] = s8 - s5;
    y[6] = s0 + s3;
    y[7] = s2 - s1;
    y[8] = s4 + s7;
}

fn imdct36(grbuf: &mut [f32], overlap: &mut [f32], window: &[f32; 18], nbands: usize) {
    for band in 0..nbands {
        let samples = &mut grbuf[band * 18..band * 18 + 18];
        let history = &mut overlap[band * 9..band * 9 + 9];
        let mut co: [f32; 9] = [0.; 9];
        let mut si: [f32; 9] = [0.; 9];
        co[0] = -samples[0];
        si[0] = samples[17];
        for index in 0..4 {
            si[8 - 2 * index] = samples[4 * index + 1] - samples[4 * index + 2];
            co[1 + 2 * index] = samples[4 * index + 1] + samples[4 * index + 2];
            si[7 - 2 * index] = samples[4 * index + 4] - samples[4 * index + 3];
            co[2 + 2 * index] = -(samples[4 * index + 3] + samples[4 * index + 4]);
        }
        dct3_9(&mut co);
        dct3_9(&mut si);
        for index in [1usize, 3, 5, 7] {
            si[index] = -si[index];
        }

        for index in 0..9 {
            let previous = history[index];
            let sum =
                co[index] * L3_IMDCT36_G_TWID9[9 + index] + si[index] * L3_IMDCT36_G_TWID9[index];
            history[index] =
                co[index] * L3_IMDCT36_G_TWID9[index] - si[index] * L3_IMDCT36_G_TWID9[9 + index];
            samples[index] = previous * window[index] - sum * window[9 + index];
            samples[17 - index] = previous * window[9 + index] + sum * window[index];
        }
    }
}

#[allow(clippy::excessive_precision)]
fn idct3(x0: f32, x1: f32, x2: f32, dst: &mut [f32]) {
    let m1: f32 = x1 * 0.86602540f32;
    let a1: f32 = x0 - x2 * 0.5f32;
    dst[1] = x0 + x2;
    dst[0] = a1 + m1;
    dst[2] = a1 - m1;
}
fn imdct12(x: &[f32], dst: &mut [f32], overlap: &mut [f32]) {
    let mut co: [f32; 3] = [0.; 3];
    let mut si: [f32; 3] = [0.; 3];
    idct3(-x[0], x[6] + x[3], x[12] + x[9], &mut co);
    idct3(x[15], x[12] - x[9], x[6] - x[3], &mut si);
    si[1] = -si[1];
    for index in 0..3 {
        let previous = overlap[index];
        let sum = co[index] * L3_IMDCT12_G_TWID3[3 + index] + si[index] * L3_IMDCT12_G_TWID3[index];
        overlap[index] =
            co[index] * L3_IMDCT12_G_TWID3[index] - si[index] * L3_IMDCT12_G_TWID3[3 + index];
        dst[index] = previous * L3_IMDCT12_G_TWID3[2 - index] - sum * L3_IMDCT12_G_TWID3[5 - index];
        dst[5 - index] =
            previous * L3_IMDCT12_G_TWID3[5 - index] + sum * L3_IMDCT12_G_TWID3[2 - index];
    }
}
fn imdct_short(grbuf: &mut [f32], overlap: &mut [f32], nbands: usize) {
    for band in 0..nbands {
        let samples = &mut grbuf[band * 18..band * 18 + 18];
        let history = &mut overlap[band * 9..band * 9 + 9];
        let mut input = [0.0f32; 18];
        input.copy_from_slice(samples);
        samples[..6].copy_from_slice(&history[..6]);
        imdct12(&input, &mut samples[6..12], &mut history[6..9]);
        imdct12(&input[1..], &mut samples[12..18], &mut history[6..9]);
        let (destination, tail) = history.split_at_mut(6);
        imdct12(&input[2..], destination, tail);
    }
}
fn change_subband_sign(grbuf: &mut [f32]) {
    for band in (1..32).step_by(2) {
        let samples = &mut grbuf[band * 18..band * 18 + 18];
        for index in (1..18).step_by(2) {
            samples[index] = -samples[index];
        }
    }
}
fn imdct_granule(
    grbuf: &mut [f32; 576],
    overlap: &mut [f32; 288],
    block_type: u32,
    n_long_bands: u32,
) {
    let long_bands = n_long_bands as usize;
    if long_bands != 0 {
        imdct36(grbuf, overlap, &L3_IMDCT_GR_G_MDCT_WINDOW[0], long_bands);
    }
    let sample_offset = 18 * long_bands;
    let overlap_offset = 9 * long_bands;
    if block_type == 2 {
        imdct_short(
            &mut grbuf[sample_offset..],
            &mut overlap[overlap_offset..],
            32 - long_bands,
        );
    } else {
        imdct36(
            &mut grbuf[sample_offset..],
            &mut overlap[overlap_offset..],
            &L3_IMDCT_GR_G_MDCT_WINDOW[usize::from(block_type == 3)],
            32 - long_bands,
        );
    };
}

fn save_reservoir(h: &mut Layer3Decoder, s_bs: &mut BitReader) {
    let mut pos: i32 = ((s_bs.pos + 7) as u32).wrapping_div(8) as i32;
    let mut remains: i32 = (s_bs.limit as u32).wrapping_div(8).wrapping_sub(pos as u32) as i32;
    if remains > 511 {
        pos += remains - 511;
        remains = 511;
    }
    if remains > 0 {
        h.reserv_buf[..remains as usize]
            .copy_from_slice(&s_bs.buf[pos as usize..(pos + remains) as usize]);
    }
    h.reserv = remains;
}

fn restore_reservoir<'a>(
    h: &mut Layer3Decoder,
    bs: &mut BitReader,
    s_maindata: &'a mut [u8; 2815],
    s_bs: &mut BitReader<'a>,
    main_data_begin: i32,
) -> i32 {
    let frame_bytes = (bs.limit - bs.pos) / 8;
    let bytes_have = if h.reserv > main_data_begin {
        main_data_begin
    } else {
        h.reserv
    };

    {
        let off = if 0 < h.reserv - main_data_begin {
            h.reserv - main_data_begin
        } else {
            0
        };
        let cnt = if h.reserv > main_data_begin {
            main_data_begin
        } else {
            h.reserv
        };
        s_maindata[..cnt as usize]
            .copy_from_slice(&h.reserv_buf[off as usize..off as usize + cnt as usize]);
    }

    s_maindata[bytes_have as usize..bytes_have as usize + frame_bytes as usize].copy_from_slice(
        &bs.buf[(bs.pos / 8) as usize..(bs.pos / 8) as usize + frame_bytes as usize],
    );

    *s_bs = bit_reader(&s_maindata[..], bytes_have + frame_bytes);
    i32::from(h.reserv >= main_data_begin)
}
fn decode_granule(
    h: &mut Layer3Decoder,
    s: &mut DecodeScratch,
    s_bs: &mut BitReader,
    gr_info: &[GranuleInfo],
    nch: u32,
) {
    let channels = nch as usize;
    for (ch, info) in gr_info.iter().enumerate().take(channels) {
        let layer3gr_limit = s_bs.pos + i32::from(info.part_23_length);
        decode_scalefactors(
            &h.header,
            &mut s.ist_pos[ch],
            s_bs,
            info,
            &mut s.scf,
            ch as u32,
        );
        decode_huffman(&mut s.grbuf[ch], s_bs, info, &s.scf, layer3gr_limit);
    }
    if h.header[3] & 0x10 != 0 {
        apply_intensity_stereo(
            s.grbuf.as_flattened_mut(),
            &mut s.ist_pos[1],
            gr_info,
            &h.header,
        );
    } else if h.header[3] & 0xe0 == 0x60 {
        apply_mid_side_stereo(s.grbuf.as_flattened_mut(), 576);
    }
    for (ch, info) in gr_info.iter().enumerate().take(channels) {
        let mut aa_bands = 31;
        let mixed_long_bands = if info.mixed_block_flag != 0 { 2 } else { 0 };
        let sample_rate_index = i32::from(h.header[2] >> 2 & 3)
            + (i32::from(h.header[1] >> 3 & 1) + i32::from(h.header[1] >> 4 & 1)) * 3;
        let n_long_bands = mixed_long_bands << i32::from(sample_rate_index == 2);
        if info.n_short_sfb != 0 {
            aa_bands = n_long_bands - 1;
            reorder_short_blocks(
                &mut s.grbuf[ch][n_long_bands as usize * 18..],
                s.syn.as_flattened_mut(),
                &info.sfbtab[usize::from(info.n_long_sfb)..],
            );
        }
        antialias(&mut s.grbuf[ch], aa_bands);
        imdct_granule(
            &mut s.grbuf[ch],
            &mut h.mdct_overlap[ch],
            u32::from(info.block_type),
            n_long_bands as u32,
        );
        change_subband_sign(&mut s.grbuf[ch]);
    }
}
fn synthesis_dct(grbuf: &mut [f32; 576], n: usize) {
    synthesis_dct_scalar(grbuf, 0, n);
}

#[allow(clippy::excessive_precision)]
fn synthesis_dct_scalar(grbuf: &mut [f32; 576], start: usize, n: usize) {
    for k in start..n {
        let mut t: [[f32; 8]; 4] = [[0.; 8]; 4];
        for i in 0..8 {
            let x0 = grbuf[k + i * 18];
            let x1 = grbuf[k + (15 - i) * 18];
            let x2 = grbuf[k + (16 + i) * 18];
            let x3 = grbuf[k + (31 - i) * 18];
            let t0: f32 = x0 + x3;
            let t1: f32 = x1 + x2;
            let t2 = (x1 - x2) * MP3D_DCT_II_G_SEC[3 * i];
            let t3 = (x0 - x3) * MP3D_DCT_II_G_SEC[3 * i + 1];
            t[0][i] = t0 + t1;
            t[1][i] = (t0 - t1) * MP3D_DCT_II_G_SEC[3 * i + 2];
            t[2][i] = t3 + t2;
            t[3][i] = (t3 - t2) * MP3D_DCT_II_G_SEC[3 * i + 2];
        }
        for x in &mut t {
            let mut x0_0 = x[0];
            let mut x1_0 = x[1];
            let mut x2_0 = x[2];
            let mut x3_0 = x[3];
            let mut x4 = x[4];
            let mut x5 = x[5];
            let mut x6 = x[6];
            let mut x7 = x[7];
            let mut xt;
            xt = x0_0 - x7;
            x0_0 += x7;
            x7 = x1_0 - x6;
            x1_0 += x6;
            x6 = x2_0 - x5;
            x2_0 += x5;
            x5 = x3_0 - x4;
            x3_0 += x4;
            x4 = x0_0 - x3_0;
            x0_0 += x3_0;
            x3_0 = x1_0 - x2_0;
            x1_0 += x2_0;
            x[0] = x0_0 + x1_0;
            x[4] = (x0_0 - x1_0) * 0.70710677f32;
            x5 += x6;
            x6 = (x6 + x7) * 0.70710677f32;
            x7 += xt;
            x3_0 = (x3_0 + x4) * 0.70710677f32;
            x5 -= x7 * 0.198912367f32;
            x7 += x5 * 0.382683432f32;
            x5 -= x7 * 0.198912367f32;
            x0_0 = xt - x6;
            xt += x6;
            x[1] = (xt + x7) * 0.50979561f32;
            x[2] = (x4 + x3_0) * 0.54119611f32;
            x[3] = (x0_0 - x5) * 0.60134488f32;
            x[5] = (x0_0 + x5) * 0.89997619f32;
            x[6] = (x4 - x3_0) * 1.30656302f32;
            x[7] = (xt - x7) * 2.56291556f32;
        }
        let mut y = k;
        for i in 0..7 {
            grbuf[y] = t[0][i];
            grbuf[y + 18] = t[2][i] + t[3][i] + t[3][i + 1];
            grbuf[y + 36] = t[1][i] + t[1][i + 1];
            grbuf[y + 54] = t[2][i + 1] + t[3][i] + t[3][i + 1];
            y += 72;
        }
        grbuf[y] = t[0][7];
        grbuf[y + 18] = t[2][7] + t[3][7];
        grbuf[y + 36] = t[1][7];
        grbuf[y + 54] = t[3][7];
    }
}

fn scale_pcm(sample: f32) -> f32 {
    sample * (1f32 / 32768f32)
}

fn synthesize_pair(pcm: &mut [f32], nch: usize, z: &[f32]) {
    let mut a = (z[14 * 64] - z[0]) * 29.0;
    a += (z[64] + z[13 * 64]) * 213.0;
    a += (z[12 * 64] - z[2 * 64]) * 459.0;
    a += (z[3 * 64] + z[11 * 64]) * 2037.0;
    a += (z[10 * 64] - z[4 * 64]) * 5153.0;
    a += (z[5 * 64] + z[9 * 64]) * 6574.0;
    a += (z[8 * 64] - z[6 * 64]) * 37489.0;
    a += z[7 * 64] * 75038.0;
    pcm[0] = scale_pcm(a);
    a = z[2 + 14 * 64] * 104.0;
    a += z[2 + 12 * 64] * 1567.0;
    a += z[2 + 10 * 64] * 9727.0;
    a += z[2 + 8 * 64] * 64019.0;
    a += z[2 + 6 * 64] * -9975.0;
    a += z[2 + 4 * 64] * -45.0;
    a += z[2 + 2 * 64] * 146.0;
    a += z[2] * -5.0;
    pcm[16 * nch] = scale_pcm(a);
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn prepare_synthesis_x86(xl: *const f32, history: *mut f32, xr: usize, i: usize) {
    const ZLIN: usize = 15 * 64;
    let source = 18 * (31 - i);
    *history.add(ZLIN + 4 * i) = *xl.add(source);
    *history.add(ZLIN + 4 * i + 1) = *xl.add(xr + source);
    *history.add(ZLIN + 4 * i + 2) = *xl.add(source + 1);
    *history.add(ZLIN + 4 * i + 3) = *xl.add(xr + source + 1);
    let source = 1 + 18 * (1 + i);
    *history.add(ZLIN + 4 * (i + 16)) = *xl.add(source);
    *history.add(ZLIN + 4 * (i + 16) + 1) = *xl.add(xr + source);
    let source = 18 * (1 + i);
    *history.add(ZLIN + 4 * i - 64 + 2) = *xl.add(source);
    *history.add(ZLIN + 4 * i - 64 + 3) = *xl.add(xr + source);
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn store_synthesis_x86(
    dst: *mut f32,
    nch: usize,
    i: usize,
    a: core::arch::x86_64::__m128,
    b: core::arch::x86_64::__m128,
) {
    use core::arch::x86_64::*;

    let dstr_off = nch - 1;
    *dst.add(dstr_off + (15 - i) * nch) = _mm_cvtss_f32(_mm_shuffle_ps::<0x55>(a, a));
    *dst.add(dstr_off + (17 + i) * nch) = _mm_cvtss_f32(_mm_shuffle_ps::<0x55>(b, b));
    *dst.add((15 - i) * nch) = _mm_cvtss_f32(a);
    *dst.add((17 + i) * nch) = _mm_cvtss_f32(b);
    *dst.add(dstr_off + (47 - i) * nch) = _mm_cvtss_f32(_mm_shuffle_ps::<0xff>(a, a));
    *dst.add(dstr_off + (49 + i) * nch) = _mm_cvtss_f32(_mm_shuffle_ps::<0xff>(b, b));
    *dst.add((47 - i) * nch) = _mm_cvtss_f32(_mm_shuffle_ps::<0xaa>(a, a));
    *dst.add((49 + i) * nch) = _mm_cvtss_f32(_mm_shuffle_ps::<0xaa>(b, b));
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn synthesize_avx2(xl: &[f32], dstl: &mut [f32], nch: usize, lins: &mut [f32; 1088]) {
    use core::arch::x86_64::*;

    const ZLIN: usize = 15 * 64;
    const SCALE: f32 = 1.0 / 32768.0;
    let xr = 576 * (nch - 1);
    let xl = xl.as_ptr();
    let dst = dstl.as_mut_ptr();
    let history = lins.as_mut_ptr();
    let weights = MP3D_SYNTH_G_WIN.as_ptr();
    let scale = _mm256_set1_ps(SCALE);
    let mut weight = 0usize;
    let mut i = 14usize;

    while i >= 2 {
        let lower_i = i - 1;
        prepare_synthesis_x86(xl, history, xr, i);
        prepare_synthesis_x86(xl, history, xr, lower_i);
        let mut a = _mm256_setzero_ps();
        let mut b = _mm256_setzero_ps();
        for phase in 0..8 {
            let upper_weight = weight + phase * 2;
            let lower_weight = upper_weight + 16;
            let w0 = _mm256_set_m128(
                _mm_set1_ps(*weights.add(upper_weight)),
                _mm_set1_ps(*weights.add(lower_weight)),
            );
            let w1 = _mm256_set_m128(
                _mm_set1_ps(*weights.add(upper_weight + 1)),
                _mm_set1_ps(*weights.add(lower_weight + 1)),
            );
            let vz = _mm256_loadu_ps(history.add(ZLIN + 4 * lower_i - phase * 64));
            let vy = _mm256_loadu_ps(history.add(ZLIN + 4 * lower_i - (15 - phase) * 64));
            let b_term = _mm256_add_ps(_mm256_mul_ps(vz, w1), _mm256_mul_ps(vy, w0));
            let a_term = if phase & 1 == 0 {
                _mm256_sub_ps(_mm256_mul_ps(vz, w0), _mm256_mul_ps(vy, w1))
            } else {
                _mm256_sub_ps(_mm256_mul_ps(vy, w1), _mm256_mul_ps(vz, w0))
            };
            if phase == 0 {
                a = a_term;
                b = b_term;
            } else {
                a = _mm256_add_ps(a, a_term);
                b = _mm256_add_ps(b, b_term);
            }
        }
        a = _mm256_mul_ps(a, scale);
        b = _mm256_mul_ps(b, scale);
        store_synthesis_x86(
            dst,
            nch,
            lower_i,
            _mm256_castps256_ps128(a),
            _mm256_castps256_ps128(b),
        );
        store_synthesis_x86(
            dst,
            nch,
            i,
            _mm256_extractf128_ps::<1>(a),
            _mm256_extractf128_ps::<1>(b),
        );
        weight += 32;
        i -= 2;
    }

    prepare_synthesis_x86(xl, history, xr, 0);
    let mut a = _mm_setzero_ps();
    let mut b = _mm_setzero_ps();
    for phase in 0..8 {
        let w0 = _mm_set1_ps(*weights.add(weight + phase * 2));
        let w1 = _mm_set1_ps(*weights.add(weight + phase * 2 + 1));
        let vz = _mm_loadu_ps(history.add(ZLIN - phase * 64));
        let vy = _mm_loadu_ps(history.add(ZLIN - (15 - phase) * 64));
        let b_term = _mm_add_ps(_mm_mul_ps(vz, w1), _mm_mul_ps(vy, w0));
        let a_term = if phase & 1 == 0 {
            _mm_sub_ps(_mm_mul_ps(vz, w0), _mm_mul_ps(vy, w1))
        } else {
            _mm_sub_ps(_mm_mul_ps(vy, w1), _mm_mul_ps(vz, w0))
        };
        if phase == 0 {
            a = a_term;
            b = b_term;
        } else {
            a = _mm_add_ps(a, a_term);
            b = _mm_add_ps(b, b_term);
        }
    }
    a = _mm_mul_ps(a, _mm_set1_ps(SCALE));
    b = _mm_mul_ps(b, _mm_set1_ps(SCALE));
    store_synthesis_x86(dst, nch, 0, a, b);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn synthesize_sse(xl: &[f32], dstl: &mut [f32], nch: usize, lins: &mut [f32; 1088]) {
    use core::arch::x86_64::*;

    const ZLIN: usize = 15 * 64;
    const SCALE: f32 = 1.0 / 32768.0;
    let xr = 576 * (nch - 1);
    let dstr_off = nch - 1;
    let xl = xl.as_ptr();
    let dst = dstl.as_mut_ptr();
    let history = lins.as_mut_ptr();
    let weights = MP3D_SYNTH_G_WIN.as_ptr();
    let scale = _mm_set1_ps(SCALE);
    let mut weight = 0usize;

    for i in (0..=14).rev() {
        let source = 18 * (31 - i);
        *history.add(ZLIN + 4 * i) = *xl.add(source);
        *history.add(ZLIN + 4 * i + 1) = *xl.add(xr + source);
        *history.add(ZLIN + 4 * i + 2) = *xl.add(source + 1);
        *history.add(ZLIN + 4 * i + 3) = *xl.add(xr + source + 1);
        let source = 1 + 18 * (1 + i);
        *history.add(ZLIN + 4 * (i + 16)) = *xl.add(source);
        *history.add(ZLIN + 4 * (i + 16) + 1) = *xl.add(xr + source);
        let source = 18 * (1 + i);
        *history.add(ZLIN + 4 * i - 64 + 2) = *xl.add(source);
        *history.add(ZLIN + 4 * i - 64 + 3) = *xl.add(xr + source);

        let mut a = _mm_setzero_ps();
        let mut b = _mm_setzero_ps();
        for phase in 0..8 {
            let w0 = _mm_set1_ps(*weights.add(weight));
            let w1 = _mm_set1_ps(*weights.add(weight + 1));
            weight += 2;
            let vz = _mm_loadu_ps(history.add(ZLIN + 4 * i - phase * 64));
            let vy = _mm_loadu_ps(history.add(ZLIN + 4 * i - (15 - phase) * 64));
            let b_term = _mm_add_ps(_mm_mul_ps(vz, w1), _mm_mul_ps(vy, w0));
            let a_term = if phase & 1 == 0 {
                _mm_sub_ps(_mm_mul_ps(vz, w0), _mm_mul_ps(vy, w1))
            } else {
                _mm_sub_ps(_mm_mul_ps(vy, w1), _mm_mul_ps(vz, w0))
            };
            if phase == 0 {
                a = a_term;
                b = b_term;
            } else {
                a = _mm_add_ps(a, a_term);
                b = _mm_add_ps(b, b_term);
            }
        }
        a = _mm_mul_ps(a, scale);
        b = _mm_mul_ps(b, scale);

        *dst.add(dstr_off + (15 - i) * nch) = _mm_cvtss_f32(_mm_shuffle_ps::<0x55>(a, a));
        *dst.add(dstr_off + (17 + i) * nch) = _mm_cvtss_f32(_mm_shuffle_ps::<0x55>(b, b));
        *dst.add((15 - i) * nch) = _mm_cvtss_f32(a);
        *dst.add((17 + i) * nch) = _mm_cvtss_f32(b);
        *dst.add(dstr_off + (47 - i) * nch) = _mm_cvtss_f32(_mm_shuffle_ps::<0xff>(a, a));
        *dst.add(dstr_off + (49 + i) * nch) = _mm_cvtss_f32(_mm_shuffle_ps::<0xff>(b, b));
        *dst.add((47 - i) * nch) = _mm_cvtss_f32(_mm_shuffle_ps::<0xaa>(a, a));
        *dst.add((49 + i) * nch) = _mm_cvtss_f32(_mm_shuffle_ps::<0xaa>(b, b));
    }
}

fn synthesize_slot(xl: &[f32], dstl: &mut [f32], nch: usize, lins: &mut [f32; 1088]) {
    const ZLIN: usize = 15 * 64;
    let xr = 576 * (nch - 1);
    let dstr_off = nch - 1;

    lins[ZLIN + 4 * 15] = xl[18 * 16];
    lins[ZLIN + 4 * 15 + 1] = xl[xr + 18 * 16];
    lins[ZLIN + 4 * 15 + 2] = xl[0];
    lins[ZLIN + 4 * 15 + 3] = xl[xr];
    lins[ZLIN + 4 * 31] = xl[1 + 18 * 16];
    lins[ZLIN + 4 * 31 + 1] = xl[xr + 1 + 18 * 16];
    lins[ZLIN + 4 * 31 + 2] = xl[1];
    lins[ZLIN + 4 * 31 + 3] = xl[xr + 1];
    synthesize_pair(&mut dstl[dstr_off..], nch, &lins[4 * 15 + 1..]);
    synthesize_pair(
        &mut dstl[dstr_off + 32 * nch..],
        nch,
        &lins[4 * 15 + 64 + 1..],
    );
    synthesize_pair(dstl, nch, &lins[4 * 15..]);
    synthesize_pair(&mut dstl[32 * nch..], nch, &lins[4 * 15 + 64..]);

    #[cfg(target_arch = "x86_64")]
    unsafe {
        // SAFETY: `synthesize_granule` supplies complete channel and history
        // windows. Runtime feature detection guards the AVX2 implementation;
        // SSE2 is guaranteed by the x86-64 architecture.
        if std::arch::is_x86_feature_detected!("avx2") {
            synthesize_avx2(xl, dstl, nch, lins);
        } else {
            synthesize_sse(xl, dstl, nch, lins);
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    synthesize_scalar(xl, dstl, nch, lins);
}

#[cfg(not(target_arch = "x86_64"))]
fn synthesize_scalar(xl: &[f32], dstl: &mut [f32], nch: usize, lins: &mut [f32; 1088]) {
    const ZLIN: usize = 15 * 64;
    let xr = 576 * (nch - 1);
    let dstr_off = nch - 1;
    let mut weight = 0;
    for i in (0..=14).rev() {
        let mut a: [f32; 4] = [0.; 4];
        let mut b: [f32; 4] = [0.; 4];
        let source = 18 * (31 - i);
        lins[ZLIN + 4 * i] = xl[source];
        lins[ZLIN + 4 * i + 1] = xl[xr + source];
        lins[ZLIN + 4 * i + 2] = xl[source + 1];
        lins[ZLIN + 4 * i + 3] = xl[xr + source + 1];
        let source = 1 + 18 * (1 + i);
        lins[ZLIN + 4 * (i + 16)] = xl[source];
        lins[ZLIN + 4 * (i + 16) + 1] = xl[xr + source];
        let source = 18 * (1 + i);
        lins[ZLIN + 4 * i - 64 + 2] = xl[source];
        lins[ZLIN + 4 * i - 64 + 3] = xl[xr + source];

        for phase in 0..8 {
            let w0 = MP3D_SYNTH_G_WIN[weight];
            let w1 = MP3D_SYNTH_G_WIN[weight + 1];
            weight += 2;
            let vz = ZLIN + 4 * i - phase * 64;
            let vy = ZLIN + 4 * i - (15 - phase) * 64;
            for j in 0..4 {
                let vz_sample = lins[vz + j];
                let vy_sample = lins[vy + j];
                let b_term = vz_sample * w1 + vy_sample * w0;
                let a_term = if phase & 1 == 0 {
                    vz_sample * w0 - vy_sample * w1
                } else {
                    vy_sample * w1 - vz_sample * w0
                };
                if phase == 0 {
                    b[j] = b_term;
                    a[j] = a_term;
                } else {
                    b[j] += b_term;
                    a[j] += a_term;
                }
            }
        }
        dstl[dstr_off + (15 - i) * nch] = scale_pcm(a[1]);
        dstl[dstr_off + (17 + i) * nch] = scale_pcm(b[1]);
        dstl[(15 - i) * nch] = scale_pcm(a[0]);
        dstl[(17 + i) * nch] = scale_pcm(b[0]);
        dstl[dstr_off + (47 - i) * nch] = scale_pcm(a[3]);
        dstl[dstr_off + (49 + i) * nch] = scale_pcm(b[3]);
        dstl[(47 - i) * nch] = scale_pcm(a[2]);
        dstl[(49 + i) * nch] = scale_pcm(b[2]);
    }
}
fn synthesize_granule(
    qmf_state: &mut [f32; 960],
    grbuf: &mut [[f32; 576]; 2],
    nbands: usize,
    nch: usize,
    pcm: &mut [f32],
    lins: &mut [[f32; 64]; 33],
) {
    for channel in grbuf.iter_mut().take(nch) {
        synthesis_dct(channel, nbands);
    }
    let grbuf = grbuf.as_flattened();
    let lins = lins.as_flattened_mut();
    lins[..960].copy_from_slice(qmf_state);
    for i in (0..nbands).step_by(2) {
        let window: &mut [f32; 1088] = (&mut lins[i * 64..i * 64 + 1088])
            .try_into()
            .expect("synthesis history window");
        synthesize_slot(&grbuf[i..], &mut pcm[32 * nch * i..], nch, window);
    }
    qmf_state.copy_from_slice(&lins[nbands * 64..nbands * 64 + 960]);
}

fn match_frame(hdr: &[u8], frame_bytes: usize) -> bool {
    let mut i: usize = 0;
    for matched in 0..10 {
        i += hdr_frame_bytes(&hdr[i..], frame_bytes) + hdr_padding(&hdr[i..]);
        if i + 4 > hdr.len() {
            return matched > 0;
        }
        if !hdr_compare(hdr, &hdr[i..]) {
            return false;
        }
    }
    true
}

fn find_frame(mut mp3: &[u8], free_format_bytes: &mut usize, ptr_frame_bytes: &mut usize) -> usize {
    let mp3_bytes = mp3.len();
    let mut i: usize = 0;
    while i < mp3_bytes - 4 {
        if hdr_valid(mp3) {
            let mut frame_bytes = hdr_frame_bytes(mp3, *free_format_bytes);
            let mut frame_and_padding = frame_bytes + hdr_padding(mp3);
            let mut k = 4;
            while frame_bytes == 0 && k < 2304 && i + 2 * k < mp3_bytes - 4 {
                if hdr_compare(mp3, &mp3[k..]) {
                    let fb = k - hdr_padding(mp3);
                    let nextfb = fb + hdr_padding(&mp3[k..]);
                    if i + k + nextfb + 4 <= mp3_bytes && hdr_compare(mp3, &mp3[k + nextfb..]) {
                        frame_and_padding = k;
                        frame_bytes = fb;
                        *free_format_bytes = fb;
                    }
                }
                k += 1;
            }
            if frame_bytes != 0
                && i + frame_and_padding <= mp3_bytes
                && match_frame(mp3, frame_bytes)
                || i == 0 && frame_and_padding == mp3_bytes
            {
                *ptr_frame_bytes = frame_and_padding;
                return i;
            }
            *free_format_bytes = 0;
        }
        i += 1;
        mp3 = &mp3[1..];
    }
    *ptr_frame_bytes = 0;
    mp3_bytes
}

fn reset_decoder(dec: &mut Layer3Decoder) {
    dec.header[0] = 0;
}

pub(super) fn decode_frame(
    dec: &mut Layer3Decoder,
    mp3: &[u8],
    mut pcm: &mut [f32],
    info: &mut CoreFrameInfo,
) -> usize {
    let mut i: usize = 0;
    let mut frame_size: usize = 0;
    let mut scratch: DecodeScratch = DecodeScratch {
        grbuf: [[0.; 576]; 2],
        scf: [0.; 40],
        syn: [[0.; 64]; 33],
        ist_pos: [[0; 39]; 2],
    };
    let mut scratch_maindata = [0u8; 2815];
    let mut scratch_bs = BitReader {
        buf: &[],
        pos: 0,
        limit: 0,
    };
    let mut scratch_gr_info = [GranuleInfo {
        sfbtab: &[],
        part_23_length: 0,
        big_values: 0,
        scalefac_compress: 0,
        global_gain: 0,
        block_type: 0,
        mixed_block_flag: 0,
        n_long_sfb: 0,
        n_short_sfb: 0,
        table_select: [0; 3],
        region_count: [0; 3],
        subblock_gain: [0; 3],
        preflag: 0,
        scalefac_scale: 0,
        count1_table: 0,
        scfsi: 0,
    }; 4];
    if mp3.len() > 4 && dec.header[0] == 0xff && hdr_compare(&dec.header, mp3) {
        frame_size = hdr_frame_bytes(mp3, dec.free_format_bytes) + hdr_padding(mp3);
        if frame_size != mp3.len()
            && (frame_size + 4 > mp3.len() || !hdr_compare(mp3, &mp3[frame_size..]))
        {
            frame_size = 0;
        }
    }
    if frame_size == 0 {
        *dec = Layer3Decoder::new();
        i = find_frame(mp3, &mut dec.free_format_bytes, &mut frame_size);
        if frame_size == 0 || i + frame_size > mp3.len() {
            info.frame_bytes = i;
            return 0;
        }
    }
    let hdr = &mp3[i..];
    dec.header.copy_from_slice(&hdr[..4]);
    info.frame_bytes = i + frame_size;
    info.frame_offset = i;
    info.channels = if hdr[3] & 0xc0 == 0xc0 { 1 } else { 2 };
    info.sample_rate = hdr_sample_rate_hz(hdr);
    info.layer = 4 - (hdr[1] >> 1 & 3);
    info.bitrate_kbps = hdr_bitrate_kbps(hdr);
    if pcm.is_empty() {
        return hdr_frame_samples(hdr) as usize;
    }
    let mut bs_frame = bit_reader(&hdr[4..], (frame_size - 4) as i32);
    if hdr[1] & 1 == 0 {
        get_bits(&mut bs_frame, 16);
    }
    if info.layer != 3 {
        return 0;
    }
    let main_data_begin = read_side_info(&mut bs_frame, &mut scratch_gr_info, hdr);
    if main_data_begin < 0 || bs_frame.pos > bs_frame.limit {
        reset_decoder(dec);
        return 0;
    }
    let success = restore_reservoir(
        dec,
        &mut bs_frame,
        &mut scratch_maindata,
        &mut scratch_bs,
        main_data_begin,
    );
    if success != 0 {
        let granules = if hdr[1] & 0x8 != 0 { 2 } else { 1 };
        for granule in 0..granules {
            scratch.grbuf.as_flattened_mut().fill(0.0);
            decode_granule(
                dec,
                &mut scratch,
                &mut scratch_bs,
                &scratch_gr_info[granule * info.channels as usize..],
                info.channels,
            );
            synthesize_granule(
                &mut dec.qmf_state,
                &mut scratch.grbuf,
                18,
                info.channels as usize,
                pcm,
                &mut scratch.syn,
            );
            pcm = &mut pcm[576 * info.channels as usize..];
        }
    }
    save_reservoir(dec, &mut scratch_bs);
    success as usize * hdr_frame_samples(&dec.header) as usize
}
