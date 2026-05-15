use crate::constants::Bandwidth;
use crate::{Error, Result};

pub(crate) const MAX_FRAMES: usize = 48;
pub(crate) const MAX_FRAME_BYTES: i32 = 1275;
pub(crate) const MAX_PACKET_SAMPLES_48K: i32 = 5760;

#[derive(Clone, Debug)]
pub(crate) struct ParsedPacket {
    pub toc: u8,
    pub count: usize,
    pub sizes: [i16; MAX_FRAMES],
    pub frame_offsets: [usize; MAX_FRAMES],
    pub payload_offset: usize,
    pub packet_offset: usize,
    pub padding_offset: usize,
    pub padding_len: usize,
}

impl ParsedPacket {
    fn new(toc: u8) -> Self {
        Self {
            toc,
            count: 0,
            sizes: [0; MAX_FRAMES],
            frame_offsets: [0; MAX_FRAMES],
            payload_offset: 0,
            packet_offset: 0,
            padding_offset: 0,
            padding_len: 0,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PacketFrame<'a> {
    pub offset: usize,
    pub data: &'a [u8],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Packet<'a> {
    pub toc: u8,
    pub payload_offset: usize,
    pub packet_offset: usize,
    pub padding_offset: usize,
    pub padding_len: usize,
    frames: Vec<PacketFrame<'a>>,
}

impl<'a> Packet<'a> {
    pub fn frames(&self) -> &[PacketFrame<'a>] {
        &self.frames
    }

    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    pub fn frame_size(&self, index: usize) -> Option<usize> {
        self.frames.get(index).map(|frame| frame.data.len())
    }
}

pub(crate) fn encode_size(size: i32, data: &mut [u8]) -> usize {
    debug_assert!(data.len() >= 2);
    if size < 252 {
        data[0] = size as u8;
        1
    } else {
        data[0] = (252 + (size & 0x3)) as u8;
        data[1] = ((size - data[0] as i32) >> 2) as u8;
        2
    }
}

fn parse_size(data: &[u8], offset: &mut usize, remaining: &mut i32) -> Result<i16> {
    if *remaining < 1 {
        return Err(Error::InvalidPacket);
    }
    let first = data[*offset];
    *offset += 1;
    *remaining -= 1;
    if first < 252 {
        Ok(first as i16)
    } else {
        if *remaining < 1 {
            return Err(Error::InvalidPacket);
        }
        let second = data[*offset];
        *offset += 1;
        *remaining -= 1;
        Ok((4 * second as i32 + first as i32) as i16)
    }
}

pub(crate) fn packet_get_samples_per_frame_byte(toc: u8, fs: i32) -> i32 {
    if toc & 0x80 != 0 {
        let audiosize = ((toc >> 3) & 0x3) as i32;
        (fs << audiosize) / 400
    } else if (toc & 0x60) == 0x60 {
        if toc & 0x08 != 0 {
            fs / 50
        } else {
            fs / 100
        }
    } else {
        let audiosize = ((toc >> 3) & 0x3) as i32;
        if audiosize == 3 {
            fs * 60 / 1000
        } else {
            (fs << audiosize) / 100
        }
    }
}

pub(crate) fn make_celt_only_fullband_toc(lm: usize, channels: usize) -> Result<u8> {
    if lm > 3 || !(1..=2).contains(&channels) {
        return Err(Error::BadArg);
    }
    Ok(0xE0 | ((lm as u8) << 3) | if channels == 2 { 0x04 } else { 0x00 })
}

pub(crate) fn celt_only_lm(toc: u8) -> Result<usize> {
    if toc & 0x80 == 0 {
        return Err(Error::InvalidPacket);
    }
    Ok(((toc >> 3) & 0x03) as usize)
}

pub(crate) fn is_celt_only(toc: u8) -> bool {
    toc & 0x80 != 0
}

pub(crate) fn parse_packet_slice(data: &[u8], self_delimited: bool) -> Result<ParsedPacket> {
    if data.is_empty() {
        return Err(Error::InvalidPacket);
    }
    let mut offset = 0usize;
    let mut remaining = data.len() as i32;
    let framesize = packet_get_samples_per_frame_byte(data[0], 48_000);
    let toc = data[offset];
    offset += 1;
    remaining -= 1;
    let mut packet = ParsedPacket::new(toc);
    let mut cbr = false;
    let mut last_size = remaining;

    match toc & 0x3 {
        0 => {
            packet.count = 1;
        }
        1 => {
            packet.count = 2;
            cbr = true;
            if !self_delimited {
                if remaining & 0x1 != 0 {
                    return Err(Error::InvalidPacket);
                }
                last_size = remaining / 2;
                packet.sizes[0] = last_size as i16;
            }
        }
        2 => {
            packet.count = 2;
            let size = parse_size(data, &mut offset, &mut remaining)?;
            if size < 0 || size as i32 > remaining {
                return Err(Error::InvalidPacket);
            }
            packet.sizes[0] = size;
            last_size = remaining - size as i32;
        }
        _ => {
            if remaining < 1 {
                return Err(Error::InvalidPacket);
            }
            let ch = data[offset];
            offset += 1;
            remaining -= 1;
            let count = (ch & 0x3f) as usize;
            if count == 0 || framesize * count as i32 > MAX_PACKET_SAMPLES_48K {
                return Err(Error::InvalidPacket);
            }
            packet.count = count;

            let mut pad = 0i32;
            if ch & 0x40 != 0 {
                loop {
                    if remaining <= 0 {
                        return Err(Error::InvalidPacket);
                    }
                    let p = data[offset] as i32;
                    offset += 1;
                    remaining -= 1;
                    let tmp = if p == 255 { 254 } else { p };
                    remaining -= tmp;
                    pad += tmp;
                    if p != 255 {
                        break;
                    }
                }
            }
            if remaining < 0 {
                return Err(Error::InvalidPacket);
            }
            packet.padding_len = pad as usize;
            cbr = ch & 0x80 == 0;
            if !cbr {
                last_size = remaining;
                for i in 0..count - 1 {
                    let size = parse_size(data, &mut offset, &mut remaining)?;
                    if size < 0 || size as i32 > remaining {
                        return Err(Error::InvalidPacket);
                    }
                    packet.sizes[i] = size;
                    last_size -= size as i32 + if size < 252 { 1 } else { 2 };
                }
                if last_size < 0 {
                    return Err(Error::InvalidPacket);
                }
            } else if !self_delimited {
                last_size = remaining / count as i32;
                if last_size * count as i32 != remaining {
                    return Err(Error::InvalidPacket);
                }
                for i in 0..count - 1 {
                    packet.sizes[i] = last_size as i16;
                }
            }
        }
    }

    if self_delimited {
        let last = parse_size(data, &mut offset, &mut remaining)?;
        if last < 0 || last as i32 > remaining {
            return Err(Error::InvalidPacket);
        }
        let count = packet.count;
        packet.sizes[count - 1] = last;
        if cbr {
            if last as i32 * count as i32 > remaining {
                return Err(Error::InvalidPacket);
            }
            for i in 0..count - 1 {
                packet.sizes[i] = last;
            }
        } else {
            let bytes = if last < 252 { 1 } else { 2 };
            if bytes + last as i32 > last_size {
                return Err(Error::InvalidPacket);
            }
        }
    } else {
        if last_size > MAX_FRAME_BYTES {
            return Err(Error::InvalidPacket);
        }
        let count = packet.count;
        packet.sizes[count - 1] = last_size as i16;
    }

    packet.payload_offset = offset;
    for i in 0..packet.count {
        packet.frame_offsets[i] = offset;
        offset += packet.sizes[i] as usize;
    }
    packet.padding_offset = offset;
    packet.packet_offset = offset + packet.padding_len;
    Ok(packet)
}

fn public_packet<'a>(data: &'a [u8], parsed: ParsedPacket) -> Packet<'a> {
    let frames = (0..parsed.count)
        .map(|i| {
            let offset = parsed.frame_offsets[i];
            let len = parsed.sizes[i] as usize;
            PacketFrame {
                offset,
                data: &data[offset..offset + len],
            }
        })
        .collect();
    Packet {
        toc: parsed.toc,
        payload_offset: parsed.payload_offset,
        packet_offset: parsed.packet_offset,
        padding_offset: parsed.padding_offset,
        padding_len: parsed.padding_len,
        frames,
    }
}

pub fn parse_packet(data: &[u8]) -> Result<Packet<'_>> {
    let parsed = parse_packet_slice(data, false)?;
    Ok(public_packet(data, parsed))
}

pub fn parse_self_delimited_packet(data: &[u8]) -> Result<Packet<'_>> {
    let parsed = parse_packet_slice(data, true)?;
    Ok(public_packet(data, parsed))
}

pub fn samples_per_frame(packet: &[u8], fs: i32) -> Result<i32> {
    let toc = *packet.first().ok_or(Error::InvalidPacket)?;
    Ok(packet_get_samples_per_frame_byte(toc, fs))
}

pub fn bandwidth(packet: &[u8]) -> Result<Bandwidth> {
    let toc = *packet.first().ok_or(Error::InvalidPacket)?;
    if toc & 0x80 != 0 {
        Ok(match (toc >> 5) & 0x3 {
            0 => Bandwidth::Narrowband,
            1 => Bandwidth::Wideband,
            2 => Bandwidth::SuperWideband,
            _ => Bandwidth::Fullband,
        })
    } else if (toc & 0x60) == 0x60 {
        if toc & 0x10 != 0 {
            Ok(Bandwidth::Fullband)
        } else {
            Ok(Bandwidth::SuperWideband)
        }
    } else {
        Ok(match (toc >> 5) & 0x3 {
            0 => Bandwidth::Narrowband,
            1 => Bandwidth::Mediumband,
            2 => Bandwidth::Wideband,
            _ => Bandwidth::SuperWideband,
        })
    }
}

pub fn channels(packet: &[u8]) -> Result<usize> {
    let toc = *packet.first().ok_or(Error::InvalidPacket)?;
    Ok(if toc & 0x4 != 0 { 2 } else { 1 })
}

pub fn frame_count(packet: &[u8]) -> Result<usize> {
    let toc = *packet.first().ok_or(Error::InvalidPacket)?;
    let count = toc & 0x3;
    if count == 0 {
        Ok(1)
    } else if count != 3 {
        Ok(2)
    } else {
        let ch = *packet.get(1).ok_or(Error::InvalidPacket)?;
        Ok((ch & 0x3f) as usize)
    }
}

pub fn sample_count(packet: &[u8], fs: i32) -> Result<i32> {
    let count = frame_count(packet)? as i64;
    let toc = packet[0];
    let samples = count * packet_get_samples_per_frame_byte(toc, fs) as i64;
    if samples * 25 > fs as i64 * 3 {
        Err(Error::InvalidPacket)
    } else {
        Ok(samples as i32)
    }
}

fn packet_get_mode_byte(toc: u8) -> i32 {
    if toc & 0x80 != 0 {
        1002
    } else if (toc & 0x60) == 0x60 {
        1001
    } else {
        1000
    }
}

pub fn has_lbrr(packet: &[u8]) -> Result<bool> {
    let toc = *packet.first().ok_or(Error::InvalidPacket)?;
    if packet_get_mode_byte(toc) == 1002 {
        return Ok(false);
    }
    let parsed = parse_packet_slice(packet, false)?;
    if parsed.count == 0 || parsed.sizes[0] <= 0 {
        return Ok(false);
    }
    let packet_frame_size = packet_get_samples_per_frame_byte(toc, 48_000);
    let nb_frames = if packet_frame_size > 960 {
        packet_frame_size / 960
    } else {
        1
    };
    let stream_channels = channels(packet)?;
    let first = packet[parsed.frame_offsets[0]];
    let mut lbrr = ((first >> (7 - nb_frames)) & 0x1) != 0;
    if stream_channels == 2 {
        lbrr |= ((first >> (6 - 2 * nb_frames)) & 0x1) != 0;
    }
    Ok(lbrr)
}

pub(crate) fn packet_offset_for_stream(data: &[u8], self_delimited: bool) -> Result<ParsedPacket> {
    parse_packet_slice(data, self_delimited)
}
