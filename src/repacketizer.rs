use crate::packet::{
    encode_size, packet_get_samples_per_frame_byte, packet_offset_for_stream, parse_packet_slice,
    MAX_FRAMES,
};
use crate::{Error, Result};

#[derive(Clone, Debug, Default)]
pub struct Repacketizer {
    toc: u8,
    framesize: i32,
    frames: Vec<Vec<u8>>,
}

impl Repacketizer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }

    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    pub fn cat(&mut self, packet: &[u8]) -> Result<()> {
        self.cat_impl(packet, false)
    }

    fn cat_impl(&mut self, packet: &[u8], self_delimited: bool) -> Result<()> {
        if packet.is_empty() {
            return Err(Error::InvalidPacket);
        }
        let parsed = parse_packet_slice(packet, self_delimited)?;
        if self.frames.is_empty() {
            self.toc = packet[0];
            self.framesize = packet_get_samples_per_frame_byte(packet[0], 8_000);
        } else if (self.toc & 0xfc) != (packet[0] & 0xfc) {
            return Err(Error::InvalidPacket);
        }
        if parsed.count < 1 {
            return Err(Error::InvalidPacket);
        }
        if (parsed.count + self.frames.len()) as i32 * self.framesize > 960 {
            return Err(Error::InvalidPacket);
        }
        if self.frames.len() + parsed.count > MAX_FRAMES {
            return Err(Error::InvalidPacket);
        }

        for i in 0..parsed.count {
            let offset = parsed.frame_offsets[i];
            let len = parsed.sizes[i] as usize;
            self.frames.push(packet[offset..offset + len].to_vec());
        }
        Ok(())
    }

    pub fn out(&self) -> Result<Vec<u8>> {
        self.out_range(0, self.frames.len())
    }

    pub fn out_range(&self, begin: usize, end: usize) -> Result<Vec<u8>> {
        self.out_range_impl(begin, end, false, None)
    }

    fn out_range_self_delimited(&self, begin: usize, end: usize) -> Result<Vec<u8>> {
        self.out_range_impl(begin, end, true, None)
    }

    fn out_range_padded(&self, begin: usize, end: usize, target_len: usize) -> Result<Vec<u8>> {
        self.out_range_impl(begin, end, false, Some(target_len))
    }

    fn out_range_impl(
        &self,
        begin: usize,
        end: usize,
        self_delimited: bool,
        pad_to: Option<usize>,
    ) -> Result<Vec<u8>> {
        if begin >= end || end > self.frames.len() {
            return Err(Error::BadArg);
        }

        let frames = &self.frames[begin..end];
        let count = frames.len();
        let lengths: Vec<i16> = frames.iter().map(|frame| frame.len() as i16).collect();
        let self_delim_size = if self_delimited {
            size_code_len(lengths[count - 1])
        } else {
            0
        };
        let pad = pad_to.is_some();
        let mut out = Vec::<u8>::new();

        if count == 1 && !pad {
            out.push(self.toc & 0xfc);
        } else if count == 2 && lengths[0] == lengths[1] && !pad {
            out.push((self.toc & 0xfc) | 0x1);
        } else if count == 2 && !pad {
            out.push((self.toc & 0xfc) | 0x2);
            let mut tmp = [0u8; 2];
            let n = encode_size(lengths[0] as i32, &mut tmp);
            out.extend_from_slice(&tmp[..n]);
        } else {
            let vbr = lengths.iter().any(|&value| value != lengths[0]);
            let mut base_size = 2 + self_delim_size;
            if vbr {
                for length in lengths.iter().take(count - 1) {
                    base_size += size_code_len(*length) + *length as usize;
                }
                base_size += lengths[count - 1] as usize;
            } else {
                base_size += count * lengths[0] as usize;
            }
            let pad_amount = match pad_to {
                Some(target_len) if base_size > target_len => return Err(Error::BufferTooSmall),
                Some(target_len) => target_len - base_size,
                None => 0,
            };

            out.push((self.toc & 0xfc) | 0x3);
            let ch_index = out.len();
            out.push(count as u8 | if vbr { 0x80 } else { 0 });
            if pad_amount != 0 {
                out[ch_index] |= 0x40;
                let nb_255s = (pad_amount - 1) / 255;
                for _ in 0..nb_255s {
                    out.push(255);
                }
                out.push((pad_amount - 255 * nb_255s - 1) as u8);
            }
            if vbr {
                let mut tmp = [0u8; 2];
                for length in lengths.iter().take(count - 1) {
                    let n = encode_size(*length as i32, &mut tmp);
                    out.extend_from_slice(&tmp[..n]);
                }
            }
        }

        if self_delimited {
            let mut tmp = [0u8; 2];
            let n = encode_size(lengths[count - 1] as i32, &mut tmp);
            out.extend_from_slice(&tmp[..n]);
        }

        for frame in frames {
            out.extend_from_slice(frame);
        }

        if let Some(target_len) = pad_to {
            if out.len() > target_len {
                return Err(Error::BufferTooSmall);
            }
            out.resize(target_len, 0);
        }
        Ok(out)
    }
}

fn size_code_len(size: i16) -> usize {
    1 + usize::from(size >= 252)
}

pub fn packet_pad(packet: &[u8], new_len: usize) -> Result<Vec<u8>> {
    if packet.is_empty() || packet.len() > new_len {
        return Err(Error::BadArg);
    }
    if packet.len() == new_len {
        return Ok(packet.to_vec());
    }
    let mut rp = Repacketizer::new();
    rp.cat(packet)?;
    rp.out_range_padded(0, rp.frame_count(), new_len)
}

pub fn packet_unpad(packet: &[u8]) -> Result<Vec<u8>> {
    if packet.is_empty() {
        return Err(Error::BadArg);
    }
    let mut rp = Repacketizer::new();
    rp.cat(packet)?;
    rp.out()
}

pub fn multistream_packet_pad(packet: &[u8], new_len: usize, nb_streams: usize) -> Result<Vec<u8>> {
    if packet.is_empty() || nb_streams == 0 || packet.len() > new_len {
        return Err(Error::BadArg);
    }
    if packet.len() == new_len {
        return Ok(packet.to_vec());
    }

    let amount = new_len - packet.len();
    let mut cursor = 0usize;
    for _ in 0..nb_streams - 1 {
        let parsed = packet_offset_for_stream(&packet[cursor..], true)?;
        cursor += parsed.packet_offset;
    }

    let mut out = packet[..cursor].to_vec();
    out.extend_from_slice(&packet_pad(
        &packet[cursor..],
        packet.len() - cursor + amount,
    )?);
    Ok(out)
}

pub fn multistream_packet_unpad(packet: &[u8], nb_streams: usize) -> Result<Vec<u8>> {
    if packet.is_empty() || nb_streams == 0 {
        return Err(Error::BadArg);
    }

    let mut out = Vec::new();
    let mut cursor = 0usize;
    for stream in 0..nb_streams {
        let self_delimited = stream != nb_streams - 1;
        let parsed = packet_offset_for_stream(&packet[cursor..], self_delimited)?;
        let stream_packet = &packet[cursor..cursor + parsed.packet_offset];
        let mut rp = Repacketizer::new();
        rp.cat_impl(stream_packet, self_delimited)?;
        let repacketized = if self_delimited {
            rp.out_range_self_delimited(0, rp.frame_count())?
        } else {
            rp.out()?
        };
        out.extend_from_slice(&repacketized);
        cursor += parsed.packet_offset;
    }
    Ok(out)
}
