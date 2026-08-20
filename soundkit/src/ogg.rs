use std::collections::VecDeque;

const MAX_INPUT_CHUNK_BYTES: usize = 4 * 1024 * 1024;
const MAX_PACKET_BYTES: usize = 16 * 1024 * 1024;
const MAX_BUFFER_BYTES: usize = MAX_PACKET_BYTES + 64 * 1024;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OggPacket {
    pub data: Vec<u8>,
    pub serial: u32,
    pub first_in_stream: bool,
    pub last_in_page: bool,
    pub last_in_stream: bool,
    pub granule_position: Option<u64>,
}

#[derive(Clone, Copy, Debug)]
struct OggPageHeader {
    version: u8,
    header_type: u8,
    granule_position: u64,
    serial: u32,
    sequence: u32,
    checksum: u32,
    segment_count: usize,
}

impl OggPageHeader {
    fn parse(data: &[u8]) -> Option<Self> {
        if data.len() < 27 || &data[..4] != b"OggS" {
            return None;
        }
        Some(Self {
            version: data[4],
            header_type: data[5],
            granule_position: u64::from_le_bytes(data[6..14].try_into().ok()?),
            serial: u32::from_le_bytes(data[14..18].try_into().ok()?),
            sequence: u32::from_le_bytes(data[18..22].try_into().ok()?),
            checksum: u32::from_le_bytes(data[22..26].try_into().ok()?),
            segment_count: usize::from(data[26]),
        })
    }

    fn continued(self) -> bool {
        self.header_type & 0x01 != 0
    }

    fn bos(self) -> bool {
        self.header_type & 0x02 != 0
    }

    fn eos(self) -> bool {
        self.header_type & 0x04 != 0
    }
}

/// Bounded incremental Ogg page and packet parser shared by all codecs.
pub struct OggPacketParser {
    buffer: Vec<u8>,
    cursor: usize,
    packet_buffer: Vec<u8>,
    pending: VecDeque<OggPacket>,
    serial: Option<u32>,
    next_sequence: Option<u32>,
    seen_eos: bool,
    last_granule: Option<u64>,
}

impl OggPacketParser {
    pub fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(8192),
            cursor: 0,
            packet_buffer: Vec::with_capacity(4096),
            pending: VecDeque::new(),
            serial: None,
            next_sequence: None,
            seen_eos: false,
            last_granule: None,
        }
    }

    pub fn add(&mut self, data: &[u8]) -> Result<Vec<OggPacket>, String> {
        if data.len() > MAX_INPUT_CHUNK_BYTES {
            return Err(format!(
                "Ogg input chunk exceeds the {MAX_INPUT_CHUNK_BYTES} byte streaming budget"
            ));
        }
        let new_len = self
            .buffer
            .len()
            .checked_add(data.len())
            .ok_or_else(|| "Ogg input buffer size overflow".to_string())?;
        if new_len > MAX_BUFFER_BYTES {
            return Err(format!(
                "Ogg input buffer exceeds the {MAX_BUFFER_BYTES} byte budget"
            ));
        }
        self.buffer.extend_from_slice(data);
        self.parse_available()?;
        Ok(self.pending.drain(..).collect())
    }

    pub fn finish(&mut self) -> Result<Vec<OggPacket>, String> {
        self.parse_available()?;
        if !self.packet_buffer.is_empty() {
            return Err("truncated Ogg packet at end of input".to_string());
        }
        if self.buffer.len() != self.cursor {
            return Err(format!(
                "truncated Ogg page: {} bytes remain",
                self.buffer.len() - self.cursor
            ));
        }
        if self.serial.is_some() && !self.seen_eos {
            return Err("Ogg logical stream is missing EOS".to_string());
        }
        Ok(self.pending.drain(..).collect())
    }

    fn parse_available(&mut self) -> Result<(), String> {
        loop {
            let remaining = &self.buffer[self.cursor..];
            if remaining.len() < 27 {
                break;
            }
            if &remaining[..4] != b"OggS" {
                return Err(format!(
                    "invalid Ogg capture pattern at byte {}",
                    self.cursor
                ));
            }
            let header = OggPageHeader::parse(remaining)
                .ok_or_else(|| "invalid Ogg page header".to_string())?;
            let header_size = 27usize
                .checked_add(header.segment_count)
                .ok_or_else(|| "Ogg page header size overflow".to_string())?;
            if remaining.len() < header_size {
                break;
            }
            let body_size = remaining[27..header_size]
                .iter()
                .try_fold(0usize, |total, size| total.checked_add(usize::from(*size)))
                .ok_or_else(|| "Ogg page body size overflow".to_string())?;
            let total_size = header_size
                .checked_add(body_size)
                .ok_or_else(|| "Ogg page size overflow".to_string())?;
            if remaining.len() < total_size {
                break;
            }
            let checksum = ogg_page_crc(&remaining[..total_size]);
            self.validate_page(header, checksum)?;
            let page = &self.buffer[self.cursor..self.cursor + total_size];
            let body_start = header_size;
            let mut body_offset = 0usize;
            let mut completed = Vec::new();
            for segment_index in 0..header.segment_count {
                let size = usize::from(page[27 + segment_index]);
                let start = body_start + body_offset;
                let end = start + size;
                let packet_len = self
                    .packet_buffer
                    .len()
                    .checked_add(size)
                    .ok_or_else(|| "Ogg packet size overflow".to_string())?;
                if packet_len > MAX_PACKET_BYTES {
                    return Err(format!(
                        "Ogg packet exceeds the {MAX_PACKET_BYTES} byte packet budget"
                    ));
                }
                self.packet_buffer.extend_from_slice(&page[start..end]);
                body_offset += size;
                if size < 255 {
                    completed.push(OggPacket {
                        data: std::mem::take(&mut self.packet_buffer),
                        serial: header.serial,
                        first_in_stream: header.bos(),
                        last_in_page: false,
                        last_in_stream: false,
                        granule_position: None,
                    });
                }
            }
            if let Some(last) = completed.last_mut() {
                last.last_in_page = true;
                last.last_in_stream = header.eos();
                last.granule_position =
                    (header.granule_position != u64::MAX).then_some(header.granule_position);
            }
            if header.eos() {
                if !self.packet_buffer.is_empty() || completed.is_empty() {
                    return Err("Ogg EOS page does not end with a complete packet".to_string());
                }
                if header.granule_position == u64::MAX {
                    return Err("Ogg EOS page has no final granule position".to_string());
                }
                self.seen_eos = true;
            }
            self.pending.extend(completed);
            self.cursor += total_size;
        }
        if self.cursor > 64 * 1024 || self.cursor == self.buffer.len() {
            self.buffer.drain(..self.cursor);
            self.cursor = 0;
        }
        Ok(())
    }

    fn validate_page(&mut self, header: OggPageHeader, crc: u32) -> Result<(), String> {
        if header.version != 0 {
            return Err(format!("unsupported Ogg page version {}", header.version));
        }
        if crc != header.checksum {
            return Err(format!("Ogg CRC mismatch on page {}", header.sequence));
        }
        match self.serial {
            None => {
                if !header.bos() || header.sequence != 0 {
                    return Err("Ogg logical stream does not begin with BOS page zero".to_string());
                }
                self.serial = Some(header.serial);
            }
            Some(serial) if serial != header.serial => {
                return Err("chained or multiplexed Ogg streams are unsupported".to_string())
            }
            Some(_) if header.bos() => {
                return Err("Ogg BOS flag appears after the first page".to_string())
            }
            Some(_) => {}
        }
        if self.seen_eos {
            return Err("Ogg page appears after EOS".to_string());
        }
        if let Some(expected) = self.next_sequence {
            if header.sequence != expected {
                return Err(format!(
                    "Ogg page sequence discontinuity: expected {expected}, got {}",
                    header.sequence
                ));
            }
        }
        self.next_sequence = Some(header.sequence.wrapping_add(1));
        if header.granule_position != u64::MAX {
            if self
                .last_granule
                .is_some_and(|previous| header.granule_position < previous)
            {
                return Err("Ogg granule position moved backwards".to_string());
            }
            self.last_granule = Some(header.granule_position);
        }
        if header.continued() != !self.packet_buffer.is_empty() {
            return Err(if header.continued() {
                "Ogg continued-packet flag has no preceding packet".to_string()
            } else {
                "Ogg packet continuation is missing its continued flag".to_string()
            });
        }
        Ok(())
    }
}

impl Default for OggPacketParser {
    fn default() -> Self {
        Self::new()
    }
}

pub fn ogg_page_crc(page: &[u8]) -> u32 {
    let mut crc = 0u32;
    for (index, byte) in page.iter().copied().enumerate() {
        let byte = if (22..26).contains(&index) { 0 } else { byte };
        crc ^= u32::from(byte) << 24;
        for _ in 0..8 {
            crc = if crc & 0x8000_0000 != 0 {
                (crc << 1) ^ 0x04C1_1DB7
            } else {
                crc << 1
            };
        }
    }
    crc
}

#[cfg(test)]
mod tests {
    use super::*;

    fn page(
        header_type: u8,
        granule: u64,
        serial: u32,
        sequence: u32,
        lacing: &[u8],
        body: &[u8],
    ) -> Vec<u8> {
        assert_eq!(
            lacing.iter().map(|size| usize::from(*size)).sum::<usize>(),
            body.len()
        );
        let mut page = vec![0; 27 + lacing.len() + body.len()];
        page[..4].copy_from_slice(b"OggS");
        page[5] = header_type;
        page[6..14].copy_from_slice(&granule.to_le_bytes());
        page[14..18].copy_from_slice(&serial.to_le_bytes());
        page[18..22].copy_from_slice(&sequence.to_le_bytes());
        page[26] = lacing.len() as u8;
        page[27..27 + lacing.len()].copy_from_slice(lacing);
        page[27 + lacing.len()..].copy_from_slice(body);
        let checksum = ogg_page_crc(&page);
        page[22..26].copy_from_slice(&checksum.to_le_bytes());
        page
    }

    #[test]
    fn crc_matches_the_ogg_polynomial_reference() {
        assert_eq!(ogg_page_crc(&[]), 0);
        assert_ne!(ogg_page_crc(b"OggS"), 0);
    }

    #[test]
    fn validates_crc_sequence_continuation_and_eos() {
        let first = page(0x02, 0, 7, 0, &[1], &[0xaa]);
        let last = page(0x04, 1, 7, 1, &[1], &[0xbb]);
        let mut parser = OggPacketParser::new();
        assert_eq!(parser.add(&first).unwrap().len(), 1);
        let packets = parser.add(&last).unwrap();
        assert_eq!(packets.len(), 1);
        assert!(packets[0].last_in_stream);
        parser.finish().unwrap();

        let mut corrupt = first.clone();
        *corrupt.last_mut().unwrap() ^= 1;
        assert!(OggPacketParser::new()
            .add(&corrupt)
            .unwrap_err()
            .contains("CRC"));

        let mut parser = OggPacketParser::new();
        parser.add(&first).unwrap();
        assert!(parser
            .add(&page(0x04, 1, 7, 2, &[1], &[0xbb]))
            .unwrap_err()
            .contains("sequence"));

        let mut parser = OggPacketParser::new();
        parser.add(&page(0x02, 0, 7, 0, &[255], &[0; 255])).unwrap();
        assert!(parser
            .add(&page(0x04, 1, 7, 1, &[1], &[0xbb]))
            .unwrap_err()
            .contains("continued flag"));
    }

    #[test]
    fn rejects_truncation_and_enforces_streaming_budgets() {
        let first = page(0x02, 0, 9, 0, &[1], &[0xaa]);
        let mut parser = OggPacketParser::new();
        parser.add(&first[..first.len() - 1]).unwrap();
        assert!(parser.finish().unwrap_err().contains("truncated Ogg page"));

        assert!(OggPacketParser::new()
            .add(&vec![0; MAX_INPUT_CHUNK_BYTES + 1])
            .unwrap_err()
            .contains("streaming budget"));

        let mut parser = OggPacketParser::new();
        parser.packet_buffer = vec![0; MAX_PACKET_BYTES];
        parser.serial = Some(9);
        parser.next_sequence = Some(1);
        assert!(parser
            .add(&page(0x01, 1, 9, 1, &[1], &[0xbb]))
            .unwrap_err()
            .contains("packet budget"));
    }
}
