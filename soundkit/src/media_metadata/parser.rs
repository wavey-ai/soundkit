use super::{ArtworkMetadata, AudioTrackMetadata, MediaMetadata};

const MAX_TAG_BYTES: usize = 64 * 1024 * 1024;
const MAX_TAG_COUNT: usize = 100_000;
const MAX_ARTWORK_BYTES: usize = 32 * 1024 * 1024;
const MAX_ARTWORK_COUNT: usize = 64;

/// Extract normalized tags and technical audio information without decoding
/// the media payload. Unknown formats return an empty metadata value.
pub fn extract_metadata(bytes: &[u8]) -> Result<MediaMetadata, String> {
    let mut metadata = MediaMetadata::default();
    if bytes.starts_with(b"fLaC") {
        metadata.container = Some("flac".to_owned());
        parse_flac(bytes, &mut metadata)?;
    } else if bytes.starts_with(b"OggS") {
        metadata.container = Some("ogg".to_owned());
        parse_ogg(bytes, &mut metadata)?;
    } else if bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WAVE") {
        metadata.container = Some("wav".to_owned());
        parse_riff(bytes, &mut metadata)?;
    } else if bytes.starts_with(b"FORM") {
        metadata.container = Some("aiff".to_owned());
        parse_aiff(bytes, &mut metadata)?;
    } else if bytes.starts_with(&ASF_HEADER_GUID) {
        metadata.container = Some("asf".to_owned());
        parse_asf(bytes, &mut metadata)?;
    } else if looks_like_mp4(bytes) {
        metadata.container = Some("mp4".to_owned());
        parse_mp4(bytes, &mut metadata)?;
    } else if bytes.starts_with(b"\x1a\x45\xdf\xa3") {
        metadata.container = Some(
            if bytes.windows(4).take(4_096).any(|window| window == b"webm") {
                "matroska/webm".to_owned()
            } else {
                "matroska".to_owned()
            },
        );
        parse_matroska(bytes, &mut metadata)?;
    }

    if bytes.starts_with(b"ID3") {
        metadata.container.get_or_insert_with(|| "id3".to_owned());
        parse_id3v2(bytes, &mut metadata)?;
    }
    parse_apev2(bytes, &mut metadata)?;
    parse_id3v1(bytes, &mut metadata);
    Ok(metadata)
}

fn parse_flac(bytes: &[u8], metadata: &mut MediaMetadata) -> Result<(), String> {
    let mut offset = 4usize;
    let mut blocks = 0usize;
    loop {
        let header = bytes
            .get(offset..offset + 4)
            .ok_or_else(|| "FLAC metadata header is truncated".to_owned())?;
        offset += 4;
        blocks += 1;
        if blocks > MAX_TAG_COUNT {
            return Err("FLAC metadata block count exceeds budget".to_owned());
        }
        let is_last = header[0] & 0x80 != 0;
        let block_type = header[0] & 0x7f;
        let length =
            (usize::from(header[1]) << 16) | (usize::from(header[2]) << 8) | usize::from(header[3]);
        if length > MAX_TAG_BYTES {
            return Err("FLAC metadata block exceeds budget".to_owned());
        }
        let payload = bytes
            .get(offset..offset + length)
            .ok_or_else(|| "FLAC metadata payload is truncated".to_owned())?;
        offset += length;
        match block_type {
            0 => parse_flac_streaminfo(payload, metadata)?,
            4 => parse_vorbis_comments(payload, metadata)?,
            6 => parse_flac_picture(payload, metadata)?,
            _ => {}
        }
        if is_last {
            return Ok(());
        }
    }
}

fn parse_flac_streaminfo(payload: &[u8], metadata: &mut MediaMetadata) -> Result<(), String> {
    if payload.len() != 34 {
        return Err("FLAC STREAMINFO must contain 34 bytes".to_owned());
    }
    let packed = u64::from_be_bytes(
        payload[10..18]
            .try_into()
            .map_err(|_| "FLAC STREAMINFO is truncated")?,
    );
    let sample_rate = (packed >> 44) as u32;
    let channels = u16::try_from(((packed >> 41) & 7) + 1).unwrap();
    let bits_per_sample = u8::try_from(((packed >> 36) & 31) + 1).unwrap();
    let total_samples = packed & 0x0fff_fffff;
    let duration_micros = (sample_rate != 0 && total_samples != 0)
        .then(|| total_samples.saturating_mul(1_000_000) / u64::from(sample_rate));
    metadata.duration_micros = metadata.duration_micros.or(duration_micros);
    metadata.audio_tracks.push(AudioTrackMetadata {
        codec: Some("flac".to_owned()),
        sample_rate: Some(sample_rate),
        channels: Some(channels),
        bits_per_sample: Some(bits_per_sample),
        duration_micros,
        ..AudioTrackMetadata::default()
    });
    Ok(())
}

fn parse_vorbis_comments(payload: &[u8], metadata: &mut MediaMetadata) -> Result<(), String> {
    let mut cursor = 0usize;
    let vendor = take_le_string(payload, &mut cursor, "Vorbis vendor")?;
    metadata.insert_tag("VENDOR", vendor);
    let count = take_le_u32(payload, &mut cursor, "Vorbis comment count")? as usize;
    if count > MAX_TAG_COUNT {
        return Err("Vorbis comment count exceeds budget".to_owned());
    }
    for _ in 0..count {
        let comment = take_le_string(payload, &mut cursor, "Vorbis comment")?;
        if let Some((key, value)) = comment.split_once('=') {
            if key.eq_ignore_ascii_case("METADATA_BLOCK_PICTURE") {
                let picture = decode_base64_bounded(value.as_bytes())?;
                parse_flac_picture(&picture, metadata)?;
            } else {
                metadata.insert_tag(key, value);
            }
        }
    }
    Ok(())
}

fn take_le_u32(bytes: &[u8], cursor: &mut usize, context: &str) -> Result<u32, String> {
    let value = bytes
        .get(*cursor..*cursor + 4)
        .ok_or_else(|| format!("{context} is truncated"))?;
    *cursor += 4;
    Ok(u32::from_le_bytes(value.try_into().unwrap()))
}

fn take_le_string(bytes: &[u8], cursor: &mut usize, context: &str) -> Result<String, String> {
    let length = take_le_u32(bytes, cursor, context)? as usize;
    if length > MAX_TAG_BYTES {
        return Err(format!("{context} exceeds budget"));
    }
    let value = bytes
        .get(*cursor..*cursor + length)
        .ok_or_else(|| format!("{context} is truncated"))?;
    *cursor += length;
    String::from_utf8(value.to_vec()).map_err(|_| format!("{context} is not UTF-8"))
}

fn parse_flac_picture(payload: &[u8], metadata: &mut MediaMetadata) -> Result<(), String> {
    let mut cursor = 0usize;
    let picture_type = take_be_u32(payload, &mut cursor, "FLAC picture type")?;
    let mime_type = take_be_string(payload, &mut cursor, "FLAC picture MIME type")?;
    let description = take_be_string(payload, &mut cursor, "FLAC picture description")?;
    let width = take_be_u32(payload, &mut cursor, "FLAC picture width")?;
    let height = take_be_u32(payload, &mut cursor, "FLAC picture height")?;
    let color_depth = take_be_u32(payload, &mut cursor, "FLAC picture color depth")?;
    let indexed_colors = take_be_u32(payload, &mut cursor, "FLAC picture palette size")?;
    let length = take_be_u32(payload, &mut cursor, "FLAC picture data length")? as usize;
    if length > MAX_ARTWORK_BYTES {
        return Err("embedded artwork exceeds budget".to_owned());
    }
    let data = payload
        .get(cursor..cursor + length)
        .ok_or_else(|| "FLAC picture data is truncated".to_owned())?;
    if cursor + length != payload.len() {
        return Err("FLAC picture block has trailing data".to_owned());
    }
    push_artwork(
        metadata,
        ArtworkMetadata {
            picture_type: Some(picture_type),
            mime_type: nonempty(mime_type),
            description: nonempty(description),
            width: Some(width).filter(|value| *value != 0),
            height: Some(height).filter(|value| *value != 0),
            color_depth: Some(color_depth).filter(|value| *value != 0),
            indexed_colors: Some(indexed_colors).filter(|value| *value != 0),
            data: data.to_vec(),
        },
    )
}

fn take_be_u32(bytes: &[u8], cursor: &mut usize, context: &str) -> Result<u32, String> {
    let value = bytes
        .get(*cursor..*cursor + 4)
        .ok_or_else(|| format!("{context} is truncated"))?;
    *cursor += 4;
    Ok(u32::from_be_bytes(value.try_into().unwrap()))
}

fn take_be_string(bytes: &[u8], cursor: &mut usize, context: &str) -> Result<String, String> {
    let length = take_be_u32(bytes, cursor, context)? as usize;
    if length > MAX_TAG_BYTES {
        return Err(format!("{context} exceeds budget"));
    }
    let value = bytes
        .get(*cursor..*cursor + length)
        .ok_or_else(|| format!("{context} is truncated"))?;
    *cursor += length;
    String::from_utf8(value.to_vec()).map_err(|_| format!("{context} is not UTF-8"))
}

fn decode_base64_bounded(bytes: &[u8]) -> Result<Vec<u8>, String> {
    let useful = bytes
        .iter()
        .filter(|byte| !byte.is_ascii_whitespace())
        .count();
    if useful > MAX_ARTWORK_BYTES.saturating_add(2) / 3 * 4 + 4 {
        return Err("base64 artwork exceeds budget".to_owned());
    }
    if useful == 0 || useful % 4 != 0 {
        return Err("invalid base64 artwork length".to_owned());
    }
    let mut output = Vec::with_capacity(useful / 4 * 3);
    let mut quartet = [0u8; 4];
    let mut count = 0usize;
    let mut finished = false;
    for &byte in bytes {
        if byte.is_ascii_whitespace() {
            continue;
        }
        if finished {
            return Err("base64 artwork has data after padding".to_owned());
        }
        quartet[count] = byte;
        count += 1;
        if count != 4 {
            continue;
        }
        let padding = usize::from(quartet[3] == b'=') + usize::from(quartet[2] == b'=');
        if quartet[2] == b'=' && quartet[3] != b'=' {
            return Err("invalid base64 artwork padding".to_owned());
        }
        let a = base64_value(quartet[0])?;
        let b = base64_value(quartet[1])?;
        let c = if quartet[2] == b'=' {
            0
        } else {
            base64_value(quartet[2])?
        };
        let d = if quartet[3] == b'=' {
            0
        } else {
            base64_value(quartet[3])?
        };
        output.push((a << 2) | (b >> 4));
        if padding < 2 {
            output.push((b << 4) | (c >> 2));
        }
        if padding == 0 {
            output.push((c << 6) | d);
        }
        if output.len() > MAX_ARTWORK_BYTES {
            return Err("decoded artwork exceeds budget".to_owned());
        }
        finished = padding != 0;
        count = 0;
    }
    Ok(output)
}

fn base64_value(byte: u8) -> Result<u8, String> {
    match byte {
        b'A'..=b'Z' => Ok(byte - b'A'),
        b'a'..=b'z' => Ok(byte - b'a' + 26),
        b'0'..=b'9' => Ok(byte - b'0' + 52),
        b'+' => Ok(62),
        b'/' => Ok(63),
        _ => Err("invalid base64 artwork character".to_owned()),
    }
}

fn parse_ogg(bytes: &[u8], metadata: &mut MediaMetadata) -> Result<(), String> {
    let mut offset = 0usize;
    let mut packet = Vec::new();
    let mut packets = 0usize;
    while offset < bytes.len() && packets < 16 {
        let header = bytes
            .get(offset..offset + 27)
            .ok_or_else(|| "Ogg page header is truncated".to_owned())?;
        if &header[..4] != b"OggS" || header[4] != 0 {
            return Err("invalid Ogg page header".to_owned());
        }
        let segments = usize::from(header[26]);
        let laces = bytes
            .get(offset + 27..offset + 27 + segments)
            .ok_or_else(|| "Ogg segment table is truncated".to_owned())?;
        let payload_length = laces
            .iter()
            .map(|&length| usize::from(length))
            .sum::<usize>();
        let mut payload_offset = offset + 27 + segments;
        let page_end = payload_offset
            .checked_add(payload_length)
            .filter(|end| *end <= bytes.len())
            .ok_or_else(|| "Ogg page payload is truncated".to_owned())?;
        for &lace in laces {
            let length = usize::from(lace);
            if packet.len().saturating_add(length) > MAX_TAG_BYTES {
                return Err("Ogg packet exceeds metadata budget".to_owned());
            }
            packet.extend_from_slice(&bytes[payload_offset..payload_offset + length]);
            payload_offset += length;
            if lace < 255 {
                parse_ogg_packet(&packet, metadata)?;
                packet.clear();
                packets += 1;
                if packets >= 16 {
                    break;
                }
            }
        }
        offset = page_end;
    }
    Ok(())
}

fn parse_ogg_packet(packet: &[u8], metadata: &mut MediaMetadata) -> Result<(), String> {
    if packet.starts_with(b"OpusHead") && packet.len() >= 19 {
        metadata.audio_tracks.push(AudioTrackMetadata {
            codec: Some("opus".to_owned()),
            sample_rate: Some(48_000),
            channels: Some(u16::from(packet[9])),
            ..AudioTrackMetadata::default()
        });
    } else if let Some(comments) = packet.strip_prefix(b"OpusTags") {
        parse_vorbis_comments(comments, metadata)?;
    } else if packet.starts_with(b"\x01vorbis") && packet.len() >= 30 {
        metadata.audio_tracks.push(AudioTrackMetadata {
            codec: Some("vorbis".to_owned()),
            sample_rate: Some(u32::from_le_bytes(packet[12..16].try_into().unwrap())),
            channels: Some(u16::from(packet[11])),
            ..AudioTrackMetadata::default()
        });
    } else if let Some(comments) = packet.strip_prefix(b"\x03vorbis") {
        parse_vorbis_comments(comments, metadata)?;
    }
    Ok(())
}

fn parse_riff(bytes: &[u8], metadata: &mut MediaMetadata) -> Result<(), String> {
    let declared = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let end = declared.saturating_add(8).min(bytes.len());
    let mut offset = 12usize;
    let mut audio = AudioTrackMetadata {
        codec: Some("pcm".to_owned()),
        ..AudioTrackMetadata::default()
    };
    let mut data_bytes = None;
    while offset + 8 <= end {
        let id = &bytes[offset..offset + 4];
        let length = u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().unwrap()) as usize;
        let payload_start = offset + 8;
        let payload_end = payload_start
            .checked_add(length)
            .filter(|value| *value <= end)
            .ok_or_else(|| "RIFF chunk is truncated".to_owned())?;
        let payload = &bytes[payload_start..payload_end];
        match id {
            b"fmt " if payload.len() >= 16 => {
                audio.channels = Some(u16::from_le_bytes(payload[2..4].try_into().unwrap()));
                audio.sample_rate = Some(u32::from_le_bytes(payload[4..8].try_into().unwrap()));
                audio.bits_per_sample =
                    Some(u16::from_le_bytes(payload[14..16].try_into().unwrap()).min(255) as u8);
            }
            b"data" => data_bytes = Some(length as u64),
            b"LIST" if payload.starts_with(b"INFO") => parse_riff_info(&payload[4..], metadata)?,
            b"ID3 " | b"id3 " => parse_id3v2(payload, metadata)?,
            _ => {}
        }
        offset = payload_end + (length & 1);
    }
    if let (Some(rate), Some(channels), Some(bits), Some(data_bytes)) = (
        audio.sample_rate,
        audio.channels,
        audio.bits_per_sample,
        data_bytes,
    ) {
        let bytes_per_frame = u64::from(channels) * u64::from(bits).div_ceil(8);
        if bytes_per_frame != 0 && rate != 0 {
            audio.duration_micros =
                Some((data_bytes / bytes_per_frame).saturating_mul(1_000_000) / u64::from(rate));
            metadata.duration_micros = metadata.duration_micros.or(audio.duration_micros);
        }
    }
    metadata.audio_tracks.push(audio);
    Ok(())
}

fn parse_riff_info(bytes: &[u8], metadata: &mut MediaMetadata) -> Result<(), String> {
    let mut offset = 0usize;
    while offset + 8 <= bytes.len() {
        let id = &bytes[offset..offset + 4];
        let length = u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().unwrap()) as usize;
        let start = offset + 8;
        let end = start
            .checked_add(length)
            .filter(|value| *value <= bytes.len())
            .ok_or_else(|| "RIFF INFO tag is truncated".to_owned())?;
        let value = decode_latin1(
            bytes[start..end]
                .strip_suffix(&[0])
                .unwrap_or(&bytes[start..end]),
        );
        let key = match id {
            b"INAM" => "TITLE",
            b"IPRD" => "ALBUM",
            b"IART" => "ARTIST",
            b"ICMT" => "COMMENT",
            b"ICRD" => "DATE",
            b"IGNR" => "GENRE",
            b"ICOP" => "COPYRIGHT",
            b"ISFT" => "ENCODER",
            _ => std::str::from_utf8(id).unwrap_or("RIFF"),
        };
        metadata.insert_tag(key, value);
        offset = end + (length & 1);
    }
    Ok(())
}

fn parse_aiff(bytes: &[u8], metadata: &mut MediaMetadata) -> Result<(), String> {
    if bytes.len() < 12 || (&bytes[8..12] != b"AIFF" && &bytes[8..12] != b"AIFC") {
        return Err("invalid AIFF FORM header".to_owned());
    }
    let end = (u32::from_be_bytes(bytes[4..8].try_into().unwrap()) as usize)
        .saturating_add(8)
        .min(bytes.len());
    let mut offset = 12usize;
    let mut audio = AudioTrackMetadata {
        codec: Some("pcm".to_owned()),
        ..AudioTrackMetadata::default()
    };
    while offset + 8 <= end {
        let id = &bytes[offset..offset + 4];
        let length = u32::from_be_bytes(bytes[offset + 4..offset + 8].try_into().unwrap()) as usize;
        let start = offset + 8;
        let payload_end = start
            .checked_add(length)
            .filter(|value| *value <= end)
            .ok_or_else(|| "AIFF chunk is truncated".to_owned())?;
        let payload = &bytes[start..payload_end];
        match id {
            b"COMM" if payload.len() >= 18 => {
                audio.channels = Some(u16::from_be_bytes(payload[..2].try_into().unwrap()));
                let frames = u32::from_be_bytes(payload[2..6].try_into().unwrap());
                audio.bits_per_sample =
                    Some(u16::from_be_bytes(payload[6..8].try_into().unwrap()).min(255) as u8);
                if let Some(rate) = parse_extended_80(&payload[8..18]) {
                    audio.sample_rate = Some(rate);
                    audio.duration_micros =
                        Some(u64::from(frames).saturating_mul(1_000_000) / u64::from(rate));
                    metadata.duration_micros = metadata.duration_micros.or(audio.duration_micros);
                }
            }
            b"NAME" => metadata.insert_tag("TITLE", decode_latin1(payload)),
            b"AUTH" => metadata.insert_tag("ARTIST", decode_latin1(payload)),
            b"(c) " => metadata.insert_tag("COPYRIGHT", decode_latin1(payload)),
            b"ANNO" => metadata.insert_tag("COMMENT", decode_latin1(payload)),
            b"ID3 " => parse_id3v2(payload, metadata)?,
            _ => {}
        }
        offset = payload_end + (length & 1);
    }
    metadata.audio_tracks.push(audio);
    Ok(())
}

fn parse_extended_80(bytes: &[u8]) -> Option<u32> {
    if bytes.len() != 10 {
        return None;
    }
    let exponent = i32::from(u16::from_be_bytes([bytes[0], bytes[1]]) & 0x7fff) - 16_383;
    let mantissa = u64::from_be_bytes(bytes[2..10].try_into().ok()?);
    if exponent == -16_383 || mantissa == 0 {
        return None;
    }
    let rate = (mantissa as f64) * 2_f64.powi(exponent - 63);
    (rate.is_finite() && rate >= 1.0 && rate <= u32::MAX as f64).then(|| rate.round() as u32)
}

fn parse_id3v2(bytes: &[u8], metadata: &mut MediaMetadata) -> Result<(), String> {
    let header = bytes
        .get(..10)
        .filter(|header| header.starts_with(b"ID3"))
        .ok_or_else(|| "ID3v2 header is truncated".to_owned())?;
    let version = header[3];
    if !(2..=4).contains(&version) {
        return Err(format!("unsupported ID3v2 version {version}"));
    }
    let size = synchsafe(&header[6..10])?;
    if size > MAX_TAG_BYTES {
        return Err("ID3v2 tag exceeds budget".to_owned());
    }
    let mut body = bytes
        .get(10..10 + size)
        .ok_or_else(|| "ID3v2 tag is truncated".to_owned())?
        .to_vec();
    if header[5] & 0x80 != 0 {
        body = remove_unsynchronization(&body);
    }
    let mut offset = if header[5] & 0x40 != 0 {
        extended_header_length(version, &body)?
    } else {
        0
    };
    let mut frames = 0usize;
    while offset < body.len() && frames < MAX_TAG_COUNT {
        let (id, size, header_size) = if version == 2 {
            let Some(header) = body.get(offset..offset + 6) else {
                break;
            };
            if header[..3].iter().all(|byte| *byte == 0) {
                break;
            }
            (
                String::from_utf8_lossy(&header[..3]).into_owned(),
                (usize::from(header[3]) << 16)
                    | (usize::from(header[4]) << 8)
                    | usize::from(header[5]),
                6,
            )
        } else {
            let Some(header) = body.get(offset..offset + 10) else {
                break;
            };
            if header[..4].iter().all(|byte| *byte == 0) {
                break;
            }
            let size = if version == 4 {
                synchsafe(&header[4..8])?
            } else {
                u32::from_be_bytes(header[4..8].try_into().unwrap()) as usize
            };
            (String::from_utf8_lossy(&header[..4]).into_owned(), size, 10)
        };
        let start = offset + header_size;
        let end = start
            .checked_add(size)
            .filter(|value| *value <= body.len())
            .ok_or_else(|| "ID3v2 frame is truncated".to_owned())?;
        parse_id3_frame(&id, &body[start..end], metadata)?;
        offset = end;
        frames += 1;
    }
    Ok(())
}

fn parse_id3_frame(id: &str, payload: &[u8], metadata: &mut MediaMetadata) -> Result<(), String> {
    if matches!(id, "APIC" | "PIC") {
        return parse_id3_picture(id, payload, metadata);
    }
    if id.starts_with('T') && id != "TXXX" && id != "TXX" {
        if let Some(value) = decode_id3_text(payload) {
            metadata.insert_tag(id, value);
        }
        return Ok(());
    }
    if matches!(id, "COMM" | "COM" | "USLT" | "ULT") && payload.len() >= 4 {
        let encoding = payload[0];
        let body = &payload[4..];
        let value_start = id3_terminated_len(body, encoding).min(body.len());
        if let Some(value) = decode_encoded_text(encoding, &body[value_start..]) {
            metadata.insert_tag(
                if id.starts_with('U') {
                    "LYRICS"
                } else {
                    "COMMENT"
                },
                value,
            );
        }
    } else if matches!(id, "TXXX" | "TXX") && !payload.is_empty() {
        let encoding = payload[0];
        let body = &payload[1..];
        let split = id3_terminated_len(body, encoding).min(body.len());
        let key = decode_encoded_text(encoding, &body[..split]).unwrap_or_else(|| id.to_owned());
        if let Some(value) = decode_encoded_text(encoding, &body[split..]) {
            metadata.insert_tag(key, value);
        }
    }
    Ok(())
}

fn parse_id3_picture(id: &str, payload: &[u8], metadata: &mut MediaMetadata) -> Result<(), String> {
    let (&encoding, body) = payload
        .split_first()
        .ok_or_else(|| "ID3 picture frame is truncated".to_owned())?;
    let (mime_type, body) = if id == "PIC" {
        let format = body
            .get(..3)
            .ok_or_else(|| "ID3v2.2 picture format is truncated".to_owned())?;
        let mime = match format {
            b"PNG" => Some("image/png".to_owned()),
            b"JPG" => Some("image/jpeg".to_owned()),
            b"GIF" => Some("image/gif".to_owned()),
            _ => nonempty(decode_latin1(format)),
        };
        (mime, &body[3..])
    } else {
        let mime_end = body
            .iter()
            .position(|byte| *byte == 0)
            .ok_or_else(|| "ID3 APIC MIME type is unterminated".to_owned())?;
        (
            nonempty(decode_latin1(&body[..mime_end])),
            &body[mime_end + 1..],
        )
    };
    let (&picture_type, body) = body
        .split_first()
        .ok_or_else(|| "ID3 picture type is truncated".to_owned())?;
    let split = id3_terminated_len(body, encoding);
    if split > body.len() {
        return Err("ID3 picture description is truncated".to_owned());
    }
    let terminator = if matches!(encoding, 1 | 2) { 2 } else { 1 };
    let description_end = split.saturating_sub(terminator);
    let description = decode_encoded_text(encoding, &body[..description_end]);
    let data = &body[split..];
    if data.len() > MAX_ARTWORK_BYTES {
        return Err("embedded artwork exceeds budget".to_owned());
    }
    push_artwork(
        metadata,
        ArtworkMetadata {
            picture_type: Some(u32::from(picture_type)),
            mime_type: mime_type.or_else(|| sniff_image_mime(data).map(str::to_owned)),
            description,
            data: data.to_vec(),
            ..ArtworkMetadata::default()
        },
    )
}

fn decode_id3_text(payload: &[u8]) -> Option<String> {
    let (&encoding, body) = payload.split_first()?;
    decode_encoded_text(encoding, body)
}

fn decode_encoded_text(encoding: u8, bytes: &[u8]) -> Option<String> {
    let value = match encoding {
        0 => decode_latin1(bytes),
        3 => String::from_utf8_lossy(bytes).into_owned(),
        1 => decode_utf16(bytes, None)?,
        2 => decode_utf16(bytes, Some(true))?,
        _ => return None,
    };
    let value = value.trim_matches('\0').trim().to_owned();
    (!value.is_empty()).then_some(value)
}

fn decode_utf16(bytes: &[u8], big_endian: Option<bool>) -> Option<String> {
    let (big_endian, bytes) = match (big_endian, bytes.get(..2)) {
        (None, Some([0xfe, 0xff])) => (true, &bytes[2..]),
        (None, Some([0xff, 0xfe])) => (false, &bytes[2..]),
        (None, _) => (false, bytes),
        (Some(value), _) => (value, bytes),
    };
    let words = bytes
        .chunks_exact(2)
        .map(|pair| {
            if big_endian {
                u16::from_be_bytes([pair[0], pair[1]])
            } else {
                u16::from_le_bytes([pair[0], pair[1]])
            }
        })
        .collect::<Vec<_>>();
    String::from_utf16(&words).ok()
}

fn id3_terminated_len(bytes: &[u8], encoding: u8) -> usize {
    if matches!(encoding, 1 | 2) {
        bytes
            .chunks_exact(2)
            .position(|pair| pair == [0, 0])
            .map_or(bytes.len(), |index| (index + 1) * 2)
    } else {
        bytes
            .iter()
            .position(|byte| *byte == 0)
            .map_or(bytes.len(), |index| index + 1)
    }
}

fn extended_header_length(version: u8, body: &[u8]) -> Result<usize, String> {
    let size_bytes = body
        .get(..4)
        .ok_or_else(|| "ID3v2 extended header is truncated".to_owned())?;
    let size = if version == 4 {
        synchsafe(size_bytes)?
    } else {
        u32::from_be_bytes(size_bytes.try_into().unwrap()) as usize + 4
    };
    (size <= body.len())
        .then_some(size)
        .ok_or_else(|| "ID3v2 extended header exceeds tag".to_owned())
}

fn synchsafe(bytes: &[u8]) -> Result<usize, String> {
    if bytes.len() != 4 || bytes.iter().any(|byte| byte & 0x80 != 0) {
        return Err("invalid synchsafe integer".to_owned());
    }
    Ok(bytes
        .iter()
        .fold(0usize, |value, byte| (value << 7) | usize::from(*byte)))
}

fn remove_unsynchronization(bytes: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(bytes.len());
    let mut offset = 0usize;
    while offset < bytes.len() {
        output.push(bytes[offset]);
        if bytes[offset] == 0xff && bytes.get(offset + 1) == Some(&0) {
            offset += 1;
        }
        offset += 1;
    }
    output
}

fn parse_id3v1(bytes: &[u8], metadata: &mut MediaMetadata) {
    let Some(tag) = bytes.get(bytes.len().saturating_sub(128)..) else {
        return;
    };
    if tag.len() != 128 || &tag[..3] != b"TAG" {
        return;
    }
    for (key, value) in [
        ("TITLE", &tag[3..33]),
        ("ARTIST", &tag[33..63]),
        ("ALBUM", &tag[63..93]),
        ("DATE", &tag[93..97]),
        ("COMMENT", &tag[97..127]),
    ] {
        metadata.insert_tag(key, decode_latin1(value).trim_matches('\0').trim());
    }
    if tag[125] == 0 && tag[126] != 0 {
        metadata.insert_tag("TRACK", tag[126].to_string());
    }
}

fn parse_apev2(bytes: &[u8], metadata: &mut MediaMetadata) -> Result<(), String> {
    let id3v1_start = bytes
        .len()
        .checked_sub(128)
        .filter(|offset| bytes.get(*offset..*offset + 3) == Some(b"TAG"));
    let footer_end = id3v1_start.unwrap_or(bytes.len());
    let Some(footer_start) = footer_end.checked_sub(32) else {
        return Ok(());
    };
    let footer = &bytes[footer_start..footer_end];
    if &footer[..8] != b"APETAGEX" {
        return Ok(());
    }
    let version = u32::from_le_bytes(footer[8..12].try_into().unwrap());
    if !matches!(version, 1000 | 2000) {
        return Err(format!("unsupported APE tag version {version}"));
    }
    let size = u32::from_le_bytes(footer[12..16].try_into().unwrap()) as usize;
    let count = u32::from_le_bytes(footer[16..20].try_into().unwrap()) as usize;
    if size < 32 || size > MAX_TAG_BYTES || size > footer_end {
        return Err("APE tag size is invalid or exceeds budget".to_owned());
    }
    if count > MAX_TAG_COUNT {
        return Err("APE tag item count exceeds budget".to_owned());
    }
    let items_start = footer_end - size;
    let items = &bytes[items_start..footer_start];
    let mut cursor = 0usize;
    for _ in 0..count {
        let header = items
            .get(cursor..cursor + 8)
            .ok_or_else(|| "APE item header is truncated".to_owned())?;
        let value_length = u32::from_le_bytes(header[..4].try_into().unwrap()) as usize;
        let flags = u32::from_le_bytes(header[4..8].try_into().unwrap());
        cursor += 8;
        if value_length > MAX_TAG_BYTES {
            return Err("APE item exceeds budget".to_owned());
        }
        let key_end = items[cursor..]
            .iter()
            .position(|byte| *byte == 0)
            .map(|length| cursor + length)
            .ok_or_else(|| "APE item key is unterminated".to_owned())?;
        if key_end - cursor > 255 {
            return Err("APE item key exceeds budget".to_owned());
        }
        let key = std::str::from_utf8(&items[cursor..key_end])
            .map_err(|_| "APE item key is not UTF-8".to_owned())?;
        cursor = key_end + 1;
        let value_end = cursor
            .checked_add(value_length)
            .filter(|end| *end <= items.len())
            .ok_or_else(|| "APE item value is truncated".to_owned())?;
        let value = &items[cursor..value_end];
        cursor = value_end;
        match (flags >> 1) & 3 {
            0 => {
                for text in value.split(|byte| *byte == 0) {
                    let text = std::str::from_utf8(text)
                        .map_err(|_| "APE text item is not UTF-8".to_owned())?;
                    metadata.insert_tag(key, text);
                }
            }
            1 if key.to_ascii_lowercase().starts_with("cover art (") => {
                let split = value.iter().position(|byte| *byte == 0).unwrap_or(0);
                let (filename, data) = if value.get(split) == Some(&0) {
                    (&value[..split], &value[split + 1..])
                } else {
                    (&[][..], value)
                };
                if data.len() > MAX_ARTWORK_BYTES {
                    return Err("embedded artwork exceeds budget".to_owned());
                }
                let lower = key.to_ascii_lowercase();
                let picture_type = if lower.contains("front") {
                    Some(3)
                } else if lower.contains("back") {
                    Some(4)
                } else {
                    Some(0)
                };
                push_artwork(
                    metadata,
                    ArtworkMetadata {
                        picture_type,
                        mime_type: sniff_image_mime(data).map(str::to_owned),
                        description: nonempty(String::from_utf8_lossy(filename).into_owned()),
                        data: data.to_vec(),
                        ..ArtworkMetadata::default()
                    },
                )?;
            }
            _ => {}
        }
    }
    Ok(())
}

const ASF_HEADER_GUID: [u8; 16] = [
    0x30, 0x26, 0xb2, 0x75, 0x8e, 0x66, 0xcf, 0x11, 0xa6, 0xd9, 0x00, 0xaa, 0x00, 0x62, 0xce, 0x6c,
];
const ASF_CONTENT_DESCRIPTION_GUID: [u8; 16] = [
    0x33, 0x26, 0xb2, 0x75, 0x8e, 0x66, 0xcf, 0x11, 0xa6, 0xd9, 0x00, 0xaa, 0x00, 0x62, 0xce, 0x6c,
];
const ASF_EXTENDED_CONTENT_DESCRIPTION_GUID: [u8; 16] = [
    0x40, 0xa4, 0xd0, 0xd2, 0x07, 0xe3, 0xd2, 0x11, 0x97, 0xf0, 0x00, 0xa0, 0xc9, 0x5e, 0xa8, 0x50,
];
const ASF_FILE_PROPERTIES_GUID: [u8; 16] = [
    0xa1, 0xdc, 0xab, 0x8c, 0x47, 0xa9, 0xcf, 0x11, 0x8e, 0xe4, 0x00, 0xc0, 0x0c, 0x20, 0x53, 0x65,
];
const ASF_STREAM_PROPERTIES_GUID: [u8; 16] = [
    0x91, 0x07, 0xdc, 0xb7, 0xb7, 0xa9, 0xcf, 0x11, 0x8e, 0xe6, 0x00, 0xc0, 0x0c, 0x20, 0x53, 0x65,
];
const ASF_AUDIO_MEDIA_GUID: [u8; 16] = [
    0x40, 0x9e, 0x69, 0xf8, 0x4d, 0x5b, 0xcf, 0x11, 0xa8, 0xfd, 0x00, 0x80, 0x5f, 0x5c, 0x44, 0x2b,
];
const ASF_HEADER_EXTENSION_GUID: [u8; 16] = [
    0xb5, 0x03, 0xbf, 0x5f, 0x2e, 0xa9, 0xcf, 0x11, 0x8e, 0xe3, 0x00, 0xc0, 0x0c, 0x20, 0x53, 0x65,
];
const ASF_METADATA_GUID: [u8; 16] = [
    0xea, 0xcb, 0xf8, 0xc5, 0xaf, 0x5b, 0x77, 0x48, 0x84, 0x67, 0xaa, 0x8c, 0x44, 0xfa, 0x4c, 0xca,
];
const ASF_METADATA_LIBRARY_GUID: [u8; 16] = [
    0x94, 0x1c, 0x23, 0x44, 0x98, 0x94, 0xd1, 0x49, 0xa1, 0x41, 0x1d, 0x13, 0x4e, 0x45, 0x70, 0x54,
];

fn parse_asf(bytes: &[u8], metadata: &mut MediaMetadata) -> Result<(), String> {
    let header = bytes
        .get(..30)
        .ok_or_else(|| "ASF header is truncated".to_owned())?;
    let header_size = usize::try_from(u64::from_le_bytes(header[16..24].try_into().unwrap()))
        .map_err(|_| "ASF header size exceeds this platform".to_owned())?;
    if header_size < 30 || header_size > bytes.len() || header_size > MAX_TAG_BYTES {
        return Err("ASF header size is invalid or exceeds budget".to_owned());
    }
    let object_count = u32::from_le_bytes(header[24..28].try_into().unwrap()) as usize;
    if object_count > MAX_TAG_COUNT {
        return Err("ASF header object count exceeds budget".to_owned());
    }
    parse_asf_objects(bytes, 30, header_size, Some(object_count), metadata, 0)?;
    for track in &mut metadata.audio_tracks {
        track.duration_micros = track.duration_micros.or(metadata.duration_micros);
    }
    Ok(())
}

fn parse_asf_objects(
    bytes: &[u8],
    mut cursor: usize,
    end: usize,
    object_count: Option<usize>,
    metadata: &mut MediaMetadata,
    depth: usize,
) -> Result<(), String> {
    if depth > 4 {
        return Err("ASF metadata nesting exceeds budget".to_owned());
    }
    let mut parsed = 0usize;
    while cursor < end && object_count.map_or(true, |count| parsed < count) {
        let object_header = bytes
            .get(cursor..cursor + 24)
            .ok_or_else(|| "ASF object header is truncated".to_owned())?;
        let guid: [u8; 16] = object_header[..16].try_into().unwrap();
        let size = usize::try_from(u64::from_le_bytes(
            object_header[16..24].try_into().unwrap(),
        ))
        .map_err(|_| "ASF object size exceeds this platform".to_owned())?;
        if size < 24 || size > MAX_TAG_BYTES {
            return Err("ASF object size is invalid or exceeds budget".to_owned());
        }
        let object_end = cursor
            .checked_add(size)
            .filter(|object_end| *object_end <= end)
            .ok_or_else(|| "ASF object extends past its parent".to_owned())?;
        let payload = &bytes[cursor + 24..object_end];
        match guid {
            ASF_CONTENT_DESCRIPTION_GUID => parse_asf_content_description(payload, metadata)?,
            ASF_EXTENDED_CONTENT_DESCRIPTION_GUID => parse_asf_extended_content(payload, metadata)?,
            ASF_FILE_PROPERTIES_GUID => parse_asf_file_properties(payload, metadata)?,
            ASF_STREAM_PROPERTIES_GUID => parse_asf_stream_properties(payload, metadata)?,
            ASF_HEADER_EXTENSION_GUID => parse_asf_header_extension(payload, metadata, depth + 1)?,
            ASF_METADATA_GUID | ASF_METADATA_LIBRARY_GUID => {
                parse_asf_metadata_records(payload, metadata)?
            }
            _ => {}
        }
        cursor = object_end;
        parsed += 1;
        if parsed > MAX_TAG_COUNT {
            return Err("ASF object count exceeds budget".to_owned());
        }
    }
    if object_count.is_some_and(|count| parsed != count) {
        return Err("ASF header contains fewer objects than declared".to_owned());
    }
    Ok(())
}

fn parse_asf_content_description(
    payload: &[u8],
    metadata: &mut MediaMetadata,
) -> Result<(), String> {
    let lengths = payload
        .get(..10)
        .ok_or_else(|| "ASF content description is truncated".to_owned())?;
    let lengths = (0..5)
        .map(|index| {
            usize::from(u16::from_le_bytes([
                lengths[index * 2],
                lengths[index * 2 + 1],
            ]))
        })
        .collect::<Vec<_>>();
    let mut cursor = 10usize;
    for (key, length) in ["TITLE", "ARTIST", "COPYRIGHT", "COMMENT", "RATING"]
        .into_iter()
        .zip(lengths)
    {
        let value = payload
            .get(cursor..cursor + length)
            .ok_or_else(|| "ASF content-description string is truncated".to_owned())?;
        cursor += length;
        metadata.insert_tag(
            key,
            decode_utf16le(value, "ASF content-description string")?,
        );
    }
    Ok(())
}

fn parse_asf_extended_content(payload: &[u8], metadata: &mut MediaMetadata) -> Result<(), String> {
    let mut cursor = 0usize;
    let count = usize::from(take_asf_u16(payload, &mut cursor, "ASF descriptor count")?);
    if count > MAX_TAG_COUNT {
        return Err("ASF descriptor count exceeds budget".to_owned());
    }
    for _ in 0..count {
        let name_length = usize::from(take_asf_u16(payload, &mut cursor, "ASF tag name length")?);
        let name = take_asf_utf16(payload, &mut cursor, name_length, "ASF tag name")?;
        let data_type = take_asf_u16(payload, &mut cursor, "ASF tag data type")?;
        let value_length = usize::from(take_asf_u16(payload, &mut cursor, "ASF tag value length")?);
        let value = take_asf_bytes(payload, &mut cursor, value_length, "ASF tag value")?;
        parse_asf_value(&name, data_type, value, metadata)?;
    }
    Ok(())
}

fn parse_asf_metadata_records(payload: &[u8], metadata: &mut MediaMetadata) -> Result<(), String> {
    let mut cursor = 0usize;
    let count = usize::from(take_asf_u16(
        payload,
        &mut cursor,
        "ASF metadata record count",
    )?);
    if count > MAX_TAG_COUNT {
        return Err("ASF metadata record count exceeds budget".to_owned());
    }
    for _ in 0..count {
        take_asf_u16(payload, &mut cursor, "ASF metadata language index")?;
        take_asf_u16(payload, &mut cursor, "ASF metadata stream number")?;
        let name_length = usize::from(take_asf_u16(
            payload,
            &mut cursor,
            "ASF metadata name length",
        )?);
        let data_type = take_asf_u16(payload, &mut cursor, "ASF metadata data type")?;
        let value_length = usize::try_from(take_asf_u32(
            payload,
            &mut cursor,
            "ASF metadata value length",
        )?)
        .unwrap();
        if value_length > MAX_TAG_BYTES {
            return Err("ASF metadata value exceeds budget".to_owned());
        }
        let name = take_asf_utf16(payload, &mut cursor, name_length, "ASF metadata name")?;
        let value = take_asf_bytes(payload, &mut cursor, value_length, "ASF metadata value")?;
        parse_asf_value(&name, data_type, value, metadata)?;
    }
    Ok(())
}

fn parse_asf_value(
    name: &str,
    data_type: u16,
    value: &[u8],
    metadata: &mut MediaMetadata,
) -> Result<(), String> {
    if name.eq_ignore_ascii_case("WM/Picture") && data_type == 1 {
        return parse_asf_picture(value, metadata);
    }
    if data_type == 1 && value.starts_with(b"ID3") {
        return parse_id3v2(value, metadata);
    }
    let text = match data_type {
        0 => decode_utf16le(value, "ASF Unicode tag")?,
        2 | 3 if value.len() == 4 => u32::from_le_bytes(value.try_into().unwrap()).to_string(),
        4 if value.len() == 8 => u64::from_le_bytes(value.try_into().unwrap()).to_string(),
        5 if value.len() == 2 => u16::from_le_bytes(value.try_into().unwrap()).to_string(),
        _ => return Ok(()),
    };
    metadata.insert_tag(name, text);
    Ok(())
}

fn parse_asf_picture(payload: &[u8], metadata: &mut MediaMetadata) -> Result<(), String> {
    let picture_type = payload
        .first()
        .copied()
        .ok_or_else(|| "ASF picture is truncated".to_owned())?;
    let length = usize::try_from(u32::from_le_bytes(
        payload
            .get(1..5)
            .ok_or_else(|| "ASF picture length is truncated".to_owned())?
            .try_into()
            .unwrap(),
    ))
    .unwrap();
    if length > MAX_ARTWORK_BYTES {
        return Err("ASF embedded artwork exceeds budget".to_owned());
    }
    let mut cursor = 5usize;
    let mime_type = take_asf_utf16z(payload, &mut cursor, "ASF picture MIME type")?;
    let description = take_asf_utf16z(payload, &mut cursor, "ASF picture description")?;
    let data = payload
        .get(cursor..cursor + length)
        .ok_or_else(|| "ASF picture data is truncated".to_owned())?;
    push_artwork(
        metadata,
        ArtworkMetadata {
            picture_type: Some(u32::from(picture_type)),
            mime_type: nonempty(mime_type).or_else(|| sniff_image_mime(data).map(str::to_owned)),
            description: nonempty(description),
            data: data.to_vec(),
            ..ArtworkMetadata::default()
        },
    )
}

fn parse_asf_header_extension(
    payload: &[u8],
    metadata: &mut MediaMetadata,
    depth: usize,
) -> Result<(), String> {
    let header = payload
        .get(..22)
        .ok_or_else(|| "ASF header extension is truncated".to_owned())?;
    let length = usize::try_from(u32::from_le_bytes(header[18..22].try_into().unwrap())).unwrap();
    if length > MAX_TAG_BYTES {
        return Err("ASF header extension exceeds budget".to_owned());
    }
    let end = 22usize
        .checked_add(length)
        .filter(|end| *end <= payload.len())
        .ok_or_else(|| "ASF header extension payload is truncated".to_owned())?;
    parse_asf_objects(payload, 22, end, None, metadata, depth)
}

fn parse_asf_file_properties(payload: &[u8], metadata: &mut MediaMetadata) -> Result<(), String> {
    let fields = payload
        .get(..64)
        .ok_or_else(|| "ASF file properties are truncated".to_owned())?;
    let play_duration = u64::from_le_bytes(fields[40..48].try_into().unwrap()) / 10;
    let preroll_micros =
        u64::from_le_bytes(fields[56..64].try_into().unwrap()).saturating_mul(1_000);
    metadata.duration_micros = Some(play_duration.saturating_sub(preroll_micros));
    Ok(())
}

fn parse_asf_stream_properties(payload: &[u8], metadata: &mut MediaMetadata) -> Result<(), String> {
    if payload.get(..16) != Some(&ASF_AUDIO_MEDIA_GUID) {
        return Ok(());
    }
    let fields = payload
        .get(..72)
        .ok_or_else(|| "ASF audio stream properties are truncated".to_owned())?;
    let format_length =
        usize::try_from(u32::from_le_bytes(fields[40..44].try_into().unwrap())).unwrap();
    if format_length < 16 || 54usize.saturating_add(format_length) > payload.len() {
        return Err("ASF audio format is truncated".to_owned());
    }
    let format = &payload[54..54 + format_length];
    let format_tag = u16::from_le_bytes(format[0..2].try_into().unwrap());
    let channels = u16::from_le_bytes(format[2..4].try_into().unwrap());
    let sample_rate = u32::from_le_bytes(format[4..8].try_into().unwrap());
    let bitrate = u64::from(u32::from_le_bytes(format[8..12].try_into().unwrap())) * 8;
    let bits_per_sample = u8::try_from(u16::from_le_bytes(format[14..16].try_into().unwrap())).ok();
    let codec = match format_tag {
        0x0001 => "pcm",
        0x0160 => "wma1",
        0x0161 => "wma2",
        0x0162 => "wmapro",
        0x0163 => "wmalossless",
        0x000a => "wmavoice",
        _ => "asf-audio",
    };
    metadata.audio_tracks.push(AudioTrackMetadata {
        codec: Some(codec.to_owned()),
        codec_id: Some(format!("0x{format_tag:04x}")),
        sample_rate: Some(sample_rate),
        channels: Some(channels),
        bits_per_sample,
        bitrate: Some(bitrate),
        ..AudioTrackMetadata::default()
    });
    Ok(())
}

fn take_asf_u16(bytes: &[u8], cursor: &mut usize, context: &str) -> Result<u16, String> {
    let value = take_asf_bytes(bytes, cursor, 2, context)?;
    Ok(u16::from_le_bytes(value.try_into().unwrap()))
}

fn take_asf_u32(bytes: &[u8], cursor: &mut usize, context: &str) -> Result<u32, String> {
    let value = take_asf_bytes(bytes, cursor, 4, context)?;
    Ok(u32::from_le_bytes(value.try_into().unwrap()))
}

fn take_asf_bytes<'a>(
    bytes: &'a [u8],
    cursor: &mut usize,
    length: usize,
    context: &str,
) -> Result<&'a [u8], String> {
    let value = bytes
        .get(*cursor..cursor.saturating_add(length))
        .ok_or_else(|| format!("{context} is truncated"))?;
    *cursor += length;
    Ok(value)
}

fn take_asf_utf16(
    bytes: &[u8],
    cursor: &mut usize,
    length: usize,
    context: &str,
) -> Result<String, String> {
    decode_utf16le(take_asf_bytes(bytes, cursor, length, context)?, context)
}

fn take_asf_utf16z(bytes: &[u8], cursor: &mut usize, context: &str) -> Result<String, String> {
    let start = *cursor;
    let mut end = start;
    while end + 2 <= bytes.len() {
        if bytes[end] == 0 && bytes[end + 1] == 0 {
            let value = decode_utf16le(&bytes[start..end], context)?;
            *cursor = end + 2;
            return Ok(value);
        }
        end += 2;
    }
    Err(format!("{context} is not null terminated"))
}

fn decode_utf16le(bytes: &[u8], context: &str) -> Result<String, String> {
    if bytes.len() > MAX_TAG_BYTES || bytes.len() % 2 != 0 {
        return Err(format!("{context} has an invalid length"));
    }
    let units = bytes
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .take_while(|unit| *unit != 0)
        .collect::<Vec<_>>();
    String::from_utf16(&units).map_err(|_| format!("{context} is not valid UTF-16"))
}

fn decode_latin1(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| char::from(*byte)).collect()
}

fn looks_like_mp4(bytes: &[u8]) -> bool {
    bytes.len() >= 12
        && matches!(
            &bytes[4..8],
            b"ftyp" | b"moov" | b"free" | b"wide" | b"mdat"
        )
}

fn parse_mp4(bytes: &[u8], metadata: &mut MediaMetadata) -> Result<(), String> {
    for_each_mp4_box(bytes, 0, bytes.len(), |kind, payload| {
        if kind == b"moov" {
            parse_mp4_container(payload, metadata, 0)?;
        }
        Ok(())
    })
}

fn parse_mp4_container(
    bytes: &[u8],
    metadata: &mut MediaMetadata,
    depth: usize,
) -> Result<(), String> {
    if depth > 16 {
        return Err("MP4 metadata nesting exceeds budget".to_owned());
    }
    for_each_mp4_box(bytes, 0, bytes.len(), |kind, payload| {
        match kind {
            b"ilst" => parse_mp4_ilst(payload, metadata)?,
            b"meta" => {
                let nested = payload
                    .get(4..)
                    .ok_or_else(|| "MP4 meta full-box header is truncated".to_owned())?;
                parse_mp4_container(nested, metadata, depth + 1)?;
            }
            b"moov" | b"udta" => parse_mp4_container(payload, metadata, depth + 1)?,
            b"mvhd" => parse_mp4_duration(payload, metadata),
            _ => {}
        }
        Ok(())
    })
}

fn parse_mp4_duration(payload: &[u8], metadata: &mut MediaMetadata) {
    let Some(&version) = payload.first() else {
        return;
    };
    let fields = if version == 1 {
        payload.get(20..32).map(|value| {
            (
                u32::from_be_bytes(value[..4].try_into().unwrap()),
                u64::from_be_bytes(value[4..12].try_into().unwrap()),
            )
        })
    } else {
        payload.get(12..20).map(|value| {
            (
                u32::from_be_bytes(value[..4].try_into().unwrap()),
                u64::from(u32::from_be_bytes(value[4..8].try_into().unwrap())),
            )
        })
    };
    if let Some((timescale, duration)) = fields.filter(|(timescale, _)| *timescale != 0) {
        metadata.duration_micros = Some(duration.saturating_mul(1_000_000) / u64::from(timescale));
    }
}

fn parse_mp4_ilst(bytes: &[u8], metadata: &mut MediaMetadata) -> Result<(), String> {
    for_each_mp4_box(bytes, 0, bytes.len(), |kind, payload| {
        if kind == b"covr" {
            return for_each_mp4_box(payload, 0, payload.len(), |child, data| {
                if child != b"data" || data.len() < 8 {
                    return Ok(());
                }
                let image = &data[8..];
                if image.len() > MAX_ARTWORK_BYTES {
                    return Err("embedded artwork exceeds budget".to_owned());
                }
                let data_type = u32::from_be_bytes(data[..4].try_into().unwrap()) & 0x00ff_ffff;
                let mime_type = match data_type {
                    13 => Some("image/jpeg".to_owned()),
                    14 => Some("image/png".to_owned()),
                    _ => sniff_image_mime(image).map(str::to_owned),
                };
                push_artwork(
                    metadata,
                    ArtworkMetadata {
                        picture_type: Some(3),
                        mime_type,
                        data: image.to_vec(),
                        ..ArtworkMetadata::default()
                    },
                )
            });
        }
        let Some(key) = mp4_tag_key(kind) else {
            return Ok(());
        };
        for_each_mp4_box(payload, 0, payload.len(), |child, data| {
            if child == b"data" && data.len() >= 8 {
                let value = &data[8..];
                if matches!(kind, b"trkn" | b"disk") && value.len() >= 6 {
                    let number = u16::from_be_bytes([value[2], value[3]]);
                    let total = u16::from_be_bytes([value[4], value[5]]);
                    metadata.insert_tag(key, format!("{number}/{total}"));
                } else if kind == b"gnre" && value.len() >= 2 {
                    let index = u16::from_be_bytes(value[value.len() - 2..].try_into().unwrap());
                    if let Some(genre) = index.checked_sub(1).and_then(id3_genre) {
                        metadata.insert_tag(key, genre);
                    }
                } else {
                    let text = String::from_utf8_lossy(value)
                        .trim_matches('\0')
                        .trim()
                        .to_owned();
                    metadata.insert_tag(key, text);
                }
            }
            Ok(())
        })?;
        Ok(())
    })
}

fn push_artwork(metadata: &mut MediaMetadata, artwork: ArtworkMetadata) -> Result<(), String> {
    if artwork.data.len() > MAX_ARTWORK_BYTES {
        return Err("embedded artwork exceeds budget".to_owned());
    }
    if metadata.artwork.len() >= MAX_ARTWORK_COUNT {
        return Err("embedded artwork count exceeds budget".to_owned());
    }
    metadata.artwork.push(artwork);
    Ok(())
}

fn nonempty(value: String) -> Option<String> {
    let value = value.trim_matches('\0').trim().to_owned();
    (!value.is_empty()).then_some(value)
}

fn sniff_image_mime(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(b"\xff\xd8\xff") {
        Some("image/jpeg")
    } else if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        Some("image/png")
    } else if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        Some("image/gif")
    } else if bytes.starts_with(b"BM") {
        Some("image/bmp")
    } else if bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WEBP") {
        Some("image/webp")
    } else {
        None
    }
}

fn mp4_tag_key(kind: &[u8; 4]) -> Option<&'static str> {
    Some(match kind {
        [0xa9, b'n', b'a', b'm'] => "TITLE",
        [0xa9, b'a', b'l', b'b'] => "ALBUM",
        [0xa9, b'A', b'R', b'T'] => "ARTIST",
        b"aART" => "ALBUMARTIST",
        [0xa9, b'w', b'r', b't'] => "COMPOSER",
        [0xa9, b'g', b'e', b'n'] | b"gnre" => "GENRE",
        [0xa9, b'd', b'a', b'y'] => "DATE",
        [0xa9, b'c', b'm', b't'] => "COMMENT",
        [0xa9, b'l', b'y', b'r'] => "LYRICS",
        [0xa9, b't', b'o', b'o'] => "ENCODER",
        [0xa9, b'c', b'p', b'y'] => "COPYRIGHT",
        b"trkn" => "TRACK",
        b"disk" => "DISC",
        _ => return None,
    })
}

fn id3_genre(index: u16) -> Option<&'static str> {
    const GENRES: &[&str] = &[
        "Blues",
        "Classic Rock",
        "Country",
        "Dance",
        "Disco",
        "Funk",
        "Grunge",
        "Hip-Hop",
        "Jazz",
        "Metal",
        "New Age",
        "Oldies",
        "Other",
        "Pop",
        "R&B",
        "Rap",
        "Reggae",
        "Rock",
        "Techno",
        "Industrial",
        "Alternative",
        "Ska",
        "Death Metal",
        "Pranks",
        "Soundtrack",
        "Euro-Techno",
        "Ambient",
        "Trip-Hop",
        "Vocal",
        "Jazz+Funk",
        "Fusion",
        "Trance",
        "Classical",
        "Instrumental",
        "Acid",
        "House",
        "Game",
        "Sound Clip",
        "Gospel",
        "Noise",
        "Alternative Rock",
        "Bass",
        "Soul",
        "Punk",
        "Space",
        "Meditative",
        "Instrumental Pop",
        "Instrumental Rock",
        "Ethnic",
        "Gothic",
        "Darkwave",
        "Techno-Industrial",
        "Electronic",
        "Pop-Folk",
        "Eurodance",
        "Dream",
        "Southern Rock",
        "Comedy",
        "Cult",
        "Gangsta",
        "Top 40",
        "Christian Rap",
        "Pop/Funk",
        "Jungle",
        "Native American",
        "Cabaret",
        "New Wave",
        "Psychedelic",
        "Rave",
        "Showtunes",
        "Trailer",
        "Lo-Fi",
        "Tribal",
        "Acid Punk",
        "Acid Jazz",
        "Polka",
        "Retro",
        "Musical",
        "Rock & Roll",
        "Hard Rock",
    ];
    GENRES.get(usize::from(index)).copied()
}

fn for_each_mp4_box(
    bytes: &[u8],
    mut offset: usize,
    end: usize,
    mut visit: impl FnMut(&[u8; 4], &[u8]) -> Result<(), String>,
) -> Result<(), String> {
    let end = end.min(bytes.len());
    let mut boxes = 0usize;
    while offset + 8 <= end {
        boxes += 1;
        if boxes > MAX_TAG_COUNT {
            return Err("MP4 box count exceeds budget".to_owned());
        }
        let short = u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap());
        let kind: &[u8; 4] = bytes[offset + 4..offset + 8].try_into().unwrap();
        let (size, header) = if short == 1 {
            let extended = bytes
                .get(offset + 8..offset + 16)
                .ok_or_else(|| "MP4 extended box header is truncated".to_owned())?;
            (u64::from_be_bytes(extended.try_into().unwrap()), 16usize)
        } else if short == 0 {
            ((end - offset) as u64, 8)
        } else {
            (u64::from(short), 8)
        };
        let size = usize::try_from(size).map_err(|_| "MP4 box size exceeds usize")?;
        let box_end = offset
            .checked_add(size)
            .filter(|value| size >= header && *value <= end)
            .ok_or_else(|| "MP4 box range is invalid".to_owned())?;
        visit(kind, &bytes[offset + header..box_end])?;
        offset = box_end;
    }
    Ok(())
}

fn parse_matroska(bytes: &[u8], metadata: &mut MediaMetadata) -> Result<(), String> {
    let mut offset = 0usize;
    while offset < bytes.len() {
        let (id, id_len) = ebml_id(&bytes[offset..])?;
        let (size, size_len) = ebml_size(&bytes[offset + id_len..])?;
        let start = offset + id_len + size_len;
        let end = size
            .and_then(|size| start.checked_add(size))
            .unwrap_or(bytes.len())
            .min(bytes.len());
        if id == 0x1853_8067 {
            return parse_matroska_segment(&bytes[start..end], metadata);
        }
        offset = end;
    }
    Ok(())
}

fn parse_matroska_segment(bytes: &[u8], metadata: &mut MediaMetadata) -> Result<(), String> {
    let mut offset = 0usize;
    let mut elements = 0usize;
    while offset < bytes.len() {
        elements += 1;
        if elements > MAX_TAG_COUNT {
            return Err("Matroska element count exceeds budget".to_owned());
        }
        let (id, id_len) = ebml_id(&bytes[offset..])?;
        let (size, size_len) = ebml_size(&bytes[offset + id_len..])?;
        let start = offset + id_len + size_len;
        let end = size
            .and_then(|size| start.checked_add(size))
            .filter(|value| *value <= bytes.len())
            .ok_or_else(|| "Matroska element is truncated or unbounded".to_owned())?;
        if id == 0x1254_c367 {
            parse_matroska_tags(&bytes[start..end], metadata)?;
        }
        offset = end;
    }
    Ok(())
}

fn parse_matroska_tags(bytes: &[u8], metadata: &mut MediaMetadata) -> Result<(), String> {
    let mut stack = vec![bytes];
    while let Some(container) = stack.pop() {
        let mut offset = 0usize;
        while offset < container.len() {
            let (id, id_len) = ebml_id(&container[offset..])?;
            let (size, size_len) = ebml_size(&container[offset + id_len..])?;
            let start = offset + id_len + size_len;
            let end = size
                .and_then(|size| start.checked_add(size))
                .filter(|value| *value <= container.len())
                .ok_or_else(|| "Matroska tag element is truncated".to_owned())?;
            let payload = &container[start..end];
            if id == 0x67c8 {
                parse_matroska_simple_tag(payload, metadata)?;
            } else if matches!(id, 0x7373 | 0x1254_c367) {
                stack.push(payload);
            }
            offset = end;
        }
    }
    Ok(())
}

fn parse_matroska_simple_tag(bytes: &[u8], metadata: &mut MediaMetadata) -> Result<(), String> {
    let mut offset = 0usize;
    let mut name = None;
    let mut value = None;
    while offset < bytes.len() {
        let (id, id_len) = ebml_id(&bytes[offset..])?;
        let (size, size_len) = ebml_size(&bytes[offset + id_len..])?;
        let start = offset + id_len + size_len;
        let end = size
            .and_then(|size| start.checked_add(size))
            .filter(|end| *end <= bytes.len())
            .ok_or_else(|| "Matroska SimpleTag is truncated".to_owned())?;
        match id {
            0x45a3 => name = Some(String::from_utf8_lossy(&bytes[start..end]).into_owned()),
            0x4487 => value = Some(String::from_utf8_lossy(&bytes[start..end]).into_owned()),
            0x67c8 => parse_matroska_simple_tag(&bytes[start..end], metadata)?,
            _ => {}
        }
        offset = end;
    }
    if let (Some(name), Some(value)) = (name, value) {
        metadata.insert_tag(name, value);
    }
    Ok(())
}

fn ebml_id(bytes: &[u8]) -> Result<(u64, usize), String> {
    let first = *bytes
        .first()
        .ok_or_else(|| "EBML ID is truncated".to_owned())?;
    let length = first.leading_zeros() as usize + 1;
    if length > 4 || bytes.len() < length {
        return Err("invalid EBML ID".to_owned());
    }
    Ok((
        bytes[..length]
            .iter()
            .fold(0u64, |value, byte| (value << 8) | u64::from(*byte)),
        length,
    ))
}

fn ebml_size(bytes: &[u8]) -> Result<(Option<usize>, usize), String> {
    let first = *bytes
        .first()
        .ok_or_else(|| "EBML size is truncated".to_owned())?;
    let length = first.leading_zeros() as usize + 1;
    if length > 8 || bytes.len() < length {
        return Err("invalid EBML size".to_owned());
    }
    let marker = 1u8 << (8 - length);
    let mut value = u64::from(first & (marker - 1));
    for byte in &bytes[1..length] {
        value = (value << 8) | u64::from(*byte);
    }
    let unknown = value == (1u64 << (7 * length)) - 1;
    let size = (!unknown).then(|| usize::try_from(value).ok()).flatten();
    Ok((size, length))
}

#[cfg(test)]
mod tests {
    use super::{extract_metadata, ASF_HEADER_GUID};

    fn synchsafe(value: usize) -> [u8; 4] {
        [
            ((value >> 21) & 0x7f) as u8,
            ((value >> 14) & 0x7f) as u8,
            ((value >> 7) & 0x7f) as u8,
            (value & 0x7f) as u8,
        ]
    }

    fn id3_text(id: &[u8; 4], value: &str) -> Vec<u8> {
        let mut payload = vec![3];
        payload.extend_from_slice(value.as_bytes());
        let mut frame = id.to_vec();
        frame.extend_from_slice(&synchsafe(payload.len()));
        frame.extend_from_slice(&[0, 0]);
        frame.extend_from_slice(&payload);
        frame
    }

    #[test]
    fn extracts_id3_album_artist_and_title() {
        let mut body = id3_text(b"TIT2", "Test title");
        body.extend(id3_text(b"TALB", "Test album"));
        body.extend(id3_text(b"TPE1", "Test artist"));
        body.extend(id3_text(b"TPE2", "Test album artist"));
        body.extend(id3_text(b"TRCK", "4/11"));
        let mut tag = b"ID3\x04\0\0".to_vec();
        tag.extend_from_slice(&synchsafe(body.len()));
        tag.extend(body);

        let metadata = extract_metadata(&tag).unwrap();
        assert_eq!(metadata.title.as_deref(), Some("Test title"));
        assert_eq!(metadata.album.as_deref(), Some("Test album"));
        assert_eq!(metadata.artists, ["Test artist"]);
        assert_eq!(metadata.album_artists, ["Test album artist"]);
        assert_eq!(metadata.track_number, Some(4));
        assert_eq!(metadata.track_total, Some(11));
    }

    #[test]
    fn malformed_inputs_do_not_panic() {
        for length in 0..64 {
            let bytes = vec![0xff; length];
            let _ = extract_metadata(&bytes);
        }
    }

    #[test]
    fn asf_rejects_missing_declared_objects() {
        let mut bytes = ASF_HEADER_GUID.to_vec();
        bytes.extend_from_slice(&30_u64.to_le_bytes());
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&[1, 2]);

        let error = extract_metadata(&bytes).unwrap_err();
        assert!(error.contains("fewer objects"));
    }
}
