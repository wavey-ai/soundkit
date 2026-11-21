//! Opus Custom packet framing, as `opus_custom_encode` and
//! `opus_custom_decode` write and read it.
//!
//! An Opus Custom packet has no Opus table-of-contents byte. With signalling
//! on (the libopus default), the CELT frame is preceded by one byte:
//!
//! ```text
//! bits 7..5  (effEBands - end) / 2   bands removed from the top
//! bits 4..3  LM                      log2 of the short-block count
//! bit  2     stereo                  1 for a two-channel stream
//! bits 1..0  0
//! ```
//!
//! For the 48 kHz standard mode (2.5 ms short blocks), libopus rewrites that
//! byte as an Opus TOC (`toOpus`/`fromOpus` in `celt/celt.h`), so 120 to
//! 960-sample Opus Custom packets are also valid CELT-only Opus packets. With
//! signalling off (`CELT_SET_SIGNALLING(0)`) there is no byte at all and the
//! frame size, channel count and end band are known out of band.

use crate::celt::modes::CeltMode;
use crate::{Error, Result};

const TO_OPUS: [u8; 20] = [
    0xE0, 0xE8, 0xF0, 0xF8, 0xC0, 0xC8, 0xD0, 0xD8, 0xA0, 0xA8, 0xB0, 0xB8, 0x00, 0x00, 0x00, 0x00,
    0x80, 0x88, 0x90, 0x98,
];
const FROM_OPUS: [u8; 16] = [
    0x80, 0x88, 0x90, 0x98, 0x40, 0x48, 0x50, 0x58, 0x20, 0x28, 0x30, 0x38, 0x00, 0x08, 0x10, 0x18,
];

fn to_opus(c: u8) -> Option<u8> {
    let table = if c < 0xA0 {
        TO_OPUS[(c >> 3) as usize]
    } else {
        0
    };
    (table != 0).then_some(table | (c & 0x7))
}

fn from_opus(c: u8) -> Option<u8> {
    (c >= 0x80).then(|| FROM_OPUS[((c >> 3) - 16) as usize] | (c & 0x7))
}

/// True for modes whose signalling byte libopus rewrites as an Opus TOC.
fn uses_opus_toc(mode: &CeltMode) -> bool {
    mode.fs == 48_000 && mode.short_mdct_size == 120
}

/// The signalling byte `celt_encode_with_ec` writes, and the end band it
/// codes with.
pub(crate) fn signalling_byte(
    mode: &CeltMode,
    end: usize,
    lm: usize,
    stereo: bool,
) -> Result<(u8, usize)> {
    let removed = (mode.eff_ebands - end.min(mode.eff_ebands)) >> 1;
    let end = 1.max(mode.eff_ebands - removed);
    let byte = ((removed as u8) << 5) | ((lm as u8) << 3) | (u8::from(stereo) << 2);
    let byte = if uses_opus_toc(mode) {
        to_opus(byte).ok_or(Error::BadArg)?
    } else {
        byte
    };
    Ok((byte, end))
}

/// A parsed Opus Custom packet.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CustomFrame<'a> {
    pub frame: &'a [u8],
    pub lm: usize,
    pub stream_channels: usize,
    pub end: usize,
}

/// The end band a signalling byte sets, or `None` when the byte is not a
/// valid CELT-only TOC in the standard mode. `celt_decode_with_ec` stores it
/// in the decoder state before it validates the rest of the packet, so a
/// packet it then rejects still changes the end band.
pub(crate) fn signalled_end(mode: &CeltMode, first: u8) -> Option<usize> {
    let byte = if uses_opus_toc(mode) {
        from_opus(first)?
    } else {
        first
    };
    Some(1.max(mode.eff_ebands as isize - 2 * (byte >> 5) as isize) as usize)
}

/// Parses the signalling byte as `celt_decode_with_ec` does and returns the
/// CELT frame that follows it.
pub(crate) fn parse_signalled<'a>(mode: &CeltMode, data: &'a [u8]) -> Result<CustomFrame<'a>> {
    let first = *data.first().ok_or(Error::InvalidPacket)?;
    let byte = if uses_opus_toc(mode) {
        from_opus(first).ok_or(Error::InvalidPacket)?
    } else {
        first
    };
    let end = signalled_end(mode, first).ok_or(Error::InvalidPacket)?;
    let lm = ((byte >> 3) & 0x3) as usize;
    let stream_channels = 1 + ((byte >> 2) & 0x1) as usize;
    let mut frame = &data[1..];
    if first & 0x03 == 0x03 {
        // A code-3 packet with padding: libopus skips the frame-count byte
        // and drops the padding from the end, as it does before it reads a
        // QEXT extension. Without the padding flag the count byte stays.
        let count = *frame.first().ok_or(Error::InvalidPacket)?;
        if count & 0x40 != 0 {
            frame = &frame[1..];
            let mut padding = 0isize;
            loop {
                let p = *frame.first().ok_or(Error::InvalidPacket)?;
                frame = &frame[1..];
                let chunk = if p == 255 { 254 } else { p as usize };
                let remaining = frame.len() as isize - chunk as isize;
                if remaining <= 0 {
                    return Err(Error::InvalidPacket);
                }
                frame = &frame[..remaining as usize];
                padding += chunk as isize;
                if p != 255 {
                    break;
                }
            }
            if padding < 1 {
                return Err(Error::InvalidPacket);
            }
        }
    }
    if lm > mode.max_lm {
        return Err(Error::InvalidPacket);
    }
    Ok(CustomFrame {
        frame,
        lm,
        stream_channels,
        end,
    })
}

#[cfg(test)]
mod tests {
    use super::{from_opus, parse_signalled, signalling_byte, to_opus};
    use crate::celt::modes::CeltMode;

    #[test]
    fn to_opus_and_from_opus_round_trip_celt_bytes() {
        for lm in 0..4u8 {
            for stereo in 0..2u8 {
                let c = (lm << 3) | (stereo << 2);
                let toc = to_opus(c).expect("CELT byte maps to a TOC");
                assert_eq!(toc & 0x80, 0x80);
                assert_eq!(from_opus(toc), Some(c));
            }
        }
    }

    #[test]
    fn custom_mode_byte_carries_lm_channels_and_end() {
        let mode = CeltMode::custom_48k_shared(48).unwrap();
        let (byte, end) = signalling_byte(mode, mode.eff_ebands, 0, true).unwrap();
        assert_eq!((byte, end), (0x04, 15));
        let packet = [byte, 0xAA, 0x55];
        let parsed = parse_signalled(mode, &packet).unwrap();
        assert_eq!(parsed.frame, &[0xAA, 0x55]);
        assert_eq!((parsed.lm, parsed.stream_channels, parsed.end), (0, 2, 15));
        // As in libopus, the encoder codes `effEBands - removed` bands but
        // the decoder reads `effEBands - 2 * removed`; only a lowered end
        // band (CELT_SET_END_BAND) reaches this.
        let (byte, end) = signalling_byte(mode, 11, 0, false).unwrap();
        assert_eq!((byte, end), (0x40, 13));
        assert_eq!(parse_signalled(mode, &[byte, 0]).unwrap().end, 11);
    }

    #[test]
    fn standard_mode_byte_is_an_opus_toc() {
        let mode = CeltMode::custom_48k_shared(120).unwrap();
        let (byte, end) = signalling_byte(mode, 21, 1, true).unwrap();
        assert_eq!((byte, end), (0xEC, 21));
        let packet = [byte, 1, 2];
        let parsed = parse_signalled(mode, &packet).unwrap();
        assert_eq!((parsed.lm, parsed.stream_channels, parsed.end), (1, 2, 21));
        assert!(parse_signalled(mode, &[0x10, 1]).is_err());
    }
}
