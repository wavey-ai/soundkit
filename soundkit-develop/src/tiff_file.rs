//! A frame from a TIFF file. A TIFF holds an encoded photograph, so the
//! frame keeps the file's 8-bit or 16-bit samples and reads them through the
//! file's ICC profile, or as sRGB.

use crate::develop::{encoded, Encoded16, Frame, Samples};
use std::io::Cursor;
use tiff::decoder::{Decoder, DecodingResult, Limits};
use tiff::tags::Tag;
use tiff::{ColorType, TiffError, TiffUnsupportedError};

const MAX_PIXELS: usize = 80_000_000;

fn message(error: TiffError) -> String {
    match error {
        TiffError::UnsupportedError(TiffUnsupportedError::UnsupportedCompressionMethod(_) | TiffUnsupportedError::UnknownCompressionMethod) =>
            "This TIFF uses a compression that is not supported. Save it with LZW or ZIP compression and try again.",
        TiffError::UnsupportedError(_) => "This TIFF uses a format that is not supported. Save it as 8-bit or 16-bit RGB and try again.",
        TiffError::LimitsExceeded => "This TIFF is too large to open.",
        _ => "This TIFF could not be read.",
    }.into()
}
fn colour_model(name: &str) -> String { format!("This TIFF uses {name} colour. Save it as RGB and try again.") }

/// A sample of 8 or 16 bits.
trait Sample: Copy {
    const MAX: u32;
    fn get(self) -> u32;
    fn make(value: u32) -> Self;
}
impl Sample for u8 { const MAX: u32 = 255; fn get(self) -> u32 { self as u32 } fn make(value: u32) -> u8 { value as u8 } }
impl Sample for u16 { const MAX: u32 = 65535; fn get(self) -> u32 { self as u32 } fn make(value: u32) -> u16 { value as u16 } }

#[derive(Clone, Copy, PartialEq)]
enum Model { Grey, Rgb, YCbCr }
/// How the file stores a pixel: the colour model, the samples in a pixel,
/// planes in place of interleaved samples, and the opacity sample with
/// `true` when the colours are multiplied by it.
struct Layout { model: Model, channels: usize, planar: bool, alpha: Option<(usize, bool)> }

/// Interleaved RGB and 8-bit opacity from the stored samples.
fn unpack<S: Sample>(samples: Vec<S>, pixels: usize, layout: &Layout) -> (Vec<S>, Option<Vec<u8>>) {
    let Layout { model, channels, planar, alpha } = *layout;
    if model == Model::Rgb && channels == 3 && !planar { return (samples, None); }
    let at = |p: usize, c: usize| if planar { samples[c * pixels + p] } else { samples[p * channels + c] }.get();
    let mut rgb = Vec::with_capacity(pixels * 3);
    let mut opacity = alpha.map(|_| Vec::with_capacity(pixels));
    for p in 0..pixels {
        let mut colour = match model {
            Model::Grey => [at(p, 0); 3],
            Model::Rgb => [at(p, 0), at(p, 1), at(p, 2)],
            // Full-range BT.601, the TIFF and JFIF default.
            Model::YCbCr => {
                let (y, cb, cr) = (at(p, 0) as f32, at(p, 1) as f32 - 128.0, at(p, 2) as f32 - 128.0);
                [y + 1.402 * cr, y - 0.344136 * cb - 0.714136 * cr, y + 1.772 * cb].map(|v| v.round().clamp(0.0, 255.0) as u32)
            }
        };
        if let (Some((index, associated)), Some(opacity)) = (alpha, opacity.as_mut()) {
            let a = at(p, index);
            if associated && a > 0 && a < S::MAX { colour = colour.map(|v| ((v * S::MAX + a / 2) / a).min(S::MAX)); }
            opacity.push(((a * 255 + S::MAX / 2) / S::MAX) as u8);
        }
        rgb.extend(colour.map(S::make));
    }
    (rgb, opacity.filter(|values| values.iter().any(|a| *a != 255)))
}

/// Turns stored pixels as the Orientation tag says. `channels` is 3 for
/// colour and 1 for opacity. Returns the pixels with their width and height.
fn orient<T: Copy>(data: Vec<T>, width: usize, height: usize, channels: usize, orientation: u32) -> (Vec<T>, usize, usize) {
    if !(2..=8).contains(&orientation) { return (data, width, height); }
    let (w, h) = if orientation >= 5 { (height, width) } else { (width, height) };
    let mut out = Vec::with_capacity(data.len());
    for y in 0..h { for x in 0..w {
        let (sx, sy) = match orientation {
            2 => (width - 1 - x, y),
            3 => (width - 1 - x, height - 1 - y),
            4 => (x, height - 1 - y),
            5 => (y, x),
            6 => (y, height - 1 - x),
            7 => (width - 1 - y, height - 1 - x),
            _ => (width - 1 - y, x),
        };
        let source = (sy * width + sx) * channels;
        out.extend_from_slice(&data[source..source + channels]);
    } }
    (out, w, h)
}

/// Decodes the first image of a TIFF file. The file can use 8-bit or 16-bit
/// grey, RGB or YCbCr samples, with or without opacity, in strips, tiles or
/// planes. The Orientation tag is applied. A file with an RGB matrix ICC
/// profile is read through the profile, and any other file is read as sRGB.
pub fn from_tiff(bytes: &[u8]) -> Result<Frame, String> {
    let mut limits = Limits::default();
    // Sixteen-bit RGBA at the pixel limit.
    limits.decoding_buffer_size = MAX_PIXELS * 8;
    limits.intermediate_buffer_size = MAX_PIXELS * 8;
    limits.ifd_value_size = 32 * 1024 * 1024;
    let mut decoder = Decoder::new(Cursor::new(bytes)).map_err(message)?.with_limits(limits);
    let (width, height) = decoder.dimensions().map_err(message)?;
    let (width, height) = (width as usize, height as usize);
    let pixels = width.checked_mul(height).filter(|pixels| *pixels > 0 && *pixels <= MAX_PIXELS)
        .ok_or("This photograph exceeds the 80 megapixel browser limit.")?;
    let colour = decoder.colortype().map_err(message)?;
    let model = match colour {
        ColorType::Gray(_) | ColorType::GrayA(_) | ColorType::Multiband { .. } => Model::Grey,
        ColorType::RGB(_) | ColorType::RGBA(_) => Model::Rgb,
        ColorType::YCbCr(8) => Model::YCbCr,
        ColorType::CMYK(_) | ColorType::CMYKA(_) => return Err(colour_model("CMYK")),
        ColorType::Lab(_) => return Err(colour_model("Lab")),
        ColorType::Palette(_) => return Err(colour_model("indexed")),
        _ => return Err(message(TiffUnsupportedError::UnsupportedColorType(colour).into())),
    };
    let mut result = DecodingResult::U8(Vec::new());
    let buffer = decoder.read_image_to_buffer(&mut result).map_err(message)?;
    let samples = match &result { DecodingResult::U8(data) => data.len(), DecodingResult::U16(data) => data.len(),
        _ => return Err("This TIFF does not have 8-bit or 16-bit samples. Save it with 8 or 16 bits and try again.".into()) };
    let channels = samples / pixels;
    let base = if model == Model::Grey { 1 } else { 3 };
    if samples != channels * pixels || channels < base || (buffer.planes > 1 && buffer.planes != channels) || colour.bit_depth() % 8 != 0 {
        return Err(message(TiffUnsupportedError::UnsupportedColorType(colour).into()));
    }
    // The first extra sample is opacity when the file says so. A sample
    // without a stated meaning is a mask or a selection, and is not used.
    let alpha = match decoder.find_tag_unsigned_vec::<u16>(Tag::ExtraSamples).ok().flatten().and_then(|extra| extra.first().copied()) {
        Some(kind @ (1 | 2)) if channels > base => Some((base, kind == 1)),
        _ => None,
    };
    let orientation = decoder.find_tag_unsigned::<u32>(Tag::Orientation).ok().flatten().unwrap_or(1);
    let profile = decoder.get_tag_u8_vec(Tag::IccProfile).ok();
    let layout = Layout { model, channels, planar: buffer.planes > 1, alpha };
    let mut frame = match result {
        DecodingResult::U8(data) => {
            let (rgb, opacity) = unpack(data, pixels, &layout);
            let (rgb, w, h) = orient(rgb, width, height, 3, orientation);
            encoded(w, h, Samples::Srgb8(rgb), opacity.map(|values| orient(values, width, height, 1, orientation).0))
        }
        DecodingResult::U16(data) => {
            let (rgb, opacity) = unpack(data, pixels, &layout);
            let (rgb, w, h) = orient(rgb, width, height, 3, orientation);
            encoded(w, h, Samples::Srgb16(rgb.into_iter().map(Encoded16).collect()), opacity.map(|values| orient(values, width, height, 1, orientation).0))
        }
        _ => unreachable!(),
    };
    if let Some(icc) = profile { frame.set_profile(&icc); }
    Ok(frame)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::develop::{develop, Curve, Options, Pixels};
    use crate::recipe::Recipe;
    use tiff::encoder::{colortype, Compression, DeflateLevel, TiffEncoder};
    use tiff::tags::{ExtraSamples, Predictor};

    /// Samples that use the low bits of a 16-bit value, and opacity that is not all opaque.
    fn card16(width: usize, height: usize, channels: usize) -> Vec<u16> {
        (0..width * height * channels).map(|i| {
            let (p, c) = (i / channels, i % channels);
            ((p % width) * 4099 + (p / width) * 263 + c * 12345 + 7) as u16
        }).collect()
    }
    fn card8(width: usize, height: usize, channels: usize) -> Vec<u8> { card16(width, height, channels).iter().map(|v| (v >> 3) as u8).collect() }
    fn encode<C: colortype::ColorType>(compression: Compression, predictor: Predictor, width: usize, height: usize, data: &[C::Inner]) -> Vec<u8>
    where [C::Inner]: tiff::encoder::TiffValue {
        let mut file = Cursor::new(Vec::new());
        TiffEncoder::new(&mut file).unwrap().with_compression(compression).with_predictor(predictor)
            .write_image::<C>(width as u32, height as u32, data).unwrap();
        file.into_inner()
    }
    fn rgb8(frame: &Frame) -> &[u8] { match &frame.data { Samples::Srgb8(data) => data, _ => panic!("not an 8-bit frame") } }
    fn rgb16(frame: &Frame) -> Vec<u16> { match &frame.data { Samples::Srgb16(data) => data.iter().map(|v| v.0).collect(), _ => panic!("not a 16-bit frame") } }
    /// The colour samples of interleaved data, with grey repeated three times.
    fn colours<T: Copy>(data: &[T], channels: usize) -> Vec<T> {
        data.chunks(channels).flat_map(|pixel| if channels < 3 { [pixel[0]; 3] } else { [pixel[0], pixel[1], pixel[2]] }).collect()
    }
    const COMPRESSIONS: [(&str, Compression); 4] = [("uncompressed", Compression::Uncompressed), ("LZW", Compression::Lzw),
        ("Deflate", Compression::Deflate(DeflateLevel::Balanced)), ("PackBits", Compression::Packbits)];
    const SIZE: (usize, usize) = (37, 23);

    #[test]
    fn every_compression_decodes_grey_rgb_and_rgba_at_8_and_16_bits() {
        let (w, h) = SIZE;
        for (name, compression) in COMPRESSIONS {
            for predictor in [Predictor::None, Predictor::Horizontal] {
                let label = format!("{name}, {predictor:?}");
                for (channels, file) in [(1, encode::<colortype::Gray8>(compression, predictor, w, h, &card8(w, h, 1))),
                    (3, encode::<colortype::RGB8>(compression, predictor, w, h, &card8(w, h, 3))),
                    (4, encode::<colortype::RGBA8>(compression, predictor, w, h, &card8(w, h, 4)))] {
                    let frame = from_tiff(&file).unwrap_or_else(|error| panic!("{label}, {channels} channels: {error}"));
                    let source = card8(w, h, channels);
                    assert_eq!((frame.width, frame.height, frame.bit_depth()), (w, h, 8), "{label}");
                    assert_eq!(rgb8(&frame), colours(&source, channels), "{label}, {channels} channels");
                    // The encoder writes four samples with no ExtraSamples tag: a sample without a stated meaning.
                    assert!(frame.alpha.is_none(), "{label}");
                }
                for (channels, file) in [(1, encode::<colortype::Gray16>(compression, predictor, w, h, &card16(w, h, 1))),
                    (3, encode::<colortype::RGB16>(compression, predictor, w, h, &card16(w, h, 3))),
                    (4, encode::<colortype::RGBA16>(compression, predictor, w, h, &card16(w, h, 4)))] {
                    let frame = from_tiff(&file).unwrap_or_else(|error| panic!("{label}, {channels} channels: {error}"));
                    let source = card16(w, h, channels);
                    assert_eq!((frame.width, frame.height, frame.bit_depth()), (w, h, 16), "{label}");
                    assert_eq!(rgb16(&frame), colours(&source, channels), "{label}, {channels} channels");
                    assert!(frame.alpha.is_none(), "{label}");
                }
            }
        }
    }

    /// An image with tags the convenience encoder does not write.
    fn encode_with<C: colortype::ColorType>(width: usize, height: usize, data: &[C::Inner], extra: &[ExtraSamples], tags: &[(Tag, u16)], icc: Option<&[u8]>) -> Vec<u8>
    where [C::Inner]: tiff::encoder::TiffValue {
        let mut file = Cursor::new(Vec::new());
        let mut encoder = TiffEncoder::new(&mut file).unwrap().with_compression(Compression::Lzw);
        let mut image = encoder.new_image::<C>(width as u32, height as u32).unwrap();
        if !extra.is_empty() { image.extra_samples(extra).unwrap(); }
        for (tag, value) in tags { image.encoder().write_tag(*tag, *value).unwrap(); }
        if let Some(icc) = icc { image.encoder().write_tag(Tag::IccProfile, icc).unwrap(); }
        image.write_data(data).unwrap();
        file.into_inner()
    }

    #[test]
    fn opacity_is_read_when_the_file_states_it() {
        let (w, h) = SIZE;
        // Unassociated: the colours are stored as they are.
        let source = card16(w, h, 4);
        let frame = from_tiff(&encode_with::<colortype::RGB16>(w, h, &source, &[ExtraSamples::UnassociatedAlpha], &[], None)).unwrap();
        assert_eq!(rgb16(&frame), colours(&source, 4));
        let alpha = frame.alpha.as_ref().expect("opacity");
        for (p, a) in alpha.iter().enumerate() { assert_eq!(*a as u32, (source[p * 4 + 3] as u32 * 255 + 32767) / 65535); }
        // Associated: the stored colours are multiplied by the opacity, and the frame holds them divided again.
        let straight = [200u8, 100, 40];
        let stored: Vec<u8> = (0..w * h).flat_map(|p| { let a = (p * 255 / (w * h - 1)) as u32; [straight[0] as u32 * a / 255, straight[1] as u32 * a / 255, straight[2] as u32 * a / 255, a].map(|v| v as u8) }).collect();
        let frame = from_tiff(&encode_with::<colortype::RGB8>(w, h, &stored, &[ExtraSamples::AssociatedAlpha], &[], None)).unwrap();
        let (rgb, alpha) = (rgb8(&frame), frame.alpha.as_ref().expect("opacity"));
        for p in 0..w * h {
            assert_eq!(alpha[p], stored[p * 4 + 3]);
            // Division by a small opacity magnifies the rounding of the stored value.
            if alpha[p] >= 128 { for c in 0..3 { assert!((rgb[p * 3 + c] as i32 - straight[c] as i32).abs() <= 2, "pixel {p}: {:?}", &rgb[p * 3..p * 3 + 3]); } }
        }
        // A grey image with opacity.
        let source = card8(w, h, 2);
        let frame = from_tiff(&encode_with::<colortype::Gray8>(w, h, &source, &[ExtraSamples::UnassociatedAlpha], &[], None)).unwrap();
        assert_eq!(rgb8(&frame), colours(&source, 2));
        assert_eq!(frame.alpha.as_deref().unwrap(), source.chunks(2).map(|pixel| pixel[1]).collect::<Vec<u8>>());
        // A sample without a stated meaning is not opacity.
        let frame = from_tiff(&encode_with::<colortype::RGB8>(w, h, &card8(w, h, 4), &[ExtraSamples::Unspecified], &[], None)).unwrap();
        assert!(frame.alpha.is_none());
        // Opacity that is opaque in every pixel is not kept.
        let opaque: Vec<u8> = card8(w, h, 4).chunks(4).flat_map(|pixel| [pixel[0], pixel[1], pixel[2], 255]).collect();
        assert!(from_tiff(&encode_with::<colortype::RGB8>(w, h, &opaque, &[ExtraSamples::UnassociatedAlpha], &[], None)).unwrap().alpha.is_none());
    }

    #[test]
    fn the_orientation_tag_is_applied() {
        // 3 by 2, each pixel with its own red value: 0 1 2 / 3 4 5.
        let source: Vec<u8> = (0..6u8).flat_map(|i| [i, 100, 200]).collect();
        let expected: [(u16, (usize, usize), [u8; 6]); 8] = [
            (1, (3, 2), [0, 1, 2, 3, 4, 5]), (2, (3, 2), [2, 1, 0, 5, 4, 3]), (3, (3, 2), [5, 4, 3, 2, 1, 0]), (4, (3, 2), [3, 4, 5, 0, 1, 2]),
            (5, (2, 3), [0, 3, 1, 4, 2, 5]), (6, (2, 3), [3, 0, 4, 1, 5, 2]), (7, (2, 3), [5, 2, 4, 1, 3, 0]), (8, (2, 3), [2, 5, 1, 4, 0, 3]),
        ];
        for (orientation, size, reds) in expected {
            let frame = from_tiff(&encode_with::<colortype::RGB8>(3, 2, &source, &[], &[(Tag::Orientation, orientation)], None)).unwrap();
            assert_eq!((frame.width, frame.height), size, "orientation {orientation}");
            assert_eq!(rgb8(&frame).chunks(3).map(|pixel| pixel[0]).collect::<Vec<u8>>(), reds, "orientation {orientation}");
        }
        // Opacity turns with the colours.
        let source: Vec<u8> = (0..6u8).flat_map(|i| [i, 100, 200, 10 * i]).collect();
        let frame = from_tiff(&encode_with::<colortype::RGB8>(3, 2, &source, &[ExtraSamples::UnassociatedAlpha], &[(Tag::Orientation, 6)], None)).unwrap();
        assert_eq!(frame.alpha.unwrap(), [30, 0, 40, 10, 50, 20]);
    }

    #[test]
    fn an_icc_profile_sets_the_colours_and_the_curve() {
        let (w, h) = SIZE;
        let adobe = crate::icc::tests::adobe_rgb();
        let frame = from_tiff(&encode_with::<colortype::RGB16>(w, h, &card16(w, h, 3), &[], &[], Some(&adobe))).unwrap();
        let Some(Curve::Sixteen(curve)) = &frame.curve else { panic!("no 16-bit curve") };
        assert!((curve[32768] - (32768.0f32 / 65535.0).powf(2.19921875)).abs() < 1e-5);
        assert!(frame.matrix[1] < -0.3);
        let frame = from_tiff(&encode_with::<colortype::RGB8>(w, h, &card8(w, h, 3), &[], &[], Some(&crate::icc::tests::display_p3()))).unwrap();
        assert!(matches!(frame.curve, Some(Curve::Eight(_))) && frame.matrix[0] > 1.2);
        // A file without a profile, and a file with a profile that is not read, are sRGB.
        for icc in [None, Some(&b"not a profile"[..])] {
            let frame = from_tiff(&encode_with::<colortype::RGB8>(w, h, &card8(w, h, 3), &[], &[], icc)).unwrap();
            assert!(frame.curve.is_none() && frame.matrix == crate::colour::IDENTITY);
        }
        // Saturated Adobe RGB green is more saturated than the same values read as sRGB.
        let green: Vec<u16> = (0..16).flat_map(|_| [20000u16, 50000, 20000]).collect();
        let plain = develop(&from_tiff(&encode_with::<colortype::RGB16>(4, 4, &green, &[], &[], None)).unwrap(), &Recipe::default(), &Options::default());
        let managed = develop(&from_tiff(&encode_with::<colortype::RGB16>(4, 4, &green, &[], &[], Some(&adobe))).unwrap(), &Recipe::default(), &Options::default());
        let (Pixels::U8(plain), Pixels::U8(managed)) = (&plain.data, &managed.data) else { panic!() };
        assert!(managed[1] as i32 - managed[0] as i32 > plain[1] as i32 - plain[0] as i32 + 10, "{:?} {:?}", &plain[..3], &managed[..3]);
    }

    #[test]
    fn a_16_bit_file_keeps_its_16_bits_through_development() {
        // A ramp that 8 bits cannot hold: 1024 steps inside 16 levels of 8-bit.
        let (w, h) = (1024usize, 2usize);
        let source: Vec<u16> = (0..w * h).flat_map(|p| [(30000 + (p % w) * 4) as u16; 3]).collect();
        let file = encode::<colortype::RGB16>(Compression::Lzw, Predictor::Horizontal, w, h, &source);
        let frame = from_tiff(&file).unwrap();
        assert_eq!(rgb16(&frame), source);
        let deep = develop(&frame, &Recipe::default(), &Options { bit_depth: 12, ..Options::default() });
        let Pixels::U16(deep) = &deep.data else { panic!() };
        let levels: std::collections::BTreeSet<u16> = deep.chunks(4).map(|pixel| pixel[0]).collect();
        // 4096 source steps cover 256 of the 12-bit levels. An 8-bit frame gives 16.
        assert!(levels.len() > 200, "{} levels", levels.len());
        // The default recipe returns the picture: 16-bit to 12-bit.
        for (p, pixel) in deep.chunks(4).enumerate() { assert!((pixel[0] as i32 - (source[p * 3] as f32 * 4095.0 / 65535.0).round() as i32).abs() <= 1); }
        // The preview of a 16-bit frame is the same picture.
        let preview = develop(&frame.linear_preview(256), &Recipe::default(), &Options::default());
        let whole = develop(&frame, &Recipe::default(), &Options { edge: 256, ..Options::default() });
        let (Pixels::U8(preview), Pixels::U8(whole)) = (&preview.data, &whole.data) else { panic!() };
        for (a, b) in preview.iter().zip(whole) { assert!((*a as i32 - *b as i32).abs() <= 1); }
    }

    /// A TIFF written tag by tag, for the layouts the encoder does not write.
    fn raw(big: bool, mut entries: Vec<(u16, Vec<u32>)>, chunks: &[Vec<u8>], tiled: bool) -> Vec<u8> {
        let w16 = |v: u16| if big { v.to_be_bytes() } else { v.to_le_bytes() };
        let w32 = |v: u32| if big { v.to_be_bytes() } else { v.to_le_bytes() };
        let mut out = if big { b"MM\0\x2a".to_vec() } else { b"II\x2a\0".to_vec() };
        out.extend([0; 4]);
        let mut offsets = Vec::new();
        for chunk in chunks { offsets.push(out.len() as u32); out.extend(chunk); if out.len() % 2 == 1 { out.push(0); } }
        entries.push((if tiled { 324 } else { 273 }, offsets));
        entries.push((if tiled { 325 } else { 279 }, chunks.iter().map(|chunk| chunk.len() as u32).collect()));
        entries.sort_by_key(|entry| entry.0);
        let mut fields = Vec::new();
        for (id, values) in &entries {
            let long = matches!(id, 256 | 257 | 273 | 279 | 324 | 325) || values.iter().any(|v| *v > 65535);
            let mut bytes: Vec<u8> = values.iter().flat_map(|v| if long { w32(*v).to_vec() } else { w16(*v as u16).to_vec() }).collect();
            if bytes.len() <= 4 { bytes.resize(4, 0); } else {
                let at = out.len() as u32;
                out.extend(&bytes); if out.len() % 2 == 1 { out.push(0); }
                bytes = w32(at).to_vec();
            }
            fields.push((*id, if long { 4u16 } else { 3 }, values.len() as u32, bytes));
        }
        let directory = out.len() as u32;
        out[4..8].copy_from_slice(&w32(directory));
        out.extend(w16(fields.len() as u16));
        for (id, kind, count, value) in fields { out.extend(w16(id)); out.extend(w16(kind)); out.extend(w32(count)); out.extend(value); }
        out.extend([0; 4]);
        out
    }
    /// The tags of an uncompressed image in one strip.
    fn tags(width: usize, height: usize, bits: u32, samples: u32, photometric: u32) -> Vec<(u16, Vec<u32>)> {
        vec![(256, vec![width as u32]), (257, vec![height as u32]), (258, vec![bits; samples as usize]), (259, vec![1]), (262, vec![photometric]),
            (277, vec![samples]), (278, vec![height as u32])]
    }

    #[test]
    fn planes_tiles_and_big_endian_files_decode() {
        let (w, h) = SIZE;
        // Planar: one strip for each of red, green and blue.
        let source = card8(w, h, 3);
        let planes: Vec<Vec<u8>> = (0..3).map(|c| source.chunks(3).map(|pixel| pixel[c]).collect()).collect();
        let mut entries = tags(w, h, 8, 3, 2); entries.push((284, vec![2]));
        assert_eq!(rgb8(&from_tiff(&raw(false, entries, &planes, false)).unwrap()), source);
        // Tiles of 16 by 16 that overhang the right and bottom edges.
        let (across, down) = (w.div_ceil(16), h.div_ceil(16));
        let tiles: Vec<Vec<u8>> = (0..across * down).map(|t| {
            let (tx, ty) = (t % across * 16, t / across * 16);
            (0..16 * 16).flat_map(|i| { let (x, y) = (tx + i % 16, ty + i / 16); if x < w && y < h { source[(y * w + x) * 3..(y * w + x) * 3 + 3].to_vec() } else { vec![0; 3] } }).collect()
        }).collect();
        let mut entries: Vec<(u16, Vec<u32>)> = tags(w, h, 8, 3, 2).into_iter().filter(|entry| entry.0 != 278).collect();
        entries.extend([(322, vec![16]), (323, vec![16])]);
        assert_eq!(rgb8(&from_tiff(&raw(false, entries, &tiles, true)).unwrap()), source);
        // Big-endian 16-bit samples, as an editor on a Mac can write them.
        let source = card16(w, h, 3);
        let bytes: Vec<u8> = source.iter().flat_map(|v| v.to_be_bytes()).collect();
        assert_eq!(rgb16(&from_tiff(&raw(true, tags(w, h, 16, 3, 2), &[bytes], false)).unwrap()), source);
        // White is zero: the samples are inverted.
        let source = card8(w, h, 1);
        let frame = from_tiff(&raw(false, tags(w, h, 8, 1, 0), &[source.clone()], false)).unwrap();
        assert_eq!(rgb8(&frame), colours(&source.iter().map(|v| 255 - v).collect::<Vec<u8>>(), 1));
    }

    #[test]
    fn the_first_page_of_a_file_with_more_pages_is_decoded() {
        let (w, h) = SIZE;
        let mut file = Cursor::new(Vec::new());
        let mut encoder = TiffEncoder::new(&mut file).unwrap();
        encoder.write_image::<colortype::RGB8>(w as u32, h as u32, &card8(w, h, 3)).unwrap();
        encoder.write_image::<colortype::Gray8>(5, 4, &[9; 20]).unwrap();
        let frame = from_tiff(&file.into_inner()).unwrap();
        assert_eq!((frame.width, frame.height), (w, h));
        assert_eq!(rgb8(&frame), card8(w, h, 3));
    }

    #[test]
    fn jpeg_compressed_files_decode_as_rgb_and_as_ycbcr() {
        // Each reference is the same file as libtiff decodes it.
        for (name, file, reference) in [("RGB", &include_bytes!("../test/fixtures/jpeg-rgb.tif")[..], &include_bytes!("../test/fixtures/jpeg-rgb.rgb")[..]),
            ("YCbCr", &include_bytes!("../test/fixtures/jpeg-ycbcr.tif")[..], &include_bytes!("../test/fixtures/jpeg-ycbcr.rgb")[..])] {
            let frame = from_tiff(file).unwrap_or_else(|error| panic!("{name}: {error}"));
            assert_eq!((frame.width, frame.height), (32, 24), "{name}");
            let rgb = rgb8(&frame);
            assert_eq!(rgb.len(), reference.len());
            let difference: Vec<i32> = rgb.iter().zip(reference).map(|(a, b)| (*a as i32 - *b as i32).abs()).collect();
            let (worst, mean) = (*difference.iter().max().unwrap(), difference.iter().sum::<i32>() as f32 / difference.len() as f32);
            // The two JPEG decoders round the inverse transform and the chroma differently.
            assert!(worst <= 4 && mean < 1.0, "{name}: worst {worst}, mean {mean}");
        }
    }

    #[test]
    fn files_that_are_not_supported_give_a_clear_error() {
        let (w, h) = SIZE;
        let error = |file: &[u8]| from_tiff(file).err().expect("an error");
        let cmyk = encode::<colortype::CMYK8>(Compression::Uncompressed, Predictor::None, w, h, &card8(w, h, 4));
        assert_eq!(error(&cmyk), "This TIFF uses CMYK colour. Save it as RGB and try again.");
        let float = encode::<colortype::RGB32Float>(Compression::Uncompressed, Predictor::None, 2, 2, &[0.5; 12]);
        assert_eq!(error(&float), "This TIFF does not have 8-bit or 16-bit samples. Save it with 8 or 16 bits and try again.");
        let mut palette = tags(w, h, 8, 1, 3); palette.push((320, vec![0; 768]));
        assert!(error(&raw(false, palette, &[card8(w, h, 1)], false)).contains("not supported"));
        let bilevel = raw(false, tags(16, 4, 1, 1, 1), &[vec![0xAA; 8]], false);
        assert!(error(&bilevel).contains("8-bit or 16-bit") || error(&bilevel).contains("not supported"), "{}", error(&bilevel));
        // Old-style JPEG, compression 6.
        let mut old = tags(w, h, 8, 3, 2); old.iter_mut().find(|entry| entry.0 == 259).unwrap().1 = vec![6];
        assert_eq!(error(&raw(false, old, &[card8(w, h, 3)], false)), "This TIFF uses a compression that is not supported. Save it with LZW or ZIP compression and try again.");
        assert_eq!(error(b"not a TIFF"), "This TIFF could not be read.");
        let whole = encode::<colortype::RGB8>(Compression::Lzw, Predictor::None, w, h, &card8(w, h, 3));
        assert_eq!(error(&whole[..whole.len() / 2]), "This TIFF could not be read.");
        // 10000 by 10000 is more than the pixel limit. The file needs no pixel data for this check.
        let large = raw(false, tags(10000, 10000, 8, 3, 2), &[vec![0; 16]], false);
        assert_eq!(error(&large), "This photograph exceeds the 80 megapixel browser limit.");
    }
}
