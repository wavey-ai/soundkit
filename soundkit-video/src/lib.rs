//! Pure-Rust video decoding for the shared SoundKit media pipeline.
//!
//! Container demuxers feed complete codec access units to [`VideoDecoder`].
//! Decoded frames use one bounded, platform-neutral planar surface so native
//! and WASM callers observe the same pixels and metadata.

const MAX_FRAME_PIXELS: usize = 7680 * 4320;

/// Decoder-ready parameter sets and the source container's NAL length width.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NalDecoderConfiguration {
    pub length_size: u8,
    pub annex_b: Vec<u8>,
}

/// Parse ISO/IEC 14496-15 AVCDecoderConfigurationRecord (`avcC`) bytes.
pub fn parse_avc_decoder_configuration(data: &[u8]) -> Result<NalDecoderConfiguration, String> {
    if data.len() < 7 || data[0] != 1 {
        return Err("invalid avcC decoder configuration".to_string());
    }
    let length_size = (data[4] & 0x03) + 1;
    let mut pos = 6usize;
    let mut annex_b = Vec::new();
    let sps_count = data[5] & 0x1f;
    for _ in 0..sps_count {
        append_configuration_nal(data, &mut pos, &mut annex_b, "avcC")?;
    }
    let pps_count = *data
        .get(pos)
        .ok_or_else(|| "truncated avcC PPS count".to_string())?;
    pos += 1;
    for _ in 0..pps_count {
        append_configuration_nal(data, &mut pos, &mut annex_b, "avcC")?;
    }
    Ok(NalDecoderConfiguration {
        length_size,
        annex_b,
    })
}

/// Parse ISO/IEC 14496-15 HEVCDecoderConfigurationRecord (`hvcC`) bytes.
pub fn parse_hevc_decoder_configuration(data: &[u8]) -> Result<NalDecoderConfiguration, String> {
    if data.len() < 23 || data[0] != 1 {
        return Err("invalid hvcC decoder configuration".to_string());
    }
    let length_size = (data[21] & 0x03) + 1;
    let array_count = data[22] as usize;
    let mut pos = 23usize;
    let mut annex_b = Vec::new();
    for _ in 0..array_count {
        pos = pos
            .checked_add(1)
            .ok_or_else(|| "hvcC array offset overflow".to_string())?;
        let nal_count = read_be_u16(data, pos).ok_or_else(|| "truncated hvcC array".to_string())?;
        pos += 2;
        for _ in 0..nal_count {
            append_configuration_nal(data, &mut pos, &mut annex_b, "hvcC")?;
        }
    }
    Ok(NalDecoderConfiguration {
        length_size,
        annex_b,
    })
}

/// Convert one container sample containing length-prefixed NAL units to Annex B.
pub fn length_prefixed_nals_to_annex_b(data: &[u8], length_size: u8) -> Result<Vec<u8>, String> {
    if !(1..=4).contains(&length_size) {
        return Err(format!("invalid NAL length size {length_size}"));
    }
    let mut output = Vec::with_capacity(data.len().saturating_add(16));
    let mut pos = 0usize;
    while pos < data.len() {
        let header_end = pos
            .checked_add(length_size as usize)
            .ok_or_else(|| "NAL header offset overflow".to_string())?;
        let header = data
            .get(pos..header_end)
            .ok_or_else(|| "truncated NAL length".to_string())?;
        let size = header.iter().try_fold(0usize, |size, byte| {
            size.checked_mul(256)
                .and_then(|value| value.checked_add(*byte as usize))
                .ok_or_else(|| "NAL size overflow".to_string())
        })?;
        if size == 0 {
            return Err("container sample contains an empty NAL unit".to_string());
        }
        let nal_end = header_end
            .checked_add(size)
            .ok_or_else(|| "NAL byte range overflow".to_string())?;
        let nal = data
            .get(header_end..nal_end)
            .ok_or_else(|| "NAL extends past its container sample".to_string())?;
        output.extend_from_slice(&[0, 0, 0, 1]);
        output.extend_from_slice(nal);
        pos = nal_end;
    }
    Ok(output)
}

fn append_configuration_nal(
    data: &[u8],
    pos: &mut usize,
    output: &mut Vec<u8>,
    format: &str,
) -> Result<(), String> {
    let size = read_be_u16(data, *pos)
        .ok_or_else(|| format!("truncated {format} codec configuration"))? as usize;
    *pos += 2;
    let end = pos
        .checked_add(size)
        .ok_or_else(|| format!("{format} NAL size overflow"))?;
    let nal = data
        .get(*pos..end)
        .ok_or_else(|| format!("truncated {format} configuration NAL"))?;
    output.extend_from_slice(&[0, 0, 0, 1]);
    output.extend_from_slice(nal);
    *pos = end;
    Ok(())
}

fn read_be_u16(data: &[u8], pos: usize) -> Option<u16> {
    Some(u16::from_be_bytes([*data.get(pos)?, *data.get(pos + 1)?]))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VideoCodec {
    H264,
    Hevc,
    Vp9,
    Av1,
    ProRes,
    DnxHd,
}

impl VideoCodec {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::H264 => "h264",
            Self::Hevc => "hevc",
            Self::Vp9 => "vp9",
            Self::Av1 => "av1",
            Self::ProRes => "prores",
            Self::DnxHd => "dnxhd",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "h264" | "avc" | "avc1" | "avc3" => Some(Self::H264),
            "hevc" | "h265" | "hev1" | "hvc1" => Some(Self::Hevc),
            "vp9" | "vp09" | "v_vp9" => Some(Self::Vp9),
            "av1" | "av01" | "v_av1" => Some(Self::Av1),
            "prores" | "apco" | "apcs" | "apcn" | "apch" | "ap4h" | "ap4x" => Some(Self::ProRes),
            "dnxhd" | "dnxhr" | "avdn" => Some(Self::DnxHd),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChromaSampling {
    Monochrome,
    Cs420,
    Cs422,
    Cs444,
}

impl ChromaSampling {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Monochrome => "400",
            Self::Cs420 => "420",
            Self::Cs422 => "422",
            Self::Cs444 => "444",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VideoPlane {
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    /// Little-endian samples. Eight-bit surfaces use one byte per sample;
    /// higher bit depths use two bytes per sample.
    pub data: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VideoFrame {
    pub width: u32,
    pub height: u32,
    pub bit_depth: u8,
    pub chroma_sampling: ChromaSampling,
    /// True when plane 3 contains full-resolution alpha. Color planes are
    /// always ordered Y, Cb, Cr; an alpha plane, when present, follows them.
    pub has_alpha: bool,
    pub pts: Option<i64>,
    pub duration: Option<u64>,
    pub planes: Vec<VideoPlane>,
}

impl VideoFrame {
    pub fn validate(&self) -> Result<(), String> {
        let pixels = (self.width as usize)
            .checked_mul(self.height as usize)
            .ok_or_else(|| "decoded video dimensions overflow".to_string())?;
        if self.width == 0 || self.height == 0 || pixels > MAX_FRAME_PIXELS {
            return Err(format!(
                "decoded video frame {}x{} exceeds the SoundKit pixel budget",
                self.width, self.height
            ));
        }
        let bytes_per_sample = if self.bit_depth <= 8 { 1 } else { 2 };
        let color_planes = if self.chroma_sampling == ChromaSampling::Monochrome {
            1
        } else {
            3
        };
        let expected_planes = color_planes + usize::from(self.has_alpha);
        if self.planes.len() != expected_planes {
            return Err(format!(
                "decoded video frame expected {expected_planes} planes, got {}",
                self.planes.len()
            ));
        }
        let expected_plane_dimensions = |index: usize| match (self.chroma_sampling, index) {
            (_, 0) => (self.width, self.height),
            (_, index) if self.has_alpha && index + 1 == expected_planes => {
                (self.width, self.height)
            }
            (ChromaSampling::Cs420, _) => (self.width.div_ceil(2), self.height.div_ceil(2)),
            (ChromaSampling::Cs422, _) => (self.width.div_ceil(2), self.height),
            (ChromaSampling::Cs444, _) => (self.width, self.height),
            (ChromaSampling::Monochrome, _) => (0, 0),
        };
        for (index, plane) in self.planes.iter().enumerate() {
            if plane.width == 0 || plane.height == 0 || plane.stride < plane.width {
                return Err("decoded video plane has invalid dimensions".to_string());
            }
            let expected_dimensions = expected_plane_dimensions(index);
            if (plane.width, plane.height) != expected_dimensions {
                return Err(format!(
                    "decoded video plane {index} expected {}x{}, got {}x{}",
                    expected_dimensions.0, expected_dimensions.1, plane.width, plane.height
                ));
            }
            let required = (plane.stride as usize)
                .checked_mul(plane.height as usize)
                .and_then(|samples| samples.checked_mul(bytes_per_sample))
                .ok_or_else(|| "decoded video plane size overflow".to_string())?;
            if plane.data.len() != required {
                return Err(format!(
                    "decoded video plane expected {required} bytes, got {}",
                    plane.data.len()
                ));
            }
        }
        Ok(())
    }
}

enum DecoderState {
    #[cfg(feature = "h264")]
    H264(rusty_h264_decoder::Decoder),
    #[cfg(feature = "hevc")]
    Hevc(rust_h265::Decoder),
    #[cfg(feature = "vp9")]
    Vp9(vp9dec::Decoder),
    #[cfg(feature = "av1")]
    Av1(rusty_av1d::Decoder),
    #[cfg(feature = "prores")]
    ProRes,
}

pub struct VideoDecoder {
    codec: VideoCodec,
    state: DecoderState,
}

impl VideoDecoder {
    pub fn new(codec: VideoCodec) -> Result<Self, String> {
        let state = match codec {
            #[cfg(feature = "h264")]
            VideoCodec::H264 => DecoderState::H264(rusty_h264_decoder::Decoder::new()),
            #[cfg(feature = "hevc")]
            VideoCodec::Hevc => DecoderState::Hevc(rust_h265::Decoder::new()),
            #[cfg(feature = "vp9")]
            VideoCodec::Vp9 => DecoderState::Vp9(vp9dec::Decoder::new()),
            #[cfg(feature = "av1")]
            VideoCodec::Av1 => DecoderState::Av1(av1_decoder()?),
            #[cfg(feature = "prores")]
            VideoCodec::ProRes => DecoderState::ProRes,
            VideoCodec::DnxHd => {
                return Err(
                    "DNxHD/DNxHR native decoding is not yet available in SoundKit".to_string(),
                );
            }
            #[allow(unreachable_patterns)]
            _ => {
                return Err(format!(
                    "{} decoder is disabled in this build",
                    codec.as_str()
                ))
            }
        };
        Ok(Self { codec, state })
    }

    pub fn codec(&self) -> VideoCodec {
        self.codec
    }

    /// Decode one complete codec access unit in decode order.
    pub fn decode(
        &mut self,
        access_unit: &[u8],
        pts: Option<i64>,
        duration: Option<u64>,
    ) -> Result<Vec<VideoFrame>, String> {
        if access_unit.is_empty() {
            return Ok(Vec::new());
        }
        let mut frames = match &mut self.state {
            #[cfg(feature = "h264")]
            DecoderState::H264(decoder) => decoder
                .decode(access_unit)
                .map_err(|error| format!("H.264 decode failed: {error}"))?
                .map(|frame| vec![h264_frame(frame, pts, duration)])
                .unwrap_or_default(),
            #[cfg(feature = "hevc")]
            DecoderState::Hevc(decoder) => {
                let mut output = Vec::new();
                for nal in rust_h265::parse_annex_b(access_unit) {
                    if let Some(frame) = decoder
                        .decode_nal(&nal)
                        .map_err(|error| format!("HEVC decode failed: {error}"))?
                    {
                        output.push(hevc_frame(frame, pts, duration));
                    }
                }
                output
            }
            #[cfg(feature = "vp9")]
            DecoderState::Vp9(decoder) => decoder
                .decode_frame(access_unit)
                .map_err(|error| format!("VP9 decode failed: {error:?}"))?
                .into_iter()
                .filter_map(|decoded| decoded.frame)
                .map(|frame| vp9_frame(frame, pts, duration))
                .collect(),
            #[cfg(feature = "av1")]
            DecoderState::Av1(decoder) => decode_av1(decoder, access_unit, pts, duration)?,
            #[cfg(feature = "prores")]
            DecoderState::ProRes => {
                let output = prores_output_format(access_unit)?;
                let frame = oxideav_prores::decoder::decode_packet_with_format(
                    access_unit,
                    pts,
                    Some(output.pixel_format),
                    oxideav_prores::decoder::OutputRange::Full,
                )
                .map_err(|error| format!("ProRes decode failed: {error}"))?;
                vec![prores_frame(frame, output.bit_depth, pts, duration)?]
            }
        };
        for frame in &frames {
            frame.validate()?;
        }
        Ok(std::mem::take(&mut frames))
    }

    /// Decode a complete elementary stream when the codec defines stream
    /// framing. Container callers should normally pass one access unit to
    /// [`Self::decode`] so frames can be released incrementally.
    pub fn decode_stream(&mut self, stream: &[u8]) -> Result<Vec<VideoFrame>, String> {
        let frames: Vec<VideoFrame> = match &mut self.state {
            #[cfg(feature = "h264")]
            DecoderState::H264(decoder) => decoder
                .decode_stream(stream)
                .map_err(|error| format!("H.264 stream decode failed: {error}"))?
                .into_iter()
                .map(|frame| h264_frame(frame, None, None))
                .collect(),
            _ => return self.decode(stream, None, None),
        };
        for frame in &frames {
            frame.validate()?;
        }
        Ok(frames)
    }

    pub fn flush(&mut self) -> Result<Vec<VideoFrame>, String> {
        let frames = match &mut self.state {
            #[cfg(feature = "hevc")]
            DecoderState::Hevc(decoder) => decoder
                .flush()
                .map(|frame| vec![hevc_frame(frame, None, None)])
                .unwrap_or_default(),
            #[cfg(feature = "av1")]
            DecoderState::Av1(decoder) => {
                let frames = drain_av1(decoder)?;
                decoder.flush();
                frames
            }
            _ => Vec::new(),
        };
        for frame in &frames {
            frame.validate()?;
        }
        Ok(frames)
    }
}

#[cfg(feature = "av1")]
fn av1_decoder() -> Result<rusty_av1d::Decoder, String> {
    let mut settings = rusty_av1d::Settings::new();
    settings.set_n_threads(1);
    settings.set_max_frame_delay(1);
    settings.set_frame_size_limit(MAX_FRAME_PIXELS as u32);
    settings.set_strict_std_compliance(true);
    rusty_av1d::Decoder::with_settings(&settings)
        .map_err(|error| format!("could not initialize AV1 decoder: {error:?}"))
}

#[cfg(feature = "av1")]
fn decode_av1(
    decoder: &mut rusty_av1d::Decoder,
    access_unit: &[u8],
    pts: Option<i64>,
    duration: Option<u64>,
) -> Result<Vec<VideoFrame>, String> {
    use rusty_av1d::Rav1dError;

    let duration = duration.and_then(|value| i64::try_from(value).ok());
    let mut output = Vec::new();
    match decoder.send_data(access_unit.into(), None, pts, duration) {
        Ok(()) => {}
        Err(Rav1dError::TryAgain) => loop {
            let pending = drain_av1(decoder)?;
            if pending.is_empty() {
                return Err("AV1 decoder requested output but produced no frame".to_string());
            }
            output.extend(pending);
            match decoder.send_pending_data() {
                Ok(()) => break,
                Err(Rav1dError::TryAgain) => continue,
                Err(error) => return Err(format!("AV1 decode failed: {error:?}")),
            }
        },
        Err(error) => return Err(format!("AV1 decode failed: {error:?}")),
    }
    output.extend(drain_av1(decoder)?);
    Ok(output)
}

fn plane_u8(width: u32, height: u32, data: Vec<u8>) -> VideoPlane {
    VideoPlane {
        width,
        height,
        stride: width,
        data,
    }
}

fn plane_u16(width: u32, height: u32, data: Vec<u16>) -> VideoPlane {
    let mut bytes = Vec::with_capacity(data.len() * 2);
    for sample in data {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    VideoPlane {
        width,
        height,
        stride: width,
        data: bytes,
    }
}

#[cfg(feature = "h264")]
fn h264_frame(
    frame: rusty_h264_common::YuvFrame,
    pts: Option<i64>,
    duration: Option<u64>,
) -> VideoFrame {
    let width = frame.width as u32;
    let height = frame.height as u32;
    VideoFrame {
        width,
        height,
        bit_depth: 8,
        chroma_sampling: ChromaSampling::Cs420,
        has_alpha: false,
        pts,
        duration,
        planes: vec![
            plane_u8(width, height, frame.y),
            plane_u8(width.div_ceil(2), height.div_ceil(2), frame.u),
            plane_u8(width.div_ceil(2), height.div_ceil(2), frame.v),
        ],
    }
}

#[cfg(feature = "hevc")]
fn hevc_frame(frame: rust_h265::Frame, pts: Option<i64>, duration: Option<u64>) -> VideoFrame {
    let width = frame.width;
    let height = frame.height;
    let make_plane = |data: rust_h265::PixelData, width, height| match data {
        rust_h265::PixelData::U8(values) => plane_u8(width, height, values),
        rust_h265::PixelData::U16(values) => plane_u16(width, height, values),
    };
    VideoFrame {
        width,
        height,
        bit_depth: frame.bit_depth,
        chroma_sampling: ChromaSampling::Cs420,
        has_alpha: false,
        pts,
        duration,
        planes: vec![
            make_plane(frame.y, width, height),
            make_plane(frame.u, width.div_ceil(2), height.div_ceil(2)),
            make_plane(frame.v, width.div_ceil(2), height.div_ceil(2)),
        ],
    }
}

#[cfg(feature = "vp9")]
fn vp9_frame(frame: vp9dec::Frame, pts: Option<i64>, duration: Option<u64>) -> VideoFrame {
    let width = frame.width;
    let height = frame.height;
    let cw = (width + frame.subsampling_x) >> frame.subsampling_x;
    let ch = (height + frame.subsampling_y) >> frame.subsampling_y;
    let make_plane = |data: vp9dec::PlaneData, width, height| match data {
        vp9dec::PlaneData::U8(values) => plane_u8(width, height, values),
        vp9dec::PlaneData::U16(values) => plane_u16(width, height, values),
    };
    let chroma_sampling = match (frame.subsampling_x, frame.subsampling_y) {
        (1, 1) => ChromaSampling::Cs420,
        (1, 0) => ChromaSampling::Cs422,
        (0, 0) => ChromaSampling::Cs444,
        _ => ChromaSampling::Monochrome,
    };
    VideoFrame {
        width,
        height,
        bit_depth: frame.bit_depth,
        chroma_sampling,
        has_alpha: false,
        pts,
        duration,
        planes: vec![
            make_plane(frame.y, width, height),
            make_plane(frame.u, cw, ch),
            make_plane(frame.v, cw, ch),
        ],
    }
}

#[cfg(feature = "av1")]
fn drain_av1(decoder: &mut rusty_av1d::Decoder) -> Result<Vec<VideoFrame>, String> {
    use rusty_av1d::{PixelLayout, PlanarImageComponent, Rav1dError};

    let mut output = Vec::new();
    loop {
        let picture = match decoder.get_picture() {
            Ok(picture) => picture,
            Err(Rav1dError::TryAgain) => break,
            Err(error) => return Err(format!("AV1 output failed: {error:?}")),
        };
        let width = picture.width();
        let height = picture.height();
        let bit_depth = picture.bits_per_component().map(|bits| bits.0).unwrap_or(8);
        let (sampling, chroma_width, chroma_height) = match picture.pixel_layout() {
            PixelLayout::I400 => (ChromaSampling::Monochrome, 0, 0),
            PixelLayout::I420 => (ChromaSampling::Cs420, width.div_ceil(2), height.div_ceil(2)),
            PixelLayout::I422 => (ChromaSampling::Cs422, width.div_ceil(2), height),
            PixelLayout::I444 => (ChromaSampling::Cs444, width, height),
        };
        let copy_plane =
            |component, plane_width: u32, plane_height: u32| -> Result<VideoPlane, String> {
                let stride = picture.stride(component) as usize;
                let bytes_per_sample = if bit_depth <= 8 { 1 } else { 2 };
                let mut data = Vec::with_capacity(
                    plane_width as usize * plane_height as usize * bytes_per_sample,
                );
                if bit_depth <= 8 {
                    let source = picture.plane(component);
                    validate_source_plane(source.len(), stride, plane_width, plane_height)?;
                    for row in 0..plane_height as usize {
                        let start = row * stride;
                        data.extend_from_slice(&source[start..start + plane_width as usize]);
                    }
                } else {
                    let source = picture.plane16(component);
                    validate_source_plane(source.len(), stride, plane_width, plane_height)?;
                    for row in 0..plane_height as usize {
                        let start = row * stride;
                        for sample in &source[start..start + plane_width as usize] {
                            data.extend_from_slice(&sample.to_le_bytes());
                        }
                    }
                }
                Ok(VideoPlane {
                    width: plane_width,
                    height: plane_height,
                    stride: plane_width,
                    data,
                })
            };
        let mut planes = vec![copy_plane(PlanarImageComponent::Y, width, height)?];
        if sampling != ChromaSampling::Monochrome {
            planes.push(copy_plane(
                PlanarImageComponent::U,
                chroma_width,
                chroma_height,
            )?);
            planes.push(copy_plane(
                PlanarImageComponent::V,
                chroma_width,
                chroma_height,
            )?);
        }
        output.push(VideoFrame {
            width,
            height,
            bit_depth,
            chroma_sampling: sampling,
            has_alpha: false,
            pts: picture.timestamp(),
            duration: u64::try_from(picture.duration())
                .ok()
                .filter(|value| *value > 0),
            planes,
        });
    }
    Ok(output)
}

#[cfg(feature = "av1")]
fn validate_source_plane(
    source_samples: usize,
    stride: usize,
    width: u32,
    height: u32,
) -> Result<(), String> {
    let required = (height as usize)
        .saturating_sub(1)
        .checked_mul(stride)
        .and_then(|offset| offset.checked_add(width as usize))
        .ok_or_else(|| "AV1 source plane size overflow".to_string())?;
    if stride < width as usize || source_samples < required {
        return Err(format!(
            "AV1 source plane needs {required} samples, got {source_samples} (stride {stride}, width {width}, height {height})"
        ));
    }
    Ok(())
}

#[cfg(feature = "prores")]
fn prores_frame(
    frame: oxideav_core::VideoFrame,
    bit_depth: u8,
    pts: Option<i64>,
    duration: Option<u64>,
) -> Result<VideoFrame, String> {
    let planes = frame.image_planes();
    let y = planes
        .first()
        .ok_or_else(|| "ProRes frame has no luma plane".to_string())?;
    let u = planes
        .get(1)
        .ok_or_else(|| "ProRes frame has no chroma plane".to_string())?;
    let v = planes
        .get(2)
        .ok_or_else(|| "ProRes frame has no chroma plane".to_string())?;
    let bytes_per_sample = if bit_depth <= 8 { 1 } else { 2 };
    let width = (y.stride / bytes_per_sample) as u32;
    let height = (y.data.len() / y.stride) as u32;
    let chroma_sampling = if u.stride * 2 == y.stride {
        ChromaSampling::Cs422
    } else {
        ChromaSampling::Cs444
    };
    let has_alpha = planes.len() == 4;
    let mut output_planes = vec![
        VideoPlane {
            width,
            height,
            stride: (y.stride / bytes_per_sample) as u32,
            data: y.data.clone(),
        },
        VideoPlane {
            width: (u.stride / bytes_per_sample) as u32,
            height,
            stride: (u.stride / bytes_per_sample) as u32,
            data: u.data.clone(),
        },
        VideoPlane {
            width: (v.stride / bytes_per_sample) as u32,
            height,
            stride: (v.stride / bytes_per_sample) as u32,
            data: v.data.clone(),
        },
    ];
    if let Some(alpha) = planes.get(3) {
        output_planes.push(VideoPlane {
            width: (alpha.stride / bytes_per_sample) as u32,
            height,
            stride: (alpha.stride / bytes_per_sample) as u32,
            data: alpha.data.clone(),
        });
    }
    Ok(VideoFrame {
        width,
        height,
        bit_depth,
        chroma_sampling,
        has_alpha,
        pts: frame.pts.or(pts),
        duration,
        planes: output_planes,
    })
}

#[cfg(feature = "prores")]
#[derive(Clone, Copy)]
struct ProresOutput {
    pixel_format: oxideav_core::PixelFormat,
    bit_depth: u8,
}

#[cfg(feature = "prores")]
fn prores_output_format(data: &[u8]) -> Result<ProresOutput, String> {
    use oxideav_core::PixelFormat;
    use oxideav_prores::frame::ChromaFormat;

    let (header, _) = oxideav_prores::frame::parse_frame(data)
        .map_err(|error| format!("ProRes header decode failed: {error}"))?;
    let has_alpha = header.alpha_channel_type != 0;
    match (header.chroma_format, has_alpha) {
        (ChromaFormat::Y422, false) => Ok(ProresOutput {
            pixel_format: PixelFormat::Yuv422P10Le,
            bit_depth: 10,
        }),
        (ChromaFormat::Y422, true) => Ok(ProresOutput {
            pixel_format: PixelFormat::Yuva422P10Le,
            bit_depth: 10,
        }),
        (ChromaFormat::Y444, false) => Ok(ProresOutput {
            pixel_format: PixelFormat::Yuv444P12Le,
            bit_depth: 12,
        }),
        (ChromaFormat::Y444, true) => Ok(ProresOutput {
            pixel_format: PixelFormat::Yuva444P12Le,
            bit_depth: 12,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_common_container_codec_names() {
        assert_eq!(VideoCodec::parse("avc1"), Some(VideoCodec::H264));
        assert_eq!(VideoCodec::parse("hvc1"), Some(VideoCodec::Hevc));
        assert_eq!(VideoCodec::parse("V_VP9"), Some(VideoCodec::Vp9));
        assert_eq!(VideoCodec::parse("av01"), Some(VideoCodec::Av1));
        assert_eq!(VideoCodec::parse("apch"), Some(VideoCodec::ProRes));
        assert_eq!(VideoCodec::parse("AVdn"), Some(VideoCodec::DnxHd));
    }

    #[test]
    fn parses_avcc_and_normalizes_length_prefixed_nals() {
        let avcc = [1, 100, 0, 31, 0xff, 0xe1, 0, 2, 0x67, 0x64, 1, 0, 1, 0x68];
        let config = parse_avc_decoder_configuration(&avcc).unwrap();
        assert_eq!(config.length_size, 4);
        assert_eq!(config.annex_b, [0, 0, 0, 1, 0x67, 0x64, 0, 0, 0, 1, 0x68]);
        assert_eq!(
            length_prefixed_nals_to_annex_b(&[0, 0, 0, 2, 0x65, 0x88], 4).unwrap(),
            [0, 0, 0, 1, 0x65, 0x88]
        );
    }

    #[test]
    fn parses_hvcc_and_rejects_truncated_nals() {
        let mut hvcc = vec![0; 23];
        hvcc[0] = 1;
        hvcc[21] = 3;
        hvcc[22] = 1;
        hvcc.extend_from_slice(&[0x20, 0, 1, 0, 2, 0x40, 0x01]);
        let config = parse_hevc_decoder_configuration(&hvcc).unwrap();
        assert_eq!(config.length_size, 4);
        assert_eq!(config.annex_b, [0, 0, 0, 1, 0x40, 0x01]);
        assert!(length_prefixed_nals_to_annex_b(&[0, 0, 0, 2, 0x65], 4).is_err());
    }

    #[test]
    fn rejects_unbounded_frame_surfaces() {
        let frame = VideoFrame {
            width: 100_000,
            height: 100_000,
            bit_depth: 8,
            chroma_sampling: ChromaSampling::Monochrome,
            has_alpha: false,
            pts: None,
            duration: None,
            planes: vec![],
        };
        assert!(frame.validate().unwrap_err().contains("pixel budget"));
    }

    #[test]
    fn dnx_reports_an_explicit_native_gap() {
        let error = match VideoDecoder::new(VideoCodec::DnxHd) {
            Ok(_) => panic!("DNx decoder unexpectedly initialized"),
            Err(error) => error,
        };
        assert!(error.contains("not yet available"));
    }
}
