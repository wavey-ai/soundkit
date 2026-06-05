use crate::bitreader::BitReader;
use crate::channel::{ChannelPairElementHeader, IndividualChannelStream, MidSideMask};
use crate::config::AudioSpecificConfig;
use crate::dsp::{AacDsp, DspChannel};
use crate::error::{AacLcError, Result};
use crate::ics::{IcsInfo, WindowSequence};
use crate::scalefactor::StandardScaleFactorDecoder;
use crate::sfb::{long_window_band_layout, short_window_band_layout};
use crate::spectral::{
    warm_standard_spectral_tables, BandLayout, SpectralCoefficients, PNS_LCG_SEED,
};
use crate::stereo::{
    apply_intensity_stereo_long, apply_intensity_stereo_short,
    apply_mid_side_long_excluding_intensity, apply_mid_side_short_excluding_intensity,
};
use crate::syntax::{ElementId, ElementInstanceTag, RawElementHeader};
use crate::tns::{apply_tns, lc_tns_max_bands};

const EXT_SBR_DATA: u8 = 13;
const EXT_SBR_DATA_CRC: u8 = 14;

#[derive(Debug)]
pub struct PlanarF32<'a> {
    channels: &'a [Vec<f32>],
    frames: usize,
}

impl<'a> PlanarF32<'a> {
    pub const fn channels(&self) -> &'a [Vec<f32>] {
        self.channels
    }

    pub const fn frames(&self) -> usize {
        self.frames
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AacLcFrame {
    pub sample_rate: u32,
    pub channels: usize,
    pub frames: usize,
}

#[derive(Debug)]
pub struct AacLcDecoder {
    config: AudioSpecificConfig,
    pcm: Vec<Vec<f32>>,
    coefficients: Vec<SpectralCoefficients>,
    dsp: AacDsp,
    dsp_channels: Vec<DspChannel>,
    pns_state: u32,
}

impl AacLcDecoder {
    pub fn new(config: AudioSpecificConfig) -> Result<Self> {
        config.validate_aac_lc_packet_path()?;
        warm_standard_spectral_tables();
        let channels = config
            .channels()
            .ok_or(AacLcError::InvalidConfig("AAC-LC channel count is unknown"))?;
        let pcm = vec![vec![0.0; config.frame_length]; channels];
        let coefficients = (0..channels)
            .map(|_| SpectralCoefficients::new(config.frame_length))
            .collect();
        let dsp_channels = (0..channels)
            .map(|_| DspChannel::new(config.frame_length))
            .collect();

        Ok(Self {
            config,
            pcm,
            coefficients,
            dsp: AacDsp::new(),
            dsp_channels,
            pns_state: PNS_LCG_SEED,
        })
    }

    pub fn from_audio_specific_config(data: &[u8]) -> Result<Self> {
        Self::new(AudioSpecificConfig::parse(data)?)
    }

    pub const fn config(&self) -> &AudioSpecificConfig {
        &self.config
    }

    pub fn frame_info(&self) -> AacLcFrame {
        AacLcFrame {
            sample_rate: self.config.sample_rate(),
            channels: self.config.channels().unwrap_or(0),
            frames: self.config.frame_length,
        }
    }

    pub fn output_capacity(&self) -> AacLcFrame {
        AacLcFrame {
            sample_rate: self.config.sample_rate(),
            channels: self.dsp_channels.len(),
            frames: self.pcm.first().map_or(0, Vec::len),
        }
    }

    pub fn decode_access_unit<'a>(&'a mut self, data: &[u8]) -> Result<PlanarF32<'a>> {
        let mut reader = BitReader::new(data);
        let mut decoded_audio = false;

        while reader.remaining_bits() >= 3 {
            if decoded_audio && remaining_bits_are_zero(&mut reader)? {
                break;
            }

            let header = RawElementHeader::read(&mut reader)?;

            match header.id {
                ElementId::SingleChannel => {
                    if decoded_audio {
                        return Err(AacLcError::InvalidBitstream(
                            "raw access unit contains multiple channel elements",
                        ));
                    }
                    self.decode_single_channel(&mut reader, header.tag)?;
                    decoded_audio = true;
                }
                ElementId::ChannelPair => {
                    if decoded_audio {
                        return Err(AacLcError::InvalidBitstream(
                            "raw access unit contains multiple channel elements",
                        ));
                    }
                    self.decode_channel_pair(&mut reader, header.tag)?;
                    decoded_audio = true;
                }
                ElementId::ChannelCoupling => {
                    return Err(AacLcError::UnsupportedFeature("channel coupling element"));
                }
                ElementId::LowFrequency => {
                    return Err(AacLcError::UnsupportedFeature("low frequency element"));
                }
                ElementId::DataStream => {
                    return Err(AacLcError::UnsupportedFeature("data stream element"));
                }
                ElementId::ProgramConfig => {
                    return Err(AacLcError::UnsupportedFeature("program config element"));
                }
                ElementId::Fill => skip_fill_element(&mut reader)?,
                ElementId::End => break,
            }
        }

        if !decoded_audio {
            return Err(AacLcError::InvalidBitstream(
                "raw access unit does not contain an AAC-LC channel element",
            ));
        }

        if !remaining_bits_are_zero(&mut reader)? {
            return Err(AacLcError::InvalidBitstream(
                "raw access unit has non-zero trailing bits",
            ));
        }

        Ok(self.output_view())
    }

    fn decode_single_channel(
        &mut self,
        reader: &mut BitReader<'_>,
        tag: Option<ElementInstanceTag>,
    ) -> Result<()> {
        if self.pcm.len() != 1 {
            return Err(AacLcError::InvalidBitstream(
                "single channel element does not match configured channel count",
            ));
        }
        let _tag = tag.ok_or(AacLcError::InvalidBitstream("missing SCE instance tag"))?;
        let mut scale_factor_decoder = StandardScaleFactorDecoder;
        let stream = IndividualChannelStream::read(reader, None, &mut scale_factor_decoder)?;

        self.decode_channel_spectrum(0, reader, &stream, false)?;
        self.synthesize_channel(0, &stream.prefix.ics_info)?;
        Ok(())
    }

    fn decode_channel_pair(
        &mut self,
        reader: &mut BitReader<'_>,
        tag: Option<ElementInstanceTag>,
    ) -> Result<()> {
        if self.pcm.len() != 2 {
            return Err(AacLcError::InvalidBitstream(
                "channel pair element does not match configured channel count",
            ));
        }

        let tag = tag.ok_or(AacLcError::InvalidBitstream("missing CPE instance tag"))?;
        let header = ChannelPairElementHeader::read(reader, tag)?;
        let mut scale_factor_decoder = StandardScaleFactorDecoder;

        let left =
            IndividualChannelStream::read(reader, header.common_ics, &mut scale_factor_decoder)?;
        self.decode_channel_spectrum(0, reader, &left, false)?;

        let right =
            IndividualChannelStream::read(reader, header.common_ics, &mut scale_factor_decoder)?;
        self.decode_channel_spectrum(1, reader, &right, true)?;

        self.apply_common_stereo_tools(&header, &left.prefix.ics_info, &left, &right)?;
        self.synthesize_channel(0, &left.prefix.ics_info)?;
        self.synthesize_channel(1, &right.prefix.ics_info)?;
        Ok(())
    }

    fn decode_channel_spectrum(
        &mut self,
        channel: usize,
        reader: &mut BitReader<'_>,
        stream: &IndividualChannelStream,
        allow_intensity: bool,
    ) -> Result<()> {
        if stream.prefix.section_data.has_intensity_stereo() && !allow_intensity {
            return Err(AacLcError::InvalidBitstream(
                "intensity stereo is only valid in the right channel of a channel pair",
            ));
        }

        let layout = self.band_layout(&stream.prefix.ics_info)?;
        self.coefficients[channel].decode_standard_with_pulse_and_pns(
            reader,
            &stream.prefix,
            &stream.scale_factors,
            layout,
            stream.pulse_data.as_ref(),
            &mut self.pns_state,
        )?;

        if let Some(tns) = stream.tns_data {
            let max_tns_bands = lc_tns_max_bands(
                self.config.sampling_frequency,
                stream.prefix.ics_info.window_sequence,
            )?;
            apply_tns(
                &tns,
                &stream.prefix.ics_info,
                layout,
                max_tns_bands,
                self.coefficients[channel].dequantized_mut(),
            )?;
        }
        Ok(())
    }

    fn apply_common_stereo_tools(
        &mut self,
        header: &ChannelPairElementHeader,
        info: &IcsInfo,
        left_stream: &IndividualChannelStream,
        right_stream: &IndividualChannelStream,
    ) -> Result<()> {
        if !header.common_window
            && (header.mid_side != MidSideMask::None
                || right_stream.prefix.section_data.has_intensity_stereo())
        {
            return Err(AacLcError::InvalidBitstream(
                "common stereo tools require common window",
            ));
        }
        if !header.common_window {
            return Ok(());
        }

        let layout = self.band_layout(info)?;
        let (left_coefficients, right_coefficients) = self.coefficients.split_at_mut(1);
        let left = left_coefficients[0].dequantized_mut();
        let right = right_coefficients[0].dequantized_mut();

        match info.window_sequence {
            WindowSequence::EightShort => {
                apply_intensity_stereo_short(
                    header.mid_side,
                    info,
                    layout,
                    &right_stream.prefix.section_data,
                    &right_stream.scale_factors,
                    left,
                    right,
                )?;
                apply_mid_side_short_excluding_intensity(
                    header.mid_side,
                    info,
                    layout,
                    &left_stream.prefix.section_data,
                    &right_stream.prefix.section_data,
                    left,
                    right,
                )
            }
            WindowSequence::OnlyLong | WindowSequence::LongStart | WindowSequence::LongStop => {
                apply_intensity_stereo_long(
                    header.mid_side,
                    info.max_sfb,
                    layout,
                    &right_stream.prefix.section_data,
                    &right_stream.scale_factors,
                    left,
                    right,
                )?;
                apply_mid_side_long_excluding_intensity(
                    header.mid_side,
                    info.max_sfb,
                    layout,
                    &left_stream.prefix.section_data,
                    &right_stream.prefix.section_data,
                    left,
                    right,
                )
            }
        }
    }

    fn synthesize_channel(&mut self, channel: usize, info: &IcsInfo) -> Result<()> {
        let previous_window_shape = self.dsp_channels[channel].previous_window_shape();

        match info.window_sequence {
            WindowSequence::OnlyLong | WindowSequence::LongStart | WindowSequence::LongStop => {
                let coeffs = self.coefficients[channel].dequantized();
                let previous_long_window = self.dsp.long_window(previous_window_shape);
                let long_window = self.dsp.long_window(info.window_shape);
                let previous_short_window = self.dsp.short_window(previous_window_shape);
                let short_window = self.dsp.short_window(info.window_shape);
                self.dsp_channels[channel].synthesize_long_sequence(
                    coeffs,
                    info.window_sequence,
                    self.dsp.long_imdct_transform(),
                    previous_long_window,
                    long_window,
                    previous_short_window,
                    short_window,
                    &mut self.pcm[channel],
                )
            }
            WindowSequence::EightShort => {
                let coeffs = self.coefficients[channel].dequantized();
                let previous_short_window = self.dsp.short_window(previous_window_shape);
                let short_window = self.dsp.short_window(info.window_shape);
                self.dsp_channels[channel].synthesize_eight_short(
                    coeffs,
                    self.dsp.short_imdct_transform(),
                    previous_short_window,
                    short_window,
                    &mut self.pcm[channel],
                )
            }
        }?;

        self.dsp_channels[channel].set_previous_window_shape(info.window_shape);

        Ok(())
    }

    fn band_layout(&self, info: &IcsInfo) -> Result<BandLayout<'static>> {
        match info.window_sequence {
            WindowSequence::EightShort => short_window_band_layout(self.config.sampling_frequency),
            WindowSequence::OnlyLong | WindowSequence::LongStart | WindowSequence::LongStop => {
                long_window_band_layout(self.config.sampling_frequency)
            }
        }
    }

    fn output_view(&self) -> PlanarF32<'_> {
        PlanarF32 {
            channels: &self.pcm,
            frames: self.config.frame_length,
        }
    }
}

fn skip_fill_element(reader: &mut BitReader<'_>) -> Result<()> {
    let mut count = reader.read_u8(4)? as usize;
    if count == 15 {
        let extended = reader.read_u8(8)?;
        if extended == 0 {
            return Err(AacLcError::InvalidBitstream("invalid fill element length"));
        }
        count += usize::from(extended - 1);
    }

    if count == 0 {
        return Ok(());
    }

    if reader.remaining_bits() < count * 8 {
        return Err(AacLcError::UnexpectedEof {
            requested_bits: (count * 8).min(u8::MAX as usize) as u8,
            remaining_bits: reader.remaining_bits(),
        });
    }

    let extension_type = reader.peek_u32(4)? as u8;
    if matches!(extension_type, EXT_SBR_DATA | EXT_SBR_DATA_CRC) {
        return Err(AacLcError::UnsupportedFeature(
            "SBR/HE-AAC extension payload",
        ));
    }

    reader.skip_bits(count * 8)
}

fn remaining_bits_are_zero(reader: &mut BitReader<'_>) -> Result<bool> {
    let mut probe = reader.clone();
    while probe.remaining_bits() >= 32 {
        if probe.read_u32(32)? != 0 {
            return Ok(false);
        }
    }

    let remaining = probe.remaining_bits();
    if remaining == 0 {
        return Ok(true);
    }

    Ok(probe.read_u32(remaining as u8)? == 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{AudioObjectType, ChannelConfig};

    #[test]
    fn creates_decoder_from_aac_lc_stereo_asc() {
        let decoder = AacLcDecoder::from_audio_specific_config(&[0x12, 0x10]).unwrap();
        let info = decoder.frame_info();

        assert_eq!(decoder.config().object_type, AudioObjectType::AacLc);
        assert_eq!(decoder.config().channel_config, ChannelConfig::Stereo);
        assert_eq!(info.sample_rate, 44_100);
        assert_eq!(info.channels, 2);
        assert_eq!(info.frames, 1024);
        assert_eq!(decoder.output_capacity(), info);
    }

    #[test]
    fn rejects_he_aac_config_at_initialization() {
        let config = AudioSpecificConfig {
            object_type: AudioObjectType::AacLc,
            sampling_frequency: crate::config::SamplingFrequency {
                index: Some(4),
                hz: 44_100,
            },
            channel_config: ChannelConfig::Stereo,
            frame_length: 1024,
            depends_on_core_coder: false,
            extension_flag: false,
            sbr_present: true,
            ps_present: false,
            extension_sampling_frequency: None,
        };

        assert_eq!(
            AacLcDecoder::new(config).unwrap_err(),
            AacLcError::UnsupportedFeature("SBR/HE-AAC")
        );
    }

    #[test]
    fn raw_access_unit_entrypoint_reads_element_id() {
        let mut decoder = AacLcDecoder::from_audio_specific_config(&[0x12, 0x10]).unwrap();

        assert_eq!(
            decoder.decode_access_unit(&[0b0010_0000]).unwrap_err(),
            AacLcError::UnexpectedEof {
                requested_bits: 8,
                remaining_bits: 0,
            }
        );
    }

    #[test]
    fn decodes_silent_single_channel_zero_spectral_frame() {
        let mut decoder = AacLcDecoder::from_audio_specific_config(&[0x12, 0x08]).unwrap();
        let data = build_bits(&silent_sce_fields());

        let pcm = decoder.decode_access_unit(&data).unwrap();

        assert_eq!(pcm.frames(), 1024);
        assert_eq!(pcm.channels().len(), 1);
        assert!(pcm.channels()[0].iter().all(|sample| *sample == 0.0));
    }

    #[test]
    fn decodes_trailing_end_element_after_audio() {
        let mut decoder = AacLcDecoder::from_audio_specific_config(&[0x12, 0x08]).unwrap();
        let mut fields = silent_sce_fields();
        fields.extend_from_slice(&[(7, 3)]); // END
        let data = build_bits(&fields);

        let pcm = decoder.decode_access_unit(&data).unwrap();

        assert_eq!(pcm.frames(), 1024);
        assert_eq!(pcm.channels().len(), 1);
        assert!(pcm.channels()[0].iter().all(|sample| *sample == 0.0));
    }

    #[test]
    fn skips_trailing_fill_element_after_audio() {
        let mut decoder = AacLcDecoder::from_audio_specific_config(&[0x12, 0x08]).unwrap();
        let mut fields = silent_sce_fields();
        fields.extend_from_slice(&[
            (6, 3), // FIL
            (1, 4), // count
            (0, 8), // one fill payload byte with non-SBR extension type
            (7, 3), // END
        ]);
        let data = build_bits(&fields);

        let pcm = decoder.decode_access_unit(&data).unwrap();

        assert_eq!(pcm.frames(), 1024);
        assert_eq!(pcm.channels().len(), 1);
        assert!(pcm.channels()[0].iter().all(|sample| *sample == 0.0));
    }

    #[test]
    fn rejects_sbr_fill_extension_payload() {
        let mut decoder = AacLcDecoder::from_audio_specific_config(&[0x12, 0x08]).unwrap();
        let mut fields = silent_sce_fields();
        fields.extend_from_slice(&[
            (6, 3),                   // FIL
            (1, 4),                   // count
            (EXT_SBR_DATA as u32, 4), // extension_type
            (0, 4),                   // payload tail
        ]);
        let data = build_bits(&fields);

        assert_eq!(
            decoder.decode_access_unit(&data).unwrap_err(),
            AacLcError::UnsupportedFeature("SBR/HE-AAC extension payload")
        );
    }

    #[test]
    fn rejects_second_channel_element_in_raw_block() {
        let mut decoder = AacLcDecoder::from_audio_specific_config(&[0x12, 0x08]).unwrap();
        let mut fields = silent_sce_fields();
        fields.extend_from_slice(&[
            (0, 3), // SCE
            (1, 4), // tag
        ]);
        let data = build_bits(&fields);

        assert_eq!(
            decoder.decode_access_unit(&data).unwrap_err(),
            AacLcError::InvalidBitstream("raw access unit contains multiple channel elements")
        );
    }

    #[test]
    fn decodes_nonzero_single_channel_long_spectral_frame() {
        let mut decoder = AacLcDecoder::from_audio_specific_config(&[0x12, 0x08]).unwrap();
        let data = build_bits(&[
            (0, 3),       // SCE
            (0, 4),       // tag
            (100, 8),     // global gain
            (0, 1),       // reserved
            (0, 2),       // only long
            (0, 1),       // sine
            (1, 6),       // max_sfb
            (0, 1),       // predictor
            (1, 4),       // codebook 1
            (1, 5),       // section length
            (0, 1),       // scalefactor delta 0
            (0, 1),       // pulse_data_present
            (0, 1),       // tns_data_present
            (0, 1),       // gain_control_data_present
            (0b10100, 5), // codebook 1 tuple [0, 0, 0, 1]
        ]);

        let pcm = decoder.decode_access_unit(&data).unwrap();

        assert_eq!(pcm.frames(), 1024);
        assert_eq!(pcm.channels().len(), 1);
        assert!(pcm.channels()[0].iter().all(|sample| sample.is_finite()));
        assert!(pcm.channels()[0].iter().any(|sample| *sample != 0.0));
    }

    #[test]
    fn decodes_silent_channel_pair_zero_spectral_frame() {
        let mut decoder = AacLcDecoder::from_audio_specific_config(&[0x12, 0x10]).unwrap();
        let data = build_bits(&[
            (1, 3),   // CPE
            (0, 4),   // tag
            (1, 1),   // common_window
            (0, 1),   // reserved
            (0, 2),   // only long
            (0, 1),   // sine
            (1, 6),   // max_sfb
            (0, 1),   // predictor
            (0, 2),   // no mid-side
            (100, 8), // left global gain
            (0, 4),   // left zero codebook
            (1, 5),   // left section length
            (0, 1),   // left pulse
            (0, 1),   // left tns
            (0, 1),   // left gain control
            (100, 8), // right global gain
            (0, 4),   // right zero codebook
            (1, 5),   // right section length
            (0, 1),   // right pulse
            (0, 1),   // right tns
            (0, 1),   // right gain control
        ]);

        let pcm = decoder.decode_access_unit(&data).unwrap();

        assert_eq!(pcm.frames(), 1024);
        assert_eq!(pcm.channels().len(), 2);
        assert!(pcm.channels()[0].iter().all(|sample| *sample == 0.0));
        assert!(pcm.channels()[1].iter().all(|sample| *sample == 0.0));
    }

    #[test]
    fn decodes_non_common_window_pair_without_common_stereo_tools() {
        let mut decoder = AacLcDecoder::from_audio_specific_config(&[0x11, 0x90]).unwrap();
        let mut fields = vec![
            (1, 3),   // CPE
            (0, 4),   // tag
            (0, 1),   // common_window
            (100, 8), // left global gain
            (0, 1),   // left reserved
            (2, 2),   // left eight short
            (0, 1),   // left sine
            (1, 4),   // left max_sfb
            (0, 7),   // left ungrouped short windows
        ];
        for _ in 0..8 {
            fields.extend_from_slice(&[
                (0, 4), // left zero codebook
                (1, 3), // left section length
            ]);
        }
        fields.extend_from_slice(&[
            (0, 1),   // left pulse
            (0, 1),   // left tns
            (0, 1),   // left gain control
            (100, 8), // right global gain
            (0, 1),   // right reserved
            (0, 2),   // right only long
            (0, 1),   // right sine
            (1, 6),   // right max_sfb
            (0, 1),   // right predictor
            (0, 4),   // right zero codebook
            (1, 5),   // right section length
            (0, 1),   // right pulse
            (0, 1),   // right tns
            (0, 1),   // right gain control
        ]);
        let data = build_bits(&fields);

        let pcm = decoder.decode_access_unit(&data).unwrap();

        assert_eq!(pcm.frames(), 1024);
        assert_eq!(pcm.channels().len(), 2);
        assert!(pcm.channels()[0].iter().all(|sample| *sample == 0.0));
        assert!(pcm.channels()[1].iter().all(|sample| *sample == 0.0));
    }

    #[test]
    fn decodes_channel_pair_spectral_data_before_next_channel_header() {
        let mut decoder = AacLcDecoder::from_audio_specific_config(&[0x12, 0x10]).unwrap();
        let data = build_bits(&[
            (1, 3),       // CPE
            (0, 4),       // tag
            (1, 1),       // common_window
            (0, 1),       // reserved
            (0, 2),       // only long
            (0, 1),       // sine
            (1, 6),       // max_sfb
            (0, 1),       // predictor
            (0, 2),       // no mid-side
            (100, 8),     // left global gain
            (1, 4),       // left codebook 1
            (1, 5),       // left section length
            (0, 1),       // left scalefactor delta 0
            (0, 1),       // left pulse
            (0, 1),       // left tns
            (0, 1),       // left gain control
            (0b10100, 5), // left codebook 1 tuple [0, 0, 0, 1]
            (100, 8),     // right global gain
            (0, 4),       // right zero codebook
            (1, 5),       // right section length
            (0, 1),       // right pulse
            (0, 1),       // right tns
            (0, 1),       // right gain control
        ]);

        let pcm = decoder.decode_access_unit(&data).unwrap();

        assert_eq!(pcm.frames(), 1024);
        assert_eq!(pcm.channels().len(), 2);
        assert!(pcm.channels()[0].iter().any(|sample| *sample != 0.0));
        assert!(pcm.channels()[1].iter().all(|sample| *sample == 0.0));
    }

    fn silent_sce_fields() -> Vec<(u32, u8)> {
        vec![
            (0, 3),   // SCE
            (0, 4),   // tag
            (100, 8), // global gain
            (0, 1),   // reserved
            (0, 2),   // only long
            (0, 1),   // sine
            (1, 6),   // max_sfb
            (0, 1),   // predictor
            (0, 4),   // zero codebook
            (1, 5),   // section length
            (0, 1),   // pulse_data_present
            (0, 1),   // tns_data_present
            (0, 1),   // gain_control_data_present
        ]
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
}
