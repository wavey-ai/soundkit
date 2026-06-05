use crate::bitreader::BitReader;
use crate::error::{AacLcError, Result};
use crate::ics::{IcsInfo, MAX_WINDOW_GROUPS};
use crate::pulse::PulseData;
use crate::scalefactor::{ScaleFactorData, ScaleFactorDecoder, StandardScaleFactorDecoder};
use crate::section::{SectionData, MAX_SCALE_FACTOR_BANDS};
use crate::spectral::{BandLayout, SpectralCoefficients, SpectralDecoder};
use crate::syntax::ElementInstanceTag;
use crate::tns::TnsData;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndividualChannelStreamPrefix {
    pub global_gain: u8,
    pub ics_info: IcsInfo,
    pub section_data: SectionData,
}

impl IndividualChannelStreamPrefix {
    pub fn read(reader: &mut BitReader<'_>, common_ics: Option<IcsInfo>) -> Result<Self> {
        let global_gain = reader.read_u8(8)?;
        let ics_info = match common_ics {
            Some(info) => info,
            None => IcsInfo::read(reader)?,
        };
        let section_data = SectionData::read(reader, &ics_info)?;

        Ok(Self {
            global_gain,
            ics_info,
            section_data,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndividualChannelStream {
    pub prefix: IndividualChannelStreamPrefix,
    pub scale_factors: ScaleFactorData,
    pub pulse_data: Option<PulseData>,
    pub tns_data: Option<TnsData>,
}

impl IndividualChannelStream {
    pub fn read<D: ScaleFactorDecoder>(
        reader: &mut BitReader<'_>,
        common_ics: Option<IcsInfo>,
        scale_factor_decoder: &mut D,
    ) -> Result<Self> {
        let prefix = IndividualChannelStreamPrefix::read(reader, common_ics)?;
        let scale_factors = ScaleFactorData::read(reader, &prefix, scale_factor_decoder)?;

        let pulse_data = if reader.read_bool()? {
            Some(PulseData::read(reader)?)
        } else {
            None
        };

        let tns_data = if reader.read_bool()? {
            Some(TnsData::read(reader, &prefix.ics_info)?)
        } else {
            None
        };

        let gain_control_data_present = reader.read_bool()?;
        if gain_control_data_present {
            return Err(AacLcError::UnsupportedFeature("gain control"));
        }

        Ok(Self {
            prefix,
            scale_factors,
            pulse_data,
            tns_data,
        })
    }

    pub fn has_spectral_data(&self) -> bool {
        !self.prefix.section_data.is_all_zero()
    }

    pub fn ensure_zero_spectral_payload(&self) -> Result<()> {
        if self.pulse_data.is_some() {
            return Err(AacLcError::NotImplemented("pulse tool synthesis"));
        }
        if self.has_spectral_data() {
            return Err(AacLcError::NotImplemented("Huffman spectral decode"));
        }
        Ok(())
    }

    pub fn decode_long_spectral<'a, D: SpectralDecoder>(
        &self,
        reader: &mut BitReader<'_>,
        coefficients: &'a mut SpectralCoefficients,
        layout: BandLayout<'_>,
        spectral_decoder: &mut D,
    ) -> Result<&'a [f32]> {
        if self.tns_data.is_some() {
            return Err(AacLcError::NotImplemented("TNS filtering"));
        }

        coefficients.decode_long_with_pulse(
            reader,
            &self.prefix,
            &self.scale_factors,
            layout,
            spectral_decoder,
            self.pulse_data.as_ref(),
        )
    }

    pub fn decode_spectral<'a, D: SpectralDecoder>(
        &self,
        reader: &mut BitReader<'_>,
        coefficients: &'a mut SpectralCoefficients,
        layout: BandLayout<'_>,
        spectral_decoder: &mut D,
    ) -> Result<&'a [f32]> {
        if self.tns_data.is_some() {
            return Err(AacLcError::NotImplemented("TNS filtering"));
        }

        if self.pulse_data.is_some()
            && self.prefix.ics_info.window_sequence == crate::ics::WindowSequence::EightShort
        {
            return Err(AacLcError::InvalidBitstream(
                "pulse data is not allowed for short windows",
            ));
        }

        if self.pulse_data.is_some() {
            return coefficients.decode_long_with_pulse(
                reader,
                &self.prefix,
                &self.scale_factors,
                layout,
                spectral_decoder,
                self.pulse_data.as_ref(),
            );
        }

        coefficients.decode(
            reader,
            &self.prefix,
            &self.scale_factors,
            layout,
            spectral_decoder,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SingleChannelElement {
    pub tag: ElementInstanceTag,
    pub channel: IndividualChannelStreamPrefix,
}

impl SingleChannelElement {
    pub fn read_zero_spectral(reader: &mut BitReader<'_>, tag: ElementInstanceTag) -> Result<Self> {
        let channel = read_zero_spectral_channel(reader, None)?;
        Ok(Self { tag, channel })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MidSideMask {
    None,
    Some {
        max_sfb: u8,
        num_window_groups: u8,
        used: [[bool; MAX_SCALE_FACTOR_BANDS]; MAX_WINDOW_GROUPS],
    },
    All,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChannelPairElementHeader {
    pub tag: ElementInstanceTag,
    pub common_window: bool,
    pub common_ics: Option<IcsInfo>,
    pub mid_side: MidSideMask,
}

impl ChannelPairElementHeader {
    pub fn read(reader: &mut BitReader<'_>, tag: ElementInstanceTag) -> Result<Self> {
        let common_window = reader.read_bool()?;
        if !common_window {
            return Ok(Self {
                tag,
                common_window,
                common_ics: None,
                mid_side: MidSideMask::None,
            });
        }

        let common_ics = IcsInfo::read(reader)?;
        let mid_side = read_mid_side_mask(reader, &common_ics)?;

        Ok(Self {
            tag,
            common_window,
            common_ics: Some(common_ics),
            mid_side,
        })
    }
}

pub fn read_zero_spectral_channel(
    reader: &mut BitReader<'_>,
    common_ics: Option<IcsInfo>,
) -> Result<IndividualChannelStreamPrefix> {
    let mut scale_factor_decoder = StandardScaleFactorDecoder;
    let stream = IndividualChannelStream::read(reader, common_ics, &mut scale_factor_decoder)?;
    stream.ensure_zero_spectral_payload()?;
    Ok(stream.prefix)
}

fn read_mid_side_mask(reader: &mut BitReader<'_>, info: &IcsInfo) -> Result<MidSideMask> {
    match reader.read_u8(2)? {
        0 => Ok(MidSideMask::None),
        1 => {
            if info.max_sfb as usize > MAX_SCALE_FACTOR_BANDS {
                return Err(AacLcError::InvalidBitstream(
                    "max_sfb exceeds parser capacity",
                ));
            }

            let mut used = [[false; MAX_SCALE_FACTOR_BANDS]; MAX_WINDOW_GROUPS];
            for group in 0..info.num_window_groups as usize {
                for sfb in 0..info.max_sfb as usize {
                    used[group][sfb] = reader.read_bool()?;
                }
            }

            Ok(MidSideMask::Some {
                max_sfb: info.max_sfb,
                num_window_groups: info.num_window_groups,
                used,
            })
        }
        2 => Ok(MidSideMask::All),
        3 => Err(AacLcError::InvalidBitstream("reserved mid/side mask mode")),
        _ => unreachable!("2-bit mid/side mask mode cannot exceed 3"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ics::{WindowSequence, WindowShape};
    use crate::pulse::Pulse;
    use crate::scalefactor::ScaleFactorValue;
    use crate::spectral::{BandLayout, SpectralCoefficients, SpectralDecoder};

    #[test]
    fn parses_zero_spectral_long_channel_prefix() {
        let bytes = build_bits(&[
            (128, 8), // global gain
            (0, 1),   // reserved
            (0, 2),   // only long
            (0, 1),   // sine
            (2, 6),   // max_sfb
            (0, 1),   // predictor
            (0, 4),   // zero codebook
            (2, 5),   // section length
            (0, 1),   // pulse_data_present
            (0, 1),   // tns_data_present
            (0, 1),   // gain_control_data_present
        ]);
        let mut reader = BitReader::new(&bytes);
        let prefix = read_zero_spectral_channel(&mut reader, None).unwrap();

        assert_eq!(prefix.global_gain, 128);
        assert_eq!(prefix.ics_info.max_sfb, 2);
        assert!(prefix.section_data.is_all_zero());
    }

    #[test]
    fn rejects_nonzero_section_before_spectral_decode_exists() {
        let bytes = build_bits(&[
            (128, 8), // global gain
            (0, 1),   // reserved
            (0, 2),   // only long
            (0, 1),   // sine
            (1, 6),   // max_sfb
            (0, 1),   // predictor
            (1, 4),   // spectral codebook
            (1, 5),   // section length
            (0, 1),   // scalefactor delta 0
            (0, 1),   // pulse_data_present
            (0, 1),   // tns_data_present
            (0, 1),   // gain_control_data_present
        ]);
        let mut reader = BitReader::new(&bytes);

        assert_eq!(
            read_zero_spectral_channel(&mut reader, None).unwrap_err(),
            AacLcError::NotImplemented("Huffman spectral decode")
        );
    }

    #[test]
    fn parses_common_window_mid_side_some_mask() {
        let bytes = build_bits(&[
            (1, 1), // common window
            (0, 1), // reserved
            (0, 2), // only long
            (0, 1), // sine
            (3, 6), // max_sfb
            (0, 1), // predictor
            (1, 2), // ms some
            (1, 1),
            (0, 1),
            (1, 1),
        ]);
        let mut reader = BitReader::new(&bytes);
        let header = ChannelPairElementHeader::read(&mut reader, ElementInstanceTag(2)).unwrap();

        assert!(header.common_window);
        assert_eq!(header.common_ics.unwrap().max_sfb, 3);
        match header.mid_side {
            MidSideMask::Some {
                max_sfb,
                num_window_groups,
                used,
            } => {
                assert_eq!(max_sfb, 3);
                assert_eq!(num_window_groups, 1);
                assert!(used[0][0]);
                assert!(!used[0][1]);
                assert!(used[0][2]);
            }
            other => panic!("unexpected mid-side mask: {other:?}"),
        }
    }

    #[test]
    fn rejects_reserved_mid_side_mode() {
        let bytes = build_bits(&[
            (1, 1), // common window
            (0, 1), // reserved
            (0, 2), // only long
            (0, 1), // sine
            (1, 6), // max_sfb
            (0, 1), // predictor
            (3, 2), // reserved ms mode
        ]);
        let mut reader = BitReader::new(&bytes);

        assert_eq!(
            ChannelPairElementHeader::read(&mut reader, ElementInstanceTag(0)).unwrap_err(),
            AacLcError::InvalidBitstream("reserved mid/side mask mode")
        );
    }

    #[test]
    fn parses_channel_with_common_ics() {
        let info = IcsInfo {
            window_sequence: WindowSequence::OnlyLong,
            window_shape: WindowShape::Sine,
            max_sfb: 1,
            num_windows: 1,
            num_window_groups: 1,
            window_group_len: [1, 0, 0, 0, 0, 0, 0, 0],
        };
        let bytes = build_bits(&[
            (90, 8), // global gain
            (0, 4),  // zero codebook
            (1, 5),  // section length
            (0, 1),  // pulse
            (0, 1),  // tns
            (0, 1),  // gain control
        ]);
        let mut reader = BitReader::new(&bytes);
        let prefix = read_zero_spectral_channel(&mut reader, Some(info)).unwrap();

        assert_eq!(prefix.global_gain, 90);
        assert_eq!(prefix.ics_info, info);
    }

    #[test]
    fn reads_nonzero_payload_with_injected_scalefactor_decoder() {
        let bytes = build_bits(&[
            (100, 8), // global gain
            (0, 1),   // reserved
            (0, 2),   // only long
            (0, 1),   // sine
            (1, 6),   // max_sfb
            (0, 1),   // predictor
            (1, 4),   // spectral codebook
            (1, 5),   // section length
            (1, 1),   // scalefactor delta marker for the test decoder
            (0, 1),   // pulse
            (0, 1),   // tns
            (0, 1),   // gain control
        ]);
        let mut reader = BitReader::new(&bytes);
        let mut decoder = OneBitScaleFactorDecoder::default();

        let stream = IndividualChannelStream::read(&mut reader, None, &mut decoder).unwrap();

        assert_eq!(decoder.calls, 1);
        assert!(stream.has_spectral_data());
        assert_eq!(
            stream.scale_factors.value(0, 0),
            Some(ScaleFactorValue::Spectral(101))
        );
        assert_eq!(stream.pulse_data, None);
        assert_eq!(stream.tns_data, None);
    }

    #[test]
    fn zero_spectral_compatibility_path_still_rejects_pulse_synthesis() {
        let bytes = build_bits(&[
            (128, 8), // global gain
            (0, 1),   // reserved
            (0, 2),   // only long
            (0, 1),   // sine
            (1, 6),   // max_sfb
            (0, 1),   // predictor
            (0, 4),   // zero codebook
            (1, 5),   // section length
            (1, 1),   // pulse_data_present
            (0, 2),   // one pulse
            (0, 6),   // pulse_start_sfb
            (0, 5),   // pulse offset
            (0, 4),   // pulse amp
            (0, 1),   // tns_data_present
            (0, 1),   // gain_control_data_present
        ]);
        let mut reader = BitReader::new(&bytes);

        assert_eq!(
            read_zero_spectral_channel(&mut reader, None).unwrap_err(),
            AacLcError::NotImplemented("pulse tool synthesis")
        );
    }

    #[test]
    fn channel_stream_dispatches_long_spectral_decode() {
        let stream = nonzero_stream();
        let mut reader = BitReader::new(&[]);
        let mut coefficients = SpectralCoefficients::new(2);
        let mut decoder = FillSpectralDecoder::default();
        let layout = BandLayout::new(&[0, 2]);

        let spectrum = stream
            .decode_long_spectral(&mut reader, &mut coefficients, layout, &mut decoder)
            .unwrap();

        assert_eq!(decoder.calls, 1);
        assert_eq!(decoder.codebook, Some(1));
        assert!(spectrum[0] > 0.0);
        assert!(spectrum[1] < 0.0);
        assert_eq!(&coefficients.quantized()[..2], &[1, -1]);
    }

    #[test]
    fn channel_stream_applies_long_pulse_data() {
        let mut stream = nonzero_stream();
        stream.pulse_data = Some(PulseData {
            pulse_start_sfb: 0,
            count: 1,
            pulses: [
                Pulse { offset: 0, amp: 2 },
                Pulse::default(),
                Pulse::default(),
                Pulse::default(),
            ],
        });
        let mut reader = BitReader::new(&[]);
        let mut coefficients = SpectralCoefficients::new(2);
        let mut decoder = FillSpectralDecoder::default();
        let layout = BandLayout::new(&[0, 2]);

        stream
            .decode_long_spectral(&mut reader, &mut coefficients, layout, &mut decoder)
            .unwrap();

        assert_eq!(decoder.calls, 1);
        assert_eq!(&coefficients.quantized()[..2], &[3, -1]);
    }

    #[test]
    fn channel_stream_dispatches_short_spectral_decode() {
        let bytes = build_bits(&[
            (100, 8),        // global gain
            (0, 1),          // reserved
            (2, 2),          // eight short
            (0, 1),          // sine
            (1, 4),          // max_sfb
            (0b111_1111, 7), // one group covering all eight windows
            (1, 4),          // spectral codebook
            (1, 3),          // section length
            (0, 1),          // scalefactor delta marker
            (0, 1),          // pulse
            (0, 1),          // tns
            (0, 1),          // gain control
        ]);
        let mut stream_reader = BitReader::new(&bytes);
        let mut scale_decoder = OneBitScaleFactorDecoder::default();
        let stream =
            IndividualChannelStream::read(&mut stream_reader, None, &mut scale_decoder).unwrap();
        let mut spectral_reader = BitReader::new(&[]);
        let mut coefficients = SpectralCoefficients::new(1024);
        let mut decoder = FillSpectralDecoder::default();

        stream
            .decode_spectral(
                &mut spectral_reader,
                &mut coefficients,
                BandLayout::new(&[0, 4]),
                &mut decoder,
            )
            .unwrap();

        assert_eq!(decoder.calls, 8);
        assert_eq!(decoder.codebook, Some(1));
        assert_eq!(&coefficients.quantized()[0..4], &[1, -1, 1, -1]);
        assert_eq!(&coefficients.quantized()[128..132], &[1, -1, 1, -1]);
    }

    #[test]
    fn channel_stream_rejects_real_tns_filtering_before_spectral_decode() {
        let mut stream = nonzero_stream();
        stream.tns_data = Some(TnsData {
            window_count: 1,
            windows: [Default::default(); 8],
        });
        let mut reader = BitReader::new(&[]);
        let mut coefficients = SpectralCoefficients::new(2);
        let mut decoder = FillSpectralDecoder::default();
        let layout = BandLayout::new(&[0, 2]);

        assert_eq!(
            stream
                .decode_long_spectral(&mut reader, &mut coefficients, layout, &mut decoder)
                .unwrap_err(),
            AacLcError::NotImplemented("TNS filtering")
        );
        assert_eq!(decoder.calls, 0);
    }

    fn nonzero_stream() -> IndividualChannelStream {
        let bytes = build_bits(&[
            (100, 8), // global gain
            (0, 1),   // reserved
            (0, 2),   // only long
            (0, 1),   // sine
            (1, 6),   // max_sfb
            (0, 1),   // predictor
            (1, 4),   // spectral codebook
            (1, 5),   // section length
            (0, 1),   // scalefactor delta marker
            (0, 1),   // pulse
            (0, 1),   // tns
            (0, 1),   // gain control
        ]);
        let mut reader = BitReader::new(&bytes);
        let mut decoder = OneBitScaleFactorDecoder::default();

        IndividualChannelStream::read(&mut reader, None, &mut decoder).unwrap()
    }

    #[derive(Default)]
    struct OneBitScaleFactorDecoder {
        calls: usize,
    }

    impl ScaleFactorDecoder for OneBitScaleFactorDecoder {
        fn read_delta(&mut self, reader: &mut BitReader<'_>) -> Result<i16> {
            self.calls += 1;
            Ok(i16::from(reader.read_bool()?))
        }
    }

    #[derive(Default)]
    struct FillSpectralDecoder {
        calls: usize,
        codebook: Option<u8>,
    }

    impl SpectralDecoder for FillSpectralDecoder {
        fn read_quantized(
            &mut self,
            _reader: &mut BitReader<'_>,
            codebook: u8,
            out: &mut [i32],
        ) -> Result<()> {
            self.calls += 1;
            self.codebook = Some(codebook);
            for (idx, value) in out.iter_mut().enumerate() {
                *value = if idx % 2 == 0 { 1 } else { -1 };
            }
            Ok(())
        }
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
