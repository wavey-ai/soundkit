use crate::bitreader::BitReader;
use crate::error::{AacLcError, Result};

pub const SAMPLES_PER_AAC_LC_FRAME: usize = 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioObjectType {
    AacMain,
    AacLc,
    AacSsr,
    AacLtp,
    Sbr,
    AacScalable,
    TwinVq,
    Celp,
    Hxvc,
    Ttsi,
    MainSynthetic,
    WavetableSynthesis,
    GeneralMidi,
    AlgorithmicSynthesis,
    ErAacLc,
    ErAacLtp,
    ErAacScalable,
    ErTwinVq,
    ErBsac,
    ErAacLd,
    ErCelp,
    ErHvxc,
    ErHiln,
    ErParametric,
    Ssc,
    ParametricStereo,
    MpegSurround,
    Escape(u8),
    Unknown(u8),
}

impl AudioObjectType {
    pub fn from_raw(value: u8) -> Result<Self> {
        let object_type = match value {
            0 => return Err(AacLcError::InvalidAudioObjectType(value)),
            1 => Self::AacMain,
            2 => Self::AacLc,
            3 => Self::AacSsr,
            4 => Self::AacLtp,
            5 => Self::Sbr,
            6 => Self::AacScalable,
            7 => Self::TwinVq,
            8 => Self::Celp,
            9 => Self::Hxvc,
            12 => Self::Ttsi,
            13 => Self::MainSynthetic,
            14 => Self::WavetableSynthesis,
            15 => Self::GeneralMidi,
            16 => Self::AlgorithmicSynthesis,
            17 => Self::ErAacLc,
            19 => Self::ErAacLtp,
            20 => Self::ErAacScalable,
            21 => Self::ErTwinVq,
            22 => Self::ErBsac,
            23 => Self::ErAacLd,
            24 => Self::ErCelp,
            25 => Self::ErHvxc,
            26 => Self::ErHiln,
            27 => Self::ErParametric,
            28 => Self::Ssc,
            29 => Self::ParametricStereo,
            30 => Self::MpegSurround,
            31 => Self::Escape(value),
            value => Self::Unknown(value),
        };
        Ok(object_type)
    }

    pub const fn raw(self) -> u8 {
        match self {
            AudioObjectType::AacMain => 1,
            AudioObjectType::AacLc => 2,
            AudioObjectType::AacSsr => 3,
            AudioObjectType::AacLtp => 4,
            AudioObjectType::Sbr => 5,
            AudioObjectType::AacScalable => 6,
            AudioObjectType::TwinVq => 7,
            AudioObjectType::Celp => 8,
            AudioObjectType::Hxvc => 9,
            AudioObjectType::Ttsi => 12,
            AudioObjectType::MainSynthetic => 13,
            AudioObjectType::WavetableSynthesis => 14,
            AudioObjectType::GeneralMidi => 15,
            AudioObjectType::AlgorithmicSynthesis => 16,
            AudioObjectType::ErAacLc => 17,
            AudioObjectType::ErAacLtp => 19,
            AudioObjectType::ErAacScalable => 20,
            AudioObjectType::ErTwinVq => 21,
            AudioObjectType::ErBsac => 22,
            AudioObjectType::ErAacLd => 23,
            AudioObjectType::ErCelp => 24,
            AudioObjectType::ErHvxc => 25,
            AudioObjectType::ErHiln => 26,
            AudioObjectType::ErParametric => 27,
            AudioObjectType::Ssc => 28,
            AudioObjectType::ParametricStereo => 29,
            AudioObjectType::MpegSurround => 30,
            AudioObjectType::Escape(value) | AudioObjectType::Unknown(value) => value,
        }
    }

    pub const fn is_aac_lc(self) -> bool {
        matches!(self, AudioObjectType::AacLc)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SamplingFrequency {
    pub index: Option<u8>,
    pub hz: u32,
}

impl SamplingFrequency {
    pub fn from_index(index: u8) -> Result<Self> {
        let hz = match index {
            0 => 96_000,
            1 => 88_200,
            2 => 64_000,
            3 => 48_000,
            4 => 44_100,
            5 => 32_000,
            6 => 24_000,
            7 => 22_050,
            8 => 16_000,
            9 => 12_000,
            10 => 11_025,
            11 => 8_000,
            12 => 7_350,
            value => return Err(AacLcError::UnsupportedSamplingFrequencyIndex(value)),
        };
        Ok(Self {
            index: Some(index),
            hz,
        })
    }

    pub const fn explicit(hz: u32) -> Self {
        Self { index: None, hz }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelConfig {
    ProgramConfigElement,
    Mono,
    Stereo,
    Unsupported(u8),
}

impl ChannelConfig {
    pub const fn from_raw(value: u8) -> Self {
        match value {
            0 => Self::ProgramConfigElement,
            1 => Self::Mono,
            2 => Self::Stereo,
            value => Self::Unsupported(value),
        }
    }

    pub const fn raw(self) -> u8 {
        match self {
            ChannelConfig::ProgramConfigElement => 0,
            ChannelConfig::Mono => 1,
            ChannelConfig::Stereo => 2,
            ChannelConfig::Unsupported(value) => value,
        }
    }

    pub const fn channels(self) -> Option<usize> {
        match self {
            ChannelConfig::Mono => Some(1),
            ChannelConfig::Stereo => Some(2),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioSpecificConfig {
    pub object_type: AudioObjectType,
    pub sampling_frequency: SamplingFrequency,
    pub channel_config: ChannelConfig,
    pub frame_length: usize,
    pub depends_on_core_coder: bool,
    pub extension_flag: bool,
    pub sbr_present: bool,
    pub ps_present: bool,
    pub extension_sampling_frequency: Option<SamplingFrequency>,
}

impl AudioSpecificConfig {
    pub fn parse(data: &[u8]) -> Result<Self> {
        let mut reader = BitReader::new(data);
        Self::read(&mut reader)
    }

    pub fn read(reader: &mut BitReader<'_>) -> Result<Self> {
        let mut object_type = read_audio_object_type(reader)?;
        let sampling_frequency = read_sampling_frequency(reader)?;
        let channel_config = ChannelConfig::from_raw(reader.read_u8(4)?);

        let mut sbr_present = false;
        let mut ps_present = false;
        let mut extension_sampling_frequency = None;

        if matches!(
            object_type,
            AudioObjectType::Sbr | AudioObjectType::ParametricStereo
        ) {
            sbr_present = true;
            ps_present = matches!(object_type, AudioObjectType::ParametricStereo);
            extension_sampling_frequency = Some(read_sampling_frequency(reader)?);
            object_type = read_audio_object_type(reader)?;
        }

        let (frame_length, depends_on_core_coder, extension_flag) =
            read_ga_specific_config(reader, object_type)?;

        Ok(Self {
            object_type,
            sampling_frequency,
            channel_config,
            frame_length,
            depends_on_core_coder,
            extension_flag,
            sbr_present,
            ps_present,
            extension_sampling_frequency,
        })
    }

    pub fn validate_aac_lc_packet_path(&self) -> Result<()> {
        if !self.object_type.is_aac_lc() {
            return Err(AacLcError::UnsupportedAudioObjectType(
                self.object_type.raw(),
            ));
        }
        if self.ps_present {
            return Err(AacLcError::UnsupportedFeature("parametric stereo"));
        }
        if self.sbr_present {
            return Err(AacLcError::UnsupportedFeature("SBR/HE-AAC"));
        }
        if self.frame_length != SAMPLES_PER_AAC_LC_FRAME {
            return Err(AacLcError::UnsupportedFeature("960-sample AAC frames"));
        }
        match self.channel_config {
            ChannelConfig::Mono | ChannelConfig::Stereo => Ok(()),
            ChannelConfig::ProgramConfigElement => Err(AacLcError::UnsupportedFeature(
                "program config element channels",
            )),
            ChannelConfig::Unsupported(value) => Err(AacLcError::UnsupportedChannelConfig(value)),
        }
    }

    pub const fn sample_rate(&self) -> u32 {
        self.sampling_frequency.hz
    }

    pub const fn channels(&self) -> Option<usize> {
        self.channel_config.channels()
    }
}

fn read_audio_object_type(reader: &mut BitReader<'_>) -> Result<AudioObjectType> {
    let value = reader.read_u8(5)?;
    if value == 31 {
        let escaped = 32 + reader.read_u8(6)?;
        AudioObjectType::from_raw(escaped)
    } else {
        AudioObjectType::from_raw(value)
    }
}

fn read_sampling_frequency(reader: &mut BitReader<'_>) -> Result<SamplingFrequency> {
    let index = reader.read_u8(4)?;
    if index == 15 {
        Ok(SamplingFrequency::explicit(reader.read_u32(24)?))
    } else {
        SamplingFrequency::from_index(index)
    }
}

fn read_ga_specific_config(
    reader: &mut BitReader<'_>,
    object_type: AudioObjectType,
) -> Result<(usize, bool, bool)> {
    match object_type {
        AudioObjectType::AacMain
        | AudioObjectType::AacLc
        | AudioObjectType::AacSsr
        | AudioObjectType::AacLtp
        | AudioObjectType::AacScalable
        | AudioObjectType::ErAacLc
        | AudioObjectType::ErAacLtp
        | AudioObjectType::ErAacScalable => {
            let frame_length_flag = reader.read_bool()?;
            let depends_on_core_coder = reader.read_bool()?;
            if depends_on_core_coder {
                let _core_coder_delay = reader.read_u16(14)?;
            }
            let extension_flag = reader.read_bool()?;
            let frame_length = if frame_length_flag {
                960
            } else {
                SAMPLES_PER_AAC_LC_FRAME
            };
            Ok((frame_length, depends_on_core_coder, extension_flag))
        }
        _ => Err(AacLcError::UnsupportedAudioObjectType(object_type.raw())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_aac_lc_44100_stereo_config() {
        let config = AudioSpecificConfig::parse(&[0x12, 0x10]).unwrap();

        assert_eq!(config.object_type, AudioObjectType::AacLc);
        assert_eq!(config.sample_rate(), 44_100);
        assert_eq!(config.sampling_frequency.index, Some(4));
        assert_eq!(config.channel_config, ChannelConfig::Stereo);
        assert_eq!(config.channels(), Some(2));
        assert_eq!(config.frame_length, 1024);
        assert!(!config.sbr_present);
        config.validate_aac_lc_packet_path().unwrap();
    }

    #[test]
    fn parses_aac_lc_48000_mono_config() {
        let config = AudioSpecificConfig::parse(&[0x11, 0x88]).unwrap();

        assert_eq!(config.object_type, AudioObjectType::AacLc);
        assert_eq!(config.sample_rate(), 48_000);
        assert_eq!(config.channel_config, ChannelConfig::Mono);
        assert_eq!(config.channels(), Some(1));
        config.validate_aac_lc_packet_path().unwrap();
    }

    #[test]
    fn parses_explicit_sample_rate() {
        let bytes = build_config_bits(&[
            (2, 5),  // AAC-LC
            (15, 4), // explicit sample rate
            (44_100, 24),
            (2, 4), // stereo
            (0, 1), // frameLengthFlag
            (0, 1), // dependsOnCoreCoder
            (0, 1), // extensionFlag
        ]);
        let config = AudioSpecificConfig::parse(&bytes).unwrap();

        assert_eq!(config.sample_rate(), 44_100);
        assert_eq!(config.sampling_frequency.index, None);
        assert_eq!(config.channel_config, ChannelConfig::Stereo);
    }

    #[test]
    fn rejects_sbr_as_unsupported_for_packet_path() {
        let bytes = build_config_bits(&[
            (5, 5), // SBR
            (4, 4), // 44.1 kHz
            (2, 4), // stereo
            (4, 4), // extension 44.1 kHz
            (2, 5), // AAC-LC
            (0, 1), // frameLengthFlag
            (0, 1), // dependsOnCoreCoder
            (0, 1), // extensionFlag
        ]);
        let config = AudioSpecificConfig::parse(&bytes).unwrap();

        assert_eq!(config.object_type, AudioObjectType::AacLc);
        assert!(config.sbr_present);
        assert_eq!(
            config.validate_aac_lc_packet_path().unwrap_err(),
            AacLcError::UnsupportedFeature("SBR/HE-AAC")
        );
    }

    #[test]
    fn rejects_parametric_stereo_as_unsupported_for_packet_path() {
        let bytes = build_config_bits(&[
            (29, 5), // Parametric Stereo
            (4, 4),  // 44.1 kHz
            (2, 4),  // stereo
            (4, 4),  // extension 44.1 kHz
            (2, 5),  // AAC-LC
            (0, 1),  // frameLengthFlag
            (0, 1),  // dependsOnCoreCoder
            (0, 1),  // extensionFlag
        ]);
        let config = AudioSpecificConfig::parse(&bytes).unwrap();

        assert_eq!(config.object_type, AudioObjectType::AacLc);
        assert!(config.sbr_present);
        assert!(config.ps_present);
        assert_eq!(
            config.validate_aac_lc_packet_path().unwrap_err(),
            AacLcError::UnsupportedFeature("parametric stereo")
        );
    }

    #[test]
    fn rejects_channel_layouts_beyond_stereo() {
        let bytes = build_config_bits(&[
            (2, 5), // AAC-LC
            (4, 4), // 44.1 kHz
            (6, 4), // 5.1 channel config
            (0, 1), // frameLengthFlag
            (0, 1), // dependsOnCoreCoder
            (0, 1), // extensionFlag
        ]);
        let config = AudioSpecificConfig::parse(&bytes).unwrap();

        assert_eq!(config.channel_config, ChannelConfig::Unsupported(6));
        assert_eq!(
            config.validate_aac_lc_packet_path().unwrap_err(),
            AacLcError::UnsupportedChannelConfig(6)
        );
    }

    #[test]
    fn rejects_program_config_element_for_initial_decoder_scope() {
        let config = AudioSpecificConfig::parse(&[0x12, 0x00]).unwrap();

        assert_eq!(config.channel_config, ChannelConfig::ProgramConfigElement);
        assert_eq!(
            config.validate_aac_lc_packet_path().unwrap_err(),
            AacLcError::UnsupportedFeature("program config element channels")
        );
    }

    fn build_config_bits(fields: &[(u32, u8)]) -> Vec<u8> {
        let bit_count: usize = fields.iter().map(|(_, width)| *width as usize).sum();
        let mut out = vec![0u8; (bit_count + 7) / 8];
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
