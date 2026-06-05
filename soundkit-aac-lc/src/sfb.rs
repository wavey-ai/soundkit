use crate::config::SamplingFrequency;
use crate::error::{AacLcError, Result};
use crate::spectral::BandLayout;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleFactorBandWindow {
    Long1024,
    Short128,
}

#[derive(Debug, Clone, Copy)]
pub struct ScaleFactorBandOffsets {
    pub window: ScaleFactorBandWindow,
    pub sampling_frequency_index: u8,
    pub offsets: &'static [usize],
}

impl ScaleFactorBandOffsets {
    pub fn layout(self) -> BandLayout<'static> {
        BandLayout::new(self.offsets)
    }

    pub fn band_count(self) -> usize {
        self.offsets.len().saturating_sub(1)
    }
}

pub fn long_window_band_offsets(
    sampling_frequency: SamplingFrequency,
) -> Result<ScaleFactorBandOffsets> {
    indexed_offsets(sampling_frequency, ScaleFactorBandWindow::Long1024)
}

pub fn short_window_band_offsets(
    sampling_frequency: SamplingFrequency,
) -> Result<ScaleFactorBandOffsets> {
    indexed_offsets(sampling_frequency, ScaleFactorBandWindow::Short128)
}

pub fn long_window_band_layout(
    sampling_frequency: SamplingFrequency,
) -> Result<BandLayout<'static>> {
    Ok(long_window_band_offsets(sampling_frequency)?.layout())
}

pub fn short_window_band_layout(
    sampling_frequency: SamplingFrequency,
) -> Result<BandLayout<'static>> {
    Ok(short_window_band_offsets(sampling_frequency)?.layout())
}

fn indexed_offsets(
    sampling_frequency: SamplingFrequency,
    window: ScaleFactorBandWindow,
) -> Result<ScaleFactorBandOffsets> {
    let index = sampling_frequency
        .index
        .ok_or(AacLcError::UnsupportedFeature(
            "explicit sample-rate scalefactor bands",
        ))?;
    let offsets = match window {
        ScaleFactorBandWindow::Long1024 => long_offsets_for_index(index)?,
        ScaleFactorBandWindow::Short128 => short_offsets_for_index(index)?,
    };

    Ok(ScaleFactorBandOffsets {
        window,
        sampling_frequency_index: index,
        offsets,
    })
}

fn long_offsets_for_index(index: u8) -> Result<&'static [usize]> {
    match index {
        0 | 1 => Ok(&SWB_OFFSET_1024_96),
        2 => Ok(&SWB_OFFSET_1024_64),
        3 | 4 => Ok(&SWB_OFFSET_1024_48),
        5 => Ok(&SWB_OFFSET_1024_32),
        6 | 7 => Ok(&SWB_OFFSET_1024_24),
        8 | 9 | 10 => Ok(&SWB_OFFSET_1024_16),
        11 | 12 => Ok(&SWB_OFFSET_1024_8),
        value => Err(AacLcError::UnsupportedSamplingFrequencyIndex(value)),
    }
}

fn short_offsets_for_index(index: u8) -> Result<&'static [usize]> {
    match index {
        0 | 1 | 2 => Ok(&SWB_OFFSET_128_96),
        3 | 4 | 5 => Ok(&SWB_OFFSET_128_48),
        6 | 7 => Ok(&SWB_OFFSET_128_24),
        8 | 9 | 10 => Ok(&SWB_OFFSET_128_16),
        11 | 12 => Ok(&SWB_OFFSET_128_8),
        value => Err(AacLcError::UnsupportedSamplingFrequencyIndex(value)),
    }
}

const SWB_OFFSET_1024_96: [usize; 42] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 64, 72, 80, 88, 96, 108, 120, 132,
    144, 156, 172, 188, 212, 240, 276, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024,
];

const SWB_OFFSET_1024_64: [usize; 48] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 64, 72, 80, 88, 100, 112, 124, 140,
    156, 172, 192, 216, 240, 268, 304, 344, 384, 424, 464, 504, 544, 584, 624, 664, 704, 744, 784,
    824, 864, 904, 944, 984, 1024,
];

const SWB_OFFSET_1024_48: [usize; 50] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 80, 88, 96, 108, 120, 132, 144, 160,
    176, 196, 216, 240, 264, 292, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704,
    736, 768, 800, 832, 864, 896, 928, 1024,
];

const SWB_OFFSET_1024_32: [usize; 52] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 80, 88, 96, 108, 120, 132, 144, 160,
    176, 196, 216, 240, 264, 292, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704,
    736, 768, 800, 832, 864, 896, 928, 960, 992, 1024,
];

const SWB_OFFSET_1024_24: [usize; 48] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 52, 60, 68, 76, 84, 92, 100, 108, 116, 124, 136,
    148, 160, 172, 188, 204, 220, 240, 260, 284, 308, 336, 364, 396, 432, 468, 508, 552, 600, 652,
    704, 768, 832, 896, 960, 1024,
];

const SWB_OFFSET_1024_16: [usize; 44] = [
    0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 100, 112, 124, 136, 148, 160, 172, 184, 196, 212,
    228, 244, 260, 280, 300, 320, 344, 368, 396, 424, 456, 492, 532, 572, 616, 664, 716, 772, 832,
    896, 960, 1024,
];

const SWB_OFFSET_1024_8: [usize; 41] = [
    0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 172, 188, 204, 220, 236, 252, 268,
    288, 308, 328, 348, 372, 396, 420, 448, 476, 508, 544, 580, 620, 664, 712, 764, 820, 880, 944,
    1024,
];

const SWB_OFFSET_128_96: [usize; 13] = [0, 4, 8, 12, 16, 20, 24, 32, 40, 48, 64, 92, 128];

const SWB_OFFSET_128_48: [usize; 15] = [0, 4, 8, 12, 16, 20, 28, 36, 44, 56, 68, 80, 96, 112, 128];

const SWB_OFFSET_128_24: [usize; 16] = [
    0, 4, 8, 12, 16, 20, 24, 28, 36, 44, 52, 64, 76, 92, 108, 128,
];

const SWB_OFFSET_128_16: [usize; 16] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 60, 72, 88, 108, 128,
];

const SWB_OFFSET_128_8: [usize; 16] = [
    0, 4, 8, 12, 16, 20, 24, 28, 36, 44, 52, 60, 72, 88, 108, 128,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_apple_music_rates_to_1024_long_window_offsets() {
        for index in [3, 4] {
            let layout =
                long_window_band_layout(SamplingFrequency::from_index(index).unwrap()).unwrap();

            assert_eq!(layout.band_count(), 49);
            assert_eq!(&layout.offsets()[..4], &[0, 4, 8, 12]);
            assert_eq!(layout.offsets().last(), Some(&1024));
            assert_eq!(layout.band_range(48).unwrap(), 928..1024);
        }
    }

    #[test]
    fn maps_16000_hz_long_window_offsets() {
        let layout = long_window_band_layout(SamplingFrequency::from_index(8).unwrap()).unwrap();

        assert_eq!(layout.band_count(), 43);
        assert_eq!(layout.band_range(0).unwrap(), 0..8);
        assert_eq!(layout.offsets().last(), Some(&1024));
    }

    #[test]
    fn maps_7350_hz_to_8000_hz_long_window_offsets() {
        let layout = long_window_band_layout(SamplingFrequency::from_index(12).unwrap()).unwrap();

        assert_eq!(layout.band_count(), 40);
        assert_eq!(layout.band_range(39).unwrap(), 944..1024);
    }

    #[test]
    fn maps_short_window_offsets_for_common_rates() {
        let layout = short_window_band_layout(SamplingFrequency::from_index(4).unwrap()).unwrap();

        assert_eq!(layout.band_count(), 14);
        assert_eq!(layout.band_range(6).unwrap(), 28..36);
        assert_eq!(layout.offsets().last(), Some(&128));
    }

    #[test]
    fn rejects_explicit_sample_rate_without_table_index() {
        assert_eq!(
            long_window_band_layout(SamplingFrequency::explicit(44_100)).unwrap_err(),
            AacLcError::UnsupportedFeature("explicit sample-rate scalefactor bands")
        );
    }
}
