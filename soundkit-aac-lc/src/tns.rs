use crate::bitreader::BitReader;
use crate::config::SamplingFrequency;
use crate::error::{AacLcError, Result};
use crate::ics::{IcsInfo, WindowSequence};
use crate::spectral::BandLayout;

pub const MAX_TNS_FILTERS: usize = 4;
pub const MAX_TNS_ORDER: usize = 20;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TnsFilter {
    pub length: u8,
    pub order: u8,
    pub direction: bool,
    pub coef_compress: bool,
    pub coef_bits: u8,
    pub coeffs: [i8; MAX_TNS_ORDER],
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TnsWindow {
    pub filter_count: u8,
    pub coef_res: Option<bool>,
    pub filters: [TnsFilter; MAX_TNS_FILTERS],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TnsData {
    pub window_count: u8,
    pub windows: [TnsWindow; 8],
}

impl TnsData {
    pub fn read(reader: &mut BitReader<'_>, info: &IcsInfo) -> Result<Self> {
        let short = info.window_sequence.is_short();
        let window_count = info.num_windows;
        let filter_count_bits = if short { 1 } else { 2 };
        let length_bits = if short { 4 } else { 6 };
        let order_bits = if short { 3 } else { 5 };
        let mut windows = [TnsWindow::default(); 8];

        for window in windows.iter_mut().take(window_count as usize) {
            window.filter_count = reader.read_u8(filter_count_bits)?;
            if window.filter_count as usize > MAX_TNS_FILTERS {
                return Err(AacLcError::InvalidBitstream("too many TNS filters"));
            }
            if window.filter_count == 0 {
                continue;
            }

            let coef_res = reader.read_bool()?;
            window.coef_res = Some(coef_res);

            for filter in window.filters.iter_mut().take(window.filter_count as usize) {
                filter.length = reader.read_u8(length_bits)?;
                filter.order = reader.read_u8(order_bits)?;
                if filter.order as usize > MAX_TNS_ORDER {
                    return Err(AacLcError::UnsupportedFeature("TNS order above 20"));
                }
                if filter.order == 0 {
                    continue;
                }

                filter.direction = reader.read_bool()?;
                filter.coef_compress = reader.read_bool()?;
                let base_bits = if coef_res { 4 } else { 3 };
                filter.coef_bits = base_bits - u8::from(filter.coef_compress);

                for coeff in filter.coeffs.iter_mut().take(filter.order as usize) {
                    *coeff = read_signed(reader, filter.coef_bits)?;
                }
            }
        }

        Ok(Self {
            window_count,
            windows,
        })
    }
}

pub fn lc_tns_max_bands(
    sampling_frequency: SamplingFrequency,
    window_sequence: WindowSequence,
) -> Result<usize> {
    let index = sampling_frequency
        .index
        .ok_or(AacLcError::UnsupportedFeature(
            "explicit sample-rate TNS max bands",
        ))? as usize;
    let bands = if window_sequence.is_short() {
        TNS_MAX_BANDS_128
    } else {
        TNS_MAX_BANDS_1024
    };

    bands
        .get(index)
        .map(|value| *value as usize)
        .ok_or(AacLcError::UnsupportedSamplingFrequencyIndex(index as u8))
}

pub fn apply_tns(
    tns: &TnsData,
    info: &IcsInfo,
    layout: BandLayout<'_>,
    max_tns_bands: usize,
    coefficients: &mut [f32],
) -> Result<()> {
    if tns.window_count != info.num_windows {
        return Err(AacLcError::InvalidBitstream(
            "TNS window count does not match ICS",
        ));
    }

    let window_len = if info.window_sequence.is_short() {
        128
    } else {
        1024
    };
    if coefficients.len() < window_len * info.num_windows as usize {
        return Err(AacLcError::InvalidConfig(
            "TNS coefficient buffer is too small",
        ));
    }

    let num_swb = layout.band_count();
    let max_tns_bands = max_tns_bands.min(info.max_sfb as usize).min(num_swb);

    for window_index in 0..tns.window_count as usize {
        let window = tns.windows[window_index];
        let coef_res_bits = if window.coef_res.unwrap_or(false) {
            4
        } else {
            3
        };
        let mut bottom = num_swb;

        for filter in window.filters.iter().take(window.filter_count as usize) {
            let top = bottom;
            bottom = top.saturating_sub(filter.length as usize);
            let order = filter.order as usize;
            if order == 0 {
                continue;
            }

            let start_band = bottom.min(max_tns_bands);
            let end_band = top.min(max_tns_bands);
            let offsets = layout.offsets();
            let start = *offsets.get(start_band).ok_or(AacLcError::InvalidConfig(
                "missing TNS start scale-factor band offset",
            ))?;
            let end = *offsets.get(end_band).ok_or(AacLcError::InvalidConfig(
                "missing TNS end scale-factor band offset",
            ))?;
            if end <= start {
                continue;
            }

            let mut lpc = [0.0f32; MAX_TNS_ORDER];
            tns_lpc_coefficients(filter, coef_res_bits, &mut lpc)?;
            apply_tns_filter(
                coefficients,
                window_index * window_len,
                start,
                end,
                filter.direction,
                &lpc[..order],
            )?;
        }
    }

    Ok(())
}

fn tns_lpc_coefficients(
    filter: &TnsFilter,
    coef_res_bits: u8,
    lpc: &mut [f32; MAX_TNS_ORDER],
) -> Result<()> {
    let order = filter.order as usize;
    if order > MAX_TNS_ORDER {
        return Err(AacLcError::UnsupportedFeature("TNS order above 20"));
    }

    let mut previous = [0.0f32; MAX_TNS_ORDER];
    for index in 0..order {
        let reflection = -inverse_quantized_tns_coefficient(
            filter.coeffs[index],
            filter.coef_bits,
            coef_res_bits,
        )?;
        lpc[index] = reflection;

        for inner in 0..((index + 1) >> 1) {
            let forward = previous[inner];
            let backward = previous[index - 1 - inner];
            lpc[inner] = forward + reflection * backward;
            lpc[index - 1 - inner] = backward + reflection * forward;
        }

        previous[..=index].copy_from_slice(&lpc[..=index]);
    }

    Ok(())
}

fn inverse_quantized_tns_coefficient(encoded: i8, coef_bits: u8, coef_res_bits: u8) -> Result<f32> {
    if coef_bits == 0 || coef_bits > 4 || coef_res_bits < 3 || coef_res_bits > 4 {
        return Err(AacLcError::InvalidBitstream(
            "invalid TNS coefficient resolution",
        ));
    }

    let raw_mask = (1i32 << coef_bits) - 1;
    let raw = encoded as i32 & raw_mask;
    let sign_boundary = 1i32 << (coef_bits - 1);
    let signed = if raw < sign_boundary {
        -raw
    } else {
        (1i32 << coef_bits) - raw
    };

    if signed == 0 {
        return Ok(0.0);
    }

    let divisor = if signed < 0 {
        (1i32 << coef_res_bits) - 1
    } else {
        (1i32 << coef_res_bits) + 1
    } as f32;

    Ok((signed as f32 * std::f32::consts::PI / divisor).sin())
}

fn apply_tns_filter(
    coefficients: &mut [f32],
    window_offset: usize,
    start: usize,
    end: usize,
    reverse: bool,
    lpc: &[f32],
) -> Result<()> {
    let window_start = window_offset + start;
    let window_end = window_offset + end;
    if window_end > coefficients.len() {
        return Err(AacLcError::InvalidConfig(
            "TNS filter range exceeds coefficient buffer",
        ));
    }

    if reverse {
        for position in (window_start..window_end).rev() {
            let processed = window_end - 1 - position;
            let max_order = processed.min(lpc.len());
            let mut value = coefficients[position];
            for order in 1..=max_order {
                value -= coefficients[position + order] * lpc[order - 1];
            }
            coefficients[position] = value;
        }
    } else {
        for position in window_start..window_end {
            let processed = position - window_start;
            let max_order = processed.min(lpc.len());
            let mut value = coefficients[position];
            for order in 1..=max_order {
                value -= coefficients[position - order] * lpc[order - 1];
            }
            coefficients[position] = value;
        }
    }

    Ok(())
}

fn read_signed(reader: &mut BitReader<'_>, bits: u8) -> Result<i8> {
    let raw = reader.read_u8(bits)?;
    let shift = 8 - bits;
    Ok(((raw << shift) as i8) >> shift)
}

const TNS_MAX_BANDS_1024: &[u8; 13] = &[31, 31, 34, 40, 42, 51, 46, 46, 42, 42, 42, 39, 39];
const TNS_MAX_BANDS_128: &[u8; 13] = &[9, 9, 10, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ics::{IcsInfo, WindowSequence, WindowShape};

    #[test]
    fn parses_empty_long_tns() {
        let info = long_info();
        let bytes = build_bits(&[(0, 2)]);
        let mut reader = BitReader::new(&bytes);
        let tns = TnsData::read(&mut reader, &info).unwrap();

        assert_eq!(tns.window_count, 1);
        assert_eq!(tns.windows[0].filter_count, 0);
    }

    #[test]
    fn parses_long_tns_filter_coefficients() {
        let info = long_info();
        let bytes = build_bits(&[
            (1, 2),      // n_filt
            (1, 1),      // coef_res = 4-bit base
            (12, 6),     // length
            (2, 5),      // order
            (1, 1),      // direction
            (0, 1),      // coef_compress
            (0b0111, 4), // 7
            (0b1111, 4), // -1
        ]);
        let mut reader = BitReader::new(&bytes);
        let tns = TnsData::read(&mut reader, &info).unwrap();
        let filter = tns.windows[0].filters[0];

        assert_eq!(filter.length, 12);
        assert_eq!(filter.order, 2);
        assert!(filter.direction);
        assert_eq!(filter.coef_bits, 4);
        assert_eq!(&filter.coeffs[..2], &[7, -1]);
    }

    #[test]
    fn parses_empty_short_tns_windows() {
        let info = IcsInfo {
            window_sequence: WindowSequence::EightShort,
            window_shape: WindowShape::Sine,
            max_sfb: 4,
            num_windows: 8,
            num_window_groups: 1,
            window_group_len: [8, 0, 0, 0, 0, 0, 0, 0],
        };
        let bytes = build_bits(&[(0, 1); 8]);
        let mut reader = BitReader::new(&bytes);
        let tns = TnsData::read(&mut reader, &info).unwrap();

        assert_eq!(tns.window_count, 8);
        assert!(tns.windows.iter().all(|window| window.filter_count == 0));
    }

    #[test]
    fn maps_lc_tns_max_bands_for_common_sample_rates() {
        assert_eq!(
            lc_tns_max_bands(
                SamplingFrequency::from_index(4).unwrap(),
                WindowSequence::OnlyLong
            )
            .unwrap(),
            42
        );
        assert_eq!(
            lc_tns_max_bands(
                SamplingFrequency::from_index(4).unwrap(),
                WindowSequence::EightShort
            )
            .unwrap(),
            14
        );
    }

    #[test]
    fn inverse_quantizes_tns_coefficients_from_transmitted_bits() {
        assert_close(
            inverse_quantized_tns_coefficient(1, 4, 4).unwrap(),
            -0.207_911_7,
        );
        assert_close(
            inverse_quantized_tns_coefficient(-8, 4, 4).unwrap(),
            0.995_734_16,
        );
        assert_close(
            inverse_quantized_tns_coefficient(-1, 4, 4).unwrap(),
            0.183_749_51,
        );
        assert_close(
            inverse_quantized_tns_coefficient(-4, 3, 4).unwrap(),
            0.673_695_62,
        );
    }

    #[test]
    fn applies_forward_tns_filter_to_long_coefficients() {
        let mut tns = TnsData {
            window_count: 1,
            windows: [TnsWindow::default(); 8],
        };
        tns.windows[0].filter_count = 1;
        tns.windows[0].coef_res = Some(true);
        tns.windows[0].filters[0] = TnsFilter {
            length: 2,
            order: 1,
            direction: false,
            coef_compress: false,
            coef_bits: 4,
            coeffs: {
                let mut coeffs = [0; MAX_TNS_ORDER];
                coeffs[0] = 1;
                coeffs
            },
        };
        let info = long_info();
        let offsets = [0, 4, 8];
        let layout = BandLayout::new(&offsets);
        let mut coefficients = vec![0.0f32; 1024];
        coefficients[0] = 1.0;
        coefficients[1] = 2.0;

        apply_tns(&tns, &info, layout, 2, &mut coefficients).unwrap();

        assert_close(coefficients[0], 1.0);
        assert_close(coefficients[1], 2.0 - 0.207_911_7);
    }

    fn long_info() -> IcsInfo {
        IcsInfo {
            window_sequence: WindowSequence::OnlyLong,
            window_shape: WindowShape::Sine,
            max_sfb: 1,
            num_windows: 1,
            num_window_groups: 1,
            window_group_len: [1, 0, 0, 0, 0, 0, 0, 0],
        }
    }

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 1.0e-6,
            "actual={actual} expected={expected}"
        );
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
