use crate::channel::MidSideMask;
use crate::error::{AacLcError, Result};
use crate::ics::{IcsInfo, WindowSequence};
use crate::scalefactor::ScaleFactorData;
use crate::section::{SectionCodebook, SectionData};
use crate::spectral::BandLayout;

pub fn apply_mid_side_long(
    mask: MidSideMask,
    max_sfb: u8,
    layout: BandLayout<'_>,
    left: &mut [f32],
    right: &mut [f32],
) -> Result<()> {
    if left.len() != right.len() {
        return Err(AacLcError::InvalidConfig(
            "mid/side channel buffers have different lengths",
        ));
    }

    match mask {
        MidSideMask::None => Ok(()),
        MidSideMask::All => apply_mid_side_bands(max_sfb, layout, left, right, |_, _| true),
        MidSideMask::Some {
            max_sfb: mask_max_sfb,
            num_window_groups,
            used,
        } => {
            if num_window_groups != 1 {
                return Err(AacLcError::NotImplemented(
                    "grouped mid/side stereo reconstruction",
                ));
            }
            if max_sfb > mask_max_sfb {
                return Err(AacLcError::InvalidConfig(
                    "mid/side mask does not cover requested scale-factor bands",
                ));
            }
            apply_mid_side_bands(max_sfb, layout, left, right, |_, sfb| used[0][sfb])
        }
    }
}

pub fn apply_mid_side_long_excluding_intensity(
    mask: MidSideMask,
    max_sfb: u8,
    layout: BandLayout<'_>,
    left_sections: &SectionData,
    right_sections: &SectionData,
    left: &mut [f32],
    right: &mut [f32],
) -> Result<()> {
    apply_mid_side_long_bands(mask, max_sfb, layout, left, right, |group, sfb| {
        mid_side_allowed(
            left_sections.codebook(group, sfb),
            right_sections.codebook(group, sfb),
        )
    })
}

pub fn apply_mid_side_short(
    mask: MidSideMask,
    info: &IcsInfo,
    layout: BandLayout<'_>,
    left: &mut [f32],
    right: &mut [f32],
) -> Result<()> {
    if info.window_sequence != WindowSequence::EightShort {
        return Err(AacLcError::InvalidConfig(
            "short mid/side stereo requires eight short windows",
        ));
    }
    if left.len() != right.len() {
        return Err(AacLcError::InvalidConfig(
            "mid/side channel buffers have different lengths",
        ));
    }

    match mask {
        MidSideMask::None => Ok(()),
        MidSideMask::All => apply_mid_side_short_bands(info, layout, left, right, |_, _| true),
        MidSideMask::Some {
            max_sfb,
            num_window_groups,
            used,
        } => {
            if info.max_sfb > max_sfb || info.num_window_groups > num_window_groups {
                return Err(AacLcError::InvalidConfig(
                    "mid/side mask does not cover requested short-window groups/bands",
                ));
            }
            apply_mid_side_short_bands(info, layout, left, right, |group, sfb| used[group][sfb])
        }
    }
}

pub fn apply_mid_side_short_excluding_intensity(
    mask: MidSideMask,
    info: &IcsInfo,
    layout: BandLayout<'_>,
    left_sections: &SectionData,
    right_sections: &SectionData,
    left: &mut [f32],
    right: &mut [f32],
) -> Result<()> {
    apply_mid_side_short_bands_with_filter(mask, info, layout, left, right, |group, sfb| {
        mid_side_allowed(
            left_sections.codebook(group, sfb),
            right_sections.codebook(group, sfb),
        )
    })
}

pub fn apply_intensity_stereo_long(
    mask: MidSideMask,
    max_sfb: u8,
    layout: BandLayout<'_>,
    right_sections: &SectionData,
    right_scale_factors: &ScaleFactorData,
    left: &[f32],
    right: &mut [f32],
) -> Result<()> {
    if left.len() != right.len() {
        return Err(AacLcError::InvalidConfig(
            "intensity stereo channel buffers have different lengths",
        ));
    }

    for sfb in 0..max_sfb as usize {
        let Some(codebook) = right_sections.codebook(0, sfb) else {
            return Err(AacLcError::InvalidBitstream(
                "missing right-channel section codebook for intensity stereo",
            ));
        };
        let Some(sign) = intensity_codebook_sign(codebook) else {
            continue;
        };
        let range = layout.band_range(sfb)?;
        if range.end > left.len() {
            return Err(AacLcError::InvalidConfig(
                "intensity scale-factor band exceeds channel buffer",
            ));
        }

        let scale = intensity_scale(right_scale_factors, 0, sfb)?;
        let sign = if mid_side_selected(mask, 0, sfb)? {
            -sign
        } else {
            sign
        };
        for index in range {
            right[index] = left[index] * scale * sign;
        }
    }

    Ok(())
}

pub fn apply_intensity_stereo_short(
    mask: MidSideMask,
    info: &IcsInfo,
    layout: BandLayout<'_>,
    right_sections: &SectionData,
    right_scale_factors: &ScaleFactorData,
    left: &[f32],
    right: &mut [f32],
) -> Result<()> {
    if info.window_sequence != WindowSequence::EightShort {
        return Err(AacLcError::InvalidConfig(
            "short intensity stereo requires eight short windows",
        ));
    }
    if left.len() != right.len() {
        return Err(AacLcError::InvalidConfig(
            "intensity stereo channel buffers have different lengths",
        ));
    }

    let mut window_start = 0usize;
    for group in 0..info.num_window_groups as usize {
        let group_len = info.window_group_len[group] as usize;
        if group_len == 0 {
            return Err(AacLcError::InvalidBitstream(
                "short-window group has zero length",
            ));
        }
        if window_start + group_len > 8 {
            return Err(AacLcError::InvalidBitstream(
                "short-window groups exceed eight windows",
            ));
        }

        for sfb in 0..info.max_sfb as usize {
            let Some(codebook) = right_sections.codebook(group, sfb) else {
                return Err(AacLcError::InvalidBitstream(
                    "missing right-channel section codebook for short intensity stereo",
                ));
            };
            let Some(sign) = intensity_codebook_sign(codebook) else {
                continue;
            };
            let range = layout.band_range(sfb)?;
            if range.end > SHORT_WINDOW_COEFFICIENTS {
                return Err(AacLcError::InvalidConfig(
                    "short intensity scale-factor band exceeds window length",
                ));
            }

            let scale = intensity_scale(right_scale_factors, group, sfb)?;
            let sign = if mid_side_selected(mask, group, sfb)? {
                -sign
            } else {
                sign
            };
            for window in window_start..window_start + group_len {
                let start = window * SHORT_WINDOW_COEFFICIENTS + range.start;
                let end = window * SHORT_WINDOW_COEFFICIENTS + range.end;
                if end > left.len() {
                    return Err(AacLcError::InvalidConfig(
                        "short intensity scale-factor band exceeds channel buffer",
                    ));
                }

                for index in start..end {
                    right[index] = left[index] * scale * sign;
                }
            }
        }

        window_start += group_len;
    }

    if window_start != 8 {
        return Err(AacLcError::InvalidBitstream(
            "short-window groups do not cover eight windows",
        ));
    }

    Ok(())
}

fn apply_mid_side_bands(
    max_sfb: u8,
    layout: BandLayout<'_>,
    left: &mut [f32],
    right: &mut [f32],
    mut should_apply: impl FnMut(usize, usize) -> bool,
) -> Result<()> {
    for sfb in 0..max_sfb as usize {
        let range = layout.band_range(sfb)?;
        if range.end > left.len() {
            return Err(AacLcError::InvalidConfig(
                "mid/side scale-factor band exceeds channel buffer",
            ));
        }
        if !should_apply(0, sfb) {
            continue;
        }

        for index in range {
            let mid = left[index];
            let side = right[index];
            left[index] = mid + side;
            right[index] = mid - side;
        }
    }

    Ok(())
}

fn apply_mid_side_long_bands(
    mask: MidSideMask,
    max_sfb: u8,
    layout: BandLayout<'_>,
    left: &mut [f32],
    right: &mut [f32],
    mut allow_band: impl FnMut(usize, usize) -> bool,
) -> Result<()> {
    match mask {
        MidSideMask::None => Ok(()),
        MidSideMask::All => apply_mid_side_bands(max_sfb, layout, left, right, |group, sfb| {
            allow_band(group, sfb)
        }),
        MidSideMask::Some {
            max_sfb: mask_max_sfb,
            num_window_groups,
            used,
        } => {
            if num_window_groups != 1 {
                return Err(AacLcError::NotImplemented(
                    "grouped mid/side stereo reconstruction",
                ));
            }
            if max_sfb > mask_max_sfb {
                return Err(AacLcError::InvalidConfig(
                    "mid/side mask does not cover requested scale-factor bands",
                ));
            }
            apply_mid_side_bands(max_sfb, layout, left, right, |group, sfb| {
                used[0][sfb] && allow_band(group, sfb)
            })
        }
    }
}

fn apply_mid_side_short_bands(
    info: &IcsInfo,
    layout: BandLayout<'_>,
    left: &mut [f32],
    right: &mut [f32],
    mut should_apply: impl FnMut(usize, usize) -> bool,
) -> Result<()> {
    let mut window_start = 0usize;

    for group in 0..info.num_window_groups as usize {
        let group_len = info.window_group_len[group] as usize;
        if group_len == 0 {
            return Err(AacLcError::InvalidBitstream(
                "short-window group has zero length",
            ));
        }
        if window_start + group_len > 8 {
            return Err(AacLcError::InvalidBitstream(
                "short-window groups exceed eight windows",
            ));
        }

        for sfb in 0..info.max_sfb as usize {
            let range = layout.band_range(sfb)?;
            if range.end > SHORT_WINDOW_COEFFICIENTS {
                return Err(AacLcError::InvalidConfig(
                    "short mid/side scale-factor band exceeds window length",
                ));
            }
            if !should_apply(group, sfb) {
                continue;
            }

            for window in window_start..window_start + group_len {
                let start = window * SHORT_WINDOW_COEFFICIENTS + range.start;
                let end = window * SHORT_WINDOW_COEFFICIENTS + range.end;
                if end > left.len() {
                    return Err(AacLcError::InvalidConfig(
                        "mid/side scale-factor band exceeds channel buffer",
                    ));
                }

                for index in start..end {
                    let mid = left[index];
                    let side = right[index];
                    left[index] = mid + side;
                    right[index] = mid - side;
                }
            }
        }

        window_start += group_len;
    }

    if window_start != 8 {
        return Err(AacLcError::InvalidBitstream(
            "short-window groups do not cover eight windows",
        ));
    }

    Ok(())
}

fn apply_mid_side_short_bands_with_filter(
    mask: MidSideMask,
    info: &IcsInfo,
    layout: BandLayout<'_>,
    left: &mut [f32],
    right: &mut [f32],
    mut allow_band: impl FnMut(usize, usize) -> bool,
) -> Result<()> {
    match mask {
        MidSideMask::None => Ok(()),
        MidSideMask::All => apply_mid_side_short_bands(info, layout, left, right, |group, sfb| {
            allow_band(group, sfb)
        }),
        MidSideMask::Some {
            max_sfb,
            num_window_groups,
            used,
        } => {
            if info.max_sfb > max_sfb || info.num_window_groups > num_window_groups {
                return Err(AacLcError::InvalidConfig(
                    "mid/side mask does not cover requested short-window groups/bands",
                ));
            }
            apply_mid_side_short_bands(info, layout, left, right, |group, sfb| {
                used[group][sfb] && allow_band(group, sfb)
            })
        }
    }
}

fn is_intensity_codebook(codebook: Option<SectionCodebook>) -> bool {
    matches!(
        codebook,
        Some(SectionCodebook::Intensity | SectionCodebook::IntensityNegative)
    )
}

fn is_noise_codebook(codebook: Option<SectionCodebook>) -> bool {
    matches!(codebook, Some(SectionCodebook::Noise))
}

fn mid_side_allowed(
    left_codebook: Option<SectionCodebook>,
    right_codebook: Option<SectionCodebook>,
) -> bool {
    !is_intensity_codebook(right_codebook)
        && !is_noise_codebook(left_codebook)
        && !is_noise_codebook(right_codebook)
}

fn intensity_codebook_sign(codebook: SectionCodebook) -> Option<f32> {
    match codebook {
        SectionCodebook::Intensity => Some(1.0),
        SectionCodebook::IntensityNegative => Some(-1.0),
        _ => None,
    }
}

fn intensity_scale(scale_factors: &ScaleFactorData, group: usize, sfb: usize) -> Result<f32> {
    scale_factors.intensity_multiplier(group, sfb)
}

fn mid_side_selected(mask: MidSideMask, group: usize, sfb: usize) -> Result<bool> {
    match mask {
        MidSideMask::None => Ok(false),
        MidSideMask::All => Ok(true),
        MidSideMask::Some {
            max_sfb,
            num_window_groups,
            used,
        } => {
            if group >= num_window_groups as usize || sfb >= max_sfb as usize {
                return Err(AacLcError::InvalidConfig(
                    "mid/side mask does not cover intensity band",
                ));
            }
            Ok(used[group][sfb])
        }
    }
}

const SHORT_WINDOW_COEFFICIENTS: usize = 128;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitreader::BitReader;
    use crate::channel::IndividualChannelStreamPrefix;
    use crate::ics::WindowShape;
    use crate::scalefactor::ScaleFactorDecoder;
    use crate::section::{SectionData, MAX_SCALE_FACTOR_BANDS};
    use crate::MAX_WINDOW_GROUPS;

    #[test]
    fn none_mask_does_not_change_coefficients() {
        let layout = BandLayout::new(&[0, 2, 4]);
        let mut left = [10.0, 20.0, 30.0, 40.0];
        let mut right = [1.0, 2.0, 3.0, 4.0];

        apply_mid_side_long(MidSideMask::None, 2, layout, &mut left, &mut right).unwrap();

        assert_eq!(left, [10.0, 20.0, 30.0, 40.0]);
        assert_eq!(right, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn all_mask_applies_every_band() {
        let layout = BandLayout::new(&[0, 2, 4]);
        let mut left = [10.0, 20.0, 30.0, 40.0];
        let mut right = [1.0, 2.0, 3.0, 4.0];

        apply_mid_side_long(MidSideMask::All, 2, layout, &mut left, &mut right).unwrap();

        assert_eq!(left, [11.0, 22.0, 33.0, 44.0]);
        assert_eq!(right, [9.0, 18.0, 27.0, 36.0]);
    }

    #[test]
    fn some_mask_applies_selected_bands() {
        let layout = BandLayout::new(&[0, 2, 4]);
        let mut left = [10.0, 20.0, 30.0, 40.0];
        let mut right = [1.0, 2.0, 3.0, 4.0];
        let mut used = [[false; MAX_SCALE_FACTOR_BANDS]; MAX_WINDOW_GROUPS];
        used[0][1] = true;

        apply_mid_side_long(
            MidSideMask::Some {
                max_sfb: 2,
                num_window_groups: 1,
                used,
            },
            2,
            layout,
            &mut left,
            &mut right,
        )
        .unwrap();

        assert_eq!(left, [10.0, 20.0, 33.0, 44.0]);
        assert_eq!(right, [1.0, 2.0, 27.0, 36.0]);
    }

    #[test]
    fn intensity_long_reconstructs_right_and_skips_mid_side_transform() {
        let layout = BandLayout::new(&[0, 2, 4]);
        let left_prefix = long_prefix(2, &[(1, 4), (2, 5)]);
        let right_prefix = long_prefix(2, &[(14, 4), (1, 5), (1, 4), (1, 5)]);
        let scale_factors = scale_factors(&right_prefix, &[4, 0]);
        let mut left = [8.0, 16.0, 10.0, 20.0];
        let mut right = [0.0, 0.0, 1.0, 2.0];

        apply_intensity_stereo_long(
            MidSideMask::All,
            2,
            layout,
            &right_prefix.section_data,
            &scale_factors,
            &left,
            &mut right,
        )
        .unwrap();
        apply_mid_side_long_excluding_intensity(
            MidSideMask::All,
            2,
            layout,
            &left_prefix.section_data,
            &right_prefix.section_data,
            &mut left,
            &mut right,
        )
        .unwrap();

        assert_eq!(left, [8.0, 16.0, 11.0, 22.0]);
        assert_eq!(right, [-4.0, -8.0, 9.0, 18.0]);
    }

    #[test]
    fn mid_side_long_skips_noise_bands_on_either_channel() {
        let layout = BandLayout::new(&[0, 2, 4, 6]);
        let left_prefix = long_prefix(3, &[(1, 4), (1, 5), (13, 4), (1, 5), (1, 4), (1, 5)]);
        let right_prefix = long_prefix(3, &[(1, 4), (2, 5), (13, 4), (1, 5)]);
        let mut left = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let mut right = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        apply_mid_side_long_excluding_intensity(
            MidSideMask::All,
            3,
            layout,
            &left_prefix.section_data,
            &right_prefix.section_data,
            &mut left,
            &mut right,
        )
        .unwrap();

        assert_eq!(left, [11.0, 22.0, 30.0, 40.0, 50.0, 60.0]);
        assert_eq!(right, [9.0, 18.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn rejects_grouped_mid_side_for_long_helper() {
        let layout = BandLayout::new(&[0, 2, 4]);
        let mut left = [10.0, 20.0, 30.0, 40.0];
        let mut right = [1.0, 2.0, 3.0, 4.0];
        let used = [[true; MAX_SCALE_FACTOR_BANDS]; MAX_WINDOW_GROUPS];

        assert_eq!(
            apply_mid_side_long(
                MidSideMask::Some {
                    max_sfb: 2,
                    num_window_groups: 2,
                    used,
                },
                2,
                layout,
                &mut left,
                &mut right,
            )
            .unwrap_err(),
            AacLcError::NotImplemented("grouped mid/side stereo reconstruction")
        );
    }

    #[test]
    fn rejects_mismatched_channel_lengths() {
        let layout = BandLayout::new(&[0, 2, 4]);
        let mut left = [10.0, 20.0, 30.0, 40.0];
        let mut right = [1.0, 2.0, 3.0];

        assert_eq!(
            apply_mid_side_long(MidSideMask::All, 2, layout, &mut left, &mut right).unwrap_err(),
            AacLcError::InvalidConfig("mid/side channel buffers have different lengths")
        );
    }

    #[test]
    fn rejects_bands_that_exceed_channel_length() {
        let layout = BandLayout::new(&[0, 2, 5]);
        let mut left = [10.0, 20.0, 30.0, 40.0];
        let mut right = [1.0, 2.0, 3.0, 4.0];

        assert_eq!(
            apply_mid_side_long(MidSideMask::All, 2, layout, &mut left, &mut right).unwrap_err(),
            AacLcError::InvalidConfig("mid/side scale-factor band exceeds channel buffer")
        );
    }

    #[test]
    fn short_none_mask_does_not_change_coefficients() {
        let info = short_info(2, &[8]);
        let layout = BandLayout::new(&[0, 2, 4]);
        let mut left = sample_channel();
        let mut right = vec![1.0; 1024];
        let original_left = left.clone();
        let original_right = right.clone();

        apply_mid_side_short(MidSideMask::None, &info, layout, &mut left, &mut right).unwrap();

        assert_eq!(left, original_left);
        assert_eq!(right, original_right);
    }

    #[test]
    fn short_all_mask_applies_every_grouped_window_band() {
        let info = short_info(2, &[2, 6]);
        let layout = BandLayout::new(&[0, 2, 4]);
        let mut left = sample_channel();
        let mut right = vec![1.0; 1024];

        apply_mid_side_short(MidSideMask::All, &info, layout, &mut left, &mut right).unwrap();

        assert_eq!(&left[0..4], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&right[0..4], &[-1.0, 0.0, 1.0, 2.0]);
        assert_eq!(&left[128..132], &[129.0, 130.0, 131.0, 132.0]);
        assert_eq!(&right[128..132], &[127.0, 128.0, 129.0, 130.0]);
        assert_eq!(&left[896..900], &[897.0, 898.0, 899.0, 900.0]);
        assert_eq!(&right[896..900], &[895.0, 896.0, 897.0, 898.0]);
    }

    #[test]
    fn short_some_mask_applies_selected_grouped_bands() {
        let info = short_info(2, &[2, 6]);
        let layout = BandLayout::new(&[0, 2, 4]);
        let mut left = sample_channel();
        let mut right = vec![1.0; 1024];
        let mut used = [[false; MAX_SCALE_FACTOR_BANDS]; MAX_WINDOW_GROUPS];
        used[0][0] = true;
        used[1][1] = true;

        apply_mid_side_short(
            MidSideMask::Some {
                max_sfb: 2,
                num_window_groups: 2,
                used,
            },
            &info,
            layout,
            &mut left,
            &mut right,
        )
        .unwrap();

        assert_eq!(&left[0..4], &[1.0, 2.0, 2.0, 3.0]);
        assert_eq!(&right[0..4], &[-1.0, 0.0, 1.0, 1.0]);
        assert_eq!(&left[128..132], &[129.0, 130.0, 130.0, 131.0]);
        assert_eq!(&right[128..132], &[127.0, 128.0, 1.0, 1.0]);
        assert_eq!(&left[256..260], &[256.0, 257.0, 259.0, 260.0]);
        assert_eq!(&right[256..260], &[1.0, 1.0, 257.0, 258.0]);
        assert_eq!(&left[896..900], &[896.0, 897.0, 899.0, 900.0]);
        assert_eq!(&right[896..900], &[1.0, 1.0, 897.0, 898.0]);
    }

    #[test]
    fn intensity_short_reconstructs_grouped_windows() {
        let info = short_info(1, &[8]);
        let layout = BandLayout::new(&[0, 2]);
        let prefix = prefix_with_info(info, &[(15, 4), (1, 3)]);
        let scale_factors = scale_factors(&prefix, &[0]);
        let left = sample_channel();
        let mut right = vec![0.0; 1024];

        apply_intensity_stereo_short(
            MidSideMask::None,
            &info,
            layout,
            &prefix.section_data,
            &scale_factors,
            &left,
            &mut right,
        )
        .unwrap();

        assert_eq!(&right[0..4], &[-0.0, -1.0, 0.0, 0.0]);
        assert_eq!(&right[128..132], &[-128.0, -129.0, 0.0, 0.0]);
        assert_eq!(&right[896..900], &[-896.0, -897.0, 0.0, 0.0]);
    }

    #[test]
    fn short_some_mask_rejects_missing_group_coverage() {
        let info = short_info(2, &[2, 6]);
        let layout = BandLayout::new(&[0, 2, 4]);
        let mut left = sample_channel();
        let mut right = vec![1.0; 1024];
        let used = [[false; MAX_SCALE_FACTOR_BANDS]; MAX_WINDOW_GROUPS];

        assert_eq!(
            apply_mid_side_short(
                MidSideMask::Some {
                    max_sfb: 2,
                    num_window_groups: 1,
                    used,
                },
                &info,
                layout,
                &mut left,
                &mut right,
            )
            .unwrap_err(),
            AacLcError::InvalidConfig(
                "mid/side mask does not cover requested short-window groups/bands"
            )
        );
    }

    fn short_info(max_sfb: u8, group_lens: &[u8]) -> IcsInfo {
        let mut window_group_len = [0u8; MAX_WINDOW_GROUPS];
        window_group_len[..group_lens.len()].copy_from_slice(group_lens);
        IcsInfo {
            window_sequence: WindowSequence::EightShort,
            window_shape: WindowShape::Sine,
            max_sfb,
            num_windows: 8,
            num_window_groups: group_lens.len() as u8,
            window_group_len,
        }
    }

    fn sample_channel() -> Vec<f32> {
        (0..1024).map(|value| value as f32).collect()
    }

    fn long_prefix(max_sfb: u8, section_fields: &[(u32, u8)]) -> IndividualChannelStreamPrefix {
        let info = IcsInfo {
            window_sequence: WindowSequence::OnlyLong,
            window_shape: WindowShape::Sine,
            max_sfb,
            num_windows: 1,
            num_window_groups: 1,
            window_group_len: [1, 0, 0, 0, 0, 0, 0, 0],
        };
        prefix_with_info(info, section_fields)
    }

    fn prefix_with_info(
        info: IcsInfo,
        section_fields: &[(u32, u8)],
    ) -> IndividualChannelStreamPrefix {
        let bytes = build_bits(section_fields);
        let mut reader = BitReader::new(&bytes);
        let section_data = SectionData::read(&mut reader, &info).unwrap();

        IndividualChannelStreamPrefix {
            global_gain: 100,
            ics_info: info,
            section_data,
        }
    }

    fn scale_factors(
        prefix: &IndividualChannelStreamPrefix,
        deltas: &'static [i16],
    ) -> ScaleFactorData {
        let mut decoder = SequenceScaleFactorDecoder { deltas, pos: 0 };
        let mut reader = BitReader::new(&[]);
        ScaleFactorData::read(&mut reader, prefix, &mut decoder).unwrap()
    }

    struct SequenceScaleFactorDecoder {
        deltas: &'static [i16],
        pos: usize,
    }

    impl ScaleFactorDecoder for SequenceScaleFactorDecoder {
        fn read_delta(&mut self, _reader: &mut BitReader<'_>) -> Result<i16> {
            let value = self.deltas[self.pos];
            self.pos += 1;
            Ok(value)
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
