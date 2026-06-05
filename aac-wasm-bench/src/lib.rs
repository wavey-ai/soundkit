pub const FIXTURE_NAME: &str = "WESTSIDE_MIX_4_CONFIRMATION_130323_256k.aac";
pub const FIXTURE: &[u8] =
    include_bytes!("../../golden/aac/WESTSIDE_MIX_4_CONFIRMATION_130323_256k.aac");
pub const SOURCE_WAV_ENV: &str = "SOUNDKIT_AAC_SOURCE_WAV";
pub const DEFAULT_SOURCE_WAV_PATH: &str =
    "../../bitneedle/apps/press/testdata/audio-regression/WESTSIDE_MIX 4 CONFIRMATION_130323.wav";
pub const DEFAULT_ITERATIONS: usize = 5;
pub const DEFAULT_RMSE_TOLERANCE: f64 = 0.005;
pub const DEFAULT_MEAN_ABS_TOLERANCE: f64 = 0.001;
pub const DEFAULT_MAX_ABS_TOLERANCE: f64 = 0.50;
pub const DEFAULT_MIN_SNR_DB: f64 = 35.0;

#[derive(Clone, Copy, Debug)]
pub struct AacFixture {
    pub name: &'static str,
    pub data: &'static [u8],
}

pub const DEFAULT_FIXTURE: AacFixture = AacFixture {
    name: FIXTURE_NAME,
    data: FIXTURE,
};

#[derive(Clone, Copy, Debug)]
pub struct AdtsFrame<'a> {
    pub full: &'a [u8],
    pub raw: &'a [u8],
    pub audio_object_type: u8,
    pub sample_rate_index: u8,
    pub sample_rate: u32,
    pub channels: u8,
}

impl AdtsFrame<'_> {
    pub fn audio_specific_config(self) -> [u8; 2] {
        [
            (self.audio_object_type << 3) | (self.sample_rate_index >> 1),
            ((self.sample_rate_index & 1) << 7) | (self.channels << 3),
        ]
    }
}

#[derive(Clone, Debug)]
pub struct DecodeOutput {
    pub pcm: Vec<f32>,
    pub decoded_frames: u64,
    pub samples_per_channel: u64,
    pub sample_rate: u32,
    pub channels: u8,
}

impl DecodeOutput {
    pub fn stats(&self) -> PcmStats {
        PcmStats::from_pcm(&self.pcm)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PcmStats {
    pub sample_count: usize,
    pub rms: f64,
    pub peak_abs: f64,
    pub checksum: u64,
}

impl PcmStats {
    pub fn from_pcm(pcm: &[f32]) -> Self {
        let mut sum_squares = 0.0f64;
        let mut peak_abs = 0.0f64;
        let mut checksum = 0xcbf29ce484222325u64;

        for sample_f32 in pcm {
            let sample = *sample_f32 as f64;
            sum_squares += sample * sample;
            peak_abs = peak_abs.max(sample.abs());
            checksum ^= sample_f32.to_bits() as u64;
            checksum = checksum.wrapping_mul(0x100000001b3);
        }

        let rms = if pcm.is_empty() {
            0.0
        } else {
            (sum_squares / pcm.len() as f64).sqrt()
        };

        Self {
            sample_count: pcm.len(),
            rms,
            peak_abs,
            checksum,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct QualityComparison {
    pub compared_samples: usize,
    pub reference_samples: usize,
    pub candidate_samples: usize,
    pub length_delta: isize,
    pub candidate_sample_offset: isize,
    pub reference_rms: f64,
    pub candidate_rms: f64,
    pub max_abs_error: f64,
    pub p99_abs_error: f64,
    pub p999_abs_error: f64,
    pub p9999_abs_error: f64,
    pub samples_over_001: usize,
    pub samples_over_01: usize,
    pub samples_over_02: usize,
    pub mean_abs_error: f64,
    pub rmse: f64,
    pub snr_db: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct FrameQualityHotspot {
    pub frame_index: usize,
    pub sample_start: usize,
    pub compared_samples: usize,
    pub max_abs_error: f64,
    pub mean_abs_error: f64,
    pub rmse: f64,
    pub reference_rms: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct FrameRegionError {
    pub frame_index: usize,
    pub channel: usize,
    pub region_start: usize,
    pub region_end: usize,
    pub compared_samples: usize,
    pub max_abs_error: f64,
    pub max_abs_frame_sample: usize,
    pub mean_abs_error: f64,
    pub rmse: f64,
    pub reference_rms: f64,
}

#[cfg(feature = "soundkit-lc")]
#[derive(Clone, Debug)]
pub struct AacFrameFeatures {
    pub frame_index: usize,
    pub element: &'static str,
    pub common_window: bool,
    pub mid_side: &'static str,
    pub mid_side_bands: usize,
    pub left: AacChannelFeatures,
    pub right: Option<AacChannelFeatures>,
}

#[cfg(feature = "soundkit-lc")]
#[derive(Clone, Debug)]
pub struct AacChannelFeatures {
    pub global_gain: u8,
    pub window_sequence: soundkit_aac_lc::WindowSequence,
    pub window_shape: soundkit_aac_lc::WindowShape,
    pub max_sfb: u8,
    pub num_window_groups: u8,
    pub window_group_len: [u8; soundkit_aac_lc::MAX_WINDOW_GROUPS],
    pub zero_bands: usize,
    pub spectral_bands: usize,
    pub noise_bands: usize,
    pub intensity_bands: usize,
    pub tns_filters: usize,
    pub tns_order: usize,
    pub tns_backward_filters: usize,
    pub pulse_count: u8,
}

impl QualityComparison {
    pub fn compare(reference: &[f32], candidate: &[f32]) -> Self {
        compare_slices(
            reference,
            candidate,
            reference.len(),
            candidate.len(),
            0,
            true,
        )
    }

    pub fn compare_aligned(
        reference: &[f32],
        candidate: &[f32],
        channels: usize,
        max_frame_offset: usize,
    ) -> Self {
        let step = channels.max(1) as isize;
        let max_sample_offset = max_frame_offset as isize * step;
        let probe_samples = reference
            .len()
            .min(candidate.len())
            .min(channels.max(1) * 1024 * 32);
        let mut best_offset = 0isize;
        let mut best = compare_with_offset(reference, candidate, 0, probe_samples, false);

        for sample_offset in (-max_sample_offset..=max_sample_offset).step_by(step as usize) {
            if sample_offset == 0 {
                continue;
            }

            let comparison =
                compare_with_offset(reference, candidate, sample_offset, probe_samples, false);
            if comparison.compared_samples == 0 {
                continue;
            }

            if comparison.rmse < best.rmse {
                best_offset = sample_offset;
                best = comparison;
            }
        }

        compare_with_offset(reference, candidate, best_offset, usize::MAX, true)
    }

    pub fn passes_default_thresholds(&self) -> bool {
        self.length_delta == 0
            && self.rmse <= DEFAULT_RMSE_TOLERANCE
            && self.mean_abs_error <= DEFAULT_MEAN_ABS_TOLERANCE
            && self.max_abs_error <= DEFAULT_MAX_ABS_TOLERANCE
            && self.snr_db >= DEFAULT_MIN_SNR_DB
    }
}

fn compare_with_offset(
    reference: &[f32],
    candidate: &[f32],
    candidate_sample_offset: isize,
    max_compared_samples: usize,
    collect_distribution: bool,
) -> QualityComparison {
    if candidate_sample_offset > 0 {
        let offset = candidate_sample_offset as usize;
        if offset >= candidate.len() {
            return compare_slices(
                &[],
                &[],
                reference.len(),
                candidate.len(),
                candidate_sample_offset,
                collect_distribution,
            );
        }
        let compared = reference
            .len()
            .min(candidate.len() - offset)
            .min(max_compared_samples);
        compare_slices(
            &reference[..compared],
            &candidate[offset..offset + compared],
            reference.len(),
            candidate.len(),
            candidate_sample_offset,
            collect_distribution,
        )
    } else if candidate_sample_offset < 0 {
        let offset = (-candidate_sample_offset) as usize;
        if offset >= reference.len() {
            return compare_slices(
                &[],
                &[],
                reference.len(),
                candidate.len(),
                candidate_sample_offset,
                collect_distribution,
            );
        }
        let compared = (reference.len() - offset)
            .min(candidate.len())
            .min(max_compared_samples);
        compare_slices(
            &reference[offset..offset + compared],
            &candidate[..compared],
            reference.len(),
            candidate.len(),
            candidate_sample_offset,
            collect_distribution,
        )
    } else {
        let compared = reference
            .len()
            .min(candidate.len())
            .min(max_compared_samples);
        compare_slices(
            &reference[..compared],
            &candidate[..compared],
            reference.len(),
            candidate.len(),
            0,
            collect_distribution,
        )
    }
}

fn compare_slices(
    reference: &[f32],
    candidate: &[f32],
    reference_samples: usize,
    candidate_samples: usize,
    candidate_sample_offset: isize,
    collect_distribution: bool,
) -> QualityComparison {
    let compared_samples = reference.len().min(candidate.len());
    let mut sum_reference_squares = 0.0f64;
    let mut sum_candidate_squares = 0.0f64;
    let mut max_abs_error = 0.0f64;
    let mut sum_abs_error = 0.0f64;
    let mut sum_square_error = 0.0f64;
    let mut samples_over_0_01 = 0usize;
    let mut samples_over_0_10 = 0usize;
    let mut samples_over_0_20 = 0usize;
    let mut abs_errors = if collect_distribution {
        Vec::with_capacity(compared_samples)
    } else {
        Vec::new()
    };

    for index in 0..compared_samples {
        let reference_sample = reference[index] as f64;
        let candidate_sample = candidate[index] as f64;
        let error = reference_sample - candidate_sample;
        let abs_error = error.abs();

        sum_reference_squares += reference_sample * reference_sample;
        sum_candidate_squares += candidate_sample * candidate_sample;
        max_abs_error = max_abs_error.max(abs_error);
        sum_abs_error += abs_error;
        sum_square_error += error * error;
        samples_over_0_01 += usize::from(abs_error > 0.01);
        samples_over_0_10 += usize::from(abs_error > 0.10);
        samples_over_0_20 += usize::from(abs_error > 0.20);
        if collect_distribution {
            abs_errors.push(abs_error);
        }
    }

    let (reference_rms, candidate_rms, mean_abs_error, rmse) = if compared_samples == 0 {
        (0.0, 0.0, 0.0, 0.0)
    } else {
        (
            (sum_reference_squares / compared_samples as f64).sqrt(),
            (sum_candidate_squares / compared_samples as f64).sqrt(),
            sum_abs_error / compared_samples as f64,
            (sum_square_error / compared_samples as f64).sqrt(),
        )
    };
    let snr_db = if rmse == 0.0 {
        f64::INFINITY
    } else if reference_rms == 0.0 {
        f64::NEG_INFINITY
    } else {
        20.0 * (reference_rms / rmse).log10()
    };
    let (p99_abs_error, p999_abs_error, p9999_abs_error) = if collect_distribution {
        abs_errors
            .sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
        (
            percentile_sorted(&abs_errors, 0.99),
            percentile_sorted(&abs_errors, 0.999),
            percentile_sorted(&abs_errors, 0.9999),
        )
    } else {
        (0.0, 0.0, 0.0)
    };

    QualityComparison {
        compared_samples,
        reference_samples,
        candidate_samples,
        length_delta: candidate_samples as isize - reference_samples as isize,
        candidate_sample_offset,
        reference_rms,
        candidate_rms,
        max_abs_error,
        p99_abs_error,
        p999_abs_error,
        p9999_abs_error,
        samples_over_001: samples_over_0_01,
        samples_over_01: samples_over_0_10,
        samples_over_02: samples_over_0_20,
        mean_abs_error,
        rmse,
        snr_db,
    }
}

fn percentile_sorted(values: &[f64], percentile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let percentile = percentile.clamp(0.0, 1.0);
    let index = ((values.len() - 1) as f64 * percentile).round() as usize;
    values[index]
}

impl QualityComparison {
    pub fn compare_unaligned(reference: &[f32], candidate: &[f32]) -> Self {
        Self::compare(reference, candidate)
    }

    pub fn offset_in_frames(&self, channels: usize) -> f64 {
        if channels == 0 {
            0.0
        } else {
            self.candidate_sample_offset as f64 / channels as f64
        }
    }
}

pub fn format_frame_hotspot(hotspot: &FrameQualityHotspot) -> String {
    format!(
        "frame={} sample_start={} compared_samples={} rmse={:.9} mean_abs_error={:.9} max_abs_error={:.9} ref_rms={:.9}",
        hotspot.frame_index,
        hotspot.sample_start,
        hotspot.compared_samples,
        hotspot.rmse,
        hotspot.mean_abs_error,
        hotspot.max_abs_error,
        hotspot.reference_rms,
    )
}

pub fn format_frame_region_error(error: &FrameRegionError) -> String {
    format!(
        "frame={} ch={} region={}..{} compared_samples={} rmse={:.9} mean_abs_error={:.9} max_abs_error={:.9} max_at={} ref_rms={:.9}",
        error.frame_index,
        error.channel,
        error.region_start,
        error.region_end,
        error.compared_samples,
        error.rmse,
        error.mean_abs_error,
        error.max_abs_error,
        error.max_abs_frame_sample,
        error.reference_rms,
    )
}

#[cfg(feature = "soundkit-lc")]
pub fn format_frame_features(features: &AacFrameFeatures) -> String {
    let right = features
        .right
        .as_ref()
        .map(format_channel_features)
        .unwrap_or_else(|| "none".to_string());

    format!(
        "frame={} element={} common_window={} mid_side={} ms_bands={} left=[{}] right=[{}]",
        features.frame_index,
        features.element,
        features.common_window,
        features.mid_side,
        features.mid_side_bands,
        format_channel_features(&features.left),
        right,
    )
}

#[cfg(feature = "soundkit-lc")]
fn format_channel_features(features: &AacChannelFeatures) -> String {
    format!(
        "seq={:?} shape={:?} gg={} max_sfb={} groups={} lens={} zero={} spectral={} pns={} intensity={} tns_filters={} tns_order={} tns_back={} pulses={}",
        features.window_sequence,
        features.window_shape,
        features.global_gain,
        features.max_sfb,
        features.num_window_groups,
        format_group_lens(features),
        features.zero_bands,
        features.spectral_bands,
        features.noise_bands,
        features.intensity_bands,
        features.tns_filters,
        features.tns_order,
        features.tns_backward_filters,
        features.pulse_count,
    )
}

#[cfg(feature = "soundkit-lc")]
fn format_group_lens(features: &AacChannelFeatures) -> String {
    let groups = features.num_window_groups as usize;
    let mut out = String::new();
    out.push('[');
    for (index, value) in features
        .window_group_len
        .iter()
        .copied()
        .take(groups)
        .enumerate()
    {
        if index != 0 {
            out.push(',');
        }
        out.push_str(&value.to_string());
    }
    out.push(']');
    out
}

#[derive(Clone, Debug)]
pub struct BenchResult {
    pub name: &'static str,
    pub iterations: usize,
    pub frames: u64,
    pub decoded_frames: u64,
    pub samples_per_channel: u64,
    pub sample_rate: u32,
    pub channels: u8,
    pub elapsed_ms: f64,
    pub pcm_stats: PcmStats,
}

impl BenchResult {
    pub fn audio_seconds(&self) -> f64 {
        if self.sample_rate == 0 {
            0.0
        } else {
            self.samples_per_channel as f64 / self.sample_rate as f64
        }
    }

    pub fn real_time_factor(&self) -> f64 {
        let audio_seconds = self.audio_seconds();
        if audio_seconds == 0.0 {
            0.0
        } else {
            (self.elapsed_ms / 1000.0) / audio_seconds
        }
    }

    pub fn frames_per_second(&self) -> f64 {
        if self.elapsed_ms == 0.0 {
            0.0
        } else {
            self.decoded_frames as f64 / (self.elapsed_ms / 1000.0)
        }
    }
}

pub fn format_bench_result(result: &BenchResult) -> String {
    format!(
        "{:<18} iterations={} frames={} decoded={} samples/ch={} sr={} ch={} elapsed_ms={:.3} rtf={:.6} frames_per_sec={:.1} quality_samples={} rms={:.9} peak={:.9} checksum={:016x}",
        result.name,
        result.iterations,
        result.frames,
        result.decoded_frames,
        result.samples_per_channel,
        result.sample_rate,
        result.channels,
        result.elapsed_ms,
        result.real_time_factor(),
        result.frames_per_second(),
        result.pcm_stats.sample_count,
        result.pcm_stats.rms,
        result.pcm_stats.peak_abs,
        result.pcm_stats.checksum,
    )
}

pub fn format_decode_summary(
    name: &str,
    iterations: usize,
    frames: u64,
    decoded_frames: u64,
    samples_per_channel: u64,
    sample_rate: u32,
    channels: u8,
    pcm_stats: PcmStats,
) -> String {
    format!(
        "{:<18} iterations={} frames={} decoded={} samples/ch={} sr={} ch={} quality_samples={} rms={:.9} peak={:.9} checksum={:016x}",
        name,
        iterations,
        frames,
        decoded_frames,
        samples_per_channel,
        sample_rate,
        channels,
        pcm_stats.sample_count,
        pcm_stats.rms,
        pcm_stats.peak_abs,
        pcm_stats.checksum,
    )
}

pub fn format_quality_comparison(name: &str, comparison: &QualityComparison) -> String {
    let status = if comparison.passes_default_thresholds() {
        "pass"
    } else {
        "fail"
    };
    format!(
        "{:<18} status={} compared_samples={} reference_samples={} candidate_samples={} length_delta={} candidate_sample_offset={} ref_rms={:.9} candidate_rms={:.9} max_abs_error={:.9} p99_abs_error={:.9} p999_abs_error={:.9} p9999_abs_error={:.9} samples_over_0.01={} samples_over_0.10={} samples_over_0.20={} mean_abs_error={:.9} rmse={:.9} snr_db={:.3}",
        name,
        status,
        comparison.compared_samples,
        comparison.reference_samples,
        comparison.candidate_samples,
        comparison.length_delta,
        comparison.candidate_sample_offset,
        comparison.reference_rms,
        comparison.candidate_rms,
        comparison.max_abs_error,
        comparison.p99_abs_error,
        comparison.p999_abs_error,
        comparison.p9999_abs_error,
        comparison.samples_over_001,
        comparison.samples_over_01,
        comparison.samples_over_02,
        comparison.mean_abs_error,
        comparison.rmse,
        comparison.snr_db,
    )
}

pub fn format_quality_measurement(name: &str, comparison: &QualityComparison) -> String {
    format!(
        "{:<18} compared_samples={} reference_samples={} candidate_samples={} length_delta={} candidate_sample_offset={} ref_rms={:.9} candidate_rms={:.9} max_abs_error={:.9} p99_abs_error={:.9} p999_abs_error={:.9} p9999_abs_error={:.9} samples_over_0.01={} samples_over_0.10={} samples_over_0.20={} mean_abs_error={:.9} rmse={:.9} snr_db={:.3}",
        name,
        comparison.compared_samples,
        comparison.reference_samples,
        comparison.candidate_samples,
        comparison.length_delta,
        comparison.candidate_sample_offset,
        comparison.reference_rms,
        comparison.candidate_rms,
        comparison.max_abs_error,
        comparison.p99_abs_error,
        comparison.p999_abs_error,
        comparison.p9999_abs_error,
        comparison.samples_over_001,
        comparison.samples_over_01,
        comparison.samples_over_02,
        comparison.mean_abs_error,
        comparison.rmse,
        comparison.snr_db,
    )
}

#[cfg(not(any(target_arch = "wasm32", target_arch = "wasm64")))]
pub fn source_wav_path() -> std::path::PathBuf {
    let path = std::env::var_os(SOURCE_WAV_ENV)
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(DEFAULT_SOURCE_WAV_PATH)
        });
    std::fs::canonicalize(&path).unwrap_or(path)
}

#[cfg(not(any(target_arch = "wasm32", target_arch = "wasm64")))]
pub fn decode_source_wav_pcm() -> Result<DecodeOutput, String> {
    let path = source_wav_path();
    let data = std::fs::read(&path)
        .map_err(|err| format!("read source WAV {} failed: {err}", path.display()))?;
    decode_wav_pcm_bytes(&data)
}

#[cfg(not(any(target_arch = "wasm32", target_arch = "wasm64")))]
pub fn compare_to_source_wav(decoded: &DecodeOutput) -> Result<QualityComparison, String> {
    let source = decode_source_wav_pcm()?;
    if decoded.sample_rate != source.sample_rate {
        return Err(format!(
            "sample-rate mismatch: source={} decoded={}",
            source.sample_rate, decoded.sample_rate
        ));
    }
    if decoded.channels != source.channels {
        return Err(format!(
            "channel-count mismatch: source={} decoded={}",
            source.channels, decoded.channels
        ));
    }

    Ok(QualityComparison::compare_aligned(
        &source.pcm,
        &decoded.pcm,
        source.channels as usize,
        8192,
    ))
}

#[cfg(all(
    feature = "fdk",
    not(any(target_arch = "wasm32", target_arch = "wasm64"))
))]
pub fn compare_fdk_to_source_wav() -> Result<QualityComparison, String> {
    let decoded = decode_fdk_fixture_pcm()?;
    compare_to_source_wav(&decoded)
}

#[cfg(all(
    feature = "soundkit-lc",
    not(any(target_arch = "wasm32", target_arch = "wasm64"))
))]
pub fn compare_soundkit_lc_to_source_wav() -> Result<QualityComparison, String> {
    let decoded = decode_soundkit_lc_fixture_pcm()?;
    compare_to_source_wav(&decoded)
}

#[cfg(all(
    feature = "soundkit-lc",
    not(any(target_arch = "wasm32", target_arch = "wasm64"))
))]
pub fn source_soundkit_lc_frame_hotspots(limit: usize) -> Result<Vec<FrameQualityHotspot>, String> {
    let reference = decode_source_wav_pcm()?;
    let candidate = decode_soundkit_lc_fixture_pcm()?;
    frame_quality_hotspots(&reference, &candidate, 8192, limit)
}

#[cfg(all(
    feature = "fdk",
    not(any(target_arch = "wasm32", target_arch = "wasm64"))
))]
pub fn source_fdk_frame_hotspots(limit: usize) -> Result<Vec<FrameQualityHotspot>, String> {
    let reference = decode_source_wav_pcm()?;
    let candidate = decode_fdk_fixture_pcm()?;
    frame_quality_hotspots(&reference, &candidate, 8192, limit)
}

#[cfg(all(
    feature = "symphonia",
    not(any(target_arch = "wasm32", target_arch = "wasm64"))
))]
pub fn source_symphonia_frame_hotspots(limit: usize) -> Result<Vec<FrameQualityHotspot>, String> {
    let reference = decode_source_wav_pcm()?;
    let candidate = decode_symphonia_fixture_pcm()?;
    frame_quality_hotspots(&reference, &candidate, 8192, limit)
}

#[cfg(feature = "soundkit-lc")]
pub fn soundkit_lc_frame_features(frame_index: usize) -> Result<AacFrameFeatures, String> {
    let frames = parse_adts_frames(FIXTURE)?;
    let frame = frames
        .get(frame_index)
        .ok_or_else(|| format!("frame index {frame_index} exceeds fixture frame count"))?;
    parse_soundkit_lc_frame_features(frame_index, frame)
}

#[cfg(feature = "soundkit-lc")]
fn parse_soundkit_lc_frame_features(
    frame_index: usize,
    frame: &AdtsFrame<'_>,
) -> Result<AacFrameFeatures, String> {
    use soundkit_aac_lc::{
        AudioSpecificConfig, BitReader, ChannelPairElementHeader, ElementId,
        IndividualChannelStream, MidSideMask, RawElementHeader, StandardScaleFactorDecoder,
    };

    let config = AudioSpecificConfig::parse(&frame.audio_specific_config())
        .map_err(|err| format!("parse ASC for frame {frame_index} failed: {err}"))?;
    let mut reader = BitReader::new(frame.raw);
    let header = RawElementHeader::read(&mut reader)
        .map_err(|err| format!("parse element header for frame {frame_index} failed: {err}"))?;

    match header.id {
        ElementId::SingleChannel => {
            let mut scale_factor_decoder = StandardScaleFactorDecoder;
            let stream =
                IndividualChannelStream::read(&mut reader, None, &mut scale_factor_decoder)
                    .map_err(|err| format!("parse SCE frame {frame_index} failed: {err}"))?;
            Ok(AacFrameFeatures {
                frame_index,
                element: "SCE",
                common_window: false,
                mid_side: "none",
                mid_side_bands: 0,
                left: channel_features(&stream),
                right: None,
            })
        }
        ElementId::ChannelPair => {
            let tag = header
                .tag
                .ok_or_else(|| format!("missing CPE tag in frame {frame_index}"))?;
            let pair = ChannelPairElementHeader::read(&mut reader, tag)
                .map_err(|err| format!("parse CPE header for frame {frame_index} failed: {err}"))?;
            let mut scale_factor_decoder = StandardScaleFactorDecoder;
            let left = IndividualChannelStream::read(
                &mut reader,
                pair.common_ics,
                &mut scale_factor_decoder,
            )
            .map_err(|err| format!("parse left ICS frame {frame_index} failed: {err}"))?;
            skip_spectral_payload(&mut reader, &config, &left)
                .map_err(|err| format!("skip left spectral frame {frame_index} failed: {err}"))?;
            let right = IndividualChannelStream::read(
                &mut reader,
                pair.common_ics,
                &mut scale_factor_decoder,
            )
            .map_err(|err| format!("parse right ICS frame {frame_index} failed: {err}"))?;

            let (mid_side, mid_side_bands) = match pair.mid_side {
                MidSideMask::None => ("none", 0),
                MidSideMask::All => ("all", left.prefix.ics_info.max_sfb as usize),
                MidSideMask::Some {
                    max_sfb,
                    num_window_groups,
                    used,
                } => {
                    let mut bands = 0usize;
                    for group in 0..num_window_groups as usize {
                        for sfb in 0..max_sfb as usize {
                            bands += usize::from(used[group][sfb]);
                        }
                    }
                    ("some", bands)
                }
            };

            Ok(AacFrameFeatures {
                frame_index,
                element: "CPE",
                common_window: pair.common_window,
                mid_side,
                mid_side_bands,
                left: channel_features(&left),
                right: Some(channel_features(&right)),
            })
        }
        _ => Err(format!(
            "frame {frame_index} starts with unsupported element {:?}",
            header.id
        )),
    }
}

#[cfg(feature = "soundkit-lc")]
fn skip_spectral_payload(
    reader: &mut soundkit_aac_lc::BitReader<'_>,
    config: &soundkit_aac_lc::AudioSpecificConfig,
    stream: &soundkit_aac_lc::IndividualChannelStream,
) -> soundkit_aac_lc::Result<()> {
    let layout = match stream.prefix.ics_info.window_sequence {
        soundkit_aac_lc::WindowSequence::EightShort => {
            soundkit_aac_lc::short_window_band_layout(config.sampling_frequency)
        }
        soundkit_aac_lc::WindowSequence::OnlyLong
        | soundkit_aac_lc::WindowSequence::LongStart
        | soundkit_aac_lc::WindowSequence::LongStop => {
            soundkit_aac_lc::long_window_band_layout(config.sampling_frequency)
        }
    }?;
    let mut coefficients = soundkit_aac_lc::SpectralCoefficients::new(config.frame_length);
    coefficients.decode_standard_with_pulse(
        reader,
        &stream.prefix,
        &stream.scale_factors,
        layout,
        stream.pulse_data.as_ref(),
    )?;
    Ok(())
}

#[cfg(feature = "soundkit-lc")]
fn channel_features(stream: &soundkit_aac_lc::IndividualChannelStream) -> AacChannelFeatures {
    let mut zero_bands = 0usize;
    let mut spectral_bands = 0usize;
    let mut noise_bands = 0usize;
    let mut intensity_bands = 0usize;

    for group in 0..stream.prefix.ics_info.num_window_groups as usize {
        for sfb in 0..stream.prefix.ics_info.max_sfb as usize {
            match stream.prefix.section_data.codebook(group, sfb) {
                Some(soundkit_aac_lc::SectionCodebook::Zero) => zero_bands += 1,
                Some(soundkit_aac_lc::SectionCodebook::Spectral(_)) => spectral_bands += 1,
                Some(soundkit_aac_lc::SectionCodebook::Noise) => noise_bands += 1,
                Some(
                    soundkit_aac_lc::SectionCodebook::Intensity
                    | soundkit_aac_lc::SectionCodebook::IntensityNegative,
                ) => intensity_bands += 1,
                None => {}
            }
        }
    }

    let mut tns_filters = 0usize;
    let mut tns_order = 0usize;
    let mut tns_backward_filters = 0usize;
    if let Some(tns) = stream.tns_data {
        for window in tns.windows.iter().take(tns.window_count as usize) {
            tns_filters += window.filter_count as usize;
            for filter in window.filters.iter().take(window.filter_count as usize) {
                tns_order += filter.order as usize;
                tns_backward_filters += usize::from(filter.direction);
            }
        }
    }

    AacChannelFeatures {
        global_gain: stream.prefix.global_gain,
        window_sequence: stream.prefix.ics_info.window_sequence,
        window_shape: stream.prefix.ics_info.window_shape,
        max_sfb: stream.prefix.ics_info.max_sfb,
        num_window_groups: stream.prefix.ics_info.num_window_groups,
        window_group_len: stream.prefix.ics_info.window_group_len,
        zero_bands,
        spectral_bands,
        noise_bands,
        intensity_bands,
        tns_filters,
        tns_order,
        tns_backward_filters,
        pulse_count: stream.pulse_data.map_or(0, |pulse| pulse.count),
    }
}

#[cfg(all(
    feature = "fdk",
    feature = "soundkit-lc",
    not(any(target_arch = "wasm32", target_arch = "wasm64"))
))]
pub fn soundkit_lc_fdk_frame_hotspots(limit: usize) -> Result<Vec<FrameQualityHotspot>, String> {
    let reference = decode_fdk_fixture_pcm()?;
    let candidate = decode_soundkit_lc_fixture_pcm()?;
    frame_quality_hotspots(&reference, &candidate, 4096, limit)
}

#[cfg(all(
    feature = "fdk",
    feature = "soundkit-lc",
    not(any(target_arch = "wasm32", target_arch = "wasm64"))
))]
pub fn soundkit_lc_fdk_frame_region_errors(
    frame_index: usize,
) -> Result<Vec<FrameRegionError>, String> {
    let reference = decode_fdk_fixture_pcm()?;
    let candidate = decode_soundkit_lc_fixture_pcm()?;
    frame_region_errors(&reference, &candidate, 4096, frame_index)
}

#[cfg(not(any(target_arch = "wasm32", target_arch = "wasm64")))]
fn frame_quality_hotspots(
    reference: &DecodeOutput,
    candidate: &DecodeOutput,
    max_frame_offset: usize,
    limit: usize,
) -> Result<Vec<FrameQualityHotspot>, String> {
    if reference.sample_rate != candidate.sample_rate {
        return Err(format!(
            "sample-rate mismatch: reference={} candidate={}",
            reference.sample_rate, candidate.sample_rate
        ));
    }
    if reference.channels != candidate.channels {
        return Err(format!(
            "channel-count mismatch: reference={} candidate={}",
            reference.channels, candidate.channels
        ));
    }

    let alignment = QualityComparison::compare_aligned(
        &reference.pcm,
        &candidate.pcm,
        reference.channels as usize,
        max_frame_offset,
    );
    let frame_samples = 1024usize * reference.channels as usize;
    if frame_samples == 0 {
        return Err("invalid zero channel frame size".to_string());
    }

    let (reference_start, candidate_start) = if alignment.candidate_sample_offset >= 0 {
        (0usize, alignment.candidate_sample_offset as usize)
    } else {
        ((-alignment.candidate_sample_offset) as usize, 0usize)
    };
    if reference_start >= reference.pcm.len() || candidate_start >= candidate.pcm.len() {
        return Ok(Vec::new());
    }

    let compared =
        (reference.pcm.len() - reference_start).min(candidate.pcm.len() - candidate_start);
    let frame_count = compared / frame_samples;
    let mut hotspots = Vec::with_capacity(frame_count);
    for frame_index in 0..frame_count {
        let local_start = frame_index * frame_samples;
        let ref_start = reference_start + local_start;
        let cand_start = candidate_start + local_start;
        let ref_frame = &reference.pcm[ref_start..ref_start + frame_samples];
        let cand_frame = &candidate.pcm[cand_start..cand_start + frame_samples];
        hotspots.push(compare_frame(frame_index, ref_start, ref_frame, cand_frame));
    }

    hotspots.sort_by(|left, right| {
        right
            .rmse
            .partial_cmp(&left.rmse)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                right
                    .max_abs_error
                    .partial_cmp(&left.max_abs_error)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });
    hotspots.truncate(limit.max(1));
    Ok(hotspots)
}

#[cfg(all(
    feature = "fdk",
    feature = "soundkit-lc",
    not(any(target_arch = "wasm32", target_arch = "wasm64"))
))]
fn frame_region_errors(
    reference: &DecodeOutput,
    candidate: &DecodeOutput,
    max_frame_offset: usize,
    frame_index: usize,
) -> Result<Vec<FrameRegionError>, String> {
    if reference.sample_rate != candidate.sample_rate {
        return Err(format!(
            "sample-rate mismatch: reference={} candidate={}",
            reference.sample_rate, candidate.sample_rate
        ));
    }
    if reference.channels != candidate.channels {
        return Err(format!(
            "channel-count mismatch: reference={} candidate={}",
            reference.channels, candidate.channels
        ));
    }

    let channels = reference.channels as usize;
    let alignment = QualityComparison::compare_aligned(
        &reference.pcm,
        &candidate.pcm,
        channels,
        max_frame_offset,
    );
    let frame_samples = 1024usize * channels;
    if frame_samples == 0 {
        return Err("invalid zero channel frame size".to_string());
    }

    let (reference_start, candidate_start) = if alignment.candidate_sample_offset >= 0 {
        (0usize, alignment.candidate_sample_offset as usize)
    } else {
        ((-alignment.candidate_sample_offset) as usize, 0usize)
    };
    let local_start = frame_index
        .checked_mul(frame_samples)
        .ok_or_else(|| "frame index overflow".to_string())?;
    let ref_frame_start = reference_start
        .checked_add(local_start)
        .ok_or_else(|| "reference frame start overflow".to_string())?;
    let cand_frame_start = candidate_start
        .checked_add(local_start)
        .ok_or_else(|| "candidate frame start overflow".to_string())?;
    if ref_frame_start + frame_samples > reference.pcm.len()
        || cand_frame_start + frame_samples > candidate.pcm.len()
    {
        return Err(format!(
            "frame {frame_index} exceeds aligned comparison length"
        ));
    }

    let mut errors = Vec::with_capacity(channels * 8);
    for channel in 0..channels {
        for tile in 0..8 {
            let region_start = tile * 128;
            let region_end = region_start + 128;
            errors.push(compare_frame_region(
                frame_index,
                channel,
                region_start,
                region_end,
                channels,
                &reference.pcm[ref_frame_start..ref_frame_start + frame_samples],
                &candidate.pcm[cand_frame_start..cand_frame_start + frame_samples],
            ));
        }
    }

    errors.sort_by(|left, right| {
        right
            .rmse
            .partial_cmp(&left.rmse)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                right
                    .max_abs_error
                    .partial_cmp(&left.max_abs_error)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });
    Ok(errors)
}

#[cfg(all(
    feature = "fdk",
    feature = "soundkit-lc",
    not(any(target_arch = "wasm32", target_arch = "wasm64"))
))]
fn compare_frame_region(
    frame_index: usize,
    channel: usize,
    region_start: usize,
    region_end: usize,
    channels: usize,
    reference: &[f32],
    candidate: &[f32],
) -> FrameRegionError {
    let mut sum_reference_squares = 0.0f64;
    let mut sum_abs_error = 0.0f64;
    let mut sum_square_error = 0.0f64;
    let mut max_abs_error = 0.0f64;
    let mut max_abs_frame_sample = region_start;
    let mut compared_samples = 0usize;

    for frame_sample in region_start..region_end {
        let index = frame_sample * channels + channel;
        if index >= reference.len() || index >= candidate.len() {
            break;
        }
        let reference_sample = reference[index] as f64;
        let error = reference_sample - candidate[index] as f64;
        let abs_error = error.abs();
        sum_reference_squares += reference_sample * reference_sample;
        sum_abs_error += abs_error;
        sum_square_error += error * error;
        if abs_error > max_abs_error {
            max_abs_error = abs_error;
            max_abs_frame_sample = frame_sample;
        }
        compared_samples += 1;
    }

    let (reference_rms, mean_abs_error, rmse) = if compared_samples == 0 {
        (0.0, 0.0, 0.0)
    } else {
        (
            (sum_reference_squares / compared_samples as f64).sqrt(),
            sum_abs_error / compared_samples as f64,
            (sum_square_error / compared_samples as f64).sqrt(),
        )
    };

    FrameRegionError {
        frame_index,
        channel,
        region_start,
        region_end,
        compared_samples,
        max_abs_error,
        max_abs_frame_sample,
        mean_abs_error,
        rmse,
        reference_rms,
    }
}

#[cfg(not(any(target_arch = "wasm32", target_arch = "wasm64")))]
fn compare_frame(
    frame_index: usize,
    sample_start: usize,
    reference: &[f32],
    candidate: &[f32],
) -> FrameQualityHotspot {
    let compared_samples = reference.len().min(candidate.len());
    let mut sum_reference_squares = 0.0f64;
    let mut sum_abs_error = 0.0f64;
    let mut sum_square_error = 0.0f64;
    let mut max_abs_error = 0.0f64;

    for index in 0..compared_samples {
        let reference_sample = reference[index] as f64;
        let error = reference_sample - candidate[index] as f64;
        let abs_error = error.abs();
        sum_reference_squares += reference_sample * reference_sample;
        sum_abs_error += abs_error;
        sum_square_error += error * error;
        max_abs_error = max_abs_error.max(abs_error);
    }

    let (reference_rms, mean_abs_error, rmse) = if compared_samples == 0 {
        (0.0, 0.0, 0.0)
    } else {
        (
            (sum_reference_squares / compared_samples as f64).sqrt(),
            sum_abs_error / compared_samples as f64,
            (sum_square_error / compared_samples as f64).sqrt(),
        )
    };

    FrameQualityHotspot {
        frame_index,
        sample_start,
        compared_samples,
        max_abs_error,
        mean_abs_error,
        rmse,
        reference_rms,
    }
}

#[cfg(all(
    feature = "symphonia",
    not(any(target_arch = "wasm32", target_arch = "wasm64"))
))]
pub fn compare_symphonia_to_source_wav() -> Result<QualityComparison, String> {
    let decoded = decode_symphonia_fixture_pcm()?;
    compare_to_source_wav(&decoded)
}

pub fn decode_wav_pcm_bytes(data: &[u8]) -> Result<DecodeOutput, String> {
    if data.len() < 12 || &data[..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        return Err("source WAV is not RIFF/WAVE".to_string());
    }

    let mut offset = 12usize;
    let mut format: Option<WavPcmFormat> = None;
    let mut pcm_data: Option<&[u8]> = None;
    while offset + 8 <= data.len() {
        let chunk_id = &data[offset..offset + 4];
        let chunk_len = read_le_u32(&data[offset + 4..offset + 8])? as usize;
        let chunk_start = offset + 8;
        let chunk_end = chunk_start
            .checked_add(chunk_len)
            .ok_or_else(|| "WAV chunk length overflow".to_string())?;
        if chunk_end > data.len() {
            return Err("truncated WAV chunk".to_string());
        }

        match chunk_id {
            b"fmt " => format = Some(parse_wav_fmt_chunk(&data[chunk_start..chunk_end])?),
            b"data" => pcm_data = Some(&data[chunk_start..chunk_end]),
            _ => {}
        }

        offset = chunk_end + (chunk_len & 1);
    }

    let format = format.ok_or_else(|| "WAV missing fmt chunk".to_string())?;
    let pcm_data = pcm_data.ok_or_else(|| "WAV missing data chunk".to_string())?;
    decode_wav_pcm_data(format, pcm_data)
}

#[derive(Clone, Copy, Debug)]
struct WavPcmFormat {
    channels: u16,
    sample_rate: u32,
    block_align: u16,
    bits_per_sample: u16,
}

fn parse_wav_fmt_chunk(data: &[u8]) -> Result<WavPcmFormat, String> {
    if data.len() < 16 {
        return Err("WAV fmt chunk is too short".to_string());
    }

    let audio_format = read_le_u16(&data[0..2])?;
    if audio_format != 1 {
        return Err(format!(
            "unsupported WAV format code {audio_format}; expected PCM"
        ));
    }

    let channels = read_le_u16(&data[2..4])?;
    let sample_rate = read_le_u32(&data[4..8])?;
    let block_align = read_le_u16(&data[12..14])?;
    let bits_per_sample = read_le_u16(&data[14..16])?;
    if channels == 0 {
        return Err("WAV channel count is zero".to_string());
    }
    if !matches!(bits_per_sample, 16 | 24 | 32) {
        return Err(format!(
            "unsupported PCM bits_per_sample {bits_per_sample}; expected 16, 24, or 32"
        ));
    }

    let expected_align = channels
        .checked_mul(bits_per_sample / 8)
        .ok_or_else(|| "WAV block-align overflow".to_string())?;
    if block_align != expected_align {
        return Err(format!(
            "WAV block_align {block_align} does not match expected {expected_align}"
        ));
    }

    Ok(WavPcmFormat {
        channels,
        sample_rate,
        block_align,
        bits_per_sample,
    })
}

fn decode_wav_pcm_data(format: WavPcmFormat, data: &[u8]) -> Result<DecodeOutput, String> {
    let bytes_per_sample = (format.bits_per_sample / 8) as usize;
    let channels = format.channels as usize;
    let block_align = format.block_align as usize;
    if data.len() % block_align != 0 {
        return Err("WAV data chunk is not aligned to sample frames".to_string());
    }

    let frames = data.len() / block_align;
    let mut pcm = Vec::with_capacity(frames * channels);
    for frame in data.chunks_exact(block_align) {
        for channel in 0..channels {
            let start = channel * bytes_per_sample;
            let sample = match format.bits_per_sample {
                16 => read_pcm_i16(&frame[start..start + 2]),
                24 => read_pcm_i24(&frame[start..start + 3]),
                32 => read_pcm_i32(&frame[start..start + 4]),
                _ => unreachable!("bits_per_sample was validated"),
            };
            pcm.push(sample);
        }
    }

    Ok(DecodeOutput {
        pcm,
        decoded_frames: frames as u64,
        samples_per_channel: frames as u64,
        sample_rate: format.sample_rate,
        channels: format.channels as u8,
    })
}

fn read_le_u16(data: &[u8]) -> Result<u16, String> {
    let bytes: [u8; 2] = data
        .try_into()
        .map_err(|_| "expected 2 little-endian bytes".to_string())?;
    Ok(u16::from_le_bytes(bytes))
}

fn read_le_u32(data: &[u8]) -> Result<u32, String> {
    let bytes: [u8; 4] = data
        .try_into()
        .map_err(|_| "expected 4 little-endian bytes".to_string())?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_pcm_i16(data: &[u8]) -> f32 {
    i16::from_le_bytes([data[0], data[1]]) as f32 / 32768.0
}

fn read_pcm_i24(data: &[u8]) -> f32 {
    let raw = i32::from(data[0]) | (i32::from(data[1]) << 8) | (i32::from(data[2]) << 16);
    ((raw << 8) >> 8) as f32 / 8_388_608.0
}

fn read_pcm_i32(data: &[u8]) -> f32 {
    i32::from_le_bytes([data[0], data[1], data[2], data[3]]) as f32 / 2_147_483_648.0
}

pub fn parse_adts_frames(data: &[u8]) -> Result<Vec<AdtsFrame<'_>>, String> {
    let mut frames = Vec::new();
    let mut offset = 0;

    while offset + 7 <= data.len() {
        while offset + 7 <= data.len()
            && !(data[offset] == 0xff && (data[offset + 1] & 0xf0) == 0xf0)
        {
            offset += 1;
        }

        if offset + 7 > data.len() {
            break;
        }

        let protection_absent = (data[offset + 1] & 0x01) != 0;
        let header_len = if protection_absent { 7 } else { 9 };
        let audio_object_type = ((data[offset + 2] & 0xc0) >> 6) + 1;
        let sample_rate_index = (data[offset + 2] & 0x3c) >> 2;
        let sample_rate = adts_sample_rate(sample_rate_index)
            .ok_or_else(|| format!("unsupported ADTS sample-rate index {sample_rate_index}"))?;
        let channels = ((data[offset + 2] & 0x01) << 2) | ((data[offset + 3] & 0xc0) >> 6);
        let frame_len = (((data[offset + 3] & 0x03) as usize) << 11)
            | ((data[offset + 4] as usize) << 3)
            | (((data[offset + 5] & 0xe0) as usize) >> 5);

        if frame_len <= header_len {
            return Err(format!(
                "invalid ADTS frame length {frame_len} at offset {offset}"
            ));
        }
        if offset + frame_len > data.len() {
            return Err(format!("truncated ADTS frame at offset {offset}"));
        }

        frames.push(AdtsFrame {
            full: &data[offset..offset + frame_len],
            raw: &data[offset + header_len..offset + frame_len],
            audio_object_type,
            sample_rate_index,
            sample_rate,
            channels,
        });
        offset += frame_len;
    }

    if frames.is_empty() {
        Err("no ADTS frames found".to_string())
    } else {
        Ok(frames)
    }
}

pub fn adts_sample_rate(index: u8) -> Option<u32> {
    const RATES: [u32; 13] = [
        96_000, 88_200, 64_000, 48_000, 44_100, 32_000, 24_000, 22_050, 16_000, 12_000, 11_025,
        8_000, 7_350,
    ];
    RATES.get(index as usize).copied()
}

#[cfg(feature = "fdk")]
pub fn decode_fdk_fixture_pcm() -> Result<DecodeOutput, String> {
    decode_fdk_fixture_pcm_for(DEFAULT_FIXTURE)
}

#[cfg(feature = "fdk")]
pub fn decode_fdk_fixture_pcm_for(fixture: AacFixture) -> Result<DecodeOutput, String> {
    let frames = parse_adts_frames(fixture.data)?;
    decode_fdk_frames(&frames, true)
}

#[cfg(all(
    feature = "fdk",
    not(any(target_arch = "wasm32", target_arch = "wasm64"))
))]
pub fn bench_fdk_fixture(iterations: usize) -> Result<BenchResult, String> {
    use std::hint::black_box;
    use std::time::Instant;

    let iterations = iterations.max(1);
    let frames = parse_adts_frames(FIXTURE)?;
    let quality = decode_fdk_frames(&frames, true)?;
    let sample_rate = frames[0].sample_rate;
    let channels = frames[0].channels;
    let mut decoded_frames = 0u64;
    let mut samples_per_channel = 0u64;
    let started = Instant::now();

    for _ in 0..iterations {
        let decoded = decode_fdk_frames(&frames, false)?;
        decoded_frames += decoded.decoded_frames;
        samples_per_channel += decoded.samples_per_channel;
        black_box(decoded.decoded_frames);
    }

    Ok(BenchResult {
        name: "fdk-aac-sys",
        iterations,
        frames: frames.len() as u64 * iterations as u64,
        decoded_frames,
        samples_per_channel,
        sample_rate,
        channels,
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        pcm_stats: quality.stats(),
    })
}

#[cfg(feature = "fdk")]
fn decode_fdk_frames(frames: &[AdtsFrame<'_>], capture_pcm: bool) -> Result<DecodeOutput, String> {
    use std::hint::black_box;

    use fdk_aac::dec::{Decoder, DecoderError, Transport};

    let sample_rate = frames
        .first()
        .map(|frame| frame.sample_rate)
        .ok_or_else(|| "fixture has no ADTS frames".to_string())?;
    let channels = frames
        .first()
        .map(|frame| frame.channels)
        .ok_or_else(|| "fixture has no ADTS frames".to_string())?;
    let mut decoder = Decoder::new(Transport::Adts);
    let mut pcm_i16 = vec![0i16; 16_384];
    let mut pcm = if capture_pcm {
        Vec::with_capacity(frames.len() * 1024 * channels as usize)
    } else {
        Vec::new()
    };
    let mut decoded_frames = 0u64;
    let mut samples_per_channel = 0u64;

    for frame in frames {
        let consumed = decoder
            .fill(frame.full)
            .map_err(|err| format!("fill failed: {err:?}"))?;
        if consumed == 0 {
            return Err("FDK consumed no input bytes".to_string());
        }

        match decoder.decode_frame(&mut pcm_i16) {
            Ok(()) => {
                let info = decoder.stream_info();
                if info.frameSize <= 0 || info.numChannels <= 0 {
                    continue;
                }

                let sample_count = info.frameSize as usize * info.numChannels as usize;
                decoded_frames += 1;
                samples_per_channel += info.frameSize as u64;

                if capture_pcm {
                    pcm.extend(
                        pcm_i16[..sample_count]
                            .iter()
                            .map(|sample| *sample as f32 / 32768.0),
                    );
                } else {
                    black_box(&pcm_i16[..sample_count]);
                }
            }
            Err(err) if err == DecoderError::NOT_ENOUGH_BITS => {}
            Err(err) => return Err(format!("decode failed: {err:?}")),
        }
    }

    Ok(DecodeOutput {
        pcm,
        decoded_frames,
        samples_per_channel,
        sample_rate,
        channels,
    })
}

#[cfg(feature = "soundkit-lc")]
pub fn decode_soundkit_lc_fixture_pcm() -> Result<DecodeOutput, String> {
    decode_soundkit_lc_fixture_pcm_for(DEFAULT_FIXTURE)
}

#[cfg(feature = "soundkit-lc")]
pub fn decode_soundkit_lc_fixture_pcm_for(fixture: AacFixture) -> Result<DecodeOutput, String> {
    let frames = parse_adts_frames(fixture.data)?;
    decode_soundkit_lc_frames(&frames, true)
}

#[cfg(all(
    feature = "soundkit-lc",
    not(any(target_arch = "wasm32", target_arch = "wasm64"))
))]
pub fn bench_soundkit_lc_fixture(iterations: usize) -> Result<BenchResult, String> {
    use std::hint::black_box;
    use std::time::Instant;

    let iterations = iterations.max(1);
    let frames = parse_adts_frames(FIXTURE)?;
    let quality = decode_soundkit_lc_frames(&frames, true)?;
    let sample_rate = frames[0].sample_rate;
    let channels = frames[0].channels;
    let mut decoded_frames = 0u64;
    let mut samples_per_channel = 0u64;
    let started = Instant::now();

    for _ in 0..iterations {
        let decoded = decode_soundkit_lc_frames(&frames, false)?;
        decoded_frames += decoded.decoded_frames;
        samples_per_channel += decoded.samples_per_channel;
        black_box(decoded.decoded_frames);
    }

    Ok(BenchResult {
        name: "soundkit-aac-lc",
        iterations,
        frames: frames.len() as u64 * iterations as u64,
        decoded_frames,
        samples_per_channel,
        sample_rate,
        channels,
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        pcm_stats: quality.stats(),
    })
}

#[cfg(all(
    feature = "soundkit-lc",
    not(any(target_arch = "wasm32", target_arch = "wasm64"))
))]
pub fn bench_soundkit_lc_fixture_reused(iterations: usize) -> Result<BenchResult, String> {
    use std::hint::black_box;
    use std::time::Instant;

    use soundkit_aac_lc::AacLcDecoder;

    let iterations = iterations.max(1);
    let frames = parse_adts_frames(FIXTURE)?;
    let first = frames
        .first()
        .ok_or_else(|| "fixture has no ADTS frames".to_string())?;
    let quality = decode_soundkit_lc_frames(&frames, true)?;
    let sample_rate = first.sample_rate;
    let channels = first.channels;
    let asc = first.audio_specific_config();
    let mut decoder = AacLcDecoder::from_audio_specific_config(&asc)
        .map_err(|err| format!("create SoundKit AAC-LC decoder failed: {err}"))?;
    let mut decoded_frames = 0u64;
    let mut samples_per_channel = 0u64;
    let started = Instant::now();

    for _ in 0..iterations {
        for (frame_index, frame) in frames.iter().enumerate() {
            let decoded = decoder.decode_access_unit(frame.raw).map_err(|err| {
                format!("SoundKit AAC-LC decode failed at frame {frame_index}: {err}")
            })?;
            decoded_frames += 1;
            samples_per_channel += decoded.frames() as u64;
            black_box(decoded.frames());
        }
    }

    Ok(BenchResult {
        name: "soundkit-lc-reuse",
        iterations,
        frames: frames.len() as u64 * iterations as u64,
        decoded_frames,
        samples_per_channel,
        sample_rate,
        channels,
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        pcm_stats: quality.stats(),
    })
}

#[cfg(feature = "soundkit-lc")]
fn decode_soundkit_lc_frames(
    frames: &[AdtsFrame<'_>],
    capture_pcm: bool,
) -> Result<DecodeOutput, String> {
    use std::hint::black_box;

    use soundkit_aac_lc::AacLcDecoder;

    let first = frames
        .first()
        .ok_or_else(|| "fixture has no ADTS frames".to_string())?;
    let sample_rate = first.sample_rate;
    let channels = first.channels;
    let asc = first.audio_specific_config();
    let mut decoder = AacLcDecoder::from_audio_specific_config(&asc)
        .map_err(|err| format!("create SoundKit AAC-LC decoder failed: {err}"))?;
    let mut pcm = if capture_pcm {
        Vec::with_capacity(frames.len() * 1024 * channels as usize)
    } else {
        Vec::new()
    };
    let mut decoded_frames = 0u64;
    let mut samples_per_channel = 0u64;

    for (frame_index, frame) in frames.iter().enumerate() {
        let decoded = decoder.decode_access_unit(frame.raw).map_err(|err| {
            format!("SoundKit AAC-LC decode failed at frame {frame_index}: {err}")
        })?;
        if decoded.channels().len() != channels as usize {
            return Err(format!(
                "SoundKit AAC-LC channel count changed at frame {frame_index}: expected {channels}, got {}",
                decoded.channels().len()
            ));
        }

        decoded_frames += 1;
        samples_per_channel += decoded.frames() as u64;

        if capture_pcm {
            for sample in 0..decoded.frames() {
                for channel in decoded.channels() {
                    pcm.push(channel[sample]);
                }
            }
        } else {
            black_box(decoded.frames());
        }
    }

    Ok(DecodeOutput {
        pcm,
        decoded_frames,
        samples_per_channel,
        sample_rate,
        channels,
    })
}

#[cfg(feature = "symphonia")]
pub fn decode_symphonia_fixture_pcm() -> Result<DecodeOutput, String> {
    decode_symphonia_fixture_once(true)
}

#[cfg(all(
    feature = "symphonia",
    not(any(target_arch = "wasm32", target_arch = "wasm64"))
))]
pub fn bench_symphonia_fixture(iterations: usize) -> Result<BenchResult, String> {
    use std::hint::black_box;
    use std::time::Instant;

    let iterations = iterations.max(1);
    let frames = parse_adts_frames(FIXTURE)?;
    let quality = decode_symphonia_fixture_once(true)?;
    let sample_rate = frames[0].sample_rate;
    let channels = frames[0].channels;
    let mut decoded_frames = 0u64;
    let mut samples_per_channel = 0u64;
    let started = Instant::now();

    for _ in 0..iterations {
        let decoded = decode_symphonia_fixture_once(false)?;
        decoded_frames += decoded.decoded_frames;
        samples_per_channel += decoded.samples_per_channel;
        black_box(decoded.decoded_frames);
    }

    Ok(BenchResult {
        name: "symphonia-aac",
        iterations,
        frames: frames.len() as u64 * iterations as u64,
        decoded_frames,
        samples_per_channel,
        sample_rate,
        channels,
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        pcm_stats: quality.stats(),
    })
}

#[cfg(feature = "symphonia")]
fn decode_symphonia_fixture_once(capture_pcm: bool) -> Result<DecodeOutput, String> {
    use std::hint::black_box;
    use std::io::Cursor;

    use symphonia_codec_aac::{AacDecoder, AdtsReader};
    use symphonia_core::codecs::audio::{AudioDecoder, AudioDecoderOptions};
    use symphonia_core::codecs::CodecParameters;
    use symphonia_core::formats::probe::ProbeableFormat;
    use symphonia_core::formats::TrackType;
    use symphonia_core::io::MediaSourceStream;

    let frames = parse_adts_frames(FIXTURE)?;
    let sample_rate = frames[0].sample_rate;
    let channels = frames[0].channels;
    let cursor = Cursor::new(FIXTURE.to_vec());
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());
    let mut reader = AdtsReader::try_probe_new(mss, Default::default())
        .map_err(|err| format!("create ADTS reader failed: {err}"))?;
    let track = reader
        .default_track(TrackType::Audio)
        .ok_or_else(|| "ADTS reader produced no default track".to_string())?;
    let codec_params = match &track.codec_params {
        Some(CodecParameters::Audio(params)) => params.clone(),
        None => return Err("ADTS track has no codec parameters".to_string()),
        _ => return Err("ADTS track is not audio".to_string()),
    };
    let mut decoder = AacDecoder::try_new(&codec_params, &AudioDecoderOptions::default())
        .map_err(|err| format!("create AAC decoder failed: {err}"))?;
    let mut pcm = if capture_pcm {
        Vec::with_capacity(frames.len() * 1024 * channels as usize)
    } else {
        Vec::new()
    };
    let mut scratch = Vec::<f32>::new();
    let mut decoded_frames = 0u64;
    let mut samples_per_channel = 0u64;

    while let Some(packet) = reader
        .next_packet()
        .map_err(|err| format!("read ADTS packet failed: {err}"))?
    {
        let decoded = decoder
            .decode(&packet)
            .map_err(|err| format!("decode failed: {err}"))?;
        decoded_frames += 1;
        samples_per_channel += decoded.frames() as u64;

        if capture_pcm {
            decoded.copy_to_vec_interleaved::<f32>(&mut scratch);
            pcm.extend_from_slice(&scratch);
        } else {
            black_box(decoded);
        }
    }

    Ok(DecodeOutput {
        pcm,
        decoded_frames,
        samples_per_channel,
        sample_rate,
        channels,
    })
}

#[cfg(all(feature = "fdk", feature = "symphonia"))]
pub fn compare_symphonia_to_fdk() -> Result<QualityComparison, String> {
    let fdk = decode_fdk_fixture_pcm()?;
    let symphonia = decode_symphonia_fixture_pcm()?;
    Ok(QualityComparison::compare_aligned(
        &fdk.pcm,
        &symphonia.pcm,
        fdk.channels as usize,
        2048,
    ))
}

#[cfg(all(feature = "fdk", feature = "soundkit-lc"))]
pub fn compare_soundkit_lc_to_fdk() -> Result<QualityComparison, String> {
    compare_soundkit_lc_to_fdk_for(DEFAULT_FIXTURE)
}

#[cfg(all(feature = "fdk", feature = "soundkit-lc"))]
pub fn compare_soundkit_lc_to_fdk_for(fixture: AacFixture) -> Result<QualityComparison, String> {
    let fdk = decode_fdk_fixture_pcm_for(fixture)?;
    let soundkit = decode_soundkit_lc_fixture_pcm_for(fixture)?;
    Ok(QualityComparison::compare_aligned(
        &fdk.pcm,
        &soundkit.pcm,
        fdk.channels as usize,
        4096,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_fixture_adts_frames_with_raw_access_units() {
        let frames = parse_adts_frames(FIXTURE).unwrap();

        assert_eq!(frames.len(), 9171);
        assert_eq!(frames[0].sample_rate, 48_000);
        assert_eq!(frames[0].channels, 2);
        assert!(frames
            .iter()
            .all(|frame| frame.full.len() > frame.raw.len()));
        assert_eq!(frames[0].audio_specific_config(), [0x11, 0x90]);
        assert!(frames.iter().all(|frame| frame.audio_object_type == 2));
        assert!(frames.iter().all(|frame| frame.sample_rate == 48_000));
        assert!(frames.iter().all(|frame| frame.channels == 2));
    }

    #[test]
    fn parses_24_bit_stereo_pcm_wav() {
        let wav = tiny_wav_24_stereo(&[
            -8_388_608, 8_388_607, //
            0, 4_194_304,
        ]);

        let decoded = decode_wav_pcm_bytes(&wav).unwrap();

        assert_eq!(decoded.sample_rate, 48_000);
        assert_eq!(decoded.channels, 2);
        assert_eq!(decoded.samples_per_channel, 2);
        assert_eq!(decoded.pcm.len(), 4);
        assert!((decoded.pcm[0] + 1.0).abs() < 1.0e-7);
        assert!((decoded.pcm[1] - (8_388_607.0 / 8_388_608.0)).abs() < 1.0e-7);
        assert_eq!(decoded.pcm[2], 0.0);
        assert!((decoded.pcm[3] - 0.5).abs() < 1.0e-7);
    }

    #[cfg(feature = "soundkit-lc")]
    #[test]
    fn soundkit_lc_decodes_fixture_access_units() {
        let decoded = decode_soundkit_lc_fixture_pcm().unwrap();

        assert_eq!(decoded.decoded_frames, 9171);
        assert_eq!(decoded.samples_per_channel, 9171 * 1024);
        assert_eq!(decoded.sample_rate, 48_000);
        assert_eq!(decoded.channels, 2);
        assert!(decoded.pcm.iter().any(|sample| *sample != 0.0));
        assert!(decoded.pcm.iter().all(|sample| sample.is_finite()));
    }

    #[cfg(all(feature = "fdk", feature = "soundkit-lc"))]
    #[test]
    fn soundkit_lc_matches_fdk_fixture_tolerance() {
        let comparison = compare_soundkit_lc_to_fdk().unwrap();

        assert!(
            comparison.passes_default_thresholds(),
            "{}",
            format_quality_comparison("fdk-vs-soundkit", &comparison)
        );
        assert_eq!(comparison.length_delta, 0);
        assert!(comparison.compared_samples >= 9171 * 1024 * 2 - 4096 * 2);
    }

    fn tiny_wav_24_stereo(samples: &[i32]) -> Vec<u8> {
        assert_eq!(samples.len() % 2, 0);
        let data_len = samples.len() * 3;
        let riff_len = 36 + data_len;
        let mut wav = Vec::with_capacity(44 + data_len);
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(riff_len as u32).to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&48_000u32.to_le_bytes());
        wav.extend_from_slice(&(48_000u32 * 2 * 3).to_le_bytes());
        wav.extend_from_slice(&(2u16 * 3).to_le_bytes());
        wav.extend_from_slice(&24u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(data_len as u32).to_le_bytes());
        for sample in samples {
            let clamped = (*sample).clamp(-8_388_608, 8_388_607);
            wav.push((clamped & 0xff) as u8);
            wav.push(((clamped >> 8) & 0xff) as u8);
            wav.push(((clamped >> 16) & 0xff) as u8);
        }
        wav
    }
}
