/// Test utilities for audio codec tests
/// Provides waveform visualization and audio comparison helpers

const WAVEFORM_WIDTH: usize = 60;
const WAVEFORM_HEIGHT: usize = 8;

/// Result from decoding
pub struct DecodeResult {
    pub bytes: usize,
    pub sample_count: usize,
    pub sample_rate: u32,
    pub channels: u8,
    pub rms: f64,
    pub waveform: Vec<f32>,
}

impl DecodeResult {
    pub fn new(samples: &[i16], sample_rate: u32, channels: u8) -> Self {
        let sample_count = samples.len();
        let bytes = sample_count * 2;

        // Calculate RMS
        let mut sum_of_squares = 0.0f64;
        for &sample in samples {
            let normalized = sample as f64 / 32768.0;
            sum_of_squares += normalized * normalized;
        }
        let rms = if sample_count > 0 {
            (sum_of_squares / sample_count as f64).sqrt()
        } else {
            0.0
        };

        // Compute waveform peaks
        let waveform = compute_waveform_peaks_i16(samples, WAVEFORM_WIDTH * 2);

        Self {
            bytes,
            sample_count,
            sample_rate,
            channels,
            rms,
            waveform,
        }
    }

    /// Create from i32 samples with specified bits per sample (e.g., 16 for FLAC 16-bit)
    pub fn from_i32(samples: &[i32], sample_rate: u32, channels: u8) -> Self {
        // Default to 32-bit range
        Self::from_i32_with_bits(samples, sample_rate, channels, 32)
    }

    /// Create from i32 samples with explicit bits_per_sample for proper normalization
    /// (FLAC stores 16-bit samples as i32 without scaling to full 32-bit range)
    pub fn from_i32_with_bits(
        samples: &[i32],
        sample_rate: u32,
        channels: u8,
        bits_per_sample: u8,
    ) -> Self {
        let sample_count = samples.len();
        let bytes = sample_count * 4;

        // Calculate the normalization factor based on actual bit depth
        let max_value = (1i64 << (bits_per_sample - 1)) as f64;

        // Calculate RMS
        let mut sum_of_squares = 0.0f64;
        for &sample in samples {
            let normalized = sample as f64 / max_value;
            sum_of_squares += normalized * normalized;
        }
        let rms = if sample_count > 0 {
            (sum_of_squares / sample_count as f64).sqrt()
        } else {
            0.0
        };

        // Compute waveform peaks with proper bit depth
        let waveform = compute_waveform_peaks_i32_bits(samples, WAVEFORM_WIDTH * 2, bits_per_sample);

        Self {
            bytes,
            sample_count,
            sample_rate,
            channels,
            rms,
            waveform,
        }
    }

    pub fn duration_secs(&self) -> f64 {
        if self.sample_rate == 0 || self.channels == 0 {
            return 0.0;
        }
        self.sample_count as f64 / self.channels as f64 / self.sample_rate as f64
    }

    pub fn rms_db(&self) -> f64 {
        if self.rms > 0.0 {
            20.0 * self.rms.log10()
        } else {
            -96.0
        }
    }
}

/// Compute waveform peaks from i16 samples for visualization
fn compute_waveform_peaks_i16(samples: &[i16], num_bins: usize) -> Vec<f32> {
    if samples.is_empty() || num_bins == 0 {
        return Vec::new();
    }

    let bin_size = (samples.len() + num_bins - 1) / num_bins;

    samples
        .chunks(bin_size)
        .map(|chunk| {
            let max_abs = chunk.iter().map(|&s| (s as f32).abs()).fold(0.0f32, f32::max);
            max_abs / 32768.0
        })
        .collect()
}

/// Compute waveform peaks from i32 samples with explicit bit depth
fn compute_waveform_peaks_i32_bits(samples: &[i32], num_bins: usize, bits_per_sample: u8) -> Vec<f32> {
    if samples.is_empty() || num_bins == 0 {
        return Vec::new();
    }

    let max_value = (1i64 << (bits_per_sample - 1)) as f64;
    let bin_size = (samples.len() + num_bins - 1) / num_bins;

    samples
        .chunks(bin_size)
        .map(|chunk| {
            let max_abs = chunk.iter().map(|&s| (s as f64).abs()).fold(0.0f64, f64::max);
            (max_abs / max_value) as f32
        })
        .collect()
}

/// Print ASCII waveform comparison for multiple decoded results
pub fn print_waveform_comparison(results: &[(&str, &DecodeResult)]) {
    if results.is_empty() {
        return;
    }

    println!();
    println!("  Decoded Audio Waveforms");
    println!("  {}", "═".repeat(70));
    println!();

    for (name, result) in results {
        println!(
            "  {} ({:.2}s, {} Hz, {} ch, {:.1} dB)",
            name,
            result.duration_secs(),
            result.sample_rate,
            result.channels,
            result.rms_db()
        );
        print_waveform(&result.waveform);
        println!();
    }
}

/// Print a single ASCII waveform with the format name
pub fn print_waveform_with_header(name: &str, result: &DecodeResult) {
    println!();
    println!("  {} Decode Test", name);
    println!("  {}", "─".repeat(50));
    println!(
        "  Duration: {:.2}s | Sample Rate: {} Hz | Channels: {} | RMS: {:.1} dB",
        result.duration_secs(),
        result.sample_rate,
        result.channels,
        result.rms_db()
    );
    print_waveform(&result.waveform);
    println!();
}

/// Print a single ASCII waveform
fn print_waveform(peaks: &[f32]) {
    if peaks.is_empty() {
        println!("  (no audio data)");
        return;
    }

    // Characters for different amplitude levels
    let chars = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

    // Resample peaks to fit display width
    let display_peaks: Vec<f32> = if peaks.len() > WAVEFORM_WIDTH {
        (0..WAVEFORM_WIDTH)
            .map(|i| {
                let start = i * peaks.len() / WAVEFORM_WIDTH;
                let end = ((i + 1) * peaks.len() / WAVEFORM_WIDTH).min(peaks.len());
                peaks[start..end]
                    .iter()
                    .map(|x| x.abs())
                    .fold(0.0f32, f32::max)
            })
            .collect()
    } else {
        peaks.iter().map(|x| x.abs()).collect()
    };

    // Find max for normalization
    let max_peak = display_peaks.iter().fold(0.0f32, |a, &b| a.max(b)).max(0.001);

    let half_height = WAVEFORM_HEIGHT / 2;

    // Top half (positive)
    for row in (0..half_height).rev() {
        let threshold = (row as f32 + 0.5) / half_height as f32;
        let line: String = display_peaks
            .iter()
            .map(|&p| {
                let normalized = p / max_peak;
                if normalized >= threshold {
                    let level =
                        ((normalized - threshold) * half_height as f32 * (chars.len() - 1) as f32)
                            as usize;
                    chars[level.min(chars.len() - 1)]
                } else {
                    ' '
                }
            })
            .collect();
        println!("  │{}│", line);
    }

    // Center line
    println!("  ├{}┤", "─".repeat(display_peaks.len()));

    // Bottom half (mirrored)
    for row in 0..half_height {
        let threshold = (row as f32 + 0.5) / half_height as f32;
        let line: String = display_peaks
            .iter()
            .map(|&p| {
                let normalized = p / max_peak;
                if normalized >= threshold {
                    let level =
                        ((normalized - threshold) * half_height as f32 * (chars.len() - 1) as f32)
                            as usize;
                    chars[level.min(chars.len() - 1)]
                } else {
                    ' '
                }
            })
            .collect();
        println!("  │{}│", line);
    }
}
