//! Stereo, frequency-banded waveform summaries for SoundKit streams.
//!
//! A player draws a "DJ app" waveform: both channels, split into low, mid and
//! high, one peak per pixel column. Reading that off the decoded PCM at load
//! time is a whole extra pass over the side, and it is the same side every
//! time — so it belongs where the side is already being read: the encoder.
//!
//! This crate computes the summary from the interleaved PCM the stream
//! encoder is handed (`soundkit::encode_interleaved_i16_to_soundkit_streams`),
//! and writes it as a small sidecar. A player then draws a waveform by reading
//! bytes, not by decoding audio.
//!
//! The summary is deliberately compact and lossless enough to look right:
//! per bucket, per channel, per band, one `u8` peak (`0..=255`). Bands are a
//! one-pole split at fixed crossovers — cheap, deterministic, and identical on
//! every host, so a waveform cut on the phone is the waveform a browser draws.

/// Crossovers that split a full-band signal into low, mid and high.
pub const DEFAULT_CROSSOVERS_HZ: [f64; 2] = [200.0, 2_000.0];
/// The bucket count a full side is summarised at — about a 1080-canvas a
/// second of audio would want, and small enough to hold for every track.
pub const DEFAULT_BUCKETS: usize = 2_048;
const FULL_SCALE: f64 = 32_768.0;
const MAGIC: &[u8; 4] = b"SKWV";
const VERSION: u8 = 1;
const HEADER_LEN: usize = 4 + 1 + 4 + 1 + 1 + 8 + 4;

/// How a waveform is summarised.
#[derive(Clone, Debug, PartialEq)]
pub struct WaveformOptions {
    /// One peak per bucket, per channel, per band.
    pub buckets: usize,
    /// Frequencies that split the bands. `N` crossovers make `N + 1` bands;
    /// empty makes one full-band band.
    pub crossovers_hz: Vec<f64>,
}

impl Default for WaveformOptions {
    fn default() -> Self {
        Self {
            buckets: DEFAULT_BUCKETS,
            crossovers_hz: DEFAULT_CROSSOVERS_HZ.to_vec(),
        }
    }
}

impl WaveformOptions {
    /// The bands this option makes: crossovers plus one.
    pub const fn band_count(&self) -> usize {
        self.crossovers_hz.len() + 1
    }
}

/// A waveform summary: per bucket, per channel, per band, one peak.
#[derive(Clone, Debug, PartialEq)]
pub struct Waveform {
    pub sample_rate: u32,
    pub channels: u8,
    pub band_count: u8,
    pub frame_count: u64,
    pub buckets: u32,
    /// `buckets * channels * band_count` bytes,
    /// `[bucket][channel][band]`.
    pub data: Vec<u8>,
}

impl Waveform {
    /// The peak for one bucket, channel and band, `0..=255`.
    pub fn peak(&self, bucket: usize, channel: usize, band: usize) -> u8 {
        let channels = self.channels as usize;
        let bands = self.band_count as usize;
        if bucket >= self.buckets as usize || channel >= channels || band >= bands {
            return 0;
        }
        self.data[(bucket * channels + channel) * bands + band]
    }

    /// The bytes of the sidecar: a fixed header and the peaks.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(HEADER_LEN + self.data.len());
        out.extend_from_slice(MAGIC);
        out.push(VERSION);
        out.extend_from_slice(&self.sample_rate.to_le_bytes());
        out.push(self.channels);
        out.push(self.band_count);
        out.extend_from_slice(&self.frame_count.to_le_bytes());
        out.extend_from_slice(&self.buckets.to_le_bytes());
        out.extend_from_slice(&self.data);
        out
    }

    /// Reads a sidecar written by [`Waveform::encode`].
    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < HEADER_LEN {
            return Err("a waveform sidecar is shorter than its header".to_owned());
        }
        if &bytes[0..4] != MAGIC {
            return Err("a waveform sidecar does not carry the SoundKit Visuals magic".to_owned());
        }
        if bytes[4] != VERSION {
            return Err(format!("waveform sidecar version {} is unsupported", bytes[4]));
        }
        let sample_rate = u32::from_le_bytes(bytes[5..9].try_into().unwrap());
        let channels = bytes[9];
        let band_count = bytes[10];
        let frame_count = u64::from_le_bytes(bytes[11..19].try_into().unwrap());
        let buckets = u32::from_le_bytes(bytes[19..23].try_into().unwrap());
        let expected = buckets as usize * channels as usize * band_count as usize;
        let data = bytes[HEADER_LEN..].to_vec();
        if data.len() != expected {
            return Err(format!(
                "waveform sidecar holds {} bytes, not the {} its header declares",
                data.len(),
                expected
            ));
        }
        Ok(Self {
            sample_rate,
            channels,
            band_count,
            frame_count,
            buckets,
            data,
        })
    }
}

/// A one-pole low pass, the band split's one primitive.
struct OnePole {
    coefficient: f64,
    state: f64,
}

impl OnePole {
    fn new(cutoff_hz: f64, sample_rate: f64) -> Self {
        let nyquist = sample_rate / 2.0;
        let cutoff = cutoff_hz.max(1.0).min(nyquist - 1.0);
        let coefficient = 1.0 - (-2.0 * std::f64::consts::PI * cutoff / sample_rate).exp();
        Self { coefficient, state: 0.0 }
    }

    fn step(&mut self, value: f64) -> f64 {
        self.state += self.coefficient * (value - self.state);
        self.state
    }
}

/// The per-channel filters a band split runs.
struct BandSplit {
    poles: Vec<OnePole>,
}

impl BandSplit {
    fn new(crossovers_hz: &[f64], sample_rate: f64) -> Self {
        Self {
            poles: crossovers_hz
                .iter()
                .map(|cutoff| OnePole::new(*cutoff, sample_rate))
                .collect(),
        }
    }

    /// Writes one band per output slot: the lowest is the first crossover's
    /// low pass, the middle bands are the span between two crossovers, and the
    /// top band is everything above the last.
    fn step(&mut self, value: f64, bands: &mut [f64]) {
        let mut previous = 0.0;
        for (index, pole) in self.poles.iter_mut().enumerate() {
            let low = pole.step(value);
            bands[index] = low - previous;
            previous = low;
        }
        bands[self.poles.len()] = value - previous;
    }
}

fn validate(channels: u8, sample_rate: u32, options: &WaveformOptions) -> Result<(), String> {
    if channels == 0 || channels > 2 {
        return Err(format!("a waveform needs one or two channels, not {channels}"));
    }
    if sample_rate == 0 {
        return Err("a waveform needs a sample rate".to_owned());
    }
    if options.buckets == 0 {
        return Err("a waveform needs at least one bucket".to_owned());
    }
    if !options.crossovers_hz.iter().all(|value| value.is_finite() && *value > 0.0) {
        return Err("waveform crossovers must be finite and positive".to_owned());
    }
    Ok(())
}

/// Summarises interleaved `i16` PCM into a stereo, banded waveform.
///
/// `pcm` is interleaved by frame: `[left, right, left, right, …]`, or one
/// sample per frame in mono. The summary has `options.buckets` buckets, the
/// last holding the remainder, so the whole side is covered whatever the
/// frame count.
pub fn compute_waveform(
    pcm: &[i16],
    sample_rate: u32,
    channels: u8,
    options: &WaveformOptions,
) -> Result<Waveform, String> {
    validate(channels, sample_rate, options)?;
    let channels_usize = channels as usize;
    let bands = options.band_count();
    let frame_count = pcm.len() / channels_usize;
    let buckets = options.buckets;
    let mut data = vec![0u8; buckets * channels_usize * bands];
    if frame_count == 0 {
        return Ok(Waveform {
            sample_rate,
            channels,
            band_count: bands as u8,
            frame_count: 0,
            buckets: buckets as u32,
            data,
        });
    }

    let rate = sample_rate as f64;
    let mut splits: Vec<BandSplit> = (0..channels_usize)
        .map(|_| BandSplit::new(&options.crossovers_hz, rate))
        .collect();
    let mut band_values = vec![0.0f64; bands];
    let frames_per_bucket = (frame_count as f64 / buckets as f64).max(1.0);

    for frame in 0..frame_count {
        let bucket = ((frame as f64 / frames_per_bucket) as usize).min(buckets - 1);
        for channel in 0..channels_usize {
            let sample = pcm[frame * channels_usize + channel] as f64 / FULL_SCALE;
            let split = &mut splits[channel];
            split.step(sample, &mut band_values);
            for band in 0..bands {
                let at = (bucket * channels_usize + channel) * bands + band;
                let level = (band_values[band].abs() * 255.0).min(255.0).round() as u8;
                if level > data[at] {
                    data[at] = level;
                }
            }
        }
    }

    Ok(Waveform {
        sample_rate,
        channels,
        band_count: bands as u8,
        frame_count: frame_count as u64,
        buckets: buckets as u32,
        data,
    })
}

/// A convenience for the common case: a stereo, three-band summary.
pub fn compute_stereo_waveform(
    pcm: &[i16],
    sample_rate: u32,
    options: &WaveformOptions,
) -> Result<Waveform, String> {
    compute_waveform(pcm, sample_rate, 2, options)
}

/// The buckets a streaming summary is cut at, per second of audio. Twenty
/// milliseconds is finer than a drawn pixel and coarser than a frame, so the
/// sidecar is a few kilobytes a minute and the player downsamples freely.
pub const DEFAULT_BUCKETS_PER_SECOND: f64 = 50.0;

/// A streaming waveform summary.
///
/// The Library encoder reads a source in bounded passes and never holds the
/// whole side, so the summary is accumulated as the PCM goes past: one bucket
/// every `1 / buckets_per_second`, peaks taken as they come. [`finish`] gives
/// the same [`Waveform`] [`compute_waveform`] would, without a second pass.
///
/// [`finish`]: WaveformAccumulator::finish
pub struct WaveformAccumulator {
    sample_rate: u32,
    channels: u8,
    bands: usize,
    frames_per_bucket: f64,
    splits: Vec<BandSplit>,
    band_values: Vec<f64>,
    data: Vec<u8>,
    /// Frames seen in the bucket being filled, and its index.
    bucket_frames: usize,
    bucket: usize,
    frame_count: u64,
}

impl WaveformAccumulator {
    /// Opens a summary. `sample_rate` and `channels` are the normalized PCM's
    /// geometry — bitneedle's Library import hands over 48 kHz stereo.
    pub fn new(sample_rate: u32, channels: u8, crossovers_hz: &[f64]) -> Result<Self, String> {
        let options = WaveformOptions {
            buckets: 1,
            crossovers_hz: crossovers_hz.to_vec(),
        };
        validate(channels, sample_rate, &options)?;
        let bands = options.band_count();
        let frames_per_bucket =
            (sample_rate as f64 / DEFAULT_BUCKETS_PER_SECOND).max(1.0);
        Ok(Self {
            sample_rate,
            channels,
            bands,
            frames_per_bucket,
            splits: (0..channels as usize)
                .map(|_| BandSplit::new(crossovers_hz, sample_rate as f64))
                .collect(),
            band_values: vec![0.0; bands],
            data: Vec::new(),
            bucket_frames: 0,
            bucket: 0,
            frame_count: 0,
        })
    }

    /// The three-band default, for the common case.
    pub fn stereo(sample_rate: u32) -> Result<Self, String> {
        Self::new(sample_rate, 2, &DEFAULT_CROSSOVERS_HZ)
    }

    /// Adds interleaved PCM: `[left, right, …]`, or one sample a frame in
    /// mono. A chunk may end mid-bucket; the bucket carries on next chunk.
    pub fn push_interleaved(&mut self, pcm: &[i16]) {
        let channels = self.channels as usize;
        let bands = self.bands;
        for frame in pcm.chunks_exact(channels) {
            let per_bucket = self.frames_per_bucket.max(1.0);
            if self.bucket_frames as f64 >= per_bucket {
                self.bucket_frames = 0;
                self.bucket += 1;
            }
            let at = self.bucket * channels * bands;
            if self.data.len() < at + channels * bands {
                self.data.resize(at + channels * bands, 0);
            }
            for channel in 0..channels {
                let sample = frame[channel] as f64 / FULL_SCALE;
                let split = &mut self.splits[channel];
                split.step(sample, &mut self.band_values);
                for band in 0..bands {
                    let slot = at + channel * bands + band;
                    let level = (self.band_values[band].abs() * 255.0).min(255.0).round() as u8;
                    if level > self.data[slot] {
                        self.data[slot] = level;
                    }
                }
            }
            self.bucket_frames += 1;
            self.frame_count += 1;
        }
    }

    /// The summary so far, as one waveform. Safe to call more than once; a
    /// later push after `finish` simply adds buckets.
    pub fn finish(&self) -> Waveform {
        let buckets = if self.data.is_empty() {
            0
        } else {
            self.data.len() / (self.channels as usize * self.bands)
        };
        Waveform {
            sample_rate: self.sample_rate,
            channels: self.channels,
            band_count: self.bands as u8,
            frame_count: self.frame_count,
            buckets: buckets as u32,
            data: self.data.clone(),
        }
    }

    /// Whether any audio has been pushed.
    pub fn is_empty(&self) -> bool {
        self.frame_count == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options(buckets: usize) -> WaveformOptions {
        WaveformOptions { buckets, crossovers_hz: DEFAULT_CROSSOVERS_HZ.to_vec() }
    }

    #[test]
    fn silence_is_all_zero() {
        let pcm = vec![0i16; 48_000 * 2];
        let waveform = compute_waveform(&pcm, 48_000, 2, &options(64)).unwrap();
        assert_eq!(waveform.buckets, 64);
        assert_eq!(waveform.band_count, 3);
        assert!(waveform.data.iter().all(|level| *level == 0));
    }

    #[test]
    fn full_scale_is_loud_across_the_bands() {
        // A tone in every band, at full scale.
        let pcm: Vec<i16> = (0..48_000 * 2)
            .map(|index| {
                let frame = index / 2;
                let t = frame as f64 / 48_000.0;
                let mix = (t * 50.0 * std::f64::consts::TAU).sin()
                    + (t * 1_000.0 * std::f64::consts::TAU).sin()
                    + (t * 8_000.0 * std::f64::consts::TAU).sin();
                (mix / 3.0 * 32_000.0) as i16
            })
            .collect();
        let waveform = compute_waveform(&pcm, 48_000, 2, &options(8)).unwrap();
        for band in 0..3 {
            assert!(waveform.peak(4, 0, band) > 80, "band {band} was quiet");
        }
    }

    #[test]
    fn a_low_tone_lands_in_the_low_band() {
        // 50 Hz is well under the first crossover.
        let pcm: Vec<i16> = (0..48_000 * 2)
            .map(|index| {
                let frame = index / 2;
                ((frame as f64 / 48_000.0 * 50.0 * std::f64::consts::TAU).sin() * 30_000.0) as i16
            })
            .collect();
        let waveform = compute_waveform(&pcm, 48_000, 2, &options(16)).unwrap();
        let low = waveform.peak(8, 0, 0);
        let high = waveform.peak(8, 0, 2);
        assert!(low > 100, "the low band was quiet: {low}");
        assert!(high < 40, "the high band carried a low tone: {high}");
    }

    #[test]
    fn a_high_tone_lands_in_the_high_band() {
        // 8 kHz is well over the last crossover.
        let pcm: Vec<i16> = (0..48_000 * 2)
            .map(|index| {
                let frame = index / 2;
                ((frame as f64 / 48_000.0 * 8_000.0 * std::f64::consts::TAU).sin() * 30_000.0) as i16
            })
            .collect();
        let waveform = compute_waveform(&pcm, 48_000, 2, &options(16)).unwrap();
        let low = waveform.peak(8, 0, 0);
        let high = waveform.peak(8, 0, 2);
        assert!(high > 100, "the high band was quiet: {high}");
        assert!(low < 40, "the low band carried a high tone: {low}");
    }

    #[test]
    fn the_channels_are_kept_apart() {
        // The left channel only.
        let pcm: Vec<i16> = (0..48_000 * 2)
            .map(|index| if index % 2 == 0 { 30_000 } else { 0 })
            .collect();
        let waveform = compute_waveform(&pcm, 48_000, 2, &options(8)).unwrap();
        assert!(waveform.peak(4, 0, 0) > 100);
        assert_eq!(waveform.peak(4, 1, 0), 0);
    }

    #[test]
    fn a_sidecar_round_trips() {
        let pcm = vec![12_000i16; 48_000 * 2];
        let waveform = compute_waveform(&pcm, 44_100, 2, &options(32)).unwrap();
        let bytes = waveform.encode();
        let decoded = Waveform::decode(&bytes).unwrap();
        assert_eq!(decoded, waveform);
    }

    #[test]
    fn a_truncated_sidecar_is_refused() {
        assert!(Waveform::decode(&[b'S', b'K', b'W', b'V']).is_err());
        let pcm = vec![100i16; 1_000 * 2];
        let mut bytes = compute_waveform(&pcm, 48_000, 2, &options(8)).unwrap().encode();
        bytes.truncate(bytes.len() - 1);
        assert!(Waveform::decode(&bytes).is_err());
    }

    #[test]
    fn a_bad_geometry_is_refused() {
        assert!(compute_waveform(&[], 48_000, 3, &options(8)).is_err());
        assert!(compute_waveform(&[], 0, 2, &options(8)).is_err());
        assert!(compute_waveform(&[], 48_000, 2, &options(0)).is_err());
    }
}
