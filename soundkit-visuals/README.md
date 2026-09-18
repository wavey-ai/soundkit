# SoundKit Visuals

Stereo, frequency-banded waveform summaries for SoundKit streams, computed at
encode time. Part of the [`soundkit`](https://github.com/wavey-ai/soundkit)
workspace.

A player draws a "DJ app" waveform: both channels, split into low, mid and
high, one peak per pixel column. Reading that off decoded PCM at load time is a
whole extra pass over the side, and it is the same side every time. This crate
computes the summary while the side is already being read — the interleaved
PCM handed to the stream encoder — and writes it as a small sidecar. A player
then draws a waveform by reading bytes.

- Per bucket, per channel, per band: one `u8` peak (`0..=255`).
- Bands are a one-pole split at fixed crossovers, so the summary is identical
  on every host.
- `Waveform::encode`/`decode` is a fixed header and the peaks.

```rust
use soundkit_visuals::{compute_waveform, WaveformOptions, DEFAULT_CROSSOVERS_HZ};

let waveform = compute_waveform(&interleaved_i16, 48_000, 2, &WaveformOptions {
    buckets: 2_048,
    crossovers_hz: DEFAULT_CROSSOVERS_HZ.to_vec(),
})?;
let sidecar = waveform.encode();
```
