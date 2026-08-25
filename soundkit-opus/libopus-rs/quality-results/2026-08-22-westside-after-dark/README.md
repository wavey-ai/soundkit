# CELT Real-Music Quality Result: 2026-08-22

Official ViSQOL Audio found no material overall quality gap between this crate
and libopus 1.6.1 on this corpus.

Rust scored `4.5840` MOS-LQO. C scored `4.5850`. The paired Rust-minus-C
difference was `-0.0010`. Its 95% confidence interval was `-0.0019` to
`-0.0001`.

This result is a checkpoint, not proof of transparency. It does not replace a
controlled listening test.

## Scope

The test used five repeatable random eight-second excerpts from each source.
The seed was `20260822`.

Both sources were 48 kHz stereo PCM24 masters:

| Source | Duration | SHA-256 |
|---|---:|---|
| `WESTSIDE_MIX 4 CONFIRMATION_130323.wav` | 195.625 s | `dec1b383d58aea9848728126efab169f42e54375b64ca55363d2f234696474c9` |
| `AFTER DARK_MIX 4 CONFIRMATION_130323.wav` | 198.571 s | `1812df5e6062549a4947a78b5bb08474b6d11026292c019c2da9b5b1950bc655` |

The matrix covered 96, 128, and 192 kb/s. It used 5 and 20 ms frames with CBR
and constrained VBR.

Each implementation produced 120 matched round trips. The comparison therefore
contains 120 paired measurements and 240 ViSQOL measurements.

## Results

| Metric | Result |
|---|---:|
| Rust mean MOS-LQO | 4.5840 |
| C mean MOS-LQO | 4.5850 |
| Rust minus C | -0.0010 |
| 95% confidence interval | -0.0019 to -0.0001 |
| Pair wins | Rust 58, C 62 |

| Configuration | Rust MOS | C MOS | Rust - C | 95% CI | Rust SNR | C SNR | Rust side SNR | C side SNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cbr_5ms_96k` | 4.4306 | 4.4311 | -0.0005 | -0.0035 to +0.0024 | 18.31 dB | 18.32 dB | 10.57 dB | 10.56 dB |
| `cbr_5ms_128k` | 4.5591 | 4.5592 | -0.0001 | -0.0012 to +0.0011 | 20.85 dB | 20.86 dB | 13.19 dB | 13.18 dB |
| `cbr_5ms_192k` | 4.7203 | 4.7202 | +0.0001 | -0.0002 to +0.0003 | 23.95 dB | 23.95 dB | 16.92 dB | 16.91 dB |
| `cbr_20ms_96k` | 4.4361 | 4.4359 | +0.0002 | -0.0001 to +0.0006 | 18.70 dB | 18.69 dB | 10.40 dB | 10.40 dB |
| `cbr_20ms_128k` | 4.6451 | 4.6448 | +0.0003 | -0.0001 to +0.0007 | 20.66 dB | 20.65 dB | 12.46 dB | 12.46 dB |
| `cbr_20ms_192k` | 4.7234 | 4.7235 | -0.0000 | -0.0002 to +0.0001 | 23.17 dB | 23.17 dB | 15.75 dB | 15.75 dB |
| `vbr_5ms_96k` | 4.4341 | 4.4345 | -0.0004 | -0.0050 to +0.0041 | 18.46 dB | 18.53 dB | 10.69 dB | 10.73 dB |
| `vbr_5ms_128k` | 4.5622 | 4.5622 | -0.0000 | -0.0018 to +0.0018 | 20.86 dB | 20.92 dB | 13.17 dB | 13.25 dB |
| `vbr_5ms_192k` | 4.7202 | 4.7204 | -0.0001 | -0.0011 to +0.0008 | 23.91 dB | 23.96 dB | 16.84 dB | 16.91 dB |
| `vbr_20ms_96k` | 4.4406 | 4.4396 | +0.0010 | +0.0000 to +0.0019 | 18.63 dB | 18.67 dB | 10.29 dB | 10.34 dB |
| `vbr_20ms_128k` | 4.6139 | 4.6258 | -0.0119 | -0.0194 to -0.0044 | 20.64 dB | 20.65 dB | 12.47 dB | 12.48 dB |
| `vbr_20ms_192k` | 4.7230 | 4.7230 | -0.0000 | -0.0003 to +0.0002 | 23.11 dB | 23.14 dB | 15.63 dB | 15.69 dB |

CBR showed no material difference. The only repeatable signal was 20 ms
constrained VBR at 128 kb/s. Its mean difference was `-0.0119` MOS-LQO.

The per-track Rust-minus-C differences were `-0.0008` for *Westside* and
`-0.0011` for *After Dark Confirmation*.

## Method

The harness converted signed 24-bit input exactly to `f32` codec samples.
It called both codecs through direct native interfaces.

The control used libopus 1.6.1 through C. It did not use the FFmpeg command-line
codec.

Each decoded result had the same 120-sample alignment offset. The harness
removed this 2.5 ms offset before scoring.

The harness made matched PCM16 ViSQOL inputs with shared gain. This conversion
was only for the metric. It did not change the codec input.

Official ViSQOL Audio used commit
`38d0b0163e441047d4429bf07ad09e5b9031d02c`. ViSQOL Audio downmixes stereo to
mono. Separate side-channel SNR checks found no material stereo difference.

Effective bitrate differences were at most `0.006` kb/s. Bitrate differences
do not explain the 128 kb/s VBR result.

The tested Rust revision was
`91217a5bb431bd2bfbecb6a42dd75f6980384e3c`, with uncommitted codec work.
`report.json` records the exact binary and harness hashes.

## Included Data

- `report.json` contains settings, source hashes, excerpts, packet data,
  diagnostics, ViSQOL scores, and summaries.
- `visqol-results.csv` contains all 240 raw ViSQOL result rows and feature
  values.
- `pairs.csv` lists every reference and degraded-file pairing.
- `blind-manifest.json` lists the selected listening cases.
- `blind-answers.json` contains the hidden A/B assignments and scores.
- `files.sha256` verifies the included result files.

The repository does not include source or generated audio. Paths in the data
are portable logical identifiers, not local file-system paths.

Do not read `blind-answers.json` before a blind listening session. Rebuild the
audio with `tools/run_visqol_quality.py` when the source masters are available.

## Limits

The corpus contains two mastered music tracks and ten excerpts. It does not
cover speech, packet loss, lower rates, surround audio, or all music genres.

ViSQOL results support regression decisions. They cannot establish subjective
transparency by themselves.
