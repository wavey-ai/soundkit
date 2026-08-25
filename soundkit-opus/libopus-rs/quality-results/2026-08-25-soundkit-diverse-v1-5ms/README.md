# 5 ms High-Rate Perceptual Quality Result: 2026-08-25

Official ViSQOL Audio found no detectable overall quality difference between
`libopus-rs` and trunk libopus in this test.

Rust scored `4.5731` MOS-LQO. C scored `4.5724`. The paired Rust-minus-C
difference was `+0.0006`.

The excerpt-level 95% confidence interval was `-0.0012` to `+0.0024`.
This analysis treats each of the 20 source excerpts as one independent unit.

All 12 configuration intervals include zero. The test did not find a repeatable
advantage for either implementation.

This result shows parity with trunk libopus. It does not prove transparency.

## Scope

The test used the SoundKit `diverse-v1` corpus. The corpus contains four
100-second, 48 kHz stereo, signed 24-bit PCM sources.

The harness selected five nonoverlapping eight-second excerpts from each
source. It used seed `20260825` and a five-second edge margin.

The test matrix was:

| Item | Values |
|---|---|
| Application | Audio |
| Frame size | 240 samples per channel, 5 ms |
| Rates | 192, 256, and 320 kb/s |
| Rate modes | CBR and constrained VBR |
| Codec PCM APIs | Signed 16-bit and signed 24-bit |
| Excerpts | 20 |
| Configurations per excerpt | 12 |
| Matched Rust/C pairs | 240 |
| ViSQOL measurements | 480 |

The sources were:

| Source | Duration | SHA-256 |
|---|---:|---|
| `bill-evans-secret-sessions` | 100.000 s | `e3c96fffa05eefd2c8d96fbf71b31be295b5890cb70c8fb9d73356a373b21281` |
| `blue-nile-hats` | 100.000 s | `a5ea40b18c8717c2d51772255ad4afaec674abd19c2a5d32672ddbedacf12a95` |
| `lori-asha` | 100.000 s | `7d06b937e7012fb7fc701f82c6ec8d69a8ec8cd9dac9c1f1055d48c192c77b2e` |
| `nocturnal-animals` | 100.000 s | `b961127010d0d576ab430b3101953a86c708dc851266535ffaee4ed8a3dd8435` |

## Headline Results

| Metric | Result |
|---|---:|
| Rust mean MOS-LQO | 4.5731 |
| C mean MOS-LQO | 4.5724 |
| Rust minus C | +0.0006 |
| Excerpt-level 95% confidence interval | -0.0012 to +0.0024 |
| Pair wins | Rust 124, C 116 |
| Full-channel mean SNR | Rust 27.9056 dB, C 27.9051 dB |

The grouped intervals below also use excerpt-level differences. This method
does not count repeated configurations as independent source material.

| Rate | Rust MOS | C MOS | Rust - C | Excerpt-level 95% CI |
|---:|---:|---:|---:|---:|
| 192 kb/s | 4.5368 | 4.5349 | +0.0019 | -0.0010 to +0.0048 |
| 256 kb/s | 4.5848 | 4.5848 | +0.0000 | -0.0023 to +0.0024 |
| 320 kb/s | 4.5975 | 4.5976 | -0.0001 | -0.0011 to +0.0009 |

| PCM API | Rust MOS | C MOS | Rust - C | Excerpt-level 95% CI |
|---:|---:|---:|---:|---:|
| Signed 16-bit | 4.5730 | 4.5723 | +0.0007 | -0.0010 to +0.0023 |
| Signed 24-bit | 4.5731 | 4.5725 | +0.0006 | -0.0014 to +0.0026 |

| Rate mode | Rust MOS | C MOS | Rust - C | Excerpt-level 95% CI |
|---|---:|---:|---:|---:|
| CBR | 4.5729 | 4.5723 | +0.0006 | -0.0011 to +0.0022 |
| Constrained VBR | 4.5732 | 4.5725 | +0.0007 | -0.0014 to +0.0027 |

## Absolute Quality by Source and Rate

The following table shows the Rust MOS-LQO score. Absolute ViSQOL values depend
on the source, so they are not a transparency threshold.

| Source | 192 kb/s | 256 kb/s | 320 kb/s |
|---|---:|---:|---:|
| `bill-evans-secret-sessions` | 4.5442 | 4.5752 | 4.5777 |
| `blue-nile-hats` | 4.7130 | 4.7215 | 4.7233 |
| `lori-asha` | 4.7205 | 4.7268 | 4.7284 |
| `nocturnal-animals` | 4.1695 | 4.3159 | 4.3607 |

The average Rust score increased from `4.5368` at 192 kb/s to `4.5975` at
320 kb/s. The difficult `nocturnal-animals` source had the largest rate benefit.

The 192 kb/s result is reference parity, not proof of transparency. Controlled
listening must decide whether its remaining artifacts are acceptable.

## Configuration Results

Each configuration interval uses 20 paired excerpt differences.

| Configuration | Rust MOS | C MOS | Rust - C | 95% CI | Rust SNR | C SNR | Rust side SNR | C side SNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `i16_cbr_5ms_192k` | 4.5369 | 4.5340 | +0.0029 | -0.0003 to +0.0061 | 25.98 dB | 25.96 dB | 22.61 dB | 22.60 dB |
| `i16_cbr_5ms_256k` | 4.5844 | 4.5849 | -0.0005 | -0.0031 to +0.0021 | 28.35 dB | 28.37 dB | 25.44 dB | 25.47 dB |
| `i16_cbr_5ms_320k` | 4.5976 | 4.5974 | +0.0002 | -0.0009 to +0.0014 | 29.42 dB | 29.40 dB | 27.71 dB | 27.80 dB |
| `i16_vbr_5ms_192k` | 4.5362 | 4.5352 | +0.0010 | -0.0025 to +0.0044 | 25.93 dB | 25.95 dB | 22.57 dB | 22.60 dB |
| `i16_vbr_5ms_256k` | 4.5852 | 4.5848 | +0.0004 | -0.0017 to +0.0026 | 28.29 dB | 28.31 dB | 25.29 dB | 25.39 dB |
| `i16_vbr_5ms_320k` | 4.5977 | 4.5977 | -0.0000 | -0.0012 to +0.0011 | 29.42 dB | 29.41 dB | 27.71 dB | 27.79 dB |
| `i24_cbr_5ms_192k` | 4.5367 | 4.5349 | +0.0018 | -0.0020 to +0.0056 | 25.98 dB | 25.97 dB | 22.60 dB | 22.63 dB |
| `i24_cbr_5ms_256k` | 4.5843 | 4.5851 | -0.0008 | -0.0036 to +0.0021 | 28.38 dB | 28.38 dB | 25.50 dB | 25.55 dB |
| `i24_cbr_5ms_320k` | 4.5975 | 4.5978 | -0.0003 | -0.0015 to +0.0010 | 29.45 dB | 29.41 dB | 27.82 dB | 27.89 dB |
| `i24_vbr_5ms_192k` | 4.5374 | 4.5355 | +0.0020 | -0.0020 to +0.0059 | 25.92 dB | 25.95 dB | 22.55 dB | 22.60 dB |
| `i24_vbr_5ms_256k` | 4.5855 | 4.5845 | +0.0010 | -0.0015 to +0.0034 | 28.31 dB | 28.33 dB | 25.37 dB | 25.45 dB |
| `i24_vbr_5ms_320k` | 4.5972 | 4.5975 | -0.0003 | -0.0020 to +0.0013 | 29.44 dB | 29.43 dB | 27.80 dB | 27.90 dB |

## Stereo Diagnostics

ViSQOL Audio downmixes stereo to mono. The harness also measured full-channel,
mid-channel, and side-channel SNR before the ViSQOL PCM16 conversion.

Rust had `27.9056` dB mean full-channel SNR. C had `27.9051` dB.

The Bill Evans source is dual mono, so side-channel SNR is not defined for that
source. The other sources produced 180 side-channel measurements.

Rust had `25.2483` dB mean side-channel SNR. C had `25.3040` dB. The mean
Rust-minus-C difference was `-0.0557` dB.

The largest paired side-channel difference was `-0.4379` dB. It occurred in
`blue-nile-hats`, excerpt 4, at signed 24-bit, 256 kb/s constrained VBR.

These diagnostics do not show a large aggregate stereo difference. They do not
replace stereo blind listening.

## Packet Rate and Delay

Both implementations produced the exact requested byte count in every CBR
case.

Across constrained-VBR pairs, the mean Rust-minus-C rate difference was
`+0.0157` kb/s. The maximum absolute difference was `0.1990` kb/s.

The harness found a 120-sample decoded-signal offset for every Rust result. It
found a 312-sample offset for every C result.

These offsets are 2.5 ms and 6.5 ms at 48 kHz. The harness removed each offset
before it made matched evaluation files.

These values describe decoded-signal alignment. They are not a complete system
latency measurement.

## Method

The harness called both codecs through direct native APIs. It did not use
FFmpeg for encoding or decoding.

Rust used `Application::Audio`. The current Rust encoder uses its complete
CELT analysis path.

C used `OPUS_APPLICATION_AUDIO`, complexity 10, fullband bandwidth, and the
music signal setting. The C control was trunk libopus
`1.6.1-51-g03647f52` at commit
`03647f524a40b05a1898522e92033810b58103c7`.

For the signed 16-bit test, the harness shifted each signed 24-bit sample right
by eight bits. It gave the same result to both codecs.

For the signed 24-bit test, the harness gave each codec the exact sign-extended
`i32` sample. Rust used `encode_i24` and `decode_i24`. C used `opus_encode24`
and `opus_decode24`.

The harness made matched PCM16 ViSQOL files with `0.1` dB headroom. This
conversion occurred after codec decoding and alignment.

Official ViSQOL Audio used commit
`38d0b0163e441047d4429bf07ad09e5b9031d02c`. The run used NumPy `1.26.4` and
SoundFile `0.12.1`.

The tested Rust codec revision was
`1ce6ec6ae3611611ab3c630b1e78ab390e063897`. The worktree contained only the
new quality harness changes. `report.json` records all binary and harness
hashes.

The harness made 720 evaluation WAV files. Every file had 383,688 stereo frames
at 48 kHz in PCM16 format.

## Statistical Method

Every MOS difference is Rust minus C for the same source excerpt and
configuration.

The overall interval first averaged all 12 configuration differences for each
excerpt. It then calculated a two-sided 95% Student interval across 20 excerpt
means.

The grouped rate, PCM, and mode intervals use the same excerpt-level method.
Each configuration interval uses its 20 paired excerpts directly.

The pair-level interval in `report.json` treats all 240 pairs separately. The
excerpt-level interval in this document is the more conservative headline.

## Reproduction

Build the Rust round-trip tool from the tested checkout:

```sh
RUSTFLAGS='-C target-cpu=native' \
  cargo build --release --example raw_pcm_roundtrip
```

Build the C control against the tested trunk libopus checkout:

```sh
cc -std=c11 -O3 -DNDEBUG -march=native -Wall -Wextra -Werror \
  -I/path/to/libopus/include tools/raw_pcm_roundtrip.c \
  /path/to/libopus/.libs/libopus.a -lm -o target/raw_pcm_roundtrip_c
```

Run the matrix after you build official ViSQOL Audio:

```sh
CORPUS_DIR=/path/to/soundkit/testdata/flac-packet-bench/diverse-v1

python3 tools/run_visqol_quality.py \
  --track bill-evans-secret-sessions="$CORPUS_DIR/bill-evans-secret-sessions-48k-s24.s32le" \
  --track blue-nile-hats="$CORPUS_DIR/blue-nile-hats-48k-s24.s32le" \
  --track lori-asha="$CORPUS_DIR/lori-asha-48k-s24.s32le" \
  --track nocturnal-animals="$CORPUS_DIR/nocturnal-animals-48k-s24.s32le" \
  --rust-bin target/release/examples/raw_pcm_roundtrip \
  --c-bin target/raw_pcm_roundtrip_c \
  --libopus-source /path/to/libopus \
  --visqol /path/to/visqol/bazel-bin/visqol \
  --out-dir quality-output \
  --seed 20260825 --excerpts-per-track 5 --excerpt-seconds 8 \
  --frame-sizes 240 --bitrates 192000,256000,320000 \
  --modes cbr,vbr --pcm-bits 16,24 --visqol-jobs 4 \
  --blind-cases 20
```

## Included Data

- `report.json` contains settings, source hashes, excerpt positions, packet
  data, diagnostics, ViSQOL scores, and summaries.
- `visqol-results.csv` contains all 480 raw ViSQOL result rows and feature
  values.
- `pairs.csv` lists all 480 reference and degraded-file inputs.
- `blind-manifest.json` lists 20 selected A/B listening cases.
- `blind-answers.json` contains the hidden assignments and ViSQOL scores.
- `files.sha256` verifies all included result files.

The repository does not include source or generated audio. Paths in the data
are portable logical identifiers.

Do not read `blind-answers.json` before a blind listening session. Run the
harness again to rebuild the selected audio.

## Limits

The corpus has four sources and 20 excerpts. It does not cover all music,
speech, packet loss, surround audio, or lower rates.

ViSQOL Audio downmixes stereo to mono. It can miss stereo-only artifacts.

ViSQOL MOS-LQO is an objective perceptual estimate. It does not replace a
controlled ABX or MUSHRA listening test.

The selected blind set contains the largest ViSQOL differences. It is useful
for investigation, but it is not a random listening sample.
