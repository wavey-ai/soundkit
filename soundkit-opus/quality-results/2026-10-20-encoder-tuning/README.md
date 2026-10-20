# Encoder tuning: 2026-10-20

This report gives the perceptual measurements for four encoder decision
changes. It compares the changed encoder with trunk libopus.

## Result

With the four changes, `soundkit-opus` scored higher than trunk libopus in
PEAQ and ViSQOL. The table gives Rust minus libopus over 360 paired items.

| Metric | Before the changes | After the changes |
|---|---:|---:|
| PEAQ advanced Distortion Index | -0.0038 | +0.0118 |
| PEAQ basic Distortion Index | +0.0014 | +0.0112 |
| Items where the PEAQ advanced DI of Rust was higher | 124 of 360 | 250 of 360 |
| ViSQOL Audio MOS-LQO | Not measured | +0.0025 |
| Items where the ViSQOL MOS-LQO of Rust was higher | Not measured | 267 of 360 |

A higher value is better for each metric. On this corpus, the PEAQ advanced DI
increased by 0.23 from 256 to 320 kb/s. Thus an increase of 0.012 is
approximately equal to 3 kb/s at 256 kb/s.

The PEAQ advanced DI of Rust was higher in all ten rate and mode cells:

| Rate | CBR | Constrained VBR |
|---:|---:|---:|
| 96 kb/s | +0.016 | +0.019 |
| 128 kb/s | +0.010 | +0.018 |
| 192 kb/s | +0.017 | +0.015 |
| 256 kb/s | +0.010 | +0.011 |
| 320 kb/s | +0.001 | +0.001 |

The ViSQOL MOS-LQO of Rust was higher in nine cells. At 128 kb/s constrained
VBR, it was 0.0001 lower.

The PEAQ advanced DI was also higher for each source:

| Source | Rust minus libopus |
|---|---:|
| `bill-evans-secret-sessions` | +0.0037 |
| `blue-nile-hats` | +0.0286 |
| `lori-asha` | +0.0114 |
| `nocturnal-animals` | +0.0034 |

## Held-out validation

The tuning used the `diverse-v1` corpus. A second corpus of 12 other tracks
tested the result. It has rock band masters, a pop release, a metal recording,
and other studio masters. Each track gave three 10-second excerpts, at 30, 75,
and 120 seconds, for 36 items per configuration.

| Metric | Rust minus libopus | 95% interval | Items where Rust was higher |
|---|---:|---|---:|
| PEAQ advanced DI | +0.0393 | +0.0299 to +0.0486 | 278 of 360 |
| PEAQ basic DI | +0.0219 | -0.0011 to +0.0449 | 246 of 360 |
| ViSQOL MOS-LQO | +0.0018 | +0.0007 to +0.0029 | 216 of 360 |

The gains on the held-out corpus were larger than on the tuning corpus.

## Results per cell

The table gives Rust minus libopus with the 95% interval over 36 items, for
both corpora. A cell is marked when the interval is fully below zero.

| Cell | Corpus | PEAQ advanced DI | PEAQ basic DI | ViSQOL MOS-LQO |
|---|---|---|---|---|
| CBR 96 | tuning | +0.016 (-0.002, +0.035) | +0.012 (+0.001, +0.024) | +0.0046 (+0.0003, +0.0089) |
| CBR 96 | held-out | +0.090 (+0.030, +0.149) | +0.070 (+0.027, +0.113) | +0.0051 (+0.0009, +0.0093) |
| CBR 128 | tuning | +0.010 (-0.001, +0.021) | +0.005 (-0.005, +0.014) | +0.0037 (-0.0001, +0.0076) |
| CBR 128 | held-out | +0.067 (+0.031, +0.103) | +0.045 (+0.016, +0.074) | +0.0030 (-0.0023, +0.0083) |
| CBR 192 | tuning | +0.017 (+0.010, +0.023) | +0.054 (-0.021, +0.128) | +0.0020 (+0.0001, +0.0040) |
| CBR 192 | held-out | +0.044 (+0.028, +0.060) | -0.114 (-0.182, -0.046), marked | +0.0024 (+0.0004, +0.0044) |
| CBR 256 | tuning | +0.010 (+0.006, +0.015) | +0.057 (-0.036, +0.150) | +0.0021 (+0.0010, +0.0031) |
| CBR 256 | held-out | +0.019 (+0.012, +0.027) | +0.079 (-0.001, +0.159) | +0.0017 (+0.0003, +0.0030) |
| CBR 320 | tuning | +0.001 (-0.001, +0.004) | +0.053 (-0.012, +0.118) | +0.0014 (+0.0006, +0.0022) |
| CBR 320 | held-out | +0.012 (+0.002, +0.022) | +0.030 (-0.043, +0.102) | +0.0004 (-0.0001, +0.0009) |
| VBR 96 | tuning | +0.019 (+0.006, +0.032) | +0.014 (+0.005, +0.024) | +0.0067 (+0.0014, +0.0121) |
| VBR 96 | held-out | +0.075 (+0.036, +0.114) | +0.052 (+0.020, +0.085) | -0.0006 (-0.0064, +0.0051) |
| VBR 128 | tuning | +0.018 (+0.006, +0.029) | +0.015 (+0.006, +0.024) | -0.0001 (-0.0036, +0.0035) |
| VBR 128 | held-out | +0.059 (+0.031, +0.088) | +0.030 (+0.009, +0.052) | +0.0027 (-0.0023, +0.0076) |
| VBR 192 | tuning | +0.015 (+0.008, +0.021) | -0.077 (-0.182, +0.029) | +0.0012 (-0.0013, +0.0036) |
| VBR 192 | held-out | +0.025 (+0.011, +0.038) | -0.113 (-0.236, +0.010) | +0.0016 (-0.0008, +0.0040) |
| VBR 256 | tuning | +0.011 (+0.006, +0.015) | +0.010 (-0.047, +0.066) | +0.0017 (+0.0004, +0.0029) |
| VBR 256 | held-out | +0.005 (-0.006, +0.016) | +0.059 (-0.022, +0.139) | +0.0016 (+0.0003, +0.0029) |
| VBR 320 | tuning | +0.001 (-0.001, +0.003) | -0.032 (-0.087, +0.024) | +0.0018 (+0.0009, +0.0026) |
| VBR 320 | held-out | -0.003 (-0.008, +0.001) | +0.082 (+0.002, +0.162) | +0.0005 (+0.0001, +0.0009) |

One of the 60 metric cells has an interval fully below zero: PEAQ basic at
192 kb/s CBR on the held-out corpus. In the same cell, the PEAQ advanced DI and
the ViSQOL MOS-LQO of Rust were significantly higher.

The PEAQ basic deficit in that cell is not from the four changes. The Rust
encoder without them gave -0.103 (-0.204, -0.002) against libopus. Allocation
trim, dynalloc, dual stereo, spread, transient threshold, and disabled
tonality analysis each left the value between -0.068 and -0.139. The two
decoders gave the same SNR on each packet set, so the difference comes from
the encoders. On the item with the largest deficit, the two decodes had the
same bandwidth. The Rust noise was 0.2-0.7 dB higher between 4 and 16 kHz.

The PEAQ basic DI changed by up to 0.8 between the two encoders on single
items with a PEAQ ODG of about +0.1. Near transparency, the basic version
varies more than the advanced version.

## Listening kit

A blind A/B kit holds the items where the metrics disagree most: two at
128 kb/s VBR, two at 96 kb/s VBR, and four at 192 kb/s CBR. Each item has the
reference and two decodes in random order. A separate file holds the answer
key. Rebuild the kit from the reproduction steps below. It is not in the
repository.

## Changes

| Decision | libopus value | SoundKit value | Frame sizes |
|---|---|---|---|
| TF-change penalty in `tf_analysis` | `0.04 * max(-0.25, 0.5 - tf_estimate)` | 3 times that value | All |
| Constrained-VBR target blend | 0.67 | 0.4 | 5 ms |
| Pitch prefilter gain before quantization | gain | 0.8 times gain | All |
| Pitch prefilter threshold | libopus threshold | libopus threshold + 0.1 | All |

The Rust encoder already used a blend of 0.5 for frame sizes other than 5 ms.

All three metrics agreed on the combination. Against the unchanged Rust
encoder at 5 ms, 128-256 kb/s:

| Metric | Change | Paired items with a gain |
|---|---:|---:|
| PEAQ advanced DI | +0.0187 | 184 of 216 |
| PEAQ basic DI | +0.0094 | |
| ViSQOL MOS-LQO | +0.0022 | 154 of 216 |

At 10 ms frames, the TF and prefilter changes raised the PEAQ advanced DI by
0.0088 and the basic DI by 0.052. At 20 ms frames, the values were 0.0063 and
0.020. The `nocturnal-animals` advanced DI decreased by 0.003 at both sizes.

## Rejected changes

| Change | PEAQ advanced DI | Other metrics |
|---|---:|---|
| Allocation trim +1, with the four changes | +0.0092 more | PEAQ basic -0.026, ViSQOL -0.0037 |
| Unconstrained VBR at the same mean rate | -0.049 | PEAQ basic -0.057 |
| Pulse-move refinement after the greedy PVQ search | +0.004 | Encode time +15% |
| Dual stereo always on | -0.310 | `nocturnal-animals` +0.022 |
| Dual stereo always off | +0.002 | `nocturnal-animals` -0.013 |
| Dual-stereo smoothing or threshold change | -0.023 to +0.005 | |
| Spread decision forced to one value | -0.008 to +0.002 | |
| Dynalloc boost scaled by 0 or 2 | -0.010 to 0.000 | |
| Transient threshold 120 or 400 | -0.002 to +0.001 | |

The PVQ figure comes from a 20-second excerpt matrix. The other rows use the
sweep method below.

The encoder now has `set_vbr_constraint`, which selects unconstrained VBR. In
both VBR modes, a drift correction holds the long-term rate at the target.
libopus unconstrained VBR does not do this. On this corpus at a 128 kb/s
target, libopus unconstrained VBR produced 101.5-139.5 kb/s. SoundKit produced
127.3-128.8 kb/s. Constrained VBR remains the default because it scored higher.

## Method

- Codecs: both encoders and decoders compiled to WebAssembly and run in one
  Node.js process; see the
  [wasm performance report](../../performance-results/2026-10-20-wasm-48k/README.md)
- libopus: trunk commit `03647f524a40b05a1898522e92033810b58103c7`, complexity
  10, music signal, fullband, `OPUS_APPLICATION_AUDIO`
- Corpus: the four SoundKit `diverse-v1` sources, 100 seconds each, 48 kHz
  stereo signed 24-bit PCM; each source joins 10-second slots from 10 tracks
- Items: the first nine 10-second slots of each source, 36 items per
  configuration
- Configuration: 5 ms frames and the signed 24-bit API, unless the text gives
  another frame size
- Alignment: the harness removed each decoder delay before scoring
- PEAQ: GstPEAQ commit `7397e49`, basic and advanced versions, 32-bit float WAV
- ViSQOL: ViSQOL Audio commit `38d0b0163e441047d4429bf07ad09e5b9031d02c`,
  nu-SVR model, PCM16 WAV with a 0.1 dB gain reduction
- Sweeps: one knob per run against the unchanged encoder, at 128, 192, and
  256 kb/s, CBR and constrained VBR, 216 paired items; even and odd slots
  were checked separately
- Final comparison: 96, 128, 192, 256, and 320 kb/s, CBR and constrained VBR,
  360 paired items

The Distortion Index is the PEAQ output before the mapping to the Objective
Difference Grade. The ODG was within 0.1 of zero for most items at 192 kb/s and
higher, so the DI was used for comparisons.

## Limits

The corpus has 40 tracks from four albums. Solo instruments, speech, and
electronic music have little coverage. PEAQ and ViSQOL are objective
estimates. They do not replace a controlled listening test. ViSQOL Audio
downmixes stereo to mono, and PEAQ averages the channels.

## Reproduction

Build the modules with `FEATURES=tuning tools/wasm_ab/rust_variant.sh <name>`.
The tuning feature makes each decision constant in `src/tuning.rs` adjustable.
Run `tools/run_wasm_ab.mjs` with `--wav-dir`, `--slot-seconds 10`, and
`--tune-b` to write WAV pairs for each setting. Then score the pairs with
GstPEAQ and ViSQOL.
