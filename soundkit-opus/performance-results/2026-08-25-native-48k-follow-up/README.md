# Native 48 kHz follow-up — 2026-08-25

This follow-up checked the remaining native CELT hotspots after publishing
`soundkit-opus` 0.13.0. The production target remained high-fidelity,
5 ms stereo audio at 48 kHz, using `OPUS_APPLICATION_AUDIO`. No WebAssembly
optimization was attempted.

## Environment

- Host: Google Cloud `yl-encodec-1`, `c4-highcpu-4`, Intel Xeon Platinum 8581C
- OS: Debian 12 x86-64
- SoundKit baseline: commit `35aa144`
- Trunk libopus: commit `03647f524a40b05a1898522e92033810b58103c7`
- Rust release profile: ThinLTO, one codegen unit, `-C target-cpu=native`
- Corpus: the four-source SoundKit `diverse-v1` 48 kHz FLAC-derived corpus
- Benchmark frame size: 240 samples per channel (5 ms)
- PCM depth: signed 24-bit input carried in `i32`

The GCP instance was stopped after this pass. Raw data remains on its persistent
disk under `/home/jamie/bench/results`.

## Fresh trunk comparison

A fresh single-source check used 10 seconds of Lori Asha material, three
alternating rounds, and seven repetitions per measurement at 256 kbit/s.
Negative deltas mean SoundKit took less CPU time than trunk libopus.

| Mode | SoundKit encode | Encode delta | SoundKit decode | Decode delta |
| --- | ---: | ---: | ---: | ---: |
| CBR | 101.82x realtime | -2.36% | 385.91x realtime | -1.25% |
| constrained VBR | 101.81x realtime | -1.75% | 384.46x realtime | -2.58% |

This reconfirmed that the published native implementation was already faster
than the matched C reference for both encode and decode in the target cell.

## Matched profiles

The same five-second VBR/256 kbit/s input was repeated 200 times for each
profile. The workload timings recorded alongside the profiles were:

| Implementation | Encode | Decode |
| --- | ---: | ---: |
| SoundKit | 49.2112 ms | 13.0871 ms |
| trunk libopus | 49.9299 ms | 13.1861 ms |

The largest SoundKit symbol was `op_pvq_search_with_scratch` at 12.05% of total
samples, compared with `op_pvq_search_sse2` at 11.70% for C. SoundKit spectral
rotation accounted for 2.99% in `exp_rotation1` and 0.98% in `exp_rotation`; the
corresponding C symbols accounted for 2.64% and 0.59%. Percentages from separate
whole-process profiles are directional evidence, not an isolated A/B result.

Profile data is retained as:

- `/home/jamie/bench/results/soundkit-opus-rust-current.data`
- `/home/jamie/bench/results/soundkit-opus-c-current.data`

## Focused A/B results

Each candidate was compared with the same baseline binary over all four corpus
sources. Runs used 20 seconds per source, five alternating rounds, seven
repetitions, CPU 2, constrained VBR at 256 kbit/s, and the 24-bit/5 ms audio
configuration. A negative delta favours the candidate.

| Candidate | Metric | Corpus delta | Paired median | Paired mean | Faster pairs | Identity changes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| reuse resolved pulse-cache row | encode | -0.2163% | -0.0968% | -0.1444% | 11/20 | 0 |
| reuse resolved pulse-cache row | decode | +0.9018% | +0.9040% | +1.0793% | 2/20 | 0 |
| inverse MDCT directly into output work buffer | encode | +0.3671% | -0.1653% | +0.0733% | 11/20 | 0 |
| inverse MDCT directly into output work buffer | decode | +1.2053% | +0.8550% | +1.1919% | 2/20 | 0 |

Both candidates preserved packet statistics and decoded checksums in every
pair, but both regressed aggregate decode time. They were rejected and fully
reverted. Their raw results are retained at:

- `/home/jamie/bench/results/soundkit-opus-pulse-row-ab.json`
- `/home/jamie/bench/results/soundkit-opus-mdct-output-ab.json`

A possible removal of the unit-stride spectral-rotation specialization was
prepared but not timed before the pass was stopped. It was reverted, so no
performance conclusion is recorded for it.

## Reusable runner

`tools/run_native_celt_ab.py` now performs the warm-up, alternating order,
paired aggregation, identity checks, and optional JSON capture used above. For
example:

```sh
tools/run_native_celt_ab.py BASELINE CANDIDATE \
  --corpus /home/jamie/bench/flac-packet-corpus/diverse-v1 \
  --seconds 20 --rounds 5 --repeats 7 --cpu 2 \
  --pcm-bits 24 --mode vbr --bitrate 256000 \
  --frame-size 240 --application audio --json result.json
```

## Outcome

No production codec change from this follow-up met the acceptance bar. The
published native path remains the best measured implementation: faster than
trunk libopus in the target cell, packet-stable, and without a decode tradeoff.
