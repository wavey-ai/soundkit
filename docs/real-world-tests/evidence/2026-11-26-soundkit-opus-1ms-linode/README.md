# SoundKit Opus 1 ms CPU benchmark on Linode

On 26 November 2026, SoundKit `d0259d8` with 1 ms Opus Custom frames took 52.8% more time to encode and 38.6% more time to decode than standard C libopus with 2.5 ms frames. Both codecs processed the same four-second, 48 kHz stereo signed-24 fixture at 256 kb/s CBR. Nominal codec delay was 2 ms for Rust and 5 ms for C, including one frame and the codec overlap. [Benchmark protocol](../../../../soundkit-opus/CODEC.md#raw-celt-benchmark)

| Measurement | Rust Opus Custom, 1 ms | C libopus, 2.5 ms |
| --- | ---: | ---: |
| Median encode speed | 35.51× realtime | 54.26× realtime |
| Median decode speed | 156.42× realtime | 216.76× realtime |
| Raw Opus bytes per packet | 32 | 80 |
| Packets per second | 1,000 | 400 |
| Encoded bytes over four seconds | 128,000 | 128,000 |
| Decoded alignment lag (absolute samples) | 48 | 120 |
| Fixture aligned SNR | 27.27 dB | 32.49 dB |

The documented runner measured elapsed wall-clock time in the codec loops over 21 repeats, after a warm-up pass. The C encoder used restricted-lowdelay at complexity 10. Both codecs were pinned to vCPU 6 on the existing AMD EPYC 7713 Linode dev box. Rust used native code generation with AVX2. C used the static Release build of xiph/opus `316c675fa1c9982d76e9cdcba7231076f75047da` with runtime SSE and AVX2 dispatch and `FLOAT_APPROX` off. The Rust build used the two-crate workspace described by the earlier verification, which preserves the release profile and avoids the unrelated Speex dependency resolution failure. [Runner output](documented.md), [build environment](environment.txt), [two-crate setup](../../../../soundkit-opus/quality-results/2026-11-26-opus-1ms-vs-libopus/verification.md)

A second check alternated the launch order in 21 pairs with a single four-second timing pass per launch. The median paired Rust elapsed-time increase was 53.9% for encode (interquartile range 46.3–55.3%) and 27.0% for decode (24.0–32.4%). Rust took longer in all 21 pairs in both phases. The paired runs used `--skip-quality`. The synthetic fixture's aligned SNR from the documented run is a signal-error diagnostic. [Paired samples](timings.csv), [summary statistics](summary.json)

The raw archive is `~/needletail-data/soundkit-opus-1ms-linode-d0259d8-20261126T201120Z/` on `wavey-linode-codex`. It includes the individual TSVs, CPU samples, build records, and the commands used to run and summarize the benchmark. The run used the existing dev box with no additional Linode instances. [Run manifest](manifest.json)
