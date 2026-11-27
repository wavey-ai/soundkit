# Quality-adjusted Opus timing

At the bitrate that matched mean advanced PEAQ against C at 256 kb/s, Rust at 1 ms still took 53.5% more time to encode and 30.9% more to decode. Raising Rust from 256 to 320 kb/s increased the median times by 1.8% and 5.4% in the same run. Nominal codec delay stayed at 2 ms for Rust and 5 ms for C across the tested rates.

| Configuration | Payload bitrate | Median encode time | Median decode time |
| --- | ---: | ---: | ---: |
| C libopus at 2.5 ms | 256 kb/s | 75.83 ms | 19.96 ms |
| Rust Opus Custom at 1 ms, equal bitrate | 256 kb/s | 114.35 ms | 24.79 ms |
| Rust Opus Custom at 1 ms, matched mean advanced PEAQ | 320 kb/s | 116.43 ms | 26.12 ms |
| Rust Opus Custom at 1 ms, matched both mean PEAQ scores | 368 kb/s | 116.97 ms | 26.67 ms |

Each measurement covered four seconds of the generated mixed 48 kHz stereo signed-24 fixture at CBR. The benchmark reused the verified SoundKit `d0259d8` and xiph/opus `316c675f` binaries from the first Linode run. Each configuration had one warm-up followed by 21 measured rounds with its position shuffled within each round. Both codecs ran on vCPU 6 of the AMD EPYC 7713 Linode dev box. [Initial benchmark](../README.md), [recorded samples](timings.tsv), [timing and alignment checks](summary.json)

The rate choices come from the eight-excerpt PEAQ study. At 320 kb/s, Rust matched C's mean advanced PEAQ but had a lower mean basic PEAQ. At 368 kb/s, both Rust means met or exceeded C's, with basic PEAQ still lower on individual excerpts. The follow-up timed the synthetic fixture. [Eight-excerpt quality study](../../../../../soundkit-opus/quality-results/2026-11-26-opus-1ms-vs-libopus/README.md)

The raw data are in `~/needletail-data/soundkit-opus-quality-adjusted-d0259d8-20261126T222431Z/` on `wavey-linode-codex`. [Manifest](manifest.json)
