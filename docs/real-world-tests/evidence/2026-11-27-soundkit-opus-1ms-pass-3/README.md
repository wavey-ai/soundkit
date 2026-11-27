# SoundKit Opus 1 ms pass 3 on Linode

At 320 kb/s, pass 3 used **10.0% less encode time and 5.8% less decode time** than the previously merged Rust build in 21 paired trials. The native build at `65e776f` reproduced the earlier packet dumps at all three default bitrates on the test fixture. It retains 1 ms frames and 2 ms nominal codec delay. [Direct Rust comparison](pass3-vs-pr2-320-core.json), [packet checks](fixture-packet-verification.txt)

| Rust 1 ms operating point | Payload (kb/s) | C encode (ms/audio s) | Rust encode (ms/audio s) | C decode (ms/audio s) | Rust decode (ms/audio s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Default, equal payload | 256 | 17.846 | 21.046 (+17.9%) | 4.624 | 5.323 (+15.1%) |
| Default, matches mean advanced PEAQ | 320 | 18.030 | 21.381 (+18.6%) | 4.608 | 5.524 (+19.9%) |
| Default, matches both mean PEAQ scores | 368 | 17.975 | 21.444 (+19.3%) | 4.593 | 5.594 (+21.8%) |
| Adaptive +4 trial, matches mean advanced PEAQ | 312 | 17.904 | 21.245 (+18.7%) | 4.655 | 5.380 (+15.6%) |
| Adaptive +4 trial, matches both mean PEAQ scores | 344 | 17.871 | 21.438 (+20.0%) | 4.572 | 5.483 (+19.9%) |

Each C comparison used standard 2.5 ms CELT restricted-lowdelay at 256 kb/s and the SoundKit slice API for Rust decode. The direct Rust comparison used the core decoder API for both builds. Each pair processed the same four-second, 48 kHz stereo signed-24 fixture at CBR. Both codecs warmed up and then ran 21 times each with alternating launch order on one AMD EPYC 7713 vCPU. The adaptive trial used a separate target directory and produced different packets from the default build at its tested rates. [Timing samples and manifest](manifest.json), [previous Linode replay](../2026-11-27-soundkit-opus-1ms-candidate/README.md)

The PEAQ labels are based on the original eight excerpts. Later passages require higher rates to reach both mean PEAQ scores, and controlled stereo listening remains open. The shipped encoder retains the default allocation decisions. [Quality trial and held-out results](../../../../soundkit-opus/quality-results/2026-11-27-opus-1ms-fixed-trim/README.md)

On `99a38b6`, the independent macOS arm64 release suite passed 152 tests with five ignored. It used the two-crate workspace and locked dependencies. Linode raw data are in `~/needletail-data/soundkit-opus-1ms-pass3-65e776f-20261127T020526Z/`. The replay used the existing dev box. [Release test results](arm64-test-results.txt), [timing summaries](summary.json)
