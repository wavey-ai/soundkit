# SoundKit Opus 1 ms optimization on Linode

The optimized Rust candidate `27da89b` used **12.0% less encode time and 5.9% less decode time** than the original Rust build `d0259d8` in paired runs at 320 kb/s. Both used the core decoder API and produced identical benchmark checksums. Standard C libopus at 2.5 ms still used less encode and decode time across all five operating points. Nominal codec delay was 2 ms for Rust and 5 ms for C. [Paired Rust comparison](rust-320-ab-core.json)

| Rust 1 ms operating point | Payload (kb/s) | C encode (ms/audio s) | Rust encode (ms/audio s) | C decode (ms/audio s) | Rust decode (ms/audio s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Default, equal payload | 256 | 18.137 | 23.279 (+28.4%) | 4.803 | 5.634 (+17.3%) |
| Default, matches mean advanced PEAQ | 320 | 18.031 | 23.629 (+31.0%) | 4.570 | 5.821 (+27.4%) |
| Default, matches both mean PEAQ scores | 368 | 17.863 | 23.731 (+32.9%) | 4.576 | 5.909 (+29.1%) |
| Trim trial, matches mean advanced PEAQ | 312 | 17.942 | 23.565 (+31.3%) | 4.545 | 5.744 (+26.4%) |
| Trim trial, matches both mean PEAQ scores | 344 | 17.984 | 23.586 (+31.1%) | 4.599 | 5.871 (+27.7%) |

Each row uses its paired C reference at 256 kb/s and the Rust SoundKit slice decoder API. Each implementation ran a four-second warm-up and 21 timed passes, alternating launch order on the same AMD EPYC 7713 vCPU. The signed 24-bit stereo mixed fixture was regenerated with the pinned benchmark code. The file replay reproduced all 1,600 C packets and all 4,000 original Rust packets from the in-memory fixture at 256 kb/s. The first trim build shared the default build's target directory and reused its executable. The reported trim measurements use a separate rebuild with verified executable and packet hashes. [Fixture check](fixture-verification.txt), [trim packet check](trim-packet-verification.txt), [results and manifest](manifest.json)

The PEAQ labels come from the prior eight-excerpt study. At 344 kb/s, the trim build meets both mean scores and still trails on basic PEAQ for some excerpts. With one packet per datagram and SoundKit and IPv4/UDP headers, it uses 696 kb/s against C's 396.8 kb/s. The published encoder retains its original allocation decisions. [Per-excerpt quality and wire rates](../../../../soundkit-opus/quality-results/2026-11-26-opus-1ms-trim/README.md)

The macOS arm64 release suite passed 150 tests with five ignored on PR head `7ff8e6f`. The raw Linode archive is `~/needletail-data/soundkit-opus-1ms-candidate-27da89b-20261127T013930Z/` on `wavey-linode-codex`. No additional Linode instances were used. [Release test results](arm64-test-results.txt), [timing summaries](summary.json), [original Linode baseline](../2026-11-26-soundkit-opus-1ms-linode/README.md)
