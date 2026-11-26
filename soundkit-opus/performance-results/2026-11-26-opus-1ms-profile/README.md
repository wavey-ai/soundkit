# Opus Custom 1 ms native profiling

The encoder and decoder changes reduce instruction counts while preserving
packets and decoded PCM. The separate allocation trim experiment reduces
the payload needed to reach the mean PEAQ scores of C Opus at 2.5 ms.
The production encoder keeps its libopus-compatible allocation decisions.
[Trim experiment and per-excerpt scores](../../quality-results/2026-11-26-opus-1ms-trim/README.md)

## Linode reference

SoundKit `d0259d8` and xiph/opus `316c675f` ran on one pinned AMD EPYC 7713
vCPU. Each configuration had one warm-up, followed by 21 rounds in shuffled
order. Each run encoded and decoded four seconds of mixed 48 kHz stereo
signed 24-bit PCM at CBR. The table divides the median wall time by four.

| Configuration | Encode ms per audio s | Decode ms per audio s | Frame plus codec delay |
|---|---:|---:|---:|
| C restricted-lowdelay 2.5 ms, 256 kb/s | 18.957 | 4.989 | 5 ms |
| Rust Custom 1 ms, 256 kb/s | 28.586 | 6.197 | 2 ms |
| Rust Custom 1 ms, 320 kb/s | 29.107 | 6.531 | 2 ms |
| Rust Custom 1 ms, 368 kb/s | 29.241 | 6.668 | 2 ms |

At 320 kb/s, the original Rust code spent 53.5% more time encoding and
30.9% more time decoding than C at 256 kb/s. The 320 and 368 kb/s points
match the mean advanced PEAQ score and both mean PEAQ scores respectively
on the original eight-excerpt study. Individual basic PEAQ deficits remain.
The raw timing archive is `~/needletail-data/soundkit-opus-quality-adjusted-d0259d8-20261126T222431Z/`
on the Linode. [Original quality study](../../quality-results/2026-11-26-opus-1ms-vs-libopus/README.md)

## Profiling and verification

Callgrind 3.24 attributed 43.5% of the reference encoder's instructions to
pitch prefiltering on the orchestra excerpt at 256 kb/s. The overlapping
pitch-downsampling and inner-product totals were 22.9% and 12.8%. Separate
decoder sampling concentrated in spectral band coding and allocation.

- A sliding contiguous pitch history avoids copying the entire history
  on each frame. Exact inner products use packed accumulators and retain
  libopus's four-lane reduction order.
- The encoder reuses analysed log-band energies and skips spread weights
  when the last band forces spreading off. Only transient frames evaluate
  the alternative time-frequency selection.
- The x86-64 decoder batches exact signed integer PCM rounding. Other
  targets retain their existing output conversion.

Callgrind instruction counts include the complete `opus_custom_interop`
encode or decode process. All cases used matching 8.020-second excerpts,
48 kHz stereo, signalled Custom packets and signed 24-bit PCM. The two
binaries used Rust 1.100.0 and the workspace release profile. The candidate
used 12.9–13.6% fewer encode instructions and 4.7–5.1% fewer decode
instructions across these six comparisons. These are instruction counts
from the development VM. Linode wall-time comparisons for the candidate
are pending. [Counts and output hashes](instruction-counts.csv)

With the trim experiment applied to the optimized build, the 312 and
344 kb/s points used 13.7–14.0% fewer encode instructions and 5.3–6.2%
fewer decode instructions than the original 320 and 368 kb/s encoder
and decoder on these two excerpts. Those rates match the mean advanced
score and both mean scores, respectively, against C at 256 kb/s.
[Quality-adjusted counts](quality-adjusted-instruction-counts.csv)

The paired Rust builds produced 772,048 identical packets and 98,548,224
identical decoded i24 samples in 128 comparisons. The matrix covered eight
excerpts, CBR and constrained VBR, 48-, 64-, 72-, 120- and 240-sample Custom
modes, and 192–368 kb/s for the 48-sample mode. The release crate suite and
live C Custom comparison passed in the same two-crate workspace used for
the original report. [Crate test procedure](../../quality-results/2026-11-26-opus-1ms-vs-libopus/verification.md)

With both the optimization and trim patch applied, the 320,800 packets
from the eight excerpts at 192, 256, 312, 320 and 344 kb/s matched the
isolated trim experiment's packets. C Custom and Rust decoded those
packets to identical i24 samples.

## Equal-rate and quality-adjusted timing runs

Use the Linode archive's C binary and Rust build flags for the next run.
Replay the archive's four-second mixed fixture for the primary comparison.
Build `raw_celt_bench` from this candidate, and build it again with the
[trim experiment patch](../../quality-results/2026-11-26-opus-1ms-trim/trim4-experiment.patch)
applied in a separate checkout. For the primary comparison, put the archived
fixture in `CORPUS` and set `SECONDS_PER_EXCERPT=4`. To collect timings per
excerpt, use the signed 24-bit corpus prepared by the
[quality report](../../quality-results/2026-11-26-opus-1ms-vs-libopus/README.md#reproduce)
and set `SECONDS_PER_EXCERPT=8`. The input filenames must end in
`-48k-s24.s32le`. Set `CPU` to the vCPU used for both binaries.

```sh
timing() {
  python3 soundkit-opus/tools/run_native_celt_ab.py "$C_BENCH" "$1" \
    --corpus "$CORPUS" --cpu "$CPU" --seconds "$SECONDS_PER_EXCERPT" --warmup-seconds 4 \
    --rounds 21 --repeats 1 --mode cbr --pcm-bits 24 \
    --application restricted-lowdelay --baseline-decode-api c \
    --frame-size 120 --bitrate 256000 --candidate-frame-size 48 \
    --candidate-bitrate "$2" --json "$3"
}
timing "$RUST_BENCH" 256000 equal-rate.json
timing "$RUST_BENCH" 320000 default-advanced-mean.json
timing "$RUST_BENCH" 368000 default-both-means.json
timing "$TRIM_BENCH" 312000 trim-advanced-mean.json
timing "$TRIM_BENCH" 344000 trim-both-means.json
```

| Configuration | Payload | Packets/s | With 16-byte SoundKit header | With SoundKit header and IPv4/UDP |
|---|---:|---:|---:|---:|
| C 2.5 ms reference | 256 kb/s | 400 | 307.2 kb/s | 396.8 kb/s |
| Rust 1 ms, equal payload | 256 kb/s | 1,000 | 384 kb/s | 608 kb/s |
| Rust 1 ms default, mean advanced PEAQ | 320 kb/s | 1,000 | 448 kb/s | 672 kb/s |
| Rust 1 ms default, both PEAQ means | 368 kb/s | 1,000 | 496 kb/s | 720 kb/s |
| Rust 1 ms trim, mean advanced PEAQ | 312 kb/s | 1,000 | 440 kb/s | 664 kb/s |
| Rust 1 ms trim, both PEAQ means | 344 kb/s | 1,000 | 472 kb/s | 696 kb/s |

The IPv4/UDP column assumes one packet per datagram, a 20-byte IPv4
header and an 8-byte UDP header. It excludes AEP1, link-layer headers
and FEC. When the endpoints agree on the Custom mode out of band,
implicit signalling saves 8 kb/s at both tested rates with identical
decoded audio. Controlled stereo listening remains the gate for
perceptual equivalence.
