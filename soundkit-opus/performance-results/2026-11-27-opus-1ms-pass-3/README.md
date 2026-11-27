# Opus Custom 1 ms native optimization pass 3

This pass uses the native source from `27da89b`, merged in PR #2 at
`b015ea6`. The codec keeps the 48-sample frame and 48-sample alignment
offset at 48 kHz. Nominal frame-plus-codec delay is 2 ms for Rust and 5 ms
for C. The default encode decisions and packet format are preserved.

Paired Linode runs at 320 kb/s measured 10.0% less encode time and 5.8%
less decode time than PR #2 with the core decoder API. At equal payload
of 256 kb/s, pass 3 took 17.9% more encode time and 15.1% more decode time
than its paired C reference. [Linode results and samples](../../../docs/real-world-tests/evidence/2026-11-27-soundkit-opus-1ms-pass-3/README.md)

## Pinned Linode reference

PR #2 measured the four-second mixed signed-24 stereo fixture in 21 paired
passes on one AMD EPYC 7713 vCPU. Each row compares against the C 2.5 ms
restricted-lowdelay build at 256 kb/s. Both implementations use CBR.

| PR #2 Rust operating point | C encode ms/audio s | Rust encode ms/audio s | C decode ms/audio s | Rust decode ms/audio s |
|---|---:|---:|---:|---:|
| Default 256 kb/s, equal payload | 18.137 | 23.279 | 4.803 | 5.634 |
| Default 320 kb/s, mean Advanced PEAQ | 18.031 | 23.629 | 4.570 | 5.821 |
| Default 368 kb/s, both PEAQ means | 17.863 | 23.731 | 4.576 | 5.909 |
| Trim +4 at 312 kb/s, mean Advanced PEAQ | 17.942 | 23.565 | 4.545 | 5.744 |
| Trim +4 at 344 kb/s, both PEAQ means | 17.984 | 23.586 | 4.599 | 5.871 |

At equal payload, PR #2 still needed 22.1% less encode time and 14.7% less
decode time to catch its paired C reference. At the 344 kb/s quality-adjusted
point, the corresponding reductions are 23.7% and 21.7%. The direct Rust
320 kb/s core-API comparison measured a 12.0% encode and 5.9% decode
wall-time reduction from `d0259d8` to `27da89b`.
[Raw paired times, fixture checks and manifests](../../../docs/real-world-tests/evidence/2026-11-27-soundkit-opus-1ms-candidate/README.md)

## Compare wire rates at 1 ms and 2.5 ms

With one frame per datagram, the two frame sizes send 1,000 and 400
packets/s. The payload plus 16-byte SoundKit framing adds 128 and 51.2
kb/s respectively to the two payloads. IPv4 and UDP add a further 224
and 89.6 kb/s. These rates exclude AEP1, link-layer headers and FEC.

| Operating point | Payload kb/s | SoundKit framing included | SoundKit and IPv4/UDP included |
|---|---:|---:|---:|
| C 2.5 ms at 256 kb/s | 256 | 307.2 | 396.8 |
| Rust 1 ms, equal payload | 256 | 384 | 608 |
| Rust 1 ms, trim +4, original mean Advanced PEAQ | 312 | 440 | 664 |
| Rust 1 ms, trim +4, original both PEAQ means | 344 | 472 | 696 |
| Rust 1 ms, fixed trim 8 at 320 kb/s on the original excerpts | 320 | 448 | 672 |
| Same fixed-trim stream with implicit signalling | 312 | 440 | 664 |

At this packet rate the C wire-rate target of 396.8 kb/s leaves 44.8 kb/s
for Rust audio payload when both streams use the stated 44 bytes of
headers. At C payload targets of 192 and 320 kb/s, the equivalent Rust
payload budgets are below zero and 108.8 kb/s, respectively. A 1 ms Rust
payload of 320 kb/s leaves at most 9.6 bytes of
total headers per datagram to meet that target. IPv4 and UDP use 28 bytes
before SoundKit framing. The 8 kb/s implicit-signalling reduction
requires endpoints with the same pre-agreed Opus Custom configuration.

Batching frames reduces datagram overhead and makes the first frame wait
for the remaining frames. Grouping at most three 1 ms frames would keep
their maximum frame-plus-codec delay below C's 5 ms before network and
jitter buffering. At 320 kb/s even that grouping would use at least
437.3 kb/s under the same header model, excluding any packet-length
fields inside the datagram. This comparison uses immediate frame sending
and retains the 2 ms nominal codec delay.

## Changes in this pass

- The pitch search evaluates fine correlations only in the two windows
  found by the coarse scan. Rolling pitch power and tone LPC use checked
  slices, and packed dual correlations preserve the reference reduction.
- Exact x86-64 PVQ specializes the 2-, 4- and 8-bin shapes used by the
  48-sample mode. Both short and general searches use the same arithmetic
  and decision order.
- Allocation searches switch to a cap-only loop below the first qualifying
  band. Fine allocation distributes leftover bits while consuming them
  in one pass. The 24-point native FFT uses its fixed stages and contiguous
  butterfly groups.
- The range decoder uses independent slice cursors for both packet ends.
  It reads packed raw bits, decodes a stereo fine-energy pair together,
  and uses an exact inverse CDF for the first five coarse-energy symbols
  before falling back to the full model.

## Native instruction counts

Callgrind 3.24.0 counted the complete encode and decode example processes
on an AMD EPYC 9V74 development VM. Both builds used Rust 1.100.0 with thin
LTO and one release codegen unit. The equal-rate runs used the same two
8.020-second orchestra and electronic excerpts at 256, 320 and 368 kb/s.
The quality-adjusted runs used the same excerpts at the listed rates.

| Comparison against PR #2 | Encode instruction reduction | Decode instruction reduction |
|---|---:|---:|
| Default x86-64 target, six equal-rate pairs | 13.8–14.3% | 7.2–7.4% |
| Zen 3 and AVX2 flags, six equal-rate pairs | 16.3–17.0% | 7.6–7.8% |
| Adaptive +4 in both builds at 312 and 344 kb/s | 13.8–14.1% | 7.3–7.4% |
| PR #2 adaptive +4 at 344 versus pass 3 fixed 8 at 320 kb/s | 14.5% | 7.8–8.3% |

The final row uses the rate that met both PEAQ means on the original
eight excerpts for each trial. It remains exploratory because the later
passages favor adaptive +4 at the rates that meet both means.
[Default counts](default-instructions.csv), [Zen 3 counts](zen3-instructions.csv),
[adaptive trim counts](adaptive-instructions.csv), [fixed trim counts](fixed-instructions.csv)

## Compatibility

The two Rust builds produced identical packets and i24 PCM across 128
streams, 772,048 packets and 98,548,224 samples. The matrix includes eight
excerpts, CBR and constrained VBR, 48/64/72/120/240-sample modes, and
192/256/320/368 kb/s in the 48-sample mode. The rebuilt SIMD128 package
matched 128,320 packets and 12,318,720 i24 samples, including 5,872
concealed frames. A separate narrow range-decoder test checks the raw window
at every starting bit count and Laplace model boundaries.

The release suite passed 152 tests, with five ignored. The explicitly
enabled live C Custom comparison passed separately. In the initial
concurrent run the existing allocation test counted four allocations.
The serial release suite, a full default rerun, and 24 isolated default
runs of that test passed.

The release build passed an AArch64 Linux compile check. Clippy completed
with `clippy::approx_constant` allowed for the existing reference constants.

The adaptive +4 pass 3 build reproduced the previous trial packets, C
Custom decode, and implicit-signalling results across eight excerpts at
248, 312, 344, and 368 kb/s. At 320 and 400 kb/s the fixed trim build
matched 176,440 archived packets and C Custom decoded 16,938,240 i24 samples
identically. Turning off signalling saved 8 kb/s and preserved the decoded
samples in all 22 fixed-trim streams. [Fixed-trim interop checks](../../quality-results/2026-11-27-opus-1ms-fixed-trim/integrated-interop.csv)

A mixed-fixture smoke run measured a 48-sample alignment offset for all
three Rust builds and a 120-sample offset for C. The 256, 344 and 320 kb/s
Rust configurations produced 32-, 43- and 40-byte CBR packets respectively.

## Reproduce the instruction profiles

Use the same release settings for both builds and build the
`opus_custom_interop` example from PR #2's `27da89b` and this pass's
`65e776f` in separate target directories. The
[baseline procedure](../../quality-results/2026-11-26-opus-1ms-vs-libopus/README.md#reproduce)
prepares the stereo corpus and two-crate workspaces. Set `CORPUS` to its
signed-24 PCM input directory.

```sh
python3 soundkit-opus/tools/run_celt_instruction_ab.py "$PR2" "$PASS3" \
  --input "$CORPUS/65_orchestra_strauss_stereo.s32le" \
  --input "$CORPUS/four_tet_baby.s32le" \
  --rate 256 --rate 320 --rate 368 --require-identical --results "$RESULTS"
```

The wrapper saves separate Callgrind profiles, packet and PCM hashes, and
the exact instruction count for each phase. The Zen 3 run builds both
binaries with `RUSTFLAGS='-C target-cpu=znver3 -C target-feature=+avx2'`.
For the established quality-adjusted points, apply the
[adaptive +4 patch](../../quality-results/2026-11-26-opus-1ms-trim/trim4-experiment.patch)
to separate copies of both builds and replace the rate options with
`--rate 312 --rate 344`. For the exploratory comparison on the original
excerpts, apply the [fixed trim 8 patch](../../quality-results/2026-11-27-opus-1ms-fixed-trim/fixed8-experiment.patch)
to a separate pass 3 copy and compare PR #2 adaptive +4 to pass 3 fixed
trim using `--rate 344:320` without the identity requirement. Its encoding
decisions produce different valid packets. The
[per-excerpt quality report](../../quality-results/2026-11-27-opus-1ms-fixed-trim/README.md)
includes the additional passages and the limitation of the 320 kb/s result.

## Repeat the pinned Linode gate

Build `raw_celt_bench` from `65e776f`. Replay the archived four-second
input and pinned C binary in the [PR #2 manifest](../../../docs/real-world-tests/evidence/2026-11-27-soundkit-opus-1ms-candidate/manifest.json).
Run one warm-up per configuration and 21 pairs with alternating launch
order on the same vCPU. Use the native flags in the archive where the
default Rust build differs from the development profile. The baseline
uses standard 2.5 ms CELT restricted-lowdelay at 256 kb/s in every pair.

```sh
replay() {
  python3 soundkit-opus/tools/run_native_celt_ab.py "$C_BENCH" "$1" \
    --corpus "$LINODE_FIXTURE" --cpu "$CPU" --seconds 4 --warmup-seconds 4 \
    --rounds 21 --repeats 1 --mode cbr --pcm-bits 24 \
    --application restricted-lowdelay --baseline-decode-api c --candidate-decode-api soundkit \
    --frame-size 120 --bitrate 256000 --candidate-frame-size 48 \
    --candidate-bitrate "$2" --json "$3"
}
replay "$PASS3_BENCH" 256000 pass3-equal-rate.json
replay "$PASS3_BENCH" 320000 pass3-default-advanced-mean.json
replay "$PASS3_BENCH" 368000 pass3-default-both-means.json
replay "$PASS3_ADAPTIVE_BENCH" 312000 pass3-adaptive-advanced-mean.json
replay "$PASS3_ADAPTIVE_BENCH" 344000 pass3-adaptive-both-means.json
```

The quality variant uses an independently built target directory and must
have a distinct executable hash and different packets. Use the same Rust
decoder API in direct PR #2 versus pass 3 pairs. The archived direct
comparison uses the core API on both builds. Controlled stereo listening
remains pending.
