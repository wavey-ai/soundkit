# Native 48 kHz performance checkpoint

Date: 2026-08-24

## Result

Rust was faster than trunk libopus in all 12 aggregate timing cells.

Encoding used `1.19-2.36%` less time. Decoding used `1.50-4.64%` less time.

Decoder-method correction (2026-08-25): these decoder cells timed each
implementation on packets produced by its own encoder. CBR packet bytes match,
but constrained-VBR streams differ, so those VBR cells are end-to-end results,
not an isolated decoder comparison. The replacement same-packet i24 gate at
192 kb/s measured SoundKit at `+0.51%` on C packets, `-0.30%` on SoundKit
packets, and `+0.10%` combined across 40 alternating music-corpus pairs.

The comparison used direct, in-process codec calls. FFmpeg did not run inside
the measured loops.

Negative deltas in this report mean Rust used less elapsed time than C.

## Scope

Each test used 48 kHz stereo CELT with 240-sample, 5 ms frames.

Both encoders used `OPUS_APPLICATION_AUDIO`, fullband audio, and maximum
complexity. The test did not use `OPUS_APPLICATION_RESTRICTED_LOWDELAY`.

The matrix covered signed 16-bit and signed 24-bit PCM. It covered CBR and
constrained VBR at 192, 256, and 320 kb/s.

The Rust path used standard PVQ. The experimental direct-cubic mode was off.

## Timing matrix

Each elapsed value is the sum of four source medians. It represents 40 seconds
of source audio for one encode or decode pass.

| PCM | Mode | Rate | Rust encode | C encode | Encode delta | Rust decode | C decode | Decode delta |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16-bit | CBR | 192 kb/s | 371.9126 ms | 380.9057 ms | -2.36% | 95.3447 ms | 99.9845 ms | -4.64% |
| 16-bit | CBR | 256 kb/s | 400.3750 ms | 407.1714 ms | -1.67% | 107.8502 ms | 112.4864 ms | -4.12% |
| 16-bit | CBR | 320 kb/s | 411.3018 ms | 420.9495 ms | -2.29% | 118.0561 ms | 122.4393 ms | -3.58% |
| 16-bit | constrained VBR | 192 kb/s | 374.5963 ms | 382.2683 ms | -2.01% | 96.1460 ms | 100.2788 ms | -4.12% |
| 16-bit | constrained VBR | 256 kb/s | 402.3984 ms | 407.7622 ms | -1.32% | 108.5093 ms | 112.8688 ms | -3.86% |
| 16-bit | constrained VBR | 320 kb/s | 413.2046 ms | 421.0595 ms | -1.87% | 118.2946 ms | 122.6479 ms | -3.55% |
| 24-bit | CBR | 192 kb/s | 373.0409 ms | 380.0931 ms | -1.86% | 93.0120 ms | 94.8794 ms | -1.97% |
| 24-bit | CBR | 256 kb/s | 397.7392 ms | 402.9012 ms | -1.28% | 104.8518 ms | 106.6332 ms | -1.67% |
| 24-bit | CBR | 320 kb/s | 408.7642 ms | 415.8744 ms | -1.71% | 114.5355 ms | 116.2834 ms | -1.50% |
| 24-bit | constrained VBR | 192 kb/s | 374.9450 ms | 380.8431 ms | -1.55% | 93.6590 ms | 95.4408 ms | -1.87% |
| 24-bit | constrained VBR | 256 kb/s | 399.6411 ms | 404.4352 ms | -1.19% | 105.4151 ms | 107.1323 ms | -1.60% |
| 24-bit | constrained VBR | 320 kb/s | 409.7085 ms | 416.2045 ms | -1.56% | 114.8206 ms | 116.8595 ms | -1.74% |

The Rust encoder ran at `96.80-107.55x` realtime. The Rust decoder ran at
`338.14-430.05x` realtime.

## Checkpoint progression

Each checkpoint repeated the complete 12-cell timing matrix. The weighted
delta uses the sum of every source and configuration median.

| Checkpoint | Encode range | Weighted encode | Encode wins | Decode range | Weighted decode | Decode wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| native baseline after i24 work | -0.70% to +0.59% | -0.19% | 10/12 | -5.12% to -1.97% | -3.30% | 12/12 |
| static pitch-tail mask | -1.38% to -0.04% | -0.74% | 12/12 | -5.65% to -1.56% | -3.12% | 12/12 |
| checked MDCT rotations | -2.49% to -0.39% | -1.40% | 12/12 | -5.24% to -1.06% | -3.38% | 12/12 |
| final allocation-clone removal | -2.36% to -1.19% | -1.72% | 12/12 | -4.64% to -1.50% | -2.86% | 12/12 |

Decode-only movement between encoder checkpoints shows normal timing variance.
The focused A/B results determine attribution.

## Timing method

The isolated host was a GCP `c4-highcpu-4` instance. It had four virtual CPUs
on an Intel Xeon Platinum 8581C.

The runner pinned every measured process to CPU 2 with `taskset`.

Each source supplied a 10-second timing excerpt. A five-second pass warmed each
source and configuration before measurement.

Each cell used three alternating Rust/C rounds. Each implementation processed
seven internal repeats during every round.

The runner calculated a median for each source and implementation. It summed
the four medians before calculating the reported percentage.

The percentage formula was `(Rust time - C time) / C time * 100`.

File loading and corpus conversion were outside the timed loops. The same
prepared PCM input reached both implementations.

## Corpus

The SoundKit `diverse-v1` corpus supplied four 100-second, 48 kHz signed-24
stereo sources.

The timing pass used the first 10 seconds. The quality pass used all 100
seconds from every source.

| Source | Prepared PCM bytes | SHA-256 |
| --- | ---: | --- |
| `bill-evans-secret-sessions` | 38,400,000 | `e3c96fffa05eefd2c8d96fbf71b31be295b5890cb70c8fb9d73356a373b21281` |
| `blue-nile-hats` | 38,400,000 | `a5ea40b18c8717c2d51772255ad4afaec674abd19c2a5d32672ddbedacf12a95` |
| `lori-asha` | 38,400,000 | `7d06b937e7012fb7fc701f82c6ec8d69a8ec8cd9dac9c1f1055d48c192c77b2e` |
| `nocturnal-animals` | 38,400,000 | `b961127010d0d576ab430b3101953a86c708dc851266535ffaee4ed8a3dd8435` |

The corpus manifest SHA-256 was
`7d0b87840a5ba23c6fde7e3226c402f0095751bbf052efdb9504bea583fd5bb4`.

The corpus audio is licensed local test data. It is not stored in this
repository.

## Builds

The reference was Opus trunk commit
`03647f524a40b05a1898522e92033810b58103c7`.

The C benchmark driver used `-O3 -DNDEBUG -march=native`. It linked the static
`libopus.a` from that source tree.

The Rust benchmark used release mode, ThinLTO, one code generation unit, and
`-C target-cpu=native`.

The final timing runner reused existing binaries. Therefore, its JSON records
both build-flag fields as `unknown`.

The recorded toolchain was Rust 1.97.1, Cargo 1.97.1, and GCC 12.2.0.

The recorded Opus configure string was unavailable. This limits exact C build
reproduction beyond its revision, library, and benchmark-driver flags.

The final timing capture started at `2026-08-24T20:28:15Z`. The quality capture
started at `2026-08-24T18:32:42Z`.

## Packet identity gate

Every timing round produced stable packet byte totals, packet ranges, and
decode checksums for each implementation.

All 48 Rust source and configuration cells matched the preceding checkpoint's
packet sizes and checksums.

Each accepted focused A/B test also reported zero packet or checksum changes
across its 20 pairs.

This gate isolates speed changes from encoder-decision changes on the measured
inputs. It does not prove identity on untested audio.

## Full-length quality gate

The quality pass processed 400 seconds per configuration. It aligned each
decode against its source before calculating SNR.

Rust used a 120-sample alignment offset. C used a 312-sample alignment offset
for the audio application.

| PCM | Mode | Rate | Rust SNR | C SNR | Mean delta | Source range | Rust bytes versus C |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16-bit | CBR | 192 kb/s | 25.03 dB | 25.02 dB | +0.015 dB | -0.05 to +0.09 dB | 0 B |
| 16-bit | CBR | 256 kb/s | 27.39 dB | 27.40 dB | -0.005 dB | -0.01 to +0.00 dB | 0 B |
| 16-bit | CBR | 320 kb/s | 28.34 dB | 28.34 dB | +0.000 dB | -0.06 to +0.05 dB | 0 B |
| 16-bit | constrained VBR | 192 kb/s | 24.97 dB | 25.01 dB | -0.035 dB | -0.09 to +0.07 dB | +350 B |
| 16-bit | constrained VBR | 256 kb/s | 27.32 dB | 27.36 dB | -0.040 dB | -0.09 to +0.02 dB | +471 B |
| 16-bit | constrained VBR | 320 kb/s | 28.33 dB | 28.35 dB | -0.018 dB | -0.07 to +0.03 dB | +527 B |
| 24-bit | CBR | 192 kb/s | 25.03 dB | 25.03 dB | +0.008 dB | -0.06 to +0.07 dB | 0 B |
| 24-bit | CBR | 256 kb/s | 27.41 dB | 27.40 dB | +0.010 dB | +0.00 to +0.03 dB | 0 B |
| 24-bit | CBR | 320 kb/s | 28.36 dB | 28.35 dB | +0.010 dB | -0.03 to +0.05 dB | 0 B |
| 24-bit | constrained VBR | 192 kb/s | 24.96 dB | 24.98 dB | -0.025 dB | -0.10 to +0.09 dB | +350 B |
| 24-bit | constrained VBR | 256 kb/s | 27.34 dB | 27.36 dB | -0.023 dB | -0.06 to +0.03 dB | +471 B |
| 24-bit | constrained VBR | 320 kb/s | 28.35 dB | 28.35 dB | +0.003 dB | -0.05 to +0.05 dB | +525 B |

CBR byte totals matched C exactly. Constrained-VBR differences were 350-527
bytes over 400 seconds.

These SNR values are regression diagnostics. They are not perceptual scores or
a transparency claim.

The final speed changes preserved packet identities on their A/B inputs and
the complete 10-second timing matrix. This evidence carried the quality gate
forward.

## Focused A/B method

Profiler-selected changes used one difficult native cell: 48 kHz stereo,
5 ms, signed 24-bit PCM, constrained VBR, and 256 kb/s.

Each variant processed 20 seconds from every corpus source. Five alternating
rounds produced 20 old/new source pairs.

The corpus delta compares the sum of four per-source medians. Pair statistics
compare each adjacent old/new source result.

| Accepted change | Corpus delta | Paired median | Paired mean | Faster pairs | Identity changes |
| --- | ---: | ---: | ---: | ---: | ---: |
| checked pulse-cache lookup | -0.1186% | -0.0409% | -0.0826% | 11/20 | 0 |
| static AVX2 pitch-tail mask | -0.6762% | -0.5841% | -0.6244% | 20/20 | 0 |
| checked forward-MDCT rotations | -0.6625% | -0.7059% | -0.6238% | 18/20 | 0 |
| remove internal allocation-vector clones | -0.3084% | -0.2589% | -0.2751% | 18/20 | 0 |

The accepted unsafe kernels have checked safe wrappers. Their safety
invariants are documented in [SAFETY.md](../../SAFETY.md).

## Rejected experiments

Candidates with a worse corpus aggregate were reverted. None remain in the
checkpoint.

| Rejected change | Corpus delta | Paired median | Paired mean | Faster pairs | Identity changes |
| --- | ---: | ---: | ---: | ---: | ---: |
| ordinary pulse-search inline hint | +0.0333% | +0.0802% | +0.0139% | 9/20 | 0 |
| forced pulse-search inline hint | +0.4448% | +0.3107% | +0.3928% | 4/20 | 0 |
| fused forward-MDCT fold loop | +0.0425% | -0.2334% | -0.2066% | 13/20 | 0 |
| skip allocation scratch zeroing | +0.1401% | +0.1373% | +0.0890% | 6/20 | 0 |
| reuse pulse-cache state | +0.1877% | +0.0445% | +0.0109% | 9/20 | 0 |
| reuse transient-analysis scratch | +0.8339% | +0.8120% | +0.7662% | 2/20 | 0 |

The fold experiment had a favorable paired median but a worse aggregate. The
aggregate gate prevented selection from a mixed result.

## Earlier combined A/B checkpoint

An earlier native build checkpoint was archived as `i24-avx-ab` at 256 kb/s.

It used four 20-second sources, three rounds, and seven internal repeats. The
label records the combined build, not isolated attribution.

| PCM | Mode | Encode delta | Decode delta |
| ---: | --- | ---: | ---: |
| 16-bit | CBR | -1.62% | +0.10% |
| 16-bit | constrained VBR | -1.65% | -0.16% |
| 24-bit | CBR | -2.38% | -0.17% |
| 24-bit | constrained VBR | -2.33% | -0.31% |

Packet byte totals and checksums were unchanged in every before/after pair.

## Profiler evidence

Native profiles selected the focused changes. Sample shares are diagnostic and
are not substitutes for the alternating A/B timings.

The Rust pitch-correlation tail initially used 1.48% of encode samples. The
static mask reduced it to 0.76%; C used 0.72%.

Forward-MDCT work fell from 1.74% to 1.52% of encode samples after the checked
rotation kernel.

Allocation-related work fell from 2.50% to 2.07% after removing three internal
vector clones.

A difficult single-source profile cell moved from 100.5959 ms encode and
23.8675 ms decode to 98.9203 ms and 23.7062 ms.

The matching C diagnostic was 100.9576 ms encode and 23.9792 ms decode. These
single runs are supporting evidence only.

## Reproduction

Build trunk libopus first. Set `OPUS_DIR` to that source tree.

Run the full local corpus gate:

```sh
OPUS_DIR=/path/to/opus-trunk \
  tools/run_soundkit_flac_corpus.py --cpu 2 --json corpus-results.json
```

Run a shorter timing check:

```sh
OPUS_DIR=/path/to/opus-trunk \
  tools/run_soundkit_flac_corpus.py --seconds 10 --rounds 3 --repeats 7 \
  --skip-quality --cpu 2 --json timing-results.json
```

Run the full-length quality gate:

```sh
OPUS_DIR=/path/to/opus-trunk \
  tools/run_soundkit_flac_corpus.py --quality-only --cpu 2 \
  --json quality-results.json
```

The runner expects the sibling SoundKit corpus by default. Use `--corpus` for
another prepared corpus directory.

## Evidence hashes

The final benchmark binary SHA-256 was
`64443c8c0e5c8a938f7072b41f87c77cbccb903d45a5232b49629fbf3bccbaf4`.

The final timing JSON SHA-256 was
`a1ff0fb6629d8e3104a4c2e33221a9e8a5737427c2c4c8a73361d7edc0c26ea1`.

| Timing checkpoint | JSON SHA-256 |
| --- | --- |
| native baseline after i24 work | `c0df74dbe049c9a94555ba61f6d8fcab7eba5b9aba303db1d4c5f46167901470` |
| static pitch-tail mask | `5ca35e5c9ef5fdca2ddd5d2c15671b81dc7daa6870ad3797482e8d673f888d72` |
| checked MDCT rotations | `e79ef871fa9d936d78990c60b2a4ec0af4b9be547b6b4b099b2e4b5f5f49f32a` |
| final allocation-clone removal | `a1ff0fb6629d8e3104a4c2e33221a9e8a5737427c2c4c8a73361d7edc0c26ea1` |

Its recorded Rust source SHA-256 was
`6cee2cca11df35063ec4b69fda5b7c5129a2cae518d104769026d642e9b12893`.

The full-length quality JSON SHA-256 was
`10a321e9cc2ec21741d5867e6aa6551eb16866b1a727c64a3c6eba9372524783`.

The earlier combined A/B JSON SHA-256 was
`26b8b2cd4044f33716080fb02220ef284a18e6d6637b417ad195843d77db38f7`.

| Focused A/B artifact | SHA-256 |
| --- | --- |
| accepted pulse-cache lookup | `6af6abd2e14410fe4a6e5c47ea4ab835e2acaad088422737bde803eba0288471` |
| accepted pitch-tail mask | `b9543aecd1bfb98459618c57db12be95046402c62c3ecf11ec267d880a0ffb39` |
| accepted MDCT rotations | `1de9e6a09ca90cdac9d57434757d9316cef256bc1b8682f9372aa2a0715acaf9` |
| accepted allocation-clone removal | `7ffc02bd450c5f8683a33f917a710998e9e0dd3af71169c0e908ec214c504fe6` |
| rejected ordinary inline | `cbd1c74d95b44be4820fa1563a8483c8c5c61789d143f9ca36827352135a60c5` |
| rejected forced inline | `6bb0e9aca5d64bab94c65cf12c075ebcc31b24e8263aa5da03a7948ab56ef30d` |
| rejected MDCT fold | `8af0a4446765423d154a7651e9c5ba8bcf389b03156156fd46bd97b7c301156d` |
| rejected scratch zeroing | `34d3e023747a7b288858f6c018fe912ffd52f60d75b17c22487cc3af6e655bc5` |
| rejected pulse-cache reuse | `a58e7f2fc9d8bc547bd6c2e1f6089be41e1b69326fe183eb07c30b04dc23557e` |
| rejected transient scratch | `8e9e80d26520ad9d17117c8f613a085c119603abf1ba13d2f6e2bca5332036d1` |

Raw timing artifacts remain local because they contain corpus paths. This
Markdown report records the durable results and verification hashes.

## Repository verification

`cargo test --all-targets` passed.

The kernel-specific Clippy gate passed before the kernel moved into the main package.

`cargo fmt --all -- --check` passed. `git diff --check` also passed.

The full public crate still reports existing Clippy findings. Therefore, the
whole-crate `-D warnings` command was not a clean gate for this checkpoint.

## Limits

This checkpoint covers one isolated x86 host. It does not establish AArch64 or
Wasm performance parity.

It covers only the current 48 kHz CELT path. SILK, hybrid Opus, Ogg Opus, and
other sample rates remain outside the implemented codec scope.

The quality measurements are diagnostic. Controlled listening tests remain
the gate for a transparency claim.
