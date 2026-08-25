# SoundKit AAC-LC Benchmark

This crate is the performance and conformance harness for SoundKit's owned
AAC-LC decoder. It measures the decoder core directly on native targets and in
WebAssembly, without container parsing or subprocess startup in the timed
region.

The current decoder beats the direct FFmpeg, FDK-AAC, and Symphonia reference
implementations in WebAssembly on every checked fixture while retaining the
exact SoundKit output checksum and the FDK quality gates.

## Verified result

The canonical run was made on 2026-08-25 from commit `fb3e1f2` on the clean
`yl-encodec-1` Google Cloud performance host:

- Google Cloud `c4-highcpu-4`, four virtual CPUs
- Intel Xeon Platinum 8581C (Emerald Rapids)
- Debian 12 x86-64
- benchmark process pinned to CPU 2 with `taskset`
- `rustc 1.97.1`, `cargo 1.97.1`
- Node.js 18.20.4
- Binaryen `wasm-opt` 132
- FFmpeg 5.1.9 installed on the host
- FLAC 1.4.2 installed on the host
- CPU-only execution; no hardware codec or GPU path

Every runner received the same ADTS bytes. Each runner was called twice before
measurement, and each native benchmark entry point also performed a complete
untimed decoder warm-up. Measured rounds alternated forward and reverse runner
order. A round was rejected if any decoder changed its checksum or produced a
different number of frames or samples.

### Primary fixture

Fixture: `WESTSIDE_MIX_4_CONFIRMATION_130323_256k.aac`, 6,428,342 bytes,
9,171 ADTS frames, 48 kHz stereo, and 195.648 seconds per iteration.

The table reports the median of 11 alternating rounds with three complete
decode iterations per measurement. `x realtime` is decoded audio duration
divided by wall time; higher is better.

| Decoder | Median | Best | x realtime | Frames/sec | SoundKit faster |
| --- | ---: | ---: | ---: | ---: | ---: |
| SoundKit AAC-LC Wasm | **423.000 ms** | **421.000 ms** | **1,387.6x** | **65,042.6** | — |
| FFmpeg C Wasm | 497.785 ms | 494.913 ms | 1,179.1x | 55,270.8 | **15.02%** |
| FDK-AAC C++ Wasm | 3,312.287 ms | 3,270.430 ms | 177.2x | 8,306.3 | **87.23%** |
| Symphonia Wasm | 677.000 ms | 674.000 ms | 867.0x | 40,639.6 | **37.52%** |

Stable sampled checksums from all 11 rounds:

| Decoder | Checksum |
| --- | --- |
| SoundKit | `b37b53039cb16347` |
| FFmpeg | `c601fe670d2b4106` |
| FDK-AAC | `0c3dd6f7b7bb85e9` |
| Symphonia | `1d73629a86c9cb7b` |

Different implementations are not expected to emit bit-identical floating
point PCM. These checksums detect instability within each implementation; the
full SoundKit bit-exact golden and the cross-decoder quality gates are described
below.

### Diverse music corpus

The FLAC performance corpus was encoded as 48 kHz stereo AAC-LC at 256 kbit/s.
Each file contains 4,689 ADTS frames and approximately 100.032 seconds of audio.
The table reports medians from seven alternating rounds with three iterations
per measurement.

| Fixture | SoundKit median / xRT | FFmpeg median / xRT | Faster | FDK median / xRT | Faster | Symphonia median / xRT | Faster |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bill Evans, *The Secret Sessions* | **220.000 ms / 1,364.1x** | 257.056 ms / 1,167.4x | **14.42%** | 1,548.122 ms / 193.8x | **85.79%** | 323.000 ms / 929.1x | **31.89%** |
| The Blue Nile, *Hats* | **217.000 ms / 1,382.9x** | 251.085 ms / 1,195.2x | **13.57%** | 1,577.150 ms / 190.3x | **86.24%** | 329.000 ms / 912.1x | **34.04%** |
| Lori Asha | **216.000 ms / 1,389.3x** | 251.598 ms / 1,192.8x | **14.15%** | 1,635.777 ms / 183.5x | **86.80%** | 337.000 ms / 890.5x | **35.91%** |
| *Nocturnal Animals* | **205.000 ms / 1,463.9x** | 242.682 ms / 1,236.6x | **15.53%** | 1,725.687 ms / 173.9x | **88.12%** | 358.000 ms / 838.3x | **42.74%** |

The weakest observed margins across the corpus are therefore 13.57% over
FFmpeg, 85.79% over FDK-AAC, and 31.89% over Symphonia.

### Raw results and artifact identity

- [Primary 11-round log](results/2026-08-25-gcp-primary-fb3e1f2.txt), SHA-256
  `2d0558dcc15e405546d02c78d85d4070cdb630216696de906fd7ee4555e102ae`
- [Four-track corpus log](results/2026-08-25-gcp-corpus-fb3e1f2.txt), SHA-256
  `c284ff98bf9114cca6958ba7c7b4939fac7212c828d810a07e755258f4aaaa6b`

The logs record `lscpu`, compiler/runtime versions, every measured round, output
lengths, checksums, medians, and comparisons. The measured Wasm binaries were:

| Artifact | SHA-256 |
| --- | --- |
| SoundKit + Symphonia benchmark module | `2c170faf695a71d8a62db846d0e9b87075c70d0d005d837c7aac8e84643b5543` |
| FFmpeg direct decoder module | `cdc919ecffba1ef63d740825affeaba21b8ec071f71c386bc51b99ab5c7652ad` |
| FDK-AAC direct decoder module | `526bbd378bd32c0d0459bc1a54732f2389e0a403c44ec8029a86bec0ae72fa52` |

The FFmpeg reference was compiled from commit
`ca821e458aabe2fa` with Emscripten 4.0.23, `-O3`, and `-msimd128`. It contains
only the direct libavcodec AAC decoder path needed by the harness. The FDK-AAC
module uses the Wasm32 coefficient profile in `reference/build-fdk-wasm.sh`,
with `-O3`, LTO, and `-msimd128`.

## Native C checkpoint

The final checkpoint measures the public `soundkit-aac` production API against
an equivalent direct native libavcodec C harness. Eleven alternating rounds of
three complete decodes were used for each music fixture; medians below are
normalized to one decode.

| Fixture | SoundKit | FFmpeg C | SoundKit faster |
| --- | ---: | ---: | ---: |
| WESTSIDE full mix, 48 kHz stereo | **94.896 ms** | 99.764 ms | **4.88%** |
| Bill Evans — Secret Sessions | **49.684 ms** | 51.293 ms | **3.14%** |
| The Blue Nile — Hats | **49.619 ms** | 50.461 ms | **1.67%** |
| Lori Asha | **48.827 ms** | 49.885 ms | **2.12%** |
| Nocturnal Animals | **45.821 ms** | 47.955 ms | **4.45%** |

This corpus contains music only. The C harness performs the same production
work: fresh decoder construction, ADTS parsing, complete decode, optimized
planar-`f32` to interleaved-`i16` conversion, and full output consumption. See
[`../soundkit-aac/BENCHMARK_NATIVE_2026-08-25.md`](../soundkit-aac/BENCHMARK_NATIVE_2026-08-25.md)
for the complete methodology, checksums, hashes, and release matrix.

## Correctness and quality gates

The committed decoder passed all of the following after optimization:

```sh
cargo test -p soundkit-aac-lc
cargo test -p aac-wasm-bench --release \
  --no-default-features --features fdk,soundkit-lc -- --nocapture
RUSTFLAGS='-C target-feature=+simd128' \
  wasm-pack test --node --release soundkit-wasm \
  --no-default-features --features aac-lc-bench -- --nocapture
```

Results:

- 121 AAC-LC unit tests passed.
- The malformed-frame integration test passed without panics.
- The steady-state fixture test passed with zero allocations.
- Six benchmark/conformance tests passed, including both FDK comparisons.
- Two Wasm benchmark/quality tests and five Wasm AAC-LC API tests passed.
- The complete Wasm PCM golden remained exactly 18,782,208 samples, RMS
  `0.162843870`, peak `0.918334067`, and checksum
  `39efaeb0d96395e6`.

FDK decoder-oracle measurements:

| Fixture | RMSE | Mean absolute error | Maximum error | SNR | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| WESTSIDE 48 kHz stereo | 0.000745894 | 0.000044252 | 0.228224099 | 46.783 dB | pass |
| Stereo music 44.1 kHz | 0.000814123 | 0.000433316 | 0.020074591 | 37.865 dB | pass |

The enforced decoder-oracle limits are RMSE at most 0.005, mean absolute error
at most 0.001, maximum absolute error at most 0.50, SNR of at least 35 dB, and
equal decoded sample counts.

## Running the benchmark

Build the Rust benchmark library for Wasm SIMD:

```sh
RUSTFLAGS='-C target-feature=+simd128' \
  cargo build --lib -p aac-wasm-bench \
  --target wasm32-unknown-unknown --release \
  --no-default-features \
  --features soundkit-lc,symphonia,wasm-bindgen-api

wasm-bindgen \
  target/wasm32-unknown-unknown/release/aac_wasm_bench.wasm \
  --target web --out-dir aac-wasm-bench/pkg

wasm-opt -O4 --enable-simd \
  aac-wasm-bench/pkg/aac_wasm_bench_bg.wasm \
  -o aac-wasm-bench/pkg/aac_wasm_bench_bg.opt.wasm

mv aac-wasm-bench/pkg/aac_wasm_bench_bg.opt.wasm \
  aac-wasm-bench/pkg/aac_wasm_bench_bg.wasm
```

The direct FDK-AAC Wasm reference can be rebuilt from an FDK-AAC source tree:

```sh
aac-wasm-bench/reference/build-fdk-wasm.sh \
  /path/to/fdk-aac aac-wasm-bench/reference/pkg-fdk
```

Once the Rust, FFmpeg, and FDK packages are present, run all references in one
Node process:

```sh
taskset -c 2 node aac-wasm-bench/bench-all-wasm-node.mjs \
  3 11 pkg pkg-ffmpeg pkg-fdk \
  golden/aac/WESTSIDE_MIX_4_CONFIRMATION_130323_256k.aac
```

Arguments are iterations, rounds, Rust package name, FFmpeg package name, FDK
package name, and an optional ADTS fixture. Generated package directories are
ignored by Git; reference source and build scripts are committed.

## Native development loop

Use SoundKit-only checks while changing decoder internals:

```sh
cargo test -p aac-wasm-bench --no-default-features --features soundkit-lc
cargo test -p soundkit-aac-lc scalefactor
cargo test -p soundkit-aac-lc imdct_fast
```

Run the full native FDK conformance gate before accepting a change:

```sh
cargo test -p aac-wasm-bench --release \
  --no-default-features --features fdk,soundkit-lc -- --nocapture
```

Useful diagnostics remain available through the native CLI:

```sh
cargo run -p aac-wasm-bench --release -- quality-hotspots 8
cargo run -p aac-wasm-bench --release -- frame-features 1865 1630
cargo run -p aac-wasm-bench --release -- frame-errors 1865 1630
```
