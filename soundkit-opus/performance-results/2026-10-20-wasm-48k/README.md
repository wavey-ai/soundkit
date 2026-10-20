# WebAssembly 48 kHz performance: 2026-10-20

This report compares the `soundkit-opus` WebAssembly encoder and decoder with
trunk libopus compiled to WebAssembly. Both modules ran in the same Node.js
process on the same PCM and the same packets.

## Result

The table gives the Rust time minus the libopus time, as a percentage of the
libopus time. A negative value means that Rust used less time.

| Host | Build | Encode | Decode |
|---|---|---:|---:|
| x86-64, Xeon Platinum 8488C | Before this work | +11.46% | +14.55% |
| x86-64, Xeon Platinum 8488C | This work, three runs | -3.44% to -3.59% | -2.82% to -3.70% |
| AArch64, Graviton3 (Neoverse V1) | Before this work | +18.99% | +16.37% |
| AArch64, Graviton3 (Neoverse V1) | This work, three runs | -0.82% to -0.88% | -0.64% to -0.74% |

Each value is the sum of 24 cell medians: 4 sources, 3 rates, and 2 rate
modes. The decoder values use the same packets for both decoders.

## Changes

The changes in this table do not change encoder packets or decoder PCM. The
harness compared packet bytes and a hash of the decoded PCM for every cell at
frame sizes of 2.5, 5, 10, and 20 ms.

| Change | Encode | Decode | Host |
|---|---:|---:|---|
| SIMD128 kernels for PVQ search, pitch correlation, and inner products; checked CWRS decoder on wasm32 | -1.76% | +0.71% | x86-64 |
| Fat LTO instead of thin LTO | -1.25% | -1.91% | x86-64 |
| Two-pass stereo i24 deemphasis with SIMD128 conversion | +0.75% | -3.42% | x86-64 |
| CWRS sign without a branch; PVQ candidate update with selects | -2.95% | -0.59% | x86-64 |
| Allocation loops over checked slices | -0.46% | -1.29% | x86-64 |
| `-C no-vectorize-slp` | -0.89% | -2.07% | x86-64 |
| SIMD128 radix-3, radix-4, and radix-5 FFT stages | -2.41% | -3.09% | x86-64 |
| SIMD128 radix-4 (`m == 1`) and radix-2 (`m == 4`) FFT stages | +1.36% | -1.16% | x86-64 |
| SIMD128 inverse-MDCT rotations and overlap mirror | -0.10% | -1.77% | AArch64 |
| SIMD128 forward-MDCT rotations | -1.09% | +1.16% | x86-64 |
| Analysis downmix state in locals | -0.60% | +0.34% | AArch64 |
| SIMD128 band-coding inner product | -0.55% | -0.11% | AArch64 |

These values are A/B deltas between consecutive builds at 192 and 320 kb/s, or
at 192, 256, and 320 kb/s. Values within about 1% are inside the run-to-run
variation.

The SLP vectorizer packed short scalar sequences, such as the two
`bitexact_cos` calls in `compute_theta`, into 128-bit operations. V8 builds
each 128-bit constant from immediates on every call. Before the change, this
code used 40% of the `compute_theta` decode cycles. The loop vectorizer
remains on: without it, encoding used 4.59% more time.

libopus has no WebAssembly SIMD FFT. With the SIMD128 butterflies, the Rust
FFT used 1,915 cycle samples in the x86-64 decode profile. The libopus FFT used
4,775.

The encoder settings in the
[2026-10-20 tuning report](../../quality-results/2026-10-20-encoder-tuning/README.md)
change the packets. They reduced encode time by 3.67%. Decode time changed by
+0.31%. Fewer frames use the pitch prefilter and fewer bands change their
time-frequency resolution.

## Build configuration

The `soundkit-opus` npm build uses `-C target-feature=+simd128`,
`-C no-vectorize-slp`, and fat LTO. Binaryen `wasm-opt -O3` and `-O4` made the
module 10% smaller. Neither changed the time by more than the run-to-run
variation. LLVM inline thresholds of 100 and 500 and `opt-level = 2` were also
within that variation.

The `soundkit-wasm` bundle now builds with `+simd128` and fat LTO, without
`-C no-vectorize-slp`, so the other codecs keep the SLP vectorizer. `make
wasm-nosimd` keeps a build without SIMD128. The harness measured the Opus code
in each bundle configuration:

| Configuration | libopus build | x86-64 encode | x86-64 decode | AArch64 encode | AArch64 decode |
|---|---|---:|---:|---:|---:|
| SIMD128, fat LTO, SLP on (`make wasm`) | SIMD | -2.79% | -3.26% | -0.35%, +0.02% | -0.51%, -1.06% |
| SIMD128, thin LTO, SLP on | SIMD | -2.19% | -1.35% | +3.77% | +2.22% |
| No SIMD, thin LTO (`make wasm-nosimd`) | no SIMD | +6.31% | +11.31% | +11.06% | +14.10% |
| No SIMD, fat LTO | no SIMD | +3.82% | +8.69% | | |

`bitexact_cos` and `bitexact_log2tan` are `#[inline(never)]`. With the SLP
vectorizer on, `compute_theta` then has no 128-bit operations.

The bundle with SIMD128 and fat LTO passed `scripts/test-video-wasm.mjs`, the
same as the checked-in package. Wasm SIMD needs Safari 16.4 or later.

## Method

- Runtime: Node.js 26.7.0 (V8), one process, pinned to one CPU with `taskset`
- x86-64 host: AWS `c7i.xlarge`, Ubuntu 24.04
- AArch64 host: AWS `c7g.xlarge`, Ubuntu 24.04
- libopus: trunk commit `03647f524a40b05a1898522e92033810b58103c7`, float build,
  Emscripten 4.0.23, `-O3 -flto -msimd128`, complexity 10, music signal,
  fullband, stack protector off
- Rust: rustc 1.96.0, `wasm32-unknown-unknown`, `-C target-feature=+simd128`,
  `-C no-vectorize-slp`, one codegen unit, fat LTO
- Corpus: the four SoundKit `diverse-v1` 48 kHz stereo signed 24-bit sources,
  20 seconds from each
- Frames: 240 samples per channel (5 ms), signed 24-bit PCM API
- Rates: 192, 256, and 320 kb/s, CBR and constrained VBR
- Five rounds per cell in alternating order; the report uses the median

Results from separate runs differed by up to 4% on the same host. Compare
builds only inside one run.

Linux `perf` with V8 `--perf-prof` gave cycle, instruction, and branch-miss
profiles per jitted wasm function. The profiles located the branch
mispredictions in the CWRS and PVQ code and the SLP vector code in
`compute_theta`.

## Reproduction

`tools/build_wasm_ab.sh` builds both modules. `tools/run_wasm_ab.mjs` runs the
comparison:

```sh
OPUS_SRC=/path/to/libopus EMSDK=/path/to/emsdk tools/build_wasm_ab.sh
node tools/run_wasm_ab.mjs --c ../target/opus-wasm-ab/wasm_ab_c.wasm \
  --rust ../target/opus-wasm-ab/wasm_ab_rust.wasm \
  --sources a.s32le,b.s32le --seconds 20 --rounds 5
```

The runner prints `identical pcm` for a cell when the two modules produced the
same packets and the same decoded PCM. Use two Rust builds as the two modules
to check that a change keeps the output.

`tools/wasm_ab/profile.mjs` prints self time per wasm function from the V8
profiler.
