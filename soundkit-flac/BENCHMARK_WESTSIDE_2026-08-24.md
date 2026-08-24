# Five-millisecond FLAC packet benchmark — 2026-08-24

This report covers commit `a976f73`. `soundkit-flac` is optimized for one
independently decodable raw FLAC frame per call, using a persistent codec and
reused buffers. It is not a whole-file throughput benchmark.

The main input is the full 195.62-second Westside confirmation master: 39,124
complete stereo S24 frames at each rate. A 5 ms frame contains 240 samples per
channel at 48 kHz or 480 samples per channel at 96 kHz. `Realtime` is compared
with libFLAC level 0; `Balanced` is compared with libFLAC level 2.

All timings are median p50 microseconds per 5 ms call. “Lower” means less
latency than the named reference. FLAC is lossless: every decoded PCM sample
matched the source exactly. SoundKit decoded the libFLAC packets, FFmpeg
decoded the SoundKit packets, and the Wasm packets matched native SoundKit
byte-for-byte.

## Result summary

| Environment and corpus | Encode reference | Encode latency lower | Decode reference | Decode latency lower |
|---|---|---:|---|---:|
| Apple M1 native, full Westside | libFLAC | **60.7%** (2.56x throughput) | FFmpeg | **29.3%** (1.42x) |
| GCP Emerald Rapids native, full Westside | libFLAC | **64.0%** (2.79x) | FFmpeg | **38.7%** (1.63x) |
| GCP Emerald Rapids native, diverse corpus | libFLAC | **63.8%** (2.79x) | FFmpeg | **43.0%** (1.76x) |
| Apple M1 Node/Wasm, full Westside | native libFLAC | **33.1%** | native FFmpeg | **4.4%** |
| GCP Node/Wasm, full Westside | native libFLAC | **35.0%** | native FFmpeg | **1.7%** |

The native and Wasm cross-runtime rows answer deployment questions but are not
strict like-for-like runtime comparisons. The direct Wasm comparison below is:
SoundKit Wasm encoding averaged **38.6% lower latency (1.63x throughput)** than
libFLAC 1.5.0 built with Emscripten at `-O3 -msimd128`.

## Full Westside: Apple M1 native

Host: Apple M1, macOS 26.5, Rust/Cargo 1.96.0, FFmpeg 8.1.2, FLAC 1.5.0.

| Rate and profile | SoundKit encode | libFLAC encode | Lower | SoundKit decode | FFmpeg decode | Lower |
|---|---:|---:|---:|---:|---:|---:|
| 48 kHz Realtime / L0 | 1.583 | 3.750 | **57.8%** | 2.334 | 3.458 | **32.5%** |
| 48 kHz Balanced / L2 | 2.000 | 5.458 | **63.4%** | 2.416 | 3.458 | **30.1%** |
| 96 kHz Realtime / L0 | 2.958 | 7.042 | **58.0%** | 4.875 | 6.792 | **28.2%** |
| 96 kHz Balanced / L2 | 3.750 | 10.333 | **63.7%** | 4.875 | 6.625 | **26.4%** |

## Full Westside: GCP x86-64 native

Host: `yl-encodec-1`, `c4-highcpu-4`, four vCPUs on Intel Xeon Platinum
8581C (Emerald Rapids), Debian 12, Rust/Cargo 1.97.1, FFmpeg 5.1.9, FLAC
1.4.2.

| Rate and profile | SoundKit encode | libFLAC encode | Lower | SoundKit decode | FFmpeg decode | Lower |
|---|---:|---:|---:|---:|---:|---:|
| 48 kHz Realtime / L0 | 1.739 | 4.721 | **63.2%** | 2.019 | 3.334 | **39.4%** |
| 48 kHz Balanced / L2 | 2.119 | 6.330 | **66.5%** | 2.010 | 3.333 | **39.7%** |
| 96 kHz Realtime / L0 | 3.265 | 8.529 | **61.7%** | 3.807 | 6.170 | **38.3%** |
| 96 kHz Balanced / L2 | 4.022 | 11.417 | **64.8%** | 3.880 | 6.204 | **37.5%** |

## Diverse GCP corpus

The wider corpus contains ten deterministic 10-second excerpts from each of
four materially different sources, for 100 seconds (20,000 frames) per source
and rate:

- Lori Asha confirmation masters and album premixes
- The Blue Nile, *Hats*
- Bill Evans, *The Secret Sessions*
- Abel Korzeniowski, *Nocturnal Animals*

Each source was measured at 48/96 kHz and Realtime/Balanced. The table averages
the four rate/profile cells after taking the median of three alternating rounds
for each cell.

| Source | Encode latency lower than libFLAC | Decode latency lower than FFmpeg |
|---|---:|---:|
| Lori Asha | **63.7%** | **39.9%** |
| The Blue Nile — *Hats* | **63.6%** | **40.8%** |
| Bill Evans — *The Secret Sessions* | **65.8%** | **46.0%** |
| Abel Korzeniowski — *Nocturnal Animals* | **62.4%** | **45.5%** |
| **All 16 cells** | **63.8%** | **43.0%** |

Across genres, the 48 kHz average was 64.6% lower for encode and 45.0% lower
for decode. The 96 kHz average was 63.1% and 41.0%, respectively. Every one of
the 16 cells favored SoundKit for both operations.

## Node/WebAssembly

The WebAssembly build used Rust `opt-level=3`, one codegen unit, SIMD128, and
`wasm-opt`; calls used the buffered API so codec-owned linear-memory buffers
were reused. The final optimized module is 422,169 bytes. Node was warmed up,
and five alternating 50,000-call rounds were measured.

### Apple M1, Node 26.3.0

| Rate and profile | Wasm encode | native libFLAC | Lower | Wasm decode | native FFmpeg | Lower |
|---|---:|---:|---:|---:|---:|---:|
| 48 kHz Realtime / L0 | 2.750 | 3.750 | **26.7%** | 3.292 | 3.458 | **4.8%** |
| 48 kHz Balanced / L2 | 3.458 | 5.458 | **36.6%** | 3.375 | 3.458 | **2.4%** |
| 96 kHz Realtime / L0 | 4.875 | 7.042 | **30.8%** | 6.209 | 6.792 | **8.6%** |
| 96 kHz Balanced / L2 | 6.375 | 10.333 | **38.3%** | 6.500 | 6.625 | **1.9%** |

### GCP x86-64, Node 18.20.4

| Rate and profile | Wasm encode | native libFLAC | Lower | Wasm decode | native FFmpeg | Lower |
|---|---:|---:|---:|---:|---:|---:|
| 48 kHz Realtime / L0 | 3.153 | 4.721 | **33.2%** | 3.214 | 3.334 | **3.6%** |
| 48 kHz Balanced / L2 | 4.044 | 6.330 | **36.1%** | 3.352 | 3.333 | 0.6% higher |
| 96 kHz Realtime / L0 | 5.579 | 8.529 | **34.6%** | 5.865 | 6.170 | **4.9%** |
| 96 kHz Balanced / L2 | 7.314 | 11.417 | **35.9%** | 6.268 | 6.204 | 1.0% higher |

The exact same Wasm binary ran on ARM64 and x86-64. WebAssembly is portable
bytecode, but it is compiled by each engine to the host ISA, so latency still
depends on V8/JSC version, tiering, bounds checks, and the underlying CPU.

### Direct Wasm encode comparison

libFLAC 1.5.0 was built with Emscripten 4.0.23 at `-O3 -msimd128`. Its C
harness stayed inside Wasm for the timed loop, which avoids giving SoundKit an
artificial advantage from a per-frame JavaScript boundary.

| Rate and profile | SoundKit Wasm | Emscripten libFLAC | Lower |
|---|---:|---:|---:|
| 48 kHz Realtime / L0 | 2.750 | 4.500 | **38.9%** |
| 48 kHz Balanced / L2 | 3.458 | 5.750 | **39.9%** |
| 96 kHz Realtime / L0 | 4.875 | 7.958 | **38.7%** |
| 96 kHz Balanced / L2 | 6.375 | 10.125 | **37.0%** |

Compression stayed in the same ballpark and differed by at most roughly 0.04%
of packed PCM size in this comparison. An equivalent Emscripten libFLAC decode
harness has not yet been measured, so no direct same-runtime Wasm decode claim
is made here.

## Method and integrity

- Persistent codec instances and reusable packet/PCM buffers.
- 1,024 warm-up calls before each timed region.
- Full Westside: five alternating 50,000-call rounds on each native host.
- Diverse corpus: three alternating 20,000-call rounds per cell on GCP.
- SoundKit/reference order rotated to reduce scheduler and thermal bias.
- Single-threaded CPU paths only; no GPU or hardware codec acceleration.
- Rust release build; C references built with `-O3`.
- Identical input and packet fixtures for each encode/decode comparison.
- CRC-valid raw FLAC frames and comparable encoded sizes.
- Source PCM equality checked for every corpus frame before timing.

The ignored local corpus lives at
`testdata/flac-packet-bench/{westside,diverse-v1}` in the SoundKit workspace.
The manifests and corpus-building script make the diverse excerpts
deterministic; copyrighted audio is not committed.
