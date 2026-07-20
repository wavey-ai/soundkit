# soundkit-aac-lc

SoundKit-owned AAC-LC decoder core.

This crate contains a pure Rust, WebAssembly-oriented AAC-LC decoder. It accepts
raw AAC access units from the SoundKit packet stream and returns PCM.

## Initial Scope

- MPEG-4 AAC-LC only.
- Mono and stereo only.
- 1024-sample AAC-LC frames.
- Raw access-unit input. No MP4/M4A demuxing, deboxing, or ADTS streaming API.
- `AudioSpecificConfig` or equivalent metadata is required at initialization.
- HE-AAC, SBR, PS, ER AAC, AAC-LD, AAC-ELD, USAC/xHE-AAC, and channel layouts beyond
  mono/stereo are unsupported and must be routed to a fallback decoder.

## Decoder Plan

1. Parse and validate `AudioSpecificConfig`. Done.
2. Parse raw AAC access units with a zero-copy bitreader. SCE/CPE plus
   trailing `END`/`FIL` handling is done for the mono/stereo packet path.
   SBR extension payloads are rejected for fallback.
3. Implement AAC-LC syntax:
   - SCE/CPE element headers. Initial parser done.
   - individual channel stream payloads. Prefix, scalefactors, pulse/TNS flags, and
     long/short-window spectral handoff now share a reusable parser path.
   - section data. Initial parser done.
   - pulse data. Parser plus long-window quantized pulse application are done.
   - TNS data. Parser and inverse filtering are done for the AAC-LC packet path.
   - scale factor control flow. Parser and standard AAC scalefactor Huffman
     table are done.
   - scale-factor band layouts. Standard 1024-sample and 128-sample band tables
     for indexed MPEG-4 sampling frequencies are done.
   - Huffman spectral data. Pluggable spectral decoder interface plus signed/unsigned
     tuple and escape handling done. Standard AAC spectral codebooks 1-11 have
     table-backed decoder paths, escape magnitude handling for codebook 11, and
     generated VLC lookup tables to avoid per-bit/table scans.
   - dequantization. Table-backed pow43/scalefactor helpers plus long-window and
     grouped short-window spectral placement scaffolds are done. The production
     path decodes standard spectral data directly into scaled `f32` coefficients.
   - PNS/noise substitution. Stateful deterministic long-window and grouped
     short-window noise-band reconstruction is done.
   - stereo tools. Long-window and grouped short-window mid/side helpers plus
     long/short intensity stereo reconstruction are done. Mid/side skips PNS
     bands and right-channel intensity bands.
4. Implement DSP:
   - inverse quantization support tables
   - IMDCT. `rustfft`-backed DCT-IV path done and checked against the scalar
     reference. WASM SIMD tuning is still pending.
   - windowing. AAC sine and KBD windows done.
   - overlap-add. Only-long, long-start, long-stop, and eight-short reusable-state
     paths are in place.
5. Add native benchmarks against FDK AAC as a correctness/performance oracle.
   Browser/worker wasm coverage lives in `soundkit-wasm-decoder` and is built
   with Rust `wasm32-unknown-unknown` plus `wasm-bindgen`.

The public API is already shaped around the final packet path. The decoder emits
planar `f32` PCM for raw AAC-LC access units. It can decode the
`golden/aac/WESTSIDE_MIX_4_CONFIRMATION_130323_256k.aac` fixture after stripping
ADTS headers: 9171/9171 access units decode and pass the native FDK oracle
tolerance. Run:

```sh
cargo run -p soundkit-aac-lc --example probe_adts_fixture
```

`aac-wasm-bench` has an automated native FDK oracle conformance test for this
fixture. It also has native SoundKit AAC-LC benchmark lines and source-WAV
quality measurements for WESTSIDE.
`tests/no_alloc_decode.rs` also guards the current fixture path against
steady-state heap allocations after decoder warmup.

## Fast Native Loop

Use this native-only loop while iterating on `soundkit-aac-lc`. Reserve wasm,
FDK/Symphonia, and full release benchmarks for integration checkpoints:

```bash
cargo test -p aac-wasm-bench --no-default-features --features soundkit-lc
cargo test -p soundkit-aac-lc scalefactor
cargo test -p soundkit-aac-lc imdct_fast
```

The full native benchmark uses the WESTSIDE AAC fixture and, when the sibling
`bitneedle` checkout is present, compares decoded AAC with the source WAV. It
reports the result as `source-vs-*` measurements. Override the source WAV path with
`SOUNDKIT_AAC_SOURCE_WAV=/path/to/source.wav`.

## Current Benchmarks

Native command:

```bash
cargo run -p aac-wasm-bench --release -- 5
```

Fixture: `golden/aac/WESTSIDE_MIX_4_CONFIRMATION_130323_256k.aac`, 9171 ADTS
frames, 48 kHz stereo, 195.648 seconds of AAC frame audio.

| Decoder | Iterations | Decoded frames | Elapsed | RTF | Frames/sec | RMS | Peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fdk-aac-sys` | 5 | 45,855 | 2,045.506 ms | 0.002091 | 22,417.4 | 0.162843351 | 0.918304443 |
| `soundkit-aac-lc` | 5 | 45,855 | 1,250.680 ms | 0.001278 | 36,664.1 | 0.162846547 | 0.918334007 |
| `soundkit-lc-reuse` | 5 | 45,855 | 1,355.441 ms | 0.001386 | 33,830.3 | 0.162846547 | 0.918334007 |
| `symphonia-aac` | 5 | 45,855 | 743.409 ms | 0.000760 | 61,682.1 | 0.162845552 | 1.001383424 |

Decoder-oracle checks compare decoded PCM against FDK with channel-aligned
offset search:

| Comparison | Status | RMSE | Mean abs | Max abs | SNR |
| --- | --- | ---: | ---: | ---: | ---: |
| `fdk-vs-symphonia` | pass | 0.001002403 | 0.000045033 | 0.260961175 | 44.215 dB |
| `fdk-vs-soundkit` | pass | 0.001242798 | 0.000132421 | 0.370039735 | 42.348 dB |

Source-WAV quality measurements compare each AAC decode against
`WESTSIDE_MIX 4 CONFIRMATION_130323.wav`:

| Comparison | RMSE | Mean abs | Max abs | SNR | p99 abs | p999 abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `source-vs-fdk` | 0.006896303 | 0.004594121 | 0.334225774 | 27.510 dB | 0.023480892 | 0.038754225 |
| `source-vs-soundkit` | 0.006919222 | 0.004613361 | 0.383684549 | 27.480 dB | 0.023648702 | 0.038796075 |
| `source-vs-symphonia` | 0.006894503 | 0.004592889 | 0.363226533 | 27.511 dB | 0.023478046 | 0.038671419 |

Wasm command:

```bash
wasm-pack test --node --release soundkit-wasm-decoder --no-default-features --features aac-lc-bench -- --nocapture
```

The wasm quality test computes SoundKit PCM inside Rust wasm and reports saved
native FDK/Symphonia baselines without linking either comparator into wasm:

| Comparison | Source | Compared samples | Offset | RMSE | Mean abs | Max abs | p99 abs | p999 abs | SNR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `saved-source-vs-fdk` | saved native | - | - | 0.006896303 | 0.004594121 | 0.334225774 | 0.023480892 | 0.038754225 | 27.510 dB |
| `saved-source-vs-symphonia` | saved native | - | - | 0.006894503 | 0.004592889 | 0.363226533 | 0.023478046 | 0.038671419 | 27.511 dB |
| `wasm-source-vs-sk` | computed in wasm | 18,779,958 | 2,048 samples | 0.006919222 | 0.004613361 | 0.383684535 | 0.023648679 | 0.038796104 | 27.480 dB |

| Wasm path | Iterations | Decoded frames | Elapsed | RTF | Frames/sec | Checksum |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `wasm-core-raw` | 5 | 45,855 | 1,476.331 ms | 0.001509 | 31,060.1 | `24eb18ebd8fc27e4` |
| `wasm-js-interleaved` | 5 | 45,855 | 1,642.932 ms | 0.001679 | 27,910.5 | `8c2d7264b07abe0a` |
| `wasm-js-into` | 5 | 45,855 | 1,605.535 ms | 0.001641 | 28,560.6 | `0d4a1cec5595b60a` |

## Wasm Build

The pure decoder builds directly for Rust wasm:

```sh
cargo build -p soundkit-aac-lc --target wasm32-unknown-unknown --release
```

The browser/worker API is exposed from `soundkit-wasm-decoder` with
`wasm-bindgen`:

```sh
wasm-pack build soundkit-wasm-decoder --target web -- --no-default-features --features aac-lc
wasm-pack test --node soundkit-wasm-decoder --no-default-features --features aac-lc
wasm-pack test --node --release soundkit-wasm-decoder --no-default-features --features aac-lc-bench -- --nocapture
```

The existing SoundKit wasm feature path also builds after the frame-header API
update:

```sh
cargo build -p soundkit --target wasm32-unknown-unknown --release --no-default-features --features wasm
```

## Status Notes

Full production acceptance still needs:

- broader fixture coverage
- a browser and worker packet harness
- WebAssembly SIMD
- more fallback coverage for unsupported tools beyond AAC-LC.

The current config and packet paths report fallback errors for SBR/HE-AAC, PS,
program-config element channels, channel layouts beyond stereo, and SBR
raw-block fill payloads. The current DSP path uses a `rustfft`-backed IMDCT. This
follows the local `mel-spec` CPU FFT precedent. It keeps transform scratch
buffers in reusable decoder state.

The hot bitreader uses a buffered MSB-first
reservoir. Synthesis tracks previous and current window shapes for each channel
to get correct AAC overlap-add.
Further performance work should focus on a tighter MDCT, wasm SIMD, and
fixture-driven edge cases rather than another scalar IMDCT path.
