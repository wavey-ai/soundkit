# soundkit-aac-lc

SoundKit-owned AAC-LC decoder core.

This crate is the home for a pure Rust, wasm-oriented AAC-LC decoder that accepts
raw AAC access units from the SoundKit packet stream and returns PCM.

## Initial Scope

- MPEG-4 AAC-LC only.
- Mono and stereo only.
- 1024-sample AAC-LC frames.
- Raw access-unit input; no MP4/M4A demuxing, deboxing, or ADTS streaming API.
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
     reference. wasm SIMD tuning is still pending.
   - windowing. AAC sine and KBD windows done.
   - overlap-add. Only-long, long-start, long-stop, and eight-short reusable-state
     paths are in place.
5. Add native benchmarks against FDK AAC as a correctness/performance oracle.
   Browser/worker wasm coverage lives in `soundkit-wasm-decoder` and is built
   with Rust `wasm32-unknown-unknown` plus `wasm-bindgen`.

The public API is already shaped around the final packet path. The decoder now emits
planar `f32` PCM for raw AAC-LC access units and can decode the
`golden/aac/WESTSIDE_MIX_4_CONFIRMATION_130323_256k.aac` fixture after stripping
ADTS headers: 9171/9171 access units decode and pass the native FDK oracle
tolerance. Run:

```sh
cargo run -p soundkit-aac-lc --example probe_adts_fixture
```

`aac-wasm-bench` now has an automated native FDK oracle conformance test for
this fixture, native benchmark lines for the SoundKit AAC-LC path, and
source-WAV quality measurements for WESTSIDE.
`tests/no_alloc_decode.rs` also guards the current fixture path against
steady-state heap allocations after decoder warmup.

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

Full production acceptance still needs broader fixture coverage, a
browser/worker packet harness, wasm SIMD, and wider fallback coverage for
remaining unsupported tools beyond AAC-LC. The current config and packet paths
already report explicit fallback errors for SBR/HE-AAC, PS, program-config
element channels, channel layouts beyond stereo, and SBR raw-block fill
payloads. The current DSP path
uses a `rustfft`-backed IMDCT, following the local `mel-spec`
CPU FFT precedent, and keeps its transform scratch buffers in reusable decoder
state. The hot bitreader path uses a buffered MSB-first reservoir, and synthesis
tracks previous/current window shapes per channel for AAC-correct overlap-add.
The current native WESTSIDE release baseline is about 36.7k frames/sec for
direct SoundKit AAC-LC decode, 33.8k frames/sec for reusable SoundKit decoder
state, 22.4k frames/sec for `fdk-aac-sys`, and 61.7k frames/sec for Symphonia.
Against the WESTSIDE source WAV, saved baselines are FDK RMSE 0.006896303 /
SNR 27.510 dB, Symphonia RMSE 0.006894503 / SNR 27.511 dB, and SoundKit
RMSE 0.006919222 / SNR 27.480 dB. The wasm-bindgen Node checkpoint computes
the SoundKit source-WAV quality metrics inside Rust wasm and reports the saved
FDK/Symphonia baselines without linking either comparator into the wasm build.
The current `wasm32-unknown-unknown` + `wasm-bindgen` Node release baseline is
about 31.1k frames/sec for core raw access-unit decode, 27.9k frames/sec through
the JS-facing interleaved API, and 28.6k frames/sec through the caller-provided
reusable `Float32Array` output API on the WESTSIDE fixture. Further performance
work should focus on a tighter MDCT, wasm SIMD, and fixture-driven edge cases
rather than another scalar IMDCT path.
