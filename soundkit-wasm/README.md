# soundkit-wasm

This crate provides the SoundKit browser and worker WebAssembly API. It includes
pure Rust media demuxers and supported codec decoders.

The controlled AAC production profile is stereo MPEG-4 AAC-LC at 44.1 or 48
kHz. All AAC paths enter through `soundkit-aac`: `Decoder` handles
ADTS and streamable M4A/MP4, while `WasmAacLcDecoder` handles indexed raw access
units.

Use `decodeSeekableStereoAacLc` for seekable M4A/MP4 files. The helper uses
bounded range reads and reuses one PCM output array.

Unsupported AAC profiles return an error. Route these inputs to a platform or
FDK AAC fallback decoder.

The compatibility Cargo feature `aac-lc` now enables the production `aac`
feature; `soundkit-wasm` no longer depends directly on `soundkit-aac-lc`.

## Build

The package uses WebAssembly SIMD128 and fat LTO. Chrome 91, Firefox 89, and
Safari 16.4 and later support WebAssembly SIMD.

```sh
CARGO_PROFILE_RELEASE_LTO=fat \
CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS='-C target-feature=+simd128' \
  wasm-pack build soundkit-wasm --target web --out-dir pkg --release -- --features default
```

`make wasm` runs this command. For an engine without WebAssembly SIMD, run
`make wasm-nosimd`, which writes `pkg-nosimd`. In that build, the Opus encoder
used 3.8-6.3% more time than libopus without SIMD, and the decoder 8.7-11.3%
more. With SIMD128 and fat LTO, the Opus encoder and decoder used less time
than libopus with SIMD on x86-64, and the same time within 1% on AArch64.

## Test

```sh
wasm-pack test --node --release soundkit-wasm \
  --no-default-features --features aac-lc-bench -- --nocapture
node scripts/test-video-wasm.mjs
```

See [`WASM_API.md`](../WASM_API.md) for the complete API.
