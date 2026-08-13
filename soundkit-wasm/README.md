# soundkit-wasm

This crate provides the SoundKit browser and worker WebAssembly API. It includes
pure Rust media demuxers and supported codec decoders.

The controlled AAC production profile is stereo MPEG-4 AAC-LC at 44.1 or 48
kHz. Use `WasmAacLcDecoder` for raw access units.

Use `decodeSeekableStereoAacLc` for seekable M4A/MP4 files. The helper uses
bounded range reads and reuses one PCM output array.

Unsupported AAC profiles return an error. Route these inputs to a platform or
FDK AAC fallback decoder.

## Build

```sh
wasm-pack build soundkit-wasm --target web --out-dir pkg --release -- --features default
```

## Test

```sh
wasm-pack test --node --release soundkit-wasm \
  --no-default-features --features aac-lc-bench -- --nocapture
node scripts/test-video-wasm.mjs
```

See [`WASM_API.md`](../WASM_API.md) for the complete API.
