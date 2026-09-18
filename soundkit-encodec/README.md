# soundkit-encodec

A SoundKit handler around `encodec-rs`, not another EnCodec implementation.

- `decode_to_sink(codec, lm, ecdc, emit)` decodes an extracted ECDC object into
  final, bounded SoundKit `AudioData` blocks. The caller owns the inference
  backend; returning an error from the sink stops further model work.
- `EncodecPcmDecoder` accepts existing model windows, cached s16 spans and
  silent windows. It delegates overlap-add, guard cropping and rounding to
  EnCodec's `SeamCursor`. Planar `PcmSegment` values can go straight to a
  platform PCM writer or be converted to SoundKit's interleaved `AudioData`.
- `browser::BrowserDecoder` (feature `browser`) adds the shared Rust entropy
  decoder and model hash check. `soundkit-wasm` exposes it as
  `WasmEncodecDecoder` with feature `encodec`. The browser keeps its lazy
  WebGPU/custom-WASM inference backend; JavaScript moves buffers, not symbols.
- `decode_planar_f32` preserves the existing BITNEEDLE native ABI's float
  output, including its established batch triangle accumulation. It is a
  compatibility adapter, not the streaming/s16 entry point.

PNG extraction, record validation, model downloads, cache lookup policy,
source-length alignment, storage and realtime deck DSP are outside this
codec handler. No model weights or inference runtime are bundled here.

For native pipeline output conversion/resampling, enable `encodec` in
`soundkit-decoder` and use `decode_encodec_to_sink`. Models are explicit:
automatic format detection never downloads or guesses a model.

Correctness tests (not performance benchmarks):

```sh
cargo test -p soundkit-encodec
cargo test -p soundkit-decoder --features encodec --test encodec
```

The Cargo manifests specify a Git revision of `encodec-rs` that provides the
shared `chunk_decode` API. Builds do not require an adjacent EnCodec checkout.
Existing EnCodec WASM entry points remain backward compatible.
