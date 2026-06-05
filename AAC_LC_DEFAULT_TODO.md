# AAC-LC Default Decoder TODO

Goal: move the SoundKit-owned pure Rust AAC-LC decoder work to main and make it
the default SoundKit AAC decode path where it is safe to do so.

## Current Status

- `soundkit-aac-lc` decodes the WESTSIDE AAC-LC music fixture: 9171/9171 raw
  access units.
- Native FDK oracle tolerance passes for the fixture.
- Source-WAV quality is close to the saved native FDK/Symphonia baselines:
  - FDK: RMSE 0.006896303, SNR 27.510 dB.
  - Symphonia: RMSE 0.006894503, SNR 27.511 dB.
  - SoundKit AAC-LC: RMSE 0.006919222, SNR 27.480 dB.
- `wasm32-unknown-unknown` + `wasm-bindgen` Node tests pass.
- `WasmAacLcDecoder` exposes raw access-unit decode to planar/interleaved
  `f32` PCM.
- SoundKit v2 frame stream payloads are covered in wasm: frame payloads can be
  parsed with `WasmSoundKitFrameDecoder` and decoded with `WasmAacLcDecoder`.
- Warmed steady-state fixture decode has a no-allocation guard.
- Unsupported AAC Main, HE-AAC/SBR, PS, PCE channels, channel layouts beyond
  stereo, and SBR fill payloads fail explicitly for fallback routing.
- The pure decoder does not depend on Symphonia.

## Before Merge To Main

- Keep the new workspace crates/files together:
  - `soundkit-aac-lc`
  - `aac-wasm-bench`
  - `soundkit-wasm-decoder` AAC-LC API/tests
  - README / WASM API updates
- Run the core gate:
  - `cargo test -p soundkit-aac-lc`
  - `cargo test -p aac-wasm-bench --no-default-features --features soundkit-lc`
  - `cargo test -p aac-wasm-bench --no-default-features --features fdk,soundkit-lc -- --nocapture`
  - `wasm-pack test --node --release soundkit-wasm-decoder --no-default-features --features aac-lc-bench -- --nocapture`
- Check the untracked crate directories are added intentionally and not missed
  by the merge.
- Do not include Symphonia as a decoder dependency. It remains benchmark-only in
  `aac-wasm-bench`.

## Make It The Default AAC Decoder

- In native SoundKit AAC decode:
  - Route AAC-LC mono/stereo raw access units to `soundkit-aac-lc` first.
  - Keep FDK fallback for HE-AAC/SBR/PS, AAC Main/LTP/ER profiles, PCE channel
    layouts, surround layouts, and any unsupported raw-block tools.
- In wasm/browser decode:
  - Use `WasmAacLcDecoder` for raw AAC-LC access units from the SoundKit packet
    stream.
  - Keep deboxing/demuxing separate. `WasmAacDeboxer` and
    `WasmAudioTrackDemuxer` should emit packets/config; AAC-LC decode consumes
    the raw access units after that.
  - If the stream is unsupported, return the explicit error to the caller so
    the app can choose a fallback decoder.
- Add feature flags that make the intended default clear:
  - `aac-lc` for the pure Rust decoder.
  - `aac-fdk-fallback` or equivalent for native fallback builds.
  - Avoid making FDK part of the wasm path.

## Remaining Hardening

- Add more AAC-LC fixtures before relying on the new decoder as the only
  production path:
  - 44.1 kHz stereo music.
  - 48 kHz stereo music.
  - Mono speech/music.
  - Low-complexity files without PNS/TNS.
  - Files with short-window transients, TNS, PNS, intensity stereo, and pulse
    where available.
- Add a small browser-style packet-stream example that wires:
  - SoundKit frame stream bytes
  - `WasmSoundKitFrameDecoder`
  - `WasmAacLcDecoder`
  - reusable `Float32Array` output
- Stabilize benchmark reporting:
  - Keep saved native FDK/Symphonia quality baselines.
  - Report wasm speed as a release checkpoint, but do not chase single-run
    variance when only tests/docs changed.
- Measure wasm binary size before enabling by default in browser bundles.
- Decide whether to add wasm SIMD for IMDCT/window/overlap hot loops before or
  after the default switch.
- Keep fallback behavior explicit for every unsupported profile/tool; do not
  silently decode partial HE-AAC/SBR/PS streams as AAC-LC.

## Good Pause Point

This is good enough to leave parked for now. The decoder is not yet broad enough
to make it the only AAC decoder everywhere, but it is strong enough to merge as
the SoundKit-owned AAC-LC path and start wiring as the preferred/default path
with FDK fallback for unsupported streams.
