# Opus 1 ms change verification

The encoder change uses a stack array for channel slices during the pitch
search. The allocation test uses the SoundKit encoder with caller-owned
packet storage and signed 24-bit PCM. After warm-up, the original mono path
allocated 100 times in 100 pitch-search frames. The revised mono and stereo
paths each allocated zero times in 100 frames.

At 192, 256 and 320 kb/s CBR, the revised encoder produced the same packet
bytes as SoundKit `28120cc3f537b607833fbd64d2900e18bad1ffd7` across all 24
streams in the [quality comparison](README.md). Each stream has 8,020 1 ms
frames. The allocation change therefore preserves the measured quality and
bitrate for those inputs.

## Native tests

147 release tests passed and five were ignored by default. The live libopus
test was run separately and passed against xiph/opus
`316c675fa1c9982d76e9cdcba7231076f75047da`. The other ignored tests generate
diagnostic files or refresh golden vectors.

These tests ran with Rust 1.100.0 in a workspace containing `soundkit` and
`soundkit-opus`, with the repository's release profile. Resolution in the
full workspace stopped at `soundkit-speex`, which requests `oxideav-speex`
`0.0.9` from an unpinned Git branch whose head reports `0.0.12`. The
[reproduction guide](README.md) describes the two-crate setup and pinned C
build. In that workspace, run:

```sh
cargo test --release -p soundkit-opus --no-fail-fast
OPUS_CUSTOM_INTEROP_C=/path/to/opus_custom_interop_c \
  cargo test --release -p soundkit-opus --test opus_custom_vectors \
  live_libopus_comparison -- --ignored
```

## Native benchmark integration

One-second synthetic 48 kHz stereo runs through the
[benchmark runner](../../CODEC.md#raw-celt-benchmark) verified the CBR
parameters, raw packet bytes and decoded delay. Rust used CBR and C used
restricted-lowdelay CBR.

| Rust / C frame | Rust / C target, kb/s | Rust / C encoded bytes | Rust / C packet bytes | Rust / C decoded delay, samples |
|---|---:|---:|---:|---:|
| 2.5 / 2.5 ms | 256 / 256 | 32,000 / 32,000 | 80 / 80 | 120 / 120 |
| 1 / 2.5 ms | 256 / 256 | 32,000 / 32,000 | 32 / 80 | 48 / 120 |
| 1 / 2.5 ms | 264 / 192 | 33,000 / 24,000 | 33 / 60 | 48 / 120 |

With `--skip-quality`, the runner displayed `n/a` in both SNR and delay
columns. Timing measurements on the repository's
[performance host](../../../AGENTS.md) are pending.

## Standalone WebAssembly package

The checked-in `soundkit-opus/pkg` bundle was regenerated with the package's
`npm run build` command using Rust 1.100.0, wasm-pack 0.15.1, wasm-bindgen
0.2.130 and Binaryen 130. The build uses SIMD128, fat LTO and disabled SLP
vectorization. A Node smoke test passed for 48, 64, 72 and 120 samples, custom
loss concealment and 48-sample signalling-off mode.

`npm pack` ran the `prepare` build. The extracted package passed the same
smoke tests and its WASM matched the checked-in binary byte for byte.
