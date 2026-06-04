# Opus Benchmark Comparison (SoundKit vs libopus-rs vs C)

## Date
Run on 2026-05-29.

## Command
`cargo run -p lori-asha-premix-bench`

## C binding support check
- `soundkit-opus/Cargo.toml` depends on `libopus` and sets
  `crate-type = ["cdylib", "rlib"]`, so the C-compatible crate path is already part of the existing SoundKit setup.
- `soundkit-opus/src/lib.rs` includes opus encoder/decoder tests under `#[cfg(test)]`, including decode/streaming and encode/decode roundtrip fixtures.

## Results (full 22-track run)

### Aggregate

Soundkit-opus:
- Tracks: 22
- Encode RTF: `0.004x` (18.064s total)
- Decode RTF: `0.002x` (6.919s total)
- Encoded output: `68259789` bytes (`126.99` kbps)
- Decoded output: `412846080` bytes (`768.04` kbps)
- Avg SNR(dB): `-4.01`
- Avg RMSE: `7695.212`
- Avg MAE: `5625.791`
- Max abs err: `61896.000`
- Avg length delta samples: `523.32`

libopus-rs:
- Tracks: 22
- Encode RTF: `0.004x` (18.442s total)
- Decode RTF: `0.002x` (6.972s total)
- Encoded output: `68259789` bytes (`126.99` kbps)
- Decoded output: `412846080` bytes (`768.04` kbps)
- Avg SNR(dB): `-4.01`
- Avg RMSE: `7695.212`
- Avg MAE: `5625.791`
- Max abs err: `61896.000`
- Avg length delta samples: `523.32`

libopus C (FFI via `opus-sys` in benchmark):
- Tracks: 22
- Encode RTF: `0.004x` (18.154s total)
- Decode RTF: `0.001x` (5.973s total)
- Encoded output: `68259789` bytes (`126.99` kbps)
- Decoded output: `412846080` bytes (`768.04` kbps)
- Avg SNR(dB): `-4.01`
- Avg RMSE: `7695.212`
- Avg MAE: `5625.791`
- Max abs err: `61896.000`
- Avg length delta samples: `523.32`

## Results (query: 1979_MIX)

Soundkit-opus:
- Tracks: 3
- Encode RTF: `0.005x` (3.566s total)
- Decode RTF: `0.002x` (1.349s total)
- Encoded output: `10765104` bytes (`125.49` kbps)
- Decoded output: `65886720` bytes (`768.02` kbps)
- Avg SNR(dB): `-3.83`
- Avg RMSE: `26.903`
- Avg MAE: `18.235`
- Max abs err: `231.000`
- Avg length delta samples: `278.67`

libopus-rs:
- Tracks: 3
- Encode RTF: `0.005x` (3.540s total)
- Decode RTF: `0.002x` (1.337s total)
- Encoded output: `10765104` bytes (`125.49` kbps)
- Decoded output: `65886720` bytes (`768.02` kbps)
- Avg SNR(dB): `-3.83`
- Avg RMSE: `26.903`
- Avg MAE: `18.235`
- Max abs err: `231.000`
- Avg length delta samples: `278.67`

libopus C (FFI):
- Tracks: 3
- Encode RTF: `0.005x` (3.506s total)
- Decode RTF: `0.002x` (1.136s total)
- Encoded output: `10765104` bytes (`125.49` kbps)
- Decoded output: `65886720` bytes (`768.02` kbps)
- Avg SNR(dB): `-3.83`
- Avg RMSE: `26.903`
- Avg MAE: `18.235`
- Max abs err: `231.000`
- Avg length delta samples: `278.67`

## Summary
- The C backend (`libopus-sys` path used in benchmark) is materially in-band with SoundKit output quality and bitrate for all tested tracks.
- decode-side throughput is marginally better in the C path than both Rust wrapper paths in this run.
