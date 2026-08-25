# AAC-LC Production Status

The SoundKit-owned AAC-LC decoder is wired into the default `soundkit-aac`
production API for ADTS and M4A/MP4. Supported MPEG-4 AAC-LC mono/stereo streams
use the owned backend; native builds automatically route unsupported AAC to
FDK-AAC. Each owned access unit contains 1,024 samples per channel.

`soundkit-wasm` now enters exclusively through `soundkit-aac` for ADTS, M4A/MP4,
and indexed raw access units. Its former direct `soundkit-aac-lc` dependency and
duplicate ADTS/MP4 adapters have been removed; `aac-lc` remains only as a Cargo
feature compatibility alias.

## Accepted Evidence

- The decoder processes all 9,171 access units in the full 195.648-second
  WESTSIDE music fixture.
- The decoder processes every access unit in the 44.1 kHz music fixture.
- Both fixtures pass the native FDK AAC quality thresholds.
- The fixtures cover short windows, TNS, PNS, intensity stereo, and mid-side stereo.
- The public production API beats native FFmpeg C on all five tracks in the
  music performance corpus by 1.73–5.27%.
- MP4 media tests use bounded range reads.
- Tests route explicit and implicit HE-AAC plus unsupported profiles to FDK.
- Warm decoder operation has a no-allocation test.
- The pure decoder does not depend on Symphonia.

## Acceptance Commands

```sh
cargo test -p soundkit-aac --release --all-features
cargo test -p soundkit-aac --release --no-default-features --features owned-lc
cargo test -p soundkit-aac --release --no-default-features --features fdk
cargo test -p soundkit-aac --release --no-default-features --features mp4-decoder
cargo test -p soundkit-aac --release --no-default-features --features mp4-fdk-fallback
cargo test -p soundkit-aac-lc --release
cargo test -p aac-wasm-bench --release --no-default-features --features fdk,soundkit-lc -- --nocapture
```

## Routing Policy

Use `soundkit_aac::AacDecoder::new()` for ADTS and `AacDecoderMp4` for M4A/MP4.
The default native feature set includes both the owned AAC-LC decoder and FDK
fallback. Use `new_soundkit_aac_lc()` when fallback must be prohibited.

Keep FDK AAC or a platform decoder for all other AAC inputs. This fallback
includes HE-AAC, SBR, PS, AAC Main, PCE layouts, and surround layouts.

## Residual Risks

- Pulse data has unit coverage but no collected real-music pulse fixture.
- The malformed-input test is deterministic smoke coverage, not continuous fuzzing.
- Performance results can change with system load.

## Release Decision

Ship the owned decoder as the default AAC-LC backend. Keep FDK enabled on native
targets for all unsupported AAC inputs. The full native evidence is recorded in
[`soundkit-aac/BENCHMARK_NATIVE_2026-08-25.md`](soundkit-aac/BENCHMARK_NATIVE_2026-08-25.md).
