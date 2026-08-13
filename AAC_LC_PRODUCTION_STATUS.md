# AAC-LC Production Status

The production profile is MPEG-4 AAC-LC stereo at 44.1 or 48 kHz. Each access
unit contains 1,024 samples per channel.

## Accepted Evidence

- The decoder processes all 9,171 access units in the 48 kHz music fixture.
- The decoder processes all access units in the 44.1 kHz music fixture.
- Both fixtures pass the native FDK AAC quality thresholds.
- The fixtures cover short windows, TNS, PNS, intensity stereo, and mid-side stereo.
- Release WASM tests pass in Node.
- MP4 media tests use bounded range reads.
- The browser path reuses one `Float32Array` for packet output.
- Tests reject HE-AAC and unsupported profiles for fallback.
- Warm decoder operation has a no-allocation test.
- The pure decoder does not depend on Symphonia.

## Acceptance Commands

```sh
cargo test -p soundkit-aac-lc
cargo test -p aac-wasm-bench --no-default-features --features soundkit-lc
cargo test -p aac-wasm-bench --release --no-default-features --features fdk,soundkit-lc -- --nocapture
wasm-pack test --node --release soundkit-wasm --no-default-features --features aac-lc-bench -- --nocapture
wasm-pack build soundkit-wasm --target web --out-dir pkg --release -- --features default
node scripts/test-video-wasm.mjs
```

## Routing Policy

Use `soundkit-aac-lc` for controlled stereo AAC-LC access units. Use
`decodeSeekableStereoAacLc` for seekable M4A/MP4 files in browsers.

Keep FDK AAC or a platform decoder for all other AAC inputs. This fallback
includes HE-AAC, SBR, PS, AAC Main, PCE layouts, and surround layouts.

## Residual Risks

- Pulse data has unit coverage but no real music fixture.
- The malformed-input test is deterministic smoke coverage, not continuous fuzzing.
- The default WASM bundle is 3,234,574 bytes before compression.
- The full bundle is 1,221,718 bytes with gzip.
- Performance results can change with system load.

## Release Decision

Use the Rust decoder for controlled stereo AAC-LC music. Keep an explicit
fallback for all other AAC inputs.
