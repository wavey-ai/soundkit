# TODO

Parity notes for `libopus-rs` against upstream libopus 1.5.2.

Reference against the exact upstream `v1.5.2` tag. The local C checkout at
`/Users/jamie/wavey.ai/opus` may be ahead of that tag, so use `git show
v1.5.2:...` or a pinned worktree when validating behavior.

## Current Baseline

- Pure Rust port, not a C wrapper.
- `#![forbid(unsafe_code)]` is active and should stay active.
- Current usable codec path is 48 kHz, CELT-only, fullband, raw Opus packets.
- `Encoder` supports 2.5, 5, 10, and 20 ms CELT frame sizes with CBR, heuristic
  constrained VBR, and exact compressed-frame-byte controls.
- `Decoder` currently handles single-frame CELT-only fullband packets at 48 kHz.
- Packet parsing, repacketizing, padding/unpadding, soft clipping, and many CELT
  primitives are already ported.

## P0: Finish Raw CELT Byte Parity

- Trace the next 2.5 ms / 128 kb/s CBR mismatch. The `tonality_slope` fix moved
  the first mismatch from frame 7 to frame 15. At frame 15, transient, spread,
  trim, and TF controls match libopus, but decoded allocation shows libopus
  coded bands = 18 while Rust coded bands = 20.
- Trace `clt_compute_allocation` inputs and loop state for that 2.5 ms frame-15
  mismatch: offsets, caps, trim, intensity, dual stereo, available bits,
  `lastCodedBands`, `signalBandwidth`, `band_bits`, `thresh`, and the skip
  flags written to the range coder.
- Verify the minimal Rust `AnalysisInfo.bandwidth` port against libopus. The
  bandwidth value is now computed and wired into encoder allocation, but it did
  not move the 2.5 ms coded-band mismatch, so either the analyzer bandwidth, the
  derived `signalBandwidth`, or allocation inputs still differ.
- Trace the next 5 ms / 128 kb/s CBR mismatch. The `tonality_slope` fix moved
  the first mismatch from frame 6 to frame 7. On frame 7, transient, spread,
  trim, TF, and coded-band controls match libopus, so the divergence is deeper
  in energy quantization, allocation bookkeeping, or PVQ band coding.
- Extend 2.5 ms CBR byte parity beyond the current 40-packet fixture at 48, 96,
  and 128 kb/s.
- Trace 10 and 20 ms / 128 kb/s CBR frame-0 parity after the matching
  high-level CELT controls. Transient, TF, spread, trim, and coded-band symbols
  now match libopus on that first frame, so the remaining divergence is in
  coarse/fine energy, allocation, or PVQ band quantization.
- Continue porting analysis-driven encoder inputs used by CELT quality
  decisions. The current Rust analyzer only covers the subset needed for
  `tonality_slope` and bandwidth; it still omits the full activity/music
  probability path, tone detection, leak boost, and any non-zero surround
  dynalloc plumbing needed by the public Opus encoder path.
- Replace the current Rust heuristic VBR sizing with libopus' constrained VBR
  target and reservoir logic after CBR is byte-identical.
- Keep exact-byte packet dump comparisons in the loop; every parity fix should
  identify the first divergent frame, symbol, and bit range.

## P1: Complete CELT Decode Coverage

- Decode all valid packet frame layouts, not only single-frame packets.
- Support CELT-only narrowband, mediumband, wideband, superwideband, and fullband
  packets, matching libopus packet bandwidth handling.
- Support valid Opus API sample rates: 8, 12, 16, 24, and 48 kHz.
- Implement packet-loss concealment for missing packets and the public
  `decode_fec` path instead of ignoring it.
- Track decoder state needed for CTL parity: final range, pitch, gain, last
  packet duration, bandwidth, phase inversion, and lookahead.
- Validate against upstream `tests/test_opus_decode.c` and `opus_demo` decode
  vectors.

## P2: Match the Opus Encoder API Surface

- Add typed Rust equivalents for the important `opus_encoder_ctl` settings:
  application, bitrate, max bandwidth, force bandwidth, VBR, constrained VBR,
  complexity, in-band FEC, packet-loss percentage, DTX, force channels, signal
  type, LSB depth, expert frame duration, prediction disabled, phase inversion
  disabled, DRED duration, DNN blob, lookahead, final range, and in-DTX query.
- Make `Application::Voip`, `Application::Audio`, and
  `Application::RestrictedLowDelay` drive the same mode, bandwidth, and delay
  decisions as libopus. They are currently recorded but the implemented path is
  still CELT-only.
- Add reset semantics and query methods equivalent to the C API behavior where
  they matter for deterministic tests.
- Unsupported public paths should return explicit errors until implemented; do
  not silently substitute CELT-only behavior for SILK, hybrid, FEC, or DRED.

## P3: Port SILK, Hybrid, and DRED

- Port SILK fixed and float common primitives plus their unit tests.
- Port the SILK decoder, including PLC, CNG, LBRR/FEC, stereo prediction, and
  resampling behavior.
- Port hybrid packet decode, including SILK/CELT transition handling and
  redundancy paths.
- Port the SILK encoder and the libopus mode-selection pipeline from
  `src/analysis.c`, `src/mlp.c`, and `src/opus_encoder.c`.
- Port DRED/deep redundancy support from libopus 1.5.2, including extension
  packet parsing and `tests/test_opus_dred.c` / `tests/test_opus_extensions.c`.
- Keep neural/deep PLC or external DNN-weight support behind explicit features
  if the model artifacts are too large for the default crate.

## P4: Multistream and Projection

- Port multistream encoder/decoder behavior, channel mapping validation, and
  self-delimited stream packing.
- Port projection and ambisonics encoder/decoder support if the crate intends
  to match the full libopus public surface.
- Extend padding/unpadding tests to include multistream packet validation against
  upstream `src/repacketizer.c`.
- Keep Ogg Opus container work separate from codec parity; libopus itself
  operates on Opus packets, not an Ogg container.

## P5: Upstream Test Parity

- Port or wrap upstream tests as Rust tests:
  - `tests/test_opus_api.c`
  - `tests/test_opus_decode.c`
  - `tests/test_opus_encode.c`
  - `tests/test_opus_padding.c`
  - `tests/test_opus_extensions.c`
  - `tests/test_opus_dred.c`
  - `tests/test_opus_projection.c`
  - CELT unit tests under `celt/tests/`
  - SILK unit tests under `silk/tests/`
- Build a deterministic parity harness that can run Rust and libopus 1.5.2 over
  the same PCM fixtures, applications, channel counts, frame sizes, bandwidths,
  bitrates, and CTL settings.
- The harness should fail with the first divergent frame and enough trace output
  to map the mismatch back to entropy symbols.
- Add fuzz/property tests for packet parsing, repacketizing, padding/unpadding,
  and invalid-packet handling.

## P6: Performance and Wasm

- Keep performance work behind correctness: profile after CELT byte parity is
  stable.
- Measure allocation hot spots in encode/decode and remove avoidable `Vec`
  churn in inner loops.
- Compare wasm encode/decode throughput against the C/libopusjs path using the
  existing benchmark scripts.
- Add SIMD only behind feature gates and checksum/parity tests.
