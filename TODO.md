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
  the first mismatch from frame 7 to frame 15, and the `leak_boost` dynalloc
  port moved it to frame 22. Porting CELT's `FLOAT_APPROX` log2/exp2 helpers
  fixed the frame-22 fine-energy bit flip. Matching C's scaled-energy
  `band_log2` path for analysis leak boost fixed the frame-25 dynalloc split;
  mirroring libopus' energy-error feedback before trim analysis fixed the
  frame-29 trim split. The current 128 kb/s one-second dump first differs at
  frame 91.
- Trace that 2.5 ms frame-91 mismatch from the first divergent post-control
  entropy symbol. Decoded controls match C through trim, coded bands, intensity,
  balance, fine-energy bit counts, pulses, and collapse masks, so the next
  divergence is in the energy/allocation/PVQ payload path. A temporary
  primitive range-coder trace showed the committed stream matches through coarse
  energy, fine energy, and into theta/PVQ RDO when scratch-encoder candidate
  calls are accounted for. The first meaningful decision split found so far is
  theta RDO at band 14 on frame 91: C has `dist_down=112.171409607`,
  `dist_up=87.302574158`, and chooses the down-rounded candidate, while Rust has
  `dist_down=112.171386719`, `dist_up=128.589492798`, and chooses the
  up-rounded candidate. The next trace should compare band-14 stereo RDO
  resynthesis inputs/outputs, especially `quant_band_stereo`, `quant_band_mono`,
  `alg_quant`, `lowband_out`, and `stereo_merge`. A speculative Rust
  `stereo_merge` `mid2` change was tested and made the trace diverge earlier, so
  do not reapply it without a narrower proof.
- Trace the next 5 ms / 128 kb/s CBR mismatch. The `tonality_slope` fix moved
  the first mismatch from frame 6 to frame 7, and the `leak_boost` dynalloc port
  moved it to frame 9. Porting CELT's `FLOAT_APPROX` log2/exp2 helpers moved the
  first mismatch to frame 24, and matching C's analysis leak log scaling moved
  it to frame 139. Mirroring energy-error feedback before trim analysis makes
  the full one-second, 200-frame 128 kb/s CBR dump byte-identical; extend that
  fixture before treating 5 ms CBR as done.
- After the current first mismatches are fixed, extend 2.5 ms CBR byte parity
  beyond one-second fixtures at 48, 96, and 128 kb/s.
- Trace 10 and 20 ms / 128 kb/s CBR frame-0 parity after the matching
  high-level CELT controls. Transient, TF, spread, trim, and coded-band symbols
  now match libopus on that first frame, so the remaining divergence is in
  coarse/fine energy, allocation, or PVQ band quantization.
- Continue porting analysis-driven encoder inputs used by CELT quality
  decisions. The current Rust analyzer only covers the subset needed for
  `tonality_slope`, bandwidth, and leak boost; it still omits the full
  activity/music probability path, tone detection, and any non-zero surround
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
- Compare wasm encode/decode throughput against the C/libopusjs path with
  `tools/run_browser_wasm_compare.mjs` when system load is low.
- Decode perf finding: browser output transfer is not the main loss; a
  2026-06-01 microbench showed `DecodeResult.output` was only about 1-3% of a
  Rust decode pass. Native allocation probing showed hundreds of heap
  allocations per 20 ms stereo frame, mostly from recursive band/PVQ and
  synthesis scratch buffers. Prioritise reusable decoder scratch storage before
  JS API copy work.
- Decode perf progress from 2026-06-01: `decode_i16` now bypasses the
  intermediate f32 PCM allocation, decoder synthesis reuses channel/MDCT
  scratch, `alg_unquant`/`decode_pulses` use stack scratch for normal decode
  sizes, band folding avoids most per-band `lowband`/`lowband_out` `Vec` churn,
  and `decode_spectral_frame_into` reuses decode/allocation/band scratch across
  frames. The native allocation probe dropped from about 227/375 allocs per
  48/128 kb/s frame before the first pass to about 28/36 allocs per frame with
  `decode_i16`, or about 27/35 allocs per frame with `decode_i16_into`.
- Follow-up decode perf pass from 2026-06-01: the decoder now uses
  `parse_packet_slice` directly to avoid the public packet frame `Vec`, reuses
  CWRS/PVQ row scratch for decode-side `alg_unquant`, and reuses decoder-owned
  postfilter work/source buffers instead of allocating/cloning them when the
  CELT postfilter is active.
- CWRS follow-up from 2026-06-01: high-rate decode now caches computed
  `U(n,k)` rows in a decoder-owned four-way cache and `decode_index` exits as
  soon as all pulses are consumed. The symbolized 196 kb/s VBR reuse profile had
  `ncwrs_urow` at about 13.6% before this cache and about 3.6% after the first
  row-cache pass; the release browser spot checks are noisy but show the 196
  kb/s reuse decode path around parity or ahead of `libopusjs`.
- Settled direction from 2026-06-01: keep the scratch/cache decode path as the
  main path rather than carrying a feature-gated alternative. These changes are
  decode-side storage and lookup optimizations only; they are intended to keep
  identical entropy decisions, MDCT/postfilter math, and decoded samples for the
  same packet. Treat any difference as a correctness bug, not as a quality
  tradeoff.
- Browser-loaded synthetic grids after the CWRS cache are still load-sensitive.
  The focused 30-second 196 kb/s run
  (`/tmp/libopus-browser-wasm-after-cwrs-4way-cache-196.json`) showed Rust CBR
  reuse at `+10.6%` and VBR reuse at `+14.0%` decode versus `libopusjs`; a later
  VBR-only rerun
  (`/tmp/libopus-browser-wasm-after-cwrs-4way-cache-196-vbr-rerun.json`) showed
  `+2.6%`. The broader 10-second grid
  (`/tmp/libopus-browser-wasm-after-cwrs-4way-cache-full.json`) confirmed 48 and
  128 kb/s decode stayed ahead of `libopusjs`, but its late 196 kb/s rows were
  noisy. Prefer focused 30-second runs for high-rate comparisons.
- Chrome decode CPU profile support now lives in
  `tools/run_browser_wasm_compare.mjs --profile-rust-decode`. A 30-second
  128 kb/s CBR reuse profile was saved at
  `/tmp/libopus-rust-decode-128k.cpuprofile.json`; optimized wasm symbols are
  stripped, but the sampled hot area is in wasm, with visible GC now low.
- Build tuning check: one-run `wasm-opt` comparisons for 128 kb/s CBR reuse
  produced noisy but similar decode results for default wasm-pack output,
  no-opt, `-O3`, and `-O4`; do not change default wasm build flags without a
  lower-load repeated run.
- Decode perf next targets: profile the four-way CWRS cache under lower system
  load, then focus on the remaining high-rate hotspots: `decode_index`,
  `quant_partition`, `exp_rotation1`, range decoder division/update cost, and
  MDCT synthesis.
- Add SIMD only behind feature gates and checksum/parity tests.
