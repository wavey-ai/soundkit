# TODO

Roadmap and parity notes for the pure Rust `libopus-rs` port against upstream
libopus `v1.5.2`.

Use the exact upstream `v1.5.2` tag for reference checks. A local C checkout may
be ahead of that tag, so prefer a pinned worktree, `git show v1.5.2:...`, or the
benchmark helper's configured `OPUS_DIR` when validating behavior.

## Current Baseline

- The crate is a safe Rust implementation, not a C wrapper; keep
  `#![forbid(unsafe_code)]` intact.
- The usable codec path is 48 kHz CELT-only fullband raw Opus packets.
- `Encoder` supports 2.5, 5, 10, and 20 ms CELT frames with CBR,
  libopus-style constrained VBR target/reservoir control, and exact
  compressed-frame-byte controls.
- `Decoder` supports single-frame CELT-only fullband packets and exposes
  reusable-output `decode_i16_into` and `decode_f32_into` paths.
- Packet parsing, repacketizing, padding/unpadding, soft clipping, entropy,
  CELT MDCT/PVQ/band coding, quantized energy, pitch prefilter/postfilter,
  dynamic allocation, and synthesis/deemphasis are already ported.
- Raw CELT parity tooling exists on both sides:
  `examples/raw_celt_bench.rs`, `tools/raw_celt_bench.c`,
  `examples/celt_packet_probe.rs`, `examples/raw_celt_decode_dump.rs`, and
  `tools/raw_celt_decode_dump.c`.

## Current Parity Checkpoint

- The decoder quality checkpoint is healthy for same-packet checks. Cross-decode
  runs with C-generated and Rust-generated packet dumps agree within about
  `0.01 dB` max aligned-SNR delta and no lag mismatches on the checked matrix.
- The decoder now mirrors libopus' in-place CELT postfilter, tracks
  `oldLogE`/`oldLogE2` for anti-collapse, applies the decoded-silence energy
  floor, and has an allocation-sensitive f32 decode path.
- The 2.5 ms / 128 kb/s CBR one-second packet dump is byte-identical to C for
  all 400 frames.
- A real coarse-energy bug was fixed: Rust used to record `qi0` before the
  max-decay clamp, while libopus records it after the clamp. The regression is
  `celt_encoder_counts_decay_limited_coarse_energy_badness_like_c`.
- The remaining 2.5 ms high-rate CBR packet splits currently look like floating
  threshold sensitivity, not confirmed Rust algorithm bugs:
  - 160 kb/s frame 226: first split is fine-energy band 0 channel 0
    (`q2=15` Rust, `q2=14` C) at the `-0.03125` residual threshold.
  - 256/384/512 kb/s frame 17: first split is fine-energy band 0 channel 1
    (`q2=26` Rust, `q2=27` C), with about `0.0009 dB` drift already visible
    before MDCT.
- VBR now uses the libopus-style target path for activity, stereo saving,
  dynalloc, transient/TF boost, tonality, pitch-change boost, temporal VBR, and
  the constrained reservoir. The temporal-VBR rate factor now uses CELT
  `equiv_rate`, matching upstream `compute_vbr()`, instead of the nominal Opus
  bitrate. In the latest one-second VBR matrix the largest observed gap remains
  about `0.21 dB`; the mean absolute quality gap is down from about
  `0.0567 dB` to `0.0392 dB`.
- Rust still keeps a `0.50` constrained-VBR blend for non-5 ms frames while
  libopus uses `0.67` everywhere. The C value fixes the 5 ms / 48 kb/s gap, but
  a traced post-`equiv_rate` experiment still worsened the VBR mean abs quality
  gap to about `0.0658 dB` and worsened mean per-row packet-length closeness
  from about `1.83` to `2.28` bytes, mostly on 10/20 ms rows. Keep the
  conservative blend until the lower-level CELT coding/allocation mismatch is
  removed.
- The largest CBR gap is still 5 ms / 48 kb/s, now about `0.30 dB` after
  porting the Opus-layer stereo-width fade and analysis `max_pitch_ratio`
  prefilter scaling.

## Stop Point: 5 ms / 48 kb/s CBR

This is the best next correctness target because it is the largest CBR quality
gap and diverges at frame 1.

- Frame 0 is byte-identical.
- Frame 1 no longer splits at allocation trim; Rust and C both select trim
  index `3` after the Opus-layer stereo-width fade is applied before CELT
  preemphasis.
- The benchmark fixture generation is textually aligned between Rust and C.
- The Opus-layer high-pass filter and CELT preemphasis formulas match libopus;
  the previous apparent preemphasis mismatch was caused by tracing before C's
  stereo-width fade.
- Packet probing shows frames 1-4 now align on transient, prefilter, spread,
  trim, coded bands, intensity, dual stereo, balance, and fine-energy bits. The
  next visible split is pulse distribution, likely caused by coarse-energy
  symbol/tell drift before allocation rather than a remaining high-level control
  mismatch.

## P0: Finish CELT CBR/VBR Quality Parity

- Resume the 5 ms / 48 kb/s CBR trace inside coarse energy/PVQ after the
  high-level controls. Patch any confirmed Rust algorithm issue; otherwise
  document the drift with C/Rust evidence and move to the next quality gap.
- Continue the VBR trace from the target/reservoir checkpoint. Current evidence
  shows the target inputs are closer after the `equiv_rate` fix, but C's
  universal `0.67` constrained blend exposes lower-level CELT coding/allocation
  quality gaps instead of improving the matrix.
- Extend the clean 2.5 ms CBR packet checks beyond one second, especially the
  now-clean 128 kb/s fixture.
- Trace 10 ms and 20 ms / 128 kb/s CBR frame-0 divergence after the high-level
  controls. Transient, TF, spread, trim, and coded-band symbols were previously
  aligned, so the remaining split should be in energy, allocation, or PVQ.
- Finish constrained-VBR parity by removing the remaining allocation/PVQ or
  state-history mismatch that prevents using libopus' `0.67` constrained blend
  everywhere.
- Keep exact packet comparisons in the loop, but treat quality/cross-decode
  checks as the higher-value signal when byte drift is caused by harmless float
  thresholds.

## P1: Complete CELT Decode Coverage

- Decode all valid packet frame layouts, not only single-frame CELT packets.
- Support CELT narrowband, mediumband, wideband, superwideband, and fullband
  packet bandwidths.
- Support public Opus API sample rates: 8, 12, 16, 24, and 48 kHz.
- Implement packet-loss concealment and the public `decode_fec` path.
- Track decoder CTL state for final range, pitch, gain, last packet duration,
  bandwidth, phase inversion, and lookahead.
- Validate against upstream `tests/test_opus_decode.c`, `opus_demo`, and CELT
  unit vectors.

## P2: Match the Public Opus Encoder API

- Add typed equivalents for important `opus_encoder_ctl` settings: application,
  bitrate, max/forced bandwidth, VBR, constrained VBR, complexity, FEC,
  packet-loss percentage, DTX, forced channels, signal type, LSB depth, expert
  frame duration, prediction disabled, phase inversion disabled, DRED duration,
  DNN blob, lookahead, final range, and in-DTX query.
- Make `Application::Voip`, `Application::Audio`, and
  `Application::RestrictedLowDelay` drive the same mode, bandwidth, and delay
  decisions as libopus.
- Add reset/query semantics where they affect deterministic behavior.
- Return explicit errors for unsupported SILK, hybrid, FEC, or DRED paths until
  those paths are implemented.

## P3: Port SILK, Hybrid, DRED, and Multistream

- Port SILK common primitives and unit tests.
- Port the SILK decoder, including PLC, CNG, LBRR/FEC, stereo prediction, and
  resampling behavior.
- Port hybrid packet decode, including SILK/CELT transition and redundancy.
- Port the SILK encoder and libopus mode-selection pipeline from
  `src/analysis.c`, `src/mlp.c`, and `src/opus_encoder.c`.
- Port DRED/deep redundancy support from libopus 1.5.2.
- Port multistream and projection behavior if this crate intends to match the
  full libopus public surface.

## P4: Test Harness and Fuzzing

- Port or wrap upstream tests:
  `test_opus_api.c`, `test_opus_decode.c`, `test_opus_encode.c`,
  `test_opus_padding.c`, `test_opus_extensions.c`, `test_opus_dred.c`,
  `test_opus_projection.c`, CELT unit tests, and SILK unit tests.
- Keep a deterministic C/Rust parity harness that reports the first divergent
  frame plus enough symbol/bit-range context to map the mismatch back to code.
- Add fuzz/property tests for packet parsing, repacketizing, padding/unpadding,
  invalid packet handling, and decode error boundaries.

## P5: Performance and Wasm

- Keep performance work correctness-gated. Storage reuse is welcome when it does
  not change entropy decisions, quality decisions, or decoded samples.
- Current safe perf progress includes reusable decoder output paths, decoder
  scratch/cache reuse, in-place decoder postfiltering, encoder DC-reject
  scratch reuse, and encoder transient-analysis scratch reuse.
- Continue removing avoidable `Vec` churn from encode/decode inner loops after
  parity or same-packet quality checks are in place for the touched path.
- Use native raw CELT runs for tight loop measurements and the browser wasm
  comparison only for public API/browser-shape checks.
