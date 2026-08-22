# libopus Rust Port

This directory contains a Rust-first Opus implementation. It does not expose a
C API, and it does not target one libopus release.

Use stable libopus and upstream main as behavioral evidence. Record the source
commit for each imported behavior. Quality and interoperability take priority
over encoder byte identity.

## Current Slice

Implemented in safe Rust:

- packet parsing, packet helpers, repacketizing, and padding
- soft clipping and the CELT entropy coder
- CELT math, Laplace, CWRS, PVQ, DFT, MDCT, and mode construction
- CELT allocation, energy coding, band coding, and synthesis
- pitch prefilter signaling, filtering, and decoder postfiltering
- LPC tone detection and tone-aware allocation
- transient-aware pitch-filter cancellation
- stereo band handling when one channel is silent
- mono and stereo 48 kHz CELT-only raw packet encoding and decoding
- 2.5, 5, 10, and 20 ms fullband frames
- CBR, constrained VBR, and exact compressed-frame-byte controls
- signed 24-bit PCM input and output through sign-extended `i32` samples

Full Opus encoding and decoding are not complete. The working audio path is the
CELT-only fullband subset.

## Upstream Evidence

Rust tests draw behavior from these upstream areas:

- packet API, padding, decode, and extension tests
- CELT entropy, math, Laplace, CWRS, DFT, MDCT, and rotation tests
- CELT mode, rate, energy, band, pitch, and encoder sources
- `opus_demo` and direct C/Rust packet cross-decode runs

The current 48 kHz slice imports these upstream changes:

- tone detection: `3b68a486`, `738c29fc`, `d7eceaea`, and `9394c7ce`
- tone allocation and control: `86101c0e`, `6f4f3e89`, `e082ddc1`,
  `a80d72f5`, and `5b8c9fae`
- ineffective pitch-filter cancellation: `462c50d3`
- one-silent-channel stereo handling: `2329ed17`

The repository also contains deterministic packet probes and raw CELT
benchmarks. Use them to locate differences. Do not use synthetic SNR alone to
approve an encoder change.

Run the main checks:

```sh
cargo test --all-features
cargo build --release --all-features
```

## Remaining Port Order

1. Keep all default crate builds under `#![forbid(unsafe_code)]`.
2. Establish a licensed real-audio corpus and perceptual quality gate.
3. Finish 48 kHz CELT encoder tracing, quality work, and longer regressions.
4. Complete valid CELT packet layouts, bandwidths, PLC, FEC, and API rates.
5. Complete CELT controls, Ogg framing, fuzzing, and performance work.
6. Add QEXT 96 kHz after the 48 kHz quality gate and 24-bit I/O.
7. Add multistream, channel mappings, multichannel tests, and then projection.
8. Port SILK, hybrid mode, DRED, and later optional extensions separately.

See [TODO.md](TODO.md) for detailed checkpoints and acceptance criteria.
