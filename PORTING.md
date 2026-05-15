# libopus Rust Port

This directory is a Rust port of Opus 1.5.2. The local C checkout is currently
slightly past the `v1.5.2` tag, so new port slices should use the 1.5.2 codec
behavior and test expectations as the target unless explicitly noted. The crate
is Rust-first and does not expose a C API.

## Current Slice

Implemented in Rust:

- safe packet parser and packet helpers
- safe repacketizer with packet and multistream pad/unpad helpers
- safe soft clipping
- safe CELT entropy/range coder
- safe CELT mathops, laplace, CWRS/PVQ, DFT, MDCT, rotation, and algebraic VQ
- encoder and decoder Rust types that currently return `Error::Unimplemented`
  for actual audio encode/decode

Real CELT/SILK encode/decode is not implemented yet.

## Upstream Tests Used

The Rust tests in `tests/packet_api.rs` are derived from:

- `tests/test_opus_api.c`, especially packet parser and repacketizer sections
- `tests/test_opus_padding.c`, especially the padding overflow regression
- `tests/test_opus_decode.c`, for soft clipping expectations
- `celt/tests/test_unit_entropy.c`, for CELT entropy/range coder behavior
- `celt/tests/test_unit_mathops.c`
- `celt/tests/test_unit_laplace.c`
- `celt/tests/test_unit_cwrs32.c`
- `celt/tests/test_unit_dft.c`
- `celt/tests/test_unit_mdct.c`
- `celt/tests/test_unit_rotation.c`

Run:

```sh
cargo test
cargo build --release
```

## Remaining Port Order

1. Keep default crate builds under `#![forbid(unsafe_code)]`.
2. Port remaining CELT bands, rate allocation, mode tables, pitch helpers, and
   quantized energy logic.
3. Port CELT decoder and validate with `tests/test_opus_decode.c`.
4. Port SILK fixed/float common signal-processing primitives and unit tests.
5. Port SILK decoder, then hybrid packet decode.
6. Port CELT and SILK encoders, then `tests/test_opus_encode.c`.
7. Port multistream and projection encoders/decoders.
8. Port DRED/deep PLC/OSCE extensions.
9. Replace `Error::Unimplemented` encode/decode returns with the real signal
   path and port the upstream C test cases into Rust.
