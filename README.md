# libopus-rs

Pure Rust port of libopus 1.5.2.

This repository is intentionally not a wrapper around the C library. The target
is a Rust implementation of the Opus 1.5.2 codec, with the upstream C test suite
used as behavioral reference material during the port.

## Status

This is an active port, not a complete Opus codec yet.

Implemented:

- safe packet parser and packet helper APIs
- safe repacketizer and packet padding/unpadding APIs
- soft clipping
- CELT entropy/range coder
- CELT mathops, laplace, CWRS/PVQ, DFT, MDCT, mode construction, rate
  allocation, quantized energy, band helpers, rotation, and algebraic VQ
- encoder/decoder Rust types that currently report real encode/decode as
  unimplemented

Still to port:

- remaining CELT pitch helpers, band quantization loop, and codec wiring
- SILK signal path
- real Opus encode/decode
- multistream/projection codec internals
- DRED/deep PLC/OSCE extensions

See [PORTING.md](PORTING.md) for the module-by-module plan and test status.
See [SAFETY.md](SAFETY.md) for the unsafe-code policy.

## Build

```sh
cargo test
cargo build --release
```

The crate is built with `#![forbid(unsafe_code)]`. It does not expose a C API.

## License

BSD-3-Clause, matching upstream libopus. See [LICENSE](LICENSE).
