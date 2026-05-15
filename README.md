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
  allocation, frame control symbols, spectral frame coding, quantized energy,
  band quantization, band helpers, synthesis/deemphasis, rotation, and
  algebraic VQ
- experimental 48 kHz CELT-only raw packet encode/decode through the Rust
  `Encoder`/`Decoder` types

Still to port:

- remaining CELT pitch helpers and full public codec wiring
- SILK signal path
- full Opus encode/decode
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

## WAV smoke test

The `wav_celt` example can round-trip 48 kHz mono/stereo PCM16 WAV through the
current pure-Rust CELT-only packet path:

```sh
cargo run --release --example wav_celt -- roundtrip input.wav output.lors decoded.wav
```

`output.lors` is a simple length-prefixed raw packet stream for testing this
port. It is not Ogg Opus yet.

## License

BSD-3-Clause, matching upstream libopus. See [LICENSE](LICENSE).
