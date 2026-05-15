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
  `Encoder`/`Decoder` types for 2.5, 5, 10, and 20 ms fullband frames
  with bitrate or exact compressed-frame-byte controls

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
cargo run --release --example wav_celt -- roundtrip --frame-size 240 --bitrate 128000 input.wav output.lors decoded.wav
cargo run --release --example wav_celt -- roundtrip --frame-size 960 --frame-bytes 120 input.wav output.lors decoded.wav
```

`output.lors` is a simple length-prefixed raw packet stream for testing this
port. It is not Ogg Opus yet.

## CELT benchmark

Upstream Opus has standard conformance vectors, but they are distributed as a
separate `opus_testvectors-rfc8251` download rather than checked into the source
tree. The benchmark helper here can use a local WAV file or generate a
deterministic synthetic 48 kHz stereo fixture.

This snapshot uses `/Users/jamie/Downloads/westside-20260515g.det.wav`:

```sh
python3 tools/bench_celt.py \
  --input-wav /Users/jamie/Downloads/westside-20260515g.det.wav \
  --repeats 3 \
  --golden-dir /Users/jamie/Downloads/westside-celt-goldens \
  --testdata-dir /tmp/libopus-rs-westside-testdata
```

The C reference is upstream `opus_demo` in restricted-lowdelay/fullband/CBR
mode so the comparison stays on CELT-only frames. Timings include process
startup and file I/O. The generated Westside goldens are kept outside the repo.
The temp testdata directory contains the Rust/C encoded streams and decoded WAVs
for each row. Byte counts are the local harness stream sizes, so they include
different framing overheads for `.lors` and `opus_demo` `.bit` files. The
remaining notable quality gaps are 2.5 ms at 96 kb/s and the 5-10 ms rows at
48 kb/s.

| Frame | Bitrate | Rust enc | Enc vs C | Rust dec | Dec vs C | C enc | C dec | Rust bytes | C bytes | Rust SNR | C SNR |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2.5 ms | 48 kb/s | 16.84 ms | -61.3% | 11.84 ms | -49.7% | 43.50 ms | 23.56 ms | 27224 | 36823 | 8.8 dB | 8.9 dB |
| 2.5 ms | 96 kb/s | 18.64 ms | -63.6% | 15.86 ms | -37.2% | 51.22 ms | 25.27 ms | 51224 | 60838 | 16.8 dB | 19.3 dB |
| 2.5 ms | 128 kb/s | 21.51 ms | -62.8% | 18.06 ms | -31.7% | 57.78 ms | 26.45 ms | 67224 | 76848 | 23.4 dB | 22.5 dB |
| 5 ms | 48 kb/s | 13.37 ms | -67.3% | 12.18 ms | -46.1% | 40.86 ms | 22.61 ms | 25624 | 30438 | 14.4 dB | 16.0 dB |
| 5 ms | 96 kb/s | 18.17 ms | -63.2% | 14.59 ms | -41.7% | 49.44 ms | 25.01 ms | 49624 | 54468 | 22.6 dB | 22.3 dB |
| 5 ms | 128 kb/s | 18.41 ms | -64.9% | 15.84 ms | -37.2% | 52.49 ms | 25.22 ms | 65624 | 70488 | 25.0 dB | 24.8 dB |
| 10 ms | 48 kb/s | 12.95 ms | -66.3% | 11.64 ms | -48.5% | 38.38 ms | 22.62 ms | 24824 | 27268 | 15.1 dB | 16.7 dB |
| 10 ms | 96 kb/s | 16.31 ms | -63.7% | 13.86 ms | -42.9% | 44.89 ms | 24.29 ms | 48824 | 51328 | 21.8 dB | 21.2 dB |
| 10 ms | 128 kb/s | 17.36 ms | -63.2% | 15.49 ms | -37.7% | 47.22 ms | 24.86 ms | 64824 | 67368 | 24.3 dB | 23.0 dB |
| 20 ms | 48 kb/s | 12.82 ms | -65.8% | 11.75 ms | -47.5% | 37.52 ms | 22.40 ms | 24424 | 25728 | 17.0 dB | 17.2 dB |
| 20 ms | 96 kb/s | 15.58 ms | -63.7% | 13.57 ms | -42.5% | 42.94 ms | 23.59 ms | 48424 | 49848 | 22.4 dB | 21.7 dB |
| 20 ms | 128 kb/s | 16.80 ms | -62.1% | 14.94 ms | -39.4% | 44.33 ms | 24.67 ms | 64424 | 65928 | 25.2 dB | 23.5 dB |

## License

BSD-3-Clause, matching upstream libopus. See [LICENSE](LICENSE).
