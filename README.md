# libopus-rs

Pure Rust port of libopus 1.5.2.

This repository is intentionally not a wrapper around the C library. The target
is a Rust implementation of the Opus 1.5.2 codec, with the upstream C test suite
used as behavioral reference material during the port.

## Current Support

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

This is not a complete Opus codec yet. The usable audio path today is CELT-only
raw frames, not Ogg Opus and not SILK/hybrid speech coding.

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

## Raw CELT benchmark

The raw benchmark compares this crate against libopus through direct in-process
encode/decode calls with no file I/O in the measured loops. The input is a
deterministic in-memory 48 kHz stereo fixture.

```sh
tools/run_raw_celt_bench.sh --repeats 21 --seconds 4
```

Set `OPUS_DIR=path/to/opus-1.5.2` to compare against a built upstream source
tree; otherwise the script uses `pkg-config opus`. The C reference is configured
for restricted-lowdelay/fullband/CBR mode so the comparison stays on CELT-only
frames. Negative speed deltas mean the Rust path was faster than C. Byte counts
are raw Opus packet bytes, not wrapper/container bytes.

| Frame | Bitrate | Rust enc | Enc vs C | Rust dec | Dec vs C | C enc | C dec | Rust bytes | C bytes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2.5 ms | 48 kb/s | 9.42 ms | -61.5% | 7.57 ms | +37.8% | 24.48 ms | 5.49 ms | 24000 | 24000 |
| 2.5 ms | 96 kb/s | 13.67 ms | -56.7% | 10.97 ms | +43.7% | 31.58 ms | 7.63 ms | 48000 | 48000 |
| 2.5 ms | 128 kb/s | 16.63 ms | -54.2% | 12.82 ms | +59.0% | 36.32 ms | 8.06 ms | 64000 | 64000 |
| 5.0 ms | 48 kb/s | 9.10 ms | -57.8% | 7.61 ms | +43.6% | 21.55 ms | 5.30 ms | 24000 | 24000 |
| 5.0 ms | 96 kb/s | 14.14 ms | -49.1% | 10.92 ms | +66.9% | 27.78 ms | 6.54 ms | 48000 | 48000 |
| 5.0 ms | 128 kb/s | 15.89 ms | -49.5% | 12.29 ms | +73.4% | 31.43 ms | 7.09 ms | 64000 | 64000 |
| 10.0 ms | 48 kb/s | 8.39 ms | -55.0% | 6.89 ms | +54.4% | 18.67 ms | 4.46 ms | 24000 | 24000 |
| 10.0 ms | 96 kb/s | 14.62 ms | -40.2% | 10.59 ms | +77.8% | 24.43 ms | 5.96 ms | 48000 | 48000 |
| 10.0 ms | 128 kb/s | 15.09 ms | -41.6% | 11.50 ms | +74.2% | 25.84 ms | 6.60 ms | 64000 | 64000 |
| 20.0 ms | 48 kb/s | 8.06 ms | -54.4% | 6.51 ms | +61.8% | 17.66 ms | 4.03 ms | 24000 | 24000 |
| 20.0 ms | 96 kb/s | 11.96 ms | -50.8% | 9.28 ms | +60.2% | 24.30 ms | 5.79 ms | 48000 | 48000 |
| 20.0 ms | 128 kb/s | 12.81 ms | -49.9% | 10.42 ms | +65.2% | 25.59 ms | 6.31 ms | 64000 | 64000 |

## License

BSD-3-Clause, matching upstream libopus. See [LICENSE](LICENSE).
