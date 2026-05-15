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
  band quantization, dynamic allocation analysis, theta RDO, energy-error
  feedback, pitch prefilter signaling/filtering, decoder postfiltering, spread
  decision state, band helpers, synthesis/deemphasis, rotation, and algebraic
  VQ
- experimental 48 kHz CELT-only raw packet encode/decode through the Rust
  `Encoder`/`Decoder` types for 2.5, 5, 10, and 20 ms fullband frames
  with CBR, constrained VBR, or exact compressed-frame-byte controls

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
cargo run --release --example wav_celt -- roundtrip --frame-size 240 --bitrate 128000 --vbr input.wav output.lors decoded.wav
cargo run --release --example wav_celt -- roundtrip --frame-size 960 --frame-bytes 120 input.wav output.lors decoded.wav
```

`output.lors` is a simple length-prefixed raw packet stream for testing this
port. It is not Ogg Opus yet.

To export side-by-side decoded WAVs for listening comparisons:

```sh
tools/export_roundtrip_wavs.sh --input input-audio --out-dir path/to/roundtrips --mode both
```

The helper normalizes the input to 48 kHz stereo PCM16 before running both
implementations. Each case directory contains the Rust packet stream and
decoded WAV plus the upstream `opus_demo` packet stream and decoded WAV.

## Raw CELT benchmark

The raw benchmark compares this crate against libopus through direct in-process
encode/decode calls with no file I/O in the measured loops. The input is a
deterministic in-memory 48 kHz stereo fixture.

```sh
tools/run_raw_celt_bench.sh --repeats 21 --seconds 4 --mode both
```

Set `OPUS_DIR=path/to/opus-1.5.2` to compare against a built upstream source
tree; otherwise the script uses `pkg-config opus`. The C reference is configured
for restricted-lowdelay/fullband mode with CBR or constrained VBR. Negative
speed deltas mean the Rust path was faster than C. Byte counts are raw Opus
packet bytes, not wrapper/container bytes. Packet ranges show per-frame
compressed packet byte sizes.

| Mode | Frame | Bitrate | Rust enc | Enc vs C | Rust dec | Dec vs C | C enc | C dec | Rust bytes | C bytes | Rust pkt | C pkt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cbr | 2.5 ms | 48 kb/s | 9.30 ms | -61.7% | 7.51 ms | +37.2% | 24.31 ms | 5.47 ms | 24000 | 24000 | 15-15 | 15-15 |
| cbr | 2.5 ms | 96 kb/s | 13.57 ms | -56.8% | 10.84 ms | +43.7% | 31.41 ms | 7.54 ms | 48000 | 48000 | 30-30 | 30-30 |
| cbr | 2.5 ms | 128 kb/s | 16.43 ms | -54.4% | 12.68 ms | +57.5% | 35.99 ms | 8.05 ms | 64000 | 64000 | 40-40 | 40-40 |
| cbr | 5.0 ms | 48 kb/s | 9.02 ms | -58.0% | 7.54 ms | +43.9% | 21.47 ms | 5.24 ms | 24000 | 24000 | 30-30 | 30-30 |
| cbr | 5.0 ms | 96 kb/s | 13.92 ms | -49.4% | 10.74 ms | +67.8% | 27.51 ms | 6.40 ms | 48000 | 48000 | 60-60 | 60-60 |
| cbr | 5.0 ms | 128 kb/s | 15.69 ms | -49.8% | 12.07 ms | +72.0% | 31.24 ms | 7.02 ms | 64000 | 64000 | 80-80 | 80-80 |
| cbr | 10.0 ms | 48 kb/s | 8.20 ms | -55.6% | 6.82 ms | +55.7% | 18.46 ms | 4.38 ms | 24000 | 24000 | 60-60 | 60-60 |
| cbr | 10.0 ms | 96 kb/s | 14.41 ms | -40.3% | 10.50 ms | +77.7% | 24.15 ms | 5.91 ms | 48000 | 48000 | 120-120 | 120-120 |
| cbr | 10.0 ms | 128 kb/s | 14.99 ms | -41.6% | 11.43 ms | +75.5% | 25.69 ms | 6.51 ms | 64000 | 64000 | 160-160 | 160-160 |
| cbr | 20.0 ms | 48 kb/s | 7.96 ms | -54.6% | 6.46 ms | +63.9% | 17.55 ms | 3.94 ms | 24000 | 24000 | 120-120 | 120-120 |
| cbr | 20.0 ms | 96 kb/s | 11.80 ms | -50.9% | 9.17 ms | +61.0% | 24.04 ms | 5.70 ms | 48000 | 48000 | 240-240 | 240-240 |
| cbr | 20.0 ms | 128 kb/s | 12.63 ms | -50.1% | 10.31 ms | +65.3% | 25.30 ms | 6.24 ms | 64000 | 64000 | 320-320 | 320-320 |
| vbr | 2.5 ms | 48 kb/s | 9.73 ms | -61.2% | 7.58 ms | +36.7% | 25.07 ms | 5.54 ms | 23995 | 25614 | 14-17 | 13-21 |
| vbr | 2.5 ms | 96 kb/s | 14.08 ms | -55.7% | 10.90 ms | +52.6% | 31.79 ms | 7.14 ms | 47989 | 49629 | 27-34 | 26-41 |
| vbr | 2.5 ms | 128 kb/s | 16.89 ms | -53.6% | 12.64 ms | +54.0% | 36.39 ms | 8.21 ms | 63985 | 65637 | 36-46 | 35-57 |
| vbr | 5.0 ms | 48 kb/s | 9.46 ms | -57.3% | 7.57 ms | +43.6% | 22.13 ms | 5.27 ms | 23988 | 24800 | 28-33 | 27-41 |
| vbr | 5.0 ms | 96 kb/s | 14.48 ms | -47.7% | 11.08 ms | +72.8% | 27.66 ms | 6.41 ms | 47976 | 48808 | 55-66 | 56-88 |
| vbr | 5.0 ms | 128 kb/s | 16.17 ms | -48.3% | 12.09 ms | +71.1% | 31.27 ms | 7.07 ms | 63968 | 64865 | 74-88 | 75-116 |
| vbr | 10.0 ms | 48 kb/s | 8.68 ms | -53.2% | 6.79 ms | +54.5% | 18.56 ms | 4.40 ms | 23977 | 24452 | 57-67 | 57-101 |
| vbr | 10.0 ms | 96 kb/s | 14.54 ms | -39.9% | 10.29 ms | +73.9% | 24.19 ms | 5.92 ms | 47956 | 48520 | 113-135 | 119-181 |
| vbr | 10.0 ms | 128 kb/s | 15.42 ms | -39.9% | 11.44 ms | +74.7% | 25.67 ms | 6.55 ms | 63940 | 64560 | 151-180 | 155-233 |
| vbr | 20.0 ms | 48 kb/s | 8.34 ms | -52.8% | 6.46 ms | +61.8% | 17.67 ms | 3.99 ms | 23954 | 24319 | 115-136 | 118-177 |
| vbr | 20.0 ms | 96 kb/s | 12.21 ms | -49.3% | 9.27 ms | +61.5% | 24.07 ms | 5.74 ms | 47909 | 48440 | 231-271 | 241-312 |
| vbr | 20.0 ms | 128 kb/s | 13.08 ms | -48.4% | 10.42 ms | +66.3% | 25.36 ms | 6.26 ms | 63878 | 64520 | 307-362 | 321-407 |

## Encoder Parity Next Steps

CBR byte parity remains the active target before VBR parity. On the
deterministic raw CELT fixture, the first six 2.5 ms CBR packets at 48, 96, and
128 kb/s are byte-identical with libopus. Across a 40-packet run, the first
divergence is frame 8 at 48 and 96 kb/s, and frame 7 at 128 kb/s.

The 2.5 ms / 128 kb/s frame-7 mismatch is narrowed to allocation trim:
prefilter signaling, coarse energy, TF/spread decisions, dynalloc signaling,
and total boost match libopus before Rust writes trim 5 where C writes trim 4.

The 5, 10, and 20 ms CBR paths still diverge from frame 0. The first traced
5 ms / 128 kb/s mismatch happens after matching coarse energy: Rust currently
encodes all TF flags as 1 with 192 dynalloc boost, while libopus encodes all TF
flags as 0 with 288 boost.

Ported in this checkpoint:

- energy-error feedback
- dynalloc analysis
- theta RDO for stereo CELT bands
- CELT pitch prefilter signaling and input filtering
- CELT decoder postfilter state and filtering
- spread decision state

Resume from this checkpoint:

1. Fix `alloc_trim_analysis` or its encoder state inputs for the 2.5 ms
   frame-7 trim 5 vs 4 mismatch.
2. Extend 2.5 ms CBR byte parity past the 40-packet fixture at 48, 96, and
   128 kb/s.
3. Port the remaining official TF analysis and transient-path details for
   `LM > 0`, then repeat 5, 10, and 20 ms CBR packet dumps.
4. After CBR is bit-identical for the raw CELT matrix, port libopus'
   constrained VBR target/reservoir logic and repeat VBR packet dumps.

## License

BSD-3-Clause, matching upstream libopus. See [LICENSE](LICENSE).
