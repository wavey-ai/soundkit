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

For local speed runs, benchmark the Rust side with host-native codegen:

```sh
RUST_BENCH_RUSTFLAGS='-C target-cpu=native -C target-feature=+avx2' tools/run_raw_celt_bench.sh --repeats 21 --seconds 4 --mode both
```

Set `OPUS_DIR=path/to/opus-1.5.2` to compare against a built upstream source
tree; otherwise the script uses `pkg-config opus`. The C reference is configured
for restricted-lowdelay/fullband mode with CBR or constrained VBR. Reported
speed columns are normalized as realtime speedup:
`RTFx = (seconds * 1000) / elapsed_ms`, where 1.0x is realtime, and larger is
faster. Negative deltas mean Rust was faster than C. Byte counts are raw Opus
packet bytes, not wrapper/container bytes. Packet ranges show per-frame compressed
packet byte sizes.

Run `tools/run_raw_celt_bench.sh` to generate the current table on your machine.
For one quick check, use:

```sh
AUDIO_SECONDS=1 REPEATS=1 MODE=both tools/run_raw_celt_bench.sh
```

The raw CELT benchmark output is generated dynamically and will vary slightly by
host CPU and optimization flags, so it is intentionally not hardcoded in this
README.

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
