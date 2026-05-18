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
for restricted-lowdelay/fullband mode with CBR or constrained VBR. Reported
speed columns are normalized as real-time factors:
`RTFx = elapsed_ms / (seconds * 1000)`, where 1.0x is realtime. Negative
deltas mean Rust was faster than C. Byte counts are raw Opus packet bytes, not
wrapper/container bytes. Packet ranges show per-frame compressed packet byte
sizes.

| Mode | Frame | Bitrate | Rust enc (RTFx) | Enc vs C | Rust dec (RTFx) | Dec vs C | C enc (RTFx) | C dec (RTFx) | Rust bytes | C bytes | Rust pkt | C pkt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cbr | 2.5 ms | 48 kb/s | 0.0023x | -61.7% | 0.0019x | +37.2% | 0.0061x | 0.0014x | 24000 | 24000 | 15-15 | 15-15 |
| cbr | 2.5 ms | 96 kb/s | 0.0034x | -56.8% | 0.0027x | +43.7% | 0.0079x | 0.0019x | 48000 | 48000 | 30-30 | 30-30 |
| cbr | 2.5 ms | 128 kb/s | 0.0041x | -54.4% | 0.0032x | +57.5% | 0.0090x | 0.0020x | 64000 | 64000 | 40-40 | 40-40 |
| cbr | 5.0 ms | 48 kb/s | 0.0023x | -58.0% | 0.0019x | +43.9% | 0.0054x | 0.0013x | 24000 | 24000 | 30-30 | 30-30 |
| cbr | 5.0 ms | 96 kb/s | 0.0035x | -49.4% | 0.0027x | +67.8% | 0.0069x | 0.0016x | 48000 | 48000 | 60-60 | 60-60 |
| cbr | 5.0 ms | 128 kb/s | 0.0039x | -49.8% | 0.0030x | +72.0% | 0.0078x | 0.0018x | 64000 | 64000 | 80-80 | 80-80 |
| cbr | 10.0 ms | 48 kb/s | 0.0020x | -55.6% | 0.0017x | +55.7% | 0.0046x | 0.0011x | 24000 | 24000 | 60-60 | 60-60 |
| cbr | 10.0 ms | 96 kb/s | 0.0036x | -40.3% | 0.0026x | +77.7% | 0.0060x | 0.0015x | 48000 | 48000 | 120-120 | 120-120 |
| cbr | 10.0 ms | 128 kb/s | 0.0037x | -41.6% | 0.0029x | +75.5% | 0.0064x | 0.0016x | 64000 | 64000 | 160-160 | 160-160 |
| cbr | 20.0 ms | 48 kb/s | 0.0020x | -54.6% | 0.0016x | +63.9% | 0.0044x | 0.0010x | 24000 | 24000 | 120-120 | 120-120 |
| cbr | 20.0 ms | 96 kb/s | 0.0030x | -50.9% | 0.0023x | +61.0% | 0.0060x | 0.0014x | 48000 | 48000 | 240-240 | 240-240 |
| cbr | 20.0 ms | 128 kb/s | 0.0032x | -50.1% | 0.0026x | +65.3% | 0.0063x | 0.0016x | 64000 | 64000 | 320-320 | 320-320 |
| vbr | 2.5 ms | 48 kb/s | 0.0024x | -61.2% | 0.0019x | +36.7% | 0.0063x | 0.0014x | 23995 | 25614 | 14-17 | 13-21 |
| vbr | 2.5 ms | 96 kb/s | 0.0035x | -55.7% | 0.0027x | +52.6% | 0.0080x | 0.0018x | 47989 | 49629 | 27-34 | 26-41 |
| vbr | 2.5 ms | 128 kb/s | 0.0042x | -53.6% | 0.0032x | +54.0% | 0.0091x | 0.0021x | 63985 | 65637 | 36-46 | 35-57 |
| vbr | 5.0 ms | 48 kb/s | 0.0024x | -57.3% | 0.0019x | +43.6% | 0.0055x | 0.0013x | 23988 | 24800 | 28-33 | 27-41 |
| vbr | 5.0 ms | 96 kb/s | 0.0036x | -47.7% | 0.0028x | +72.8% | 0.0069x | 0.0016x | 47976 | 48808 | 55-66 | 56-88 |
| vbr | 5.0 ms | 128 kb/s | 0.0040x | -48.3% | 0.0030x | +71.1% | 0.0078x | 0.0018x | 63968 | 64865 | 74-88 | 75-116 |
| vbr | 10.0 ms | 48 kb/s | 0.0022x | -53.2% | 0.0017x | +54.5% | 0.0046x | 0.0011x | 23977 | 24452 | 57-67 | 57-101 |
| vbr | 10.0 ms | 96 kb/s | 0.0036x | -39.9% | 0.0026x | +73.9% | 0.0060x | 0.0015x | 47956 | 48520 | 113-135 | 119-181 |
| vbr | 10.0 ms | 128 kb/s | 0.0039x | -39.9% | 0.0029x | +74.7% | 0.0064x | 0.0016x | 63940 | 64560 | 151-180 | 155-233 |
| vbr | 20.0 ms | 48 kb/s | 0.0021x | -52.8% | 0.0016x | +61.8% | 0.0044x | 0.0010x | 23954 | 24319 | 115-136 | 118-177 |
| vbr | 20.0 ms | 96 kb/s | 0.0031x | -49.3% | 0.0023x | +61.5% | 0.0060x | 0.0014x | 47909 | 48440 | 231-271 | 241-312 |
| vbr | 20.0 ms | 128 kb/s | 0.0033x | -48.4% | 0.0026x | +66.3% | 0.0063x | 0.0016x | 63878 | 64520 | 307-362 | 321-407 |

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
