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
faster. Positive deltas mean Rust was faster than C. Byte counts are raw Opus
packet bytes, not wrapper/container bytes. Packet ranges show per-frame compressed
packet byte sizes.

Run `tools/run_raw_celt_bench.sh` to generate the current table on your machine.
For one quick check, use:

```sh
AUDIO_SECONDS=1 REPEATS=1 MODE=both tools/run_raw_celt_bench.sh
```

## Wasm CELT benchmark

The wasm benchmark builds scalar and `simd128` versions of a small exported CELT
encode kernel, then times both from Node.js:

```sh
tools/run_wasm_celt_bench.sh
```

Use `AUDIO_SECONDS`, `REPEATS`, `BITRATE`, or `SIMD_RUSTFLAGS` to adjust the
run. `tools/build_wasm_simd.sh --example wasm_celt_bench` builds only the
`simd128` artifact when you just need the wasm output.

Current local measurements do not justify enabling wasm SIMD by default. The
`simd128` build produced matching checksums, but it was generally slower than
the scalar wasm build in Node on Apple Silicon, with the 5 ms frame case showing
the clearest regression. Treat the scalar wasm build as the baseline for now and
use this benchmark only to validate future targeted SIMD work.

## Full-track wasm comparison

A full-track browser-shape comparison was run on:

```text
/Users/jamie/Downloads/Lori Asha - Lori Asha Album Premix/02 - Lori Asha - Westside.mp3
```

The source was decoded and resampled once with `ffmpeg` to identical `48 kHz`
stereo `s16` PCM, then encoded and decoded in Node.js with 20 ms frames through
`libopusjs` C wasm and pure Rust `libopus-rs` wasm. Quality metrics below are
delay-aligned against the source PCM; unaligned RMSE/SNR numbers are misleading
because the two paths have different codec delay. The current `libopusjs`
wrapper does not expose a CBR/VBR toggle and uses libopus' default VBR behavior.
Rust wasm defaults to CBR for Bitneedle, and also exposes constrained VBR via
`encoder.set_vbr(true)`.

| Target | Codec/mode | Encoded bytes | Effective kb/s | Packet bytes | Encode xRTF | Decode xRTF | Delay | Aligned SNR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 48 kb/s | C `libopusjs` default VBR | 1,260,065 | 48.35 | 3-206 (avg 120.86) | 147.4x | 332.5x | 6.5 ms | 15.33 dB |
| 48 kb/s | Rust `libopus-rs` CBR | 1,251,120 | 48.00 | 120-120 | 127.2x | 377.3x | 2.5 ms | 14.34 dB |
| 48 kb/s | Rust `libopus-rs` VBR | 1,260,198 | 48.35 | 109-157 (avg 120.87) | 128.0x | 380.1x | 2.5 ms | 14.41 dB |
| 128 kb/s | C `libopusjs` default VBR | 3,346,824 | 128.41 | 3-543 (avg 321.01) | 110.9x | 279.7x | 6.5 ms | 22.14 dB |
| 128 kb/s | Rust `libopus-rs` CBR | 3,336,320 | 128.01 | 320-320 | 81.4x | 263.6x | 2.5 ms | 19.01 dB |
| 128 kb/s | Rust `libopus-rs` VBR | 3,343,132 | 128.27 | 290-416 (avg 320.65) | 82.4x | 268.2x | 2.5 ms | 19.11 dB |

All wasm paths encoded and decoded the full track without failures. Rust VBR
tracks the target byte budget and varies packet sizes, but on this workload it
does not close the quality gap to C. Rust decode is competitive and faster at
48 kb/s; Rust encode is still slower, especially at 128 kb/s, and needs targeted
profiling before claiming parity with the C encoder.

A snapshot of a local run (`AUDIO_SECONDS=1 REPEATS=1 MODE=both`) is included below:

| Mode | Frame | Bitrate | Rust enc (xRTF) | Enc vs C | Rust dec (xRTF) | Dec vs C | C enc (xRTF) | C dec (xRTF) | Rust bytes | C bytes | Rust pkt | C pkt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cbr | 2.5 ms | 48 kb/s | 114.30x | +4.6% | 371.13x | +59.6% | 119.50x | 592.42x | 6000 | 6000 | 15-15 | 15-15 |
| cbr | 2.5 ms | 96 kb/s | 82.26x | +25.2% | 292.41x | +54.1% | 103.03x | 450.65x | 12000 | 12000 | 30-30 | 30-30 |
| cbr | 2.5 ms | 128 kb/s | 70.95x | +37.8% | 246.55x | +89.2% | 97.77x | 466.42x | 16000 | 16000 | 40-40 | 40-40 |
| cbr | 2.5 ms | 160 kb/s | 63.09x | +38.9% | 221.83x | +68.8% | 87.61x | 374.53x | 20000 | 20000 | 50-50 | 50-50 |
| cbr | 2.5 ms | 192 kb/s | 58.12x | +42.1% | 203.25x | +70.9% | 82.61x | 347.34x | 24000 | 24000 | 60-60 | 60-60 |
| cbr | 2.5 ms | 256 kb/s | 55.66x | +34.5% | 192.21x | +66.8% | 74.84x | 320.51x | 32000 | 32000 | 80-80 | 80-80 |
| cbr | 2.5 ms | 320 kb/s | 56.16x | +27.9% | 181.28x | +71.3% | 71.81x | 310.56x | 40000 | 40000 | 100-100 | 100-100 |
| cbr | 2.5 ms | 384 kb/s | 54.77x | +45.9% | 168.75x | +49.1% | 79.91x | 251.64x | 48000 | 48000 | 120-120 | 120-120 |
| cbr | 2.5 ms | 512 kb/s | 46.08x | +19.5% | 120.80x | +130.2% | 55.05x | 278.09x | 64000 | 64000 | 160-160 | 160-160 |
| cbr | 5.0 ms | 48 kb/s | 156.40x | +19.9% | 408.18x | +83.8% | 187.55x | 750.19x | 6000 | 6000 | 30-30 | 30-30 |
| cbr | 5.0 ms | 96 kb/s | 136.61x | +16.6% | 331.14x | +95.0% | 159.29x | 645.58x | 12000 | 12000 | 60-60 | 60-60 |
| cbr | 5.0 ms | 128 kb/s | 123.23x | +11.9% | 298.86x | +101.8% | 137.89x | 603.14x | 16000 | 16000 | 80-80 | 80-80 |
| cbr | 5.0 ms | 160 kb/s | 119.60x | +24.2% | 275.07x | +115.5% | 148.54x | 592.77x | 20000 | 20000 | 100-100 | 100-100 |
| cbr | 5.0 ms | 192 kb/s | 111.41x | +29.8% | 246.14x | +112.4% | 144.63x | 522.74x | 24000 | 24000 | 120-120 | 120-120 |
| cbr | 5.0 ms | 256 kb/s | 102.88x | +27.6% | 210.03x | +106.5% | 131.25x | 433.65x | 32000 | 32000 | 160-160 | 160-160 |
| cbr | 5.0 ms | 320 kb/s | 97.82x | +26.1% | 182.60x | +111.5% | 123.35x | 386.10x | 40000 | 40000 | 200-200 | 200-200 |
| cbr | 5.0 ms | 384 kb/s | 90.33x | +39.5% | 153.41x | +124.5% | 126.04x | 344.47x | 48000 | 48000 | 240-240 | 240-240 |
| cbr | 5.0 ms | 512 kb/s | 71.89x | +63.8% | 114.77x | +161.6% | 117.72x | 300.21x | 64000 | 64000 | 320-320 | 320-320 |
| cbr | 10.0 ms | 48 kb/s | 193.91x | +12.4% | 541.36x | +71.4% | 217.91x | 927.64x | 6000 | 6000 | 60-60 | 60-60 |
| cbr | 10.0 ms | 96 kb/s | 171.13x | +16.3% | 398.41x | +106.6% | 198.97x | 823.05x | 12000 | 12000 | 120-120 | 120-120 |
| cbr | 10.0 ms | 128 kb/s | 160.70x | +12.8% | 357.35x | +87.2% | 181.26x | 668.90x | 16000 | 16000 | 160-160 | 160-160 |
| cbr | 10.0 ms | 160 kb/s | 145.13x | +21.9% | 298.92x | +61.1% | 176.87x | 481.70x | 20000 | 20000 | 200-200 | 200-200 |
| cbr | 10.0 ms | 192 kb/s | 133.56x | +23.2% | 271.44x | +89.5% | 164.55x | 514.40x | 24000 | 24000 | 240-240 | 240-240 |
| cbr | 10.0 ms | 256 kb/s | 127.05x | +18.9% | 214.38x | +101.5% | 151.08x | 431.97x | 32000 | 32000 | 320-320 | 320-320 |
| cbr | 10.0 ms | 320 kb/s | 115.97x | +29.3% | 188.81x | +101.6% | 149.99x | 380.66x | 40000 | 40000 | 400-400 | 400-400 |
| cbr | 10.0 ms | 384 kb/s | 105.79x | +36.8% | 154.67x | +126.2% | 144.70x | 349.90x | 48000 | 48000 | 480-480 | 480-480 |
| cbr | 10.0 ms | 512 kb/s | 82.55x | +65.5% | 115.60x | +143.7% | 136.63x | 281.77x | 64000 | 64000 | 640-640 | 640-640 |
| cbr | 20.0 ms | 48 kb/s | 231.28x | -1.0% | 606.13x | +65.3% | 229.04x | 1002.00x | 6000 | 6000 | 120-120 | 120-120 |
| cbr | 20.0 ms | 96 kb/s | 182.82x | -6.0% | 414.65x | +79.6% | 171.94x | 744.60x | 12000 | 12000 | 240-240 | 240-240 |
| cbr | 20.0 ms | 128 kb/s | 176.30x | +12.5% | 367.31x | +73.3% | 198.41x | 636.54x | 16000 | 16000 | 320-320 | 320-320 |
| cbr | 20.0 ms | 160 kb/s | 157.92x | +13.7% | 294.83x | +90.0% | 179.53x | 560.22x | 20000 | 20000 | 400-400 | 400-400 |
| cbr | 20.0 ms | 192 kb/s | 142.08x | +21.7% | 270.93x | +81.5% | 172.92x | 491.64x | 24000 | 24000 | 480-480 | 480-480 |
| cbr | 20.0 ms | 256 kb/s | 134.13x | +16.5% | 222.86x | +86.5% | 156.27x | 415.63x | 32000 | 32000 | 640-640 | 640-640 |
| cbr | 20.0 ms | 320 kb/s | 123.68x | +23.0% | 189.76x | +91.4% | 152.14x | 363.24x | 40000 | 40000 | 800-800 | 800-800 |
| cbr | 20.0 ms | 384 kb/s | 113.10x | +28.0% | 161.38x | +118.3% | 144.80x | 352.36x | 48000 | 48000 | 960-960 | 960-960 |
| cbr | 20.0 ms | 512 kb/s | 86.86x | +51.8% | 110.00x | +157.5% | 131.86x | 283.21x | 63800 | 63750 | 1276-1276 | 1275-1275 |
| vbr | 2.5 ms | 48 kb/s | 122.81x | +18.0% | 416.86x | +62.0% | 144.93x | 675.22x | 6402 | 6402 | 15-17 | 13-20 |
| vbr | 2.5 ms | 96 kb/s | 86.17x | +34.6% | 296.32x | +73.2% | 115.96x | 513.08x | 12404 | 12427 | 30-32 | 25-42 |
| vbr | 2.5 ms | 128 kb/s | 67.67x | +27.1% | 242.98x | +47.2% | 86.01x | 357.65x | 16404 | 16439 | 40-43 | 34-54 |
| vbr | 2.5 ms | 160 kb/s | 62.27x | +22.8% | 225.02x | +17.3% | 76.50x | 263.85x | 20406 | 20448 | 49-53 | 43-68 |
| vbr | 2.5 ms | 192 kb/s | 57.06x | +32.5% | 203.49x | +58.7% | 75.60x | 323.00x | 24408 | 24458 | 59-63 | 52-82 |
| vbr | 2.5 ms | 256 kb/s | 54.90x | +33.7% | 189.74x | +71.7% | 73.43x | 325.84x | 32411 | 32477 | 78-84 | 72-111 |
| vbr | 2.5 ms | 320 kb/s | 55.16x | +27.1% | 179.00x | +76.0% | 70.13x | 315.06x | 40413 | 40497 | 98-105 | 92-139 |
| vbr | 2.5 ms | 384 kb/s | 54.46x | +42.7% | 163.23x | +90.0% | 77.69x | 310.17x | 48416 | 48517 | 117-125 | 112-160 |
| vbr | 2.5 ms | 512 kb/s | 45.28x | +73.9% | 128.66x | +122.4% | 78.74x | 286.20x | 64421 | 64000 | 155-167 | 160-160 |
| vbr | 5.0 ms | 48 kb/s | 156.98x | +11.7% | 409.37x | +83.1% | 175.38x | 749.63x | 6204 | 6202 | 31-32 | 29-41 |
| vbr | 5.0 ms | 96 kb/s | 132.64x | +21.5% | 336.97x | +78.0% | 161.16x | 599.88x | 12207 | 12260 | 60-62 | 53-91 |
| vbr | 5.0 ms | 128 kb/s | 123.20x | +20.7% | 301.76x | +94.9% | 148.65x | 588.24x | 16209 | 16280 | 80-83 | 76-119 |
| vbr | 5.0 ms | 160 kb/s | 117.71x | +24.2% | 269.32x | +110.0% | 146.16x | 565.61x | 20211 | 20300 | 100-103 | 96-148 |
| vbr | 5.0 ms | 192 kb/s | 109.02x | +23.0% | 253.63x | +101.5% | 134.10x | 510.99x | 24213 | 24320 | 120-124 | 115-176 |
| vbr | 5.0 ms | 256 kb/s | 104.88x | +22.9% | 205.81x | +110.4% | 128.95x | 433.09x | 32218 | 32360 | 160-165 | 154-231 |
| vbr | 5.0 ms | 320 kb/s | 96.63x | +10.5% | 172.47x | +119.8% | 106.79x | 379.08x | 40221 | 40400 | 199-206 | 192-287 |
| vbr | 5.0 ms | 384 kb/s | 90.68x | +28.8% | 160.26x | +98.9% | 116.80x | 318.78x | 48226 | 48440 | 239-246 | 230-319 |
| vbr | 5.0 ms | 512 kb/s | 71.80x | +53.7% | 115.35x | +155.4% | 110.39x | 294.55x | 64234 | 63800 | 318-328 | 319-319 |
| vbr | 10.0 ms | 48 kb/s | 198.75x | -4.9% | 528.76x | +19.8% | 188.93x | 633.31x | 6107 | 6160 | 60-62 | 58-100 |
| vbr | 10.0 ms | 96 kb/s | 167.94x | +14.4% | 409.89x | +55.2% | 192.05x | 636.13x | 12113 | 12220 | 120-124 | 121-182 |
| vbr | 10.0 ms | 128 kb/s | 159.11x | +11.2% | 341.58x | -77.4% | 176.99x | 77.22x | 16117 | 16260 | 160-164 | 157-236 |
| vbr | 10.0 ms | 160 kb/s | 141.96x | +18.8% | 298.65x | +71.2% | 168.58x | 511.25x | 20121 | 20300 | 200-205 | 194-291 |
| vbr | 10.0 ms | 192 kb/s | 137.52x | -53.9% | 266.83x | +88.3% | 63.38x | 502.51x | 24126 | 24340 | 240-246 | 232-345 |
| vbr | 10.0 ms | 256 kb/s | 124.63x | -24.8% | 220.98x | -4.2% | 93.69x | 211.69x | 32135 | 32420 | 319-328 | 308-453 |
| vbr | 10.0 ms | 320 kb/s | 114.13x | +7.6% | 185.92x | +93.0% | 122.80x | 358.81x | 40143 | 40500 | 398-410 | 385-560 |
| vbr | 10.0 ms | 384 kb/s | 105.52x | +20.6% | 159.48x | -4.5% | 127.21x | 152.37x | 48151 | 48580 | 478-491 | 461-638 |
| vbr | 10.0 ms | 512 kb/s | 81.65x | +60.5% | 116.10x | +155.7% | 131.06x | 296.82x | 64168 | 63800 | 637-655 | 638-638 |
| vbr | 20.0 ms | 48 kb/s | 223.19x | +0.5% | 611.62x | -6.7% | 224.37x | 570.78x | 6063 | 6170 | 121-124 | 119-190 |
| vbr | 20.0 ms | 96 kb/s | 188.78x | -2.8% | 414.54x | +29.3% | 183.49x | 535.91x | 12075 | 12290 | 240-246 | 241-375 |
| vbr | 20.0 ms | 128 kb/s | 169.51x | -14.0% | 355.20x | +75.0% | 145.79x | 621.50x | 16083 | 16370 | 320-328 | 321-511 |
| vbr | 20.0 ms | 160 kb/s | 153.83x | -1.4% | 299.45x | +75.9% | 151.75x | 526.87x | 20092 | 20450 | 400-410 | 401-631 |
| vbr | 20.0 ms | 192 kb/s | 144.66x | +16.5% | 262.87x | +86.8% | 168.52x | 491.16x | 24100 | 24530 | 480-491 | 481-707 |
| vbr | 20.0 ms | 256 kb/s | 132.47x | +11.1% | 221.74x | +87.0% | 147.23x | 414.59x | 32117 | 32690 | 639-655 | 641-867 |
| vbr | 20.0 ms | 320 kb/s | 121.32x | +19.1% | 193.39x | +94.5% | 144.47x | 376.22x | 40134 | 40850 | 799-818 | 798-1075 |
| vbr | 20.0 ms | 384 kb/s | 112.48x | +33.3% | 160.69x | +108.3% | 149.97x | 334.67x | 48151 | 49010 | 958-982 | 953-1275 |
| vbr | 20.0 ms | 512 kb/s | 87.33x | +31.9% | 114.53x | +143.4% | 115.22x | 278.71x | 63800 | 63750 | 1276-1276 | 1275-1275 |

Run `tools/run_raw_celt_bench.sh` to generate your machine's current table.

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
