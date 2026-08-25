# soundkit-opus

This crate lives in the [SoundKit repository](https://github.com/wavey-ai/soundkit/tree/main/soundkit-opus).

Pure Rust implementation of Opus, with a current focus on 48 kHz CELT.

Current encoder quality and performance work targets 48 kHz stereo at 192,
256, and 320 kb/s. These are transparency candidates, not a transparency
claim; lower bitrates remain in the regression matrix, and controlled listening
tests remain the gate for perceptual changes.

This crate does not wrap libopus or expose a C API. Stable releases and
upstream main provide behavioral evidence. Encoder packets do not need to be
byte-identical when quality and interoperability agree.

## Native Performance

On an isolated x86 GCP host, native Rust encoded `1.19-2.36%` faster and
decoded `1.50-4.64%` faster than trunk libopus across all 12 tested cells.

Tests used 5 ms, 48 kHz stereo audio at maximum complexity. They covered
16-bit and 24-bit PCM, CBR, constrained VBR, and 192-320 kb/s.

All packet sizes and checksums matched the preceding Rust checkpoint. See the
[complete native performance report](performance-results/2026-08-24-native-48k/README.md)
for the matrix, test method, quality gate, and profiler evidence.

## Current Support

- safe packet parser and packet helper APIs
- safe repacketizer and packet padding/unpadding APIs
- soft clipping
- CELT entropy/range coder
- CELT math, Laplace, CWRS/PVQ, DFT, and MDCT
- CELT mode construction, rate allocation, and frame control symbols
- spectral frame coding, quantized energy, and band quantization
- dynamic allocation analysis, theta RDO, and energy-error feedback
- pitch prefilter signaling, filtering, and decoder postfiltering
- LPC tone detection and tone-aware allocation
- transient-aware pitch filtering and ineffective-filter cancellation
- safe stereo band handling when one channel is silent
- spread decision state, band helpers, synthesis, deemphasis, rotation, and
  algebraic VQ
- experimental 48 kHz CELT-only raw packet encode/decode through the Rust
  `Encoder`/`Decoder` types for 2.5, 5, 10, and 20 ms fullband frames
  with CBR, constrained VBR, or exact compressed-frame-byte controls
- signed 24-bit PCM encode/decode APIs using sign-extended `i32` samples

The codec is incomplete. The supported audio path is limited to CELT-only raw
frames. Ogg Opus, SILK, and hybrid speech coding are not implemented.

## Codec Status

The encoder is experimental. CELT-only CBR and constrained VBR are suitable for
internal testing and non-critical generated-audio workflows. They are not
production replacements for libopus.

The current encoder follows the best available upstream 48 kHz CELT behavior.
It does not target byte identity with one libopus release.

### High-Rate Real-Music Quality Checkpoint

Official ViSQOL Audio scored 240 matched Rust and trunk libopus round trips from
20 excerpts in the four-source SoundKit `diverse-v1` corpus.

The matrix used 5 ms frames at 192, 256, and 320 kb/s. It covered CBR,
constrained VBR, and the signed 16-bit and signed 24-bit APIs.

Rust scored `4.5731` MOS-LQO. C scored `4.5724`. The paired Rust-minus-C
difference was `+0.0006`.

The conservative excerpt-level 95% confidence interval was `-0.0012` to
`+0.0024`. All 12 configuration intervals included zero.

Rust scored `4.5368` at 192 kb/s, `4.5848` at 256 kb/s, and `4.5975` at
320 kb/s. The difficult `nocturnal-animals` source gained the most from the
higher rates.

This result shows parity with trunk libopus. It does not prove transparency or
replace a controlled listening test. ViSQOL Audio downmixes stereo to mono.

See [the complete 2026-08-25 quality result](quality-results/2026-08-25-soundkit-diverse-v1-5ms/README.md).
The [2026-08-22 result](quality-results/2026-08-22-westside-after-dark/README.md)
contains the earlier 96-192 kb/s and 5-20 ms matrix.

A one-second, 72-row mixed fixture compares Rust with libopus 1.6.1. The mean
absolute aligned-SNR gap is `0.032 dB`. The maximum gap is `0.18 dB`.

At 5 ms and 48 kb/s, Rust CBR measures `13.15 dB`; C measures `13.28 dB`.
Rust constrained VBR measures `13.40 dB`; C measures `13.32 dB`.

The deterministic fixture is regression evidence, not a listening test.
Synthetic SNR can favor an obsolete filter decision. Real-audio perceptual
tests must gate further encoder changes.

The first 5 ms, 128 kb/s pure-tone packet is byte-identical to C. Later spectral
payloads differ, so tone and PVQ tracing remains active work.

See [PORTING.md](PORTING.md) for the module-by-module plan and test status.
See [SAFETY.md](SAFETY.md) for the unsafe-code policy.

## Build

```sh
cargo test
cargo build --release
```

The public API uses checked slices and owned state. Audited SIMD and MDCT code
lives in a private `kernels` module behind checked safe functions. The crate
does not expose a C API.

The 24-bit methods use the range `-8_388_608..=8_388_607` in `i32`. Conversion
to the internal `f32` path preserves every 24-bit input value exactly. Opus is
still lossy; this avoids an extra 16-bit PCM boundary but does not make the
compressed stream lossless.

Streaming callers can reuse compressed-packet storage with `encode_i16_into`,
`encode_i24_into`, or `encode_f32_into`. The matching `decode_*_into` APIs also
avoid a packet allocation on every short frame.

## WAV Round Trip

The `wav_celt` example round-trips 48 kHz mono/stereo PCM16 WAV through the
implemented CELT-only packet path:

```sh
cargo run --release --example wav_celt -- roundtrip input.wav output.lors decoded.wav
cargo run --release --example wav_celt -- roundtrip --frame-size 240 --bitrate 192000 input.wav output.lors decoded.wav
cargo run --release --example wav_celt -- roundtrip --frame-size 240 --bitrate 192000 --vbr input.wav output.lors decoded.wav
cargo run --release --example wav_celt -- roundtrip --frame-size 960 --frame-bytes 120 input.wav output.lors decoded.wav
```

`output.lors` is a simple length-prefixed raw packet stream for testing this
port. It is not Ogg Opus yet.

Export side-by-side decoded WAVs for listening comparisons:

```sh
tools/export_roundtrip_wavs.sh --input input-audio --out-dir path/to/roundtrips --mode both
```

The helper normalizes the input to 48 kHz stereo PCM16 before running both
implementations. Each case directory contains the Rust packet stream and
decoded WAV plus the upstream `opus_demo` packet stream and decoded WAV.

## Raw CELT benchmark

The raw benchmark compares this crate against libopus through direct in-process
encode/decode calls with no file I/O in the measured loops. Its default matrix
uses a deterministic in-memory 48 kHz stereo fixture at 192, 256, and 320 kb/s;
lower rates remain available through an explicit `--bitrate`.

```sh
tools/run_raw_celt_bench.sh --repeats 21 --seconds 4 --mode both
```

Filter a trace to one row or select the pure-tone fixture:

```sh
tools/run_raw_celt_bench.sh --seconds 1 --repeats 1 --frame-size 240 --bitrate 192000
tools/run_raw_celt_bench.sh --seconds 1 --repeats 1 --frame-size 240 --bitrate 256000 --fixture tone
```

Use `--pcm-bits 16` or `--pcm-bits 24` to benchmark the corresponding integer
encoder and decoder APIs. A real signed-24 source is stored as sign-extended
little-endian `i32` samples. The 16-bit case derives the same source at its
16-bit precision before either implementation sees it:

```sh
tools/run_raw_celt_bench.sh --seconds 10 --repeats 21 --frame-size 240 \
  --bitrate 192000 --input-s32le input-48k-stereo-s24.s32le --pcm-bits 16
tools/run_raw_celt_bench.sh --seconds 10 --repeats 21 --frame-size 240 \
  --bitrate 192000 --input-s32le input-48k-stereo-s24.s32le --pcm-bits 24
```

Add `--skip-quality` to repeated timing runs to skip alignment and SNR
calculation. Packet sizes and checksums remain active. Run at least one pass
without this option as the quality gate.

Use the SoundKit FLAC corpus for the full 48 kHz quality and performance gate.
The runner builds each implementation once. It warms each cell and alternates
Rust with the specified trunk libopus build:

```sh
OPUS_DIR=/path/to/opus-trunk \
  tools/run_soundkit_flac_corpus.py --cpu 2 --json corpus-results.json
```

By default, the runner tests four 100-second sources. It tests 16-bit and
24-bit PCM at 192, 256, and 320 kb/s. Each test uses 5 ms frames, CBR, and
constrained VBR. Set `--seconds 10 --rounds 1 --repeats 3` for a quick check.
Use `--quality-only` for a full-length quality pass without timing rounds. Use
`--skip-quality` for a timing-only pass. The corpus is local test data and is
not part of this repository.

Add `--direct-cubic` to test the experimental high-depth shape coder on the
Rust path. The C reference continues to use standard Opus. Both Rust endpoints
must enable this mode. Its packets are not compatible with standard Opus
decoders.

The mode retains PVQ below its measured high-depth crossover. Therefore, 5 ms
CBR packets at 192, 256, and 320 kb/s remain byte-identical to the default Rust
path. Constrained-VBR packets can differ. The SoundKit 400-second FLAC corpus
showed no aligned-SNR regression at 384 or 512 kb/s. This result is a
regression gate, not a transparency claim.

For local speed runs, benchmark the Rust side with host-native codegen:

```sh
RUST_BENCH_RUSTFLAGS='-C target-cpu=native -C target-feature=+avx2' tools/run_raw_celt_bench.sh --repeats 21 --seconds 4 --mode both
```

Set `OPUS_DIR=path/to/built-libopus` to compare against a built upstream source
tree. Otherwise the script uses `pkg-config opus`. Both encoders use
audio/fullband mode and maximum encoder complexity by default, with CBR or
constrained VBR. Reported speed columns are normalized as realtime speedup:
`RTFx = (seconds * 1000) / elapsed_ms`, where 1.0x is realtime, and larger is
faster. Record the libopus build configuration with performance results. Use
`--application restricted-lowdelay` to
isolate low-delay application behavior on both implementations. Architecture
intrinsics and `--enable-float-approx` materially affect the C
baseline.

Positive deltas mean Rust took longer than C. Negative deltas mean Rust
was faster. Byte counts are raw Opus packet bytes, not wrapper/container bytes.

Packet ranges show per-frame compressed packet byte sizes. The SNR and lag
columns are aligned decode-quality checks against the deterministic fixture.
The script also builds `raw_celt_decode_dump_c` in the benchmark directory for
same-packet C decode checks against `--dump-packets` output.

See the [2026-08-24 native x86 checkpoint](performance-results/2026-08-24-native-48k/README.md)
for the complete timing matrix, isolated A/B tests, profiler findings, and
full-length quality checks.

Run `tools/run_raw_celt_bench.sh` to generate a local table. For a minimal
check, use:

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
`simd128` artifact.

The `simd128` build produces matching checksums. It is approximately 7.5%
faster than scalar WASM for the current Apple Silicon 5 ms encode checkpoint.
Thus, the npm release build enables `simd128`. The benchmark still builds both
variants for revalidation on other runtimes.

## Full-track wasm comparison

For a repeatable browser-loaded comparison against the sibling `libopusjs`
Emscripten build, run:

```sh
npm run bench:browser-wasm -- --seconds 10 --repeats 3 \
  --frame-size 240 \
  --cases c,rust-cbr,rust-vbr,rust-cbr-reuse,rust-vbr-reuse
```

The benchmark serves the local `pkg/soundkit_opus_bg.wasm` output and
`../libopusjs/release/libopus.wasm` through localhost. It starts a new headless
Chrome for each codec and bitrate case. It measures the public browser
JavaScript API for 2.5, 5, 10, or 20 ms stereo CELT frames. The default bitrate
set is 192, 256, and 320 kb/s. Use `--json /tmp/wasm-browser-bench.json` to keep
the raw samples.
The fixture is deterministic synthetic 48 kHz stereo audio for repeatable runs
under lower system load.

The Rust cases also support `rust-cbr-reuse` and `rust-vbr-reuse`. These cases
stage encoder input and reuse encoder/decoder output storage. The 16-bit path
exposes `inputPtr`, `inputLen`, `outputPtr`, and `outputLen`; the signed-24 path
uses the corresponding `inputI24*` and `outputI24*` properties. Conventional
signed-24 calls are also available as `enc_frame_i24` and `dec_frame_i24`.

Run the Rust signed-24 browser path with:

```sh
npm run bench:browser-wasm -- --seconds 10 --frame-size 240 --pcm-bits 24 \
  --bitrates 192000,256000,320000 --cases rust-cbr-reuse,rust-vbr-reuse
```

The sibling `libopusjs` wrapper does not expose libopus 1.6's signed-24 entry
points, so the browser tool rejects the C case with `--pcm-bits 24`. To capture
a Chrome CPU profile around a Rust decode loop, run a single Rust case with
`--profile-rust-decode /tmp/rust-decode.cpuprofile.json`.

A full-track browser API comparison used the *Westside* premix source.

`ffmpeg` decoded and resampled the source one time to identical `48 kHz` stereo
`s16` PCM. Node.js then encoded and decoded 20 ms frames. It used `libopusjs` C
WASM and pure Rust `soundkit-opus` WASM. Quality metrics below are
delay-aligned against the source PCM. Unaligned RMSE/SNR numbers are misleading
because the two paths have different codec delay. The `libopusjs` wrapper does
not expose a CBR/VBR toggle and uses libopus' default VBR behavior.

Rust wasm defaults to CBR for Bitneedle, and also exposes constrained VBR via
`encoder.set_vbr(true)`.

| Target | Codec/mode | Encoded bytes | Effective kb/s | Packet bytes | Encode xRTF | Decode xRTF | Delay | Aligned SNR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 48 kb/s | C `libopusjs` default VBR | 1,260,065 | 48.35 | 3-206 (avg 120.86) | 147.4x | 332.5x | 6.5 ms | 15.33 dB |
| 48 kb/s | Rust `soundkit-opus` CBR | 1,251,120 | 48.00 | 120-120 | 127.2x | 377.3x | 2.5 ms | 14.34 dB |
| 48 kb/s | Rust `soundkit-opus` VBR | 1,260,198 | 48.35 | 109-157 (avg 120.87) | 128.0x | 380.1x | 2.5 ms | 14.41 dB |
| 128 kb/s | C `libopusjs` default VBR | 3,346,824 | 128.41 | 3-543 (avg 321.01) | 110.9x | 279.7x | 6.5 ms | 22.14 dB |
| 128 kb/s | Rust `soundkit-opus` CBR | 3,336,320 | 128.01 | 320-320 | 81.4x | 263.6x | 2.5 ms | 19.01 dB |
| 128 kb/s | Rust `soundkit-opus` VBR | 3,343,132 | 128.27 | 290-416 (avg 320.65) | 82.4x | 268.2x | 2.5 ms | 19.11 dB |

All wasm paths encoded and decoded the full track without failures. Rust VBR
tracks the target byte budget and varies packet sizes, but on this workload it
does not reduce the quality gap relative to C. Rust decode performance is faster
at 48 kb/s. Rust encode remains slower, especially at 128 kb/s, and requires
targeted profiling before parity claims with the C encoder.

A historical local snapshot is included below. Run the tool for current results.

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
| cbr | 20.0 ms | 512 kb/s | 86.86x | +51.8% | 110.00x | +157.5% | 131.86x | 283.21x | 63750 | 63750 | 1275-1275 | 1275-1275 |
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

Run `tools/run_raw_celt_bench.sh` to generate a local current table.

## Historical Encoder Parity Notes

This section records the earlier 1.5.2-focused trace. It is not the active
roadmap. See [TODO.md](TODO.md) for the current quality-led plan.

CBR byte parity remains the active target before VBR parity. On the
deterministic 128 kb/s raw CELT fixture, the first 2.5 ms mismatch was frame 7.
The `AnalysisInfo.tonality_slope` port moved it to frame 15. The analysis
leak-boost port moved the 2.5 ms mismatch to frame 22 and the 5 ms mismatch to
frame 9.

Porting CELT's `FLOAT_APPROX` `celt_log2`/`celt_exp2` helpers fixed
the 2.5 ms frame-22 fine-energy bit flip. Matching C's scaled-energy
`band_log2` path for analysis leak boost fixed the 2.5 ms frame-25 dynalloc
split. Mirroring libopus' energy-error feedback on the local `bandLogE` copy
before trim analysis fixed the 2.5 ms frame-29 trim split.

Mirroring libopus'
post-frame RNG handoff from the folding LCG seed to the range coder's final
`rng` fixed the 2.5 ms frame-91 theta-RDO split. Matching libopus' coarse-energy
badness baseline after the max-decay adjustment fixed the 2.5 ms frame-227
inter/intra split. The one-second CBR dump is byte-identical at 2.5 ms for 48,
96, 128, 192, and 320 kb/s. It is also byte-identical at 5 ms and 128 kb/s.

The 2.5 ms / 128 kb/s frame-15 allocation mismatch was caused by missing
`AnalysisInfo.leak_boost` dynalloc input. The previous frame-22 payload mismatch
occurred because Rust used exact `f32::log2` and `exp2`. This pinned libopus
build uses `FLOAT_APPROX`. That difference changed fine-energy quantization by
one raw bit.

The previous frame-25 allocation mismatch was caused by Rust adding C's
`1e-10f` analysis-log epsilon before applying the same energy scale C uses.
That rounded `leak_boost[3]` from 63 to 64, crossing the dynalloc threshold for
band 3. The previous frame-29 trim/coded-band split was caused by Rust computing
allocation trim from the uncorrected local `band_log_e`. Libopus applies the
prior `energyError` feedback to `bandLogE` before `alloc_trim_analysis`.

The previous 2.5 ms / 128 kb/s frame-91 split was caused by Rust carrying the
band folding seed into the next frame. Libopus uses that seed for folding and
anti-collapse inside the current frame, then stores the range coder's final
`rng` for the following frame. The fixed frame-91 RDO trace matches C's seed,
down/up candidate reconstruction, and selected down-rounded candidate.

The previous 2.5 ms / 128 kb/s frame-227 split was caused by Rust counting
max-decay-limited coarse-energy deltas as badness. Libopus records the
coarse-energy `qi0` baseline after the max-decay clamp. Rust recorded it before
the clamp, causing the intra pass to appear better and leaving only 713
allocation fractional bits where C had 1045. With the baseline moved, frame 227
matches C's inter decision, allocation, and packet bytes.

The 5 ms, 128 kb/s one-second CBR dump is byte-identical for all 200 frames. The
2.5 ms CBR dumps are byte-identical for all 400 frames at these rates: 48, 96,
128, 192, and 320 kb/s. Packet sizes match across the CBR matrix. Remaining byte
mismatches require first-symbol traces. The remaining 2.5 ms CBR mismatches are
frame 226 at 160 kb/s and frame 17 at 256, 384, and 512 kb/s.

The 2.5 ms / 160 kb/s frame-226 trace also reaches matching high-level controls
and matching allocation. The first split is the first fine-energy raw bit for
band 0, channel 0. Rust encodes `q2=15`, and C encodes `q2=14`. The coarse-energy
residual crosses the exact `-0.03125` threshold
(`-0.031249762` Rust versus `-0.031250238` C). Scalar C matches default C
through this frame cluster, so this is tracked as floating sensitivity rather
than a confirmed Rust algorithm bug.

The 2.5 ms / 256 kb/s frame-17 trace reached matching high-level controls and
matching raw-bit call order. The first split is fine energy for band 0,
channel 1. C encodes `q2=27`, and Rust encodes `q2=26`. The band log energy
differs by approximately `0.0009 dB` and crosses the raw-bit threshold. The
drift is present in the pre-MDCT input and first MDCT bin.

Scalar C still
matches default C. Thus, this is floating sensitivity, not a confirmed Rust
algorithm bug.

The 10 and 20 ms CBR paths diverge from frame 0. A 128 kb/s control-symbol trace
shows that frame-0 high-level decisions match libopus. These decisions include
transient, TF, spread, trim, and coded-band values. Thus, the next mismatch is
in the energy/PVQ payload path.

Decoder quality checkpoint from 2026-07-13: the native raw CELT benchmark
reports aligned SNR and lag for Rust and C. The Rust and C dump tools can decode
packets from either implementation with either decoder.

This work found a decoder bug. Libopus applies the CELT decoder postfilter in
place. Thus, delayed taps can read samples filtered earlier in the same frame.
Rust now uses the same in-place feedback. It tracks decoder `oldLogE` and
`oldLogE2` for anti-collapse and applies the decoded silence energy floor.

In the latest one-second, 72-row CBR/VBR matrix, the largest aligned-SNR gap is
`0.43 dB`. All aligned lags match. Symmetric same-packet checks agree within a
`0.01 dB` maximum aligned-SNR delta. There are no lag mismatches for C-generated
or Rust-generated packets. Thus, remaining matrix quality gaps are on the
packet or encoder side until a same-packet check shows otherwise.

Rust decoding of C-generated 512 kb/s packets tracks C at all tested durations.
At 2.5, 5, 10, and 20 ms, results are `42.39`, `41.26`, `41.56`, and `40.89 dB`.

Ported in this checkpoint:

- energy-error feedback
- dynalloc analysis
- theta RDO for stereo CELT bands
- CELT pitch prefilter signaling and input filtering
- CELT decoder postfilter state and in-place filtering
- CELT decoder anti-collapse energy history
- spread decision state
- LM>0 TF analysis and transient patch decision
- transient second-MDCT `bandLogE2` dynalloc input
- dynalloc TF-importance ordering and spread-weight masking
- minimal `AnalysisInfo` tonality-slope, bandwidth, and C-scaled leak-boost
  analysis
- CELT `FLOAT_APPROX` log2/exp2 helpers
- pre-trim energy-error feedback on `bandLogE`
- aligned raw CELT quality reporting and symmetric packet cross-decode helpers
- reusable-output `Decoder::decode_f32_into` for f32 decode loops
- final-range RNG handoff after CELT encode/decode frames

The historical continuation list was:

1. Trace the next 2.5 ms high-rate CBR mismatch, starting with 160 kb/s frame
   226 or the 256/384/512 kb/s frame-17 split.
2. Extend the now-matching 2.5 ms 48/96/128/192/320 kb/s and 5 ms / 128 kb/s
   CBR fixtures beyond one second before classifying those paths as complete.
3. Extend the remaining 5 ms CBR rates after the 128 kb/s fixture stays clean on
   longer deterministic inputs.
4. Trace 10 and 20 ms frame-0 parity after the matching control symbols through
   coarse/fine energy, allocation, and PVQ band quantization.
5. After CBR is bit-identical for the raw CELT matrix, port libopus'
   constrained VBR target/reservoir logic and repeat VBR packet dumps.
6. Keep the aligned quality matrix and same-packet cross-decode helpers in the
   loop. Quality regressions should block byte-parity work even when packet
   sizes remain plausible.

## License

BSD-3-Clause, matching upstream libopus. See [LICENSE](LICENSE).
