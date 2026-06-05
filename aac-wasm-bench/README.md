# AAC Bench

Native AAC-LC correctness and speed harness for SoundKit.

This crate is intentionally not the browser wasm path. Browser/worker wasm is
built from `soundkit-wasm-decoder` with Rust `wasm32-unknown-unknown` and
`wasm-bindgen`.

The default native build includes benchmark comparators:

- `fdk-aac` / `fdk-aac-sys`
- `soundkit-aac-lc`
- `symphonia-codec-aac`

`fdk-vs-*` lines are decoder-oracle checks. Actual lossy-audio quality is
reported against the original source WAV as `source-vs-*` measurements when the
WESTSIDE source WAV is available.

## Fast Loop

Use this native-only loop while iterating on the SoundKit AAC-LC decoder:

```sh
cargo test -p aac-wasm-bench --no-default-features --features soundkit-lc
cargo test -p soundkit-aac-lc scalefactor
cargo test -p soundkit-aac-lc imdct_fast
```

This avoids wasm, FDK, Symphonia, and full release benchmarking during small
decoder edits.

## Full Native Runs

SoundKit-only native decode:

```sh
cargo run -p aac-wasm-bench --no-default-features --features soundkit-lc --release -- 5
```

Native SoundKit-vs-FDK conformance:

```sh
cargo test -p aac-wasm-bench --no-default-features --features fdk,soundkit-lc -- --nocapture
```

Full native comparison:

```sh
cargo run -p aac-wasm-bench --release -- 5
```

The optional argument is the number of fixture decode iterations. The CLI also
reports `soundkit-lc-reuse`, which reuses one SoundKit AAC-LC decoder across all
fixture iterations to approximate the steady-state raw packet stream path.

Native quality diagnostics:

```sh
cargo run -p aac-wasm-bench --release -- quality-hotspots 8
cargo run -p aac-wasm-bench --release -- frame-features 1865 1630
cargo run -p aac-wasm-bench --release -- frame-errors 1865 1630
```

`quality-hotspots` reports the highest-RMSE AAC-frame regions for source-vs-FDK,
source-vs-SoundKit, source-vs-Symphonia, and FDK-vs-SoundKit. `frame-features`
prints the AAC tools used by selected raw access units. `frame-errors` breaks
aligned FDK-vs-SoundKit residuals into 128-sample tiles per channel, which is
useful for separating transition-overlap errors from ordinary spectral decode.

## Source WAV

The current fixture is:

```text
golden/aac/WESTSIDE_MIX_4_CONFIRMATION_130323_256k.aac
```

It was encoded from the bitneedle source WAV:

```text
bitneedle/apps/press/testdata/audio-regression/WESTSIDE_MIX 4 CONFIRMATION_130323.wav
```

If the sibling `bitneedle` checkout is not present, set:

```sh
SOUNDKIT_AAC_SOURCE_WAV=/path/to/WESTSIDE_MIX\ 4\ CONFIRMATION_130323.wav
```

## Output

Each decoder line reports speed and single-pass PCM stats:

- `elapsed_ms`
- `rtf`
- `frames_per_sec`
- `quality_samples`
- `rms`
- `peak`
- `checksum`

The `fdk-vs-*` lines compare decoded PCM with FDK as the reference. The
comparison searches a channel-aligned offset because decoders can report
equivalent AAC PCM with different priming delay. Current decoder-oracle pass
thresholds are:

- `rmse <= 0.005`
- `mean_abs_error <= 0.001`
- `max_abs_error <= 0.50`
- `snr_db >= 35`
- equal decoded sample counts

The `source-vs-*` lines compare decoded AAC output with the original source WAV.
These lines are measurements, not pass/fail oracle checks.

## Current Native Result

Command:

```sh
cargo run -p aac-wasm-bench --release -- 5
```

Fixture: 6,428,342 bytes, 9171 ADTS frames, 48 kHz stereo, 195.648 seconds of
AAC frame audio.

| Decoder | Iterations | Decoded frames | Elapsed | RTF | Frames/sec | RMS | Peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fdk-aac-sys` | 5 | 45,855 | 2,045.506 ms | 0.002091 | 22,417.4 | 0.162843351 | 0.918304443 |
| `soundkit-aac-lc` | 5 | 45,855 | 1,250.680 ms | 0.001278 | 36,664.1 | 0.162846547 | 0.918334007 |
| `soundkit-lc-reuse` | 5 | 45,855 | 1,355.441 ms | 0.001386 | 33,830.3 | 0.162846547 | 0.918334007 |
| `symphonia-aac` | 5 | 45,855 | 743.409 ms | 0.000760 | 61,682.1 | 0.162845552 | 1.001383424 |

Decoder-oracle checks:

| Comparison | Status | RMSE | Mean abs | Max abs | SNR |
| --- | --- | ---: | ---: | ---: | ---: |
| `fdk-vs-symphonia` | pass | 0.001002403 | 0.000045033 | 0.260961175 | 44.215 dB |
| `fdk-vs-soundkit` | pass | 0.001242798 | 0.000132421 | 0.370039735 | 42.348 dB |

Source-WAV quality measurements:

| Comparison | RMSE | Mean abs | Max abs | SNR | p99 abs | p999 abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `source-vs-fdk` | 0.006896303 | 0.004594121 | 0.334225774 | 27.510 dB | 0.023480892 | 0.038754225 |
| `source-vs-soundkit` | 0.006919222 | 0.004613361 | 0.383684549 | 27.480 dB | 0.023648702 | 0.038796075 |
| `source-vs-symphonia` | 0.006894503 | 0.004592889 | 0.363226533 | 27.511 dB | 0.023478046 | 0.038671419 |

Current hotspot diagnostics show the largest SoundKit-vs-FDK residuals in
short/long transition overlap tiles, especially frames whose previous short
window used PNS/TNS/intensity. The same regions also appear in the source-vs-FDK
and source-vs-Symphonia hotspot lists, so those frames are lossy/PNS-sensitive
quality checks rather than clean decoder-only failures.

## Wasm Checkpoint

Browser/worker wasm is exercised through the `soundkit-wasm-decoder`
`wasm-bindgen` facade. Use this as an integration checkpoint, not for the inner
decoder loop:

```sh
wasm-pack test --node --release soundkit-wasm-decoder --no-default-features --features aac-lc-bench -- --nocapture
```

Current Node release quality result on the WESTSIDE fixture decodes AAC-LC to
PCM in Rust wasm and compares that PCM directly to the source WAV:

| Comparison | Source | Compared samples | Offset | RMSE | Mean abs | Max abs | p99 abs | p999 abs | SNR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `saved-source-vs-fdk` | saved native | - | - | 0.006896303 | 0.004594121 | 0.334225774 | 0.023480892 | 0.038754225 | 27.510 dB |
| `saved-source-vs-symphonia` | saved native | - | - | 0.006894503 | 0.004592889 | 0.363226533 | 0.023478046 | 0.038671419 | 27.511 dB |
| `wasm-source-vs-sk` | computed in wasm | 18,779,958 | 2,048 samples | 0.006919222 | 0.004613361 | 0.383684535 | 0.023648679 | 0.038796104 | 27.480 dB |

Current Node release speed result on the WESTSIDE fixture:

| Path | Iterations | Decoded frames | Elapsed | RTF | Frames/sec | Checksum |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `wasm-core-raw` | 5 | 45,855 | 1,476.331 ms | 0.001509 | 31,060.1 | `24eb18ebd8fc27e4` |
| `wasm-js-interleaved` | 5 | 45,855 | 1,642.932 ms | 0.001679 | 27,910.5 | `8c2d7264b07abe0a` |
| `wasm-js-into` | 5 | 45,855 | 1,605.535 ms | 0.001641 | 28,560.6 | `0d4a1cec5595b60a` |
