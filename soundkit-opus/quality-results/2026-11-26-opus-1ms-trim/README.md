# Opus Custom 1 ms stereo allocation trim experiment

Adding four to the CELT allocation trim for 48-sample stereo frames improved
mean advanced PEAQ at every measured equal-rate CBR point. Across the same
eight excerpts and libopus build as the initial 1 ms comparison, the trim
experiment still needed more payload than C 2.5 ms to match quality scores.
The patch adjusts the encoder's signalled allocation trim and uses the
existing decoder format.

## Equal-rate CBR

Each row averages eight 48 kHz stereo excerpts. Changes are the trim
experiment minus baseline Rust 1 ms. The C comparison uses standard
restricted-lowdelay at 2.5 ms.

| Payload rate | Rust advanced DI | Change | C advanced DI | Rust basic DI | Change | C basic DI |
|---:|---:|---:|---:|---:|---:|---:|
| 192 kb/s | 1.153 | +0.293 | 2.105 | 1.730 | +0.172 | 2.455 |
| 256 kb/s | 2.271 | +0.199 | 2.745 | 2.386 | +0.101 | 2.923 |
| 320 kb/s | 2.847 | +0.096 | 3.033 | 2.847 | +0.142 | 3.305 |

Advanced DI improved on every clip at all three rates. Basic DI fell
in three of 24 comparisons. The metal excerpt fell by 0.046 at 192 kb/s
and 0.095 at 256 kb/s. The cymbal excerpt fell by 0.035 at 320 kb/s.

## Quality-matched CBR

The table gives the lowest tested Rust 1 ms rate for each criterion. Each
criterion compares the measured score with C 2.5 ms using matched excerpts.

| C payload rate | Rust mean advanced DI | Rust mean advanced and basic DI | Rust per-clip advanced DI across all eight |
|---:|---:|---:|---:|
| 192 kb/s | 248 kb/s | 272 kb/s | 280 kb/s |
| 256 kb/s | 312 kb/s | 344 kb/s | 344 kb/s |
| 320 kb/s | 368 kb/s | 432 kb/s | 400 kb/s |

The original 1 ms encoder met the mean advanced DI criterion at 264,
320, and 384 kb/s. The trim experiment cuts that payload by 16, 8,
and 16 kb/s respectively. The mean advanced DI at 304 versus C 256 and
360 versus C 320 was within 0.001 of C. Basic DI still fell short there.
At 344 kb/s versus C at 256 kb/s, the trim experiment led in both means
by 0.228 advanced DI and 0.035 basic DI, with a largest per-clip basic
DI deficit of 0.270.

At the strict mean advanced DI rates of 248, 312, and 368 kb/s, the Python
ViSQOL Audio means were 4.659, 4.709, and 4.719 versus C's 4.681, 4.715,
and 4.719. The largest per-excerpt deficits were 0.097, 0.024, and 0.004.
The eight excerpts include four used to screen candidate trim offsets and
four held back until validation. Perceptual equivalence calls for a
controlled listening test. The objective ViSQOL measurement downmixes
stereo to mono.

## Compatibility and signalling

C Custom decoded all 192,480 equal-rate test packets to the same i24
samples as the Rust decoder. At 248, 312, 344, and 368 kb/s CBR, the
static patch reproduced the tuning-run encoder packets. The C Custom
decoder matched the Rust decoder over these 256,640 packets as well.
At the four rates, implicit signalling saved exactly 8 kb/s and preserved
every remaining payload byte and decoded sample across all eight excerpts.
This option requires both endpoints to agree on the Custom mode.

## Data and reproduction

- [Equal-rate CBR](equal-rate.csv) and [CBR rate sweep](rate-sweep.csv) retain per-excerpt advanced and basic PEAQ scores.
- [Constrained VBR](cvbr.csv) retains the equal-target-rate measurements. Use the reported payload column when comparing VBR.
- [Exact experiment patch](trim4-experiment.patch) contains the six-line change to the 48-sample stereo encoder.
- [C decoding and signalling checks](signalling.csv) compare the static patch's packets against the tuning run and measure the signalling byte.
- [ViSQOL Audio](visqol-python.csv) compares the strict mean advanced DI operating points.

The measurements use SoundKit `d0259d8675a5c6887934215236ffaf641f7be436`,
Rust 1.100.0, the release profile with one codegen unit and thin LTO,
and the original report's audio, PEAQ method and C reference. Apply the
patch in a separate copy of the pinned source. Follow the
[baseline build and corpus instructions](../2026-11-26-opus-1ms-vs-libopus/README.md#reproduce)
to prepare the scratch workspace, C binaries, and original results
under `$WORK/direct`. Then prepare the isolated trim build separately.

```sh
TRIM_WORKSPACE="$WORK/opus-trim4"
RESULTS="$REVIEW_CHECKOUT/soundkit-opus/quality-results/2026-11-26-opus-1ms-trim"
mkdir -p "$TRIM_WORKSPACE"
cp -R "$BUILD_WORKSPACE/Cargo.toml" "$BUILD_WORKSPACE/Cargo.lock" \
  "$BUILD_WORKSPACE/soundkit" "$BUILD_WORKSPACE/soundkit-opus" "$TRIM_WORKSPACE/"
(
  cd "$TRIM_WORKSPACE"
  git apply "$RESULTS/trim4-experiment.patch"
  cargo build --release -p soundkit-opus --example opus_custom_interop
)
TRIM_RUST="$TRIM_WORKSPACE/target/release/examples/opus_custom_interop"

quality --rust "$TRIM_RUST" --codecs rust --label trim4 --output "$WORK/trim4-equal"
quality --rust "$TRIM_RUST" --codecs rust --label trim4 \
  --rates 232000,240000,248000,264000,272000,280000,288000,296000,304000,312000,328000,336000,344000,352000,360000,368000,376000,384000,392000,400000,408000,416000,432000,448000 \
  --output "$WORK/trim4-rates"
quality --rust "$TRIM_RUST" --codecs rust --label trim4 --modes vbr --output "$WORK/trim4-cvbr"
PYTHONPATH="$TOOL" python3 -B "$RESULTS/verify-trim4.py" --rust "$TRIM_RUST" \
  --c-custom "$WORK/c_custom" --inputs "$WORK/inputs" --renders "$WORK/trim4-rates" \
  --output "$WORK/trim4-signalling.csv"
python3 -B "$RESULTS/score-trim4-visqol.py" --references "$WORK/inputs" \
  --rust-frames "$WORK/trim4-rates" --c-frames "$WORK/direct" \
  --output "$WORK/trim4-visqol-python.csv"
```
