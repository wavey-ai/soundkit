# SoundKit 1 ms vs C libopus 2.5 ms

SoundKit's 1 ms Opus Custom reduces frame time plus codec delay from 5 ms to
2 ms versus standard C Opus in restricted-lowdelay mode. At the same CBR
payload bitrate, SoundKit scored lower on both PEAQ metrics on all eight
excerpts. Matching the mean advanced PEAQ Distortion Index took 64–72 kb/s
more payload on the tested grid. Basic PEAQ still scored lower at those
operating points.

## Quality and bytes

Each row compares eight matched excerpts at 48 kHz stereo. PEAQ and
ViSQOL scores are higher when they indicate better quality.

| C 2.5 ms payload | Rust 1 ms payload | PEAQ advanced DI, C / Rust | PEAQ basic DI, C / Rust | ViSQOL Audio MOS-LQO, C / Rust | With 16-byte packet headers, C / Rust |
|---:|---:|---:|---:|---:|---:|
| 192 kb/s | 264 kb/s | 2.105 / 2.204 | 2.455 / 2.356 | 4.681 / 4.698 | 243.2 / 392 kb/s |
| 256 kb/s | 320 kb/s | 2.745 / 2.751 | 2.923 / 2.706 | 4.715 / 4.714 | 307.2 / 448 kb/s |
| 320 kb/s | 384 kb/s | 3.033 / 3.036 | 3.305 / 2.940 | 4.719 / 4.720 | 371.2 / 512 kb/s |

The largest per-excerpt advanced DI deficits at these operating points were
0.352 for speech, 0.236 for speech, and 0.093 for violin. The lowest Rust
rates that equalled or exceeded C on all eight excerpts were 304 kb/s
against C at 192 kb/s and 368 kb/s against C at 256 kb/s. Against C at
320 kb/s, the sweep reached 384 kb/s with at least one excerpt still below C.

At equal payload rates of 192, 256, and 320 kb/s, Rust's mean advanced DI
was 0.860, 2.072, and 2.751. C's standard Audio mode gave means of 2.116,
2.762, and 3.035 at the same rates. For constrained VBR the mean measured payload rates were
191.38, 255.15, and 318.91 kb/s in Rust and 195.23, 259.23, and 323.24 kb/s
in C restricted-lowdelay.

With one frame per packet, 1 ms yields 1,000 packets/s and 2.5 ms yields
400 packets/s. The header column includes the 16-byte SoundKit v2 header.
AEP1, UDP, IP, and FEC would add to these rates.

Rust and C **Custom** at 1 ms produced identical packets and decoded i24
samples across all 192,480 frames in the CBR control. In 48 comparisons
across CBR and constrained VBR, turning off Custom signalling at both ends
and reducing the configured bitrate by 8 kb/s preserved every remaining
packet byte and decoded sample. This setting requires both endpoints to
agree on the Custom mode without reading a signalling byte.

The source selection and objective metrics give a limited quality estimate.
ViSQOL uses a third-party Python implementation here, downmixes stereo to
mono, and reached 4.732 on an identical reference. A controlled listening
test on representative stereo programme material is still needed to
establish perceptual equivalence.

### Encoder experiments

Using actual frame cadence in the Custom encoder's equivalent-rate
calculation improved mean advanced DI by 0.080 at 192 kb/s and 0.079 at
256 kb/s. Gains were 0.029 at 272 kb/s and 0.013 at 288 kb/s. Scores at 304
and 320 kb/s were unchanged. This experiment changes packet decisions and
breaks the byte-for-byte C Custom encoder match. The default keeps C Custom
parity. [Exact experiment patch](duration-experiment.patch)

Switching only the existing SoundKit encoder decision constants on for
Custom gave essentially the same advanced DI at 192, 256, and 320 kb/s, and
a lower mean basic DI at 256 and 320 kb/s.
[SoundKit decision experiment](soundkit-decisions-experiment.csv)

## Data

- [Direct comparison](direct.csv) covers four codecs, three CBR rates, and eight excerpts.
- [Rust 1 ms rate sweep](rate-sweep.csv) extends the CBR grid by eleven bitrates.
- [Constrained VBR](cvbr.csv) records per-excerpt quality and measured payload rates.
- [ViSQOL Audio](visqol-python.csv) includes the identity reference scores.
- [Signalling check](signalling.csv) records every measured byte and decode comparison.
- [Duration experiment](duration-experiment.csv) records the separate encoder variant.
- [Source manifest](source-manifest.json) records the original file and prepared PCM SHA-256 values.
- [Verification](verification.md) records the native tests, interop checks, and timing status.

## Method

- SoundKit `28120cc3f537b607833fbd64d2900e18bad1ffd7`, release build with
  thin LTO and one codegen unit, Rust 1.100.0 on x86-64.
- xiph/opus `316c675fa1c9982d76e9cdcba7231076f75047da`, GCC 14.2,
  release with Custom modes enabled, DRED, Deep PLC, and OSCE disabled.
  Standard C sets fullband, music signal, and complexity 10. Custom controls
  use the default libopus arithmetic, encoder decisions, and signalling.
- FFmpeg 7.1.5 with the SoX resampler converted the original 44.1 kHz WAVs to
  stereo 48 kHz. Each 8.020-second input was attenuated only if its peak
  exceeded −1 dBFS, then rounded to signed 24-bit PCM. The score uses the
  first 8.000 seconds of the reference with one shared gain per source.
- The scorer drops the codec delay of 48, 120, or 312 samples for Custom,
  restricted-lowdelay, and Audio respectively. Cross-correlation confirmed
  those values, with a peak one sample earlier on the standard C orchestra
  excerpt.
- GstPEAQ `7397e49f5ede2f2429fb78f6663784e548401fec` scores the aligned
  stereo float WAVs. Python 3.12.14, NumPy 2.3.5, and SciPy 1.17.0 prepared
  the inputs and checked the lag. The supplemental ViSQOL Python 3.8.0
  run used NumPy 2.5.5, SciPy 1.18.1, and Numba 0.68.1.
- SNR in the CSV is a signal error diagnostic. Runtime requires the
  [native performance procedure](../../CODEC.md#raw-celt-benchmark).

## Reproduce

The [measurement tool](../../tools/opus_1ms_quality/quality.py) creates
the aligned files used by the
[packet checks](../../tools/opus_1ms_quality/verify_packets.py) and
[ViSQOL scorer](../../tools/opus_1ms_quality/score_visqol.py).

Download the original WAVs into a scratch directory named `source`. The
input stem and offset for each excerpt are in the
[source manifest](source-manifest.json).

| Clip | Excerpt start, seconds | Original WAV |
|---|---:|---|
| Castanets | 6 | [QOA sample](https://qoaformat.org/samples/sqam/27_castanets_single_tone_rhythm_stereo.wav) |
| Cymbal | 3 | [QOA sample](https://qoaformat.org/samples/sqam/31_cymbal.wav) |
| English speech | 5 | [QOA sample](https://qoaformat.org/samples/sqam/49_female_speech_english_mono.wav) |
| Violin | 4 | [QOA sample](https://qoaformat.org/samples/sqam/59_violin_ravel_stereo.wav) |
| Orchestra | 20 | [QOA sample](https://qoaformat.org/samples/sqam/65_orchestra_strauss_stereo.wav) |
| Allegaeon, Beasts and Worms | 22 | [QOA sample](https://qoaformat.org/samples/bandcamp/allegaeon-beasts-and-worms.wav) |
| Four Tet, Baby | 20 | [QOA sample](https://qoaformat.org/samples/bandcamp/four_tet_baby.wav) |
| Julien Baker, Sprained Ankle | 12 | [QOA sample](https://qoaformat.org/samples/bandcamp/julien_baker_sprained_ankle.wav) |

Set `SOUNDKIT` and `OPUS_SRC` to the pinned source checkouts,
`REVIEW_CHECKOUT` to the checkout containing this report, `PEAQ` to the
pinned GstPEAQ command-line executable, and `WORK` to the scratch directory.
Use absolute paths. Install NumPy and SciPy for Python and use a build of FFmpeg
with libsoxr. The GstPEAQ `peaq` executable must be able to load the
`libgstpeaq.so` plugin. If the plugin is outside the default GStreamer
path, pass its directory through `GST_PLUGIN_PATH`.
[GstPEAQ source and build instructions](https://github.com/HSU-ANT/gstpeaq/tree/7397e49f5ede2f2429fb78f6663784e548401fec)

The reported build used an isolated workspace with the `soundkit` and
`soundkit-opus` crates because an unrelated Git dependency blocked
resolution of the full workspace. The commands below create the same
two-crate workspace with the repository's release profile.

```sh
BUILD_WORKSPACE="$WORK/opus-only"
mkdir -p "$BUILD_WORKSPACE"
cp -R "$SOUNDKIT/soundkit" "$SOUNDKIT/soundkit-opus" "$BUILD_WORKSPACE/"
cat > "$BUILD_WORKSPACE/Cargo.toml" <<'TOML'
[workspace]
resolver = "2"
members = ["soundkit", "soundkit-opus"]

[profile.release]
codegen-units = 1
lto = "thin"
TOML
cargo build --manifest-path "$BUILD_WORKSPACE/Cargo.toml" --release \
  -p soundkit-opus --example opus_custom_interop
RUST="$BUILD_WORKSPACE/target/release/examples/opus_custom_interop"
TOOL="$REVIEW_CHECKOUT/soundkit-opus/tools/opus_1ms_quality"
OPUS_TOOLS="$SOUNDKIT/soundkit-opus/tools"
OPUS_BUILD="$WORK/libopus"

cmake -S "$OPUS_SRC" -B "$OPUS_BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DOPUS_CUSTOM_MODES=ON -DOPUS_BUILD_PROGRAMS=OFF -DOPUS_BUILD_TESTING=OFF \
  -DOPUS_DRED=OFF -DOPUS_DEEP_PLC=OFF -DOPUS_OSCE=OFF
cmake --build "$OPUS_BUILD" --target opus
sed 's/OPUS_APPLICATION_AUDIO/OPUS_APPLICATION_RESTRICTED_LOWDELAY/' \
  "$OPUS_TOOLS/raw_pcm_roundtrip.c" > "$WORK/raw_pcm_roundtrip_lowdelay.c"
cc -O2 -I"$OPUS_SRC/include" "$WORK/raw_pcm_roundtrip_lowdelay.c" "$OPUS_BUILD/libopus.a" -lm -o "$WORK/c_lowdelay"
cc -O2 -I"$OPUS_SRC/include" "$OPUS_TOOLS/raw_pcm_roundtrip.c" "$OPUS_BUILD/libopus.a" -lm -o "$WORK/c_audio"
cc -O2 -DOPUS_CUSTOM_INTERNALS -DHAVE_CONFIG_H -I"$OPUS_BUILD" \
  -I"$OPUS_SRC/include" -I"$OPUS_SRC/celt" -I"$OPUS_SRC" \
  "$OPUS_TOOLS/opus_custom_interop.c" "$OPUS_BUILD/libopus.a" -lm -o "$WORK/c_custom"

quality() {
  python3 -B "$TOOL/quality.py" --rust "$RUST" --source-dir "$WORK/source" \
    --work-dir "$WORK" --peaq "$PEAQ" "$@"
}
quality --codecs rust,lowdelay,custom,audio --output "$WORK/direct"
quality --codecs rust --rates 224000,240000,264000,272000,288000,304000,328000,336000,352000,368000,384000 \
  --output "$WORK/rate_sweep"
quality --codecs rust,lowdelay --modes vbr --output "$WORK/cvbr"
python3 -B "$TOOL/verify_packets.py" --work-dir "$WORK" --rust "$RUST"

# Optional ViSQOL Audio comparison. Numba speeds up scoring with the Python port.
python3 -m pip install "visqol-python[accel]==3.8.0"
python3 -B "$TOOL/score_visqol.py" --work-dir "$WORK"
```

To reproduce the duration experiment, apply the linked patch at the pinned
SoundKit revision in a separate checkout or scratch workspace. Build the
same example and run the measurement with that binary. The second command
uses the unpatched binary.

```sh
quality --rust "$DURATION_RUST" --codecs rust --label rust_duration \
  --rates 192000,256000,272000,288000,304000,320000 --output "$WORK/duration"
quality --codecs rust --label rust_tuned --encode-options='--decisions soundkit' \
  --output "$WORK/tuned"
```
