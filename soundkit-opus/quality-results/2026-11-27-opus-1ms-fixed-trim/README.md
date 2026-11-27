# Fixed allocation trim 8 for 1 ms stereo Opus Custom

A fixed trim of 8 improved mean basic PEAQ at 320 kb/s in two groups of
excerpts. The improvement moved the original group's C 2.5 ms at
256 kb/s mean-score threshold from 344 to 320 kb/s. On seven later passages,
both trim policies still needed a higher rate and the adaptive +4 trim
matched C at a lower tested rate. Keep adaptive +4 as the comparison point
for further quality experiments.

## Original eight excerpts

Each trial encoded signalled 1 ms Opus Custom at 48 kHz stereo CBR. Each
score compares the decoded audio with the same reference excerpt.

| Encoder and payload | Advanced PEAQ DI | Basic PEAQ DI |
|---|---:|---:|
| C 2.5 ms at 256 kb/s | 2.7453 | 2.9229 |
| Adaptive +4 at 256 kb/s | 2.2713 | 2.3858 |
| Fixed 8 at 256 kb/s | 2.2651 | 2.3774 |
| Adaptive +4 at 320 kb/s | 2.8466 | 2.8474 |
| Fixed 8 at 320 kb/s | 2.8485 | 3.0360 |
| Adaptive +4 at 344 kb/s | 2.9735 | 2.9575 |
| Fixed 8 at 344 kb/s | 2.9726 | 2.9675 |

At 320 kb/s, fixed 8 improved basic DI on six of eight clips relative
to adaptive +4. The gains were largest for cymbal, metal, and electronic
music. Basic DI fell on two clips, with a maximum deficit of 0.064 for
orchestra. Compared with C256, the largest basic DI deficit was 0.425 for
speech and the largest advanced DI deficit was 0.055 for electronic music.

The Python ViSQOL Audio means were 4.7117 for fixed 8 at 320 kb/s, 4.7120
for adaptive +4 at 320 kb/s, and 4.7150 for C256. The tool downmixes stereo
to mono. Perceptual equivalence calls for a controlled listening test.

## Later passages from seven source files

The initial parameter screen used cymbal, speech, violin, and metal from
the original eight excerpts. The first comparison then scored all eight,
so that aggregate includes discovery clips. The later passages were
nonoverlapping excerpts from seven of the same source files and were
held back from parameter selection. This checks a new set of passages
within the same source collection.

C256 scored 2.8151 advanced and 3.1886 basic DI on these passages.

| Rust payload | Fixed 8 advanced / basic DI | Adaptive +4 advanced / basic DI |
|---:|---:|---:|
| 256 kb/s | 2.3114 / 2.5074 | 2.3014 / 2.4990 |
| 320 kb/s | 2.8819 / 2.9856 | 2.8787 / 2.8821 |
| 328 kb/s | 2.9299 / 2.9029 | 2.9259 / 2.8856 |
| 336 kb/s | 2.9757 / 2.8806 | 2.9679 / 2.8799 |
| 344 kb/s | 3.0111 / 2.9027 | 3.0066 / 2.8851 |
| 352 kb/s | 3.0369 / 2.9379 | 3.0369 / 2.9609 |
| 368 kb/s | 3.0970 / 2.9666 | 3.0880 / 2.9310 |
| 384 kb/s | 3.1286 / 3.0829 | 3.1187 / 3.1321 |
| 400 kb/s | 3.1536 / 3.1037 | 3.1490 / 3.2143 |
| 432 kb/s | 3.1990 / 3.3501 | — |

Fixed 8 at 320 kb/s improved mean basic DI by 0.103 over adaptive +4 at
320 kb/s and by 0.100 over adaptive +4 at 344 kb/s. The first tested rate
meeting both C256 means was 400 kb/s for adaptive +4 and 432 kb/s for
fixed 8. The grid after 400 kb/s has one additional point, 432 kb/s.

Fixed 8 met C256 advanced DI on every excerpt in both sets at 336 kb/s.
The first tested rates meeting that criterion with adaptive +4 were
344 kb/s on the original set and 352 kb/s on the later set.

At 272 kb/s, fixed 8 met both C192 means on both sets of excerpts. At
432 kb/s compared with C320, it fell 0.018 below the original set's basic
DI mean and exceeded the later set's basic DI mean by 0.063. Advanced DI
was higher in both cases. These were spot checks.

## Other candidates

Adaptive +5 scored 0.045 below C256 basic DI at 320 kb/s on the original
set and 0.325 below on the later passages. Reducing the pitch prefilter
gain to 80% with adaptive +4 came within 0.021 basic DI of C256 at
312 kb/s on the original set and fell 0.288 below on the later passages.
Allowing intensity stereo across all bands on near-mono frames improved
mean advanced DI by about 0.01 in the four-clip screen. The mean basic DI
change varied with the bitrate and the side-to-mid energy threshold.

The same four discovery excerpts also tested fixed trim 8 with forced
spreading off, light or aggressive, and three changes to dynamic allocation.
Light and aggressive spreading improved Basic PEAQ at 312 kb/s and lost
more at 320 kb/s. Changing the dynamic-allocation band cutoffs from indices
8 and 12 to the first bands at or above 1,600 and 3,200 Hz, and lowering
the packet-size gate to 12 bytes, gained 0.105 mean Basic DI at 312 kb/s and
lost 0.191 at 320 kb/s. The other settings disabled the boost or scaled
high-band boosts by 1.0 instead of 0.5. Their scores also fell at 320 kb/s.
[Spread and dynamic allocation screen](fixed8-spread-dynalloc-screen.csv)

## Compatibility and provenance

The [fixed trim experiment patch](fixed8-experiment.patch) changes the
encoder's existing allocation trim symbol after the usual analysis. It
restricts the change to stereo, 48 kHz, `LM = 0`, and a 48-sample short
block. The experiment scored CBR. The measurements used an equivalent
tuning override that set both trim clamp bounds to 8. Both implementations
were built from SoundKit `d0259d8675a5c6887934215236ffaf641f7be436`.

A fresh build of the attached patch reproduced the archived Rust packets
across 176,440 frames. The check covered eight original and seven later
excerpts at 320 kb/s plus seven later excerpts at 400 kb/s. Pinned C
Custom decoded all 16,938,240 i24 values identically to Rust.

The experiment used the same Rust release profile, C libopus revision,
GstPEAQ revision, alignment, and preparation settings as the
[baseline comparison](../2026-11-26-opus-1ms-vs-libopus/README.md#method).
The [held-out manifest](heldout-source-manifest.json) records the offsets
and source and PCM hashes for the later passages.

- [Primary fixed 8 sweep](primary-cbr.csv) includes 72 per-excerpt results.
- [Primary other candidates](primary-other-candidates.csv) records trim +5 and prefilter gain ×0.8.
- [Held-out sweep](heldout-cbr.csv) includes the C 192, 256, and 320 kb/s references.
- [Held-out other candidates](heldout-other-candidates.csv) records trim +5 and prefilter gain ×0.8.
- [Focused discovery sweep](discovery-cbr.csv) records the trim, pitch, dynamic allocation, and stereo variants.
- [Primary ViSQOL Audio results](primary-visqol.csv) record the three original eight-excerpt operating points.

## Reproduce the fixed 8 comparison

Prepare the C binaries and source WAVs using the
[baseline instructions](../2026-11-26-opus-1ms-vs-libopus/README.md#reproduce).
Build adaptive +4 using its
[patch](../2026-11-26-opus-1ms-trim/trim4-experiment.patch). In a fresh
isolated two-crate copy of the pinned SoundKit source, apply this report's
fixed 8 patch and build the `opus_custom_interop` example. Set `TRIM_RUST`
and `FIXED_RUST` to the resulting binaries and `RESULTS` to this report's
directory.

```sh
quality --rust "$FIXED_RUST" --codecs rust --label fixed8 \
  --rates 256000,272000,304000,312000,320000,328000,336000,344000,432000 \
  --output "$WORK/fixed8-primary"

PYTHONPATH="$TOOL" python3 -B "$RESULTS/measure-heldout.py" \
  --work-dir "$WORK" --source-dir "$WORK/source" --peaq "$PEAQ" \
  --adaptive "$TRIM_RUST" --fixed "$FIXED_RUST" --output "$WORK/fixed8-heldout"
```
