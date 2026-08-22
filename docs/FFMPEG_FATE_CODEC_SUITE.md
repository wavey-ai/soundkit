# FFmpeg FATE codec suite

SoundKit uses a curated part of the FFmpeg FATE sample suite as its formal
codec integration suite.

The suite requires `ffmpeg`, `ffprobe`, `rsync`, and `shasum`. Samples are
downloaded from the upstream FATE server and verified against the SHA-256 in
`scripts/ffmpeg-fate-codec-manifest.tsv`; corpus files are not committed.

Run the correctness suite:

```sh
make codec-fate-test
```

Run the correctness suite and the local FFmpeg comparison benchmark:

```sh
make codec-fate-bench
```

The fetch step stores hash-verified samples in the ignored
`build/fate-codec-corpus` directory. The manifest records accepted codecs and
valid codec profiles that SoundKit does not support.

For accepted audio, the runner converts SoundKit and FFmpeg output to signed
16-bit PCM. It aligns whole sample frames and enforces the manifest SNR limit.
For accepted video, the runner compares normalized planar samples and enforces
the manifest PSNR limit.

Every manifest row has one of three outcomes:

| Outcome | Required result |
| --- | --- |
| `accept` | SoundKit decodes the sample and meets the SNR or PSNR threshold against FFmpeg. |
| `mismatch` | SoundKit decodes the sample but remains below the declared quality threshold or emits the wrong video extent. If it starts passing, the suite fails until the row is promoted to `accept`. |
| `reject` | FFmpeg decodes the sample and SoundKit returns an error or no decoded frames. Unexpected decoded output fails the suite. |

A panic, hang, checksum change, or silent profile substitution fails the suite.

## Current accepted baselines

| Codec | FATE baseline | Oracle result |
| --- | --- | --- |
| AAC-LC | mono MP4, extracted losslessly to ADTS | 78.35 dB SNR |
| MP3 Layer III | mono conformance bitstream | 80.03 dB SNR |
| Vorbis | stereo Ogg | 54.66 dB SNR |
| AMR-NB | 12.2 kbit/s | 30.92 dB SNR |
| H.264 | constrained baseline, progressive 4:2:0 | pixel-exact |
| HEVC | Main, 8-bit 4:2:0 IPCM | pixel-exact |
| VP9 | profile 0, static-resolution 4:2:0 | pixel-exact |
| AV1 | Main, 8-bit 4:2:0 | pixel-exact |

## Current codec gaps

| Codec surface | Reproduced gap |
| --- | --- |
| AC-3 stereo | Decodes, but output is -9.33 dB SNR against FFmpeg. |
| Opus stereo | Decodes, but output is 11.47 dB SNR against FFmpeg. |
| Opus surround | Rejects mapping family 1. |
| E-AC-3 | No decoded audio output. |
| DTS-ES | No decoded audio output. |
| AMR-WB | No decoded audio output. |
| H.264 | Rejects interlaced/field coding and a 4:2:0 8-bit to 4:4:4 10-bit transition. |
| HEVC | A Main-profile merge vector decodes incorrectly; one Main10 stream fails `cu_qp_delta`; 4:2:2 10-bit RExt is unsupported. |
| VP9 | A midstream resolution change emits the wrong output extent. |

A green run currently means all eight accepted baselines agree with FFmpeg and
all twelve declared gaps reproduce. It does not mean every SoundKit codec or
profile is represented yet.

## Local benchmark

The benchmark runs only `accept` rows and refuses to start unless the complete
correctness suite is green. Override the default five iterations with, for
example:

```sh
make codec-fate-bench CODEC_FATE_BENCH_ITERATIONS=20
```

SoundKit time includes in-process decoder construction and decode. FFmpeg time
uses FFmpeg's own single-thread `rtime`, excluding CLI launch. Container-to-
elementary-stream extraction for SoundKit is performed before timing. The
reported `SK/FF` ratio is SoundKit time divided by FFmpeg time, so lower is
faster.

ProRes, DNx, FLAC, and ALAC are not represented in this first matrix. Future
rows should remain small and record the upstream sample's SHA-256 in the
manifest.
