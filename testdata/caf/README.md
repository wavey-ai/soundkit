# CAF PCM fixtures

FFmpeg 8.1.2 generated these fixtures from the committed 48 kHz mono PCM source.

Each file contains 4,800 stereo frames. The fixtures cover integer and floating-point PCM formats.

Run these commands from the SoundKit repository:

```sh
ffmpeg -f s16le -ar 48000 -ac 1 \
  -i testdata/linear16_48/A_Tusk_is_used_to_make_costly_gifts.s16le \
  -t 0.1 -ac 2 -c:a pcm_s16be testdata/caf/pcm-s16be.caf
```

Change `pcm_s16be` and the output name to generate the other formats.

The test matrix uses `pcm_s24le`, `pcm_s32be`, `pcm_f32le`, and `pcm_f64be`.

`SHA256SUMS` records the container hashes. The Rust tests also record decoded PCM hashes from FFmpeg.
