# MOV PCM fixtures

FFmpeg 8.1.2 generated these fixtures from the committed 48 kHz mono PCM source.

Each file contains 4,800 stereo frames. The fixtures test the QuickTime `enda` atom.

The matrix covers `in24`, `in32`, `fl32`, and `fl64`. Each format has both byte orders.

Run this command from the SoundKit repository:

```sh
ffmpeg -f s16le -ar 48000 -ac 1 \
  -i testdata/linear16_48/A_Tusk_is_used_to_make_costly_gifts.s16le \
  -t 0.1 -ac 2 -c:a pcm_s24be testdata/mov-pcm/pcm-s24be.mov
```

Change the codec and output name to generate the other fixtures.

`SHA256SUMS` records the container hashes. The Rust tests record decoded PCM hashes from FFmpeg.
