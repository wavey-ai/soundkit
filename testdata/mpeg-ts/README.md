# MPEG-TS conformance fixtures

These one-second fixtures derive from `testdata/wav_stereo/A_Tusk_is_used_to_make_costly_gifts.wav`.

They were generated with FFmpeg 8.1.2:

```sh
ffmpeg -i INPUT -t 1 -ar 48000 -ac 2 -c:a aac -b:a 128k -f mpegts aac-stereo-48k.ts
ffmpeg -i INPUT -t 1 -ar 48000 -ac 2 -c:a aac -b:a 128k -mpegts_m2ts_mode 1 -f mpegts aac-stereo-48k.m2ts
ffmpeg -i INPUT -t 1 -ar 48000 -ac 2 -c:a mp2 -b:a 192k -f mpegts mp2-stereo-48k.ts
```

FFprobe 8.1.2 reports 48 AAC packets in each AAC fixture and 42 packets in the MP2 fixture.

SHA-256:

```text
443540b05358818709fa49da3be04fb4721abaf5e2b5ce65456fcdaa3f4fd711  aac-stereo-48k.m2ts
92d1fdc51f925de12f56294f14ac134618d9ad6860b4876aca5a17eb9e08dc8e  aac-stereo-48k.ts
3409957fe29cd270b137cd14dff7a182e7b292b39f3f06269c98e9fa571a0c8c  mp2-stereo-48k.ts
```
