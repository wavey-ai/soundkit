# Specialty DNx fixtures

`dnxhd-1080i-cid1241.mov` is a one-frame, 1920x1080, 10-bit interlaced
DNxHD fixture generated with FFmpeg 8.1.2:

```sh
ffmpeg -f lavfi -i testsrc2=size=1920x1080:rate=50 \
  -vf tinterlace=interleave_top -frames:v 1 -c:v dnxhd -b:v 185M \
  -pix_fmt yuv422p10le -flags +ildct dnxhd-1080i-cid1241.mov
```

`dnxhr-hqx-12bit.mov` is the official FFmpeg FATE fixture
`dnxhd/dnxhr_cid1271_12bit.mov`, downloaded from
`https://fate-suite.ffmpeg.org/`. It is a one-frame 1920x1080, 12-bit
DNxHR HQX file.

The Rust decoder tests compare tightly packed planar output with FFmpeg 8.1.2
using the scalar simple IDCT (`-cpuflags 0 -idct simple`). `SHA256SUMS` records
the committed container hashes.
