# MXF Avid and AES3 conformance fixtures

`track_01_v02.mxf` and `track_02_a01.mxf` are unmodified Avid Media
Composer 8.6.3 OP-Atom samples from the public FFmpeg FATE suite
(`mxf/track_01_v02.mxf` and `mxf/track_02_a01.mxf`). The former contains ten
1280x720 DNxHR LB coding units at 24000/1001. The latter contains 22022 frames
of 48 kHz, mono, packed 24-bit PCM plus Avid private metadata KLVs.

`d10-aes3-stereo-16bit.mxf` was generated with FFmpeg 8.1.2. Its D-10 audio
essence contains one 1920-frame SMPTE 331M packet: a four-byte header followed
by eight 32-bit AES3 slots per frame, with two significant 16-bit channels.

```sh
ffmpeg -f lavfi -i testsrc2=size=720x576:rate=25:duration=0.04 \
  -f lavfi -i sine=frequency=997:sample_rate=48000:duration=0.04 \
  -map 0:v -map 1:a -vf 'pad=720:608:0:32,setfield=tff' \
  -c:v mpeg2video -g 0 -flags +ildct+low_delay -intra_dc_precision 2 \
  -non_linear_quant 1 -intra_vlc 1 -qscale 1 -ps 1 -qmin 1 \
  -rc_max_vbv_use 1 -rc_min_vbv_use 1 -pix_fmt yuv422p \
  -minrate 30000k -maxrate 30000k -b:v 30000k -bufsize 1200000 \
  -rc_init_occupancy 1200000 -qmax 12 -c:a pcm_s16le -ar 48000 -ac 2 \
  -f mxf_d10 d10-aes3-stereo-16bit.mxf
```

`malformed-opatom-truncated.mxf` is the committed FFmpeg-generated
`pcm24-mono-48k.mxf` truncated to 149000 bytes, inside its clip-wrapped essence
KLV. It verifies deterministic rejection of a declared range beyond EOF.

`Avid-00005.mxf` is another unmodified FFmpeg FATE sample, authored by AVID
TRMG 2.97. It contains one unsupported DV picture track and two independent
48 kHz mono PCM tracks. SoundKit intentionally ignores the unsupported picture
essence while indexing both audio tracks (25 edit units each).

Hashes are pinned in `SHA256SUMS`.

FFmpeg 8.1.2 elementary PCM SHA-256 references:

```text
a52dbfc5af845dce23f28bc5b7b95bd75aabd55bd1641a90e64075973e8381e4  track_02_a01.s24le
0a538ff65fd57a22526103706538ec42b1370e9fb5cf89924a143023d7b4f93c  d10-aes3-stereo-16bit.s16le
c6649bfc9efb59ac9753e1704b6ee8e7c27e0dfdedc7d2b56227964a664a0649  Avid-00005-audio-track-{1,2}.s16le
```
