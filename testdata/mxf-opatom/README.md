# MXF OP-Atom conformance fixtures

These fixtures were generated with FFmpeg 8.1.2.

`pcm24-mono-48k.mxf` contains one second of 48 kHz, 24-bit mono PCM. It derives from the committed stereo WAV fixture.

`dnxhr-hqx-one-frame.mxf` contains one copied DNxHR HQX frame. It derives from the committed DNxHR HQX MOV fixture.

FFprobe 8.1.2 reports 25 audio packets and one video packet.

SHA-256:

```text
472aca6dfea95e8ba15459cf9f7787d026360693759affd4bf844c2cce06621e  dnxhr-hqx-one-frame.mxf
aedfba2377da01721eecd7d475962c63b12f4f258915c468bb0ede5c395d847e  pcm24-mono-48k.mxf
```
