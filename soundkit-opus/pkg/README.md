# soundkit-opus

This package provides SoundKit's authored native Opus codec and incremental
streaming APIs.

`Encoder` and `Decoder` are the production 48 kHz CELT codec types and
implement SoundKit's packet contracts directly. `OpusEncoder` and
`OpusDecoder` are zero-cost type aliases.

Use `OpusStreamEncoder` and `OpusStreamDecoder` for arbitrary input chunks.

The SoundKit encoder uses `OPUS_APPLICATION_AUDIO` and CBR by default.

The current production target is 48 kHz stereo audio with 5 ms frames. The main bitrate range is 192-320 kb/s.

The decoder accepts 48 kHz CELT packets. SILK, hybrid, FEC, and mode
transitions are rejected explicitly; there is no external compatibility
decoder in the feature graph.

## Results

On the isolated x86 host, the encoder used 1.19-2.36% less time than trunk
libopus. The corrected byte-identical-packet i24 gate measured the decoder at
0.10% slower combined across C- and SoundKit-produced music packets: native
parity within run-to-run noise.

The test covered 5 ms frames, 16-bit and 24-bit PCM, CBR, constrained VBR, and 192-320 kb/s.

Read the [native performance report](performance-results/2026-08-24-native-48k/README.md) for the complete method and matrix.

The four-source perceptual test scored Rust at 4.5731 MOS-LQO. Trunk libopus scored 4.5724.

Read the [perceptual quality report](quality-results/2026-08-25-soundkit-diverse-v1-5ms/README.md) for the complete results and limits.

Read [CODEC.md](CODEC.md) for the codec API, support limits, and benchmark commands.

## Repository history

SoundKit imported the complete earlier codec history without squashing it. See [HISTORY.md](HISTORY.md) for migration details.
