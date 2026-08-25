# soundkit-opus

This package provides SoundKit's authored native Opus codec and incremental
streaming APIs.

`Encoder` and `Decoder` are the production 48 kHz CELT codec types. They
implement SoundKit's packet traits directly and write into caller-owned
storage; there is no codec wrapper in front of them. `OpusEncoder` and
`OpusDecoder` are zero-cost type aliases retained for existing callers.

Use `OpusStreamEncoder` and `OpusStreamDecoder` when PCM or compressed bytes
arrive in arbitrary chunks. The stream format is `OpusHead` followed by
little-endian `u16` packet lengths.

The SoundKit encoder uses `OPUS_APPLICATION_AUDIO` and CBR by default.

The current production target is 48 kHz stereo audio with 5 ms frames. The main bitrate range is 192-320 kb/s.

The authored decoder accepts 48 kHz CELT packets directly. SILK, hybrid, FEC,
and mode transitions are currently rejected explicitly; no external Opus
decoder or hidden compatibility fallback exists in any feature graph.

## Results

On the isolated Emerald Rapids host, the authored encoder used 1.19-2.36%
less time than trunk libopus across the 5 ms native matrix.

The corrected same-packet i24 decoder gate feeds byte-identical packets to
both decoders. At 192 kb/s constrained VBR across four music sources,
SoundKit was 0.51% slower on C-produced packets, 0.30% faster on
SoundKit-produced packets, and 0.10% slower combined: native parity within
run-to-run noise. The 40 alternating pairs consumed identical packet counts
and bytes.

The earlier 1.50-4.64% decoder result used each implementation's own encoded
packets. It remains useful as an end-to-end measurement but is not the decoder
performance gate because constrained-VBR bitstreams differ.

Read the [native performance report](performance-results/2026-08-24-native-48k/README.md) for the complete method and matrix.

The four-source perceptual test scored Rust at 4.5731 MOS-LQO. Trunk libopus scored 4.5724.

Read the [perceptual quality report](quality-results/2026-08-25-soundkit-diverse-v1-5ms/README.md) for the complete results and limits.

Read [CODEC.md](CODEC.md) for the codec API, support limits, and benchmark commands.

## Repository history

SoundKit imported the complete earlier codec history without squashing it. See [HISTORY.md](HISTORY.md) for migration details.
