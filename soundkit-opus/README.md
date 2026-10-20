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

In WebAssembly, the encoder used 3.44-3.59% less time than trunk libopus on
x86-64 and 0.82-0.88% less on Graviton3. The decoder used 2.82-3.70% and
0.64-0.74% less time. Read the
[wasm performance report](performance-results/2026-10-20-wasm-48k/README.md).

The encoder changes four libopus decision constants. Over 360 paired 10-second
items at 96-320 kb/s, CBR and constrained VBR, Rust scored higher than trunk
libopus. The PEAQ advanced Distortion Index was 0.0118 higher, and the ViSQOL
MOS-LQO was 0.0025 higher. Read the
[encoder tuning report](quality-results/2026-10-20-encoder-tuning/README.md).

`set_vbr_constraint(false)` selects unconstrained VBR. A drift correction holds
its long-term rate at the target. Constrained VBR scored higher and remains
the default.

Before the encoder changes, the 2026-08-25 four-source test scored Rust at
4.5731 MOS-LQO and trunk libopus at 4.5724.

Read the [2026-08-25 perceptual quality report](quality-results/2026-08-25-soundkit-diverse-v1-5ms/README.md) for the complete results and limits.

Read [CODEC.md](CODEC.md) for the codec API, support limits, and benchmark commands.

## Repository history

SoundKit imported the complete earlier codec history without squashing it. See [HISTORY.md](HISTORY.md) for migration details.
