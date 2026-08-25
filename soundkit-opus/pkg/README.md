# soundkit-opus

This package provides a pure Rust Opus codec and SoundKit streaming adapters.

Use `Encoder` and `Decoder` for the direct 48 kHz CELT codec API.

Use `OpusEncoder`, `OpusDecoder`, and `OpusStreamDecoder` with SoundKit packet streams.

The SoundKit encoder uses `OPUS_APPLICATION_AUDIO` and CBR by default.

The current production target is 48 kHz stereo audio with 5 ms frames. The main bitrate range is 192-320 kb/s.

The allocation-light decoder accepts 48 kHz CELT packets. The general decoder also accepts SILK, hybrid, and legal mode transitions.

## Results

On the isolated x86 test host, Rust encoded 1.19-2.36% faster than trunk libopus. Rust decoded 1.50-4.64% faster.

The test covered 5 ms frames, 16-bit and 24-bit PCM, CBR, constrained VBR, and 192-320 kb/s.

Read the [native performance report](performance-results/2026-08-24-native-48k/README.md) for the complete method and matrix.

The four-source perceptual test scored Rust at 4.5731 MOS-LQO. Trunk libopus scored 4.5724.

Read the [perceptual quality report](quality-results/2026-08-25-soundkit-diverse-v1-5ms/README.md) for the complete results and limits.

Read [CODEC.md](CODEC.md) for the codec API, support limits, and benchmark commands.

## Repository history

SoundKit imported the complete earlier codec history without squashing it. See [HISTORY.md](HISTORY.md) for migration details.
