# Opus benchmark results

The current Opus results use the in-tree [`soundkit-opus`](soundkit-opus) codec.

## Native performance

Rust encoded 1.19-2.36% faster than trunk libopus on the isolated x86 test host.

Rust decoded 1.50-4.64% faster than trunk libopus on the same host.

The matrix used 48 kHz stereo audio with 5 ms frames. It covered 16-bit and 24-bit PCM at 192-320 kb/s.

Read the [complete native report](soundkit-opus/performance-results/2026-08-24-native-48k/README.md).

## Perceptual quality

The four-source ViSQOL Audio test scored Rust at 4.5731 MOS-LQO. Trunk libopus scored 4.5724.

The paired Rust-minus-C result was +0.0006. Its conservative 95% confidence interval was -0.0012 to +0.0024.

Read the [complete quality report](soundkit-opus/quality-results/2026-08-25-soundkit-diverse-v1-5ms/README.md).

These results show parity with the tested libopus trunk revision. They do not prove transparency or replace controlled listening tests.
