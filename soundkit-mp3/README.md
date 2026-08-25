# soundkit-mp3

SoundKit's streaming MPEG Layer III codec crate.

## Decode

`Mp3Decoder` incrementally accepts arbitrary byte chunks and emits bounded
`f32`, `i16`, or `i32` PCM without an external decoder crate or C library. The
in-tree decoder owns frame synchronization, the bit reservoir, scale factors,
Huffman decoding, stereo processing, IMDCT, and polyphase synthesis. x86-64
uses runtime-selected SoundKit AVX2 synthesis with an SSE2 fallback; other
targets retain the scalar Rust path.

The former `nanomp3` Git dependency has been removed. Provenance notices for
the permissively licensed implementation lineage and MPEG tables are retained
in [THIRD_PARTY.md](THIRD_PARTY.md).

```rust
use soundkit::audio_packet::Decoder;
use soundkit_mp3::Mp3Decoder;

let input = std::fs::read("music.mp3")?;
let mut decoder = Mp3Decoder::new();
let mut pcm = vec![0.0_f32; input.len() * 64];
let samples = decoder.decode_f32(&input, &mut pcm, false)?;
pcm.truncate(samples);
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Encode

Encoding remains feature-gated behind the default `encode` feature and
currently uses LAME through `mp3lame-encoder`. Decoder-only builds are fully
Rust and dependency-free at the codec-core boundary:

```sh
cargo build -p soundkit-mp3 --no-default-features
```

## Native verification

Results from 2026-08-25 on the SoundKit CPU performance host:

- Intel Xeon Platinum 8581C (Emerald Rapids), 4 vCPUs, Debian 12 x86-64
- Rust 1.97.1, Cargo 1.97.1, FFmpeg 5.1.9, FLAC 1.4.2
- release build with thin LTO, one codegen unit, and `target-cpu=native`
- C reference: lieff/minimp3 at `ea99364f`, Clang
  `-O3 -DNDEBUG -march=native`
- 20 ten-second stereo excerpts from two real FLAC albums, each encoded as
  CBR 128, CBR 320, and VBR V2: 60 MP3 files total, split evenly between
  44.1 kHz and 48 kHz
- three warm-ups per process; 20 measured decodes per file; SoundKit/C order
  alternated for each input

| Decoder | Files | Interleaved samples | Total time | Time/sample |
| --- | ---: | ---: | ---: | ---: |
| SoundKit AVX2 | 60 | 1,111,449,600 | **5.919182 s** | **5.325641 ns** |
| minimp3 C SIMD | 60 | 1,111,449,600 | 6.050758 s | 5.444024 ns |

SoundKit used 2.17% less elapsed decode time per sample (2.22% greater
throughput) across the full corpus. A sample-for-sample differential decode
against minimp3 C covered 55,572,480 samples and measured 145.567 dB SNR,
5.56e-9 RMS error, and 3.58e-7 maximum absolute error. All 60 files matched
channel count, sample rate, and decoded sample count. The four decoder-only
regression/streaming tests pass with `--no-default-features`; all five tests,
including the feature-gated LAME encoder test, pass with default features.
