# soundkit-vorbis

`soundkit-vorbis` provides SoundKit's streaming Vorbis decoder.

The crate accepts Ogg Vorbis streams and raw Vorbis packets from WebM. It emits
interleaved signed 16-bit PCM before end of file. The codec core is in this
crate. It does not call an external decoder crate or C library.

SoundKit uses Vorbis only as an import format. This crate does not provide a
Vorbis encoder. SoundKit currently writes Opus and FLAC; its final encoding
phase will add AAC writing and fragmented-MP4 boxing for SoundKit LL-HLS, not a
Vorbis encoder.

## Decode APIs

Use `VorbisDecoder` for Ogg streams. Call `add` with arbitrary input chunks,
then call `finish` at end of file.

Use `VorbisPacketDecoder` for raw Vorbis packets from Matroska or WebM. Send the
three header packets before the audio packets.

The decoder keeps Ogg granule-position trimming. It also keeps packet state
across input chunks. The x86 path selects AVX2 at run time. Other CPUs use the
portable Rust path.

## Native performance

These results are from the SoundKit CPU performance host on 2026-08-25:

- Intel Xeon Platinum 8581C, Emerald Rapids, four virtual CPUs, Debian 12 x86-64.
- Rust 1.97.1 and Cargo 1.97.1.
- FFmpeg 5.1.9 and FLAC 1.4.2.
- libvorbisfile 1.3.7 and Clang 14.0.6.
- Release builds used thin LTO, one codegen unit, and `target-cpu=native`.
- The C reference used `-O3 -DNDEBUG -march=native`.

The corpus contains 20 ten-second excerpts from two real FLAC albums. One
album uses 44.1 kHz stereo. The other uses 192 kHz stereo. Each excerpt was
encoded at Vorbis quality 2, 5, and 8. The result contains 60 Ogg files.

Each process performed three warm-up decodes. Each measured run decoded a file
20 times. The test pinned both decoders to one CPU. Test order alternated for
each file.

| Decoder | Files | Interleaved samples | Total time | Time/sample |
| --- | ---: | ---: | ---: | ---: |
| SoundKit Rust | 60 | 2,833,200,000 | **24.477061 s** | **8.639369 ns** |
| libvorbis C | 60 | 2,833,200,000 | 26.796176 s | 9.457919 ns |

SoundKit used 8.65% less elapsed time per sample. It delivered 9.47% more
throughput than libvorbis C.

SoundKit finished first on all 60 files. Its elapsed-time lead was 8.65% at
quality 2, 8.72% at quality 5, and 8.60% at quality 8.

The PCM comparison decoded every file once with each decoder. It compared
141,660,000 interleaved samples. Every file matched its sample rate, channel
count, and sample count. The maximum difference was one integer PCM unit. The
result measured 74.7071 dB SNR and 0.706918 RMS error.

Run the crate tests with:

```sh
cargo test -p soundkit-vorbis
```

The test suite contains streaming, parser, entropy, transform, and golden PCM
checks. One ignored test regenerates the small committed fixture with FFmpeg.

## Provenance

SoundKit bootstrapped the parser and scalar decoder from Lewton revision
`bb2955b717094b40260902cf2f8dd9c5ea62a84a`. SoundKit now maintains that code
inside this crate. It added the streaming integration, bit reader, FFT IMDCT,
AVX2 kernels, and direct PCM output.

See [THIRD_PARTY.md](THIRD_PARTY.md) and [LICENSE-LEWTON](LICENSE-LEWTON) for
the retained notices.
