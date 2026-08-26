# soundkit-alac

`soundkit-alac` provides SoundKit's bounded, pure-Rust ALAC decoder for M4A,
MP4, MOV, and CAF imports. The codec core is maintained in this crate. It does
not call an external decoder package, C library, FFI boundary, or platform
codec.

ALAC is an import format for SoundKit, so this crate is decoder-only. SoundKit
currently writes Opus and FLAC; its final encoding phase will add AAC writing
and fragmented-MP4 boxing for SoundKit LL-HLS, not an ALAC encoder.

## Decode APIs

Use `AlacPacketDecoder` with the ALAC magic cookie from a container track, then
pass it one indexed access unit at a time. `decode_packet_into` is the
allocation-free streaming hot path: it clears a caller-owned `Vec<u8>` while
retaining its capacity and writes interleaved signed little-endian PCM.
`decode_packet` remains as an owned-`AudioData` convenience API.

Use `Mp4MediaIndex` from `soundkit-audio-demux` for M4A, MP4, or MOV. Use
`CafAlacPacketIndex` for CAF. Both paths retain bounded metadata and read only
the requested packet ranges; they do not retain a complete media extent.

The decoder accepts 16-, 24-, and 32-bit ALAC. The real-music validation below
covers compressed 16- and 24-bit streams. A full-width uncompressed 32-bit
packet is covered by the crate tests because FFmpeg 5.1's ALAC encoder emits at
most 24-bit streams.

## Native performance

These results are from the SoundKit CPU performance host on 2026-08-26:

- Intel Xeon Platinum 8581C, Emerald Rapids, four virtual CPUs, Debian 12 x86-64.
- Rust 1.97.1 and Cargo 1.97.1.
- FFmpeg/libavcodec 5.1.9, Clang 14.0.6, and FLAC 1.4.2.
- Release builds used thin LTO, one codegen unit, and `target-cpu=native`.
- The FFmpeg C harness used `-O3 -DNDEBUG -march=native`.

The corpus contains 20 ten-second excerpts from two real music albums. One
album is 44.1 kHz stereo and the other is 192 kHz stereo. The 16-bit set keeps
the source precision. The 24-bit set applies a fixed float-domain gain before
ALAC encoding so its low precision bits are populated instead of containing an
upshifted 16-bit signal.

Each process performed three warm-up decodes. Each measured run decoded a file
20 times. Both decoders were pinned to one CPU and their order alternated for
each file. Container indexing and packet extraction happened before the timed
region. Both implementations decoded identical access units into their normal
PCM output contract.

| Depth | Decoder | Files | Interleaved samples | Total time | Time/sample |
| --- | --- | ---: | ---: | ---: | ---: |
| 16-bit | SoundKit Rust | 20 | 944,400,000 | **19.280441 s** | **20.415545 ns** |
| 16-bit | FFmpeg C | 20 | 944,400,000 | 22.291951 s | 23.604353 ns |
| 24-bit | SoundKit Rust | 20 | 944,400,000 | **21.159028 s** | **22.404731 ns** |
| 24-bit | FFmpeg C | 20 | 944,400,000 | 23.930285 s | 25.339142 ns |

At 16-bit, SoundKit used 13.51% less elapsed time and delivered 15.62% more
throughput than FFmpeg C. At 24-bit, it used 11.58% less elapsed time and
delivered 13.10% more throughput. SoundKit finished first on all 40 per-file
comparisons.

Every file was also decoded once by both implementations and compared as raw
PCM. All 20 16-bit files and all 20 24-bit files were byte-for-byte identical.

Run the crate tests with:

```sh
cargo test -p soundkit-alac --all-targets
```

## Performance Gate

Run the ALAC gate on the SoundKit CPU performance host. It builds the Rust and
FFmpeg harnesses with native CPU tuning, confirms byte-identical PCM first,
warms both decoders, alternates the decoder order for every file and round,
then compares the corpus median time per sample.

```sh
python3 tools/run_decode_bench.py --build --cpu 0 \
  --corpus /path/to/alac-corpus --rounds 5 --iterations 20 \
  --json performance-results/alac-decode.json
```

The default gate rejects a SoundKit corpus result slower than FFmpeg. Use
`--max-rust-slower-percent` only when intentionally changing the performance
contract. The JSON result records `lscpu`, Rust, Cargo, FFmpeg, and FLAC
versions together with the paired timings and PCM hashes.

## Provenance

SoundKit bootstrapped the scalar packet decoder from the permissively licensed
`alac` crate version 0.5.0 by Edward Barnard. SoundKit now maintains the code
inside `src/decoder`. It added the 64-bit buffered Rice reader, direct unary
decoding, optimized adaptive LPC, reusable PCM output, bounded validation, and
the owned M4A/CAF packet integration.

See [THIRD_PARTY.md](THIRD_PARTY.md),
[LICENSE-ALAC-RS-MIT](LICENSE-ALAC-RS-MIT), and
[LICENSE-ALAC-RS-APACHE](LICENSE-ALAC-RS-APACHE) for retained notices.
