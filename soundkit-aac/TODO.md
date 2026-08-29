# soundkit-aac TODO

## Decoder comparison benchmark: soundkit-aac-lc vs rusty_aac

### Status: benchmark binary written, pending perf-host run

### Binary
`soundskit-aac-lc/examples/decode_compare.rs` — standalone comparison that:
- Parses both ADTS fixtures (Westside 256k 48k, stereo-music 192k 44.1k)
- Decodes with both decoders, reports wall time, RTF, output checksum
- Warmup runs before measured runs

### Preliminary findings (macOS ARM scalar, release mode)

| Decoder | WESTSIDE 256k (9171 frames, 195.6s audio) |
| --- | --- |
| soundkit-aac-lc | 120.7 ms (RTF 0.00062) |
| rusty_aac (scalar, no SIMD) | ~240 s (RTF ~1.23) — **~2000x slower** |

The scalar `rusty_aac` path is not competitive on this platform. With AVX2 on
the Intel perf host, the crate claims ~450x realtime encode and fast decode.
The SIMD path (`simd` feature, default) is x86-only.

### Plan
1. Run `decode_compare` on `yl-encodec-1` (Intel Xeon, AVX2 available)
2. Measure rusty_aac with SIMD enabled vs soundkit-aac-lc
3. Both decoders produce valid PCM — compare checksums for output equivalence
4. If rusty_aac decode is competitive, evaluate the encoder for porting/forking

---

## Encoder approach: research notes

### Options considered
1. **Port rusty_aac encoder to SoundKit-owned crate** — full rewrite, not vendoring
2. **Clean-room from ISO 14496-3 specs** — we already have the decoder primitives
3. **Fork+improve rusty_aac** — not desired, we want clean ownership

### rusty_aac encoder capabilities
- Psychoacoustic Bark-scale masking model
- Two-phase bitrate rate loop
- Transient-driven block switching
- Per-SFB M/S stereo
- Frame-parallel `encode_stream` (~450x realtime)
- ADTS, MP4, LATM-LOAS output
- 5.1 layouts, HE-AAC signalling
- Zero dependencies (Apache-2.0)

### FFmpeg 9.1 NMR encoder (June 2026, by Lynne)
- Beats FDK-AAC on Zimtohrli/ViSQOL
- Trellis RDO with masked band energy
- Not yet released in stable FFmpeg
- Reference: https://code.ffmpeg.org/FFmpeg/FFmpeg/pulls/23430

### AAC encoder gates
- Must be faster than FFmpeg C encoder
- Must beat FFmpeg 9.1 NMR encoder on perceived quality (ViSQOL/PEAQ)

---

## rusty_aac provenance

`rusty_aac` v0.5.0 on crates.io is extracted from the `remade_ffmpeg_rs`
monorepo: https://github.com/Remade-With-Rust/remade_ffmpeg_rs

Same repo, same author. The crate is the standalone AAC component of that
project.

---

## Completed work

- [x] soundkit-aac-lc decoder: faster than FFmpeg C on all 5 music fixtures (see BENCHMARK_NATIVE_2026-08-25.md)
- [x] ALAC in-tree decoder committed as `13fb090` — 13 tests, 40 corpus byte-identical
- [x] Full codec dependency audit (10 in-tree, 2 Rust-wrapping, 3 C FFI)
- [x] WASM pipeline audit: streaming API production-ready
- [x] Renamed `WasmMusicDecoder` -> `Decoder` with `PacketDecoder` trait alias
- [x] Container routing: MKV/MPEG-TS/fMP4 through `ContainerAudioDecoder`
- [x] MKV AAC: audio-specific config from Matroska, ADTS header injection
- [x] MKV test asserts real decoded PCM data
- [x] AAC encoder research (rusty_aac, FFmpeg 9.1 NMR, Symphonia)
- [x] Copy `access-unit` into workspace, add container detection variants
