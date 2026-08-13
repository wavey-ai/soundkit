# SoundKit media pipeline

SoundKit owns deterministic audio extraction and video decoding for native and WASM clients. Platform code transports bytes and presents decoded frames. It does not validate or decode media.

## Pipeline

1. The container demuxer identifies tracks from container metadata.
2. The audio path emits typed AAC, Opus, Vorbis, FLAC, ALAC, MP3, AC-3, or PCM packets.
3. The video path emits codec access units with timestamps and sync metadata.
4. Pure-Rust decoders validate dimensions and decode into bounded planar frames.
5. WASM exports copy validated Rust frames across the browser boundary.

`Mp4MediaIndex` parses only `moov` and returns validated absolute sample ranges. This matters for camera and NLE files that store a large `mdat` before metadata. Browser code reads the ranges from the `File`; it does not parse or validate MP4. Contiguous QuickTime PCM samples are grouped into bounded 4,096-frame packets.

`WebmMediaDemuxer` streams all supported WebM video and audio tracks. It resolves cluster-relative timestamps and durations to nanoseconds, preserves keyframe state, and timestamps laced frames independently.

The decoded video contract is planar YUV with explicit dimensions, stride, bit depth, chroma sampling, alpha presence, presentation timestamp, and duration. Planes use Y, Cb, Cr, and optional alpha order. Eight-bit samples use one byte. Deeper samples use little-endian 16-bit words.

## Compatibility baseline

The deterministic `never-final.mov` matrix covers common artist delivery and upload formats.

| Container | Video | Audio | Native/WASM status |
| --- | --- | --- | --- |
| MP4 | H.264 High 8-bit 4:2:0 | AAC-LC 48 kHz stereo | Passing |
| MP4 | H.264 High 8-bit 4:2:0 | FLAC 24-bit 48 kHz stereo | Passing |
| MOV | HEVC Main 8-bit 4:2:0 | AAC-LC 48 kHz stereo | Passing |
| MOV | HEVC Main10 10-bit 4:2:0 | PCM 24-bit 48 kHz stereo | Passing |
| MOV | ProRes 422 HQ 10-bit 4:2:2 | PCM 24-bit 48 kHz stereo | Passing |
| MOV | ProRes 4444 12-bit 4:4:4 + alpha | PCM 24-bit 48 kHz stereo | Passing |
| MOV | DNxHR HQX 10-bit 4:2:2 | PCM 24-bit 48 kHz stereo | Audio passing; video decoder pending |
| WebM | VP9 Profile 0 8-bit 4:2:0 | Opus 48 kHz stereo | Passing |
| WebM | AV1 Main 8-bit 4:2:0 | Opus 48 kHz stereo | Passing |
| IVF | AV1 Main 10-bit 4:4:4 | None | Passing |
| IVF | AV1 Main 10-bit monochrome | None | Passing |
| Annex B | HEVC Main10 HDR10 4:2:0 | None | Passing |

FLAC decode and streaming encode are required pipeline capabilities. MP4 `dfLa` metadata is normalized into a decoder-ready FLAC stream in Rust. Streaming encoders must call `finish()` once, then use the final `streamHeader()` metadata.

## Reproduce

Generate the ignored local corpus from a source music video:

```sh
make media-fixtures SOURCE_MEDIA=/absolute/path/to/never-final.mov
```

Build optimized WASM and decode both video and audio from each complete container:

```sh
make media-conformance
```

The repository stores nine deterministic three-second container fixtures under `testdata/video-compat/never-final`. It does not store the artist source. The generator recreates the fixtures under `build/`, and `media-conformance` verifies the committed SHA-256 manifest before decoding.

Fetch the pinned Chromium corpus and verify its SHA-256 values:

```sh
make media-upstream-corpus
```

Run the upstream corpus through the optimized release WASM artifact:

```sh
make media-upstream-conformance
```

Run deterministic codec and container mutations in isolated processes:

```sh
make media-fuzz
```

## Safety and ownership

- Rust rejects zero-sized, overflowing, or larger-than-8K frame declarations.
- Malformed codec input returns a bounded result or typed error.
- The container demuxer reports PCM depth, endianness, and integer/float representation.
- JavaScript performs no media validation. It only feeds bytes and consumes exported Rust values.
- A dependency-specific release profile keeps `vp9dec` at optimization level 2 because LLVM 21 crashes at level 3 on `wasm32`.
- The vendored `rusty_av1d` patch fixes high-bit-depth plane access and one malformed-input cleanup panic.

## Remaining format work

DNxHD/DNxHR needs a production-quality pure-Rust decoder. The current API rejects it explicitly instead of silently invoking a device decoder. Additional corpus work should cover H.264 4:2:2/4:4:4, HEVC 4:2:2, variable frame rates, edit lists, fragmented MP4, and broader Matroska variants.
