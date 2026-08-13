# SoundKit media pipeline

SoundKit owns deterministic audio extraction and video decoding for native and WASM clients. Platform code transports bytes and presents decoded frames. It does not validate or decode media.

## Pipeline

1. The container demuxer identifies tracks from container metadata.
2. The audio path emits typed AAC, Opus, Vorbis, FLAC, ALAC, MP3, AC-3, or PCM packets.
3. The video path emits codec access units with timestamps and sync metadata.
4. Pure-Rust decoders validate dimensions and decode into bounded planar frames.
5. WASM exports copy validated Rust frames across the browser boundary.

`Mp4MediaIndex` parses only `moov` and returns validated absolute sample ranges. Browser code reads only Rust-requested ranges from the `File`.

The browser reads 16-byte top-level headers. Rust validates each box and skips a large `mdat` without reading its payload.

This path supports NLE files with metadata after multi-gigabyte media data. Contiguous QuickTime PCM samples use bounded 4,096-frame packets.

`Mp4MediaDemuxer` incrementally parses fragmented MP4 and CMAF. It releases each complete sample before the current `mdat` reaches EOF.

The demuxer retains only incomplete metadata, one incomplete sample, and pending sample records. It applies Rust timeline and NAL normalization rules.

Sequential regular MP4 accepts `moov` before `mdat`. A tail-`moov` file must use `Mp4MediaIndex` and seekable ranges.

Unknown top-level boxes are skipped incrementally. Rust limits metadata, input chunks, and individual compressed packets before allocation.

MP4 edit lists are parsed and normalized in Rust. Each track exposes a linear timeline in its own timescale. Packet presentation timestamps include the edit, and `pcm_packet_trim` removes AAC preroll and tail padding exactly. Platform adapters only apply the returned source-frame slice.

`WebmMediaDemuxer` streams all supported WebM video and audio tracks. It resolves cluster-relative timestamps and durations to nanoseconds, preserves keyframe state, and timestamps laced frames independently.

Known-size clusters emit blocks before cluster EOF. Rust bounds metadata elements, packet elements, caller chunks, and the parser buffer.

`MxfMediaDemuxer` incrementally parses bounded KLV records, header metadata, frame-wrapped Generic Container essence, DNx coding-unit headers, and BWF/AES3 PCM in Rust. The browser only provides byte chunks. The first production profile is OP1a DNxHD/DNxHR picture essence with 16- or 24-bit PCM sound essence.

The decoded video contract is planar YUV with explicit dimensions, stride, bit depth, chroma sampling, alpha presence, presentation timestamp, and duration. Planes use Y, Cb, Cr, and optional alpha order. Eight-bit samples use one byte. Deeper samples use little-endian 16-bit words.

## Compatibility baseline

The deterministic `never-final.mov` matrix covers common artist delivery and upload formats.

| Container | Video | Audio | Native/WASM status |
| --- | --- | --- | --- |
| MP4 | H.264 High 8-bit 4:2:0 | AAC-LC 48 kHz stereo | Passing |
| Fragmented MP4/CMAF/DASH | H.264 High 8-bit 4:2:0 | AAC-LC 48 kHz stereo | Passing across default-base, explicit-base, and separate-moof layouts |
| MP4 VFR | H.264 High 8-bit 4:2:0 | AAC-LC 48 kHz stereo | Passing, including edit timeline |
| MP4 | H.264 High 8-bit 4:2:0 | FLAC 24-bit 48 kHz stereo | Passing |
| MP4 | H.264 High 4:2:2 8-bit | AAC-LC 48 kHz stereo | Demux/audio passing; native video profile pending |
| MP4 | H.264 High 4:4:4 8-bit | AAC-LC 48 kHz stereo | Demux/audio passing; native video profile pending |
| MOV | HEVC Main 8-bit 4:2:0 | AAC-LC 48 kHz stereo | Passing |
| MOV | HEVC Main10 10-bit 4:2:0 | PCM 24-bit 48 kHz stereo | Passing |
| MOV | HEVC Main 4:2:2 10-bit | AAC-LC 48 kHz stereo | Demux/audio passing; native video profile pending |
| MOV | HEVC Main 4:4:4 10-bit | AAC-LC 48 kHz stereo | Demux/audio passing; native video profile pending |
| MOV | ProRes Proxy 10-bit 4:2:2 | PCM 24-bit 48 kHz stereo | Passing |
| MOV | ProRes LT 10-bit 4:2:2 | PCM 24-bit 48 kHz stereo | Passing |
| MOV | ProRes 422 10-bit 4:2:2 | PCM 24-bit 48 kHz stereo | Passing |
| MOV | ProRes 422 HQ 10-bit 4:2:2 | PCM 24-bit 48 kHz stereo | Passing |
| MOV | ProRes 4444 12-bit 4:4:4 + alpha | PCM 24-bit 48 kHz stereo | Passing |
| MOV | ProRes 4444 XQ 12-bit 4:4:4 + alpha | PCM 24-bit 48 kHz stereo | Passing |
| MOV | DNxHR HQX 10-bit 4:2:2 | PCM 24-bit 48 kHz stereo | Passing |
| MOV | DNxHR HQ/SQ/LB 8-bit 4:2:2 | PCM 24-bit 48 kHz stereo | Passing for every profile |
| MOV | DNxHR 444 10-bit GBR and YCbCr | PCM 24-bit 48 kHz stereo | Passing for both color models |
| MOV | DNxHD 1080p 36/120/185 8-bit 4:2:2 | PCM 24-bit 48 kHz stereo | Passing |
| MOV | DNxHD 1080p 185 10-bit 4:2:2 | PCM 24-bit 48 kHz stereo | Passing |
| MXF OP1a | DNxHR HQX 10-bit 4:2:2 | PCM 24-bit 48 kHz stereo | Passing |
| WebM | VP9 Profile 0 8-bit 4:2:0 | Opus 48 kHz stereo | Passing |
| WebM | VP9 Profile 2 10-bit 4:2:0 | Opus 48 kHz stereo | Passing |
| WebM | AV1 Main 8-bit 4:2:0 | Opus 48 kHz stereo | Passing |
| WebM | AV1 Main 10-bit 4:2:0 | Opus 48 kHz stereo | Passing |
| Matroska | H.264 High 8-bit 4:2:0 | AAC-LC 48 kHz stereo | Passing |
| Matroska | HEVC Main 8-bit 4:2:0 | AAC-LC 48 kHz stereo | Passing |
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

The repository stores 36 deterministic container fixtures under `testdata/video-compat/never-final`. It does not store the artist source. The generator recreates the fixtures under `build/`, and `media-conformance` verifies the committed SHA-256 manifest before decoding. Most fixtures are three seconds. The DNxHR 444 and legacy DNxHD fixtures are shorter to limit repository size.

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
- Seekable MOV and MP4 imports never copy the complete source into JavaScript or WASM memory.
- Seekable M4A and CAF ALAC imports read one Rust-indexed packet range at a time.
- Sequential demuxers reject oversized caller chunks, metadata elements, and compressed packets before allocation.
- WebM and fragmented MP4 emit complete packets before their enclosing media extent ends.
- Audio autodetection retains at most 64 KiB and forwards later bytes to the selected decoder.
- WAV, AIFF, FLAC, Ogg, MP3, AAC, AC-3, Opus, raw PCM, and MPEG-TS release consumed input incrementally.
- WAV output emits an exact RIFF or RF64 header followed by bounded PCM chunks.
- Browser push calls reject chunks larger than 4 MiB so JavaScript cannot force an unbounded WASM copy.
- A dependency-specific release profile keeps `vp9dec` at optimization level 2 because LLVM 21 crashes at level 3 on `wasm32`.
- The vendored `rusty_av1d` patch fixes high-bit-depth plane access and one malformed-input cleanup panic.
- The memory-safe Rust DNx decoder is isolated in `soundkit-dnx` under LGPL-2.1-or-later. Its scalar output matches pinned FFmpeg output across DNxHR and progressive 1080p DNxHD profiles.

## Remaining format work

DNxHR 444 12-bit and DNxHD interlaced profiles remain. H.264 and HEVC 4:2:2/4:4:4 need decoder extensions; their container and audio paths already pass. The current API rejects these gaps explicitly instead of silently invoking a device decoder. Additional corpus work should cover OP-Atom, clip-wrapped MXF, and broader Matroska variants.
