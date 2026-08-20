# soundkit

[![CI](https://github.com/wavey-ai/soundkit/actions/workflows/ci.yml/badge.svg)](https://github.com/wavey-ai/soundkit/actions/workflows/ci.yml)

Rust media tooling for deterministic audio extraction, video decoding, PCM conversion,
resampling, codec wrappers, and browser-safe streaming.

## At A Glance

| Area | Crates / APIs | Notes |
| --- | --- | --- |
| PCM utilities | `soundkit::audio_bytes`, `soundkit::raw_pcm` | Sample-width conversion, endian conversion, interleave/deinterleave, headerless PCM streams. |
| WAV / RF64 | `soundkit::wav` | Incremental PCM parser and bounded `WavStreamEncoder`; `generate_wav_buffer` remains a convenience wrapper. |
| Resampling | `soundkit::downsample_audio` | `rubato` sinc resampling. |
| Codecs | `soundkit-*` codec crates | Small wrappers around native Rust decoders where available, with C FFI only where useful or required. |
| Decode pipeline | `soundkit-decoder` | Ring-buffered worker thread, `access-unit` autodetection, explicit telephony paths, optional output conversion. |
| Media demux | `soundkit-audio-demux`, `soundkit-webm` | Rust-owned MOV, MP4, fragmented MP4, WebM, Matroska, MPEG-TS, and MXF parsing. |
| Video decode | `soundkit-video`, `soundkit-dnx` | Pure-Rust H.264, HEVC, VP9, AV1, ProRes, DNxHD, and DNxHR decoding. |
| WASM | `soundkit-wasm` | Seekable browser media adapters and deterministic Rust audio/video decode. |

## Platform Integration Policy

SoundKit complements platform media APIs rather than replacing device I/O or
ordinary presentation playback. Rust owns every operation that must be bounded,
portable, byte-addressable, or deterministic. Apple frameworks and Web APIs own
the device and presentation services they are designed to provide.

| Operation | Apple platforms | Browser / WebCodecs | SoundKit responsibility |
| --- | --- | --- | --- |
| Audio session, routes, capture, and device output | `AVAudioSession`, Audio Unit, or `AVAudioEngine` | Web Audio and `AudioWorklet` | Codec packets, deterministic DSP, and stored stream formats. |
| Ordinary original-file playback | `AVPlayer` is appropriate | An HTML media element is appropriate | Optional inspection and compatibility fallback. This path does not define canonical media. |
| File and container inspection | SoundKit first | SoundKit first in a worker | Detect and validate bytes. Never trust a suffix or MIME type as the parser contract. |
| Container demux and sample indexing | SoundKit | SoundKit | Own MOV/MP4, fragmented MP4, WebM/Matroska, MPEG-TS/M2TS, Ogg, CAF, and MXF ranges, timestamps, edit lists, and packet limits. |
| Codec decode supported by SoundKit | SoundKit first | SoundKit first | Produce the same bounded decoded-frame contract across native and WASM targets. |
| Unsupported codec profile | `AVAssetReader` or `AVAudioConverter` fallback | `AudioDecoder` fallback after `isConfigSupported()` | Demux first, supply access units, validate output, and consume fallback PCM immediately in bounded blocks. |
| Canonical normalization and resampling | SoundKit | SoundKit | Keep channel mapping, sample rate, priming, finite-sample handling, and hashes consistent across platforms. |
| Application Opus/FLAC creation | SoundKit only | SoundKit only | Frame, encode, hash, and build byte indexes with one cross-platform contract. |
| Random-access editing or cached playback | SoundKit indexed stream | SoundKit indexed stream | Resolve byte ranges and decode only the requested packets. Do not create a persistent PCM working copy. |
| Video presentation decode | VideoToolbox may be selected | WebCodecs may be selected | Retain container/index ownership and provide Rust fallback for supported codecs and profiles. |

The selection order for canonical imports is:

1. SoundKit detects and demuxes the source.
2. A SoundKit decoder handles the codec when supported.
3. A platform decoder handles only a codec profile SoundKit explicitly rejects.
4. The adapter returns bounded PCM blocks directly to Rust.
5. SoundKit normalizes, hashes, encodes, indexes, and stores the result.

This ordering avoids two incompatible canonical pipelines. Platform decoders can
change priming, channel mapping, sample conversion, or supported profiles between
OS and browser releases. Those differences are acceptable for fallback playback,
but they must not silently change stored packet bytes or cache identities.

`AVAssetReader` and `AVAudioConverter` provide the sequential and pull-based
decode APIs needed for a bounded fallback. WebCodecs `AudioDecoder` provides an
asynchronous access-unit API with a decode queue and flush operation. WebCodecs
does not demux containers, generate byte-range indexes, or frame SoundKit streams.
Its availability and codec profiles must be checked at runtime. Whole-file browser
APIs such as `decodeAudioData()` are not a canonical SoundKit import path.

References: [AVAssetReader](https://developer.apple.com/documentation/avfoundation/avassetreader),
[AVAudioConverter](https://developer.apple.com/documentation/avfaudio/avaudioconverter),
and [WebKit WebCodecs audio support](https://webkit.org/blog/16993/news-from-wwdc25-web-technology-coming-this-fall-in-safari-26-beta/).

## Streaming Decode Matrix

`Stream output` means the decoder can emit PCM before EOF from chunked input.
`EOF` means the wrapper accepts chunks but currently buffers the full container
and emits only after an empty EOF chunk. `Limited` means chunked files work, but
the container layout can require enough metadata/media to be buffered first.

| Format | Package / backend | Pipeline path | Stream output | Notes |
| --- | --- | --- | --- | --- |
| Raw PCM (`linear16`, `linear32`, `s16le`, `f32le`, `L16`) | `soundkit::raw_pcm` | Explicit | Yes | Caller supplies sample rate, channels, sample format. |
| WAV / RIFF PCM | `soundkit::wav` | Auto | Yes | Emits complete PCM frame runs after the `data` chunk starts. |
| MP3 | `soundkit-mp3` / `nanomp3` | Auto | Yes | Pure Rust decode; native decoder output is `f32`. |
| AAC ADTS | `soundkit-aac` / `fdk-aac` | Auto | Yes | Frame-stream friendly C FFI path. Use `soundkit-aac-lc` for the controlled pure Rust stereo profile. |
| AAC in M4A/MP4 | `soundkit-aac` / Rust MP4 demux + `fdk-aac` | Auto or seekable MP4 index | Yes | Fast-start files stream sequentially. Tail-`moov` files use one bounded packet range at a time. |
| FLAC | `soundkit-flac` / `claxon` | Auto | Yes | Pure Rust decode retains metadata state and only the current incomplete compressed frame. |
| Raw Opus stream | `soundkit-opus` / `libopus` | Auto | Yes | Soundkit `OpusHead` plus length-prefixed packets. |
| Ogg Opus | `soundkit-ogg-opus` / Ogg parser + `libopus` | Auto | Yes | Ogg pages parsed incrementally. |
| WebM Opus | `soundkit-webm` / EBML parser + `libopus` | Auto | Yes | Known and unknown-size clusters emit bounded blocks before cluster EOF. |
| Ogg Speex | `soundkit-speex` / `oxideav-speex` | Explicit | Yes | Pure Rust codec core and streaming Ogg packet parser. |
| Ogg Vorbis | `soundkit-vorbis` / `lewton` | Auto or explicit | Yes | Pure Rust decode and streaming Ogg packet parser. |
| ALAC in M4A/MP4 | `soundkit-alac` / `alac` | Seekable MP4 index | Yes | Rust reads `moov`, then decodes one ranged ALAC packet at a time. |
| ALAC in CAF | `soundkit-alac` / `alac` | Seekable CAF index | Yes | Rust scans bounded metadata, skips `data`, then decodes one ranged packet at a time. |
| AIFF / AIFF-C | `soundkit-aiff` | Auto or explicit | Yes | Incremental Rust FORM parser supports integer PCM, float PCM, A-law, u-law, and IMA4. |
| Raw AC-3 syncframes | `soundkit-ac3` / `oxideav-ac3` | Auto or explicit | Yes | Raw elementary AC-3 stream, not containerized AC-3. |
| AMR-NB | `soundkit-amr` / OpenCORE AMR-NB | Explicit | Yes | 3GPP `.amr` magic and raw frame streams; C FFI backend. |
| G.711 u-law / A-law | `soundkit-g711` | Explicit | Yes | Pure Rust PCMU/PCMA decode. |
| G.722 | `soundkit-g722` / `ezk-g722` | Explicit | Yes | Pure Rust 64 kbit/s wideband speech decode. |
| G.726 | `soundkit-g726` | Explicit | Yes | Pure Rust 16/24/32/40 kbit/s profiles. |
| G.729 | `soundkit-g729` / `g729-sys` | Explicit | Yes | Frame-buffered 8 kbit/s speech decode. |
| GSM 06.10 / WAV-49 | `soundkit-gsm` / `libgsm` | Explicit | Yes | Standard raw GSM and Microsoft WAV-49 packet framing. |

## Pure Rust Decode Boundary

For native builds, SoundKit can mix Rust wrappers and C-backed codec libraries.
For WASM, Cloudflare Workers, and other Rust-only targets, the codec decode
boundary is narrower:

| Format / area | Current decode path | Pure Rust decode? | Notes |
| --- | --- | --- | --- |
| AAC ADTS | `soundkit-aac` / `fdk-aac` | No | Frame streaming is supported, but the production AAC codec decode is FDK-AAC C FFI. |
| AAC-LC raw access units | `soundkit-aac-lc` + `soundkit-wasm` | Controlled production profile | Pure Rust decoding supports stereo AAC-LC at 44.1 and 48 kHz. Other profiles return explicit fallback errors. See [`AAC_LC_PRODUCTION_STATUS.md`](AAC_LC_PRODUCTION_STATUS.md). |
| AAC in M4A/MP4 | `mp4` demux + `fdk-aac` | No | MP4 demux/debox can be Rust, but production AAC frame decode still uses FDK-AAC. |
| AMR-NB | OpenCORE AMR-NB | No | Requires the native `opencore-amrnb` library via `pkg-config`. |
| G.729 | `g729-sys` | No | Uses a native codec binding. |
| GSM 06.10 / WAV-49 | `gsm-sys` / `libgsm` | No | Uses the native libgsm codec. |
| Opus / Ogg Opus / WebM Opus | `soundkit-opus` pure Rust backend | Partial | Supported for the current packet path, but FEC is not implemented and the backend is not full libopus parity yet. |
| FLAC | `claxon` in `soundkit-decoder` | Yes | The aggregate decoder selects pure Rust `claxon`; the standalone `soundkit-flac` crate defaults to libFLAC unless `claxon-decoder` is selected. |
| H.264, HEVC, VP9, AV1, ProRes | `soundkit-video` | Yes | Rust produces bounded planar frames on native and WASM targets. |
| DNxHD and DNxHR | `soundkit-dnx` | Yes | Rust supports progressive DNxHD and current DNxHR delivery profiles. |

Everything else in the decode matrix is currently on a pure-Rust decode path.
MP3 decode uses `nanomp3`. MP3 encode is the part that pulls in LAME.
Rubber Band is also a native dependency, but it is a time-stretch/resampling
tool rather than a codec decoder.

## Native Media Pipeline

SoundKit can extract audio and video without browser or device media decoders.
Integrations may use a platform fallback under the policy above. Rust still owns
container validation, timestamps, codec normalization, and decoded-frame validation.

Browser `File` and `Blob` sources use seekable byte ranges for MOV, MP4, M4A, and CAF.
Large `mdat` and CAF `data` extents never cross the WASM boundary as complete buffers.

Fragmented MP4, WebM, Matroska, MPEG-TS, and MXF consume bounded sequential chunks.
Malformed lengths and oversized metadata, packets, frames, or input chunks fail before allocation.

WAV, AIFF, FLAC, Ogg, MP3, AAC, AC-3, Opus, raw PCM, and MPEG-TS use bounded incremental parsers.
Automatic detection retains at most 64 KiB. Browser push calls accept at most 4 MiB per chunk.

The current artist-delivery matrix covers H.264, HEVC, VP9, AV1, ProRes, DNxHD, and DNxHR.
See [`docs/media-pipeline.md`](docs/media-pipeline.md) for exact container, profile, audio, and conformance coverage.

## Decode Pipeline APIs

| Need | API |
| --- | --- |
| Autodetect common media files | `DecodePipeline::spawn()` |
| Override output rate, depth, or channels | `DecodePipeline::spawn_with_options(options)` |
| Headerless PCM | `DecodePipeline::spawn_raw_pcm(format)` |
| Telephony and speech codecs | `spawn_g711`, `spawn_g722`, `spawn_g726_with_rate`, `spawn_g729`, `spawn_gsm`, `spawn_amr_nb`, `spawn_speex` |
| Consumer containers with explicit format | `spawn_vorbis`, `spawn_alac`, `spawn_aiff`, `spawn_ac3` |

```rust
use soundkit_decoder::{Bytes, DecodePipeline};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut pipeline = DecodePipeline::spawn();
    pipeline.send(Bytes::from(std::fs::read("audio.ogg")?))?;
    pipeline.send(Bytes::new())?; // EOF / flush

    while let Some(frame) = pipeline.try_recv() {
        let audio = frame?;
        println!(
            "{} Hz, {} ch, {} bits",
            audio.sampling_rate(),
            audio.channel_count(),
            audio.bits_per_sample()
        );
    }

    Ok(())
}
```

## Encode Support

| Format | Encoder | Streaming-friendly | Notes |
| --- | --- | --- | --- |
| Raw PCM | Core byte helpers | Yes | Headerless PCM is just framed bytes. |
| WAV / RF64 | `WavStreamEncoder` / `WasmWavEncoder` | Yes | Emits an exact header, then bounded interleaved PCM chunks. Uses RF64 beyond 4 GiB. |
| MP3 | `mp3lame` | Yes | Feature-gated encoder path. |
| AAC ADTS | `fdk-aac` | Yes | ADTS output. |
| FLAC | `flacenc` | Yes | The default encoder is pure Rust. Enable `oxideav-encoder` only for compatibility tests. |
| Opus | `libopus` | Yes | Packet encoder. |
| AMR-NB | OpenCORE AMR-NB | Yes | 160-sample speech frames. |
| G.711 / G.722 / G.726 / G.729 / GSM | Codec crates | Yes | Frame or sample streaming, depending on codec. |
| Vorbis / Speex / ALAC / AIFF / AC-3 / WebM | Decode-only today | No | Add only when fixture generation and licensing are clear. |

## Test Fixture Rule

| Requirement | Current pattern |
| --- | --- |
| Codec fixture | Generate with FFmpeg into `testdata/<format>/...` when FFmpeg can encode it. |
| Golden output | Decode with soundkit and write WAV under `golden/<format>/...`. |
| Decoder tests | Compare chunked-vs-whole decode and run pipeline explicit/autodetect tests where available. |
| External comparison | Compare native PCM with FFmpeg PCM where practical. |
| Manual playback | Play decoded golden WAVs with `ffplay` after implementation. |

## License Notes

| Dependency family | Distribution note |
| --- | --- |
| Pure Rust codec crates (`lewton`, `alac`, `aifc`, `oxideav-*`, `ezk-g722`) | Mostly permissive; keep crate license notices in packaged distributions. |
| `mp4parse` on the ALAC M4A path | MPL-2.0 dependency. |
| `libopus`, `libFLAC`, `mp3lame`, `fdk-aac`, OpenCORE AMR-NB, `libgsm`, Rubber Band | C/C++ library dependencies; ship notices and review binary distribution requirements. |
| `libgsm` | Preserve the upstream notice in source and binary distributions. |

## Pending Formats

| Format | Status |
| --- | --- |
| AMR-WB | Pending a fixture-safe encoder path. |
| Monkey's Audio / APE | Deferred because the local FFmpeg build can decode APE but cannot encode fixtures. |
