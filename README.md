# soundkit

[![CI](https://github.com/wavey-ai/soundkit/actions/workflows/ci.yml/badge.svg)](https://github.com/wavey-ai/soundkit/actions/workflows/ci.yml)

Rust media tooling for deterministic audio extraction, video decoding, PCM conversion,
resampling, authored codec implementations, and browser-safe streaming.

## At A Glance

| Area | Crates / APIs | Notes |
| --- | --- | --- |
| PCM utilities | `soundkit::audio_bytes`, `soundkit::raw_pcm` | Sample-width conversion, endian conversion, interleave/deinterleave, headerless PCM streams. |
| WAV / RF64 | `soundkit::wav` | Incremental PCM parser and bounded `WavStreamEncoder`; `generate_wav_buffer` remains a convenience wrapper. |
| Resampling | `soundkit::downsample_audio` | `rubato` sinc resampling. |
| Codecs | `soundkit-*` codec crates | SoundKit-authored codec cores first, with explicit native fallbacks only for profiles not yet owned. |
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
| Application media creation | SoundKit only | SoundKit only | Opus and FLAC are writable today. Owned AAC writing and fragmented-MP4 boxing for SoundKit LL-HLS are the final planned encoding phase. |
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
| MP3 | `soundkit-mp3` | Auto | Yes | SoundKit-owned Layer III decoder; bounded Rust core with runtime AVX2/SSE2 synthesis on x86-64. |
| AAC ADTS | `soundkit-aac` / owned AAC-LC + FDK fallback | Auto | Yes | Supported mono/stereo AAC-LC uses SoundKit's owned decoder by default; unsupported profiles and tools route to FDK on native builds. |
| AAC-LC in M4A/MP4/MOV or Matroska | `soundkit-aac` + owned Rust demuxers | Streaming or seekable index | Limited | Browser AAC-LC uses the production owned decoder. Native builds retain FDK fallback for HE-AAC/SBR/PS and unsupported profiles. |
| MP2 in MPEG-TS | `soundkit-decoder` / Rust TS demux + Wavey Symphonia fork | Complete-file | EOF | Pure Rust; collected TS fixture measures 50.36 dB against FFmpeg. |
| FLAC | `soundkit-flac` / `wavey-flac` | Auto | Yes | Unified pure-Rust encode/decode retains metadata state and only the current incomplete compressed frame. |
| Raw Opus stream | `soundkit-opus` / Rust Opus codecs | Auto | Yes | SoundKit `OpusHead` plus length-prefixed packets. |
| Ogg Opus | `soundkit-ogg-opus` / Ogg parser + Rust Opus decoder | Auto | Yes | Ogg pages parsed incrementally. |
| WebM Opus | `soundkit-webm` / EBML parser + Rust Opus decoder | Auto | Yes | Known and unknown-size clusters emit bounded blocks before cluster EOF. |
| Ogg Speex | `soundkit-speex` / `oxideav-speex` | Explicit | Yes | Pure Rust codec core and streaming Ogg packet parser. |
| Ogg Vorbis | `soundkit-vorbis` owned core | Auto or explicit | Yes | In-tree Rust decoder and streaming Ogg packet parser. |
| ALAC in M4A/MP4 | `soundkit-alac` owned core | Seekable MP4 index | Yes | Rust reads `moov`, then decodes one ranged ALAC packet at a time. |
| ALAC in CAF | `soundkit-alac` owned core | Seekable CAF index | Yes | Rust scans bounded metadata, skips `data`, then decodes one ranged packet at a time. |
| AIFF / AIFF-C | `soundkit-aiff` | Auto or explicit | Yes | Incremental Rust FORM parser supports integer PCM, float PCM, A-law, u-law, and IMA4. |
| Raw AC-3 syncframes | `soundkit-ac3` / `oxideav-ac3` | Auto or explicit | Yes | Raw elementary AC-3 stream, not containerized AC-3. |
| AMR-NB | `soundkit-amr` / OpenCORE AMR-NB | Explicit | Yes | 3GPP `.amr` magic and raw frame streams; C FFI backend. |
| G.711 u-law / A-law | `soundkit-g711` | Explicit | Yes | Pure Rust PCMU/PCMA decode. |
| G.722 | `soundkit-g722` | Explicit | Yes | SoundKit-authored, allocation-free 64 kbit/s encode/decode; bit-exact with the FFmpeg fixtures. |
| G.726 | `soundkit-g726` | Explicit | Yes | Pure Rust 16/24/32/40 kbit/s profiles. |
| G.729 | `soundkit-g729` / `g729-sys` | Explicit | Yes | Frame-buffered 8 kbit/s speech decode. |
| GSM 06.10 / WAV-49 | `soundkit-gsm` / `libgsm` | Explicit | Yes | Standard raw GSM and Microsoft WAV-49 packet framing. |

## Pure Rust Decode Boundary

Native builds may retain explicit C-backed fallbacks for codec profiles the
authored cores do not yet support. For Rust-only targets, the decode boundary
is narrower:

The current consolidation pass is decoder-first for import formats. SoundKit
Opus and FLAC are writable today. After decoder consolidation, the final
encoding phase will add an owned AAC writer and fragmented-MP4 boxing for
SoundKit LL-HLS. Other import formats remain decode-only.

| Format / area | Current decode path | Pure Rust decode? | Notes |
| --- | --- | --- | --- |
| MP3 | `soundkit-mp3` owned decoder | Yes | No decoder package or FFI boundary. On the 60-file multi-album music corpus, SoundKit used 2.17% less time/sample than optimized minimp3 C and measured 145.567 dB differential SNR. |
| AAC ADTS | `soundkit-aac` / owned AAC-LC + optional FDK fallback | Yes for supported AAC-LC | The production API selects the owned decoder for mono/stereo AAC-LC; native FDK handles unsupported AAC. |
| AAC-LC raw access units | `soundkit-aac` + `soundkit-wasm` | Controlled production profile | `soundkit-aac-lc` is the internal owned engine behind the production facade. Pure Rust decoding supports 1,024-sample mono/stereo AAC-LC. Other profiles return explicit fallback errors. See [`AAC_LC_PRODUCTION_STATUS.md`](AAC_LC_PRODUCTION_STATUS.md). |
| AAC in MP4/MOV/Matroska | `soundkit-aac` facade + owned demuxers | AAC-LC only in Rust-only builds | AAC-LC uses the owned access-unit decoder. HE-AAC/SBR/PS requires native FDK or a platform fallback. |
| AMR-NB | OpenCORE AMR-NB | No | Requires the native `opencore-amrnb` library via `pkg-config`. |
| G.729 | `g729-sys` | No | Uses a native codec binding. |
| GSM 06.10 / WAV-49 | `gsm-sys` / `libgsm` | No | Uses the native libgsm codec. |
| Opus / Ogg Opus / WebM Opus | `soundkit-opus` Rust core | Partial | The in-tree decoder handles 48 kHz CELT. It rejects SILK, hybrid, FEC, and mode transitions. |
| FLAC | `wavey-flac` | Yes | SoundKit uses the standalone Wavey-owned pure-Rust codec for both encoding and decoding. |
| Vorbis | `soundkit-vorbis` owned decoder | Yes | No external codec package or FFI boundary. SoundKit beat libvorbis C by 8.65% in elapsed time per sample. |
| ALAC | `soundkit-alac` owned decoder | Yes | No external codec package or FFI boundary. Across the 20-file real-music corpus, SoundKit used 13.51% less elapsed time than FFmpeg C at 16-bit and 11.58% less at 24-bit; all PCM was byte-exact. |
| H.264, HEVC, VP9, AV1, ProRes | `soundkit-video` | Yes | Rust produces bounded planar frames on native and WASM targets. |
| DNxHD and DNxHR | `soundkit-dnx` | Yes | Rust supports progressive DNxHD and current DNxHR delivery profiles. |

Everything else in the decode matrix is currently on a pure-Rust decode path.
MP3 encoding, unlike decoding, still pulls in LAME when the `encode` feature is
enabled. Rubber Band is also a native dependency, but it is a
time-stretch/resampling tool rather than a codec decoder.

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

With the `soundkit-audio-demux` `decode-video` feature, MOV and MP4 sources
produce a sealed keyframe timeline. The timeline lists every sync sample with
its presentation time. Callers decode a capped subset to keep the work light.
`decode_mp4_keyframes_from_file` reads only the `moov` box and the selected
sample ranges, so multi-gigabyte sources do not enter memory whole.

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

Canonical application encoding currently uses SoundKit Opus and FLAC. Import
formats are decoder-only in the current consolidation pass. After the decoder
cores are consolidated, the final encoding phase will add SoundKit-owned AAC
writing and fragmented-MP4 boxing for SoundKit LL-HLS. Older utility encoder
surfaces remain listed for codebase accuracy; they are not targets for new
codec work or canonical SoundKit storage.

| Format | Encoder | Streaming-friendly | Notes |
| --- | --- | --- | --- |
| Raw PCM | Core byte helpers | Yes | Headerless PCM is just framed bytes. |
| WAV / RF64 | `WavStreamEncoder` / `WasmWavEncoder` | Yes | Emits an exact header, then bounded interleaved PCM chunks. Uses RF64 beyond 4 GiB. |
| MP3 | `mp3lame` | Yes | Feature-gated encoder path. |
| AAC ADTS | `fdk-aac` | Yes | Existing native compatibility encoder. A SoundKit-owned AAC writer is planned last, after decoder consolidation. |
| Fragmented MP4 / SoundKit LL-HLS | Planned SoundKit boxer | Planned | The final encoding phase will box SoundKit-authored AAC access units into fragmented MP4 for LL-HLS. |
| FLAC | `wavey-flac` | Yes | The default encoder and decoder come from the standalone Wavey-owned codec. |
| Opus | `soundkit-opus` | Yes | Pure Rust 48 kHz CELT packet encoder with 16-bit and 24-bit APIs. |
| AMR-NB | OpenCORE AMR-NB | Yes | 160-sample speech frames. |
| G.711 / G.722 / G.726 / G.729 / GSM | Codec crates | Yes | Frame or sample streaming, depending on codec. |
| Vorbis / Speex / ALAC / AIFF / AC-3 / WebM | Decode-only | No | No encoder is planned for these import formats. AAC writing and fMP4 LL-HLS boxing are tracked separately. |

## Test Fixture Rule

| Requirement | Current pattern |
| --- | --- |
| Codec fixture | Generate with FFmpeg into `testdata/<format>/...` when FFmpeg can encode it. |
| Golden output | Ignored regeneration tools may write WAV under `golden/<format>/...`; normal tests are read-only. |
| Decoder tests | Compare chunked-vs-whole decode and run pipeline explicit/autodetect tests where available. |
| External comparison | Compare native PCM with FFmpeg PCM where practical. |
| Formal codec integration | Run `make codec-fate-test`; see the [FFmpeg FATE codec suite](docs/FFMPEG_FATE_CODEC_SUITE.md). |
| Manual playback | Play decoded golden WAVs with `ffplay` after implementation. |

## License Notes

| Dependency family | Distribution note |
| --- | --- |
| Bootstrapped in-tree codec cores | Preserve the upstream notices stored beside each `soundkit-*` crate. |
| `libFLAC`, `mp3lame`, `fdk-aac`, OpenCORE AMR-NB, `libgsm`, Rubber Band | C/C++ library dependencies; ship notices and review binary distribution requirements. |
| `libgsm` | Preserve the upstream notice in source and binary distributions. |

## Pending Formats

| Format | Status |
| --- | --- |
| AMR-WB | Pending a fixture-safe encoder path. |
| Monkey's Audio / APE | Deferred because the local FFmpeg build can decode APE but cannot encode fixtures. |
