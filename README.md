# soundkit

[![CI](https://github.com/wavey-ai/soundkit/actions/workflows/ci.yml/badge.svg)](https://github.com/wavey-ai/soundkit/actions/workflows/ci.yml)

Rust audio tooling for PCM conversion, WAV handling, resampling, codec wrappers,
and a thread-based decode pipeline with automatic format detection.

## At A Glance

| Area | Crates / APIs | Notes |
| --- | --- | --- |
| PCM utilities | `soundkit::audio_bytes`, `soundkit::raw_pcm` | Sample-width conversion, endian conversion, interleave/deinterleave, headerless PCM streams. |
| WAV | `soundkit::wav` | Incremental RIFF/WAVE PCM parser plus `generate_wav_buffer`. |
| Resampling | `soundkit::downsample_audio`, `soundkit-rubberband` | `rubato` sinc resampling and Rubber Band wrapper. |
| Codecs | `soundkit-*` codec crates | Small wrappers around native Rust decoders where available, with C FFI only where useful or required. |
| Decode pipeline | `soundkit-decoder` | Ring-buffered worker thread, autodetection, explicit telephony paths, optional output conversion. |
| WASM | `soundkit::wasm` | Browser-oriented WAV-to-packet and WAV-to-PCM helpers. |

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
| AAC ADTS | `soundkit-aac` / `fdk-aac` | Auto | Yes | Frame-stream friendly; C FFI backend. |
| AAC in M4A/MP4 | `soundkit-aac` / `mp4` + `fdk-aac` | Auto | Limited | MP4 sample tables make this less suitable for live chunking than ADTS. |
| FLAC | `soundkit-flac` / `claxon` | Auto | Yes | Pure Rust decode; current wrapper keeps input history while decoding. |
| Raw Opus stream | `soundkit-opus` / `libopus` | Auto | Yes | Soundkit `OpusHead` plus length-prefixed packets. |
| Ogg Opus | `soundkit-ogg-opus` / Ogg parser + `libopus` | Auto | Yes | Ogg pages parsed incrementally. |
| WebM Opus | `soundkit-webm` / EBML parser + `libopus` | Auto | Yes | WebM clusters parsed incrementally. |
| Ogg Speex | `soundkit-speex` / `oxideav-speex` | Explicit | Yes | Pure Rust codec core and streaming Ogg packet parser. |
| Ogg Vorbis | `soundkit-vorbis` / `lewton` | Auto or explicit | Yes | Pure Rust decode and streaming Ogg packet parser. |
| ALAC in M4A/MP4 or CAF | `soundkit-alac` / `alac` | Auto or explicit | EOF | Pure Rust codec; current container reader requires `Read + Seek`. |
| AIFF / AIFF-C | `soundkit-aiff` / `aifc` | Auto or explicit | EOF | Pure Rust reader; current wrapper decodes after EOF. |
| Raw AC-3 syncframes | `soundkit-ac3` / `oxideav-ac3` | Auto or explicit | Yes | Raw elementary AC-3 stream, not containerized AC-3. |
| AMR-NB | `soundkit-amr` / OpenCORE AMR-NB | Explicit | Yes | 3GPP `.amr` magic and raw frame streams; C FFI backend. |
| G.711 u-law / A-law | `soundkit-g711` | Explicit | Yes | Pure Rust PCMU/PCMA decode. |
| G.722 | `soundkit-g722` / `ezk-g722` | Explicit | Yes | Pure Rust 64 kbit/s wideband speech decode. |
| G.726-32 | `soundkit-g726` | Explicit | Yes | Pure Rust 32 kbit/s profile; 16/24/40 kbit/s profiles pending. |
| G.729 | `soundkit-g729` / `g729-sys` | Explicit | Yes | Frame-buffered 8 kbit/s speech decode. |
| GSM 06.10 / WAV-49 | `soundkit-gsm` / `libgsm` | Explicit | Yes | Standard raw GSM and Microsoft WAV-49 packet framing. |

## Decode Pipeline APIs

| Need | API |
| --- | --- |
| Autodetect common media files | `DecodePipeline::spawn()` |
| Override output rate, depth, or channels | `DecodePipeline::spawn_with_options(options)` |
| Headerless PCM | `DecodePipeline::spawn_raw_pcm(format)` |
| Telephony and speech codecs | `spawn_g711`, `spawn_g722`, `spawn_g726`, `spawn_g729`, `spawn_gsm`, `spawn_amr_nb`, `spawn_speex` |
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
| WAV | `generate_wav_buffer` | No | Writes a complete WAV buffer. |
| MP3 | `mp3lame` | Yes | Feature-gated encoder path. |
| AAC ADTS | `fdk-aac` | Yes | ADTS output. |
| FLAC | `libFLAC` | Yes | Reference encoder. |
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
| G.726 16/24/40 kbit/s | Decoder API currently implements the common 32 kbit/s profile. |
| Monkey's Audio / APE | Deferred because the local FFmpeg build can decode APE but cannot encode fixtures. |
