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
| Decode pipeline | `soundkit-decoder` | Ring-buffered worker thread, `access-unit` autodetection, explicit telephony paths, optional output conversion. |
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
| AAC ADTS | `soundkit-aac` / `fdk-aac` | Auto | Yes | Frame-stream friendly; C FFI backend. `soundkit-aac-lc` is the in-progress pure Rust raw access-unit decoder path. |
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
| AAC-LC raw access units | `soundkit-aac-lc` + `soundkit-wasm-decoder` | In progress | SoundKit-owned decoder crate. It validates AAC-LC `AudioSpecificConfig`, parses raw SCE/CPE access units with trailing `END`/`FIL` element handling, rejects SBR extension payloads for fallback, decodes standard scalefactors and spectral codebooks 1-11, applies table-backed dequantization, stateful deterministic PNS/noise-band reconstruction, TNS filtering, long/short mid-side helpers that skip PNS and right-channel intensity bands, long/short intensity stereo, long-window pulse data, AAC windows, previous/current window-shape tracking, `rustfft`-backed N/2 FFT IMDCT, buffered bit reading, and reusable overlap-add for only-long, long-start, long-stop, and eight-short frames. The current WESTSIDE AAC-LC fixture decodes 9171/9171 raw access units, passes the native `aac-wasm-bench` FDK oracle tolerance, and has an allocation guard proving zero heap allocations during a warmed steady-state fixture decode pass. Unsupported SBR/HE-AAC, PS, program-config-element channels, channel layouts beyond stereo, and SBR raw-block fill payloads are reported as explicit fallback errors. Current native WESTSIDE release baselines are about 36.7k frames/sec for SoundKit, 33.8k frames/sec through the reusable decoder path, 22.4k frames/sec for `fdk-aac-sys`, and 61.7k frames/sec for Symphonia. Against the WESTSIDE source WAV, saved baselines are FDK RMSE 0.006896303 / SNR 27.510 dB, Symphonia RMSE 0.006894503 / SNR 27.511 dB, and SoundKit RMSE 0.006919222 / SNR 27.480 dB. The browser/worker API is `WasmAacLcDecoder`, built with Rust `wasm32-unknown-unknown` and `wasm-bindgen`; the Node wasm checkpoint decodes AAC to PCM in Rust wasm, compares it directly to the source WAV plus the saved FDK/Symphonia baselines, and currently measures about 31.1k frames/sec for core raw access-unit decode, 27.9k frames/sec through the JS interleaved API, and 28.6k frames/sec through caller-provided reusable `Float32Array` output. Broader fixtures, wasm SIMD, and more edge-case AAC-LC tools remain open. |
| AAC in M4A/MP4 | `mp4` demux + `fdk-aac` | No | MP4 demux/debox can be Rust, but production AAC frame decode still uses FDK-AAC. |
| AMR-NB | OpenCORE AMR-NB | No | Requires the native `opencore-amrnb` library via `pkg-config`. |
| G.729 | `g729-sys` | No | Uses a native codec binding. |
| GSM 06.10 / WAV-49 | `gsm-sys` / `libgsm` | No | Uses the native libgsm codec. |
| Opus / Ogg Opus / WebM Opus | `soundkit-opus` pure Rust backend | Partial | Supported for the current packet path, but FEC is not implemented and the backend is not full libopus parity yet. |
| FLAC | `claxon` in `soundkit-decoder` | Yes | The aggregate decoder selects pure Rust `claxon`; the standalone `soundkit-flac` crate defaults to libFLAC unless `claxon-decoder` is selected. |
| Video codecs | Out of scope | No | SoundKit can demux/debox some audio containers, but it does not decode video codecs such as H.264, H.265, VPx, or AV1. |

Everything else in the decode matrix is currently on a pure-Rust decode path.
MP3 decode uses `nanomp3`; MP3 encode is the part that pulls in LAME.
Rubber Band is also a native dependency, but it is a time-stretch/resampling
tool rather than a codec decoder.

## AAC-LC Fast Native Loop

Use this native-only loop while iterating on `soundkit-aac-lc`; reserve wasm,
FDK/Symphonia, and full release benchmarks for integration checkpoints:

```bash
cargo test -p aac-wasm-bench --no-default-features --features soundkit-lc
cargo test -p soundkit-aac-lc scalefactor
cargo test -p soundkit-aac-lc imdct_fast
```

The full native benchmark uses the WESTSIDE AAC fixture and, when the sibling
`bitneedle` checkout is present, reports decoded AAC quality against the source
WAV as `source-vs-*` measurements. Override the source WAV path with
`SOUNDKIT_AAC_SOURCE_WAV=/path/to/source.wav`.

### Current AAC-LC Benchmarks

Native command:

```bash
cargo run -p aac-wasm-bench --release -- 5
```

Fixture: `golden/aac/WESTSIDE_MIX_4_CONFIRMATION_130323_256k.aac`, 9171 ADTS
frames, 48 kHz stereo, 195.648 seconds of AAC frame audio.

| Decoder | Iterations | Decoded frames | Elapsed | RTF | Frames/sec | RMS | Peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fdk-aac-sys` | 5 | 45,855 | 2,045.506 ms | 0.002091 | 22,417.4 | 0.162843351 | 0.918304443 |
| `soundkit-aac-lc` | 5 | 45,855 | 1,250.680 ms | 0.001278 | 36,664.1 | 0.162846547 | 0.918334007 |
| `soundkit-lc-reuse` | 5 | 45,855 | 1,355.441 ms | 0.001386 | 33,830.3 | 0.162846547 | 0.918334007 |
| `symphonia-aac` | 5 | 45,855 | 743.409 ms | 0.000760 | 61,682.1 | 0.162845552 | 1.001383424 |

Decoder-oracle checks compare decoded PCM against FDK with channel-aligned
offset search:

| Comparison | Status | RMSE | Mean abs | Max abs | SNR |
| --- | --- | ---: | ---: | ---: | ---: |
| `fdk-vs-symphonia` | pass | 0.001002403 | 0.000045033 | 0.260961175 | 44.215 dB |
| `fdk-vs-soundkit` | pass | 0.001242798 | 0.000132421 | 0.370039735 | 42.348 dB |

Source-WAV quality measurements compare each AAC decode against
`WESTSIDE_MIX 4 CONFIRMATION_130323.wav`:

| Comparison | RMSE | Mean abs | Max abs | SNR | p99 abs | p999 abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `source-vs-fdk` | 0.006896303 | 0.004594121 | 0.334225774 | 27.510 dB | 0.023480892 | 0.038754225 |
| `source-vs-soundkit` | 0.006919222 | 0.004613361 | 0.383684549 | 27.480 dB | 0.023648702 | 0.038796075 |
| `source-vs-symphonia` | 0.006894503 | 0.004592889 | 0.363226533 | 27.511 dB | 0.023478046 | 0.038671419 |

Wasm command:

```bash
wasm-pack test --node --release soundkit-wasm-decoder --no-default-features --features aac-lc-bench -- --nocapture
```

The wasm quality test computes SoundKit PCM inside Rust wasm and reports saved
native FDK/Symphonia baselines without linking either comparator into wasm:

| Comparison | Source | Compared samples | Offset | RMSE | Mean abs | Max abs | p99 abs | p999 abs | SNR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `saved-source-vs-fdk` | saved native | - | - | 0.006896303 | 0.004594121 | 0.334225774 | 0.023480892 | 0.038754225 | 27.510 dB |
| `saved-source-vs-symphonia` | saved native | - | - | 0.006894503 | 0.004592889 | 0.363226533 | 0.023478046 | 0.038671419 | 27.511 dB |
| `wasm-source-vs-sk` | computed in wasm | 18,779,958 | 2,048 samples | 0.006919222 | 0.004613361 | 0.383684535 | 0.023648679 | 0.038796104 | 27.480 dB |

| Wasm path | Iterations | Decoded frames | Elapsed | RTF | Frames/sec | Checksum |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `wasm-core-raw` | 5 | 45,855 | 1,476.331 ms | 0.001509 | 31,060.1 | `24eb18ebd8fc27e4` |
| `wasm-js-interleaved` | 5 | 45,855 | 1,642.932 ms | 0.001679 | 27,910.5 | `8c2d7264b07abe0a` |
| `wasm-js-into` | 5 | 45,855 | 1,605.535 ms | 0.001641 | 28,560.6 | `0d4a1cec5595b60a` |

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
| Monkey's Audio / APE | Deferred because the local FFmpeg build can decode APE but cannot encode fixtures. |
