# soundkit

[![CI](https://github.com/wavey-ai/soundkit/actions/workflows/ci.yml/badge.svg)](https://github.com/wavey-ai/soundkit/actions/workflows/ci.yml)

A lightweight, extensible Rust audio toolbox providing:

- **Raw sample conversions** (16 ↔ 32 bit, 24 bit, float ↔ int)
- **Interleave / deinterleave** helpers for multi-channel PCM
- **WAV I/O**: streaming parser (`WavStreamProcessor`) and generator (`generate_wav_buffer`)
- **Resampling** via `rubato` sinc-kernel interpolator
- **Downmixing** multi-channel audio to mono/stereo with proper coefficients
- **Packetized codecs** through uniform `Encoder` / `Decoder` traits:
  - **Opus** (using `libopus`)
  - **FLAC** (via `libflac_sys`)
  - **MP3** (via `mp3lame-sys` / `minimp3`)
  - **AAC** (using `fdk-aac`)
- **Container formats**:
  - **Ogg Opus** streaming decoder
  - **WebM** container decoder (Opus audio)
  - **M4A/MP4** container support for AAC
- **Streaming decode pipeline** with automatic format detection
- **WebAssembly bindings** (`wasm-bindgen`) for in-browser audio framing
- **Pipeline helpers** for framing, encoding, mixing and more  

---

## Features

- **Sample-width conversions**
  Convert arbitrary raw buffers:
  `i16le_to_f32`, `s32le_to_f32`, `s24le_to_i32`, `f32le_to_i16`, …

- **Channel packing / unpacking**
  `interleave_vecs_i16` / `deinterleave_vecs_i16`
  Supports 16, 24, 32-bit and float PCM.

- **WAV streaming**
  Incremental WAV parser that handles arbitrary chunk boundaries
  Supports classic PCM and extensible (24-bit) headers.
  `generate_wav_buffer` to write PCM back to RIFF/WAVE.

- **Resampling**
  `downsample_audio` using `rubato::SincFixedIn` for high-quality offline resampling.

- **Downmixing**
  Convert multi-channel audio (5.1, 7.1, etc.) to mono or stereo using proper ITU-R BS.775 coefficients.

- **Unified codec interface**
  Traits `Encoder` / `Decoder` let you plug in any supported format
  Helpers for framing, packetizing, and byte-order management.

- **Streaming decode pipeline** (`soundkit-decoder`)
  Thread-based pipeline with automatic format detection for:
  - MP3, FLAC, AAC (M4A/MP4), Opus, Ogg Opus, WebM
  - Ring buffer I/O for backpressure handling
  - Optional output transformations (sample rate, bit depth, channel count)

- **Container decoders**
  - `OggOpusDecoder`: streaming Ogg container parser with Opus decoding
  - `WebmDecoder`: EBML/WebM container parser with Opus audio extraction

- **WASM support**
  `WavToPkt` and `WavToPcm` bindings for streaming audio in the browser.

---

## Contact-Center and STT Format Roadmap

These formats are the next practical additions for Deepgram-style speech-to-text
ingestion, SIP/RTP integrations, and call-center recording workflows. The order
reflects expected product value for telephony audio rather than general media
coverage.

| Priority | Format / codec | Why it matters | Status / implementation path |
| --- | --- | --- | --- |
| 1 | **G.711 u-law / A-law** (`mulaw`, `alaw`, `PCMU`, `PCMA`) | Baseline PSTN and SIP trunk audio. Twilio Media Streams use 8 kHz mono u-law, and many contact-center recording/transcription paths accept PCMU/PCMA directly. | Implemented as native Rust in `soundkit-g711`, with streaming encode/decode helpers and explicit `DecodePipeline::spawn_g711(...)` support. |
| 2 | **Raw PCM stream modes** (`linear16`, `linear32`, `L16`, `s16le`, `f32le`) | Common STT and contact-center handoff format. Deepgram raw streaming requires explicit encoding and sample rate; Amazon Connect-style streams are typically 8 kHz mono PCM. | Implemented in `soundkit::raw_pcm` with explicit stream descriptors, partial-frame buffering, and `DecodePipeline::spawn_raw_pcm(...)` support. |
| 3 | **G.722** | Common wideband VoIP codec for higher-quality speech at 16 kHz while staying telephony-friendly. | Implemented as `soundkit-g722` using the native Rust `ezk-g722` codec core, with odd-sample buffering and `DecodePipeline::spawn_g722()` support. |
| 4 | **G.729** | Still appears in bandwidth-constrained SIP and call-center deployments, and is accepted by Deepgram as an explicit raw/containerized encoding. | Implemented as `soundkit-g729` using the pure-Rust `g729-sys` backend, with 80-sample / 10-byte frame buffering and `DecodePipeline::spawn_g729()` support. |
| 5 | **AMR-NB / AMR-WB** | Used in mobile/carrier audio and supported by Deepgram. AMR-NB is narrowband speech; AMR-WB is wideband speech. | Investigate OpenCORE/VisualOn-style C libraries and Rust FFI availability. Treat licensing and deployment packaging as part of the design. |
| 6 | **Speex** | Legacy speech codec that still appears in older VoIP and recording archives; also supported by Deepgram. | Use `speex-sys` / `speex-safe` or a narrowly scoped FFI wrapper around `libspeex`. |
| 7 | **GSM 06.10** | Legacy PBX and call-recording codec. Useful for archive compatibility rather than new integrations. | Use `gsm-sys` / `libgsm` if fixtures show real demand. Keep it optional. |
| 8 | **G.726** | Shows up in PBX, recorder, and surveillance-adjacent voice systems. Lower priority than G.711/G.722/G.729. | Evaluate whether to implement the ADPCM codec directly or bind an existing telecom codec library. |

Vorbis/Ogg Vorbis remains useful for consumer media ingestion, but it is lower
priority for call-center pipelines than the telephony codecs above. Ogg is only
the container; Ogg Vorbis and Ogg Opus use different audio codecs.

---
