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
