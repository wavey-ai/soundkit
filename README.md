# soundkit

A lightweight, extensible Rust audio toolbox providing:

- **Raw sample conversions** (16 ↔ 32 bit, 24 bit, float ↔ int)  
- **Interleave / deinterleave** helpers for multi-channel PCM  
- **WAV I/O**: streaming parser (`WavStreamProcessor`) and generator (`generate_wav_buffer`)  
- **Resampling** via `rubato` sinc-kernel interpolator  
- **Packetized codecs** through uniform `Encoder` / `Decoder` traits:  
  - **Opus** (using `libopus`)  
  - **FLAC** (via `libflac_sys`)  
  - **MP3** (via `mp3lame-sys` / `mp3lame_encoder`)  
  - **AAC** (using `fdk-aac`)  
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

- **Unified codec interface**  
  Traits `Encoder` / `Decoder` let you plug in any supported format  
  Helpers for framing, packetizing, and byte-order management.

- **WASM support**  
  `WavToPkt` and `WavToPcm` bindings for streaming audio in the browser.

---
