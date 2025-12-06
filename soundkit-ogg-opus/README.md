# soundkit-ogg-opus

Streaming Ogg Opus decoder built on `ogg` + `soundkit-opus`.

- Feed arbitrary byte chunks via `add(&[u8])`.
- When a complete Opus packet (or series) is decoded, you get back `AudioData` (16-bit PCM, little-endian).
