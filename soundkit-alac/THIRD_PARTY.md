# Decoder provenance

The packet decoder in `src/decoder` was bootstrapped from `alac` version 0.5.0,
copyright Edward Barnard. That project is available under the MIT or
Apache-2.0 license. SoundKit retains both upstream license texts in
`LICENSE-ALAC-RS-MIT` and `LICENSE-ALAC-RS-APACHE`.

SoundKit maintains the decoder directly in this crate. The external `alac`,
`caf`, `mp4parse`, and `bitreader` packages are not in the production dependency
tree. SoundKit's own seekable M4A/MP4/MOV and CAF indexes supply bounded ALAC
access units to the codec core.

SoundKit replaced the original 32-bit bit cursor with a 64-bit buffered Rice
reader and direct unary decoding. It added bounded stream validation, optimized
adaptive LPC coefficient updates, right-aligned internal samples, and a
caller-owned reusable PCM output path. The normal crate API emits interleaved
little-endian PCM.
