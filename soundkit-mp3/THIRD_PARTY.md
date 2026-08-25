# Decoder provenance

The SoundKit Layer III decoder in `src/decoder.rs` was bootstrapped from the
permissively licensed `nanomp3` 0.1.1/minimp3 code lineage (nanomp3 revision
`c1bad1b`). Its standard MPEG lookup data lives in `src/decoder/tables.rs`.
The corresponding MIT and Apache-2.0 notices remain in
`LICENSE-NANOMP3-MIT` and `LICENSE-NANOMP3-APACHE`.

SoundKit now owns and maintains the decoder implementation directly: there is
no nanomp3/minimp3 package, FFI boundary, wrapper, callback layer, or runtime
dispatch into third-party decoder code. The in-tree core uses bounded Rust
state, SoundKit streaming semantics, safe entropy/DSP code, and authored SSE2
and AVX2 synthesis kernels. The provenance notices are retained even though
the former dependency boundary no longer exists.
