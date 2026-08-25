# Decoder provenance

The decoder in `src/decoder` was bootstrapped from Lewton revision
`bb2955b717094b40260902cf2f8dd9c5ea62a84a`. Lewton is available under the MIT
or Apache-2.0 license. SoundKit retains the upstream copyright and license text
in `LICENSE-LEWTON`.

SoundKit maintains the decoder directly in this crate. No Lewton package, C
library, FFI boundary, wrapper, or callback layer remains in the codec path.

SoundKit added an allocation-free FFT IMDCT, cached transform tables, and a
buffered bit reader. It also added AVX2 FFT and PCM kernels with portable Rust
fallbacks. The SoundKit streaming facade performs Ogg granule trimming and
writes signed 16-bit PCM directly.
