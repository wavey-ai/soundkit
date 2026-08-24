# SoundKit rusty_av1d patch

SoundKit vendors `rusty_av1d` 1.2.0 from crates.io. The source uses the BSD-2-Clause license in `LICENSE`.

SoundKit changes one cleanup condition in `src/decode.rs`. Malformed AV1 input can clear `frame_hdr` before `on_error` runs. The upstream code then calls `unwrap()` and aborts the WASM instance. The patch treats a missing header as no refresh context.

SoundKit adds `Picture::plane16()` in `src/rust_api.rs`. Upstream `Picture::plane()` always creates an 8-bit slice. The new method exposes the decoder's existing 16-bit plane storage through the same read-only guard.

SoundKit also fixes `Picture::bit_depth()` and `Picture::stride()`. The first method now returns the 8-bit or 16-bit storage width. The second method now returns samples as documented.

SoundKit restores upstream's `src/arm/asm-offsets.h`, which the crates.io
package excludes even though the native ARM assembly sources include it. This
allows native AArch64 builds to use dav1d's assembly fast paths; WASM builds
continue to use the no-assembly implementation.

Remove this patch after upstream publishes an equivalent repair.
