# Safety Policy

`libopus-rs` is a Rust-first port. The public codec crate forbids unsafe Rust.
It uses a private path dependency for audited low-level kernels.

The rules are:

- no C API
- no C ABI artifacts
- no raw-pointer public API
- no unsafe in the public codec crate
- no unchecked public kernel API
- validate slice lengths, indices, and CPU features before each unsafe kernel
- keep each unsafe block next to its safety argument
- no codec logic hidden behind FFI wrappers

This is enforced at the crate root:

```rust
#![forbid(unsafe_code)]
```

This attribute does not apply recursively to dependencies. The private
`libopus-rs-kernels` crate contains the current unsafe boundary. It provides
checked safe functions for MDCT indexing, AArch64 NEON kernels, and x86 SIMD
kernels. The codec crate does not receive raw pointers from that boundary.

The kernel crate also enables `unsafe_op_in_unsafe_fn`. Each pointer operation
therefore needs an explicit unsafe block. Unit tests compare kernel reductions
with checked reference calculations and verify codec invariants.

The forward and inverse MDCT wrappers validate transform sizes, trig-table
lengths, bit-reversal indices, strides, and output capacity. Their inner loops
then use unchecked indexing without changing the scalar operation order.

The pulse-cache wrapper validates the maximum index in the first cache byte.
The fixed six-step search cannot read outside that checked range.

The x86 pitch-correlation wrapper checks AVX2 and FMA support, all input
ranges, and every output group. Its short tail uses a fixed 15-value mask. The
selected eight-value mask window remains inside that array for tail lengths
from one through seven.

The AArch64 radix-5 FFT wrapper checks multiplication overflow, the five value
blocks, the four packed-twiddle blocks, and NEON support before it uses raw
pointers. `Complex32` has a C-compatible pair-of-`f32` layout for interleaved
NEON loads. A direct test compares every output bit with the scalar butterfly.

The stereo i24 CELT deemphasis wrapper checks output-length multiplication,
equal channel lengths, two filter-state values, and output capacity before it
uses raw pointers. The inner kernel clamps each sample to a finite signed
24-bit range before it uses unchecked float-to-integer conversion. Direct tests
cover odd and even lengths, compare every output value and final filter state
with the scalar calculation, and reject invalid slice lengths.
