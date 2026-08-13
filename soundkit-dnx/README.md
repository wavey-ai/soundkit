# soundkit-dnx

`soundkit-dnx` is SoundKit's memory-safe Rust decoder for VC-3, DNxHD, and
DNxHR coding units. It is deliberately isolated from the MIT-licensed codec
facade because the implementation and normative code tables are derived from
FFmpeg's LGPL decoder.

The implementation is licensed under LGPL-2.1-or-later. The corresponding
source used for this port is FFmpeg commit
`ca821e458aabe2fa211d9e94eac38cd69fe2ea09`.

The public API accepts one complete coding unit and returns bounded planar
pixels. It reports YCbCr and GBR planes explicitly for DNxHR 444. Container
parsing and all untrusted-media validation remain in Rust.
