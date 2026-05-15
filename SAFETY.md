# Safety Policy

`libopus-rs` is a Rust-first port. The default crate forbids unsafe Rust.

The rules are:

- no C API
- no C ABI artifacts
- no raw-pointer public API
- no unsafe in the default crate
- no codec logic hidden behind FFI wrappers

This is enforced at the crate root:

```rust
#![forbid(unsafe_code)]
```
