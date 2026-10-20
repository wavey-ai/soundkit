#!/usr/bin/env bash
# Builds the Rust A/B module from the working tree and copies it to
# ../target/opus-wasm-ab/<name>.wasm.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="$ROOT/../target/opus-wasm-ab"
cd "$ROOT"
CARGO_PROFILE_RELEASE_LTO="${CARGO_PROFILE_RELEASE_LTO:-fat}" \
  RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }-C target-feature=+simd128 -C no-vectorize-slp" \
  cargo build -q --release --target wasm32-unknown-unknown --example wasm_ab ${FEATURES:+--features $FEATURES} \
  --target-dir "$OUT_DIR/rust-target" 2>&1 | grep -E '^error' -A12 || true
cp "$OUT_DIR/rust-target/wasm32-unknown-unknown/release/examples/wasm_ab.wasm" "$OUT_DIR/$1.wasm"
