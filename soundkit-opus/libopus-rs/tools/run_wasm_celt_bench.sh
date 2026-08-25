#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCALAR_TARGET_DIR="${SCALAR_TARGET_DIR:-/tmp/libopus-rs-wasm-scalar}"
SIMD_TARGET_DIR="${SIMD_TARGET_DIR:-/tmp/libopus-rs-wasm-simd}"
TARGET="${TARGET:-wasm32-unknown-unknown}"
SIMD_RUSTFLAGS="${SIMD_RUSTFLAGS--C target-feature=+simd128}"

cd "$ROOT"
RUSTFLAGS="${RUSTFLAGS:-}" cargo build --release --target "$TARGET" \
  --target-dir "$SCALAR_TARGET_DIR" --example wasm_celt_bench >/dev/null
RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }${SIMD_RUSTFLAGS}" cargo build --release --target "$TARGET" \
  --target-dir "$SIMD_TARGET_DIR" --example wasm_celt_bench >/dev/null

node "$ROOT/tools/run_wasm_celt_bench.js" \
  "$SCALAR_TARGET_DIR/$TARGET/release/examples/wasm_celt_bench.wasm" \
  "$SIMD_TARGET_DIR/$TARGET/release/examples/wasm_celt_bench.wasm"
