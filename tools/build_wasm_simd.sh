#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${TARGET:-wasm32-unknown-unknown}"
SIMD_RUSTFLAGS="${SIMD_RUSTFLAGS--C target-feature=+simd128}"
BUILD_RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }${SIMD_RUSTFLAGS}"

cd "$ROOT"
RUSTFLAGS="$BUILD_RUSTFLAGS" cargo build --release --target "$TARGET" "$@"
