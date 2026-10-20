#!/usr/bin/env bash
# Builds the two modules that tools/run_wasm_ab.mjs compares.
#
# OPUS_SRC: libopus source checkout (required)
# EMSDK:    emsdk root that contains upstream/emscripten/emcc (required)
# OUT_DIR:  output directory (default: ../target/opus-wasm-ab)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT/../target/opus-wasm-ab}"
: "${OPUS_SRC:?set OPUS_SRC to a libopus checkout}"
: "${EMSDK:?set EMSDK to an emsdk root}"
export PATH="$EMSDK/upstream/emscripten:$PATH"
export EM_NODE_JS="${EM_NODE_JS:-$(command -v node)}"

mkdir -p "$OUT_DIR/libopus"
if [ ! -f "$OUT_DIR/libopus/libopus.a" ]; then
  (cd "$OUT_DIR/libopus" &&
    emcmake cmake "$OPUS_SRC" -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_FLAGS_RELEASE="-O3 -DNDEBUG -msimd128 -flto" \
      -DOPUS_BUILD_PROGRAMS=OFF -DOPUS_BUILD_TESTING=OFF \
      -DOPUS_DRED=OFF -DOPUS_OSCE=OFF -DOPUS_DEEP_PLC=OFF \
      -DOPUS_STACK_PROTECTOR=OFF -DOPUS_FORTIFY_SOURCE=OFF >/dev/null &&
    emmake make -j8 opus >/dev/null)
fi

emcc -O3 -flto -msimd128 -I"$OPUS_SRC/include" "$ROOT/tools/wasm_ab/wasm_ab.c" \
  "$OUT_DIR/libopus/libopus.a" --no-entry -sSTANDALONE_WASM \
  -sINITIAL_MEMORY=256MB -sALLOW_MEMORY_GROWTH=1 -sFILESYSTEM=0 \
  -o "$OUT_DIR/wasm_ab_c.wasm"

(cd "$ROOT" && CARGO_PROFILE_RELEASE_LTO="${CARGO_PROFILE_RELEASE_LTO:-fat}" RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }-C target-feature=+simd128 -C no-vectorize-slp" \
  cargo build --release --target wasm32-unknown-unknown --example wasm_ab \
  --target-dir "$OUT_DIR/rust-target")
cp "$OUT_DIR/rust-target/wasm32-unknown-unknown/release/examples/wasm_ab.wasm" \
  "$OUT_DIR/wasm_ab_rust.wasm"
ls -l "$OUT_DIR"/*.wasm
