#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "usage: benchmark_packet_wasm.sh PCM_48K_S32LE PCM_96K_S32LE [ITERATIONS] [ROUNDS]" >&2
  exit 2
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
crate_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
workspace_dir=$(CDPATH= cd -- "$crate_dir/.." && pwd)
wasm_crate="$workspace_dir/soundkit-wasm"
pcm_48=$1
pcm_96=$2
iterations=${3:-20000}
rounds=${4:-3}
wasm_transfer=${SOUNDKIT_WASM_TRANSFER:-buffered}
bench_tmp=$(mktemp -d "${TMPDIR:-/tmp}/soundkit-flac-wasm-bench.XXXXXX")
trap 'rm -rf -- "$bench_tmp"' EXIT

bench_target=${SOUNDKIT_BENCH_TARGET:-"$crate_dir/target/packet-wasm-bench"}
native_rustflags=${SOUNDKIT_NATIVE_RUSTFLAGS:-${RUSTFLAGS:-}}
wasm_rustflags=${SOUNDKIT_WASM_RUSTFLAGS:-${RUSTFLAGS:-}}
RUSTFLAGS="$native_rustflags" CARGO_TARGET_DIR="$bench_target" cargo build --quiet --release \
  --manifest-path "$crate_dir/Cargo.toml" --example flac_packet_bench
RUSTFLAGS="${wasm_rustflags:+$wasm_rustflags }-C target-feature=+simd128" \
  wasm-pack build "$wasm_crate" --target web --out-dir "$bench_tmp/wasm-pkg" \
    --release -- --no-default-features --features flac

rust_bench="$bench_target/release/examples/flac_packet_bench"

run_case() {
  local rate=$1
  local profile=$2
  local pcm=$3
  local bundle="$bench_tmp/${rate}-${profile}.bundle"

  "$rust_bench" "$rate" 2 24 "$profile" 1 "$pcm" - "$bundle" >/dev/null

  run_native() {
    "$rust_bench" "$rate" 2 24 "$profile" "$iterations" "$pcm" \
      "$bundle"
  }
  run_wasm() {
    node --expose-gc "$script_dir/benchmark_flac_packet_wasm.mjs" \
      "$bench_tmp/wasm-pkg" "$rate" "$profile" "$iterations" "$pcm" "$bundle" \
      "$wasm_transfer"
  }

  echo "case rate=$rate profile=$profile"
  local round
  for ((round = 0; round < rounds; round++)); do
    echo "round=$((round + 1))"
    if ((round % 2 == 0)); then
      run_native
      run_wasm
    else
      run_wasm
      run_native
    fi
  done
}

echo "benchmark host"
uname -a
if command -v lscpu >/dev/null 2>&1; then lscpu; fi
if command -v sysctl >/dev/null 2>&1; then
  sysctl -n machdep.cpu.brand_string 2>/dev/null || true
fi
uptime
rustc --version
cargo --version
node --version
wasm-pack --version
echo "native rustflags=${native_rustflags:-<default>}"
echo "wasm rustflags=${wasm_rustflags:+$wasm_rustflags }-C target-feature=+simd128 transfer=$wasm_transfer"

run_case 48000 realtime "$pcm_48"
run_case 48000 balanced "$pcm_48"
run_case 96000 realtime "$pcm_96"
run_case 96000 balanced "$pcm_96"
