#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "usage: benchmark_packet_sequence.sh PCM_48K_S32LE PCM_96K_S32LE [ITERATIONS] [ROUNDS]" >&2
  exit 2
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
crate_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
pcm_48=$1
pcm_96=$2
iterations=${3:-20000}
rounds=${4:-3}
bench_tmp=$(mktemp -d "${TMPDIR:-/tmp}/soundkit-flac-bench.XXXXXX")
trap 'rm -rf -- "$bench_tmp"' EXIT

bench_target=${SOUNDKIT_BENCH_TARGET:-"$crate_dir/target/packet-sequence-bench"}
CARGO_TARGET_DIR="$bench_target" cargo build --quiet --release \
  --manifest-path "$crate_dir/Cargo.toml" --example flac_packet_bench
cc -O3 "$script_dir/libflac_frame_bench.c" \
  $(pkg-config --cflags --libs flac) -lm \
  -o "$bench_tmp/libflac_frame_bench"
cc -O3 "$script_dir/ffmpeg_flac_packet_bench.c" \
  $(pkg-config --cflags --libs libavcodec libavutil) \
  -o "$bench_tmp/ffmpeg_flac_packet_bench"

rust_bench="$bench_target/release/examples/flac_packet_bench"

run_case() {
  local rate=$1
  local profile=$2
  local level=$3
  local pcm=$4
  local stem="$bench_tmp/${rate}-${profile}"

  "$bench_tmp/libflac_frame_bench" \
    "$rate" "$level" 1 1 "$pcm" "$stem.libflac.bundle" >/dev/null
  "$rust_bench" "$rate" 2 24 "$profile" 1 "$pcm" \
    "$stem.libflac.bundle" "$stem.soundkit.bundle" >/dev/null

  run_soundkit() {
    "$rust_bench" "$rate" 2 24 "$profile" "$iterations" "$pcm" \
      "$stem.libflac.bundle" "$stem.soundkit.bundle"
  }
  run_libflac() {
    "$bench_tmp/libflac_frame_bench" \
      "$rate" "$level" "$iterations" 1 "$pcm"
  }
  run_ffmpeg() {
    "$bench_tmp/ffmpeg_flac_packet_bench" \
      "$stem.soundkit.bundle" "$pcm" "$rate" "$iterations" 1
  }

  echo "case rate=$rate profile=$profile libflac_level=$level"
  local round
  for ((round = 0; round < rounds; round++)); do
    echo "round=$((round + 1))"
    case $((round % 3)) in
      0) run_soundkit; run_libflac; run_ffmpeg ;;
      1) run_libflac; run_ffmpeg; run_soundkit ;;
      2) run_ffmpeg; run_soundkit; run_libflac ;;
    esac
  done
}

echo "benchmark host"
uname -a
if command -v lscpu >/dev/null 2>&1; then lscpu; fi
rustc --version
cargo --version
ffmpeg -version 2>/dev/null | sed -n '1p'
flac --version

run_case 48000 realtime 0 "$pcm_48"
run_case 48000 balanced 2 "$pcm_48"
run_case 96000 realtime 0 "$pcm_96"
run_case 96000 balanced 2 "$pcm_96"
