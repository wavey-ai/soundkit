#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${TARGET_DIR:-/tmp/libopus-rs-raw-bench-target}"
BENCH_DIR="${BENCH_DIR:-/tmp/libopus-rs-raw-bench}"
REPEATS="${REPEATS:-21}"
AUDIO_SECONDS="${AUDIO_SECONDS:-4}"
MODE="${MODE:-both}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repeats)
      REPEATS="$2"
      shift 2
      ;;
    --seconds)
      AUDIO_SECONDS="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    *)
      echo "usage: tools/run_raw_celt_bench.sh [--repeats n] [--seconds n] [--mode cbr|vbr|both]" >&2
      exit 2
      ;;
  esac
done

case "$MODE" in
  cbr | vbr | both) ;;
  *)
    echo "--mode must be cbr, vbr, or both" >&2
    exit 2
    ;;
esac

mkdir -p "$BENCH_DIR"

c_bin="$BENCH_DIR/raw_celt_bench_c"
rust_bin="$TARGET_DIR/release/examples/raw_celt_bench"

if [[ -n "${OPUS_DIR:-}" ]]; then
  if [[ ! -f "$OPUS_DIR/.libs/libopus.a" ]]; then
    echo "OPUS_DIR must point at a built libopus tree containing .libs/libopus.a" >&2
    exit 1
  fi
  cc -O3 -DNDEBUG -I"$OPUS_DIR/include" \
    "$ROOT/tools/raw_celt_bench.c" "$OPUS_DIR/.libs/libopus.a" -lm -o "$c_bin"
else
  if ! pkg-config --exists opus; then
    echo "Set OPUS_DIR to a built libopus tree, or install opus for pkg-config." >&2
    exit 1
  fi
  # shellcheck disable=SC2046
  cc -O3 -DNDEBUG $(pkg-config --cflags opus) \
    "$ROOT/tools/raw_celt_bench.c" $(pkg-config --libs opus) -lm -o "$c_bin"
fi

cargo build --release --target-dir "$TARGET_DIR" --example raw_celt_bench >/dev/null

echo "Raw in-memory CELT benchmark: generated 48 kHz stereo fixture, no file I/O in measured loops." >&2
echo "Repeats: $REPEATS, seconds: $AUDIO_SECONDS, mode: $MODE" >&2

rust_out="$("$rust_bin" --repeats "$REPEATS" --seconds "$AUDIO_SECONDS" --mode "$MODE")"
c_out="$("$c_bin" --repeats "$REPEATS" --seconds "$AUDIO_SECONDS" --mode "$MODE")"

rt_factor=$((AUDIO_SECONDS * 1000))

printf '%s\n%s\n' "$rust_out" "$c_out" | awk -F '\t' -v rt_factor="$rt_factor" '
  $1 == "impl" { next }
  {
    key = $2 "|" $3 "|" $5
    if ($1 == "rust" && !(key in seen)) {
      order[++count] = key
      seen[key] = 1
      mode[key] = $2
      frame_ms[key] = $4
      bitrate[key] = $5
    }
    encode[$1, key] = $6
    decode[$1, key] = $7
    bytes[$1, key] = $8
    min_packet[$1, key] = $9
    max_packet[$1, key] = $10
  }
  END {
    print "| Mode | Frame | Bitrate | Rust enc (RTFx) | Enc vs C | Rust dec (RTFx) | Dec vs C | C enc (RTFx) | C dec (RTFx) | Rust bytes | C bytes | Rust pkt | C pkt |"
    print "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    for (i = 1; i <= count; i++) {
      key = order[i]
      if ((("rust" SUBSEP key) in encode) && (("c" SUBSEP key) in encode)) {
        enc_delta = 100.0 * (encode["rust", key] - encode["c", key]) / encode["c", key]
        dec_delta = 100.0 * (decode["rust", key] - decode["c", key]) / decode["c", key]
        printf "| %s | %.1f ms | %d kb/s | %.4fx | %+.1f%% | %.4fx | %+.1f%% | %.4fx | %.4fx | %d | %d | %d-%d | %d-%d |\n", \
          mode[key], frame_ms[key], bitrate[key] / 1000, encode["rust", key] / rt_factor, enc_delta, \
          decode["rust", key] / rt_factor, dec_delta, encode["c", key] / rt_factor, decode["c", key] / rt_factor, \
          bytes["rust", key], bytes["c", key], \
          min_packet["rust", key], max_packet["rust", key], min_packet["c", key], max_packet["c", key]
      }
    }
  }
'
