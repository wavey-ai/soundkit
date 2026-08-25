#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${TARGET_DIR:-/tmp/soundkit-opus-raw-bench-target}"
BENCH_DIR="${BENCH_DIR:-/tmp/soundkit-opus-raw-bench}"
REPEATS="${REPEATS:-21}"
AUDIO_SECONDS="${AUDIO_SECONDS:-4}"
MODE="${MODE:-both}"
FRAME_SIZE="${FRAME_SIZE:-}"
BITRATE="${BITRATE:-}"
FIXTURE="${FIXTURE:-mixed}"
INPUT_S32LE="${INPUT_S32LE:-}"
PCM_BITS="${PCM_BITS:-}"
APPLICATION="${APPLICATION:-audio}"
DIRECT_CUBIC="${DIRECT_CUBIC:-0}"
SKIP_QUALITY="${SKIP_QUALITY:-0}"
C_BENCH_CFLAGS="${C_BENCH_CFLAGS--O3 -DNDEBUG}"
RUST_BENCH_RUSTFLAGS="${RUST_BENCH_RUSTFLAGS--C target-cpu=native}"
BUILD_RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }${RUST_BENCH_RUSTFLAGS}"

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
    --application)
      APPLICATION="$2"
      shift 2
      ;;
    --direct-cubic)
      DIRECT_CUBIC=1
      shift
      ;;
    --skip-quality)
      SKIP_QUALITY=1
      shift
      ;;
    --frame-size)
      FRAME_SIZE="$2"
      shift 2
      ;;
    --bitrate)
      BITRATE="$2"
      shift 2
      ;;
    --fixture)
      FIXTURE="$2"
      shift 2
      ;;
    --input-s32le)
      INPUT_S32LE="$2"
      shift 2
      ;;
    --pcm-bits)
      PCM_BITS="$2"
      shift 2
      ;;
    *)
      echo "usage: tools/run_raw_celt_bench.sh [--repeats n] [--seconds n] [--mode cbr|vbr|both] [--application audio|restricted-lowdelay] [--direct-cubic] [--frame-size n] [--bitrate n] [--fixture mixed|tone] [--input-s32le path] [--pcm-bits 16|24] [--skip-quality]" >&2
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

case "$APPLICATION" in
  audio | restricted-lowdelay) ;;
  *)
    echo "--application must be audio or restricted-lowdelay" >&2
    exit 2
    ;;
esac

case "$DIRECT_CUBIC" in
  0 | 1) ;;
  *)
    echo "DIRECT_CUBIC must be 0 or 1" >&2
    exit 2
    ;;
esac

case "$SKIP_QUALITY" in
  0 | 1) ;;
  *)
    echo "SKIP_QUALITY must be 0 or 1" >&2
    exit 2
    ;;
esac

case "$FIXTURE" in
  mixed | tone) ;;
  *)
    echo "--fixture must be mixed or tone" >&2
    exit 2
    ;;
esac

case "$PCM_BITS" in
  "" | 16 | 24) ;;
  *)
    echo "--pcm-bits must be 16 or 24" >&2
    exit 2
    ;;
esac

if [[ -n "$INPUT_S32LE" && -z "$PCM_BITS" ]]; then
  PCM_BITS=24
fi

mkdir -p "$BENCH_DIR"

c_bin="$BENCH_DIR/raw_celt_bench_c"
c_decode_dump_bin="$BENCH_DIR/raw_celt_decode_dump_c"
rust_bin="$TARGET_DIR/release/examples/raw_celt_bench"

if [[ -n "${OPUS_DIR:-}" ]]; then
  if [[ ! -f "$OPUS_DIR/.libs/libopus.a" ]]; then
    echo "OPUS_DIR must point at a built libopus tree containing .libs/libopus.a" >&2
    exit 1
  fi
  # shellcheck disable=SC2086
  cc $C_BENCH_CFLAGS -I"$OPUS_DIR/include" \
    "$ROOT/tools/raw_celt_bench.c" "$OPUS_DIR/.libs/libopus.a" -lm -o "$c_bin"
  # shellcheck disable=SC2086
  cc $C_BENCH_CFLAGS -I"$OPUS_DIR/include" \
    "$ROOT/tools/raw_celt_decode_dump.c" "$OPUS_DIR/.libs/libopus.a" -lm -o "$c_decode_dump_bin"
else
  if ! pkg-config --exists opus; then
    echo "Set OPUS_DIR to a built libopus tree, or install opus for pkg-config." >&2
    exit 1
  fi
  # shellcheck disable=SC2046
  # shellcheck disable=SC2086
  cc $C_BENCH_CFLAGS $(pkg-config --cflags opus) \
    "$ROOT/tools/raw_celt_bench.c" $(pkg-config --libs opus) -lm -o "$c_bin"
  # shellcheck disable=SC2046
  # shellcheck disable=SC2086
  cc $C_BENCH_CFLAGS $(pkg-config --cflags opus) \
    "$ROOT/tools/raw_celt_decode_dump.c" $(pkg-config --libs opus) -lm -o "$c_decode_dump_bin"
fi

RUSTFLAGS="$BUILD_RUSTFLAGS" cargo build --release --target-dir "$TARGET_DIR" --example raw_celt_bench >/dev/null

if [[ -n "$INPUT_S32LE" ]]; then
  if [[ "$PCM_BITS" == 16 ]]; then
    echo "Raw in-memory CELT benchmark: 48 kHz stereo signed-16 PCM derived from an S32LE signed-24 source, no file I/O in measured loops." >&2
  else
    echo "Raw in-memory CELT benchmark: 48 kHz stereo signed-24 S32LE input, no file I/O in measured loops." >&2
  fi
  echo "Repeats: $REPEATS, seconds: $AUDIO_SECONDS, mode: $MODE, application: $APPLICATION, direct cubic: $DIRECT_CUBIC, input: $INPUT_S32LE" >&2
else
  echo "Raw in-memory CELT benchmark: generated 48 kHz stereo fixture, no file I/O in measured loops." >&2
  echo "Repeats: $REPEATS, seconds: $AUDIO_SECONDS, mode: $MODE, application: $APPLICATION, direct cubic: $DIRECT_CUBIC, fixture: $FIXTURE" >&2
fi

bench_args=(--repeats "$REPEATS" --seconds "$AUDIO_SECONDS" --mode "$MODE" --application "$APPLICATION" --fixture "$FIXTURE")
if [[ -n "$INPUT_S32LE" ]]; then
  bench_args+=(--input-s32le "$INPUT_S32LE")
fi
if [[ -n "$PCM_BITS" ]]; then
  bench_args+=(--pcm-bits "$PCM_BITS")
fi
if [[ -n "$FRAME_SIZE" ]]; then
  bench_args+=(--frame-size "$FRAME_SIZE")
fi
if [[ -n "$BITRATE" ]]; then
  bench_args+=(--bitrate "$BITRATE")
fi
if [[ "$SKIP_QUALITY" == 1 ]]; then
  bench_args+=(--skip-quality)
fi

rust_bench_args=("${bench_args[@]}")
if [[ "$DIRECT_CUBIC" == 1 ]]; then
  rust_bench_args+=(--direct-cubic)
fi

rust_out="$("$rust_bin" "${rust_bench_args[@]}")"
c_out="$("$c_bin" "${bench_args[@]}")"

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
    quality_lag[$1, key] = $12
    quality_snr[$1, key] = $13
  }
  END {
    print "| Mode | Frame | Bitrate | Rust enc (xRTF) | Enc vs C | Rust dec (xRTF) | Dec vs C | C enc (xRTF) | C dec (xRTF) | Rust bytes | C bytes | Rust pkt | C pkt | Rust SNR | C SNR | Rust lag | C lag |"
    print "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    for (i = 1; i <= count; i++) {
      key = order[i]
      if ((("rust" SUBSEP key) in encode) && (("c" SUBSEP key) in encode)) {
        enc_delta = 100.0 * (encode["rust", key] - encode["c", key]) / encode["c", key]
        dec_delta = 100.0 * (decode["rust", key] - decode["c", key]) / decode["c", key]
        printf "| %s | %.1f ms | %d kb/s | %.2fx | %+.1f%% | %.2fx | %+.1f%% | %.2fx | %.2fx | %d | %d | %d-%d | %d-%d | %.2f dB | %.2f dB | %d | %d |\n", \
          mode[key], frame_ms[key], bitrate[key] / 1000, rt_factor / encode["rust", key], enc_delta, \
          rt_factor / decode["rust", key], dec_delta, rt_factor / encode["c", key], rt_factor / decode["c", key], \
          bytes["rust", key], bytes["c", key], \
          min_packet["rust", key], max_packet["rust", key], min_packet["c", key], max_packet["c", key], \
          quality_snr["rust", key], quality_snr["c", key], quality_lag["rust", key], quality_lag["c", key]
      }
    }
  }
'
