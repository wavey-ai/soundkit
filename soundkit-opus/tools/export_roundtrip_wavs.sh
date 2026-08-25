#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${TARGET_DIR:-/tmp/soundkit-opus-roundtrip-target}"
OUT_DIR="${OUT_DIR:-/tmp/soundkit-opus-roundtrip-wavs}"
OPUS_DEMO="${OPUS_DEMO:-$ROOT/../opus/opus_demo}"

usage() {
  echo "usage: tools/export_roundtrip_wavs.sh --input path/to/input-audio [--out-dir path] [--mode cbr|vbr|both]" >&2
  exit 2
}

INPUT=""
MODE="both"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      INPUT="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    *)
      usage
      ;;
  esac
done

case "$MODE" in
  cbr | vbr | both) ;;
  *) usage ;;
esac

if [[ -z "$INPUT" || ! -f "$INPUT" ]]; then
  usage
fi
if [[ ! -x "$OPUS_DEMO" ]]; then
  echo "Set OPUS_DEMO to an executable upstream opus_demo." >&2
  exit 1
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required to wrap opus_demo raw PCM output as WAV." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
cargo build --release --target-dir "$TARGET_DIR" --example wav_celt >/dev/null
rust_bin="$TARGET_DIR/release/examples/wav_celt"

input_wav="$OUT_DIR/input.48k-s16le.wav"
input_raw="$OUT_DIR/input.s16le"
ffmpeg -y -hide_banner -loglevel error -i "$INPUT" \
  -acodec pcm_s16le -ar 48000 -ac 2 "$input_wav"
ffmpeg -y -hide_banner -loglevel error -i "$INPUT" \
  -f s16le -acodec pcm_s16le -ar 48000 -ac 2 "$input_raw"

manifest="$OUT_DIR/manifest.tsv"
printf 'mode\tframe_ms\tframe_size\tbitrate\trust_stream\trust_wav\tc_stream\tc_wav\n' >"$manifest"

for mode in cbr vbr; do
  if [[ "$MODE" != "both" && "$MODE" != "$mode" ]]; then
    continue
  fi
  for frame in 120 240 480 960; do
    case "$frame" in
      120) frame_ms="2.5" ;;
      240) frame_ms="5" ;;
      480) frame_ms="10" ;;
      960) frame_ms="20" ;;
      *) exit 1 ;;
    esac
    for bitrate in 192000 256000 320000; do
      case_dir="$OUT_DIR/${mode}_${frame_ms}ms_$((bitrate / 1000))k"
      mkdir -p "$case_dir"

      rust_stream="$case_dir/rust.lors"
      rust_wav="$case_dir/rust.decoded.wav"
      c_stream="$case_dir/c.bit"
      c_raw="$case_dir/c.decoded.s16le"
      c_wav="$case_dir/c.decoded.wav"

      rust_args=(roundtrip --frame-size "$frame" --bitrate "$bitrate")
      c_args=(-e restricted-lowdelay 48000 2 "$bitrate" -bandwidth FB -framesize "$frame_ms")
      if [[ "$mode" == "vbr" ]]; then
        rust_args+=(--vbr)
        c_args+=(-cvbr)
      else
        c_args+=(-cbr)
      fi

      "$rust_bin" "${rust_args[@]}" "$input_wav" "$rust_stream" "$rust_wav"
      "$OPUS_DEMO" "${c_args[@]}" "$input_raw" "$c_stream" >/dev/null 2>&1
      "$OPUS_DEMO" -d 48000 2 "$c_stream" "$c_raw" >/dev/null 2>&1
      ffmpeg -y -hide_banner -loglevel error \
        -f s16le -ar 48000 -ac 2 -i "$c_raw" "$c_wav"
      rm -f "$c_raw"

      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$mode" "$frame_ms" "$frame" "$bitrate" \
        "$rust_stream" "$rust_wav" "$c_stream" "$c_wav" >>"$manifest"
    done
  done
done

echo "Wrote roundtrip WAVs to $OUT_DIR"
echo "Manifest: $manifest"
