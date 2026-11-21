#!/usr/bin/env bash
# Builds the Opus Custom interop tools and runs the comparison matrix.
#
# OPUS_SRC:   libopus source checkout (required)
# OPUS_BUILD: a CMake build of OPUS_SRC with -DOPUS_CUSTOM_MODES=ON; built
#             under OUT_DIR when unset
# OUT_DIR:    work directory (default: ../target/opus-custom-interop)
# MUSIC:      optional comma-separated 48 kHz stereo s32le files (24-bit
#             samples, e.g. `ffmpeg -f s32le`), converted and added to the
#             inputs. Keep real music out of the repository.
# Any arguments are passed to `opus_custom_interop.py run`, for example
# `--frames 48,120 --rates 256000 --jobs 8`.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT/../target/opus-custom-interop}"
: "${OPUS_SRC:?set OPUS_SRC to a libopus checkout}"
mkdir -p "$OUT_DIR"

if [[ -z "${OPUS_BUILD:-}" ]]; then
  OPUS_BUILD="$OUT_DIR/libopus-custom"
  if [[ ! -f "$OPUS_BUILD/libopus.a" ]]; then
    cmake -S "$OPUS_SRC" -B "$OPUS_BUILD" -DCMAKE_BUILD_TYPE=Release \
      -DOPUS_CUSTOM_MODES=ON -DOPUS_BUILD_PROGRAMS=OFF -DOPUS_BUILD_TESTING=OFF \
      -DOPUS_DRED=OFF -DOPUS_DEEP_PLC=OFF -DOPUS_OSCE=OFF >/dev/null
    cmake --build "$OPUS_BUILD" --target opus -j 8 >/dev/null
  fi
fi

cc -O2 -DOPUS_CUSTOM_INTERNALS -DHAVE_CONFIG_H -I"$OPUS_BUILD" -I"$OPUS_SRC/include" \
  -I"$OPUS_SRC/celt" -I"$OPUS_SRC" "$ROOT/tools/opus_custom_interop.c" \
  "$OPUS_BUILD/libopus.a" -lm -o "$OUT_DIR/opus_custom_interop_c"

(cd "$ROOT" && cargo build --release --example opus_custom_interop \
  --target-dir "$OUT_DIR/rust-target")
RUST_TOOL="$OUT_DIR/rust-target/release/examples/opus_custom_interop"

DATA="$OUT_DIR/data"
mkdir -p "$DATA"
python3 -B "$ROOT/tools/opus_custom_interop.py" synth "$DATA" --seconds 3
INPUTS="synth"
if [[ -n "${MUSIC:-}" ]]; then
  i=0
  IFS=',' read -ra files <<<"$MUSIC"
  for file in "${files[@]}"; do
    i=$((i + 1))
    python3 -B "$ROOT/tools/opus_custom_interop.py" convert "$file" "$DATA/music$i"
    INPUTS="$INPUTS,music$i"
  done
fi

python3 -B "$ROOT/tools/opus_custom_interop.py" run --c "$OUT_DIR/opus_custom_interop_c" \
  --rust "$RUST_TOOL" --data "$DATA" --out "$OUT_DIR/results" --inputs "$INPUTS" "$@"
echo "results: $OUT_DIR/results/summary.md and $OUT_DIR/results/results.tsv"
