#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
output=${1:-"$script_dir/ffmpeg-aac-production-bench"}

cc -O3 -march=native -DNDEBUG \
  "$script_dir/ffmpeg-aac-production-bench.c" \
  -o "$output" \
  $(pkg-config --cflags --libs libavcodec libavutil libswresample) \
  -lm
