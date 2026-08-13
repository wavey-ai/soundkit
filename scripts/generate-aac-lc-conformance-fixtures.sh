#!/bin/sh
set -eu

input="golden/aac/WESTSIDE_MIX_4_CONFIRMATION_130323_256k.aac"
adts="golden/aac/stereo-music-44100-192k.aac"
m4a="golden/aac/stereo-music-44100-192k.m4a"

ffmpeg -hide_banner -loglevel error -y \
  -i "$input" \
  -t 3 \
  -ar 44100 \
  -ac 2 \
  -c:a aac \
  -profile:a aac_low \
  -b:a 192k \
  -f adts \
  "$adts"

ffmpeg -hide_banner -loglevel error -y \
  -i "$adts" \
  -c:a copy \
  -movflags +faststart \
  "$m4a"
