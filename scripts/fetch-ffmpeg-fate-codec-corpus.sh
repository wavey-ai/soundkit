#!/bin/sh
set -eu

output_root="${1:-build/fate-codec-corpus}"
manifest="${2:-scripts/ffmpeg-fate-codec-manifest.tsv}"
fate_url="rsync://fate-suite.ffmpeg.org/fate-suite"

command -v rsync >/dev/null 2>&1 || {
  echo "rsync is required to download FFmpeg FATE samples" >&2
  exit 2
}

mkdir -p "$output_root"

awk '!/^#/ && NF { print $6, $7 }' "$manifest" |
while read -r relative_path expected_sha256; do
  destination="$output_root/$relative_path"
  mkdir -p "${destination%/*}"
  if [ ! -f "$destination" ] || [ "$(shasum -a 256 "$destination" | awk '{print $1}')" != "$expected_sha256" ]; then
    rsync -a "$fate_url/$relative_path" "${destination%/*}/"
  fi
  actual_sha256="$(shasum -a 256 "$destination" | awk '{print $1}')"
  if [ "$actual_sha256" != "$expected_sha256" ]; then
    echo "$relative_path checksum mismatch: expected $expected_sha256, got $actual_sha256" >&2
    exit 2
  fi
  echo "$relative_path $actual_sha256"
done
