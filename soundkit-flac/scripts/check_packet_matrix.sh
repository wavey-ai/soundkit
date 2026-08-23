#!/usr/bin/env bash
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
crate_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
matrix_tmp=$(mktemp -d "${TMPDIR:-/tmp}/soundkit-flac-matrix.XXXXXX")
trap 'rm -rf -- "$matrix_tmp"' EXIT

matrix_target="$matrix_tmp/target"
CARGO_TARGET_DIR="$matrix_target" cargo build --quiet --release \
  --manifest-path "$crate_dir/Cargo.toml" \
  --example flac_packet_matrix_fixture \
  --example flac_packet_verify

cc -O2 "$script_dir/libflac_packet_fixture.c" \
  $(pkg-config --cflags --libs flac) \
  -o "$matrix_tmp/libflac_packet_fixture"
cc -O2 "$script_dir/ffmpeg_flac_packet_verify.c" \
  $(pkg-config --cflags --libs libavcodec libavutil) \
  -o "$matrix_tmp/ffmpeg_flac_packet_verify"

for rate in 48000 96000; do
  for channels in 1 2 8; do
    for bits in 16 24; do
      for profile_level in realtime:0 balanced:2; do
        profile=${profile_level%:*}
        level=${profile_level#*:}
        stem="$matrix_tmp/${rate}-${channels}-${bits}-${profile}"
        "$matrix_target/release/examples/flac_packet_matrix_fixture" \
          "$rate" "$channels" "$bits" "$profile" 64 \
          "$stem.pcm" "$stem.soundkit.bundle"
        "$matrix_tmp/ffmpeg_flac_packet_verify" \
          "$stem.soundkit.bundle" "$stem.pcm" "$rate" "$channels" "$bits"
        "$matrix_tmp/libflac_packet_fixture" \
          "$stem.pcm" "$stem.libflac.bundle" "$rate" "$channels" "$bits" "$level"
        "$matrix_target/release/examples/flac_packet_verify" \
          "$rate" "$channels" "$bits" "$stem.pcm" "$stem.libflac.bundle"
      done
    done
  done
done

CARGO_TARGET_DIR="$matrix_target" cargo test --quiet \
  --manifest-path "$crate_dir/Cargo.toml" \
  randomized_malformed_packet_matrix_never_panics_or_escapes_checksum_validation

echo "packet differential matrix passed"
