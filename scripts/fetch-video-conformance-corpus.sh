#!/bin/sh
set -eu

chromium_revision="146.0.7650.0"
output_root="${1:-build/video-compat/upstream}"
base_url="https://raw.githubusercontent.com/chromium/chromium/${chromium_revision}/media/test/data"

mkdir -p "$output_root"

fetch() {
  name="$1"
  expected_sha256="$2"
  destination="$output_root/$name"
  if [ ! -f "$destination" ] || [ "$(shasum -a 256 "$destination" | awk '{print $1}')" != "$expected_sha256" ]; then
    curl --fail --location --silent --show-error "$base_url/$name" --output "$destination"
  fi
  actual_sha256="$(shasum -a 256 "$destination" | awk '{print $1}')"
  if [ "$actual_sha256" != "$expected_sha256" ]; then
    echo "$name checksum mismatch: expected $expected_sha256, got $actual_sha256" >&2
    exit 2
  fi
  echo "$name $actual_sha256"
}

fetch test-25fps.h264 c018857e2eb964837dc033d77e56505e2417494f6542864b71a33c6895077db8
fetch test-25fps.hevc 4cb52ca41a34f28c951324e6e08c5cdd41159e66c6faf11bb46071a354c4e45b
fetch test-25fps.hevc10 46f2f338d6ace186e5f9cb5d89e8af460217495b6c42783c29afd8e14004490b
fetch test-25fps.av1.ivf 787a7d9476e4557741ed4ca6bf9cff621ace478896b5bfd743438ade789ef319
fetch vp90_2_10_show_existing_frame2.vp9.ivf feef0243ad5317e526902493333d64342c75e8f0b4fcd83342007fdbee80d193
fetch bear-1280x720-hevc-10bit-hdr10.hevc b6a869e766161d92064117bdefe412e28b66f77f3db843e5dff4c04aad858b0c
fetch av1-monochrome-I-frame-320x240-10bpp bf6f70489c77d34f025774229225f4b448d5d5891ff25903cd51ed7bdcf61ea9
fetch bear_av1_720p_444_10bit.ivf ed8928ec2952cbda78a1c1c7fa43f6a98e3d4ae0289ed610112514b5ca6af7e6
