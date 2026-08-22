#!/bin/sh
set -eu

matroska_revision="e6965e5ca666322ed93e2748a10a4f132309e005"
output_root="${1:-build/container-corpus}"
matroska_root="$output_root/matroska"
matroska_url="https://raw.githubusercontent.com/ietf-wg-cellar/matroska-test-files/$matroska_revision/test_files"

mkdir -p "$matroska_root"

fetch() {
  name="$1"
  expected_sha256="$2"
  destination="$matroska_root/$name"
  if [ ! -f "$destination" ] || [ "$(shasum -a 256 "$destination" | awk '{print $1}')" != "$expected_sha256" ]; then
    curl --fail --location --silent --show-error \
      "$matroska_url/$name" \
      --output "$destination"
  fi
  actual_sha256="$(shasum -a 256 "$destination" | awk '{print $1}')"
  if [ "$actual_sha256" != "$expected_sha256" ]; then
    echo "$name checksum mismatch: expected $expected_sha256, got $actual_sha256" >&2
    exit 2
  fi
  echo "$name $actual_sha256"
}

fetch test1.mkv 0996a309ff2095910b9d30d5253b044d637154297ddf7d0bda7f3adedf5addc1
fetch test2.mkv 5b53d306e56f9bda6e80c3fbd9f3ccd20cc885770449d1fc0b5bec35c71d61e2
fetch test3.mkv 1722b0d93a6ef1a14dd513bd031cd5901c233b45aa3e3c87be0b0d7348d7d1b5
fetch test4.mkv 43df750a2a01a37949791b717051b41522081a266b71d113be4b713063843699
fetch test5.mkv 92acdc33bb0b5d7a4d9b0d6ca792230a78c786a30179dc9999cee41c28642842
fetch test6.mkv 7cad84b434116e023d340dd584ac833b93f03fb1bd7ea2727fa45de50af0abb9
fetch test7.mkv 95b21c92ad5a4fe00914ff5009e2a64f12fd4c5fb9cb1c3c888ab50bf0ffe483
fetch test8.mkv 9dddcd1550b814dae44d62e2b9f27c0eca31d5e190df2220cbf7492e3d6c63da
