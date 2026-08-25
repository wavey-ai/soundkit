#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: build-fdk-wasm.sh FDK_AAC_SOURCE [OUTPUT_DIR]" >&2
  exit 2
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
fdk_source=$1
output_dir=${2:-"$script_dir/pkg-fdk"}
emxx=${EMXX:-em++}
arch_profile=${FDK_WASM_ARCH_PROFILE:-wasm32}

# Some vendored emsdk installations retain a stale absolute path to their
# bundled Node binary. Prefer the active Node executable for compiler helpers.
if [[ -z ${EM_NODE_JS:-} ]] && command -v node >/dev/null 2>&1; then
  export EM_NODE_JS
  EM_NODE_JS=$(command -v node)
fi

if [[ ! -f "$fdk_source/libAACdec/include/aacdecoder_lib.h" ]]; then
  echo "FDK_AAC_SOURCE must be the directory containing libAACdec" >&2
  exit 2
fi

mkdir -p "$output_dir"

source_dirs=(
  libAACdec
  libPCMutils
  libFDK
  libSYS
  libMpegTPDec
  libSBRdec
  libArithCoding
  libDRCdec
  libSACdec
)

sources=("$script_dir/fdk-wasm-bench.cpp")
include_flags=()
for library in "${source_dirs[@]}"; do
  while IFS= read -r source; do
    sources+=("$source")
  done < <(find "$fdk_source/$library/src" -maxdepth 1 -name '*.cpp' -print | sort)
  include_flags+=("-I$fdk_source/$library/include")
done

case "$arch_profile" in
  portable)
    architecture_flags=()
    ;;
  wasm32)
    # WebAssembly has fast 32-bit integer multiplication but scalar i64 shifts
    # are relatively costly. This matches FDK's 32x16 coefficient profile
    # without pretending that the target supports x86 or ARM assembly.
    architecture_flags=(
      -DARCH_PREFER_MULT_32x16
      -DSINETABLE_16BIT
      -DWINDOWTABLE_16BIT
      -DPOW2COEFF_16BIT
      -DLDCOEFF_16BIT
    )
    ;;
  *)
    echo "unsupported FDK_WASM_ARCH_PROFILE: $arch_profile" >&2
    exit 2
    ;;
esac

"$emxx" "${sources[@]}" "${include_flags[@]}" "${architecture_flags[@]}" \
  -std=c++17 -O3 -flto -msimd128 -DNDEBUG \
  -DFDK_FALLTHROUGH= -DSUPPRESS_BUILD_DATE_INFO \
  -Wno-cpp -Wno-deprecated-register -Wno-unused-parameter \
  --no-entry \
  -sMODULARIZE=1 \
  -sEXPORT_ES6=1 \
  -sENVIRONMENT=node \
  -sFILESYSTEM=0 \
  -sALLOW_MEMORY_GROWTH=1 \
  -sASSERTIONS=0 \
  -sDISABLE_EXCEPTION_CATCHING=1 \
  -sEXPORTED_FUNCTIONS='["_malloc","_free","_fdk_aac_bench","_fdk_aac_last_decoded_frames","_fdk_aac_last_samples_per_channel","_fdk_aac_last_checksum_high","_fdk_aac_last_checksum_low","_fdk_aac_last_error"]' \
  -sEXPORTED_RUNTIME_METHODS='["HEAPU8"]' \
  -o "$output_dir/fdk-aac.mjs"

ls -lh "$output_dir/fdk-aac.mjs" "$output_dir/fdk-aac.wasm"
