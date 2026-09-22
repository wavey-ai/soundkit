cmake_minimum_required(VERSION 3.24)
project(soundkit_avif_wasm C)

include(FetchContent)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(SOUNDKIT_LIBAVIF_REF "v1.2.1" CACHE STRING "libavif git ref to build")

set(AVIF_CODEC_AOM "LOCAL" CACHE STRING "Build libaom locally for AVIF encode")
set(AOM_TARGET_CPU "generic" CACHE STRING "Disable native CPU assembly/intrinsics for WASM")
set(AVIF_CODEC_DAV1D "OFF" CACHE STRING "Disable dav1d decode")
set(AVIF_CODEC_LIBGAV1 "OFF" CACHE STRING "Disable libgav1 decode")
set(AVIF_CODEC_RAV1E "OFF" CACHE STRING "Disable rav1e")
set(AVIF_CODEC_SVT "OFF" CACHE STRING "Disable SVT-AV1")
set(AVIF_LIBYUV "OFF" CACHE BOOL "Disable libyuv")
set(AVIF_ZLIBPNG "OFF" CACHE BOOL "Disable PNG helper support")
set(AVIF_JPEG "OFF" CACHE BOOL "Disable JPEG helper support")
set(AVIF_BUILD_APPS "OFF" CACHE BOOL "Do not build avifenc/avifdec")
set(AVIF_BUILD_TESTS "OFF" CACHE BOOL "Do not build tests")
set(AVIF_BUILD_EXAMPLES "OFF" CACHE BOOL "Do not build examples")
set(AVIF_ENABLE_WERROR "OFF" CACHE BOOL "No warnings as errors")
set(BUILD_SHARED_LIBS "OFF" CACHE BOOL "Static libraries only")

FetchContent_Declare(
  libavif
  GIT_REPOSITORY https://github.com/AOMediaCodec/libavif.git
  GIT_TAG ${SOUNDKIT_LIBAVIF_REF}
)
FetchContent_MakeAvailable(libavif)

add_executable(soundkit_avif src/soundkit_avif.c)
target_link_libraries(soundkit_avif PRIVATE avif)

target_compile_options(soundkit_avif PRIVATE -Oz -flto)

if (EMSCRIPTEN)
  set_target_properties(soundkit_avif PROPERTIES
    OUTPUT_NAME "soundkit_avif"
    SUFFIX ".js"
  )
  target_link_options(soundkit_avif PRIVATE
    -Oz
    -flto
    "-sMODULARIZE=1"
    "-sEXPORT_ES6=1"
    "-sENVIRONMENT=web,worker,node"
    "-sALLOW_MEMORY_GROWTH=1"
    # aom's slower search paths carry deep stacks and large frames; the 64KB
    # emscripten default overflows into "memory access out of bounds" at any
    # speed below 5, which is exactly where the quality per byte is.
    "-sSTACK_SIZE=8MB"
    "-sFILESYSTEM=0"
    "-sMALLOC=emmalloc"
    "-sEXPORTED_FUNCTIONS=['_skavif_malloc','_skavif_free','_skavif_encode_rgba','_skavif_output_ptr','_skavif_output_size','_skavif_last_error_ptr']"
    "-sEXPORTED_RUNTIME_METHODS=['HEAPU8','UTF8ToString']"
  )
endif()
