# Native AAC production benchmark — 2026-08-25

## Result

The default SoundKit production decoder selected `SoundKitAacLc` and beat the
native FFmpeg C reference on every track in the five-fixture music corpus.

| Music fixture | Audio duration | SoundKit | FFmpeg C | SoundKit faster | SoundKit realtime | FFmpeg realtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| WESTSIDE full mix | 195.648 s | **277.171 ms** | 292.586 ms | **5.27%** | 705.9x | 668.7x |
| Bill Evans — Secret Sessions | 100.032 s | **145.231 ms** | 149.858 ms | **3.09%** | 688.8x | 667.5x |
| The Blue Nile — Hats | 100.032 s | **144.233 ms** | 146.771 ms | **1.73%** | 693.5x | 681.6x |
| Lori Asha | 100.032 s | **141.392 ms** | 147.184 ms | **3.94%** | 707.5x | 679.6x |
| Nocturnal Animals | 100.032 s | **133.659 ms** | 140.784 ms | **5.06%** | 748.4x | 710.5x |

Times are medians for one complete decode, normalized from batches of three.
The corpus contains music only. The short A Tusk speech fixture remains a unit
test for streaming behavior and was not used as performance evidence.

## Host and toolchain

- Google Cloud instance: `yl-encodec-1`
- Project: `steadfast-slate-498623-r2`
- Zone: `europe-west2-b`
- Machine: `c4-highcpu-4`, four virtual CPUs
- CPU: Intel Xeon Platinum 8581C, Emerald Rapids
- OS: Debian 12 x86-64
- Rust: `rustc 1.97.1`, `cargo 1.97.1`
- FFmpeg: `5.1.9`
- FLAC: `1.4.2`
- C compiler: GCC `12.2.0`

The benchmark was CPU-only. No GPU or hardware codec path was enabled.

## Compared work

The measurement includes the production work an application has to perform:

1. Construct and initialize a fresh decoder for each iteration.
2. Parse the complete ADTS stream.
3. Decode every AAC access unit.
4. Convert planar floating-point decoder output to interleaved signed 16-bit
   PCM.
5. Consume the complete PCM output so the compiler cannot discard work.

SoundKit used the public `soundkit_aac::AacDecoder` API and its automatically
selected owned backend. The C reference directly used libavcodec, copied each
raw AAC packet into FFmpeg-padded storage, and used libswresample for its native
planar-float-to-interleaved-`i16` conversion. File reading and the full-output
quality checksum were outside the timed region for both implementations.

Each process first performed one complete untimed warm-up decode. The measured
run used 11 rounds, three full decodes per round, pinned to logical CPU 0 with
`taskset`. SoundKit and C process order alternated between rounds to reduce
thermal and scheduler bias. The table reports the median batch time divided by
three.

This production comparison supersedes the narrower core-only checkpoint. That
checkpoint excluded wrapper work and measured a 2.4–6.4% SoundKit lead; it is
not mixed into the production table above.

## Fixtures and stable output

Every implementation produced a stable sample count and checksum in all 11
rounds. Checksums differ between decoders because the independently implemented
synthesis paths are compared by the quality oracle rather than bit identity.

| Fixture file | Fixture SHA-256 | SoundKit FNV-1a PCM | FFmpeg C FNV-1a PCM |
| --- | --- | --- | --- |
| `WESTSIDE_MIX_4_CONFIRMATION_130323_256k.aac` | `e4b3eae14f2fb1fbb3e70f41d7cb879a49951455a8d05fec244f91006391f711` | `1f7b96f0d1f34bdc` | `a7d5d343624c6d87` |
| `bill-evans-secret-sessions-48k-256k.aac` | `e450bcd4e8be08ac148ece6aba718f031971eecc086e4763d5426514c8a40db0` | `aa4ec56aafc2cabe` | `39099fc3b90aa823` |
| `blue-nile-hats-48k-256k.aac` | `2cd1d10902d2e888e6a54734d68c53d7805ab2480715d358c47a085c81b631b5` | `f4eccea8e27bd60c` | `2c5cb89d957e457e` |
| `lori-asha-48k-256k.aac` | `aaece88f15aadc9e2030baed294600965a673fdc0073cd1398f0b33eac34acaf` | `d71f6532a499c206` | `5aeaecff5194a381` |
| `nocturnal-animals-48k-256k.aac` | `93049e89a667f97e9fcc080bcd7201aa688550874111e6e3f0f25a0f9dd820ca` | `816a72ce54374d5b` | `46bea4a642fb8482` |

Measured executable identity:

| Artifact | SHA-256 |
| --- | --- |
| `bench_adts_decode` | `ffda1301590781f6d017b490a3b366f3830c8de85749a7e9f3a3e695d11f679d` |
| `ffmpeg-aac-production-bench` | `e75fad6de2950c2f53cb71835644452af8569b258be653c61854aeaaaa00c99e` |

## Correctness gates

The owned decoder was compared with FDK-AAC using equal decoded sample counts
and these enforced limits: RMSE at most `0.005`, mean absolute error at most
`0.001`, maximum absolute error at most `0.50`, and SNR at least `35 dB`.

| Music fixture | RMSE | Mean absolute error | Maximum error | SNR | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| WESTSIDE, 48 kHz stereo | 0.000745894 | 0.000044259 | 0.228224084 | 46.783 dB | pass |
| Stereo music, 44.1 kHz | 0.000814121 | 0.000433312 | 0.020074576 | 37.865 dB | pass |

Production PCM is required to be exactly invariant across different input chunk
sizes on the same host. Cross-architecture synthesis can differ by a few least
significant `i16` bits, so release gates use the quality oracle across machines
instead of an invalid architecture-independent PCM hash.

## Release test matrix

All commands ran in release mode on the benchmark host after the final
optimization:

| Command | Result |
| --- | ---: |
| `cargo test -p soundkit-aac --release --all-features` | 12 passed |
| `cargo test -p soundkit-aac --release --no-default-features --features owned-lc` | 5 passed |
| `cargo test -p soundkit-aac --release --no-default-features --features fdk` | 3 passed |
| `cargo test -p soundkit-aac --release --no-default-features --features mp4-decoder` | 8 passed |
| `cargo test -p soundkit-aac --release --no-default-features --features mp4-fdk-fallback` | 12 passed |
| `cargo test -p soundkit-aac-lc --release` | 121 unit + 2 integration passed |
| `cargo test -p aac-wasm-bench --release --no-default-features --features fdk,soundkit-lc -- --nocapture` | 6 passed |

The integration tests include deterministic malformed-input coverage and a
steady-state no-allocation decode check. Production wrapper tests cover split
ADTS sync words, arbitrary chunk sizes, small-output recovery, owned/FDK backend
routing, implicit HE-AAC detection, 44.1 kHz M4A owned decoding, and HE-AAC MP4
fallback.

## Build and run

```sh
cargo build -p soundkit-aac --release --all-features \
  --example bench_adts_decode

./aac-wasm-bench/reference/build-ffmpeg-native-production.sh

taskset -c 0 target/release/examples/bench_adts_decode \
  golden/aac/WESTSIDE_MIX_4_CONFIRMATION_130323_256k.aac 3 soundkit

taskset -c 0 aac-wasm-bench/reference/ffmpeg-aac-production-bench \
  golden/aac/WESTSIDE_MIX_4_CONFIRMATION_130323_256k.aac 3 1
```

Repeat each command for 11 rounds, alternating which executable goes first.
The C harness is built with `-O3 -march=native -DNDEBUG`; Cargo uses the release
profile.
