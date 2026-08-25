# soundkit-aac

Production AAC encoding and decoding for SoundKit.

The default decoder now selects the SoundKit-owned AAC-LC implementation for
supported mono/stereo AAC-LC streams. Native builds retain FDK-AAC as an
automatic compatibility fallback for HE-AAC/SBR/PS, unsupported profiles,
program-config-element or surround layouts, and unsupported coding tools.

Encoding is supported on native targets through FDK-AAC. SoundKit does not yet
have an owned AAC encoder.

## Decoder routing

`AacDecoder::new()` starts without a selected backend. The first complete ADTS
header, or the MP4 `AudioSpecificConfig`, selects the backend:

| Input | Default backend |
| --- | --- |
| MPEG-4 AAC-LC, 1,024-sample frames, mono or stereo | `SoundKitAacLc` |
| HE-AAC, SBR, PS, non-LC profiles, PCE/surround, unsupported tools | `FdkAac` on native builds |

Use `AacDecoder::new_soundkit_aac_lc()` to require the owned decoder and return
an explicit error instead of falling back. Use `AacDecoder::new_fdk()` to force
FDK-AAC. `AacDecoder::backend()` reports the selected backend.

The ADTS decoder accepts arbitrary input chunk boundaries, including a sync
word split across calls. Call `decode_i16(&[], ...)` between chunks to drain
available frames, then `finish_i16(...)` at EOF. Finishing rejects a truncated
last frame and prevents subsequent input. Input and pending compressed data are
bounded to 4 MiB.

```rust
use soundkit::audio_packet::Decoder;
use soundkit_aac::{AacDecoder, AacDecoderBackend};

let adts_chunk: &[u8] = obtain_adts_bytes();
let mut decoder = AacDecoder::new();
decoder.init()?;

let mut pcm = vec![0_i16; 128 * 1024];
let written = decoder.decode_i16(adts_chunk, &mut pcm, false)?;
consume_interleaved_pcm(&pcm[..written]);

loop {
    let written = decoder.finish_i16(&mut pcm)?;
    if written == 0 {
        break;
    }
    consume_interleaved_pcm(&pcm[..written]);
}

assert_ne!(decoder.backend(), AacDecoderBackend::Pending);
# Ok::<(), String>(())
```

With `mp4-decoder`, `AacDecoderMp4` applies the track's complete
`AudioSpecificConfig` before decoding. With `mp4-fdk-fallback`, implicit and
explicit HE-AAC configurations route to FDK-AAC rather than being decoded as
their lower-rate AAC-LC core.

For container-indexed raw AAC-LC access units, construct
`AacLcAccessUnitDecoder` from the track `AudioSpecificConfig`. This exported
facade keeps callers on `soundkit-aac`; the lower-level `soundkit-aac-lc` crate
is an internal codec engine.

## Features

| Feature | Function |
| --- | --- |
| `default` | Enables `owned-lc` and `fdk`. |
| `owned-lc` | SoundKit-owned AAC-LC backend for ADTS and raw access-unit decoding. |
| `fdk` | Native FDK-AAC fallback, forced decoding, and AAC-LC ADTS encoding. |
| `mp4-demux` | Streaming AAC packet/config extraction from M4A/MP4. |
| `mp4-decoder` | MP4 demux plus the owned AAC-LC decoder. |
| `mp4-fdk-fallback` | MP4 decoding with owned AAC-LC and native FDK fallback. |

## Native performance

The production `soundkit-aac` API beats a directly linked FFmpeg C decoder on
all five music fixtures in the native corpus:

| Music fixture | Audio | SoundKit | FFmpeg C | SoundKit faster |
| --- | ---: | ---: | ---: | ---: |
| WESTSIDE full mix | 195.648 s | **94.896 ms** | 99.764 ms | **4.88%** |
| Bill Evans — Secret Sessions | 100.032 s | **49.684 ms** | 51.293 ms | **3.14%** |
| The Blue Nile — Hats | 100.032 s | **49.619 ms** | 50.461 ms | **1.67%** |
| Lori Asha | 100.032 s | **48.827 ms** | 49.885 ms | **2.12%** |
| Nocturnal Animals | 100.032 s | **45.821 ms** | 47.955 ms | **4.45%** |

These are median times for one complete decode: the median three-decode batch
from 11 alternating rounds divided by three. The test ran on an Intel Emerald
Rapids CPU. Both paths construct a decoder, parse the same ADTS input, decode
and finish every frame, convert to interleaved signed 16-bit PCM, and consume
the full output inside the timed region. Every process performs an untimed
full-file warm-up first. No speech fixture is included in this performance
result.

See [BENCHMARK_NATIVE_2026-08-25.md](BENCHMARK_NATIVE_2026-08-25.md) for the
host, commands, checksums, artifact hashes, methodology, and complete release
test matrix.

## What changed

- Wired the owned decoder into the public ADTS and MP4 production APIs.
- Added automatic FDK routing without sending supported AAC-LC through C.
- Detects implicit MPEG-4 SBR/HE-AAC sync extensions before backend selection.
- Replaced repeated streaming-buffer drains with a cursor and bounded
  compaction.
- Reuses decoder and fallback PCM storage, including recovery when the caller's
  output slice cannot hold the pending FDK frame.
- Converts planar `f32` output directly to interleaved `i16`; x86-64 uses an
  exact AVX2 conversion/packing path when available.
- Preserves exact same-host PCM across arbitrary ADTS chunk boundaries.

## Verification

The 2026-08-25 release run passed:

```sh
cargo test -p soundkit-aac --release --all-features
cargo test -p soundkit-aac --release --no-default-features --features owned-lc
cargo test -p soundkit-aac --release --no-default-features --features fdk
cargo test -p soundkit-aac --release --no-default-features --features mp4-decoder
cargo test -p soundkit-aac --release --no-default-features --features mp4-fdk-fallback
cargo test -p soundkit-aac-lc --release
cargo test -p aac-wasm-bench --release \
  --no-default-features --features fdk,soundkit-lc -- --nocapture
```

The SoundKit-vs-FDK music oracle passed equal-length checks and the enforced
RMSE, mean-error, maximum-error, and 35 dB minimum-SNR gates. WESTSIDE measured
46.783 dB SNR; the 44.1 kHz stereo music fixture measured 37.865 dB.

## Reproduce the native comparison

```sh
cargo build -p soundkit-aac --release --all-features \
  --example bench_adts_decode
./aac-wasm-bench/reference/build-ffmpeg-native-production.sh

taskset -c 0 target/release/examples/bench_adts_decode \
  golden/aac/WESTSIDE_MIX_4_CONFIRMATION_130323_256k.aac 3 soundkit
taskset -c 0 aac-wasm-bench/reference/ffmpeg-aac-production-bench \
  golden/aac/WESTSIDE_MIX_4_CONFIRMATION_130323_256k.aac 3 1
```

Run 11 rounds and alternate which implementation goes first. The benchmark
binary reports elapsed time, decoded frames, sample count, sample rate,
channels, backend, realtime factor, and a full-output checksum.
