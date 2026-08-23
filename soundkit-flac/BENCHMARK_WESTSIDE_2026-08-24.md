# Westside five-millisecond FLAC packet benchmark — 2026-08-24

This benchmark uses the full `WESTSIDE_MIX 4 CONFIRMATION_130323.wav`
master. The source is 195.625 seconds of stereo, 48 kHz, 24-bit PCM.

The test contains 39,124 complete 5 ms frames at each sample rate. A 48 kHz
frame contains 240 samples per channel. A 96 kHz frame contains 480 samples
per channel. The 96 kHz input is resampled from the same master.

`Realtime` is compared with libFLAC level 0. `Balanced` is compared with
libFLAC level 2. SoundKit packets are decoded by FFmpeg. libFLAC packets are
decoded by SoundKit. All decoded PCM is bit-exact.

## Method

- Keep one codec instance for all calls.
- Reuse the packet and PCM buffers.
- Run 1,024 warm-up calls before each timed region.
- Run 20,000 timed calls in each round.
- Rotate the SoundKit, libFLAC, and FFmpeg order in each round.
- Use five rounds on Apple M1 and three rounds on GCP.
- Report the median value for each statistic across the rounds.
- Use CPU-only, single-threaded codec paths.
- Build the C references with `-O3`.
- Build Rust with release optimization and `target-cpu=native`.

FFmpeg stores decoded S24 values left-aligned in S32. The corpus preparation
step divides those S32 values by 256. This operation restores the original
signed 24-bit sample values before the benchmark.

Timings below are microseconds per 5 ms call. A positive p50 change means that
SoundKit is faster. A negative change means that SoundKit is slower.

## Apple M1 results

Host: Apple M1, macOS 26.5, Rust and Cargo 1.96.0, FFmpeg 8.1.2, and FLAC
1.5.0. The C reference timer reports whole microseconds on this host. Treat
small differences as directional results.

### Encode

| Rate and profile | SoundKit p50 / p95 / p99 | libFLAC p50 / p95 / p99 | p50 change |
|---|---:|---:|---:|
| 48 kHz Realtime / level 0 | 3.791 / 4.000 / 4.167 | 4 / 4 / 5 | **5.2% faster** |
| 48 kHz Balanced / level 2 | 4.917 / 5.209 / 5.417 | 5 / 6 / 7 | **1.7% faster** |
| 96 kHz Realtime / level 0 | 7.208 / 7.667 / 8.000 | 7 / 8 / 9 | **3.0% slower** |
| 96 kHz Balanced / level 2 | 9.458 / 10.000 / 10.416 | 10 / 11 / 12 | **5.4% faster** |

### Decode

| Rate and profile | SoundKit p50 / p95 / p99 | FFmpeg p50 / p95 / p99 | p50 change |
|---|---:|---:|---:|
| 48 kHz Realtime | 3.292 / 3.708 / 3.917 | 3 / 4 / 4 | **9.7% slower** |
| 48 kHz Balanced | 3.292 / 3.667 / 3.875 | 3 / 4 / 4 | **9.7% slower** |
| 96 kHz Realtime | 6.416 / 6.875 / 7.250 | 7 / 7 / 8 | **8.3% faster** |
| 96 kHz Balanced | 6.458 / 6.875 / 7.125 | 7 / 7 / 7 | **7.7% faster** |

## GCP x86-64 results

Host: `yl-encodec-1`, `c4-highcpu-4`, four vCPUs on Intel Xeon Platinum
8581C (Emerald Rapids), Debian 12. Rust and Cargo are 1.97.1. FFmpeg is 5.1.9.
FLAC is 1.4.2.

### Encode

| Rate and profile | SoundKit p50 / p95 / p99 | libFLAC p50 / p95 / p99 | p50 change |
|---|---:|---:|---:|
| 48 kHz Realtime / level 0 | 3.594 / 3.861 / 3.966 | 4.838 / 5.423 / 5.725 | **25.7% faster** |
| 48 kHz Balanced / level 2 | 4.711 / 4.947 / 5.042 | 6.488 / 7.058 / 7.431 | **27.4% faster** |
| 96 kHz Realtime / level 0 | 6.896 / 7.387 / 7.678 | 8.822 / 9.649 / 10.051 | **21.8% faster** |
| 96 kHz Balanced / level 2 | 9.032 / 9.577 / 9.940 | 11.660 / 13.040 / 15.629 | **22.5% faster** |

### Decode

| Rate and profile | SoundKit p50 / p95 / p99 | FFmpeg p50 / p95 / p99 | p50 change |
|---|---:|---:|---:|
| 48 kHz Realtime | 2.247 / 2.580 / 2.746 | 3.406 / 4.274 / 4.836 | **34.0% faster** |
| 48 kHz Balanced | 2.222 / 2.512 / 2.658 | 3.406 / 4.196 / 4.682 | **34.8% faster** |
| 96 kHz Realtime | 4.262 / 4.677 / 4.882 | 6.302 / 7.333 / 8.109 | **32.4% faster** |
| 96 kHz Balanced | 4.356 / 4.682 / 4.824 | 6.328 / 7.364 / 8.158 | **31.2% faster** |

## Encoded size

The encoders produce nearly equal packet sizes for the same profile. Ratios
are encoded bytes divided by packed S24 PCM bytes.

| Rate and profile | SoundKit ratio | libFLAC ratio |
|---|---:|---:|
| 48 kHz Realtime / level 0 | 0.7933 | 0.7932 |
| 48 kHz Balanced / level 2 | 0.7788 | 0.7788 |
| 96 kHz Realtime / level 0 | 0.6701 | 0.6698 |
| 96 kHz Balanced / level 2 | 0.6549 | 0.6547 |

## Result

On GCP, SoundKit is 21.8% to 27.4% faster than libFLAC for encoding. It is
31.2% to 34.8% faster than FFmpeg for decoding.

On Apple M1, encoding ranges from 3.0% slower to 5.4% faster. Decoding is 9.7%
slower at 48 kHz and 7.7% to 8.3% faster at 96 kHz. Every p99 result remains
far below the 5,000 microsecond PCM interval.
