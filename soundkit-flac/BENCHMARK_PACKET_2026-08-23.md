# Five-millisecond FLAC packet benchmark — 2026-08-23

This benchmark measures one persistent codec instance processing a varied
2.96-second speech sequence as 592 independent stereo S24 FLAC frames. Every
timed call contains 5 ms of PCM: 240 samples per channel at 48 kHz or 480 at
96 kHz.

`Realtime` is compared with libFLAC level 0; `Balanced` is compared with
libFLAC level 2. SoundKit packets are decoded by FFmpeg and libFLAC packets are
decoded by SoundKit before timing. All comparisons are bit-exact.

## Method

- Source: `testdata/flac/A_Tusk_is_used_to_make_costly_gifts.flac`, resampled
  to 48/96 kHz stereo S32LE and attenuated by 1/256 to retain signed 24-bit
  sample range.
- Persistent encoder/decoder state and caller-owned reusable buffers.
- 1,024 warm-up calls before each timed region.
- 20,000 calls per round on GCP; 10,000 on ARM.
- Three rounds with SoundKit, libFLAC, and FFmpeg order rotated each round.
- Each cell below is the median of that statistic across the three rounds.
- Timings are per-call microseconds; the timer read is included equally in all
  implementations.
- CPU-only, single-threaded codec paths; C references built with `-O3` and Rust
  with release optimization plus `target-cpu=native`.

Packet bundles use the same sequence and geometry on each side. Ratios shown
are actual bundle bytes divided by packed S24 PCM bytes, excluding the
four-byte benchmark-only packet-length prefix.

## GCP x86-64 results

Host: `yl-encodec-1`, `c4-highcpu-4`, four vCPUs on Intel Xeon Platinum 8581C
(Emerald Rapids), Debian 12. Rust/Cargo 1.97.1, FFmpeg 5.1.9, FLAC 1.4.2.

### Encode

| Rate/profile | SoundKit p50/p95/p99 µs | libFLAC p50/p95/p99 µs | SoundKit/libFLAC ratio | p50 change |
|---|---:|---:|---:|---:|
| 48 kHz Realtime / level 0 | 3.196 / 3.372 / 3.475 | 4.154 / 4.563 / 4.764 | 0.4800 / 0.4780 | 23.1% faster |
| 48 kHz Balanced / level 2 | 2.966 / 3.078 / 3.142 | 4.626 / 4.878 / 5.003 | 0.2464 / 0.2454 | 35.9% faster |
| 96 kHz Realtime / level 0 | 6.183 / 6.644 / 6.959 | 7.086 / 7.680 / 8.050 | 0.3183 / 0.3172 | 12.7% faster |
| 96 kHz Balanced / level 2 | 5.504 / 5.875 / 6.242 | 8.320 / 8.797 / 9.154 | 0.1625 / 0.1620 | 33.8% faster |

### Decode

| Rate/profile | SoundKit p50/p95/p99 µs | FFmpeg p50/p95/p99 µs | p50 change |
|---|---:|---:|---:|
| 48 kHz Realtime | 2.138 / 2.361 / 2.458 | 3.169 / 3.248 / 3.392 | 32.5% faster |
| 48 kHz Balanced | 1.156 / 1.277 / 1.325 | 1.977 / 2.050 / 2.099 | 41.5% faster |
| 96 kHz Realtime | 3.997 / 4.285 / 4.471 | 5.842 / 5.931 / 6.149 | 31.6% faster |
| 96 kHz Balanced | 2.109 / 2.280 / 2.346 | 3.383 / 3.434 / 3.577 | 37.7% faster |

The worst observed median-of-rounds SoundKit p99 was 6.959 µs for encode and
4.471 µs for decode, respectively 0.14% and 0.09% of the 5 ms PCM interval.

## Apple Silicon spot check

Host: Apple M1, macOS 26.5, Rust/Cargo 1.96.0, FFmpeg 8.1.2, FLAC 1.5.0.
The macOS monotonic timer exposed coarser whole-microsecond C results, so this
is a directional architecture check rather than the primary comparison.

| Rate/profile | SoundKit encode p50/p95/p99 µs | libFLAC p50/p95/p99 µs | SoundKit decode p50/p95/p99 µs | FFmpeg p50/p95/p99 µs |
|---|---:|---:|---:|---:|
| 48 kHz Realtime | 3.792 / 4.125 / 4.708 | 4 / 4 / 5 | 3.125 / 3.750 / 3.959 | 3 / 4 / 4 |
| 48 kHz Balanced | 3.125 / 3.334 / 3.584 | 4 / 5 / 6 | 1.583 / 1.792 / 1.958 | 2 / 3 / 3 |
| 96 kHz Realtime | 6.750 / 7.541 / 11.334 | 6 / 8 / 9 | 5.375 / 6.167 / 6.667 | 7 / 7 / 8 |
| 96 kHz Balanced | 5.916 / 6.542 / 6.834 | 8 / 10 / 11 | 2.958 / 3.458 / 3.584 | 4 / 4 / 5 |

The ARM check confirms the optimized path is portable. The one notable result
is 96 kHz Realtime encode, where libFLAC level 0 is about 12.5% faster at p50;
`Balanced`, the recommended profile, remains faster for both encode and decode.

## Reproduction

Run the full 48/96 kHz, 1/2/8-channel, S16/S24 differential matrix:

```sh
scripts/check_packet_matrix.sh
```

Run the alternating real-sequence benchmark after preparing matching S32LE
corpora:

```sh
scripts/benchmark_packet_sequence.sh real-48k.s32le real-96k.s32le 20000 3
```
