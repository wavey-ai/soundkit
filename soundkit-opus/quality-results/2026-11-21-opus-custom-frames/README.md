# Opus Custom 1, 1.33 and 1.5 ms frames: 2026-11-21

This report measures delay, quality, CPU and bytes for the Opus Custom frame
sizes against standard 2.5 and 5 ms Opus. The interop runs were on
2026-11-21. The measurements ran on 2026-11-22 (UTC).

## Result

A 1 ms frame cuts the frame plus codec delay from 5 ms (2.5 ms frames) or
7.5 ms (5 ms frames) to 2 ms. It needs about 64 kb/s more than a 2.5 ms frame for the same
PEAQ advanced Distortion Index.

At 320 kb/s CBR, 1 ms frames scored at least as well as the shipped SoundKit
production setting, 5 ms frames at 192 kb/s CBR, on all three measures:

| Configuration | PEAQ advanced DI, mean | ViSQOL MOS-LQO, mean | ViSQOL MOS-LQO, worst excerpt |
|---|---:|---:|---:|
| 5 ms, 192 kb/s CBR (the bar) | 2.50 | 4.721 | 4.693 |
| 1 ms, 320 kb/s CBR | 2.87 | 4.723 | 4.699 |
| 1 ms, 256 kb/s CBR | 2.29 | 4.708 | 4.646 |
| 2.5 ms, 192 kb/s CBR | 2.18 | 4.711 | 4.648 |

The recommended low-latency default is therefore 48-sample (1 ms) Opus
Custom frames at 320 kb/s CBR, with the signalling byte on. The
[quality bar and choice](#quality-bar-and-choice) section gives the reasons.

The results are objective estimates. They do not replace a controlled
listening test.

## Opus Custom

Standard Opus (RFC 6716) has no frame shorter than 2.5 ms. libopus's Opus
Custom API (`opus_custom_mode_create`) builds CELT modes for other frame
sizes. This crate implements those modes; see the Opus Custom section of
[CODEC.md](../../CODEC.md#opus-custom-modes-1-133-and-15-ms).

libopus sets these limits. They were checked against `celt/modes.c` at
xiph/opus commit `316c675`:

- The frame must last at least 1 ms: `frame_size * 1000 >= Fs`. At 48 kHz,
  48 samples is the shortest frame.
- The frame size must be even and from 40 to 1024 samples. A short block
  must last at most 3.3 ms.
- 48, 64 and 72-sample frames get `LM = 0`. Each frame is one short block.
  There are no transient short blocks, no dual stereo and no anti-collapse.
- The MDCT overlap equals the frame size: 48, 64 and 72 samples. The
  standard modes use 120 samples at every frame size.
- The modes have 15, 17 and 18 coded bands. The standard modes have 21.

Custom packets use the `opus_custom_encode` format. A one-byte signalling
header carries the mode and channel count. There is no Opus TOC byte.

## Interop proof

With its default libopus arithmetic and decision constants, the Rust codec
equals libopus's custom encoder and decoder. The comparison used libopus
built with `-DOPUS_CUSTOM_MODES=ON` at commit `316c675`, complexity 10, on
x86-64.

| Check | Cases | Result |
|---|---:|---|
| Four rate and format matrices: 4 kb/s to 510 kb/s and maximum, CBR and both VBR modes, s16, s24 and float, signalling on and off, 48 to 240 samples | 3480 | 17,321,352 of 17,321,352 packets identical; every decoded sample identical, with and without losses |
| Low and extreme rates, 501 b/s to 2 Mb/s, 48, 64 and 72 samples | 2352 | 4,978,288 packets and every decoded sample identical |
| Packet limits from 2 to 1276 bytes | 8448 | 15,650,216 packets and every decoded sample identical |
| Corrupted streams | 576 | Same 66,417 packets rejected; 61,966,264 decoded samples identical |

The 3480 matrix cases check four things: Rust packets through both
decoders, libopus packets through both decoders, Rust packets against
libopus packets, and libopus packets with losses through both decoders.
The section
[Interop with libopus custom modes](../../CODEC.md#interop-with-libopus-custom-modes)
in CODEC.md gives the full method and the remaining exceptions.

## Results

Each row is a mean over 16 excerpts. "Worst" is the lowest-scoring excerpt.
Higher is better for every quality measure. Delay is in samples at 48 kHz.

CPU is milliseconds of CPU time on one core per second of stereo audio. Each
value is the median of seven rounds. Bytes are per second of audio.
Packets/s is also the AEP1/UDP datagram rate. The header column adds the
16-byte SoundKit v2 packet header. It does not include AEP1, UDP, IP or FEC
overhead.

All rows use CBR, the production mode. The CSV files also hold the
constrained-VBR rows. For the custom sizes at 192 kb/s and above, constrained
VBR is within noise of CBR.

### 192, 256 and 320 kb/s

| Frame | kb/s | Codec delay | Frame + delay | PEAQ adv DI | PEAQ adv ODG | PEAQ basic ODG | ViSQOL mean | ViSQOL worst |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 ms (48) | 192 | 48 | 96 | 1.26 | -0.71 | -0.22 | 4.607 | 4.151 |
| 1.33 ms (64) | 192 | 64 | 128 | 1.62 | -0.48 | -0.16 | 4.604 | 4.124 |
| 1.5 ms (72) | 192 | 72 | 144 | 1.68 | -0.44 | -0.14 | 4.574 | 4.258 |
| 2.5 ms (120) | 192 | 120 | 240 | 2.18 | -0.21 | 0.01 | 4.711 | 4.648 |
| 5 ms (240) | 192 | 120 | 360 | 2.50 | -0.10 | 0.07 | 4.721 | 4.693 |
| 1 ms (48) | 256 | 48 | 96 | 2.29 | -0.17 | -0.01 | 4.708 | 4.646 |
| 1.33 ms (64) | 256 | 64 | 128 | 2.53 | -0.09 | 0.07 | 4.716 | 4.658 |
| 1.5 ms (72) | 256 | 72 | 144 | 2.59 | -0.07 | 0.08 | 4.718 | 4.673 |
| 2.5 ms (120) | 256 | 120 | 240 | 2.85 | -0.01 | 0.08 | 4.725 | 4.709 |
| 5 ms (240) | 256 | 120 | 360 | 2.97 | 0.02 | 0.08 | 4.727 | 4.719 |
| 1 ms (48) | 320 | 48 | 96 | 2.87 | -0.00 | 0.08 | 4.723 | 4.699 |
| 1.33 ms (64) | 320 | 64 | 128 | 2.97 | 0.01 | 0.07 | 4.727 | 4.710 |
| 1.5 ms (72) | 320 | 72 | 144 | 3.01 | 0.02 | 0.08 | 4.727 | 4.715 |
| 2.5 ms (120) | 320 | 120 | 240 | 3.09 | 0.04 | 0.08 | 4.728 | 4.719 |
| 5 ms (240) | 320 | 120 | 360 | 3.13 | 0.04 | 0.09 | 4.729 | 4.724 |

| Frame | kb/s | Encode native | Decode native | Encode WASM | Decode WASM | Payload B/s | Packets/s | With header B/s (kb/s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 ms | 192 | 13.5 | 5.3 | 24.5 | 10.0 | 24000 | 1000.0 | 40000 (320.0) |
| 1.33 ms | 192 | 21.4 | 5.3 | 51.1 | 7.5 | 24000 | 750.0 | 36000 (288.0) |
| 1.5 ms | 192 | 21.4 | 5.1 | 53.1 | 7.2 | 24000 | 666.7 | 34667 (277.3) |
| 2.5 ms | 192 | 18.3 | 4.0 | 26.2 | 6.0 | 24000 | 400.0 | 30400 (243.2) |
| 5 ms | 192 | 14.0 | 3.3 | 20.5 | 4.7 | 24000 | 200.0 | 27200 (217.6) |
| 1 ms | 256 | 26.3 | 5.9 | 65.8 | 9.9 | 32000 | 1000.0 | 48000 (384.0) |
| 1.33 ms | 256 | 23.8 | 5.7 | 55.7 | 8.4 | 32250 | 750.0 | 44250 (354.0) |
| 1.5 ms | 256 | 23.1 | 5.6 | 55.1 | 7.7 | 32000 | 666.7 | 42667 (341.3) |
| 2.5 ms | 256 | 19.9 | 4.4 | 28.7 | 6.1 | 32000 | 400.0 | 38400 (307.2) |
| 5 ms | 256 | 14.8 | 3.8 | 21.8 | 4.9 | 32000 | 200.0 | 35200 (281.6) |
| 1 ms | 320 | 30.2 | 6.4 | 64.8 | 9.9 | 40000 | 1000.0 | 56000 (448.0) |
| 1.33 ms | 320 | 25.1 | 6.0 | 60.5 | 8.7 | 39750 | 750.0 | 51750 (414.0) |
| 1.5 ms | 320 | 23.8 | 5.7 | 58.9 | 9.2 | 40000 | 666.7 | 50667 (405.3) |
| 2.5 ms | 320 | 19.7 | 4.7 | 31.5 | 6.5 | 40000 | 400.0 | 46400 (371.2) |
| 5 ms | 320 | 15.7 | 4.0 | 23.4 | 5.5 | 40000 | 200.0 | 43200 (345.6) |

### 96, 128 and 384 kb/s

| Frame | kb/s | PEAQ adv DI | PEAQ adv ODG | PEAQ basic ODG | ViSQOL mean | ViSQOL worst | Encode native | Decode native | Encode WASM | Decode WASM | Payload B/s | With header B/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 ms | 96 | -1.33 | -2.92 | -2.74 | 3.987 | 3.694 | 10.3 | 4.6 | 15.3 | 8.8 | 12000 | 28000 |
| 1.33 ms | 96 | -2.42 | -3.60 | -2.05 | 4.177 | 3.821 | 9.5 | 4.3 | 16.5 | 7.1 | 12000 | 24000 |
| 1.5 ms | 96 | -1.40 | -3.13 | -1.56 | 4.190 | 3.822 | 9.4 | 4.3 | 15.9 | 7.3 | 12000 | 22667 |
| 2.5 ms | 96 | -0.17 | -2.05 | -0.81 | 4.390 | 4.007 | 13.8 | 3.4 | 20.9 | 5.2 | 12000 | 18400 |
| 5 ms | 96 | 0.86 | -1.03 | -0.29 | 4.517 | 3.938 | 11.7 | 2.8 | 17.7 | 4.0 | 12000 | 15200 |
| 1 ms | 128 | -0.65 | -2.52 | -1.22 | 4.226 | 3.876 | 11.9 | 5.0 | 17.5 | 7.9 | 16000 | 32000 |
| 1.33 ms | 128 | 0.01 | -1.87 | -0.70 | 4.480 | 4.165 | 11.2 | 4.6 | 17.0 | 6.7 | 15750 | 27750 |
| 1.5 ms | 128 | 0.23 | -1.65 | -0.52 | 4.457 | 4.053 | 10.9 | 4.5 | 16.3 | 6.5 | 16000 | 26667 |
| 2.5 ms | 128 | 0.91 | -0.99 | -0.28 | 4.528 | 3.992 | 16.4 | 3.7 | 23.7 | 5.0 | 16000 | 22400 |
| 5 ms | 128 | 1.53 | -0.53 | -0.14 | 4.584 | 4.203 | 12.7 | 3.0 | 16.9 | 3.7 | 16000 | 19200 |
| 1 ms | 384 | 3.10 | 0.04 | 0.08 | 4.728 | 4.721 | 28.0 | 6.4 | 61.1 | 9.6 | 48000 | 64000 |
| 1.33 ms | 384 | 3.12 | 0.04 | 0.10 | 4.729 | 4.725 | 25.4 | 6.2 | 55.1 | 8.7 | 48000 | 60000 |
| 1.5 ms | 384 | 3.13 | 0.04 | 0.11 | 4.729 | 4.723 | 24.2 | 5.9 | 55.3 | 8.5 | 48000 | 58667 |
| 2.5 ms | 384 | 3.16 | 0.05 | 0.09 | 4.729 | 4.725 | 20.0 | 4.8 | 27.7 | 6.7 | 48000 | 54400 |
| 5 ms | 384 | 3.20 | 0.06 | 0.09 | 4.730 | 4.725 | 16.3 | 4.3 | 21.5 | 5.7 | 48000 | 51200 |

Packets/s and delay depend only on the frame size, as in the first table.
PEAQ advanced saturates near DI -2.3, so it orders the 96 kb/s custom rows
unreliably. ViSQOL and PEAQ basic rank 1.33 ms above 1 ms there.

### Delay

The decoded stream lags the input by the MDCT overlap. The encoder also
buffers one frame. The decoder adds nothing.

| Frame | Codec delay | Frame + delay | Measured lag, music | Measured lag, click train |
|---|---:|---:|---|---|
| 1 ms (48) | 48 (1 ms) | 96 (2 ms) | 48 in 192 of 192 | 48 in 10 of 12 configurations |
| 1.33 ms (64) | 64 (1.33 ms) | 128 (2.67 ms) | 64 in 192 of 192 | 64 in 12 of 12 |
| 1.5 ms (72) | 72 (1.5 ms) | 144 (3 ms) | 72 in 192 of 192 | 72 in 12 of 12 |
| 2.5 ms (120) | 120 (2.5 ms) | 240 (5 ms) | 120 in 192 of 192 | 120 in 12 of 12 |
| 5 ms (240) | 120 (2.5 ms) | 360 (7.5 ms) | 120 in 192 of 192 | 120 in 12 of 12 |

The two 1 ms click-train misses were 96 kb/s CBR (40 samples) and 128 kb/s
constrained VBR (49 samples). Their 12 to 16-byte packets smear the clicks.
Every click-train configuration from 192 kb/s up gave the expected lag.

SoundKit's standard encoder has no Opus-layer delay compensation. Its 2.5 ms
delay is 120 samples. libopus's standard encoder reports 312 samples with
`OPUS_APPLICATION_AUDIO`. Opus Custom has no Opus layer.

These values are algorithmic delay. A real chain also waits for encode and
decode CPU time, any jitter buffer and the network. Natively, one 1 ms frame
took about 10-30 us to encode and 5-7 us to decode.

### Signalling byte

Custom packets carry one signalling byte by default. With signalling off at
the same `set_bitrate`, packets keep the same size. libopus gives the byte to
the CELT frame.

Signalling off at a bitrate lower by 8 bits per packet gave bit-identical
decoded audio. This held for all 16 excerpts in every CBR configuration and
in every constrained-VBR configuration from 192 kb/s. The byte therefore
costs 8 kb/s at 1 ms, 6 kb/s at 1.33 ms and 5.33 kb/s at 1.5 ms.

## Quality bar and choice

The bar: the lane default must be at least as good as the shipped SoundKit
Opus production setting. That setting is standard 5 ms frames at 192 kb/s
CBR, the bottom of the 192-320 kb/s production range. The comparison uses
three measures over the 16 excerpts: mean PEAQ advanced DI (2.50), mean
ViSQOL MOS-LQO (4.721) and worst-excerpt ViSQOL MOS-LQO (4.693).

| Configuration | PEAQ adv DI | ViSQOL mean | ViSQOL worst | Result |
|---|---:|---:|---:|---|
| 1 ms, 320 kb/s CBR | 2.87 | 4.723 | 4.699 | Passes |
| 1 ms, 256 kb/s CBR | 2.29 | 4.708 | 4.646 | Fails |
| 1.33 ms, 256 kb/s CBR | 2.53 | 4.716 | 4.658 | Fails on the worst excerpt |
| 1.5 ms, 256 kb/s CBR | 2.59 | 4.718 | 4.673 | Fails on the worst excerpt |
| 1.33 ms, 320 kb/s CBR | 2.97 | 4.727 | 4.710 | Passes, larger frame |
| 1.5 ms, 320 kb/s CBR | 3.01 | 4.727 | 4.715 | Passes, larger frame |

The smallest acceptable frame is 1 ms (48 samples) at 320 kb/s CBR.

1 ms at 256 kb/s equals standard 2.5 ms at 192 kb/s: PEAQ advanced DI 2.29
against 2.18, and ViSQOL 4.708 against 4.711. It is the documented
lower-bandwidth option.

The lane default is 48-sample Opus Custom frames, 320000 b/s CBR, with
custom signalling on, the libopus default. Every SoundKit and libopus
decoder default then decodes lane packets without extra configuration. This
includes the browser WebAssembly decoder, built with `new Decoder(channels,
48000, frameSize)`. The signalling byte also checks LM and the channel count.
It costs 8 kb/s.

Standard 2.5 ms (120-sample) Opus stays the interop setting. Use it for
anything in fMP4 or Ogg, WebRTC and RTP, talkback, the public programme
rendition and other players.

## Findings

- **The pitch prefilter doubles custom encode CPU.** libopus, and this port,
  run the prefilter pitch search only when a frame has more than 12 bytes
  per channel (`celt_encoder.c`, `nbAvailableBytes > 12*C`). In stereo at
  1 ms, that is above about 200 kb/s CBR. At 1.33 ms it is above about
  150 kb/s, and at 1.5 ms above about 133 kb/s. Constrained VBR crosses it
  at lower rates. The search costs about the same per frame at every frame
  size, so it roughly doubles the 1 ms encoder: 13.5 ms/s at 192 kb/s CBR,
  26.3 ms/s at 256 kb/s. Standard 2.5 and 5 ms frames are above the
  threshold at every rate measured. This explanation comes from the source
  and fits the data. It was not profiled.
- **WebAssembly custom encoding is slower relative to native.** Where the
  pitch search runs, the custom encoder took 2.1-2.7 times its native CPU
  time in WebAssembly. The standard encoder took 1.3-1.6 times. Custom
  modes run libopus arithmetic by default, with the portable scalar
  summation kernels in `src/celt/refkern.rs`. Those kernels are the likely
  cause. This was not profiled.
- **Standard constrained VBR runs 1 byte per packet over target.** At every
  rate, standard constrained VBR wrote 1.00-1.04 bytes per packet more than
  the target: 401-404 B/s at 2.5 ms and 203-208 B/s at 5 ms. The TOC byte
  may not be charged to the VBR budget. Standard packets on this branch
  equal origin/main's, so the behaviour is pre-existing. It was not compared
  with libopus here. Custom constrained VBR was within 2 B/s of target,
  except 64-sample frames at 128, 256 and 320 kb/s (0.16-0.22 bytes per
  packet under).
- **64-sample CBR packets are whole bytes.** 64-sample CBR wrote 15750 B/s
  at 128 kb/s (-1.6%), 32250 B/s at 256 kb/s (+0.8%) and 39750 B/s at
  320 kb/s (-0.6%).

## Method

- Codec: `soundkit-opus` on branch `opus-custom-1ms` at
  `4cc934c3cb5196c79b97a6b2e4e67ab8409a88eb`, 48 kHz stereo, signed 24-bit
  PCM through the i24 API.
- Custom sizes (48, 64, 72): `Encoder::new_custom(48000, 2, N)` with
  `set_bitrate(R)`, and `Decoder::new_for_frame_count(48000, 2, N)`. Defaults
  were kept: libopus decisions and arithmetic, complexity-10 analysis, the
  x86-64 PVQ search and the signalling byte. These packets equal
  `Encoder::try_new(48000, 24, 2, N, R)`'s.
- Standard sizes (120, 240): `Encoder::try_new(48000, 24, 2, N, R)`, the
  SoundKit encoder with `OPUS_APPLICATION_AUDIO`, SoundKit decisions and CBR.
- Constrained VBR added `set_vbr(true)`. Rates were 96, 128, 192, 256, 320
  and 384 kb/s, for 60 configurations.
- Source: eight Lori Asha confirmation mixes (owner's masters, not
  distributed). Two random 8.1-second excerpts per track, seed `20261121`,
  5-second edge margin, for 16 excerpts.
- Quality: each decode dropped the known codec delay and scored the first
  8.000 s. PEAQ used GstPEAQ commit `7397e49`, basic and advanced, on 32-bit
  float WAV. ViSQOL Audio used commit
  `38d0b0163e441047d4429bf07ad09e5b9031d02c`, nu-SVR model, on PCM16 WAV with
  0.1 dB headroom. This is the method of the earlier reports. All 960 pairs
  scored on the first attempt. The ViSQOL ceiling (identity) on this corpus
  is 4.7321.
- Delay: cross-correlation of each decoded signal with its input, for every
  music decode (960) and for a click train (80 windowed clicks, 60
  configurations).
- CPU: one excerpt per track, 64.8 s of audio per configuration. Seven
  rounds; the order of frame sizes rotated each round. A fresh encoder and
  decoder per excerpt. Thread CPU time over the encode or decode calls only.
  Native and WebAssembly each ran pinned to one core at `nice -n 10`.
- Native build: `cargo build --release`, the workspace profile
  (`codegen-units = 1`, `lto = "thin"`), the x86-64 baseline target, rustc
  1.100.0. Calls: `encode_i24_into` and `decode_i24_into_slice`.
- WebAssembly build: the npm release flags (fat LTO, `+simd128`,
  `no-vectorize-slp`, `--features wasm`), then `wasm-bindgen 0.2.130 --target
  web`, with no `wasm-opt` pass. Node.js v26.13.1 ran the JS constructors
  `new Encoder(2, 48000, R, N)` and `new Decoder(2, 48000, N)` with
  `enc_frame_i24_reuse` and `dec_frame_i24_reuse`, copying PCM in and
  packets out.
- Host: AMD EPYC 7713 VM, 32 vCPUs, shared with other workloads.
- Bytes: all 16 excerpts of the quality run, 129.6 s per configuration.

## Limits

- Objective metrics only. There was no listening test.
- The corpus is 16 excerpts from eight masters by one artist. A worst-excerpt
  value is a single item, not a percentile.
- The CPU host was a shared VM without frequency control. Other workloads ran
  during the measurements: the load average was 3.3-41.7 (mean 15.7) during
  the native run and 3.3-16.6 (mean 8.7) during the WebAssembly run. CPU time
  is less sensitive to this than wall time, but not immune. Treat single
  maxima as contention.
- Custom modes use libopus's decision constants, the production default.
  Standard modes use SoundKit's tuned constants. Differences between sizes
  include that difference.
- WebAssembly figures have no `wasm-opt` pass and ran in Node.js, not a
  browser.
- The custom encoder's default x86-64 PVQ search and float differences make
  some WebAssembly packets differ from native ones. Total bytes differed by
  1-2 in 5 of the constrained-VBR configurations. CBR totals are equal.
- PEAQ advanced saturates near DI -2.3 (ODG -3.6). PEAQ basic does not
  separate 256 and 320 kb/s.

## Included Data

These files hold aggregate values per configuration. Each has 60 rows: five
frame sizes, six rates, CBR and constrained VBR.

- `scores.csv`: codec delay, PEAQ advanced and basic means, worst-excerpt
  PEAQ and ViSQOL values, ViSQOL mean and mean SNR.
- `cpu.csv`: native and WebAssembly encode and decode CPU, in ms per second
  of audio, as the median, minimum and maximum over seven rounds.
- `bytes.csv`: payload bytes/s, packets/s, packet size range, bytes/s with
  the 16-byte SoundKit v2 header, and bytes/s with signalling off at the
  same audio quality.

The repository does not include source audio, decoded audio, packets or
per-excerpt results. The source masters are not distributed.

## Reproduction

The measurement drivers live in a local lab directory, not in this
repository. The paths below are local to the measuring machine.

- `measure/src/main.rs` builds `skmeasure`. `skmeasure rt` round-trips a
  file and `skmeasure bench` times encode and decode. It depends on this
  crate by path.
- `measure/run_quality.py decode`, `xcorr` and `score` encode, decode,
  measure delay and score every configuration. Scoring follows the method
  of the earlier reports in this directory.
- `measure/delay.py` builds the click train and measures lag.
- `measure/sigcheck.sh` checks the signalling-off equivalence.
- `measure/run_cpu.sh` and `measure/wasm_bench.mjs` run the native and
  WebAssembly CPU rounds.
- `measure/summarize.py` writes the per-configuration summary that the CSV
  files here come from.

The interop matrix runs from this repository with
`tools/run_opus_custom_interop.sh`; see CODEC.md.
