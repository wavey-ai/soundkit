# soundkit-flac

`soundkit-flac` is a pure-Rust FLAC frame codec for low-latency packet audio.
Its primary workload is one independently decodable 5 ms PCM frame per call at
48 or 96 kHz: 240 or 480 samples per channel. The optimized path supports one
to eight channels of signed 16-bit or 24-bit PCM.

Each encoded payload is exactly one raw FLAC frame. It has no `fLaC` marker or
`STREAMINFO`; the surrounding track or transport must carry the fixed sample
rate, channel count, sample depth, and frame length out of band.

## Raw packet API

Keep one encoder and decoder per track and reuse the packet and PCM buffers.
After their initial allocations, the `*_into` methods do not allocate in the
steady-state 48/96 kHz hot path.

```rust
use soundkit_flac::{
    FlacFrameConfig, FlacFrameDecoder, FlacFrameEncoder, FlacProfile,
};

let sample_rate = 48_000;
let frame_length = sample_rate / 200; // 5 ms: 240 samples per channel
let config = FlacFrameConfig::new(
    sample_rate,
    2,
    24,
    frame_length,
    FlacProfile::Balanced,
)?;

let mut encoder = FlacFrameEncoder::new(config)?;
let mut decoder = FlacFrameDecoder::new(config)?;
let pcm = vec![0_i32; config.sample_count()?];
let mut packet = Vec::new();
let mut decoded = vec![0_i32; config.sample_count()?];

encoder.encode_i32_into(&pcm, &mut packet)?;
let written = decoder.decode_into(&packet, &mut decoded)?;

assert_eq!(written, pcm.len());
assert_eq!(decoded, pcm);
# Ok::<(), Box<dyn std::error::Error>>(())
```

`encode_i16_into` accepts interleaved `i16`; `encode_s24le_into` accepts packed
interleaved S24LE. `encode_i32_into` clips samples to the configured 16- or
24-bit range. Decoding can write directly to reusable interleaved `i32`, `i16`,
or packed S24LE storage. The `*_block_into` variants are reserved for a shorter
final transport packet.

`Balanced` is the normal choice and matches libFLAC compression level 2's
fixed-predictor, partitioned-Rice, and stereo-decorrelation strategy.
`Realtime` matches libFLAC compression level 0 by using the same fixed
predictors and Rice analysis with independent channel coding.
`Maximum` retains the generic LPC encoder for compatibility and is outside the
specialized 5 ms path.

The encoder always writes valid FLAC CRC-8 and CRC-16 checksums. Raw packet
decoding skips checksum verification by default, matching FFmpeg's packet
decoder and assuming the transport supplies integrity protection. Enable it
when packets are not otherwise protected:

```rust
decoder.set_verify_checksums(true);
```

## Native FLAC compatibility

The explicit `stream::{Encoder, Decoder}` API remains available for native FLAC
streams, incremental input, `STREAMINFO`, and MD5 tracking. The SoundKit
`FlacEncoder` adapter emits raw packet frames; `FlacDecoder` remains a native
incremental decoder for existing import callers. Native whole-stream support is
compatibility functionality, not the latency-optimized usage model.

## Verification and benchmarks

Run the complete crate test suite with:

```sh
cargo test -p soundkit-flac --all-targets
```

Run the packet benchmark directly with:

```sh
cargo run --release -p soundkit-flac --example flac_frame_cost -- \
  48000 balanced 50000
```

`examples/flac_packet_fixture.rs` produces identical PCM and packet fixtures
for the direct libFLAC encode and FFmpeg decode harnesses in `scripts/`.

See [`BENCHMARK_WESTSIDE_2026-08-24.md`](BENCHMARK_WESTSIDE_2026-08-24.md) for
the full-song Apple M1 and GCP results.
