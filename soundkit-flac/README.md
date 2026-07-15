# soundkit-flac

## Pure-Rust low-latency frames

Enable `packet-codec` without default features to encode independent FLAC
frames without libFLAC or another native dependency:

```toml
soundkit-flac = {
  path = "../soundkit-flac",
  default-features = false,
  features = ["packet-codec"]
}
```

The surrounding transport owns timestamps and format generations. Each track
owns one encoder and decoder, and every returned payload is one independently
decodable raw FLAC frame:

```rust
use soundkit_flac::{
    FlacFrameConfig, FlacFrameDecoder, FlacFrameEncoder, FlacProfile,
};

let config = FlacFrameConfig::new(48_000, 2, 24, 240, FlacProfile::Realtime)?;
let mut encoder = FlacFrameEncoder::new(config)?;
let mut decoder = FlacFrameDecoder::new(config)?;
let input = vec![0_u8; config.raw_pcm_bytes()?];

let encoded = encoder.encode_s24le(&input)?;
let decoded = decoder.decode(&encoded.payload)?;
assert_eq!(decoded.to_s24le()?, input);
# Ok::<(), Box<dyn std::error::Error>>(())
```

`240` samples at 48 kHz is a 5 ms frame. FLAC itself supports blocks down to
32 samples; this API does not substitute a hidden 4096-sample block. Use
`reset()` on only the affected track when its format generation or continuity
segment changes.

Run the non-asserting component cost harness with:

```sh
cargo run -p soundkit-flac --release --no-default-features \
  --features packet-codec --example flac_frame_cost
```

The legacy default features remain available for existing users. New
latency-sensitive code should disable defaults and select `packet-codec` so the
codec path is pure Rust.
