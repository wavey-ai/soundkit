# soundkit-flac

## Default stream encoder

The default `FlacEncoder` uses `flacenc`. It emits complete FLAC frames and updates the final `STREAMINFO` block.

Enable `oxideav-encoder` only when you must test the compatibility backend.

## Verification

Run the default FLAC tests:

```sh
cargo test -p soundkit-flac --lib
```

Run the `flacenc` streaming round-trip test without default features:

```sh
cargo test -p soundkit-flac --no-default-features \
  --features claxon-decoder,flacenc-encoder \
  test_flacenc_stream_encoder_roundtrip_with_a_short_final_block
```

Run the WASM PCM conversion and codec tests:

```sh
cargo test -p soundkit-wasm --lib
```

The PCM conversion test verifies normalized 24-bit samples. It includes positive and negative limit values.

The streaming test verifies a short final block. It also verifies the final `STREAMINFO` metadata and decoded PCM.

### Browser worker reference

The vin.yl.web worker benchmark uses the built SoundKit WASM package. It encodes and decodes every source frame.

`Source PCM` means the decoded master before FLAC encoding. `Codec PCM` means the quantized PCM that enters the FLAC encoder.

The reference run used a 148-second, 48 kHz, stereo, 24-bit PCM master on 2026-08-17.

| Measurement | Result |
| --- | ---: |
| FLAC encode time | 2.60 seconds |
| Encode rate | 57 times realtime |
| Previous OxideAV encode time | 339 seconds |
| FLAC size | 31.0 MB |
| FLAC bitrate | 1,674 kbit/s |
| FLAC size compared to raw PCM | 72.7% |
| Codec PCM differences | 0 of 14,208,000 samples |
| Source PCM differences | 0 of 14,208,000 samples |
| Decoded frames | 7,104,000 of 7,104,000 frames |

This measurement is a reference result. Hardware and browser changes can change the encode time.

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
32 samples. This API does not substitute a hidden 4096-sample block. Use
`reset()` on only the affected track when its format generation or continuity
segment changes.

Run the non-asserting component cost harness with:

```sh
cargo run -p soundkit-flac --release --no-default-features \
  --features packet-codec --example flac_frame_cost
```

Use `packet-codec` when a transport needs independent FLAC frames. This feature also uses the `flacenc` core.
