# soundkit-flac

SoundKit's FLAC adapter uses the standalone
[`wavey-ai/flac`](https://github.com/wavey-ai/flac) pure-Rust codec for both
encoding and decoding. The dependency follows its `main` branch; SoundKit no
longer selects separate Claxon, flacenc, OxideAV, or libFLAC backends.

The default feature is `wavey-codec`. The `packet-codec` feature enables the
same codec and exposes its raw-frame API for transports and fragmented MP4.

```toml
soundkit-flac = { path = "../soundkit-flac" }
```

The adapter retains SoundKit's `audio_packet::Encoder` and `Decoder` traits and
re-exports the standalone codec's frame types:

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

Run the adapter tests with:

```sh
cargo test -p soundkit-flac
```

The standalone codec owns the full corpus, differential, and FFmpeg benchmark
suites. See its `TODO.md` for the current quality and performance handoff.
