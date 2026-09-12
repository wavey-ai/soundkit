# SoundKit Stream

SoundKit v2 frame-stream encoding and sidecar-index tooling, part of the
[`soundkit`](https://github.com/wavey-ai/soundkit) workspace.

A **stream** is a continuous concatenation of SoundKit v2 frames. A **sidecar
index** maps decoded PCM frames to byte offsets so a player can range-fetch and
seek without holding the whole stream. The format itself is described in
[`docs/soundkit-stream-format.md`](../../docs/soundkit-stream-format.md).

- `*.soundkit` — a continuous SoundKit v2 frame stream.
- `*.soundkit.idx` — a compact sidecar byte-offset index.
- `*.soundkit.json` — optional metadata for object stores and players.

The player fetches the sidecar index first, binary-searches by target PCM frame,
then range-fetches the stream from the indexed byte offset. To seek to an exact
time, decode from the nearest earlier frame boundary and trim decoded samples to
the requested target frame.

## Codecs

One interleaved i16 PCM pass is cut into both streams:

- **Opus** — the lossy house profile, from [`soundkit-opus`](../soundkit-opus).
- **FLAC** — the lossless copy, from [`soundkit-flac`](../soundkit-flac).

Each packet is wrapped in a SoundKit v2 header from `frame-header`, and each
stream carries one sidecar index entry per packet.

## Layout

- `soundkit-stream` — the Rust core encoder and index codec.
- `soundkit-stream-wasm` — the `wasm-bindgen` wrapper.
- `packages/soundkit-stream-js` — the JavaScript parser, frame scanner, sidecar
  index utilities, and WASM loader helpers.
- `workers/soundkit-store` — the Cloudflare Worker storage API for a stream and
  its index.

## Rust API

```rust
use soundkit_stream::{
    encode_interleaved_i16_to_soundkit_streams, PcmI16StreamOptions,
};

let encoded = encode_interleaved_i16_to_soundkit_streams(
    &pcm,
    PcmI16StreamOptions::default(),
)?;

let opus_stream = encoded.opus.stream;
let opus_index = encoded.opus.index_bytes()?;
let flac_stream = encoded.flac.stream;
let flac_index = encoded.flac.index_bytes()?;
```

`encode_interleaved_i16_to_opus_soundkit_stream` remains available when only the
lossy stream is wanted.

## JavaScript Usage

```js
import {
  decodeSoundKitIndex,
  frameForTimeSeconds,
  soundKitRangeRequestForFrame
} from "@wavey/soundkit-stream";

const indexBytes = new Uint8Array(await (await fetch("/audio.soundkit.idx")).arrayBuffer());
const index = decodeSoundKitIndex(indexBytes);
const targetFrame = frameForTimeSeconds(42.25, index.timescale);
const request = soundKitRangeRequestForFrame(index, targetFrame);

const streamResponse = await fetch("/audio.soundkit", {
  headers: request.headers
});
```

## WASM Usage

After building the WASM package:

```js
import init, * as wasm from "./pkg/soundkit_stream_wasm.js";
import { encodePcmI16ToSoundKitStreamsWithWasm } from "@wavey/soundkit-stream/wasm";

await init();

const encoded = encodePcmI16ToSoundKitStreamsWithWasm(wasm, pcmInt16Array, {
  sampleRate: 48_000,
  channels: 2,
  bitrate: 128_000,
  frameSize: 960
});

const opusStream = encoded.opusStream;
const opusIndex = encoded.opusIndex;
const flacStream = encoded.flacStream;
const flacIndex = encoded.flacIndex;
const metadata = JSON.parse(encoded.metadataJson());
```

For arbitrary source files, compose the SoundKit decoder WASM with the stream
encoder WASM:

```js
import {
  transcodeSourceToSoundKitStreamsWithWasm
} from "@wavey/soundkit-stream/transcode";

const result = await transcodeSourceToSoundKitStreamsWithWasm({
  decoderModule: soundkitDecoderWasm,
  encoderModule: soundkitStreamWasm,
  chunks: [sourceFileBytes],
  sampleRate: 48_000,
  bitrate: 128_000,
  frameSize: 960
});
```

The transcode helper expects the decoder to emit 48 kHz 16-bit PCM; the stream
encoder normalizes to the geometry the format carries.

## Development

```sh
npm run test:js
npm run test:rust
cargo build -p soundkit-stream-wasm --target wasm32-unknown-unknown
```

`wasm-pack` packaging is exposed as:

```sh
npm run build:wasm
```
