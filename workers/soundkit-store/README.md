# SoundKit Store Worker

Rust Cloudflare Worker scaffold for live SoundKit stream ingest.

The design mirrors `store-stream` behavior, but uses:

- R2 for audio bytes.
- D1 for the mutable upload manifest, chunk table, and SoundKit frame index.
- A sealed R2 sidecar index once upload is finalized.

Chunks must be aligned to complete SoundKit v2 frame boundaries. The encoder
should treat `targetChunkBytes` as a target and flush only between complete
Opus/FLAC SoundKit frames.

## API

Create an upload:

```sh
curl -X POST https://store.example.com/v1/objects/track-1/uploads \
  -H 'content-type: application/json' \
  --data '{"targetChunkBytes":262144,"opusFrameMs":20,"timescale":48000}'
```

Append a SoundKit-frame-aligned chunk:

```sh
curl -X POST https://store.example.com/v1/objects/track-1/chunks \
  -H 'content-type: application/octet-stream' \
  --data-binary @chunk.soundkit
```

Read a live range:

```sh
curl https://store.example.com/v1/objects/track-1/stream \
  -H 'range: bytes=1048576-1310719'
```

Fetch the current sidecar index:

```sh
curl https://store.example.com/v1/objects/track-1/index \
  -o stream.soundkit.idx
```

Find the closest byte offset for a playback time:

```sh
curl 'https://store.example.com/v1/objects/track-1/seek?timeMs=12345'
```

Seal upload state and persist manifest/index objects into R2:

```sh
curl -X POST https://store.example.com/v1/objects/track-1/seal
```

## R2 Layout

```text
objects/{objectId}/chunks/0000000000.soundkit
objects/{objectId}/chunks/0000000001.soundkit
objects/{objectId}/stream.soundkit.idx
objects/{objectId}/manifest.json
```

The Worker does not compact chunks into one contiguous `stream.soundkit` object
yet. For high-scale delivery, add an asynchronous compaction job. The job must
read committed chunks and write the sealed stream object. Then, serve that
object directly through Cloudflare cache.

## Development

```sh
npm install
cargo install worker-build
npm run check -w @wavey/soundkit-store-worker
npm run dev -w @wavey/soundkit-store-worker
```

Before deploying, create the R2 bucket and D1 database, then replace
`database_id` in `wrangler.jsonc`.
