# SoundKit WASM Streaming API

This document describes the browser-facing WASM API exposed by
`soundkit-wasm-decoder`.

The WASM package is designed around push-based streaming:

1. Create a decoder or deboxer.
2. Call `push(bytes)` with arbitrary byte chunks.
3. Process all returned frames/events immediately.
4. Call `flush()` exactly once at EOF.

After `flush()`, the instance is drained and should not be reused.

## Build

```sh
wasm-pack build soundkit-wasm-decoder --target web --out-dir pkg -- --offline
```

The generated browser package is written to:

```text
soundkit-wasm-decoder/pkg/
```

The default feature set is intended for the browser and avoids the known C
codec blockers for AAC and Opus decode:

```text
detect, raw-pcm, wav, mp3, vorbis, aiff, alac, flac, webm,
opus-debox, aac-debox, aac-lc, audio-demux
```

## Import

```js
import init, {
  WasmMusicDecoder,
  WasmAudioTrackDemuxer,
  WasmAacDeboxer,
  WasmAacLcDecoder,
  WasmOpusDeboxer,
} from "./soundkit-wasm-decoder/pkg/soundkit_wasm_decoder.js";

await init();
```

For synchronous initialization with already-loaded wasm bytes:

```js
import { initSync } from "./soundkit-wasm-decoder/pkg/soundkit_wasm_decoder.js";

initSync({ module: wasmBytes });
```

## General Streaming Contract

`push(bytes)` accepts a `Uint8Array` of at most 4 MiB. Split larger reads before
calling WASM. A call may return zero, one, or many output objects depending on
container buffering and codec frame availability.

`flush()` performs the final EOF drain. It may return additional output objects.
It should be called once when no more bytes are available.

Errors are thrown as JavaScript exceptions.

Automatic detection retains at most 64 KiB before choosing a format. It begins
detection at 8 KiB and forwards bytes beyond the retained probe immediately.

## PCM Decoder

Use `WasmMusicDecoder` when SoundKit should fully decode the audio to PCM inside
WASM.

```js
const decoder = WasmMusicDecoder.newWithFormat("mp3");
const pcmFrames = [];

for await (const chunk of encodedByteStream) {
  pcmFrames.push(...decoder.push(chunk));
}

pcmFrames.push(...decoder.flush());
```

Constructors:

```ts
new WasmMusicDecoder()
WasmMusicDecoder.newAuto()
WasmMusicDecoder.newWithFormat(format: string)
WasmMusicDecoder.newRawLinear16(sampleRate: number, channels: number)
WasmMusicDecoder.newRawLinear32(sampleRate: number, channels: number)
```

PCM frame shape:

```ts
type PcmFrame = {
  sampleRate: number;
  channels: number;
  bitsPerSample: number;
  data: Uint8Array;
};
```

`data` is interleaved little-endian PCM.

Supported `newWithFormat` strings depend on enabled Cargo features:

| Format string | Output | Default browser build |
| --- | --- | --- |
| `wav`, `wave` | PCM | yes |
| `mp3` | PCM | yes |
| `ogg`, `ogg-vorbis`, `vorbis` | PCM | yes |
| `webm` | PCM from WebM/Vorbis | yes |
| `aiff`, `aifc` | PCM | yes |
| `alac`, `caf-alac` | Rejected; use the seekable packet APIs | yes |
| `flac` | PCM | yes |
| `aac`, `adts` | PCM via FDK AAC | no |
| `m4a`, `mp4`, `aac-mp4` | PCM via FDK AAC | no |
| `opus` | PCM via native Opus path | no |
| `ogg-opus`, `opus-ogg` | PCM via native Opus path | no |

For ADTS/M4A AAC container streams, prefer the debox APIs below. For AAC-LC raw
access units in the SoundKit packet stream, use `WasmAacLcDecoder`.

The generic decoder rejects ALAC containers. Use the seekable ALAC APIs because
required MP4 or CAF metadata can follow a large media extent.

## AAC-LC Raw Access Unit Decoder

Use `WasmAacLcDecoder` when SoundKit already has AAC-LC access units and an
MPEG-4 `AudioSpecificConfig`. This is the Rust-built wasm path for the
SoundKit-owned AAC-LC decoder.

```js
const decoder = new WasmAacLcDecoder(audioSpecificConfig);

for await (const accessUnit of rawAacAccessUnits) {
  const pcm = decoder.decodeInterleaved(accessUnit);
  audioWorklet.enqueue(pcm, decoder.sampleRate, decoder.channels);
}
```

Constructor:

```ts
new WasmAacLcDecoder(audioSpecificConfig: Uint8Array)
```

Properties:

```ts
sampleRate: number
channels: number
framesPerAccessUnit: number
```

Methods:

```ts
decodeInterleaved(accessUnit: Uint8Array): Float32Array
decodeInterleavedInto(accessUnit: Uint8Array, output: Float32Array): number
decodePlanar(accessUnit: Uint8Array): Float32Array[]
```

`decodeInterleaved` returns interleaved `f32` PCM. `decodePlanar` returns one
`Float32Array` per channel. `decodeInterleavedInto` writes interleaved `f32` PCM
into a caller-owned `Float32Array` and returns the number of samples written,
allowing worker/player code to reuse output buffers.

Current AAC-LC wasm support is raw access-unit only. MP4/M4A deboxing is handled
by `WasmAacDeboxer` / `WasmAudioTrackDemuxer`; raw-block `END`/`FIL` elements
are handled. Unsupported SBR/HE-AAC, PS, program-config-element channels,
channel layouts beyond stereo, and SBR raw-block fill payloads fail with
explicit error messages so callers can route those streams to a fallback
decoder.

## Generic Audio Track Demuxing

Use `WasmAudioTrackDemuxer` when SoundKit should extract the first supported
audio track from a video/audio container and another decoder should consume the
encoded packets.

This is the preferred API for extracting audio from video containers.

```js
const demuxer = WasmAudioTrackDemuxer.newWithFormat("mp4");

for await (const chunk of containerByteStream) {
  for (const event of demuxer.push(chunk)) {
    if (event.type === "config") {
      configureDecoder(event);
    } else if (event.type === "packet") {
      encodedDecoder.decode(event.data);
    }
  }
}

for (const event of demuxer.flush()) {
  if (event.type === "packet") {
    encodedDecoder.decode(event.data);
  }
}
```

Rust API:

```rust
use soundkit_audio_demux::{AudioDemuxEvent, AudioTrackDemuxer};

let mut demuxer = AudioTrackDemuxer::new_with_format("mpeg-ts")?;
for event in demuxer.push(bytes)? {
    match event {
        AudioDemuxEvent::Config(config) => { /* configure decoder */ }
        AudioDemuxEvent::Packet(packet) => { /* decode packet.data */ }
    }
}
for event in demuxer.flush()? {
    // final drain
}
```

Constructors:

```ts
new WasmAudioTrackDemuxer()
WasmAudioTrackDemuxer.newAuto()
WasmAudioTrackDemuxer.newWithFormat(format: string)
```

Supported explicit format strings:

```text
mp4, m4a, m4v, mov, quicktime, aac-mp4, mp4-aac,
fmp4, fragmented-mp4, cmaf, cmf,
webm, matroska, mkv,
ts, mpeg-ts, mpegts, hls-ts
```

Generic config event:

```ts
type AudioTrackConfigEvent = {
  type: "config";
  container: "mp4" | "webm" | "mpeg-ts";
  codec: "aac" | "opus" | "vorbis" | "mp3" | "ac3" | string;
  format?: "adts" | "latm" | "raw";
  codecId?: string;
  trackId?: number;
  pid?: number;
  streamType?: number;
  sampleRate?: number;
  channels?: number;
  sampleCount?: number;
  codecPrivate: Uint8Array;
  preSkip?: number;
  outputGain?: number;
  mappingFamily?: number;
};
```

Generic packet event:

```ts
type AudioTrackPacketEvent = {
  type: "packet";
  container: "mp4" | "webm" | "mpeg-ts";
  codec: "aac" | "opus" | "vorbis" | "mp3" | "ac3" | string;
  format: "adts" | "latm" | "raw";
  data: Uint8Array;
  rawData?: Uint8Array;
  trackId?: number;
  pid?: number;
  streamType?: number;
  sampleId?: number;
  startTime?: number;
  duration?: number;
  renderingOffset?: number;
  isSync?: boolean;
  timecode?: number;
};
```

Current container coverage:

| Container | Audio output | Notes |
| --- | --- | --- |
| MP4/M4A/M4V/MOV | AAC as ADTS packets | Sequential input requires `moov` before `mdat`; browser files should use seekable ranges. |
| fragmented MP4/CMAF | AAC as ADTS packets | Parses init `moov`, then `moof`/`mdat` fragments. Use `fmp4`, `fragmented-mp4`, or `cmaf`. |
| WebM/Matroska | Opus/Vorbis/other audio packets as raw blocks | Emits CodecPrivate in config. |
| MPEG-TS/HLS `.ts` | AAC ADTS packets, AAC LATM payloads, MP3 payloads | Parses PAT/PMT/PES and chooses the first supported audio PID. |

### Streaming WAV output

`WasmWavEncoder` writes RIFF or RF64 without assembling a complete WAV in
JavaScript or WASM memory. Supply the final frame count, emit `header()` once,
then forward each bounded result from `encodePlanarI16`, `encodePlanarI32`, or
`encodePlanarF32`. Call `finish()` to verify the exact frame count.

```js
const writer = new WasmWavEncoder(48_000, 2, "f32", totalFrames);
await sink.write(writer.header());
for (const { planar, frames } of pcmChunks) {
  await sink.write(writer.encodePlanarF32(planar, frames));
}
writer.finish();
```

Each input call is limited to 4 MiB. Outputs beyond RIFF's size range receive
an exact RF64 `ds64` header before the first PCM chunk.

Use the seekable API below for each browser `File` or `Blob`. Do not send a
complete MOV or MP4 source across the WASM boundary.

Import `openSeekableMp4` from `soundkit-wasm/runtime/streaming-media.mjs`. The
adapter reads 16-byte headers and calls Rust to validate each box range.

```js
const media = await openSeekableMp4(file, {
  inspectMp4TopLevelBox,
  WasmMp4MediaIndex,
});

try {
  for await (const { sampleIndex, packet } of media.packets()) {
    consume(packet, media.pcmTrim.bind(media), sampleIndex);
  }
} finally {
  media.close();
}
```

The adapter skips `mdat` without reading its payload. It reads `moov` once and
then reads one Rust-indexed sample range at a time.

Do not use the sequential regular MP4 demuxer for tail-`moov` files. It rejects
`mdat` before `moov` instead of buffering the media extent.

Use `decodeSeekableAlac` from the same module for ALAC M4A files. It constructs
`WasmAlacPacketDecoder` from Rust-owned track metadata and decodes one packet
range per iteration.

Use `openSeekableCafAlac` for CAF ALAC files. Rust validates each chunk range
and builds the packet index from `desc`, `kuki`, `pakt`, and `data` metadata.

The adapter skips the `data` extent during the metadata scan. It then reads and
decodes one Rust-indexed ALAC packet at a time.

Rust limits `moov` metadata to 64 MiB. Rust validates box sizes, sample ranges,
timestamps, edit lists, and codec normalization.

Use `WasmMp4MediaDemuxer` only for fragmented MP4 and CMAF byte streams. It now
releases complete samples before the current `mdat` reaches EOF.

Sequential container calls accept bounded chunks only. Rust also limits MP4
metadata and compressed packets before allocating their declared sizes.

WebM parsers consume known-size Cluster headers immediately. They emit complete
blocks before Cluster EOF and bound metadata, packet, and parser-buffer sizes.

## AAC In M4A/MP4 Deboxing

Use `WasmAacDeboxer` when SoundKit should extract AAC access units from M4A/MP4
and another JS/WASM decoder should decode AAC.

This sequential API requires `moov` before `mdat`. Use `openSeekableMp4` for
browser files so tail-`moov` exports and multi-gigabyte sources remain ranged.

```js
const deboxer = WasmAacDeboxer.newWithFormat("m4a");

for await (const chunk of m4aByteStream) {
  for (const event of deboxer.push(chunk)) {
    if (event.type === "config") {
      configureAacDecoder(event);
    } else if (event.type === "packet") {
      // event.data is ADTS-framed AAC.
      aacDecoder.decode(event.data);
    }
  }
}

for (const event of deboxer.flush()) {
  if (event.type === "packet") {
    aacDecoder.decode(event.data);
  }
}
```

Constructors:

```ts
new WasmAacDeboxer()
WasmAacDeboxer.newAuto()
WasmAacDeboxer.newWithFormat(format: string)
```

Supported explicit format strings:

```text
m4a, mp4, aac-mp4, mp4-aac
```

AAC config event:

```ts
type AacConfigEvent = {
  type: "config";
  container: "mp4";
  codec: "aac";
  sampleRate: number;
  channels: number;
  trackId: number;
  sampleCount: number;
};
```

AAC packet event:

```ts
type AacPacketEvent = {
  type: "packet";
  container: "mp4";
  codec: "aac";
  format: "adts";
  data: Uint8Array;
  rawData: Uint8Array;
  sampleId: number;
  startTime: number;
  duration: number;
  renderingOffset: number;
  isSync: boolean;
};
```

`data` contains an ADTS header plus the AAC access unit. This is the preferred
payload for most JS AAC decoders.

`rawData` contains the AAC access unit exactly as extracted from MP4, without
the generated ADTS header.

The current M4A/MP4 path uses the `mp4` crate. It buffers until the MP4 header
can be parsed, then emits available samples. Ordinary M4A files are supported.
For fragmented MP4/CMAF, use the generic `WasmAudioTrackDemuxer` with `fmp4` or
`cmaf`; the AAC-specific `WasmAacDeboxer` remains the regular M4A/MP4 path.

## Opus Deboxing

Use `WasmOpusDeboxer` when SoundKit should extract Opus packets and a JS Opus
decoder, such as `libopusjs`, should decode them.

```js
const deboxer = WasmOpusDeboxer.newWithFormat("ogg-opus");

for await (const chunk of opusContainerByteStream) {
  for (const event of deboxer.push(chunk)) {
    if (event.type === "config") {
      configureOpusDecoder(event);
    } else if (event.type === "packet") {
      opusDecoder.decode(event.data);
    }
  }
}

for (const event of deboxer.flush()) {
  if (event.type === "packet") {
    opusDecoder.decode(event.data);
  }
}
```

Constructors:

```ts
new WasmOpusDeboxer()
WasmOpusDeboxer.newAuto()
WasmOpusDeboxer.newWithFormat(format: string)
```

Supported explicit format strings:

```text
ogg, ogg-opus, opus-ogg, webm, webm-opus, opus, raw-opus
```

`opus` and `raw-opus` refer to SoundKit's raw Opus stream format:

```text
OpusHead bytes followed by u16-le length-prefixed Opus packets
```

Opus config event:

```ts
type OpusConfigEvent = {
  type: "config";
  container: "ogg" | "webm" | "raw";
  codec: "opus";
  sampleRate: number;
  channels: number;
  preSkip: number;
  outputGain: number;
  mappingFamily: number;
  codecPrivate: Uint8Array;
};
```

Opus tags event:

```ts
type OpusTagsEvent = {
  type: "tags";
  container: "ogg";
  codec: "opus";
  data: Uint8Array;
};
```

Opus packet event:

```ts
type OpusPacketEvent = {
  type: "packet";
  container: "ogg" | "webm" | "raw";
  codec: "opus";
  data: Uint8Array;
  timecode?: number;
};
```

`data` is the encoded Opus packet to pass into the JS Opus decoder.

`timecode` is currently emitted for WebM packets. Ogg and raw Opus packets do
not include a `timecode` field.

## Feature Flags And C Codec Boundaries

The browser-safe default avoids pulling FDK AAC or native libopus into the WASM
build.

| Feature | Purpose | Browser default |
| --- | --- | --- |
| `audio-demux` | Generic audio-track demux for MP4/fMP4/CMAF/WebM/MPEG-TS | yes |
| `aac-debox` | Demux AAC from M4A/MP4, emit ADTS packets | yes |
| `opus-debox` | Demux Opus from Ogg/WebM/raw, emit Opus packets | yes |
| `aac` | Decode ADTS AAC through FDK AAC | no |
| `m4a` | Decode AAC-in-M4A through FDK AAC | no |
| `opus` | Decode raw Opus through native Opus dependency | no |
| `ogg-opus` | Decode Ogg Opus through native Opus dependency | no |
| `webm-opus` | Decode WebM Opus through native Opus dependency | no |

For browser playback/editing, the intended split is:

```text
SoundKit WASM: container detection, demux/debox, Rust-native music decoders
JS/WASM codecs: AAC decode, Opus decode
```

## Notes

- `push()` drains immediately: callers should consume every object returned by
  every call.
- `flush()` is required for final buffered output.
- Returned `Uint8Array` values are new JS-owned arrays created from WASM data.
- The API does not depend on WebCodecs.
- AAC-in-video containers use the same MP4 AAC demux path when the MP4 reader can
  find an AAC audio track.
