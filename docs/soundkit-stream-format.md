# SoundKit Stream Format

## Stream Object

The stream object is a concatenation of SoundKit v2 frames:

```text
[FrameHeaderV2][payload][FrameHeaderV2][payload]...
```

The v2 header is the existing `frame-header` crate format:

- Big-endian base header.
- Explicit payload byte size.
- Explicit decoded PCM frame count.
- Optional packet ID.
- Optional PTS.
- Optional packet CRC32.
- Public packet flags for discontinuity and encryption.

For Opus and FLAC streams produced by this workspace:

- Opus: `encoding = Opus`, `sampleRate = 48000`, `bitsPerSample = 0`.
- FLAC: `encoding = FLAC`, `sampleRate = 48000`, `bitsPerSample = 24`.
- `frameCount = decoded PCM frames represented by the packet`
- `pts = start PCM frame`
- `id = packet index`

The final Opus packet may be zero-padded for encoder input. Its SoundKit
`frameCount` records the real source frame count so playback can trim exactly at
the original duration.

## Sidecar Index Object

The sidecar index is little-endian and fixed-entry:

```text
offset  size  field
0       8     magic: "SKIDX2\0\0"
8       2     version: u16le, currently 1
10      2     entry_bytes: u16le, currently 16
12      4     timescale: u32le, usually sample rate
16      8     entry_count: u64le
24      8     duration_frames: u64le
32      ...   entries
```

Each entry is:

```text
offset  size  field
0       8     stream byte_offset: u64le
8       8     start_frame: u64le
```

Entries must be sorted by `start_frame`. A player seeks by finding the greatest
entry with `start_frame <= target_frame`, range-fetching from `byte_offset`, then
decoding and trimming to `target_frame`.

## Object Store Layout

Recommended durable layout:

```text
source/{id}/audio.soundkit
source/{id}/audio.soundkit.idx
source/{id}/audio.soundkit.json
```

The stream and index can be persisted independently in D2 or any byte-range
addressable object store. Random access requires the stream object to support
HTTP range reads or equivalent byte-range API calls.

