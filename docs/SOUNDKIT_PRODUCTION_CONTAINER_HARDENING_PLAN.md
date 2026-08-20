# SoundKit Production Container Hardening Plan

Date: 2026-08-20

Primary implementation repository: `/Users/jamie/wavey.ai/soundkit`

Integration repositories:

- `/Users/jamie/wavey.ai/vin.yl.app/Native/vin.yl.native`
- `/Users/jamie/wavey.ai/vin.yl.app`

## Progress ledger

Updated: 2026-08-20

This ledger is the operational TODO. The detailed phases below remain the complete implementation plan.

### Completed in this hardening pass

- [x] Make Opus construction fallible and validate supported rates and channel counts.
- [x] Bound hostile MP4 table counts and avoid expanding constant-size PCM tables.
- [x] Correct MOV PCM versions, flags, `enda` endianness, and fragmented MP4 classification.
- [x] Correct fragmented MP4 offsets, inheritance, durations, edit lists, and sample queuing.
- [x] Consolidate active WebM audio and media paths around one parser.
- [x] Preserve WebM `BlockGroup`, timestamps, codec delay, seek pre-roll, discard padding, lacing, and Opus timing.
- [x] Share bounded Ogg page parsing across Opus, Vorbis, and Speex.
- [x] Validate Ogg CRC, sequencing, continuation state, granules, mapping, pre-skip, output gain, and final trimming.
- [x] Detect 188-byte TS, 192-byte M2TS, and 204-byte transport layouts.
- [x] Preserve PAT, PMT, CRC, continuity, PTS, DTS, wraparound, and PES boundaries.
- [x] Extract AAC ADTS, AAC LOAS, MP2, and MP3 from transport streams.
- [x] Add general CAF indexing for common PCM layouts and ALAC.
- [x] Add bounded MXF KLV parsing, partition/RIP validation, primer UL lookup, OP1a indexing, and exact essence ranges.
- [x] Add OP-Atom DNxHR and clip-wrapped PCM fixtures with clip-level indexing.
- [x] Remove repeated front-draining from fragmented MP4 and MXF hot paths.
- [x] Run the full SoundKit workspace test suite after the parser changes.
- [x] Resolve warnings introduced by the parser and Opus constructor changes.
- [x] Verify demux-only no-default builds for Ogg Opus, WebM, and audio demux.
- [x] Remove the obsolete July M4A decoder-options plan.

### Remaining before common production formats are ready to integrate

- [ ] Add or finish real-fixture reference comparisons for MOV/MP4/fMP4, WebM, Ogg, TS/M2TS, and CAF.
- [ ] Run chunk-boundary invariance tests for every common container at 1 byte, 188 bytes, 4 KiB, 64 KiB, and 4 MiB where applicable.
- [ ] Add malformed-input and allocation-budget regression coverage to CI.
- [ ] Add release throughput and peak-buffer benchmarks for the common container paths.
- [ ] Confirm no large-push performance cliff remains in MP4, WebM, Ogg, TS/M2TS, or CAF.
- [ ] Commit and push the SoundKit implementation and fixtures.
- [ ] Update `vin.yl.native` to the pushed SoundKit revision.
- [ ] Route YL.VIN imports through the SoundKit-owned container path.
- [ ] Run focused native tests, the iOS Debug build, and the iOS test suite.

### Deferred specialty work

These items remain in the full plan but do not block the current common-format checkpoint.

- [ ] Extract M2TS Blu-ray LPCM and DTS access units.
- [ ] Resolve the full MXF Preface, package, sequence, source-clip, and track-origin graph.
- [ ] Split clip-wrapped OP-Atom PCM into edit-unit seek points using resolved package timing.
- [ ] Unpack AES3 PCM instead of exposing its container words as packed PCM.
- [ ] Add Avid-generated multi-channel, AES3, and malformed OP-Atom fixtures.
- [ ] Add interlaced DNxHD and 12-bit DNxHR 4:4:4 support.

## Objective

Make SoundKit the canonical container inspection, demux, timing, and indexing layer for YL.VIN imports.

Prioritize common production media before specialized archive variants.

Support these container families without an AVFoundation or browser demux fallback:

1. MOV and MP4.
2. Fragmented MP4, CMAF, and DASH segments.
3. WebM and Matroska.
4. MPEG-TS and 192-byte M2TS.
5. Ogg Opus and Ogg Vorbis.
6. CAF with PCM and ALAC.
7. MXF with Avid-oriented DNx and PCM media.

Platform decoders may handle an unsupported codec profile after SoundKit demuxes validated access units.

## Repository boundaries

- Implement reusable container, codec, index, and timing logic in SoundKit.
- Keep reusable YL.VIN engine adapters in `vin.yl.native`.
- Keep Swift limited to file access, platform codec fallback, and application workflows.
- Do not create a parallel Swift format catalog.
- Do not add an AVFoundation fallback for a missing SoundKit demuxer.
- Update the SoundKit revision only after its changes are pushed and validated.

## Working-tree safety

At plan creation, another agent owned active changes in both repositories.

Status on 2026-08-20: that overlap has cleared. Keep the original list below as provenance and recheck ownership before future work.

At plan creation time, these SoundKit files contain unrelated work:

- `soundkit-wasm/Cargo.toml`
- `soundkit-wasm/src/lib.rs`
- generated `soundkit-wasm/pkg` files
- `soundkit/src/audio_pipeline.rs`
- `soundkit/src/wav.rs`
- `README.md`

Preserve every existing modification.

Do not reset, restore, reformat, or rewrite another agent's files.

Implement core parser phases without touching those files.

Integrate wrappers only after the current owner finishes or coordinates ownership.

Before each phase:

1. Run `git status --short` in every affected repository.
2. Record the starting commit.
3. Confirm that the phase does not overlap active edits.
4. Use a separate worktree when ownership remains ambiguous.

## Definition of done

The work is complete when all statements below are true.

- Public parsing and decoding APIs return errors instead of panicking on arbitrary input.
- Declared table counts cannot force unbounded allocation.
- Results remain identical across supported input chunk sizes.
- MOV PCM endianness matches the sample entry and `enda` metadata.
- Fragmented MP4 detection does not depend on an early `moof` box.
- WebM audio preserves `BlockGroup`, timestamps, codec delay, and discard padding.
- Ogg Opus obeys pre-skip, final granule trimming, mapping, and output gain rules.
- TS and M2TS preserve PTS, continuity, program selection, and packet boundaries.
- CAF supports common PCM layouts and the existing ALAC path.
- MXF provides a seekable sample index for supported OP1a and OP-Atom media.
- Avid PCM and DNx fixtures pass without platform container parsing.
- Container parsers have focused tests, malformed-input tests, and CI coverage.
- Release benchmarks show no large input-chunk performance cliff.
- YL.VIN imports supported containers through the SoundKit-owned path.

## Confirmed baseline defects

Turn every item below into a regression test before changing its implementation.

### MP4 and MOV

- A valid fragmented MP4 fails auto-detection when `moof` begins after byte 8,192.
- A 12,000-byte top-level `free` box reproduces the failure.
- The current error is `MP4 audio track is missing stsz sample sizes`.
- Valid little-endian and big-endian `in24` MOV files both report little-endian.
- The parser ignores the QuickTime `enda` child atom.
- The parser does not support the version-two `lpcm` sample entry.
- Several sample-table parsers allocate directly from untrusted counts.

### WebM and Matroska

- The committed VP9/Opus fixture contains 151 audio packets.
- `WebmAudioDemuxer` emits only 150 packets.
- `WebmMediaDemuxer` emits all 151 packets.
- The missing packet lives in a final `BlockGroup` with discard padding.
- The audio-only event path exposes a relative block timecode.
- The audio-only event path drops cluster time and timecode scale.
- The decoder ignores `CodecDelay`, `SeekPreRoll`, and `DiscardPadding`.

### Ogg Opus

- `OpusDecoder::new(44_100, 1)` panics with `BadArg`.
- Ogg Opus treats the informational input sample rate as the decoder rate.
- The Ogg parser discards granule positions.
- The decoder never performs final granule trimming.
- The decoder exposes output gain but does not apply it.
- The parser does not validate mapping-family payloads.

### MPEG-TS and M2TS

- A normal 188-byte AAC transport stream emits 48 AAC packets.
- A valid FFmpeg-generated 192-byte M2TS file fails auto-detection.
- Explicit MPEG-TS parsing of that M2TS file emits no packets.
- Emitted packets contain no PTS, DTS, duration, or continuity information.
- A release probe falls to low-single-digit MiB/s for one 4 MiB push.
- Small pushes process the same bytes at hundreds of MiB/s.

### CAF

- SoundKit supports seekable ALAC-in-CAF.
- SoundKit has no general CAF demuxer.
- Common linear PCM CAF input therefore lacks a SoundKit-owned path.

### MXF and DNx

- Frame-wrapped OP1a DNxHR HQX with 24-bit PCM demuxes successfully.
- The fixture emits 75 video packets and 75 audio packets.
- Both emitted track configurations report `sample_count=0`.
- The demuxer emits sequential events but provides no seekable MXF index.
- The parser stores the Primer Pack but resolves properties through fixed tags.
- OP-Atom, clip-wrapped PCM, and Avid-generated files lack conformance coverage.
- AES3 descriptors use the same packed-PCM assumptions as BWF descriptors.
- Interlaced DNxHD and 12-bit DNxHR 444 remain unsupported.

## Phase 1: Eliminate crash and allocation hazards

Complete this phase before expanding format coverage.

### 1.1 Make Opus construction fallible

Files:

- `soundkit-opus/src/lib.rs`
- every `OpusDecoder::new` caller

Implementation:

- Change decoder construction to return `Result`.
- Remove the constructor `expect` call.
- Validate supported decode rates before calling `libopus-rs`.
- Validate mono and stereo channel counts for the single-stream decoder.
- Return a precise unsupported-mapping error for multistream input.
- Update every native and WASM caller through compiler-guided changes.

Tests:

- Accept 8, 12, 16, 24, and 48 kHz decode rates.
- Reject 44.1 kHz without panicking.
- Reject zero channels without panicking.
- Reject more than two channels without panicking.
- Run each public constructor inside `catch_unwind` with invalid inputs.

Exit gate:

- No public Opus constructor can panic from caller-controlled metadata.

### 1.2 Bound MP4 sample-table expansion

File:

- `soundkit-audio-demux/src/lib.rs`

Implementation:

- Make every table parser return `Result`.
- Validate entry counts against the remaining payload length.
- Use checked arithmetic for every entry width calculation.
- Use `try_reserve` and convert allocation failure into an error.
- Represent constant `stsz` entries without materializing one value per sample.
- Coalesce constant-size PCM while building the index.
- Define a documented maximum for materialized compressed access units.
- Reject inconsistent table lengths before building sample records.
- Reject sample counts that exceed `u32` event identifiers.

Apply the same rules to:

- `stts`
- `ctts`
- `stsc`
- `stsz`
- `stco`
- `co64`
- `stss`
- `elst`
- `trex`, `tfhd`, and `trun` arrays

Tests:

- Add tiny boxes with counts near `u32::MAX`.
- Assert a normal error without a large allocation.
- Add truncated tables whose declared count exceeds their payload.
- Add a large constant-size PCM table without per-frame expansion.
- Preserve current sample offsets and timestamps for committed fixtures.

Exit gate:

- No MP4 metadata field can request an allocation unrelated to validated bytes.

## Phase 2: Correct MOV, MP4, and fragmented MP4

### 2.1 Replace fragmented-MP4 window detection

File:

- `soundkit-audio-demux/src/lib.rs`

Implementation:

- Replace the 8 KiB classification decision with a stateful box scanner.
- Keep detecting while the current top-level box is incomplete.
- Classify fragmented MP4 when `moov` contains `mvex`.
- Also classify fragmented MP4 when a complete top-level `moof` appears.
- Do not require media payload bytes for classification.
- Retain the existing metadata budget.
- Return a clear ambiguity error when the budget expires.
- Keep explicit `mp4` and `fmp4` constructors available.

Tests:

- Put `moof` after 8 KiB, 32 KiB, and 64 KiB.
- Put `mvex` inside a complete initialization `moov` without `moof`.
- Split every top-level header across input chunks.
- Verify identical events for 1-byte, 4 KiB, 64 KiB, and 4 MiB pushes.

Exit gate:

- Valid initialization segments classify correctly before media payload arrives.

### 2.2 Implement QuickTime PCM sample entries

File:

- `soundkit-audio-demux/src/lib.rs`

Implementation:

- Parse audio sample-entry versions zero, one, and two.
- Parse the `enda` child atom for `in24`, `in32`, `fl32`, and `fl64`.
- Apply QuickTime defaults when `enda` is absent.
- Parse version-two `lpcm` format flags.
- Preserve signed, floating-point, packed, aligned-high, and endian attributes.
- Reject unsupported non-interleaved layouts explicitly.
- Expose enough PCM geometry for deterministic decode.

Tests:

- Add `in24` little-endian and big-endian MOV fixtures.
- Add `in32`, `fl32`, and `fl64` endian variants.
- Add at least one version-two `lpcm` fixture.
- Compare decoded PCM hashes against pinned FFmpeg output.

Exit gate:

- SoundKit reports and decodes every fixture with the correct byte order.

### 2.3 Finish fragmented sample semantics

File:

- `soundkit-audio-demux/src/lib.rs`

Implementation:

- Implement implicit base-data-offset inheritance across `traf` boxes.
- Honor `default-base-is-moof` explicitly.
- Honor sample-description indexes.
- Remove the universal 1,024-sample audio-duration fallback.
- Derive duration from codec metadata only when the specification permits it.
- Reject fragments with no valid duration source.
- Support multiple linear edit-list entries.
- Preserve empty edits and media-time offsets.

Tests:

- Cover multiple `traf` boxes sharing one `mdat`.
- Cover audio codecs with durations other than 1,024 samples.
- Cover explicit and implicit base offsets.
- Cover multiple valid edit entries.

Exit gate:

- Fragment offsets and timestamps match Bento4 or FFmpeg for every fixture.

## Phase 3: Consolidate WebM and Matroska parsing

SoundKit currently has several partially duplicated WebM parsers.

Use one internal parser to prevent packet and timing divergence.

Files:

- `soundkit-webm/src/lib.rs`
- `soundkit-audio-demux/src/lib.rs`

### 3.1 Create one internal Matroska event model

Implementation:

- Parse EBML, Segment, Info, Tracks, Clusters, and BlockGroups once.
- Make media, audio-only, Opus-only, and decoder adapters consume shared events.
- Accept valid audio-only and video-only files.
- Validate DocType and supported read versions.
- Preserve unknown elements within bounded size rules.
- Support `ContentEncodings` header stripping or reject it precisely.

### 3.2 Preserve packet timing and trimming

Implementation:

- Combine cluster timecode and signed block timecode.
- Apply `TimecodeScale` to normalized timestamps.
- preserve `TrackTimestampScale` where supported.
- Parse `CodecDelay` and subtract it from presentation timestamps.
- Parse `SeekPreRoll` for random-access decode plans.
- Parse positive and negative `DiscardPadding`.
- Attach discard information to the affected packet.
- Derive laced Opus durations from packet TOC data when needed.
- Reject laced timing that cannot be resolved deterministically.

Tests:

- Assert 151 audio packets for `vp9-profile0-opus.webm`.
- Assert exact decoded frame count against FFmpeg.
- Assert monotonic absolute timestamps across multiple clusters.
- Cover SimpleBlock and BlockGroup packet forms.
- Cover Xiph, fixed, and EBML lacing.
- Cover audio-only WebM and video-only Matroska.
- Cover a final BlockGroup containing discard padding.

Exit gate:

- Every public WebM adapter emits the same packet set and normalized timing.

## Phase 4: Make Ogg Opus specification-correct

Files:

- `soundkit-ogg-opus/src/lib.rs`
- `soundkit-vorbis/src/lib.rs`
- `soundkit-speex/src/lib.rs`
- optionally a new shared Ogg parser crate or module

### 4.1 Correct OpusHead semantics

Implementation:

- Preserve the original input sample rate as metadata.
- Decode at 48 kHz unless a caller selects another supported Opus rate.
- Validate the OpusHead version.
- Validate channel counts and mapping-family payload lengths.
- Reject unsupported multistream mappings without panicking.
- Apply header output gain exactly once.

### 4.2 Preserve granule positions

Implementation:

- Attach page granule positions to completed packets.
- Apply pre-skip at the decoder output rate.
- Trim the final page using its EOS granule position.
- Handle packets spanning page boundaries.
- Reject broken continuation sequences.
- Decide whether to support chained streams.
- Return an explicit error when chaining remains unsupported.

### 4.3 Share Ogg page validation

Implementation:

- Validate page version zero.
- Validate CRC checksums.
- Track serial numbers and page sequence numbers.
- Enforce continued-packet flags.
- Enforce BOS and EOS ordering.
- Use a cursor with occasional compaction instead of front-draining every page.

Tests:

- Use a valid 44.1 kHz input-rate metadata field.
- Cover nonzero output gain.
- Cover exact pre-skip and final trim.
- Cover mono, stereo, and rejected multichannel mappings.
- Corrupt CRC, sequence, continuation, and EOS fields independently.
- Preserve Vorbis final granule trimming.

Exit gate:

- Ogg Opus decoded length and amplitude match a pinned reference decoder.

## Phase 5: Rebuild MPEG-TS and M2TS around packet strides

File:

- `soundkit-audio-demux/src/lib.rs`

Move the implementation into `soundkit-audio-demux/src/mpeg_ts.rs` when practical.

### 5.1 Detect transport layout

Implementation:

- Detect 188-byte MPEG-TS packets.
- Detect 192-byte M2TS packets with a four-byte arrival timestamp prefix.
- Detect 204-byte protected TS packets when useful.
- Confirm at least five sync positions before locking.
- Reconfirm stride after corruption or discontinuity.
- Add `m2ts` and `bdav` format aliases.
- Expose the selected stride and prefix geometry in diagnostics.

### 5.2 Parse transport state

Implementation:

- Replace per-packet `Vec::drain` with a read cursor.
- Avoid allocating a new vector for each transport packet.
- Validate transport-error and scrambling flags.
- Track continuity counters per PID.
- Parse adaptation-field discontinuity indicators.
- Reassemble PAT and PMT sections across packets.
- Validate PSI CRC and table versions.
- Select programs and audio tracks deterministically.

### 5.3 Preserve PES timing

Implementation:

- Reassemble PES payloads across packet boundaries.
- Parse 33-bit PTS and DTS values.
- Handle timestamp wraparound with a monotonic epoch.
- Add an explicit packet timescale of 90,000.
- Preserve discontinuities instead of inventing continuous time.
- Split ADTS and LOAS/LATM access units correctly.

### 5.4 Expand common audio stream types

Prioritize:

1. AAC ADTS.
2. AAC LOAS/LATM.
3. MPEG Layer II and Layer III.
4. AC-3 and E-AC-3 through descriptors.
5. M2TS LPCM.
6. DTS labeling and extraction when decode remains unavailable.

Tests:

- Add real FFmpeg-generated TS and M2TS fixtures.
- Add a Blu-ray-style LPCM M2TS fixture.
- Split PAT, PMT, PES, and audio frames across arbitrary chunks.
- Inject continuity gaps and transport errors.
- Assert monotonic PTS across wraparound.
- Compare packet counts and timestamps against FFprobe.
- Benchmark 188-byte, 4 KiB, 64 KiB, and 4 MiB pushes.

Exit gate:

- A 4 MiB push has no order-of-magnitude throughput penalty.
- TS and M2TS produce identical elementary audio for equivalent sources.

## Phase 6: Add general CAF support

Files:

- new `soundkit-audio-demux/src/caf.rs`
- `soundkit-audio-demux/src/lib.rs`
- `soundkit-alac/src/lib.rs`

Implementation:

- Move generic CAF chunk parsing into the audio-demux layer.
- Keep compatibility re-exports for existing ALAC callers.
- Create a seekable `CafAudioIndex`.
- Parse `desc`, `data`, `pakt`, `kuki`, and channel-layout metadata.
- Preserve priming, remainder, and valid-frame counts for every codec.
- Support linear PCM before less common CAF codecs.
- Parse integer, float, endian, packed, aligned, and interleaved PCM flags.
- Retain the existing bounded ALAC packet decoder.
- Add AAC and IMA4 only when a bounded decoder path exists.

Tests:

- Add PCM16, PCM24, PCM32, float32, and float64 CAF fixtures.
- Cover both endian modes.
- Cover constant and variable packet sizes.
- Cover data before and after packet metadata.
- Compare PCM hashes against Core Audio or FFmpeg.
- Verify ALAC priming and remainder trimming.

Exit gate:

- Common PCM and ALAC CAF files never require a platform container parser.

## Phase 7: Build an Avid-ready MXF index

Files:

- `soundkit-audio-demux/src/mxf.rs`
- new focused MXF modules when the file becomes too large
- `soundkit-dnx/src/lib.rs`
- `soundkit-video/src/lib.rs`

### 7.1 Separate sequential demux from seekable indexing

Implementation:

- Retain `MxfMediaDemuxer` for bounded sequential input.
- Add `MxfMediaIndex` for seekable files.
- Expose validated byte ranges through `MediaSampleIndex`-equivalent records.
- Parse Header, Body, and Footer Partition Packs.
- Parse BodySID, IndexSID, body offsets, and partition status.
- Parse IndexTableSegment records.
- Parse DeltaEntryArray and IndexEntryArray data.
- Parse the Random Index Pack.
- Fall back to bounded KLV scanning only when index metadata is absent.

### 7.2 Resolve structural metadata through ULs

Implementation:

- Resolve local tags through the Primer Pack.
- Identify properties by their universal labels.
- Build the Preface, ContentStorage, Package, Track, Sequence, and Descriptor graph.
- Support MultipleDescriptor relationships.
- Preserve Track Origin and edit rates.
- Resolve source and material package timelines.
- Report real packet counts instead of zero.

### 7.3 Support production essence layouts

Prioritize:

1. OP1a frame-wrapped DNxHD or DNxHR with BWF PCM.
2. OP-Atom DNx picture files.
3. OP-Atom PCM audio files.
4. Clip-wrapped BWF PCM.
5. AES3 PCM with correct 32-bit subframe unpacking.

Implementation:

- Distinguish BWF and AES3 packing.
- Preserve stored versus significant sample depth.
- Preserve channel assignment metadata.
- Validate edit-unit byte counts.
- Support audio-only MXF files.
- Return precise errors for unsupported operational patterns.

### 7.4 Complete common DNx profiles

Implementation:

- Add interlaced DNxHD decoding.
- Add 12-bit DNxHR 444 decoding.
- Keep malformed-input allocation checks ahead of plane allocation.
- Add optional parallel slice decoding after correctness is locked.
- Preserve byte-identical output against pinned FFmpeg references.

Fixtures:

- Add short files exported from Avid Media Composer.
- Include OP-Atom picture and separate PCM audio media.
- Include DNxHD 1080i.
- Include DNxHR HQX and 444 variants.
- Include 16-bit, packed 24-bit, and AES3 24-in-32 audio.
- Store hashes and provenance for every fixture.
- Keep committed media small enough for normal CI checkout.

Tests:

- Compare sample counts, byte ranges, and timestamps against FFprobe and MediaInfo.
- Seek to the first, middle, and final edit units.
- Decode audio-only OP-Atom without scanning unrelated files.
- Verify material and source package origins.
- Assert deterministic behavior across input chunk sizes.

Exit gate:

- Avid OP-Atom picture and audio files index and decode without AVFoundation demux.

## Phase 8: Remove structural performance cliffs

Affected parsers:

- fragmented MP4
- MPEG-TS and M2TS
- MXF
- Ogg
- WebM and Matroska

Implementation:

- Replace repeated front-draining with a cursor and periodic compaction.
- Use borrowed slices until an owned packet crosses the public API.
- Remove `Vec::remove` from ordered packet loops.
- Use queues or sorted cursors for fragmented samples.
- Avoid retaining both raw and wrapped packet copies by default.
- Add an explicit opt-in when callers need both packet forms.
- Keep all metadata and packet budgets.

Benchmarks:

- Add release benchmarks for every container parser.
- Test 1-byte, packet-sized, 4 KiB, 64 KiB, and 4 MiB pushes.
- Measure throughput, peak retained bytes, and allocation count.
- Add DNx 1080p25, 1080p60, and 4K benchmarks.
- Record native and WASM results separately.

Performance gates:

- Chunk-size changes must not cause an order-of-magnitude slowdown.
- Streaming parsers must retain only bounded metadata and incomplete packets.
- 1080p25 DNx decode must keep comfortable real-time headroom.
- The plan must document 4K limitations when real-time decode remains unavailable.

## Phase 9: Put conformance and fuzzing in CI

Files:

- `.github/workflows/ci.yml`
- `Makefile`
- `scripts/fuzz-video-wasm.mjs`
- new fuzz targets and fixture scripts

Implementation:

- Run `soundkit-audio-demux` tests in CI.
- Run `soundkit-alac`, `soundkit-vorbis`, `soundkit-dnx`, and `soundkit-video` tests.
- Run container conformance tests without writing golden artifacts.
- Keep golden generation as an explicit maintenance command.
- Add coverage-guided fuzz targets for each parser entry point.
- Seed fuzzers with every committed container fixture.
- Add allocation-budget assertions to fuzz harnesses.
- Add deterministic mutation cases for TS, M2TS, CAF, Ogg, and MXF.
- Treat panics, traps, excessive allocation, and timeouts as failures.

Required checks:

```sh
cargo fmt --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
make media-conformance
make media-fuzz
```

Exit gate:

- Every supported container parser runs in normal pull-request CI.

## Phase 10: Integrate the pushed SoundKit revision into YL.VIN

Complete this phase only after SoundKit passes all relevant gates.

### 10.1 Update `vin.yl.native`

Files:

- `crates/bitneedle-native-core/Cargo.toml`
- `crates/bitneedle-native-core/Cargo.lock`
- reusable Rust bridge modules
- C headers generated from the bridge contract

Implementation:

- Pin every SoundKit crate to one pushed commit.
- Add direct dependencies for the demux and index crates used by the bridge.
- Expose seekable range plans instead of whole-file byte copies.
- Emit bounded decoded PCM blocks into the existing normalization pipeline.
- Replace Symphonia container ownership for newly supported formats.
- Keep a platform decoder fallback only for rejected codec profiles.
- Preserve SoundKit timing, trim, channel, and sample-rate decisions.

### 10.2 Update the Apple application

Implementation:

- Let Swift provide validated file-range reads requested by Rust.
- Keep container selection and packet interpretation in Rust.
- Consume decoded blocks incrementally.
- Do not retain a complete decoded PCM working copy.
- Keep AVFoundation for device I/O and approved codec fallback only.
- Record the selected demux and decoder path in technical diagnostics.
- Keep user-facing messages free of internal service names.

Validation:

```sh
make build
make test
```

Use the repository-owned DerivedData caches from `AGENTS.md`.

Do not start a release archive during this implementation plan.

Exit gate:

- Every supported fixture imports through SoundKit in an app integration test.

## Suggested commit sequence

Keep each commit independently testable.

1. `soundkit: make opus construction fallible`
2. `soundkit: bound mp4 table allocations`
3. `soundkit: fix fragmented mp4 detection and quicktime pcm`
4. `soundkit: unify webm audio and media block parsing`
5. `soundkit: implement ogg opus timing and trim semantics`
6. `soundkit: add stride-aware mpeg-ts and m2ts demux`
7. `soundkit: add generic pcm caf indexing`
8. `soundkit: add seekable avid mxf indexing`
9. `soundkit: remove streaming front-drain hotspots`
10. `soundkit: run media conformance and fuzzing in ci`
11. `vin.yl.native: adopt the hardened soundkit revision`
12. `yl.vin: route supported imports through soundkit`

## Final handoff checklist

- [ ] List every changed public API.
- [ ] List every newly supported container and codec combination.
- [ ] List every intentionally unsupported profile.
- [ ] Include fixture provenance and hashes.
- [ ] Include focused test results.
- [ ] Include full workspace test results.
- [ ] Include native and WASM benchmark results.
- [ ] Include peak-memory measurements.
- [ ] Confirm no public parser panic remains.
- [ ] Confirm no unrelated dirty changes were overwritten.
- [ ] Confirm the SoundKit commit is pushed.
- [ ] Confirm `vin.yl.native` pins that exact commit.
- [ ] Confirm the YL.VIN Debug build passes.
- [ ] Confirm the YL.VIN iOS test suite passes.
