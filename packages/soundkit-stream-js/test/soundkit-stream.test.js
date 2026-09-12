import assert from "node:assert/strict";
import test from "node:test";
import {
  SoundKitEncoding,
  buildSoundKitFrame,
  buildSoundKitIndexFromStream,
  computeSoundKitPacketCrc32,
  decodeSoundKitFrameHeader,
  decodeSoundKitIndex,
  encodeSoundKitFrameHeader,
  encodeSoundKitIndex,
  findSoundKitSeekEntry,
  frameForTimeSeconds,
  pcmI16FromDecodedFrames,
  readSoundKitFrame,
  scanSoundKitFrameStream,
  soundKitRangeRequestForFrame,
  verifySoundKitPacketCrc32
} from "../src/index.js";

test("encodes and decodes SoundKit v2 frame headers", () => {
  const payload = new Uint8Array([1, 2, 3, 4]);
  const headerInit = {
    encoding: SoundKitEncoding.Opus,
    payloadSize: payload.length,
    frameCount: 960,
    sampleRate: 48_000,
    channels: 2,
    id: 7n,
    pts: 960n
  };
  const packetCrc32 = computeSoundKitPacketCrc32(headerInit, payload);
  const encoded = encodeSoundKitFrameHeader({ ...headerInit, packetCrc32 });
  const header = decodeSoundKitFrameHeader(encoded);

  assert.equal(header.encoding, SoundKitEncoding.Opus);
  assert.equal(header.payloadSize, payload.length);
  assert.equal(header.frameCount, 960);
  assert.equal(header.sampleRate, 48_000);
  assert.equal(header.channels, 2);
  assert.equal(header.id, 7n);
  assert.equal(header.pts, 960n);
  assert.equal(verifySoundKitPacketCrc32(header, encoded, payload), true);
});

test("encodes extended SoundKit v2 frame sizes", () => {
  const encoded = encodeSoundKitFrameHeader({
    encoding: SoundKitEncoding.Opus,
    payloadSize: 70_000,
    frameCount: 70_001,
    sampleRate: 48_000,
    channels: 2
  });
  const header = decodeSoundKitFrameHeader(encoded);

  assert.equal(header.payloadSize, 70_000);
  assert.equal(header.frameCount, 70_001);
});

test("scans a SoundKit v2 frame stream", () => {
  const first = buildSoundKitFrame({
    encoding: SoundKitEncoding.Opus,
    frameCount: 960,
    sampleRate: 48_000,
    channels: 2,
    id: 0n,
    pts: 0n
  }, new Uint8Array([1, 2, 3]));
  const second = buildSoundKitFrame({
    encoding: SoundKitEncoding.Opus,
    frameCount: 960,
    sampleRate: 48_000,
    channels: 2,
    id: 1n,
    pts: 960n
  }, new Uint8Array([4, 5]));
  const stream = new Uint8Array(first.length + second.length);
  stream.set(first, 0);
  stream.set(second, first.length);

  const frames = scanSoundKitFrameStream(stream);
  assert.equal(frames.length, 2);
  assert.equal(frames[0].byteOffset, 0n);
  assert.equal(frames[1].byteOffset, BigInt(first.length));
  assert.deepEqual([...readSoundKitFrame(stream).payload], [1, 2, 3]);
});

test("encodes, decodes, and seeks sidecar indexes", () => {
  const indexBytes = encodeSoundKitIndex({
    timescale: 48_000,
    durationFrames: 2_880n,
    entries: [
      { byteOffset: 0n, startFrame: 0n },
      { byteOffset: 123n, startFrame: 960n },
      { byteOffset: 246n, startFrame: 1_920n }
    ]
  });
  const index = decodeSoundKitIndex(indexBytes);

  assert.equal(index.timescale, 48_000);
  assert.equal(index.durationFrames, 2_880n);
  assert.equal(findSoundKitSeekEntry(index, 959n).byteOffset, 0n);
  assert.equal(findSoundKitSeekEntry(index, 960n).byteOffset, 123n);
  assert.equal(findSoundKitSeekEntry(index, frameForTimeSeconds(0.05, 48_000)).byteOffset, 246n);
  assert.deepEqual(soundKitRangeRequestForFrame(index, 1_920n).headers, {
    Range: "bytes=246-"
  });
});

test("builds a sidecar index from a complete frame stream", () => {
  const first = buildSoundKitFrame({
    encoding: SoundKitEncoding.Opus,
    frameCount: 960,
    sampleRate: 48_000,
    channels: 2,
    pts: 0n
  }, new Uint8Array([1]));
  const second = buildSoundKitFrame({
    encoding: SoundKitEncoding.Opus,
    frameCount: 480,
    sampleRate: 48_000,
    channels: 2,
    pts: 960n
  }, new Uint8Array([2]));
  const stream = new Uint8Array(first.length + second.length);
  stream.set(first, 0);
  stream.set(second, first.length);

  const index = decodeSoundKitIndex(buildSoundKitIndexFromStream(stream));
  assert.equal(index.timescale, 48_000);
  assert.equal(index.durationFrames, 1_440n);
  assert.equal(index.entries[0].byteOffset, 0n);
  assert.equal(index.entries[1].byteOffset, BigInt(first.length));
});

test("converts SoundKit decoder PCM frames to interleaved Int16Array", () => {
  const decoded = pcmI16FromDecodedFrames([
    {
      sampleRate: 48_000,
      channels: 2,
      bitsPerSample: 16,
      data: new Uint8Array([0x01, 0x00, 0xff, 0xff, 0x00, 0x80, 0xff, 0x7f])
    }
  ]);

  assert.equal(decoded.sampleRate, 48_000);
  assert.equal(decoded.channels, 2);
  assert.equal(decoded.frameCount, 2);
  assert.deepEqual([...decoded.pcm], [1, -1, -32768, 32767]);
});
