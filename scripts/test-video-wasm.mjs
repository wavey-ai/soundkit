import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import init, { WasmAudioTrackDemuxer, WasmVideoDecoder } from "../soundkit-wasm/pkg/soundkit_wasm.js";

const fixtureRoot = resolve(process.argv[2] ?? "build/video-compat/never-final");
await init({ module_or_path: await readFile(new URL("../soundkit-wasm/pkg/soundkit_wasm_bg.wasm", import.meta.url)) });

function ivfPackets(bytes) {
  assert.equal(Buffer.from(bytes.subarray(0, 4)).toString("ascii"), "DKIF");
  const packets = [];
  let cursor = bytes.readUInt16LE(6);
  while (cursor < bytes.length) {
    assert.ok(cursor + 12 <= bytes.length, "truncated IVF packet header");
    const size = bytes.readUInt32LE(cursor);
    const timestamp = Number(bytes.readBigUInt64LE(cursor + 4));
    cursor += 12;
    assert.ok(cursor + size <= bytes.length, "truncated IVF packet");
    packets.push({ data: bytes.subarray(cursor, cursor + size), timestamp });
    cursor += size;
  }
  return packets;
}

function proresPackets(bytes) {
  const packets = [];
  let cursor = 0;
  while (cursor < bytes.length) {
    assert.ok(cursor + 8 <= bytes.length, "truncated ProRes frame header");
    const size = bytes.readUInt32BE(cursor);
    assert.ok(size >= 8 && cursor + size <= bytes.length, "invalid ProRes frame size");
    packets.push(bytes.subarray(cursor, cursor + size));
    cursor += size;
  }
  return packets;
}

function assertFrames(codec, frames, expected) {
  assert.equal(frames.length, expected.frames, `${codec} frame count`);
  for (const frame of frames) {
    assert.equal(frame.width, 640, `${codec} width`);
    assert.equal(frame.height, 360, `${codec} height`);
    assert.ok(frame.planes.length >= 1, `${codec} planes`);
    for (const plane of frame.planes) {
      const bytesPerSample = frame.bitDepth <= 8 ? 1 : 2;
      assert.equal(
        plane.data.byteLength,
        plane.stride * plane.height * bytesPerSample,
        `${codec} plane byte contract`,
      );
    }
  }
  assert.equal(frames[0].bitDepth, expected.bitDepth, `${codec} bit depth`);
  assert.equal(frames[0].chromaSampling, expected.chroma, `${codec} chroma`);
  console.log(`${codec}: ${frames.length} frames, ${frames[0].width}x${frames[0].height}, ${frames[0].bitDepth}-bit ${frames[0].chromaSampling}`);
}

async function decodeStream(codec, file, expected) {
  const decoder = new WasmVideoDecoder(codec);
  try {
    const frames = decoder.decodeStream(await readFile(resolve(fixtureRoot, file)));
    assertFrames(codec, frames, expected);
  } finally {
    decoder.free();
  }
}

async function decodePackets(codec, file, packetizer, expected) {
  const decoder = new WasmVideoDecoder(codec);
  const frames = [];
  try {
    for (const [index, packet] of packetizer(await readFile(resolve(fixtureRoot, file))).entries()) {
      const data = packet.data ?? packet;
      frames.push(...decoder.decode(data, packet.timestamp ?? index, Number.NaN));
    }
    frames.push(...decoder.flush());
    assertFrames(codec, frames, expected);
  } finally {
    decoder.free();
  }
}

await decodeStream("h264", "h264-high.264", { frames: 75, bitDepth: 8, chroma: "420" });
await decodeStream("hevc", "hevc-main.265", { frames: 75, bitDepth: 8, chroma: "420" });
await decodeStream("hevc", "hevc-main10.265", { frames: 75, bitDepth: 10, chroma: "420" });
await decodePackets("vp9", "vp9-profile0.ivf", ivfPackets, { frames: 75, bitDepth: 8, chroma: "420" });
await decodePackets("av1", "av1-main.ivf", ivfPackets, { frames: 75, bitDepth: 8, chroma: "420" });
await decodePackets("prores", "prores-422-hq.bin", proresPackets, { frames: 75, bitDepth: 10, chroma: "422" });

for (const codec of ["h264", "hevc", "vp9", "av1", "prores"]) {
  const decoder = new WasmVideoDecoder(codec);
  try {
    let completed = false;
    try {
      const outcome = decoder.decode(Uint8Array.from([0xff, 0, 1, 2, 3]), 0, 1);
      assert.ok(Array.isArray(outcome), `${codec} malformed input returns frames or an error`);
      completed = true;
    } catch (error) {
      assert.notEqual(error, undefined, `${codec} malformed input reports an error`);
      completed = true;
    }
    assert.equal(completed, true, `${codec} malformed input stays bounded`);
  } finally {
    decoder.free();
  }
}

assert.throws(() => new WasmVideoDecoder("dnxhr"), /not yet available/);

async function inspectAudio(file, expected) {
  const demuxer = WasmAudioTrackDemuxer.newAuto();
  let config;
  let packets = 0;
  try {
    const bytes = await readFile(resolve(fixtureRoot, file));
    for (let offset = 0; offset < bytes.length; offset += 64 * 1024) {
      for (const event of demuxer.push(bytes.subarray(offset, offset + 64 * 1024))) {
        if (event.type === "config") config = event;
        if (event.type === "packet") packets += 1;
      }
    }
    for (const event of demuxer.flush()) {
      if (event.type === "config") config = event;
      if (event.type === "packet") packets += 1;
    }
  } finally {
    demuxer.free();
  }
  assert.ok(config, `${file} audio config`);
  assert.equal(config.codec, expected.codec, `${file} audio codec`);
  assert.equal(config.sampleRate, 48_000, `${file} audio rate`);
  assert.equal(config.channels, 2, `${file} audio channels`);
  if (expected.bits) assert.equal(config.bitsPerSample, expected.bits, `${file} PCM depth`);
  assert.ok(packets > 0, `${file} audio packets`);
  console.log(`${file}: ${config.codec} ${config.sampleRate}Hz/${config.channels}ch, ${packets} packets`);
}

await inspectAudio("h264-high-aac.mp4", { codec: "aac" });
await inspectAudio("hevc-main-aac.mov", { codec: "aac" });
await inspectAudio("hevc-main10-pcm.mov", { codec: "pcm", bits: 24 });
await inspectAudio("prores-422-hq-pcm.mov", { codec: "pcm", bits: 24 });
await inspectAudio("dnxhr-hqx-pcm.mov", { codec: "pcm", bits: 24 });
await inspectAudio("vp9-profile0-opus.webm", { codec: "opus" });
await inspectAudio("av1-main-opus.webm", { codec: "opus" });

console.log("SoundKit WASM media conformance passed");
