import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import init, { WasmVideoDecoder } from "../soundkit-wasm/pkg/soundkit_wasm.js";

const corpusRoot = resolve(process.argv[2] ?? "build/video-compat/upstream");
await init({
  module_or_path: await readFile(
    new URL("../soundkit-wasm/pkg/soundkit_wasm_bg.wasm", import.meta.url),
  ),
});

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

function assertFrames(label, frames, expected) {
  assert.equal(frames.length, expected.frames, `${label} frame count`);
  for (const frame of frames) {
    assert.equal(frame.width, expected.width, `${label} width`);
    assert.equal(frame.height, expected.height, `${label} height`);
    assert.equal(frame.bitDepth, expected.bitDepth, `${label} bit depth`);
    assert.equal(frame.chromaSampling, expected.chroma, `${label} chroma`);
    assert.equal(
      frame.planes.length,
      expected.chroma === "400" ? 1 : 3,
      `${label} plane count`,
    );
    for (const plane of frame.planes) {
      const bytesPerSample = frame.bitDepth <= 8 ? 1 : 2;
      assert.equal(
        plane.data.byteLength,
        plane.stride * plane.height * bytesPerSample,
        `${label} plane byte contract`,
      );
    }
  }
  console.log(
    `${label}: ${frames.length} frames, ${expected.width}x${expected.height}, ${expected.bitDepth}-bit ${expected.chroma}`,
  );
}

async function decodeStream(codec, file, expected) {
  const decoder = new WasmVideoDecoder(codec);
  try {
    const frames = decoder.decodeStream(await readFile(resolve(corpusRoot, file)));
    frames.push(...decoder.flush());
    assertFrames(file, frames, expected);
  } finally {
    decoder.free();
  }
}

async function decodeIvf(codec, file, expected) {
  const decoder = new WasmVideoDecoder(codec);
  const frames = [];
  try {
    for (const packet of ivfPackets(await readFile(resolve(corpusRoot, file)))) {
      frames.push(...decoder.decode(packet.data, packet.timestamp, Number.NaN));
    }
    frames.push(...decoder.flush());
    assertFrames(file, frames, expected);
  } finally {
    decoder.free();
  }
}

async function decodePacket(codec, file, expected) {
  const decoder = new WasmVideoDecoder(codec);
  try {
    const frames = decoder.decode(await readFile(resolve(corpusRoot, file)), 0, Number.NaN);
    frames.push(...decoder.flush());
    assertFrames(file, frames, expected);
  } finally {
    decoder.free();
  }
}

await decodeStream("h264", "test-25fps.h264", {
  frames: 250,
  width: 320,
  height: 240,
  bitDepth: 8,
  chroma: "420",
});
await decodeStream("hevc", "test-25fps.hevc", {
  frames: 250,
  width: 320,
  height: 240,
  bitDepth: 8,
  chroma: "420",
});
await decodeStream("hevc", "test-25fps.hevc10", {
  frames: 250,
  width: 320,
  height: 240,
  bitDepth: 10,
  chroma: "420",
});
await decodeStream("hevc", "bear-1280x720-hevc-10bit-hdr10.hevc", {
  frames: 82,
  width: 1280,
  height: 720,
  bitDepth: 10,
  chroma: "420",
});
await decodeIvf("vp9", "vp90_2_10_show_existing_frame2.vp9.ivf", {
  frames: 16,
  width: 352,
  height: 288,
  bitDepth: 8,
  chroma: "420",
});
await decodeIvf("av1", "test-25fps.av1.ivf", {
  frames: 250,
  width: 320,
  height: 240,
  bitDepth: 8,
  chroma: "420",
});
await decodeIvf("av1", "bear_av1_720p_444_10bit.ivf", {
  frames: 2,
  width: 1280,
  height: 720,
  bitDepth: 10,
  chroma: "444",
});
await decodePacket("av1", "av1-monochrome-I-frame-320x240-10bpp", {
  frames: 1,
  width: 320,
  height: 240,
  bitDepth: 10,
  chroma: "400",
});

console.log("SoundKit upstream release-WASM video conformance passed");
