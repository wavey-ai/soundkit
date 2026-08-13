import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

Error.stackTraceLimit = 100;

const scriptPath = fileURLToPath(import.meta.url);
const upstreamRoot = resolve("build/video-compat/upstream");
const generatedRoot = resolve(process.env.SOUNDKIT_MEDIA_FIXTURE_ROOT ?? "testdata/video-compat/never-final");
const wasmPackageRoot = resolve(process.env.SOUNDKIT_WASM_PACKAGE_ROOT ?? "soundkit-wasm/pkg");

function ivfLargestPacket(bytes) {
  assert.equal(Buffer.from(bytes.subarray(0, 4)).toString("ascii"), "DKIF");
  let cursor = bytes.readUInt16LE(6);
  let largest = Buffer.alloc(0);
  while (cursor < bytes.length) {
    assert.ok(cursor + 12 <= bytes.length);
    const size = bytes.readUInt32LE(cursor);
    cursor += 12;
    assert.ok(cursor + size <= bytes.length);
    const packet = bytes.subarray(cursor, cursor + size);
    if (packet.length > largest.length) largest = packet;
    cursor += size;
  }
  return largest;
}

function mutate(source, mutation) {
  if (mutation === "empty") return Buffer.alloc(0);
  if (mutation === "truncate-quarter") return source.subarray(0, Math.max(1, source.length >> 2));
  if (mutation === "truncate-half") return source.subarray(0, Math.max(1, source.length >> 1));
  if (mutation === "truncate-tail") return source.subarray(0, Math.max(1, source.length - 1));
  const output = Buffer.from(source);
  if (mutation === "flip-head") {
    output[Math.min(3, output.length - 1)] ^= 0xff;
  } else if (mutation === "flip-middle") {
    output[output.length >> 1] ^= 0xff;
  } else if (mutation === "zero-middle") {
    const start = Math.max(0, (output.length >> 1) - 16);
    output.fill(0, start, Math.min(output.length, start + 32));
  } else {
    throw new Error(`unknown mutation: ${mutation}`);
  }
  return output;
}

async function runCase(specification) {
  const initModule = await import(pathToFileURL(resolve(wasmPackageRoot, "soundkit_wasm.js")));
  await initModule.default({
    module_or_path: await readFile(resolve(wasmPackageRoot, "soundkit_wasm_bg.wasm")),
  });
  const source = await readFile(specification.path);
  let packet = specification.framing === "ivf" ? ivfLargestPacket(source) : source;
  if (specification.framing === "mp4-video-packet") {
    const index = initModule.WasmMp4MediaIndex.fromFile(source);
    try {
      let videoSample;
      for (let sampleIndex = 0; sampleIndex < index.sampleCount; sampleIndex += 1) {
        const candidate = index.sample(sampleIndex);
        if (candidate.kind === "video") {
          videoSample = { ...candidate, sampleIndex };
          break;
        }
      }
      assert.ok(videoSample, `${specification.path} has a video sample`);
      packet = index.packet(
        videoSample.sampleIndex,
        source.subarray(videoSample.offset, videoSample.offset + videoSample.size),
      ).data;
    } finally {
      index.free();
    }
  }
  const input = mutate(packet, specification.mutation);
  if (specification.framing === "mp4") {
    let index;
    let fatalError;
    try {
      try {
        index = initModule.WasmMp4MediaIndex.fromFile(input);
        if (index.sampleCount > 0) {
          const sample = index.sample(0);
          index.packet(0, input.subarray(sample.offset, sample.offset + sample.size));
        }
      } catch (error) {
        const message = String(error);
        if (
          error instanceof WebAssembly.RuntimeError ||
          /unreachable|out of bounds|memory access|stack overflow/i.test(message)
        ) fatalError = error;
      }
    } finally {
      try {
        index?.free();
      } catch (error) {
        fatalError ??= error;
      }
    }
    if (fatalError) throw fatalError;
    return;
  }
  if (specification.framing === "fmp4") {
    const demuxer = new initModule.WasmMp4MediaDemuxer();
    let fatalError;
    try {
      try {
        for (let offset = 0; offset < input.length; offset += 4093) {
          demuxer.push(input.subarray(offset, offset + 4093));
        }
        demuxer.flush();
      } catch (error) {
        const message = String(error);
        if (
          error instanceof WebAssembly.RuntimeError ||
          /unreachable|out of bounds|memory access|stack overflow/i.test(message)
        ) fatalError = error;
      }
    } finally {
      try {
        demuxer.free();
      } catch (error) {
        fatalError ??= error;
      }
    }
    if (fatalError) throw fatalError;
    return;
  }
  if (specification.framing === "webm") {
    const demuxer = new initModule.WasmWebmMediaDemuxer();
    let fatalError;
    try {
      try {
        for (let offset = 0; offset < input.length; offset += 4093) {
          demuxer.push(input.subarray(offset, offset + 4093));
        }
        demuxer.flush();
      } catch (error) {
        const message = String(error);
        if (
          error instanceof WebAssembly.RuntimeError ||
          /unreachable|out of bounds|memory access|stack overflow/i.test(message)
        ) fatalError = error;
      }
    } finally {
      try {
        demuxer.free();
      } catch (error) {
        fatalError ??= error;
      }
    }
    if (fatalError) throw fatalError;
    return;
  }
  if (specification.framing === "mxf") {
    const demuxer = new initModule.WasmMxfMediaDemuxer();
    let fatalError;
    try {
      try {
        for (let offset = 0; offset < input.length; offset += 32749) {
          demuxer.push(input.subarray(offset, offset + 32749));
        }
        demuxer.flush();
      } catch (error) {
        const message = String(error);
        if (
          error instanceof WebAssembly.RuntimeError ||
          /unreachable|out of bounds|memory access|stack overflow/i.test(message)
        ) fatalError = error;
      }
    } finally {
      try {
        demuxer.free();
      } catch (error) {
        fatalError ??= error;
      }
    }
    if (fatalError) throw fatalError;
    return;
  }
  const decoder = new initModule.WasmVideoDecoder(specification.codec);
  let fatalError;
  try {
    try {
      if (specification.framing === "stream") {
        decoder.decodeStream(input);
      } else {
        decoder.decode(input, 0, Number.NaN);
      }
      decoder.flush();
    } catch (error) {
      const message = String(error);
      if (
        error instanceof WebAssembly.RuntimeError ||
        /unreachable|out of bounds|memory access|stack overflow/i.test(message)
      ) {
        fatalError = error;
      }
    }
  } finally {
    try {
      decoder.free();
    } catch (error) {
      fatalError ??= error;
    }
  }
  if (fatalError) throw fatalError;
}

if (process.argv[2] === "--case") {
  const specification = JSON.parse(Buffer.from(process.argv[3], "base64url").toString("utf8"));
  await runCase(specification);
  process.exit(0);
}

const sources = [
  { codec: "h264", path: resolve(upstreamRoot, "test-25fps.h264"), framing: "stream" },
  { codec: "hevc", path: resolve(upstreamRoot, "test-25fps.hevc10"), framing: "stream" },
  { codec: "vp9", path: resolve(upstreamRoot, "vp90_2_10_show_existing_frame2.vp9.ivf"), framing: "ivf" },
  { codec: "av1", path: resolve(upstreamRoot, "test-25fps.av1.ivf"), framing: "ivf" },
  { codec: "av1", path: resolve(upstreamRoot, "bear_av1_720p_444_10bit.ivf"), framing: "ivf" },
  {
    codec: "prores",
    path: resolve(generatedRoot, "prores-422-hq-pcm.mov"),
    framing: "mp4-video-packet",
  },
  {
    codec: "prores",
    path: resolve(generatedRoot, "prores-4444xq-alpha-pcm.mov"),
    framing: "mp4-video-packet",
  },
  { codec: "container", path: resolve(generatedRoot, "h264-high-aac.mp4"), framing: "mp4" },
  {
    codec: "container",
    path: resolve(generatedRoot, "h264-aac-fragmented.mp4"),
    framing: "fmp4",
  },
  { codec: "container", path: resolve(generatedRoot, "h264-aac-cmaf.mp4"), framing: "fmp4" },
  { codec: "container", path: resolve(generatedRoot, "h264-aac-dash.mp4"), framing: "fmp4" },
  {
    codec: "container",
    path: resolve(generatedRoot, "h264-aac-separate-moof.mp4"),
    framing: "fmp4",
  },
  { codec: "container", path: resolve(generatedRoot, "h264-vfr-aac.mp4"), framing: "mp4" },
  { codec: "container", path: resolve(generatedRoot, "h264-high422-aac.mp4"), framing: "mp4" },
  { codec: "container", path: resolve(generatedRoot, "hevc-main10-pcm.mov"), framing: "mp4" },
  { codec: "container", path: resolve(generatedRoot, "hevc-main422-10-aac.mov"), framing: "mp4" },
  { codec: "container", path: resolve(generatedRoot, "prores-4444-alpha-pcm.mov"), framing: "mp4" },
  { codec: "container", path: resolve(generatedRoot, "vp9-profile0-opus.webm"), framing: "webm" },
  { codec: "container", path: resolve(generatedRoot, "vp9-profile2-10bit-opus.webm"), framing: "webm" },
  { codec: "container", path: resolve(generatedRoot, "matroska-h264-aac.mkv"), framing: "webm" },
  { codec: "container", path: resolve(generatedRoot, "matroska-hevc-aac.mkv"), framing: "webm" },
  { codec: "container", path: resolve(generatedRoot, "av1-main-opus.webm"), framing: "webm" },
  { codec: "container", path: resolve(generatedRoot, "av1-main10-opus.webm"), framing: "webm" },
  { codec: "container", path: resolve(generatedRoot, "dnxhr-hqx-pcm.mxf"), framing: "mxf" },
];
const mutations = [
  "empty",
  "truncate-quarter",
  "truncate-half",
  "truncate-tail",
  "flip-head",
  "flip-middle",
  "zero-middle",
];

let completed = 0;
for (const source of sources) {
  for (const mutation of mutations) {
    const specification = { ...source, mutation };
    const encoded = Buffer.from(JSON.stringify(specification)).toString("base64url");
    const result = spawnSync(process.execPath, [scriptPath, "--case", encoded], {
      encoding: "utf8",
      timeout: 10_000,
      maxBuffer: 1024 * 1024,
    });
    assert.equal(
      result.status,
      0,
      `${source.codec}/${source.path.split("/").at(-1)}/${mutation} failed or exceeded 10 seconds\n${result.stderr}`,
    );
    completed += 1;
  }
}

console.log(`SoundKit release-WASM mutation fuzz passed ${completed} bounded cases`);
