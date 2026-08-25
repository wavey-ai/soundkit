#!/usr/bin/env node
// Direct FFmpeg AAC decoder benchmark compiled with Emscripten and simd128.
// Usage: node aac-wasm-bench/bench-ffmpeg-wasm-node.mjs \
//   [iterations=5] [rounds=5] [package=pkg-ffmpeg]

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const iterations = Number(process.argv[2] ?? 5);
const rounds = Number(process.argv[3] ?? 5);
const packageName = process.argv[4] ?? "pkg-ffmpeg";
if (!Number.isSafeInteger(iterations) || iterations < 1) {
  throw new Error(`iterations must be a positive integer, got ${process.argv[2]}`);
}
if (!Number.isSafeInteger(rounds) || rounds < 1) {
  throw new Error(`rounds must be a positive integer, got ${process.argv[3]}`);
}

const scriptDir = dirname(fileURLToPath(import.meta.url));
const { default: createFfmpegModule } = await import(
  `./reference/${packageName}/ffmpeg-aac.mjs`
);
const fixture = readFileSync(
  join(scriptDir, "../golden/aac/WESTSIDE_MIX_4_CONFIRMATION_130323_256k.aac"),
);
const module = await createFfmpegModule();
const fixturePointer = module._malloc(fixture.length);
module.HEAPU8.set(fixture, fixturePointer);

console.log(
  `wasm-aac-decode fixture=WESTSIDE_MIX_4_CONFIRMATION_130323_256k.aac iterations=${iterations} rounds=${rounds} mode=ffmpeg package=${packageName} node=${process.version}`,
);

const records = [];
for (let round = 0; round < rounds; ++round) {
  const elapsedMs = module._ffmpeg_aac_bench(
    fixturePointer,
    fixture.length,
    iterations,
  );
  const error = module._ffmpeg_aac_last_error();
  if (elapsedMs < 0 || error !== 0) {
    throw new Error(`FFmpeg AAC benchmark failed with AVERROR ${error}`);
  }

  const decodedFrames = module._ffmpeg_aac_last_decoded_frames() >>> 0;
  const samplesPerChannel = module._ffmpeg_aac_last_samples_per_channel();
  const checksum =
    (BigInt(module._ffmpeg_aac_last_checksum_high() >>> 0) << 32n) |
    BigInt(module._ffmpeg_aac_last_checksum_low() >>> 0);
  const audioSeconds = samplesPerChannel / 48000;
  const rtf = elapsedMs / 1000 / audioSeconds;
  const framesPerSecond = decodedFrames / (elapsedMs / 1000);
  const record = { elapsedMs, decodedFrames, rtf, framesPerSecond, checksum };
  records.push(record);
  console.log(
    `ffmpeg-aac-wasm decoded_frames=${decodedFrames} elapsed_ms=${elapsedMs.toFixed(3)} rtf=${rtf.toFixed(6)} frames_per_sec=${framesPerSecond.toFixed(1)} checksum=${checksum.toString(16).padStart(16, "0")}`,
  );
}

module._free(fixturePointer);
const elapsed = records.map((record) => record.elapsedMs);
const medianElapsedMs = median(elapsed);
const bestElapsedMs = Math.min(...elapsed);
const medianFramesPerSecond =
  records[0].decodedFrames / (medianElapsedMs / 1000);
console.log(
  `ffmpeg-aac-wasm median_elapsed_ms=${medianElapsedMs.toFixed(3)} best_elapsed_ms=${bestElapsedMs.toFixed(3)} median_frames_per_sec=${medianFramesPerSecond.toFixed(1)}`,
);

function median(values) {
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[middle - 1] + sorted[middle]) / 2
    : sorted[middle];
}
