#!/usr/bin/env node
// Alternating, single-process AAC-LC Wasm benchmark for SoundKit, FFmpeg,
// FDK-AAC, and Symphonia.
//
// Usage: node aac-wasm-bench/bench-all-wasm-node.mjs \
//   [iterations=3] [rounds=11] [rust-package=pkg] \
//   [ffmpeg-package=pkg-ffmpeg] [fdk-package=pkg-fdk] [fixture]

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const iterations = positiveInteger(process.argv[2] ?? 3, "iterations");
const rounds = positiveInteger(process.argv[3] ?? 11, "rounds");
const rustPackage = process.argv[4] ?? "pkg";
const ffmpegPackage = process.argv[5] ?? "pkg-ffmpeg";
const fdkPackage = process.argv[6] ?? "pkg-fdk";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const fixturePath = process.argv[7] ??
  join(scriptDir, "../golden/aac/WESTSIDE_MIX_4_CONFIRMATION_130323_256k.aac");
const fixture = readFileSync(fixturePath);
const sampleRates = [
  96000, 88200, 64000, 48000, 44100, 32000, 24000,
  22050, 16000, 12000, 11025, 8000, 7350,
];
const sampleRate = sampleRates[(fixture[2] >> 2) & 15];

const rust = await import(`./${rustPackage}/aac_wasm_bench.js`);
await rust.default({
  module_or_path: readFileSync(
    join(scriptDir, rustPackage, "aac_wasm_bench_bg.wasm"),
  ),
});

const { default: createFfmpegModule } = await import(
  `./reference/${ffmpegPackage}/ffmpeg-aac.mjs`
);
const ffmpeg = await createFfmpegModule();
const ffmpegFixture = copyIntoModule(ffmpeg, fixture);

const { default: createFdkModule } = await import(
  `./reference/${fdkPackage}/fdk-aac.mjs`
);
const fdk = await createFdkModule();
const fdkFixture = copyIntoModule(fdk, fixture);

const runners = [
  {
    key: "soundkit",
    run(count) {
      return rustRecord(rust.bench_soundkit_lc_data_wasm(fixture, count));
    },
  },
  {
    key: "ffmpeg",
    run(count) {
      return cRecord(
        ffmpeg,
        "ffmpeg_aac",
        "ffmpeg-aac-wasm",
        ffmpegFixture,
        fixture.length,
        count,
        sampleRate,
      );
    },
  },
  {
    key: "fdk",
    run(count) {
      return cRecord(
        fdk,
        "fdk_aac",
        "fdk-aac-wasm",
        fdkFixture,
        fixture.length,
        count,
        sampleRate,
      );
    },
  },
  {
    key: "symphonia",
    run(count) {
      return rustRecord(rust.bench_symphonia_data_wasm(fixture, count));
    },
  },
];

console.log(
  `wasm-aac-decode fixture=${fixturePath.split("/").at(-1)} iterations=${iterations} rounds=${rounds} mode=all rust_package=${rustPackage} ffmpeg_package=${ffmpegPackage} fdk_package=${fdkPackage} node=${process.version}`,
);

// Compile every hot Wasm body before any measured round. Each native entry
// point also performs one complete codec warm-up pass outside its timer.
for (const runner of runners) {
  runner.run(1);
  runner.run(1);
}

const records = new Map(runners.map(({ key }) => [key, []]));
for (let round = 0; round < rounds; ++round) {
  const order = round % 2 === 0 ? runners : [...runners].reverse();
  console.log(`round=${round + 1} order=${order.map(({ key }) => key).join(",")}`);
  const roundRecords = [];
  for (const runner of order) {
    const record = runner.run(iterations);
    const prior = records.get(runner.key);
    if (prior.length > 0 && record.checksum !== prior[0].checksum) {
      throw new Error(
        `${runner.key} checksum changed: ${prior[0].checksum} -> ${record.checksum}`,
      );
    }
    prior.push(record);
    roundRecords.push(record);
    printRecord(record);
  }
  const expected = roundRecords[0];
  for (const record of roundRecords.slice(1)) {
    if (
      record.decodedFrames !== expected.decodedFrames ||
      record.samplesPerChannel !== expected.samplesPerChannel
    ) {
      throw new Error(
        `${record.name} output length ${record.decodedFrames}/${record.samplesPerChannel} does not match ${expected.name} ${expected.decodedFrames}/${expected.samplesPerChannel}`,
      );
    }
  }
}

const summaries = new Map();
for (const runner of runners) {
  summaries.set(runner.key, report(runner.key, records.get(runner.key)));
}

const soundkitMs = summaries.get("soundkit").medianElapsedMs;
for (const key of ["ffmpeg", "fdk", "symphonia"]) {
  const referenceMs = summaries.get(key).medianElapsedMs;
  console.log(
    `wasm-comparison soundkit_vs_${key}=${(soundkitMs / referenceMs).toFixed(4)}x soundkit_delta_pct=${(((soundkitMs - referenceMs) / referenceMs) * 100).toFixed(2)}`,
  );
}

ffmpeg._free(ffmpegFixture);
fdk._free(fdkFixture);

function positiveInteger(value, name) {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < 1) {
    throw new Error(`${name} must be a positive integer, got ${value}`);
  }
  return parsed;
}

function copyIntoModule(module, data) {
  const pointer = module._malloc(data.length);
  module.HEAPU8.set(data, pointer);
  return pointer;
}

function rustRecord(line) {
  const values = {};
  for (const token of line.split(/\s+/)) {
    const separator = token.indexOf("=");
    if (separator !== -1) {
      values[token.slice(0, separator)] = token.slice(separator + 1);
    }
  }
  return {
    name: values.name,
    decodedFrames: Number(values.decoded_frames),
    samplesPerChannel: Number(values.samples_per_channel),
    elapsedMs: Number(values.elapsed_ms),
    rtf: Number(values.rtf),
    framesPerSecond: Number(values.frames_per_sec),
    checksum: values.checksum,
  };
}

function cRecord(
  module,
  prefix,
  name,
  fixturePointer,
  fixtureLength,
  count,
  rate,
) {
  const elapsedMs = module[`_${prefix}_bench`](
    fixturePointer,
    fixtureLength,
    count,
  );
  const error = module[`_${prefix}_last_error`]();
  if (elapsedMs < 0 || error !== 0) {
    throw new Error(`${name} failed with error ${error}`);
  }
  const decodedFrames = module[`_${prefix}_last_decoded_frames`]() >>> 0;
  const samplesPerChannel = module[`_${prefix}_last_samples_per_channel`]();
  const checksum =
    (BigInt(module[`_${prefix}_last_checksum_high`]() >>> 0) << 32n) |
    BigInt(module[`_${prefix}_last_checksum_low`]() >>> 0);
  const audioSeconds = samplesPerChannel / rate;
  return {
    name,
    decodedFrames,
    samplesPerChannel,
    elapsedMs,
    rtf: elapsedMs / 1000 / audioSeconds,
    framesPerSecond: decodedFrames / (elapsedMs / 1000),
    checksum: checksum.toString(16).padStart(16, "0"),
  };
}

function printRecord(record) {
  console.log(
    `${record.name} decoded_frames=${record.decodedFrames} samples_per_channel=${record.samplesPerChannel} elapsed_ms=${record.elapsedMs.toFixed(3)} rtf=${record.rtf.toFixed(6)} frames_per_sec=${record.framesPerSecond.toFixed(1)} checksum=${record.checksum}`,
  );
}

function report(key, recordsForRunner) {
  const elapsed = recordsForRunner.map(({ elapsedMs }) => elapsedMs);
  const medianElapsedMs = median(elapsed);
  const bestElapsedMs = Math.min(...elapsed);
  const decodedFrames = recordsForRunner[0].decodedFrames;
  const medianFramesPerSecond = decodedFrames / (medianElapsedMs / 1000);
  console.log(
    `${key} median_elapsed_ms=${medianElapsedMs.toFixed(3)} best_elapsed_ms=${bestElapsedMs.toFixed(3)} median_frames_per_sec=${medianFramesPerSecond.toFixed(1)} checksum=${recordsForRunner[0].checksum}`,
  );
  return { medianElapsedMs, bestElapsedMs, medianFramesPerSecond };
}

function median(values) {
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[middle - 1] + sorted[middle]) / 2
    : sorted[middle];
}
