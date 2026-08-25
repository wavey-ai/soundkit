#!/usr/bin/env node
// Like-for-like WASM AAC decode benchmark: SoundKit AAC-LC vs Symphonia.
// Both decoders are compiled into the same wasm module and run in the same
// Node process with alternating warm rounds to reduce bias.
//
// Usage: node aac-wasm-bench/bench-wasm-node.mjs \
//   [iterations=5] [rounds=5] [both|soundkit|symphonia] [package=pkg]

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const iterations = Number(process.argv[2] ?? 5);
const rounds = Number(process.argv[3] ?? 5);
const mode = process.argv[4] ?? "both";
const packageName = process.argv[5] ?? "pkg";

if (!Number.isSafeInteger(iterations) || iterations < 1) {
  throw new Error(`iterations must be a positive integer, got ${process.argv[2]}`);
}
if (!Number.isSafeInteger(rounds) || rounds < 1) {
  throw new Error(`rounds must be a positive integer, got ${process.argv[3]}`);
}
if (!new Set(["both", "soundkit", "symphonia"]).has(mode)) {
  throw new Error(`mode must be both, soundkit, or symphonia; got ${mode}`);
}

const pkgDir = dirname(fileURLToPath(import.meta.url));
const packageDir = join(pkgDir, packageName);
const {
  default: init,
  bench_soundkit_lc_wasm,
  bench_symphonia_wasm,
} = await import(`./${packageName}/aac_wasm_bench.js`);
await init({
  module_or_path: readFileSync(join(packageDir, "aac_wasm_bench_bg.wasm")),
});

console.log(
  `wasm-aac-decode fixture=WESTSIDE_MIX_4_CONFIRMATION_130323_256k.aac iterations=${iterations} rounds=${rounds} mode=${mode} package=${packageName} node=${process.version}`,
);

const runners = [
  {
    key: "soundkit",
    reportName: "soundkit-lc-wasm",
    run: bench_soundkit_lc_wasm,
  },
  {
    key: "symphonia",
    reportName: "symphonia-wasm",
    run: bench_symphonia_wasm,
  },
].filter(({ key }) => mode === "both" || mode === key);

// Warm both implementations before measured rounds so V8 has compiled their
// hot Wasm bodies. The Rust entry points also reset codec state outside their
// timed regions.
for (const { run } of runners) {
  run(1);
}

const records = new Map(runners.map(({ key }) => [key, []]));
for (let round = 0; round < rounds; ++round) {
  const order = round % 2 === 0 ? runners : [...runners].reverse();
  for (const { key, run } of order) {
    records.get(key).push(toRecord(run(iterations)));
  }
}

const summaries = new Map();
for (const { key, reportName } of runners) {
  summaries.set(key, report(reportName, records.get(key)));
}
if (summaries.has("soundkit") && summaries.has("symphonia")) {
  const soundkitMs = summaries.get("soundkit").medianElapsedMs;
  const symphoniaMs = summaries.get("symphonia").medianElapsedMs;
  console.log(
    `wasm-comparison soundkit_vs_symphonia=${(soundkitMs / symphoniaMs).toFixed(4)}x soundkit_delta_pct=${(((soundkitMs - symphoniaMs) / symphoniaMs) * 100).toFixed(2)}`,
  );
}

function toRecord(line) {
  const record = {};
  for (const token of line.split(/\s+/)) {
    const [key, value] = token.split("=");
    record[key] = value;
  }
  return record;
}

function report(name, records) {
  for (const record of records) {
    console.log(
      `${record.name} decoded_frames=${record.decoded_frames} elapsed_ms=${record.elapsed_ms} rtf=${record.rtf} frames_per_sec=${record.frames_per_sec} checksum=${record.checksum}`,
    );
  }
  const elapsed = records.map((record) => Number(record.elapsed_ms));
  const medianElapsedMs = median(elapsed);
  const bestElapsedMs = Math.min(...elapsed);
  const decodedFrames = Number(records[0].decoded_frames);
  const medianFramesPerSecond = decodedFrames / (medianElapsedMs / 1000);
  console.log(
    `${name} median_elapsed_ms=${medianElapsedMs.toFixed(3)} best_elapsed_ms=${bestElapsedMs.toFixed(3)} median_frames_per_sec=${medianFramesPerSecond.toFixed(1)}`,
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
