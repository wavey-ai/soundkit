#!/usr/bin/env node

const fs = require("fs");
const { performance } = require("perf_hooks");

const [, , scalarPath, simdPath] = process.argv;
if (!scalarPath || !simdPath) {
  console.error("usage: run_wasm_celt_bench.js scalar.wasm simd.wasm");
  process.exit(2);
}

async function loadBench(path) {
  const { instance } = await WebAssembly.instantiate(fs.readFileSync(path), {});
  const bench = instance.exports.raw_celt_encode_bench;
  if (typeof bench !== "function") {
    throw new Error(`${path} does not export raw_celt_encode_bench`);
  }
  return bench;
}

function median(samples) {
  samples.sort((a, b) => a - b);
  return samples[Math.floor(samples.length / 2)];
}

function timeBench(bench, frameSize, bitrate, seconds, repeats) {
  const frames = Math.floor((48_000 * seconds) / frameSize);
  bench(10, frameSize, bitrate);

  const samples = [];
  let checksum = 0;
  for (let i = 0; i < repeats; i += 1) {
    const start = performance.now();
    checksum = bench(frames, frameSize, bitrate);
    samples.push(performance.now() - start);
  }

  return { frames, ms: median(samples), checksum };
}

(async () => {
  const repeats = Number(process.env.REPEATS || 9);
  const seconds = Number(process.env.AUDIO_SECONDS || 2);
  const bitrate = Number(process.env.BITRATE || 128_000);
  const scalar = await loadBench(scalarPath);
  const simd = await loadBench(simdPath);

  console.log("| Frame | Frames | Scalar | SIMD | Delta | Checksum |");
  console.log("| ---: | ---: | ---: | ---: | ---: | ---: |");
  for (const frameSize of [120, 240, 480, 960]) {
    const scalarResult = timeBench(scalar, frameSize, bitrate, seconds, repeats);
    const simdResult = timeBench(simd, frameSize, bitrate, seconds, repeats);
    const delta = ((simdResult.ms - scalarResult.ms) / scalarResult.ms) * 100;
    const checksum =
      scalarResult.checksum === simdResult.checksum
        ? `${scalarResult.checksum}`
        : `${scalarResult.checksum}/${simdResult.checksum}`;
    console.log(
      `| ${frameSize} | ${scalarResult.frames} | ${scalarResult.ms.toFixed(3)} ms | ` +
        `${simdResult.ms.toFixed(3)} ms | ${delta.toFixed(1)}% | ${checksum} |`,
    );
  }
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
