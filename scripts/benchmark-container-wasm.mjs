import { spawnSync } from "node:child_process";
import { readFile } from "node:fs/promises";
import { performance } from "node:perf_hooks";
import { fileURLToPath } from "node:url";

const chunkSizes = [1, 188, 4 * 1024, 64 * 1024, 4 * 1024 * 1024];
const cases = {
  fMP4: {
    file: "testdata/video-compat/never-final/h264-aac-fragmented.mp4",
    format: "fmp4",
    type: "stream",
  },
  WebM: {
    file: "testdata/video-compat/never-final/vp9-profile0-opus.webm",
    format: "webm",
    type: "stream",
  },
  TS: {
    file: "testdata/mpeg-ts/aac-stereo-48k.ts",
    format: "mpeg-ts",
    type: "stream",
  },
  M2TS: {
    file: "testdata/mpeg-ts/aac-stereo-48k.m2ts",
    format: "m2ts",
    type: "stream",
  },
  Ogg: {
    file: "testdata/ogg_opus/A_Tusk_is_used_to_make_costly_gifts_48khz.ogg",
    format: "ogg-opus",
    type: "ogg",
  },
  MP4: {
    file: "testdata/video-compat/never-final/h264-high-aac.mp4",
    type: "mp4-index",
  },
  CAF: {
    file: "testdata/alac/A_Tusk_is_used_to_make_costly_gifts.caf",
    type: "caf-index",
  },
};

function iterationsFor(chunkSize) {
  if (chunkSize === 1) return 1;
  if (chunkSize === 188) return 3;
  if (chunkSize === 4096) return 10;
  return 20;
}

function label(chunkSize) {
  if (chunkSize === null) return "seekable";
  if (chunkSize === 1) return "1 B";
  if (chunkSize === 188) return "188 B";
  if (chunkSize === 4096) return "4 KiB";
  if (chunkSize === 65_536) return "64 KiB";
  return "4 MiB";
}

async function runChild(caseName, chunkSize) {
  const specification = cases[caseName];
  if (!specification) throw new Error(`unknown benchmark case: ${caseName}`);
  const module = await import("../soundkit-wasm/pkg/soundkit_wasm.js");
  await module.default({
    module_or_path: await readFile(
      new URL("../soundkit-wasm/pkg/soundkit_wasm_bg.wasm", import.meta.url),
    ),
  });
  const data = await readFile(specification.file);
  const wasmStart = module.wasmMemoryBytes();

  const operation = () => {
    let events = 0;
    if (specification.type === "stream") {
      const demuxer = module.WasmAudioTrackDemuxer.newWithFormat(specification.format);
      try {
        for (let offset = 0; offset < data.length; offset += chunkSize) {
          events += demuxer.push(data.subarray(offset, offset + chunkSize)).length;
        }
        events += demuxer.flush().length;
      } finally {
        demuxer.free();
      }
    } else if (specification.type === "ogg") {
      const demuxer = module.WasmOpusDeboxer.newWithFormat(specification.format);
      try {
        for (let offset = 0; offset < data.length; offset += chunkSize) {
          events += demuxer.push(data.subarray(offset, offset + chunkSize)).length;
        }
        events += demuxer.flush().length;
      } finally {
        demuxer.free();
      }
    } else if (specification.type === "mp4-index") {
      const index = module.WasmMp4MediaIndex.fromFile(data);
      try {
        events = index.sampleCount + index.tracks().length;
      } finally {
        index.free();
      }
    } else if (specification.type === "caf-index") {
      const index = module.WasmCafAudioIndex.fromFile(data);
      try {
        events = index.sampleCount + 1;
      } finally {
        index.free();
      }
    }
    return events;
  };

  const iterations = chunkSize === null ? 20 : iterationsFor(chunkSize);
  let events = 0;
  const started = performance.now();
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    events = operation();
  }
  const elapsedSeconds = (performance.now() - started) / 1000;
  const wasmPeak = module.wasmMemoryBytes();
  return {
    container: caseName,
    chunkSize,
    bytes: data.length,
    events,
    iterations,
    mibPerSecond: data.length * iterations / elapsedSeconds / (1024 * 1024),
    wasmBytes: wasmPeak,
    wasmGrowthBytes: wasmPeak - wasmStart,
  };
}

function runParent() {
  const script = fileURLToPath(import.meta.url);
  const rows = [];
  for (const caseName of Object.keys(cases)) {
    const sizes = cases[caseName].type.endsWith("index") ? [null] : chunkSizes;
    for (const chunkSize of sizes) {
      const child = spawnSync(
        process.execPath,
        [script, "--child", caseName, chunkSize === null ? "seekable" : String(chunkSize)],
        { cwd: process.cwd(), encoding: "utf8" },
      );
      if (child.status !== 0) {
        process.stderr.write(child.stderr);
        process.exit(child.status ?? 1);
      }
      rows.push(JSON.parse(child.stdout));
    }
  }

  console.log(
    `${"container".padEnd(10)} ${"push".padStart(9)} ${"MiB/s".padStart(11)} ` +
      `${"WASM bytes".padStart(12)} ${"growth".padStart(12)} ${"events".padStart(9)}`,
  );
  for (const row of rows) {
    console.log(
      `${row.container.padEnd(10)} ${label(row.chunkSize).padStart(9)} ` +
        `${row.mibPerSecond.toFixed(2).padStart(11)} ` +
        `${String(row.wasmBytes).padStart(12)} ` +
        `${String(row.wasmGrowthBytes).padStart(12)} ${String(row.events).padStart(9)}`,
    );
  }

  for (const caseName of ["fMP4", "WebM", "TS", "M2TS", "Ogg"]) {
    const throughput = (size) => rows.find(
      (row) => row.container === caseName && row.chunkSize === size,
    ).mibPerSecond;
    const packet = throughput(64 * 1024);
    const large = throughput(4 * 1024 * 1024);
    if (large < packet * 0.1) {
      throw new Error(
        `${caseName} 4 MiB throughput ${large.toFixed(2)} MiB/s is below 10% ` +
          `of its 64 KiB throughput ${packet.toFixed(2)} MiB/s`,
      );
    }
  }
}

if (process.argv[2] === "--child") {
  const chunkSize = process.argv[4] === "seekable" ? null : Number(process.argv[4]);
  console.log(JSON.stringify(await runChild(process.argv[3], chunkSize)));
} else {
  runParent();
}
