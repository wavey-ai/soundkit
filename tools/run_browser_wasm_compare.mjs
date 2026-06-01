#!/usr/bin/env node

import { createServer } from "node:http";
import { existsSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";
import { tmpdir } from "node:os";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const DEFAULT_CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const DEFAULT_CASES = ["c", "rust-cbr", "rust-vbr"];
const SUPPORTED_CASES = [...DEFAULT_CASES, "rust-cbr-reuse", "rust-vbr-reuse"];
const DEFAULT_BITRATES = [48_000, 128_000, 196_000];

function usage() {
  console.error(`usage: node tools/run_browser_wasm_compare.mjs [options]

Options:
  --seconds <n>        synthetic 48 kHz stereo fixture length (default: 10)
  --bitrates <csv>     bitrates in bps (default: 48000,128000,196000)
  --cases <csv>        c,rust-cbr,rust-vbr,rust-cbr-reuse,rust-vbr-reuse (default: c,rust-cbr,rust-vbr)
  --repeats <n>        fresh Chrome runs per case; median is reported (default: 1)
  --timeout-ms <n>     timeout per browser run (default: 90000)
  --chrome <path>      Chrome executable path
  --rust-pkg <dir>     wasm-pack pkg directory (default: ./pkg)
  --libopusjs <dir>    libopusjs release directory (default: ../libopusjs/release)
  --json <path>        also write raw result JSON
  --profile-rust-decode <path>
                      write a Chrome CPU profile for one Rust decode case
  --keep-open          keep the HTTP server running after the benchmark
  -h, --help           show this help
`);
}

function parseArgs(argv) {
  const options = {
    seconds: Number(process.env.BENCH_SECONDS || 10),
    bitrates: DEFAULT_BITRATES,
    cases: DEFAULT_CASES,
    repeats: Number(process.env.REPEATS || 1),
    timeoutMs: Number(process.env.TIMEOUT_MS || 90_000),
    chrome: process.env.CHROME || DEFAULT_CHROME,
    rustPkg: resolve(ROOT, "pkg"),
    libopusjs: resolve(ROOT, "../libopusjs/release"),
    json: null,
    profileRustDecode: null,
    keepOpen: false,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    const next = () => {
      if (i + 1 >= argv.length) {
        throw new Error(`${arg} requires a value`);
      }
      i += 1;
      return argv[i];
    };

    switch (arg) {
      case "-h":
      case "--help":
        usage();
        process.exit(0);
        break;
      case "--seconds":
        options.seconds = Number(next());
        break;
      case "--bitrates":
        options.bitrates = parseCsvNumbers(next(), arg);
        break;
      case "--cases":
        options.cases = next().split(",").map((value) => value.trim()).filter(Boolean);
        break;
      case "--repeats":
        options.repeats = Number(next());
        break;
      case "--timeout-ms":
        options.timeoutMs = Number(next());
        break;
      case "--chrome":
        options.chrome = next();
        break;
      case "--rust-pkg":
        options.rustPkg = resolve(next());
        break;
      case "--libopusjs":
        options.libopusjs = resolve(next());
        break;
      case "--json":
        options.json = next();
        break;
      case "--profile-rust-decode":
        options.profileRustDecode = next();
        break;
      case "--keep-open":
        options.keepOpen = true;
        break;
      default:
        throw new Error(`unknown option: ${arg}`);
    }
  }

  if (!Number.isFinite(options.seconds) || options.seconds <= 0) {
    throw new Error("--seconds must be a positive number");
  }
  if (!Number.isInteger(options.repeats) || options.repeats <= 0) {
    throw new Error("--repeats must be a positive integer");
  }
  if (!Number.isInteger(options.timeoutMs) || options.timeoutMs <= 0) {
    throw new Error("--timeout-ms must be a positive integer");
  }
  for (const name of options.cases) {
    if (!SUPPORTED_CASES.includes(name)) {
      throw new Error(`unknown case "${name}"; expected one of ${SUPPORTED_CASES.join(",")}`);
    }
  }

  return options;
}

function parseCsvNumbers(value, optionName) {
  const numbers = value
    .split(",")
    .map((item) => Number(item.trim()))
    .filter((item) => Number.isFinite(item));
  if (numbers.length === 0 || numbers.some((item) => item <= 0)) {
    throw new Error(`${optionName} must contain positive numbers`);
  }
  return numbers;
}

function requireFile(path, label) {
  if (!existsSync(path)) {
    throw new Error(`${label} not found: ${path}`);
  }
}

function makePage({ caseName, bitrate, seconds, cacheBust, profileDecode }) {
  return `<!doctype html>
<meta charset="utf-8">
<title>libopus-rs browser wasm benchmark</title>
<pre id="out">running ${caseName} ${bitrate}</pre>
<script type="module">
const SAMPLE_RATE = 48000;
const CHANNELS = 2;
const FRAME_SIZE = 960;
const FRAME_SAMPLES = FRAME_SIZE * CHANNELS;
const CASE_NAME = ${JSON.stringify(caseName)};
const BITRATE = ${bitrate};
const SECONDS = ${seconds};
const CACHE_BUST = ${JSON.stringify(cacheBust)};
const PROFILE_DECODE = ${profileDecode ? "true" : "false"};

function emit(kind, payload) {
  console.log("__BENCH_" + kind + "__" + JSON.stringify(payload));
}

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function makePcm() {
  const frames = Math.floor((SAMPLE_RATE * SECONDS) / FRAME_SIZE);
  const pcm = new Int16Array(frames * FRAME_SAMPLES);

  for (let frame = 0; frame < frames; frame += 1) {
    for (let i = 0; i < FRAME_SIZE; i += 1) {
      const t = (frame * FRAME_SIZE + i) / SAMPLE_RATE;
      const envelope = 0.55 + 0.35 * Math.sin(2 * Math.PI * 0.37 * t);
      const left =
        envelope *
        (0.48 * Math.sin(2 * Math.PI * 220 * t) +
          0.24 * Math.sin(2 * Math.PI * 659.25 * t) +
          0.12 * Math.sin(2 * Math.PI * 1760 * t));
      const right =
        envelope *
        (0.45 * Math.sin(2 * Math.PI * 246.94 * t) +
          0.22 * Math.sin(2 * Math.PI * 554.37 * t) +
          0.1 * Math.sin(2 * Math.PI * 1490 * t));
      const base = frame * FRAME_SAMPLES + i * CHANNELS;
      pcm[base] = Math.max(-32768, Math.min(32767, Math.round(left * 32767)));
      pcm[base + 1] = Math.max(-32768, Math.min(32767, Math.round(right * 32767)));
    }
  }

  return { pcm, frames, durationSec: (frames * FRAME_SIZE) / SAMPLE_RATE };
}

function packetStats(packets) {
  let bytes = 0;
  let min = Infinity;
  let max = 0;
  for (const packet of packets) {
    bytes += packet.length;
    min = Math.min(min, packet.length);
    max = Math.max(max, packet.length);
  }
  return { bytes, min, max, avg: bytes / packets.length };
}

function finish(name, fixture, packets, encodeMs, decodeMs, checksum, loadMs) {
  const stats = packetStats(packets);
  return {
    case: CASE_NAME,
    name,
    bitrate: BITRATE,
    targetKbps: BITRATE / 1000,
    frames: fixture.frames,
    durationSec: fixture.durationSec,
    loadMs,
    encodedBytes: stats.bytes,
    effectiveKbps: (stats.bytes * 8) / fixture.durationSec / 1000,
    packetMin: stats.min,
    packetMax: stats.max,
    packetAvg: stats.avg,
    encodeMs,
    decodeMs,
    encodeXrtf: (fixture.durationSec * 1000) / encodeMs,
    decodeXrtf: (fixture.durationSec * 1000) / decodeMs,
    checksum,
  };
}

async function loadC() {
  const start = performance.now();
  const moduleFactory = (await import("/c/libopus.mjs?v=" + CACHE_BUST)).default;
  const imported = performance.now();
  const module = await moduleFactory({
    locateFile: (url) => (url === "libopus.wasm" ? "/c/libopus.wasm?v=" + CACHE_BUST : url),
  });
  return { module, importMs: imported - start, initMs: performance.now() - imported, totalMs: performance.now() - start };
}

async function loadRust() {
  const start = performance.now();
  const module = await import("/rust/libopus_rs.js?v=" + CACHE_BUST);
  const imported = performance.now();
  const wasm = await module.default({ module_or_path: "/rust/libopus_rs_bg.wasm?v=" + CACHE_BUST });
  return { module, wasm, importMs: imported - start, initMs: performance.now() - imported, totalMs: performance.now() - start };
}

function benchC(module, fixture, loadMs) {
  const encoder = new module.Encoder(CHANNELS, SAMPLE_RATE, BITRATE, FRAME_SIZE);
  const packets = [];
  let started = performance.now();
  for (let frame = 0; frame < fixture.frames; frame += 1) {
    const input = fixture.pcm.subarray(frame * FRAME_SAMPLES, (frame + 1) * FRAME_SAMPLES);
    const result = encoder.enc_frame(input);
    if (!result.ok) {
      throw new Error("C/libopusjs encode failed at frame " + frame);
    }
    packets.push(result.encodedData);
  }
  const encodeMs = performance.now() - started;
  encoder.destroy();

  const decoder = new module.Decoder(CHANNELS, SAMPLE_RATE, FRAME_SIZE);
  let checksum = 0;
  started = performance.now();
  for (let frame = 0; frame < packets.length; frame += 1) {
    const result = decoder.dec_frame(packets[frame]);
    if (result.decodedSize !== FRAME_SIZE) {
      throw new Error("C/libopusjs decode failed at frame " + frame + ": " + result.decodedSize);
    }
    checksum = (checksum + (result.output[0] | 0) + (result.output[result.output.length - 1] | 0)) | 0;
  }
  const decodeMs = performance.now() - started;
  decoder.destroy();
  return finish("C libopusjs default VBR", fixture, packets, encodeMs, decodeMs, checksum, loadMs);
}

async function benchRust(module, wasm, fixture, vbr, reuseOutput, loadMs) {
  const encoder = new module.Encoder(CHANNELS, SAMPLE_RATE, BITRATE, FRAME_SIZE);
  encoder.set_vbr(vbr);
  const packets = [];
  let started = performance.now();
  for (let frame = 0; frame < fixture.frames; frame += 1) {
    const input = fixture.pcm.subarray(frame * FRAME_SAMPLES, (frame + 1) * FRAME_SAMPLES);
    const result = encoder.enc_frame(input);
    if (!result.ok) {
      throw new Error("libopus-rs encode failed at frame " + frame);
    }
    packets.push(result.encodedData);
    result.free();
  }
  const encodeMs = performance.now() - started;
  encoder.destroy();

  const decoder = new module.Decoder(CHANNELS, SAMPLE_RATE, FRAME_SIZE);
  let checksum = 0;
  if (PROFILE_DECODE) {
    emit("PROFILE_START", { case: CASE_NAME, bitrate: BITRATE });
    await wait(5);
  }
  started = performance.now();
  for (let frame = 0; frame < packets.length; frame += 1) {
    if (reuseOutput) {
      const decodedSize = decoder.dec_frame_reuse(packets[frame]);
      if (decodedSize !== FRAME_SIZE) {
        throw new Error("libopus-rs reuse decode failed at frame " + frame + ": " + decodedSize);
      }
      const output = new Int16Array(wasm.memory.buffer, decoder.outputPtr, decoder.outputLen);
      checksum = (checksum + (output[0] | 0) + (output[output.length - 1] | 0)) | 0;
    } else {
      const result = decoder.dec_frame(packets[frame]);
      if (result.decodedSize !== FRAME_SIZE) {
        throw new Error("libopus-rs decode failed at frame " + frame + ": " + result.decodedSize);
      }
      const output = result.output;
      checksum = (checksum + (output[0] | 0) + (output[output.length - 1] | 0)) | 0;
      result.free();
    }
  }
  const decodeMs = performance.now() - started;
  if (PROFILE_DECODE) {
    emit("PROFILE_END", { case: CASE_NAME, bitrate: BITRATE });
    await wait(5);
  }
  decoder.destroy();
  return finish(
    "Rust libopus-rs " + (vbr ? "VBR" : "CBR") + (reuseOutput ? " reuse" : ""),
    fixture,
    packets,
    encodeMs,
    decodeMs,
    checksum,
    loadMs,
  );
}

try {
  const fixture = makePcm();
  emit("INFO", {
    userAgent: navigator.userAgent,
    case: CASE_NAME,
    bitrate: BITRATE,
    frames: fixture.frames,
    durationSec: fixture.durationSec,
  });

  if (CASE_NAME === "c") {
    const loaded = await loadC();
    const row = benchC(loaded.module, fixture, loaded.totalMs);
    const { module: _module, ...load } = loaded;
    emit("DONE", { load, row });
  } else {
    const loaded = await loadRust();
    const row = await benchRust(
      loaded.module,
      loaded.wasm,
      fixture,
      CASE_NAME.includes("vbr"),
      CASE_NAME.endsWith("reuse"),
      loaded.totalMs,
    );
    const { module: _module, wasm: _wasm, ...load } = loaded;
    emit("DONE", { load, row });
  }
} catch (error) {
  emit("ERROR", { message: error && error.message, stack: error && error.stack });
}
</script>`;
}

function createAssetServer(paths) {
  const server = createServer((request, response) => {
    const url = new URL(request.url, "http://127.0.0.1");
    response.setHeader("Cache-Control", "no-store");

    if (url.pathname === "/bench.html") {
      const caseName = url.searchParams.get("case");
      const bitrate = Number(url.searchParams.get("bitrate"));
      const seconds = Number(url.searchParams.get("seconds"));
      const cacheBust = url.searchParams.get("v") || String(Date.now());
      const profileDecode = url.searchParams.get("profile") === "1";
      response.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
      response.end(makePage({ caseName, bitrate, seconds, cacheBust, profileDecode }));
      return;
    }

    const asset = {
      "/rust/libopus_rs.js": ["text/javascript; charset=utf-8", paths.rustJs],
      "/rust/libopus_rs_bg.wasm": ["application/wasm", paths.rustWasm],
      "/c/libopus.mjs": ["text/javascript; charset=utf-8", paths.cJs],
      "/c/libopus.wasm": ["application/wasm", paths.cWasm],
    }[url.pathname];

    if (!asset) {
      response.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
      response.end("not found\n");
      return;
    }

    response.writeHead(200, { "Content-Type": asset[0] });
    response.end(readFileSync(asset[1]));
  });
  return server;
}

async function listen(server) {
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  return server.address().port;
}

async function wait(ms) {
  await new Promise((resolve) => setTimeout(resolve, ms));
}

async function readJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${url}: ${response.status}`);
  }
  return response.json();
}

async function waitForDebuggerUrl(debugPort, stderr) {
  const deadline = Date.now() + 15_000;
  while (Date.now() < deadline) {
    try {
      const pages = await readJson(`http://127.0.0.1:${debugPort}/json/list`);
      const page = pages.find((item) => item.type === "page" && item.webSocketDebuggerUrl);
      if (page) {
        return page.webSocketDebuggerUrl;
      }
    } catch {
      // Chrome may not have opened the DevTools HTTP endpoint yet.
    }
    await wait(100);
  }
  throw new Error(`Chrome did not expose a DevTools page:\n${stderr()}`);
}

async function connectDevtools(wsUrl) {
  const socket = new WebSocket(wsUrl);
  let nextId = 1;
  const pending = new Map();

  await new Promise((resolve, reject) => {
    socket.addEventListener("open", resolve, { once: true });
    socket.addEventListener("error", reject, { once: true });
  });

  function send(method, params = {}) {
    const id = nextId;
    nextId += 1;
    socket.send(JSON.stringify({ id, method, params }));
    return new Promise((resolve, reject) => {
      pending.set(id, { resolve, reject });
    });
  }

  return { socket, pending, send };
}

async function stopChrome(chrome) {
  if (chrome.exitCode !== null || chrome.signalCode !== null) {
    return;
  }
  chrome.kill("SIGTERM");
  const closed = await Promise.race([
    new Promise((resolve) => chrome.once("close", () => resolve(true))),
    wait(2_000).then(() => false),
  ]);
  if (!closed && chrome.exitCode === null && chrome.signalCode === null) {
    chrome.kill("SIGKILL");
  }
}

async function runBrowserCase({ port, options, caseName, bitrate, repeat }) {
  const shouldProfile = Boolean(options.profileRustDecode && caseName.startsWith("rust"));
  const debugPort = 9_200 + Math.floor(Math.random() * 700);
  const userDataDir = resolve(tmpdir(), `libopus-browser-bench-${process.pid}-${Date.now()}-${repeat}`);
  const chrome = spawn(
    options.chrome,
    [
      "--headless=new",
      "--disable-gpu",
      "--disable-background-timer-throttling",
      "--disable-backgrounding-occluded-windows",
      "--disable-renderer-backgrounding",
      "--no-first-run",
      "--no-default-browser-check",
      `--user-data-dir=${userDataDir}`,
      `--remote-debugging-port=${debugPort}`,
      "about:blank",
    ],
    { stdio: ["ignore", "ignore", "pipe"] },
  );

  let chromeStderr = "";
  chrome.stderr.on("data", (chunk) => {
    chromeStderr += chunk.toString();
  });

  try {
    const wsUrl = await waitForDebuggerUrl(debugPort, () => chromeStderr);
    const devtools = await connectDevtools(wsUrl);
    let decodeProfile = null;
    let profileStop = Promise.resolve();
    const done = new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        reject(new Error(`${caseName} ${bitrate} timed out after ${options.timeoutMs} ms`));
      }, options.timeoutMs);

      devtools.socket.addEventListener("message", (event) => {
        const message = JSON.parse(event.data);
        if (message.id && devtools.pending.has(message.id)) {
          const pending = devtools.pending.get(message.id);
          devtools.pending.delete(message.id);
          if (message.error) {
            pending.reject(new Error(JSON.stringify(message.error)));
          } else {
            pending.resolve(message.result);
          }
          return;
        }

        if (message.method === "Runtime.consoleAPICalled") {
          const text = message.params.args
            ?.map((arg) => arg.value ?? arg.description ?? "")
            .join(" ") ?? "";
          if (text.startsWith("__BENCH_DONE__")) {
            clearTimeout(timer);
            resolve(JSON.parse(text.slice("__BENCH_DONE__".length)));
          } else if (text.startsWith("__BENCH_ERROR__")) {
            clearTimeout(timer);
            reject(new Error(text.slice("__BENCH_ERROR__".length)));
          } else if (text.startsWith("__BENCH_PROFILE_START__") && shouldProfile) {
            profileStop = devtools.send("Profiler.start");
          } else if (text.startsWith("__BENCH_PROFILE_END__") && shouldProfile) {
            profileStop = profileStop
              .then(() => devtools.send("Profiler.stop"))
              .then((result) => {
                decodeProfile = result.profile;
              });
          }
        } else if (message.method === "Runtime.exceptionThrown") {
          clearTimeout(timer);
          reject(new Error(message.params.exceptionDetails?.text || "browser exception"));
        }
      });
    });

    await devtools.send("Runtime.enable");
    await devtools.send("Page.enable");
    if (shouldProfile) {
      await devtools.send("Profiler.enable");
    }
    const url =
      `http://127.0.0.1:${port}/bench.html` +
      `?case=${encodeURIComponent(caseName)}` +
      `&bitrate=${bitrate}` +
      `&seconds=${options.seconds}` +
      `&v=${Date.now()}-${repeat}` +
      (shouldProfile ? "&profile=1" : "");
    await devtools.send("Page.navigate", { url });
    const result = await done;
    if (shouldProfile) {
      await profileStop;
      if (decodeProfile) {
        writeFileSync(
          options.profileRustDecode,
          `${JSON.stringify({ case: caseName, bitrate, profile: decodeProfile }, null, 2)}\n`,
        );
      }
    }
    devtools.socket.close();
    return result.row;
  } finally {
    await stopChrome(chrome);
    rmSync(userDataDir, { recursive: true, force: true });
  }
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

function aggregate(samples) {
  const row = { ...samples[0] };
  row.samples = samples;
  row.repeats = samples.length;
  for (const key of [
    "loadMs",
    "encodeMs",
    "decodeMs",
    "encodeXrtf",
    "decodeXrtf",
    "effectiveKbps",
    "packetAvg",
  ]) {
    row[key] = median(samples.map((sample) => sample[key]));
  }
  return row;
}

function formatNumber(value, digits = 1) {
  return Number.isFinite(value) ? value.toFixed(digits) : "n/a";
}

function printMarkdown(rows, options, paths) {
  console.log("");
  console.log(
    `Browser-loaded wasm comparison: ${options.seconds}s synthetic 48 kHz stereo fixture, ` +
      `${options.repeats} fresh Chrome run${options.repeats === 1 ? "" : "s"} per case.`,
  );
  console.log(`Chrome: ${options.chrome}`);
  console.log(`Rust wasm: ${paths.rustWasm} (${statSync(paths.rustWasm).size} bytes)`);
  console.log(`C wasm: ${paths.cWasm} (${statSync(paths.cWasm).size} bytes)`);
  console.log("");
  console.log("| Target | Case | Load ms | Encoded bytes | Effective kb/s | Packet bytes | Encode xRTF | Decode xRTF |");
  console.log("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |");
  for (const row of rows) {
    console.log(
      `| ${row.targetKbps} kb/s | ${row.name} | ${formatNumber(row.loadMs, 1)} | ` +
        `${row.encodedBytes} | ${formatNumber(row.effectiveKbps, 2)} | ` +
        `${row.packetMin}-${row.packetMax} (avg ${formatNumber(row.packetAvg, 2)}) | ` +
        `${formatNumber(row.encodeXrtf, 1)}x | ${formatNumber(row.decodeXrtf, 1)}x |`,
    );
  }

  const byBitrate = new Map();
  for (const row of rows) {
    if (!byBitrate.has(row.bitrate)) {
      byBitrate.set(row.bitrate, new Map());
    }
    byBitrate.get(row.bitrate).set(row.case, row);
  }

  console.log("");
  console.log("| Target | Rust case | Encode vs C | Decode vs C |");
  console.log("| --- | --- | ---: | ---: |");
  for (const [bitrate, cases] of byBitrate) {
    const c = cases.get("c");
    if (!c) {
      continue;
    }
    for (const rustCase of ["rust-cbr", "rust-vbr", "rust-cbr-reuse", "rust-vbr-reuse"]) {
      const rust = cases.get(rustCase);
      if (!rust) {
        continue;
      }
      console.log(
        `| ${bitrate / 1000} kb/s | ${rust.name} | ` +
          `${formatSignedPercent((rust.encodeXrtf / c.encodeXrtf - 1) * 100)} | ` +
          `${formatSignedPercent((rust.decodeXrtf / c.decodeXrtf - 1) * 100)} |`,
      );
    }
  }
}

function formatSignedPercent(value) {
  if (!Number.isFinite(value)) {
    return "n/a";
  }
  return `${value >= 0 ? "+" : ""}${value.toFixed(1)}%`;
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const paths = {
    rustJs: resolve(options.rustPkg, "libopus_rs.js"),
    rustWasm: resolve(options.rustPkg, "libopus_rs_bg.wasm"),
    cJs: resolve(options.libopusjs, "libopus.mjs"),
    cWasm: resolve(options.libopusjs, "libopus.wasm"),
  };

  requireFile(options.chrome, "Chrome");
  requireFile(paths.rustJs, "libopus-rs JS glue");
  requireFile(paths.rustWasm, "libopus-rs wasm");
  requireFile(paths.cJs, "libopusjs JS glue");
  requireFile(paths.cWasm, "libopusjs wasm");

  const server = createAssetServer(paths);
  const port = await listen(server);
  console.error(`serving wasm benchmark assets at http://127.0.0.1:${port}/`);

  const rows = [];
  try {
    for (const bitrate of options.bitrates) {
      for (const caseName of options.cases) {
        const samples = [];
        for (let repeat = 0; repeat < options.repeats; repeat += 1) {
          process.stderr.write(
            `running ${caseName} ${bitrate} bps (${repeat + 1}/${options.repeats})... `,
          );
          const row = await runBrowserCase({ port, options, caseName, bitrate, repeat });
          samples.push(row);
          process.stderr.write(`enc ${formatNumber(row.encodeXrtf, 1)}x dec ${formatNumber(row.decodeXrtf, 1)}x\n`);
        }
        rows.push(aggregate(samples));
      }
    }

    const result = {
      generatedAt: new Date().toISOString(),
      fixture: {
        sampleRate: 48_000,
        channels: 2,
        frameSize: 960,
        seconds: options.seconds,
      },
      options: {
        bitrates: options.bitrates,
        cases: options.cases,
        repeats: options.repeats,
        chrome: options.chrome,
      },
      artifacts: {
        rustJs: paths.rustJs,
        rustWasm: paths.rustWasm,
        rustWasmBytes: statSync(paths.rustWasm).size,
        cJs: paths.cJs,
        cWasm: paths.cWasm,
        cWasmBytes: statSync(paths.cWasm).size,
      },
      rows,
    };

    printMarkdown(rows, options, paths);
    if (options.json) {
      writeFileSync(options.json, `${JSON.stringify(result, null, 2)}\n`);
      console.error(`wrote ${options.json}`);
    }
  } finally {
    if (options.keepOpen) {
      console.error(`server left running at http://127.0.0.1:${port}/`);
    } else {
      server.close();
    }
  }
}

main().catch((error) => {
  console.error(error.stack || error.message || String(error));
  process.exit(1);
});
