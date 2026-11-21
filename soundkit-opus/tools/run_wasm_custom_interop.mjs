#!/usr/bin/env node
// Runs the Opus Custom codec compiled to WebAssembly
// (examples/wasm_custom_interop.rs) with the arguments of the native
// `opus_custom_interop` example, so the outputs can be compared byte for
// byte with the native build and with libopus:
//
//   node tools/run_wasm_custom_interop.mjs --wasm module.wasm \
//     encode --frame 48 --channels 2 --bitrate 256000 --pcm s16 in.s16 out.pkt
//   node tools/run_wasm_custom_interop.mjs --wasm module.wasm \
//     decode --frame 48 --channels 2 --pcm s16 in.pkt out.s16

import { readFileSync, writeFileSync } from "node:fs";

const argv = process.argv.slice(2);
const wasmIndex = argv.indexOf("--wasm");
if (wasmIndex < 0 || argv.length < 4) {
  console.error("usage: run_wasm_custom_interop.mjs --wasm M.wasm encode|decode [options] IN OUT");
  process.exit(2);
}
const wasmPath = argv[wasmIndex + 1];
const rest = argv.filter((_, i) => i !== wasmIndex && i !== wasmIndex + 1);
const command = rest[0];
const options = {
  frame: 0,
  channels: 2,
  bitrate: -1,
  vbr: 0,
  cvbr: 1,
  signalling: 1,
  "max-bytes": 1275,
  pcm: "s16",
  decisions: "libopus",
  arith: "libopus",
  pvq: "native",
};
const positional = [];
for (let i = 1; i < rest.length; i++) {
  if (rest[i].startsWith("--")) {
    const key = rest[i].slice(2);
    const value = rest[++i];
    if (key === "complexity") continue;
    if (!(key in options)) {
      console.error(`unknown option --${key}`);
      process.exit(2);
    }
    options[key] = key === "pcm" || key === "decisions" || key === "arith" || key === "pvq"
      ? value
      : key === "bitrate" && value === "max" ? -1 : Number(value);
  } else {
    positional.push(rest[i]);
  }
}
if (positional.length !== 2 || !options.frame) {
  console.error("need --frame, IN and OUT");
  process.exit(2);
}
const [input, output] = positional;
const format = { s16: 0, s24: 1, f32: 2 }[options.pcm];

const module = await WebAssembly.compile(readFileSync(wasmPath));
const { exports: e } = await WebAssembly.instantiate(module, {});

function copyIn(bytes) {
  const ptr = e.ci_alloc(bytes.length);
  new Uint8Array(e.memory.buffer, ptr, bytes.length).set(bytes);
  return ptr;
}

const data = readFileSync(input);
const inPtr = copyIn(data);
let cap;
let result;
let outPtr;
if (command === "encode") {
  cap = 1 + Math.ceil(data.length / (options.frame * options.channels)) * 1280 + data.length;
  outPtr = e.ci_alloc(cap);
  result = e.ci_encode(
    inPtr, data.length, format, options.frame, options.channels, options.bitrate,
    options.vbr, options.cvbr, options.signalling, options["max-bytes"],
    options.decisions === "libopus" ? 1 : 0, options.arith === "libopus" ? 1 : 0,
    options.pvq === "portable" ? 1 : 0, outPtr, cap,
  );
} else if (command === "decode") {
  const bytesPerSample = format === 0 ? 2 : 4;
  let frames = 0;
  for (let pos = 0; pos + 2 <= data.length; frames++) pos += 2 + (data[pos] | (data[pos + 1] << 8));
  cap = frames * options.frame * options.channels * bytesPerSample;
  outPtr = e.ci_alloc(cap);
  result = e.ci_decode(
    inPtr, data.length, format, options.frame, options.channels, options.signalling,
    options.arith === "libopus" ? 1 : 0, outPtr, cap,
  );
} else {
  console.error(`unknown command ${command}`);
  process.exit(2);
}
if (result < 0) {
  console.error(`${command} failed: ${result}`);
  process.exit(1);
}
writeFileSync(output, Buffer.from(new Uint8Array(e.memory.buffer, outPtr, result)));
