// Runs one module's encode or decode loop under the V8 inspector profiler and
// prints self time per wasm function.
//
// node tools/wasm_ab/profile.mjs <module.wasm> <source.s32le> <encode|decode> [rate] [vbr] [bits]
import { readFileSync } from "node:fs";
import { Session } from "node:inspector/promises";

const [path, source, what, rateArg, vbrArg, bitsArg] = process.argv.slice(2);
const rate = Number(rateArg || 256000);
const vbr = Number(vbrArg || 0);
const bits = Number(bitsArg || 24);
const frameSize = 240;
const seconds = 20;

const module = await WebAssembly.compile(readFileSync(path));
const imports = {};
for (const imp of WebAssembly.Module.imports(module)) {
  imports[imp.module] ??= {};
  if (imp.kind === "function") imports[imp.module][imp.name] = () => 0;
}
const { exports: e } = await WebAssembly.instantiate(module, imports);
e._initialize?.();

const raw = readFileSync(source);
const all = new Int32Array(raw.buffer, raw.byteOffset, raw.byteLength >> 2);
const samples = Math.floor((seconds * 48000) / frameSize) * frameSize;
const pcm = all.slice(30 * 48000 * 2, 30 * 48000 * 2 + samples * 2);
const cap = (samples / frameSize) * 1277;
const pcmPtr = e.ab_alloc(pcm.byteLength) >>> 0;
new Int32Array(e.memory.buffer, pcmPtr, pcm.length).set(pcm);
const pkt = e.ab_alloc(cap) >>> 0;
const out = e.ab_alloc((samples + frameSize) * 8) >>> 0;
const len = e.ab_encode(pcmPtr, samples, frameSize, rate, vbr, bits, pkt, cap);

const session = new Session();
session.connect();
await session.post("Profiler.enable");
await session.post("Profiler.setSamplingInterval", { interval: 100 });
await session.post("Profiler.start");
for (let i = 0; i < 4; i += 1) {
  if (what === "encode") e.ab_encode(pcmPtr, samples, frameSize, rate, vbr, bits, pkt, cap);
  else e.ab_decode(pkt, len, frameSize, bits, out, samples + frameSize);
}
const { profile } = await session.post("Profiler.stop");

const self = new Map();
const dt = new Map();
for (let i = 0; i < profile.samples.length; i += 1) {
  dt.set(profile.samples[i], (dt.get(profile.samples[i]) || 0) + (profile.timeDeltas[i] || 0));
}
let total = 0;
for (const node of profile.nodes) {
  const t = dt.get(node.id) || 0;
  const name = node.callFrame.functionName || node.callFrame.url || "(anon)";
  self.set(name, (self.get(name) || 0) + t);
  total += t;
}
const rows = [...self.entries()].sort((a, b) => b[1] - a[1]).slice(0, 40);
for (const [name, t] of rows) {
  console.log(`${((t / total) * 100).toFixed(2).padStart(6)}%  ${(t / 1000).toFixed(1).padStart(8)} ms  ${name.slice(0, 110)}`);
}
