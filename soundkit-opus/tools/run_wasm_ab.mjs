#!/usr/bin/env node
// Times the libopus and soundkit-opus wasm modules from tools/build_wasm_ab.sh
// on identical PCM and identical packets, and reports delay-aligned SNR.

import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { basename, join } from "node:path";
import { performance } from "node:perf_hooks";

const args = process.argv.slice(2);
function opt(name, fallback) {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : fallback;
}
const cWasm = opt("c");
const rustWasm = opt("rust");
const sources = opt("sources", "").split(",").filter(Boolean);
const seconds = Number(opt("seconds", 20));
const offsetSeconds = Number(opt("offset", 30));
const rounds = Number(opt("rounds", 5));
const frameSize = Number(opt("frame-size", 240));
const rates = opt("rates", "192000,256000,320000").split(",").map(Number);
const modes = opt("modes", "cbr,vbr").split(",");
const bits = opt("bits", "24").split(",").map(Number);
const jsonPath = opt("json", null);
const dumpDir = opt("dump", null);
const wavDir = opt("wav-dir", null);
const slotSeconds = Number(opt("slot-seconds", 0));
const skipSpeed = args.includes("--quality-only");
// --tune-a/--tune-b "knob=value,knob=value" call ab_tune on module A or B.
const tunes = { c: opt("tune-a", ""), rust: opt("tune-b", "") };

if (!cWasm || !rustWasm || sources.length === 0) {
  console.error(
    "usage: run_wasm_ab.mjs --c c.wasm --rust rust.wasm --sources a.s32le,b.s32le " +
      "[--seconds 20] [--offset 30] [--rounds 5] [--frame-size 240] " +
      "[--rates 192000,256000,320000] [--modes cbr,vbr] [--bits 24,16] " +
      "[--json out.json] [--dump dir] [--quality-only]",
  );
  process.exit(2);
}

async function load(path, label) {
  const module = await WebAssembly.compile(readFileSync(path));
  const imports = {};
  for (const imp of WebAssembly.Module.imports(module)) {
    imports[imp.module] ??= {};
    if (imp.kind === "function") {
      imports[imp.module][imp.name] = (...a) => {
        if (imp.name === "proc_exit") throw new Error(`${label} exit ${a[0]}`);
        if (imp.name === "emscripten_notify_memory_growth") return;
        return 0;
      };
    }
  }
  const instance = await WebAssembly.instantiate(module, imports);
  const e = instance.exports;
  e._initialize?.();
  return { label, e, mem: () => e.memory.buffer };
}

function copyIn(codec, bytes) {
  const ptr = codec.e.ab_alloc(bytes.byteLength) >>> 0;
  if (ptr === 0) throw new Error(`${codec.label} alloc failed`);
  new Uint8Array(codec.mem(), ptr, bytes.byteLength).set(
    new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength),
  );
  return ptr;
}

function encode(codec, pcmPtr, samples, rate, vbr, pcmBits, outPtr, cap) {
  const t0 = performance.now();
  const n = codec.e.ab_encode(pcmPtr, samples, frameSize, rate, vbr, pcmBits, outPtr, cap);
  const ms = performance.now() - t0;
  if (n < 0) throw new Error(`${codec.label} encode error ${n}`);
  return { n, ms };
}

function decode(codec, pktPtr, len, pcmBits, outPtr, outSamples) {
  const t0 = performance.now();
  const n = codec.e.ab_decode(pktPtr, len, frameSize, pcmBits, outPtr, outSamples);
  const ms = performance.now() - t0;
  if (n < 0) throw new Error(`${codec.label} decode error ${n}`);
  return { n, ms };
}

function fnv1a(int32s) {
  let h = 0x811c9dc5;
  for (let i = 0; i < int32s.length; i += 1) {
    h = Math.imul(h ^ int32s[i], 0x01000193);
  }
  return h >>> 0;
}

// Writes interleaved stereo s24-in-i32 samples as a 32-bit float WAV.
function writeWav(path, int32s, start, frames) {
  const data = Buffer.alloc(frames * 8);
  for (let i = 0; i < frames * 2; i += 1) {
    data.writeFloatLE(int32s[start * 2 + i] / 8388608, i * 4);
  }
  const h = Buffer.alloc(44);
  h.write("RIFF", 0);
  h.writeUInt32LE(36 + data.length, 4);
  h.write("WAVEfmt ", 8);
  h.writeUInt32LE(16, 16);
  h.writeUInt16LE(3, 20);
  h.writeUInt16LE(2, 22);
  h.writeUInt32LE(48000, 24);
  h.writeUInt32LE(48000 * 8, 28);
  h.writeUInt16LE(8, 32);
  h.writeUInt16LE(32, 34);
  h.write("data", 36);
  h.writeUInt32LE(data.length, 40);
  writeFileSync(path, Buffer.concat([h, data]));
}

function median(values) {
  const s = [...values].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
}

// Finds the decoder delay from one second of signal, then computes SNR over
// the whole aligned length, full-band and on the side channel.
function alignedSnr(ref, dec, samples) {
  const probeStart = 48000;
  const probeLen = Math.min(48000, samples - probeStart - 2000);
  let best = 0;
  let bestScore = -Infinity;
  for (let lag = 0; lag < 1500; lag += 1) {
    let s = 0;
    for (let i = 0; i < probeLen; i += 1) {
      const j = (probeStart + i) * 2;
      s += ref[j] * dec[j + lag * 2] + ref[j + 1] * dec[j + 1 + lag * 2];
    }
    if (s > bestScore) {
      bestScore = s;
      best = lag;
    }
  }
  let sig = 0;
  let err = 0;
  let sideSig = 0;
  let sideErr = 0;
  const n = samples - best - 960;
  for (let i = 960; i < n; i += 1) {
    const a = i * 2;
    const b = (i + best) * 2;
    const l = ref[a];
    const r = ref[a + 1];
    const dl = dec[b] - l;
    const dr = dec[b + 1] - r;
    sig += l * l + r * r;
    err += dl * dl + dr * dr;
    const side = l - r;
    const dside = dec[b] - dec[b + 1] - side;
    sideSig += side * side;
    sideErr += dside * dside;
  }
  return {
    delay: best,
    snr: 10 * Math.log10(sig / Math.max(err, 1e-9)),
    sideSnr: sideSig > 0 ? 10 * Math.log10(sideSig / Math.max(sideErr, 1e-9)) : null,
  };
}

const c = await load(cWasm, "c");
const rust = await load(rustWasm, "rust");
for (const codec of [c, rust]) {
  for (const pair of tunes[codec.label].split(",").filter(Boolean)) {
    const [knob, value] = pair.split("=").map(Number);
    if (!codec.e.ab_tune) throw new Error(`${codec.label} has no ab_tune export`);
    codec.e.ab_tune(knob, value);
  }
}
const results = [];

for (const source of sources) {
  const raw = readFileSync(source);
  const all = new Int32Array(raw.buffer, raw.byteOffset, raw.byteLength >> 2);
  const start = Math.floor(offsetSeconds * 48000) * 2;
  const frames = Math.floor((seconds * 48000) / frameSize);
  const samples = frames * frameSize;
  const pcm = all.slice(start, start + samples * 2);
  const cap = frames * (1275 + 2);
  const outSamples = samples + frameSize;

  const ctx = {};
  for (const codec of [c, rust]) {
    ctx[codec.label] = {
      pcm: copyIn(codec, pcm),
      pkt: codec.e.ab_alloc(cap) >>> 0,
      foreign: codec.e.ab_alloc(cap) >>> 0,
      out: codec.e.ab_alloc(outSamples * 2 * 4) >>> 0,
    };
  }

  for (const pcmBits of bits) {
    for (const mode of modes) {
      for (const rate of rates) {
        const vbr = mode === "vbr" ? 1 : mode === "uvbr" ? 2 : 0;
        const enc = { c: [], rust: [] };
        const bytes = {};
        const order = skipSpeed ? 1 : rounds;
        for (let r = 0; r < order; r += 1) {
          const pair = r % 2 === 0 ? [c, rust] : [rust, c];
          for (const codec of pair) {
            const x = ctx[codec.label];
            const { n, ms } = encode(codec, x.pcm, samples, rate, vbr, pcmBits, x.pkt, cap);
            enc[codec.label].push(ms);
            bytes[codec.label] = n;
          }
        }
        // Same-packet decode: each decoder sees each encoder's packets.
        const packets = {
          c: new Uint8Array(c.mem(), ctx.c.pkt, bytes.c).slice(),
          rust: new Uint8Array(rust.mem(), ctx.rust.pkt, bytes.rust).slice(),
        };
        for (const codec of [c, rust]) {
          const other = codec.label === "c" ? "rust" : "c";
          new Uint8Array(codec.mem(), ctx[codec.label].foreign, cap).set(packets[other]);
        }
        const dec = { c: { c: [], rust: [] }, rust: { c: [], rust: [] } };
        const quality = {};
        for (let r = 0; r < order; r += 1) {
          for (const producer of ["c", "rust"]) {
            const pair = r % 2 === 0 ? [c, rust] : [rust, c];
            for (const codec of pair) {
              const x = ctx[codec.label];
              const ptr = producer === codec.label ? x.pkt : x.foreign;
              const { ms } = decode(codec, ptr, bytes[producer], pcmBits, x.out, outSamples);
              dec[codec.label][producer].push(ms);
              // CROSS_DECODE=1 scores every packet set with module A's
              // decoder, so an encoder comparison uses one decoder.
              const scored = process.env.CROSS_DECODE ? codec.label === "c" : producer === codec.label;
              if (r === 0 && scored) {
                const decoded = new Int32Array(codec.mem(), x.out, samples * 2).slice();
                const label = process.env.CROSS_DECODE ? producer : codec.label;
                quality[label] = alignedSnr(pcm, decoded, samples);
                quality[label].hash = fnv1a(decoded);
                if (wavDir) {
                  mkdirSync(wavDir, { recursive: true });
                  const stem = basename(source).replace(/\..*$/, "");
                  const delay = quality[label].delay;
                  const slotFrames = slotSeconds > 0 ? slotSeconds * 48000 : samples - 1500;
                  const slots = Math.floor((samples - 1500) / slotFrames);
                  for (let k = 0; k < slots; k += 1) {
                    const name = slotSeconds > 0 ? `${stem}-s${String(k).padStart(2, "0")}` : stem;
                    writeWav(join(wavDir, `${name}.ref.wav`), pcm, k * slotFrames, slotFrames);
                    writeWav(
                      join(wavDir, `${name}_${pcmBits}_${mode}_${rate}_${label}.wav`),
                      decoded,
                      k * slotFrames + delay,
                      slotFrames,
                    );
                  }
                }
                if (dumpDir) {
                  mkdirSync(dumpDir, { recursive: true });
                  const name = `${basename(source).replace(/\..*$/, "")}_${pcmBits}_${mode}_${rate}_${codec.label}.s32le`;
                  writeFileSync(join(dumpDir, name), decoded);
                }
              }
            }
          }
        }
        const audioMs = (samples / 48000) * 1000;
        const row = {
          source: basename(source),
          pcmBits,
          mode,
          rate,
          frameSize,
          seconds: samples / 48000,
          kbps: {
            c: (bytes.c - frames * 2) * 8 / (samples / 48000) / 1000,
            rust: (bytes.rust - frames * 2) * 8 / (samples / 48000) / 1000,
          },
          encodeMs: { c: median(enc.c), rust: median(enc.rust) },
          decodeMs: {
            cOnC: median(dec.c.c),
            rustOnC: median(dec.rust.c),
            cOnRust: median(dec.c.rust),
            rustOnRust: median(dec.rust.rust),
          },
          quality,
          audioMs,
          identicalPackets:
            packets.c.length === packets.rust.length && packets.c.every((v, i) => v === packets.rust[i]),
        };
        results.push(row);
        const encDelta = (row.encodeMs.rust / row.encodeMs.c - 1) * 100;
        const decC = row.decodeMs.cOnC + row.decodeMs.cOnRust;
        const decR = row.decodeMs.rustOnC + row.decodeMs.rustOnRust;
        const decDelta = (decR / decC - 1) * 100;
        console.log(
          `${row.source.padEnd(40)} i${pcmBits} ${mode} ${rate / 1000}k | ` +
            `enc C ${(audioMs / row.encodeMs.c).toFixed(1)}x R ${(audioMs / row.encodeMs.rust).toFixed(1)}x (${encDelta >= 0 ? "+" : ""}${encDelta.toFixed(1)}%) | ` +
            `dec C ${((2 * audioMs) / decC).toFixed(1)}x R ${((2 * audioMs) / decR).toFixed(1)}x (${decDelta >= 0 ? "+" : ""}${decDelta.toFixed(1)}%) | ` +
            `kbps C ${row.kbps.c.toFixed(1)} R ${row.kbps.rust.toFixed(1)} | ` +
            `SNR C ${quality.c.snr.toFixed(2)} R ${quality.rust.snr.toFixed(2)} ` +
            `side C ${quality.c.sideSnr?.toFixed(2)} R ${quality.rust.sideSnr?.toFixed(2)} ` +
            `delay ${quality.c.delay}/${quality.rust.delay}${row.identicalPackets ? " | identical" : ""}${quality.c.hash === quality.rust.hash ? " pcm" : ""}`,
        );
      }
    }
  }
  for (const codec of [c, rust]) {
    const x = ctx[codec.label];
    codec.e.ab_free(x.pcm, pcm.byteLength);
    codec.e.ab_free(x.pkt, cap);
    codec.e.ab_free(x.foreign, cap);
    codec.e.ab_free(x.out, outSamples * 2 * 4);
  }
}

const sum = (f) => results.reduce((a, r) => a + f(r), 0);
const encC = sum((r) => r.encodeMs.c);
const encR = sum((r) => r.encodeMs.rust);
const decC = sum((r) => r.decodeMs.cOnC + r.decodeMs.cOnRust);
const decR = sum((r) => r.decodeMs.rustOnC + r.decodeMs.rustOnRust);
const snrC = sum((r) => r.quality.c.snr) / results.length;
const snrR = sum((r) => r.quality.rust.snr) / results.length;
console.log(
  `TOTAL enc Rust/C ${(encR / encC * 100 - 100).toFixed(2)}% | dec Rust/C ${(decR / decC * 100 - 100).toFixed(2)}% | ` +
    `mean SNR C ${snrC.toFixed(3)} R ${snrR.toFixed(3)} (R-C ${(snrR - snrC).toFixed(3)} dB)`,
);
if (jsonPath) writeFileSync(jsonPath, JSON.stringify(results, null, 2));
