#!/usr/bin/env node

import { readFile } from "node:fs/promises";
import { basename, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const WARMUP_CALLS = 1_024;
const MAX_PACKET_BYTES = 16 * 1_024 * 1_024;

function usage() {
  return "usage: node benchmark_flac_packet_wasm.mjs WASM_PACKAGE_DIR RATE realtime|balanced ITERATIONS PCM_S32LE PACKET_BUNDLE [buffered|view|copy]";
}

function parsePositiveInteger(value, name) {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed <= 0) {
    throw new Error(`${name} must be a positive integer`);
  }
  return parsed;
}

function readPackets(bytes) {
  const packets = [];
  let offset = 0;
  while (offset < bytes.byteLength) {
    if (offset + 4 > bytes.byteLength) {
      throw new Error("truncated packet bundle length");
    }
    const length = bytes.readUInt32LE(offset);
    offset += 4;
    if (length === 0 || length > MAX_PACKET_BYTES) {
      throw new Error(`invalid packet bundle frame length ${length}`);
    }
    if (offset + length > bytes.byteLength) {
      throw new Error("truncated packet bundle frame");
    }
    packets.push(
      new Uint8Array(bytes.buffer, bytes.byteOffset + offset, length),
    );
    offset += length;
  }
  if (packets.length === 0) {
    throw new Error("packet bundle contains no frames");
  }
  return packets;
}

function percentile(sortedNanos, percent) {
  const rank = Math.ceil((sortedNanos.length * percent) / 100);
  return sortedNanos[Math.max(0, rank - 1)] / 1_000;
}

function report(label, nanos, bytes, pcmBytes, checksum) {
  nanos.sort();
  console.log(
    `${label} p50_us=${percentile(nanos, 50).toFixed(3)}` +
      ` p95_us=${percentile(nanos, 95).toFixed(3)}` +
      ` p99_us=${percentile(nanos, 99).toFixed(3)}` +
      ` min_us=${(nanos[0] / 1_000).toFixed(3)}` +
      ` encoded/pcm=${(bytes / pcmBytes).toFixed(4)}` +
      ` calls=${nanos.length} checksum=${checksum >>> 0}`,
  );
}

function assertBytesEqual(actual, expected, frameIndex) {
  if (actual.byteLength !== expected.byteLength) {
    throw new Error(
      `encoded packet length mismatch in corpus frame ${frameIndex}: ` +
        `${actual.byteLength} != ${expected.byteLength}`,
    );
  }
  for (let index = 0; index < actual.byteLength; index += 1) {
    if (actual[index] !== expected[index]) {
      throw new Error(
        `encoded packet mismatch in corpus frame ${frameIndex}, byte ${index}`,
      );
    }
  }
}

function assertSamplesEqual(actual, expected, frameIndex) {
  if (actual.length !== expected.length) {
    throw new Error(
      `decoded sample count mismatch in corpus frame ${frameIndex}: ` +
        `${actual.length} != ${expected.length}`,
    );
  }
  for (let index = 0; index < actual.length; index += 1) {
    if (actual[index] !== expected[index]) {
      throw new Error(
        `decoded PCM mismatch in corpus frame ${frameIndex}, sample ${index}`,
      );
    }
  }
}

function collectTimerBaseline(iterations) {
  const nanos = new Float64Array(iterations);
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    const started = process.hrtime.bigint();
    const finished = process.hrtime.bigint();
    nanos[iteration] = Number(finished - started);
  }
  nanos.sort();
  return percentile(nanos, 50);
}

const arguments_ = process.argv.slice(2);
if (arguments_.length < 6 || arguments_.length > 7) {
  throw new Error(usage());
}

const [packageDirectory, rateArgument, profile, iterationArgument, pcmPath, bundlePath] =
  arguments_;
const transferMode = arguments_[6] ?? "view";
const sampleRate = parsePositiveInteger(rateArgument, "RATE");
const iterations = parsePositiveInteger(iterationArgument, "ITERATIONS");
if (sampleRate !== 48_000 && sampleRate !== 96_000) {
  throw new Error("RATE must be 48000 or 96000");
}
if (profile !== "realtime" && profile !== "balanced") {
  throw new Error("profile must be realtime or balanced");
}
if (
  transferMode !== "buffered" &&
  transferMode !== "view" &&
  transferMode !== "copy"
) {
  throw new Error("transfer mode must be buffered, view, or copy");
}
if (new Uint8Array(new Uint32Array([0x01020304]).buffer)[0] !== 4) {
  throw new Error("the S32LE benchmark requires a little-endian Node host");
}

const channels = 2;
const bitsPerSample = 24;
const frameSize = sampleRate / 200;
const samplesPerFrame = frameSize * channels;
const compressionLevel = profile === "realtime" ? 0 : 2;

const [pcmBytes, bundleBytes] = await Promise.all([
  readFile(pcmPath),
  readFile(bundlePath),
]);
if (pcmBytes.byteLength === 0 || pcmBytes.byteLength % 4 !== 0) {
  throw new Error(`${pcmPath} must contain non-empty S32LE PCM`);
}
const pcmBuffer = pcmBytes.buffer.slice(
  pcmBytes.byteOffset,
  pcmBytes.byteOffset + pcmBytes.byteLength,
);
const pcm = new Int32Array(pcmBuffer);
if (pcm.length % samplesPerFrame !== 0) {
  throw new Error(
    `${pcmPath} has ${pcm.length} samples; expected a multiple of ${samplesPerFrame}`,
  );
}
const packets = readPackets(bundleBytes);
const corpusFrames = pcm.length / samplesPerFrame;
if (packets.length !== corpusFrames) {
  throw new Error(
    `packet bundle has ${packets.length} frames but PCM has ${corpusFrames}`,
  );
}

const packagePath = resolve(packageDirectory);
const modulePath = join(packagePath, "soundkit_wasm.js");
const wasmPath = join(packagePath, "soundkit_wasm_bg.wasm");
const soundkit = await import(pathToFileURL(modulePath).href);
await soundkit.default({ module_or_path: await readFile(wasmPath) });

let verifierEncoder;
let verifierDecoder;
try {
  verifierEncoder = new soundkit.WasmFlacFrameEncoder(
    sampleRate,
    channels,
    bitsPerSample,
    frameSize,
    compressionLevel,
  );
  verifierDecoder = new soundkit.WasmFlacFrameDecoder(
    sampleRate,
    channels,
    bitsPerSample,
    frameSize,
  );
  const encoderInput =
    transferMode === "buffered" ? verifierEncoder.inputPcmView() : undefined;
  const decoderInput =
    transferMode === "buffered" ? verifierDecoder.inputPacketView() : undefined;
  const decoderOutput =
    transferMode === "buffered" ? verifierDecoder.decodedPcmView() : undefined;
  for (let frameIndex = 0; frameIndex < corpusFrames; frameIndex += 1) {
    const pcmOffset = frameIndex * samplesPerFrame;
    const frame = pcm.subarray(pcmOffset, pcmOffset + samplesPerFrame);
    let encoded;
    if (transferMode === "buffered") {
      encoderInput.set(frame);
      encoded = verifierEncoder.encodeBufferedView();
    } else if (transferMode === "view") {
      encoded = verifierEncoder.encodeInterleavedI32View(frame);
    } else {
      encoded = verifierEncoder.encodeInterleavedI32(frame);
    }
    assertBytesEqual(
      encoded,
      packets[frameIndex],
      frameIndex,
    );
    let decoded;
    if (transferMode === "buffered") {
      decoderInput.set(packets[frameIndex]);
      const written = verifierDecoder.decodeBuffered(packets[frameIndex].length);
      decoded = decoderOutput.subarray(0, written);
    } else if (transferMode === "view") {
      decoded = verifierDecoder.decodeInterleavedI32View(packets[frameIndex]);
    } else {
      decoded = verifierDecoder.decodeInterleavedI32(packets[frameIndex]);
    }
    assertSamplesEqual(decoded, frame, frameIndex);
  }
} finally {
  verifierEncoder?.free();
  verifierDecoder?.free();
}

globalThis.gc?.();
let encoder;
const encodeNanos = new Float64Array(iterations);
let encodedBytes = 0;
let encodeChecksum = 0;
try {
  encoder = new soundkit.WasmFlacFrameEncoder(
    sampleRate,
    channels,
    bitsPerSample,
    frameSize,
    compressionLevel,
  );
  let encoderInput =
    transferMode === "buffered" ? encoder.inputPcmView() : undefined;
  for (let iteration = 0; iteration < WARMUP_CALLS; iteration += 1) {
    const frameIndex = iteration % corpusFrames;
    const pcmOffset = frameIndex * samplesPerFrame;
    const frame = pcm.subarray(pcmOffset, pcmOffset + samplesPerFrame);
    if (transferMode === "buffered") {
      encoderInput.set(frame);
      encoder.encodeBufferedView();
    } else if (transferMode === "view") {
      encoder.encodeInterleavedI32View(frame);
    } else {
      encoder.encodeInterleavedI32(frame);
    }
  }
  encoder.reset();
  if (transferMode === "buffered") {
    encoderInput = encoder.inputPcmView();
  }
  globalThis.gc?.();
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    const frameIndex = iteration % corpusFrames;
    const pcmOffset = frameIndex * samplesPerFrame;
    const frame = pcm.subarray(pcmOffset, pcmOffset + samplesPerFrame);
    const started = process.hrtime.bigint();
    let packet;
    if (transferMode === "buffered") {
      encoderInput.set(frame);
      packet = encoder.encodeBufferedView();
    } else if (transferMode === "view") {
      packet = encoder.encodeInterleavedI32View(frame);
    } else {
      packet = encoder.encodeInterleavedI32(frame);
    }
    const finished = process.hrtime.bigint();
    encodeNanos[iteration] = Number(finished - started);
    encodedBytes += packet.byteLength;
    encodeChecksum =
      (encodeChecksum + packet.byteLength + packet[0] + packet.at(-1)) >>> 0;
  }
} finally {
  encoder?.free();
}

globalThis.gc?.();
let decoder;
const decodeNanos = new Float64Array(iterations);
let decodedPacketBytes = 0;
let decodeChecksum = 0;
try {
  decoder = new soundkit.WasmFlacFrameDecoder(
    sampleRate,
    channels,
    bitsPerSample,
    frameSize,
  );
  let decoderInput =
    transferMode === "buffered" ? decoder.inputPacketView() : undefined;
  let decoderOutput =
    transferMode === "buffered" ? decoder.decodedPcmView() : undefined;
  for (let iteration = 0; iteration < WARMUP_CALLS; iteration += 1) {
    const packet = packets[iteration % corpusFrames];
    if (transferMode === "buffered") {
      decoderInput.set(packet);
      decoder.decodeBuffered(packet.length);
    } else if (transferMode === "view") {
      decoder.decodeInterleavedI32View(packet);
    } else {
      decoder.decodeInterleavedI32(packet);
    }
  }
  decoder.reset();
  if (transferMode === "buffered") {
    decoderInput = decoder.inputPacketView();
    decoderOutput = decoder.decodedPcmView();
  }
  globalThis.gc?.();
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    const packet = packets[iteration % corpusFrames];
    const started = process.hrtime.bigint();
    let decoded;
    let decodedLength;
    if (transferMode === "buffered") {
      decoderInput.set(packet);
      decodedLength = decoder.decodeBuffered(packet.length);
      decoded = decoderOutput;
    } else if (transferMode === "view") {
      decoded = decoder.decodeInterleavedI32View(packet);
      decodedLength = decoded.length;
    } else {
      decoded = decoder.decodeInterleavedI32(packet);
      decodedLength = decoded.length;
    }
    const finished = process.hrtime.bigint();
    decodeNanos[iteration] = Number(finished - started);
    decodedPacketBytes += packet.byteLength;
    decodeChecksum =
      (decodeChecksum +
        decodedLength +
        decoded[0] +
        decoded[decodedLength - 1]) >>>
      0;
  }
} finally {
  decoder?.free();
}

const timerP50Us = collectTimerBaseline(iterations);
console.log(
  `wasm corpus rate=${sampleRate} frame=${frameSize} channels=${channels}` +
    ` bits=${bitsPerSample} profile=${profile} corpus_frames=${corpusFrames}` +
    ` corpus_ms=${(corpusFrames * 5).toFixed(1)} node=${process.version}` +
    ` module=${basename(wasmPath)} transfer=${transferMode}` +
    ` timer_p50_us=${timerP50Us.toFixed(3)}` +
    " native_packet_match=true decoded_pcm_match=true",
);
report(
  `wasm encode-${transferMode}`,
  encodeNanos,
  encodedBytes,
  iterations * samplesPerFrame * (bitsPerSample / 8),
  encodeChecksum,
);
report(
  `wasm decode-${transferMode}`,
  decodeNanos,
  decodedPacketBytes,
  iterations * samplesPerFrame * (bitsPerSample / 8),
  decodeChecksum,
);
