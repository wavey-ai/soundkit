import {
  computeSoundKitPacketCrc32,
  decodeSoundKitFrameHeader,
  encodeSoundKitFrameHeader,
  verifySoundKitPacketCrc32
} from "./frame-header.js";
import { encodeSoundKitIndex } from "./sidecar-index.js";

const asUint8Array = (bytes) => {
  if (bytes instanceof Uint8Array) return bytes;
  if (bytes instanceof ArrayBuffer) return new Uint8Array(bytes);
  if (ArrayBuffer.isView(bytes)) {
    return new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  }
  throw new TypeError("bytes must be a Uint8Array, ArrayBuffer, or typed array view");
};

const concat = (chunks) => {
  const total = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const output = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    output.set(chunk, offset);
    offset += chunk.length;
  }
  return output;
};

export const readSoundKitFrame = (bytes, offset = 0, { verifyCrc32 = true } = {}) => {
  const input = asUint8Array(bytes);
  const header = decodeSoundKitFrameHeader(input, offset);
  const payloadOffset = offset + header.headerBytes;
  const nextOffset = payloadOffset + header.payloadSize;
  if (input.length < nextOffset) {
    throw new Error("incomplete SoundKit frame payload");
  }

  const encodedHeader = input.subarray(offset, payloadOffset);
  const payload = input.subarray(payloadOffset, nextOffset);
  if (verifyCrc32 && header.packetCrc32 !== undefined && !verifySoundKitPacketCrc32(header, encodedHeader, payload)) {
    throw new Error("SoundKit frame CRC32 mismatch");
  }

  return {
    byteOffset: BigInt(offset),
    byteLength: header.headerBytes + header.payloadSize,
    header,
    encodedHeader,
    payload,
    nextOffset
  };
};

export const scanSoundKitFrameStream = (bytes, options = {}) => {
  const input = asUint8Array(bytes);
  const frames = [];
  let offset = 0;
  while (offset < input.length) {
    const frame = readSoundKitFrame(input, offset, options);
    frames.push(frame);
    offset = frame.nextOffset;
  }
  return frames;
};

export const buildSoundKitFrame = (header, payload, { packetCrc32 = true } = {}) => {
  const payloadBytes = asUint8Array(payload);
  const headerInit = {
    ...header,
    payloadSize: payloadBytes.length
  };
  if (packetCrc32) {
    headerInit.packetCrc32 = computeSoundKitPacketCrc32(headerInit, payloadBytes);
  }
  const encodedHeader = encodeSoundKitFrameHeader(headerInit);
  const output = new Uint8Array(encodedHeader.length + payloadBytes.length);
  output.set(encodedHeader, 0);
  output.set(payloadBytes, encodedHeader.length);
  return output;
};

export const joinSoundKitFrames = (frames) => concat(frames.map((frame) => asUint8Array(frame)));

export const buildSoundKitIndexFromStream = (bytes, { timescale } = {}) => {
  const input = asUint8Array(bytes);
  const frames = scanSoundKitFrameStream(input);
  const resolvedTimescale = timescale ?? frames[0]?.header.sampleRate ?? 0;
  if (!resolvedTimescale) {
    throw new Error("timescale is required for an empty SoundKit stream");
  }

  const entries = [];
  let cursorFrame = 0n;
  for (const frame of frames) {
    const startFrame = frame.header.pts ?? cursorFrame;
    entries.push({
      byteOffset: frame.byteOffset,
      startFrame
    });
    cursorFrame = startFrame + BigInt(frame.header.frameCount);
  }

  return encodeSoundKitIndex({
    timescale: resolvedTimescale,
    durationFrames: cursorFrame,
    entries
  });
};

