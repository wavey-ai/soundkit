const U16_MAX = 0xffff;
const U32_MAX = 0xffff_ffff;
const U64_MAX = (1n << 64n) - 1n;

export const SOUNDKIT_INDEX_MAGIC = new Uint8Array([0x53, 0x4b, 0x49, 0x44, 0x58, 0x32, 0x00, 0x00]);
export const SOUNDKIT_INDEX_VERSION = 1;
export const SOUNDKIT_INDEX_ENTRY_BYTES = 16;
export const SOUNDKIT_INDEX_HEADER_BYTES = 32;

const asUint8Array = (bytes) => {
  if (bytes instanceof Uint8Array) return bytes;
  if (bytes instanceof ArrayBuffer) return new Uint8Array(bytes);
  if (ArrayBuffer.isView(bytes)) {
    return new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  }
  throw new TypeError("bytes must be a Uint8Array, ArrayBuffer, or typed array view");
};

const toU64 = (value, name) => {
  if (typeof value === "number" && !Number.isSafeInteger(value)) {
    throw new RangeError(`${name} number must be a safe integer`);
  }
  const bigintValue = typeof value === "bigint" ? value : BigInt(value);
  if (bigintValue < 0n || bigintValue > U64_MAX) {
    throw new RangeError(`${name} must fit in u64`);
  }
  return bigintValue;
};

const assertU16 = (value, name) => {
  if (!Number.isInteger(value) || value < 0 || value > U16_MAX) {
    throw new RangeError(`${name} must fit in u16`);
  }
};

const assertU32 = (value, name) => {
  if (!Number.isInteger(value) || value < 0 || value > U32_MAX) {
    throw new RangeError(`${name} must fit in u32`);
  }
};

const readU16Le = (bytes, offset) => bytes[offset] | (bytes[offset + 1] << 8);

const writeU16Le = (bytes, offset, value) => {
  bytes[offset] = value & 0xff;
  bytes[offset + 1] = (value >>> 8) & 0xff;
};

const readU32Le = (bytes, offset) => (
  (
    bytes[offset]
    | (bytes[offset + 1] << 8)
    | (bytes[offset + 2] << 16)
    | (bytes[offset + 3] * 0x1_000000)
  ) >>> 0
);

const writeU32Le = (bytes, offset, value) => {
  bytes[offset] = value & 0xff;
  bytes[offset + 1] = (value >>> 8) & 0xff;
  bytes[offset + 2] = (value >>> 16) & 0xff;
  bytes[offset + 3] = (value >>> 24) & 0xff;
};

const readU64Le = (bytes, offset) => {
  let value = 0n;
  for (let index = 7; index >= 0; index -= 1) {
    value = (value << 8n) | BigInt(bytes[offset + index]);
  }
  return value;
};

const writeU64Le = (bytes, offset, value) => {
  let remaining = value;
  for (let index = 0; index < 8; index += 1) {
    bytes[offset + index] = Number(remaining & 0xffn);
    remaining >>= 8n;
  }
};

const assertMagic = (bytes) => {
  for (let index = 0; index < SOUNDKIT_INDEX_MAGIC.length; index += 1) {
    if (bytes[index] !== SOUNDKIT_INDEX_MAGIC[index]) {
      throw new Error("invalid SoundKit sidecar index magic");
    }
  }
};

export const encodeSoundKitIndex = ({
  timescale,
  durationFrames,
  entries
}) => {
  assertU32(timescale, "timescale");
  const duration = toU64(durationFrames, "durationFrames");
  if (!Array.isArray(entries)) {
    throw new TypeError("entries must be an array");
  }

  const output = new Uint8Array(SOUNDKIT_INDEX_HEADER_BYTES + entries.length * SOUNDKIT_INDEX_ENTRY_BYTES);
  output.set(SOUNDKIT_INDEX_MAGIC, 0);
  writeU16Le(output, 8, SOUNDKIT_INDEX_VERSION);
  writeU16Le(output, 10, SOUNDKIT_INDEX_ENTRY_BYTES);
  writeU32Le(output, 12, timescale);
  writeU64Le(output, 16, BigInt(entries.length));
  writeU64Le(output, 24, duration);

  let lastStartFrame = -1n;
  let offset = SOUNDKIT_INDEX_HEADER_BYTES;
  for (const entry of entries) {
    const byteOffset = toU64(entry.byteOffset, "entry.byteOffset");
    const startFrame = toU64(entry.startFrame, "entry.startFrame");
    if (startFrame <= lastStartFrame) {
      throw new RangeError("index entries must be sorted by increasing startFrame");
    }
    writeU64Le(output, offset, byteOffset);
    writeU64Le(output, offset + 8, startFrame);
    offset += SOUNDKIT_INDEX_ENTRY_BYTES;
    lastStartFrame = startFrame;
  }

  return output;
};

export const decodeSoundKitIndex = (bytes) => {
  const input = asUint8Array(bytes);
  if (input.length < SOUNDKIT_INDEX_HEADER_BYTES) {
    throw new Error("SoundKit sidecar index is too small");
  }
  assertMagic(input);

  const version = readU16Le(input, 8);
  if (version !== SOUNDKIT_INDEX_VERSION) {
    throw new Error(`unsupported SoundKit sidecar index version ${version}`);
  }
  const entryBytes = readU16Le(input, 10);
  if (entryBytes !== SOUNDKIT_INDEX_ENTRY_BYTES) {
    throw new Error(`unsupported SoundKit sidecar index entry size ${entryBytes}`);
  }
  assertU16(entryBytes, "entryBytes");

  const timescale = readU32Le(input, 12);
  const entryCount = readU64Le(input, 16);
  const durationFrames = readU64Le(input, 24);
  if (entryCount > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new Error("SoundKit sidecar index entry count is too large for JavaScript");
  }
  const entryCountNumber = Number(entryCount);
  const expectedBytes = SOUNDKIT_INDEX_HEADER_BYTES + entryCountNumber * entryBytes;
  if (input.length !== expectedBytes) {
    throw new Error("SoundKit sidecar index length does not match entry count");
  }

  const entries = [];
  let offset = SOUNDKIT_INDEX_HEADER_BYTES;
  let lastStartFrame = -1n;
  for (let index = 0; index < entryCountNumber; index += 1) {
    const byteOffset = readU64Le(input, offset);
    const startFrame = readU64Le(input, offset + 8);
    if (startFrame <= lastStartFrame) {
      throw new Error("SoundKit sidecar index entries are not sorted by startFrame");
    }
    entries.push({ byteOffset, startFrame });
    offset += entryBytes;
    lastStartFrame = startFrame;
  }

  return {
    version,
    entryBytes,
    timescale,
    durationFrames,
    entries
  };
};

export const findSoundKitSeekEntry = (index, targetFrame) => {
  const target = toU64(targetFrame, "targetFrame");
  const entries = Array.isArray(index) ? index : index.entries;
  if (!Array.isArray(entries) || entries.length === 0) return undefined;

  let low = 0;
  let high = entries.length - 1;
  let found = 0;
  while (low <= high) {
    const mid = (low + high) >> 1;
    if (entries[mid].startFrame <= target) {
      found = mid;
      low = mid + 1;
    } else {
      high = mid - 1;
    }
  }
  return entries[found];
};

export const frameForTimeSeconds = (seconds, timescale) => {
  if (!Number.isFinite(seconds) || seconds < 0) {
    throw new RangeError("seconds must be a non-negative finite number");
  }
  assertU32(timescale, "timescale");
  return BigInt(Math.floor(seconds * timescale));
};

export const timeSecondsForFrame = (frame, timescale) => {
  assertU32(timescale, "timescale");
  const value = toU64(frame, "frame");
  return Number(value) / timescale;
};

export const soundKitRangeRequestForFrame = (index, targetFrame) => {
  const entry = findSoundKitSeekEntry(index, targetFrame);
  if (!entry) return undefined;
  return {
    byteOffset: entry.byteOffset,
    startFrame: entry.startFrame,
    headers: {
      Range: `bytes=${entry.byteOffset.toString()}-`
    }
  };
};
