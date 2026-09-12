const MAGIC_WORD = 0x2b;
const VERSION = 2;
const MAGIC_SHIFT = 26;
const VERSION_SHIFT = 24;
const FLAGS_SHIFT = 16;
const ENCODING_SHIFT = 12;
const SAMPLE_RATE_SHIFT = 8;
const CHANNELS_SHIFT = 3;
const BITS_MASK = 0x7;
const U32_MAX = 0xffff_ffff;
const U64_MAX = (1n << 64n) - 1n;

const FLAG_ID_PRESENT = 1 << 0;
const FLAG_ID_U64 = 1 << 1;
const FLAG_PTS_PRESENT = 1 << 2;
const FLAG_PACKET_CRC32_PRESENT = 1 << 3;
const FLAG_BIG_ENDIAN = 1 << 4;
const FLAG_EXTENDED_SIZES = 1 << 5;
const FLAG_DISCONTINUITY = 1 << 6;
const FLAG_ENCRYPTED = 1 << 7;
const PUBLIC_PACKET_FLAGS = FLAG_DISCONTINUITY | FLAG_ENCRYPTED;
const SHORT_SIZE_SENTINEL = 0xffff;

export const SOUNDKIT_FRAME_HEADER_VERSION = VERSION;
export const SOUNDKIT_FRAME_HEADER_BASE_BYTES = 8;
export const SOUNDKIT_FRAME_HEADER_EXTENDED_SIZE_BYTES = 8;
export const SOUNDKIT_SHORT_SIZE_MAX = 0xfffe;

export const SOUNDKIT_SAMPLE_RATES = Object.freeze([
  8_000,
  12_000,
  16_000,
  24_000,
  32_000,
  44_100,
  48_000,
  88_200,
  96_000,
  176_400,
  192_000
]);

export const SOUNDKIT_BITS_PER_SAMPLE = Object.freeze([0, 8, 16, 24, 32, 64]);

export const SoundKitEncoding = Object.freeze({
  PcmSigned: 0,
  PcmFloat: 1,
  Opus: 2,
  Flac: 3,
  Aac: 4,
  H264: 5
});

export const SoundKitEncodingName = Object.freeze({
  0: "PCMSigned",
  1: "PCMFloat",
  2: "Opus",
  3: "FLAC",
  4: "AAC",
  5: "H264"
});

export const SoundKitEndianness = Object.freeze({
  LittleEndian: 0,
  BigEndian: 1
});

export const SoundKitPacketFlags = Object.freeze({
  None: 0,
  Discontinuity: FLAG_DISCONTINUITY,
  Encrypted: FLAG_ENCRYPTED
});

export class IncompleteSoundKitFrameHeaderError extends Error {
  constructor(message = "not enough bytes for SoundKit frame header") {
    super(message);
    this.name = "IncompleteSoundKitFrameHeaderError";
  }
}

const asUint8Array = (bytes) => {
  if (bytes instanceof Uint8Array) return bytes;
  if (bytes instanceof ArrayBuffer) return new Uint8Array(bytes);
  if (ArrayBuffer.isView(bytes)) {
    return new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  }
  throw new TypeError("bytes must be a Uint8Array, ArrayBuffer, or typed array view");
};

const assertAvailable = (bytes, offset, length) => {
  if (bytes.length < offset + length) {
    throw new IncompleteSoundKitFrameHeaderError();
  }
};

const assertOffset = (offset) => {
  if (!Number.isInteger(offset) || offset < 0) {
    throw new RangeError("offset must be a non-negative integer");
  }
};

const assertU32 = (value, name) => {
  if (!Number.isInteger(value) || value < 0 || value > U32_MAX) {
    throw new RangeError(`${name} must fit in u32`);
  }
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

const readU64Be = (bytes, offset) => {
  let value = 0n;
  for (let index = 0; index < 8; index += 1) {
    value = (value << 8n) | BigInt(bytes[offset + index]);
  }
  return value;
};

const readU32Be = (bytes, offset) => (
  (
    bytes[offset] * 0x1_000000
    + (bytes[offset + 1] << 16)
    + (bytes[offset + 2] << 8)
    + bytes[offset + 3]
  ) >>> 0
);

const writeU64Be = (bytes, offset, value) => {
  let remaining = value;
  for (let index = 7; index >= 0; index -= 1) {
    bytes[offset + index] = Number(remaining & 0xffn);
    remaining >>= 8n;
  }
};

const writeU32Be = (bytes, offset, value) => {
  bytes[offset] = (value >>> 24) & 0xff;
  bytes[offset + 1] = (value >>> 16) & 0xff;
  bytes[offset + 2] = (value >>> 8) & 0xff;
  bytes[offset + 3] = value & 0xff;
};

const defaultBitsPerSample = (encoding) => {
  if (encoding === SoundKitEncoding.PcmSigned) return 16;
  if (encoding === SoundKitEncoding.PcmFloat) return 32;
  return 0;
};

let crcTable;

const getCrcTable = () => {
  if (crcTable) return crcTable;

  const table = new Uint32Array(256);
  for (let index = 0; index < 256; index += 1) {
    let crc = index;
    for (let bit = 0; bit < 8; bit += 1) {
      crc = (crc & 1) !== 0 ? (0xedb8_8320 ^ (crc >>> 1)) >>> 0 : crc >>> 1;
    }
    table[index] = crc >>> 0;
  }

  crcTable = table;
  return table;
};

export const crc32IeeeUpdate = (previous, bytes) => {
  let crc = (~previous) >>> 0;
  const table = getCrcTable();
  const input = asUint8Array(bytes);

  for (let index = 0; index < input.length; index += 1) {
    crc = (table[(crc ^ input[index]) & 0xff] ^ (crc >>> 8)) >>> 0;
  }

  return (~crc) >>> 0;
};

export const crc32Ieee = (bytes) => crc32IeeeUpdate(0, bytes);

export const soundKitPacketCrc32 = (headerWithoutCrc, payload) => (
  crc32IeeeUpdate(crc32Ieee(asUint8Array(headerWithoutCrc)), asUint8Array(payload))
);

export const soundKitFrameHeaderByteLength = (bytes, offset = 0) => {
  const input = asUint8Array(bytes);
  assertOffset(offset);
  assertAvailable(input, offset, SOUNDKIT_FRAME_HEADER_BASE_BYTES);

  const word = readU32Be(input, offset);
  const magic = (word >>> MAGIC_SHIFT) & 0x3f;
  if (magic !== MAGIC_WORD) {
    throw new Error(`invalid SoundKit frame header magic 0x${magic.toString(16)}`);
  }
  const version = (word >>> VERSION_SHIFT) & 0x3;
  if (version !== VERSION) {
    throw new Error(`unsupported SoundKit frame header version ${version}`);
  }

  const flags = (word >>> FLAGS_SHIFT) & 0xff;
  if ((flags & FLAG_ID_U64) !== 0 && (flags & FLAG_ID_PRESENT) === 0) {
    throw new Error("SoundKit frame has 64-bit ID flag without ID present");
  }

  return SOUNDKIT_FRAME_HEADER_BASE_BYTES
    + ((flags & FLAG_EXTENDED_SIZES) !== 0 ? SOUNDKIT_FRAME_HEADER_EXTENDED_SIZE_BYTES : 0)
    + ((flags & FLAG_ID_PRESENT) !== 0 ? ((flags & FLAG_ID_U64) !== 0 ? 8 : 4) : 0)
    + ((flags & FLAG_PTS_PRESENT) !== 0 ? 8 : 0)
    + ((flags & FLAG_PACKET_CRC32_PRESENT) !== 0 ? 4 : 0);
};

export const decodeSoundKitFrameHeader = (bytes, offset = 0) => {
  const input = asUint8Array(bytes);
  assertOffset(offset);

  const startOffset = offset;
  assertAvailable(input, offset, SOUNDKIT_FRAME_HEADER_BASE_BYTES);
  const word = readU32Be(input, offset);
  const sizeWord = readU32Be(input, offset + 4);
  offset += SOUNDKIT_FRAME_HEADER_BASE_BYTES;

  const magic = (word >>> MAGIC_SHIFT) & 0x3f;
  if (magic !== MAGIC_WORD) {
    throw new Error(`invalid SoundKit frame header magic 0x${magic.toString(16)}`);
  }

  const version = (word >>> VERSION_SHIFT) & 0x3;
  if (version !== VERSION) {
    throw new Error(`unsupported SoundKit frame header version ${version}`);
  }

  const flags = (word >>> FLAGS_SHIFT) & 0xff;
  if ((flags & FLAG_ID_U64) !== 0 && (flags & FLAG_ID_PRESENT) === 0) {
    throw new Error("SoundKit frame has 64-bit ID flag without ID present");
  }

  const encoding = (word >>> ENCODING_SHIFT) & 0xf;
  if (SoundKitEncodingName[encoding] === undefined) {
    throw new Error(`unsupported SoundKit encoding ${encoding}`);
  }

  const sampleRateCode = (word >>> SAMPLE_RATE_SHIFT) & 0xf;
  const sampleRate = SOUNDKIT_SAMPLE_RATES[sampleRateCode];
  if (sampleRate === undefined) throw new Error("invalid SoundKit sample-rate code");

  const bitsCode = word & BITS_MASK;
  const bitsPerSample = SOUNDKIT_BITS_PER_SAMPLE[bitsCode];
  if (bitsPerSample === undefined) throw new Error("invalid SoundKit bits-per-sample code");

  const channels = ((word >>> CHANNELS_SHIFT) & 0x1f) + 1;
  if (channels < 1 || channels > 32) throw new Error("invalid SoundKit channel count");

  const hasExtendedSizes = (flags & FLAG_EXTENDED_SIZES) !== 0;
  if (hasExtendedSizes && sizeWord !== 0xffff_ffff) {
    throw new Error("extended SoundKit sizes must use short-size sentinels");
  }
  if (
    !hasExtendedSizes
    && (
      ((sizeWord >>> 16) & SHORT_SIZE_SENTINEL) === SHORT_SIZE_SENTINEL
      || (sizeWord & SHORT_SIZE_SENTINEL) === SHORT_SIZE_SENTINEL
    )
  ) {
    throw new Error("short-size sentinel requires extended SoundKit sizes");
  }

  let payloadSize;
  let frameCount;
  if (hasExtendedSizes) {
    assertAvailable(input, offset, SOUNDKIT_FRAME_HEADER_EXTENDED_SIZE_BYTES);
    payloadSize = readU32Be(input, offset);
    frameCount = readU32Be(input, offset + 4);
    offset += SOUNDKIT_FRAME_HEADER_EXTENDED_SIZE_BYTES;
  } else {
    payloadSize = (sizeWord >>> 16) & 0xffff;
    frameCount = sizeWord & 0xffff;
  }

  let id;
  const idIsU64 = (flags & FLAG_ID_U64) !== 0;
  if ((flags & FLAG_ID_PRESENT) !== 0) {
    if (idIsU64) {
      assertAvailable(input, offset, 8);
      id = readU64Be(input, offset);
      offset += 8;
    } else {
      assertAvailable(input, offset, 4);
      id = BigInt(readU32Be(input, offset));
      offset += 4;
    }
  }

  let pts;
  if ((flags & FLAG_PTS_PRESENT) !== 0) {
    assertAvailable(input, offset, 8);
    pts = readU64Be(input, offset);
    offset += 8;
  }

  let packetCrc32;
  if ((flags & FLAG_PACKET_CRC32_PRESENT) !== 0) {
    assertAvailable(input, offset, 4);
    packetCrc32 = readU32Be(input, offset);
    offset += 4;
  }

  return {
    encoding,
    encodingName: SoundKitEncodingName[encoding],
    payloadSize,
    frameCount,
    sampleRate,
    channels,
    bitsPerSample,
    endianness: (flags & FLAG_BIG_ENDIAN) === 0
      ? SoundKitEndianness.LittleEndian
      : SoundKitEndianness.BigEndian,
    id,
    idIsU64,
    pts,
    packetCrc32,
    packetFlags: flags & PUBLIC_PACKET_FLAGS,
    headerBytes: offset - startOffset
  };
};

export const encodeSoundKitFrameHeader = (header) => {
  if (SoundKitEncodingName[header.encoding] === undefined) {
    throw new RangeError(`unsupported SoundKit encoding ${header.encoding}`);
  }
  assertU32(header.payloadSize, "payloadSize");
  assertU32(header.frameCount, "frameCount");

  const sampleRateCode = SOUNDKIT_SAMPLE_RATES.indexOf(header.sampleRate);
  if (sampleRateCode < 0) throw new RangeError(`unsupported SoundKit sample rate ${header.sampleRate}`);

  const bitsPerSample = header.bitsPerSample ?? defaultBitsPerSample(header.encoding);
  const bitsCode = SOUNDKIT_BITS_PER_SAMPLE.indexOf(bitsPerSample);
  if (bitsCode < 0) throw new RangeError(`unsupported SoundKit bits per sample ${bitsPerSample}`);
  if (
    (header.encoding === SoundKitEncoding.PcmSigned || header.encoding === SoundKitEncoding.PcmFloat)
    && bitsPerSample === 0
  ) {
    throw new RangeError("PCM headers must set bitsPerSample");
  }
  if (!Number.isInteger(header.channels) || header.channels < 1 || header.channels > 32) {
    throw new RangeError("channels must be between 1 and 32");
  }

  const id = header.id === undefined ? undefined : toU64(header.id, "id");
  const idIsU64 = id !== undefined && (Boolean(header.idIsU64) || id > BigInt(U32_MAX));
  const pts = header.pts === undefined ? undefined : toU64(header.pts, "pts");
  const endianness = header.endianness ?? SoundKitEndianness.LittleEndian;
  const packetFlags = header.packetFlags ?? 0;
  if ((packetFlags & ~PUBLIC_PACKET_FLAGS) !== 0) {
    throw new RangeError("unsupported SoundKit packet flags set");
  }
  if (header.packetCrc32 !== undefined) assertU32(header.packetCrc32, "packetCrc32");

  const hasExtendedSizes = (
    header.payloadSize > SOUNDKIT_SHORT_SIZE_MAX
    || header.frameCount > SOUNDKIT_SHORT_SIZE_MAX
  );
  if (
    !hasExtendedSizes
    && (header.payloadSize === SHORT_SIZE_SENTINEL || header.frameCount === SHORT_SIZE_SENTINEL)
  ) {
    throw new RangeError("short SoundKit size fields reserve 65535 as the extension sentinel");
  }

  let flags = packetFlags;
  if (id !== undefined) flags |= FLAG_ID_PRESENT;
  if (idIsU64) flags |= FLAG_ID_U64;
  if (pts !== undefined) flags |= FLAG_PTS_PRESENT;
  if (header.packetCrc32 !== undefined) flags |= FLAG_PACKET_CRC32_PRESENT;
  if (endianness === SoundKitEndianness.BigEndian) flags |= FLAG_BIG_ENDIAN;
  if (hasExtendedSizes) flags |= FLAG_EXTENDED_SIZES;

  let byteLength = SOUNDKIT_FRAME_HEADER_BASE_BYTES;
  if (hasExtendedSizes) byteLength += SOUNDKIT_FRAME_HEADER_EXTENDED_SIZE_BYTES;
  if (id !== undefined) byteLength += idIsU64 ? 8 : 4;
  if (pts !== undefined) byteLength += 8;
  if (header.packetCrc32 !== undefined) byteLength += 4;

  const output = new Uint8Array(byteLength);
  let word = (MAGIC_WORD << MAGIC_SHIFT) >>> 0;
  word |= VERSION << VERSION_SHIFT;
  word |= flags << FLAGS_SHIFT;
  word |= header.encoding << ENCODING_SHIFT;
  word |= sampleRateCode << SAMPLE_RATE_SHIFT;
  word |= (header.channels - 1) << CHANNELS_SHIFT;
  word |= bitsCode;
  writeU32Be(output, 0, word >>> 0);

  const sizeWord = hasExtendedSizes
    ? 0xffff_ffff
    : header.payloadSize * 0x1_0000 + header.frameCount;
  writeU32Be(output, 4, sizeWord);

  let offset = SOUNDKIT_FRAME_HEADER_BASE_BYTES;
  if (hasExtendedSizes) {
    writeU32Be(output, offset, header.payloadSize);
    writeU32Be(output, offset + 4, header.frameCount);
    offset += SOUNDKIT_FRAME_HEADER_EXTENDED_SIZE_BYTES;
  }
  if (id !== undefined) {
    if (idIsU64) {
      writeU64Be(output, offset, id);
      offset += 8;
    } else {
      writeU32Be(output, offset, Number(id));
      offset += 4;
    }
  }
  if (pts !== undefined) {
    writeU64Be(output, offset, pts);
    offset += 8;
  }
  if (header.packetCrc32 !== undefined) {
    writeU32Be(output, offset, header.packetCrc32);
  }

  return output;
};

export const computeSoundKitPacketCrc32 = (header, payload) => {
  const encodedHeader = encodeSoundKitFrameHeader({ ...header, packetCrc32: 0 });
  return soundKitPacketCrc32(encodedHeader.subarray(0, encodedHeader.length - 4), asUint8Array(payload));
};

export const verifySoundKitPacketCrc32 = (header, encodedHeader, payload) => {
  if (header.packetCrc32 === undefined) return false;
  const headerBytes = asUint8Array(encodedHeader);
  if (headerBytes.length < header.headerBytes) {
    throw new RangeError("encodedHeader is shorter than header.headerBytes");
  }
  return soundKitPacketCrc32(headerBytes.subarray(0, header.headerBytes - 4), asUint8Array(payload))
    === header.packetCrc32;
};

export const isSoundKitFrameHeader = (bytes, offset = 0) => {
  try {
    decodeSoundKitFrameHeader(bytes, offset);
    return true;
  } catch {
    return false;
  }
};

export const soundKitPcmPayloadBytes = (header) => {
  if (header.encoding !== SoundKitEncoding.PcmSigned && header.encoding !== SoundKitEncoding.PcmFloat) {
    return undefined;
  }
  return header.frameCount * header.channels * (header.bitsPerSample / 8);
};

