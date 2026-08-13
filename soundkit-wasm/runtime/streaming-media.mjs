// SoundKit's browser I/O adapters. All container validation and range planning
// are owned by Rust/WASM. This file only performs requested reads and forwards
// bounded chunks or samples.

async function readSourceRange(source, start, end) {
  if (typeof source.read === "function") {
    return source.read(start, end);
  }
  return new Uint8Array(await source.slice(start, end).arrayBuffer());
}

/**
 * Open a seekable MOV/MP4 without copying the complete source into WASM.
 *
 * `source` can be a browser File/Blob, or `{ size, read(start, end) }`.
 * The returned object owns its WASM index and must be closed.
 */
export async function openSeekableMp4(source, wasm) {
  let offset = 0;
  let moovRange = null;
  while (offset < source.size) {
    const header = await readSourceRange(source, offset, Math.min(offset + 16, source.size));
    const range = wasm.inspectMp4TopLevelBox(header, offset, source.size);
    if (range.boxType === "moov") {
      moovRange = range;
      break;
    }
    offset = range.end;
  }
  if (!moovRange) {
    throw new Error("MOV/MP4 source has no moov box");
  }

  const moov = await readSourceRange(
    source,
    moovRange.payloadOffset,
    moovRange.end,
  );
  const index = new wasm.WasmMp4MediaIndex(moov);
  const tracks = Array.from(index.tracks());
  let closed = false;
  return {
    tracks,
    sampleCount: index.sampleCount,
    async *packets({ trackId = null, kind = null } = {}) {
      if (closed) throw new Error("seekable MOV/MP4 source is closed");
      for (let sampleIndex = 0; sampleIndex < index.sampleCount; sampleIndex += 1) {
        const sample = index.sample(sampleIndex);
        if (trackId !== null && sample.trackId !== trackId) continue;
        if (kind !== null && sample.kind !== kind) continue;
        const bytes = await readSourceRange(
          source,
          sample.offset,
          sample.offset + sample.size,
        );
        yield { sampleIndex, packet: index.packet(sampleIndex, bytes) };
      }
    },
    pcmTrim(sampleIndex, decodedFrames) {
      if (closed) throw new Error("seekable MOV/MP4 source is closed");
      return index.pcmTrim(sampleIndex, decodedFrames);
    },
    close() {
      if (!closed) index.free();
      closed = true;
    },
  };
}

/** Decode ALAC from seekable M4A/MP4 one Rust-indexed packet at a time. */
export async function* decodeSeekableAlac(source, wasm) {
  const media = await openSeekableMp4(source, wasm);
  const track = media.tracks.find(
    (candidate) => candidate.kind === "audio" && candidate.codec === "alac",
  );
  if (!track) {
    media.close();
    throw new Error("MOV/MP4 source has no ALAC audio track");
  }
  const decoder = new wasm.WasmAlacPacketDecoder(track.codecPrivate);
  try {
    for await (const { sampleIndex, packet } of media.packets({
      trackId: track.trackId,
    })) {
      const frame = decoder.decode(packet.data);
      const bytesPerFrame = frame.channels * Math.ceil(frame.bitsPerSample / 8);
      const decodedFrames = frame.data.byteLength / bytesPerFrame;
      const trim = media.pcmTrim(sampleIndex, decodedFrames);
      if (trim !== null) yield { packet, frame, trim };
    }
  } finally {
    decoder.free();
    media.close();
  }
}

/**
 * Open seekable CAF ALAC through a Rust-owned packet index.
 *
 * The adapter reads only the file header, chunk headers, bounded metadata, and
 * requested packets. It skips the complete `data` extent while it builds the
 * index, including when the packet table follows that extent.
 */
export async function openSeekableCafAlac(source, wasm) {
  const fileHeader = await readSourceRange(source, 0, Math.min(8, source.size));
  wasm.validateCafFileHeader(fileHeader, source.size);

  let offset = 8;
  let description = new Uint8Array();
  let magicCookie = new Uint8Array();
  let packetTable = new Uint8Array();
  let dataOffset = 0;
  let dataSize = 0;
  while (offset < source.size) {
    const header = await readSourceRange(
      source,
      offset,
      Math.min(offset + 12, source.size),
    );
    const range = wasm.inspectCafChunk(header, offset, source.size);
    if (range.chunkType === "desc") {
      description = await readSourceRange(source, range.payloadOffset, range.end);
    } else if (range.chunkType === "kuki") {
      magicCookie = await readSourceRange(source, range.payloadOffset, range.end);
    } else if (range.chunkType === "pakt") {
      packetTable = await readSourceRange(source, range.payloadOffset, range.end);
    } else if (range.chunkType === "data") {
      dataOffset = range.payloadOffset;
      dataSize = range.payloadSize;
    }
    offset = range.end;
  }

  const index = new wasm.WasmCafAlacIndex(
    description,
    magicCookie,
    packetTable,
    dataOffset,
    dataSize,
  );
  let closed = false;
  return {
    sampleRate: index.sampleRate,
    channels: index.channels,
    bitDepth: index.bitDepth,
    validFrames: index.validFrames,
    packetCount: index.packetCount,
    magicCookie: index.magicCookie,
    async *packets() {
      if (closed) throw new Error("seekable CAF source is closed");
      for (let packetIndex = 0; packetIndex < index.packetCount; packetIndex += 1) {
        const sample = index.sample(packetIndex);
        const bytes = await readSourceRange(
          source,
          sample.offset,
          sample.offset + sample.size,
        );
        yield { packetIndex, data: index.packet(packetIndex, bytes) };
      }
    },
    close() {
      if (!closed) index.free();
      closed = true;
    },
  };
}

/** Decode CAF ALAC one Rust-indexed packet at a time. */
export async function* decodeSeekableCafAlac(source, wasm) {
  const media = await openSeekableCafAlac(source, wasm);
  const decoder = new wasm.WasmAlacPacketDecoder(media.magicCookie);
  try {
    for await (const packet of media.packets()) {
      yield { packet, frame: decoder.decode(packet.data) };
    }
  } finally {
    decoder.free();
    media.close();
  }
}

/** Feed any sequential source into a Rust streaming demuxer with backpressure. */
export async function streamDemux(source, demuxer, consume) {
  for await (const chunk of source) {
    for (const event of demuxer.push(chunk)) await consume(event);
  }
  for (const event of demuxer.flush()) await consume(event);
}
