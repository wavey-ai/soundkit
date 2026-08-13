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

/** Feed any sequential source into a Rust streaming demuxer with backpressure. */
export async function streamDemux(source, demuxer, consume) {
  for await (const chunk of source) {
    for (const event of demuxer.push(chunk)) await consume(event);
  }
  for (const event of demuxer.flush()) await consume(event);
}
