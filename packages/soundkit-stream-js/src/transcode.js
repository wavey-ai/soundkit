import {
  encodePcmI16ToSoundKitOpusStreamWithWasm,
  encodePcmI16ToSoundKitStreamsWithWasm
} from "./wasm.js";

const asUint8Array = (bytes) => {
  if (bytes instanceof Uint8Array) return bytes;
  if (bytes instanceof ArrayBuffer) return new Uint8Array(bytes);
  if (ArrayBuffer.isView(bytes)) {
    return new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  }
  throw new TypeError("bytes must be a Uint8Array, ArrayBuffer, or typed array view");
};

const readI16Le = (bytes, offset) => {
  const value = bytes[offset] | (bytes[offset + 1] << 8);
  return value >= 0x8000 ? value - 0x1_0000 : value;
};

export const pcmI16FromDecodedFrames = (
  frames,
  {
    requiredSampleRate = 48_000,
    requiredChannels
  } = {}
) => {
  if (!Array.isArray(frames)) {
    throw new TypeError("frames must be an array");
  }

  let sampleRate = requiredSampleRate;
  let channels = requiredChannels;
  let sampleCount = 0;
  const byteFrames = [];

  for (const frame of frames) {
    const frameSampleRate = Number(frame.sampleRate);
    const frameChannels = Number(frame.channels);
    const bitsPerSample = Number(frame.bitsPerSample);
    if (bitsPerSample !== 16) {
      throw new Error(`decoded frame bitsPerSample must be 16, got ${bitsPerSample}`);
    }
    if (frameSampleRate !== requiredSampleRate) {
      throw new Error(`decoded frame sampleRate must be ${requiredSampleRate}, got ${frameSampleRate}`);
    }
    if (channels === undefined) {
      channels = frameChannels;
    } else if (channels !== frameChannels) {
      throw new Error(`decoded frame channel count changed from ${channels} to ${frameChannels}`);
    }
    if (sampleRate !== frameSampleRate) {
      throw new Error(`decoded frame sampleRate changed from ${sampleRate} to ${frameSampleRate}`);
    }

    const data = asUint8Array(frame.data);
    if (data.length % 2 !== 0) {
      throw new Error("decoded i16 PCM frame has an odd byte length");
    }
    byteFrames.push(data);
    sampleCount += data.length / 2;
    sampleRate = frameSampleRate;
  }

  if (channels === undefined) {
    throw new Error("no decoded PCM frames were produced");
  }

  const pcm = new Int16Array(sampleCount);
  let outputOffset = 0;
  for (const data of byteFrames) {
    for (let offset = 0; offset < data.length; offset += 2) {
      pcm[outputOffset] = readI16Le(data, offset);
      outputOffset += 1;
    }
  }

  return {
    pcm,
    sampleRate,
    channels,
    frameCount: pcm.length / channels
  };
};

export const decodeSourceToPcmI16WithSoundKitWasm = async (
  decoderModule,
  chunks,
  options = {}
) => {
  if (!decoderModule || typeof decoderModule.WasmMusicDecoder !== "function") {
    throw new TypeError("decoderModule must expose WasmMusicDecoder");
  }

  const decoder = new decoderModule.WasmMusicDecoder();
  const frames = [];
  for await (const chunk of chunks) {
    frames.push(...decoder.push(asUint8Array(chunk)));
  }
  frames.push(...decoder.finish());
  return pcmI16FromDecodedFrames(frames, options);
};

export const transcodeSourceToSoundKitOpusStreamWithWasm = async ({
  decoderModule,
  encoderModule,
  chunks,
  sampleRate = 48_000,
  channels,
  bitrate = 128_000,
  frameSize = 960
}) => {
  const decoded = await decodeSourceToPcmI16WithSoundKitWasm(decoderModule, chunks, {
    requiredSampleRate: sampleRate,
    requiredChannels: channels
  });

  return encodePcmI16ToSoundKitOpusStreamWithWasm(encoderModule, decoded.pcm, {
    sampleRate: decoded.sampleRate,
    channels: decoded.channels,
    bitrate,
    frameSize
  });
};

export const transcodeSourceToSoundKitStreamsWithWasm = async ({
  decoderModule,
  encoderModule,
  chunks,
  sampleRate = 48_000,
  channels,
  bitrate = 128_000,
  frameSize = 960
}) => {
  const decoded = await decodeSourceToPcmI16WithSoundKitWasm(decoderModule, chunks, {
    requiredSampleRate: sampleRate,
    requiredChannels: channels
  });

  return encodePcmI16ToSoundKitStreamsWithWasm(encoderModule, decoded.pcm, {
    sampleRate: decoded.sampleRate,
    channels: decoded.channels,
    bitrate,
    frameSize
  });
};

