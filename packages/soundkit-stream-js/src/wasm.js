export const loadSoundKitStreamWasm = async (moduleOrSpecifier, initInput) => {
  const module = typeof moduleOrSpecifier === "string"
    ? await import(moduleOrSpecifier)
    : moduleOrSpecifier;

  if (module && typeof module.default === "function") {
    await module.default(initInput);
  }

  return module;
};

export const encodePcmI16ToSoundKitOpusStreamWithWasm = (wasmModule, pcm, options = {}) => {
  if (!wasmModule || typeof wasmModule.encodePcmI16ToSoundKitOpusStream !== "function") {
    throw new TypeError("wasmModule must expose encodePcmI16ToSoundKitOpusStream");
  }
  const {
    sampleRate = 48_000,
    channels = 2,
    bitrate = 128_000,
    frameSize = 960
  } = options;
  return wasmModule.encodePcmI16ToSoundKitOpusStream(
    pcm,
    sampleRate,
    channels,
    bitrate,
    frameSize
  );
};

export const encodePcmI16ToSoundKitStreamsWithWasm = (wasmModule, pcm, options = {}) => {
  if (!wasmModule || typeof wasmModule.encodePcmI16ToSoundKitStreams !== "function") {
    throw new TypeError("wasmModule must expose encodePcmI16ToSoundKitStreams");
  }
  const {
    sampleRate = 48_000,
    channels = 2,
    bitrate = 128_000,
    frameSize = 960
  } = options;
  return wasmModule.encodePcmI16ToSoundKitStreams(
    pcm,
    sampleRate,
    channels,
    bitrate,
    frameSize
  );
};

