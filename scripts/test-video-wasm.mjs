import assert from "node:assert/strict";
import { open, readFile } from "node:fs/promises";
import { resolve } from "node:path";
import init, {
  WasmAacLcDecoder,
  WasmAlacPacketDecoder,
  WasmAudioTrackDemuxer,
  WasmMp4MediaDemuxer,
  WasmMp4MediaIndex,
  WasmMusicDecoder,
  WasmMxfMediaDemuxer,
  WasmOpusDecoder,
  WasmVideoDecoder,
  WasmWebmMediaDemuxer,
  inspectMp4TopLevelBox,
} from "../soundkit-wasm/pkg/soundkit_wasm.js";
import {
  decodeSeekableAlac,
  openSeekableMp4,
} from "../soundkit-wasm/runtime/streaming-media.mjs";

const fixtureRoot = resolve(process.argv[2] ?? "testdata/video-compat/never-final");
await init({ module_or_path: await readFile(new URL("../soundkit-wasm/pkg/soundkit_wasm_bg.wasm", import.meta.url)) });

// Test only: confirm that Rust-owned, Rust-validated frames retain their values
// after WASM serialization. This is not production media validation.
function assertWasmFrameSerialization(codec, frames, expected) {
  assert.equal(frames.length, expected.frames, `${codec} frame count`);
  for (const frame of frames) {
    assert.equal(frame.width, expected.width ?? 640, `${codec} width`);
    assert.equal(frame.height, expected.height ?? 360, `${codec} height`);
    assert.equal(
      frame.colorModel,
      expected.colorModel ?? "ycbcr",
      `${codec} Rust color model`,
    );
    assert.equal(frame.hasAlpha, expected.hasAlpha ?? false, `${codec} alpha contract`);
    assert.equal(
      frame.planes.length,
      expected.hasAlpha ? 4 : expected.chroma === "400" ? 1 : 3,
      `${codec} plane count`,
    );
    for (const plane of frame.planes) {
      const bytesPerSample = frame.bitDepth <= 8 ? 1 : 2;
      assert.equal(
        plane.data.byteLength,
        plane.stride * plane.height * bytesPerSample,
        `${codec} plane byte contract`,
      );
    }
  }
  assert.equal(frames[0].bitDepth, expected.bitDepth, `${codec} bit depth`);
  assert.equal(frames[0].chromaSampling, expected.chroma, `${codec} chroma`);
  console.log(`${codec}: ${frames.length} frames, ${frames[0].width}x${frames[0].height}, ${frames[0].bitDepth}-bit ${frames[0].colorModel} ${frames[0].chromaSampling}`);
}

function countPcmFrames(frames) {
  return frames.reduce((total, frame) => {
    const bytesPerFrame = frame.channels * Math.ceil(frame.bitsPerSample / 8);
    assert.equal(frame.data.byteLength % bytesPerFrame, 0, "Rust PCM frame alignment");
    return total + frame.data.byteLength / bytesPerFrame;
  }, 0);
}

async function decodeSeekableAlacFixture() {
  const file = resolve("testdata/alac/A_Tusk_is_used_to_make_costly_gifts.m4a");
  const handle = await open(file, "r");
  const { size } = await handle.stat();
  let maximumRead = 0;
  const source = {
    size,
    async read(start, end) {
      const length = end - start;
      maximumRead = Math.max(maximumRead, length);
      const bytes = new Uint8Array(length);
      const { bytesRead } = await handle.read(bytes, 0, length, start);
      assert.equal(bytesRead, length, "ALAC range read length");
      return bytes;
    },
  };
  let frames = 0;
  let packets = 0;
  try {
    for await (const { frame, trim } of decodeSeekableAlac(source, {
      inspectMp4TopLevelBox,
      WasmAlacPacketDecoder,
      WasmMp4MediaIndex,
    })) {
      assert.equal(frame.sampleRate, 8_000, "ALAC sample rate");
      assert.equal(frame.channels, 1, "ALAC channels");
      assert.equal(frame.bitsPerSample, 16, "ALAC depth");
      frames += trim.frameCount;
      packets += 1;
    }
    assert.ok(packets > 1, "ALAC emits multiple bounded packets");
    assert.equal(frames, 23_680, "ALAC frame count");
    assert.ok(maximumRead < size, "ALAC never reads the complete M4A source");
    console.log(`ALAC M4A: Rust range-decoded ${packets} packets and ${frames} frames`);
  } finally {
    await handle.close();
  }
}


async function decodeMp4MediaFile(file, expected) {
  const handle = await open(resolve(fixtureRoot, file), "r");
  const { size } = await handle.stat();
  let maximumRead = 0;
  const source = {
    size,
    async read(start, end) {
      const length = end - start;
      maximumRead = Math.max(maximumRead, length);
      const bytes = new Uint8Array(length);
      let written = 0;
      while (written < length) {
        const result = await handle.read(bytes, written, length - written, start + written);
        if (result.bytesRead === 0) throw new Error(`${file} ended during a planned range read`);
        written += result.bytesRead;
      }
      return bytes;
    },
  };
  const media = await openSeekableMp4(source, {
    inspectMp4TopLevelBox,
    WasmMp4MediaIndex,
  });
  const decoder = new WasmVideoDecoder(expected.codec);
  let audioDecoder = null;
  let audioPackets = 0;
  let audioBytes = 0;
  let audioFrames = 0;
  const frames = [];
  const videoPresentationTimes = [];
  const videoDurations = [];
  try {
    const tracks = media.tracks;
    const video = tracks.find((track) => track.kind === "video");
    const audio = tracks.find((track) => track.kind === "audio");
    assert.ok(video, `${file} has a Rust-indexed video track`);
    assert.ok(audio, `${file} has a Rust-indexed audio track`);
    assert.equal(video.codec, expected.codec, `${file} video codec`);
    assert.equal(audio.codec, expected.audioCodec, `${file} audio codec`);
    if (expected.videoTimeline) {
      assert.deepEqual(video.timeline, expected.videoTimeline, `${file} video edit timeline`);
    }
    if (expected.audioTimeline) {
      assert.deepEqual(audio.timeline, expected.audioTimeline, `${file} audio edit timeline`);
    }
    if (audio.codec === "aac") {
      audioDecoder = new WasmAacLcDecoder(audio.decoderConfiguration);
    } else if (audio.codec === "flac") {
      audioDecoder = WasmMusicDecoder.newWithFormat("flac");
      audioFrames += countPcmFrames(audioDecoder.push(audio.decoderConfiguration));
    }

    if (video.decoderConfiguration.byteLength > 0) {
      frames.push(...decoder.decode(video.decoderConfiguration, Number.NaN, Number.NaN));
    }
    for await (const { sampleIndex, packet } of media.packets()) {
      if (packet.kind === "video") {
        videoPresentationTimes.push(packet.presentationTime);
        videoDurations.push(packet.duration);
        frames.push(...decoder.decode(packet.data, packet.presentationTime, packet.duration));
      } else {
        audioPackets += 1;
        audioBytes += packet.data.byteLength;
        if (audioDecoder) {
          if (audio.codec === "aac") {
            const decodedFrames = audioDecoder.decodeInterleaved(packet.data).length / audio.channels;
            const trim = media.pcmTrim(sampleIndex, decodedFrames);
            audioFrames += trim?.frameCount ?? 0;
          } else {
            audioFrames += countPcmFrames(audioDecoder.push(packet.data));
          }
        } else if (audio.codec === "pcm") {
          const bytesPerFrame = audio.channels * Math.ceil(audio.bitsPerSample / 8);
          assert.equal(packet.data.byteLength % bytesPerFrame, 0, `${file} PCM frame alignment`);
          const decodedFrames = packet.data.byteLength / bytesPerFrame;
          const trim = media.pcmTrim(sampleIndex, decodedFrames);
          audioFrames += trim?.frameCount ?? 0;
        }
      }
    }
    if (audio.codec === "flac") {
      audioFrames += countPcmFrames(audioDecoder.flush());
    }
    frames.push(...decoder.flush());
    assertWasmFrameSerialization(file, frames, expected);
    assert.deepEqual(
      frames.map((frame) => frame.pts).sort((left, right) => left - right),
      videoPresentationTimes.sort((left, right) => left - right),
      `${file} preserves the Rust edit-list presentation timeline through decode`,
    );
    assert.deepEqual(
      frames.map((frame) => frame.duration).sort((left, right) => left - right),
      videoDurations.sort((left, right) => left - right),
      `${file} preserves Rust packet durations through decode`,
    );
    if (expected.variableFrameDurations) {
      assert.ok(new Set(videoDurations).size > 1, `${file} keeps variable frame durations`);
    }
    assert.ok(audioPackets > 0, `${file} extracts audio packets`);
    assert.ok(audioBytes > 0, `${file} extracts audio bytes`);
    assert.equal(audioFrames, expected.audioFrames, `${file} decoded audio frame count`);
    assert.ok(maximumRead < size, `${file} never reads the complete source`);
    console.log(`${file}: Rust decoded video plus ${audioFrames} audio frames`);
  } finally {
    audioDecoder?.free();
    decoder.free();
    media.close();
    await handle.close();
  }
}

async function decodeFragmentedMp4MediaFile(file, expected) {
  const bytes = await readFile(resolve(fixtureRoot, file));
  const demuxer = new WasmMp4MediaDemuxer();
  const events = [];
  let videoDecoder = null;
  let audioDecoder = null;
  const frames = [];
  const videoPresentationTimes = [];
  let audioFrames = 0;
  try {
    // Deliberately split across MP4 box fields and media samples. The Rust
    // demuxer must retain incomplete input without a JavaScript parser.
    for (let offset = 0; offset < bytes.length; offset += 4093) {
      events.push(...demuxer.push(bytes.subarray(offset, offset + 4093)));
    }
    events.push(...demuxer.flush());

    const video = events.find((event) => event.type === "config" && event.kind === "video");
    const audio = events.find((event) => event.type === "config" && event.kind === "audio");
    assert.ok(video, `${file} has a Rust-indexed fragmented video track`);
    assert.ok(audio, `${file} has a Rust-indexed fragmented audio track`);
    assert.equal(video.codec, expected.codec, `${file} video codec`);
    assert.equal(audio.codec, expected.audioCodec, `${file} audio codec`);

    videoDecoder = new WasmVideoDecoder(video.codec);
    audioDecoder = new WasmAacLcDecoder(audio.decoderConfiguration);
    if (video.decoderConfiguration.byteLength > 0) {
      frames.push(...videoDecoder.decode(video.decoderConfiguration, Number.NaN, Number.NaN));
    }
    for (const event of events) {
      if (event.type !== "packet") continue;
      if (event.kind === "video") {
        videoPresentationTimes.push(event.presentationTime);
        frames.push(...videoDecoder.decode(event.data, event.presentationTime, event.duration));
      } else {
        const decodedFrames = audioDecoder.decodeInterleaved(event.data).length / audio.channels;
        const trim = demuxer.pcmTrim(
          event.trackId,
          event.presentationTime,
          event.duration,
          decodedFrames,
        );
        audioFrames += trim?.frameCount ?? 0;
      }
    }
    frames.push(...videoDecoder.flush());

    assertWasmFrameSerialization(file, frames, expected);
    assert.deepEqual(
      frames.map((frame) => frame.pts).sort((left, right) => left - right),
      videoPresentationTimes.sort((left, right) => left - right),
      `${file} preserves fragmented MP4 presentation timestamps`,
    );
    assert.equal(audioFrames, expected.audioFrames, `${file} decoded audio frame count`);
    console.log(`${file}: Rust streamed fragmented video plus ${audioFrames} audio frames`);
  } finally {
    audioDecoder?.free();
    videoDecoder?.free();
    demuxer.free();
  }
}

async function inspectMxfMediaFile(file, expected) {
  const bytes = await readFile(resolve(fixtureRoot, file));
  const demuxer = new WasmMxfMediaDemuxer();
  const decoder = new WasmVideoDecoder("dnxhr");
  const events = [];
  const frames = [];
  try {
    // Split KLV keys, BER lengths, metadata sets, and essence payloads. Rust
    // owns reassembly, bounds checking, metadata resolution, and DNx headers.
    for (let offset = 0; offset < bytes.length; offset += 32749) {
      events.push(...demuxer.push(bytes.subarray(offset, offset + 32749)));
    }
    events.push(...demuxer.flush());

    const video = events.find((event) => event.type === "config" && event.kind === "video");
    const audio = events.find((event) => event.type === "config" && event.kind === "audio");
    assert.ok(video, `${file} has a Rust-resolved picture track`);
    assert.ok(audio, `${file} has a Rust-resolved sound track`);
    assert.equal(video.codec, "dnxhr", `${file} video codec`);
    assert.equal(video.codecId, "dnxhr-hqx", `${file} DNx profile`);
    assert.equal(video.width, 640, `${file} visible width`);
    assert.equal(video.height, 360, `${file} visible height`);
    assert.equal(video.bitsPerSample, 10, `${file} video depth`);
    assert.equal(audio.codec, "pcm", `${file} audio codec`);
    assert.equal(audio.codecId, "pcm_s24le", `${file} PCM format`);
    assert.equal(audio.sampleRate, 48_000, `${file} audio rate`);
    assert.equal(audio.channels, 2, `${file} audio channels`);
    assert.equal(audio.bitsPerSample, 24, `${file} audio depth`);

    const videoPackets = events.filter(
      (event) => event.type === "packet" && event.kind === "video",
    );
    const audioPackets = events.filter(
      (event) => event.type === "packet" && event.kind === "audio",
    );
    assert.equal(videoPackets.length, expected.videoPackets, `${file} video packet count`);
    assert.equal(audioPackets.length, expected.audioPackets, `${file} audio packet count`);
    assert.deepEqual(
      videoPackets.map((packet) => packet.decodeTime),
      Array.from({ length: expected.videoPackets }, (_, index) => index),
      `${file} video timestamps`,
    );
    assert.equal(
      audioPackets.reduce((frames, packet) => frames + packet.duration, 0),
      expected.audioFrames,
      `${file} PCM frame count`,
    );
    for (const packet of videoPackets) {
      frames.push(...decoder.decode(packet.data, packet.presentationTime, packet.duration));
    }
    frames.push(...decoder.flush());
    assertWasmFrameSerialization(file, frames, {
      frames: expected.videoPackets,
      bitDepth: 10,
      chroma: "422",
    });
    console.log(
      `${file}: Rust decoded ${frames.length} DNxHR frames plus ${expected.audioFrames} PCM frames`,
    );
  } finally {
    decoder.free();
    demuxer.free();
  }
}

async function decodeWebmMediaFile(file, expected) {
  const bytes = await readFile(resolve(fixtureRoot, file));
  const demuxer = new WasmWebmMediaDemuxer();
  const decoder = new WasmVideoDecoder(expected.codec);
  const frames = [];
  let videoConfig = null;
  let audioConfig = null;
  let audioPackets = 0;
  let audioFrames = 0;
  let audioDecoder = null;
  try {
    const events = [];
    for (let offset = 0; offset < bytes.length; offset += 64 * 1024) {
      events.push(...demuxer.push(bytes.subarray(offset, offset + 64 * 1024)));
    }
    events.push(...demuxer.flush());
    for (const event of events) {
      if (event.type === "config") {
        if (event.kind === "video") {
          videoConfig = event;
          if (event.decoderConfiguration.byteLength > 0 &&
              ["V_MPEG4/ISO/AVC", "V_MPEGH/ISO/HEVC"].includes(event.codecId)) {
            frames.push(...decoder.decode(event.decoderConfiguration, Number.NaN, Number.NaN));
          }
        }
        if (event.kind === "audio") {
          audioConfig = event;
          if (event.codecId === "A_OPUS") {
            audioDecoder = new WasmOpusDecoder(event.channels, event.sampleRate, 5760);
          } else if (event.codecId === "A_AAC") {
            audioDecoder = new WasmAacLcDecoder(event.decoderConfiguration);
          }
        }
        continue;
      }
      if (event.kind === "video") {
        frames.push(...decoder.decode(event.data, event.timestampNs, Number.NaN));
      } else {
        audioPackets += 1;
        if (audioConfig.codecId === "A_OPUS") {
          audioFrames += audioDecoder?.dec_frame_reuse(event.data) ?? 0;
        } else if (audioConfig.codecId === "A_AAC") {
          audioFrames += audioDecoder.decodeInterleaved(event.data).length / audioConfig.channels;
        }
      }
    }
    frames.push(...decoder.flush());
    assert.ok(videoConfig, `${file} has a Rust-indexed video track`);
    assert.ok(audioConfig, `${file} has a Rust-indexed audio track`);
    assert.equal(videoConfig.codecId, expected.codecId, `${file} video codec ID`);
    assert.equal(audioConfig.codecId, expected.audioCodecId, `${file} audio codec ID`);
    assertWasmFrameSerialization(file, frames, expected);
    assert.ok(audioPackets > 0, `${file} extracts audio packets`);
    assert.equal(audioFrames, expected.audioFrames, `${file} decoded audio frame count`);
    console.log(`${file}: Rust decoded video plus ${audioFrames} audio frames`);
  } finally {
    audioDecoder?.free();
    decoder.free();
    demuxer.free();
  }
}

async function inspectExplicitProfileGap(file, expected) {
  const bytes = await readFile(resolve(fixtureRoot, file));
  const index = WasmMp4MediaIndex.fromFile(bytes);
  const decoder = new WasmVideoDecoder(expected.codec);
  try {
    const video = index.tracks().find((track) => track.kind === "video");
    assert.ok(video, `${file} has a Rust-indexed video track`);
    assert.equal(video.codec, expected.codec, `${file} video codec`);
    let capabilityError = null;
    try {
      if (video.decoderConfiguration.byteLength > 0) {
        decoder.decode(video.decoderConfiguration, Number.NaN, Number.NaN);
      }
      for (let sampleIndex = 0; sampleIndex < index.sampleCount; sampleIndex += 1) {
        const sample = index.sample(sampleIndex);
        if (sample.kind !== "video") continue;
        const packet = index.packet(
          sampleIndex,
          bytes.subarray(sample.offset, sample.offset + sample.size),
        );
        decoder.decode(packet.data, packet.presentationTime, packet.duration);
      }
      decoder.flush();
    } catch (error) {
      capabilityError = error;
    }
    assert.ok(capabilityError, `${file} must not silently decode an unsupported profile`);
    assert.equal(
      capabilityError instanceof WebAssembly.RuntimeError,
      false,
      `${file} reports a typed capability error rather than trapping`,
    );
    assert.match(String(capabilityError), expected.error, `${file} capability error`);
    console.log(`${file}: Rust reported the explicit native profile gap`);
  } finally {
    decoder.free();
    index.free();
  }
}

await decodeMp4MediaFile("h264-high-aac.mp4", {
  codec: "h264",
  audioCodec: "aac",
  audioFrames: 144000,
  frames: 75,
  bitDepth: 8,
  chroma: "420",
  videoTimeline: { presentationStart: 0, mediaStart: 1024, duration: 38400 },
  audioTimeline: { presentationStart: 0, mediaStart: 1024, duration: 144000 },
});
await decodeMp4MediaFile("h264-vfr-aac.mp4", {
  codec: "h264",
  audioCodec: "aac",
  audioFrames: 144000,
  frames: 31,
  bitDepth: 8,
  chroma: "420",
  variableFrameDurations: true,
});
for (const file of [
  "h264-aac-fragmented.mp4",
  "h264-aac-cmaf.mp4",
  "h264-aac-dash.mp4",
  "h264-aac-separate-moof.mp4",
]) {
  await decodeFragmentedMp4MediaFile(file, {
    codec: "h264",
    audioCodec: "aac",
    audioFrames: 145408,
    frames: 75,
    bitDepth: 8,
    chroma: "420",
  });
}
await decodeMp4MediaFile("h264-flac.mp4", {
  codec: "h264",
  audioCodec: "flac",
  audioFrames: 144384,
  frames: 75,
  bitDepth: 8,
  chroma: "420",
});
await decodeMp4MediaFile("hevc-main-aac.mov", {
  codec: "hevc",
  audioCodec: "aac",
  audioFrames: 144000,
  frames: 75,
  bitDepth: 8,
  chroma: "420",
});
await decodeMp4MediaFile("hevc-main10-pcm.mov", {
  codec: "hevc",
  audioCodec: "pcm",
  audioFrames: 144000,
  frames: 75,
  bitDepth: 10,
  chroma: "420",
});
await decodeMp4MediaFile("prores-422-hq-pcm.mov", {
  codec: "prores",
  audioCodec: "pcm",
  audioFrames: 144000,
  frames: 75,
  bitDepth: 10,
  chroma: "422",
});
await decodeMp4MediaFile("prores-proxy-pcm.mov", {
  codec: "prores",
  audioCodec: "pcm",
  audioFrames: 144000,
  frames: 75,
  bitDepth: 10,
  chroma: "422",
});
await decodeMp4MediaFile("prores-lt-pcm.mov", {
  codec: "prores",
  audioCodec: "pcm",
  audioFrames: 144000,
  frames: 75,
  bitDepth: 10,
  chroma: "422",
});
await decodeMp4MediaFile("prores-standard-pcm.mov", {
  codec: "prores",
  audioCodec: "pcm",
  audioFrames: 144000,
  frames: 75,
  bitDepth: 10,
  chroma: "422",
});
await decodeMp4MediaFile("prores-4444-alpha-pcm.mov", {
  codec: "prores",
  audioCodec: "pcm",
  audioFrames: 144000,
  frames: 75,
  bitDepth: 12,
  chroma: "444",
  hasAlpha: true,
});
await decodeMp4MediaFile("prores-4444xq-alpha-pcm.mov", {
  codec: "prores",
  audioCodec: "pcm",
  audioFrames: 144000,
  frames: 75,
  bitDepth: 12,
  chroma: "444",
  hasAlpha: true,
});
await decodeWebmMediaFile("vp9-profile0-opus.webm", {
  codec: "vp9",
  codecId: "V_VP9",
  audioCodecId: "A_OPUS",
  audioFrames: 144960,
  frames: 75,
  bitDepth: 8,
  chroma: "420",
});
await decodeWebmMediaFile("vp9-profile2-10bit-opus.webm", {
  codec: "vp9",
  codecId: "V_VP9",
  audioCodecId: "A_OPUS",
  audioFrames: 144960,
  frames: 75,
  bitDepth: 10,
  chroma: "420",
});
await decodeWebmMediaFile("av1-main-opus.webm", {
  codec: "av1",
  codecId: "V_AV1",
  audioCodecId: "A_OPUS",
  audioFrames: 144960,
  frames: 75,
  bitDepth: 8,
  chroma: "420",
});
await decodeWebmMediaFile("av1-main10-opus.webm", {
  codec: "av1",
  codecId: "V_AV1",
  audioCodecId: "A_OPUS",
  audioFrames: 144960,
  frames: 75,
  bitDepth: 10,
  chroma: "420",
});
await decodeWebmMediaFile("matroska-h264-aac.mkv", {
  codec: "h264",
  codecId: "V_MPEG4/ISO/AVC",
  audioCodecId: "A_AAC",
  audioFrames: 145408,
  frames: 75,
  bitDepth: 8,
  chroma: "420",
});
await decodeWebmMediaFile("matroska-hevc-aac.mkv", {
  codec: "hevc",
  codecId: "V_MPEGH/ISO/HEVC",
  audioCodecId: "A_AAC",
  audioFrames: 145408,
  frames: 75,
  bitDepth: 8,
  chroma: "420",
});
await decodeMp4MediaFile("dnxhr-hqx-pcm.mov", {
  codec: "dnxhr",
  audioCodec: "pcm",
  audioFrames: 144000,
  frames: 75,
  bitDepth: 10,
  chroma: "422",
});
for (const profile of ["hq", "sq", "lb"]) {
  await decodeMp4MediaFile(`dnxhr-${profile}-pcm.mov`, {
    codec: "dnxhr",
    audioCodec: "pcm",
    audioFrames: 144000,
    frames: 75,
    bitDepth: 8,
    chroma: "422",
  });
}
for (const [model, colorModel] of [["gbr", "gbr"], ["yuv", "ycbcr"]]) {
  await decodeMp4MediaFile(`dnxhr-444-${model}10-pcm.mov`, {
    codec: "dnxhr",
    audioCodec: "pcm",
    audioFrames: 9216,
    frames: 5,
    bitDepth: 10,
    chroma: "444",
    colorModel,
  });
}
for (const [profile, bitDepth] of [
  ["1080p120-8bit", 8],
  ["1080p185-8bit", 8],
  ["1080p185-10bit", 10],
  ["1080p36-8bit", 8],
]) {
  await decodeMp4MediaFile(`dnxhd-${profile}-pcm.mov`, {
    codec: "dnxhr",
    audioCodec: "pcm",
    audioFrames: 3072,
    frames: 2,
    width: 1920,
    height: 1080,
    bitDepth,
    chroma: "422",
  });
}
await inspectMxfMediaFile("dnxhr-hqx-pcm.mxf", {
  videoPackets: 75,
  audioPackets: 75,
  audioFrames: 144000,
});
await inspectExplicitProfileGap("h264-high422-aac.mp4", {
  codec: "h264",
  error: /non-4:2:0 chroma/i,
});
await inspectExplicitProfileGap("h264-high444-aac.mp4", {
  codec: "h264",
  error: /non-4:2:0 chroma/i,
});
await inspectExplicitProfileGap("hevc-main422-10-aac.mov", {
  codec: "hevc",
  error: /only 4:2:0.*supported/i,
});
await inspectExplicitProfileGap("hevc-main444-10-aac.mov", {
  codec: "hevc",
  error: /only 4:2:0.*supported/i,
});

for (const codec of ["h264", "hevc", "vp9", "av1", "prores", "dnxhr"]) {
  const decoder = new WasmVideoDecoder(codec);
  try {
    let completed = false;
    try {
      const outcome = decoder.decode(Uint8Array.from([0xff, 0, 1, 2, 3]), 0, 1);
      assert.ok(Array.isArray(outcome), `${codec} malformed input returns frames or an error`);
      completed = true;
    } catch (error) {
      assert.notEqual(error, undefined, `${codec} malformed input reports an error`);
      completed = true;
    }
    assert.equal(completed, true, `${codec} malformed input stays bounded`);
  } finally {
    decoder.free();
  }
}

async function inspectAudio(file, expected) {
  const demuxer = WasmAudioTrackDemuxer.newAuto();
  let config;
  let packets = 0;
  try {
    const bytes = await readFile(resolve(fixtureRoot, file));
    for (let offset = 0; offset < bytes.length; offset += 64 * 1024) {
      for (const event of demuxer.push(bytes.subarray(offset, offset + 64 * 1024))) {
        if (event.type === "config") config = event;
        if (event.type === "packet") packets += 1;
      }
    }
    for (const event of demuxer.flush()) {
      if (event.type === "config") config = event;
      if (event.type === "packet") packets += 1;
    }
  } finally {
    demuxer.free();
  }
  assert.ok(config, `${file} audio config`);
  assert.equal(config.codec, expected.codec, `${file} audio codec`);
  assert.equal(config.sampleRate, 48_000, `${file} audio rate`);
  assert.equal(config.channels, 2, `${file} audio channels`);
  if (expected.bits) assert.equal(config.bitsPerSample, expected.bits, `${file} PCM depth`);
  assert.ok(packets > 0, `${file} audio packets`);
  console.log(`${file}: ${config.codec} ${config.sampleRate}Hz/${config.channels}ch, ${packets} packets`);
}

await inspectAudio("h264-high-aac.mp4", { codec: "aac" });
for (const file of [
  "h264-aac-fragmented.mp4",
  "h264-aac-cmaf.mp4",
  "h264-aac-dash.mp4",
  "h264-aac-separate-moof.mp4",
]) {
  await inspectAudio(file, { codec: "aac" });
}
await inspectAudio("h264-vfr-aac.mp4", { codec: "aac" });
await inspectAudio("h264-high422-aac.mp4", { codec: "aac" });
await inspectAudio("h264-high444-aac.mp4", { codec: "aac" });
await inspectAudio("hevc-main-aac.mov", { codec: "aac" });
await inspectAudio("hevc-main10-pcm.mov", { codec: "pcm", bits: 24 });
await inspectAudio("hevc-main422-10-aac.mov", { codec: "aac" });
await inspectAudio("hevc-main444-10-aac.mov", { codec: "aac" });
await inspectAudio("prores-proxy-pcm.mov", { codec: "pcm", bits: 24 });
await inspectAudio("prores-lt-pcm.mov", { codec: "pcm", bits: 24 });
await inspectAudio("prores-standard-pcm.mov", { codec: "pcm", bits: 24 });
await inspectAudio("prores-422-hq-pcm.mov", { codec: "pcm", bits: 24 });
await inspectAudio("prores-4444-alpha-pcm.mov", { codec: "pcm", bits: 24 });
await inspectAudio("prores-4444xq-alpha-pcm.mov", { codec: "pcm", bits: 24 });
await inspectAudio("dnxhr-hqx-pcm.mov", { codec: "pcm", bits: 24 });
await inspectAudio("vp9-profile0-opus.webm", { codec: "opus" });
await inspectAudio("vp9-profile2-10bit-opus.webm", { codec: "opus" });
await inspectAudio("av1-main-opus.webm", { codec: "opus" });
await inspectAudio("av1-main10-opus.webm", { codec: "opus" });
await decodeSeekableAlacFixture();

console.log("SoundKit WASM media conformance passed");
