import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import init, {
  WasmAacLcDecoder,
  WasmAudioTrackDemuxer,
  WasmMp4MediaIndex,
  WasmMusicDecoder,
  WasmOpusDecoder,
  WasmVideoDecoder,
  WasmWebmMediaDemuxer,
} from "../soundkit-wasm/pkg/soundkit_wasm.js";

const fixtureRoot = resolve(process.argv[2] ?? "testdata/video-compat/never-final");
await init({ module_or_path: await readFile(new URL("../soundkit-wasm/pkg/soundkit_wasm_bg.wasm", import.meta.url)) });

// These assertions verify the JS/WASM serialization boundary. Frame safety and
// media validity are already enforced by VideoFrame::validate in Rust before
// export_video_frames can return an object to JavaScript.
function assertExportedFrameContract(codec, frames, expected) {
  assert.equal(frames.length, expected.frames, `${codec} frame count`);
  for (const frame of frames) {
    assert.equal(frame.width, 640, `${codec} width`);
    assert.equal(frame.height, 360, `${codec} height`);
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
  console.log(`${codec}: ${frames.length} frames, ${frames[0].width}x${frames[0].height}, ${frames[0].bitDepth}-bit ${frames[0].chromaSampling}`);
}

function countPcmFrames(frames) {
  return frames.reduce((total, frame) => {
    const bytesPerFrame = frame.channels * Math.ceil(frame.bitsPerSample / 8);
    assert.equal(frame.data.byteLength % bytesPerFrame, 0, "Rust PCM frame alignment");
    return total + frame.data.byteLength / bytesPerFrame;
  }, 0);
}


async function decodeMp4MediaFile(file, expected) {
  const bytes = await readFile(resolve(fixtureRoot, file));
  const index = WasmMp4MediaIndex.fromFile(bytes);
  const decoder = new WasmVideoDecoder(expected.codec);
  let audioDecoder = null;
  let audioPackets = 0;
  let audioBytes = 0;
  let audioFrames = 0;
  const frames = [];
  const videoPresentationTimes = [];
  const videoDurations = [];
  try {
    const tracks = index.tracks();
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
    for (let sampleIndex = 0; sampleIndex < index.sampleCount; sampleIndex += 1) {
      const sample = index.sample(sampleIndex);
      const source = bytes.subarray(sample.offset, sample.offset + sample.size);
      const packet = index.packet(sampleIndex, source);
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
            const trim = index.pcmTrim(sampleIndex, decodedFrames);
            audioFrames += trim?.frameCount ?? 0;
          } else {
            audioFrames += countPcmFrames(audioDecoder.push(packet.data));
          }
        } else if (audio.codec === "pcm") {
          const bytesPerFrame = audio.channels * Math.ceil(audio.bitsPerSample / 8);
          assert.equal(packet.data.byteLength % bytesPerFrame, 0, `${file} PCM frame alignment`);
          const decodedFrames = packet.data.byteLength / bytesPerFrame;
          const trim = index.pcmTrim(sampleIndex, decodedFrames);
          audioFrames += trim?.frameCount ?? 0;
        }
      }
    }
    if (audio.codec === "flac") {
      audioFrames += countPcmFrames(audioDecoder.flush());
    }
    frames.push(...decoder.flush());
    assertExportedFrameContract(file, frames, expected);
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
    console.log(`${file}: Rust decoded video plus ${audioFrames} audio frames`);
  } finally {
    audioDecoder?.free();
    decoder.free();
    index.free();
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
    assertExportedFrameContract(file, frames, expected);
    assert.ok(audioPackets > 0, `${file} extracts audio packets`);
    assert.equal(audioFrames, expected.audioFrames, `${file} decoded audio frame count`);
    console.log(`${file}: Rust decoded video plus ${audioFrames} audio frames`);
  } finally {
    audioDecoder?.free();
    decoder.free();
    demuxer.free();
  }
}

async function inspectExplicitVideoGap(file, expected) {
  const bytes = await readFile(resolve(fixtureRoot, file));
  const index = WasmMp4MediaIndex.fromFile(bytes);
  try {
    const tracks = index.tracks();
    const video = tracks.find((track) => track.kind === "video");
    const audio = tracks.find((track) => track.kind === "audio");
    assert.equal(video?.codec, expected.codec, `${file} identifies the video codec`);
    assert.equal(audio?.codec, expected.audioCodec, `${file} identifies the audio codec`);
    let videoPackets = 0;
    let audioFrames = 0;
    for (let sampleIndex = 0; sampleIndex < index.sampleCount; sampleIndex += 1) {
      const sample = index.sample(sampleIndex);
      const packet = index.packet(
        sampleIndex,
        bytes.subarray(sample.offset, sample.offset + sample.size),
      );
      if (packet.kind === "video") {
        videoPackets += 1;
      } else {
        const bytesPerFrame = audio.channels * Math.ceil(audio.bitsPerSample / 8);
        audioFrames += packet.data.byteLength / bytesPerFrame;
      }
    }
    assert.equal(videoPackets, expected.videoPackets, `${file} extracts every video packet`);
    assert.equal(audioFrames, expected.audioFrames, `${file} extracted PCM frame count`);
    assert.throws(
      () => new WasmVideoDecoder(expected.codec),
      /not(?: yet)? available|unsupported/i,
      `${file} reports the native decoder gap explicitly`,
    );
    console.log(`${file}: Rust extracted both tracks and reported the explicit decoder gap`);
  } finally {
    index.free();
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
await inspectExplicitVideoGap("dnxhr-hqx-pcm.mov", {
  codec: "dnxhr",
  audioCodec: "pcm",
  audioFrames: 144000,
  videoPackets: 75,
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

for (const codec of ["h264", "hevc", "vp9", "av1", "prores"]) {
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

assert.throws(() => new WasmVideoDecoder("dnxhr"), /not yet available/);

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

console.log("SoundKit WASM media conformance passed");
