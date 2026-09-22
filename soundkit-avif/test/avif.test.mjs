import assert from 'node:assert/strict';
import test from 'node:test';
import { readFile } from 'node:fs/promises';
import { encode, encodeSRGB, decode, initEncoder, initDecoder, tagAvifSRGB } from '../dist/index.mjs';

// The eight-bit decoder returns ImageData; Node has no presentation API.
globalThis.ImageData ??= class ImageData {
    constructor(data, width, height) { Object.assign(this, { data, width, height }); }
};
await initEncoder({ wasmBinary: await readFile(new URL('../dist/codec/enc/avif_enc.wasm', import.meta.url)) });
await initDecoder({ wasmBinary: await readFile(new URL('../dist/codec/dec/avif_dec.wasm', import.meta.url)) });

function pixels(bitDepth = 8) {
    const width = 32, height = 24, max = (1 << bitDepth) - 1;
    const data = bitDepth === 8 ? new Uint8ClampedArray(width * height * 4) : new Uint16Array(width * height * 4);
    for (let i = 0; i < width * height; i++) {
        data.set([Math.round(max * .6), Math.round(max * .3), Math.round(max * .15), i % width < 8 ? 0 : max], i * 4);
    }
    return { width, height, data };
}

test('AVIF encodes and decodes RGBA with transparency and respects typed-array offsets', async () => {
    const input = pixels(), padded = new Uint8Array(input.data.length + 32).fill(255);
    padded.set(input.data, 16);
    const bytes = await encode({ ...input, data: padded.subarray(16, -16) }, { quality: 90, speed: 8, subsample: 3 });
    const result = await decode(bytes);
    assert.deepEqual([result.width, result.height], [input.width, input.height]);
    assert.equal(result.data[3], 0);
    const at = (12 * input.width + 20) * 4;
    assert.equal(result.data[at + 3], 255);
    for (let channel = 0; channel < 3; channel++) assert.ok(Math.abs(result.data[at + channel] - input.data[at + channel]) < 8);
});

test('10-bit sRGB AVIF preserves dimensions, alpha and both colour descriptors', async () => {
    const input = pixels(10);
    const raw = new Uint8Array(await encode(input, { quality: 90, speed: 8, bitDepth: 10, subsample: 3 }));
    const original = raw.slice();
    const tagged = tagAvifSRGB(raw);
    assert.deepEqual(raw, original, 'Tagging does not mutate caller bytes');
    assert.equal(tagged.length, raw.length);
    assert.deepEqual(tagAvifSRGB(tagged), tagged, 'Tagging is idempotent');
    const nclx = Buffer.from(tagged).indexOf('nclx');
    assert.ok(nclx > 0);
    const view = new DataView(tagged.buffer, tagged.byteOffset, tagged.byteLength);
    assert.equal(view.getUint16(nclx + 4), 1);
    assert.equal(view.getUint16(nclx + 6), 13);
    assert.equal(view.getUint16(nclx + 8), 6);
    const encoded = await encodeSRGB(input, { quality: 90, speed: 8 });
    const decoded = await decode(encoded, { bitDepth: 10 });
    assert.deepEqual([decoded.width, decoded.height], [32, 24]);
    assert.ok(decoded.data instanceof Uint16Array);
    assert.equal(decoded.data[3], 0);
    const at = (12 * input.width + 20) * 4;
    assert.ok(decoded.data[at] > 255, 'Ten-bit output survives encoding');
    assert.equal(decoded.data[at + 3], 1023);
    for (let channel = 0; channel < 3; channel++) assert.ok(Math.abs(decoded.data[at + channel] - input.data[at + channel]) < 32);
});

test('invalid buffers and unsupported sRGB profiles fail before encoding', async () => {
    await assert.rejects(encode({ ...pixels(), width: 0 }), /dimensions/);
    await assert.rejects(encode({ ...pixels(), data: new Uint8Array(1) }), /RGBA/);
    await assert.rejects(encode(pixels(), { bitDepth: 10 }), /RGBA/);
    await assert.rejects(encodeSRGB(pixels(), { bitDepth: 8 }), /10-bit/);
    await assert.rejects(encodeSRGB(pixels(10), { lossless: true }), /lossy/);
    assert.throws(() => tagAvifSRGB(new Uint8Array([0, 1, 2])), /colour header/);
});
