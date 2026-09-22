import assert from 'node:assert/strict';
import test from 'node:test';
import { readFile } from 'node:fs/promises';
import { createAvifEncoder } from '../dist/target/index.mjs';
import { init as initDecoder, default as decode } from '../dist/decode.js';

const encoder = await createAvifEncoder(undefined, { wasmBinary: await readFile(new URL('../dist/target/soundkit_avif.wasm', import.meta.url)) });
await initDecoder({ wasmBinary: await readFile(new URL('../dist/codec/dec/avif_dec.wasm', import.meta.url)) });
const width = 96, height = 64, data = new Uint8Array(width * height * 4);
for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    data.set([(x * 3 + y * 5) & 255, (255 - x * 2) & 255, (x ^ y) & 255, 255], (y * width + x) * 4);
}
const options = { width, height, speed: 8 };

test('fixed quantizer encodes monochrome and colour with original dimensions', async () => {
    for (const monochrome of [true, false]) {
        const bytes = encoder.encode(data, { ...options, monochrome, quantizer: 30 });
        const result = await decode(bytes, { bitDepth: 10 });
        assert.deepEqual([result.width, result.height], [width, height]);
        assert.equal(result.data[3], 1023);
        if (monochrome) for (let i = 0; i < result.data.length; i += 4) {
            assert.equal(result.data[i], result.data[i + 1]);
            assert.equal(result.data[i], result.data[i + 2]);
        }
    }
});

test('target encoding fits the byte budget and reports impossible budgets', () => {
    const reference = encoder.encode(data, { ...options, quantizer: 40 });
    const result = encoder.encodeForTarget(data, { ...options, targetBytes: reference.length + 20 });
    assert.equal(result.fits, true);
    assert.ok(result.bytes.length <= reference.length + 20);
    const impossible = encoder.encodeForTarget(data, { ...options, targetBytes: 1 });
    assert.equal(impossible.fits, false);
    assert.ok(impossible.bytes.length > 1);
});
