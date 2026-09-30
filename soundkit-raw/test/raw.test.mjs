import assert from 'node:assert/strict';
import test from 'node:test';
import { createRawDecoder, develop, linearPreview, defaultRecipe, fromRGBA } from '../dist/index.mjs';
import { makeDNG } from './dng.mjs';

test('real LibRaw demosaics 12-bit DNG and preserves editable sensor precision', async () => {
    const decoder = await createRawDecoder();
    try {
        const metadata = decoder.open(makeDNG());
        assert.equal(metadata.make, 'SoundKit');
        const frame = decoder.decode();
        assert.equal(frame.width, 320); assert.equal(frame.height, 240);
        assert.ok(frame.data instanceof Uint16Array);
        assert.ok(new Set(frame.data).size > 256);
        assert.ok(frame.matrix.some(value => value !== 0));
        const baseline = develop(frame), brighter = develop(frame, { exposure: 1 });
        const at = (120 * 320 + 100) * 4;
        assert.ok(brighter.data[at] > baseline.data[at] + 25, 'Exposure changes linear RAW data');
        assert.ok(Math.max(...baseline.data.slice(at, at + 3)) - Math.min(...baseline.data.slice(at, at + 3)) < 5, 'A neutral sensor ramp stays neutral');
        const ten = develop(frame, {}, { bitDepth: 10 });
        assert.ok(ten.data instanceof Uint16Array); assert.equal(ten.data[3], 1023);
        const proxy = linearPreview(frame, 160);
        assert.equal(proxy.width, 160); assert.equal(proxy.height, 120);
        const proxyPixels = develop(proxy);
        assert.ok(Math.abs(proxyPixels.data[(60 * 160 + 50) * 4] - baseline.data[at]) < 4, 'Proxy and export agree');
    } finally { decoder.close(); }
});
test('orientation is applied by the decoder and corrupt input reports a recoverable error', async () => {
    const decoder = await createRawDecoder();
    try {
        assert.throws(() => decoder.open(new Uint8Array([1, 2, 3])), /unsupported|format|file|input\/output/i);
        decoder.open(makeDNG({ orientation: 6 })); const frame = decoder.decode();
        assert.equal(frame.width, 240); assert.equal(frame.height, 320);
    } finally { decoder.close(); }
});
test('development keeps source geometry and scales previews proportionally', () => {
    const frame = fromRGBA({ width: 300, height: 200, data: new Uint8ClampedArray(300 * 200 * 4).fill(180) });
    const adjusted = develop(frame, { exposure: 3 });
    const before = develop(frame, { exposure: 3 }, { before: true });
    assert.deepEqual([adjusted.width, adjusted.height], [300, 200]);
    assert.equal(before.data[0], 180);
    const preview = develop(frame, {}, { edge: 150 });
    assert.deepEqual([preview.width, preview.height], [150, 100]);
    const defaults = defaultRecipe(); assert.equal(develop(frame, defaults).data[0], 180);
    assert.equal('crop' in defaults, false);
});
