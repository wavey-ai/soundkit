import assert from 'node:assert/strict';
import test from 'node:test';
import { createRawDecoder, develop, linearPreview, cropGeometry, defaultRecipe, fromRGBA } from '../dist/index.mjs';
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
test('crop and rotation change geometry without stretching; before retains the crop', () => {
    const frame = fromRGBA({ width: 300, height: 200, data: new Uint8ClampedArray(300 * 200 * 4).fill(180) });
    assert.deepEqual([cropGeometry(frame).width, cropGeometry(frame).height], [300, 200]);
    const square = develop(frame, { exposure: 3, crop: { ratio: 1 } }, { before: true });
    assert.equal(square.width, 200); assert.equal(square.height, 200); assert.equal(square.data[0], 180);
    const rotated = cropGeometry(frame, { rotation: 90 }); assert.equal(rotated.width, 200); assert.equal(rotated.height, 300);
    const straightened = cropGeometry(frame, { straighten: 10 });
    assert.ok(straightened.width < 300); assert.ok(Math.abs(straightened.width / straightened.height - 1.5) < .01);
    const defaults = defaultRecipe(); assert.equal(develop(frame, defaults).data[0], 180);
});
