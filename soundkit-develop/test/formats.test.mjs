// The engine's file and deep-pixel entry points, through the Node build in pkg-node.
import assert from 'node:assert/strict';
import test from 'node:test';
import { createRequire } from 'node:module';
import { readFileSync } from 'node:fs';
import { cardTIFF, pixel, opacity } from '../../soundkit-heif/test/card.mjs';
const { Engine } = createRequire(import.meta.url)('../pkg-node/soundkit_develop.js');

const size = { width: 96, height: 64 };
// The default recipe returns the picture, so the output shows what the engine holds.
function developed(engine, bitDepth = 8) {
    const result = engine.develop('{}', JSON.stringify({ bitDepth }));
    try { return bitDepth === 8 ? result.dataU8() : result.dataU16(); } finally { result.free(); }
}
test('an 8-bit TIFF loads as an 8-bit frame', () => {
    const engine = Engine.fromTiff(cardTIFF(size));
    assert.deepEqual([engine.width, engine.height, engine.bitDepth], [96, 64, 8]);
    const data = developed(engine);
    for (const [x, y] of [[0, 0], [95, 0], [10, 50], [95, 63]]) assert.deepEqual([...data.slice((y * 96 + x) * 4, (y * 96 + x) * 4 + 4)], [...pixel(x, y, 96, 64), 255]);
    engine.free();
});
test('a 16-bit TIFF loads as a 16-bit frame and develops to 10 bits', () => {
    const engine = Engine.fromTiff(cardTIFF({ ...size, bits: 16 }));
    assert.deepEqual([engine.width, engine.height, engine.bitDepth], [96, 64, 16]);
    const data = developed(engine, 10);
    assert.ok(data instanceof Uint16Array);
    for (let y = 0; y < 64; y += 7) for (let x = 0; x < 96; x += 5) pixel(x, y, 96, 64).forEach((value, c) =>
        assert.ok(Math.abs(data[(y * 96 + x) * 4 + c] - value * 1023 / 255) <= 1, `${x},${y}`));
    engine.free();
});
test('orientation and opacity of a TIFF are applied', () => {
    const turned = Engine.fromTiff(cardTIFF({ ...size, orientation: 6 }));
    assert.deepEqual([turned.width, turned.height], [64, 96]);
    // The stored top-left pixel shows at the top right.
    assert.deepEqual([...developed(turned).slice(63 * 4, 63 * 4 + 3)], pixel(0, 0, 96, 64));
    turned.free();
    const clear = Engine.fromTiff(cardTIFF({ ...size, alpha: true }));
    const data = developed(clear);
    for (const [x, y] of [[0, 0], [48, 32], [20, 12]]) assert.equal(data[(y * 96 + x) * 4 + 3], opacity(x, y, 96, 64), `opacity at ${x},${y}`);
    assert.ok(opacity(0, 0, 96, 64) < 10 && opacity(48, 32, 96, 64) === 255);
    clear.free();
});
test('a file that is not a TIFF gives an error', () => {
    assert.throws(() => Engine.fromTiff(new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8])), { message: 'This TIFF could not be read.' });
    assert.equal(Engine.fromTiff(cardTIFF(size)).width, 96);
});
test('10-bit RGBA loads as a 16-bit frame and returns its values', () => {
    const rgba = new Uint16Array(96 * 64 * 4);
    for (let y = 0; y < 64; y++) for (let x = 0; x < 96; x++) rgba.set([...pixel(x, y, 96, 64).map(value => value * 4 + (x & 3)), 1023], (y * 96 + x) * 4);
    const engine = Engine.fromRGBA16(96, 64, rgba, 10);
    assert.equal(engine.bitDepth, 16);
    const data = developed(engine, 10);
    for (let i = 0; i < data.length; i++) assert.ok(Math.abs(data[i] - rgba[i]) <= 1, `sample ${i}`);
    engine.free();
});
test('an ICC profile changes how RGBA values are read', () => {
    // The Display P3 profile of the soundkit-heif fixture, taken from its colour box.
    const file = readFileSync(new URL('../../soundkit-heif/test/fixtures/p3.heic', import.meta.url));
    const at = file.indexOf('colrprof') + 8, icc = file.subarray(at, at + file.readUInt32BE(at));
    const rgba = new Uint8ClampedArray([200, 60, 50, 255]);
    const plain = Engine.fromRGBA(1, 1, rgba), managed = Engine.fromRGBA(1, 1, rgba, icc), refused = Engine.fromRGBA(1, 1, rgba, new Uint8Array(200));
    assert.deepEqual([...developed(plain)], [200, 60, 50, 255]);
    assert.deepEqual([...developed(refused)], [200, 60, 50, 255]);
    // A red in Display P3 is a more saturated red in sRGB.
    const [r, g] = developed(managed);
    assert.ok(r > 205 && g < 50, `${r}, ${g}`);
    for (const engine of [plain, managed, refused]) engine.free();
});
