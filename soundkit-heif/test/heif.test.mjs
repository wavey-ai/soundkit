import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { createHeifDecoder, isHeif } from '../dist/index.mjs';
import { pixel, opacity } from './card.mjs';

const fixture = name => readFileSync(new URL(`./fixtures/${name}`, import.meta.url));
const decoder = await createHeifDecoder();

// The largest and the mean difference between a decoded picture and the card.
// `source` gives the card position of a decoded pixel. 4:2:0 chroma blends
// across the card's colour boundaries, so `inner` leaves those pixels out.
function compare(image, { source = (x, y) => [x, y], card = [image.width, image.height], inner = true } = {}) {
    const { width, height, data, bitDepth } = image, max = (1 << bitDepth) - 1;
    let worst = 0, sum = 0, count = 0;
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
        const [sx, sy] = source(x, y);
        if (inner && (Math.abs(sx - card[0] / 2 + .5) < 4 || Math.abs(sy - card[1] / 2 + .5) < 4)) continue;
        const want = pixel(sx, sy, card[0], card[1]);
        for (let c = 0; c < 3; c++) {
            const difference = Math.abs(data[(y * width + x) * 4 + c] * 255 / max - want[c]);
            worst = Math.max(worst, difference); sum += difference; count++;
        }
    }
    return { worst, mean: sum / count };
}

test('an 8-bit HEIC decodes to RGBA close to its source', () => {
    const bytes = fixture('card.heic');
    assert.equal(isHeif(bytes), true);
    const image = decoder.decode(bytes);
    assert.deepEqual([image.width, image.height, image.bitDepth], [96, 64, 8]);
    assert.ok(image.data instanceof Uint8ClampedArray);
    assert.equal(image.data.length, 96 * 64 * 4);
    assert.equal(image.hasAlpha, false); assert.equal(image.icc, null);
    for (let i = 3; i < image.data.length; i += 4) assert.equal(image.data[i], 255);
    const { worst, mean } = compare(image);
    assert.ok(worst <= 6 && mean < 1, `worst ${worst}, mean ${mean}`);
    // A few pixels by position: one inside each quadrant.
    for (const [x, y] of [[10, 10], [80, 12], [14, 50], [85, 55]]) {
        const got = [...image.data.slice((y * 96 + x) * 4, (y * 96 + x) * 4 + 3)], want = pixel(x, y, 96, 64);
        for (let c = 0; c < 3; c++) assert.ok(Math.abs(got[c] - want[c]) <= 4, `${x},${y}: ${got} for ${want}`);
    }
});
test('the outer pixels have the chroma of their own position', () => {
    // libheif 1.23.5 takes the chroma of the border pixels from another place; the build's patch corrects it.
    const image = decoder.decode(fixture('card.heic'));
    const { width, height, data } = image;
    let worst = 0;
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
        if (x > 0 && y > 0 && x < width - 1 && y < height - 1) continue;
        if (Math.abs(x - width / 2 + .5) < 4 || Math.abs(y - height / 2 + .5) < 4) continue;
        const want = pixel(x, y, width, height);
        for (let c = 0; c < 3; c++) worst = Math.max(worst, Math.abs(data[(y * width + x) * 4 + c] - want[c]));
    }
    assert.ok(worst <= 6, `worst border difference ${worst}`);
});
test('the rotation in the file is applied', () => {
    const image = decoder.decode(fixture('rotated.heic'));
    assert.deepEqual([image.width, image.height], [64, 96]);
    // The stored picture is 96 by 64 and turns a quarter clockwise for display.
    const { worst, mean } = compare(image, { source: (x, y) => [y, 63 - x], card: [96, 64] });
    assert.ok(worst <= 6 && mean < 1, `worst ${worst}, mean ${mean}`);
    // The stored top-left quadrant is red and shows at the top right.
    const at = (x, y) => [...image.data.slice((y * 64 + x) * 4, (y * 64 + x) * 4 + 3)];
    assert.ok(at(60, 4)[0] > at(60, 4)[2] + 50, `top right is red: ${at(60, 4)}`);
    assert.ok(at(4, 4)[2] > at(4, 4)[0] + 50, `top left is blue: ${at(4, 4)}`);
});
test('opacity decodes with the colours', () => {
    const image = decoder.decode(fixture('alpha.heic'));
    assert.equal(image.hasAlpha, true); assert.equal(image.premultiplied, false);
    let worst = 0;
    for (let y = 0; y < 64; y++) for (let x = 0; x < 96; x++) worst = Math.max(worst, Math.abs(image.data[(y * 96 + x) * 4 + 3] - opacity(x, y, 96, 64)));
    assert.ok(worst <= 6, `worst opacity difference ${worst}`);
    assert.ok(compare(image).worst <= 6);
});
test('a 10-bit HEIC decodes to 16-bit data at its 10-bit values', () => {
    const bytes = fixture('deep.heic');
    assert.equal(isHeif(bytes), true);
    const image = decoder.decode(bytes);
    assert.deepEqual([image.width, image.height, image.bitDepth], [96, 64, 10]);
    assert.ok(image.data instanceof Uint16Array);
    let largest = 0; for (const value of image.data) largest = Math.max(largest, value);
    assert.equal(largest, 1023, 'Opacity is the largest 10-bit value');
    assert.ok(new Set(image.data.filter((_, i) => i % 4 === 0)).size > 100, 'More red levels than a quadrant has in 8 bits');
    const { worst, mean } = compare(image);
    assert.ok(worst <= 4 && mean < 1, `worst ${worst}, mean ${mean}`);
});
test('a tiled HEIC decodes as one picture', () => {
    const image = decoder.decode(fixture('grid.heic'));
    assert.deepEqual([image.width, image.height, image.bitDepth], [1280, 960, 8]);
    const { worst, mean } = compare(image);
    assert.ok(worst <= 6 && mean < 1, `worst ${worst}, mean ${mean}`);
});
test('the ICC profile of the file is returned', () => {
    const image = decoder.decode(fixture('p3.heic'));
    assert.ok(image.icc instanceof Uint8Array && image.icc.length > 128);
    assert.equal(String.fromCharCode(...image.icc.slice(36, 40)), 'acsp');
    // The profile stores its name as UTF-16.
    assert.match(new TextDecoder('latin1').decode(image.icc).replaceAll('\0', ''), /Display P3/);
    // The pixel values are those of the file: the decoder does not convert them.
    assert.ok(compare(image).worst <= 6);
});
test('the HEVC loop filters are on', () => {
    // The reference is the macOS decoder's output. With the deblocking and
    // SAO filters off, the mean difference is 2.1 and the largest is 34.
    const image = decoder.decode(fixture('texture.heic')), reference = fixture('texture.rgb');
    assert.deepEqual([image.width, image.height], [64, 48]);
    let worst = 0, sum = 0, count = 0;
    for (let y = 0; y < 48; y++) for (let x = 0; x < 64; x++) {
        if (Math.abs(x - 31.5) < 4 || Math.abs(y - 23.5) < 4) continue;
        for (let c = 0; c < 3; c++) {
            const difference = Math.abs(image.data[(y * 64 + x) * 4 + c] - reference[(y * 64 + x) * 3 + c]);
            worst = Math.max(worst, difference); sum += difference; count++;
        }
    }
    assert.ok(worst <= 14 && sum / count < 1.5, `worst ${worst}, mean ${sum / count}`);
});
test('a file that is not HEIC gives an error and the decoder stays usable', () => {
    const bytes = fixture('card.heic');
    assert.equal(isHeif(new Uint8Array([1, 2, 3])), false);
    assert.equal(isHeif(new TextEncoder().encode('\0\0\0\x1cftypavif\0\0\0\0avifmif1miaf')), false);
    assert.equal(isHeif(new TextEncoder().encode('\0\0\0\x18ftypmif1\0\0\0\0mif1heic')), true);
    assert.throws(() => decoder.decode(new Uint8Array([1, 2, 3])), /Invalid input|could not be decoded/);
    assert.throws(() => decoder.decode(bytes.subarray(0, 600)), /Invalid input|end of file|could not be decoded/i);
    const damaged = Uint8Array.from(bytes); damaged.fill(0x55, 500, 760);
    try { decoder.decode(damaged); } catch (error) { assert.ok(error instanceof Error); }
    assert.equal(decoder.decode(bytes).width, 96);
});
test('a closed decoder refuses to decode', async () => {
    const other = await createHeifDecoder();
    other.close();
    assert.throws(() => other.decode(fixture('card.heic')), /closed/);
});
