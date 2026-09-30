import assert from 'node:assert/strict';
import test from 'node:test';
import { createRawDecoder, develop, linearPreview, defaultRecipe, normalizeRecipe, fromRGBA, autoTone, hasDaylightReference } from '../dist/index.mjs';
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

/// A test card: a luminance ramp across, a hue sweep down, so every tone
/// and every colour band has pixels.
function card(width = 180, height = 120) {
    const data = new Uint8ClampedArray(width * height * 4);
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
        const v = x / (width - 1), h = y / height * 6, k = n => (n + h) % 6, f = n => 1 - Math.max(0, Math.min(k(n), 4 - k(n), 1));
        const i = (y * width + x) * 4;
        data[i] = 255 * v * (.4 + .6 * f(5)); data[i + 1] = 255 * v * (.4 + .6 * f(3)); data[i + 2] = 255 * v * (.4 + .6 * f(1)); data[i + 3] = 255;
    }
    return fromRGBA({ width, height, data });
}
const differs = (a, b) => a.data.some((value, i) => value !== b.data[i]);
test('version 1 recipes read as version 2 with every new control at rest', () => {
    const legacy = normalizeRecipe({ version: 1, exposure: .5, contrast: 20 });
    assert.equal(legacy.version, 2); assert.equal(legacy.exposure, .5); assert.equal(legacy.profile, 'standard');
    assert.equal(legacy.mixer.orange.saturation, 0); assert.equal(legacy.grading.blending, 50);
    const frame = card();
    assert.equal(differs(develop(frame, { exposure: .5, contrast: 20 }), develop(frame, legacy)), false);
});
test('each control changes the picture and leaves its alpha alone', () => {
    const frame = card(), baseline = develop(frame);
    const edits = {
        texture: { texture: 60 }, clarity: { clarity: 60 }, dehaze: { dehaze: 60 }, haze: { dehaze: -60 },
        vibrance: { vibrance: 60 }, saturation: { saturation: -60 }, curve: { curve: { darks: 50 } },
        mixer: { mixer: { blue: { hue: 50, saturation: 40, luminance: -40 } } },
        grading: { grading: { shadows: { hue: 210, saturation: 60 }, highlights: { hue: 40, saturation: 60 } } },
        sharpening: { sharpening: { amount: 120, radius: 1 } }, noise: { noise: { luminance: 60, colour: 60 } },
        vignette: { vignette: { amount: -60 } }, grain: { grain: { amount: 60 } },
        vivid: { profile: 'vivid' }, neutral: { profile: 'neutral' }, monochrome: { profile: 'monochrome' },
        auto: { whiteBalance: 'auto', temperature: 30 },
    };
    for (const [name, edit] of Object.entries(edits)) {
        const result = develop(frame, edit);
        assert.ok(differs(result, baseline), `${name} changes the picture`);
        assert.equal(result.data[3], 255, `${name} keeps alpha`);
        assert.equal(result.histogramRGB.length, 768);
    }
    const grey = develop(frame, { profile: 'monochrome' });
    for (let i = 0; i < grey.data.length; i += 4) assert.ok(grey.data[i] === grey.data[i + 1] && grey.data[i + 1] === grey.data[i + 2]);
});
test('vibrance holds skin back while it lifts other colours', () => {
    const swatch = (r, g, b) => fromRGBA({ width: 1, height: 1, data: new Uint8ClampedArray([r, g, b, 255]) });
    const spread = px => Math.max(...px.data.slice(0, 3)) - Math.min(...px.data.slice(0, 3));
    const skin = swatch(200, 150, 120), sky = swatch(120, 150, 200);
    const skinLift = spread(develop(skin, { vibrance: 80 })) - spread(develop(skin));
    const skyLift = spread(develop(sky, { vibrance: 80 })) - spread(develop(sky));
    assert.ok(skyLift > skinLift * 2, `sky +${skyLift}, skin +${skinLift}`);
});
test('the same grain lands on the same pixels every render', () => {
    const frame = card();
    assert.equal(differs(develop(frame, { grain: { amount: 50 } }), develop(frame, { grain: { amount: 50 } })), false);
});
test('auto tone lifts a dark picture and darkens a bright one', () => {
    const dark = fromRGBA({ width: 64, height: 64, data: new Uint8ClampedArray(64 * 64 * 4).map((_, i) => i % 4 === 3 ? 255 : 20 + (i % 97)) });
    const bright = fromRGBA({ width: 64, height: 64, data: new Uint8ClampedArray(64 * 64 * 4).map((_, i) => i % 4 === 3 ? 255 : 200 + (i % 50)) });
    assert.ok(autoTone(dark).exposure > .5);
    assert.ok(autoTone(bright).exposure < 0);
});
test('RAW white balance presets move the colour from as shot', async () => {
    const decoder = await createRawDecoder();
    try {
        decoder.open(makeDNG()); const frame = decoder.decode();
        assert.ok(hasDaylightReference(frame));
        assert.equal(hasDaylightReference(card()), false);
        const proxy = linearPreview(frame, 160);
        assert.ok(hasDaylightReference(proxy));
        const at = (60 * 160 + 80) * 4;
        const tungsten = develop(proxy, { whiteBalance: 'tungsten' }), shade = develop(proxy, { whiteBalance: 'shade' });
        assert.ok(tungsten.data[at + 2] - tungsten.data[at] > shade.data[at + 2] - shade.data[at], 'Tungsten is bluer than shade');
        const full = develop(frame, { whiteBalance: 'tungsten' }, { edge: 160 });
        assert.ok(Math.abs(full.data[at + 2] - tungsten.data[at + 2]) < 6, 'Proxy and export agree');
    } finally { decoder.close(); }
});
test('clipping counts only what the edit pushed to pure white or pure black', () => {
    const frame = card();
    const untouched = develop(frame, {}, { clipping: true });
    assert.equal(untouched.clippedHigh, 0); assert.equal(untouched.clippedLow, 0);
    assert.equal(differs(untouched, develop(frame)), false, 'Nothing is marked on an untouched picture');
    assert.ok(develop(frame, { exposure: 2 }).clippedHigh > 0);
    assert.ok(develop(frame, { exposure: -4, blacks: -100 }).clippedLow > 0);
    assert.equal(develop(frame, { exposure: 2 }, { before: true }).clippedHigh, 0);
});
