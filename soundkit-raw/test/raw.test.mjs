import assert from 'node:assert/strict';
import test from 'node:test';
import { createRawDecoder, develop, linearPreview, defaultRecipe, normalizeRecipe, fromRGBA, autoTone, hasDaylightReference, whiteOf, COLOUR_BAND_HUES } from '../dist/index.mjs';
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

// OKLab of an 8-bit sRGB pixel, for checking what the colour controls keep.
const lin = v => { v /= 255; return v <= .04045 ? v / 12.92 : ((v + .055) / 1.055) ** 2.4; };
function oklab(r, g, b) {
    [r, g, b] = [lin(r), lin(g), lin(b)];
    const l = Math.cbrt(.4122214708 * r + .5363325363 * g + .0514459929 * b), m = Math.cbrt(.2119034982 * r + .6806995451 * g + .1073969566 * b), s = Math.cbrt(.0883024619 * r + .2817188376 * g + .6299787005 * b);
    const L = .2104542553 * l + .7936177850 * m - .0040720468 * s, a = 1.9779984951 * l - 2.4285922050 * m + .4505937099 * s, bb = .0259040371 * l + .7827717662 * m - .8086757660 * s;
    return { L, C: Math.hypot(a, bb), h: (Math.atan2(bb, a) * 180 / Math.PI + 360) % 360 };
}
const swatch = (r, g, b) => fromRGBA({ width: 1, height: 1, data: new Uint8ClampedArray([r, g, b, 255]) });
const pixel = result => [...result.data.slice(0, 3)];
const hueGap = (a, b) => Math.abs(((a - b + 540) % 360) - 180);
test('the white point maths lands on D65 and every band hue is measured in OKLCh', () => {
    const [X, , Z] = whiteOf(6504, .0032), x = X / (X + 1 + Z), y = 1 / (X + 1 + Z);
    assert.ok(Math.abs(x - .3127) < .002 && Math.abs(y - .3290) < .002, `D65 at ${x.toFixed(4)}, ${y.toFixed(4)}`);
    assert.equal(COLOUR_BAND_HUES.length, 8);
    for (let i = 1; i < 8; i++) assert.ok(COLOUR_BAND_HUES[i] > COLOUR_BAND_HUES[i - 1], 'Band hues run round the circle in order');
});
test('temperature and tint adapt the light: greys warm, cool and turn magenta', () => {
    const grey = swatch(128, 128, 128);
    const [wr, , wb] = pixel(develop(grey, { temperature: 50 })), [cr, , cb] = pixel(develop(grey, { temperature: -50 }));
    assert.ok(wr > wb + 8, `warmer ${wr}/${wb}`); assert.ok(cb > cr + 8, `cooler ${cr}/${cb}`);
    const [mr, mg, mb] = pixel(develop(grey, { tint: 60 }));
    assert.ok(mg < mr - 4 && mg < mb - 4, `magenta ${mr}/${mg}/${mb}`);
    assert.deepEqual(pixel(develop(grey, { temperature: 0, tint: 0 })), [128, 128, 128]);
});
test('a band hue shift turns the colour and keeps its lightness', () => {
    const source = oklab(40, 90, 200);
    const shifted = oklab(...pixel(develop(swatch(40, 90, 200), { mixer: { blue: { hue: 80 } } })));
    assert.ok(hueGap(shifted.h, source.h) > 6, `hue ${source.h.toFixed(1)} -> ${shifted.h.toFixed(1)}`);
    assert.ok(Math.abs(shifted.L - source.L) < .01, `lightness ${source.L.toFixed(3)} -> ${shifted.L.toFixed(3)}`);
});
test('saturation at -100 leaves a grey of the same lightness', () => {
    const source = oklab(200, 120, 60), [r, g, b] = pixel(develop(swatch(200, 120, 60), { saturation: -100 }));
    assert.ok(Math.max(r, g, b) - Math.min(r, g, b) <= 1, `${r}/${g}/${b}`);
    assert.ok(Math.abs(oklab(r, g, b).L - source.L) < .01);
});
test('a colour pushed past the screen keeps its hue', () => {
    for (const colour of [[30, 60, 230], [230, 40, 60], [40, 200, 90]]) {
        const source = oklab(...colour), result = oklab(...pixel(develop(swatch(...colour), { saturation: 100, vibrance: 100 })));
        assert.ok(hueGap(result.h, source.h) < 3, `${colour}: hue ${source.h.toFixed(1)} -> ${result.h.toFixed(1)}`);
        assert.ok(result.C >= source.C - .005, 'and is no less vivid');
    }
});
test('a highlight pushed past white keeps its hue on the way to white', () => {
    const source = oklab(230, 140, 60), result = oklab(...pixel(develop(swatch(230, 140, 60), { exposure: 1.2 })));
    assert.ok(hueGap(result.h, source.h) < 10, `hue ${source.h.toFixed(1)} -> ${result.h.toFixed(1)}`);
});
