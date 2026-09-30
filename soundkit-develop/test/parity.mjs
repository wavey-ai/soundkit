// The engine against the JavaScript reference in soundkit-raw, recipe by
// recipe, and both timed on a 12-megapixel frame.
import { createRequire } from 'node:module';
import { develop as reference, fromRGBA } from '../../soundkit-raw/src/develop.mjs';
const { Engine } = createRequire(import.meta.url)('../pkg-node/soundkit_develop.js');

function card(w, h) {
    const data = new Uint8ClampedArray(w * h * 4);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const i = (y * w + x) * 4, v = x / w, t = y / h;
        data[i] = 255 * v * (.55 + .45 * Math.sin(t * 6.28)); data[i + 1] = 255 * v * (.55 + .45 * Math.sin(t * 6.28 + 2.1));
        data[i + 2] = 255 * v * (.55 + .45 * Math.sin(t * 6.28 + 4.2)); data[i + 3] = 255;
    }
    return data;
}
const RECIPES = {
    default: {}, exposure: { exposure: 1.2 }, contrast: { contrast: 40 }, tone: { highlights: -50, shadows: 40, whites: 20, blacks: -20 },
    temperature: { temperature: 40, tint: -30 }, auto: { whiteBalance: 'auto' }, texture: { texture: 50 }, clarity: { clarity: 50 },
    dehaze: { dehaze: 40 }, haze: { dehaze: -40 }, vibrance: { vibrance: 60 }, saturation: { saturation: -40 }, curve: { curve: { darks: 40, highlights: -30 } },
    mixer: { mixer: { orange: { hue: 30, saturation: 40, luminance: 20 }, blue: { saturation: -50 } } }, mono: { profile: 'monochrome', mixer: { red: { luminance: 40 } } },
    vivid: { profile: 'vivid' }, grading: { grading: { shadows: { hue: 240, saturation: 40 }, highlights: { hue: 70, saturation: 30 }, balance: 20 } },
    detail: { sharpening: { amount: 80 }, noise: { luminance: 40, colour: 40 } }, vignette: { vignette: { amount: -50 } },
    skin: { highlights: -30, shadows: 25, texture: -15, mixer: { red: { hue: 20, saturation: -27 }, orange: { saturation: -10, luminance: 14 } } },
};
const w = 480, h = 320, rgba = card(w, h);
const frame = fromRGBA({ width: w, height: h, data: rgba }), engine = Engine.fromRGBA(w, h, rgba);
let worst = 0;
for (const [name, recipe] of Object.entries(RECIPES)) {
    const a = reference(frame, recipe), b = engine.develop(JSON.stringify(recipe), '{}'), bd = b.dataU8();
    let max = 0, sum = 0;
    for (let i = 0; i < a.data.length; i++) { if (i % 4 === 3) continue; const d = Math.abs(a.data[i] - bd[i]); max = Math.max(max, d); sum += d; }
    const mean = sum / (a.data.length * .75);
    worst = Math.max(worst, max);
    console.log(name.padEnd(12), 'max', String(max).padStart(3), ' mean', mean.toFixed(3), ' clipped', a.clippedHigh, b.clippedHigh, a.clippedLow, b.clippedLow);
    b.free();
}
console.log('worst channel difference', worst);

// 12 megapixels, as a phone takes them.
const W = 4032, H = 3024, big = card(W, H);
let t = performance.now();
const bigFrame = fromRGBA({ width: W, height: H, data: big });
const bigEngine = Engine.fromRGBA(W, H, big);
console.log('load 12MP   ', (performance.now() - t).toFixed(0), 'ms (JS float frame + engine)');
const look = { highlights: -30, shadows: 25, texture: -15, clarity: 10, vibrance: 12, mixer: { orange: { luminance: 14 } }, grading: { highlights: { hue: 70, saturation: 12 } } };
for (const [name, recipe] of [['untouched', {}], ['portrait look', look]]) {
    // CPU time, not wall time: the comparison holds on a busy machine.
    let c = process.cpuUsage(); reference(bigFrame, recipe); const js = process.cpuUsage(c).user / 1000;
    c = process.cpuUsage(); bigEngine.develop(JSON.stringify(recipe), '{}').free(); const wasm = process.cpuUsage(c).user / 1000;
    console.log(`12MP ${name.padEnd(14)} JS ${js.toFixed(0).padStart(5)} ms   WASM ${wasm.toFixed(0).padStart(5)} ms   ×${(js / wasm).toFixed(1)}`);
}
const preview = bigEngine.preview(1400);
const previewFrame = fromRGBA({ width: preview.width, height: preview.height, data: new Uint8ClampedArray(preview.width * preview.height * 4) });
let c = process.cpuUsage(); for (let i = 0; i < 5; i++) preview.develop(JSON.stringify(look), '{}').free();
const pw = process.cpuUsage(c).user / 5000;
c = process.cpuUsage(); for (let i = 0; i < 5; i++) reference(previewFrame, look);
const pj = process.cpuUsage(c).user / 5000;
console.log(`preview 1400 portrait look  JS ${pj.toFixed(0)} ms CPU   WASM ${pw.toFixed(0)} ms CPU`);
