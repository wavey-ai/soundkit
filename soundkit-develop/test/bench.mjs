// The WebAssembly engine's CPU time on a 12-megapixel frame and on a
// 1400-pixel working copy, through the Node build in pkg-node.
import { createRequire } from 'node:module';
const { Engine } = createRequire(import.meta.url)('../pkg-node/soundkit_develop.js');

function card(w, h) {
    const data = new Uint8Array(w * h * 4);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const i = (y * w + x) * 4, v = x / w, t = y / h;
        data[i] = 255 * v * (.55 + .45 * Math.sin(t * 6.28)); data[i + 1] = 255 * v * (.55 + .45 * Math.sin(t * 6.28 + 2.1));
        data[i + 2] = 255 * v * (.55 + .45 * Math.sin(t * 6.28 + 4.2)); data[i + 3] = 255;
    }
    return data;
}
const look = { highlights: -30, shadows: 25, texture: -15, clarity: 10, vibrance: 12, mixer: { orange: { luminance: 14 } }, grading: { highlights: { hue: 70, saturation: 12 } } };
const cpu = run => { const start = process.cpuUsage(); run(); return process.cpuUsage(start).user / 1000; };
const big = Engine.fromRGBA(4032, 3024, card(4032, 3024));
for (const [name, recipe] of [['untouched', {}], ['portrait look', look]]) {
    console.log(`12 MP ${name.padEnd(14)} ${cpu(() => big.develop(JSON.stringify(recipe), '{}').free()).toFixed(0).padStart(6)} ms CPU`);
}
const preview = big.preview(1400);
let total = 0;
for (let i = 0; i < 5; i++) total += cpu(() => preview.develop(JSON.stringify({ ...look, exposure: i / 10 }), '{}').free());
console.log(`1400 px portrait look  ${(total / 5).toFixed(0).padStart(6)} ms CPU`);
