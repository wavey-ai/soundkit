// The GPU path against the WebAssembly engine, recipe by recipe, in Chromium.
// Runs on the machine's GPU and on SwiftShader, Chromium's software GPU.
// Build the web package first: npm run build:develop.
//
//   node soundkit-develop/test/gpu.browser.mjs            both adapters
//   ADAPTER=metal node soundkit-develop/test/gpu.browser.mjs
import { chromium } from 'playwright';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const files = {
    '/soundkit_develop.js': ['pkg/soundkit_develop.js', 'text/javascript'],
    '/soundkit_develop_bg.wasm': ['pkg/soundkit_develop_bg.wasm', 'application/wasm'],
    '/gpu/index.mjs': ['gpu/index.mjs', 'text/javascript'],
    '/gpu/develop.wgsl': ['gpu/develop.wgsl', 'text/plain'],
};
const ADAPTERS = {
    metal: ['--enable-unsafe-webgpu', '--use-angle=metal'],
    swiftshader: ['--enable-unsafe-webgpu', '--use-webgpu-adapter=swiftshader'],
};
const chosen = process.env.ADAPTER ? [process.env.ADAPTER] : Object.keys(ADAPTERS);

// Every recipe the parity check covers, and the frames each runs on.
const RECIPES = {
    default: {}, exposure: { exposure: 1.2 }, contrast: { contrast: 40 }, tone: { highlights: -50, shadows: 40, whites: 20, blacks: -20 },
    temperature: { temperature: 40, tint: -30 }, auto: { whiteBalance: 'auto' }, texture: { texture: 50 }, clarity: { clarity: 50 },
    dehaze: { dehaze: 40 }, haze: { dehaze: -40 }, vibrance: { vibrance: 60 }, saturation: { saturation: -40 }, curve: { curve: { darks: 40, highlights: -30 } },
    points: { curve: { points: [[0, 12], [70, 50], [190, 210], [255, 245]], red: [[0, 0], [128, 140], [255, 255]], blue: [[0, 10], [255, 230]] } },
    mixer: { mixer: { orange: { hue: 30, saturation: 40, luminance: 20 }, blue: { saturation: -50 } } }, mono: { profile: 'monochrome', mixer: { red: { luminance: 40 } } },
    vivid: { profile: 'vivid' }, hue: { hue: 120, vibrance: 20 }, grading: { grading: { shadows: { hue: 240, saturation: 40 }, highlights: { hue: 70, saturation: 30 }, balance: 20 } },
    detail: { sharpening: { amount: 80, masking: 30 }, noise: { luminance: 40, colour: 40 } }, vignette: { vignette: { amount: -50 } }, grain: { grain: { amount: 40, size: 40 } },
    skin: { highlights: -30, shadows: 25, texture: -15, mixer: { red: { hue: 20, saturation: -27 }, orange: { saturation: -10, luminance: 14 } } },
    everything: { exposure: .3, contrast: 20, highlights: -40, shadows: 30, whites: 10, blacks: -10, clarity: 25, texture: 20, dehaze: 15, vibrance: 25,
        curve: { points: [[0, 0], [128, 136], [255, 255]] }, grading: { midtones: { hue: 30, saturation: 15 } }, sharpening: { amount: 40 }, noise: { luminance: 20, colour: 20 },
        vignette: { amount: -20 }, grain: { amount: 15 } },
};

async function run(adapter) {
    const browser = await chromium.launch({ headless: true, args: ADAPTERS[adapter] });
    const page = await browser.newPage();
    await page.route('https://develop.test/**', route => {
        const path = new URL(route.request().url()).pathname;
        if (path === '/') return route.fulfill({ contentType: 'text/html', body: '<!doctype html><title>gpu</title>' });
        const file = files[path];
        if (!file) return route.fulfill({ status: 404, body: '' });
        return route.fulfill({ contentType: file[1], body: readFileSync(join(root, file[0])) });
    });
    page.on('console', message => { if (message.type() === 'error') console.error('page:', message.text()); });
    await page.goto('https://develop.test/');
    const report = await page.evaluate(async recipes => {
        const { default: init, Engine } = await import('/soundkit_develop.js');
        await init({ module_or_path: '/soundkit_develop_bg.wasm' });
        const { createGpuDeveloper } = await import('/gpu/index.mjs');
        const gpu = await createGpuDeveloper({ allowSoftware: true });
        if (!gpu) return { error: 'no WebGPU adapter' };
        const info = (await navigator.gpu.requestAdapter())?.info ?? {};

        const w = 480, h = 320;
        const card = new Uint8Array(w * h * 4);
        for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
            const i = (y * w + x) * 4, v = x / w, t = y / h;
            card[i] = 255 * v * (.55 + .45 * Math.sin(t * 6.28)); card[i + 1] = 255 * v * (.55 + .45 * Math.sin(t * 6.28 + 2.1));
            card[i + 2] = 255 * v * (.55 + .45 * Math.sin(t * 6.28 + 4.2)); card[i + 3] = 255;
        }
        // Camera samples with a clipped sun, 14-bit, through a camera matrix.
        const sensor = new Uint16Array(w * h * 3);
        for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
            const i = (y * w + x) * 3, v = (x / w) ** 2, t = y / h, sun = Math.max(0, 1 - Math.hypot(x - 360, y - 90) / 60) * 3;
            sensor[i] = Math.min(16383, 16383 * (v * .5 * (.6 + .4 * Math.sin(t * 9)) + sun));
            sensor[i + 1] = Math.min(16383, 16383 * (v * .8 + sun * .9));
            sensor[i + 2] = Math.min(16383, 16383 * (v * .35 * (.6 + .4 * Math.cos(t * 7)) + sun * .6));
        }
        const meta = JSON.stringify({ scale: 1 / 16383, wb: [2.1, 1, 1.6], daylight: [2.0, 1, 1.5],
            matrix: [1.73, -0.58, -0.15, -0.21, 1.48, -0.27, 0.02, -0.52, 1.5] });
        const frames = {
            photo: Engine.fromRGBA(w, h, card),
            raw: Engine.fromU16(w, h, sensor, meta),
        };
        frames.proxy = frames.raw.preview(360);
        const cases = [];
        for (const [frameName, engine] of Object.entries(frames)) {
            for (const [name, recipe] of Object.entries(recipes)) cases.push({ frameName, engine, name, recipe, options: {} });
            cases.push({ frameName, engine, name: 'region', recipe: recipes.everything, options: { region: [100, 60, 200, 120] } });
            cases.push({ frameName, engine, name: '10-bit', recipe: recipes.everything, options: { bitDepth: 10 } });
            cases.push({ frameName, engine, name: 'clipping', recipe: { exposure: 1.5, blacks: -60 }, options: { clipping: true } });
            cases.push({ frameName, engine, name: 'before', recipe: recipes.everything, options: { before: true, edge: 300 } });
        }
        const rows = [];
        for (const c of cases) {
            const cpu = c.engine.develop(JSON.stringify(c.recipe), JSON.stringify(c.options));
            const cpuData = cpu.bitDepth === 8 ? cpu.dataU8() : cpu.dataU16();
            const cpuHist = cpu.histogram(), cpuClip = [cpu.clippedHigh, cpu.clippedLow], size = [cpu.width, cpu.height], eight = cpu.bitDepth === 8;
            cpu.free();
            const g = await gpu.develop(c.engine, c.recipe, c.options);
            if (!g) { rows.push({ frame: c.frameName, recipe: c.name, error: 'did not fit' }); continue; }
            let max = 0, sum = 0, over = 0, worst = 0;
            for (let i = 0; i < cpuData.length; i++) {
                const d = Math.abs(cpuData[i] - g.data[i]);
                if (d > max) { max = d; worst = i; }
                sum += d; if (d > (eight ? 1 : 4)) over++;
            }
            const at = worst - worst % 4;
            const pixels = size[0] * size[1];
            let histogram = 0; for (let i = 0; i < 256; i++) histogram += Math.abs(cpuHist[i] - g.histogram[i]);
            rows.push({ frame: c.frameName, recipe: c.name, size: `${g.width}x${g.height}`, sameSize: g.width === size[0] && g.height === size[1],
                max, mean: sum / cpuData.length, over: over / cpuData.length, histogram: histogram / pixels,
                clipped: [cpuClip[0], g.clippedHigh, cpuClip[1], g.clippedLow],
                worst: `${(at / 4) % size[0]},${Math.floor(at / 4 / size[0])} cpu ${[...cpuData.slice(at, at + 3)]} gpu ${[...g.data.slice(at, at + 3)]}` });
        }
        // Speed on a 1400-pixel working copy, as the editor drags a slider.
        const big = new Uint8Array(1400 * 933 * 4);
        for (let i = 0; i < big.length; i++) big[i] = (i * 2654435761 >>> 24) & 255;
        const preview = Engine.fromRGBA(1400, 933, big);
        const look = recipes.everything;
        await gpu.develop(preview, look);
        let t = performance.now(); for (let k = 0; k < 5; k++) await gpu.develop(preview, { ...look, exposure: k / 10 });
        const gpuMs = (performance.now() - t) / 5;
        t = performance.now(); for (let k = 0; k < 3; k++) preview.develop(JSON.stringify({ ...look, exposure: k / 10 }), '{}').free();
        const cpuMs = (performance.now() - t) / 3;
        gpu.destroy();
        return { adapter: `${info.vendor} ${info.architecture}`, rows, gpuMs, cpuMs };
    }, RECIPES);
    await browser.close();
    return report;
}

let failed = false;
for (const adapter of chosen) {
    const report = await run(adapter);
    if (report.error) { console.log(adapter, report.error); failed = true; continue; }
    console.log(`\n${adapter}: ${report.adapter}`);
    for (const r of report.rows) {
        if (r.error) { console.log(r.frame, r.recipe, r.error); failed = true; continue; }
        // Rounding differs by a level now and then; anything more is a bug.
        const ok = r.sameSize && r.max <= 3 && r.mean < 0.05 && r.over < 0.002 && r.histogram < 0.01;
        if (!ok) failed = true;
        console.log(`${ok ? 'ok  ' : 'FAIL'} ${r.frame.padEnd(6)} ${r.recipe.padEnd(12)} ${r.size.padEnd(8)} max ${String(r.max).padStart(3)} mean ${r.mean.toFixed(4)} over ${(r.over * 100).toFixed(3)}% hist ${(r.histogram * 100).toFixed(3)}% clip ${r.clipped.join('/')}${ok ? '' : '  worst ' + r.worst}`);
    }
    console.log(`1400 px "everything" look: GPU ${report.gpuMs.toFixed(1)} ms, WebAssembly ${report.cpuMs.toFixed(1)} ms`);
}
process.exit(failed ? 1 : 0);
