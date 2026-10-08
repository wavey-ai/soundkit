// The develop engine on WebGPU. The WebAssembly engine resolves a recipe into
// a plan (`engine.gpuPlan`); this module uploads it, runs the window
// operations the plan lists, then the per-pixel pass in develop.wgsl, and
// reads the result back in the shape `engine.develop` returns.
//
// `createGpuDeveloper` resolves to null where WebGPU is missing. A develop
// returns null when the picture does not fit the device's limits, and throws
// when the device fails; the caller develops on the CPU in both cases.

const OPS = { 1: 'prepare', 2: 'box_rows', 3: 'box_columns', 4: 'square', 5: 'guided', 6: 'copy_plane' };
const P = { RW: 4, RH: 5, WX0: 6, WY0: 7, WX1: 8, WY1: 9, PLANE_SIZE: 143 };
const STATS = 1026;
const UNIFORM_STRIDE = 256;

async function shaderCode() {
    const response = await fetch(new URL('./develop.wgsl', import.meta.url));
    if (!response.ok) throw new Error(`develop.wgsl: ${response.status}`);
    return response.text();
}

/// A GPU developer, or null without a hardware GPU. `options.code` supplies
/// the shader source instead of fetching it; `options.device` an existing
/// device. A software adapter, such as SwiftShader, develops no faster than
/// WebAssembly, so it is declined unless `options.allowSoftware` is set.
export async function createGpuDeveloper(options = {}) {
    const gpu = globalThis.navigator?.gpu;
    if (!gpu && !options.device) return null;
    let device = options.device;
    if (!device) {
        const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
        if (!adapter) return null;
        const software = adapter.info?.isFallbackAdapter || adapter.isFallbackAdapter || adapter.info?.architecture === 'swiftshader';
        if (software && !options.allowSoftware) return null;
        // Full-size pictures need more than the default 128 MB per binding.
        const want = Math.min(adapter.limits.maxStorageBufferBindingSize, adapter.limits.maxBufferSize, 2 ** 31 - 4);
        device = await adapter.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: want, maxBufferSize: want } });
    }
    let lost = false;
    device.lost?.then(() => { lost = true; });
    const module = device.createShaderModule({ code: options.code ?? await shaderCode() });
    const info = await module.getCompilationInfo?.();
    const errors = info?.messages?.filter(m => m.type === 'error') ?? [];
    if (errors.length) throw new Error(`develop.wgsl: ${errors.map(e => `${e.lineNum}:${e.linePos} ${e.message}`).join('; ')}`);
    const storage = type => ({ visibility: GPUShaderStage.COMPUTE, buffer: { type } });
    const layout = device.createBindGroupLayout({ entries: [
        { binding: 0, ...storage('read-only-storage') }, { binding: 1, ...storage('read-only-storage') }, { binding: 2, ...storage('read-only-storage') },
        { binding: 3, ...storage('storage') }, { binding: 4, ...storage('storage') }, { binding: 5, ...storage('storage') },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true } },
    ] });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const pipelines = {};
    for (const entryPoint of ['develop', ...Object.values(OPS)]) {
        pipelines[entryPoint] = await device.createComputePipelineAsync({ layout: pipelineLayout, compute: { module, entryPoint } });
    }
    const limit = device.limits.maxStorageBufferBindingSize;

    // Buffers grow and are reused, so a slider drag allocates nothing.
    const pool = new Map();
    function buffer(role, bytes, usage) {
        const size = Math.max(16, Math.ceil(bytes / 16) * 16);
        const held = pool.get(role);
        if (held && held.size >= size && held.usage === usage) return held;
        held?.destroy();
        const created = device.createBuffer({ size, usage });
        pool.set(role, created);
        return created;
    }
    // One source upload per frame; the last two frames stay resident.
    const sources = [];
    function sourceOf(engine) {
        const id = engine.id;
        const held = sources.find(s => s.id === id);
        if (held) return held.buffer;
        const words = engine.gpuSource();
        if (words.byteLength > limit) return null;
        const upload = device.createBuffer({ size: Math.max(16, Math.ceil(words.byteLength / 16) * 16), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(upload, 0, words);
        sources.unshift({ id, buffer: upload });
        while (sources.length > 2) sources.pop().buffer.destroy();
        return upload;
    }

    // Develops run one at a time: they share the buffers above.
    let queue = Promise.resolve();
    /// Develops `engine` with `recipe` and `options` as engine.develop does.
    function develop(engine, recipe, options = {}) {
        const run = queue.then(() => developNow(engine, recipe, options));
        queue = run.catch(() => {});
        return run;
    }
    async function developNow(engine, recipe, options) {
        if (lost) throw new Error('The GPU device was lost.');
        const plan = engine.gpuPlan(JSON.stringify(recipe), JSON.stringify(options));
        let params, tables, ops, width, height, bitDepth, planeCount;
        try {
            params = plan.params(); tables = plan.tables(); ops = plan.ops();
            ({ width, height, bitDepth } = plan); planeCount = plan.planes;
        } finally { plan.free(); }
        const planeSize = params[P.PLANE_SIZE];
        const pixels = width * height;
        const outBytes = pixels * (bitDepth === 8 ? 4 : 8);
        const planeBytes = Math.max(1, planeCount) * planeSize * 4;
        if (outBytes > limit || planeBytes > limit || tables.byteLength > limit) return null;
        const source = sourceOf(engine);
        if (!source) return null;

        const U = GPUBufferUsage;
        const paramBuffer = buffer('params', params.byteLength, U.STORAGE | U.COPY_DST);
        const tableBuffer = buffer('tables', Math.max(4, tables.byteLength), U.STORAGE | U.COPY_DST);
        const planeBuffer = buffer('planes', planeCount ? planeBytes : 16, U.STORAGE);
        const outBuffer = buffer('output', outBytes, U.STORAGE | U.COPY_SRC);
        const statsBuffer = buffer('stats', STATS * 4, U.STORAGE | U.COPY_SRC | U.COPY_DST);
        const passCount = ops.length / 5 + 1;
        const uniformBuffer = buffer('passes', passCount * UNIFORM_STRIDE, U.UNIFORM | U.COPY_DST);
        const outRead = buffer('outputRead', outBytes, U.MAP_READ | U.COPY_DST);
        const statsRead = buffer('statsRead', STATS * 4, U.MAP_READ | U.COPY_DST);

        device.queue.writeBuffer(paramBuffer, 0, params);
        if (tables.byteLength) device.queue.writeBuffer(tableBuffer, 0, tables);
        const uniforms = new Uint32Array(passCount * UNIFORM_STRIDE / 4);
        for (let k = 0; k < ops.length / 5; k++) uniforms.set(ops.subarray(k * 5, k * 5 + 4), k * UNIFORM_STRIDE / 4);
        device.queue.writeBuffer(uniformBuffer, 0, uniforms);

        const group = device.createBindGroup({ layout, entries: [
            { binding: 0, resource: { buffer: paramBuffer } }, { binding: 1, resource: { buffer: tableBuffer } },
            { binding: 2, resource: { buffer: source } }, { binding: 3, resource: { buffer: planeBuffer } },
            { binding: 4, resource: { buffer: outBuffer, size: outBytes } }, { binding: 5, resource: { buffer: statsBuffer, size: STATS * 4 } },
            { binding: 6, resource: { buffer: uniformBuffer, size: 16 } },
        ] });
        device.pushErrorScope('validation');
        device.pushErrorScope('out-of-memory');
        const encoder = device.createCommandEncoder();
        encoder.clearBuffer(statsBuffer, 0, STATS * 4);
        const pass = encoder.beginComputePass();
        const ww = params[P.WX1] - params[P.WX0], wh = params[P.WY1] - params[P.WY0];
        for (let k = 0; k < ops.length / 5; k++) {
            pass.setPipeline(pipelines[OPS[ops[k * 5]]]);
            pass.setBindGroup(0, group, [k * UNIFORM_STRIDE]);
            pass.dispatchWorkgroups(Math.ceil(ww / 8), Math.ceil(wh / 8));
        }
        pass.setPipeline(pipelines.develop);
        pass.setBindGroup(0, group, [(passCount - 1) * UNIFORM_STRIDE]);
        pass.dispatchWorkgroups(Math.ceil(params[P.RW] / 8), Math.ceil(params[P.RH] / 8));
        pass.end();
        encoder.copyBufferToBuffer(outBuffer, 0, outRead, 0, outBytes);
        encoder.copyBufferToBuffer(statsBuffer, 0, statsRead, 0, STATS * 4);
        device.queue.submit([encoder.finish()]);
        const memory = await device.popErrorScope();
        const validation = await device.popErrorScope();
        if (validation || memory) throw new Error(`GPU develop failed: ${(validation || memory).message}`);
        await Promise.all([outRead.mapAsync(GPUMapMode.READ, 0, outBytes), statsRead.mapAsync(GPUMapMode.READ, 0, STATS * 4)]);
        const bytes = outRead.getMappedRange(0, outBytes).slice(0);
        const stats = new Uint32Array(statsRead.getMappedRange(0, STATS * 4).slice(0));
        outRead.unmap(); statsRead.unmap();
        return {
            width, height, bitDepth,
            data: bitDepth === 8 ? new Uint8ClampedArray(bytes) : new Uint16Array(bytes),
            histogram: stats.slice(0, 256), histogramRGB: stats.slice(256, 1024),
            clippedHigh: stats[1024], clippedLow: stats[1025],
        };
    }

    function destroy() {
        for (const b of pool.values()) b.destroy();
        for (const s of sources) s.buffer.destroy();
        pool.clear(); sources.length = 0;
        if (!options.device) device.destroy();
    }
    return { develop, destroy, device, get lost() { return lost; } };
}
