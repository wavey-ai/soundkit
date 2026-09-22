import createModule from './raw.mjs';
export { defaultRecipe, normalizeRecipe, develop, linearPreview, fromRGBA, cropGeometry } from './develop.mjs';

// One decoder per worker. The input stays alive until LibRaw releases its buffer stream.
export async function createRawDecoder(options = {}) {
    const module = await createModule(options);
    let input = 0;
    const close = () => { module._raw_close(); if (input) module._free(input); input = 0; };
    const checked = ok => { if (!ok) throw new Error(module.UTF8ToString(module._raw_error()) || 'This RAW could not be read.'); };
    const metadata = () => JSON.parse(module.UTF8ToString(module._raw_metadata()));
    return {
        open(bytes) {
            close();
            input = module._malloc(bytes.byteLength);
            if (!input) throw new Error('Not enough memory to open this RAW.');
            module.HEAPU8.set(bytes, input);
            try { checked(module._raw_open(input, bytes.byteLength)); return metadata(); }
            catch (error) { close(); throw error; }
        },
        thumbnail() {
            const pointer = module._raw_thumbnail();
            return pointer ? module.HEAPU8.slice(pointer, pointer + module._raw_thumbnail_size()) : null;
        },
        decode({ half = false } = {}) {
            checked(module._raw_decode(half ? 1 : 0));
            const meta = metadata();
            const pointer = module._raw_pixels();
            const data = module.HEAPU16.slice(pointer / 2, (pointer + module._raw_size()) / 2);
            const green = meta.wb[1] || 1;
            return { width: module._raw_width(), height: module._raw_height(), data,
                scale: 1 / Math.max(1, meta.maximum), wb: meta.wb.slice(0, 3).map(value => (value || green) / green),
                matrix: meta.matrix, metadata: meta, linear: true };
        },
        close,
    };
}
