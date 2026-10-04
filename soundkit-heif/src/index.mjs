import createModule from './heif.mjs';

const HEVC_BRANDS = new Set(['heic', 'heix', 'heim', 'heis', 'hevc', 'hevx', 'hevm', 'hevs']);
const AV1_BRANDS = new Set(['avif', 'avis']);
const GENERIC_BRANDS = new Set(['mif1', 'msf1']);

// Reads the file type box. A HEIC file names an HEVC brand. A file that names
// only the generic image brands is HEIF unless it also names an AVIF brand.
export function isHeif(bytes) {
    if (!bytes || bytes.length < 16) return false;
    const text = at => String.fromCharCode(bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]);
    if (text(4) !== 'ftyp') return false;
    const size = Math.min(bytes.length, ((bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3]) >>> 0);
    const brands = [text(8)];
    for (let at = 16; at + 4 <= size; at += 4) brands.push(text(at));
    if (brands.some(brand => HEVC_BRANDS.has(brand))) return true;
    return brands.some(brand => GENERIC_BRANDS.has(brand)) && !brands.some(brand => AV1_BRANDS.has(brand));
}

// One decoder per worker. Each decode releases its buffers before it returns.
export async function createHeifDecoder(options = {}) {
    let module = await createModule(options);
    return {
        decode(bytes) {
            if (!module) throw new Error('This decoder is closed.');
            const input = module._malloc(bytes.byteLength);
            if (!input) throw new Error('Not enough memory to open this photograph.');
            try {
                module.HEAPU8.set(bytes, input);
                if (!module._skheif_decode(input, bytes.byteLength)) throw new Error(module.UTF8ToString(module._skheif_error()) || 'This photograph could not be decoded.');
                const width = module._skheif_width(), height = module._skheif_height(), bitDepth = module._skheif_depth();
                const pointer = module._skheif_pixels(), stride = module._skheif_stride();
                const row = width * 4;
                let data;
                if (bitDepth === 8) {
                    data = new Uint8ClampedArray(row * height);
                    for (let y = 0; y < height; y++) data.set(module.HEAPU8.subarray(pointer + y * stride, pointer + y * stride + row), y * row);
                } else {
                    data = new Uint16Array(row * height);
                    for (let y = 0; y < height; y++) data.set(module.HEAPU16.subarray((pointer + y * stride) / 2, (pointer + y * stride) / 2 + row), y * row);
                }
                const profile = module._skheif_profile();
                const icc = profile ? module.HEAPU8.slice(profile, profile + module._skheif_profile_size()) : null;
                return { width, height, data, bitDepth, icc, ...JSON.parse(module.UTF8ToString(module._skheif_metadata())) };
            } finally { module._skheif_close(); module._free(input); }
        },
        // Drops the reference to the WebAssembly heap so that it can be collected.
        close() { module = null; },
    };
}
