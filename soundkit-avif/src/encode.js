import encodeBackend, { init as initBackend } from './backend-encode.js';
import { tagAvifSRGB } from './srgb.mjs';

let ready;
export function init(...args) {
    ready = initBackend(...args).catch(error => { ready = null; throw error; });
    return ready;
}

export default async function encode(pixels, options = {}) {
    const { width, height, data } = pixels;
    const bitDepth = options.bitDepth ?? 8;
    if (![8, 10, 12].includes(bitDepth)) throw new Error('AVIF bit depth must be 8, 10 or 12.');
    if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) || width < 1 || height < 1 || width * height > 80000000) {
        throw new Error('AVIF dimensions must be positive integers within the 80 megapixel limit.');
    }
    const validType = bitDepth === 8 ? data instanceof Uint8Array || data instanceof Uint8ClampedArray : data instanceof Uint16Array;
    if (!validType || data.length !== width * height * 4) throw new Error('AVIF requires a complete RGBA buffer matching its dimensions and bit depth.');
    // The backend reads the entire backing buffer. Respect caller-provided subviews.
    if (data.byteOffset || data.byteLength !== data.buffer.byteLength) {
        pixels = { width, height, data: bitDepth === 8 ? new Uint8Array(data) : new Uint16Array(data) };
    }
    await (ready || init());
    return encodeBackend(pixels, options);
}

export async function encodeSRGB(pixels, options = {}) {
    if ((options.bitDepth ?? 10) !== 10 || (options.subsample ?? 3) !== 3 || options.lossless) {
        throw new Error('Tagged sRGB AVIF requires 10-bit 4:4:4 lossy encoding.');
    }
    const bytes = await encode(pixels, { ...options, bitDepth: 10, subsample: 3 });
    return tagAvifSRGB(new Uint8Array(bytes));
}
