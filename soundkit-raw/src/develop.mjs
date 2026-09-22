const IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0, 1];
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const finite = (x, fallback = 0) => Number.isFinite(Number(x)) ? Number(x) : fallback;
export const defaultRecipe = () => ({ version: 1, exposure: 0, temperature: 0, tint: 0, highlights: 0,
    shadows: 0, whites: 0, blacks: 0, contrast: 0, rotation: 0, straighten: 0,
    crop: { ratio: 0, zoom: 1, x: 0, y: 0 } });
export function normalizeRecipe(value = {}) {
    const r = defaultRecipe();
    r.exposure = clamp(finite(value.exposure), -5, 5);
    for (const key of ['temperature', 'tint', 'highlights', 'shadows', 'whites', 'blacks', 'contrast']) r[key] = clamp(finite(value[key]), -100, 100);
    r.rotation = ((Math.round(finite(value.rotation) / 90) % 4) + 4) % 4 * 90;
    r.straighten = clamp(finite(value.straighten), -15, 15);
    r.crop = { ratio: clamp(finite(value.crop?.ratio), 0, 4), zoom: clamp(finite(value.crop?.zoom, 1), 1, 4),
        x: clamp(finite(value.crop?.x), -1, 1), y: clamp(finite(value.crop?.y), -1, 1) };
    return r;
}
const toLinear = value => value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
const toSRGB = value => value <= 0.0031308 ? value * 12.92 : 1.055 * value ** (1 / 2.4) - 0.055;
export function fromRGBA({ width, height, data }) {
    const pixels = width * height, out = new Float32Array(pixels * 3), alpha = new Uint8Array(pixels);
    const ramp = Float32Array.from({ length: 256 }, (_, i) => toLinear(i / 255));
    for (let i = 0; i < pixels; i++) {
        for (let c = 0; c < 3; c++) out[i * 3 + c] = ramp[data[i * 4 + c]];
        alpha[i] = data[i * 4 + 3];
    }
    return { data: out, alpha, width, height, scale: 1, wb: [1, 1, 1], matrix: IDENTITY, linear: true };
}
function working(frame, offset, result) {
    const m = frame.matrix?.some(v => v !== 0) ? frame.matrix : IDENTITY;
    const wb = frame.wb || [1, 1, 1], scale = frame.scale ?? 1;
    const r = frame.data[offset] * scale * wb[0], g = frame.data[offset + 1] * scale * wb[1], b = frame.data[offset + 2] * scale * wb[2];
    result[0] = m[0] * r + m[1] * g + m[2] * b;
    result[1] = m[3] * r + m[4] * g + m[5] * b;
    result[2] = m[6] * r + m[7] * g + m[8] * b;
}
// Box-filter the linear sensor image once. Every edit reuses this small working image.
export function linearPreview(frame, edge = 1400) {
    const factor = Math.min(1, edge / Math.max(frame.width, frame.height));
    const width = Math.max(1, Math.round(frame.width * factor)), height = Math.max(1, Math.round(frame.height * factor));
    const data = new Float32Array(width * height * 3), alpha = frame.alpha ? new Uint8Array(width * height) : null;
    const rgb = [0, 0, 0];
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
        const left = Math.floor(x * frame.width / width), right = Math.max(left + 1, Math.floor((x + 1) * frame.width / width));
        const top = Math.floor(y * frame.height / height), bottom = Math.max(top + 1, Math.floor((y + 1) * frame.height / height));
        const dest = (y * width + x) * 3; let count = 0, opacity = 0;
        for (let sy = top; sy < bottom; sy++) for (let sx = left; sx < right; sx++) {
            const pixel = sy * frame.width + sx;
            working(frame, pixel * 3, rgb);
            for (let c = 0; c < 3; c++) data[dest + c] += rgb[c];
            if (alpha) opacity += frame.alpha[pixel];
            count++;
        }
        for (let c = 0; c < 3; c++) data[dest + c] /= count;
        if (alpha) alpha[y * width + x] = Math.round(opacity / count);
    }
    return { width, height, data, alpha, scale: 1, matrix: IDENTITY, wb: [1, 1, 1], linear: true };
}
export function cropGeometry(frame, recipe = {}, edge = 0) {
    const r = normalizeRecipe(recipe), swapped = r.rotation % 180 !== 0;
    const sourceWidth = swapped ? frame.height : frame.width, sourceHeight = swapped ? frame.width : frame.height;
    const aspect = r.crop.ratio || sourceWidth / sourceHeight;
    const radians = r.straighten * Math.PI / 180, cos = Math.cos(radians), sin = Math.sin(radians);
    // An inscribed rectangle: straightening never introduces empty corners or stretches pixels.
    const safe = 1 / (Math.abs(cos) + Math.abs(sin) * Math.max(sourceWidth / sourceHeight, sourceHeight / sourceWidth));
    let cropWidth = sourceWidth * safe, cropHeight = sourceHeight * safe;
    if (cropWidth / cropHeight > aspect) cropWidth = cropHeight * aspect;
    else cropHeight = cropWidth / aspect;
    cropWidth /= r.crop.zoom; cropHeight /= r.crop.zoom;
    const cx = sourceWidth / 2 + r.crop.x * (sourceWidth * safe - cropWidth) / 2;
    const cy = sourceHeight / 2 + r.crop.y * (sourceHeight * safe - cropHeight) / 2;
    const scale = edge > 0 ? Math.min(1, edge / Math.max(cropWidth, cropHeight)) : 1;
    return { width: Math.max(1, Math.round(cropWidth * scale)), height: Math.max(1, Math.round(cropHeight * scale)),
        cropWidth, cropHeight, cx, cy, sourceWidth, sourceHeight, cos, sin, rotation: r.rotation };
}
export function develop(frame, recipe = {}, { edge = 0, bitDepth = 8, before = false, clipping = false } = {}) {
    if (![8, 10, 12].includes(bitDepth)) throw new Error('Output must be 8, 10 or 12 bit.');
    const r = normalizeRecipe(recipe), g = cropGeometry(frame, r, edge);
    const settings = before ? defaultRecipe() : r;
    const { width, height } = g, max = (1 << bitDepth) - 1;
    const data = bitDepth === 8 ? new Uint8ClampedArray(width * height * 4) : new Uint16Array(width * height * 4);
    const histogram = new Uint32Array(256);
    const exposure = 2 ** settings.exposure;
    const warm = 2 ** (settings.temperature / 200), tint = 2 ** (settings.tint / 300);
    const gains = [warm * tint, 1 / tint, tint / warm];
    const contrast = 2 ** (settings.contrast / 150);
    const black = settings.blacks / 1000, white = 2 ** (settings.whites / 200);
    // All expensive curves are sampled once per edit, not once per channel.
    // The display LUT has 16-bit linear resolution before 8/10/12-bit output.
    const clipAt = .18 * (1 / .18) ** (1 / contrast);
    const displayScale = 65535 / clipAt;
    const display = new Uint16Array(65536);
    for (let i = 0; i < display.length; i++) display[i] = Math.round(clamp(toSRGB(.18 * (i / displayScale / .18) ** contrast), 0, 1) * max);
    const tones = new Float32Array(16385);
    for (let i = 0; i < tones.length; i++) {
        const luminance = i / 2048;
        tones[i] = 2 ** ((settings.shadows * Math.exp(-luminance * 6) + settings.highlights * (1 - Math.exp(-luminance * 1.5))) / 100);
    }
    let clippedHigh = 0, clippedLow = 0;
    const rgb = [0, 0, 0];
    const matrix = frame.matrix?.some(v => v !== 0) ? frame.matrix : IDENTITY;
    const wb = frame.wb || [1, 1, 1], scale = frame.scale ?? 1;
    const rr = scale * wb[0], gg = scale * wb[1], bb = scale * wb[2];
    const sensor = frame.data;
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
        const ox = g.cx + ((x + .5) / width - .5) * g.cropWidth - g.sourceWidth / 2;
        const oy = g.cy + ((y + .5) / height - .5) * g.cropHeight - g.sourceHeight / 2;
        const rx = ox * g.cos - oy * g.sin + g.sourceWidth / 2 - .5;
        const ry = ox * g.sin + oy * g.cos + g.sourceHeight / 2 - .5;
        let sx, sy;
        if (g.rotation === 90) { sx = ry; sy = frame.height - 1 - rx; }
        else if (g.rotation === 180) { sx = frame.width - 1 - rx; sy = frame.height - 1 - ry; }
        else if (g.rotation === 270) { sx = frame.width - 1 - ry; sy = rx; }
        else { sx = rx; sy = ry; }
        sx = clamp(sx, 0, frame.width - 1); sy = clamp(sy, 0, frame.height - 1);
        const x0 = Math.floor(sx), y0 = Math.floor(sy), x1 = Math.min(x0 + 1, frame.width - 1), y1 = Math.min(y0 + 1, frame.height - 1);
        const dx = sx - x0, dy = sy - y0;
        const p00 = y0 * frame.width + x0, p10 = y0 * frame.width + x1;
        const p01 = y1 * frame.width + x0, p11 = y1 * frame.width + x1;
        const w00 = (1 - dx) * (1 - dy), w10 = dx * (1 - dy), w01 = (1 - dx) * dy, w11 = dx * dy;
        const cr = (sensor[p00 * 3] * w00 + sensor[p10 * 3] * w10 + sensor[p01 * 3] * w01 + sensor[p11 * 3] * w11) * rr;
        const cg = (sensor[p00 * 3 + 1] * w00 + sensor[p10 * 3 + 1] * w10 + sensor[p01 * 3 + 1] * w01 + sensor[p11 * 3 + 1] * w11) * gg;
        const cb = (sensor[p00 * 3 + 2] * w00 + sensor[p10 * 3 + 2] * w10 + sensor[p01 * 3 + 2] * w01 + sensor[p11 * 3 + 2] * w11) * bb;
        rgb[0] = matrix[0] * cr + matrix[1] * cg + matrix[2] * cb;
        rgb[1] = matrix[3] * cr + matrix[4] * cg + matrix[5] * cb;
        rgb[2] = matrix[6] * cr + matrix[7] * cg + matrix[8] * cb;
        const a = frame.alpha;
        const alpha = a ? (a[p00] * w00 + a[p10] * w10 + a[p01] * w01 + a[p11] * w11) / 255 : 1;
        for (let c = 0; c < 3; c++) rgb[c] = Math.max(0, rgb[c] * gains[c] * exposure);
        const luminance = .2126 * rgb[0] + .7152 * rgb[1] + .0722 * rgb[2];
        const tone = tones[Math.min(16384, Math.round(luminance * 2048))];
        const dest = (y * width + x) * 4;
        let hi = false, lo = false;
        for (let c = 0; c < 3; c++) {
            const value = Math.max(0, (rgb[c] * tone + black) * white);
            hi ||= value > clipAt; lo ||= value <= 0;
            data[dest + c] = display[Math.min(65535, Math.round(value * displayScale))];
        }
        histogram[clamp(Math.round((.2126 * data[dest] + .7152 * data[dest + 1] + .0722 * data[dest + 2]) * 255 / max), 0, 255)]++;
        if (hi) clippedHigh++; if (lo) clippedLow++;
        if (clipping && hi) { data[dest] = max; data[dest + 1] = 0; data[dest + 2] = 0; }
        else if (clipping && lo) { data[dest] = 0; data[dest + 1] = 0; data[dest + 2] = max; }
        data[dest + 3] = Math.round(alpha * max);
    }
    return { width, height, data, bitDepth, histogram, clippedHigh, clippedLow };
}
