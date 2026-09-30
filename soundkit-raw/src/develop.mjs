const IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0, 1];
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const finite = (x, fallback = 0) => Number.isFinite(Number(x)) ? Number(x) : fallback;
const smooth = (a, b, x) => { const t = clamp((x - a) / (b - a || 1e-6), 0, 1); return t * t * (3 - 2 * t); };

/// The eight colour bands of the colour mixer, by hue in degrees. Skin sits
/// between red and orange, mostly in orange.
export const COLOUR_BANDS = Object.freeze(['red', 'orange', 'yellow', 'green', 'aqua', 'blue', 'purple', 'magenta']);
const BAND_HUES = [0, 30, 60, 120, 180, 240, 280, 320];
/// White balance choices. `shot` and `auto` apply to every image; the others
/// are measured from a RAW file's daylight reference.
export const WHITE_BALANCES = Object.freeze(['shot', 'auto', 'daylight', 'cloudy', 'shade', 'tungsten', 'fluorescent', 'flash']);
export const RAW_WHITE_BALANCES = Object.freeze(['daylight', 'cloudy', 'shade', 'tungsten', 'fluorescent', 'flash']);
export const PROFILES = Object.freeze(['standard', 'vivid', 'neutral', 'monochrome']);
// Illuminant temperature and green correction for each preset.
const WB_LIGHT = { daylight: [5500, 1], cloudy: [6500, 1], shade: [7500, 1], tungsten: [3200, 1], fluorescent: [4000, .93], flash: [5600, 1] };
const GRADING_ZONES = ['shadows', 'midtones', 'highlights', 'global'];

const zone = () => ({ hue: 0, saturation: 0, luminance: 0 });
export const defaultRecipe = () => ({
    version: 2, profile: 'standard', whiteBalance: 'shot',
    exposure: 0, temperature: 0, tint: 0, highlights: 0, shadows: 0, whites: 0, blacks: 0, contrast: 0,
    texture: 0, clarity: 0, dehaze: 0, vibrance: 0, saturation: 0,
    curve: { highlights: 0, lights: 0, darks: 0, shadows: 0 },
    mixer: Object.fromEntries(COLOUR_BANDS.map(band => [band, { hue: 0, saturation: 0, luminance: 0 }])),
    grading: { shadows: zone(), midtones: zone(), highlights: zone(), global: zone(), blending: 50, balance: 0 },
    sharpening: { amount: 0, radius: 1, masking: 0 },
    noise: { luminance: 0, colour: 0 },
    vignette: { amount: 0, midpoint: 50, feather: 50 },
    grain: { amount: 0, size: 25 },
});
const signed = value => clamp(finite(value), -100, 100);
const unit = (value, fallback = 0) => clamp(finite(value, fallback), 0, 100);
export function normalizeRecipe(value = {}) {
    const r = defaultRecipe();
    r.profile = PROFILES.includes(value.profile) ? value.profile : 'standard';
    r.whiteBalance = WHITE_BALANCES.includes(value.whiteBalance) ? value.whiteBalance : 'shot';
    r.exposure = clamp(finite(value.exposure), -5, 5);
    for (const key of ['temperature', 'tint', 'highlights', 'shadows', 'whites', 'blacks', 'contrast',
        'texture', 'clarity', 'dehaze', 'vibrance', 'saturation']) r[key] = signed(value[key]);
    for (const key of Object.keys(r.curve)) r.curve[key] = signed(value.curve?.[key]);
    for (const band of COLOUR_BANDS) for (const key of ['hue', 'saturation', 'luminance']) r.mixer[band][key] = signed(value.mixer?.[band]?.[key]);
    for (const name of GRADING_ZONES) {
        const source = value.grading?.[name] ?? {};
        r.grading[name] = { hue: ((finite(source.hue) % 360) + 360) % 360, saturation: unit(source.saturation), luminance: signed(source.luminance) };
    }
    r.grading.blending = unit(value.grading?.blending, 50);
    r.grading.balance = signed(value.grading?.balance);
    r.sharpening = { amount: clamp(finite(value.sharpening?.amount), 0, 150), radius: clamp(finite(value.sharpening?.radius, 1), .5, 3), masking: unit(value.sharpening?.masking) };
    r.noise = { luminance: unit(value.noise?.luminance), colour: unit(value.noise?.colour) };
    r.vignette = { amount: signed(value.vignette?.amount), midpoint: unit(value.vignette?.midpoint, 50), feather: unit(value.vignette?.feather, 50) };
    r.grain = { amount: unit(value.grain?.amount), size: unit(value.grain?.size, 25) };
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
/// The camera facts a frame was developed from. A proxy has its camera's
/// white balance baked in and keeps the facts to change it later.
const cameraOf = frame => frame.camera ?? { matrix: frame.matrix?.some(v => v !== 0) ? frame.matrix : IDENTITY, wb: frame.wb || [1, 1, 1], daylight: frame.daylight || null };
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
    return { width, height, data, alpha, scale: 1, matrix: IDENTITY, wb: [1, 1, 1], linear: true,
        camera: cameraOf(frame), sourceWidth: frame.sourceWidth ?? frame.width };
}

// ---------------------------------------------------------------- colour math

const multiply = (a, b) => Array.from({ length: 9 }, (_, i) => {
    const row = Math.floor(i / 3), col = i % 3;
    return a[row * 3] * b[col] + a[row * 3 + 1] * b[3 + col] + a[row * 3 + 2] * b[6 + col];
});
function invert(m) {
    const [a, b, c, d, e, f, g, h, i] = m;
    const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
    const det = a * A + b * B + c * C;
    if (Math.abs(det) < 1e-9) return null;
    return [A / det, -(b * i - c * h) / det, (b * f - c * e) / det,
        B / det, (a * i - c * g) / det, -(a * f - c * d) / det,
        C / det, -(a * h - b * g) / det, (a * e - b * d) / det];
}
const diagonal = ([r, g, b]) => [r, 0, 0, 0, g, 0, 0, 0, b];
/// Linear sRGB of a black body at `kelvin`, green normalised to one.
function blackBody(kelvin) {
    const t = clamp(kelvin, 1667, 25000);
    const x = t <= 4000 ? -0.2661239e9 / t ** 3 - 0.2343589e6 / t ** 2 + 0.8776956e3 / t + 0.179910
        : -3.0258469e9 / t ** 3 + 2.1070379e6 / t ** 2 + 0.2226347e3 / t + 0.240390;
    const y = t <= 2222 ? -1.1063814 * x ** 3 - 1.34811020 * x ** 2 + 2.18555832 * x - 0.20219683
        : t <= 4000 ? -0.9549476 * x ** 3 - 1.37418593 * x ** 2 + 2.09137015 * x - 0.16748867
            : 3.0817580 * x ** 3 - 5.87338670 * x ** 2 + 3.75112997 * x - 0.37001483;
    const X = x / y, Z = (1 - x - y) / y;
    const rgb = [3.2406 * X - 1.5372 - 0.4986 * Z, -0.9689 * X + 1.8758 + 0.0415 * Z, 0.0557 * X - 0.2040 + 1.0570 * Z];
    return rgb.map(value => Math.max(1e-4, value) / Math.max(1e-4, rgb[1]));
}
/// Whether a frame can take the RAW white balance presets.
export const hasDaylightReference = frame => Boolean(cameraOf(frame).daylight);
/// The working-space transform that changes the frame's as-shot white
/// balance to the chosen one, or null for as shot.
function whiteBalanceTransform(frame, choice) {
    const light = WB_LIGHT[choice];
    const camera = cameraOf(frame);
    if (!light || !camera.daylight) return null;
    const inverse = invert(camera.matrix);
    if (!inverse) return null;
    const toDaylight = camera.daylight.map((value, c) => value / (camera.wb[c] || 1));
    const reference = blackBody(5500), lamp = blackBody(light[0]);
    const correction = [reference[0] / lamp[0], light[1], reference[2] / lamp[2]];
    return multiply(diagonal(correction), multiply(camera.matrix, multiply(diagonal(toDaylight), inverse)));
}

// ------------------------------------------------------------- local contrast

/// Three box passes approximate a gaussian; `radius` is in pixels.
function blur(source, width, height, radius) {
    const r = Math.max(0, Math.round(radius));
    if (!r) return Float32Array.from(source);
    let a = Float32Array.from(source), b = new Float32Array(source.length);
    const size = 2 * r + 1;
    for (let pass = 0; pass < 3; pass++) {
        for (let y = 0; y < height; y++) {
            const row = y * width;
            let sum = 0;
            for (let i = -r; i <= r; i++) sum += a[row + clamp(i, 0, width - 1)];
            for (let x = 0; x < width; x++) {
                b[row + x] = sum / size;
                sum += a[row + Math.min(width - 1, x + r + 1)] - a[row + Math.max(0, x - r)];
            }
        }
        for (let x = 0; x < width; x++) {
            let sum = 0;
            for (let i = -r; i <= r; i++) sum += b[clamp(i, 0, height - 1) * width + x];
            for (let y = 0; y < height; y++) {
                a[y * width + x] = sum / size;
                sum += b[Math.min(height - 1, y + r + 1) * width + x] - b[Math.max(0, y - r) * width + x];
            }
        }
    }
    return a;
}
/// A small grid of the picture, sampled bilinearly: wide blurs and the haze
/// estimate read it instead of the full image.
function grid(width, height, cells = 160) {
    const gw = Math.max(2, Math.round(width >= height ? cells : cells * width / height));
    const gh = Math.max(2, Math.round(height > width ? cells : cells * height / width));
    return { gw, gh, at(values, x, y) {
        const gx = clamp((x + .5) / width * gw - .5, 0, gw - 1), gy = clamp((y + .5) / height * gh - .5, 0, gh - 1);
        const x0 = Math.floor(gx), y0 = Math.floor(gy), x1 = Math.min(gw - 1, x0 + 1), y1 = Math.min(gh - 1, y0 + 1);
        const dx = gx - x0, dy = gy - y0;
        return (values[y0 * gw + x0] * (1 - dx) + values[y0 * gw + x1] * dx) * (1 - dy) + (values[y1 * gw + x0] * (1 - dx) + values[y1 * gw + x1] * dx) * dy;
    } };
}
function downsample(values, width, height, g) {
    const out = new Float32Array(g.gw * g.gh), count = new Uint32Array(g.gw * g.gh);
    for (let y = 0; y < height; y++) {
        const gy = Math.min(g.gh - 1, Math.floor(y * g.gh / height));
        for (let x = 0; x < width; x++) {
            const cell = gy * g.gw + Math.min(g.gw - 1, Math.floor(x * g.gw / width));
            out[cell] += values[y * width + x]; count[cell]++;
        }
    }
    for (let i = 0; i < out.length; i++) out[i] /= Math.max(1, count[i]);
    return out;
}
// Hash noise: the same grain for the same pixel on every render.
const hash = (x, y) => { let h = (x * 374761393 + y * 668265263) | 0; h = (h ^ (h >>> 13)) * 1274126177 | 0; return ((h ^ (h >>> 16)) >>> 0) / 4294967295 * 2 - 1; };
function grainAt(x, y, cell) {
    const gx = x / cell, gy = y / cell, x0 = Math.floor(gx), y0 = Math.floor(gy), dx = gx - x0, dy = gy - y0;
    const sx = dx * dx * (3 - 2 * dx), sy = dy * dy * (3 - 2 * dy);
    const top = hash(x0, y0) * (1 - sx) + hash(x0 + 1, y0) * sx, bottom = hash(x0, y0 + 1) * (1 - sx) + hash(x0 + 1, y0 + 1) * sx;
    return top * (1 - sy) + bottom * sy;
}
const luma = (r, g, b) => .2126 * r + .7152 * g + .0722 * b;
function hueOf(r, g, b) {
    const max = Math.max(r, g, b), min = Math.min(r, g, b), chroma = max - min;
    if (chroma <= 1e-6) return 0;
    const h = max === r ? ((g - b) / chroma) % 6 : max === g ? (b - r) / chroma + 2 : (r - g) / chroma + 4;
    return (h * 60 + 360) % 360;
}
/// The two neighbouring bands of a hue and how much of each it takes.
function bandsOf(hue, weights) {
    weights.fill(0);
    for (let i = 0; i < BAND_HUES.length; i++) {
        const from = BAND_HUES[i], to = i + 1 < BAND_HUES.length ? BAND_HUES[i + 1] : 360;
        if (hue >= from && hue < to) { const t = (hue - from) / (to - from); weights[i] = 1 - t; weights[(i + 1) % BAND_HUES.length] = t; return; }
    }
}
const BAND_GAP = BAND_HUES.map((hue, i) => ((BAND_HUES[(i + 1) % BAND_HUES.length] - hue + 360) % 360 + (hue - BAND_HUES[(i + BAND_HUES.length - 1) % BAND_HUES.length] + 360) % 360) / 2);
const hueColour = hue => {
    const k = n => (n + hue / 60) % 6, f = n => 1 - Math.max(0, Math.min(k(n), 4 - k(n), 1));
    return [f(5), f(3), f(1)];
};

/// Auto tone: exposure, contrast and the four tone sliders from the
/// picture's own luminance, with its white balance and look left alone.
export function autoTone(frame, recipe = {}) {
    const r = normalizeRecipe(recipe);
    const sampled = develop(frame, { ...r, exposure: 0, contrast: 0, highlights: 0, shadows: 0, whites: 0, blacks: 0 }, { edge: 320, measure: true });
    const values = Float32Array.from(sampled.luminance).sort();
    const at = p => values[Math.min(values.length - 1, Math.floor(p * values.length))] || 1e-4;
    const median = Math.max(1e-4, at(.5));
    // Mid-grey for the middle of the picture, but never so bright that the
    // highlights go past what the Highlights slider can bring back: a night
    // scene stays a night scene.
    const exposure = clamp(Math.min(Math.log2(.14 / median), Math.log2(2 / Math.max(1e-4, at(.99)))), -2, 2);
    const lift = 2 ** exposure, top = at(.995) * lift, floor = at(.02) * lift, deep = at(.005) * lift;
    const highlights = top > 1 ? -clamp(Math.log2(top) * 45, 0, 80) : 0;
    const shadows = floor < .03 ? clamp(Math.log2(.03 / Math.max(1e-4, floor)) * 14, 0, 55) : 0;
    const whites = top < .8 ? clamp(Math.log2(.95 / Math.max(1e-4, top)) * 60, 0, 40) : 0;
    const blacks = -clamp(deep * 1000, 0, 25);
    const range = Math.log2(Math.max(1e-4, at(.95)) / Math.max(1e-4, at(.05)));
    const contrast = clamp((6 - range) * 6, -15, 25);
    const round = value => Math.round(value);
    return { exposure: Math.round(exposure * 20) / 20, contrast: round(contrast), highlights: round(highlights),
        shadows: round(shadows), whites: round(whites), blacks: round(blacks) };
}

/// Reads an output pixel's working-space linear colour, after white balance
/// and exposure, with its opacity in the fourth slot. Every value it closes
/// over is fixed, which keeps the hot loop optimisable.
function sampler(frame, width, height, balance, gains, exposure) {
    const matrix = frame.matrix?.some(v => v !== 0) ? frame.matrix : IDENTITY;
    const wb = frame.wb || [1, 1, 1], scale = frame.scale ?? 1;
    const rr = scale * wb[0], gg = scale * wb[1], bb = scale * wb[2];
    const sensor = frame.data, alphaSource = frame.alpha, fw = frame.width, fh = frame.height;
    const m0 = matrix[0], m1 = matrix[1], m2 = matrix[2], m3 = matrix[3], m4 = matrix[4], m5 = matrix[5], m6 = matrix[6], m7 = matrix[7], m8 = matrix[8];
    const n = balance || IDENTITY, hasBalance = Boolean(balance);
    const kr = gains[0] * exposure, kg = gains[1] * exposure, kb = gains[2] * exposure;
    const xs = fw / width, ys = fh / height;
    const out = new Float64Array(4);
    return (x, y) => {
        const sx = clamp((x + .5) * xs - .5, 0, fw - 1), sy = clamp((y + .5) * ys - .5, 0, fh - 1);
        const x0 = Math.floor(sx), y0 = Math.floor(sy), x1 = Math.min(x0 + 1, fw - 1), y1 = Math.min(y0 + 1, fh - 1);
        const dx = sx - x0, dy = sy - y0;
        const p00 = y0 * fw + x0, p10 = y0 * fw + x1, p01 = y1 * fw + x0, p11 = y1 * fw + x1;
        const w00 = (1 - dx) * (1 - dy), w10 = dx * (1 - dy), w01 = (1 - dx) * dy, w11 = dx * dy;
        const cr = (sensor[p00 * 3] * w00 + sensor[p10 * 3] * w10 + sensor[p01 * 3] * w01 + sensor[p11 * 3] * w11) * rr;
        const cg = (sensor[p00 * 3 + 1] * w00 + sensor[p10 * 3 + 1] * w10 + sensor[p01 * 3 + 1] * w01 + sensor[p11 * 3 + 1] * w11) * gg;
        const cb = (sensor[p00 * 3 + 2] * w00 + sensor[p10 * 3 + 2] * w10 + sensor[p01 * 3 + 2] * w01 + sensor[p11 * 3 + 2] * w11) * bb;
        let r = m0 * cr + m1 * cg + m2 * cb, g = m3 * cr + m4 * cg + m5 * cb, b = m6 * cr + m7 * cg + m8 * cb;
        if (hasBalance) {
            const r1 = n[0] * r + n[1] * g + n[2] * b, g1 = n[3] * r + n[4] * g + n[5] * b, b1 = n[6] * r + n[7] * g + n[8] * b;
            r = r1; g = g1; b = b1;
        }
        out[0] = Math.max(0, r * kr); out[1] = Math.max(0, g * kg); out[2] = Math.max(0, b * kb);
        out[3] = alphaSource ? (alphaSource[p00] * w00 + alphaSource[p10] * w10 + alphaSource[p01] * w01 + alphaSource[p11] * w11) / 255 : 1;
        return out;
    };
}

export function develop(frame, recipe = {}, { edge = 0, bitDepth = 8, before = false, clipping = false, measure = false } = {}) {
    if (![8, 10, 12].includes(bitDepth)) throw new Error('Output must be 8, 10 or 12 bit.');
    const r = normalizeRecipe(recipe);
    const settings = before ? defaultRecipe() : r;
    const factor = edge > 0 ? Math.min(1, edge / Math.max(frame.width, frame.height)) : 1;
    const width = Math.max(1, Math.round(frame.width * factor));
    const height = Math.max(1, Math.round(frame.height * factor));
    const pixels = width * height;
    const max = (1 << bitDepth) - 1;
    const data = bitDepth === 8 ? new Uint8ClampedArray(pixels * 4) : new Uint16Array(pixels * 4);
    const histogram = new Uint32Array(256), histogramRGB = new Uint32Array(768);
    // Output pixels per source pixel of the full-resolution image: radii
    // given in source pixels shrink with the preview.
    const detailScale = width / (frame.sourceWidth ?? frame.width);
    const longEdge = Math.max(width, height);
    const exposure = 2 ** settings.exposure;
    const warm = 2 ** (settings.temperature / 200), tint = 2 ** (settings.tint / 300);
    const gains = [warm * tint, 1 / tint, tint / warm];
    const contrast = 2 ** (settings.contrast / 150);
    const black = settings.blacks / 1000, white = 2 ** (settings.whites / 200);
    // All expensive curves are sampled once per edit, not once per channel.
    // The display LUT has 16-bit linear resolution before 8/10/12-bit output.
    const clipAt = .18 * (1 / .18) ** (1 / contrast);
    const displayScale = 65535 / clipAt;
    const display = new Float32Array(65536);
    for (let i = 0; i < display.length; i++) display[i] = clamp(toSRGB(.18 * (i / displayScale / .18) ** contrast), 0, 1);
    const tones = new Float32Array(16385);
    for (let i = 0; i < tones.length; i++) {
        const luminance = i / 2048;
        tones[i] = 2 ** ((settings.shadows * Math.exp(-luminance * 6) + settings.highlights * (1 - Math.exp(-luminance * 1.5))) / 100);
    }
    let balance = settings.whiteBalance === 'shot' || settings.whiteBalance === 'auto' ? null : whiteBalanceTransform(frame, settings.whiteBalance);
    // Auto white balance: the grey the mid-tones average to, made neutral.
    if (settings.whiteBalance === 'auto') {
        const probe = sampler(frame, width, height, null, [1, 1, 1], 1), step = Math.max(1, Math.round(longEdge / 200));
        const sum = [0, 0, 0];
        for (let y = 0; y < height; y += step) for (let x = 0; x < width; x += step) {
            const px = probe(x, y), l = luma(px[0], px[1], px[2]);
            if (l < .02 || l > .9) continue;
            sum[0] += px[0]; sum[1] += px[1]; sum[2] += px[2];
        }
        if (sum[0] > 0 && sum[1] > 0 && sum[2] > 0) balance = diagonal([sum[1] / sum[0], 1, sum[1] / sum[2]]);
    }
    const sample = sampler(frame, width, height, balance, gains, exposure);
    // Clipping is reported for what the edit did: a pixel already pure
    // white or pure black in the untouched picture is not counted.
    const untouched = before ? null : sampler(frame, width, height, null, [1, 1, 1], 1);
    const clipAt0 = 1 - 1e-6;

    // ------------------------------------------------ what this edit needs
    const tonesActive = settings.highlights !== 0 || settings.shadows !== 0;
    const clarity = settings.clarity / 100, texture = settings.texture / 100, dehaze = settings.dehaze / 100;
    const sharpen = settings.sharpening.amount / 100, nrLuminance = settings.noise.luminance / 100, nrColour = settings.noise.colour / 100;
    const local = tonesActive || clarity || texture || dehaze || sharpen || nrLuminance || nrColour;
    const g = grid(width, height);
    let haze = null, lum = null, toneGrid = null, clarityGrid = null, textureBlur = null, fineBlur = null, sharpBlur = null, chromaR = null, chromaB = null;
    if (dehaze) {
        // Dark channel on the grid; the atmosphere is the brightest haze.
        const cells = g.gw * g.gh, dark = new Float32Array(cells), colour = new Float32Array(cells * 3);
        for (let gy = 0; gy < g.gh; gy++) for (let gx = 0; gx < g.gw; gx++) {
            const cell = gy * g.gw + gx; let n = 0, lowest = Infinity;
            for (let j = 0; j < 3; j++) for (let i = 0; i < 3; i++) {
                const rgb = sample(Math.min(width - 1, Math.floor((gx + (i + .5) / 3) * width / g.gw)), Math.min(height - 1, Math.floor((gy + (j + .5) / 3) * height / g.gh)));
                lowest = Math.min(lowest, rgb[0], rgb[1], rgb[2]);
                colour[cell * 3] += rgb[0]; colour[cell * 3 + 1] += rgb[1]; colour[cell * 3 + 2] += rgb[2]; n++;
            }
            dark[cell] = lowest; for (let c = 0; c < 3; c++) colour[cell * 3 + c] /= n;
        }
        const order = Array.from(dark.keys()).sort((a, b) => dark[b] - dark[a]).slice(0, Math.max(1, Math.round(cells * .01)));
        const air = [0, 0, 0];
        for (const cell of order) for (let c = 0; c < 3; c++) air[c] += colour[cell * 3 + c] / order.length;
        const airMax = Math.max(1e-4, ...air);
        haze = { air, transmission: blur(dark.map(value => 1 - .95 * Math.min(1, value / airMax)), g.gw, g.gh, 2) };
    }
    const applyHaze = (rgb, x, y) => {
        if (!haze) return;
        if (dehaze > 0) {
            const t = Math.max(.1, 1 - (1 - g.at(haze.transmission, x, y)) * dehaze);
            for (let c = 0; c < 3; c++) rgb[c] = Math.max(0, (rgb[c] - haze.air[c]) / t + haze.air[c]);
        } else {
            const k = -dehaze * .6;
            for (let c = 0; c < 3; c++) rgb[c] = rgb[c] * (1 - k) + haze.air[c] * k;
        }
    };
    if (tonesActive || clarity) {
        // The wide blurs only need the picture's broad light: a few
        // samples per grid cell stand in for the full image.
        const small = new Float32Array(g.gw * g.gh);
        for (let gy = 0; gy < g.gh; gy++) for (let gx = 0; gx < g.gw; gx++) {
            let sum = 0;
            for (let j = 0; j < 3; j++) for (let i = 0; i < 3; i++) {
                const px = Math.min(width - 1, Math.floor((gx + (i + .5) / 3) * width / g.gw)), py = Math.min(height - 1, Math.floor((gy + (j + .5) / 3) * height / g.gh));
                const rgb = sample(px, py);
                applyHaze(rgb, px, py);
                sum += Math.log2(luma(rgb[0], rgb[1], rgb[2]) + 1e-4);
            }
            small[gy * g.gw + gx] = sum / 9;
        }
        // Stored as a linear factor: the loop multiplies instead of taking a power.
        if (tonesActive) toneGrid = blur(small, g.gw, g.gh, 5).map(value => 2 ** (value * .7));
        if (clarity) clarityGrid = blur(small, g.gw, g.gh, 2);
    }
    if (texture || nrLuminance || sharpen || nrColour) {
        lum = new Float32Array(pixels);
        if (nrColour) { chromaR = new Float32Array(pixels); chromaB = new Float32Array(pixels); }
        for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
            const rgb = sample(x, y); applyHaze(rgb, x, y);
            const i = y * width + x, l = luma(rgb[0], rgb[1], rgb[2]);
            lum[i] = Math.log2(l + 1e-4);
            if (chromaR) { chromaR[i] = rgb[0] / (l + 1e-4); chromaB[i] = rgb[2] / (l + 1e-4); }
        }
        if (texture) textureBlur = blur(lum, width, height, Math.max(1, longEdge * .004));
        if (nrLuminance) fineBlur = blur(lum, width, height, Math.max(1, 1.5 * detailScale));
        if (sharpen) sharpBlur = blur(lum, width, height, Math.max(1, settings.sharpening.radius * detailScale));
        if (chromaR) {
            const radius = Math.max(1, 4 * detailScale * (.5 + nrColour));
            chromaR = blur(chromaR, width, height, radius); chromaB = blur(chromaB, width, height, radius);
        }
        lum = null;
    }

    // ------------------------------------------------- display-side looks
    const curve = settings.curve;
    const curveActive = curve.highlights || curve.lights || curve.darks || curve.shadows || settings.profile === 'vivid' || settings.profile === 'neutral';
    let curveLut = null;
    if (curveActive) {
        const profileBend = settings.profile === 'vivid' ? .08 : settings.profile === 'neutral' ? -.06 : 0;
        curveLut = new Float32Array(4097);
        for (let i = 0; i < curveLut.length; i++) {
            const v = i / 4096;
            // Each region lifts or lowers its own quarter of the range.
            const bump = (centre, width) => Math.max(0, 1 - Math.abs(v - centre) / width);
            let out = v + (curve.shadows * bump(.125, .25) + curve.darks * bump(.375, .25) + curve.lights * bump(.625, .25) + curve.highlights * bump(.875, .25)) / 100 * .25;
            out += profileBend * Math.sin((v - .5) * Math.PI) * .5 * (1 - Math.abs(2 * v - 1)) * 2;
            curveLut[i] = clamp(out, 0, 1);
        }
    }
    const mixer = COLOUR_BANDS.map(band => settings.mixer[band]);
    const mono = settings.profile === 'monochrome';
    const mixerActive = mixer.some(band => band.hue || band.saturation || band.luminance);
    const vibrance = settings.vibrance / 100;
    const saturation = settings.saturation / 100 + (settings.profile === 'vivid' ? .15 : settings.profile === 'neutral' ? -.08 : 0);
    const grading = settings.grading;
    const zones = GRADING_ZONES.map(name => grading[name]);
    const gradingActive = zones.some(z => z.saturation || z.luminance);
    const gradingTint = zones.map(z => { const c = hueColour(z.hue), l = luma(c[0], c[1], c[2]); return c.map(v => (v - l) * z.saturation / 100 * .3); });
    const exponent = 1 + (100 - grading.blending) / 100 * 3;
    const colourActive = mono || mixerActive || vibrance || saturation || gradingActive;
    const vignette = settings.vignette, vignetteAmount = vignette.amount / 100;
    const vignetteMid = .25 + vignette.midpoint / 100 * .9, vignetteFeather = .05 + vignette.feather / 100 * .75;
    const grainAmount = settings.grain.amount / 100 * .12;
    const grainCell = Math.max(.5, (.6 + settings.grain.size / 100 * 3) * longEdge / 2000);
    const weights = new Float32Array(BAND_HUES.length);
    const out = new Float64Array(3);
    const needsBase = Boolean(fineBlur || textureBlur || clarityGrid || sharpBlur);
    const toHistogram = 255 / max;
    let clippedHigh = 0, clippedLow = 0;
    const luminance = measure ? new Float32Array(pixels) : null;

    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
        const i = y * width + x;
        const rgb = sample(x, y); applyHaze(rgb, x, y);
        let l = luma(rgb[0], rgb[1], rgb[2]);
        if (luminance) luminance[i] = l;
        let tone;
        if (local) {
            // Local luminance edits act on log luminance and scale the
            // colour with it, so hue and saturation hold.
            const base = needsBase ? Math.log2(l + 1e-4) : 0;
            let edited = base;
            if (fineBlur) {
                const edgeWeight = Math.exp(-((base - fineBlur[i]) ** 2) / (.02 + nrLuminance * .3));
                edited += (fineBlur[i] - base) * nrLuminance * edgeWeight;
            }
            if (textureBlur) edited += (edited - textureBlur[i]) * texture * .6;
            if (clarityGrid) {
                const mid = Math.exp(-((base + 2.5) ** 2) / 4);
                edited += (edited - g.at(clarityGrid, x, y)) * clarity * .5 * mid;
            }
            if (sharpBlur) {
                const detail = base - sharpBlur[i];
                const mask = settings.sharpening.masking ? smooth(0, settings.sharpening.masking / 100 * .15, Math.abs(detail)) : 1;
                edited += detail * sharpen * mask;
            }
            if (edited !== base) {
                const gain = 2 ** (edited - base);
                rgb[0] *= gain; rgb[1] *= gain; rgb[2] *= gain; l *= gain;
            }
            if (chromaR) {
                const cr = chromaR[i] * l, cb = chromaB[i] * l;
                rgb[0] += (cr - rgb[0]) * nrColour; rgb[2] += (cb - rgb[2]) * nrColour;
                rgb[1] = Math.max(0, (l - .2126 * rgb[0] - .0722 * rgb[2]) / .7152);
            }
            const toneLuminance = toneGrid ? g.at(toneGrid, x, y) * Math.pow(l + 1e-4, .3) : l;
            tone = tones[Math.min(16384, Math.round(toneLuminance * 2048))];
        } else tone = tones[Math.min(16384, Math.round(l * 2048))];
        let hi = false, lo = true;
        for (let c = 0; c < 3; c++) {
            const value = Math.max(0, (rgb[c] * tone + black) * white);
            hi ||= value > clipAt; lo &&= value <= 0;
            out[c] = display[Math.min(65535, Math.round(value * displayScale))];
        }
        if (curveLut) for (let c = 0; c < 3; c++) { const p = out[c] * 4096, k = Math.min(4095, Math.floor(p)); out[c] = curveLut[k] + (curveLut[k + 1] - curveLut[k]) * (p - k); }
        if (colourActive) {
            let lm = luma(out[0], out[1], out[2]);
            const high = Math.max(out[0], out[1], out[2]), low = Math.min(out[0], out[1], out[2]);
            const sat = high > 1e-4 ? (high - low) / high : 0;
            const hue = hueOf(out[0], out[1], out[2]);
            bandsOf(hue, weights);
            if (mono) {
                let shift = 0;
                for (let b = 0; b < weights.length; b++) shift += weights[b] * mixer[b].luminance;
                lm = clamp(lm * 2 ** (shift / 100 * sat * 1.2), 0, 1);
                out[0] = out[1] = out[2] = lm;
            } else {
                let factor = 1 + saturation;
                if (mixerActive) {
                    let hueShift = 0, satShift = 0, lumShift = 0;
                    for (let b = 0; b < weights.length; b++) {
                        if (!weights[b]) continue;
                        hueShift += weights[b] * mixer[b].hue / 100 * BAND_GAP[b] / 2;
                        satShift += weights[b] * mixer[b].saturation / 100;
                        lumShift += weights[b] * mixer[b].luminance / 100;
                    }
                    if (hueShift && sat > 0) {
                        const target = hueColour((hue + hueShift + 360) % 360), tl = luma(target[0], target[1], target[2]);
                        const chroma = high - low;
                        for (let c = 0; c < 3; c++) out[c] = lm + (target[c] - tl) * chroma;
                    }
                    factor *= Math.max(0, 1 + satShift);
                    if (lumShift) { const k = 2 ** (lumShift * sat * .9); out[0] *= k; out[1] *= k; out[2] *= k; lm *= k; }
                }
                if (vibrance) {
                    // Skin sits between red and orange; vibrance leaves it be.
                    const skin = smooth(5, 20, hue) * (1 - smooth(40, 55, hue));
                    factor *= vibrance > 0 ? 1 + vibrance * (1 - sat) * (1 - .75 * skin) : 1 + vibrance;
                }
                if (factor !== 1) for (let c = 0; c < 3; c++) out[c] = lm + (out[c] - lm) * factor;
            }
            if (gradingActive) {
                const t = clamp(lm - grading.balance / 400, 0, 1);
                const ws = (1 - t) ** exponent, wh = t ** exponent, wm = Math.max(0, 1 - ws - wh);
                const w = [ws, wm, wh, 1];
                for (let z = 0; z < 4; z++) {
                    if (!w[z]) continue;
                    const lift = zones[z].luminance / 100 * .15 * w[z];
                    for (let c = 0; c < 3; c++) out[c] += gradingTint[z][c] * w[z] + lift;
                }
            }
            for (let c = 0; c < 3; c++) out[c] = clamp(out[c], 0, 1);
        }
        if (vignetteAmount) {
            const dx = (x + .5) / width * 2 - 1, dy = (y + .5) / height * 2 - 1;
            const f = smooth(vignetteMid - vignetteFeather, vignetteMid + vignetteFeather, Math.hypot(dx, dy) / Math.SQRT2 * 1.4);
            for (let c = 0; c < 3; c++) out[c] = vignetteAmount < 0 ? out[c] * (1 + vignetteAmount * f) : out[c] + (1 - out[c]) * vignetteAmount * f;
        }
        if (grainAmount) {
            const n = grainAt(x, y, grainCell) * grainAmount, lm = luma(out[0], out[1], out[2]);
            const k = n * (.35 + 2.6 * lm * (1 - lm));
            for (let c = 0; c < 3; c++) out[c] = clamp(out[c] + k, 0, 1);
        }
        const dest = i * 4;
        const qr = Math.round(out[0] * max), qg = Math.round(out[1] * max), qb = Math.round(out[2] * max);
        data[dest] = qr; data[dest + 1] = qg; data[dest + 2] = qb;
        histogramRGB[Math.round(qr * toHistogram)]++; histogramRGB[256 + Math.round(qg * toHistogram)]++; histogramRGB[512 + Math.round(qb * toHistogram)]++;
        histogram[Math.min(255, Math.round((.2126 * qr + .7152 * qg + .0722 * qb) * toHistogram))]++;
        if (hi || lo) {
            // Untouched, contrast is zero and white clips at one, less the
            // rounding the edited threshold carries.
            const o = untouched ? untouched(x, y) : null;
            if (!o) hi = lo = false;
            else {
                if (hi && Math.max(o[0], o[1], o[2]) >= clipAt0) hi = false;
                if (lo && o[0] <= 0 && o[1] <= 0 && o[2] <= 0) lo = false;
            }
        }
        if (hi) clippedHigh++; if (lo) clippedLow++;
        if (clipping && hi) { data[dest] = max; data[dest + 1] = 0; data[dest + 2] = 0; }
        else if (clipping && lo) { data[dest] = 0; data[dest + 1] = 0; data[dest + 2] = max; }
        data[dest + 3] = Math.round(rgb[3] * max);
    }
    return { width, height, data, bitDepth, histogram, histogramRGB, clippedHigh, clippedLow, ...(luminance ? { luminance } : {}) };
}
