// A deterministic test card: four colour quadrants under a horizontal ramp.
// Each corner has its own colour, so a rotation or a mirror is visible.
const QUADRANTS = [[200, 60, 50], [60, 170, 70], [50, 80, 200], [210, 190, 60]];
export function pixel(x, y, width, height) {
    const [r, g, b] = QUADRANTS[(y * 2 >= height ? 2 : 0) + (x * 2 >= width ? 1 : 0)];
    const light = .55 + .45 * x / (width - 1);
    return [Math.round(r * light), Math.round(g * light), Math.round(b * light)];
}
// A circular opacity ramp: opaque in the center, clear in the corners.
export function opacity(x, y, width, height) {
    const distance = Math.hypot((x + .5) / width - .5, (y + .5) / height - .5) / Math.SQRT1_2;
    return Math.round(255 * Math.min(1, Math.max(0, 1.3 - 1.3 * distance)));
}
// The card as a baseline little-endian TIFF: the input for the fixture encoders.
export function cardTIFF({ width, height, bits = 8, alpha = false, orientation = 1 }) {
    const samples = alpha ? 4 : 3, bytes = bits / 8, size = width * height * samples * bytes;
    const entries = [[256, 4, [width]], [257, 4, [height]], [258, 3, Array(samples).fill(bits)], [259, 3, [1]], [262, 3, [2]], [273, 4, [0]],
        [274, 3, [orientation]], [277, 3, [samples]], [278, 4, [height]], [279, 4, [size]], [284, 3, [1]], ...(alpha ? [[338, 3, [2]]] : [])];
    let offset = 8 + 2 + entries.length * 12 + 4;
    const extra = entries.map(([, type, values]) => {
        const length = values.length * (type === 3 ? 2 : 4);
        if (length <= 4) return 0;
        const at = offset; offset += length; return at;
    });
    const out = Buffer.alloc(offset + size);
    out.write('II'); out.writeUInt16LE(42, 2); out.writeUInt32LE(8, 4); out.writeUInt16LE(entries.length, 8);
    entries.forEach(([id, type, values], index) => {
        const at = 10 + index * 12;
        if (id === 273) values = [offset];
        out.writeUInt16LE(id, at); out.writeUInt16LE(type, at + 2); out.writeUInt32LE(values.length, at + 4);
        const target = extra[index] || at + 8;
        if (extra[index]) out.writeUInt32LE(extra[index], at + 8);
        values.forEach((value, i) => type === 3 ? out.writeUInt16LE(value, target + i * 2) : out.writeUInt32LE(value, target + i * 4));
    });
    let at = offset;
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
        const values = [...pixel(x, y, width, height), ...(alpha ? [opacity(x, y, width, height)] : [])];
        for (const value of values) {
            if (bits === 8) out[at] = value; else out.writeUInt16LE(value * 257, at);
            at += bytes;
        }
    }
    return out;
}
// The card with deterministic noise: a low-quality encode of it needs the HEVC loop filters.
export function texturedTIFF({ width, height }) {
    const out = cardTIFF({ width, height });
    let seed = 12345;
    for (let at = out.length - width * height * 3; at < out.length; at++) {
        seed = (seed * 1103515245 + 12345) & 0x7fffffff;
        out[at] = Math.max(0, Math.min(255, out[at] + ((seed >> 16) % 61) - 30));
    }
    return out;
}
