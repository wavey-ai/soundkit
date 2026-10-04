// Writes test/fixtures with the macOS HEVC encoder. Run on macOS: node test/make-fixtures.mjs
import { execFileSync } from 'node:child_process';
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { cardTIFF, texturedTIFF } from './card.mjs';

const fixtures = join(dirname(fileURLToPath(import.meta.url)), 'fixtures');
const work = mkdtempSync(join(tmpdir(), 'soundkit-heif-'));
mkdirSync(fixtures, { recursive: true });
const sips = (...args) => execFileSync('sips', args, { stdio: ['ignore', 'ignore', 'inherit'] });
const encode = (name, tiff, { quality = 90, profile } = {}) => {
    const source = join(work, `${name}.tif`);
    writeFileSync(source, tiff);
    if (profile) sips('--embedProfile', profile, source);
    sips('-s', 'format', 'heic', '-s', 'formatOptions', String(quality), source, '--out', join(fixtures, `${name}.heic`));
};
const size = { width: 96, height: 64 };
encode('card', cardTIFF(size));
// Orientation 6 becomes a rotation property in the file, not rotated pixels.
encode('rotated', cardTIFF({ ...size, orientation: 6 }));
encode('alpha', cardTIFF({ ...size, alpha: true }));
// A 16-bit source gives a 10-bit file.
encode('deep', cardTIFF({ ...size, bits: 16 }));
// Larger than one 512-pixel tile: a grid of six tiles, cropped.
encode('grid', cardTIFF({ width: 1280, height: 960 }));
encode('p3', cardTIFF(size), { profile: '/System/Library/ColorSync/Profiles/Display P3.icc' });
encode('texture', texturedTIFF({ width: 64, height: 48 }), { quality: 10 });
// The reference for the low-quality file is the macOS decoder's output, as RGB.
const bitmap = join(work, 'texture.bmp');
sips('-s', 'format', 'bmp', join(fixtures, 'texture.heic'), '--out', bitmap);
const bmp = readFileSync(bitmap);
const offset = bmp.readUInt32LE(10), width = bmp.readInt32LE(18), rows = bmp.readInt32LE(22), bytes = bmp.readUInt16LE(28) / 8;
const height = Math.abs(rows), stride = Math.ceil(width * bytes / 4) * 4, rgb = Buffer.alloc(width * height * 3);
for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const at = offset + (rows > 0 ? height - 1 - y : y) * stride + x * bytes;
    rgb.set([bmp[at + 2], bmp[at + 1], bmp[at]], (y * width + x) * 3);
}
writeFileSync(join(fixtures, 'texture.rgb'), rgb);
rmSync(work, { recursive: true });
