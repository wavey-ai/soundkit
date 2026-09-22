import { createRequire } from 'node:module';
import { cpSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const require = createRequire(join(root, 'package.json'));
const backend = dirname(require.resolve('@jsquash/avif/package.json'));
const version = JSON.parse(readFileSync(join(backend, 'package.json'))).version;
if (version !== '2.1.1') throw new Error(`Unsupported AVIF backend ${version}; verify the colour-header contract before upgrading.`);
const dist = join(root, 'dist');
rmSync(dist, { recursive: true, force: true });
mkdirSync(dist, { recursive: true });
for (const name of ['meta.js', 'utils.js', 'decode.js']) cpSync(join(backend, name), join(dist, name));
for (const kind of ['enc', 'dec']) {
    mkdirSync(join(dist, 'codec', kind), { recursive: true });
    for (const extension of ['js', 'wasm']) {
        const name = `avif_${kind}.${extension}`;
        cpSync(join(backend, 'codec', kind, name), join(dist, 'codec', kind, name));
    }
}
// Ship the single-thread runtime so the package also works without cross-origin isolation.
const encoder = readFileSync(join(backend, 'encode.js'), 'utf8');
if (!encoder.includes("from 'wasm-feature-detect'")) throw new Error('The AVIF backend thread selector changed.');
writeFileSync(join(dist, 'backend-encode.js'), encoder.replace("from 'wasm-feature-detect'", "from './threads.js'"));
writeFileSync(join(dist, 'threads.js'), 'export const threads = async () => false;\n');
for (const name of ['index.mjs', 'encode.js', 'srgb.mjs']) cpSync(join(root, 'src', name), join(dist, name));
cpSync(join(backend, 'LICENSE'), join(dist, 'LICENSE.jsquash'));
cpSync(join(root, 'LICENSE'), join(dist, 'LICENSE'));
writeFileSync(join(dist, 'package.json'), JSON.stringify({ type: 'module', private: true }) + '\n');
const hashes = Object.fromEntries(['enc', 'dec'].map(kind => [kind, createHash('sha256')
    .update(readFileSync(join(dist, 'codec', kind, `avif_${kind}.wasm`))).digest('hex')]));
writeFileSync(join(dist, 'provenance.json'), JSON.stringify({ package: '@wavey-ai/soundkit-avif', backend: '@jsquash/avif', version, sha256: hashes }, null, 2) + '\n');
console.log(`SoundKit AVIF: jSquash ${version}; encoder and decoder installed.`);
