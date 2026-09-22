import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { cpSync, existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync, statSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { availableParallelism } from 'node:os';
import { spawn } from 'node:child_process';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const version = '0.22.0';
const checksum = '1071e6e8011593c366ffdadc3d3513f57c90202d526e133174945ec1dd53f2a1';
const build = join(root, 'build'), dist = join(root, 'dist'), source = join(build, `LibRaw-${version}`);
mkdirSync(build, { recursive: true }); mkdirSync(dist, { recursive: true });
const archive = join(build, `LibRaw-${version}.tar.gz`);
if (!existsSync(archive)) {
    const response = await fetch(`https://www.libraw.org/data/LibRaw-${version}.tar.gz`);
    if (!response.ok) throw new Error(`LibRaw download: ${response.status}`);
    writeFileSync(archive, Buffer.from(await response.arrayBuffer()));
}
if (createHash('sha256').update(readFileSync(archive)).digest('hex') !== checksum) throw new Error('LibRaw source checksum mismatch');
if (!existsSync(source)) execFileSync('tar', ['-xzf', archive, '-C', build]);
const env = { ...process.env, EM_NODE_JS: process.execPath };
const compiler = process.env.EMXX || 'em++';
const flags = ['-O2', '-std=c++17', '-fexceptions', '-DLIBRAW_NOTHREADS', '-DUSE_ZLIB', '-DUSE_JPEG', '-sUSE_ZLIB=1', '-sUSE_LIBJPEG=1', `-I${source}`];
const sources = directory => readdirSync(directory, { withFileTypes: true }).flatMap(entry => entry.isDirectory()
    ? sources(join(directory, entry.name)) : entry.name.endsWith('.cpp') ? [join(directory, entry.name)] : []);
const files = [...sources(join(source, 'src')).filter(file => !file.endsWith('_ph.cpp')), join(root, 'src/raw.cpp')];
const objects = files.map(file => join(build, file.replace(source, '').replace(root, '').replace(/[^a-zA-Z0-9]/g, '_') + '.o'));
// Prepare Emscripten ports once before parallel compiles touch their cache.
if (!existsSync(objects[0])) execFileSync(compiler, [...flags, '-c', files[0], '-o', objects[0]], { env, stdio: 'inherit' });
let next = 0, compiled = 0;
await Promise.all(Array.from({ length: Math.min(6, availableParallelism()) }, async () => {
    while (next < files.length) {
        const index = next++, file = files[index], object = objects[index];
        if (existsSync(object) && statSync(object).mtimeMs >= statSync(file).mtimeMs) continue;
        await new Promise((accept, reject) => {
            const child = spawn(compiler, [...flags, '-c', file, '-o', object], { env, stdio: 'inherit' });
            child.on('error', reject); child.on('exit', code => code === 0 ? accept() : reject(new Error(`Compilation failed: ${file}`)));
        });
        compiled++;
    }
}));
const output = join(dist, 'raw.mjs');
if (compiled || !existsSync(output) || !existsSync(join(dist, 'raw.wasm'))) {
    execFileSync(compiler, [...flags, ...objects, '-sMODULARIZE=1', '-sEXPORT_ES6=1', '-sENVIRONMENT=web,worker,node',
        '-sALLOW_MEMORY_GROWTH=1', '-sMAXIMUM_MEMORY=2147483648', '-sFILESYSTEM=0', '-sDISABLE_EXCEPTION_CATCHING=0',
        '-sEXPORTED_FUNCTIONS=["_malloc","_free","_raw_open","_raw_decode","_raw_close","_raw_error","_raw_metadata","_raw_pixels","_raw_size","_raw_width","_raw_height","_raw_thumbnail","_raw_thumbnail_size"]',
        '-sEXPORTED_RUNTIME_METHODS=["HEAPU8","HEAPU16","UTF8ToString"]', '-o', output], { env, stdio: 'inherit' });
}
cpSync(join(root, 'src/index.mjs'), join(dist, 'index.mjs'));
cpSync(join(root, 'src/develop.mjs'), join(dist, 'develop.mjs'));
cpSync(join(root, 'LICENSE'), join(dist, 'LICENSE'));
cpSync(join(source, 'LICENSE.CDDL'), join(dist, 'LICENSE.LibRaw'));
cpSync(join(source, 'COPYRIGHT'), join(dist, 'COPYRIGHT.LibRaw'));
cpSync(archive, join(dist, 'LibRaw-source.tar.gz'));
writeFileSync(join(dist, 'NOTICE'), `LibRaw ${version}: https://www.libraw.org/ (CDDL-1.0). Source SHA256 ${checksum}.\nThis software is based in part on the work of the Independent JPEG Group.\nzlib: Copyright (C) 1995-2024 Jean-loup Gailly and Mark Adler.\nEmscripten JPEG and zlib ports are used unmodified.\n`);
writeFileSync(join(dist, 'provenance.json'), JSON.stringify({ libraw: version, sourceSha256: checksum, compiler: execFileSync(compiler, ['--version'], { env, encoding: 'utf8' }).split('\n')[0] }, null, 2));
console.log(`RAW WASM: ${(statSync(join(dist, 'raw.wasm')).size / 1048576).toFixed(2)} MB; LibRaw ${version}`);
