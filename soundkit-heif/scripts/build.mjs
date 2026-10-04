import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { cpSync, existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { availableParallelism } from 'node:os';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const sha256 = file => createHash('sha256').update(readFileSync(file)).digest('hex');
// LIBDE265_SOURCE and LIBHEIF_SOURCE name a changed source tree to link in
// place of the pinned release. The build applies no patch to such a tree.
const libraries = {
    libde265: { version: '1.1.3', checksum: '554228bd17788c99a7e63b37ab5634722190e6e2bf60c1dcb01cef328e133905', patches: [], override: process.env.LIBDE265_SOURCE },
    // The patch is upstream commit 215cbdad of 3 October 2026. Remove it when a release contains the commit.
    libheif: { version: '1.23.5', checksum: 'fd9036064c4432f0550d15072ddf34956a248279ee9aeaff0fba3fa0f77d8f1a', patches: ['libheif-bilinear-chroma-border.patch'], override: process.env.LIBHEIF_SOURCE },
};
const build = join(root, 'build'), dist = join(root, 'dist'), prefix = join(build, 'prefix');
mkdirSync(build, { recursive: true }); mkdirSync(join(dist, 'source/patches'), { recursive: true });
for (const [name, library] of Object.entries(libraries)) {
    library.archive = join(build, `${name}-${library.version}.tar.gz`);
    if (!existsSync(library.archive)) {
        const response = await fetch(`https://github.com/strukturag/${name}/releases/download/v${library.version}/${name}-${library.version}.tar.gz`);
        if (!response.ok) throw new Error(`${name} download: ${response.status}`);
        writeFileSync(library.archive, Buffer.from(await response.arrayBuffer()));
    }
    if (sha256(library.archive) !== library.checksum) throw new Error(`${name} source checksum mismatch`);
    library.source = join(build, `${name}-${library.version}`);
    library.objects = join(build, name);
    const patched = join(library.source, '.soundkit-patched');
    if (!existsSync(patched)) {
        execFileSync('tar', ['-xzf', library.archive, '-C', build]);
        for (const patch of library.patches) execFileSync('patch', ['-p1', '-i', join(root, 'patches', patch)], { cwd: library.source, stdio: 'inherit' });
        writeFileSync(patched, library.patches.join('\n'));
    }
    if (library.override) { library.source = resolve(library.override); library.objects = join(build, `${name}-changed`); }
}
const env = { ...process.env, EM_NODE_JS: process.execPath };
const run = (command, args) => execFileSync(command, args, { cwd: root, env, stdio: 'inherit' });
const jobs = String(Math.min(6, availableParallelism()));
// WebAssembly exceptions make a decode error recoverable, and a call that
// does not throw has no added cost. The build has one thread, so the module
// runs without SharedArrayBuffer.
const flags = '-fwasm-exceptions';
const library = (name, options, define = '') => {
    const { source, objects } = libraries[name];
    run('emcmake', ['cmake', '-S', source, '-B', objects, '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_SHARED_LIBS=OFF',
        `-DCMAKE_INSTALL_PREFIX=${prefix}`, `-DCMAKE_C_FLAGS=${flags}${define}`, `-DCMAKE_CXX_FLAGS=${flags}${define}`, ...options]);
    run('cmake', ['--build', objects, '--parallel', jobs]);
    run('cmake', ['--install', objects]);
};
library('libde265', ['-DENABLE_SDL=OFF', '-DENABLE_DECODER=OFF', '-DENABLE_ENCODER=OFF', '-DENABLE_SIMD=OFF']);
library('libheif', ['-DENABLE_PLUGIN_LOADING=OFF', '-DENABLE_MULTITHREADING_SUPPORT=OFF', '-DENABLE_PARALLEL_TILE_DECODING=OFF',
    '-DWITH_LIBDE265=ON', `-DLIBDE265_INCLUDE_DIR=${join(prefix, 'include')}`, `-DLIBDE265_LIBRARY=${join(prefix, 'lib/libde265.a')}`,
    ...['X265', 'KVAZAAR', 'UVG266', 'VVDEC', 'VVENC', 'X264', 'OpenH264_DECODER', 'DAV1D', 'AOM_DECODER', 'AOM_ENCODER', 'SvtEnc', 'RAV1E',
        'JPEG_DECODER', 'JPEG_ENCODER', 'OpenJPEG_ENCODER', 'OpenJPEG_DECODER', 'FFMPEG_DECODER', 'OPENJPH_ENCODER',
        'UNCOMPRESSED_CODEC', 'WEBCODECS', 'LIBSHARPYUV', 'HEADER_COMPRESSION', 'EXAMPLES', 'GDK_PIXBUF'].map(name => `-DWITH_${name}=OFF`),
    '-DBUILD_TESTING=OFF', '-DBUILD_DOCUMENTATION=OFF'],
    // libheif's switch for a build without its embind JavaScript interface.
    ' -D__EMSCRIPTEN_STANDALONE_WASM__=1');
const compiler = process.env.EMXX || 'em++';
const exported = ['malloc', 'free', 'skheif_decode', 'skheif_close', 'skheif_error', 'skheif_pixels', 'skheif_stride', 'skheif_width', 'skheif_height',
    'skheif_depth', 'skheif_metadata', 'skheif_profile', 'skheif_profile_size'].map(name => `"_${name}"`).join(',');
run(compiler, ['-O2', '-std=c++17', flags, `-I${join(prefix, 'include')}`, join(root, 'src/heif.cpp'), join(prefix, 'lib/libheif.a'), join(prefix, 'lib/libde265.a'),
    // src/heif.cpp gives the reason for the wrapped symbol.
    '-Wl,--wrap=de265_set_parameter_bool',
    '-sMODULARIZE=1', '-sEXPORT_ES6=1', '-sENVIRONMENT=web,worker,node', '-sALLOW_MEMORY_GROWTH=1', '-sMAXIMUM_MEMORY=2147483648',
    '-sSTACK_SIZE=1MB', '-sFILESYSTEM=0', `-sEXPORTED_FUNCTIONS=[${exported}]`, '-sEXPORTED_RUNTIME_METHODS=["HEAPU8","HEAPU16","UTF8ToString"]',
    '-o', join(dist, 'heif.mjs')]);
cpSync(join(root, 'src/index.mjs'), join(dist, 'index.mjs'));
cpSync(join(root, 'LICENSE'), join(dist, 'LICENSE'));
// The LGPL asks for the license texts, the library sources with the changes
// made to them, and the means to link the wrapper with a changed library.
const { libde265, libheif } = libraries;
const changed = Boolean(libde265.override || libheif.override);
const patches = Object.values(libraries).flatMap(library => library.override ? [] : library.patches);
for (const [name, library] of Object.entries(libraries)) {
    cpSync(join(library.source, 'COPYING'), join(dist, `COPYING.${name}`));
    cpSync(library.archive, join(dist, `${name}-source.tar.gz`));
}
for (const name of ['src/heif.cpp', 'src/index.mjs', 'scripts/build.mjs']) cpSync(join(root, name), join(dist, 'source', name.split('/').pop()));
for (const patch of patches) cpSync(join(root, 'patches', patch), join(dist, 'source/patches', patch));
writeFileSync(join(dist, 'NOTICE'), `This module contains libheif ${libheif.version} and libde265 ${libde265.version}.
libheif: https://github.com/strukturag/libheif (LGPL-3.0-or-later). Copyright (c) 2017-2025 Dirk Farin. Source SHA256 ${libheif.checksum}.
libde265: https://github.com/strukturag/libde265 (LGPL-3.0-or-later). Copyright (c) 2013-2014 struktur AG, Dirk Farin. Source SHA256 ${libde265.checksum}.
COPYING.libheif and COPYING.libde265 contain the GNU Lesser General Public License version 3 and the GNU General Public License version 3.
libheif-source.tar.gz and libde265-source.tar.gz contain the complete sources of the two releases.
${changed ? 'This build links a changed library source tree in place of a pinned release.'
        : `Wavey AI modified libheif on 4 October 2026: the build applies source/patches/${libheif.patches[0]} to the release. The patch is libheif commit 215cbdadef517b43b30889eb277ca1ba14b09bfa. The libde265 source is unmodified.`}
The SoundKit wrapper is MIT licensed (LICENSE). source/ contains the wrapper and its build script.
To link the wrapper with a changed library, run "LIBHEIF_SOURCE=<directory> LIBDE265_SOURCE=<directory> node scripts/build.mjs" in the soundkit-heif package.
`);
writeFileSync(join(dist, 'provenance.json'), JSON.stringify({ libheif: libheif.version, libheifSha256: libheif.checksum,
    libde265: libde265.version, libde265Sha256: libde265.checksum,
    patches: Object.fromEntries(patches.map(patch => [patch, sha256(join(root, 'patches', patch))])), changedSourceTree: changed,
    wasmSha256: sha256(join(dist, 'heif.wasm')),
    compiler: execFileSync(compiler, ['--version'], { env, encoding: 'utf8' }).split('\n')[0] }, null, 2) + '\n');
console.log(`HEIF WASM: ${(statSync(join(dist, 'heif.wasm')).size / 1048576).toFixed(2)} MB; libheif ${libheif.version}, libde265 ${libde265.version}`);
