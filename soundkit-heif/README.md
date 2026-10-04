# soundkit-heif

HEIC still-image decoding for browser applications.
The module contains libheif 1.23.5 and libde265 1.1.3, built from checksum-pinned release sources.
The build applies one patch to libheif.
Browser applications run this module in a dedicated worker.

## Build and test

The build requires Node.js, an Emscripten SDK (`emcmake` and `em++` on PATH), CMake 3.16 or later, `tar`, and `patch`.
The first build downloads the two release archives from GitHub.
Subsequent builds use the archives and objects in `build/`.

From the SoundKit repository:

```sh
npm install --ignore-scripts
npm run build:heif
npm test --workspace @wavey-ai/soundkit-heif
```

`soundkit-heif/dist/` contains these files:

- `index.mjs`, `heif.mjs` and `heif.wasm`: the ES module and the 1.3 MB decoder.
- `NOTICE`, `LICENSE`, `COPYING.libheif` and `COPYING.libde265`: the notices and license texts.
- `libheif-source.tar.gz` and `libde265-source.tar.gz`: the library sources.
- `source/`: the wrapper, the build script and the patch.
- `provenance.json`: the versions, the checksums and the compiler.

Serve the whole directory.
`heif.mjs` loads `heif.wasm` from its own directory.
The module uses one thread.
It operates without SharedArrayBuffer and without the COOP and COEP headers.
The decoder uses WebAssembly exception handling, which Chrome 95, Firefox 100 and Safari 15.2 introduced.

The tests use the files in `test/fixtures/`.
`node test/make-fixtures.mjs` makes these files again with the macOS `sips` command.

## Worker API

```js
import { createHeifDecoder, isHeif } from './dist/index.mjs';

const bytes = new Uint8Array(await file.arrayBuffer());
if (isHeif(bytes)) {
  const decoder = await createHeifDecoder();
  try {
    const image = decoder.decode(bytes);
    // image: { width, height, data, bitDepth, icc, hasAlpha, premultiplied, nclx }
  } finally {
    decoder.close();
  }
}
```

`isHeif(bytes)` reads the file type box.
It returns `true` for a file with an HEVC brand, such as `heic` or `heix`.
It returns `true` for a file with only the `mif1` or `msf1` brand.
It returns `false` for an AVIF file.

`createHeifDecoder(options)` loads the WASM and returns a decoder.
`options` goes to the Emscripten module, and can contain `locateFile` or `wasmBinary`.

`decoder.decode(bytes)` decodes the primary image of the file and returns these fields:

| Field | Value |
| --- | --- |
| `width`, `height` | The dimensions after rotation, mirror and crop. |
| `bitDepth` | The bit depth of the file: 8, 10 or 12. |
| `data` | Interleaved RGBA. A `Uint8ClampedArray` when `bitDepth` is 8. A `Uint16Array` with values from 0 to 1023 or 4095 when `bitDepth` is 10 or 12. |
| `icc` | The ICC profile of the image as a `Uint8Array`, or `null`. |
| `nclx` | `{ primaries, transfer, matrix, fullRange }` with the code points of the file, or `null`. |
| `hasAlpha` | `true` when the file has an opacity channel. Without one, each opacity value is the maximum. |
| `premultiplied` | `true` when the colors of the file are multiplied by the opacity. |

`decode` throws an `Error` when the file cannot be decoded.
The decoder stays usable after an error.
Each call releases its decoder buffers before it returns.
`decoder.close()` releases the WASM memory for garbage collection.

## Color and precision

The decoder applies the rotation, mirror and crop properties of the file.
An application must not apply the Exif orientation to the result.

The pixel values are in the color space of the file.
The decoder does not convert them to sRGB.
A photograph from an iPhone is in Display P3, and its profile is in `icc`.
`Engine.fromRGBA(width, height, data, icc)` and `Engine.fromRGBA16(width, height, data, bitDepth, icc)` in `soundkit-develop` read the pixels through the profile.

The decoder converts YCbCr to RGB with the matrix and range of the file.
It makes 4:2:0 and 4:2:2 chroma full size with bilinear interpolation.
The HEVC deblocking and sample adaptive offset filters are on.
A 10-bit or 12-bit file keeps its bit depth.

## Format coverage and limits

The decoder reads HEVC-coded HEIF images: single images, tiled grid images, and images with an opacity channel.
For a file with more than one image, it decodes the primary image.
It ignores depth maps, HDR gain maps, thumbnails, Exif and XMP.
It returns an error for images with other codecs: AV1, VVC, JPEG, JPEG 2000 and uncompressed.
It returns an error for an image of more than 80 megapixels.

The decoder returns the values of a PQ or HLG image without tone mapping.
The WASM memory limit is 2 GB.
libde265 uses its scalar code paths.

## libheif patch

`patches/libheif-bilinear-chroma-border.patch` is libheif commit 215cbdad of 3 October 2026.
libheif 1.23.5 reads the chroma of the outer pixel rows and columns from an incorrect position during bilinear interpolation.
The patch corrects the position.
Remove the patch from `scripts/build.mjs` when a libheif release contains the commit.

libheif turns the HEVC deblocking and sample adaptive offset filters off in an Emscripten build.
The build links with `--wrap=de265_set_parameter_bool`, and `src/heif.cpp` keeps the two filters on.
This method leaves the library sources as they are.

## Licenses

The SoundKit wrapper and build script are MIT licensed. See `LICENSE`.

libheif and libde265 are licensed under the GNU Lesser General Public License, version 3 or later.
The WASM file links the two libraries statically.
The distribution therefore contains:

- the notice in `dist/NOTICE`, which names the libraries, the license and the change to libheif;
- the LGPL and GPL texts in `dist/COPYING.libheif` and `dist/COPYING.libde265`;
- the complete library sources and the patch;
- the wrapper source and the build script in `dist/source/`.

To link the wrapper with a changed library, run this command in `soundkit-heif`:

```sh
LIBHEIF_SOURCE=<directory> LIBDE265_SOURCE=<directory> node scripts/build.mjs
```

Each variable is optional and names a source tree to use in place of the pinned release.

An application that gives `heif.wasm` to its users must also give them the notice, the license texts and the sources.
An application that serves the whole `dist/` directory and shows a link to `dist/NOTICE` does this.

HEVC is subject to patents that third parties hold.
The licenses of this package do not include a license to those patents.
