# soundkit-avif

AVIF encoding and decoding for browser workers. jSquash 2.1.1 provides
pixel-buffer encoding, decoding and a checked 10-bit sRGB path. A libavif
1.2.1/libaom backend provides fixed-quantizer and target-byte encoding,
including monochrome YUV400 output.

## Build and test

Requires Node.js, Emscripten (`emcmake`, `emcc`) and CMake 3.24+.
From the SoundKit repository:

```sh
npm install --ignore-scripts
npm run build:avif
npm test --workspace @wavey-ai/soundkit-avif
```

Serve the entire `soundkit-avif/dist/` directory. WASM files resolve relative
to their modules; `initEncoder` and `initDecoder` accept Emscripten options,
including `locateFile` or `wasmBinary`.

## API

```js
import { encode, encodeSRGB, decode } from './dist/index.mjs';

const avif = await encode(rgba8, { quality: 80, speed: 8 });
const image = await decode(avif);
const tagged = await encodeSRGB(rgba10, { quality: 90, speed: 8 });
```

Inputs have `width`, `height` and interleaved RGBA `data`. Eight-bit input uses
`Uint8Array` or `Uint8ClampedArray`; 10- and 12-bit input uses `Uint16Array`
with values in 0–1023 or 0–4095. `encode` returns an `ArrayBuffer`.
Encoding accepts images up to 80 megapixels and preserves dimensions.

`encodeSRGB` expects sRGB transfer-encoded, full-range 10-bit samples and
returns a `Uint8Array`. It uses 4:4:4 chroma and writes matching sRGB colour
descriptors into the AVIF container and AV1 sequence header. It preserves
the encoder's YUV matrix, sample range, payload sizes and offsets. The
colour-header parser is verified against the pinned backend version.

Decoding defaults to eight-bit RGBA. Pass `{ bitDepth: 10 }` or
`{ bitDepth: 12 }` as the second argument for high-bit-depth samples.
Eight-bit decoding uses the browser's `ImageData` constructor.

`encode.js` and `decode.js` also expose a default codec function and `init`.
Applications choose quality, speed and export settings.

## Quantizer and target-byte API

```js
import { createAvifEncoder } from './dist/target/index.mjs';

const encoder = await createAvifEncoder();
const result = encoder.encodeForTarget(rgba8, {
  width: 576, height: 576, targetBytes: 6800,
  monochrome: false, speed: 5,
});
// result: { bytes, size, quantizer, fits }
```

The package subpath is `@wavey-ai/soundkit-avif/target`. `encode` accepts a
fixed `quantizer` from 0 (best) to 63. `encodeForTarget` searches that range
for a candidate within the requested byte budget and reports `fits: false`
when the budget cannot be reached. Width and height stay unchanged.

This backend accepts eight-bit RGBA, produces opaque output and defaults
to monochrome, speed 2 and 10-bit AVIF. Set `monochrome: false` for YUV444.
`denoise` (0–50) enables AV1 film-grain synthesis; it defaults to zero.
Use the pixel-buffer API above for transparency and 10/12-bit source data.

## Licenses

SoundKit wrapper: MIT. jSquash: Apache-2.0. The generated distribution
includes `LICENSE` and `LICENSE.jsquash`, and records WASM hashes and the
backend version in `provenance.json`.

The libavif adapter retains its PolyForm Noncommercial 1.0.0 license in
`libavif/LICENSE`. Its distribution includes that license and the libavif
and libaom notices under `dist/target/`.
