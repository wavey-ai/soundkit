# soundkit-raw

Camera RAW decoding and photographic development for browser applications.
LibRaw 0.22.0 is built from checksum-pinned, unmodified source. Browser applications
run this module in a dedicated worker.

## Build and test

Requires Node.js, an Emscripten SDK (`em++` on PATH), and `tar`. The first build
downloads LibRaw and Emscripten's JPEG/zlib ports. Subsequent builds reuse objects.

From the SoundKit repository:

```sh
npm install --ignore-scripts
npm run build:raw
npm test --workspace @wavey-ai/soundkit-raw
```

`soundkit-raw/dist/` contains the ES module, roughly 1 MB decoder WASM, processing kernel,
source provenance, and LibRaw's source and notices. Serve the whole directory.
The build does not require COOP/COEP or SharedArrayBuffer.

## Worker API

```js
import { createRawDecoder, linearPreview, develop } from './dist/index.mjs';

const decoder = await createRawDecoder();
try {
  const metadata = decoder.open(new Uint8Array(await file.arrayBuffer()));
  const jpeg = decoder.thumbnail(); // Optional, temporary camera preview.
  const frame = decoder.decode({ half: true });
  const preview = linearPreview(frame, 1400);
  const display = develop(preview, { exposure: .5, temperature: 10 });
  // display.data is RGBA Uint8ClampedArray; display.histogram has 256 bins.
} finally {
  decoder.close();
}
```

Reopen and decode with `half: false` for a full-resolution export. Use
`develop(frame, recipe, { bitDepth: 10 })` to obtain sRGB transfer-encoded,
full-range RGBA values (0–1023) in a `Uint16Array`, suitable for jSquash AVIF.
The package does not encode AVIF or manage application storage.

## Colour and precision

LibRaw removes sensor black levels, demosaics, and applies file orientation.
It returns camera RGB at 16-bit precision, with automatic scaling, white balance,
gamma, and integer colour conversion disabled. The development kernel normalizes
by the sensor white level, applies as-shot white balance and LibRaw's camera-to-sRGB
matrix in floating point. Values above white remain available to exposure and tone
adjustments before the final display transform. The original is never mutated.

Temperature/tint are relative to as-shot white balance, not absolute Kelvin.
Controls include exposure, highlights/shadows, whites/blacks, contrast, crop,
quarter-turn rotation, and straightening with an inscribed crop. Geometry always
uses proportional resizing and cropping. Cropped previews and exports share one
kernel. Tone/display lookup tables avoid expensive curves in each pixel operation.

This is an SDR sRGB workflow. It does not yet provide lens profiles, advanced
denoising, local masks, custom ICC/DCP profiles, HDR output, or reconstruction of
fully clipped sensor channels. Highlights adjusts retained data, not missing data.

## Camera coverage and limits

Uses LibRaw's built-in decoders, including CR2/CR3, NEF, ARW, RAF, and DNG, plus
zlib and JPEG. Support depends on the camera and compression mode, not just the
extension. Optional DNG SDK/JPEG XL, RawSpeed, LCMS, and GPL demosaic packs are not
linked. Unusual non-RGB sensor layouts fail explicitly. The wrapper limits RAW
images to 80 megapixels; applications should add their own input/memory limits.
Decode errors are recoverable by closing and reopening the decoder.

Tests generate their own 12-bit Bayer DNG: genuine LibRaw decoding, preserved
precision, neutral colour, exposure response, proxy/export agreement, rotation,
crop geometry, and error recovery.

## Licenses

Wrapper and development code: MIT. LibRaw: CDDL 1.0; the generated distribution
includes its license, copyright notices, and complete matching source archive.
JPEG and zlib are built using Emscripten's unmodified ports. See `LICENSE` and
`dist/NOTICE`.
