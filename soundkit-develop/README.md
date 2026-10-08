# soundkit-develop

Photographic development for SoundKit. The crate takes a linear frame and a
develop recipe and gives display pixels. The same Rust code runs natively and
in the browser through WebAssembly. In the browser, a WebGPU path runs the same
plan on the GPU. The WebAssembly engine is the fallback.

```
soundkit-develop/
  src/develop.rs     frames, the plan, the per-pixel pass, the CPU runner
  src/gpu.rs         the plan laid out for the GPU path
  src/filter.rs      box blurs, Gaussian, guided filter
  src/highlight.rs   highlight reconstruction of RAW frames
  src/curve.rs       monotone point curves
  src/colour.rs      working space, white balance, OKLab, gamut map
  src/recipe.rs      the recipe, read from JSON
  src/wasm.rs        the browser API (Engine, GpuPlan)
  gpu/develop.wgsl   the GPU passes
  gpu/index.mjs      the WebGPU host
  test/              Node, browser and benchmark scripts
```

## Build

```
npm run build:develop      # pkg/ (web) and pkg-node/ (Node)
```

A browser application serves `pkg/soundkit_develop.js`,
`pkg/soundkit_develop_bg.wasm`, `gpu/index.mjs` and `gpu/develop.wgsl`. The
WGSL file must stay next to `index.mjs`.

## Tests

```
cargo test -p soundkit-develop
node --test soundkit-develop/test/*.test.mjs
node soundkit-develop/test/gpu.browser.mjs     # GPU against WebAssembly, in Chromium
node soundkit-develop/test/bench.mjs           # WebAssembly CPU time
```

`gpu.browser.mjs` develops a photograph, a RAW frame and a proxy with each
recipe on both paths. It runs on the machine's GPU and on SwiftShader. A case
passes when no channel differs by more than 3 levels, the mean difference is
below 0.05 levels and the histograms and clipping counts agree.

## Browser API

```js
import init, { Engine } from './soundkit_develop.js';
import { createGpuDeveloper } from './gpu/index.mjs';

await init({ module_or_path: './soundkit_develop_bg.wasm' });
const engine = Engine.fromU16(width, height, samples, JSON.stringify({ scale, wb, matrix, daylight }));
const preview = engine.preview(1400);
const gpu = await createGpuDeveloper();           // null without a hardware GPU
const result = (gpu && await gpu.develop(preview, recipe, { edge: 1200 }))
    ?? preview.develop(JSON.stringify(recipe), JSON.stringify({ edge: 1200 }));
```

`gpu.develop` returns `{ width, height, bitDepth, data, histogram,
histogramRGB, clippedHigh, clippedLow }`. It returns null when the picture does
not fit the device's limits. It throws when the device fails. Use the
WebAssembly engine in both cases. `createGpuDeveloper` declines a software
adapter unless `allowSoftware` is set, because a software adapter is not faster
than WebAssembly.

## Pipeline

A recipe is resolved into a `Plan` once per develop. The plan holds every
matrix, table and low-resolution plane. The CPU runner and the GPU shader then
run the plan for each pixel in the same order:

1. Read the stored sample and take it to the working space: scale, white
   balance, camera matrix, the recipe's white balance change and exposure.
2. Dehaze, from the dark channel on a grid of about 160 cells.
3. Local tone in log luminance: Shadows, Highlights, Clarity, Texture,
   sharpening and luminance noise reduction. Then colour noise reduction.
4. Tone: the scene tone curve for a RAW frame, or the display curve for a
   photograph.
5. The gamut map into sRGB, at constant hue.
6. The parametric curve, then the point curves.
7. The colour controls in OKLab: the colour mixer, vibrance, saturation and
   colour grading.
8. Vignette and grain, then quantisation.

### Working space

The working space is linear Rec.2020 with the sRGB white. A saturated colour
from a camera keeps its hue through every edit. The final gamut map lowers its
chroma at constant lightness and hue until it fits sRGB.

### Highlight reconstruction

`from_sensor` rebuilds clipped channels of a RAW frame before any edit. The
reference colour is the white-balanced log ratio of red and blue to green on
bright unclipped pixels, on a grid of 8-pixel cells. A push-pull pyramid fills
the cells inside large clipped areas. A clipped channel takes the value that
the reference colour gives at the brightness of the unclipped channels. A pixel
with all channels clipped becomes neutral. When rebuilt values pass 16 bits,
the samples are halved and the scale doubles.

### Local tone

Shadows and Highlights read an edge-aware base: a guided filter of log
luminance with a radius of 2% of the long edge. Clarity reads a second guided
filter at 1%. Both are solved on a grid of cells a quarter of the radius wide,
over the whole output, so a 100% region reads the same values as the whole
picture. The planes are cached between develops. Exposure moves the planes by
a fixed amount, so a change of Exposure does not solve them again.

Texture uses a Gaussian of 0.25% of the long edge. Luminance noise reduction
uses a guided filter whose radius and epsilon increase with the slider. These
filters run over a window that is the region plus their reach.

### Tone

A RAW frame holds scene light. The scene tone curve maps luminance in stops
from middle grey to display luminance. Contrast changes the slope about grey.
A shoulder takes the highlights smoothly to white. Whites changes the upper
tones and Blacks changes the toe. The colour keeps its ratios, and a colour
past white goes to white at its own hue.

A photograph is already made for a display. With the default recipe it is
returned unchanged. Contrast is a curve about middle grey on each channel,
Whites is a gain and Blacks is an offset.

### Curves

The parametric curve has four regions. The point curves are the master curve
and one curve each for red, green and blue. Points are pairs from 0 to 255.
The curve through them is a monotone cubic (Fritsch and Carlson), so it never
overshoots between points.

### Colour mixer

The eight bands sit at the OKLCh hues of the sRGB colours at HSV hues 0, 30,
60, 120, 180, 225, 270 and 315 degrees. A hue between two bands is shared
between them along a raised cosine.
