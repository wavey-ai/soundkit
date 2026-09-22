const DEFAULT_MIN_QUANTIZER = 0;
const DEFAULT_MAX_QUANTIZER = 63;
const DEFAULT_SPEED = 2;
const DEFAULT_DENOISE = 0;
const DEFAULT_BIT_DEPTH = 10;

function assertByte(value, name) {
  if (!Number.isInteger(value) || value < 0 || value > 255) {
    throw new RangeError(`${name} must be an integer byte`);
  }
}

function assertDimension(value, name) {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(`${name} must be a positive integer`);
  }
}

function normalizeRgbaBytes(input, width, height) {
  const bytes = (input instanceof Uint8Array || input instanceof Uint8ClampedArray) ? input : input?.data;
  if (!(bytes instanceof Uint8Array || bytes instanceof Uint8ClampedArray)) {
    throw new TypeError("expected RGBA bytes or ImageData-like { data }");
  }
  if (width * height > 80000000) throw new RangeError('Image exceeds the 80 megapixel limit');
  const expected = width * height * 4;
  if (bytes.length !== expected) {
    throw new RangeError(`expected ${expected} RGBA bytes, got ${bytes.length}`);
  }
  return bytes;
}

function clampQuantizer(value, fallback, name) {
  const q = value ?? fallback;
  if (!Number.isInteger(q) || q < 0 || q > 63) {
    throw new RangeError(`${name} must be an integer in the range 0..63`);
  }
  return q;
}

function clampSpeed(value) {
  const speed = value ?? DEFAULT_SPEED;
  if (!Number.isInteger(speed) || speed < 0 || speed > 10) {
    throw new RangeError("speed must be an integer in the range 0..10");
  }
  return speed;
}

function clampDenoise(value) {
  const denoise = value ?? DEFAULT_DENOISE;
  if (!Number.isInteger(denoise) || denoise < 0 || denoise > 50) {
    throw new RangeError("denoise must be an integer in the range 0..50");
  }
  return denoise;
}

function clampBitDepth(value) {
  const depth = value ?? DEFAULT_BIT_DEPTH;
  if (depth !== 8 && depth !== 10 && depth !== 12) {
    throw new RangeError("bitDepth must be 8, 10 or 12");
  }
  return depth;
}

export async function createAvifEncoder(moduleFactory, init = {}) {
  const factory = moduleFactory ?? (await import("./soundkit_avif.js")).default;
  if (typeof factory !== "function") {
    throw new TypeError("moduleFactory must be an Emscripten module factory");
  }

  const module = await factory(init);

  function lastError() {
    const ptr = module._skavif_last_error_ptr();
    return ptr ? module.UTF8ToString(ptr) : "AVIF encode failed";
  }

  function encode(input, options) {
    const width = options?.width ?? input?.width;
    const height = options?.height ?? input?.height;
    assertDimension(width, "width");
    assertDimension(height, "height");

    const quantizer = clampQuantizer(options?.quantizer, 32, "quantizer");
    const speed = clampSpeed(options?.speed);
    const denoise = clampDenoise(options?.denoise);
    const bitDepth = clampBitDepth(options?.bitDepth);
    const monochrome = options?.monochrome !== false;
    const rgba = normalizeRgbaBytes(input, width, height);

    const ptr = module._skavif_malloc(rgba.length);
    if (!ptr) {
      throw new Error("failed to allocate WASM input buffer");
    }

    try {
      module.HEAPU8.set(rgba, ptr);
      const ok = module._skavif_encode_rgba(
        ptr,
        width,
        height,
        quantizer,
        speed,
        monochrome ? 1 : 0,
        denoise,
        bitDepth,
      );
      if (!ok) {
        throw new Error(lastError());
      }
      const outPtr = module._skavif_output_ptr();
      const outSize = module._skavif_output_size();
      if (!outPtr || outSize <= 0) {
        throw new Error("encoder returned empty AVIF output");
      }
      return new Uint8Array(module.HEAPU8.subarray(outPtr, outPtr + outSize));
    } finally {
      module._skavif_free(ptr);
    }
  }

  function encodeForTarget(input, options) {
    const width = options?.width ?? input?.width;
    const height = options?.height ?? input?.height;
    assertDimension(width, "width");
    assertDimension(height, "height");

    const targetBytes = options?.targetBytes;
    if (!Number.isInteger(targetBytes) || targetBytes <= 0) {
      throw new RangeError("targetBytes must be a positive integer");
    }

    let low = clampQuantizer(options?.minQuantizer, DEFAULT_MIN_QUANTIZER, "minQuantizer");
    let high = clampQuantizer(options?.maxQuantizer, DEFAULT_MAX_QUANTIZER, "maxQuantizer");
    if (low > high) {
      throw new RangeError("minQuantizer must be <= maxQuantizer");
    }

    const speed = clampSpeed(options?.speed);
    const denoise = clampDenoise(options?.denoise);
    const bitDepth = clampBitDepth(options?.bitDepth);
    const monochrome = options?.monochrome !== false;

    let bestFit = null;
    let smallest = null;

    while (low <= high) {
      const quantizer = Math.floor((low + high) / 2);
      const bytes = encode(input, {
        width,
        height,
        quantizer,
        speed,
        denoise,
        bitDepth,
        monochrome,
      });
      const candidate = { bytes, quantizer, size: bytes.length, fits: bytes.length <= targetBytes };

      if (!smallest || candidate.size < smallest.size) {
        smallest = candidate;
      }

      if (candidate.fits) {
        bestFit = candidate;
        high = quantizer - 1;
      } else {
        low = quantizer + 1;
      }
    }

    return bestFit ?? { ...smallest, fits: false };
  }

  function rgbaToGreyscale(input, options = {}) {
    const width = options.width ?? input?.width;
    const height = options.height ?? input?.height;
    assertDimension(width, "width");
    assertDimension(height, "height");
    const rgba = normalizeRgbaBytes(input, width, height);
    const out = new Uint8Array(rgba.length);
    const rWeight = options.rWeight ?? 77;
    const gWeight = options.gWeight ?? 150;
    const bWeight = options.bWeight ?? 29;
    assertByte(rWeight, "rWeight");
    assertByte(gWeight, "gWeight");
    assertByte(bWeight, "bWeight");
    for (let i = 0; i < rgba.length; i += 4) {
      const y = (rgba[i] * rWeight + rgba[i + 1] * gWeight + rgba[i + 2] * bWeight + 128) >> 8;
      out[i] = y;
      out[i + 1] = y;
      out[i + 2] = y;
      out[i + 3] = rgba[i + 3];
    }
    return out;
  }

  return {
    module,
    encode,
    encodeForTarget,
    rgbaToGreyscale
  };
}
