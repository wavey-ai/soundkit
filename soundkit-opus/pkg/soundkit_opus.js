/* @ts-self-types="./soundkit_opus.d.ts" */

export class DecodeResult {
    static __wrap(ptr) {
        const obj = Object.create(DecodeResult.prototype);
        obj.__wbg_ptr = ptr;
        DecodeResultFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        DecodeResultFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_decoderesult_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    get decodedSize() {
        const ret = wasm.decoderesult_decodedSize(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {Int16Array}
     */
    get output() {
        const ret = wasm.decoderesult_output(this.__wbg_ptr);
        var v1 = getArrayI16FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 2, 2);
        return v1;
    }
}
if (Symbol.dispose) DecodeResult.prototype[Symbol.dispose] = DecodeResult.prototype.free;

export class DecodeResultI24 {
    static __wrap(ptr) {
        const obj = Object.create(DecodeResultI24.prototype);
        obj.__wbg_ptr = ptr;
        DecodeResultI24Finalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        DecodeResultI24Finalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_decoderesulti24_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    get decodedSize() {
        const ret = wasm.decoderesulti24_decodedSize(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {Int32Array}
     */
    get output() {
        const ret = wasm.decoderesulti24_output(this.__wbg_ptr);
        var v1 = getArrayI32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
}
if (Symbol.dispose) DecodeResultI24.prototype[Symbol.dispose] = DecodeResultI24.prototype.free;

export class Decoder {
    static __wrap(ptr) {
        const obj = Object.create(Decoder.prototype);
        obj.__wbg_ptr = ptr;
        DecoderFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        DecoderFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_decoder_free(ptr, 0);
    }
    /**
     * @param {Uint8Array | number[]} packet
     * @returns {DecodeResult}
     */
    dec_frame(packet) {
        const ptr0 = passArray8ToWasm0(packet, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.decoder_dec_frame(this.__wbg_ptr, ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return DecodeResult.__wrap(ret[0]);
    }
    /**
     * Decodes to interleaved floating-point samples.
     * @param {Uint8Array | number[]} packet
     * @returns {Float32Array}
     */
    dec_frame_f32(packet) {
        const ptr0 = passArray8ToWasm0(packet, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.decoder_dec_frame_f32(this.__wbg_ptr, ptr0, len0);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * @param {Uint8Array | number[]} packet
     * @returns {DecodeResultI24}
     */
    dec_frame_i24(packet) {
        const ptr0 = passArray8ToWasm0(packet, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.decoder_dec_frame_i24(this.__wbg_ptr, ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return DecodeResultI24.__wrap(ret[0]);
    }
    /**
     * @param {Uint8Array | number[]} packet
     * @returns {number}
     */
    dec_frame_i24_reuse(packet) {
        const ptr0 = passArray8ToWasm0(packet, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.decoder_dec_frame_i24_reuse(this.__wbg_ptr, ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] >>> 0;
    }
    /**
     * @param {Uint8Array | number[]} packet
     * @returns {number}
     */
    dec_frame_reuse(packet) {
        const ptr0 = passArray8ToWasm0(packet, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.decoder_dec_frame_reuse(this.__wbg_ptr, ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] >>> 0;
    }
    /**
     * Conceals one lost frame (packet-loss concealment) into the i16
     * output, for standard and Opus Custom decoders alike.
     * @returns {DecodeResult}
     */
    dec_lost() {
        const ret = wasm.decoder_dec_lost(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return DecodeResult.__wrap(ret[0]);
    }
    /**
     * @returns {number}
     */
    get decodedSize() {
        const ret = wasm.decoder_decodedSize(this.__wbg_ptr);
        return ret >>> 0;
    }
    destroy() {
        const ptr = this.__destroy_into_raw();
        wasm.decoder_destroy(ptr);
    }
    /**
     * @returns {boolean}
     */
    get isCustom() {
        const ret = wasm.decoder_isCustom(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Creates an Opus Custom decoder at 48 kHz for `frame_size`-sample
     * frames (48, 64, 72 for 1, 1.33, 1.5 ms). An empty packet conceals a
     * lost frame.
     * @param {number} channels
     * @param {number} frame_size
     * @returns {Decoder}
     */
    static newCustom(channels, frame_size) {
        const ret = wasm.decoder_newCustom(channels, frame_size);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return Decoder.__wrap(ret[0]);
    }
    /**
     * Creates a decoder. A `frame_size` that standard Opus cannot carry but
     * Opus Custom can (48, 64 and 72 samples; see `custom_frame_size` in the
     * Rust API) gives an Opus Custom decoder, as `newCustom` does. Any other
     * value gives the standard decoder, which reads every standard frame
     * duration whatever `frame_size` says.
     * @param {number} channels
     * @param {number} sample_rate
     * @param {number} frame_size
     */
    constructor(channels, sample_rate, frame_size) {
        const ret = wasm.decoder_new(channels, sample_rate, frame_size);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        DecoderFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @returns {number}
     */
    get outputI24Len() {
        const ret = wasm.decoder_outputI24Len(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get outputI24Ptr() {
        const ret = wasm.decoder_outputI24Ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get outputLen() {
        const ret = wasm.decoder_outputLen(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get outputPtr() {
        const ret = wasm.decoder_outputPtr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Selects libopus's handling of empty, short and damaged packets for a
     * standard decoder (off by default: such packets are errors, as they
     * always were). Opus Custom decoders always conceal.
     * @param {boolean} enabled
     */
    set_conceal_errors(enabled) {
        const ret = wasm.decoder_set_conceal_errors(this.__wbg_ptr, enabled);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Opus Custom: expects (`true`, the default) or omits the signalling
     * byte.
     * @param {boolean} enabled
     */
    set_custom_signalling(enabled) {
        const ret = wasm.decoder_set_custom_signalling(this.__wbg_ptr, enabled);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
}
if (Symbol.dispose) Decoder.prototype[Symbol.dispose] = Decoder.prototype.free;

export class EncodeResult {
    static __wrap(ptr) {
        const obj = Object.create(EncodeResult.prototype);
        obj.__wbg_ptr = ptr;
        EncodeResultFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        EncodeResultFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_encoderesult_free(ptr, 0);
    }
    /**
     * @returns {Uint8Array}
     */
    get encodedData() {
        const ret = wasm.encoderesult_encodedData(this.__wbg_ptr);
        var v1 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        return v1;
    }
    /**
     * @returns {boolean}
     */
    get ok() {
        const ret = wasm.encoderesult_ok(this.__wbg_ptr);
        return ret !== 0;
    }
}
if (Symbol.dispose) EncodeResult.prototype[Symbol.dispose] = EncodeResult.prototype.free;

export class Encoder {
    static __wrap(ptr) {
        const obj = Object.create(Encoder.prototype);
        obj.__wbg_ptr = ptr;
        EncoderFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        EncoderFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_encoder_free(ptr, 0);
    }
    destroy() {
        const ptr = this.__destroy_into_raw();
        wasm.encoder_destroy(ptr);
    }
    /**
     * @param {Int16Array | number[]} input
     * @returns {EncodeResult}
     */
    enc_frame(input) {
        const ptr0 = passArray16ToWasm0(input, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.encoder_enc_frame(this.__wbg_ptr, ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return EncodeResult.__wrap(ret[0]);
    }
    /**
     * @param {Float32Array | number[]} input
     * @returns {EncodeResult}
     */
    enc_frame_f32(input) {
        const ptr0 = passArrayF32ToWasm0(input, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.encoder_enc_frame_f32(this.__wbg_ptr, ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return EncodeResult.__wrap(ret[0]);
    }
    /**
     * @param {Int32Array | number[]} input
     * @returns {EncodeResult}
     */
    enc_frame_i24(input) {
        const ptr0 = passArray32ToWasm0(input, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.encoder_enc_frame_i24(this.__wbg_ptr, ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return EncodeResult.__wrap(ret[0]);
    }
    /**
     * Encodes signed 24-bit samples from this encoder's staged `i32` input.
     * @returns {number}
     */
    enc_frame_i24_reuse() {
        const ret = wasm.encoder_enc_frame_i24_reuse(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] >>> 0;
    }
    /**
     * Encodes the samples in this encoder's input storage into its output
     * storage and returns the packet size.
     *
     * JavaScript callers fill `inputPtr..inputLen`, then read the packet through
     * `outputPtr..outputLen`. Views must be refreshed after Wasm memory grows.
     * @returns {number}
     */
    enc_frame_reuse() {
        const ret = wasm.encoder_enc_frame_reuse(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] >>> 0;
    }
    /**
     * @returns {number}
     */
    get inputI24Len() {
        const ret = wasm.encoder_inputI24Len(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get inputI24Ptr() {
        const ret = wasm.encoder_inputI24Ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get inputLen() {
        const ret = wasm.encoder_inputLen(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get inputPtr() {
        const ret = wasm.encoder_inputPtr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {boolean}
     */
    get isCustom() {
        const ret = wasm.encoder_isCustom(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Creates an Opus Custom encoder at 48 kHz for `frame_size`-sample
     * frames: 48, 64 and 72 give 1, 1.33 and 1.5 ms frames. Packets are
     * the `opus_custom_encode` format (one signalling byte, no TOC).
     * Defaults follow libopus: the maximum bitrate (every packet fills the
     * maximum packet size) and CBR; call `set_bitrate` for a target rate.
     * @param {number} channels
     * @param {number} frame_size
     * @returns {Encoder}
     */
    static newCustom(channels, frame_size) {
        const ret = wasm.encoder_newCustom(channels, frame_size);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return Encoder.__wrap(ret[0]);
    }
    /**
     * Creates an encoder for `frame_size`-sample frames. A frame size that
     * standard Opus cannot carry but Opus Custom can (48, 64 and 72 samples
     * for 1, 1.33 and 1.5 ms; see `custom_frame_size` in the Rust API) gives
     * an Opus Custom encoder, as `newCustom` does, at `bitrate` in CBR; its
     * packets need a custom decoder of the same frame size. Other frame
     * sizes give the standard encoder.
     * @param {number} channels
     * @param {number} sample_rate
     * @param {number} bitrate
     * @param {number} frame_size
     */
    constructor(channels, sample_rate, bitrate, frame_size) {
        const ret = wasm.encoder_new(channels, sample_rate, bitrate, frame_size);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        EncoderFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @returns {number}
     */
    get outputLen() {
        const ret = wasm.encoder_outputLen(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get outputPtr() {
        const ret = wasm.encoder_outputPtr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Sets the bitrate in bits per second; -1 selects the maximum (Opus
     * Custom only).
     * @param {number} bitrate
     */
    set_bitrate(bitrate) {
        const ret = wasm.encoder_set_bitrate(this.__wbg_ptr, bitrate);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Opus Custom: writes (`true`, the default) or omits the signalling byte.
     * @param {boolean} enabled
     */
    set_custom_signalling(enabled) {
        const ret = wasm.encoder_set_custom_signalling(this.__wbg_ptr, enabled);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Selects libopus's encoder decision constants (`true`) or SoundKit's.
     * @param {boolean} enabled
     */
    set_libopus_decisions(enabled) {
        wasm.encoder_set_libopus_decisions(this.__wbg_ptr, enabled);
    }
    /**
     * Opus Custom: the largest packet, signalling byte included (2-1276).
     * @param {number} bytes
     */
    set_max_packet_bytes(bytes) {
        const ret = wasm.encoder_set_max_packet_bytes(this.__wbg_ptr, bytes);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * @param {boolean} enabled
     */
    set_vbr(enabled) {
        const ret = wasm.encoder_set_vbr(this.__wbg_ptr, enabled);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Selects constrained VBR (`true`, the default) or unconstrained VBR.
     * @param {boolean} constrained
     */
    set_vbr_constraint(constrained) {
        const ret = wasm.encoder_set_vbr_constraint(this.__wbg_ptr, constrained);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
}
if (Symbol.dispose) Encoder.prototype[Symbol.dispose] = Encoder.prototype.free;
function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg___wbindgen_throw_4c88d8bf264fd984: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbindgen_generic_0000000000000001: function(arg0, arg1) {
            // Cast intrinsic for `Ref(String) -> Externref`.
            const ret = getStringFromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            table.set(0, undefined);
            table.set(1024, undefined);
            table.set(1025, null);
            table.set(1026, true);
            table.set(1027, false);
        },
    };
    return {
        __proto__: null,
        "./soundkit_opus_bg.js": import0,
    };
}

const DecodeResultFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_decoderesult_free(ptr, 1));
const DecodeResultI24Finalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_decoderesulti24_free(ptr, 1));
const DecoderFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_decoder_free(ptr, 1));
const EncodeResultFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_encoderesult_free(ptr, 1));
const EncoderFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_encoder_free(ptr, 1));

function getArrayF32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function getArrayI16FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getInt16ArrayMemory0().subarray(ptr / 2, ptr / 2 + len);
}

function getArrayI32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getInt32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function getArrayU8FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint8ArrayMemory0().subarray(ptr / 1, ptr / 1 + len);
}

let cachedFloat32ArrayMemory0 = null;
function getFloat32ArrayMemory0() {
    if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.byteLength === 0) {
        cachedFloat32ArrayMemory0 = new Float32Array(wasm.memory.buffer);
    }
    return cachedFloat32ArrayMemory0;
}

let cachedInt16ArrayMemory0 = null;
function getInt16ArrayMemory0() {
    if (cachedInt16ArrayMemory0 === null || cachedInt16ArrayMemory0.byteLength === 0) {
        cachedInt16ArrayMemory0 = new Int16Array(wasm.memory.buffer);
    }
    return cachedInt16ArrayMemory0;
}

let cachedInt32ArrayMemory0 = null;
function getInt32ArrayMemory0() {
    if (cachedInt32ArrayMemory0 === null || cachedInt32ArrayMemory0.byteLength === 0) {
        cachedInt32ArrayMemory0 = new Int32Array(wasm.memory.buffer);
    }
    return cachedInt32ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return decodeText(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

let cachedUint16ArrayMemory0 = null;
function getUint16ArrayMemory0() {
    if (cachedUint16ArrayMemory0 === null || cachedUint16ArrayMemory0.byteLength === 0) {
        cachedUint16ArrayMemory0 = new Uint16Array(wasm.memory.buffer);
    }
    return cachedUint16ArrayMemory0;
}

let cachedUint32ArrayMemory0 = null;
function getUint32ArrayMemory0() {
    if (cachedUint32ArrayMemory0 === null || cachedUint32ArrayMemory0.byteLength === 0) {
        cachedUint32ArrayMemory0 = new Uint32Array(wasm.memory.buffer);
    }
    return cachedUint32ArrayMemory0;
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function passArray16ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 2, 2) >>> 0;
    getUint16ArrayMemory0().set(arg, ptr / 2);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArray32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getUint32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArray8ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 1, 1) >>> 0;
    getUint8ArrayMemory0().set(arg, ptr / 1);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArrayF32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getFloat32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_externrefs.get(idx);
    wasm.__externref_table_dealloc(idx);
    return value;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(bytes) {
    numBytesDecoded += bytes.length;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true });
        cachedTextDecoder.decode();
        numBytesDecoded = bytes.length;
    }
    return cachedTextDecoder.decode(bytes);
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasmInstance, wasm;
function __wbg_finalize_init(instance, module) {
    wasmInstance = instance;
    wasm = instance.exports;
    wasmModule = module;
    cachedFloat32ArrayMemory0 = null;
    cachedInt16ArrayMemory0 = null;
    cachedInt32ArrayMemory0 = null;
    cachedUint16ArrayMemory0 = null;
    cachedUint32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (!module.ok) {
            throw new Error(`failed to fetch Wasm: ${module.status} ${module.statusText} fetching '${module.url}'`);
        }

        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

let __wbg_init_promise;
async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;
    if (__wbg_init_promise === undefined) {
        __wbg_init_promise = __wbg_init_once(module_or_path);
        __wbg_init_promise.catch(() => { __wbg_init_promise = undefined; });
    }
    return __wbg_init_promise;
}

async function __wbg_init_once(module_or_path) {

    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('soundkit_opus_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
