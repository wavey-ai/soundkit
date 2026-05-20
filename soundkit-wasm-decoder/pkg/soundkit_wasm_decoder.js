/* @ts-self-types="./soundkit_wasm_decoder.d.ts" */

export class WasmAacDeboxer {
    static __wrap(ptr) {
        const obj = Object.create(WasmAacDeboxer.prototype);
        obj.__wbg_ptr = ptr;
        WasmAacDeboxerFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmAacDeboxerFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmaacdeboxer_free(ptr, 0);
    }
    /**
     * Final drain call. The deboxer should not be reused after this.
     * @returns {Array<any>}
     */
    flush() {
        const ret = wasm.wasmaacdeboxer_flush(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    constructor() {
        const ret = wasm.wasmaacdeboxer_new();
        this.__wbg_ptr = ret;
        WasmAacDeboxerFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @returns {WasmAacDeboxer}
     */
    static newAuto() {
        const ret = wasm.wasmaacdeboxer_newAuto();
        return WasmAacDeboxer.__wrap(ret);
    }
    /**
     * @param {string} format
     * @returns {WasmAacDeboxer}
     */
    static newWithFormat(format) {
        const ptr0 = passStringToWasm0(format, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmaacdeboxer_newWithFormat(ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmAacDeboxer.__wrap(ret[0]);
    }
    /**
     * Push arbitrary MP4/M4A bytes and receive AAC config/packet events.
     *
     * Packet events contain ADTS AAC frames in `data` and the original MP4
     * access unit in `rawData`.
     * @param {Uint8Array} bytes
     * @returns {Array<any>}
     */
    push(bytes) {
        const ptr0 = passArray8ToWasm0(bytes, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmaacdeboxer_push(this.__wbg_ptr, ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
}
if (Symbol.dispose) WasmAacDeboxer.prototype[Symbol.dispose] = WasmAacDeboxer.prototype.free;

export class WasmAudioTrackDemuxer {
    static __wrap(ptr) {
        const obj = Object.create(WasmAudioTrackDemuxer.prototype);
        obj.__wbg_ptr = ptr;
        WasmAudioTrackDemuxerFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmAudioTrackDemuxerFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmaudiotrackdemuxer_free(ptr, 0);
    }
    /**
     * Final drain call. The demuxer should not be reused after this.
     * @returns {Array<any>}
     */
    flush() {
        const ret = wasm.wasmaudiotrackdemuxer_flush(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    constructor() {
        const ret = wasm.wasmaudiotrackdemuxer_new();
        this.__wbg_ptr = ret;
        WasmAudioTrackDemuxerFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @returns {WasmAudioTrackDemuxer}
     */
    static newAuto() {
        const ret = wasm.wasmaudiotrackdemuxer_newAuto();
        return WasmAudioTrackDemuxer.__wrap(ret);
    }
    /**
     * @param {string} format
     * @returns {WasmAudioTrackDemuxer}
     */
    static newWithFormat(format) {
        const ptr0 = passStringToWasm0(format, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmaudiotrackdemuxer_newWithFormat(ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmAudioTrackDemuxer.__wrap(ret[0]);
    }
    /**
     * Push arbitrary container bytes and receive audio-track config/packet events.
     * @param {Uint8Array} bytes
     * @returns {Array<any>}
     */
    push(bytes) {
        const ptr0 = passArray8ToWasm0(bytes, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmaudiotrackdemuxer_push(this.__wbg_ptr, ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
}
if (Symbol.dispose) WasmAudioTrackDemuxer.prototype[Symbol.dispose] = WasmAudioTrackDemuxer.prototype.free;

export class WasmMusicDecoder {
    static __wrap(ptr) {
        const obj = Object.create(WasmMusicDecoder.prototype);
        obj.__wbg_ptr = ptr;
        WasmMusicDecoderFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmMusicDecoderFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmmusicdecoder_free(ptr, 0);
    }
    /**
     * Final EOF/drain call. The decoder should not be reused after this.
     * @returns {Array<any>}
     */
    flush() {
        const ret = wasm.wasmmusicdecoder_flush(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    constructor() {
        const ret = wasm.wasmmusicdecoder_new();
        this.__wbg_ptr = ret;
        WasmMusicDecoderFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @returns {WasmMusicDecoder}
     */
    static newAuto() {
        const ret = wasm.wasmmusicdecoder_newAuto();
        return WasmMusicDecoder.__wrap(ret);
    }
    /**
     * @param {number} sample_rate
     * @param {number} channels
     * @returns {WasmMusicDecoder}
     */
    static newRawLinear16(sample_rate, channels) {
        const ret = wasm.wasmmusicdecoder_newRawLinear16(sample_rate, channels);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmMusicDecoder.__wrap(ret[0]);
    }
    /**
     * @param {number} sample_rate
     * @param {number} channels
     * @returns {WasmMusicDecoder}
     */
    static newRawLinear32(sample_rate, channels) {
        const ret = wasm.wasmmusicdecoder_newRawLinear32(sample_rate, channels);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmMusicDecoder.__wrap(ret[0]);
    }
    /**
     * @param {string} format
     * @returns {WasmMusicDecoder}
     */
    static newWithFormat(format) {
        const ptr0 = passStringToWasm0(format, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmmusicdecoder_newWithFormat(ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmMusicDecoder.__wrap(ret[0]);
    }
    /**
     * Push arbitrary encoded bytes and receive all PCM frames currently available.
     *
     * This method drains decoder output after each push. Use `flush()` once at EOF
     * to force final container/codec drain.
     * @param {Uint8Array} bytes
     * @returns {Array<any>}
     */
    push(bytes) {
        const ptr0 = passArray8ToWasm0(bytes, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmmusicdecoder_push(this.__wbg_ptr, ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
}
if (Symbol.dispose) WasmMusicDecoder.prototype[Symbol.dispose] = WasmMusicDecoder.prototype.free;

export class WasmOpusDeboxer {
    static __wrap(ptr) {
        const obj = Object.create(WasmOpusDeboxer.prototype);
        obj.__wbg_ptr = ptr;
        WasmOpusDeboxerFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmOpusDeboxerFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmopusdeboxer_free(ptr, 0);
    }
    /**
     * Final drain call. The deboxer should not be reused after this.
     * @returns {Array<any>}
     */
    flush() {
        const ret = wasm.wasmopusdeboxer_flush(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    constructor() {
        const ret = wasm.wasmopusdeboxer_new();
        this.__wbg_ptr = ret;
        WasmOpusDeboxerFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @returns {WasmOpusDeboxer}
     */
    static newAuto() {
        const ret = wasm.wasmopusdeboxer_newAuto();
        return WasmOpusDeboxer.__wrap(ret);
    }
    /**
     * @param {string} format
     * @returns {WasmOpusDeboxer}
     */
    static newWithFormat(format) {
        const ptr0 = passStringToWasm0(format, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmopusdeboxer_newWithFormat(ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmOpusDeboxer.__wrap(ret[0]);
    }
    /**
     * Push arbitrary container bytes and receive Opus config/packet events.
     *
     * Packet events contain encoded Opus packet bytes suitable for a JS Opus
     * decoder. Config events carry channel/sample-rate/pre-skip metadata.
     * @param {Uint8Array} bytes
     * @returns {Array<any>}
     */
    push(bytes) {
        const ptr0 = passArray8ToWasm0(bytes, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmopusdeboxer_push(this.__wbg_ptr, ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
}
if (Symbol.dispose) WasmOpusDeboxer.prototype[Symbol.dispose] = WasmOpusDeboxer.prototype.free;
function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg___wbindgen_throw_9c31b086c2b26051: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbg_new_02d162bc6cf02f60: function() {
            const ret = new Object();
            return ret;
        },
        __wbg_new_310879b66b6e95e1: function() {
            const ret = new Array();
            return ret;
        },
        __wbg_new_from_slice_269e35316ed2d061: function(arg0, arg1) {
            const ret = new Uint8Array(getArrayU8FromWasm0(arg0, arg1));
            return ret;
        },
        __wbg_push_b77c476b01548d0a: function(arg0, arg1) {
            const ret = arg0.push(arg1);
            return ret;
        },
        __wbg_set_a0e911be3da02782: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = Reflect.set(arg0, arg1, arg2);
            return ret;
        }, arguments); },
        __wbindgen_cast_0000000000000001: function(arg0) {
            // Cast intrinsic for `F64 -> Externref`.
            const ret = arg0;
            return ret;
        },
        __wbindgen_cast_0000000000000002: function(arg0, arg1) {
            // Cast intrinsic for `Ref(String) -> Externref`.
            const ret = getStringFromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
    };
    return {
        __proto__: null,
        "./soundkit_wasm_decoder_bg.js": import0,
    };
}

const WasmAacDeboxerFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmaacdeboxer_free(ptr, 1));
const WasmAudioTrackDemuxerFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmaudiotrackdemuxer_free(ptr, 1));
const WasmMusicDecoderFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmmusicdecoder_free(ptr, 1));
const WasmOpusDeboxerFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmopusdeboxer_free(ptr, 1));

function addToExternrefTable0(obj) {
    const idx = wasm.__externref_table_alloc();
    wasm.__wbindgen_externrefs.set(idx, obj);
    return idx;
}

function getArrayU8FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint8ArrayMemory0().subarray(ptr / 1, ptr / 1 + len);
}

function getStringFromWasm0(ptr, len) {
    return decodeText(ptr >>> 0, len);
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function handleError(f, args) {
    try {
        return f.apply(this, args);
    } catch (e) {
        const idx = addToExternrefTable0(e);
        wasm.__wbindgen_exn_store(idx);
    }
}

function passArray8ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 1, 1) >>> 0;
    getUint8ArrayMemory0().set(arg, ptr / 1);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_externrefs.get(idx);
    wasm.__externref_table_dealloc(idx);
    return value;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    };
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasmInstance, wasm;
function __wbg_finalize_init(instance, module) {
    wasmInstance = instance;
    wasm = instance.exports;
    wasmModule = module;
    cachedUint8ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && expectedResponseType(module.type);

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

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('soundkit_wasm_decoder_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
