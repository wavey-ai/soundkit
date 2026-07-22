/* @ts-self-types="./soundkit_wasm.d.ts" */

export class WasmAudioContentCipher {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmAudioContentCipherFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmaudiocontentcipher_free(ptr, 0);
    }
    /**
     * @param {Uint8Array} key
     */
    constructor(key) {
        const ptr0 = passArray8ToWasm0(key, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmaudiocontentcipher_new(ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        WasmAudioContentCipherFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {number} expected_key_epoch
     * @param {Uint8Array} envelope
     * @param {Uint8Array} authenticated_data
     * @returns {Uint8Array}
     */
    open(expected_key_epoch, envelope, authenticated_data) {
        const ptr0 = passArray8ToWasm0(envelope, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArray8ToWasm0(authenticated_data, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmaudiocontentcipher_open(this.__wbg_ptr, expected_key_epoch, ptr0, len0, ptr1, len1);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @param {number} key_epoch
     * @param {Uint8Array} nonce
     * @param {Uint8Array} plaintext
     * @param {Uint8Array} authenticated_data
     * @returns {Uint8Array}
     */
    seal(key_epoch, nonce, plaintext, authenticated_data) {
        const ptr0 = passArray8ToWasm0(nonce, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArray8ToWasm0(plaintext, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passArray8ToWasm0(authenticated_data, wasm.__wbindgen_malloc);
        const len2 = WASM_VECTOR_LEN;
        const ret = wasm.wasmaudiocontentcipher_seal(this.__wbg_ptr, key_epoch, ptr0, len0, ptr1, len1, ptr2, len2);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
}
if (Symbol.dispose) WasmAudioContentCipher.prototype[Symbol.dispose] = WasmAudioContentCipher.prototype.free;

/**
 * Opens the endpoint-specific envelope that transports an audio content key.
 *
 * The wrapping key comes from P-256 ECDH and HKDF-SHA256. The caller supplies
 * the canonical key-exchange context as additional authenticated data.
 */
export class WasmAudioContentKeyUnwrapper {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmAudioContentKeyUnwrapperFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmaudiocontentkeyunwrapper_free(ptr, 0);
    }
    /**
     * @param {Uint8Array} key
     */
    constructor(key) {
        const ptr0 = passArray8ToWasm0(key, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmaudiocontentkeyunwrapper_new(ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        WasmAudioContentKeyUnwrapperFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {Uint8Array} nonce
     * @param {Uint8Array} ciphertext
     * @param {Uint8Array} authenticated_data
     * @returns {Uint8Array}
     */
    open(nonce, ciphertext, authenticated_data) {
        const ptr0 = passArray8ToWasm0(nonce, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArray8ToWasm0(ciphertext, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passArray8ToWasm0(authenticated_data, wasm.__wbindgen_malloc);
        const len2 = WASM_VECTOR_LEN;
        const ret = wasm.wasmaudiocontentkeyunwrapper_open(this.__wbg_ptr, ptr0, len0, ptr1, len1, ptr2, len2);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @param {Uint8Array} nonce
     * @param {Uint8Array} plaintext
     * @param {Uint8Array} authenticated_data
     * @returns {Uint8Array}
     */
    seal(nonce, plaintext, authenticated_data) {
        const ptr0 = passArray8ToWasm0(nonce, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArray8ToWasm0(plaintext, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passArray8ToWasm0(authenticated_data, wasm.__wbindgen_malloc);
        const len2 = WASM_VECTOR_LEN;
        const ret = wasm.wasmaudiocontentkeyunwrapper_seal(this.__wbg_ptr, ptr0, len0, ptr1, len1, ptr2, len2);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
}
if (Symbol.dispose) WasmAudioContentKeyUnwrapper.prototype[Symbol.dispose] = WasmAudioContentKeyUnwrapper.prototype.free;

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

export class WasmSoundKitFrameDecoder {
    static __wrap(ptr) {
        const obj = Object.create(WasmSoundKitFrameDecoder.prototype);
        obj.__wbg_ptr = ptr;
        WasmSoundKitFrameDecoderFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmSoundKitFrameDecoderFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmsoundkitframedecoder_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    bufferedBytes() {
        const ret = wasm.wasmsoundkitframedecoder_bufferedBytes(this.__wbg_ptr);
        return ret >>> 0;
    }
    clearKey() {
        wasm.wasmsoundkitframedecoder_clearKey(this.__wbg_ptr);
    }
    finish() {
        const ret = wasm.wasmsoundkitframedecoder_finish(this.__wbg_ptr);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    constructor() {
        const ret = wasm.wasmsoundkitframedecoder_new();
        this.__wbg_ptr = ret;
        WasmSoundKitFrameDecoderFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @returns {WasmSoundKitFrameDecoder}
     */
    static newUnencrypted() {
        const ret = wasm.wasmsoundkitframedecoder_newUnencrypted();
        return WasmSoundKitFrameDecoder.__wrap(ret);
    }
    /**
     * @param {string} key
     * @returns {WasmSoundKitFrameDecoder}
     */
    static newWithDecimalKey(key) {
        const ptr0 = passStringToWasm0(key, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmsoundkitframedecoder_newWithDecimalKey(ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmSoundKitFrameDecoder.__wrap(ret[0]);
    }
    /**
     * @param {Uint8Array} key
     * @returns {WasmSoundKitFrameDecoder}
     */
    static newWithKeyBytes(key) {
        const ptr0 = passArray8ToWasm0(key, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmsoundkitframedecoder_newWithKeyBytes(ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmSoundKitFrameDecoder.__wrap(ret[0]);
    }
    /**
     * @param {Uint8Array} bytes
     * @returns {Array<any>}
     */
    push(bytes) {
        const ptr0 = passArray8ToWasm0(bytes, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmsoundkitframedecoder_push(this.__wbg_ptr, ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    reset() {
        wasm.wasmsoundkitframedecoder_reset(this.__wbg_ptr);
    }
    /**
     * @param {string} key
     */
    setDecimalKey(key) {
        const ptr0 = passStringToWasm0(key, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmsoundkitframedecoder_setDecimalKey(this.__wbg_ptr, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * @param {Uint8Array} key
     */
    setKeyBytes(key) {
        const ptr0 = passArray8ToWasm0(key, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmsoundkitframedecoder_setKeyBytes(this.__wbg_ptr, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
}
if (Symbol.dispose) WasmSoundKitFrameDecoder.prototype[Symbol.dispose] = WasmSoundKitFrameDecoder.prototype.free;

/**
 * @param {string} session_context
 * @param {string} transport_session_id
 * @param {number} config_generation
 * @param {string} epoch_id
 * @param {string} pts_samples
 * @param {number} sample_rate
 * @param {number} frame_count
 * @param {number} group_count
 * @param {number} group_id
 * @param {number} group_index
 * @param {number} channel_start
 * @param {number} channel_count
 * @param {number} payload_kind
 * @param {number} sample_format
 * @param {number} flags
 * @returns {Uint8Array}
 */
export function buildAudioGroupAssociatedData(session_context, transport_session_id, config_generation, epoch_id, pts_samples, sample_rate, frame_count, group_count, group_id, group_index, channel_start, channel_count, payload_kind, sample_format, flags) {
    const ptr0 = passStringToWasm0(session_context, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passStringToWasm0(transport_session_id, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passStringToWasm0(epoch_id, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passStringToWasm0(pts_samples, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len3 = WASM_VECTOR_LEN;
    const ret = wasm.buildAudioGroupAssociatedData(ptr0, len0, ptr1, len1, config_generation, ptr2, len2, ptr3, len3, sample_rate, frame_count, group_count, group_id, group_index, channel_start, channel_count, payload_kind, sample_format, flags);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * @param {number} encoding
 * @param {number} payload_size
 * @param {number} sample_size
 * @param {number} sample_rate
 * @param {number} channels
 * @param {number} bits_per_sample
 * @param {number} pts
 * @returns {Uint8Array}
 */
export function buildSoundKitFrameHeaderV2(encoding, payload_size, sample_size, sample_rate, channels, bits_per_sample, pts) {
    const ret = wasm.buildSoundKitFrameHeaderV2(encoding, payload_size, sample_size, sample_rate, channels, bits_per_sample, pts);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

/**
 * @param {number} encoding
 * @param {Uint8Array} payload
 * @param {number} sample_size
 * @param {number} sample_rate
 * @param {number} channels
 * @param {number} bits_per_sample
 * @param {number} pts
 * @returns {Uint8Array}
 */
export function buildSoundKitFrameV2(encoding, payload, sample_size, sample_rate, channels, bits_per_sample, pts) {
    const ptr0 = passArray8ToWasm0(payload, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.buildSoundKitFrameV2(encoding, ptr0, len0, sample_size, sample_rate, channels, bits_per_sample, pts);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}
function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg___wbindgen_throw_344f42d3211c4765: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbg_new_32b398fb48b6d94a: function() {
            const ret = new Array();
            return ret;
        },
        __wbg_new_da52cf8fe3429cb2: function() {
            const ret = new Object();
            return ret;
        },
        __wbg_new_from_slice_77cdfb7977362f3c: function(arg0, arg1) {
            const ret = new Uint8Array(getArrayU8FromWasm0(arg0, arg1));
            return ret;
        },
        __wbg_push_d2ae3af0c1217ae6: function(arg0, arg1) {
            const ret = arg0.push(arg1);
            return ret;
        },
        __wbg_set_8535240470bf2500: function() { return handleError(function (arg0, arg1, arg2) {
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
        "./soundkit_wasm_bg.js": import0,
    };
}

const WasmAudioContentCipherFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmaudiocontentcipher_free(ptr, 1));
const WasmAudioContentKeyUnwrapperFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmaudiocontentkeyunwrapper_free(ptr, 1));
const WasmMusicDecoderFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmmusicdecoder_free(ptr, 1));
const WasmSoundKitFrameDecoderFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmsoundkitframedecoder_free(ptr, 1));

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
        module_or_path = new URL('soundkit_wasm_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
