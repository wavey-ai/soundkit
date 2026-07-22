/* tslint:disable */
/* eslint-disable */

export class WasmAudioContentCipher {
    free(): void;
    [Symbol.dispose](): void;
    constructor(key: Uint8Array);
    open(expected_key_epoch: number, envelope: Uint8Array, authenticated_data: Uint8Array): Uint8Array;
    seal(key_epoch: number, nonce: Uint8Array, plaintext: Uint8Array, authenticated_data: Uint8Array): Uint8Array;
}

/**
 * Opens the endpoint-specific envelope that transports an audio content key.
 *
 * The wrapping key comes from P-256 ECDH and HKDF-SHA256. The caller supplies
 * the canonical key-exchange context as additional authenticated data.
 */
export class WasmAudioContentKeyUnwrapper {
    free(): void;
    [Symbol.dispose](): void;
    constructor(key: Uint8Array);
    open(nonce: Uint8Array, ciphertext: Uint8Array, authenticated_data: Uint8Array): Uint8Array;
    seal(nonce: Uint8Array, plaintext: Uint8Array, authenticated_data: Uint8Array): Uint8Array;
}

export class WasmMusicDecoder {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Final EOF/drain call. The decoder should not be reused after this.
     */
    flush(): Array<any>;
    constructor();
    static newAuto(): WasmMusicDecoder;
    static newRawLinear16(sample_rate: number, channels: number): WasmMusicDecoder;
    static newRawLinear32(sample_rate: number, channels: number): WasmMusicDecoder;
    static newWithFormat(format: string): WasmMusicDecoder;
    /**
     * Push arbitrary encoded bytes and receive all PCM frames currently available.
     *
     * This method drains decoder output after each push. Use `flush()` once at EOF
     * to force final container/codec drain.
     */
    push(bytes: Uint8Array): Array<any>;
}

export class WasmSoundKitFrameDecoder {
    free(): void;
    [Symbol.dispose](): void;
    bufferedBytes(): number;
    clearKey(): void;
    finish(): void;
    constructor();
    static newUnencrypted(): WasmSoundKitFrameDecoder;
    static newWithDecimalKey(key: string): WasmSoundKitFrameDecoder;
    static newWithKeyBytes(key: Uint8Array): WasmSoundKitFrameDecoder;
    push(bytes: Uint8Array): Array<any>;
    reset(): void;
    setDecimalKey(key: string): void;
    setKeyBytes(key: Uint8Array): void;
}

export function buildAudioGroupAssociatedData(session_context: string, transport_session_id: string, config_generation: number, epoch_id: string, pts_samples: string, sample_rate: number, frame_count: number, group_count: number, group_id: number, group_index: number, channel_start: number, channel_count: number, payload_kind: number, sample_format: number, flags: number): Uint8Array;

export function buildSoundKitFrameHeaderV2(encoding: number, payload_size: number, sample_size: number, sample_rate: number, channels: number, bits_per_sample: number, pts: number): Uint8Array;

export function buildSoundKitFrameV2(encoding: number, payload: Uint8Array, sample_size: number, sample_rate: number, channels: number, bits_per_sample: number, pts: number): Uint8Array;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_wasmaudiocontentcipher_free: (a: number, b: number) => void;
    readonly __wbg_wasmaudiocontentkeyunwrapper_free: (a: number, b: number) => void;
    readonly __wbg_wasmmusicdecoder_free: (a: number, b: number) => void;
    readonly __wbg_wasmsoundkitframedecoder_free: (a: number, b: number) => void;
    readonly buildAudioGroupAssociatedData: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number, r: number, s: number) => [number, number, number];
    readonly buildSoundKitFrameHeaderV2: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number, number];
    readonly buildSoundKitFrameV2: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => [number, number, number];
    readonly wasmaudiocontentcipher_new: (a: number, b: number) => [number, number, number];
    readonly wasmaudiocontentcipher_open: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number];
    readonly wasmaudiocontentcipher_seal: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => [number, number, number];
    readonly wasmaudiocontentkeyunwrapper_new: (a: number, b: number) => [number, number, number];
    readonly wasmaudiocontentkeyunwrapper_open: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number, number];
    readonly wasmaudiocontentkeyunwrapper_seal: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number, number];
    readonly wasmmusicdecoder_flush: (a: number) => [number, number, number];
    readonly wasmmusicdecoder_new: () => number;
    readonly wasmmusicdecoder_newRawLinear16: (a: number, b: number) => [number, number, number];
    readonly wasmmusicdecoder_newRawLinear32: (a: number, b: number) => [number, number, number];
    readonly wasmmusicdecoder_newWithFormat: (a: number, b: number) => [number, number, number];
    readonly wasmmusicdecoder_push: (a: number, b: number, c: number) => [number, number, number];
    readonly wasmsoundkitframedecoder_bufferedBytes: (a: number) => number;
    readonly wasmsoundkitframedecoder_clearKey: (a: number) => void;
    readonly wasmsoundkitframedecoder_finish: (a: number) => [number, number];
    readonly wasmsoundkitframedecoder_new: () => number;
    readonly wasmsoundkitframedecoder_newWithDecimalKey: (a: number, b: number) => [number, number, number];
    readonly wasmsoundkitframedecoder_newWithKeyBytes: (a: number, b: number) => [number, number, number];
    readonly wasmsoundkitframedecoder_push: (a: number, b: number, c: number) => [number, number, number];
    readonly wasmsoundkitframedecoder_reset: (a: number) => void;
    readonly wasmsoundkitframedecoder_setDecimalKey: (a: number, b: number, c: number) => [number, number];
    readonly wasmsoundkitframedecoder_setKeyBytes: (a: number, b: number, c: number) => [number, number];
    readonly wasmmusicdecoder_newAuto: () => number;
    readonly wasmsoundkitframedecoder_newUnencrypted: () => number;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
