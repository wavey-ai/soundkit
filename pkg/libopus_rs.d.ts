/* tslint:disable */
/* eslint-disable */

export class DecodeResult {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    readonly decodedSize: number;
    readonly output: Int16Array;
}

export class DecodeResultI24 {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    readonly decodedSize: number;
    readonly output: Int32Array;
}

export class Decoder {
    free(): void;
    [Symbol.dispose](): void;
    dec_frame(packet: Uint8Array): DecodeResult;
    dec_frame_i24(packet: Uint8Array): DecodeResultI24;
    dec_frame_i24_reuse(packet: Uint8Array): number;
    dec_frame_reuse(packet: Uint8Array): number;
    destroy(): void;
    constructor(channels: number, sample_rate: number, _frame_size: number);
    readonly decodedSize: number;
    readonly outputI24Len: number;
    readonly outputI24Ptr: number;
    readonly outputLen: number;
    readonly outputPtr: number;
}

export class EncodeResult {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    readonly encodedData: Uint8Array;
    readonly ok: boolean;
}

export class Encoder {
    free(): void;
    [Symbol.dispose](): void;
    destroy(): void;
    enc_frame(input: Int16Array): EncodeResult;
    enc_frame_i24(input: Int32Array): EncodeResult;
    /**
     * Encodes signed 24-bit samples from this encoder's staged `i32` input.
     */
    enc_frame_i24_reuse(): number;
    /**
     * Encodes the samples in this encoder's input storage into its output
     * storage and returns the packet size.
     *
     * JavaScript callers fill `inputPtr..inputLen`, then read the packet through
     * `outputPtr..outputLen`. Views must be refreshed after Wasm memory grows.
     */
    enc_frame_reuse(): number;
    constructor(channels: number, sample_rate: number, bitrate: number, frame_size: number);
    set_vbr(enabled: boolean): void;
    readonly inputI24Len: number;
    readonly inputI24Ptr: number;
    readonly inputLen: number;
    readonly inputPtr: number;
    readonly outputLen: number;
    readonly outputPtr: number;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_decoder_free: (a: number, b: number) => void;
    readonly __wbg_decoderesult_free: (a: number, b: number) => void;
    readonly __wbg_decoderesulti24_free: (a: number, b: number) => void;
    readonly __wbg_encoder_free: (a: number, b: number) => void;
    readonly __wbg_encoderesult_free: (a: number, b: number) => void;
    readonly decoder_dec_frame: (a: number, b: number, c: number) => [number, number, number];
    readonly decoder_dec_frame_i24: (a: number, b: number, c: number) => [number, number, number];
    readonly decoder_dec_frame_i24_reuse: (a: number, b: number, c: number) => [number, number, number];
    readonly decoder_dec_frame_reuse: (a: number, b: number, c: number) => [number, number, number];
    readonly decoder_decodedSize: (a: number) => number;
    readonly decoder_destroy: (a: number) => void;
    readonly decoder_new: (a: number, b: number, c: number) => [number, number, number];
    readonly decoder_outputI24Len: (a: number) => number;
    readonly decoder_outputI24Ptr: (a: number) => number;
    readonly decoder_outputLen: (a: number) => number;
    readonly decoder_outputPtr: (a: number) => number;
    readonly decoderesult_decodedSize: (a: number) => number;
    readonly decoderesult_output: (a: number) => [number, number];
    readonly decoderesulti24_output: (a: number) => [number, number];
    readonly encoder_destroy: (a: number) => void;
    readonly encoder_enc_frame: (a: number, b: number, c: number) => [number, number, number];
    readonly encoder_enc_frame_i24: (a: number, b: number, c: number) => [number, number, number];
    readonly encoder_enc_frame_i24_reuse: (a: number) => [number, number, number];
    readonly encoder_enc_frame_reuse: (a: number) => [number, number, number];
    readonly encoder_inputI24Len: (a: number) => number;
    readonly encoder_inputI24Ptr: (a: number) => number;
    readonly encoder_inputLen: (a: number) => number;
    readonly encoder_inputPtr: (a: number) => number;
    readonly encoder_new: (a: number, b: number, c: number, d: number) => [number, number, number];
    readonly encoder_outputLen: (a: number) => number;
    readonly encoder_outputPtr: (a: number) => number;
    readonly encoder_set_vbr: (a: number, b: number) => [number, number];
    readonly encoderesult_encodedData: (a: number) => [number, number];
    readonly encoderesult_ok: (a: number) => number;
    readonly decoderesulti24_decodedSize: (a: number) => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
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
