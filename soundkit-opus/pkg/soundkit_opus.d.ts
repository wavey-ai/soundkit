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
    dec_frame(packet: Uint8Array | number[]): DecodeResult;
    /**
     * Decodes to interleaved floating-point samples.
     */
    dec_frame_f32(packet: Uint8Array | number[]): Float32Array;
    dec_frame_i24(packet: Uint8Array | number[]): DecodeResultI24;
    dec_frame_i24_reuse(packet: Uint8Array | number[]): number;
    dec_frame_reuse(packet: Uint8Array | number[]): number;
    /**
     * Conceals one lost frame (packet-loss concealment) into the i16
     * output, for standard and Opus Custom decoders alike.
     */
    dec_lost(): DecodeResult;
    destroy(): void;
    /**
     * Creates an Opus Custom decoder at 48 kHz for `frame_size`-sample
     * frames (48, 64, 72 for 1, 1.33, 1.5 ms). An empty packet conceals a
     * lost frame.
     */
    static newCustom(channels: number, frame_size: number): Decoder;
    /**
     * Creates a decoder. A `frame_size` that standard Opus cannot carry but
     * Opus Custom can (48, 64 and 72 samples; see `custom_frame_size` in the
     * Rust API) gives an Opus Custom decoder, as `newCustom` does. Any other
     * value gives the standard decoder, which reads every standard frame
     * duration whatever `frame_size` says.
     */
    constructor(channels: number, sample_rate: number, frame_size: number);
    /**
     * Selects libopus's handling of empty, short and damaged packets for a
     * standard decoder (off by default: such packets are errors, as they
     * always were). Opus Custom decoders always conceal.
     */
    set_conceal_errors(enabled: boolean): void;
    /**
     * Opus Custom: expects (`true`, the default) or omits the signalling
     * byte.
     */
    set_custom_signalling(enabled: boolean): void;
    readonly decodedSize: number;
    readonly isCustom: boolean;
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
    enc_frame(input: Int16Array | number[]): EncodeResult;
    enc_frame_f32(input: Float32Array | number[]): EncodeResult;
    enc_frame_i24(input: Int32Array | number[]): EncodeResult;
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
    /**
     * Creates an Opus Custom encoder at 48 kHz for `frame_size`-sample
     * frames: 48, 64 and 72 give 1, 1.33 and 1.5 ms frames. Packets are
     * the `opus_custom_encode` format (one signalling byte, no TOC).
     * Defaults follow libopus: the maximum bitrate (every packet fills the
     * maximum packet size) and CBR; call `set_bitrate` for a target rate.
     */
    static newCustom(channels: number, frame_size: number): Encoder;
    /**
     * Creates an encoder for `frame_size`-sample frames. A frame size that
     * standard Opus cannot carry but Opus Custom can (48, 64 and 72 samples
     * for 1, 1.33 and 1.5 ms; see `custom_frame_size` in the Rust API) gives
     * an Opus Custom encoder, as `newCustom` does, at `bitrate` in CBR; its
     * packets need a custom decoder of the same frame size. Other frame
     * sizes give the standard encoder.
     */
    constructor(channels: number, sample_rate: number, bitrate: number, frame_size: number);
    /**
     * Sets the bitrate in bits per second; -1 selects the maximum (Opus
     * Custom only).
     */
    set_bitrate(bitrate: number): void;
    /**
     * Opus Custom: writes (`true`, the default) or omits the signalling byte.
     */
    set_custom_signalling(enabled: boolean): void;
    /**
     * Selects libopus's encoder decision constants (`true`) or SoundKit's.
     */
    set_libopus_decisions(enabled: boolean): void;
    /**
     * Opus Custom: the largest packet, signalling byte included (2-1276).
     */
    set_max_packet_bytes(bytes: number): void;
    set_vbr(enabled: boolean): void;
    /**
     * Selects constrained VBR (`true`, the default) or unconstrained VBR.
     */
    set_vbr_constraint(constrained: boolean): void;
    readonly inputI24Len: number;
    readonly inputI24Ptr: number;
    readonly inputLen: number;
    readonly inputPtr: number;
    readonly isCustom: boolean;
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
    readonly decoder_dec_frame_f32: (a: number, b: number, c: number) => [number, number, number, number];
    readonly decoder_dec_frame_i24: (a: number, b: number, c: number) => [number, number, number];
    readonly decoder_dec_frame_i24_reuse: (a: number, b: number, c: number) => [number, number, number];
    readonly decoder_dec_frame_reuse: (a: number, b: number, c: number) => [number, number, number];
    readonly decoder_dec_lost: (a: number) => [number, number, number];
    readonly decoder_decodedSize: (a: number) => number;
    readonly decoder_destroy: (a: number) => void;
    readonly decoder_isCustom: (a: number) => number;
    readonly decoder_new: (a: number, b: number, c: number) => [number, number, number];
    readonly decoder_newCustom: (a: number, b: number) => [number, number, number];
    readonly decoder_outputI24Len: (a: number) => number;
    readonly decoder_outputI24Ptr: (a: number) => number;
    readonly decoder_outputLen: (a: number) => number;
    readonly decoder_outputPtr: (a: number) => number;
    readonly decoder_set_conceal_errors: (a: number, b: number) => [number, number];
    readonly decoder_set_custom_signalling: (a: number, b: number) => [number, number];
    readonly decoderesult_decodedSize: (a: number) => number;
    readonly decoderesult_output: (a: number) => [number, number];
    readonly decoderesulti24_decodedSize: (a: number) => number;
    readonly decoderesulti24_output: (a: number) => [number, number];
    readonly encoder_destroy: (a: number) => void;
    readonly encoder_enc_frame: (a: number, b: number, c: number) => [number, number, number];
    readonly encoder_enc_frame_f32: (a: number, b: number, c: number) => [number, number, number];
    readonly encoder_enc_frame_i24: (a: number, b: number, c: number) => [number, number, number];
    readonly encoder_enc_frame_i24_reuse: (a: number) => [number, number, number];
    readonly encoder_enc_frame_reuse: (a: number) => [number, number, number];
    readonly encoder_inputI24Len: (a: number) => number;
    readonly encoder_inputI24Ptr: (a: number) => number;
    readonly encoder_inputLen: (a: number) => number;
    readonly encoder_inputPtr: (a: number) => number;
    readonly encoder_isCustom: (a: number) => number;
    readonly encoder_new: (a: number, b: number, c: number, d: number) => [number, number, number];
    readonly encoder_newCustom: (a: number, b: number) => [number, number, number];
    readonly encoder_outputLen: (a: number) => number;
    readonly encoder_outputPtr: (a: number) => number;
    readonly encoder_set_bitrate: (a: number, b: number) => [number, number];
    readonly encoder_set_custom_signalling: (a: number, b: number) => [number, number];
    readonly encoder_set_libopus_decisions: (a: number, b: number) => void;
    readonly encoder_set_max_packet_bytes: (a: number, b: number) => [number, number];
    readonly encoder_set_vbr: (a: number, b: number) => [number, number];
    readonly encoder_set_vbr_constraint: (a: number, b: number) => [number, number];
    readonly encoderesult_encodedData: (a: number) => [number, number];
    readonly encoderesult_ok: (a: number) => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
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
