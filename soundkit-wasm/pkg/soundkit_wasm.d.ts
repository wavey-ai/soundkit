/* tslint:disable */
/* eslint-disable */

export class WasmAacDeboxer {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Final drain call. The deboxer should not be reused after this.
     */
    flush(): Array<any>;
    constructor();
    static newAuto(): WasmAacDeboxer;
    static newWithFormat(format: string): WasmAacDeboxer;
    /**
     * Push arbitrary MP4/M4A bytes and receive AAC config/packet events.
     *
     * Packet events contain ADTS AAC frames in `data` and the original MP4
     * access unit in `rawData`.
     */
    push(bytes: Uint8Array): Array<any>;
}

export class WasmAacLcDecoder {
    free(): void;
    [Symbol.dispose](): void;
    decodeInterleaved(access_unit: Uint8Array): Float32Array;
    decodeInterleavedInto(access_unit: Uint8Array, output: Float32Array): number;
    decodePlanar(access_unit: Uint8Array): Array<any>;
    constructor(audio_specific_config: Uint8Array);
    readonly channels: number;
    readonly framesPerAccessUnit: number;
    readonly sampleRate: number;
}

export class WasmAudioTrackDemuxer {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Final drain call. The demuxer should not be reused after this.
     */
    flush(): Array<any>;
    constructor();
    static newAuto(): WasmAudioTrackDemuxer;
    static newWithFormat(format: string): WasmAudioTrackDemuxer;
    /**
     * Push arbitrary container bytes and receive audio-track config/packet events.
     */
    push(bytes: Uint8Array): Array<any>;
}

export class WasmFlacEncoder {
    free(): void;
    [Symbol.dispose](): void;
    encodePlanarF32(planar: Float32Array, frames_per_channel: number): Uint8Array;
    constructor(sample_rate: number, channels: number, bits_per_sample: number, frame_size: number, compression_level: number);
    reset(): void;
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

export class WasmOpusDeboxer {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Final drain call. The deboxer should not be reused after this.
     */
    flush(): Array<any>;
    constructor();
    static newAuto(): WasmOpusDeboxer;
    static newWithFormat(format: string): WasmOpusDeboxer;
    /**
     * Push arbitrary container bytes and receive Opus config/packet events.
     *
     * Packet events contain encoded Opus packet bytes suitable for a JS Opus
     * decoder. Config events carry channel/sample-rate/pre-skip metadata.
     */
    push(bytes: Uint8Array): Array<any>;
}

export class WasmOpusDecodeResult {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    readonly decodedSize: number;
    readonly output: Int16Array;
}

export class WasmOpusDecoder {
    free(): void;
    [Symbol.dispose](): void;
    dec_frame(packet: Uint8Array): WasmOpusDecodeResult;
    dec_frame_reuse(packet: Uint8Array): number;
    destroy(): void;
    constructor(channels: number, sample_rate: number, _frame_size: number);
    readonly decodedSize: number;
    readonly outputLen: number;
    readonly outputPtr: number;
}

export class WasmOpusEncoder {
    free(): void;
    [Symbol.dispose](): void;
    encodeInterleavedI16(interleaved: Int16Array): Uint8Array;
    constructor(sample_rate: number, channels: number, bitrate: number, frame_size: number);
    reset(): void;
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

export function buildSoundKitFrameHeaderV2(encoding: number, payload_size: number, sample_size: number, sample_rate: number, channels: number, bits_per_sample: number, pts: number): Uint8Array;

export function buildSoundKitFrameV2(encoding: number, payload: Uint8Array, sample_size: number, sample_rate: number, channels: number, bits_per_sample: number, pts: number): Uint8Array;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_wasmaacdeboxer_free: (a: number, b: number) => void;
    readonly __wbg_wasmaaclcdecoder_free: (a: number, b: number) => void;
    readonly __wbg_wasmaudiotrackdemuxer_free: (a: number, b: number) => void;
    readonly __wbg_wasmflacencoder_free: (a: number, b: number) => void;
    readonly __wbg_wasmmusicdecoder_free: (a: number, b: number) => void;
    readonly __wbg_wasmopusdeboxer_free: (a: number, b: number) => void;
    readonly __wbg_wasmopusdecoder_free: (a: number, b: number) => void;
    readonly __wbg_wasmopusdecoderesult_free: (a: number, b: number) => void;
    readonly __wbg_wasmopusencoder_free: (a: number, b: number) => void;
    readonly __wbg_wasmsoundkitframedecoder_free: (a: number, b: number) => void;
    readonly buildSoundKitFrameHeaderV2: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number, number];
    readonly buildSoundKitFrameV2: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => [number, number, number];
    readonly wasmaacdeboxer_flush: (a: number) => [number, number, number];
    readonly wasmaacdeboxer_new: () => number;
    readonly wasmaacdeboxer_newWithFormat: (a: number, b: number) => [number, number, number];
    readonly wasmaacdeboxer_push: (a: number, b: number, c: number) => [number, number, number];
    readonly wasmaaclcdecoder_channels: (a: number) => number;
    readonly wasmaaclcdecoder_decodeInterleaved: (a: number, b: number, c: number) => [number, number, number];
    readonly wasmaaclcdecoder_decodeInterleavedInto: (a: number, b: number, c: number, d: any) => [number, number, number];
    readonly wasmaaclcdecoder_decodePlanar: (a: number, b: number, c: number) => [number, number, number];
    readonly wasmaaclcdecoder_framesPerAccessUnit: (a: number) => number;
    readonly wasmaaclcdecoder_new: (a: number, b: number) => [number, number, number];
    readonly wasmaaclcdecoder_sampleRate: (a: number) => number;
    readonly wasmaudiotrackdemuxer_flush: (a: number) => [number, number, number];
    readonly wasmaudiotrackdemuxer_new: () => number;
    readonly wasmaudiotrackdemuxer_newWithFormat: (a: number, b: number) => [number, number, number];
    readonly wasmaudiotrackdemuxer_push: (a: number, b: number, c: number) => [number, number, number];
    readonly wasmflacencoder_encodePlanarF32: (a: number, b: number, c: number, d: number) => [number, number, number];
    readonly wasmflacencoder_new: (a: number, b: number, c: number, d: number, e: number) => [number, number, number];
    readonly wasmflacencoder_reset: (a: number) => [number, number];
    readonly wasmmusicdecoder_flush: (a: number) => [number, number, number];
    readonly wasmmusicdecoder_new: () => number;
    readonly wasmmusicdecoder_newRawLinear16: (a: number, b: number) => [number, number, number];
    readonly wasmmusicdecoder_newRawLinear32: (a: number, b: number) => [number, number, number];
    readonly wasmmusicdecoder_newWithFormat: (a: number, b: number) => [number, number, number];
    readonly wasmmusicdecoder_push: (a: number, b: number, c: number) => [number, number, number];
    readonly wasmopusdeboxer_flush: (a: number) => [number, number, number];
    readonly wasmopusdeboxer_new: () => number;
    readonly wasmopusdeboxer_newWithFormat: (a: number, b: number) => [number, number, number];
    readonly wasmopusdeboxer_push: (a: number, b: number, c: number) => [number, number, number];
    readonly wasmopusdecoder_dec_frame: (a: number, b: number, c: number) => [number, number, number];
    readonly wasmopusdecoder_dec_frame_reuse: (a: number, b: number, c: number) => [number, number, number];
    readonly wasmopusdecoder_decodedSize: (a: number) => number;
    readonly wasmopusdecoder_destroy: (a: number) => void;
    readonly wasmopusdecoder_new: (a: number, b: number, c: number) => [number, number, number];
    readonly wasmopusdecoder_outputLen: (a: number) => number;
    readonly wasmopusdecoder_outputPtr: (a: number) => number;
    readonly wasmopusdecoderesult_decodedSize: (a: number) => number;
    readonly wasmopusdecoderesult_output: (a: number) => [number, number];
    readonly wasmopusencoder_encodeInterleavedI16: (a: number, b: number, c: number) => [number, number, number];
    readonly wasmopusencoder_new: (a: number, b: number, c: number, d: number) => [number, number, number];
    readonly wasmopusencoder_reset: (a: number) => [number, number];
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
    readonly wasmaacdeboxer_newAuto: () => number;
    readonly wasmaudiotrackdemuxer_newAuto: () => number;
    readonly wasmopusdeboxer_newAuto: () => number;
    readonly wasmsoundkitframedecoder_newUnencrypted: () => number;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
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
