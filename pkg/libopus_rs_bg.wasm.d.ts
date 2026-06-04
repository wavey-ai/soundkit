/* tslint:disable */
/* eslint-disable */
export const memory: WebAssembly.Memory;
export const __wbg_encoder_free: (a: number, b: number) => void;
export const encoder_new: (a: number, b: number, c: number, d: number) => [number, number, number];
export const encoder_enc_frame: (a: number, b: number, c: number) => [number, number, number];
export const encoder_set_vbr: (a: number, b: number) => [number, number];
export const encoder_destroy: (a: number) => void;
export const __wbg_encoderesult_free: (a: number, b: number) => void;
export const encoderesult_ok: (a: number) => number;
export const encoderesult_encodedData: (a: number) => [number, number];
export const __wbg_decoder_free: (a: number, b: number) => void;
export const decoder_new: (a: number, b: number, c: number) => [number, number, number];
export const decoder_dec_frame: (a: number, b: number, c: number) => [number, number, number];
export const decoder_dec_frame_reuse: (a: number, b: number, c: number) => [number, number, number];
export const decoder_decodedSize: (a: number) => number;
export const decoder_outputPtr: (a: number) => number;
export const decoder_outputLen: (a: number) => number;
export const decoder_destroy: (a: number) => void;
export const __wbg_decoderesult_free: (a: number, b: number) => void;
export const decoderesult_decodedSize: (a: number) => number;
export const decoderesult_output: (a: number) => [number, number];
export const __wbindgen_externrefs: WebAssembly.Table;
export const __wbindgen_malloc: (a: number, b: number) => number;
export const __externref_table_dealloc: (a: number) => void;
export const __wbindgen_free: (a: number, b: number, c: number) => void;
export const __wbindgen_start: () => void;
