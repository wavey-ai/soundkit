#![cfg(all(target_arch = "wasm32", feature = "browser-audio"))]

use soundkit_wasm::{Decoder, WasmCanonicalPcmDecoder};
use wasm_bindgen_test::wasm_bindgen_test;


fn fixture(name: &'static [u8]) -> Vec<u8> {
    name.to_vec()
}

fn decode_all(format: &str, data: &[u8], chunk_size: usize) -> Vec<u8> {
    let mut decoder = Decoder::new_with_format(format).unwrap();
    let mut pcm = Vec::new();
    for chunk in data.chunks(chunk_size) {
        for frame in decoder.push_rust(chunk).unwrap() {
            pcm.extend_from_slice(frame.data());
        }
    }
    for frame in decoder.flush_rust().unwrap() {
        pcm.extend_from_slice(frame.data());
    }
    pcm
}

fn decode_all_auto(data: &[u8], chunk_size: usize) -> Vec<u8> {
    let mut decoder = Decoder::new_auto();
    let mut pcm = Vec::new();
    for chunk in data.chunks(chunk_size) {
        for frame in decoder.push_rust(chunk).unwrap() {
            pcm.extend_from_slice(frame.data());
        }
    }
    for frame in decoder.flush_rust().unwrap() {
        pcm.extend_from_slice(frame.data());
    }
    pcm
}

fn decode_canonical(data: &[u8], format: &str, chunk_size: usize) -> Vec<u8> {
    let mut decoder = WasmCanonicalPcmDecoder::new_with_format(format).unwrap();
    let mut pcm = Vec::new();
    for chunk in data.chunks(chunk_size) {
        let batch = decoder.push_rust(chunk).unwrap();
        for block in &batch.blocks {
            pcm.extend_from_slice(&block.pcm_s16_planar);
        }
    }
    let batch = decoder.finish_rust().unwrap();
    for block in &batch.blocks {
        pcm.extend_from_slice(&block.pcm_s16_planar);
    }
    pcm
}

// ---------------------------------------------------------------------------
// Decoder: format-specific streaming tests
// ---------------------------------------------------------------------------

#[wasm_bindgen_test]
fn wav_streaming_decodes_to_pcm() {
    let data = fixture(include_bytes!("../../testdata/wav_stereo/A_Tusk_is_used_to_make_costly_gifts.wav"));
    let pcm = decode_all("wav", &data, 997);
    assert!(!pcm.is_empty(), "WAV decode must produce PCM output");
}

#[wasm_bindgen_test]
fn flac_streaming_decodes_to_pcm() {
    let data = fixture(include_bytes!("../../testdata/flac/A_Tusk_is_used_to_make_costly_gifts.flac"));
    let pcm = decode_all("flac", &data, 997);
    assert!(!pcm.is_empty(), "FLAC decode must produce PCM output");
}

#[wasm_bindgen_test]
fn mp3_streaming_decodes_to_pcm() {
    let data = fixture(include_bytes!("../../testdata/mp3/A_Tusk_is_used_to_make_costly_gifts.mp3"));
    let pcm = decode_all("mp3", &data, 997);
    assert!(!pcm.is_empty(), "MP3 decode must produce PCM output");
}

#[wasm_bindgen_test]
fn vorbis_streaming_decodes_to_pcm() {
    let data = fixture(include_bytes!("../../testdata/vorbis/A_Tusk_is_used_to_make_costly_gifts.ogg"));
    let pcm = decode_all("ogg-vorbis", &data, 641);
    assert!(!pcm.is_empty(), "Vorbis decode must produce PCM output");
}

#[wasm_bindgen_test]
fn opus_streaming_decodes_to_pcm() {
    let data = fixture(include_bytes!("../../testdata/ogg_opus/A_Tusk_is_used_to_make_costly_gifts_48khz.ogg"));
    let pcm = decode_all("ogg-opus", &data, 641);
    assert!(!pcm.is_empty(), "Opus decode must produce PCM output");
}

#[wasm_bindgen_test]
fn ac3_streaming_decodes_to_pcm() {
    let data = fixture(include_bytes!("../../testdata/ac3/A_Tusk_is_used_to_make_costly_gifts.ac3"));
    let pcm = decode_all("ac3", &data, 997);
    assert!(!pcm.is_empty(), "AC-3 decode must produce PCM output");
}

#[wasm_bindgen_test]
fn aiff_streaming_decodes_to_pcm() {
    let data = fixture(include_bytes!("../../testdata/aiff/A_Tusk_is_used_to_make_costly_gifts.aiff"));
    let pcm = decode_all("aiff", &data, 997);
    assert!(!pcm.is_empty(), "AIFF decode must produce PCM output");
}

#[wasm_bindgen_test]
fn aifc_streaming_decodes_to_pcm() {
    let data = fixture(include_bytes!("../../testdata/aifc/A_Tusk_is_used_to_make_costly_gifts.aifc"));
    let pcm = decode_all("aifc", &data, 997);
    assert!(!pcm.is_empty(), "AIFC decode must produce PCM output");
}

// ---------------------------------------------------------------------------
// Container extraction: audio from video files
// These files need specific demuxer APIs (WasmAudioTrackDemuxer with "fmp4",
// "mpeg-ts" hints, or WasmWebmMediaDemuxer) and are covered by tests in
// soundkit-audio-demux and JS test scripts.  Decoder auto-detect
// handles MP4/MOV and WebM/MKV only when they are pure-audio containers.
// ---------------------------------------------------------------------------

#[wasm_bindgen_test]
fn mp4_h264_aac_extracts_audio() {
    let data = fixture(include_bytes!("../../testdata/video-compat/never-final/h264-high-aac.mp4"));
    let pcm = decode_all_auto(&data, 997);
    assert!(!pcm.is_empty(), "MP4 H.264+AAC must extract audio");
}

#[wasm_bindgen_test]
fn hevc_aac_mov_extracts_audio() {
    let data = fixture(include_bytes!("../../testdata/video-compat/never-final/hevc-main-aac.mov"));
    let pcm = decode_all_auto(&data, 997);
    assert!(!pcm.is_empty(), "MOV HEVC+AAC must extract audio");
}

// ---------------------------------------------------------------------------
// Auto-detection: format without explicit hint
// ---------------------------------------------------------------------------

#[wasm_bindgen_test]
fn auto_detect_wav() {
    let data = fixture(include_bytes!("../../testdata/wav_stereo/A_Tusk_is_used_to_make_costly_gifts.wav"));
    let pcm = decode_all_auto(&data, 997);
    assert!(!pcm.is_empty(), "Auto-detect WAV must produce output");
}

#[wasm_bindgen_test]
fn auto_detect_flac() {
    let data = fixture(include_bytes!("../../testdata/flac/A_Tusk_is_used_to_make_costly_gifts.flac"));
    let pcm = decode_all_auto(&data, 997);
    assert!(!pcm.is_empty(), "Auto-detect FLAC must produce output");
}

#[wasm_bindgen_test]
fn auto_detect_mp3() {
    let data = fixture(include_bytes!("../../testdata/mp3/A_Tusk_is_used_to_make_costly_gifts.mp3"));
    let pcm = decode_all_auto(&data, 997);
    assert!(!pcm.is_empty(), "Auto-detect MP3 must produce output");
}

#[wasm_bindgen_test]
fn auto_detect_vorbis() {
    let data = fixture(include_bytes!("../../testdata/vorbis/A_Tusk_is_used_to_make_costly_gifts.ogg"));
    let pcm = decode_all_auto(&data, 641);
    assert!(!pcm.is_empty(), "Auto-detect Vorbis must produce output");
}

#[wasm_bindgen_test]
fn auto_detect_opus() {
    let data = fixture(include_bytes!("../../testdata/ogg_opus/A_Tusk_is_used_to_make_costly_gifts_48khz.ogg"));
    let pcm = decode_all_auto(&data, 641);
    assert!(!pcm.is_empty(), "Auto-detect Opus must produce output");
}

#[wasm_bindgen_test]
fn auto_detect_aac_adts() {
    let data = fixture(include_bytes!("../../golden/aac/stereo-music-44100-192k.aac"));
    let pcm = decode_all_auto(&data, 997);
    assert!(!pcm.is_empty(), "Auto-detect AAC ADTS must produce output");
}

// ---------------------------------------------------------------------------
// Chunk size invariance: same output regardless of push granularity
// ---------------------------------------------------------------------------

#[wasm_bindgen_test]
fn chunk_size_invariance_wav() {
    let data = fixture(include_bytes!("../../testdata/wav_stereo/A_Tusk_is_used_to_make_costly_gifts.wav"));
    let reference = decode_all("wav", &data, 997);
    for &chunk_size in &[1, 7, 256, 4096, 65536] {
        let pcm = decode_all("wav", &data, chunk_size);
        assert_eq!(
            pcm, reference,
            "WAV output changed for {chunk_size}-byte chunks"
        );
    }
}

#[wasm_bindgen_test]
fn chunk_size_invariance_flac() {
    let data = fixture(include_bytes!("../../testdata/flac/A_Tusk_is_used_to_make_costly_gifts.flac"));
    let reference = decode_all("flac", &data, 997);
    for &chunk_size in &[7, 256, 4096] {
        let pcm = decode_all("flac", &data, chunk_size);
        assert_eq!(
            pcm, reference,
            "FLAC output changed for {chunk_size}-byte chunks"
        );
    }
}

// ---------------------------------------------------------------------------
// Canonical PCM decoder: 48kHz stereo normalization
// ---------------------------------------------------------------------------

#[wasm_bindgen_test]
fn canonical_pcm_normalizes_wav_to_48k_stereo() {
    let data = fixture(include_bytes!("../../testdata/wav_stereo/A_Tusk_is_used_to_make_costly_gifts.wav"));
    let pcm = decode_canonical(&data, "wav", 997);
    assert!(!pcm.is_empty(), "Canonical WAV must produce output");
}

#[wasm_bindgen_test]
fn canonical_pcm_normalizes_flac_to_48k_stereo() {
    let data = fixture(include_bytes!("../../testdata/flac/A_Tusk_is_used_to_make_costly_gifts.flac"));
    let pcm = decode_canonical(&data, "flac", 997);
    assert!(!pcm.is_empty(), "Canonical FLAC must produce output");
}

#[wasm_bindgen_test]
fn canonical_pcm_normalizes_mp3_to_48k_stereo() {
    let data = fixture(include_bytes!("../../testdata/mp3/A_Tusk_is_used_to_make_costly_gifts.mp3"));
    let pcm = decode_canonical(&data, "mp3", 997);
    assert!(!pcm.is_empty(), "Canonical MP3 must produce output");
}

#[wasm_bindgen_test]
fn canonical_pcm_normalizes_opus_to_48k_stereo() {
    let data = fixture(include_bytes!("../../testdata/ogg_opus/A_Tusk_is_used_to_make_costly_gifts_48khz.ogg"));
    let pcm = decode_canonical(&data, "ogg-opus", 641);
    assert!(!pcm.is_empty(), "Canonical Opus must produce output");
}

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

#[wasm_bindgen_test]
fn rejects_oversized_chunk() {
    let oversized = vec![0u8; 4 * 1024 * 1024 + 1];
    let mut decoder = Decoder::new_auto();
    let error = decoder.push_rust(&oversized);
    assert!(error.is_err(), "Must reject oversized chunks");
}

#[wasm_bindgen_test]
fn rejects_unknown_format() {
    let result = Decoder::new_with_format("nonexistent-format");
    assert!(result.is_err(), "Must reject unknown format");
}

// ---------------------------------------------------------------------------
// Container extraction via auto-detect
// ---------------------------------------------------------------------------

#[wasm_bindgen_test]
fn matroska_h264_aac_auto_detect() {
    let data = fixture(include_bytes!(
        "../../testdata/video-compat/never-final/matroska-h264-aac.mkv"
    ));
    let mut decoder = Decoder::new_auto();
    let mut pcm = Vec::new();
    for chunk in data.chunks(4096) {
        for frame in decoder.push_rust(chunk).unwrap() {
            pcm.extend_from_slice(frame.data());
        }
    }
    for frame in decoder.flush_rust().unwrap() {
        pcm.extend_from_slice(frame.data());
    }
    assert!(
        !pcm.is_empty(),
        "Matroska MKV demux must reach AAC decoder and produce PCM"
    );
}

#[wasm_bindgen_test]
fn cmaf_h264_aac_auto_detect() {
    let data = fixture(include_bytes!(
        "../../testdata/video-compat/never-final/h264-aac-cmaf.mp4"
    ));
    let mut decoder = Decoder::new_auto();
    let mut pcm = Vec::new();
    for chunk in data.chunks(4096) {
        for frame in decoder.push_rust(chunk).unwrap() {
            pcm.extend_from_slice(frame.data());
        }
    }
    for frame in decoder.flush_rust().unwrap() {
        pcm.extend_from_slice(frame.data());
    }
    assert!(!pcm.is_empty(), "CMAF H264+AAC must produce PCM");
}

#[wasm_bindgen_test]
fn fragmented_mp4_h264_aac_auto_detect() {
    let data = fixture(include_bytes!(
        "../../testdata/video-compat/never-final/h264-aac-fragmented.mp4"
    ));
    let mut decoder = Decoder::new_auto();
    let mut pcm = Vec::new();
    for chunk in data.chunks(4096) {
        for frame in decoder.push_rust(chunk).unwrap() {
            pcm.extend_from_slice(frame.data());
        }
    }
    for frame in decoder.flush_rust().unwrap() {
        pcm.extend_from_slice(frame.data());
    }
    assert!(!pcm.is_empty(), "Fragmented MP4 H264+AAC must produce PCM");
}

#[wasm_bindgen_test]
fn mpeg_ts_aac_auto_detect() {
    let data = fixture(include_bytes!("../../testdata/mpeg-ts/aac-stereo-48k.ts"));
    let mut decoder = Decoder::new_auto();
    let mut pcm = Vec::new();
    for chunk in data.chunks(4096) {
        for frame in decoder.push_rust(chunk).unwrap() {
            pcm.extend_from_slice(frame.data());
        }
    }
    for frame in decoder.flush_rust().unwrap() {
        pcm.extend_from_slice(frame.data());
    }
    assert!(!pcm.is_empty(), "MPEG-TS AAC must produce PCM");
}
