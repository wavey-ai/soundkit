#![cfg(all(target_arch = "wasm32", feature = "aac-lc"))]

use frame_header::{EncodingFlag, Endianness, FrameHeaderV2};
use js_sys::{Float32Array, Reflect, Uint8Array};
use soundkit_wasm_decoder::{WasmAacLcDecoder, WasmSoundKitFrameDecoder};
use wasm_bindgen::JsCast;
use wasm_bindgen_test::wasm_bindgen_test;

#[wasm_bindgen_test]
fn decodes_silent_single_channel_access_unit() {
    let mut decoder = WasmAacLcDecoder::new(&[0x12, 0x08]).unwrap();

    assert_eq!(decoder.sample_rate(), 44_100);
    assert_eq!(decoder.channels(), 1);
    assert_eq!(decoder.frames_per_access_unit(), 1024);

    let access_unit = silent_mono_access_unit();

    let interleaved = decoder.decode_interleaved(&access_unit).unwrap();
    assert_eq!(interleaved.length(), 1024);

    let mut samples = vec![1.0; interleaved.length() as usize];
    interleaved.copy_to(&mut samples);
    assert!(samples.iter().all(|sample| *sample == 0.0));

    let reusable_output = Float32Array::new_with_length(1024);
    let written = decoder
        .decode_interleaved_into(&access_unit, &reusable_output)
        .unwrap();
    assert_eq!(written, 1024);
    reusable_output.copy_to(&mut samples);
    assert!(samples.iter().all(|sample| *sample == 0.0));

    let short_output = Float32Array::new_with_length(1023);
    assert!(decoder
        .decode_interleaved_into(&access_unit, &short_output)
        .is_err());

    let planar = decoder.decode_planar(&access_unit).unwrap();
    assert_eq!(planar.length(), 1);

    let channel: Float32Array = planar.get(0).unchecked_into();
    assert_eq!(channel.length(), 1024);
}

#[wasm_bindgen_test]
fn decodes_aac_lc_payloads_from_soundkit_frame_stream() {
    let access_unit = silent_mono_access_unit();
    let mut stream_bytes = encode_soundkit_aac_frame(&access_unit, 7, 0);
    stream_bytes.extend_from_slice(&encode_soundkit_aac_frame(&access_unit, 8, 1024));

    let mut frame_decoder = WasmSoundKitFrameDecoder::new_unencrypted();
    let partial = frame_decoder.push(&stream_bytes[..5]).unwrap();
    assert_eq!(partial.length(), 0);
    assert_eq!(frame_decoder.buffered_bytes(), 5);

    let frames = frame_decoder.push(&stream_bytes[5..]).unwrap();
    assert_eq!(frames.length(), 2);
    frame_decoder.finish().unwrap();

    let mut aac_decoder = WasmAacLcDecoder::new(&[0x12, 0x08]).unwrap();
    for index in 0..frames.length() {
        let frame = frames.get(index);
        let header = Reflect::get(&frame, &"header".into()).unwrap();
        assert_eq!(reflect_string(&header, "encoding"), "AAC");
        assert_eq!(reflect_number(&header, "encodingCode") as u8, 4);
        assert_eq!(reflect_number(&header, "sampleRate") as u32, 44_100);
        assert_eq!(reflect_number(&header, "channels") as u8, 1);
        assert_eq!(reflect_number(&header, "frameCount") as usize, 1024);
        assert_eq!(
            reflect_number(&header, "payloadSize") as usize,
            access_unit.len()
        );

        let payload_array: Uint8Array = Reflect::get(&frame, &"data".into())
            .unwrap()
            .unchecked_into();
        let mut payload = vec![0u8; payload_array.length() as usize];
        payload_array.copy_to(&mut payload);
        assert_eq!(payload, access_unit);

        let interleaved = aac_decoder.decode_interleaved(&payload).unwrap();
        assert_eq!(interleaved.length(), 1024);
        let mut samples = vec![1.0; interleaved.length() as usize];
        interleaved.copy_to(&mut samples);
        assert!(samples.iter().all(|sample| *sample == 0.0));
    }
}

#[wasm_bindgen_test]
fn rejects_non_aac_lc_config() {
    let aac_main = build_bits(&[
        (1, 5), // AAC Main
        (4, 4), // 44.1 kHz
        (2, 4), // stereo
        (0, 1), // frameLengthFlag
        (0, 1), // dependsOnCoreCoder
        (0, 1), // extensionFlag
    ]);
    assert_error_contains(&aac_main, "unsupported AAC audio object type 1");
}

#[wasm_bindgen_test]
fn reports_unsupported_profiles_and_channel_layouts_for_fallback() {
    let sbr = build_bits(&[
        (5, 5), // SBR
        (4, 4), // 44.1 kHz
        (2, 4), // stereo
        (4, 4), // extension 44.1 kHz
        (2, 5), // AAC-LC
        (0, 1), // frameLengthFlag
        (0, 1), // dependsOnCoreCoder
        (0, 1), // extensionFlag
    ]);
    assert_error_contains(&sbr, "unsupported AAC feature: SBR/HE-AAC");

    let ps = build_bits(&[
        (29, 5), // Parametric Stereo
        (4, 4),  // 44.1 kHz
        (2, 4),  // stereo
        (4, 4),  // extension 44.1 kHz
        (2, 5),  // AAC-LC
        (0, 1),  // frameLengthFlag
        (0, 1),  // dependsOnCoreCoder
        (0, 1),  // extensionFlag
    ]);
    assert_error_contains(&ps, "unsupported AAC feature: parametric stereo");

    let surround = build_bits(&[
        (2, 5), // AAC-LC
        (4, 4), // 44.1 kHz
        (6, 4), // 5.1 channel config
        (0, 1), // frameLengthFlag
        (0, 1), // dependsOnCoreCoder
        (0, 1), // extensionFlag
    ]);
    assert_error_contains(&surround, "unsupported AAC channel configuration 6");
}

fn assert_error_contains(audio_specific_config: &[u8], expected: &str) {
    let error = match WasmAacLcDecoder::new(audio_specific_config) {
        Ok(_) => panic!("expected decoder initialization to fail"),
        Err(error) => error,
    };
    let message = error.as_string().unwrap_or_default();
    assert!(
        message.contains(expected),
        "expected error containing {expected:?}, got {message:?}"
    );
}

fn silent_mono_access_unit() -> Vec<u8> {
    build_bits(&[
        (0, 3),   // SCE
        (0, 4),   // tag
        (100, 8), // global gain
        (0, 1),   // reserved
        (0, 2),   // only long
        (0, 1),   // sine
        (1, 6),   // max_sfb
        (0, 1),   // predictor
        (0, 4),   // zero codebook
        (1, 5),   // section length
        (0, 1),   // pulse_data_present
        (0, 1),   // tns_data_present
        (0, 1),   // gain_control_data_present
    ])
}

fn encode_soundkit_aac_frame(payload: &[u8], id: u64, pts: u64) -> Vec<u8> {
    let header = FrameHeaderV2::new(
        EncodingFlag::AAC,
        payload.len() as u32,
        1024,
        44_100,
        1,
        0,
        Endianness::LittleEndian,
        Some(id),
        Some(pts),
        None,
    )
    .unwrap()
    .with_packet_crc32(payload)
    .unwrap();

    let mut output = Vec::with_capacity(header.size() + payload.len());
    header.encode(&mut output).unwrap();
    output.extend_from_slice(payload);
    output
}

fn reflect_string(object: &wasm_bindgen::JsValue, key: &str) -> String {
    Reflect::get(object, &key.into())
        .unwrap()
        .as_string()
        .unwrap()
}

fn reflect_number(object: &wasm_bindgen::JsValue, key: &str) -> f64 {
    Reflect::get(object, &key.into()).unwrap().as_f64().unwrap()
}

fn build_bits(fields: &[(u32, u8)]) -> Vec<u8> {
    let bit_count: usize = fields.iter().map(|(_, width)| *width as usize).sum();
    let mut out = vec![0u8; bit_count.div_ceil(8)];
    let mut bit_pos = 0usize;

    for &(value, width) in fields {
        for bit in (0..width).rev() {
            if ((value >> bit) & 1) != 0 {
                out[bit_pos / 8] |= 1 << (7 - (bit_pos % 8));
            }
            bit_pos += 1;
        }
    }

    out
}
