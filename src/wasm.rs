use crate::audio_bytes::s24le_to_i32;
use crate::audio_packet::encode_audio_packet_header;
use crate::audio_types::get_audio_config;
use crate::audio_types::EncodingFlag;
use crate::wav::WavStreamProcessor;
use js_sys::{Array, Int16Array, Object, Reflect};
use wasm_bindgen::prelude::*;
use web_sys::Worker;

#[wasm_bindgen]
struct WavToPacket {
    wav_reader: WavStreamProcessor,
    audio_packets: Vec<u8>,
    frame_size: usize,
    packets: Vec<Vec<u8>>,
    bitrate: usize,
    widow: Vec<Vec<u8>>,
}

#[wasm_bindgen]
impl WavToPacket {
    #[wasm_bindgen]
    pub fn new(bitrate: usize, frame_size: usize) -> Self {
        let wav_reader = WavStreamProcessor::new();

        Self {
            wav_reader,
            audio_packets: Vec::new(),
            frame_size,
            packets: Vec::new(),
            bitrate,
            widow: Vec::new(),
        }
    }

    #[wasm_bindgen]
    pub fn into_frames(&mut self, data: &[u8]) -> JsValue {
        let result = Object::new();
        Reflect::set(&result, &JsValue::from_str("ok"), &JsValue::from(true)).unwrap();
        match self.wav_reader.add(data) {
            Ok(Some(audio_data)) => self._into_frames(audio_data.data(), false),
            Ok(None) => result.into(),
            Err(err) => {
                Reflect::set(&result, &JsValue::from_str("ok"), &JsValue::from(false)).unwrap();
                return result.into();
            }
        }
    }

    #[wasm_bindgen]
    pub fn set_frame(&mut self, data: &[u8]) {
        if let Some(config) = get_audio_config(
            self.wav_reader.sampling_rate() as u32,
            self.wav_reader.bits_per_sample() as u8,
        ) {
            let mut packet_data = encode_audio_packet_header(
                &EncodingFlag::Opus,
                config,
                self.wav_reader.channel_count() as u8,
                data.len() as u16,
            );

            packet_data.extend_from_slice(&data);

            self.packets.push(packet_data);
        }
    }

    #[wasm_bindgen]
    pub fn flush(&mut self) -> Vec<u8> {
        if let Some(widow) = self.widow.pop() {
            let _ = self._into_frames(&widow, true);
        }

        let mut offset = 0;
        let mut offsets = Vec::new();
        let mut encoded_data: Vec<u8> = Vec::new();
        for chunk in &self.packets {
            offsets.push(offset);
            offset += chunk.len();
            encoded_data.extend(chunk);
        }

        let mut final_encoded_data = Vec::new();
        for i in 0..4 {
            final_encoded_data.push(((offsets.len() >> (i * 8)) & 0xFF) as u8);
        }

        for offset in offsets {
            for i in 0..4 {
                final_encoded_data.push((offset >> (i * 8) & 0xFF) as u8);
            }
        }

        final_encoded_data.extend(encoded_data);

        self.reset();

        final_encoded_data
    }

    fn _into_frames(&mut self, data: &[u8], is_last: bool) -> JsValue {
        let bits_per_sample = self.wav_reader.bits_per_sample() as usize;
        let channel_count = self.wav_reader.channel_count() as usize;
        let result = Object::new();
        Reflect::set(&result, &JsValue::from_str("ok"), &JsValue::from(false)).unwrap();

        let chunk_size = self.frame_size * channel_count * bits_per_sample;

        let mut owned_data = data.to_owned();
        if let Some(widow) = self.widow.pop() {
            owned_data.extend_from_slice(&widow);
        }

        Reflect::set(&result, &JsValue::from_str("ok"), &JsValue::from(true)).unwrap();

        let mut data = Vec::new();

        for chunk in owned_data.chunks(chunk_size) {
            let Some(config) = get_audio_config(
                self.wav_reader.sampling_rate() as u32,
                bits_per_sample as u8,
            ) else {
                return result.into();
            };

            let chunk_size = self.frame_size as usize * channel_count * bits_per_sample;

            if chunk.len() < chunk_size {
                self.widow.push(chunk.to_vec());
                return result.into();
            };

            let mut src: Vec<i16> = Vec::new();
            match bits_per_sample {
                16 => {
                    for bytes in chunk.chunks_exact(2) {
                        let sample = i16::from_le_bytes([bytes[0], bytes[1]]);
                        src.push(sample);
                    }
                }
                24 => {
                    for bytes in chunk.chunks_exact(3) {
                        let sample = s24le_to_i32([bytes[0], bytes[1], bytes[2]]) as i16;
                        src.push(sample);
                    }
                }
                32 => {
                    for bytes in chunk.chunks_exact(4) {
                        let sample = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                        let scaled_sample = (sample * 32767.0) as i16;
                        src.push(scaled_sample);
                    }
                }
                _ => {
                    return result.into();
                }
            }

            data.push(src);
        }
        let mut nested_array = Array::new();
        for src in data {
            let frame_array = Int16Array::from(&src[..]);
            nested_array.push(&frame_array.into());
        }

        Reflect::set(&result, &JsValue::from_str("frames"), &nested_array).unwrap();

        Reflect::set(
            &result,
            &JsValue::from_str("channel_count"),
            &JsValue::from(channel_count),
        )
        .unwrap();

        result.into()
    }

    fn reset(&mut self) {
        self.wav_reader = WavStreamProcessor::new();
    }
}

/// Run entry point for the main thread.
#[wasm_bindgen]
pub fn startup(path: String) -> Worker {
    let worker_handle = Worker::new(&path).unwrap();
    worker_handle
}
