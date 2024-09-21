use crate::audio_bytes::{
    deinterleave_vecs_f32, deinterleave_vecs_i16, deinterleave_vecs_s24, f32be_to_i16,
    f32le_to_i16, s16be_to_i16, s16le_to_i16, s24be_to_i16, s24le_to_i16, s32be_to_i16,
    s32le_to_i16,
};
use crate::audio_packet::FrameHeader;
use crate::audio_pipeline::{vec_i16_to_f32, vec_i32_to_f32};
use crate::audio_types::{EncodingFlag, Endianness};
use crate::wav::WavStreamProcessor;
use js_sys::{Array, Float32Array, Int16Array, Object, Reflect};
use wasm_bindgen::prelude::*;
use web_sys::Worker;

#[wasm_bindgen]
struct WavToPkt {
    wav_reader: WavStreamProcessor,
    audio_packets: Vec<u8>,
    frame_size: usize,
    packets: Vec<Vec<u8>>,
    bitrate: usize,
    widow: Vec<u8>,
    idx: usize,
}

#[wasm_bindgen]
impl WavToPkt {
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
            idx: 0,
        }
    }

    #[wasm_bindgen]
    pub fn bits_per_sample(&self) -> usize {
        self.wav_reader.bits_per_sample()
    }

    #[wasm_bindgen]
    pub fn channel_count(&self) -> usize {
        self.wav_reader.channel_count()
    }

    #[wasm_bindgen]
    pub fn sampling_rate(&self) -> usize {
        self.wav_reader.sampling_rate()
    }

    #[wasm_bindgen]
    pub fn into_frames(&mut self, data: &[u8]) -> JsValue {
        self.idx = self.idx.wrapping_add(1);

        let result = Object::new();

        Reflect::set(&result, &JsValue::from_str("ok"), &JsValue::from(false)).unwrap();

        match self.wav_reader.add(data) {
            Ok(Some(audio_data)) => self._into_frames(audio_data.data(), false),
            Ok(None) => {
                Reflect::set(&result, &JsValue::from_str("ok"), &JsValue::from(true)).unwrap();
                Reflect::set(
                    &result,
                    &JsValue::from_str("msg"),
                    &JsValue::from("no wav data"),
                )
                .unwrap();

                return result.into();
            }
            Err(err) => {
                Reflect::set(
                    &result,
                    &JsValue::from_str("err"),
                    &JsValue::from(err.to_string()),
                )
                .unwrap();

                return result.into();
            }
        }
    }

    #[wasm_bindgen]
    pub fn set_frame(&mut self, data: &[u8]) {
        let mut packet_data: Vec<u8> = Vec::new();
        let header = FrameHeader::new(
            EncodingFlag::Opus,
            self.frame_size.try_into().unwrap(),
            self.wav_reader.sampling_rate().try_into().unwrap(),
            self.wav_reader.channel_count().try_into().unwrap(),
            self.wav_reader.bits_per_sample().try_into().unwrap(),
            Endianness::LittleEndian,
        );
        header.encode(&mut packet_data).unwrap();
        packet_data.extend_from_slice(&data);
        self.packets.push(packet_data);
    }

    #[wasm_bindgen]
    pub fn flush(&mut self) -> Vec<u8> {
        if self.widow.len() > 0 {
            let _ = self._into_frames(&self.widow.clone(), true);
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
        let sampling_rate = self.wav_reader.sampling_rate() as usize;
        let bytes_per_sample = bits_per_sample / 8;
        let audio_format = self.wav_reader.audio_format();

        let result = Object::new();
        Reflect::set(
            &result,
            &JsValue::from_str("len"),
            &JsValue::from(data.len()),
        )
        .unwrap();

        Reflect::set(&result, &JsValue::from_str("ok"), &JsValue::from(false)).unwrap();
        Reflect::set(&result, &JsValue::from_str("seq"), &JsValue::from(self.idx)).unwrap();

        let chunk_size = self.frame_size * channel_count * bytes_per_sample as usize;

        let mut owned_data;
        if self.widow.len() > 0 {
            owned_data = self.widow.clone();
            owned_data.extend_from_slice(&data);
            self.widow.drain(..);
        } else {
            owned_data = data.to_owned();
        }

        let mut converted_data: Vec<Vec<i16>> = Vec::new();

        for chunk in owned_data.chunks(chunk_size) {
            if chunk.len() < chunk_size {
                self.widow.extend_from_slice(&chunk);
                break;
            }

            let src = match bits_per_sample {
                16 => s16le_to_i16(chunk),
                24 => s24le_to_i16(chunk),
                32 => {
                    if audio_format == EncodingFlag::PCMFloat {
                        f32le_to_i16(chunk)
                    } else {
                        s32le_to_i16(chunk)
                    }
                }
                _ => {
                    Reflect::set(
                        &result,
                        &JsValue::from_str("msg"),
                        &JsValue::from("unsupported bits_per_sample"),
                    )
                    .unwrap();

                    Reflect::set(
                        &result,
                        &JsValue::from_str("val"),
                        &JsValue::from(bits_per_sample),
                    )
                    .unwrap();

                    return result.into();
                }
            };
            converted_data.push(src);
        }

        Reflect::set(&result, &JsValue::from_str("ok"), &JsValue::from(true)).unwrap();

        if converted_data.is_empty() {
            return result.into();
        }

        let mut nested_array = Array::new();
        for src in converted_data {
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
        Reflect::set(
            &result,
            &JsValue::from_str("bits_per_sample"),
            &JsValue::from(bits_per_sample),
        )
        .unwrap();
        Reflect::set(
            &result,
            &JsValue::from_str("sampling_rate"),
            &JsValue::from(sampling_rate),
        )
        .unwrap();

        result.into()
    }

    fn reset(&mut self) {
        self.wav_reader = WavStreamProcessor::new();
    }
}

#[wasm_bindgen]
pub struct WavToPcm {
    wav: WavStreamProcessor,
}

#[wasm_bindgen]
impl WavToPcm {
    #[wasm_bindgen]
    pub fn new() -> Self {
        let wav = WavStreamProcessor::new();
        Self { wav }
    }

    #[wasm_bindgen]
    pub fn add(&mut self, data: &[u8]) -> JsValue {
        let result = Object::new();
        Reflect::set(&result, &JsValue::from_str("ok"), &JsValue::from(false)).unwrap();
        match self.wav.add(data) {
            Ok(Some(audio)) => {
                Reflect::set(&result, &JsValue::from_str("ok"), &JsValue::from(true)).unwrap();
                Reflect::set(
                    &result,
                    &JsValue::from_str("bits_per_sample"),
                    &JsValue::from(audio.bits_per_sample()),
                )
                .unwrap();
                Reflect::set(
                    &result,
                    &JsValue::from_str("sampling_rate"),
                    &JsValue::from(audio.sampling_rate()),
                )
                .unwrap();
                Reflect::set(
                    &result,
                    &JsValue::from_str("channel_count"),
                    &JsValue::from(audio.channel_count()),
                )
                .unwrap();

                let channels = match audio.bits_per_sample() {
                    16 => deinterleave_vecs_i16(&data, audio.channel_count() as usize)
                        .iter()
                        .map(|a| vec_i16_to_f32(a.clone()))
                        .collect(),
                    24 => deinterleave_vecs_s24(&data, audio.channel_count() as usize)
                        .iter()
                        .map(|a| vec_i32_to_f32(a.clone()))
                        .collect(),
                    32 => deinterleave_vecs_f32(&data, audio.channel_count() as usize),
                    _ => todo!(),
                };

                let js_array = channels
                    .iter()
                    .map(|channel| {
                        channel
                            .iter()
                            .map(|&value| JsValue::from_f64(f64::from(value)))
                            .collect::<Array>()
                    })
                    .collect::<Array>();
                Reflect::set(&result, &JsValue::from_str("channels"), &js_array).unwrap();
            }
            Ok(None) => {
                return JsValue::from(result);
            }
            Err(error) => {
                // Handle the error
                println!("Error: {}", error);
            }
        }

        return JsValue::from(result);
    }
}

/// Run entry point for the main thread.
#[wasm_bindgen]
pub fn startup(path: String) -> Worker {
    let worker_handle = Worker::new(&path).unwrap();
    worker_handle
}
