use crate::audio_bytes::{f32le_to_i16, s16le_to_i16, s24le_to_i16, s32le_to_i16};
use crate::audio_pipeline::{audio_to_f32_channels, PcmFrameSplitter};
use crate::wav::WavStreamProcessor;
use frame_header::{EncodingFlag, Endianness, FrameHeader};
use js_sys::{Array, Int16Array, Object, Reflect};
use wasm_bindgen::prelude::*;
use web_sys::Worker;

/// Splits a WAV stream into 16-bit frames for a JavaScript encoder and
/// collects the encoded packets it returns.
///
/// Call `into_frames` for each chunk, `finish_frames` for the last partial
/// frame (padded with silence), `set_frame` for each encoded packet, and
/// `flush` for the packed stream, which also resets the object.
#[wasm_bindgen]
struct WavToPkt {
    wav_reader: WavStreamProcessor,
    frame_size: usize,
    packets: Vec<Vec<u8>>,
    bitrate: usize,
    splitter: Option<PcmFrameSplitter>,
    idx: usize,
}

#[wasm_bindgen]
impl WavToPkt {
    #[wasm_bindgen]
    pub fn new(bitrate: usize, frame_size: usize) -> Self {
        let wav_reader = WavStreamProcessor::new();

        Self {
            wav_reader,
            frame_size,
            packets: Vec::new(),
            bitrate,
            splitter: None,
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
            None,
            None,
        )
        .unwrap();
        header.encode(&mut packet_data).unwrap();
        packet_data.extend_from_slice(&data);
        self.packets.push(packet_data);
    }

    /// The last partial frame, padded with silence, in the result shape of
    /// `into_frames`; no frames when none is held.
    #[wasm_bindgen]
    pub fn finish_frames(&mut self) -> JsValue {
        self._into_frames(&[], true)
    }

    #[wasm_bindgen]
    pub fn flush(&mut self) -> Vec<u8> {
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

        let chunk_size = self.frame_size * channel_count * bytes_per_sample;
        let splitter = self
            .splitter
            .get_or_insert_with(|| PcmFrameSplitter::new(chunk_size));
        let mut frames = match splitter.push(data) {
            Ok(frames) => frames,
            Err(error) => {
                Reflect::set(&result, &JsValue::from_str("msg"), &JsValue::from(error)).unwrap();
                return result.into();
            }
        };
        if is_last {
            frames.extend(splitter.finish());
        }

        let mut converted_data: Vec<Vec<i16>> = Vec::with_capacity(frames.len());
        for chunk in &frames {
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

        let nested_array = Array::new();
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
        self.packets.clear();
        self.splitter = None;
        self.idx = 0;
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

                // The samples of the decoded block, scaled by their own depth.
                let channels = match audio_to_f32_channels(&audio) {
                    Ok(channels) => channels,
                    Err(error) => {
                        Reflect::set(&result, &JsValue::from_str("ok"), &JsValue::from(false))
                            .unwrap();
                        Reflect::set(&result, &JsValue::from_str("err"), &JsValue::from(error))
                            .unwrap();
                        return JsValue::from(result);
                    }
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
                Reflect::set(&result, &JsValue::from_str("err"), &JsValue::from(error)).unwrap();
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
