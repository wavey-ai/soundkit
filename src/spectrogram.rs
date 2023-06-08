use rustfft::algorithm::Radix4;
use rustfft::num_traits::Zero;
use rustfft::{num_complex::Complex, FftPlanner};
use std::sync::Arc;

extern crate image;
extern crate imageproc;
use image::codecs::png::PngEncoder;
use image::{DynamicImage, GrayImage};
use imageproc::contrast::equalize_histogram;
use std::io::Cursor;

pub fn save_spectrogram(frames: Vec<Vec<Vec<f32>>>, eq: bool) {
    for i in 0..frames.len() {
        let width = frames[i].len();
        let height = frames[i][0].len();

        let mut imgbuf = GrayImage::new(width as u32, height as u32);
        for (x, frame) in frames[i].iter().enumerate() {
            let frame: Vec<f32> = frame.iter().rev().copied().collect();
            for (y, &value) in frame.iter().enumerate() {
                let pixel = ((value * 255.0) as u8).min(255).max(0);
                imgbuf.put_pixel(x as u32, y as u32, image::Luma([pixel]));
            }
        }

        let name: String = format!("out/spectrogram{}.png", i);
        if eq {
            let equalized_image = equalize_histogram(&imgbuf);
            let dynamic_image = DynamicImage::ImageLuma8(equalized_image);
            dynamic_image.save(name).unwrap();
        } else {
            imgbuf.save(name).unwrap();
        }
    }
}

pub fn spectrogram_frame_bytes(frame: Vec<f32>) -> Vec<u8> {
    let height = frame.len();
    let mut imgbuf = GrayImage::new(1, height as u32);

    for (y, value) in frame.into_iter().rev().enumerate() {
        let pixel = ((value * 255.0) as u8).min(255).max(0);
        imgbuf.put_pixel(0, y as u32, image::Luma([pixel]));
    }

    let equalized_image = equalize_histogram(&imgbuf);
    let dynamic_image = DynamicImage::ImageLuma8(equalized_image);

    let mut bytes: Vec<u8> = Vec::new();
    let encoder = PngEncoder::new(Cursor::new(&mut bytes));
    encoder
        .encode(
            dynamic_image.to_bytes().as_slice(),
            1,
            height as u32,
            image::ColorType::L8,
        )
        .unwrap();

    bytes
}

pub enum PcmData {
    F32(Vec<f32>),
    I16(Vec<i16>),
}

pub fn frame_spectrogram(
    pcm_data: PcmData,
    fft_size: usize,
    hop_size: usize,
    percent_norm: f32,
) -> Vec<Vec<f32>> {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);

    // Compute Hamming window.
    let hamming_window: Vec<f32> = (0..fft_size)
        .map(|i| {
            0.54 - 0.46 * ((2.0 * std::f32::consts::PI * i as f32) / (fft_size as f32 - 1.0)).cos()
        })
        .collect();

    match pcm_data {
        PcmData::F32(data) => process_frame(
            &mut planner,
            &fft,
            &hamming_window,
            &data,
            fft_size,
            hop_size,
            percent_norm,
        ),
        PcmData::I16(data) => {
            let data_f32: Vec<f32> = data.into_iter().map(|x| x as f32).collect();
            process_frame(
                &mut planner,
                &fft,
                &hamming_window,
                &data_f32,
                fft_size,
                hop_size,
                percent_norm,
            )
        }
    }
}

fn process_frame(
    planner: &mut FftPlanner<f32>,
    fft: &Arc<dyn rustfft::Fft<f32>>,
    hamming_window: &Vec<f32>,
    data: &Vec<f32>,
    fft_size: usize,
    hop_size: usize,
    percent_norm: f32,
) -> Vec<Vec<f32>> {
    // Initialize empty vector for storing all the processed FFT frames
    let mut all_fft_frames: Vec<Vec<f32>> = Vec::new();

    let mut start = 0;
    while start + fft_size <= data.len() {
        // Apply the window function and FFT to the current portion of data
        let mut input_output: Vec<Complex<f32>> = data[start..start + fft_size]
            .into_iter()
            .enumerate()
            .map(|(i, x)| Complex::new(x * hamming_window[i], 0.0))
            .collect();

        fft.process(&mut input_output);

        // Normalize the magnitudes to 0.0 - 1.0 range.
        let max_magnitude = input_output.iter().map(|c| c.norm()).fold(0.0, f32::max);
        if percent_norm > 0.0 {
            let normalized_magnitudes = percentile_norm(
                &input_output
                    .iter()
                    .take(fft_size / 2)
                    .map(|c| c.norm())
                    .collect::<Vec<_>>(),
                percent_norm,
            );
            all_fft_frames.push(normalized_magnitudes);
        } else {
            let normalized_magnitudes: Vec<f32> = input_output
                .iter()
                .map(|c| c.norm() / max_magnitude)
                .take(fft_size / 2)
                .collect();

            all_fft_frames.push(normalized_magnitudes);
        }
        start += hop_size;
    }

    all_fft_frames
}

fn percentile_norm(input: &[f32], percentile: f32) -> Vec<f32> {
    assert!(percentile > 0.0 && percentile <= 1.0);

    let mut sorted = input.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let index = (sorted.len() as f32 * percentile).ceil() as usize - 1;
    let norm_factor = sorted[index];

    input.iter().map(|&x| x / norm_factor).collect()
}
