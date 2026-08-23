// Copyright 2022-2024 Google LLC
// Copyright 2025- flacenc-rs developers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Module for input source handling.

use std::fmt;

use md5::Digest;

use super::arrayutils::deinterleave;
use super::arrayutils::find_min_and_max;
use super::arrayutils::i32s_to_le_bytes;
use super::arrayutils::le_bytes_to_i32s;
use super::constant::MAX_BLOCK_SIZE;
use super::constant::MAX_CHANNELS;
use super::constant::MIN_BLOCK_SIZE;
use super::error::verify_range;
use super::error::verify_true;
use super::error::SourceError;
use super::error::VerifyError;

/// Traits for buffer-like objects that accumulate encoded input samples.
///
/// An encoder is expected to call one of the `fill_*` method declared in
/// this trait.
///
/// An impl of `Fill` must accept the samples that is shorter than the pre-
/// defined length for e.g. the last frame handling. On the other hand,
/// `Fill` is expected to return an error if the number of samples is larger
/// than the block size.
pub trait Fill {
    /// Fills the target variable with the given interleaved samples.
    ///
    /// # Errors
    ///
    /// This may fail when configuration of `Fill` is not consistent with the
    /// input `interleaved` values.
    ///
    /// # Examples
    ///
    /// [`FrameBuf`] implements `Fill`.
    ///
    /// ```
    /// # use soundkit_flac::source::{Fill, FrameBuf};
    /// let mut fb = FrameBuf::with_size(8, 1024).unwrap();
    /// fb.fill_interleaved(&[0i32; 8 * 1024]);
    /// ```
    fn fill_interleaved(&mut self, interleaved: &[i32]) -> Result<(), SourceError>;

    /// Fills target with the little-endian bytes that represent samples.
    ///
    /// # Errors
    ///
    /// This may fail when configuration of `Fill` is not consistent with the
    /// input `bytes` or `bytes_per_sample` values.
    ///
    /// # Examples
    ///
    /// [`FrameBuf`] implements `Fill`.
    ///
    /// ```
    /// # use soundkit_flac::source::{Fill, FrameBuf};
    /// let mut fb = FrameBuf::with_size(2, 64).unwrap();
    /// // Note that `FrameBuf` (or `Fill` in general) accepts shorter inputs.
    /// fb.fill_le_bytes(&[0x12, 0x34, 0x54, 0x76, 0x56, 0x78, 0x10, 0x32], 2);
    /// // this FrameBuf now has 2 channels with elements:
    /// //   - channel-1 (left) : [0x3412, 0x7856]
    /// //   - channel-2 (right): [0x7654, 0x3210]
    /// ```
    fn fill_le_bytes(&mut self, bytes: &[u8], bytes_per_sample: usize) -> Result<(), SourceError>;
}

impl<T: Fill, U: Fill> Fill for (T, U) {
    #[inline]
    fn fill_interleaved(&mut self, interleaved: &[i32]) -> Result<(), SourceError> {
        self.0.fill_interleaved(interleaved)?;
        self.1.fill_interleaved(interleaved)
    }

    #[inline]
    fn fill_le_bytes(&mut self, bytes: &[u8], bytes_per_sample: usize) -> Result<(), SourceError> {
        self.0.fill_le_bytes(bytes, bytes_per_sample)?;
        self.1.fill_le_bytes(bytes, bytes_per_sample)
    }
}

impl<T> Fill for &mut T
where
    T: Fill,
{
    #[inline]
    fn fill_interleaved(&mut self, interleaved: &[i32]) -> Result<(), SourceError> {
        T::fill_interleaved(self, interleaved)
    }

    #[inline]
    fn fill_le_bytes(&mut self, bytes: &[u8], bytes_per_sample: usize) -> Result<(), SourceError> {
        T::fill_le_bytes(self, bytes, bytes_per_sample)
    }
}

/// Reusable buffer for multi-channel framed signals.
#[derive(Clone, Debug)]
pub struct FrameBuf {
    samples: Vec<i32>,
    size: usize,
    /// The number of loaded inter-channel samples.
    ///
    /// this can be smaller than `self.samples.len() / self.channels` for the last block of the
    /// stream.
    filled_size: usize,
    /// Working buffer.
    ///
    /// This is currently only used in `read_le_bytes` (for storing `i32`-upcasted samples).
    readbuf: Vec<i32>,
}

impl FrameBuf {
    /// Constructs new 2-channel `FrameBuf` that will be later resized.
    ///
    /// This is a safe constructor that never fails, and always produce a valid
    /// `FrameBuf`. This constructor is intended to be used for preparing
    /// reusable buffer for stereo coding.
    pub(crate) fn new_stereo_buffer() -> Self {
        Self {
            samples: vec![0i32; 256 * 2],
            size: 256,
            filled_size: 0,
            readbuf: vec![],
        }
    }

    /// Constructs `FrameBuf` of the specified size.
    ///
    /// # Errors
    ///
    /// Returns `VerifyError` if arguments are out of the ranges of FLAC
    /// specifications. Specifically `channels` must be in `1..=[MAX_CHANNELS]` and
    /// size must be in `MIN_BLOCK_SIZE]..=MAX_BLOCK_SIZE`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use soundkit_flac::source::FrameBuf;
    /// let fb = FrameBuf::with_size(2, 1024).unwrap();
    /// assert_eq!(fb.size(), 1024);
    /// ```
    pub fn with_size(channels: usize, size: usize) -> Result<Self, VerifyError> {
        verify_range!("FrameBuf::with_size (channels)", channels, 1..=MAX_CHANNELS)?;
        verify_range!(
            "FrameBuf::with_size (block size)",
            size,
            MIN_BLOCK_SIZE..=MAX_BLOCK_SIZE
        )?;
        Ok(Self {
            samples: vec![0i32; size * channels],
            size,
            filled_size: 0,
            readbuf: vec![],
        })
    }

    /// Returns the size in the number of per-channel samples.
    ///
    /// # Examples
    ///
    /// ```
    /// # use soundkit_flac::source::FrameBuf;
    /// let fb = FrameBuf::with_size(2, 1024).unwrap();
    /// assert_eq!(fb.size(), 1024);
    /// ```
    pub const fn size(&self) -> usize {
        self.size
    }

    /// Returns the number of inter-channel samples written to this `FrameBuf`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use soundkit_flac::source::Fill;
    /// # use soundkit_flac::source::FrameBuf;
    /// let mut fb = FrameBuf::with_size(1, 1024).unwrap();
    /// fb.fill_interleaved(&[0, 1, 2, 3]);
    /// assert_eq!(fb.filled_size(), 4);
    /// ```
    pub const fn filled_size(&self) -> usize {
        self.filled_size
    }

    /// Fill stereo buffer with the stereo samples from the given iterator.
    ///
    /// This is currently only used for making M/S framebuffer from the L/R buffer.
    pub(crate) fn fill_stereo_with_iter<I>(&mut self, iter: I)
    where
        I: Iterator<Item = (i32, i32)>,
    {
        assert_eq!(2, self.channels());
        let (m_slice, s_slice) = self.samples.split_at_mut(self.size);
        self.filled_size = 0;
        let dest_iter = m_slice.iter_mut().zip(s_slice.iter_mut());
        for ((m, s), (dest_m, dest_s)) in iter.take(self.size).zip(dest_iter) {
            *dest_m = m;
            *dest_s = s;
            self.filled_size += 1;
        }
    }

    /// Resizes `FrameBuf`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use soundkit_flac::source::FrameBuf;
    /// let mut fb = FrameBuf::with_size(2, 1024).unwrap();
    /// assert_eq!(fb.size(), 1024);
    /// fb.resize(2048);
    /// assert_eq!(fb.size(), 2048);
    /// ```
    pub fn resize(&mut self, new_size: usize) {
        let channels = self.channels();
        self.size = new_size;
        self.samples.resize(self.size * channels, 0i32);
    }

    /// Returns the number of channels
    ///
    /// # Examples
    ///
    /// ```
    /// # use soundkit_flac::source::FrameBuf;
    /// let fb = FrameBuf::with_size(8, 1024).unwrap();
    /// assert_eq!(fb.channels(), 8);
    /// ```
    pub fn channels(&self) -> usize {
        self.samples.len() / self.size
    }

    /// Returns samples from the given channel.
    pub(crate) fn channel_slice(&self, ch: usize) -> &[i32] {
        &self.samples[ch * self.size..(ch * self.size + self.filled_size)]
    }

    /// Returns the internal representation of multichannel signals.
    #[cfg(test)]
    pub(crate) fn raw_slice(&self) -> &[i32] {
        &self.samples
    }

    /// Verifies data consistency with the given stream info.
    pub(crate) fn verify_samples(&self, bits_per_sample: usize) -> Result<(), VerifyError> {
        let max_allowed = (1i32 << (bits_per_sample - 1)) - 1;
        let min_allowed = -(1i32 << (bits_per_sample - 1));
        for ch in 0..self.channels() {
            let (min, max) = find_min_and_max::<64>(self.channel_slice(ch), 0i32);
            if min < min_allowed || max > max_allowed {
                return Err(VerifyError::new(
                    "input.framebuf",
                    &format!("input sample must be in the range of bits={bits_per_sample}"),
                ));
            }
        }
        Ok(())
    }
}

impl Fill for FrameBuf {
    fn fill_interleaved(&mut self, interleaved: &[i32]) -> Result<(), SourceError> {
        let stride = self.size();
        let channels = self.channels();
        deinterleave(interleaved, channels, stride, &mut self.samples);
        self.filled_size = interleaved.len() / channels;
        Ok(())
    }

    #[inline]
    fn fill_le_bytes(&mut self, bytes: &[u8], bytes_per_sample: usize) -> Result<(), SourceError> {
        let sample_count = bytes.len() / bytes_per_sample;
        self.readbuf.resize(sample_count, 0);
        le_bytes_to_i32s(bytes, &mut self.readbuf, bytes_per_sample);

        let stride = self.size();
        let channels = self.channels();
        deinterleave(&self.readbuf, self.channels(), stride, &mut self.samples);
        self.filled_size = sample_count / channels;
        Ok(())
    }
}

/// Context information being updated while frames are filled.
///
/// Some information such as MD5 of the input waveform is better handled on
/// the caller side rather than via frame buffers. `Context` holds such
/// context variables.
#[derive(Clone)]
pub struct Context {
    md5: md5::Md5,
    sample_bytes: Vec<u8>,
    bytes_per_sample: usize,
    channels: usize,
    sample_count: usize,
    frame_count: usize,
}

impl Context {
    /// Creates new context.
    ///
    /// # Panics
    ///
    /// Panics if `bits_per_sample > 32`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use soundkit_flac::source::Context;
    /// let ctx = Context::new(16, 2);
    /// assert!(ctx.current_frame_number().is_none());
    /// assert_eq!(ctx.total_samples(), 0);
    /// ```
    pub fn new(bits_per_sample: usize, channels: usize) -> Self {
        let bytes_per_sample = bits_per_sample.div_ceil(8);
        assert!(
            bytes_per_sample <= 4,
            "bits_per_sample={bits_per_sample} cannot be larger than 32."
        );
        Self {
            md5: md5::Md5::new(),
            sample_bytes: Vec::new(),
            bytes_per_sample,
            channels,
            sample_count: 0,
            frame_count: 0,
        }
    }

    /// Returns bytes-per-sample configuration of this `Context`.
    #[inline]
    pub fn bytes_per_sample(&self) -> usize {
        self.bytes_per_sample
    }

    /// Returns the count of the last frame loaded.
    ///
    /// # Examples
    ///
    /// ```
    /// # use soundkit_flac::source::{Context, Fill};
    /// let mut ctx = Context::new(16, 2);
    /// assert!(ctx.current_frame_number().is_none());
    ///
    /// ctx.fill_interleaved(&[0, -1, -2, 3]);
    /// assert_eq!(ctx.current_frame_number(), Some(0usize));
    /// ```
    #[inline]
    #[allow(clippy::unnecessary_lazy_evaluations)] // false-alarm
    pub fn current_frame_number(&self) -> Option<usize> {
        (self.frame_count > 0).then(|| self.frame_count - 1)
    }

    /// Returns MD5 digest of the consumed samples.
    ///
    /// # Examples
    ///
    /// ```
    /// # use soundkit_flac::source::Context;
    /// let ctx = Context::new(16, 2);
    /// let zero_md5 = [
    ///     0xD4, 0x1D, 0x8C, 0xD9, 0x8F, 0x00, 0xB2, 0x04,
    ///     0xE9, 0x80, 0x09, 0x98, 0xEC, 0xF8, 0x42, 0x7E,
    /// ];
    /// assert_eq!(ctx.md5_digest(), zero_md5);
    /// // it doesn't change if you don't call "update" functions.
    /// assert_eq!(ctx.md5_digest(), zero_md5);
    /// ```
    #[inline]
    pub fn md5_digest(&self) -> [u8; 16] {
        self.md5.clone().finalize().into()
    }

    /// Returns the number of samples loaded.
    ///
    /// # Examples
    ///
    /// ```
    /// # use soundkit_flac::source::{Context, Fill};
    /// let mut ctx = Context::new(16, 2);
    ///
    /// ctx.fill_interleaved(&[0, -1, -2, 3]);
    /// assert_eq!(ctx.total_samples(), 2);
    /// ```
    #[inline]
    pub fn total_samples(&self) -> usize {
        self.sample_count
    }
}

impl Fill for Context {
    fn fill_interleaved(&mut self, interleaved: &[i32]) -> Result<(), SourceError> {
        if interleaved.is_empty() {
            return Ok(());
        }
        self.sample_bytes
            .resize(interleaved.len() * self.bytes_per_sample, 0);
        i32s_to_le_bytes(interleaved, &mut self.sample_bytes, self.bytes_per_sample);
        self.md5.update(&self.sample_bytes);
        self.sample_count += interleaved.len() / self.channels;
        self.frame_count += 1;
        Ok(())
    }

    #[inline]
    fn fill_le_bytes(&mut self, bytes: &[u8], bytes_per_sample: usize) -> Result<(), SourceError> {
        if bytes.is_empty() {
            return Ok(());
        }
        self.md5.update(bytes);
        self.sample_count += bytes.len() / self.channels / bytes_per_sample;
        self.frame_count += 1;
        Ok(())
    }
}

impl fmt::Debug for Context {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let digest = format!("{:?}", self.md5.clone().finalize());
        f.debug_struct("Context")
            .field("bytes_per_sample", &self.bytes_per_sample)
            .field("channels", &self.channels)
            .field("sample_count", &self.sample_count)
            .field("frame_count", &self.frame_count)
            .field("md5", &digest)
            .finish()
    }
}
