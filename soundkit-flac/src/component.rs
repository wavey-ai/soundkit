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

//! Components to be written in the output file.

mod bitrepr;
mod datatype;
mod verify;

pub use bitrepr::BitRepr;
pub(crate) use datatype::BlockSizeSpec;
pub use datatype::ChannelAssignment;
pub use datatype::Constant;
pub use datatype::FixedLpc;
pub use datatype::Frame;
pub use datatype::FrameHeader;
pub use datatype::FrameOffset;
pub use datatype::Lpc;
pub(crate) use datatype::MetadataBlock;
pub use datatype::MetadataBlockData;
pub use datatype::QuantizedParameters;
pub use datatype::Residual;
pub(crate) use datatype::SampleRateSpec;
pub(crate) use datatype::SampleSizeSpec;
pub use datatype::Stream;
pub use datatype::StreamInfo;
pub use datatype::SubFrame;
pub use datatype::Verbatim;
