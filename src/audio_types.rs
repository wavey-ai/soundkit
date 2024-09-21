pub enum PcmData {
    I16(Vec<Vec<i16>>),
    I32(Vec<Vec<i32>>),
    F32(Vec<Vec<f32>>),
}

#[derive(Debug)]
pub struct AudioData {
    bits_per_sample: u8,
    channel_count: u8,
    data: Vec<u8>,
    sampling_rate: u32,
    audio_format: EncodingFlag,
    endianness: Endianness,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Endianness {
    LittleEndian,
    BigEndian,
}

impl AudioData {
    pub fn new(
        bits_per_sample: u8,
        channel_count: u8,
        sampling_rate: u32,
        data: Vec<u8>,
        audio_format: EncodingFlag,
        endianness: Endianness,
    ) -> Self {
        AudioData {
            bits_per_sample,
            channel_count,
            sampling_rate,
            data,
            audio_format,
            endianness,
        }
    }

    pub fn bits_per_sample(&self) -> u8 {
        self.bits_per_sample
    }

    pub fn channel_count(&self) -> u8 {
        self.channel_count
    }

    pub fn sampling_rate(&self) -> u32 {
        self.sampling_rate
    }

    pub fn data(&self) -> &Vec<u8> {
        &self.data
    }

    pub fn audio_format(&self) -> EncodingFlag {
        self.audio_format
    }

    pub fn endianness(&self) -> Endianness {
        self.endianness
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum EncodingFlag {
    PCMSigned = 0,
    PCMFloat = 1,
    Opus = 2,
    FLAC = 3,
    AAC = 4,
}
