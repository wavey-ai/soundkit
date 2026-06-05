use crate::bitreader::BitReader;
use crate::error::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementId {
    SingleChannel,
    ChannelPair,
    ChannelCoupling,
    LowFrequency,
    DataStream,
    ProgramConfig,
    Fill,
    End,
}

impl ElementId {
    pub const fn from_bits(value: u8) -> Self {
        match value {
            0 => Self::SingleChannel,
            1 => Self::ChannelPair,
            2 => Self::ChannelCoupling,
            3 => Self::LowFrequency,
            4 => Self::DataStream,
            5 => Self::ProgramConfig,
            6 => Self::Fill,
            7 => Self::End,
            _ => unreachable!(),
        }
    }

    pub const fn needs_instance_tag(self) -> bool {
        matches!(
            self,
            Self::SingleChannel
                | Self::ChannelPair
                | Self::ChannelCoupling
                | Self::LowFrequency
                | Self::DataStream
                | Self::ProgramConfig
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ElementInstanceTag(pub u8);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawElementHeader {
    pub id: ElementId,
    pub tag: Option<ElementInstanceTag>,
}

impl RawElementHeader {
    pub fn read(reader: &mut BitReader<'_>) -> Result<Self> {
        let id = ElementId::from_bits(reader.read_u8(3)?);
        let tag = if id.needs_instance_tag() {
            Some(ElementInstanceTag(reader.read_u8(4)?))
        } else {
            None
        };

        Ok(Self { id, tag })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_channel_pair_header_with_instance_tag() {
        let mut reader = BitReader::new(&[0b0011_0110]);
        let header = RawElementHeader::read(&mut reader).unwrap();

        assert_eq!(
            header,
            RawElementHeader {
                id: ElementId::ChannelPair,
                tag: Some(ElementInstanceTag(0b1011)),
            }
        );
        assert_eq!(reader.bit_pos(), 7);
    }

    #[test]
    fn reads_end_header_without_instance_tag() {
        let mut reader = BitReader::new(&[0b1110_0000]);
        let header = RawElementHeader::read(&mut reader).unwrap();

        assert_eq!(
            header,
            RawElementHeader {
                id: ElementId::End,
                tag: None,
            }
        );
        assert_eq!(reader.bit_pos(), 3);
    }
}
