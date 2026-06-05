mod bitreader;
mod channel;
mod config;
mod decoder;
mod dsp;
mod error;
mod ics;
mod pulse;
mod scalefactor;
mod section;
mod sfb;
mod spectral;
mod stereo;
mod syntax;
mod tns;
mod vlc;

pub use bitreader::BitReader;
pub use channel::{
    ChannelPairElementHeader, IndividualChannelStream, IndividualChannelStreamPrefix, MidSideMask,
    SingleChannelElement,
};
pub use config::{
    AudioObjectType, AudioSpecificConfig, ChannelConfig, SamplingFrequency,
    SAMPLES_PER_AAC_LC_FRAME,
};
pub use decoder::{AacLcDecoder, AacLcFrame, PlanarF32};
pub use dsp::{
    dequantize_signed, scalefactor_multiplier, AacDsp, DspChannel, ImdctTransform,
    LONG_SPECTRUM_LEN, LONG_WINDOW_LEN, SHORT_SPECTRUM_LEN, SHORT_WINDOW_LEN,
};
pub use error::{AacLcError, Result};
pub use ics::{IcsInfo, WindowSequence, WindowShape, MAX_WINDOW_GROUPS};
pub use pulse::{Pulse, PulseData, MAX_PULSES};
pub use scalefactor::{
    NotImplementedScaleFactorDecoder, ScaleFactorData, ScaleFactorDecoder, ScaleFactorValue,
    StandardScaleFactorDecoder, VlcScaleFactorDecoder,
};
pub use section::{SectionCodebook, SectionData, MAX_SCALE_FACTOR_BANDS};
pub use sfb::{
    long_window_band_layout, long_window_band_offsets, short_window_band_layout,
    short_window_band_offsets, ScaleFactorBandOffsets, ScaleFactorBandWindow,
};
pub use spectral::{
    decode_signed_tuples, decode_unsigned_escape_pairs, decode_unsigned_tuples, BandLayout,
    NotImplementedSpectralDecoder, SpectralCodebookKind, SpectralCoefficients, SpectralDecoder,
    StandardSpectralDecoder, TupleCodebook,
};
pub use stereo::{apply_mid_side_long, apply_mid_side_short};
pub use syntax::{ElementId, ElementInstanceTag, RawElementHeader};
pub use tns::{
    apply_tns, lc_tns_max_bands, TnsData, TnsFilter, TnsWindow, MAX_TNS_FILTERS, MAX_TNS_ORDER,
};
pub use vlc::{VlcEntry, VlcTable};
