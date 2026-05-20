#[cfg(feature = "webm")]
use soundkit_webm::{WebmAudioDemuxEvent, WebmAudioDemuxer};

const MIN_DETECTION_BYTES: usize = 8192;
const MAX_DETECTION_BYTES: usize = 65_536;
#[cfg(feature = "mp4")]
const REGULAR_MP4_DEFERRED_MDAT_MAX_BYTES: u64 = 512 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AudioContainer {
    Mp4,
    WebM,
    MpegTs,
}

impl AudioContainer {
    pub fn as_str(&self) -> &'static str {
        match self {
            AudioContainer::Mp4 => "mp4",
            AudioContainer::WebM => "webm",
            AudioContainer::MpegTs => "mpeg-ts",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AudioCodec {
    Aac,
    Opus,
    Vorbis,
    Mp3,
    Ac3,
    Unknown(String),
}

impl AudioCodec {
    pub fn as_str(&self) -> &str {
        match self {
            AudioCodec::Aac => "aac",
            AudioCodec::Opus => "opus",
            AudioCodec::Vorbis => "vorbis",
            AudioCodec::Mp3 => "mp3",
            AudioCodec::Ac3 => "ac3",
            AudioCodec::Unknown(codec) => codec.as_str(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AudioPacketFormat {
    Adts,
    Latm,
    Raw,
}

impl AudioPacketFormat {
    pub fn as_str(&self) -> &'static str {
        match self {
            AudioPacketFormat::Adts => "adts",
            AudioPacketFormat::Latm => "latm",
            AudioPacketFormat::Raw => "raw",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AudioTrackConfig {
    pub container: AudioContainer,
    pub codec: AudioCodec,
    pub packet_format: Option<AudioPacketFormat>,
    pub codec_id: Option<String>,
    pub track_id: Option<u64>,
    pub pid: Option<u16>,
    pub stream_type: Option<u8>,
    pub sample_rate: Option<u32>,
    pub channels: Option<u8>,
    pub sample_count: Option<u32>,
    pub codec_private: Vec<u8>,
    pub pre_skip: Option<u16>,
    pub output_gain: Option<i16>,
    pub mapping_family: Option<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AudioTrackPacket {
    pub container: AudioContainer,
    pub codec: AudioCodec,
    pub format: AudioPacketFormat,
    pub data: Vec<u8>,
    pub raw_data: Option<Vec<u8>>,
    pub track_id: Option<u64>,
    pub pid: Option<u16>,
    pub stream_type: Option<u8>,
    pub sample_id: Option<u32>,
    pub start_time: Option<u64>,
    pub duration: Option<u32>,
    pub rendering_offset: Option<i32>,
    pub is_sync: Option<bool>,
    pub timecode: Option<i64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AudioDemuxEvent {
    Config(AudioTrackConfig),
    Packet(AudioTrackPacket),
}

pub struct AudioTrackDemuxer {
    state: DemuxerState,
}

enum DemuxerState {
    Detecting {
        buffer: Vec<u8>,
        bytes_collected: usize,
    },
    #[cfg(feature = "mp4")]
    Mp4(Mp4AudioDemuxer),
    #[cfg(feature = "webm")]
    WebM(WebmAudioDemuxer),
    #[cfg(feature = "mpeg-ts")]
    MpegTs(MpegTsAudioDemuxer),
    Finished,
}

impl AudioTrackDemuxer {
    pub fn new_auto() -> Self {
        Self {
            state: DemuxerState::Detecting {
                buffer: Vec::new(),
                bytes_collected: 0,
            },
        }
    }

    pub fn new_with_format(format: &str) -> Result<Self, String> {
        Ok(Self {
            state: demuxer_for_format(format)?,
        })
    }

    pub fn push(&mut self, bytes: &[u8]) -> Result<Vec<AudioDemuxEvent>, String> {
        let state = std::mem::replace(&mut self.state, DemuxerState::Finished);
        match state {
            DemuxerState::Detecting {
                mut buffer,
                bytes_collected,
            } => {
                buffer.extend_from_slice(bytes);
                let new_bytes_collected = bytes_collected + bytes.len();

                if new_bytes_collected < MIN_DETECTION_BYTES {
                    self.state = DemuxerState::Detecting {
                        buffer,
                        bytes_collected: new_bytes_collected,
                    };
                    return Ok(Vec::new());
                }

                match detect_and_init_demuxer(&buffer) {
                    Ok(mut demuxer) => {
                        let events = process_state(&mut demuxer, &buffer, false)?;
                        self.state = demuxer;
                        Ok(events)
                    }
                    Err(error) if new_bytes_collected < MAX_DETECTION_BYTES => {
                        self.state = DemuxerState::Detecting {
                            buffer,
                            bytes_collected: new_bytes_collected,
                        };
                        if bytes.is_empty() {
                            self.state = DemuxerState::Finished;
                            Err(error)
                        } else {
                            Ok(Vec::new())
                        }
                    }
                    Err(error) => {
                        self.state = DemuxerState::Finished;
                        Err(error)
                    }
                }
            }
            state @ DemuxerState::Finished => {
                self.state = state;
                Err("demuxer is already finished".to_string())
            }
            #[cfg(all(feature = "mp4", feature = "webm", feature = "mpeg-ts"))]
            mut
            state @ (DemuxerState::Mp4(_) | DemuxerState::WebM(_) | DemuxerState::MpegTs(_)) => {
                let events = process_state(&mut state, bytes, false)?;
                self.state = state;
                Ok(events)
            }
            #[cfg(all(feature = "mp4", feature = "webm", not(feature = "mpeg-ts")))]
            mut state @ (DemuxerState::Mp4(_) | DemuxerState::WebM(_)) => {
                let events = process_state(&mut state, bytes, false)?;
                self.state = state;
                Ok(events)
            }
            #[cfg(all(feature = "mp4", feature = "mpeg-ts", not(feature = "webm")))]
            mut state @ (DemuxerState::Mp4(_) | DemuxerState::MpegTs(_)) => {
                let events = process_state(&mut state, bytes, false)?;
                self.state = state;
                Ok(events)
            }
            #[cfg(all(feature = "webm", feature = "mpeg-ts", not(feature = "mp4")))]
            mut state @ (DemuxerState::WebM(_) | DemuxerState::MpegTs(_)) => {
                let events = process_state(&mut state, bytes, false)?;
                self.state = state;
                Ok(events)
            }
            #[cfg(all(feature = "mp4", not(any(feature = "webm", feature = "mpeg-ts"))))]
            mut state @ DemuxerState::Mp4(_) => {
                let events = process_state(&mut state, bytes, false)?;
                self.state = state;
                Ok(events)
            }
            #[cfg(all(feature = "webm", not(any(feature = "mp4", feature = "mpeg-ts"))))]
            mut state @ DemuxerState::WebM(_) => {
                let events = process_state(&mut state, bytes, false)?;
                self.state = state;
                Ok(events)
            }
            #[cfg(all(feature = "mpeg-ts", not(any(feature = "mp4", feature = "webm"))))]
            mut state @ DemuxerState::MpegTs(_) => {
                let events = process_state(&mut state, bytes, false)?;
                self.state = state;
                Ok(events)
            }
        }
    }

    pub fn flush(&mut self) -> Result<Vec<AudioDemuxEvent>, String> {
        let state = std::mem::replace(&mut self.state, DemuxerState::Finished);
        match state {
            DemuxerState::Detecting { buffer, .. } => {
                let mut demuxer = detect_and_init_demuxer(&buffer)?;
                process_state(&mut demuxer, &buffer, true)
            }
            DemuxerState::Finished => Ok(Vec::new()),
            #[cfg(all(feature = "mp4", feature = "webm", feature = "mpeg-ts"))]
            mut
            state @ (DemuxerState::Mp4(_) | DemuxerState::WebM(_) | DemuxerState::MpegTs(_)) => {
                process_state(&mut state, &[], true)
            }
            #[cfg(all(feature = "mp4", feature = "webm", not(feature = "mpeg-ts")))]
            mut state @ (DemuxerState::Mp4(_) | DemuxerState::WebM(_)) => {
                process_state(&mut state, &[], true)
            }
            #[cfg(all(feature = "mp4", feature = "mpeg-ts", not(feature = "webm")))]
            mut state @ (DemuxerState::Mp4(_) | DemuxerState::MpegTs(_)) => {
                process_state(&mut state, &[], true)
            }
            #[cfg(all(feature = "webm", feature = "mpeg-ts", not(feature = "mp4")))]
            mut state @ (DemuxerState::WebM(_) | DemuxerState::MpegTs(_)) => {
                process_state(&mut state, &[], true)
            }
            #[cfg(all(feature = "mp4", not(any(feature = "webm", feature = "mpeg-ts"))))]
            mut state @ DemuxerState::Mp4(_) => process_state(&mut state, &[], true),
            #[cfg(all(feature = "webm", not(any(feature = "mp4", feature = "mpeg-ts"))))]
            mut state @ DemuxerState::WebM(_) => process_state(&mut state, &[], true),
            #[cfg(all(feature = "mpeg-ts", not(any(feature = "mp4", feature = "webm"))))]
            mut state @ DemuxerState::MpegTs(_) => process_state(&mut state, &[], true),
        }
    }
}

impl Default for AudioTrackDemuxer {
    fn default() -> Self {
        Self::new_auto()
    }
}

fn demuxer_for_format(format: &str) -> Result<DemuxerState, String> {
    match normalize_format(format).as_str() {
        #[cfg(feature = "mp4")]
        "mp4" | "m4a" | "m4v" | "mov" | "quicktime" | "aac-mp4" | "mp4-aac" => {
            Ok(DemuxerState::Mp4(Mp4AudioDemuxer::regular()?))
        }
        #[cfg(feature = "mp4")]
        "fmp4" | "fragmented-mp4" | "cmaf" | "cmf" => {
            Ok(DemuxerState::Mp4(Mp4AudioDemuxer::fragmented()))
        }
        #[cfg(feature = "webm")]
        "webm" | "matroska" | "mkv" => {
            let mut demuxer = WebmAudioDemuxer::new();
            demuxer.init()?;
            Ok(DemuxerState::WebM(demuxer))
        }
        #[cfg(feature = "mpeg-ts")]
        "ts" | "mpeg-ts" | "mpegts" | "hls-ts" => {
            Ok(DemuxerState::MpegTs(MpegTsAudioDemuxer::new()))
        }
        other => Err(format!("unsupported audio demux format: {other}")),
    }
}

fn detect_and_init_demuxer(bytes: &[u8]) -> Result<DemuxerState, String> {
    #[cfg(feature = "mpeg-ts")]
    if looks_like_mpeg_ts(bytes) {
        return demuxer_for_format("mpeg-ts");
    }

    #[cfg(feature = "webm")]
    if bytes.starts_with(&[0x1a, 0x45, 0xdf, 0xa3]) {
        return demuxer_for_format("webm");
    }

    #[cfg(feature = "mp4")]
    if looks_like_mp4(bytes) {
        if contains_top_level_box(bytes, b"moof") {
            return demuxer_for_format("fmp4");
        }
        return demuxer_for_format("mp4");
    }

    Err("could not detect supported audio/video container".to_string())
}

fn process_state(
    state: &mut DemuxerState,
    bytes: &[u8],
    finalizing: bool,
) -> Result<Vec<AudioDemuxEvent>, String> {
    match state {
        #[cfg(feature = "mp4")]
        DemuxerState::Mp4(demuxer) => {
            if finalizing {
                demuxer.finish(bytes)
            } else {
                demuxer.add(bytes)
            }
        }
        #[cfg(feature = "webm")]
        DemuxerState::WebM(demuxer) => {
            let _ = finalizing;
            convert_webm_events(demuxer.add(bytes)?)
        }
        #[cfg(feature = "mpeg-ts")]
        DemuxerState::MpegTs(demuxer) => {
            if finalizing {
                demuxer.finish(bytes)
            } else {
                demuxer.add(bytes)
            }
        }
        DemuxerState::Detecting { .. } => Ok(Vec::new()),
        DemuxerState::Finished => Err("demuxer is already finished".to_string()),
    }
}

#[cfg(feature = "mp4")]
enum Mp4AudioDemuxer {
    Regular(RegularMp4AacDemuxer),
    Fragmented(Fmp4AacDemuxer),
}

#[cfg(feature = "mp4")]
impl Mp4AudioDemuxer {
    fn regular() -> Result<Self, String> {
        Ok(Self::Regular(RegularMp4AacDemuxer::new()))
    }

    fn fragmented() -> Self {
        Self::Fragmented(Fmp4AacDemuxer::new())
    }

    fn add(&mut self, bytes: &[u8]) -> Result<Vec<AudioDemuxEvent>, String> {
        match self {
            Self::Regular(demuxer) => demuxer.add(bytes),
            Self::Fragmented(demuxer) => demuxer.add(bytes),
        }
    }

    fn finish(&mut self, bytes: &[u8]) -> Result<Vec<AudioDemuxEvent>, String> {
        match self {
            Self::Regular(demuxer) => demuxer.finish(bytes),
            Self::Fragmented(demuxer) => demuxer.finish(bytes),
        }
    }
}

#[cfg(feature = "mp4")]
#[derive(Clone, Debug)]
struct RegularMp4AacTrack {
    track_id: u32,
    sample_rate: u32,
    channels: u8,
    codec_private: Vec<u8>,
    samples: Vec<RegularMp4Sample>,
}

#[cfg(feature = "mp4")]
#[derive(Clone, Debug)]
struct RegularMp4Sample {
    sample_id: u32,
    absolute_offset: u64,
    size: u32,
    duration: u32,
    start_time: u64,
    rendering_offset: i32,
    is_sync: bool,
}

#[cfg(feature = "mp4")]
#[derive(Clone, Debug)]
struct RegularMdatRange {
    payload_start: u64,
    payload_end: u64,
}

#[cfg(feature = "mp4")]
#[derive(Clone, Debug)]
struct DeferredMdat {
    payload_start: u64,
    payload_end: u64,
    data: Vec<u8>,
}

#[cfg(feature = "mp4")]
struct RegularMp4AacDemuxer {
    buffer: Vec<u8>,
    absolute_start: u64,
    track: Option<RegularMp4AacTrack>,
    active_mdat: Option<RegularMdatRange>,
    deferred_mdats: Vec<DeferredMdat>,
    emitted_config: bool,
    next_sample_index: usize,
}

#[cfg(feature = "mp4")]
impl RegularMp4AacDemuxer {
    fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(128 * 1024),
            absolute_start: 0,
            track: None,
            active_mdat: None,
            deferred_mdats: Vec::new(),
            emitted_config: false,
            next_sample_index: 0,
        }
    }

    fn add(&mut self, bytes: &[u8]) -> Result<Vec<AudioDemuxEvent>, String> {
        self.buffer.extend_from_slice(bytes);
        self.parse_available(false)
    }

    fn finish(&mut self, bytes: &[u8]) -> Result<Vec<AudioDemuxEvent>, String> {
        self.buffer.extend_from_slice(bytes);
        self.parse_available(true)
    }

    fn parse_available(&mut self, finalizing: bool) -> Result<Vec<AudioDemuxEvent>, String> {
        let mut events = Vec::new();

        loop {
            if self.active_mdat.is_some() {
                let before = self.absolute_start;
                events.extend(self.emit_active_mdat_samples()?);
                if self.active_mdat.is_some() && self.absolute_start == before {
                    break;
                }
                continue;
            }

            let Some(header) = Mp4BoxHeader::read(&self.buffer) else {
                if finalizing && !self.buffer.is_empty() {
                    return Err("truncated MP4 box header".to_string());
                }
                break;
            };

            if header.name == *b"mdat" {
                let box_start = self.absolute_start;
                let payload_start = box_start + header.header_size as u64;
                let payload_end = box_start + header.size as u64;

                if self.track.is_some() {
                    self.drain_front(header.header_size);
                    self.active_mdat = Some(RegularMdatRange {
                        payload_start,
                        payload_end,
                    });
                    continue;
                }

                if header.size as u64 > REGULAR_MP4_DEFERRED_MDAT_MAX_BYTES {
                    return Err(format!(
                        "MP4 mdat appears before moov and is too large to buffer ({} bytes)",
                        header.size
                    ));
                }
                if self.buffer.len() < header.size {
                    if finalizing {
                        return Err("truncated MP4 mdat before moov".to_string());
                    }
                    break;
                }

                self.deferred_mdats.push(DeferredMdat {
                    payload_start,
                    payload_end,
                    data: self.buffer[header.header_size..header.size].to_vec(),
                });
                self.drain_front(header.size);
                continue;
            }

            if self.buffer.len() < header.size {
                if finalizing {
                    return Err(format!(
                        "truncated MP4 box {}",
                        String::from_utf8_lossy(&header.name)
                    ));
                }
                break;
            }

            let payload = self.buffer[header.header_size..header.size].to_vec();
            match &header.name {
                b"moov" => {
                    if let Some(track) = parse_regular_moov(&payload)? {
                        self.track = Some(track);
                        self.emit_config_if_needed(&mut events);
                        events.extend(self.emit_deferred_mdat_samples()?);
                    }
                }
                _ => {}
            }
            self.drain_front(header.size);
        }

        Ok(events)
    }

    fn drain_front(&mut self, bytes: usize) {
        self.buffer.drain(..bytes);
        self.absolute_start += bytes as u64;
    }

    fn emit_config_if_needed(&mut self, events: &mut Vec<AudioDemuxEvent>) {
        if self.emitted_config {
            return;
        }
        let Some(track) = self.track.as_ref() else {
            return;
        };
        events.push(AudioDemuxEvent::Config(AudioTrackConfig {
            container: AudioContainer::Mp4,
            codec: AudioCodec::Aac,
            packet_format: Some(AudioPacketFormat::Adts),
            codec_id: Some("mp4a".to_string()),
            track_id: Some(track.track_id as u64),
            pid: None,
            stream_type: None,
            sample_rate: Some(track.sample_rate),
            channels: Some(track.channels),
            sample_count: Some(track.samples.len() as u32),
            codec_private: track.codec_private.clone(),
            pre_skip: None,
            output_gain: None,
            mapping_family: None,
        }));
        self.emitted_config = true;
    }

    fn emit_active_mdat_samples(&mut self) -> Result<Vec<AudioDemuxEvent>, String> {
        let mut events = Vec::new();
        let Some(mdat) = self.active_mdat.clone() else {
            return Ok(events);
        };
        self.emit_config_if_needed(&mut events);

        let available_end = (self.absolute_start + self.buffer.len() as u64).min(mdat.payload_end);
        if available_end <= self.absolute_start {
            return Ok(events);
        }

        loop {
            let Some(sample) = self.next_sample_in_mdat(&mdat).cloned() else {
                break;
            };
            let sample_end = sample.absolute_offset + sample.size as u64;
            if sample_end > available_end {
                break;
            }
            if sample.absolute_offset < self.absolute_start {
                return Err("MP4 sample offset was already discarded".to_string());
            }

            let start = (sample.absolute_offset - self.absolute_start) as usize;
            let end = start + sample.size as usize;
            if end > self.buffer.len() {
                break;
            }
            let raw = self.buffer[start..end].to_vec();
            self.next_sample_index += 1;
            events.push(self.packet_event(&sample, raw));
        }

        let drain_until = self.safe_mdat_drain_until(&mdat, available_end);
        if drain_until > self.absolute_start {
            self.drain_front((drain_until - self.absolute_start) as usize);
        }
        if self.absolute_start >= mdat.payload_end {
            self.active_mdat = None;
        }

        Ok(events)
    }

    fn emit_deferred_mdat_samples(&mut self) -> Result<Vec<AudioDemuxEvent>, String> {
        let mut events = Vec::new();
        if self.deferred_mdats.is_empty() {
            return Ok(events);
        }
        self.emit_config_if_needed(&mut events);

        let mdats = std::mem::take(&mut self.deferred_mdats);
        for mdat in mdats {
            while let Some(sample) = self
                .next_sample_in_range(mdat.payload_start, mdat.payload_end)
                .cloned()
            {
                let sample_end = sample.absolute_offset + sample.size as u64;
                if sample_end > mdat.payload_end {
                    break;
                }
                let start = (sample.absolute_offset - mdat.payload_start) as usize;
                let end = start + sample.size as usize;
                if end > mdat.data.len() {
                    break;
                }
                let raw = mdat.data[start..end].to_vec();
                self.next_sample_index += 1;
                events.push(self.packet_event(&sample, raw));
            }
        }

        Ok(events)
    }

    fn next_sample_in_mdat(&self, mdat: &RegularMdatRange) -> Option<&RegularMp4Sample> {
        self.next_sample_in_range(mdat.payload_start, mdat.payload_end)
    }

    fn next_sample_in_range(&self, start: u64, end: u64) -> Option<&RegularMp4Sample> {
        let track = self.track.as_ref()?;
        let mut index = self.next_sample_index;
        while let Some(sample) = track.samples.get(index) {
            let sample_end = sample.absolute_offset + sample.size as u64;
            if sample_end <= start {
                index += 1;
                continue;
            }
            if sample.absolute_offset >= end {
                return None;
            }
            return Some(sample);
        }
        None
    }

    fn safe_mdat_drain_until(&self, mdat: &RegularMdatRange, available_end: u64) -> u64 {
        let Some(next_sample) = self.next_sample_in_mdat(mdat) else {
            return available_end;
        };
        next_sample
            .absolute_offset
            .min(available_end)
            .max(self.absolute_start)
    }

    fn packet_event(&self, sample: &RegularMp4Sample, raw: Vec<u8>) -> AudioDemuxEvent {
        let track = self
            .track
            .as_ref()
            .expect("track is set before emitting samples");
        let mut data = create_adts_header(
            track.sample_rate,
            track.channels,
            raw.len(),
            &track.codec_private,
        );
        data.extend_from_slice(&raw);
        AudioDemuxEvent::Packet(AudioTrackPacket {
            container: AudioContainer::Mp4,
            codec: AudioCodec::Aac,
            format: AudioPacketFormat::Adts,
            data,
            raw_data: Some(raw),
            track_id: Some(track.track_id as u64),
            pid: None,
            stream_type: None,
            sample_id: Some(sample.sample_id),
            start_time: Some(sample.start_time),
            duration: Some(sample.duration),
            rendering_offset: Some(sample.rendering_offset),
            is_sync: Some(sample.is_sync),
            timecode: Some(sample.start_time as i64),
        })
    }
}

#[cfg(feature = "webm")]
fn convert_webm_events(events: Vec<WebmAudioDemuxEvent>) -> Result<Vec<AudioDemuxEvent>, String> {
    let mut output = Vec::new();
    for event in events {
        match event {
            WebmAudioDemuxEvent::Config(config) => {
                let codec = webm_codec(&config.codec_id);
                output.push(AudioDemuxEvent::Config(AudioTrackConfig {
                    container: AudioContainer::WebM,
                    codec,
                    packet_format: Some(AudioPacketFormat::Raw),
                    codec_id: Some(config.codec_id),
                    track_id: Some(config.track_number),
                    pid: None,
                    stream_type: None,
                    sample_rate: Some(config.sample_rate),
                    channels: Some(config.channels),
                    sample_count: None,
                    codec_private: config.codec_private,
                    pre_skip: config.pre_skip,
                    output_gain: config.output_gain,
                    mapping_family: config.mapping_family,
                }));
            }
            WebmAudioDemuxEvent::Packet {
                track_number,
                codec_id,
                data,
                timecode,
            } => {
                let codec = webm_codec(&codec_id);
                output.push(AudioDemuxEvent::Packet(AudioTrackPacket {
                    container: AudioContainer::WebM,
                    codec,
                    format: AudioPacketFormat::Raw,
                    data,
                    raw_data: None,
                    track_id: Some(track_number),
                    pid: None,
                    stream_type: None,
                    sample_id: None,
                    start_time: None,
                    duration: None,
                    rendering_offset: None,
                    is_sync: None,
                    timecode: Some(timecode as i64),
                }));
            }
        }
    }
    Ok(output)
}

#[cfg(feature = "webm")]
fn webm_codec(codec_id: &str) -> AudioCodec {
    match codec_id {
        "A_OPUS" => AudioCodec::Opus,
        "A_VORBIS" => AudioCodec::Vorbis,
        "A_AAC" | "A_AAC/MPEG2/LC" | "A_AAC/MPEG4/LC" => AudioCodec::Aac,
        "A_MPEG/L3" => AudioCodec::Mp3,
        "A_AC3" => AudioCodec::Ac3,
        other => AudioCodec::Unknown(other.to_ascii_lowercase()),
    }
}

fn normalize_format(format: &str) -> String {
    format.trim().to_ascii_lowercase().replace('_', "-")
}

#[cfg(feature = "mp4")]
fn looks_like_mp4(bytes: &[u8]) -> bool {
    bytes.len() >= 12 && &bytes[4..8] == b"ftyp"
}

#[cfg(feature = "mp4")]
fn contains_top_level_box(bytes: &[u8], name: &[u8; 4]) -> bool {
    let mut pos = 0usize;
    while pos + 8 <= bytes.len() {
        let Some(header) = Mp4BoxHeader::read(&bytes[pos..]) else {
            return false;
        };
        if header.name == *name {
            return true;
        }
        if header.size == 0 || pos + header.size > bytes.len() {
            return false;
        }
        pos += header.size;
    }
    false
}

#[cfg(feature = "mp4")]
#[derive(Clone, Copy, Debug)]
struct Mp4BoxHeader {
    name: [u8; 4],
    size: usize,
    header_size: usize,
}

#[cfg(feature = "mp4")]
impl Mp4BoxHeader {
    fn read(data: &[u8]) -> Option<Self> {
        if data.len() < 8 {
            return None;
        }

        let short_size = u32::from_be_bytes([data[0], data[1], data[2], data[3]]) as u64;
        let name = [data[4], data[5], data[6], data[7]];
        let (size, header_size) = if short_size == 1 {
            if data.len() < 16 {
                return None;
            }
            (
                u64::from_be_bytes([
                    data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15],
                ]),
                16usize,
            )
        } else if short_size == 0 {
            return None;
        } else {
            (short_size, 8usize)
        };

        if size < header_size as u64 || size > usize::MAX as u64 {
            return None;
        }

        Some(Self {
            name,
            size: size as usize,
            header_size,
        })
    }
}

#[cfg(feature = "mp4")]
#[derive(Clone, Debug)]
struct Fmp4AacTrack {
    track_id: u32,
    sample_rate: u32,
    channels: u8,
    codec_private: Vec<u8>,
    default_sample_duration: Option<u32>,
    default_sample_size: Option<u32>,
}

#[cfg(feature = "mp4")]
#[derive(Clone, Debug, Default)]
struct Fmp4TrackDefaults {
    track_id: u32,
    default_sample_duration: Option<u32>,
    default_sample_size: Option<u32>,
}

#[cfg(feature = "mp4")]
#[derive(Clone, Debug)]
struct Fmp4Sample {
    absolute_offset: u64,
    size: u32,
    duration: u32,
    start_time: u64,
    rendering_offset: i32,
    is_sync: bool,
}

#[cfg(feature = "mp4")]
#[derive(Clone, Debug)]
struct Fmp4Fragment {
    track_id: u32,
    samples: Vec<Fmp4Sample>,
}

#[cfg(feature = "mp4")]
struct Fmp4AacDemuxer {
    buffer: Vec<u8>,
    absolute_start: u64,
    track: Option<Fmp4AacTrack>,
    track_defaults: Vec<Fmp4TrackDefaults>,
    pending_fragments: Vec<Fmp4Fragment>,
    emitted_config: bool,
    next_sample_id: u32,
}

#[cfg(feature = "mp4")]
impl Fmp4AacDemuxer {
    fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(128 * 1024),
            absolute_start: 0,
            track: None,
            track_defaults: Vec::new(),
            pending_fragments: Vec::new(),
            emitted_config: false,
            next_sample_id: 1,
        }
    }

    fn add(&mut self, bytes: &[u8]) -> Result<Vec<AudioDemuxEvent>, String> {
        self.buffer.extend_from_slice(bytes);
        self.parse_available(false)
    }

    fn finish(&mut self, bytes: &[u8]) -> Result<Vec<AudioDemuxEvent>, String> {
        self.buffer.extend_from_slice(bytes);
        self.parse_available(true)
    }

    fn parse_available(&mut self, finalizing: bool) -> Result<Vec<AudioDemuxEvent>, String> {
        let mut events = Vec::new();

        loop {
            let Some(header) = Mp4BoxHeader::read(&self.buffer) else {
                break;
            };

            if self.buffer.len() < header.size {
                if finalizing {
                    return Err(format!(
                        "truncated fMP4 box {}",
                        String::from_utf8_lossy(&header.name)
                    ));
                }
                break;
            }

            let box_start = self.absolute_start;
            let payload_start = header.header_size;
            let payload_end = header.size;
            let payload = self.buffer[payload_start..payload_end].to_vec();

            match &header.name {
                b"moov" => {
                    self.parse_moov(&payload)?;
                    self.drain_front(header.size);
                }
                b"moof" => {
                    let fragments = self.parse_moof(&payload, box_start)?;
                    self.pending_fragments.extend(fragments);
                    self.drain_front(header.size);
                }
                b"mdat" => {
                    let mdat_payload_start = box_start + header.header_size as u64;
                    let mdat_payload_end = box_start + header.size as u64;
                    events.extend(self.emit_mdat_samples(mdat_payload_start, mdat_payload_end)?);
                    self.drain_front(header.size);
                }
                _ => {
                    self.drain_front(header.size);
                }
            }
        }

        Ok(events)
    }

    fn drain_front(&mut self, bytes: usize) {
        self.buffer.drain(..bytes);
        self.absolute_start += bytes as u64;
    }

    fn parse_moov(&mut self, data: &[u8]) -> Result<(), String> {
        self.track_defaults = parse_trex_defaults(data);
        let mut selected = None;

        for_each_child_box(data, |header, payload, _| {
            if header.name == *b"trak" {
                if let Some(track) = parse_trak(payload, &self.track_defaults) {
                    selected = Some(track);
                }
            }
            Ok(())
        })?;

        if let Some(track) = selected {
            self.track = Some(track);
        }

        Ok(())
    }

    fn parse_moof(
        &self,
        data: &[u8],
        moof_absolute_start: u64,
    ) -> Result<Vec<Fmp4Fragment>, String> {
        let mut fragments = Vec::new();

        for_each_child_box(data, |header, payload, _| {
            if header.name == *b"traf" {
                if let Some(fragment) = self.parse_traf(payload, moof_absolute_start)? {
                    fragments.push(fragment);
                }
            }
            Ok(())
        })?;

        Ok(fragments)
    }

    fn parse_traf(
        &self,
        data: &[u8],
        moof_absolute_start: u64,
    ) -> Result<Option<Fmp4Fragment>, String> {
        let mut tfhd = None;
        let mut base_decode_time = 0u64;
        let mut truns = Vec::new();

        for_each_child_box(data, |header, payload, _| {
            match &header.name {
                b"tfhd" => tfhd = parse_tfhd(payload),
                b"tfdt" => base_decode_time = parse_tfdt(payload).unwrap_or(0),
                b"trun" => {
                    if let Some(trun) = parse_trun(payload) {
                        truns.push(trun);
                    }
                }
                _ => {}
            }
            Ok(())
        })?;

        let Some(tfhd) = tfhd else {
            return Ok(None);
        };

        let Some(track) = self
            .track
            .as_ref()
            .filter(|track| track.track_id == tfhd.track_id)
        else {
            return Ok(None);
        };

        let track_defaults = self
            .track_defaults
            .iter()
            .find(|defaults| defaults.track_id == tfhd.track_id)
            .cloned()
            .unwrap_or_default();

        let default_duration = tfhd
            .default_sample_duration
            .or(track.default_sample_duration)
            .or(track_defaults.default_sample_duration)
            .unwrap_or(1024);
        let default_size = tfhd
            .default_sample_size
            .or(track.default_sample_size)
            .or(track_defaults.default_sample_size);
        let base_data_offset = tfhd.base_data_offset.unwrap_or(moof_absolute_start);

        let mut samples = Vec::new();
        let mut decode_time = base_decode_time;
        let mut fallback_data_offset = base_data_offset;

        for trun in truns {
            let mut sample_offset = if let Some(data_offset) = trun.data_offset {
                add_signed_offset(base_data_offset, data_offset)?
            } else {
                fallback_data_offset
            };

            for index in 0..trun.sample_count as usize {
                let size = trun
                    .sample_sizes
                    .get(index)
                    .copied()
                    .or(default_size)
                    .ok_or_else(|| "fMP4 trun sample has no size".to_string())?;
                let duration = trun
                    .sample_durations
                    .get(index)
                    .copied()
                    .unwrap_or(default_duration);
                let rendering_offset = trun.sample_cts.get(index).copied().unwrap_or(0);
                let is_sync = trun
                    .sample_flags
                    .get(index)
                    .copied()
                    .or(trun.first_sample_flags.filter(|_| index == 0))
                    .map(|flags| flags & 0x0001_0000 == 0)
                    .unwrap_or(true);

                samples.push(Fmp4Sample {
                    absolute_offset: sample_offset,
                    size,
                    duration,
                    start_time: decode_time,
                    rendering_offset,
                    is_sync,
                });

                sample_offset += size as u64;
                decode_time += duration as u64;
            }

            fallback_data_offset = sample_offset;
        }

        Ok(Some(Fmp4Fragment {
            track_id: tfhd.track_id,
            samples,
        }))
    }

    fn emit_mdat_samples(
        &mut self,
        mdat_payload_start: u64,
        mdat_payload_end: u64,
    ) -> Result<Vec<AudioDemuxEvent>, String> {
        let mut events = Vec::new();
        let Some(track) = self.track.clone() else {
            self.pending_fragments.clear();
            return Ok(events);
        };

        if !self.emitted_config {
            events.push(AudioDemuxEvent::Config(AudioTrackConfig {
                container: AudioContainer::Mp4,
                codec: AudioCodec::Aac,
                packet_format: Some(AudioPacketFormat::Adts),
                codec_id: Some("mp4a".to_string()),
                track_id: Some(track.track_id as u64),
                pid: None,
                stream_type: None,
                sample_rate: Some(track.sample_rate),
                channels: Some(track.channels),
                sample_count: None,
                codec_private: track.codec_private.clone(),
                pre_skip: None,
                output_gain: None,
                mapping_family: None,
            }));
            self.emitted_config = true;
        }

        let fragments = std::mem::take(&mut self.pending_fragments);
        for fragment in fragments {
            if fragment.track_id != track.track_id {
                continue;
            }

            for sample in fragment.samples {
                let sample_end = sample.absolute_offset + sample.size as u64;
                if sample.absolute_offset < mdat_payload_start || sample_end > mdat_payload_end {
                    continue;
                }

                let start = (sample.absolute_offset - self.absolute_start) as usize;
                let end = start + sample.size as usize;
                if end > self.buffer.len() {
                    continue;
                }

                let raw = self.buffer[start..end].to_vec();
                let mut data = create_adts_header(
                    track.sample_rate,
                    track.channels,
                    raw.len(),
                    &track.codec_private,
                );
                data.extend_from_slice(&raw);

                let sample_id = self.next_sample_id;
                self.next_sample_id += 1;
                events.push(AudioDemuxEvent::Packet(AudioTrackPacket {
                    container: AudioContainer::Mp4,
                    codec: AudioCodec::Aac,
                    format: AudioPacketFormat::Adts,
                    data,
                    raw_data: Some(raw),
                    track_id: Some(track.track_id as u64),
                    pid: None,
                    stream_type: None,
                    sample_id: Some(sample_id),
                    start_time: Some(sample.start_time),
                    duration: Some(sample.duration),
                    rendering_offset: Some(sample.rendering_offset),
                    is_sync: Some(sample.is_sync),
                    timecode: Some(sample.start_time as i64),
                }));
            }
        }

        Ok(events)
    }
}

#[cfg(feature = "mp4")]
#[derive(Clone, Debug, Default)]
struct ParsedTfhd {
    track_id: u32,
    base_data_offset: Option<u64>,
    default_sample_duration: Option<u32>,
    default_sample_size: Option<u32>,
}

#[cfg(feature = "mp4")]
#[derive(Clone, Debug, Default)]
struct ParsedTrun {
    sample_count: u32,
    data_offset: Option<i32>,
    first_sample_flags: Option<u32>,
    sample_durations: Vec<u32>,
    sample_sizes: Vec<u32>,
    sample_flags: Vec<u32>,
    sample_cts: Vec<i32>,
}

#[cfg(feature = "mp4")]
fn for_each_child_box<F>(data: &[u8], mut f: F) -> Result<(), String>
where
    F: FnMut(Mp4BoxHeader, &[u8], usize) -> Result<(), String>,
{
    let mut pos = 0usize;
    while pos + 8 <= data.len() {
        let Some(header) = Mp4BoxHeader::read(&data[pos..]) else {
            break;
        };
        if header.size == 0 || pos + header.size > data.len() {
            break;
        }
        let payload_start = pos + header.header_size;
        let payload_end = pos + header.size;
        f(header, &data[payload_start..payload_end], pos)?;
        pos += header.size;
    }
    Ok(())
}

#[cfg(feature = "mp4")]
#[derive(Clone, Debug, Default)]
struct RegularTrakTables {
    track_id: Option<u32>,
    is_audio: bool,
    timescale: Option<u32>,
    sample_entry: Option<Mp4aSampleEntry>,
    stts: Vec<SttsEntry>,
    ctts: Vec<CttsEntry>,
    stsc: Vec<StscEntry>,
    sample_sizes: Vec<u32>,
    chunk_offsets: Vec<u64>,
    sync_samples: Vec<u32>,
}

#[cfg(feature = "mp4")]
#[derive(Clone, Debug)]
struct SttsEntry {
    sample_count: u32,
    sample_duration: u32,
}

#[cfg(feature = "mp4")]
#[derive(Clone, Debug)]
struct CttsEntry {
    sample_count: u32,
    sample_offset: i32,
}

#[cfg(feature = "mp4")]
#[derive(Clone, Debug)]
struct StscEntry {
    first_chunk: u32,
    samples_per_chunk: u32,
}

#[cfg(feature = "mp4")]
fn parse_regular_moov(data: &[u8]) -> Result<Option<RegularMp4AacTrack>, String> {
    let mut selected = None;

    for_each_child_box(data, |header, payload, _| {
        if header.name == *b"trak" {
            if let Some(track) = parse_regular_trak(payload)? {
                selected = Some(track);
            }
        }
        Ok(())
    })?;

    Ok(selected)
}

#[cfg(feature = "mp4")]
fn parse_regular_trak(data: &[u8]) -> Result<Option<RegularMp4AacTrack>, String> {
    let mut tables = RegularTrakTables::default();

    walk_boxes(data, &mut |header, payload| {
        match &header.name {
            b"tkhd" => tables.track_id = parse_tkhd_track_id(payload),
            b"mdhd" => tables.timescale = parse_mdhd_timescale(payload),
            b"hdlr" => tables.is_audio |= parse_hdlr_is_audio(payload),
            b"stsd" => tables.sample_entry = parse_stsd_mp4a(payload),
            b"stts" => tables.stts = parse_stts(payload),
            b"ctts" => tables.ctts = parse_ctts(payload),
            b"stsc" => tables.stsc = parse_stsc(payload),
            b"stsz" => tables.sample_sizes = parse_stsz(payload),
            b"stco" => tables.chunk_offsets = parse_stco(payload),
            b"co64" => tables.chunk_offsets = parse_co64(payload),
            b"stss" => tables.sync_samples = parse_stss(payload),
            _ => {}
        }
        Ok(())
    })?;

    if !tables.is_audio {
        return Ok(None);
    }

    let track_id = tables
        .track_id
        .ok_or_else(|| "MP4 audio track is missing tkhd track id".to_string())?;
    let samples = build_regular_samples(&tables)?;
    if samples.is_empty() {
        return Err("MP4 audio track has no samples".to_string());
    }

    let mut sample_entry = tables
        .sample_entry
        .ok_or_else(|| "MP4 audio track is missing mp4a sample entry".to_string())?;
    if let Some((sample_rate, channels)) = parse_asc_audio_config(&sample_entry.codec_private) {
        sample_entry.sample_rate = sample_rate;
        sample_entry.channels = channels;
    }

    Ok(Some(RegularMp4AacTrack {
        track_id,
        sample_rate: sample_entry.sample_rate,
        channels: sample_entry.channels,
        codec_private: sample_entry.codec_private,
        samples,
    }))
}

#[cfg(feature = "mp4")]
fn build_regular_samples(tables: &RegularTrakTables) -> Result<Vec<RegularMp4Sample>, String> {
    if tables.sample_sizes.is_empty() {
        return Err("MP4 audio track is missing stsz sample sizes".to_string());
    }
    if tables.chunk_offsets.is_empty() {
        return Err("MP4 audio track is missing stco/co64 chunk offsets".to_string());
    }
    if tables.stsc.is_empty() {
        return Err("MP4 audio track is missing stsc sample-to-chunk table".to_string());
    }

    let mut samples = Vec::with_capacity(tables.sample_sizes.len());
    let mut sample_index = 0usize;
    let mut stsc_index = 0usize;
    let mut decode_time = 0u64;
    let mut stts_reader = TimeToSampleReader::new(&tables.stts);
    let mut ctts_reader = CompositionOffsetReader::new(&tables.ctts);

    for (chunk_index, chunk_offset) in tables.chunk_offsets.iter().copied().enumerate() {
        let chunk_number = chunk_index as u32 + 1;
        while stsc_index + 1 < tables.stsc.len()
            && tables.stsc[stsc_index + 1].first_chunk <= chunk_number
        {
            stsc_index += 1;
        }
        let samples_per_chunk = tables.stsc[stsc_index].samples_per_chunk as usize;
        let mut sample_offset = chunk_offset;

        for _ in 0..samples_per_chunk {
            if sample_index >= tables.sample_sizes.len() {
                break;
            }
            let size = tables.sample_sizes[sample_index];
            let duration = stts_reader.next_duration().unwrap_or(1024);
            let rendering_offset = ctts_reader.next_offset().unwrap_or(0);
            let sample_id = sample_index as u32 + 1;
            let is_sync = tables.sync_samples.is_empty()
                || tables.sync_samples.binary_search(&sample_id).is_ok();

            samples.push(RegularMp4Sample {
                sample_id,
                absolute_offset: sample_offset,
                size,
                duration,
                start_time: decode_time,
                rendering_offset,
                is_sync,
            });

            sample_offset = sample_offset
                .checked_add(size as u64)
                .ok_or_else(|| "MP4 sample offset overflow".to_string())?;
            decode_time = decode_time
                .checked_add(duration as u64)
                .ok_or_else(|| "MP4 sample timestamp overflow".to_string())?;
            sample_index += 1;
        }
    }

    if samples.len() != tables.sample_sizes.len() {
        return Err(format!(
            "MP4 sample tables described {} samples but only {} were mapped to chunks",
            tables.sample_sizes.len(),
            samples.len()
        ));
    }

    Ok(samples)
}

#[cfg(feature = "mp4")]
struct TimeToSampleReader<'a> {
    entries: &'a [SttsEntry],
    index: usize,
    remaining: u32,
}

#[cfg(feature = "mp4")]
impl<'a> TimeToSampleReader<'a> {
    fn new(entries: &'a [SttsEntry]) -> Self {
        Self {
            entries,
            index: 0,
            remaining: 0,
        }
    }

    fn next_duration(&mut self) -> Option<u32> {
        if self.remaining == 0 {
            let entry = self.entries.get(self.index)?;
            self.index += 1;
            self.remaining = entry.sample_count;
        }
        self.remaining = self.remaining.saturating_sub(1);
        self.entries
            .get(self.index.saturating_sub(1))
            .map(|entry| entry.sample_duration)
    }
}

#[cfg(feature = "mp4")]
struct CompositionOffsetReader<'a> {
    entries: &'a [CttsEntry],
    index: usize,
    remaining: u32,
}

#[cfg(feature = "mp4")]
impl<'a> CompositionOffsetReader<'a> {
    fn new(entries: &'a [CttsEntry]) -> Self {
        Self {
            entries,
            index: 0,
            remaining: 0,
        }
    }

    fn next_offset(&mut self) -> Option<i32> {
        if self.entries.is_empty() {
            return Some(0);
        }
        if self.remaining == 0 {
            let entry = self.entries.get(self.index)?;
            self.index += 1;
            self.remaining = entry.sample_count;
        }
        self.remaining = self.remaining.saturating_sub(1);
        self.entries
            .get(self.index.saturating_sub(1))
            .map(|entry| entry.sample_offset)
    }
}

#[cfg(feature = "mp4")]
fn parse_trex_defaults(data: &[u8]) -> Vec<Fmp4TrackDefaults> {
    let mut defaults = Vec::new();
    let _ = walk_boxes(data, &mut |header, payload| {
        if header.name == *b"trex" && payload.len() >= 24 {
            defaults.push(Fmp4TrackDefaults {
                track_id: be_u32(payload, 4).unwrap_or(0),
                default_sample_duration: Some(be_u32(payload, 12).unwrap_or(0)).filter(|v| *v > 0),
                default_sample_size: Some(be_u32(payload, 16).unwrap_or(0)).filter(|v| *v > 0),
            });
        }
        Ok(())
    });
    defaults
}

#[cfg(feature = "mp4")]
fn parse_trak(data: &[u8], defaults: &[Fmp4TrackDefaults]) -> Option<Fmp4AacTrack> {
    let mut track_id = None;
    let mut is_audio = false;
    let mut sample_entry = None;

    let _ = walk_boxes(data, &mut |header, payload| {
        match &header.name {
            b"tkhd" => track_id = parse_tkhd_track_id(payload),
            b"hdlr" => is_audio |= parse_hdlr_is_audio(payload),
            b"stsd" => sample_entry = parse_stsd_mp4a(payload),
            _ => {}
        }
        Ok(())
    });

    if !is_audio {
        return None;
    }
    let track_id = track_id?;
    let mut sample_entry = sample_entry?;
    let defaults = defaults
        .iter()
        .find(|defaults| defaults.track_id == track_id)
        .cloned()
        .unwrap_or_default();

    if let Some((sample_rate, channels)) = parse_asc_audio_config(&sample_entry.codec_private) {
        sample_entry.sample_rate = sample_rate;
        sample_entry.channels = channels;
    }

    Some(Fmp4AacTrack {
        track_id,
        sample_rate: sample_entry.sample_rate,
        channels: sample_entry.channels,
        codec_private: sample_entry.codec_private,
        default_sample_duration: defaults.default_sample_duration,
        default_sample_size: defaults.default_sample_size,
    })
}

#[cfg(feature = "mp4")]
#[derive(Clone, Debug)]
struct Mp4aSampleEntry {
    sample_rate: u32,
    channels: u8,
    codec_private: Vec<u8>,
}

#[cfg(feature = "mp4")]
fn parse_tkhd_track_id(data: &[u8]) -> Option<u32> {
    let version = *data.first()?;
    let offset = if version == 1 { 20 } else { 12 };
    be_u32(data, offset)
}

#[cfg(feature = "mp4")]
fn parse_mdhd_timescale(data: &[u8]) -> Option<u32> {
    let version = *data.first()?;
    let offset = if version == 1 { 20 } else { 12 };
    be_u32(data, offset)
}

#[cfg(feature = "mp4")]
fn parse_hdlr_is_audio(data: &[u8]) -> bool {
    data.len() >= 12 && &data[8..12] == b"soun"
}

#[cfg(feature = "mp4")]
fn parse_stsd_mp4a(data: &[u8]) -> Option<Mp4aSampleEntry> {
    if data.len() < 16 {
        return None;
    }

    let entry_count = be_u32(data, 4)?;
    let mut pos = 8usize;
    for _ in 0..entry_count {
        let header = Mp4BoxHeader::read(&data[pos..])?;
        if pos + header.size > data.len() {
            return None;
        }
        let payload = &data[pos + header.header_size..pos + header.size];
        if header.name == *b"mp4a" && payload.len() >= 28 {
            let channels = be_u16(payload, 16).unwrap_or(2) as u8;
            let sample_rate = be_u32(payload, 24).unwrap_or(44_100 << 16) >> 16;
            let mut codec_private = Vec::new();
            let _ = for_each_child_box(&payload[28..], |child, child_payload, _| {
                if child.name == *b"esds" {
                    codec_private =
                        parse_esds_audio_specific_config(child_payload).unwrap_or_default();
                }
                Ok(())
            });
            return Some(Mp4aSampleEntry {
                sample_rate,
                channels,
                codec_private,
            });
        }
        pos += header.size;
    }

    None
}

#[cfg(feature = "mp4")]
fn parse_stts(data: &[u8]) -> Vec<SttsEntry> {
    let Some(entry_count) = be_u32(data, 4) else {
        return Vec::new();
    };
    let mut entries = Vec::with_capacity(entry_count as usize);
    let mut pos = 8usize;
    for _ in 0..entry_count {
        let Some(sample_count) = be_u32(data, pos) else {
            break;
        };
        let Some(sample_duration) = be_u32(data, pos + 4) else {
            break;
        };
        entries.push(SttsEntry {
            sample_count,
            sample_duration,
        });
        pos += 8;
    }
    entries
}

#[cfg(feature = "mp4")]
fn parse_ctts(data: &[u8]) -> Vec<CttsEntry> {
    let Some(version) = data.first().copied() else {
        return Vec::new();
    };
    let Some(entry_count) = be_u32(data, 4) else {
        return Vec::new();
    };
    let mut entries = Vec::with_capacity(entry_count as usize);
    let mut pos = 8usize;
    for _ in 0..entry_count {
        let Some(sample_count) = be_u32(data, pos) else {
            break;
        };
        let Some(raw_offset) = be_u32(data, pos + 4) else {
            break;
        };
        let sample_offset = if version == 1 {
            raw_offset as i32
        } else {
            raw_offset.min(i32::MAX as u32) as i32
        };
        entries.push(CttsEntry {
            sample_count,
            sample_offset,
        });
        pos += 8;
    }
    entries
}

#[cfg(feature = "mp4")]
fn parse_stsc(data: &[u8]) -> Vec<StscEntry> {
    let Some(entry_count) = be_u32(data, 4) else {
        return Vec::new();
    };
    let mut entries = Vec::with_capacity(entry_count as usize);
    let mut pos = 8usize;
    for _ in 0..entry_count {
        let Some(first_chunk) = be_u32(data, pos) else {
            break;
        };
        let Some(samples_per_chunk) = be_u32(data, pos + 4) else {
            break;
        };
        entries.push(StscEntry {
            first_chunk,
            samples_per_chunk,
        });
        pos += 12;
    }
    entries
}

#[cfg(feature = "mp4")]
fn parse_stsz(data: &[u8]) -> Vec<u32> {
    let Some(sample_size) = be_u32(data, 4) else {
        return Vec::new();
    };
    let Some(sample_count) = be_u32(data, 8) else {
        return Vec::new();
    };
    if sample_size > 0 {
        return vec![sample_size; sample_count as usize];
    }
    let mut sizes = Vec::with_capacity(sample_count as usize);
    let mut pos = 12usize;
    for _ in 0..sample_count {
        let Some(size) = be_u32(data, pos) else {
            break;
        };
        sizes.push(size);
        pos += 4;
    }
    sizes
}

#[cfg(feature = "mp4")]
fn parse_stco(data: &[u8]) -> Vec<u64> {
    let Some(entry_count) = be_u32(data, 4) else {
        return Vec::new();
    };
    let mut offsets = Vec::with_capacity(entry_count as usize);
    let mut pos = 8usize;
    for _ in 0..entry_count {
        let Some(offset) = be_u32(data, pos) else {
            break;
        };
        offsets.push(offset as u64);
        pos += 4;
    }
    offsets
}

#[cfg(feature = "mp4")]
fn parse_co64(data: &[u8]) -> Vec<u64> {
    let Some(entry_count) = be_u32(data, 4) else {
        return Vec::new();
    };
    let mut offsets = Vec::with_capacity(entry_count as usize);
    let mut pos = 8usize;
    for _ in 0..entry_count {
        let Some(offset) = be_u64(data, pos) else {
            break;
        };
        offsets.push(offset);
        pos += 8;
    }
    offsets
}

#[cfg(feature = "mp4")]
fn parse_stss(data: &[u8]) -> Vec<u32> {
    let Some(entry_count) = be_u32(data, 4) else {
        return Vec::new();
    };
    let mut samples = Vec::with_capacity(entry_count as usize);
    let mut pos = 8usize;
    for _ in 0..entry_count {
        let Some(sample_id) = be_u32(data, pos) else {
            break;
        };
        samples.push(sample_id);
        pos += 4;
    }
    samples
}

#[cfg(feature = "mp4")]
fn parse_tfhd(data: &[u8]) -> Option<ParsedTfhd> {
    if data.len() < 8 {
        return None;
    }

    let flags = be_u24(data, 1)?;
    let track_id = be_u32(data, 4)?;
    let mut pos = 8usize;
    let base_data_offset = if flags & 0x000001 != 0 {
        let value = be_u64(data, pos)?;
        pos += 8;
        Some(value)
    } else {
        None
    };
    if flags & 0x000002 != 0 {
        pos += 4;
    }
    let default_sample_duration = if flags & 0x000008 != 0 {
        let value = be_u32(data, pos)?;
        pos += 4;
        Some(value)
    } else {
        None
    };
    let default_sample_size = if flags & 0x000010 != 0 {
        Some(be_u32(data, pos)?)
    } else {
        None
    };

    Some(ParsedTfhd {
        track_id,
        base_data_offset,
        default_sample_duration,
        default_sample_size,
    })
}

#[cfg(feature = "mp4")]
fn parse_tfdt(data: &[u8]) -> Option<u64> {
    if data.len() < 8 {
        return None;
    }
    match data[0] {
        1 => be_u64(data, 4),
        0 => be_u32(data, 4).map(|value| value as u64),
        _ => None,
    }
}

#[cfg(feature = "mp4")]
fn parse_trun(data: &[u8]) -> Option<ParsedTrun> {
    if data.len() < 8 {
        return None;
    }

    let version = data[0];
    let flags = be_u24(data, 1)?;
    let sample_count = be_u32(data, 4)?;
    let mut pos = 8usize;

    let data_offset = if flags & 0x000001 != 0 {
        let value = be_i32(data, pos)?;
        pos += 4;
        Some(value)
    } else {
        None
    };
    let first_sample_flags = if flags & 0x000004 != 0 {
        let value = be_u32(data, pos)?;
        pos += 4;
        Some(value)
    } else {
        None
    };

    let mut sample_durations = Vec::new();
    let mut sample_sizes = Vec::new();
    let mut sample_flags = Vec::new();
    let mut sample_cts = Vec::new();

    for _ in 0..sample_count {
        if flags & 0x000100 != 0 {
            sample_durations.push(be_u32(data, pos)?);
            pos += 4;
        }
        if flags & 0x000200 != 0 {
            sample_sizes.push(be_u32(data, pos)?);
            pos += 4;
        }
        if flags & 0x000400 != 0 {
            sample_flags.push(be_u32(data, pos)?);
            pos += 4;
        }
        if flags & 0x000800 != 0 {
            let value = if version == 1 {
                be_i32(data, pos)?
            } else {
                be_u32(data, pos)? as i32
            };
            sample_cts.push(value);
            pos += 4;
        }
    }

    Some(ParsedTrun {
        sample_count,
        data_offset,
        first_sample_flags,
        sample_durations,
        sample_sizes,
        sample_flags,
        sample_cts,
    })
}

#[cfg(feature = "mp4")]
fn walk_boxes<F>(data: &[u8], f: &mut F) -> Result<(), String>
where
    F: FnMut(Mp4BoxHeader, &[u8]) -> Result<(), String>,
{
    for_each_child_box(data, |header, payload, _| {
        f(header, payload)?;
        if is_mp4_container_box(&header.name) {
            walk_boxes(payload, f)?;
        }
        Ok(())
    })
}

#[cfg(feature = "mp4")]
fn is_mp4_container_box(name: &[u8; 4]) -> bool {
    matches!(
        name,
        b"moov" | b"trak" | b"mdia" | b"minf" | b"stbl" | b"mvex"
    )
}

#[cfg(feature = "mp4")]
fn parse_esds_audio_specific_config(data: &[u8]) -> Option<Vec<u8>> {
    if data.len() < 4 {
        return None;
    }
    find_mpeg4_descriptor(&data[4..], 0x05)
}

#[cfg(feature = "mp4")]
fn find_mpeg4_descriptor(data: &[u8], tag: u8) -> Option<Vec<u8>> {
    let mut pos = 0usize;
    while pos + 2 <= data.len() {
        let descriptor_tag = data[pos];
        pos += 1;
        let (len, len_len) = read_descriptor_len(&data[pos..])?;
        pos += len_len;
        if pos + len > data.len() {
            return None;
        }
        let body = &data[pos..pos + len];
        if descriptor_tag == tag {
            return Some(body.to_vec());
        }
        let nested_start = match descriptor_tag {
            0x03 if body.len() >= 3 => 3,
            0x04 if body.len() >= 13 => 13,
            _ => 0,
        };
        if nested_start > 0 {
            if let Some(found) = find_mpeg4_descriptor(&body[nested_start..], tag) {
                return Some(found);
            }
        }
        pos += len;
    }
    None
}

#[cfg(feature = "mp4")]
fn read_descriptor_len(data: &[u8]) -> Option<(usize, usize)> {
    let mut value = 0usize;
    for (index, byte) in data.iter().take(4).enumerate() {
        value = (value << 7) | (byte & 0x7f) as usize;
        if byte & 0x80 == 0 {
            return Some((value, index + 1));
        }
    }
    None
}

#[cfg(feature = "mp4")]
fn parse_asc_audio_config(data: &[u8]) -> Option<(u32, u8)> {
    if data.len() < 2 {
        return None;
    }
    let freq_index = ((data[0] & 0x07) << 1) | (data[1] >> 7);
    let sample_rate = adts_sample_rate(freq_index)?;
    let channels = (data[1] >> 3) & 0x0f;
    Some((sample_rate, channels))
}

#[cfg(feature = "mp4")]
fn create_adts_header(sample_rate: u32, channels: u8, raw_len: usize, asc: &[u8]) -> Vec<u8> {
    let profile = asc
        .first()
        .map(|first| ((*first >> 3).saturating_sub(1)).min(3))
        .unwrap_or(1);
    let sample_rate_index = sample_rate_index(sample_rate);
    let channel_config = channels.min(7);
    let frame_length = raw_len + 7;

    vec![
        0xff,
        0xf1,
        (profile << 6) | (sample_rate_index << 2) | (channel_config >> 2),
        ((channel_config & 0x03) << 6) | (((frame_length >> 11) & 0x03) as u8),
        ((frame_length >> 3) & 0xff) as u8,
        (((frame_length & 0x07) << 5) as u8) | 0x1f,
        0xfc,
    ]
}

#[cfg(feature = "mp4")]
fn sample_rate_index(sample_rate: u32) -> u8 {
    match sample_rate {
        96_000 => 0,
        88_200 => 1,
        64_000 => 2,
        48_000 => 3,
        44_100 => 4,
        32_000 => 5,
        24_000 => 6,
        22_050 => 7,
        16_000 => 8,
        12_000 => 9,
        11_025 => 10,
        8_000 => 11,
        7_350 => 12,
        _ => 15,
    }
}

#[cfg(feature = "mp4")]
fn add_signed_offset(base: u64, offset: i32) -> Result<u64, String> {
    if offset >= 0 {
        base.checked_add(offset as u64)
            .ok_or_else(|| "fMP4 data offset overflow".to_string())
    } else {
        base.checked_sub(offset.unsigned_abs() as u64)
            .ok_or_else(|| "fMP4 negative data offset underflow".to_string())
    }
}

#[cfg(feature = "mp4")]
fn be_u16(data: &[u8], pos: usize) -> Option<u16> {
    Some(u16::from_be_bytes([*data.get(pos)?, *data.get(pos + 1)?]))
}

#[cfg(feature = "mp4")]
fn be_u24(data: &[u8], pos: usize) -> Option<u32> {
    Some(
        ((*data.get(pos)? as u32) << 16)
            | ((*data.get(pos + 1)? as u32) << 8)
            | *data.get(pos + 2)? as u32,
    )
}

#[cfg(feature = "mp4")]
fn be_u32(data: &[u8], pos: usize) -> Option<u32> {
    Some(u32::from_be_bytes([
        *data.get(pos)?,
        *data.get(pos + 1)?,
        *data.get(pos + 2)?,
        *data.get(pos + 3)?,
    ]))
}

#[cfg(feature = "mp4")]
fn be_i32(data: &[u8], pos: usize) -> Option<i32> {
    be_u32(data, pos).map(|value| value as i32)
}

#[cfg(feature = "mp4")]
fn be_u64(data: &[u8], pos: usize) -> Option<u64> {
    Some(u64::from_be_bytes([
        *data.get(pos)?,
        *data.get(pos + 1)?,
        *data.get(pos + 2)?,
        *data.get(pos + 3)?,
        *data.get(pos + 4)?,
        *data.get(pos + 5)?,
        *data.get(pos + 6)?,
        *data.get(pos + 7)?,
    ]))
}

#[cfg(feature = "mpeg-ts")]
fn looks_like_mpeg_ts(bytes: &[u8]) -> bool {
    bytes.len() >= 188 * 3 && bytes[0] == 0x47 && bytes[188] == 0x47 && bytes[376] == 0x47
}

#[cfg(feature = "mpeg-ts")]
struct MpegTsAudioDemuxer {
    buffer: Vec<u8>,
    pmt_pid: Option<u16>,
    audio_pid: Option<u16>,
    audio_codec: Option<AudioCodec>,
    packet_format: Option<AudioPacketFormat>,
    stream_type: Option<u8>,
    current_pes: Vec<u8>,
    emitted_config: bool,
    sample_rate: Option<u32>,
    channels: Option<u8>,
}

#[cfg(feature = "mpeg-ts")]
impl MpegTsAudioDemuxer {
    fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(188 * 32),
            pmt_pid: None,
            audio_pid: None,
            audio_codec: None,
            packet_format: None,
            stream_type: None,
            current_pes: Vec::new(),
            emitted_config: false,
            sample_rate: None,
            channels: None,
        }
    }

    fn add(&mut self, data: &[u8]) -> Result<Vec<AudioDemuxEvent>, String> {
        self.buffer.extend_from_slice(data);
        self.parse_available_packets()
    }

    fn finish(&mut self, data: &[u8]) -> Result<Vec<AudioDemuxEvent>, String> {
        self.buffer.extend_from_slice(data);
        let mut events = self.parse_available_packets()?;
        events.extend(self.flush_current_pes()?);
        Ok(events)
    }

    fn parse_available_packets(&mut self) -> Result<Vec<AudioDemuxEvent>, String> {
        let mut events = Vec::new();

        loop {
            if self.buffer.len() < 188 {
                break;
            }

            if self.buffer[0] != 0x47 {
                if let Some(offset) = self.buffer.iter().position(|byte| *byte == 0x47) {
                    self.buffer.drain(..offset);
                } else {
                    self.buffer.clear();
                    break;
                }

                if self.buffer.len() < 188 {
                    break;
                }
            }

            let packet: Vec<u8> = self.buffer.drain(..188).collect();
            self.parse_packet(&packet, &mut events)?;
        }

        Ok(events)
    }

    fn parse_packet(
        &mut self,
        packet: &[u8],
        events: &mut Vec<AudioDemuxEvent>,
    ) -> Result<(), String> {
        if packet.len() != 188 || packet[0] != 0x47 {
            return Ok(());
        }

        let payload_unit_start = packet[1] & 0x40 != 0;
        let pid = (((packet[1] & 0x1f) as u16) << 8) | packet[2] as u16;
        let adaptation_field_control = (packet[3] & 0x30) >> 4;

        if adaptation_field_control == 0 || adaptation_field_control == 2 {
            return Ok(());
        }

        let mut payload_start = 4usize;
        if adaptation_field_control == 3 {
            let adaptation_len = *packet.get(payload_start).unwrap_or(&0) as usize;
            payload_start = payload_start.saturating_add(1 + adaptation_len);
        }

        if payload_start >= packet.len() {
            return Ok(());
        }

        let payload = &packet[payload_start..];
        if pid == 0 {
            self.parse_pat(payload, payload_unit_start)?;
        } else if Some(pid) == self.pmt_pid {
            self.parse_pmt(payload, payload_unit_start)?;
        } else if Some(pid) == self.audio_pid {
            if payload_unit_start {
                events.extend(self.flush_current_pes()?);
                self.current_pes.clear();
            }
            self.current_pes.extend_from_slice(payload);
        }

        Ok(())
    }

    fn parse_pat(&mut self, payload: &[u8], payload_unit_start: bool) -> Result<(), String> {
        let section = psi_section(payload, payload_unit_start)?;
        if section.len() < 12 || section[0] != 0x00 {
            return Ok(());
        }

        let section_length = (((section[1] & 0x0f) as usize) << 8) | section[2] as usize;
        let section_end = (3 + section_length).min(section.len());
        if section_end < 12 {
            return Ok(());
        }

        let entries_end = section_end.saturating_sub(4);
        let mut pos = 8usize;
        while pos + 4 <= entries_end {
            let program_number = u16::from_be_bytes([section[pos], section[pos + 1]]);
            let pid = (((section[pos + 2] & 0x1f) as u16) << 8) | section[pos + 3] as u16;
            if program_number != 0 {
                self.pmt_pid = Some(pid);
                break;
            }
            pos += 4;
        }

        Ok(())
    }

    fn parse_pmt(&mut self, payload: &[u8], payload_unit_start: bool) -> Result<(), String> {
        let section = psi_section(payload, payload_unit_start)?;
        if section.len() < 16 || section[0] != 0x02 {
            return Ok(());
        }

        let section_length = (((section[1] & 0x0f) as usize) << 8) | section[2] as usize;
        let section_end = (3 + section_length).min(section.len());
        if section_end < 16 {
            return Ok(());
        }

        let program_info_length = (((section[10] & 0x0f) as usize) << 8) | section[11] as usize;
        let mut pos = 12 + program_info_length;
        let entries_end = section_end.saturating_sub(4);

        while pos + 5 <= entries_end {
            let stream_type = section[pos];
            let pid = (((section[pos + 1] & 0x1f) as u16) << 8) | section[pos + 2] as u16;
            let es_info_length =
                (((section[pos + 3] & 0x0f) as usize) << 8) | section[pos + 4] as usize;

            if let Some((codec, packet_format)) = ts_stream_codec(stream_type) {
                if self.audio_pid.is_none() {
                    self.audio_pid = Some(pid);
                    self.audio_codec = Some(codec);
                    self.packet_format = Some(packet_format);
                    self.stream_type = Some(stream_type);
                }
                break;
            }

            pos += 5 + es_info_length;
        }

        Ok(())
    }

    fn flush_current_pes(&mut self) -> Result<Vec<AudioDemuxEvent>, String> {
        if self.current_pes.is_empty() {
            return Ok(Vec::new());
        }

        let pes = std::mem::take(&mut self.current_pes);
        let payload = match pes_payload(&pes) {
            Some(payload) => payload,
            None => return Ok(Vec::new()),
        };

        let codec = match self.audio_codec.clone() {
            Some(codec) => codec,
            None => return Ok(Vec::new()),
        };

        match self.packet_format.clone().unwrap_or(AudioPacketFormat::Raw) {
            AudioPacketFormat::Adts => self.emit_adts_frames(payload),
            AudioPacketFormat::Latm | AudioPacketFormat::Raw => {
                let mut events = Vec::new();
                self.ensure_config(&mut events, None, None);
                if !payload.is_empty() {
                    events.push(AudioDemuxEvent::Packet(AudioTrackPacket {
                        container: AudioContainer::MpegTs,
                        codec,
                        format: self.packet_format.clone().unwrap_or(AudioPacketFormat::Raw),
                        data: payload.to_vec(),
                        raw_data: None,
                        track_id: None,
                        pid: self.audio_pid,
                        stream_type: self.stream_type,
                        sample_id: None,
                        start_time: None,
                        duration: None,
                        rendering_offset: None,
                        is_sync: None,
                        timecode: None,
                    }));
                }
                Ok(events)
            }
        }
    }

    fn emit_adts_frames(&mut self, payload: &[u8]) -> Result<Vec<AudioDemuxEvent>, String> {
        let mut events = Vec::new();
        let mut pos = 0usize;
        while pos + 7 <= payload.len() {
            if payload[pos] != 0xff || payload[pos + 1] & 0xf0 != 0xf0 {
                pos += 1;
                continue;
            }

            let Some(header) = parse_adts_header(&payload[pos..]) else {
                break;
            };

            if pos + header.frame_length > payload.len() {
                break;
            }

            self.ensure_config(&mut events, Some(header.sample_rate), Some(header.channels));
            events.push(AudioDemuxEvent::Packet(AudioTrackPacket {
                container: AudioContainer::MpegTs,
                codec: AudioCodec::Aac,
                format: AudioPacketFormat::Adts,
                data: payload[pos..pos + header.frame_length].to_vec(),
                raw_data: None,
                track_id: None,
                pid: self.audio_pid,
                stream_type: self.stream_type,
                sample_id: None,
                start_time: None,
                duration: None,
                rendering_offset: None,
                is_sync: None,
                timecode: None,
            }));
            pos += header.frame_length;
        }

        Ok(events)
    }

    fn ensure_config(
        &mut self,
        events: &mut Vec<AudioDemuxEvent>,
        sample_rate: Option<u32>,
        channels: Option<u8>,
    ) {
        if self.emitted_config {
            return;
        }

        self.sample_rate = sample_rate.or(self.sample_rate);
        self.channels = channels.or(self.channels);
        events.push(AudioDemuxEvent::Config(AudioTrackConfig {
            container: AudioContainer::MpegTs,
            codec: self
                .audio_codec
                .clone()
                .unwrap_or_else(|| AudioCodec::Unknown("unknown".to_string())),
            packet_format: self.packet_format.clone(),
            codec_id: None,
            track_id: None,
            pid: self.audio_pid,
            stream_type: self.stream_type,
            sample_rate: self.sample_rate,
            channels: self.channels,
            sample_count: None,
            codec_private: Vec::new(),
            pre_skip: None,
            output_gain: None,
            mapping_family: None,
        }));
        self.emitted_config = true;
    }
}

#[cfg(feature = "mpeg-ts")]
fn psi_section(payload: &[u8], payload_unit_start: bool) -> Result<&[u8], String> {
    if !payload_unit_start {
        return Ok(payload);
    }

    let pointer = *payload
        .first()
        .ok_or_else(|| "Truncated PSI pointer field".to_string())? as usize;
    if payload.len() < 1 + pointer {
        return Err("PSI pointer exceeds payload".to_string());
    }
    Ok(&payload[1 + pointer..])
}

#[cfg(feature = "mpeg-ts")]
fn ts_stream_codec(stream_type: u8) -> Option<(AudioCodec, AudioPacketFormat)> {
    match stream_type {
        0x0f => Some((AudioCodec::Aac, AudioPacketFormat::Adts)),
        0x11 => Some((AudioCodec::Aac, AudioPacketFormat::Latm)),
        0x03 | 0x04 => Some((AudioCodec::Mp3, AudioPacketFormat::Raw)),
        0x81 => Some((AudioCodec::Ac3, AudioPacketFormat::Raw)),
        _ => None,
    }
}

#[cfg(feature = "mpeg-ts")]
fn pes_payload(pes: &[u8]) -> Option<&[u8]> {
    if pes.len() < 9 || pes[0] != 0x00 || pes[1] != 0x00 || pes[2] != 0x01 {
        return None;
    }

    let header_data_len = pes[8] as usize;
    let payload_start = 9 + header_data_len;
    if payload_start > pes.len() {
        return None;
    }

    Some(&pes[payload_start..])
}

#[cfg(feature = "mpeg-ts")]
struct AdtsHeader {
    frame_length: usize,
    sample_rate: u32,
    channels: u8,
}

#[cfg(feature = "mpeg-ts")]
fn parse_adts_header(data: &[u8]) -> Option<AdtsHeader> {
    if data.len() < 7 || data[0] != 0xff || data[1] & 0xf0 != 0xf0 {
        return None;
    }

    let layer = (data[1] & 0x06) >> 1;
    if layer != 0 {
        return None;
    }

    let sample_rate_index = (data[2] & 0x3c) >> 2;
    let sample_rate = adts_sample_rate(sample_rate_index)?;
    let channels = ((data[2] & 0x01) << 2) | ((data[3] & 0xc0) >> 6);
    let frame_length = (((data[3] as usize & 0x03) << 11)
        | ((data[4] as usize) << 3)
        | (data[5] as usize >> 5)) as usize;

    if frame_length < 7 {
        return None;
    }

    Some(AdtsHeader {
        frame_length,
        sample_rate,
        channels,
    })
}

#[cfg(any(feature = "mp4", feature = "mpeg-ts"))]
fn adts_sample_rate(index: u8) -> Option<u32> {
    match index {
        0 => Some(96_000),
        1 => Some(88_200),
        2 => Some(64_000),
        3 => Some(48_000),
        4 => Some(44_100),
        5 => Some(32_000),
        6 => Some(24_000),
        7 => Some(22_050),
        8 => Some(16_000),
        9 => Some(12_000),
        10 => Some(11_025),
        11 => Some(8_000),
        12 => Some(7_350),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    fn fixture(path: &str) -> Vec<u8> {
        fs::read(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("testdata")
                .join(path),
        )
        .unwrap()
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn demuxes_mp4_aac() {
        let data = fixture("mac_aac/A_Tusk_is_used_to_make_costly_gifts.m4a");
        let mut demuxer = AudioTrackDemuxer::new_with_format("mp4").unwrap();
        let mut events = Vec::new();
        for chunk in data.chunks(997) {
            events.extend(demuxer.push(chunk).unwrap());
        }
        events.extend(demuxer.flush().unwrap());

        assert!(events.iter().any(|event| matches!(
            event,
            AudioDemuxEvent::Config(AudioTrackConfig {
                container: AudioContainer::Mp4,
                codec: AudioCodec::Aac,
                sample_rate: Some(16_000),
                channels: Some(1),
                ..
            })
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            AudioDemuxEvent::Packet(AudioTrackPacket {
                container: AudioContainer::Mp4,
                codec: AudioCodec::Aac,
                format: AudioPacketFormat::Adts,
                data,
                ..
            }) if data.starts_with(&[0xff, 0xf1])
        )));
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn demuxes_regular_mp4_with_quicktime_alias_handler() {
        let data = synthetic_regular_mp4_with_alias_handler();
        let mut demuxer = AudioTrackDemuxer::new_with_format("mp4").unwrap();
        let mut events = Vec::new();
        for chunk in data.chunks(23) {
            events.extend(demuxer.push(chunk).unwrap());
        }
        events.extend(demuxer.flush().unwrap());

        assert!(events.iter().any(|event| matches!(
            event,
            AudioDemuxEvent::Config(AudioTrackConfig {
                container: AudioContainer::Mp4,
                codec: AudioCodec::Aac,
                packet_format: Some(AudioPacketFormat::Adts),
                sample_rate: Some(44_100),
                channels: Some(2),
                track_id: Some(1),
                sample_count: Some(2),
                ..
            })
        )));

        let packets: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                AudioDemuxEvent::Packet(packet) => Some(packet),
                _ => None,
            })
            .collect();
        assert_eq!(packets.len(), 2);
        assert_eq!(packets[0].sample_id, Some(1));
        assert_eq!(packets[0].start_time, Some(0));
        assert_eq!(
            packets[0].raw_data.as_deref(),
            Some(&[0x11, 0x22, 0x33][..])
        );
        assert_eq!(packets[1].sample_id, Some(2));
        assert_eq!(packets[1].start_time, Some(1024));
        assert_eq!(packets[1].raw_data.as_deref(), Some(&[0x44, 0x55][..]));
    }

    #[cfg(feature = "webm")]
    #[test]
    fn demuxes_webm_audio() {
        let data = fs::read(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("soundkit-webm")
                .join("testdata")
                .join("test.webm"),
        )
        .unwrap();
        let mut demuxer = AudioTrackDemuxer::new_with_format("webm").unwrap();
        let mut events = Vec::new();
        for chunk in data.chunks(997) {
            events.extend(demuxer.push(chunk).unwrap());
        }
        events.extend(demuxer.flush().unwrap());

        assert!(events.iter().any(|event| matches!(
            event,
            AudioDemuxEvent::Config(AudioTrackConfig {
                container: AudioContainer::WebM,
                codec: AudioCodec::Opus,
                sample_rate: Some(48_000),
                channels: Some(1),
                ..
            })
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            AudioDemuxEvent::Packet(AudioTrackPacket {
                container: AudioContainer::WebM,
                format: AudioPacketFormat::Raw,
                ..
            })
        )));
    }

    #[cfg(feature = "mp4")]
    #[test]
    fn demuxes_fragmented_mp4_cmaf_aac() {
        let data = synthetic_fmp4_segment();
        let mut demuxer = AudioTrackDemuxer::new_with_format("fmp4").unwrap();
        let mut events = Vec::new();
        for chunk in data.chunks(31) {
            events.extend(demuxer.push(chunk).unwrap());
        }
        events.extend(demuxer.flush().unwrap());

        assert!(events.iter().any(|event| matches!(
            event,
            AudioDemuxEvent::Config(AudioTrackConfig {
                container: AudioContainer::Mp4,
                codec: AudioCodec::Aac,
                packet_format: Some(AudioPacketFormat::Adts),
                sample_rate: Some(44_100),
                channels: Some(2),
                track_id: Some(1),
                sample_count: None,
                ..
            })
        )));

        let packets: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                AudioDemuxEvent::Packet(packet) => Some(packet),
                _ => None,
            })
            .collect();
        assert_eq!(packets.len(), 2);
        assert_eq!(packets[0].sample_id, Some(1));
        assert_eq!(packets[0].start_time, Some(0));
        assert_eq!(packets[0].duration, Some(1024));
        assert!(packets[0].data.starts_with(&[0xff, 0xf1]));
        assert_eq!(
            packets[0].raw_data.as_deref(),
            Some(&[0x11, 0x22, 0x33][..])
        );
        assert_eq!(packets[1].sample_id, Some(2));
        assert_eq!(packets[1].start_time, Some(1024));
        assert_eq!(packets[1].raw_data.as_deref(), Some(&[0x44, 0x55][..]));
    }

    #[cfg(feature = "mpeg-ts")]
    #[test]
    fn demuxes_mpeg_ts_aac_adts() {
        let data = synthetic_ts_segment();
        let mut demuxer = AudioTrackDemuxer::new_with_format("mpeg-ts").unwrap();
        let mut events = Vec::new();
        for chunk in data.chunks(113) {
            events.extend(demuxer.push(chunk).unwrap());
        }
        events.extend(demuxer.flush().unwrap());

        assert!(events.iter().any(|event| matches!(
            event,
            AudioDemuxEvent::Config(AudioTrackConfig {
                container: AudioContainer::MpegTs,
                codec: AudioCodec::Aac,
                packet_format: Some(AudioPacketFormat::Adts),
                sample_rate: Some(44_100),
                channels: Some(2),
                pid: Some(0x0101),
                stream_type: Some(0x0f),
                ..
            })
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            AudioDemuxEvent::Packet(AudioTrackPacket {
                container: AudioContainer::MpegTs,
                codec: AudioCodec::Aac,
                format: AudioPacketFormat::Adts,
                data,
                ..
            }) if data.starts_with(&[0xff, 0xf1])
        )));
    }

    #[cfg(feature = "mp4")]
    fn synthetic_fmp4_segment() -> Vec<u8> {
        let mut out = Vec::new();
        out.extend(mp4_box(
            b"ftyp",
            &[
                b"isom".as_slice(),
                &0u32.to_be_bytes(),
                b"iso6",
                b"cmfc",
                b"mp41",
            ]
            .concat(),
        ));
        out.extend(synthetic_init_moov());
        out.extend(synthetic_fragment());
        out
    }

    #[cfg(feature = "mp4")]
    fn synthetic_regular_mp4_with_alias_handler() -> Vec<u8> {
        let raw_a = vec![0x11, 0x22, 0x33];
        let raw_b = vec![0x44, 0x55];
        let ftyp = mp4_box(
            b"ftyp",
            &[b"isom".as_slice(), &0u32.to_be_bytes(), b"isom", b"mp42"].concat(),
        );
        let moov = synthetic_regular_moov(0);
        let chunk_offset = ftyp.len() + moov.len() + 8;
        let moov = synthetic_regular_moov(chunk_offset as u32);
        let mdat = mp4_box(b"mdat", &[raw_a, raw_b].concat());
        [ftyp, moov, mdat].concat()
    }

    #[cfg(feature = "mp4")]
    fn synthetic_regular_moov(chunk_offset: u32) -> Vec<u8> {
        let mvhd = full_box(b"mvhd", 0, 0, &[0u8; 20]);
        let tkhd_payload = [
            &0u32.to_be_bytes()[..],
            &0u32.to_be_bytes(),
            &1u32.to_be_bytes(),
            &0u32.to_be_bytes(),
            &0u32.to_be_bytes(),
            &[0u8; 60],
        ]
        .concat();
        let tkhd = full_box(b"tkhd", 0, 0x000007, &tkhd_payload);
        let mdhd_payload = [
            &0u32.to_be_bytes()[..],
            &0u32.to_be_bytes(),
            &44_100u32.to_be_bytes(),
            &0u32.to_be_bytes(),
            &0u16.to_be_bytes(),
            &0u16.to_be_bytes(),
        ]
        .concat();
        let mdhd = full_box(b"mdhd", 0, 0, &mdhd_payload);
        let hdlr = full_box(
            b"hdlr",
            0,
            0,
            &[&0u32.to_be_bytes()[..], b"soun", &[0u8; 12].as_slice()].concat(),
        );
        let smhd = full_box(b"smhd", 0, 0, &[0u8; 4]);
        let alias_hdlr = full_box(
            b"hdlr",
            0,
            0,
            &[&0u32.to_be_bytes()[..], b"alis", &[0u8; 12].as_slice()].concat(),
        );
        let dinf = mp4_box(b"dinf", &[]);
        let stsd = full_box(
            b"stsd",
            0,
            0,
            &[&1u32.to_be_bytes()[..], &synthetic_mp4a_entry()].concat(),
        );
        let stts = full_box(
            b"stts",
            0,
            0,
            &[
                &1u32.to_be_bytes()[..],
                &2u32.to_be_bytes(),
                &1024u32.to_be_bytes(),
            ]
            .concat(),
        );
        let stsc = full_box(
            b"stsc",
            0,
            0,
            &[
                &1u32.to_be_bytes()[..],
                &1u32.to_be_bytes(),
                &2u32.to_be_bytes(),
                &1u32.to_be_bytes(),
            ]
            .concat(),
        );
        let stsz = full_box(
            b"stsz",
            0,
            0,
            &[
                &0u32.to_be_bytes()[..],
                &2u32.to_be_bytes(),
                &3u32.to_be_bytes(),
                &2u32.to_be_bytes(),
            ]
            .concat(),
        );
        let stco = full_box(
            b"stco",
            0,
            0,
            &[&1u32.to_be_bytes()[..], &chunk_offset.to_be_bytes()].concat(),
        );
        let stbl = mp4_box(b"stbl", &[stsd, stts, stsc, stsz, stco].concat());
        let minf = mp4_box(b"minf", &[smhd, alias_hdlr, dinf, stbl].concat());
        let mdia = mp4_box(b"mdia", &[mdhd, hdlr, minf].concat());
        let trak = mp4_box(b"trak", &[tkhd, mdia].concat());
        mp4_box(b"moov", &[mvhd, trak].concat())
    }

    #[cfg(feature = "mp4")]
    fn synthetic_init_moov() -> Vec<u8> {
        let mvhd = full_box(b"mvhd", 0, 0, &[0u8; 20]);
        let tkhd_payload = [
            &0u32.to_be_bytes()[..],
            &0u32.to_be_bytes(),
            &1u32.to_be_bytes(),
            &0u32.to_be_bytes(),
            &0u32.to_be_bytes(),
            &[0u8; 60],
        ]
        .concat();
        let tkhd = full_box(b"tkhd", 0, 0x000007, &tkhd_payload);
        let mdhd_payload = [
            &0u32.to_be_bytes()[..],
            &0u32.to_be_bytes(),
            &44_100u32.to_be_bytes(),
            &0u32.to_be_bytes(),
            &0u16.to_be_bytes(),
            &0u16.to_be_bytes(),
        ]
        .concat();
        let mdhd = full_box(b"mdhd", 0, 0, &mdhd_payload);
        let hdlr = full_box(
            b"hdlr",
            0,
            0,
            &[&0u32.to_be_bytes()[..], b"soun", &[0u8; 12].as_slice()].concat(),
        );
        let smhd = full_box(b"smhd", 0, 0, &[0u8; 4]);
        let alias_hdlr = full_box(
            b"hdlr",
            0,
            0,
            &[&0u32.to_be_bytes()[..], b"alis", &[0u8; 12].as_slice()].concat(),
        );
        let dinf = mp4_box(b"dinf", &[]);
        let stsd = full_box(
            b"stsd",
            0,
            0,
            &[&1u32.to_be_bytes()[..], &synthetic_mp4a_entry()].concat(),
        );
        let stbl = mp4_box(b"stbl", &stsd);
        let minf = mp4_box(b"minf", &[smhd, alias_hdlr, dinf, stbl].concat());
        let mdia = mp4_box(b"mdia", &[mdhd, hdlr, minf].concat());
        let trak = mp4_box(b"trak", &[tkhd, mdia].concat());
        let trex_payload = [
            &1u32.to_be_bytes()[..],
            &1u32.to_be_bytes(),
            &1024u32.to_be_bytes(),
            &0u32.to_be_bytes(),
            &0u32.to_be_bytes(),
        ]
        .concat();
        let trex = full_box(b"trex", 0, 0, &trex_payload);
        let mvex = mp4_box(b"mvex", &trex);
        mp4_box(b"moov", &[mvhd, trak, mvex].concat())
    }

    #[cfg(feature = "mp4")]
    fn synthetic_mp4a_entry() -> Vec<u8> {
        let asc = [0x12, 0x10];
        let dec_specific = descriptor(0x05, &asc);
        let dec_config = descriptor(
            0x04,
            &[
                &[0x40, 0x15, 0x00, 0x00, 0x00][..],
                &0u32.to_be_bytes(),
                &0u32.to_be_bytes(),
                &dec_specific,
            ]
            .concat(),
        );
        let es = descriptor(
            0x03,
            &[&0u16.to_be_bytes()[..], &[0x00], &dec_config].concat(),
        );
        let esds = full_box(b"esds", 0, 0, &es);
        let payload = [
            &[0u8; 6][..],
            &1u16.to_be_bytes(),
            &[0u8; 8],
            &2u16.to_be_bytes(),
            &16u16.to_be_bytes(),
            &0u16.to_be_bytes(),
            &0u16.to_be_bytes(),
            &(44_100u32 << 16).to_be_bytes(),
            &esds,
        ]
        .concat();
        mp4_box(b"mp4a", &payload)
    }

    #[cfg(feature = "mp4")]
    fn synthetic_fragment() -> Vec<u8> {
        let raw_a = vec![0x11, 0x22, 0x33];
        let raw_b = vec![0x44, 0x55];
        let mfhd = full_box(b"mfhd", 0, 0, &1u32.to_be_bytes());
        let tfhd = full_box(b"tfhd", 0, 0x020000, &1u32.to_be_bytes());
        let tfdt = full_box(b"tfdt", 0, 0, &0u32.to_be_bytes());
        let trun_size = 8 + 4 + 4 + 4 + (2 * 8);
        let traf_size = 8 + tfhd.len() + tfdt.len() + trun_size;
        let moof_size = 8 + mfhd.len() + traf_size;
        let data_offset = (moof_size + 8) as i32;
        let trun_payload = [
            &2u32.to_be_bytes()[..],
            &data_offset.to_be_bytes(),
            &1024u32.to_be_bytes(),
            &(raw_a.len() as u32).to_be_bytes(),
            &1024u32.to_be_bytes(),
            &(raw_b.len() as u32).to_be_bytes(),
        ]
        .concat();
        let trun = full_box(b"trun", 0, 0x000301, &trun_payload);
        let traf = mp4_box(b"traf", &[tfhd, tfdt, trun].concat());
        let moof = mp4_box(b"moof", &[mfhd, traf].concat());
        assert_eq!(moof.len(), moof_size);
        let mdat = mp4_box(b"mdat", &[raw_a, raw_b].concat());
        [moof, mdat].concat()
    }

    #[cfg(feature = "mp4")]
    fn descriptor(tag: u8, payload: &[u8]) -> Vec<u8> {
        assert!(payload.len() < 128);
        let mut out = vec![tag, payload.len() as u8];
        out.extend_from_slice(payload);
        out
    }

    #[cfg(feature = "mp4")]
    fn full_box(name: &[u8; 4], version: u8, flags: u32, payload: &[u8]) -> Vec<u8> {
        let mut full_payload = vec![
            version,
            ((flags >> 16) & 0xff) as u8,
            ((flags >> 8) & 0xff) as u8,
            (flags & 0xff) as u8,
        ];
        full_payload.extend_from_slice(payload);
        mp4_box(name, &full_payload)
    }

    #[cfg(feature = "mp4")]
    fn mp4_box(name: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let size = 8 + payload.len();
        let mut out = Vec::with_capacity(size);
        out.extend_from_slice(&(size as u32).to_be_bytes());
        out.extend_from_slice(name);
        out.extend_from_slice(payload);
        out
    }

    #[cfg(feature = "mpeg-ts")]
    fn synthetic_ts_segment() -> Vec<u8> {
        let mut out = Vec::new();
        out.extend(ts_packet(0x0000, true, &pat_section(0x1000)));
        out.extend(ts_packet(0x1000, true, &pmt_section(0x0101, 0x0f)));
        out.extend(ts_packet(0x0101, true, &pes_packet(&adts_frame())));
        out
    }

    #[cfg(feature = "mpeg-ts")]
    fn ts_packet(pid: u16, payload_unit_start: bool, payload: &[u8]) -> Vec<u8> {
        let mut packet = vec![0xff; 188];
        packet[0] = 0x47;
        packet[1] = ((pid >> 8) as u8 & 0x1f) | if payload_unit_start { 0x40 } else { 0 };
        packet[2] = pid as u8;
        packet[3] = 0x10;
        let copy_len = payload.len().min(184);
        packet[4..4 + copy_len].copy_from_slice(&payload[..copy_len]);
        packet
    }

    #[cfg(feature = "mpeg-ts")]
    fn pat_section(pmt_pid: u16) -> Vec<u8> {
        let mut section = vec![
            0x00,
            0x00,
            0xb0,
            0x0d,
            0x00,
            0x01,
            0xc1,
            0x00,
            0x00,
            0x00,
            0x01,
            0xe0 | ((pmt_pid >> 8) as u8 & 0x1f),
            pmt_pid as u8,
            0,
            0,
            0,
            0,
        ];
        section[0] = 0;
        section
    }

    #[cfg(feature = "mpeg-ts")]
    fn pmt_section(audio_pid: u16, stream_type: u8) -> Vec<u8> {
        vec![
            0x00,
            0x02,
            0xb0,
            0x12,
            0x00,
            0x01,
            0xc1,
            0x00,
            0x00,
            0xe1,
            0x00,
            0xf0,
            0x00,
            stream_type,
            0xe0 | ((audio_pid >> 8) as u8 & 0x1f),
            audio_pid as u8,
            0xf0,
            0x00,
            0,
            0,
            0,
            0,
        ]
    }

    #[cfg(feature = "mpeg-ts")]
    fn pes_packet(payload: &[u8]) -> Vec<u8> {
        let pes_len = payload.len() + 3;
        let mut pes = vec![
            0x00,
            0x00,
            0x01,
            0xc0,
            (pes_len >> 8) as u8,
            pes_len as u8,
            0x80,
            0x00,
            0x00,
        ];
        pes.extend_from_slice(payload);
        pes
    }

    #[cfg(feature = "mpeg-ts")]
    fn adts_frame() -> Vec<u8> {
        let payload_len = 8usize;
        let frame_len = 7 + payload_len;
        let mut frame = vec![0u8; frame_len];
        frame[0] = 0xff;
        frame[1] = 0xf1;
        frame[2] = (1 << 6) | (4 << 2);
        frame[3] = (2 << 6) | (((frame_len >> 11) & 0x03) as u8);
        frame[4] = ((frame_len >> 3) & 0xff) as u8;
        frame[5] = (((frame_len & 0x07) << 5) as u8) | 0x1f;
        frame[6] = 0xfc;
        for (idx, byte) in frame[7..].iter_mut().enumerate() {
            *byte = idx as u8;
        }
        frame
    }
}
