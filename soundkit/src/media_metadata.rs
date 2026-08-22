//! Normalized descriptive and technical metadata shared by SoundKit formats.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

mod parser;
pub use parser::extract_metadata;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MediaMetadata {
    pub title: Option<String>,
    pub album: Option<String>,
    pub artists: Vec<String>,
    pub album_artists: Vec<String>,
    pub composers: Vec<String>,
    pub genres: Vec<String>,
    pub date: Option<String>,
    pub track_number: Option<u32>,
    pub track_total: Option<u32>,
    pub disc_number: Option<u32>,
    pub disc_total: Option<u32>,
    pub comment: Option<String>,
    pub lyrics: Option<String>,
    pub copyright: Option<String>,
    pub encoder: Option<String>,
    pub container: Option<String>,
    pub duration_micros: Option<u64>,
    pub audio_tracks: Vec<AudioTrackMetadata>,
    pub video_tracks: Vec<VideoTrackMetadata>,
    /// Embedded cover art. Image payloads are bounded by the metadata parser.
    pub artwork: Vec<ArtworkMetadata>,
    /// Original textual tags, retained under their source spelling.
    pub tags: BTreeMap<String, Vec<String>>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtworkMetadata {
    /// Container-specific picture type. FLAC and ID3 use the shared ID3/APIC values.
    pub picture_type: Option<u32>,
    pub mime_type: Option<String>,
    pub description: Option<String>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub color_depth: Option<u32>,
    pub indexed_colors: Option<u32>,
    pub data: Vec<u8>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioTrackMetadata {
    pub id: Option<u64>,
    pub codec: Option<String>,
    pub codec_id: Option<String>,
    pub title: Option<String>,
    pub language: Option<String>,
    pub sample_rate: Option<u32>,
    pub channels: Option<u16>,
    pub bits_per_sample: Option<u8>,
    pub duration_micros: Option<u64>,
    pub bitrate: Option<u64>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct VideoTrackMetadata {
    pub id: Option<u64>,
    pub codec: Option<String>,
    pub codec_id: Option<String>,
    pub title: Option<String>,
    pub language: Option<String>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub bit_depth: Option<u8>,
    pub duration_micros: Option<u64>,
    pub bitrate: Option<u64>,
}

impl MediaMetadata {
    /// Retains a source tag and maps common ID3, Vorbis, iTunes, RIFF, and
    /// Matroska spellings into the normalized fields.
    pub fn insert_tag(&mut self, key: impl Into<String>, value: impl Into<String>) {
        let key = key.into();
        let value = value.into().trim().to_owned();
        if key.trim().is_empty() || value.is_empty() {
            return;
        }
        self.tags
            .entry(key.clone())
            .or_default()
            .push(value.clone());

        let normalized = normalize_key(&key);
        match normalized.as_str() {
            "title" | "tit2" | "tt2" | "nam" | "name" => set_once(&mut self.title, value),
            "album" | "talb" | "tal" | "alb" | "wmalbumtitle" => set_once(&mut self.album, value),
            "artist" | "author" | "tpe1" | "tp1" | "art" | "wmauthor" => {
                push_unique(&mut self.artists, value)
            }
            "albumartist" | "tpe2" | "aart" | "wmalbumartist" => {
                push_unique(&mut self.album_artists, value);
            }
            "composer" | "tcom" | "wrt" | "wmcomposer" => push_unique(&mut self.composers, value),
            "genre" | "tcon" | "gen" | "wmgenre" => push_unique(&mut self.genres, value),
            "date" | "tdrc" | "tyer" | "tye" | "day" | "year" | "wmyear" => {
                set_once(&mut self.date, value)
            }
            "track" | "tracknumber" | "trck" | "trk" | "trkn" => {
                assign_number_pair(&value, &mut self.track_number, &mut self.track_total);
            }
            // WM/Track is the legacy zero-based ASF field. WM/TrackNumber is
            // the preferred one-based value and is allowed to replace it.
            "wmtrack" => {
                if self.track_number.is_none() {
                    self.track_number =
                        parse_number(&value).and_then(|number| number.checked_add(1));
                }
            }
            "wmtracknumber" => {
                if let Some(number) = value.split('/').next().and_then(parse_number) {
                    self.track_number = Some(number);
                }
                if self.track_total.is_none() {
                    self.track_total = value
                        .split_once('/')
                        .and_then(|(_, total)| parse_number(total));
                }
            }
            "tracktotal" | "totaltracks" => assign_number(&value, &mut self.track_total),
            "disc" | "discnumber" | "disk" | "tpos" | "tpa" => {
                assign_number_pair(&value, &mut self.disc_number, &mut self.disc_total);
            }
            "disctotal" | "totaldiscs" => assign_number(&value, &mut self.disc_total),
            "comment" | "description" | "comm" | "cmt" | "wmdescription" => {
                set_once(&mut self.comment, value)
            }
            "lyrics" | "uslt" | "unsyncedlyrics" => set_once(&mut self.lyrics, value),
            "copyright" | "tcop" | "cpy" => set_once(&mut self.copyright, value),
            "encoder" | "encodedby" | "tenc" | "tsse" | "too" | "vendor" => {
                set_once(&mut self.encoder, value);
            }
            _ => {}
        }
    }
}

fn normalize_key(key: &str) -> String {
    key.chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn set_once(target: &mut Option<String>, value: String) {
    if target.is_none() {
        *target = Some(value);
    }
}

fn push_unique(target: &mut Vec<String>, value: String) {
    if !target.iter().any(|existing| existing == &value) {
        target.push(value);
    }
}

fn assign_number(value: &str, target: &mut Option<u32>) {
    if target.is_none() {
        *target = parse_number(value);
    }
}

fn assign_number_pair(value: &str, number: &mut Option<u32>, total: &mut Option<u32>) {
    let mut fields = value.splitn(2, '/');
    let parsed_number = fields.next().and_then(parse_number);
    let parsed_total = fields.next().and_then(parse_number);
    if number.is_none() {
        *number = parsed_number;
    }
    if total.is_none() {
        *total = parsed_total;
    }
}

fn parse_number(value: &str) -> Option<u32> {
    value
        .trim()
        .trim_start_matches('0')
        .parse()
        .ok()
        .or_else(|| value.trim().parse().ok())
}

#[cfg(test)]
mod tests {
    use super::MediaMetadata;

    #[test]
    fn normalizes_common_tag_spellings_and_number_pairs() {
        let mut metadata = MediaMetadata::default();
        metadata.insert_tag("TIT2", "Track title");
        metadata.insert_tag("ARTIST", "First artist");
        metadata.insert_tag("TPE1", "Second artist");
        metadata.insert_tag("ALBUMARTIST", "Album artist");
        metadata.insert_tag("TRCK", "03/12");
        metadata.insert_tag("DISCNUMBER", "2/4");
        metadata.insert_tag("DATE", "2026-08-22");

        assert_eq!(metadata.title.as_deref(), Some("Track title"));
        assert_eq!(metadata.artists, ["First artist", "Second artist"]);
        assert_eq!(metadata.album_artists, ["Album artist"]);
        assert_eq!(metadata.track_number, Some(3));
        assert_eq!(metadata.track_total, Some(12));
        assert_eq!(metadata.disc_number, Some(2));
        assert_eq!(metadata.disc_total, Some(4));
        assert_eq!(metadata.date.as_deref(), Some("2026-08-22"));
        assert_eq!(metadata.tags["TRCK"], ["03/12"]);
    }

    #[test]
    fn retains_unknown_and_repeated_tags_without_overwriting_primary_values() {
        let mut metadata = MediaMetadata::default();
        metadata.insert_tag("TITLE", "Primary");
        metadata.insert_tag("TITLE", "Alternate");
        metadata.insert_tag("MusicBrainz Track Id", "abc");

        assert_eq!(metadata.title.as_deref(), Some("Primary"));
        assert_eq!(metadata.tags["TITLE"], ["Primary", "Alternate"]);
        assert_eq!(metadata.tags["MusicBrainz Track Id"], ["abc"]);
    }

    #[test]
    fn normalizes_legacy_zero_based_asf_track_without_inventing_a_total() {
        let mut metadata = MediaMetadata::default();
        metadata.insert_tag("WM/Track", "0");
        metadata.insert_tag("WM/TrackNumber", "1");

        assert_eq!(metadata.track_number, Some(1));
        assert_eq!(metadata.track_total, None);
    }
}
