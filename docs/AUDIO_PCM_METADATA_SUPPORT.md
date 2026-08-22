# Audio extraction and metadata support

`soundkit_decoder::decode_audio_file` is the production complete-file entry
point. It extracts the first supported audio track from audio-only or video
containers, converts it to the requested PCM representation, and returns the
same file's normalized `soundkit::media_metadata::MediaMetadata`.

## PCM extraction status

| Family | Collected fixtures | FFmpeg differential result |
| --- | --- | --- |
| WAV, AIFF/AIFF-C, CAF, MOV PCM | s16, s24, s32, f32; both byte orders where available | Bit-exact after s16 conversion |
| FLAC, ALAC | Native files, M4A/CAF, FLAC-in-MP4 video | Bit-exact |
| MP3 | Native MP3 | 72.24 dB; decoder-delay alignment measured |
| Vorbis | Ogg, WebM audio | 51.90–69.21 dB |
| Opus | Ogg and WebM behind VP9/AV1 video | 64.45–73.29 dB |
| AAC-LC | M4A, regular/fragmented/CMAF/DASH MP4, MOV, Matroska, TS/M2TS | 75.45–80.56 dB for MP4/Matroska; 43.19 dB for TS |
| HE-AAC/SBR | MP4 with backward-compatible SBR signalling | 68.59 dB |
| MP2 | MPEG-TS | 50.36 dB |
| AC-3 | Raw AC-3 | 51.97 dB |
| AVI audio | u8/s16 PCM and MP3 | PCM is bit-exact; the collected MP3 case remains a 12.49 dB quality gap |
| MPEG-PS/VOB | DVD LPCM 16-bit and 24-bit | Bit-exact after s16 conversion |
| MXF PCM/AES3 | OP-Atom, OP1a, Avid multitrack, and D-10 AES3 | Bit-exact |
| Video independence | H.264, HEVC, VP9, AV1, DNxHR, and ProRes containers | Audio extraction passes regardless of video profile |

The formal manifest currently contains 56 cases. Its expected rejects are also
checked against FFmpeg, so a gap cannot be recorded using a corrupt fixture.

## Normalized metadata

`MediaMetadata` has first-class fields for `title`, `album`, `artists`,
`album_artists`, composers, genres, date, track/disc numbers and totals,
comments, lyrics, copyright, encoder, duration, technical audio/video tracks,
and bounded embedded artwork. Unknown and repeated textual tags remain
available in the `tags` map.

The bounded parser currently normalizes:

- ID3v1 and ID3v2.2/v2.3/v2.4, including APIC/PIC artwork;
- FLAC and Ogg Vorbis comments, including Opus tags and FLAC picture blocks;
- MP4/M4A/MOV iTunes metadata, including `covr` artwork;
- APEv2 text and front/back cover-art items;
- ASF/WMA Content Description, Extended Content, Metadata/Metadata Library,
  embedded ID3, and `WM/Picture` objects;
- RIFF INFO and embedded WAV ID3;
- AIFF text chunks and embedded ID3;
- Matroska/WebM SimpleTags.

The combined decode result also enriches technical tracks from MP4/MOV,
Matroska/WebM, CAF, and MPEG-TS demux configuration.

## Formal checks

```sh
make media-pcm-fixtures
make media-metadata-fate
make media-metadata-sweep
make media-audio-fuzz
```

Current results:

- PCM/FFmpeg differential suite: 56/56 expected outcomes;
- tagged metadata conformance: 11/11;
- FATE metadata sweep: 771 accepted, 26 clean rejects, 0 panics;
- deterministic fixture mutation sweep: 336 cases, 0 panics.

## Remaining codec gaps

| Priority | Gap | Current behavior |
| --- | --- | --- |
| High | DTS core/ES | Rust TS demux succeeds, but no production-quality PCM decoder is available |
| High | ASF/WMA | No complete-file demux plus production-quality Rust WMA decoder path |
| Medium | AVI MP3 | Decodes, but the collected low-rate/corrupt-packet sample is only 12.49 dB against FFmpeg |
| Medium | MPEG-PS compressed audio and 20-bit DVD LPCM | Complete-file support currently covers byte-exact 16/24-bit DVD LPCM |
| Low | CAF/MOV 64-bit float PCM | Explicitly rejected because SoundKit PCM models f32, not f64 |
