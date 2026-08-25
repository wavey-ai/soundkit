# soundkit-decoder

Threaded audio decode pipeline for chunked input. It autodetects common media
containers and exposes explicit spawn paths for headerless telephony codecs.

## Streaming Model

| Term | Meaning |
| --- | --- |
| `Yes` | The decoder can emit PCM before EOF from chunked input. |
| `Limited` | Chunked files can work, but the container may need metadata/media buffered first. |
| `EOF` | The wrapper accepts chunks, but emits only after the caller sends an empty EOF chunk. |
| `Auto` | `DecodePipeline::spawn()` can detect the format from bytes. |
| `Explicit` | The caller must select the decoder because the stream is headerless or not autodetected. |

## Format Matrix

| Format | Spawn API | Detection | Stream output | Backend |
| --- | --- | --- | --- | --- |
| Raw PCM (`linear16`, `linear32`, `s16le`, `f32le`, `L16`) | `spawn_raw_pcm(format)` | Explicit | Yes | `soundkit::raw_pcm` |
| WAV / RIFF PCM | `spawn()` | Auto | Yes | `WavStreamProcessor` |
| MP3 | `spawn()` | Auto | Yes | `soundkit-mp3` owned decoder |
| AAC ADTS | `spawn()` | Auto | Yes | `soundkit-aac` / `fdk-aac` |
| AAC-LC/HE-AAC in M4A/MP4/MOV or Matroska | `decode_audio_file()` | Auto | EOF/Limited | owned Rust demux + Wavey Symphonia fork |
| MP2 in MPEG-TS | `decode_audio_file()` | Auto | EOF | owned Rust TS demux + Wavey Symphonia fork |
| PCM in AVI; DVD LPCM in MPEG-PS/VOB | `decode_audio_file()` | Auto | EOF | owned Rust demux/PCM conversion |
| FLAC | `spawn()` | Auto | Yes | `soundkit-flac` / `wavey-flac` |
| Raw Opus stream | `spawn()` | Auto | Yes | `soundkit-opus` owned decoder |
| Ogg Opus | `spawn()` | Auto | Yes | `soundkit-ogg-opus` + `soundkit-opus` |
| WebM Opus / Vorbis | `spawn()` | Auto | Yes | `soundkit-webm` + owned Opus/Vorbis decoders |
| Ogg Speex | `spawn_speex()` | Explicit | Yes | `soundkit-speex` / `oxideav-speex` |
| Ogg Vorbis | `spawn()` or `spawn_vorbis()` | Auto or explicit | Yes | `soundkit-vorbis` owned decoder |
| ALAC in M4A/MP4 or CAF | `spawn()` or `spawn_alac()` | Auto or explicit | EOF | `soundkit-alac` / `alac` |
| AIFF / AIFF-C | `spawn()` or `spawn_aiff()` | Auto or explicit | EOF | `soundkit-aiff` / `aifc` |
| Raw AC-3 syncframes | `spawn()` or `spawn_ac3()` | Auto or explicit | Yes | `soundkit-ac3` / `oxideav-ac3` |
| AMR-NB | `spawn_amr_nb()` | Explicit | Yes | `soundkit-amr` / OpenCORE AMR-NB |
| G.711 u-law / A-law | `spawn_g711(law, rate, channels)` | Explicit | Yes | `soundkit-g711` |
| G.722 | `spawn_g722()` | Explicit | Yes | `soundkit-g722` owned codec |
| G.726 16/24/32/40 | `spawn_g726_with_rate(rate, packing)` | Explicit | Yes | `soundkit-g726` |
| G.729 | `spawn_g729()` | Explicit | Yes | `soundkit-g729` / `g729-sys` |
| GSM 06.10 / WAV-49 | `spawn_gsm(variant)` | Explicit | Yes | `soundkit-gsm` / `libgsm` |

For complete files, prefer `decode_audio_file`. It returns `DecodedAudioFile`,
which contains normalized `MediaMetadata`, the selected container track when
applicable, and decoded PCM frames. This path handles regular and fragmented
MP4/MOV, Matroska AAC, CAF, AVI, MPEG-PS/VOB, MXF, and MPEG-TS layouts that
require packet metadata:

```rust
use soundkit_decoder::{decode_audio_file, DecodeOptions};

let source = std::fs::read("video.mp4")?;
let decoded = decode_audio_file(&source, DecodeOptions {
    output_bits_per_sample: Some(16),
    ..DecodeOptions::default()
})?;
println!("{:?} / {:?}", decoded.metadata.artists.first(), decoded.metadata.title);
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Usage

```rust
use soundkit_decoder::{Bytes, DecodePipeline};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut pipeline = DecodePipeline::spawn();
    pipeline.send(Bytes::from(std::fs::read("audio.mp3")?))?;
    pipeline.send(Bytes::new())?; // EOF / flush

    while let Some(frame) = pipeline.try_recv() {
        let audio = frame?;
        println!(
            "{} Hz, {} ch, {} bits",
            audio.sampling_rate(),
            audio.channel_count(),
            audio.bits_per_sample()
        );
    }

    Ok(())
}
```

## Output Conversion

| Option | Field | Behavior |
| --- | --- | --- |
| Bit depth | `DecodeOptions::output_bits_per_sample` | Converts decoded PCM to the requested depth. |
| Sample rate | `DecodeOptions::output_sample_rate` | Uses a stateful `rubato` resampler across frames. |
| Channels | `DecodeOptions::output_channels` | Downmixes or adapts channel count after decode. |

```rust
use soundkit_decoder::{DecodeOptions, DecodePipeline};

let pipeline = DecodePipeline::spawn_with_options(DecodeOptions {
    output_bits_per_sample: Some(16),
    output_sample_rate: Some(16_000),
    output_channels: Some(1),
});
```

## Detection Notes

| Format family | Detection path |
| --- | --- |
| MP3, AAC ADTS, M4A/MP4 AAC, FLAC, Opus, Ogg Opus, Ogg Vorbis, Ogg Speex, WebM, WAV, ALAC, AIFF/AIFF-C, AC-3 | `access-unit` detection plus owned container probes. |
| AVI, MPEG-PS/VOB, MPEG-TS/M2TS, CAF, and MXF | `decode_audio_file` container probes and bounded Rust demuxers. |
| Headerless PCM and telephony codecs | Explicit spawn APIs because metadata is not present in the byte stream. |

## YouTube Audio Itags

`DecodePipeline::spawn()` does not route by numeric itag. It routes by the downloaded
container and audio codec, which covers the common YouTube audio itag families:

| YouTube audio family | Example itags | Decode path |
| --- | --- | --- |
| MP4 AAC-LC | `140`, `141` | owned Rust MP4 demux + Wavey pure-Rust AAC fork through `decode_audio_file` |
| MP4 HE-AAC | `139`, `256`, `258`, `599` | same owned path with SBR/PS; the collected itag-139 fixture measures 68.59 dB against FFmpeg |
| WebM Opus | `249`, `250`, `251`, `600`, `774` | WebM demux + `soundkit-opus` |
| WebM Vorbis | `171`, `172` | WebM demux + `soundkit-vorbis` owned decoder |

## Current Gaps

| Format | Gap |
| --- | --- |
| Opus SILK, hybrid, FEC, and mode transitions | `soundkit-opus` currently decodes 48 kHz CELT packets. Other modes return explicit unsupported errors. |
| Streaming ALAC and AIFF/AIFF-C | Seek-based readers make the streaming wrappers EOF-buffered; complete ALAC in M4A/CAF is packet-decoded by `decode_audio_file`. |
| AAC in M4A/MP4 | MP4 sample tables make live chunking layout-dependent; use ADTS for live AAC. |
| MP4 AC-3 / E-AC-3 YouTube surround itags | Raw AC-3 syncframes decode, but MP4-contained AC-3/E-AC-3 is not wired yet. |
| DTS core/ES | TS/M2TS access units are framed, but no production-quality Rust PCM decoder is connected. |
| ASF/WMA | Metadata and artwork are parsed, but complete-file WMA PCM extraction is not connected. |
| AVI MP3 | The collected low-rate/corrupt-packet sample decodes at only 12.49 dB against FFmpeg. |
| 64-bit float PCM | CAF/MOV f64 is explicitly rejected; integer PCM through 32-bit and f32 are supported. |
| APE | Deferred until fixtures can be generated with FFmpeg in this repo's test pattern. |
