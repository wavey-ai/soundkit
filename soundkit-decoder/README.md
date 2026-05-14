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
| MP3 | `spawn()` | Auto | Yes | `soundkit-mp3` / `nanomp3` |
| AAC ADTS | `spawn()` | Auto | Yes | `soundkit-aac` / `fdk-aac` |
| AAC in M4A/MP4 | `spawn()` | Auto | Limited | `mp4` + `fdk-aac` |
| FLAC | `spawn()` | Auto | Yes | `soundkit-flac` / `claxon` |
| Raw Opus stream | `spawn()` | Auto | Yes | `soundkit-opus` / `libopus` |
| Ogg Opus | `spawn()` | Auto | Yes | `soundkit-ogg-opus` + `libopus` |
| WebM Opus | `spawn()` | Auto | Yes | `soundkit-webm` + `libopus` |
| Ogg Speex | `spawn_speex()` | Explicit | Yes | `soundkit-speex` / `oxideav-speex` |
| Ogg Vorbis | `spawn()` or `spawn_vorbis()` | Auto or explicit | Yes | `soundkit-vorbis` / `lewton` |
| ALAC in M4A/MP4 or CAF | `spawn()` or `spawn_alac()` | Auto or explicit | EOF | `soundkit-alac` / `alac` |
| AIFF / AIFF-C | `spawn()` or `spawn_aiff()` | Auto or explicit | EOF | `soundkit-aiff` / `aifc` |
| Raw AC-3 syncframes | `spawn()` or `spawn_ac3()` | Auto or explicit | Yes | `soundkit-ac3` / `oxideav-ac3` |
| AMR-NB | `spawn_amr_nb()` | Explicit | Yes | `soundkit-amr` / OpenCORE AMR-NB |
| G.711 u-law / A-law | `spawn_g711(law, rate, channels)` | Explicit | Yes | `soundkit-g711` |
| G.722 | `spawn_g722()` | Explicit | Yes | `soundkit-g722` / `ezk-g722` |
| G.726 16/24/32/40 | `spawn_g726_with_rate(rate, packing)` | Explicit | Yes | `soundkit-g726` |
| G.729 | `spawn_g729()` | Explicit | Yes | `soundkit-g729` / `g729-sys` |
| GSM 06.10 / WAV-49 | `spawn_gsm(variant)` | Explicit | Yes | `soundkit-gsm` / `libgsm` |

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
| MP3, AAC ADTS, M4A/MP4 AAC, FLAC, Opus, Ogg Opus, Ogg Vorbis, Ogg Speex, WebM, WAV, ALAC, AIFF/AIFF-C, AC-3 | `access-unit` detection. |
| Headerless PCM and telephony codecs | Explicit spawn APIs because metadata is not present in the byte stream. |

## Current Gaps

| Format | Gap |
| --- | --- |
| ALAC and AIFF/AIFF-C | Seek-based readers make current wrappers EOF-buffered. |
| AAC in M4A/MP4 | MP4 sample tables make live chunking layout-dependent; use ADTS for live AAC. |
| APE | Deferred until fixtures can be generated with FFmpeg in this repo's test pattern. |
