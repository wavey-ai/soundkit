# soundkit-aac

This crate provides the native FDK AAC encoder and decoder. It supports ADTS
streams and AAC tracks in M4A/MP4 files.

Use this crate when the application needs broad native AAC compatibility. The
default feature uses the FDK AAC C library.

Use `soundkit-aac-lc` for the pure Rust production profile. That profile is
stereo MPEG-4 AAC-LC at 44.1 or 48 kHz.

## Features

| Feature | Function |
| --- | --- |
| `fdk` | Encode and decode ADTS with FDK AAC. |
| `mp4-demux` | Extract AAC access units from M4A/MP4 files. |
| `mp4-decoder` | Decode M4A/MP4 AAC with FDK AAC. |

## Test

```sh
cargo test -p soundkit-aac --all-features
```

The tests cover ADTS streaming, MP4 demuxing, AAC-LC encoding, and HE-AAC
fallback compatibility.
