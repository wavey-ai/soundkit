# soundkit-fmp4

`soundkit-fmp4` validates and reboxes unencrypted Soundkit v2 FLAC and Opus
packet streams into fragmented MP4 without decoding or re-encoding audio.
RaptorQ recovery and AEP1 group extraction remain transport responsibilities.

Generate the ignored Westside benchmark fixture:

```sh
cargo run -p soundkit-fmp4 --example generate_fixture -- \
  ../encodec-rs/testdata/westside_4s_48khz_stereo.wav \
  target/fixtures/soundkit-fmp4
```

Benchmark the four-second stereo Opus and FLAC v2 streams:

```sh
cargo bench -p soundkit-fmp4 --bench rebox
```
