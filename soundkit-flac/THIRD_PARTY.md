# Third-party provenance

soundkit-flac vendors a complete FLAC codec derived from two established
Apache-2.0 Rust implementations instead of rewriting the format from
scratch.

## flacenc-rs

- Upstream: <https://github.com/yotarok/flacenc-rs>
- Role: encoder foundation
- License: Apache-2.0
- Original documentation: [`docs/FLACENC.md`](https://github.com/yotarok/flacenc-rs)

Copyright notices in the original files are retained.

## Claxon

- Upstream: <https://github.com/ruuda/claxon>
- Role: decoder foundation
- License: Apache-2.0
- Original license: [`LICENSE-CLAXON`](LICENSE-CLAXON)

The decoder source files retain their original Claxon copyright and license
headers.

Both foundations arrived through the intermediate Wavey FLAC fork of this
code. Performance work on hot kernels follows clean-room practice only where
an upstream license requires it, such as FFmpeg (LGPL). Apache-2.0 sources
are modified directly under their own license terms.
