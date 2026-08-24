# TODO

This roadmap targets the best Rust CELT implementation with a safe public codec
crate. It does not target one libopus release or encoder byte identity.

Use stable libopus and upstream main as evidence. Preserve packet
interoperability. Let real-audio quality tests decide between valid encoders.

## Current 48 kHz Baseline

- Keep `#![forbid(unsafe_code)]` intact in the public codec crate.
- Keep audited unsafe kernels in the private kernel crate behind checked safe
  functions.
- Support mono and stereo 48 kHz CELT-only fullband raw packets.
- Focus encoder quality and performance work on 192, 256, and 320 kb/s stereo;
  retain lower bitrates as regression coverage.
- Support 2.5, 5, 10, and 20 ms frames.
- Support CBR, constrained VBR, and exact compressed-frame-byte controls.
- Decode one CELT frame per packet through reusable i16 and f32 output paths.
- Maintain C and Rust packet probes, cross-decode tools, and quality benchmarks.

## Current SOTA Checkpoint

- The encoder now cancels pitch filtering when the filter does not help.
- Strong transients also disable discontinuous pitch filtering.
- LPC tone detection now controls pitch selection, TF analysis, and allocation.
- Stereo band coding now handles one silent channel without unstable theta input.
- A one-second mixed matrix contains 72 CBR and constrained-VBR rows.
- Against libopus 1.6.1, its mean absolute aligned-SNR gap is `0.032 dB`.
- Its maximum aligned-SNR gap is `0.18 dB`.
- At 5 ms and 48 kb/s, Rust CBR is `13.15 dB`; C is `13.28 dB`.
- At 5 ms and 48 kb/s, Rust VBR is `13.40 dB`; C is `13.32 dB`.
- The first 5 ms, 128 kb/s pure-tone packet matches C exactly.
- Later pure-tone packets diverge in the spectral payload.

Synthetic SNR does not measure transparency. It can reward a filter that
current upstream code correctly rejects.

## P0: Establish a Quality Gate

- Build a licensed corpus with music, speech, transients, ambience, and tonal signals.
- Gate the 192, 256, and 320 kb/s stereo transparency candidates first, while
  retaining representative low and medium rates as regression coverage.
- Add a reproducible perceptual metric and delay-aligned diagnostic metrics.
- Add blinded listening tests for changes that affect encoder decisions.
- Keep malformed-packet, cross-decode, and exact-budget tests as hard gates.
- Store benchmark metadata, upstream commits, settings, and packet checksums.

## P1: Finish the 48 kHz CELT Encoder

- Trace the later pure-tone spectral split through allocation and PVQ.
- Investigate the largest mixed-fixture gaps only when the corpus confirms them.
- Compare the constrained-VBR blend with upstream's `0.67` value.
- Change that blend only when the corpus shows a quality gain.
- Extend all deterministic cases beyond one second.
- Keep exact packet comparisons as diagnostic evidence, not the quality target.
- Port later ordinary-CELT improvements from upstream main when they apply.

## P2: Complete CELT Decode Coverage

- Decode all valid packet frame layouts.
- Support narrowband through fullband CELT packets.
- Support the public 8, 12, 16, 24, and 48 kHz API rates.
- Implement packet-loss concealment and the public FEC path.
- Complete decoder state queries and reset behavior.
- Validate with upstream tests, fuzzing, and symmetric cross-decode runs.

## P3: Complete the 48 kHz Product Surface

- Add Ogg Opus framing, pre-skip, granule handling, and stream finalization.
- Complete the public encoder controls that apply to CELT.
- Reject unsupported SILK, hybrid, DRED, and FEC modes explicitly.
- Add fuzz and property tests for packet and state boundaries.
- Remove avoidable allocations after correctness and quality gates pass.

## P4: Add 96 kHz/QEXT and 24-Bit I/O

- Keep ordinary 48 kHz Opus as the default interoperable path.
- The encoder and decoder now accept signed 24-bit PCM in sign-extended `i32`.
- Preserve all 24-bit values at the `f32` encoder boundary and quantize decoded
  `f32` directly to 24-bit output.
- Add upstream QEXT 96 kHz behavior after the 48 kHz quality baseline is stable.
- Test QEXT interoperability against the matching upstream implementation.
- Measure 96/24 where 48 kHz bandwidth or 16-bit I/O limits could matter.
- Do not market 96/24 Opus as a master or FLAC replacement.

## P5: Add Multistream and Multichannel

- Add multistream encoder and decoder state.
- Support channel mapping families and coupled-stream routing.
- Add projection support only after the base multistream path is stable.
- Test mixed mono, coupled stereo, silent channels, and mapping failures.

Multistream is separate from core stereo quality. It can follow 96/24 or proceed
independently when a product requires it.

## P6: Add Other Opus Modes

- Port SILK common primitives and tests.
- Port SILK decoding, PLC, CNG, LBRR, stereo prediction, and resampling.
- Port hybrid decoding and transition redundancy.
- Port SILK encoding and mode selection.
- Port DRED and later optional extensions as separate work.
