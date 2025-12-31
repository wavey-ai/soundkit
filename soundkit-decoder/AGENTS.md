# Decoder Pipeline Architecture

## Overview

The `soundkit-decoder` crate provides a unified streaming decoder pipeline that automatically detects audio format and decodes to PCM. All codecs are handled through a single `StreamingDecoder` trait interface.

## Supported Formats

| Format | Container | Status |
|--------|-----------|--------|
| MP3 | Raw | ✓ Working |
| FLAC | Raw | ✓ Working |
| AAC | M4A/MP4 | ✓ Working |
| Opus | Raw | ✓ Working |
| Opus | Ogg | ✓ Working |
| Opus | WebM | ✓ Working |

## Architecture

### Unified StreamingDecoder Trait

All decoders implement a common interface:

```rust
trait StreamingDecoder {
    fn process(&mut self, chunk: &[u8]) -> Result<Vec<AudioData>, String>;
    fn flush(&mut self) -> Result<Vec<AudioData>, String>;
}
```

The pipeline doesn't need to know codec implementation details - it just calls `process()` and `flush()`.

### Pipeline Flow

```
Input Thread              Worker Thread                    Output Thread
     |                         |                                |
     | send(Bytes)             |                                |
     |---> input_rx ---------> |                                |
     |                         |                                |
     |                   Detecting Phase                        |
     |                   (buffer MIN_DETECTION_BYTES)           |
     |                         |                                |
     |                   detect_and_init_decoder()              |
     |                         |                                |
     |                   Decoding Phase                         |
     |                   decoder.process(chunk)                 |
     |                         |                                |
     |                         |---> output_tx ----------------> try_recv()
     |                         |                                |
     | send(Bytes::new())      |                                |
     |---> EOF signal -------> |                                |
     |                   decoder.flush()                        |
     |                         |                                |
```

### Key Design Decisions

#### Bytes vs Vec<u8>

The pipeline uses `Bytes` for input because:
- Zero-copy slicing for chunked streaming (`data.slice(start..end)`)
- Callers can pass network data directly without copying
- Reference counted for cheap sharing

#### Backpressure Handling

The `push_output` function blocks with retry when the output buffer is full, preventing silent data loss:

```rust
fn push_output(output_tx: &mut Producer<DecodeOutput>, output: DecodeOutput) {
    loop {
        match output_tx.push(item) {
            Ok(_) => return,
            Err(rtrb::PushError::Full(returned_item)) => {
                item = returned_item;
                std::thread::sleep(Duration::from_micros(100));
            }
        }
    }
}
```

#### Decoder-Specific Handling

Two internal APIs exist but are abstracted by helper functions:

1. **Decoder trait** (MP3, AAC, FLAC): `decode_i16/i32(input, output) -> samples`
   - Uses `decode_i16_with_drain` or `decode_with_drain` helpers

2. **add() API** (Opus, OggOpus, WebM): `add(input) -> Option<AudioData>`
   - Uses `process_with_add_api` helper

Both are wrapped by the `StreamingDecoder` trait implementation on `FormatDecoder`.

## Testing

```bash
# Run all decoder tests
cargo test -p soundkit-decoder

# Run benchmark (requires testdata with all formats)
cargo test -p soundkit-decoder bench_all_formats -- --ignored --nocapture

# Generate golden files for comparison
cargo test -p soundkit-decoder test_decode_all_formats_to_s16le_16k_mono -- --nocapture
```

## File Structure

```
soundkit-decoder/
├── src/lib.rs          # Main pipeline implementation
├── testdata/
│   ├── aac/            # M4A test files
│   ├── flac/           # FLAC test files
│   ├── mp3/            # MP3 test files
│   ├── opus/           # Raw Opus test files
│   ├── ogg_opus/       # Ogg Opus test files
│   ├── webm/           # WebM test files
│   └── golden/         # Reference outputs (16kHz mono s16le)
```
