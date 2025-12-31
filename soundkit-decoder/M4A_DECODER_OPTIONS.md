# M4A/AAC Decoder Strategy Comparison

## Context

**Boxer** (`wavey-ai/boxer`): Takes raw AAC frames (with ADTS headers) → Boxes into fMP4 for streaming
**Inverse**: Take M4A container → Extract raw AAC → Decode with FDK

## Option 1: Standard `mp4` Crate (0.14.0)

### API
```rust
use mp4::{Mp4Reader, Mp4Sample};

let reader = Mp4Reader::read_header(file, size)?;
let track_id = /* find audio track */;
let sample_count = reader.sample_count(track_id)?;

for sample_id in 1..=sample_count {
    let sample: Mp4Sample = reader.read_sample(track_id, sample_id)?;
    // sample.bytes contains raw AAC (no ADTS header)
    // sample.duration, sample.start_time available
}
```

### Pros
- ✅ Mature, well-documented (alfg/mp4-rust)
- ✅ Simple API: `read_sample()` returns `Mp4Sample` with `.bytes` field
- ✅ Standard crate (0.14.0 on crates.io)
- ✅ Handles full MP4/M4A spec
- ✅ Works with streaming (BufReader)

### Cons
- ⚠️ AAC samples are **raw** (no ADTS headers)
- ⚠️ Need to configure FDK for raw AAC or add ADTS headers
- ⚠️ Additional dependency (~50KB)

### Implementation Effort
**Medium**: Need to handle raw AAC → FDK conversion

---

## Option 2: Inverse of Boxer Strategy

Looking at `boxer/src/fmp4.rs` lines 195-222:
- Boxer **reads ADTS** (line 203: `AdtsHeader::read_from()`)
- Boxer **strips ADTS** (line 219: `extract_aac_data()` removes headers)
- Stores raw AAC in MP4

### Inverse Process
1. Read M4A with `mp4` crate
2. Extract raw AAC samples
3. **Re-add ADTS headers** using info from MP4 metadata
4. Feed ADTS-wrapped AAC to FDK decoder

### Pros
- ✅ Leverages existing ecosystem (`access-unit` for ADTS)
- ✅ FDK already configured for ADTS transport
- ✅ Symmetrical with encoding path

### Cons
- ⚠️ Extra work: Add ADTS headers that were stripped during encoding
- ⚠️ More complex: MP4 metadata → ADTS header conversion
- ⚠️ Still needs `mp4` crate for demuxing

### Implementation Effort
**High**: Need MP4 demux + ADTS header construction

---

## Option 3: Reconfigure FDK for Raw AAC

### API Change
```rust
// Current: DecoderTransport::Adts
let decoder = AacLibDecoder::new(DecoderTransport::Adts);

// Change to: DecoderTransport::Raw
let decoder = AacLibDecoder::new(DecoderTransport::Raw);
```

### Pros
- ✅ Simplest: No ADTS header handling
- ✅ Direct: MP4 raw AAC → FDK raw decoder
- ✅ Efficient: No extra processing

### Cons
- ⚠️ Need codec-specific config (SamplingFrequency, ChannelConfiguration)
- ⚠️ May require ASC (AudioSpecificConfig) from MP4
- ⚠️ Changes current decoder setup

### Implementation Effort
**Low to Medium**: Change FDK transport + extract ASC from MP4

---

## Recommendation: **Option 3 (FDK Raw Mode)**

### Rationale
1. **Most efficient**: Direct path from MP4 → FDK without reconstruction
2. **Leverages FDK**: Best AAC decoder quality
3. **Clean separation**: `mp4` for container, `fdk-aac` for codec
4. **Minimal code**: No ADTS header manipulation

### Implementation Plan

```rust
// In soundkit-aac, add a new decoder variant
pub struct AacDecoderMp4 {
    mp4_reader: Option<Mp4Reader<BufReader<Cursor<Vec<u8>>>>>,
    fdk_decoder: AacLibDecoder, // Configured for Raw transport
    track_id: Option<u32>,
    current_sample: u32,
    sample_count: u32,
}

impl AacDecoderMp4 {
    pub fn new() -> Self {
        // FDK configured for Raw (not ADTS)
        let decoder = AacLibDecoder::new(DecoderTransport::Raw);
        // ... initialize with MP4 parsing
    }
}
```

### Key Steps
1. Add `mp4 = "0.14.0"` to `soundkit-aac/Cargo.toml`
2. Create `AacDecoderMp4` that wraps FDK in Raw mode
3. Parse M4A on first chunk, extract AudioSpecificConfig
4. Configure FDK with ASC
5. Read samples, feed raw AAC to FDK
6. Update `soundkit-decoder` to detect M4A vs ADTS and route accordingly

### Complexity
- **Lines of code**: ~200-300
- **Dependencies**: +1 (`mp4` crate)
- **Risk**: Low (well-tested components)

---

## Performance Comparison

| Approach | Demux | Process | Decode | Total |
|----------|-------|---------|--------|-------|
| Option 1 (mp4 crate) | mp4 | - | FDK raw | Fast |
| Option 2 (Add ADTS) | mp4 | +ADTS headers | FDK ADTS | Medium |
| Option 3 (FDK raw) | mp4 | - | FDK raw | **Fastest** |
| Symphonia | Symphonia | - | Symphonia | Fails ("aac too complex") |

**Winner: Option 3** - Direct path, no intermediate processing
