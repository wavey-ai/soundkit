use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use soundkit_aac_lc::{AacLcDecoder, AacLcError, Result};

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

static COUNT_ALLOCATIONS: AtomicBool = AtomicBool::new(false);
static ALLOCATION_COUNT: AtomicUsize = AtomicUsize::new(0);

const FIXTURE: &[u8] =
    include_bytes!("../../golden/aac/WESTSIDE_MIX_4_CONFIRMATION_130323_256k.aac");

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        count_allocation(ptr);
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc_zeroed(layout);
        count_allocation(ptr);
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let ptr = System.realloc(ptr, layout, new_size);
        count_allocation(ptr);
        ptr
    }
}

fn count_allocation(ptr: *mut u8) {
    if !ptr.is_null() && COUNT_ALLOCATIONS.load(Ordering::Relaxed) {
        ALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);
    }
}

#[test]
fn steady_state_fixture_decode_does_not_allocate() -> Result<()> {
    let frames = parse_adts_frames(FIXTURE)?;
    let first = frames
        .first()
        .ok_or(AacLcError::InvalidBitstream("fixture has no ADTS frames"))?;
    let asc = first.audio_specific_config();
    let mut decoder = AacLcDecoder::from_audio_specific_config(&asc)?;

    for frame in &frames {
        decoder.decode_access_unit(frame.raw)?;
    }

    ALLOCATION_COUNT.store(0, Ordering::Relaxed);
    COUNT_ALLOCATIONS.store(true, Ordering::Relaxed);
    for frame in &frames {
        decoder.decode_access_unit(frame.raw)?;
    }
    COUNT_ALLOCATIONS.store(false, Ordering::Relaxed);

    assert_eq!(
        ALLOCATION_COUNT.load(Ordering::Relaxed),
        0,
        "steady-state AAC-LC fixture decode allocated"
    );
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct AdtsFrame<'a> {
    raw: &'a [u8],
    audio_object_type: u8,
    sample_rate_index: u8,
    channels: u8,
}

impl AdtsFrame<'_> {
    fn audio_specific_config(self) -> [u8; 2] {
        [
            (self.audio_object_type << 3) | (self.sample_rate_index >> 1),
            ((self.sample_rate_index & 1) << 7) | (self.channels << 3),
        ]
    }
}

fn parse_adts_frames(data: &[u8]) -> Result<Vec<AdtsFrame<'_>>> {
    let mut frames = Vec::new();
    let mut offset = 0usize;

    while offset + 7 <= data.len() {
        while offset + 7 <= data.len()
            && !(data[offset] == 0xff && (data[offset + 1] & 0xf0) == 0xf0)
        {
            offset += 1;
        }
        if offset + 7 > data.len() {
            break;
        }

        let protection_absent = (data[offset + 1] & 0x01) != 0;
        let header_len = if protection_absent { 7 } else { 9 };
        let audio_object_type = ((data[offset + 2] & 0xc0) >> 6) + 1;
        let sample_rate_index = (data[offset + 2] & 0x3c) >> 2;
        let channels = ((data[offset + 2] & 0x01) << 2) | ((data[offset + 3] & 0xc0) >> 6);
        let frame_len = (((data[offset + 3] & 0x03) as usize) << 11)
            | ((data[offset + 4] as usize) << 3)
            | (((data[offset + 5] & 0xe0) as usize) >> 5);

        if frame_len <= header_len {
            return Err(AacLcError::InvalidBitstream("invalid ADTS frame length"));
        }
        if offset + frame_len > data.len() {
            return Err(AacLcError::UnexpectedEof {
                requested_bits: 8,
                remaining_bits: (data.len() - offset) * 8,
            });
        }

        frames.push(AdtsFrame {
            raw: &data[offset + header_len..offset + frame_len],
            audio_object_type,
            sample_rate_index,
            channels,
        });
        offset += frame_len;
    }

    Ok(frames)
}
