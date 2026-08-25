use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use soundkit::audio_packet::{Decoder, Encoder};
use soundkit_opus::{OpusDecoder, OpusEncoder};

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

static COUNT_ALLOCATIONS: AtomicBool = AtomicBool::new(false);
static ALLOCATION_COUNT: AtomicUsize = AtomicUsize::new(0);

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
fn steady_state_soundkit_opus_reuses_caller_storage() {
    const SAMPLE_RATE: u32 = 48_000;
    const CHANNELS: u32 = 2;
    const FRAME_SIZE: u32 = 960;

    let mut encoder = OpusEncoder::new(SAMPLE_RATE, 16, CHANNELS, FRAME_SIZE, 128_000);
    encoder.init().unwrap();
    let pcm = (0..FRAME_SIZE as usize)
        .flat_map(|frame| {
            let phase = frame as f32 * 440.0 * std::f32::consts::TAU / SAMPLE_RATE as f32;
            let sample = (phase.sin() * i16::MAX as f32 * 0.25) as i16;
            [sample, sample]
        })
        .collect::<Vec<_>>();
    let mut packet = vec![0u8; 4_096];
    let packet_len = encoder.encode_i16(&pcm, &mut packet).unwrap();

    let mut decoder = OpusDecoder::new_celt_only(SAMPLE_RATE as usize, CHANNELS as usize).unwrap();
    decoder.init().unwrap();
    let mut output = vec![0.0f32; 5_760 * CHANNELS as usize];
    decoder
        .decode_f32(&packet[..packet_len], &mut output, false)
        .unwrap();

    ALLOCATION_COUNT.store(0, Ordering::Relaxed);
    COUNT_ALLOCATIONS.store(true, Ordering::Relaxed);
    for _ in 0..100 {
        assert_eq!(
            decoder
                .decode_f32(&packet[..packet_len], &mut output, false)
                .unwrap(),
            FRAME_SIZE as usize,
        );
    }
    COUNT_ALLOCATIONS.store(false, Ordering::Relaxed);

    assert_eq!(
        ALLOCATION_COUNT.load(Ordering::Relaxed),
        0,
        "steady-state f32 Opus decode allocated"
    );

    let mut output_i16 = vec![0i16; 5_760 * CHANNELS as usize];
    decoder.reset().unwrap();
    decoder
        .decode_i16(&packet[..packet_len], &mut output_i16, false)
        .unwrap();

    ALLOCATION_COUNT.store(0, Ordering::Relaxed);
    COUNT_ALLOCATIONS.store(true, Ordering::Relaxed);
    for _ in 0..100 {
        assert_eq!(
            decoder
                .decode_i16(&packet[..packet_len], &mut output_i16, false)
                .unwrap(),
            FRAME_SIZE as usize,
        );
    }
    COUNT_ALLOCATIONS.store(false, Ordering::Relaxed);

    assert_eq!(
        ALLOCATION_COUNT.load(Ordering::Relaxed),
        0,
        "steady-state i16 Opus decode allocated"
    );

    encoder.encode_i16(&pcm, &mut packet).unwrap();
    ALLOCATION_COUNT.store(0, Ordering::Relaxed);
    COUNT_ALLOCATIONS.store(true, Ordering::Relaxed);
    for _ in 0..100 {
        assert!(encoder.encode_i16(&pcm, &mut packet).unwrap() > 0);
    }
    COUNT_ALLOCATIONS.store(false, Ordering::Relaxed);

    assert_eq!(
        ALLOCATION_COUNT.load(Ordering::Relaxed),
        0,
        "steady-state Opus encode allocated"
    );

    let pcm_i24 = pcm
        .iter()
        .map(|&sample| i32::from(sample) << 8)
        .collect::<Vec<_>>();
    let mut encoder_i24 = OpusEncoder::new(SAMPLE_RATE, 24, CHANNELS, FRAME_SIZE, 128_000);
    encoder_i24.init().unwrap();
    let packet_len_i24 = encoder_i24.encode_i32(&pcm_i24, &mut packet).unwrap();
    encoder_i24.encode_i32(&pcm_i24, &mut packet).unwrap();

    ALLOCATION_COUNT.store(0, Ordering::Relaxed);
    COUNT_ALLOCATIONS.store(true, Ordering::Relaxed);
    for _ in 0..100 {
        assert!(encoder_i24.encode_i32(&pcm_i24, &mut packet).unwrap() > 0);
    }
    COUNT_ALLOCATIONS.store(false, Ordering::Relaxed);

    assert_eq!(
        ALLOCATION_COUNT.load(Ordering::Relaxed),
        0,
        "steady-state 24-bit Opus encode allocated"
    );

    let mut decoder_i24 =
        OpusDecoder::new_celt_only(SAMPLE_RATE as usize, CHANNELS as usize).unwrap();
    let mut output_i24 = vec![0i32; 5_760 * CHANNELS as usize];
    decoder_i24
        .decode_i32(&packet[..packet_len_i24], &mut output_i24, false)
        .unwrap();

    ALLOCATION_COUNT.store(0, Ordering::Relaxed);
    COUNT_ALLOCATIONS.store(true, Ordering::Relaxed);
    for _ in 0..100 {
        assert_eq!(
            decoder_i24
                .decode_i32(&packet[..packet_len_i24], &mut output_i24, false)
                .unwrap(),
            FRAME_SIZE as usize,
        );
    }
    COUNT_ALLOCATIONS.store(false, Ordering::Relaxed);

    assert_eq!(
        ALLOCATION_COUNT.load(Ordering::Relaxed),
        0,
        "steady-state 24-bit Opus decode allocated"
    );
}
