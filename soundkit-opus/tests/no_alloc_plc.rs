//! Packet-loss concealment must not allocate once the decoder has warmed up:
//! at 1 ms frames a real-time thread may conceal a thousand frames a second.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use soundkit_opus::{Decoder, Encoder};

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
#[cfg_attr(
    feature = "trace",
    ignore = "the trace feature records every decoded frame, which allocates"
)]
fn steady_state_concealment_does_not_allocate() {
    const FRAME: usize = 48;
    for channels in [1usize, 2] {
        let mut encoder = Encoder::new_custom(48_000, channels, FRAME).unwrap();
        encoder.set_bitrate(128_000 * channels as i32).unwrap();
        let packets: Vec<Vec<u8>> = (0..64)
            .map(|frame| {
                let pcm: Vec<f32> = (0..FRAME)
                    .flat_map(|i| {
                        let t = (frame * FRAME + i) as f32 / 48_000.0;
                        let v = 0.25 * (t * 220.0 * std::f32::consts::TAU).sin();
                        std::iter::repeat_n(v, channels)
                    })
                    .collect();
                encoder.encode_f32_vec(&pcm, FRAME).unwrap()
            })
            .collect();
        let mut decoder = Decoder::new_custom(48_000, channels, FRAME).unwrap();
        let mut output = vec![0.0f32; FRAME * channels];
        // Good packets, a short loss (pitch concealment), more good packets,
        // then a long loss that turns into noise concealment.
        let mut run = |decoder: &mut Decoder| {
            for (i, packet) in packets.iter().enumerate() {
                let lost = (20..22).contains(&i) || i >= 30;
                let packet: &[u8] = if lost { &[] } else { packet };
                assert_eq!(
                    decoder
                        .decode_f32_into_slice(packet, false, &mut output)
                        .unwrap(),
                    FRAME
                );
            }
        };
        run(&mut decoder);
        ALLOCATION_COUNT.store(0, Ordering::Relaxed);
        COUNT_ALLOCATIONS.store(true, Ordering::Relaxed);
        run(&mut decoder);
        COUNT_ALLOCATIONS.store(false, Ordering::Relaxed);
        assert_eq!(
            ALLOCATION_COUNT.load(Ordering::Relaxed),
            0,
            "{channels}-channel concealment allocated"
        );
    }
}
