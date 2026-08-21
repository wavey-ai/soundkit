use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use soundkit_fmp4::{rebox_soundkit_v2, ReboxOptions};
use std::env;
use std::fs;
use std::path::PathBuf;

fn benchmark_rebox(c: &mut Criterion) {
    let fixture_dir = env::var_os("SOUNDKIT_FMP4_FIXTURE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .expect("soundkit-fmp4 must be in the Soundkit workspace")
                .join("target/fixtures/soundkit-fmp4")
        });
    let fixtures = [
        ("opus", fixture_dir.join("westside-4s-opus.skv2")),
        ("flac", fixture_dir.join("westside-4s-flac.skv2")),
    ];
    let mut group = c.benchmark_group("soundkit_v2_to_fmp4_4s_stereo");

    for (codec, path) in fixtures {
        let bytes = fs::read(&path).unwrap_or_else(|error| {
            panic!(
                "read {}: {error}; run the generate_fixture example first",
                path.display()
            )
        });
        group.throughput(Throughput::Bytes(bytes.len() as u64));
        group.bench_with_input(BenchmarkId::new("rebox", codec), &bytes, |b, input| {
            b.iter(|| {
                let output = rebox_soundkit_v2(
                    black_box(input),
                    ReboxOptions {
                        sequence: 1,
                        include_init: true,
                        start_pts_ms: None,
                    },
                )
                .expect("Westside Soundkit fixture must rebox");
                black_box((output.init, output.media));
            });
        });
    }
    group.finish();
}

criterion_group!(benches, benchmark_rebox);
criterion_main!(benches);
