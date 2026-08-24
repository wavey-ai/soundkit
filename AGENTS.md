# SoundKit CPU performance host

Use `yl-encodec-1` for CPU-only media codec performance tests.

The Google Cloud project is `steadfast-slate-498623-r2`.
The zone is `europe-west2-b`.
The machine type is `c4-highcpu-4`.
The CPU platform is Intel Emerald Rapids with four virtual CPUs.
The operating system is Debian 12 x86-64.

Start the host before a test:

```sh
gcloud compute instances start yl-encodec-1 --zone=europe-west2-b
```

Connect after the start operation completes:

```sh
ssh soundkit-perf
```

Use this command if the SSH alias is not available:

```sh
gcloud compute ssh yl-encodec-1 --zone=europe-west2-b
```

The host includes Rust, Cargo, Git, Clang, CMake, FFmpeg, FLAC, libFLAC headers, Python, and Linux `perf`.

Record these versions with each benchmark result:

```sh
lscpu
rustc --version
cargo --version
ffmpeg -version | head -1
flac --version
```

Use warm-up runs before measured runs.
Alternate SoundKit and reference runs to reduce thermal and scheduler bias.
Keep input files and output checks identical for each comparison.
Do not use GPU or hardware-accelerated codec paths for CPU comparisons.

Stop the host immediately after the test:

```sh
gcloud compute instances stop yl-encodec-1 --zone=europe-west2-b
```

A stopped host does not incur virtual CPU or memory charges.
The persistent disk and reserved external IP address continue to incur charges.
