#!/usr/bin/env python3
"""Benchmark pure-Rust CELT raw frames against upstream opus_demo.

The upstream Opus repository keeps conformance vectors as a separate download,
so this script uses either a supplied 48 kHz stereo PCM16 WAV or a deterministic
synthetic fixture. It writes encoded golden streams, decoded WAVs, and metrics
manifests when output directories are supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import statistics
import struct
import subprocess
import time
import wave
from array import array
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = Path("/tmp/libopus-rs-celt-bench-target")
DEFAULT_WORK = Path("/tmp/libopus-rs-celt-bench")
DEFAULT_OPUS_DEMO = Path("/Users/jamie/wavey.ai/opus/opus_demo")
FRAME_SIZES = [120, 240, 480, 960]
BITRATES = [48_000, 96_000, 128_000]
CHANNELS = 2
SAMPLE_RATE = 48_000


def run(cmd: list[str], cwd: Path = ROOT) -> None:
    proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}")


def timed(cmd: list[str], repeats: int, cwd: Path = ROOT) -> float:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        run(cmd, cwd)
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def clamp_i16(value: float) -> int:
    return max(-32768, min(32767, int(round(value * 32767.0))))


def generate_fixture(wav_path: Path, raw_path: Path, seconds: float) -> None:
    frames = int(SAMPLE_RATE * seconds)
    samples: list[int] = []
    noise = 0x1234_5678
    for i in range(frames):
        t = i / SAMPLE_RATE
        noise = (1664525 * noise + 1013904223) & 0xFFFF_FFFF
        n = ((noise >> 9) / float(1 << 23)) - 1.0
        transient = 0.35 * math.exp(-900.0 * (t - 1.37) ** 2)
        left = (
            0.29 * math.sin(2.0 * math.pi * 261.63 * t)
            + 0.17 * math.sin(2.0 * math.pi * 659.25 * t + 0.2)
            + 0.05 * math.sin(2.0 * math.pi * 4210.0 * t)
            + 0.015 * n
            + transient
        )
        right = (
            0.25 * math.sin(2.0 * math.pi * 329.63 * t + 0.4)
            - 0.13 * math.sin(2.0 * math.pi * 880.0 * t)
            + 0.05 * math.sin(2.0 * math.pi * 3910.0 * t + 0.7)
            - 0.012 * n
            - 0.8 * transient
        )
        samples.extend([clamp_i16(left), clamp_i16(right)])

    wav_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(struct.pack("<" + "h" * len(samples), *samples))
    raw_path.write_bytes(struct.pack("<" + "h" * len(samples), *samples))


def read_wav_i16(path: Path) -> tuple[int, int, list[int]]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        width = wav.getsampwidth()
        data = wav.readframes(wav.getnframes())
    if width != 2:
        raise ValueError(f"expected PCM16 WAV: {path}")
    samples = array("h")
    samples.frombytes(data)
    return sample_rate, channels, list(samples)


def read_raw_i16(path: Path) -> list[int]:
    data = path.read_bytes()
    samples = array("h")
    samples.frombytes(data)
    return list(samples)


def write_wav_i16(path: Path, samples: list[int]) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(struct.pack("<" + "h" * len(samples), *samples))


def deinterleave(samples: list[int], channels: int) -> list[list[float]]:
    return [[samples[i] / 32768.0 for i in range(c, len(samples), channels)] for c in range(channels)]


def corr_at_lag(ref: list[float], got: list[float], lag: int, stride: int = 16, max_points: int = 3_000) -> float:
    if lag >= 0:
        ref_start = 0
        got_start = lag
        available = min(len(ref), len(got) - lag)
    else:
        ref_start = -lag
        got_start = 0
        available = min(len(ref) + lag, len(got))
    if available <= 0:
        return 0.0

    rr = 0.0
    gg = 0.0
    rg = 0.0
    used = 0
    for offset in range(0, available, stride):
        x = ref[ref_start + offset]
        y = got[got_start + offset]
        rr += x * x
        gg += y * y
        rg += x * y
        used += 1
        if used >= max_points:
            break

    if used < 32:
        return 0.0
    if rr == 0.0 or gg == 0.0:
        return 0.0
    return rg / math.sqrt(rr * gg)


def channel_metrics(ref: list[float], got: list[float], expected_lag: int = 120, search_radius: int = 16) -> tuple[float, float, int]:
    # Restricted-lowdelay CELT has 120 samples of lookahead at 48 kHz. A broad
    # correlation search can lock onto later periodic music content and make the
    # SNR estimate meaningless, so only allow a small alignment tolerance.
    min_lag = max(0, expected_lag - search_radius)
    max_lag = expected_lag + search_radius
    lag = max(
        range(min_lag, max_lag + 1),
        key=lambda candidate: corr_at_lag(ref, got, candidate),
    )
    if lag >= 0:
        n = min(len(ref), len(got) - lag)
        r = ref[:n]
        g = got[lag : lag + n]
    else:
        n = min(len(ref) + lag, len(got))
        r = ref[-lag : -lag + n]
        g = got[:n]
    skip = min(960, max(0, len(r) // 4))
    r = r[skip:]
    g = g[skip:]
    signal = sum(x * x for x in r)
    noise = sum((x - y) * (x - y) for x, y in zip(r, g))
    corr = corr_at_lag(ref, got, lag)
    snr = 99.0 if noise == 0.0 else 10.0 * math.log10(signal / noise)
    return corr, snr, lag


def quality_metrics(input_wav: Path, decoded_wav: Path) -> tuple[float, float, int]:
    _, channels, ref_samples = read_wav_i16(input_wav)
    _, got_channels, got_samples = read_wav_i16(decoded_wav)
    if channels != got_channels:
        raise ValueError("channel mismatch")
    ref = deinterleave(ref_samples, channels)
    got = deinterleave(got_samples, channels)
    metrics = [channel_metrics(ref[c], got[c]) for c in range(channels)]
    return (
        statistics.mean(item[0] for item in metrics),
        statistics.mean(item[1] for item in metrics),
        round(statistics.mean(item[2] for item in metrics)),
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_ms(frame_size: int) -> str:
    value = frame_size * 1000.0 / SAMPLE_RATE
    return f"{value:g}"


def build_rust_example(target_dir: Path) -> Path:
    run(
        [
            "cargo",
            "build",
            "--release",
            "--target-dir",
            str(target_dir),
            "--example",
            "wav_celt",
        ]
    )
    return target_dir / "release" / "examples" / "wav_celt"


def prepare_input(args: argparse.Namespace) -> tuple[Path, Path]:
    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    input_raw = work / "input.s16le"
    if args.input_wav:
        input_wav = args.input_wav
        sample_rate, channels, samples = read_wav_i16(input_wav)
        if sample_rate != SAMPLE_RATE or channels != CHANNELS:
            raise SystemExit("input WAV must be 48 kHz stereo PCM16")
        input_raw.write_bytes(struct.pack("<" + "h" * len(samples), *samples))
    else:
        input_wav = work / "synthetic_48k_stereo.wav"
        generate_fixture(input_wav, input_raw, args.seconds)
    return input_wav, input_raw


def benchmark(args: argparse.Namespace) -> tuple[Path, list[dict[str, object]]]:
    input_wav, input_raw = prepare_input(args)
    work = args.work_dir

    rust_bin = build_rust_example(args.target_dir)
    opus_demo = args.opus_demo
    if not opus_demo.exists():
        raise SystemExit(f"opus_demo not found: {opus_demo}")

    rows = []
    for frame_size in FRAME_SIZES:
        ms = frame_ms(frame_size)
        for bitrate in BITRATES:
            label = f"{ms}ms_{bitrate // 1000}k"
            rust_stream = work / f"rust_{label}.lors"
            rust_wav = work / f"rust_{label}.decoded.wav"
            c_stream = work / f"c_{label}.bit"
            c_raw = work / f"c_{label}.decoded.s16le"
            c_wav = work / f"c_{label}.decoded.wav"

            rust_encode = timed(
                [
                    str(rust_bin),
                    "encode",
                    "--frame-size",
                    str(frame_size),
                    "--bitrate",
                    str(bitrate),
                    str(input_wav),
                    str(rust_stream),
                ],
                args.repeats,
            )
            rust_decode = timed([str(rust_bin), "decode", str(rust_stream), str(rust_wav)], args.repeats)
            rust_corr, rust_snr, rust_lag = quality_metrics(input_wav, rust_wav)

            c_encode = timed(
                [
                    str(opus_demo),
                    "-e",
                    "restricted-lowdelay",
                    str(SAMPLE_RATE),
                    str(CHANNELS),
                    str(bitrate),
                    "-cbr",
                    "-bandwidth",
                    "FB",
                    "-framesize",
                    ms,
                    str(input_raw),
                    str(c_stream),
                ],
                args.repeats,
            )
            c_decode = timed([str(opus_demo), "-d", str(SAMPLE_RATE), str(CHANNELS), str(c_stream), str(c_raw)], args.repeats)
            write_wav_i16(c_wav, read_raw_i16(c_raw))
            c_corr, c_snr, c_lag = quality_metrics(input_wav, c_wav)

            rows.append(
                {
                    "frame_ms": ms,
                    "frame_size": frame_size,
                    "bitrate": bitrate,
                    "rust_encode_ms": rust_encode,
                    "rust_decode_ms": rust_decode,
                    "c_encode_ms": c_encode,
                    "c_decode_ms": c_decode,
                    "rust_bytes": rust_stream.stat().st_size,
                    "c_bytes": c_stream.stat().st_size,
                    "rust_corr": rust_corr,
                    "rust_snr_db": rust_snr,
                    "rust_lag": rust_lag,
                    "c_corr": c_corr,
                    "c_snr_db": c_snr,
                    "c_lag": c_lag,
                    "rust_stream": rust_stream,
                    "rust_decoded": rust_wav,
                    "c_stream": c_stream,
                    "c_decoded": c_wav,
                }
            )
    return input_wav, rows


def write_goldens(golden_dir: Path, input_wav: Path, rows: list[dict[str, object]]) -> None:
    if golden_dir.exists():
        shutil.rmtree(golden_dir)
    golden_dir.mkdir(parents=True)

    manifest = []
    source = {
        "path": str(input_wav),
        "sha256": sha256(input_wav),
        "bytes": input_wav.stat().st_size,
    }
    for row in rows:
        copied = {}
        for key in ["rust_stream", "c_stream"]:
            src = Path(row[key])
            dst = golden_dir / src.name
            shutil.copy2(src, dst)
            copied[key] = {"file": dst.name, "sha256": sha256(dst), "bytes": dst.stat().st_size}
        for key in ["rust_decoded", "c_decoded"]:
            src = Path(row[key])
            copied[key] = {"sha256": sha256(src), "bytes": src.stat().st_size}
        manifest.append(
            {
                "frame_ms": row["frame_ms"],
                "frame_size": row["frame_size"],
                "bitrate": row["bitrate"],
                "metrics": {
                    "rust_corr": row["rust_corr"],
                    "rust_snr_db": row["rust_snr_db"],
                    "rust_lag": row["rust_lag"],
                    "c_corr": row["c_corr"],
                    "c_snr_db": row["c_snr_db"],
                    "c_lag": row["c_lag"],
                },
                "goldens": copied,
            }
        )
    (golden_dir / "source.json").write_text(json.dumps(source, indent=2) + "\n")
    (golden_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def write_testdata(testdata_dir: Path, input_wav: Path, rows: list[dict[str, object]]) -> None:
    if testdata_dir.exists():
        shutil.rmtree(testdata_dir)
    testdata_dir.mkdir(parents=True)

    source = {
        "path": str(input_wav),
        "sha256": sha256(input_wav),
        "bytes": input_wav.stat().st_size,
    }
    manifest = []
    for row in rows:
        case = f"{row['frame_ms']}ms_{int(row['bitrate']) // 1000}k"
        case_dir = testdata_dir / case
        case_dir.mkdir()
        artifacts = {}
        copies = {
            "rust_stream": "rust.lors",
            "rust_decoded": "rust.decoded.wav",
            "c_stream": "c.bit",
            "c_decoded": "c.decoded.wav",
        }
        for key, name in copies.items():
            src = Path(row[key])
            dst = case_dir / name
            shutil.copy2(src, dst)
            artifacts[key] = {"file": str(dst.relative_to(testdata_dir)), "sha256": sha256(dst), "bytes": dst.stat().st_size}
        manifest.append(
            {
                "case": case,
                "frame_ms": row["frame_ms"],
                "frame_size": row["frame_size"],
                "bitrate": row["bitrate"],
                "metrics": {
                    "rust_corr": row["rust_corr"],
                    "rust_snr_db": row["rust_snr_db"],
                    "rust_lag": row["rust_lag"],
                    "c_corr": row["c_corr"],
                    "c_snr_db": row["c_snr_db"],
                    "c_lag": row["c_lag"],
                },
                "artifacts": artifacts,
            }
        )
    (testdata_dir / "source.json").write_text(json.dumps(source, indent=2) + "\n")
    (testdata_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def markdown_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| Frame | Bitrate | Rust enc | Enc vs C | Rust dec | Dec vs C | C enc | C dec | Rust bytes | C bytes | Rust SNR | C SNR |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        enc_delta = 100.0 * (float(row["rust_encode_ms"]) - float(row["c_encode_ms"])) / float(row["c_encode_ms"])
        dec_delta = 100.0 * (float(row["rust_decode_ms"]) - float(row["c_decode_ms"])) / float(row["c_decode_ms"])
        lines.append(
            "| {frame_ms} ms | {bitrate} kb/s | {rust_encode_ms:.2f} ms | {enc_delta:+.1f}% | "
            "{rust_decode_ms:.2f} ms | {dec_delta:+.1f}% | {c_encode_ms:.2f} ms | {c_decode_ms:.2f} ms | "
            "{rust_bytes} | {c_bytes} | "
            "{rust_snr_db:.1f} dB | {c_snr_db:.1f} dB |".format(
                frame_ms=row["frame_ms"],
                bitrate=int(row["bitrate"]) // 1000,
                rust_encode_ms=row["rust_encode_ms"],
                enc_delta=enc_delta,
                rust_decode_ms=row["rust_decode_ms"],
                dec_delta=dec_delta,
                c_encode_ms=row["c_encode_ms"],
                c_decode_ms=row["c_decode_ms"],
                rust_bytes=row["rust_bytes"],
                c_bytes=row["c_bytes"],
                rust_snr_db=row["rust_snr_db"],
                c_snr_db=row["c_snr_db"],
            )
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opus-demo", type=Path, default=DEFAULT_OPUS_DEMO)
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--golden-dir", type=Path)
    parser.add_argument("--testdata-dir", type=Path)
    parser.add_argument("--input-wav", type=Path)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    input_wav, rows = benchmark(args)
    if args.golden_dir:
        write_goldens(args.golden_dir, input_wav, rows)
    if args.testdata_dir:
        write_testdata(args.testdata_dir, input_wav, rows)
    print(markdown_table(rows))


if __name__ == "__main__":
    main()
