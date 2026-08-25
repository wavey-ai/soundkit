#!/usr/bin/env python3
"""Benchmark 5 ms Rust CELT against trunk libopus on the SoundKit FLAC corpus."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import shlex
import shutil
import statistics
import subprocess
import sys
from typing import Any


SAMPLE_RATE = 48_000
CHANNELS = 2
FRAME_SIZE = 240
TARGET_BITRATES = (192_000, 256_000, 320_000)
BYTES_PER_SECOND = SAMPLE_RATE * CHANNELS * 4


@dataclass(frozen=True, order=True)
class Cell:
    source: str
    pcm_bits: int
    mode: str
    bitrate: int


@dataclass(frozen=True)
class CorpusInput:
    name: str
    path: Path


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_pcm_bits(value: str) -> tuple[int, ...]:
    try:
        bits = tuple(dict.fromkeys(int(item) for item in value.split(",")))
    except ValueError as error:
        raise argparse.ArgumentTypeError("PCM bits must be 16, 24, or 16,24") from error
    if not bits or any(item not in (16, 24) for item in bits):
        raise argparse.ArgumentTypeError("PCM bits must be 16, 24, or 16,24")
    return bits


def parse_args(repo_root: Path) -> argparse.Namespace:
    default_corpus = repo_root.parent / "soundkit/testdata/flac-packet-bench/diverse-v1"
    parser = argparse.ArgumentParser(
        description=(
            "Compare the Rust 48 kHz CELT path with trunk libopus across the "
            "SoundKit FLAC corpus. File loading is outside the reported timings."
        )
    )
    parser.add_argument("--corpus", type=Path, default=default_corpus)
    parser.add_argument(
        "--opus-dir",
        type=Path,
        default=Path(os.environ["OPUS_DIR"]) if "OPUS_DIR" in os.environ else None,
        help="built trunk libopus tree containing .libs/libopus.a",
    )
    parser.add_argument("--seconds", type=positive_int)
    parser.add_argument("--warmup-seconds", type=positive_int, default=5)
    parser.add_argument("--rounds", type=positive_int, default=3)
    parser.add_argument("--repeats", type=positive_int, default=7)
    parser.add_argument("--pcm-bits", type=parse_pcm_bits, default=(16, 24))
    parser.add_argument("--mode", choices=("cbr", "vbr", "both"), default="both")
    parser.add_argument("--bitrate", type=int, choices=TARGET_BITRATES)
    parser.add_argument(
        "--application", choices=("audio", "restricted-lowdelay"), default="audio"
    )
    parser.add_argument("--direct-cubic", action="store_true")
    quality_group = parser.add_mutually_exclusive_group()
    quality_group.add_argument("--skip-quality", action="store_true")
    quality_group.add_argument(
        "--quality-only",
        action="store_true",
        help="measure full-length quality without running timing rounds",
    )
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--cpu", type=int, help="Linux CPU number for taskset")
    parser.add_argument(
        "--target-dir", type=Path, default=repo_root / "target/soundkit-flac-corpus",
    )
    parser.add_argument(
        "--bench-dir",
        type=Path,
        default=repo_root / "target/soundkit-flac-corpus-tools",
    )
    parser.add_argument(
        "--json", type=Path, help="write raw results and metadata to JSON"
    )
    parser.add_argument("--no-build", action="store_true")
    return parser.parse_args()


def run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def command_output(command: list[str], cwd: Path) -> str:
    try:
        return run(command, cwd=cwd, capture=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def git_revision(directory: Path) -> str:
    return command_output(["git", "rev-parse", "HEAD"], directory)


def source_fingerprint(repo_root: Path) -> str:
    digest = hashlib.sha256()
    paths = [repo_root / "Cargo.toml", repo_root / "Cargo.lock"]
    paths.extend(sorted((repo_root / "src").rglob("*.rs")))
    paths.extend(sorted((repo_root / "crates").rglob("*.rs")))
    paths.append(repo_root / "examples/raw_celt_bench.rs")
    for path in paths:
        relative = path.relative_to(repo_root)
        digest.update(str(relative).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_corpus(corpus_dir: Path, filters: list[str]) -> list[CorpusInput]:
    inputs = []
    suffix = "-48k-s24.s32le"
    for path in sorted(corpus_dir.glob(f"*{suffix}")):
        name = path.name.removesuffix(suffix)
        if filters and name not in filters:
            continue
        inputs.append(CorpusInput(name, path.resolve()))
    if not inputs:
        selected = ", ".join(filters) if filters else "all sources"
        raise RuntimeError(f"no 48 kHz S32LE inputs for {selected} in {corpus_dir}")
    unknown = sorted(set(filters) - {item.name for item in inputs})
    if unknown:
        raise RuntimeError(f"unknown corpus source: {', '.join(unknown)}")
    return inputs


def corpus_seconds(inputs: list[CorpusInput], requested: int | None) -> int:
    available = []
    for item in inputs:
        size = item.path.stat().st_size
        if size % BYTES_PER_SECOND != 0:
            raise RuntimeError(
                f"{item.path} is not a whole number of stereo 48 kHz seconds"
            )
        available.append(size // BYTES_PER_SECOND)
    seconds = requested if requested is not None else min(available)
    for item, duration in zip(inputs, available, strict=True):
        if duration < seconds:
            raise RuntimeError(
                f"{item.path} has {duration} seconds; {seconds} were requested"
            )
    return seconds


def build_binaries(
    repo_root: Path, opus_dir: Path, target_dir: Path, bench_dir: Path,
) -> tuple[Path, Path, dict[str, str]]:
    opus_library = opus_dir / ".libs/libopus.a"
    if not opus_library.is_file():
        raise RuntimeError(f"{opus_dir} does not contain .libs/libopus.a")

    target_dir.mkdir(parents=True, exist_ok=True)
    bench_dir.mkdir(parents=True, exist_ok=True)
    build_env = os.environ.copy()
    native_flags = build_env.get("RUST_BENCH_RUSTFLAGS", "-C target-cpu=native")
    build_env["RUSTFLAGS"] = " ".join(
        value for value in (build_env.get("RUSTFLAGS", ""), native_flags) if value
    )
    print("Building the Rust benchmark once.", file=sys.stderr, flush=True)
    run(
        [
            "cargo",
            "build",
            "--release",
            "--target-dir",
            str(target_dir),
            "--example",
            "raw_celt_bench",
        ],
        cwd=repo_root,
        env=build_env,
    )

    c_bin = bench_dir / "raw_celt_bench_c"
    compiler = os.environ.get("CC", "cc")
    c_flags = os.environ.get("C_BENCH_CFLAGS", "-O3 -DNDEBUG -march=native")
    print("Building the trunk libopus benchmark once.", file=sys.stderr, flush=True)
    run(
        [
            compiler,
            *shlex.split(c_flags),
            f"-I{opus_dir / 'include'}",
            str(repo_root / "tools/raw_celt_bench.c"),
            str(opus_library),
            "-lm",
            "-o",
            str(c_bin),
        ],
        cwd=repo_root,
    )
    return (
        target_dir / "release/examples/raw_celt_bench",
        c_bin,
        {"rustflags": build_env["RUSTFLAGS"], "cflags": c_flags,},
    )


def existing_binaries(target_dir: Path, bench_dir: Path) -> tuple[Path, Path]:
    rust_bin = target_dir / "release/examples/raw_celt_bench"
    c_bin = bench_dir / "raw_celt_bench_c"
    for binary in (rust_bin, c_bin):
        if not binary.is_file():
            raise RuntimeError(f"{binary} does not exist; remove --no-build")
    return rust_bin, c_bin


def parse_rows(output: str, expected_impl: str) -> list[dict[str, Any]]:
    rows = list(csv.DictReader(io.StringIO(output), delimiter="\t"))
    if not rows:
        raise RuntimeError(f"{expected_impl} benchmark returned no rows")
    parsed = []
    for row in rows:
        if row["impl"] != expected_impl:
            raise RuntimeError(f"expected {expected_impl} output, got {row['impl']}")
        quality_snr_db = float(row["quality_snr_db"])
        parsed.append(
            {
                "mode": row["mode"],
                "bitrate": int(row["bitrate"]),
                "encode_ms": float(row["encode_ms"]),
                "decode_ms": float(row["decode_ms"]),
                "bytes": int(row["bytes"]),
                "min_packet": int(row["min_packet"]),
                "max_packet": int(row["max_packet"]),
                "checksum": int(row["checksum"]),
                "quality_lag": int(row["quality_lag"]),
                "quality_snr_db": None
                if math.isnan(quality_snr_db)
                else quality_snr_db,
            }
        )
    return parsed


def benchmark_command(
    binary: Path,
    implementation: str,
    corpus: CorpusInput,
    pcm_bits: int,
    args: argparse.Namespace,
    *,
    seconds: int,
    repeats: int,
    skip_quality: bool,
) -> list[str]:
    command = [
        str(binary),
        "--repeats",
        str(repeats),
        "--seconds",
        str(seconds),
        "--mode",
        args.mode,
        "--application",
        args.application,
        "--frame-size",
        str(FRAME_SIZE),
        "--input-s32le",
        str(corpus.path),
        "--pcm-bits",
        str(pcm_bits),
    ]
    if args.bitrate is not None:
        command.extend(("--bitrate", str(args.bitrate)))
    if implementation == "rust" and args.direct_cubic:
        command.append("--direct-cubic")
    if skip_quality:
        command.append("--skip-quality")
    else:
        rust_lag = -120
        c_lag = -312 if args.application == "audio" else -120
        command.extend(
            ("--quality-lag", str(rust_lag if implementation == "rust" else c_lag))
        )
    if args.cpu is not None:
        command = ["taskset", "-c", str(args.cpu), *command]
    return command


def invoke_benchmark(
    binary: Path,
    implementation: str,
    corpus: CorpusInput,
    pcm_bits: int,
    args: argparse.Namespace,
    repo_root: Path,
    *,
    seconds: int,
    repeats: int,
    skip_quality: bool,
) -> list[dict[str, Any]]:
    command = benchmark_command(
        binary,
        implementation,
        corpus,
        pcm_bits,
        args,
        seconds=seconds,
        repeats=repeats,
        skip_quality=skip_quality,
    )
    result = run(command, cwd=repo_root, capture=True)
    return parse_rows(result.stdout, implementation)


def row_cell(corpus: CorpusInput, pcm_bits: int, row: dict[str, Any]) -> Cell:
    return Cell(corpus.name, pcm_bits, row["mode"], row["bitrate"])


def percent_delta(rust: float, c_value: float) -> float:
    return 100.0 * (rust - c_value) / c_value


def format_delta(value: float) -> str:
    return f"{value:+.2f}%"


def format_snr(values: list[float]) -> str:
    if not values:
        return "n/a"
    return (
        f"{statistics.fmean(values):+.2f} dB "
        f"[{min(values):+.2f}, {max(values):+.2f}]"
    )


def selected_modes(args: argparse.Namespace) -> tuple[str, ...]:
    return ("cbr", "vbr") if args.mode == "both" else (args.mode,)


def selected_bitrates(args: argparse.Namespace) -> tuple[int, ...]:
    return TARGET_BITRATES if args.bitrate is None else (args.bitrate,)


def render_summary(
    inputs: list[CorpusInput],
    args: argparse.Namespace,
    seconds: int,
    timings: dict[Cell, dict[str, list[dict[str, Any]]]],
    quality: dict[Cell, dict[str, dict[str, Any]]],
) -> None:
    total_audio_ms = len(inputs) * seconds * 1000.0
    print()
    print(
        "| PCM | Mode | Bitrate | Rust enc | Enc vs C | Rust dec | Dec vs C | SNR vs C | Bytes vs C |"
    )
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for pcm_bits in args.pcm_bits:
        for mode in selected_modes(args):
            for bitrate in selected_bitrates(args):
                cells = [Cell(item.name, pcm_bits, mode, bitrate) for item in inputs]
                rust_enc = sum(
                    statistics.median(row["encode_ms"] for row in timings[cell]["rust"])
                    for cell in cells
                )
                c_enc = sum(
                    statistics.median(row["encode_ms"] for row in timings[cell]["c"])
                    for cell in cells
                )
                rust_dec = sum(
                    statistics.median(row["decode_ms"] for row in timings[cell]["rust"])
                    for cell in cells
                )
                c_dec = sum(
                    statistics.median(row["decode_ms"] for row in timings[cell]["c"])
                    for cell in cells
                )
                if args.skip_quality:
                    snr_deltas = []
                    byte_delta = 0
                else:
                    snr_deltas = [
                        float(quality[cell]["rust"]["quality_snr_db"])
                        - float(quality[cell]["c"]["quality_snr_db"])
                        for cell in cells
                    ]
                    byte_delta = sum(
                        quality[cell]["rust"]["bytes"] - quality[cell]["c"]["bytes"]
                        for cell in cells
                    )
                print(
                    f"| {pcm_bits}-bit | {mode.upper()} | {bitrate // 1000} kb/s | "
                    f"{total_audio_ms / rust_enc:.2f}x | {format_delta(percent_delta(rust_enc, c_enc))} | "
                    f"{total_audio_ms / rust_dec:.2f}x | {format_delta(percent_delta(rust_dec, c_dec))} | "
                    f"{format_snr(snr_deltas)} | {byte_delta:+d} B |"
                )

    print()
    print(
        "Positive timing deltas mean Rust took longer. Negative timing deltas mean Rust was faster."
    )
    if not args.skip_quality:
        print(
            "SNR deltas are Rust minus trunk libopus after each implementation's measured codec delay."
        )


def render_quality_summary(
    inputs: list[CorpusInput],
    args: argparse.Namespace,
    quality: dict[Cell, dict[str, dict[str, Any]]],
) -> None:
    print()
    print("| PCM | Mode | Bitrate | Rust SNR | C SNR | SNR vs C | Bytes vs C |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for pcm_bits in args.pcm_bits:
        for mode in selected_modes(args):
            for bitrate in selected_bitrates(args):
                cells = [Cell(item.name, pcm_bits, mode, bitrate) for item in inputs]
                rust_snr = [
                    float(quality[cell]["rust"]["quality_snr_db"]) for cell in cells
                ]
                c_snr = [float(quality[cell]["c"]["quality_snr_db"]) for cell in cells]
                snr_deltas = [rust - c_value for rust, c_value in zip(rust_snr, c_snr)]
                byte_delta = sum(
                    quality[cell]["rust"]["bytes"] - quality[cell]["c"]["bytes"]
                    for cell in cells
                )
                print(
                    f"| {pcm_bits}-bit | {mode.upper()} | {bitrate // 1000} kb/s | "
                    f"{statistics.fmean(rust_snr):.2f} dB | "
                    f"{statistics.fmean(c_snr):.2f} dB | "
                    f"{format_snr(snr_deltas)} | {byte_delta:+d} B |"
                )

    print()
    print(
        "SNR deltas are Rust minus trunk libopus after each implementation's measured codec delay."
    )


def serializable_results(
    inputs: list[CorpusInput],
    args: argparse.Namespace,
    seconds: int,
    metadata: dict[str, Any],
    timings: dict[Cell, dict[str, list[dict[str, Any]]]],
    quality: dict[Cell, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    cells = []
    for cell in sorted(timings.keys() | quality.keys()):
        cells.append(
            {
                "source": cell.source,
                "pcm_bits": cell.pcm_bits,
                "mode": cell.mode,
                "bitrate": cell.bitrate,
                "timings": timings.get(cell),
                "quality": quality.get(cell),
            }
        )
    return {
        "metadata": metadata,
        "configuration": {
            "seconds_per_source": seconds,
            "sources": [item.name for item in inputs],
            "pcm_bits": list(args.pcm_bits),
            "mode": args.mode,
            "bitrate": args.bitrate,
            "frame_size": FRAME_SIZE,
            "application": args.application,
            "direct_cubic": args.direct_cubic,
            "quality_only": args.quality_only,
            "rounds": args.rounds,
            "repeats": args.repeats,
            "warmup_seconds": args.warmup_seconds,
            "cpu": args.cpu,
        },
        "cells": cells,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    args = parse_args(repo_root)
    if args.opus_dir is None:
        raise RuntimeError(
            "set OPUS_DIR or pass --opus-dir for the built trunk libopus tree"
        )
    args.corpus = args.corpus.resolve()
    args.opus_dir = args.opus_dir.resolve()
    args.target_dir = args.target_dir.resolve()
    args.bench_dir = args.bench_dir.resolve()
    if args.cpu is not None and shutil.which("taskset") is None:
        raise RuntimeError("--cpu requires taskset")

    inputs = discover_corpus(args.corpus, args.source)
    seconds = corpus_seconds(inputs, args.seconds)
    warmup_seconds = min(args.warmup_seconds, seconds)
    if args.no_build:
        rust_bin, c_bin = existing_binaries(args.target_dir, args.bench_dir)
        build_flags = {
            "rustflags": os.environ.get("RUSTFLAGS", "unknown"),
            "cflags": os.environ.get("C_BENCH_CFLAGS", "unknown"),
        }
    else:
        rust_bin, c_bin, build_flags = build_binaries(
            repo_root, args.opus_dir, args.target_dir, args.bench_dir
        )

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.platform(),
        "rustc": command_output(["rustc", "--version"], repo_root),
        "cargo": command_output(["cargo", "--version"], repo_root),
        "cc": command_output(
            [os.environ.get("CC", "cc"), "--version"], repo_root
        ).splitlines()[0],
        "ffmpeg": command_output(["ffmpeg", "-version"], repo_root).splitlines()[0],
        "flac": command_output(["flac", "--version"], repo_root).splitlines()[0],
        "lscpu": command_output(["lscpu"], repo_root),
        "rust_revision": git_revision(repo_root),
        "rust_source_sha256": source_fingerprint(repo_root),
        "opus_revision": git_revision(args.opus_dir),
        "opus_configure": command_output(
            [str(args.opus_dir / "config.status"), "--config"], repo_root
        ),
        "corpus_seed": (args.corpus / "seed.txt").read_text().strip()
        if (args.corpus / "seed.txt").is_file()
        else "unavailable",
        "corpus_manifest_sha256": file_sha256(args.corpus / "manifest.tsv")
        if (args.corpus / "manifest.tsv").is_file()
        else "unavailable",
        "corpus_inputs": [
            {
                "name": item.name,
                "path": str(item.path),
                "bytes": item.path.stat().st_size,
                "sha256": file_sha256(item.path),
            }
            for item in inputs
        ],
        **build_flags,
    }
    print(
        f"Corpus: {len(inputs)} sources x {seconds} seconds at 48 kHz stereo; "
        f"5 ms frames; PCM {','.join(map(str, args.pcm_bits))}-bit.",
        file=sys.stderr,
        flush=True,
    )
    if args.quality_only:
        run_description = "Quality-only pass."
    else:
        run_description = (
            f"Timing: {args.rounds} alternating rounds x "
            f"{args.repeats} internal repeats."
        )
    print(
        f"{run_description} Rust {metadata['rust_revision']}; "
        f"Opus {metadata['opus_revision']}.",
        file=sys.stderr,
        flush=True,
    )

    binaries = {"rust": rust_bin, "c": c_bin}
    groups = [(corpus, pcm_bits) for corpus in inputs for pcm_bits in args.pcm_bits]
    timings: dict[Cell, dict[str, list[dict[str, Any]]]] = {}
    quality: dict[Cell, dict[str, dict[str, Any]]] = {}

    for group_index, (corpus, pcm_bits) in enumerate(groups):
        order = ("rust", "c") if group_index % 2 == 0 else ("c", "rust")
        if not args.quality_only:
            print(
                f"Warm-up {group_index + 1}/{len(groups)}: {corpus.name}, {pcm_bits}-bit.",
                file=sys.stderr,
                flush=True,
            )
            for implementation in order:
                invoke_benchmark(
                    binaries[implementation],
                    implementation,
                    corpus,
                    pcm_bits,
                    args,
                    repo_root,
                    seconds=warmup_seconds,
                    repeats=1,
                    skip_quality=True,
                )

        if not args.skip_quality:
            print(
                f"Quality {group_index + 1}/{len(groups)}: {corpus.name}, {pcm_bits}-bit.",
                file=sys.stderr,
                flush=True,
            )
            for implementation in reversed(order):
                rows = invoke_benchmark(
                    binaries[implementation],
                    implementation,
                    corpus,
                    pcm_bits,
                    args,
                    repo_root,
                    seconds=seconds,
                    repeats=1,
                    skip_quality=False,
                )
                for row in rows:
                    quality.setdefault(row_cell(corpus, pcm_bits, row), {})[
                        implementation
                    ] = row

    if not args.quality_only:
        for round_index in range(args.rounds):
            print(
                f"Timing round {round_index + 1}/{args.rounds}.",
                file=sys.stderr,
                flush=True,
            )
            for group_index, (corpus, pcm_bits) in enumerate(groups):
                print(
                    f"  Group {group_index + 1}/{len(groups)}: {corpus.name}, {pcm_bits}-bit.",
                    file=sys.stderr,
                    flush=True,
                )
                rust_first = (round_index + group_index) % 2 == 0
                order = ("rust", "c") if rust_first else ("c", "rust")
                for implementation in order:
                    rows = invoke_benchmark(
                        binaries[implementation],
                        implementation,
                        corpus,
                        pcm_bits,
                        args,
                        repo_root,
                        seconds=seconds,
                        repeats=args.repeats,
                        skip_quality=True,
                    )
                    for row in rows:
                        cell = row_cell(corpus, pcm_bits, row)
                        timings.setdefault(cell, {"rust": [], "c": []})[
                            implementation
                        ].append(row)

        render_summary(inputs, args, seconds, timings, quality)
    else:
        render_quality_summary(inputs, args, quality)
    if args.json is not None:
        output = args.json.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                serializable_results(inputs, args, seconds, metadata, timings, quality),
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
        print(f"Raw results: {output}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
