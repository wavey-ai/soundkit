#!/usr/bin/env python3
"""Gate SoundKit ALAC decode against FFmpeg on byte-identical PCM output.

The gate indexes and extracts packets before each benchmark driver's timed
region, runs a full warm-up for both decoders, alternates benchmark order, and
uses median per-file timings. It fails if SoundKit's corpus time per sample is
more than the configured threshold slower than the FFmpeg reference.

Examples:

    python3 tools/run_decode_bench.py --build --cpu 0 song.m4a
    python3 tools/run_decode_bench.py --build --cpu 0 --corpus /srv/alac-corpus \
        --rounds 5 --iterations 20 --json results/alac.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = ROOT.parent


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--pattern", default="*.m4a")
    parser.add_argument("--rounds", type=positive_int, default=5)
    parser.add_argument("--iterations", type=positive_int, default=20)
    parser.add_argument("--warmup-iterations", type=positive_int, default=1)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--max-rust-slower-percent", type=float, default=0.0)
    parser.add_argument("--target-dir", type=Path)
    parser.add_argument("--rust-binary", type=Path)
    parser.add_argument("--ffmpeg-binary", type=Path)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--json", type=Path)
    return parser.parse_args()


def command_output(command: list[str], cwd: Path | None = None) -> str:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def cargo_target_dir() -> Path:
    metadata = command_output(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"], WORKSPACE
    )
    return Path(json.loads(metadata)["target_directory"])


def resolve_inputs(args: argparse.Namespace) -> list[Path]:
    paths = [path.resolve() for path in args.inputs]
    if args.corpus is not None:
        corpus = args.corpus.resolve()
        if not corpus.is_dir():
            raise RuntimeError(f"corpus is not a directory: {corpus}")
        paths.extend(path.resolve() for path in sorted(corpus.rglob(args.pattern)))
    if not paths:
        raise RuntimeError("provide an input file or --corpus")
    unique = list(dict.fromkeys(paths))
    missing = [str(path) for path in unique if not path.is_file()]
    if missing:
        raise RuntimeError(f"input files do not exist: {', '.join(missing)}")
    return unique


def run(
    command: list[str],
    cwd: Path = WORKSPACE,
    environment: dict[str, str] | None = None,
) -> None:
    subprocess.run(command, cwd=cwd, env=environment, check=True)


def build_binaries(args: argparse.Namespace, target_dir: Path) -> tuple[Path, Path]:
    rust_binary = target_dir / "release/examples/decode_bench"
    ffmpeg_binary = target_dir / "release/ffmpeg_alac_decode_bench"
    if args.build:
        environment = os.environ.copy()
        environment["RUSTFLAGS"] = (
            f"{environment.get('RUSTFLAGS', '')} -C target-cpu=native"
        ).strip()
        run(
            [
                "cargo",
                "build",
                "--release",
                "--target-dir",
                str(target_dir),
                "-p",
                "soundkit-alac",
                "--example",
                "decode_bench",
            ],
            environment=environment,
        )
        pkg_config = subprocess.run(
            [
                "pkg-config",
                "--cflags",
                "--libs",
                "libavcodec",
                "libavformat",
                "libavutil",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.split()
        ffmpeg_binary.parent.mkdir(parents=True, exist_ok=True)
        run(
            [
                "cc",
                "-O3",
                "-DNDEBUG",
                "-march=native",
                str(ROOT / "tools/ffmpeg_alac_decode_bench.c"),
                *pkg_config,
                "-o",
                str(ffmpeg_binary),
            ]
        )
    return rust_binary, ffmpeg_binary


def parse_result(output: str, implementation: str) -> dict[str, int | str]:
    line = next(
        (line for line in output.splitlines() if line.startswith("implementation=")),
        None,
    )
    if line is None:
        raise RuntimeError(f"{implementation} benchmark did not report a result")
    result: dict[str, int | str] = {}
    for field in line.split():
        name, value = field.split("=", 1)
        result[name] = value
    if result.get("implementation") != implementation:
        raise RuntimeError(f"expected {implementation}, got {result.get('implementation')}")
    for name in (
        "input_bytes",
        "packets",
        "iterations",
        "samples",
        "sample_rate",
        "channels",
        "bit_depth",
        "elapsed_ns",
        "checksum",
    ):
        try:
            result[name] = int(str(result[name]))
        except (KeyError, ValueError) as error:
            raise RuntimeError(f"{implementation} reported invalid {name}") from error
    return result


def benchmark(
    binary: Path,
    implementation: str,
    input_path: Path,
    iterations: int,
    cpu: int | None,
    output_path: Path | None = None,
) -> dict[str, int | str]:
    command = [str(binary), str(input_path), str(iterations)]
    if output_path is not None:
        command.append(str(output_path))
    if cpu is not None:
        command = ["taskset", "--cpu-list", str(cpu), *command]
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return parse_result(completed.stdout, implementation)


def require_identical_results(
    rust: dict[str, int | str], ffmpeg: dict[str, int | str]
) -> None:
    for field in (
        "input_bytes",
        "packets",
        "iterations",
        "samples",
        "sample_rate",
        "channels",
        "bit_depth",
        "checksum",
    ):
        if rust[field] != ffmpeg[field]:
            raise RuntimeError(
                f"decoder result mismatch for {field}: {rust[field]} != {ffmpeg[field]}"
            )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_pcm(
    rust_binary: Path, ffmpeg_binary: Path, input_path: Path, cpu: int | None
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="soundkit-alac-pcm-") as directory:
        root = Path(directory)
        rust_path = root / "soundkit.pcm"
        ffmpeg_path = root / "ffmpeg.pcm"
        rust = benchmark(
            rust_binary, "soundkit-alac", input_path, 1, cpu, rust_path
        )
        ffmpeg = benchmark(
            ffmpeg_binary, "ffmpeg-c", input_path, 1, cpu, ffmpeg_path
        )
        require_identical_results(rust, ffmpeg)
        rust_hash = sha256_file(rust_path)
        ffmpeg_hash = sha256_file(ffmpeg_path)
        if rust_path.stat().st_size != ffmpeg_path.stat().st_size or rust_hash != ffmpeg_hash:
            raise RuntimeError(f"PCM output differs for {input_path}")
        return {
            "pcm_bytes": rust_path.stat().st_size,
            "pcm_sha256": rust_hash,
            "samples": rust["samples"],
            "sample_rate": rust["sample_rate"],
            "channels": rust["channels"],
            "bit_depth": rust["bit_depth"],
        }


def delta_percent(candidate: float, reference: float) -> float:
    return 100.0 * (candidate - reference) / reference


def median(values: list[int]) -> float:
    if not values:
        raise RuntimeError("cannot calculate a median without results")
    return float(statistics.median(values))


def version_metadata() -> dict[str, str]:
    return {
        "lscpu": command_output(["lscpu"]),
        "rustc": command_output(["rustc", "--version"]),
        "cargo": command_output(["cargo", "--version"]),
        "ffmpeg": command_output(["ffmpeg", "-version"]).splitlines()[0],
        "flac": command_output(["flac", "--version"]),
    }


def main() -> None:
    args = parse_args()
    inputs = resolve_inputs(args)
    target_dir = (args.target_dir or cargo_target_dir()).resolve()
    rust_binary, ffmpeg_binary = build_binaries(args, target_dir)
    rust_binary = (args.rust_binary or rust_binary).resolve()
    ffmpeg_binary = (args.ffmpeg_binary or ffmpeg_binary).resolve()
    for binary in (rust_binary, ffmpeg_binary):
        if not binary.is_file():
            raise RuntimeError(f"missing benchmark binary: {binary}; pass --build")

    metadata = version_metadata()
    print("Benchmark environment:")
    for name, value in metadata.items():
        print(f"{name}: {value}")

    identities = {}
    for input_path in inputs:
        identity = verify_pcm(rust_binary, ffmpeg_binary, input_path, args.cpu)
        identities[str(input_path)] = identity
        print(
            f"PCM identity: {input_path.name} {identity['pcm_bytes']} bytes "
            f"sha256={identity['pcm_sha256']}"
        )

    for input_index, input_path in enumerate(inputs):
        order = ("soundkit", "ffmpeg") if input_index % 2 == 0 else ("ffmpeg", "soundkit")
        for implementation in order:
            benchmark(
                rust_binary if implementation == "soundkit" else ffmpeg_binary,
                "soundkit-alac" if implementation == "soundkit" else "ffmpeg-c",
                input_path,
                args.warmup_iterations,
                args.cpu,
            )

    rows: list[dict[str, Any]] = []
    for round_index in range(args.rounds):
        for input_index, input_path in enumerate(inputs):
            soundkit_first = (round_index + input_index) % 2 == 0
            order = ("soundkit", "ffmpeg") if soundkit_first else ("ffmpeg", "soundkit")
            pair: dict[str, dict[str, int | str]] = {}
            for implementation in order:
                row = benchmark(
                    rust_binary if implementation == "soundkit" else ffmpeg_binary,
                    "soundkit-alac" if implementation == "soundkit" else "ffmpeg-c",
                    input_path,
                    args.iterations,
                    args.cpu,
                )
                pair[implementation] = row
                rows.append(
                    {
                        "round": round_index + 1,
                        "input": str(input_path),
                        "implementation": implementation,
                        "result": row,
                    }
                )
            require_identical_results(pair["soundkit"], pair["ffmpeg"])
            print(
                f"round {round_index + 1}/{args.rounds} {input_path.name}: "
                f"soundkit={pair['soundkit']['elapsed_ns']}ns "
                f"ffmpeg={pair['ffmpeg']['elapsed_ns']}ns"
            )

    summaries = []
    corpus_soundkit_ns = 0.0
    corpus_ffmpeg_ns = 0.0
    corpus_samples = 0
    for input_path in inputs:
        input_rows = [row for row in rows if row["input"] == str(input_path)]
        soundkit_times = [
            row["result"]["elapsed_ns"]
            for row in input_rows
            if row["implementation"] == "soundkit"
        ]
        ffmpeg_times = [
            row["result"]["elapsed_ns"]
            for row in input_rows
            if row["implementation"] == "ffmpeg"
        ]
        soundkit_time = median(soundkit_times)
        ffmpeg_time = median(ffmpeg_times)
        samples = int(identities[str(input_path)]["samples"]) * args.iterations
        corpus_soundkit_ns += soundkit_time
        corpus_ffmpeg_ns += ffmpeg_time
        corpus_samples += samples
        summaries.append(
            {
                "input": str(input_path),
                "samples_per_run": identities[str(input_path)]["samples"],
                "soundkit_median_ns": soundkit_time,
                "ffmpeg_median_ns": ffmpeg_time,
                "soundkit_ns_per_sample": soundkit_time / samples,
                "ffmpeg_ns_per_sample": ffmpeg_time / samples,
                "soundkit_delta_percent": delta_percent(soundkit_time, ffmpeg_time),
            }
        )

    delta = delta_percent(corpus_soundkit_ns, corpus_ffmpeg_ns)
    summary = {
        "soundkit_median_sum_ns": corpus_soundkit_ns,
        "ffmpeg_median_sum_ns": corpus_ffmpeg_ns,
        "samples": corpus_samples,
        "soundkit_ns_per_sample": corpus_soundkit_ns / corpus_samples,
        "ffmpeg_ns_per_sample": corpus_ffmpeg_ns / corpus_samples,
        "soundkit_delta_percent": delta,
    }
    print("\n| Input | SoundKit ns/sample | FFmpeg ns/sample | SoundKit delta |")
    print("| --- | ---: | ---: | ---: |")
    for row in summaries:
        print(
            f"| {Path(row['input']).name} | {row['soundkit_ns_per_sample']:.6f} | "
            f"{row['ffmpeg_ns_per_sample']:.6f} | "
            f"{row['soundkit_delta_percent']:+.2f}% |"
        )
    print(
        f"| Corpus | {summary['soundkit_ns_per_sample']:.6f} | "
        f"{summary['ffmpeg_ns_per_sample']:.6f} | {delta:+.2f}% |"
    )

    output = {
        "metadata": metadata,
        "configuration": {
            "inputs": [str(path) for path in inputs],
            "rounds": args.rounds,
            "iterations": args.iterations,
            "warmup_iterations": args.warmup_iterations,
            "cpu": args.cpu,
            "max_rust_slower_percent": args.max_rust_slower_percent,
            "rust_binary": str(rust_binary),
            "ffmpeg_binary": str(ffmpeg_binary),
        },
        "pcm_identity": identities,
        "per_input": summaries,
        "summary": summary,
        "runs": rows,
    }
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(output, indent=2) + "\n")
        print(f"Raw results: {args.json}")
    if delta > args.max_rust_slower_percent:
        raise RuntimeError(
            f"performance gate failed: SoundKit is {delta:+.2f}% versus FFmpeg, "
            f"maximum allowed is {args.max_rust_slower_percent:+.2f}%"
        )


if __name__ == "__main__":
    main()
