#!/usr/bin/env python3
"""Compare C and SoundKit CELT decode on byte-identical music packets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import platform
import statistics
import subprocess
from typing import Any


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rust-producer", type=Path, required=True)
    parser.add_argument("--c-producer", type=Path, required=True)
    parser.add_argument("--rust-decoder", type=Path, required=True)
    parser.add_argument("--c-decoder", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--seconds", type=positive_int, default=10)
    parser.add_argument("--rounds", type=positive_int, default=5)
    parser.add_argument("--repeats", type=positive_int, default=7)
    parser.add_argument("--warmup-repeats", type=positive_int, default=2)
    parser.add_argument("--pcm-bits", type=int, choices=(16, 24), default=24)
    parser.add_argument("--mode", choices=("cbr", "vbr"), default="vbr")
    parser.add_argument("--bitrate", type=positive_int, default=192_000)
    parser.add_argument("--frame-size", type=positive_int, default=240)
    parser.add_argument(
        "--application", choices=("audio", "restricted-lowdelay"), default="audio"
    )
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--json", type=Path)
    return parser.parse_args()


def command_output(command: list[str]) -> str:
    return subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def discover_sources(corpus: Path, filters: list[str]) -> list[tuple[str, Path]]:
    suffix = "-48k-s24.s32le"
    sources = [
        (path.name.removesuffix(suffix), path.resolve())
        for path in sorted(corpus.glob(f"*{suffix}"))
        if not filters or path.name.removesuffix(suffix) in filters
    ]
    known = {name for name, _ in sources}
    unknown = sorted(set(filters) - known)
    if unknown:
        raise RuntimeError(f"unknown corpus source: {', '.join(unknown)}")
    if not sources:
        raise RuntimeError(f"no 48 kHz S24LE inputs in {corpus}")
    return sources


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_packet_dump(
    binary: Path, source: Path, args: argparse.Namespace
) -> tuple[str, dict[str, Any]]:
    packet_count = args.seconds * 48_000 // args.frame_size
    command = [
        str(binary),
        "--repeats",
        "1",
        "--seconds",
        str(args.seconds),
        "--mode",
        args.mode,
        "--application",
        args.application,
        "--frame-size",
        str(args.frame_size),
        "--bitrate",
        str(args.bitrate),
        "--input-s32le",
        str(source),
        "--pcm-bits",
        str(args.pcm_bits),
        "--dump-packets",
        str(packet_count),
    ]
    result = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    rows = list(csv.DictReader(io.StringIO(result.stdout), delimiter="\t"))
    if len(rows) != packet_count:
        raise RuntimeError(
            f"expected {packet_count} dumped packets from {binary}, got {len(rows)}"
        )
    packet_bytes = sum(int(row["len"]) for row in rows)
    packet_hash = hashlib.sha256(result.stdout.encode()).hexdigest()
    return result.stdout, {
        "packets": packet_count,
        "bytes": packet_bytes,
        "dump_sha256": packet_hash,
    }


def run_decoder(
    binary: Path,
    packet_dump: str,
    packet_impl: str,
    repeats: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    command = [
        str(binary),
        "--seconds",
        str(args.seconds),
        "--repeats",
        str(repeats),
        "--pcm-bits",
        str(args.pcm_bits),
        "--impl",
        packet_impl,
        "--mode",
        args.mode,
        "--frame-size",
        str(args.frame_size),
        "--bitrate",
        str(args.bitrate),
    ]
    if args.cpu is not None:
        command = ["taskset", "-c", str(args.cpu), *command]
    result = subprocess.run(
        command,
        input=packet_dump,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    rows = list(csv.DictReader(io.StringIO(result.stdout), delimiter="\t"))
    if len(rows) != 1:
        raise RuntimeError(f"expected one decoder row, got {len(rows)}")
    row = rows[0]
    if row["packet_impl"] != packet_impl:
        raise RuntimeError("decoder selected the wrong packet stream")
    return {
        "decode_ms": float(row["decode_ms"]),
        "packets": int(row["packets"]),
        "bytes": int(row["bytes"]),
        "checksum": int(row["checksum"]),
    }


def delta(candidate: float, baseline: float) -> float:
    return 100.0 * (candidate - baseline) / baseline


def summarize(
    sources: list[tuple[str, Path]], results: list[dict[str, Any]]
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for packet_impl in ("c", "rust", "all"):
        selected = [
            row
            for row in results
            if packet_impl == "all" or row["packet_impl"] == packet_impl
        ]
        paired = []
        corpus_c = 0.0
        corpus_rust = 0.0
        origins = ("c", "rust") if packet_impl == "all" else (packet_impl,)
        for origin in origins:
            for source, _ in sources:
                c_rows = [
                    row["row"]["decode_ms"]
                    for row in selected
                    if row["source"] == source
                    and row["packet_impl"] == origin
                    and row["decoder_impl"] == "c"
                ]
                rust_rows = [
                    row["row"]["decode_ms"]
                    for row in selected
                    if row["source"] == source
                    and row["packet_impl"] == origin
                    and row["decoder_impl"] == "rust"
                ]
                if len(c_rows) != len(rust_rows):
                    raise RuntimeError("unpaired decoder results")
                corpus_c += statistics.median(c_rows)
                corpus_rust += statistics.median(rust_rows)
                paired.extend(
                    delta(rust, c)
                    for c, rust in zip(c_rows, rust_rows, strict=True)
                )
        summary[packet_impl] = {
            "c_corpus_median_sum_ms": corpus_c,
            "rust_corpus_median_sum_ms": corpus_rust,
            "rust_delta_percent": delta(corpus_rust, corpus_c),
            "paired_median_percent": statistics.median(paired),
            "paired_mean_percent": statistics.fmean(paired),
            "rust_faster_pairs": sum(value < 0.0 for value in paired),
            "pair_count": len(paired),
        }
    return summary


def main() -> None:
    args = parse_args()
    for field in ("rust_producer", "c_producer", "rust_decoder", "c_decoder"):
        path = getattr(args, field).resolve()
        if not path.is_file():
            raise RuntimeError(f"binary does not exist: {path}")
        setattr(args, field, path)
    args.corpus = args.corpus.resolve()
    sources = discover_sources(args.corpus, args.source)

    dumps: dict[tuple[str, str], str] = {}
    dump_metadata: list[dict[str, Any]] = []
    for source, path in sources:
        for packet_impl in ("c", "rust"):
            print(f"Packets: {source} ({packet_impl})", flush=True)
            dump, metadata = build_packet_dump(
                getattr(args, f"{packet_impl}_producer"), path, args
            )
            dumps[(source, packet_impl)] = dump
            dump_metadata.append(
                {"source": source, "packet_impl": packet_impl, **metadata}
            )

    for source_index, (source, _) in enumerate(sources):
        for origin_index, packet_impl in enumerate(("c", "rust")):
            order = ("c", "rust") if (source_index + origin_index) % 2 == 0 else (
                "rust",
                "c",
            )
            for decoder_impl in order:
                run_decoder(
                    getattr(args, f"{decoder_impl}_decoder"),
                    dumps[(source, packet_impl)],
                    packet_impl,
                    args.warmup_repeats,
                    args,
                )

    results: list[dict[str, Any]] = []
    for round_index in range(args.rounds):
        print(f"Round {round_index + 1}/{args.rounds}", flush=True)
        for source_index, (source, _) in enumerate(sources):
            for origin_index, packet_impl in enumerate(("c", "rust")):
                c_first = (round_index + source_index + origin_index) % 2 == 0
                order = ("c", "rust") if c_first else ("rust", "c")
                rows: dict[str, dict[str, Any]] = {}
                for decoder_impl in order:
                    row = run_decoder(
                        getattr(args, f"{decoder_impl}_decoder"),
                        dumps[(source, packet_impl)],
                        packet_impl,
                        args.repeats,
                        args,
                    )
                    rows[decoder_impl] = row
                    results.append(
                        {
                            "round": round_index + 1,
                            "source": source,
                            "packet_impl": packet_impl,
                            "decoder_impl": decoder_impl,
                            "row": row,
                        }
                    )
                if (
                    rows["c"]["packets"] != rows["rust"]["packets"]
                    or rows["c"]["bytes"] != rows["rust"]["bytes"]
                ):
                    raise RuntimeError("decoders did not consume identical packet input")

    summary = summarize(sources, results)
    print()
    print("| Packet source | Rust delta | Paired median | Paired mean | Rust faster |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for packet_impl, label in (("c", "C"), ("rust", "SoundKit"), ("all", "Combined")):
        row = summary[packet_impl]
        print(
            f"| {label} | {row['rust_delta_percent']:+.4f}% | "
            f"{row['paired_median_percent']:+.4f}% | "
            f"{row['paired_mean_percent']:+.4f}% | "
            f"{row['rust_faster_pairs']}/{row['pair_count']} |"
        )

    if args.json is not None:
        metadata = {
            "timestamp_utc": command_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"]),
            "host": platform.platform(),
            "lscpu": command_output(["lscpu"]),
            "rustc": command_output(["rustc", "--version"]),
            "cargo": command_output(["cargo", "--version"]),
            "cc": command_output(["cc", "--version"]).splitlines()[0],
            "ffmpeg": command_output(["ffmpeg", "-version"]).splitlines()[0],
            "flac": command_output(["flac", "--version"]),
            "binary_sha256": {
                field: sha256_file(getattr(args, field))
                for field in (
                    "rust_producer",
                    "c_producer",
                    "rust_decoder",
                    "c_decoder",
                )
            },
        }
        output = args.json.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "metadata": metadata,
                    "configuration": {
                        "sources": [name for name, _ in sources],
                        "seconds": args.seconds,
                        "rounds": args.rounds,
                        "repeats": args.repeats,
                        "warmup_repeats": args.warmup_repeats,
                        "pcm_bits": args.pcm_bits,
                        "mode": args.mode,
                        "bitrate": args.bitrate,
                        "frame_size": args.frame_size,
                        "application": args.application,
                        "cpu": args.cpu,
                    },
                    "packet_dumps": dump_metadata,
                    "summary": summary,
                    "results": results,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"Raw results: {output}")


if __name__ == "__main__":
    main()
