#!/usr/bin/env python3
"""Run paired native CELT timing checks between two Rust builds."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import statistics
import subprocess
from typing import Any


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Alternate two raw_celt_bench binaries across a 48 kHz corpus "
            "and report paired encode and decode deltas."
        )
    )
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--seconds", type=positive_int, default=20)
    parser.add_argument("--warmup-seconds", type=positive_int, default=5)
    parser.add_argument("--rounds", type=positive_int, default=5)
    parser.add_argument("--repeats", type=positive_int, default=7)
    parser.add_argument("--pcm-bits", type=int, choices=(16, 24), default=24)
    parser.add_argument("--mode", choices=("cbr", "vbr"), default="vbr")
    parser.add_argument("--bitrate", type=positive_int, default=256_000)
    parser.add_argument("--frame-size", type=positive_int, default=240)
    parser.add_argument(
        "--application", choices=("audio", "restricted-lowdelay"), default="audio"
    )
    parser.add_argument(
        "--baseline-decode-api", choices=("core", "adapter"), default="core"
    )
    parser.add_argument(
        "--candidate-decode-api", choices=("core", "adapter"), default="core"
    )
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--json", type=Path)
    return parser.parse_args()


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


def run_once(
    binary: Path,
    source: Path,
    args: argparse.Namespace,
    *,
    seconds: int,
    repeats: int,
    decode_api: str,
) -> dict[str, Any]:
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
        str(args.frame_size),
        "--bitrate",
        str(args.bitrate),
        "--input-s32le",
        str(source),
        "--pcm-bits",
        str(args.pcm_bits),
        "--skip-quality",
        "--decode-api",
        decode_api,
    ]
    if args.cpu is not None:
        command = ["taskset", "-c", str(args.cpu), *command]
    result = subprocess.run(
        command, check=True, text=True, stdout=subprocess.PIPE
    )
    rows = list(csv.DictReader(io.StringIO(result.stdout), delimiter="\t"))
    if len(rows) != 1:
        raise RuntimeError(f"expected one benchmark row, received {len(rows)}")
    row = rows[0]
    return {
        "encode_ms": float(row["encode_ms"]),
        "decode_ms": float(row["decode_ms"]),
        "bytes": int(row["bytes"]),
        "min_packet": int(row["min_packet"]),
        "max_packet": int(row["max_packet"]),
        "checksum": int(row["checksum"]),
    }


def delta(candidate: float, baseline: float) -> float:
    return 100.0 * (candidate - baseline) / baseline


def summarize(
    sources: list[tuple[str, Path]], results: list[dict[str, Any]]
) -> dict[str, Any]:
    pairs: dict[str, dict[str, list[dict[str, Any]]]] = {
        name: {"baseline": [], "candidate": []} for name, _ in sources
    }
    for result in results:
        pairs[result["source"]][result["implementation"]].append(result["row"])

    summary: dict[str, Any] = {}
    for metric in ("encode_ms", "decode_ms"):
        baseline_total = sum(
            statistics.median(row[metric] for row in pairs[name]["baseline"])
            for name, _ in sources
        )
        candidate_total = sum(
            statistics.median(row[metric] for row in pairs[name]["candidate"])
            for name, _ in sources
        )
        paired = []
        for name, _ in sources:
            paired.extend(
                delta(candidate[metric], baseline[metric])
                for baseline, candidate in zip(
                    pairs[name]["baseline"], pairs[name]["candidate"], strict=True
                )
            )
        summary[metric] = {
            "baseline_corpus_median_sum_ms": baseline_total,
            "candidate_corpus_median_sum_ms": candidate_total,
            "corpus_delta_percent": delta(candidate_total, baseline_total),
            "paired_median_percent": statistics.median(paired),
            "paired_mean_percent": statistics.fmean(paired),
            "faster_pairs": sum(value < 0.0 for value in paired),
            "pair_count": len(paired),
        }

    identity_fields = ("bytes", "min_packet", "max_packet", "checksum")
    summary["identity_changes"] = sum(
        any(
            baseline[field] != candidate[field]
            for field in identity_fields
        )
        for name, _ in sources
        for baseline, candidate in zip(
            pairs[name]["baseline"], pairs[name]["candidate"], strict=True
        )
    )
    return summary


def main() -> None:
    args = parse_args()
    args.baseline = args.baseline.resolve()
    args.candidate = args.candidate.resolve()
    args.corpus = args.corpus.resolve()
    for binary in (args.baseline, args.candidate):
        if not binary.is_file():
            raise RuntimeError(f"benchmark binary does not exist: {binary}")

    sources = discover_sources(args.corpus, args.source)
    warmup_seconds = min(args.seconds, args.warmup_seconds)
    results: list[dict[str, Any]] = []

    for source_index, (name, path) in enumerate(sources):
        order = ("baseline", "candidate") if source_index % 2 == 0 else (
            "candidate",
            "baseline",
        )
        print(f"Warm-up {source_index + 1}/{len(sources)}: {name}", flush=True)
        for implementation in order:
            run_once(
                getattr(args, implementation),
                path,
                args,
                seconds=warmup_seconds,
                repeats=1,
                decode_api=getattr(args, f"{implementation}_decode_api"),
            )

    for round_index in range(args.rounds):
        print(f"Round {round_index + 1}/{args.rounds}", flush=True)
        for source_index, (name, path) in enumerate(sources):
            baseline_first = (round_index + source_index) % 2 == 0
            order = ("baseline", "candidate") if baseline_first else (
                "candidate",
                "baseline",
            )
            for implementation in order:
                row = run_once(
                    getattr(args, implementation),
                    path,
                    args,
                    seconds=args.seconds,
                    repeats=args.repeats,
                    decode_api=getattr(args, f"{implementation}_decode_api"),
                )
                results.append(
                    {"source": name, "implementation": implementation, "row": row}
                )

    summary = summarize(sources, results)
    print()
    print("| Metric | Corpus delta | Paired median | Paired mean | Faster pairs |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for metric, label in (("encode_ms", "Encode"), ("decode_ms", "Decode")):
        row = summary[metric]
        print(
            f"| {label} | {row['corpus_delta_percent']:+.4f}% | "
            f"{row['paired_median_percent']:+.4f}% | "
            f"{row['paired_mean_percent']:+.4f}% | "
            f"{row['faster_pairs']}/{row['pair_count']} |"
        )
    print(f"\nPacket/checksum changes: {summary['identity_changes']}")

    if args.json is not None:
        output = args.json.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "configuration": {
                        "sources": [name for name, _ in sources],
                        "seconds": args.seconds,
                        "warmup_seconds": warmup_seconds,
                        "rounds": args.rounds,
                        "repeats": args.repeats,
                        "pcm_bits": args.pcm_bits,
                        "mode": args.mode,
                        "bitrate": args.bitrate,
                        "frame_size": args.frame_size,
                        "application": args.application,
                        "baseline_decode_api": args.baseline_decode_api,
                        "candidate_decode_api": args.candidate_decode_api,
                        "cpu": args.cpu,
                    },
                    "baseline": str(args.baseline),
                    "candidate": str(args.candidate),
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
