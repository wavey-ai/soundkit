#!/usr/bin/env python3
"""Build balanced 5 ms FLAC benchmark corpora from local album excerpts."""

from __future__ import annotations

import argparse
from array import array
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


SEED = "soundkit-flac-diverse-v1"
EXCERPT_SECONDS = 10.0
EXCERPTS_PER_GROUP = 10
AUDIO_SUFFIXES = {".aac", ".aif", ".aiff", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav"}


@dataclass(frozen=True)
class Source:
    path: Path
    digest: str


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def unique_sources(paths: list[Path]) -> list[Source]:
    by_digest: dict[str, Path] = {}
    for path in sorted(set(paths)):
        digest = file_digest(path)
        current = by_digest.get(digest)
        if current is None or str(path) < str(current):
            by_digest[digest] = path
    return [Source(path, digest) for digest, path in sorted(by_digest.items())]


def audio_files(directory: Path) -> list[Path]:
    return [
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    ]


def lori_sources(downloads: Path) -> list[Path]:
    paths = []
    for path in downloads.rglob("*"):
        if not path.is_file() or "emastered" in str(path).lower():
            continue
        lower = str(path).lower()
        if path.suffix.lower() == ".wav" and "confirmation" in path.name.lower():
            paths.append(path)
        elif path.suffix.lower() == ".mp3" and "lori asha album premix" in lower:
            paths.append(path)
    return paths


def stable_rank(group: str, source: Source) -> bytes:
    value = f"{SEED}\0{group}\0{source.digest}\0{source.path}".encode()
    return hashlib.sha256(value).digest()


def select_sources(group: str, sources: list[Source]) -> list[Source]:
    if not sources:
        raise RuntimeError(f"no audio sources found for {group}")
    ranked = sorted(sources, key=lambda source: stable_rank(group, source))
    if len(ranked) >= EXCERPTS_PER_GROUP:
        return ranked[:EXCERPTS_PER_GROUP]
    return [ranked[index % len(ranked)] for index in range(EXCERPTS_PER_GROUP)]


def duration_seconds(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    duration = float(json.loads(result.stdout)["format"]["duration"])
    if duration < EXCERPT_SECONDS:
        raise RuntimeError(f"{path} is shorter than {EXCERPT_SECONDS:.0f} seconds")
    return duration


def stable_offset(group: str, slot: int, source: Source, duration: float) -> float:
    # Avoid intros/outros when the track is long enough. Repeated selections
    # (needed for the seven-track Hats album) get independent offsets.
    low = 10.0 if duration >= 30.0 else 0.0
    high = duration - EXCERPT_SECONDS - (10.0 if duration >= 30.0 else 0.0)
    if high <= low:
        return low
    token = f"{SEED}\0{group}\0{slot}\0{source.digest}".encode()
    fraction = int.from_bytes(hashlib.sha256(token).digest()[:8], "big") / 2**64
    return low + fraction * (high - low)


def decode_excerpt(path: Path, offset: float, rate: int) -> bytes:
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-ss",
        f"{offset:.9f}",
        "-i",
        str(path),
        "-t",
        f"{EXCERPT_SECONDS:.1f}",
        "-vn",
        "-sn",
        "-dn",
        "-af",
        "volume=0.00390625:precision=fixed",
        "-ar",
        str(rate),
        "-ac",
        "2",
        "-c:a",
        "pcm_s32le",
        "-f",
        "s32le",
        "pipe:1",
    ]
    decoded = subprocess.run(command, check=True, capture_output=True).stdout
    expected = int(rate * EXCERPT_SECONDS) * 2 * 4
    if len(decoded) != expected:
        raise RuntimeError(f"{path} produced {len(decoded)} bytes at {rate} Hz; expected {expected}")
    return clamp_s24(decoded)


def clamp_s24(decoded: bytes) -> bytes:
    """Match the public 24-bit encoder contract after source resampling."""
    try:
        import numpy as np

        samples = np.frombuffer(decoded, dtype="<i4").copy()
        np.clip(samples, -8_388_608, 8_388_607, out=samples)
        return samples.tobytes()
    except ImportError:
        samples = array("i")
        samples.frombytes(decoded)
        if sys.byteorder != "little":
            samples.byteswap()
        for index, sample in enumerate(samples):
            samples[index] = max(-8_388_608, min(8_388_607, sample))
        if sys.byteorder != "little":
            samples.byteswap()
        return samples.tobytes()


def display_path(path: Path, downloads: Path) -> str:
    try:
        return f"~/Downloads/{path.relative_to(downloads)}"
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--downloads", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    downloads = args.downloads.resolve()
    groups = {
        "lori-asha": lori_sources(downloads),
        "blue-nile-hats": audio_files(downloads / "The_Blue_Nile-Hats-1989-ERP_INT"),
        "bill-evans-secret-sessions": audio_files(downloads / "Bill Evans - The Secret Sessions"),
        "nocturnal-animals": audio_files(
            downloads / "Abel Korzeniowski - 2016 - Nocturnal Animals OST FLAC"
        ),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_lines = [
        "group\tslot\toffset_seconds\tduration_seconds\tsha256\tpath"
    ]
    for group, paths in groups.items():
        sources = unique_sources(paths)
        selected = select_sources(group, sources)
        pcm_paths = {
            48_000: args.output / f"{group}-48k-s24.s32le",
            96_000: args.output / f"{group}-96k-s24.s32le",
        }
        outputs = {rate: path.open("wb") for rate, path in pcm_paths.items()}
        try:
            for slot, source in enumerate(selected, start=1):
                duration = duration_seconds(source.path)
                offset = stable_offset(group, slot, source, duration)
                print(
                    f"{group} {slot:02d}/{EXCERPTS_PER_GROUP}: "
                    f"{source.path.name} at {offset:.3f}s",
                    flush=True,
                )
                for rate, output in outputs.items():
                    output.write(decode_excerpt(source.path, offset, rate))
                manifest_lines.append(
                    "\t".join(
                        [
                            group,
                            str(slot),
                            f"{offset:.9f}",
                            f"{duration:.9f}",
                            source.digest,
                            display_path(source.path, downloads),
                        ]
                    )
                )
        finally:
            for output in outputs.values():
                output.close()

    (args.output / "manifest.tsv").write_text("\n".join(manifest_lines) + "\n")
    (args.output / "seed.txt").write_text(SEED + "\n")


if __name__ == "__main__":
    main()
