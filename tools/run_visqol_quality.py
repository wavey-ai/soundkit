#!/usr/bin/env python3
"""Compare Rust and C CELT round-trips with official ViSQOL Audio."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


SAMPLE_RATE = 48_000
CHANNELS = 2
MAX_ALIGNMENT_LAG = 960
VALID_FRAME_SIZES = {120, 240, 480, 960}
VALID_MODES = {"cbr", "vbr"}


@dataclass(frozen=True)
class Track:
    name: str
    path: Path


@dataclass(frozen=True)
class Configuration:
    frame_size: int
    bitrate: int
    mode: str

    @property
    def key(self) -> str:
        frame_ms = self.frame_size * 1000 / SAMPLE_RATE
        return f"{self.mode}_{frame_ms:g}ms_{self.bitrate // 1000}k"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--track",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Add one source track. Repeat this option for more tracks.",
    )
    parser.add_argument("--rust-bin", type=Path, required=True)
    parser.add_argument("--c-bin", type=Path, required=True)
    parser.add_argument("--visqol", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--excerpts-per-track", type=int, default=5)
    parser.add_argument("--excerpt-seconds", type=float, default=8.0)
    parser.add_argument("--edge-margin-seconds", type=float, default=5.0)
    parser.add_argument("--bitrates", default="192000,256000,320000")
    parser.add_argument("--frame-sizes", default="240,960")
    parser.add_argument("--modes", default="cbr,vbr")
    parser.add_argument("--headroom-db", type=float, default=0.1)
    parser.add_argument("--blind-cases", type=int, default=10)
    parser.add_argument("--visqol-jobs", type=int, default=4)
    parser.add_argument("--keep-raw", action="store_true")
    return parser.parse_args()


def parse_tracks(values: list[str]) -> list[Track]:
    tracks = []
    names = set()
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError(f"invalid --track value: {value}")
        if name in names:
            raise ValueError(f"duplicate track name: {name}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"track does not exist: {path}")
        tracks.append(Track(name=name, path=path))
        names.add(name)
    return tracks


def parse_int_list(value: str, label: str) -> list[int]:
    try:
        values = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise ValueError(f"invalid {label}: {value}") from error
    if not values or any(item <= 0 for item in values):
        raise ValueError(f"invalid {label}: {value}")
    return values


def parse_modes(value: str) -> list[str]:
    modes = value.split(",")
    if not modes or any(mode not in VALID_MODES for mode in modes):
        raise ValueError(f"invalid modes: {value}")
    return modes


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_json(command: list[str]) -> dict:
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"command returned no JSON: {command[0]}")
    return json.loads(lines[-1])


def git_value(directory: Path, *arguments: str) -> str | None:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=directory,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def choose_starts(
    frame_count: int,
    excerpt_frames: int,
    margin_frames: int,
    count: int,
    rng: random.Random,
) -> list[int]:
    alignment = max(VALID_FRAME_SIZES)
    first = math.ceil(margin_frames / alignment)
    last = (frame_count - margin_frames - excerpt_frames) // alignment
    if last < first:
        raise ValueError("track is too short for the requested excerpts")
    candidates = list(range(first, last + 1))
    rng.shuffle(candidates)
    starts = []
    for candidate in candidates:
        start = candidate * alignment
        end = start + excerpt_frames
        if all(end <= other or start >= other + excerpt_frames for other in starts):
            starts.append(start)
            if len(starts) == count:
                return sorted(starts)
    raise ValueError("cannot select enough non-overlapping excerpts")


def write_f32le(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(audio, dtype="<f4").tofile(path)


def read_f32le(path: Path, frame_count: int) -> np.ndarray:
    audio = np.fromfile(path, dtype="<f4")
    expected = frame_count * CHANNELS
    if audio.size != expected:
        raise ValueError(f"{path} has {audio.size} samples; expected {expected}")
    audio = audio.reshape(frame_count, CHANNELS)
    if not np.isfinite(audio).all():
        raise ValueError(f"{path} contains non-finite samples")
    return audio


def estimate_lag(reference: np.ndarray, candidate: np.ndarray) -> int:
    length = min(reference.shape[0], candidate.shape[0], SAMPLE_RATE * 4)
    ref = reference[:length].mean(axis=1).astype(np.float64)
    deg = candidate[:length].mean(axis=1).astype(np.float64)
    ref -= ref.mean()
    deg -= deg.mean()
    correlation_size = deg.size + ref.size - 1
    fft_size = 1 << (correlation_size - 1).bit_length()
    correlation = np.fft.irfft(
        np.fft.rfft(deg, fft_size) * np.fft.rfft(ref[::-1], fft_size), fft_size
    )[:correlation_size]
    lags = np.arange(-ref.size + 1, deg.size)
    selected = (lags >= -MAX_ALIGNMENT_LAG) & (lags <= MAX_ALIGNMENT_LAG)
    return int(lags[selected][np.argmax(correlation[selected])])


def aligned_audio(
    reference: np.ndarray,
    candidates: dict[str, np.ndarray],
    lags: dict[str, int],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    reference_start = max(0, max(-lag for lag in lags.values()))
    reference_end = min(
        reference.shape[0],
        min(candidate.shape[0] - lag for name, candidate in candidates.items() for lag in [lags[name]]),
    )
    if reference_end <= reference_start:
        raise ValueError("alignment produced an empty comparison")
    aligned_reference = reference[reference_start:reference_end]
    aligned_candidates = {
        name: candidate[reference_start + lags[name] : reference_end + lags[name]]
        for name, candidate in candidates.items()
    }
    return aligned_reference, aligned_candidates


def snr_db(reference: np.ndarray, candidate: np.ndarray) -> float:
    signal_energy = float(np.sum(np.square(reference, dtype=np.float64)))
    error_energy = float(np.sum(np.square(reference - candidate, dtype=np.float64)))
    return 10.0 * math.log10(signal_energy / max(error_energy, np.finfo(float).tiny))


def si_sdr_db(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = reference.astype(np.float64).reshape(-1)
    deg = candidate.astype(np.float64).reshape(-1)
    denominator = float(np.dot(ref, ref))
    scale = float(np.dot(deg, ref)) / max(denominator, np.finfo(float).tiny)
    target = scale * ref
    noise = deg - target
    return 10.0 * math.log10(
        float(np.dot(target, target))
        / max(float(np.dot(noise, noise)), np.finfo(float).tiny)
    )


def channel_correlation(audio: np.ndarray) -> float:
    left = audio[:, 0].astype(np.float64)
    right = audio[:, 1].astype(np.float64)
    denominator = math.sqrt(float(np.dot(left, left)) * float(np.dot(right, right)))
    return float(np.dot(left, right)) / max(denominator, np.finfo(float).tiny)


def diagnostics(reference: np.ndarray, candidate: np.ndarray) -> dict:
    mid_reference = 0.5 * (reference[:, 0] + reference[:, 1])
    mid_candidate = 0.5 * (candidate[:, 0] + candidate[:, 1])
    side_reference = 0.5 * (reference[:, 0] - reference[:, 1])
    side_candidate = 0.5 * (candidate[:, 0] - candidate[:, 1])
    return {
        "snr_db": snr_db(reference, candidate),
        "si_sdr_db": si_sdr_db(reference, candidate),
        "mid_snr_db": snr_db(mid_reference, mid_candidate),
        "side_snr_db": snr_db(side_reference, side_candidate),
        "channel_correlation": channel_correlation(candidate),
        "reference_channel_correlation": channel_correlation(reference),
        "peak": float(np.max(np.abs(candidate))),
    }


def write_evaluation_wavs(
    directory: Path,
    reference: np.ndarray,
    candidates: dict[str, np.ndarray],
    headroom_db: float,
) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    peak = max(
        float(np.max(np.abs(reference))),
        *(float(np.max(np.abs(candidate))) for candidate in candidates.values()),
    )
    target_peak = 10.0 ** (-headroom_db / 20.0)
    gain = min(1.0, target_peak / max(peak, np.finfo(np.float32).tiny))
    paths = {"master": directory / "master.wav"}
    paths.update({name: directory / f"{name}.wav" for name in candidates})
    sf.write(paths["master"], reference * gain, SAMPLE_RATE, subtype="PCM_16")
    for name, candidate in candidates.items():
        sf.write(paths[name], candidate * gain, SAMPLE_RATE, subtype="PCM_16")
    return paths


def confidence_interval(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        return values[0], values[0]
    mean = statistics.fmean(values)
    degrees_of_freedom = len(values) - 1
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    normal_critical = statistics.NormalDist().inv_cdf(0.975)
    z = normal_critical
    df = degrees_of_freedom
    t_critical = (
        z
        + (z**3 + z) / (4 * df)
        + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)
        + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / (384 * df**3)
    )
    half_width = t_critical * standard_error
    return mean - half_width, mean + half_width


def run_visqol_shard(
    shard_index: int,
    indexed_rows: list[tuple[int, dict]],
    visqol: Path,
    visqol_root: Path,
    visqol_model: Path,
    output_root: Path,
) -> list[tuple[int, dict]]:
    shard_root = output_root / "visqol-shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    pairs_path = shard_root / f"pairs-{shard_index:02d}.csv"
    results_path = shard_root / f"results-{shard_index:02d}.csv"
    log_path = shard_root / f"visqol-{shard_index:02d}.log"
    with pairs_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["reference", "degraded"])
        for _, row in indexed_rows:
            writer.writerow(
                [
                    row["evaluation_paths"]["master"],
                    row["evaluation_paths"][row["treatment"]],
                ]
            )
    with log_path.open("w") as log:
        subprocess.run(
            [
                str(visqol),
                "--batch_input_csv",
                str(pairs_path),
                "--results_csv",
                str(results_path),
                "--similarity_to_quality_model",
                str(visqol_model),
                "--use_lattice_model=false",
            ],
            cwd=visqol_root,
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    with results_path.open(newline="") as handle:
        result_rows = list(csv.DictReader(handle))
    if len(result_rows) != len(indexed_rows):
        raise ValueError(
            f"ViSQOL shard {shard_index} returned {len(result_rows)} rows; "
            f"expected {len(indexed_rows)}"
        )
    return [
        (row_index, result_row)
        for (row_index, _), result_row in zip(indexed_rows, result_rows, strict=True)
    ]


def aggregate(rows: list[dict], configurations: list[Configuration]) -> dict:
    paired = {}
    for row in rows:
        key = (row["track"], row["excerpt"], row["configuration"])
        paired.setdefault(key, {})[row["treatment"]] = row["mos_lqo"]
    differences = [scores["rust"] - scores["c"] for scores in paired.values()]
    overall_ci = confidence_interval(differences)
    result = {
        "overall": {
            "pairs": len(differences),
            "rust_mean": statistics.fmean(row["mos_lqo"] for row in rows if row["treatment"] == "rust"),
            "c_mean": statistics.fmean(row["mos_lqo"] for row in rows if row["treatment"] == "c"),
            "paired_difference_mean": statistics.fmean(differences),
            "paired_difference_ci95": list(overall_ci),
            "rust_wins": sum(value > 0 for value in differences),
            "ties": sum(value == 0 for value in differences),
            "c_wins": sum(value < 0 for value in differences),
        },
        "configurations": {},
    }
    for configuration in configurations:
        key = configuration.key
        selected = [row for row in rows if row["configuration"] == key]
        rust_scores = [row["mos_lqo"] for row in selected if row["treatment"] == "rust"]
        c_scores = [row["mos_lqo"] for row in selected if row["treatment"] == "c"]
        config_differences = [
            scores["rust"] - scores["c"]
            for pair_key, scores in paired.items()
            if pair_key[2] == key
        ]
        ci = confidence_interval(config_differences)
        result["configurations"][key] = {
            "pairs": len(config_differences),
            "rust_mean": statistics.fmean(rust_scores),
            "c_mean": statistics.fmean(c_scores),
            "paired_difference_mean": statistics.fmean(config_differences),
            "paired_difference_ci95": list(ci),
            "rust_snr_mean": statistics.fmean(
                row["diagnostics"]["snr_db"]
                for row in selected
                if row["treatment"] == "rust"
            ),
            "c_snr_mean": statistics.fmean(
                row["diagnostics"]["snr_db"]
                for row in selected
                if row["treatment"] == "c"
            ),
            "rust_side_snr_mean": statistics.fmean(
                row["diagnostics"]["side_snr_db"]
                for row in selected
                if row["treatment"] == "rust"
            ),
            "c_side_snr_mean": statistics.fmean(
                row["diagnostics"]["side_snr_db"]
                for row in selected
                if row["treatment"] == "c"
            ),
        }
    return result


def write_blind_set(
    output_root: Path,
    rows: list[dict],
    count: int,
    seed: int,
) -> None:
    grouped = {}
    for row in rows:
        key = (row["track"], row["excerpt"], row["configuration"])
        grouped.setdefault(key, {})[row["treatment"]] = row
    ranked = sorted(
        grouped.items(),
        key=lambda item: abs(item[1]["rust"]["mos_lqo"] - item[1]["c"]["mos_lqo"]),
        reverse=True,
    )[:count]
    blind_root = output_root / "blind"
    blind_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed ^ 0x414258)
    public_rows = []
    answer_rows = []

    def link_or_copy(source: Path, destination: Path) -> None:
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)

    for index, (key, treatments) in enumerate(ranked, start=1):
        case_dir = blind_root / f"case-{index:02d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        rust_is_a = bool(rng.getrandbits(1))
        labels = {"rust": "A" if rust_is_a else "B", "c": "B" if rust_is_a else "A"}
        reference = Path(treatments["rust"]["evaluation_paths"]["master"])
        link_or_copy(reference, case_dir / "reference.wav")
        for treatment, label in labels.items():
            link_or_copy(
                Path(treatments[treatment]["evaluation_paths"][treatment]),
                case_dir / f"{label}.wav",
            )
        public_rows.append(
            {
                "case": index,
                "track": key[0],
                "excerpt": key[1],
                "configuration": key[2],
            }
        )
        answer_rows.append(
            {
                **public_rows[-1],
                "A": "rust" if rust_is_a else "c",
                "B": "c" if rust_is_a else "rust",
                "rust_mos_lqo": treatments["rust"]["mos_lqo"],
                "c_mos_lqo": treatments["c"]["mos_lqo"],
            }
        )
    (blind_root / "manifest.json").write_text(json.dumps(public_rows, indent=2) + "\n")
    (blind_root / "answers.json").write_text(json.dumps(answer_rows, indent=2) + "\n")


def markdown_report(report: dict) -> str:
    overall = report["summary"]["overall"]
    lines = [
        "# CELT ViSQOL quality comparison",
        "",
        "Official ViSQOL Audio scored matched random excerpts from the source masters.",
        "",
        "The codec input uses exact `f32` conversion from signed 24-bit PCM.",
        "ViSQOL input uses matched PCM16 copies with shared gain.",
        "",
        "## Overall result",
        "",
        f"- Matched pairs per implementation: {overall['pairs']}",
        f"- Rust mean MOS-LQO: {overall['rust_mean']:.4f}",
        f"- C mean MOS-LQO: {overall['c_mean']:.4f}",
        f"- Mean paired Rust minus C: {overall['paired_difference_mean']:+.4f}",
        "- Paired 95% confidence interval: "
        f"{overall['paired_difference_ci95'][0]:+.4f} to "
        f"{overall['paired_difference_ci95'][1]:+.4f}",
        f"- Pair wins: Rust {overall['rust_wins']}, tie {overall['ties']}, C {overall['c_wins']}",
        "",
        "## Configuration results",
        "",
        "| Configuration | Rust MOS | C MOS | Rust - C | 95% CI | Rust SNR | C SNR | Rust side SNR | C side SNR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, values in report["summary"]["configurations"].items():
        ci = values["paired_difference_ci95"]
        lines.append(
            f"| `{key}` | {values['rust_mean']:.4f} | {values['c_mean']:.4f} | "
            f"{values['paired_difference_mean']:+.4f} | {ci[0]:+.4f} to {ci[1]:+.4f} | "
            f"{values['rust_snr_mean']:.2f} dB | {values['c_snr_mean']:.2f} dB | "
            f"{values['rust_side_snr_mean']:.2f} dB | {values['c_side_snr_mean']:.2f} dB |"
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "ViSQOL Audio downmixes stereo to mono.",
            "Use the side-channel diagnostics and blind clips to check stereo artifacts.",
            "A ViSQOL result does not replace a controlled listening test.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    tracks = parse_tracks(args.track)
    bitrates = parse_int_list(args.bitrates, "bitrates")
    frame_sizes = parse_int_list(args.frame_sizes, "frame sizes")
    if any(frame_size not in VALID_FRAME_SIZES for frame_size in frame_sizes):
        raise ValueError(f"frame sizes must be in {sorted(VALID_FRAME_SIZES)}")
    modes = parse_modes(args.modes)
    configurations = [
        Configuration(frame_size=frame_size, bitrate=bitrate, mode=mode)
        for mode in modes
        for frame_size in frame_sizes
        for bitrate in bitrates
    ]
    rust_bin = args.rust_bin.expanduser().resolve()
    c_bin = args.c_bin.expanduser().resolve()
    # Keep the Bazel symlink path because the model is relative to its workspace.
    visqol = args.visqol.expanduser().absolute()
    output_root = args.out_dir.expanduser().resolve()
    for executable in [rust_bin, c_bin, visqol]:
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise ValueError(f"executable is unavailable: {executable}")
    if output_root.exists():
        raise ValueError(f"output directory already exists: {output_root}")
    if args.visqol_jobs <= 0:
        raise ValueError("--visqol-jobs must be positive")
    output_root.mkdir(parents=True)

    excerpt_frames = round(args.excerpt_seconds * SAMPLE_RATE)
    margin_frames = round(args.edge_margin_seconds * SAMPLE_RATE)
    if excerpt_frames <= 0 or excerpt_frames % max(frame_sizes) != 0:
        raise ValueError("excerpt length must contain an exact number of all requested frames")
    rng = random.Random(args.seed)
    source_rows = []
    excerpt_rows = []
    packet_rows = []
    pending_rows = []

    for track in tracks:
        info = sf.info(track.path)
        if info.samplerate != SAMPLE_RATE or info.channels != CHANNELS:
            raise ValueError(f"{track.path} must be 48 kHz stereo")
        source_hash = sha256_file(track.path)
        source_rows.append(
            {
                "name": track.name,
                "path": str(track.path),
                "sha256": source_hash,
                "frames": info.frames,
                "seconds": info.frames / SAMPLE_RATE,
                "subtype": info.subtype,
            }
        )
        print(f"[source] {track.name}: {info.frames / SAMPLE_RATE:.3f}s {info.subtype}", flush=True)
        audio, sample_rate = sf.read(track.path, dtype="float32", always_2d=True)
        if sample_rate != SAMPLE_RATE or audio.shape != (info.frames, CHANNELS):
            raise ValueError(f"unexpected decoded shape for {track.path}")
        starts = choose_starts(
            info.frames,
            excerpt_frames,
            margin_frames,
            args.excerpts_per_track,
            rng,
        )
        for excerpt_index, start in enumerate(starts, start=1):
            excerpt = audio[start : start + excerpt_frames]
            excerpt_key = f"{track.name}_{excerpt_index:02d}"
            excerpt_dir = output_root / "raw" / excerpt_key
            input_path = excerpt_dir / "master.f32le"
            write_f32le(input_path, excerpt)
            excerpt_rows.append(
                {
                    "track": track.name,
                    "excerpt": excerpt_index,
                    "start_frame": start,
                    "start_seconds": start / SAMPLE_RATE,
                    "frames": excerpt_frames,
                }
            )
            for config_index, configuration in enumerate(configurations, start=1):
                print(
                    f"[codec] {track.name} {excerpt_index}/{len(starts)} "
                    f"{config_index}/{len(configurations)} {configuration.key}",
                    flush=True,
                )
                config_dir = excerpt_dir / configuration.key
                config_dir.mkdir(parents=True, exist_ok=True)
                raw_paths = {
                    "rust": config_dir / "rust.f32le",
                    "c": config_dir / "c.f32le",
                }
                stats_by_treatment = {
                    "rust": run_json(
                        [
                            str(rust_bin),
                            str(configuration.frame_size),
                            str(configuration.bitrate),
                            configuration.mode,
                            str(input_path),
                            str(raw_paths["rust"]),
                        ]
                    ),
                    "c": run_json(
                        [
                            str(c_bin),
                            str(configuration.frame_size),
                            str(configuration.bitrate),
                            configuration.mode,
                            str(input_path),
                            str(raw_paths["c"]),
                        ]
                    ),
                }
                candidates = {
                    name: read_f32le(path, excerpt_frames) for name, path in raw_paths.items()
                }
                lags = {
                    name: estimate_lag(excerpt, candidate)
                    for name, candidate in candidates.items()
                }
                aligned_reference, aligned_candidates = aligned_audio(excerpt, candidates, lags)
                evaluation_paths = write_evaluation_wavs(
                    output_root / "evaluation" / excerpt_key / configuration.key,
                    aligned_reference,
                    aligned_candidates,
                    args.headroom_db,
                )
                for treatment in ["rust", "c"]:
                    codec_stats = stats_by_treatment[treatment]
                    packet_rows.append(
                        {
                            "track": track.name,
                            "excerpt": excerpt_index,
                            "configuration": configuration.key,
                            "treatment": treatment,
                            **codec_stats,
                            "effective_kbps": codec_stats["packet_bytes"]
                            * 8
                            / args.excerpt_seconds
                            / 1000,
                        }
                    )
                    pending_rows.append(
                        {
                            "track": track.name,
                            "excerpt": excerpt_index,
                            "start_seconds": start / SAMPLE_RATE,
                            "configuration": configuration.key,
                            "frame_size": configuration.frame_size,
                            "bitrate": configuration.bitrate,
                            "mode": configuration.mode,
                            "treatment": treatment,
                            "lag_frames": lags[treatment],
                            "lag_ms": lags[treatment] * 1000 / SAMPLE_RATE,
                            "diagnostics": diagnostics(
                                aligned_reference, aligned_candidates[treatment]
                            ),
                            "evaluation_paths": {
                                name: str(path) for name, path in evaluation_paths.items()
                            },
                        }
                    )
                if not args.keep_raw:
                    for raw_path in raw_paths.values():
                        raw_path.unlink()
                    config_dir.rmdir()
            if not args.keep_raw:
                input_path.unlink()
                excerpt_dir.rmdir()

    visqol_root = visqol.parent.parent
    visqol_model = (visqol_root / "model/libsvm_nu_svr_model.txt").resolve()
    if not visqol_model.is_file():
        raise ValueError(f"ViSQOL audio model is unavailable: {visqol_model}")
    pairs_path = output_root / "pairs.csv"
    with pairs_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["reference", "degraded"])
        for row in pending_rows:
            writer.writerow(
                [
                    row["evaluation_paths"]["master"],
                    row["evaluation_paths"][row["treatment"]],
                ]
            )
    job_count = min(args.visqol_jobs, len(pending_rows))
    shards = [[] for _ in range(job_count)]
    for row_index, row in enumerate(pending_rows):
        shards[row_index % job_count].append((row_index, row))
    print(
        f"[visqol] scoring {len(pending_rows)} files with {job_count} jobs",
        flush=True,
    )
    indexed_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=job_count) as executor:
        futures = [
            executor.submit(
                run_visqol_shard,
                shard_index,
                shard,
                visqol,
                visqol_root,
                visqol_model,
                output_root,
            )
            for shard_index, shard in enumerate(shards, start=1)
        ]
        for future_index, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            indexed_results.extend(future.result())
            print(f"[visqol] completed shard {future_index}/{job_count}", flush=True)
    indexed_results.sort(key=lambda item: item[0])
    visqol_rows = [row for _, row in indexed_results]
    if len(visqol_rows) != len(pending_rows):
        raise ValueError(
            f"ViSQOL returned {len(visqol_rows)} rows; expected {len(pending_rows)}"
        )
    for row, visqol_row in zip(pending_rows, visqol_rows, strict=True):
        row["mos_lqo"] = float(visqol_row["moslqo"])

    results_path = output_root / "visqol-results.csv"
    with results_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=visqol_rows[0].keys())
        writer.writeheader()
        writer.writerows(visqol_rows)

    summary = aggregate(pending_rows, configurations)
    repository_root = Path(__file__).resolve().parent.parent
    report = {
        "schema": "wavey.libopus-rs.visqol-quality",
        "schema_version": 1,
        "settings": {
            "sample_rate": SAMPLE_RATE,
            "channels": CHANNELS,
            "seed": args.seed,
            "excerpts_per_track": args.excerpts_per_track,
            "excerpt_seconds": args.excerpt_seconds,
            "edge_margin_seconds": args.edge_margin_seconds,
            "headroom_db": args.headroom_db,
            "bitrates": bitrates,
            "frame_sizes": frame_sizes,
            "modes": modes,
            "visqol_jobs": job_count,
        },
        "tools": {
            "rust_bin": str(rust_bin),
            "c_bin": str(c_bin),
            "visqol": str(visqol),
            "visqol_commit": git_value(visqol_root, "rev-parse", "HEAD"),
            "libopus_rs_commit": git_value(repository_root, "rev-parse", "HEAD"),
            "libopus_rs_worktree_dirty": bool(
                git_value(repository_root, "status", "--porcelain")
            ),
            "rust_bin_sha256": sha256_file(rust_bin),
            "c_bin_sha256": sha256_file(c_bin),
            "harness_sha256": sha256_file(Path(__file__).resolve()),
            "numpy_version": np.__version__,
            "soundfile_version": sf.__version__,
        },
        "sources": source_rows,
        "excerpts": excerpt_rows,
        "packets": packet_rows,
        "rows": pending_rows,
        "summary": summary,
    }
    (output_root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (output_root / "report.md").write_text(markdown_report(report))
    write_blind_set(output_root, pending_rows, args.blind_cases, args.seed)
    print(f"[done] {output_root / 'report.md'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
