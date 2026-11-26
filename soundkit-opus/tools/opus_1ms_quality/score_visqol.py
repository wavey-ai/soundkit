#!/usr/bin/env python3
"""Score the delay-aligned renders with the Python ViSQOL reimplementation."""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from visqol import VisqolApi

from quality import FS, SOURCES

RENDERS = (
    "inputs",
    "direct/rust-cbr-192000",
    "direct/rust-cbr-256000",
    "rate_sweep/rust-cbr-264000",
    "direct/rust-cbr-320000",
    "rate_sweep/rust-cbr-384000",
    "direct/c_lowdelay-cbr-192000",
    "direct/c_lowdelay-cbr-256000",
    "direct/c_lowdelay-cbr-320000",
)


def read_mono(path):
    _, pcm = wavfile.read(path)
    return pcm.mean(axis=1).astype(np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()

    api = VisqolApi()
    api.create(mode="audio")
    with (args.work_dir / "visqol-python.csv").open("w") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(("source", "config", "moslqo"))
        for name in SOURCES:
            filename = f"{name}.wav"
            reference = read_mono(args.work_dir / "inputs" / filename)
            for path in RENDERS:
                decoded = read_mono(args.work_dir / path / filename)
                score = api.measure_from_arrays(reference, decoded, FS).moslqo
                writer.writerow((name, path, score))
                print(name, path, f"{score:.6f}", flush=True)


if __name__ == "__main__":
    main()
