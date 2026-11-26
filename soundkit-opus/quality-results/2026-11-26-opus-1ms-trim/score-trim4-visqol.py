#!/usr/bin/env python3
"""Compare the trim experiment and C at their strict mean advanced DI pairs."""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from visqol import VisqolApi

PAIRS = ((192, 248), (256, 312), (320, 368))


def mono(path):
    fs, pcm = wavfile.read(path)
    if fs != 48_000 or pcm.ndim != 2 or pcm.shape[1] != 2:
        raise ValueError(f"Expected 48 kHz stereo PCM in {path}")
    return pcm.mean(axis=1).astype(np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--rust-frames", type=Path, required=True)
    parser.add_argument("--c-frames", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    api = VisqolApi()
    api.create(mode="audio")
    with args.output.open("w") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("track", "rust_kbps", "c_kbps", "rust_moslqo", "c_moslqo"))
        for reference in sorted(args.references.glob("*.wav")):
            ref = mono(reference)
            for c_rate, rust_rate in PAIRS:
                rust = args.rust_frames / f"trim4-cbr-{rust_rate * 1000}" / reference.name
                c = args.c_frames / f"c_lowdelay-cbr-{c_rate * 1000}" / reference.name
                mos_rust = api.measure_from_arrays(ref, mono(rust), 48_000).moslqo
                mos_c = api.measure_from_arrays(ref, mono(c), 48_000).moslqo
                writer.writerow((reference.stem, rust_rate, c_rate, mos_rust, mos_c))
                handle.flush()


if __name__ == "__main__":
    main()
