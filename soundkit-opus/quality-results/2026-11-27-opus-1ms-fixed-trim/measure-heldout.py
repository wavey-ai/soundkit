#!/usr/bin/env python3
"""Reproduce the fixed and adaptive trim comparison on seven later passages."""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from quality import prepare, score

OFFSETS = {
    "31_cymbal": 45,
    "49_female_speech_english_mono": 14,
    "59_violin_ravel_stereo": 16,
    "65_orchestra_strauss_stereo": 52,
    "allegaeon-beasts-and-worms": 55,
    "four_tet_baby": 40,
    "julien_baker_sprained_ankle": 31,
}
RATES = {
    "fixed8": (256, 272, 320, 328, 336, 344, 352, 368, 384, 400, 432),
    "trim4": (256, 320, 328, 336, 344, 352, 368, 384, 400),
    "c_lowdelay": (192, 256, 320),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--peaq", type=Path, required=True)
    parser.add_argument("--gst-plugin-dir", type=Path)
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--adaptive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # The parent harness uses a built-in source-offset map for the earlier
    # excerpts. Use the same preparation method with the later offsets.
    import quality
    quality.SOURCES = OFFSETS
    inputs = prepare(args.output, args.source_dir, list(OFFSETS))
    tasks = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for label, rates in RATES.items():
            binary = args.fixed if label == "fixed8" else args.adaptive
            options = SimpleNamespace(
                work_dir=args.work_dir.resolve(),
                output=args.output,
                rust=binary.resolve(),
                peaq=args.peaq.resolve(),
                gst_plugin_dir=args.gst_plugin_dir,
                label=label,
                encode_options=[],
            )
            codec = "lowdelay" if label == "c_lowdelay" else "rust"
            for rate in rates:
                for source in OFFSETS:
                    tasks.append(pool.submit(score, options, inputs, codec, source, rate * 1000, "cbr"))
        rows = [task.result() for task in tasks]
    rows.sort(key=lambda row: (row["label"], float(row["target_kbps"]), row["track"]))
    with (args.output / "results.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
