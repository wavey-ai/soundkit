#!/usr/bin/env python3
"""Count instructions in paired native Opus Custom 1 ms runs with Callgrind."""

import argparse
import csv
import hashlib
from pathlib import Path
import re
import subprocess


def profile(binary, operation, arguments, log, valgrind):
    subprocess.run(
        [valgrind, "--tool=callgrind", f"--callgrind-out-file={log}",
         binary, operation, "--frame", "48", "--channels", "2", "--pcm", "s24",
         *arguments],
        check=True, capture_output=True,
    )
    summary = re.search(r"^summary: (\d+)\s*$", log.read_text(), re.MULTILINE)
    if summary is None:
        raise RuntimeError(f"no instruction count in {log}")
    return int(summary[1])


def run_case(binary, source, rate, output, valgrind):
    packets = output.with_name(f"{output.name}.packets")
    pcm = output.with_name(f"{output.name}.s32le")
    encode = profile(
        binary, "encode", ["--bitrate", str(rate * 1000), "--vbr", "0", source, packets],
        output.with_name(f"{output.name}.encode.callgrind"), valgrind,
    )
    decode = profile(
        binary, "decode", [packets, pcm], output.with_name(f"{output.name}.decode.callgrind"), valgrind,
    )
    return (
        encode, decode,
        hashlib.sha256(packets.read_bytes()).hexdigest(),
        hashlib.sha256(pcm.read_bytes()).hexdigest(),
    )


def rate_pair(value):
    """Read equal rates as one kb/s number or different rates as BASE:CANDIDATE."""
    try:
        baseline, _, candidate = value.partition(":")
        rates = int(baseline), int(candidate or baseline)
        if min(rates) <= 0:
            raise ValueError
        return rates
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected positive rates in kb/s") from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--rate", type=rate_pair, action="append", required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--valgrind", default="valgrind")
    parser.add_argument("--require-identical", action="store_true")
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    cases = ((source, rates) for source in args.input for rates in args.rate)
    with (args.results / "instruction-counts.csv").open("w", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(("source", "baseline_kbps", "candidate_kbps",
                         "baseline_encode_ir", "baseline_decode_ir", "baseline_packet_sha256", "baseline_pcm_sha256",
                         "candidate_encode_ir", "candidate_decode_ir", "candidate_packet_sha256", "candidate_pcm_sha256"))
        for index, (source, rates) in enumerate(cases, 1):
            baseline_rate, candidate_rate = rates
            key = f"{index:02d}-{source.stem}-{baseline_rate}-{candidate_rate}"
            baseline = run_case(
                args.baseline.resolve(), source, baseline_rate,
                args.results / f"{key}-baseline", args.valgrind,
            )
            candidate = run_case(
                args.candidate.resolve(), source, candidate_rate,
                args.results / f"{key}-candidate", args.valgrind,
            )
            if args.require_identical and baseline[2:] != candidate[2:]:
                raise RuntimeError(f"packet or PCM mismatch for {key}")
            writer.writerow((source.stem, *rates, *baseline, *candidate))
            print(key, flush=True)


if __name__ == "__main__":
    main()
