#!/usr/bin/env python3
"""Check C Custom parity and the payload savings from implicit signalling."""
import argparse
import csv
from pathlib import Path

from quality import SOURCES, command, parse_packets, read_packets


def verify_custom(work_dir):
    count = 0
    for rate in (192_000, 256_000, 320_000):
        rust = work_dir / "direct" / f"rust-cbr-{rate}"
        c = work_dir / "direct" / f"c_custom-cbr-{rate}"
        for name in SOURCES:
            for suffix in ("packets", "raw"):
                filename = f"{name}.{suffix}"
                if (rust / filename).read_bytes() != (c / filename).read_bytes():
                    raise ValueError(f"Different Rust and C Custom {suffix} for {name} at {rate}")
            _, packets, _, _ = parse_packets(rust / f"{name}.packets")
            count += packets
    print(f"Rust and C Custom matched on {count:,} packets and all decoded i24 samples.")


def verify_signalling(work_dir, rust):
    output = work_dir / "signalling"
    output.mkdir(exist_ok=True)
    with (work_dir / "signalling.csv").open("w") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("mode", "target_kbps", "source", "signalled_bits_per_second",
                         "implicit_bits_per_second", "payload_matches", "decode_matches"))
        for mode, run in (("cbr", "direct"), ("vbr", "cvbr")):
            for rate in (192_000, 256_000, 320_000):
                for name in SOURCES:
                    signalled = work_dir / run / f"rust-{mode}-{rate}" / f"{name}.packets"
                    source = work_dir / "inputs" / f"{name}.s32le"
                    implicit = output / f"rust-{mode}-{rate - 8000}-{name}.packets"
                    decoded = implicit.with_suffix(".raw")
                    common = ["--frame", 48, "--channels", 2, "--pcm", "s24", "--signalling", 0]
                    command([rust, "encode", *common, "--bitrate", rate - 8000,
                             "--vbr", int(mode == "vbr"), "--cvbr", 1, source, implicit])
                    command([rust, "decode", *common, implicit, decoded])
                    on_bytes, on_count, _, _ = parse_packets(signalled)
                    off_bytes, off_count, _, _ = parse_packets(implicit)
                    payload_matches = all(
                        on[1:] == off
                        for on, off in zip(read_packets(signalled), read_packets(implicit), strict=True)
                    )
                    decode_matches = decoded.read_bytes() == signalled.with_suffix(".raw").read_bytes()
                    writer.writerow((mode, rate // 1000, name, f"{on_bytes * 8000 / on_count:.3f}",
                                     f"{off_bytes * 8000 / off_count:.3f}", payload_matches, decode_matches))
                    if not payload_matches or not decode_matches or on_bytes - off_bytes != on_count:
                        raise ValueError(f"Implicit signalling comparison failed for {mode}, {name}, {rate}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--rust", type=Path, required=True)
    args = parser.parse_args()
    verify_custom(args.work_dir)
    verify_signalling(args.work_dir, args.rust.resolve())


if __name__ == "__main__":
    main()
