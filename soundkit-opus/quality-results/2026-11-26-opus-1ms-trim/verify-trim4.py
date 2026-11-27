#!/usr/bin/env python3
"""Verify compatibility and implicit signalling at selected trim rates."""
import argparse
import csv
from pathlib import Path
from tempfile import TemporaryDirectory

from quality import FS, SOURCES, command, parse_packets, read_packets

RATES = (248_000, 312_000, 344_000, 368_000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rust", type=Path, required=True)
    parser.add_argument("--c-custom", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--renders", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    common = ["--frame", 48, "--channels", 2, "--pcm", "s24"]
    with TemporaryDirectory() as directory, args.output.open("w") as output:
        work = Path(directory)
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(("track", "kbps", "packets", "static_encoder_matches",
                         "c_decode_matches", "payload_matches", "implicit_decode_matches",
                         "payload_saving_bps"))
        for rate in RATES:
            for track in sorted(SOURCES):
                source = args.inputs / f"{track}.s32le"
                expected = args.renders / f"trim4-cbr-{rate}" / f"{track}.packets"
                expected_pcm = expected.with_suffix(".raw").read_bytes()
                on = work / "signalled.packets"
                off = work / "implicit.packets"
                decoded = work / "decode.pcm"
                command([args.rust, "encode", *common, "--bitrate", rate, source, on])
                same_packets = on.read_bytes() == expected.read_bytes()
                command([args.c_custom, "decode", *common, on, decoded])
                same_c_decode = expected_pcm == decoded.read_bytes()
                command([args.rust, "encode", *common, "--signalling", 0, "--bitrate",
                         rate - 8000, source, off])
                command([args.rust, "decode", *common, "--signalling", 0, off, decoded])
                same_implicit_decode = expected_pcm == decoded.read_bytes()
                same_payload = all(
                    signalled[1:] == implicit
                    for signalled, implicit in zip(read_packets(on), read_packets(off), strict=True)
                )
                on_bytes, packets, _, _ = parse_packets(on)
                off_bytes, _, _, _ = parse_packets(off)
                saving_bps = (on_bytes - off_bytes) * 8 * FS // (packets * 48)
                writer.writerow((track, rate // 1000, packets, same_packets, same_c_decode,
                                 same_payload, same_implicit_decode, saving_bps))
                if not all((same_packets, same_c_decode, same_payload, same_implicit_decode,
                            saving_bps == 8000)):
                    raise ValueError(f"Packet check failed at {rate} b/s for {track}")


if __name__ == "__main__":
    main()
