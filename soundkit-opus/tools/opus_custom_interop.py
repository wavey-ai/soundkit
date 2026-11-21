#!/usr/bin/env python3
"""Opus Custom interop harness: soundkit-opus against libopus custom modes.

Subcommands:

  synth OUTDIR [--seconds S]
      Writes deterministic synthetic test signals (stereo and mono) in the
      three PCM formats: OUTDIR/synth-{stereo,mono}.{s16,s24,f32}.

  convert IN.s32raw OUTPREFIX
      Converts 48 kHz stereo signed-32 PCM holding 24-bit samples (as
      `ffmpeg -f s32le` writes it) to OUTPREFIX-{stereo,mono}.{s16,s24,f32}.

  run --c C_TOOL --rust RUST_TOOL --data DIR --out DIR [--inputs a,b]
      [--frames 48,64,72,120,240] [--rates 192000,256000,320000]
      [--modes cbr,cvbr] [--formats s16,s24,f32] [--signalling 1,0]
      [--rust-extra OPTIONS] [--jobs N]
      Runs the interop matrix and writes OUT/results.tsv and OUT/summary.md:
        a  Rust encoder -> libopus and Rust decoders, PCM compared
        b  libopus encoder -> libopus and Rust decoders, PCM compared
        c  packet identity, Rust encoder vs opus_custom_encode
        d  deterministic packet loss, the two decoders' PLC compared

Both tools take the arguments of tools/opus_custom_interop.c. No numpy is
needed. Real-music inputs stay in the lab; only synthetic signals are fit
for the repository.
"""

import argparse
import array
import math
import os
import struct
import subprocess
import sys
from pathlib import Path

RATE = 48_000
FORMATS = ("s16", "s24", "f32")


def write_pcm(path, samples, fmt):
    """samples: list of floats in [-1, 1) at 24-bit precision."""
    if fmt == "s16":
        data = array.array("h", (max(-32768, min(32767, int(math.floor(s * 32768.0)))) for s in samples))
    elif fmt == "s24":
        data = array.array("i", (max(-8388608, min(8388607, int(math.floor(s * 8388608.0)))) for s in samples))
    else:
        data = array.array("f", (math.floor(s * 8388608.0) / 8388608.0 for s in samples))
    if sys.byteorder != "little":
        data.byteswap()
    Path(path).write_bytes(data.tobytes())


def lcg(seed):
    return (1664525 * seed + 1013904223) & 0xFFFFFFFF


def synth_signal(seconds, channels):
    """A deterministic mixed signal: tones, a chirp, noise bursts, clicks,
    a pure-tone stretch and two stretches of digital silence."""
    n = int(seconds * RATE)
    out = []
    seed = 12345
    for i in range(n):
        t = i / RATE
        seg = int(t * 4) % 8
        frame = []
        for c in range(channels):
            if t < 0.05 or 1.5 <= t < 1.55:
                v = 0.0
            elif seg == 6:
                v = 0.5 * math.sin(2 * math.pi * 997.0 * t + c * 0.3)
            else:
                v = 0.25 * math.sin(2 * math.pi * (220.0 + 110.0 * c) * t)
                v += 0.15 * math.sin(2 * math.pi * 3150.0 * t + 0.5 * c)
                v += 0.1 * math.sin(2 * math.pi * (500.0 + 4000.0 * (t % 1.0)) * t)
                seed = lcg(seed)
                noise = ((seed >> 8) / 16777216.0) - 0.5
                burst = 1.0 if (int(t * 10) % 3 == 1) else 0.05
                v += 0.3 * noise * burst
                if i % 4800 == 0:
                    v += 0.6
                if seg == 3:
                    v *= 0.02
            frame.append(max(-0.99, min(0.99, v)))
        out.extend(frame)
    return out


def cmd_synth(args):
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    for channels, name in ((2, "stereo"), (1, "mono")):
        samples = synth_signal(args.seconds, channels)
        for fmt in FORMATS:
            write_pcm(out / f"synth-{name}.{fmt}", samples, fmt)


def cmd_convert(args):
    raw = array.array("i")
    raw.frombytes(Path(args.input).read_bytes())
    if sys.byteorder != "little":
        raw.byteswap()
    stereo = [v / 2147483648.0 for v in raw]
    mono = [(stereo[2 * i] + stereo[2 * i + 1]) * 0.5 for i in range(len(stereo) // 2)]
    for samples, name in ((stereo, "stereo"), (mono, "mono")):
        for fmt in FORMATS:
            write_pcm(f"{args.prefix}-{name}.{fmt}", samples, fmt)


def read_packets(path):
    data = Path(path).read_bytes()
    packets = []
    pos = 0
    while pos + 2 <= len(data):
        n = data[pos] | (data[pos + 1] << 8)
        packets.append(data[pos + 2 : pos + 2 + n])
        pos += 2 + n
    return packets


def write_packets(path, packets):
    out = bytearray()
    for p in packets:
        out += struct.pack("<H", len(p)) + p
    Path(path).write_bytes(bytes(out))


def loss_pattern(count, seed=7):
    """Deterministic losses: about 5% isolated drops plus bursts of 2, 3 and
    50 frames (the last one long enough to reach the noise concealment)."""
    lost = [False] * count
    s = seed
    for i in range(count):
        s = lcg(s)
        if (s >> 16) % 20 == 0:
            lost[i] = True
    for start, length in ((count // 5, 2), (count // 3, 3), (count // 2, 50)):
        for i in range(start, min(count, start + length)):
            lost[i] = True
    lost[0] = False
    return lost


def read_pcm(path, fmt):
    code = {"s16": "h", "s24": "i", "f32": "f"}[fmt]
    data = array.array(code)
    data.frombytes(Path(path).read_bytes())
    if sys.byteorder != "little":
        data.byteswap()
    return data


def compare_pcm(a_path, b_path, fmt):
    a = read_pcm(a_path, fmt)
    b = read_pcm(b_path, fmt)
    if len(a) != len(b):
        return {"identical": False, "differ": -1, "max": float("nan"), "count": max(len(a), len(b))}
    if a.tobytes() == b.tobytes():
        return {"identical": True, "differ": 0, "max": 0, "count": len(a)}
    differ = 0
    worst = 0.0
    scale = {"s16": 1.0, "s24": 1.0, "f32": 32768.0}[fmt]
    for x, y in zip(a, b):
        if x != y:
            differ += 1
            worst = max(worst, abs(x - y) * scale)
    return {"identical": False, "differ": differ, "max": worst, "count": len(a)}


def compare_packets(a_path, b_path):
    a = read_packets(a_path)
    b = read_packets(b_path)
    same = 0
    first = None
    for i, (x, y) in enumerate(zip(a, b)):
        if x == y:
            same += 1
        elif first is None:
            first = i
    if len(a) != len(b) and first is None:
        first = min(len(a), len(b))
    return same, max(len(a), len(b)), first


def run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(" ".join(cmd) + "\n" + result.stderr)


def cmd_run(args):
    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    work.mkdir(exist_ok=True)
    frames = [int(f) for f in args.frames.split(",")]
    inputs = args.inputs.split(",")
    rates = args.rates.split(",")
    all_modes = {
        "cbr": ["--vbr", "0"],
        "cvbr": ["--vbr", "1", "--cvbr", "1"],
        "uvbr": ["--vbr", "1", "--cvbr", "0"],
    }
    modes = [(m, all_modes[m]) for m in args.modes.split(",")]
    formats = args.formats.split(",")
    signalling = [int(v) for v in args.signalling.split(",")]
    rust_extra = args.rust_extra.split() if args.rust_extra else []
    cases = []
    for name in inputs:
        for channels, layout in ((2, "stereo"), (1, "mono")):
            for frame in frames:
                for fmt in formats:
                    pcm = data / f"{name}-{layout}.{fmt}"
                    if not pcm.exists():
                        continue
                    for rate in rates:
                        for mode_name, mode_args in modes:
                            for sig in signalling:
                                rate_tag = "max" if rate == "max" else f"{int(rate) // 1000}k"
                                tag = f"{name}-{layout}-f{frame}-{fmt}-{rate_tag}-{mode_name}-s{sig}"
                                common = ["--frame", str(frame), "--channels", str(channels),
                                          "--signalling", str(sig), "--pcm", fmt]
                                enc = common + ["--bitrate", rate] + mode_args
                                cases.append((tag, pcm, enc, common, fmt))
    from concurrent.futures import ThreadPoolExecutor
    rows = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_case, args, tag, pcm, work, enc, common, fmt, rust_extra)
                   for tag, pcm, enc, common, fmt in cases]
        for future in futures:
            row = future.result()
            rows.append(row)
            print("\t".join(str(v) for v in row.values()), flush=True)
    keys = list(rows[0].keys())
    with open(out / "results.tsv", "w") as f:
        f.write("\t".join(keys) + "\n")
        for r in rows:
            f.write("\t".join(str(r[k]) for k in keys) + "\n")
    write_summary(out / "summary.md", rows, args)


def write_summary(path, rows, args):
    """One line per frame size and channel layout: counts of cases whose
    decoded PCM is identical, and packet identity totals."""
    groups = {}
    for r in rows:
        parts = r["case"].split("-")
        key = (int(parts[2][1:]), parts[1])
        g = groups.setdefault(key, {"cases": 0, "a": 0, "b": 0, "d": 0, "same": 0, "packets": 0,
                                   "worst": []})
        g["cases"] += 1
        for label in ("a", "b", "d"):
            if r[label] == "identical":
                g[label] += 1
            else:
                g["worst"].append(f"{label} {r['case']}: {r[label]}")
        same, total = (int(v) for v in r["c_identical_packets"].split("/"))
        g["same"] += same
        g["packets"] += total
    lines = [f"# Opus Custom interop: {args.out}", "",
             f"Inputs {args.inputs}; rates {args.rates}; modes {args.modes}; "
             f"formats {args.formats}; signalling {args.signalling}; Rust options "
             f"'{args.rust_extra or 'default'}'.", "",
             "| Frame | Channels | Cases | (a) Rust packets, PCM identical | "
             "(b) libopus packets, PCM identical | (d) with losses, PCM identical | "
             "(c) identical packets |",
             "| ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for (frame, layout), g in sorted(groups.items()):
        pct = 100.0 * g["same"] / g["packets"] if g["packets"] else 0.0
        lines.append(f"| {frame} | {layout} | {g['cases']} | {g['a']} | {g['b']} | {g['d']} | "
                     f"{g['same']}/{g['packets']} ({pct:.3f}%) |")
    worst = [w for g in groups.values() for w in g["worst"]]
    if worst:
        lines += ["", "Non-identical decodes:", ""] + [f"- {w}" for w in worst[:200]]
    Path(path).write_text("\n".join(lines) + "\n")


def describe(result):
    if result["identical"]:
        return "identical"
    if result["differ"] < 0:
        return "length-mismatch"
    return f"{result['differ']}/{result['count']} max {result['max']:.3g}"


def run_case(args, tag, pcm, work, enc_args, dec_args, fmt, rust_extra=()):
    c_tool, r_tool = args.c, args.rust
    rust_extra = list(rust_extra)
    rp = work / f"{tag}.rust.pkt"
    cp = work / f"{tag}.c.pkt"
    run([r_tool, "encode"] + enc_args + rust_extra + [str(pcm), str(rp)])
    run([c_tool, "encode"] + enc_args + ["--complexity", "10", str(pcm), str(cp)])
    row = {"case": tag}
    # (c) packet identity
    same, total, first = compare_packets(rp, cp)
    row["c_identical_packets"] = f"{same}/{total}"
    row["c_first_mismatch"] = "-" if first is None else str(first)
    # (a) Rust packets, (b) libopus packets, (d) libopus packets with losses
    lossy = work / f"{tag}.c.lossy.pkt"
    packets = read_packets(cp)
    pattern = loss_pattern(len(packets))
    write_packets(lossy, [b"" if lost else p for p, lost in zip(packets, pattern)])
    for label, pkt in (("a", rp), ("b", cp), ("d", lossy)):
        out_c = work / f"{tag}.{label}.c.{fmt}"
        out_r = work / f"{tag}.{label}.rust.{fmt}"
        run([c_tool, "decode"] + dec_args + [str(pkt), str(out_c)])
        run([r_tool, "decode"] + dec_args + rust_extra + [str(pkt), str(out_r)])
        row[label] = describe(compare_pcm(out_c, out_r, fmt))
        if not args.keep:
            out_c.unlink()
            out_r.unlink()
    if not args.keep:
        for p in (rp, cp, lossy):
            p.unlink()
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("synth")
    p.add_argument("outdir")
    p.add_argument("--seconds", type=float, default=3.0)
    p.set_defaults(func=cmd_synth)
    p = sub.add_parser("convert")
    p.add_argument("input")
    p.add_argument("prefix")
    p.set_defaults(func=cmd_convert)
    p = sub.add_parser("run")
    p.add_argument("--c", required=True)
    p.add_argument("--rust", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--inputs", default="synth")
    p.add_argument("--frames", default="48,64,72,120,240")
    p.add_argument("--rates", default="192000,256000,320000")
    p.add_argument("--modes", default="cbr,cvbr")
    p.add_argument("--formats", default="s16,s24,f32")
    p.add_argument("--signalling", default="1,0")
    p.add_argument("--rust-extra", default="",
                   help="extra Rust tool options, e.g. '--arith soundkit --decisions soundkit'")
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--keep", action="store_true")
    p.set_defaults(func=cmd_run)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
