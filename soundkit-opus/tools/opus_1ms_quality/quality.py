#!/usr/bin/env python3
"""Direct 1 ms Rust vs 2.5 ms C comparison on local WAV fixtures."""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import struct
import subprocess

import numpy as np
from scipy import signal
from scipy.io import wavfile

FS = 48_000
SCORE_SAMPLES = FS * 8
INPUT_SAMPLES = SCORE_SAMPLES + 960
SOURCES = {
    "27_castanets_single_tone_rhythm_stereo": 6,
    "31_cymbal": 3,
    "49_female_speech_english_mono": 5,
    "59_violin_ravel_stereo": 4,
    "65_orchestra_strauss_stereo": 20,
    "allegaeon-beasts-and-worms": 22,
    "four_tet_baby": 20,
    "julien_baker_sprained_ankle": 12,
}
DELAYS = {"rust": 48, "custom": 48, "lowdelay": 120, "audio": 312}


def command(arguments, **kwargs):
    return subprocess.run(list(map(str, arguments)), check=True, capture_output=True,
                          timeout=60, **kwargs).stdout


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def prepare(work_dir, source_dir, names):
    work = work_dir / "inputs"
    work.mkdir(exist_ok=True)
    manifest = []
    for name in names:
        start = SOURCES[name]
        dst = work / f"{name}.s32le"
        original = source_dir / f"{name}.wav"
        pcm = command([
            "ffmpeg", "-v", "error", "-ss", start, "-i", original, "-ac", 2, "-af",
            "aresample=48000:resampler=soxr", "-t", INPUT_SAMPLES / FS,
            "-f", "f32le", "-",
        ])
        samples = np.frombuffer(pcm, "<f4").reshape(-1, 2)
        if len(samples) != INPUT_SAMPLES or not np.isfinite(samples).all():
            raise ValueError(f"Unexpected input samples for {name}")
        peak = np.max(np.abs(samples))
        gain = min(1.0, 10 ** (-1 / 20) / peak) if peak else 1.0
        i24 = np.rint(samples.astype(np.float64) * gain * (1 << 23)).astype("<i4")
        i24.tofile(dst)
        reference = i24[:SCORE_SAMPLES].astype(np.float32) / np.float32(1 << 23)
        wavfile.write(work / f"{name}.wav", FS, reference)
        manifest.append({
            "source": name,
            "offset_seconds": start,
            "file_sha256": sha256(original),
            "pcm_sha256": sha256(dst),
        })
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return work


def read_packets(path):
    data = path.read_bytes()
    i = 0
    while i < len(data):
        if len(data) - i < 2:
            raise ValueError(f"Truncated packet length in {path}")
        length = struct.unpack_from("<H", data, i)[0]
        start = i + 2
        i = start + length
        if i > len(data):
            raise ValueError(f"Truncated packet in {path}")
        yield memoryview(data)[start:i]


def parse_packets(path):
    lengths = [len(packet) for packet in read_packets(path)]
    if not lengths:
        raise ValueError(f"No packets in {path}")
    return sum(lengths), len(lengths), min(lengths), max(lengths)


def observed_delay(reference, decode):
    correlation = None
    for channel in range(reference.shape[1]):
        ref = reference[:FS * 2, channel].astype(np.float64)
        deg = decode[:FS * 2, channel].astype(np.float64)
        corr = signal.correlate(deg, ref, method="fft")[len(ref) - 1:len(ref) + 479]
        correlation = corr if correlation is None else correlation + corr
    return int(correlation.argmax())


def gstpeaq(args, reference, decode, advanced):
    cmd = [args.peaq, "--advanced" if advanced else "--basic", reference, decode]
    env = dict(os.environ, GST_REGISTRY=str(args.work_dir / "gst.cache"))
    if args.gst_plugin_dir:
        env["GST_PLUGIN_PATH"] = str(args.gst_plugin_dir)
    for attempt in range(3):
        try:
            out = subprocess.run(list(map(str, cmd)), check=True, capture_output=True,
                                 timeout=6, env=env, text=True).stdout
            break
        except subprocess.TimeoutExpired:
            if attempt == 2:
                raise
    odg = float(re.search(r"Objective Difference Grade: ([0-9.+-]+)", out)[1])
    di = float(re.search(r"Distortion Index: ([0-9.+-]+)", out)[1])
    return di, odg


def score(args, inputs, codec, name, rate, mode):
    label = args.label if codec == "rust" else f"c_{codec}"
    work = args.output / f"{label}-{mode}-{rate}"
    work.mkdir(exist_ok=True)
    raw_in = inputs / f"{name}.s32le"
    reference_wav = inputs / f"{name}.wav"
    packets = work / f"{name}.packets"
    raw_out = work / f"{name}.raw"
    decode_wav = work / f"{name}.wav"
    frame = 48 if codec in ("rust", "custom") else 120
    if codec in ("rust", "custom"):
        binary = args.rust if codec == "rust" else args.work_dir / "c_custom"
        common = ["--frame", frame, "--channels", 2, "--pcm", "s24"]
        options = args.encode_options if codec == "rust" else []
        command([binary, "encode", *common, "--bitrate", rate, "--vbr", int(mode == "vbr"),
                 "--cvbr", 1, *options, raw_in, packets])
        command([binary, "decode", *common, packets, raw_out])
        total, count, min_bytes, max_bytes = parse_packets(packets)
        decoded = np.fromfile(raw_out, "<i4").reshape(-1, 2).astype(np.float32) / np.float32(1 << 23)
    else:
        metadata = json.loads(command([args.work_dir / f"c_{codec}", "--pcm-bits", 24, frame, rate,
                                       mode, raw_in, raw_out]))
        total, count, min_bytes, max_bytes = (metadata[key] for key in
                                              ("packet_bytes", "packets", "packet_min", "packet_max"))
        decoded = np.fromfile(raw_out, "<f4").reshape(-1, 2)
    reference = wavfile.read(reference_wav)[1]
    delay = DELAYS[codec]
    measured_delay = observed_delay(reference, decoded)
    aligned = decoded[delay:delay + SCORE_SAMPLES]
    if aligned.shape != reference.shape or not np.isfinite(aligned).all():
        raise ValueError(f"Invalid decoded length or samples for {codec}, {name}, {rate}, {mode}")
    wavfile.write(decode_wav, FS, aligned)
    advanced_di, advanced_odg = gstpeaq(args, reference_wav, decode_wav, True)
    basic_di, basic_odg = gstpeaq(args, reference_wav, decode_wav, False)
    return {
        "label": label,
        "track": name,
        "mode": mode,
        "target_kbps": rate / 1000,
        "frame": frame,
        "delay_samples": delay,
        "xcorr_delay_samples": measured_delay,
        "kbps": total * 8 * FS / (1000 * count * frame),
        "packets_per_s": FS / frame,
        "min_bytes": min_bytes,
        "max_bytes": max_bytes,
        "advanced_di": advanced_di,
        "advanced_odg": advanced_odg,
        "basic_di": basic_di,
        "basic_odg": basic_odg,
        "snr_db": float(10 * np.log10(np.sum(reference ** 2) / np.sum((reference - aligned) ** 2))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rust", type=Path)
    parser.add_argument("--peaq", type=Path, required=True)
    parser.add_argument("--gst-plugin-dir", type=Path)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--label", default="rust")
    parser.add_argument("--codecs", default="rust,custom,lowdelay,audio")
    parser.add_argument("--rates", default="192000,256000,320000")
    parser.add_argument("--modes", default="cbr")
    parser.add_argument("--sources", default=",".join(SOURCES))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encode-options", default="")
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    args.work_dir = args.work_dir.resolve()
    args.peaq = args.peaq.resolve()
    if args.rust:
        args.rust = args.rust.resolve()
    codecs = args.codecs.split(",")
    modes = args.modes.split(",")
    sources = args.sources.split(",")
    rates = list(map(int, args.rates.split(",")))
    if set(codecs) - DELAYS.keys() or ("rust" in codecs and args.rust is None):
        parser.error("Choose codecs from rust,custom,lowdelay,audio and pass --rust for Rust")
    if set(sources) - SOURCES.keys() or set(modes) - {"cbr", "vbr"}:
        parser.error("Choose known sources and modes from cbr,vbr")
    if min(rates) <= 0 or args.jobs <= 0:
        parser.error("Bitrates and the number of jobs must be positive")
    args.encode_options = shlex.split(args.encode_options)
    args.output.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    inputs = prepare(args.work_dir, args.source_dir, sources)
    sweep = [
        (codec, track, rate, mode)
        for codec in codecs
        for rate in rates
        for mode in modes
        for track in sources
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(score, args, inputs, *cell): cell for cell in sweep}
        completed = []
        for future in concurrent.futures.as_completed(futures):
            try:
                row = future.result()
            except subprocess.CalledProcessError as error:
                print(error.stderr, flush=True)
                raise
            completed.append(row)
            print(f"{row['label']} {row['mode']} {row['target_kbps']:g} {row['track']} "
                  f"{row['advanced_di']:.3f}", flush=True)
    completed.sort(key=lambda row: (row["label"], row["mode"], row["target_kbps"], row["track"]))
    with (args.output / "results.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(completed[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(completed)


if __name__ == "__main__":
    main()
