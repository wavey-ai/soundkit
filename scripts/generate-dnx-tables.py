#!/usr/bin/env python3
"""Regenerate the LGPL DNxHD/DNxHR codec tables from a pinned FFmpeg checkout."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

SOURCE_REVISION = "ca821e458aabe2fa211d9e94eac38cd69fe2ea09"
TABLES = (
    ("dnxhd_1237_luma_weight", "u8"),
    ("dnxhd_1237_chroma_weight", "u8"),
    ("dnxhd_1238_luma_weight", "u8"),
    ("dnxhd_1238_chroma_weight", "u8"),
    ("dnxhd_1241_luma_weight", "u8"),
    ("dnxhd_1241_chroma_weight", "u8"),
    ("dnxhd_1235_dc_codes", "u8"),
    ("dnxhd_1235_dc_bits", "u8"),
    ("dnxhd_1237_dc_codes", "u8"),
    ("dnxhd_1237_dc_bits", "u8"),
    ("dnxhd_1235_ac_codes", "u16"),
    ("dnxhd_1235_ac_bits", "u8"),
    ("dnxhd_1235_ac_info", "u8"),
    ("dnxhd_1237_ac_codes", "u16"),
    ("dnxhd_1237_ac_bits", "u8"),
    ("dnxhd_1237_ac_info", "u8"),
    ("dnxhd_1238_ac_codes", "u16"),
    ("dnxhd_1238_ac_bits", "u8"),
    ("dnxhd_1238_ac_info", "u8"),
    ("dnxhd_1235_run_codes", "u16"),
    ("dnxhd_1235_run_bits", "u8"),
    ("dnxhd_1235_run", "u8"),
    ("dnxhd_1237_run_codes", "u16"),
    ("dnxhd_1237_run_bits", "u8"),
    ("dnxhd_1237_run", "u8"),
    ("dnxhd_1238_run", "u8"),
)


def extract(source: str, name: str) -> list[int]:
    match = re.search(
        rf"static const [^;]+\b{re.escape(name)}\s*\[[^]]*\]\s*=\s*\{{(.*?)\}};",
        source,
        re.DOTALL,
    )
    if not match:
        raise SystemExit(f"missing FFmpeg table {name}")
    body = re.sub(r"/\*.*?\*/", "", match.group(1), flags=re.DOTALL)
    values = [int(value, 0) for value in re.findall(r"(?:0x[0-9a-fA-F]+|\d+)", body)]
    if not values:
        raise SystemExit(f"empty FFmpeg table {name}")
    return values


def format_table(name: str, rust_type: str, values: list[int]) -> str:
    rust_name = name.upper()
    lines = [f"pub(crate) const {rust_name}: [{rust_type}; {len(values)}] = ["]
    for offset in range(0, len(values), 16):
        lines.append("    " + ", ".join(str(value) for value in values[offset : offset + 16]) + ",")
    lines.append("];\n")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="FFmpeg libavcodec/dnxhddata.c")
    parser.add_argument("output", type=Path, help="generated Rust destination")
    args = parser.parse_args()
    source = args.source.read_text()
    output = [
        "// SPDX-License-Identifier: LGPL-2.1-or-later",
        "//",
        "// Generated from FFmpeg libavcodec/dnxhddata.c at",
        f"// {SOURCE_REVISION}. Do not edit by hand.",
        "",
    ]
    for name, rust_type in TABLES:
        output.append(format_table(name, rust_type, extract(source, name)))
    args.output.write_text("\n".join(output))


if __name__ == "__main__":
    main()
