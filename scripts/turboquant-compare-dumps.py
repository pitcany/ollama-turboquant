#!/usr/bin/env python3
import argparse
import struct
from pathlib import Path


def load_meta(bin_path):
    meta_path = bin_path.with_suffix(".txt")
    meta = {}
    if not meta_path.exists():
        return meta
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        meta[key] = value
    return meta


def iter_f32(data):
    for (value,) in struct.iter_unpack("<f", data):
        yield value


def compare_f32(left, right, atol, rtol):
    max_abs = -1.0
    max_rel = -1.0
    max_i = 0
    max_left = 0.0
    max_right = 0.0
    mismatches = 0

    for i, (a, b) in enumerate(zip(iter_f32(left), iter_f32(right))):
        abs_diff = abs(a - b)
        rel_diff = abs_diff / max(abs(a), 1e-12)
        if abs_diff > max_abs:
            max_abs = abs_diff
            max_rel = rel_diff
            max_i = i
            max_left = a
            max_right = b
        if abs_diff > atol + rtol * abs(a):
            mismatches += 1

    if max_abs < 0:
        max_abs = 0.0
        max_rel = 0.0

    return {
        "kind": "f32",
        "mismatches": mismatches,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "index": max_i,
        "left": max_left,
        "right": max_right,
    }


def compare_bytes(left, right):
    mismatches = 0
    first = None
    first_left = 0
    first_right = 0
    for i, (a, b) in enumerate(zip(left, right)):
        if a == b:
            continue
        mismatches += 1
        if first is None:
            first = i
            first_left = a
            first_right = b
    return {
        "kind": "bytes",
        "mismatches": mismatches,
        "index": -1 if first is None else first,
        "left": first_left,
        "right": first_right,
    }


def comparable_type(left_meta, right_meta):
    left_type = left_meta.get("type", "").lower()
    right_type = right_meta.get("type", "").lower()
    return left_type == right_type and left_type == "f32"


def op_label(meta):
    op = meta.get("op", "?")
    typ = meta.get("type", "?")
    node = meta.get("node", "?")
    name = meta.get("name", "")
    return f"node={node} op={op} type={typ} name={name}"


def main():
    parser = argparse.ArgumentParser(description="Compare OLLAMA_TURBOQUANT_DUMP_DIR outputs from two runs.")
    parser.add_argument("left", type=Path, help="first dump directory, usually CPU/reference")
    parser.add_argument("right", type=Path, help="second dump directory, usually GPU/candidate")
    parser.add_argument("--atol", type=float, default=1e-5, help="absolute tolerance for f32 tensors")
    parser.add_argument("--rtol", type=float, default=1e-4, help="relative tolerance for f32 tensors")
    parser.add_argument("--top", type=int, default=20, help="maximum mismatching tensors to print")
    args = parser.parse_args()

    left_bins = sorted(args.left.glob("*.bin"))
    right_bins = sorted(args.right.glob("*.bin"))

    if not left_bins:
        raise SystemExit(f"no .bin dumps found in {args.left}")
    if not right_bins:
        raise SystemExit(f"no .bin dumps found in {args.right}")

    pairs = min(len(left_bins), len(right_bins))
    total_mismatching_tensors = 0
    printed = 0

    if len(left_bins) != len(right_bins):
        print(f"warning: dump count differs: left={len(left_bins)} right={len(right_bins)} comparing={pairs}")

    for i in range(pairs):
        left_path = left_bins[i]
        right_path = right_bins[i]
        left_meta = load_meta(left_path)
        right_meta = load_meta(right_path)
        left = left_path.read_bytes()
        right = right_path.read_bytes()

        size_mismatch = len(left) != len(right)
        if size_mismatch:
            result = {"kind": "size", "mismatches": 1}
        elif comparable_type(left_meta, right_meta):
            result = compare_f32(left, right, args.atol, args.rtol)
        else:
            result = compare_bytes(left, right)

        if result["mismatches"] == 0:
            continue

        total_mismatching_tensors += 1
        if printed >= args.top:
            continue

        print(f"[{i}] {left_path.name} <> {right_path.name}")
        print(f"  left:  {op_label(left_meta)}")
        print(f"  right: {op_label(right_meta)}")
        if result["kind"] == "f32":
            print(
                "  f32 mismatch:"
                f" count={result['mismatches']}"
                f" max_abs={result['max_abs']:.9g}"
                f" max_rel={result['max_rel']:.9g}"
                f" index={result['index']}"
                f" left={result['left']:.9g}"
                f" right={result['right']:.9g}"
            )
        elif result["kind"] == "bytes":
            print(
                "  byte mismatch:"
                f" count={result['mismatches']}"
                f" first={result['index']}"
                f" left=0x{result['left']:02x}"
                f" right=0x{result['right']:02x}"
            )
        else:
            print(f"  size mismatch: left={len(left)} right={len(right)}")

        printed += 1

    if total_mismatching_tensors == 0 and len(left_bins) == len(right_bins):
        print(f"MATCH: compared {pairs} tensors")
        return

    print(
        "SUMMARY:"
        f" compared={pairs}"
        f" mismatching_tensors={total_mismatching_tensors}"
        f" left_count={len(left_bins)}"
        f" right_count={len(right_bins)}"
    )
    raise SystemExit(1)


if __name__ == "__main__":
    main()
