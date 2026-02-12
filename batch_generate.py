#!/usr/bin/env python3
"""Repeatedly invoke `generate_data.py` for growing diagram sizes.

This helper script is meant for long-running batches (e.g., overnight).  It
starts at a configurable maximum-diagram size, increments by one each run, and
calls `generate_data.py 2 <size>` so only two-qubit Fourier matrices (4 addable
cells) are produced.  Each run writes to the same CSV naming scheme as the
main tool (`data/4_addable_size_<size>_u3_cnot.csv`), allowing you to cancel
the script at any point without losing completed files.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_OUTPUT = Path("data")
FILENAME_TEMPLATE = "4_addable_size_{size}_u3_cnot.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start-size",
        type=int,
        default=6,
        help="Initial `max_diagram_size` to hand to generate_data.py (default: 6).",
    )
    parser.add_argument(
        "--end-size",
        type=int,
        default=None,
        help="Optional inclusive upper bound for diagram sizes; omit to run indefinitely.",
    )
    parser.add_argument(
        "--max-diagrams",
        type=int,
        default=None,
        help="Forwarded to generate_data.py to bound diagrams per size.",
    )
    parser.add_argument(
        "--optimization-level",
        type=int,
        choices=[1, 2, 3, 4],
        default=3,
        help="Forwarded to generate_data.py (default: 3).",
    )
    parser.add_argument(
        "--synthesis-epsilon",
        type=float,
        default=1e-9,
        help="Forwarded tolerance for BQSKit synthesis.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional PRNG seed forwarded to generate_data.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory where CSV files are stored (default: ./data).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if the target CSV already exists.",
    )
    return parser.parse_args()


def build_command(args: argparse.Namespace, size: int) -> list[str]:
    cmd = [
        sys.executable,
        "generate_data.py",
        "2",
        str(size),
        "--output-dir",
        str(args.output_dir),
        "--optimization-level",
        str(args.optimization_level),
        "--synthesis-epsilon",
        str(args.synthesis_epsilon),
    ]
    if args.max_diagrams is not None:
        cmd.extend(["--max-diagrams", str(args.max_diagrams)])
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    return cmd


def main() -> None:
    args = parse_args()
    size = args.start_size

    try:
        while args.end_size is None or size <= args.end_size:
            filename = FILENAME_TEMPLATE.format(size=size)
            destination = args.output_dir / filename
            if destination.exists() and not args.force:
                print(f"[skip] {destination} already exists.")
                size += 1
                continue

            destination.parent.mkdir(parents=True, exist_ok=True)
            print(f"[run ] size={size} -> {destination}")
            cmd = build_command(args, size)
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as exc:
                print(f"[fail] size={size} exited with code {exc.returncode}.", file=sys.stderr)
                break

            size += 1
    except KeyboardInterrupt:
        print("\nInterrupted by user; exiting gracefully.")


if __name__ == "__main__":
    main()
