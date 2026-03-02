#!/usr/bin/env python3
"""Plot histogram of theta when phi and lambda are close to multiples of pi."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/u3_params.csv"))
    parser.add_argument("--tol", type=float, default=1e-3, help="Tolerance to pi multiples.")
    parser.add_argument("--bins", type=int, default=60)
    parser.add_argument("--output", type=Path, default=Path("data/plots/theta_histogram.png"))
    return parser.parse_args()


def _near_pi_multiple(x: float, tol: float) -> bool:
    k = round(x / np.pi)
    return abs(x - k * np.pi) <= tol


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    required = {"theta", "phi", "lambda"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns in CSV: {sorted(missing)}")

    mask = df["phi"].apply(lambda v: _near_pi_multiple(float(v), args.tol)) & df[
        "lambda"
    ].apply(lambda v: _near_pi_multiple(float(v), args.tol))

    subset = df.loc[mask, "theta"].astype(float)
    if subset.empty:
        raise SystemExit("No rows where phi and lambda are near multiples of pi.")

    plt.figure(figsize=(7, 4))
    plt.hist(subset, bins=args.bins, color="#2a6f9b", alpha=0.8)
    plt.title("Histogram of theta where phi and lambda ~ k*pi")
    plt.xlabel("theta")
    plt.ylabel("count")
    plt.grid(True, alpha=0.3)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
