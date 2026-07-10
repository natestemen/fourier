#!/usr/bin/env python3
"""Analyze the point cloud of 4-addable A-matrix Weyl coordinates.

Loads a CSV of precomputed Weyl coordinates (columns diagram, size, weyl_a,
weyl_b, weyl_c — as written by the Weyl-sweep generator) and asks three
questions extracted from play.ipynb:

(a) Is the cloud planar?  PCA via numpy SVD should show two nonzero
    explained-variance ratios — every point lies on the a = π/4 face.
(b) How are b and |c| related?  Pearson correlation and a linear fit.
(c) Does diagram symmetry organize the (b, c) slice?  Scatter colored by a
    partition symmetry score (1 = self-conjugate).

Supports report.md Finding 1 ("Weyl Chamber — All 2-Qubit A-Matrices Have
a = π/4"): the point cloud is the numerical half of that finding.

Expected result: variance ratios ≈ [0.81, 0.19, 1e-31] (rank 2), a moderate
positive b vs |c| correlation (~0.42, slope ~0.33), and a scatter at
data/plots/weyl_pointcloud.png.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def transpose_partition(part: tuple[int, ...]) -> list[int]:
    """The conjugate partition (columns of the Young diagram)."""
    return [sum(1 for p in part if p > i) for i in range(max(part))]


def symmetry_score(part: tuple[int, ...]) -> float:
    """1 − |λ − λᵀ|₁ / (2|λ|): equals 1 iff λ is self-conjugate."""
    part_t = transpose_partition(part)
    m = max(len(part), len(part_t))
    p = np.array(list(part) + [0] * (m - len(part)))
    pt = np.array(list(part_t) + [0] * (m - len(part_t)))
    return 1 - np.sum(np.abs(p - pt)) / (2 * np.sum(p))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/weyl_4_addable_size_25.csv"),
        help="CSV with diagram, weyl_a, weyl_b, weyl_c columns.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/plots/weyl_pointcloud.png")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} points from {args.csv}")

    points = df[["weyl_a", "weyl_b", "weyl_c"]].values

    # (a) Planarity: all points on the a = pi/4 face, so the centered cloud
    # has rank 2.  PCA = SVD of the centered points.
    max_dev = np.max(np.abs(points[:, 0] - np.pi / 4))
    print(f"max |a - pi/4| = {max_dev:.3e}")

    centered = points - points.mean(axis=0)
    print("rank of centered point cloud:", np.linalg.matrix_rank(centered))
    s = np.linalg.svd(centered, compute_uv=False)
    explained = s**2 / np.sum(s**2)
    print("PCA explained-variance ratios:", explained)

    # (b) b vs |c| correlation and linear fit.
    b = df["weyl_b"].values
    c_abs = np.abs(df["weyl_c"].values)
    corr = np.corrcoef(b, c_abs)[0, 1]
    slope, intercept = np.polyfit(b, c_abs, 1)
    print(f"corr(b, |c|) = {corr:.4f}")
    print(f"linear fit: |c| = {slope:.4f}*b + {intercept:.4f}")

    # (c) Scatter of the (b, c) slice colored by partition symmetry.
    diagrams = df["diagram"].apply(ast.literal_eval)
    scores = diagrams.apply(symmetry_score)

    plt.figure()
    sc = plt.scatter(df["weyl_b"], df["weyl_c"], c=scores, cmap="viridis", alpha=0.7)
    plt.xlabel("b")
    plt.ylabel("c")
    plt.title("Weyl Chamber Slice Colored by Diagram Symmetry")
    plt.colorbar(sc, label="Symmetry Score (1 = self-conjugate)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
