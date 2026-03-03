#!/usr/bin/env python3
"""Histogram theta from block-rotation check across random 2-qubit A matrices."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from compute_matrix import A_matrix
from helper import find_yds_with_fixed_addable_cells


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eig-tol", type=float, default=1e-6)
    parser.add_argument("--bins", type=int, default=60)
    parser.add_argument("--output", type=str, default="data/plots/theta_block_hist.png")
    return parser.parse_args()


def _orthonormalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


def _orthonormal_complement(V: np.ndarray) -> np.ndarray:
    basis = []
    P = V @ V.T
    for i in range(4):
        e = np.zeros(4)
        e[i] = 1.0
        v = e - P @ e
        if np.linalg.norm(v) > 1e-8:
            basis.append(v)
    M = np.stack(basis, axis=1)
    Q, _ = np.linalg.qr(M)
    return Q[:, :2]


def main() -> None:
    args = parse_args()
    diagrams = find_yds_with_fixed_addable_cells(4, args.max_size)

    thetas = []
    for d in diagrams:
        A = np.array(A_matrix(d), dtype=float)
        w, V = np.linalg.eig(A)
        idx1 = np.argmin(np.abs(w - 1.0))
        idxm1 = np.argmin(np.abs(w + 1.0))
        if abs(w[idx1] - 1.0) > args.eig_tol or abs(w[idxm1] + 1.0) > args.eig_tol:
            continue

        v1 = _orthonormalize(V[:, idx1].real)
        v2 = _orthonormalize(V[:, idxm1].real)
        v2 = v2 - np.dot(v1, v2) * v1
        v2 = _orthonormalize(v2)

        V12 = np.stack([v1, v2], axis=1)
        V34 = _orthonormal_complement(V12)
        v3, v4 = V34[:, 0], V34[:, 1]

        B = np.array(
            [
                [v3.T @ A @ v3, v3.T @ A @ v4],
                [v4.T @ A @ v3, v4.T @ A @ v4],
            ],
            dtype=float,
        )
        theta = np.arctan2(B[1, 0], B[0, 0])
        thetas.append(theta)

    if not thetas:
        raise SystemExit("No valid thetas found.")

    plt.figure(figsize=(7, 4))
    plt.hist(thetas, bins=args.bins, color="#2a6f9b", alpha=0.8)
    plt.title("Histogram of theta (block-rotation)")
    plt.xlabel("theta")
    plt.ylabel("count")
    plt.grid(True, alpha=0.3)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
