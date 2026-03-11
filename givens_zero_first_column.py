#!/usr/bin/env python3
"""Apply Givens rotations to zero out first column below the first entry.

Reports the Givens angles used (left-multiplication).
"""
from __future__ import annotations

import argparse
import math

import numpy as np

from compute_matrix import A_matrix
from helper import find_yds_with_fixed_addable_cells


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-size", type=int, default=30)
    parser.add_argument("--index", type=int, default=0, help="Diagram index to use.")
    parser.add_argument("--tol", type=float, default=1e-10)
    return parser.parse_args()


def _givens(a: float, b: float) -> tuple[float, float, float]:
    """Return (c, s, theta) s.t. [[c, s],[-s, c]] @ [a,b]^T = [r,0]^T."""
    r = math.hypot(a, b)
    if r == 0:
        return 1.0, 0.0, 0.0
    c = a / r
    s = b / r
    theta = math.atan2(s, c)
    return c, s, theta


def _apply_givens(M: np.ndarray, i: int, j: int, c: float, s: float) -> np.ndarray:
    """Left-multiply by Givens rotation acting on rows i,j."""
    G = np.eye(M.shape[0])
    G[i, i] = c
    G[i, j] = s
    G[j, i] = -s
    G[j, j] = c
    return G @ M


def main() -> None:
    args = parse_args()

    diagrams = list(find_yds_with_fixed_addable_cells(4, args.max_size))
    if not diagrams:
        raise SystemExit("No diagrams found with 4 addable cells.")
    if args.index < 0 or args.index >= len(diagrams):
        raise SystemExit(f"--index out of range (0..{len(diagrams)-1}).")

    d = diagrams[args.index]
    A = np.array(A_matrix(d), dtype=float)

    angles = []
    M = A.copy()
    # Zero out M[1,0], M[2,0], M[3,0] using Givens rotations on (0,i)
    for i in range(3, 0, -1):
        a = M[0, 0]
        b = M[i, 0]
        if abs(b) <= args.tol:
            continue
        c, s, theta = _givens(a, b)
        M = _apply_givens(M, 0, i, c, s)
        angles.append({"rows": (0, i), "theta": theta, "c": c, "s": s})

    print("diagram:", getattr(d, "partition", d))

    if any(abs(M[i, 0]) > args.tol for i in range(1, 4)):
        print("not fully zeroed within tolerance")
    else:
        print("successfully zeroed (within tolerance)")

    print("givens angles:")
    for entry in angles:
        rows = entry["rows"]
        print(f"  rows {rows}: theta={entry['theta']:.6g}, c={entry['c']:.6g}, s={entry['s']:.6g}")
    print(M)


if __name__ == "__main__":
    main()
