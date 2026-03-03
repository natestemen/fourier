#!/usr/bin/env python3
"""Check whether random 2-qubit A matrices commute with SWAP."""
from __future__ import annotations

import argparse
import random

import numpy as np

from compute_matrix import A_matrix
from helper import find_yds_with_fixed_addable_cells


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-size", type=int, default=30, help="Max diagram size to search.")
    parser.add_argument("--count", type=int, default=10, help="Number of random diagrams to test.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (0 for random).")
    parser.add_argument("--tol", type=float, default=1e-9, help="Tolerance for commutator norm.")
    return parser.parse_args()


def _swap() -> np.ndarray:
    return np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )


def main() -> None:
    args = parse_args()
    rng = random.Random(None if args.seed == 0 else args.seed)

    diagrams = list(find_yds_with_fixed_addable_cells(4, args.max_size))
    if not diagrams:
        raise SystemExit("No diagrams found with 4 addable cells.")

    if len(diagrams) < args.count:
        print(f"Warning: only {len(diagrams)} diagrams available; using all of them.")
        sample = diagrams
    else:
        sample = rng.sample(diagrams, args.count)

    SWAP = _swap()

    for i, d in enumerate(sample, start=1):
        mat = np.array(A_matrix(d), dtype=float)
        comm = mat @ SWAP - SWAP @ mat
        norm = float(np.linalg.norm(comm, ord='fro'))
        print("=" * 72)
        print(f"[{i}] diagram: {getattr(d, 'partition', d)}")
        print(f"commutator Frobenius norm: {norm:.6g}")
        print("commutes:", norm <= args.tol)


if __name__ == "__main__":
    main()
