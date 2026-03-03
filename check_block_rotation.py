#!/usr/bin/env python3
"""Check if 2-qubit A matrices block diagonalize as 1 ⊕ (-1) ⊕ R(theta)."""
from __future__ import annotations

import argparse
import random

import numpy as np

from compute_matrix import A_matrix
from helper import find_yds_with_fixed_addable_cells


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-size", type=int, default=30)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eig-tol", type=float, default=1e-6)
    parser.add_argument("--orth-tol", type=float, default=1e-6)
    return parser.parse_args()


def _orthonormalize(v: np.ndarray) -> np.ndarray:
    v = v.astype(float)
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


def _orthonormal_complement(V: np.ndarray) -> np.ndarray:
    """Return 4x2 orthonormal basis for complement of columns of V (4x2)."""
    # Project standard basis onto complement.
    basis = []
    P = V @ V.T
    for i in range(4):
        e = np.zeros(4)
        e[i] = 1.0
        v = e - P @ e
        if np.linalg.norm(v) > 1e-8:
            basis.append(v)
    if len(basis) < 2:
        raise RuntimeError("Failed to find complement basis.")
    M = np.stack(basis, axis=1)
    Q, _ = np.linalg.qr(M)
    return Q[:, :2]


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

    for i, d in enumerate(sample, start=1):
        A = np.array(A_matrix(d), dtype=float)
        w, V = np.linalg.eig(A)

        # Find eigenvectors for eigenvalues near 1 and -1.
        idx1 = np.argmin(np.abs(w - 1.0))
        idxm1 = np.argmin(np.abs(w + 1.0))
        if abs(w[idx1] - 1.0) > args.eig_tol or abs(w[idxm1] + 1.0) > args.eig_tol:
            print("=" * 72)
            print(f"[{i}] diagram: {getattr(d, 'partition', d)}")
            print("skipping: eigenvalues not close to ±1")
            continue

        v1 = _orthonormalize(V[:, idx1].real)
        v2 = _orthonormalize(V[:, idxm1].real)

        # Orthonormalize v2 against v1.
        v2 = v2 - np.dot(v1, v2) * v1
        v2 = _orthonormalize(v2)

        V12 = np.stack([v1, v2], axis=1)
        V34 = _orthonormal_complement(V12)
        v3, v4 = V34[:, 0], V34[:, 1]

        # Build B = [v_i^T A v_j] for i,j in {3,4}
        B = np.array(
            [
                [v3.T @ A @ v3, v3.T @ A @ v4],
                [v4.T @ A @ v3, v4.T @ A @ v4],
            ],
            dtype=float,
        )
        theta = np.arctan2(B[1, 0], B[0, 0])
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        Q = np.column_stack([v1, v2, v3, v4])
        QtAQ = Q.T @ A @ Q
        target = np.block(
            [
                [np.array([[1.0, 0.0], [0.0, -1.0]]), np.zeros((2, 2))],
                [np.zeros((2, 2)), R],
            ]
        )
        err = np.linalg.norm(QtAQ - target, ord="fro")
        ortho_err = np.linalg.norm(Q.T @ Q - np.eye(4), ord="fro")

        print("=" * 72)
        print(f"[{i}] diagram: {getattr(d, 'partition', d)}")
        print("rotation matrix:")
        print(B)
        print("ortho matrix:")
        print(Q)
        print(f"theta: {theta}")
        print(f"orthonormality error ||Q^T Q - I||: {ortho_err:.3e}")
        print(f"block form error ||Q^T A Q - (1⊕-1⊕R)||: {err:.3e}")


if __name__ == "__main__":
    main()
