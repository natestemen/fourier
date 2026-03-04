#!/usr/bin/env python3
"""Generate Q matrices from 2-qubit A matrices and compute their eigenvalues."""
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
    parser.add_argument("--eig-tol", type=float, default=1e-6)
    parser.add_argument("--sort", action="store_true", help="Sort eigenvalues by angle then magnitude.")
    return parser.parse_args()


def _normalize(v: np.ndarray) -> np.ndarray:
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


def _build_q(A: np.ndarray, eig_tol: float) -> np.ndarray | None:
    w, V = np.linalg.eig(A)
    idx1 = np.argmin(np.abs(w - 1.0))
    idxm1 = np.argmin(np.abs(w + 1.0))
    if abs(w[idx1] - 1.0) > eig_tol or abs(w[idxm1] + 1.0) > eig_tol:
        return None

    v1 = _normalize(V[:, idx1].real)
    v2 = _normalize(V[:, idxm1].real)
    v2 = v2 - np.dot(v1, v2) * v1
    v2 = _normalize(v2)

    V12 = np.stack([v1, v2], axis=1)
    V34 = _orthonormal_complement(V12)
    v3, v4 = V34[:, 0], V34[:, 1]

    Q = np.column_stack([v1, v2, v3, v4])
    Q, _ = np.linalg.qr(Q)
    return Q


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

    tol = 1e-10
    snap_vals = [1.0, -1.0]

    for i, d in enumerate(sample, start=1):
        A = np.array(A_matrix(d), dtype=float)
        Q = _build_q(A, args.eig_tol)
        print("=" * 72)
        print(f"[{i}] diagram: {getattr(d, 'partition', d)}")
        if Q is None:
            print("skipping: could not form Q.")
            continue

        eigvals = np.linalg.eigvals(Q)
        if args.sort:
            eigvals = sorted(eigvals, key=lambda z: (np.angle(z), abs(z)))

        print("eigenvalues:")
        for ev in eigvals:
            re = ev.real if abs(ev.real) >= tol else 0.0
            im = ev.imag if abs(ev.imag) >= tol else 0.0
            for target in snap_vals:
                if abs(re - target) < tol:
                    re = target
                if abs(im - target) < tol:
                    im = target
            if im == 0.0:
                print(f"  {re}")
            else:
                print(f"  {re}{im:+}j")


if __name__ == "__main__":
    main()
