#!/usr/bin/env python3
"""Test whether A' can be written as A' = M (1 ⊕ A) with orthogonal M.

We solve M = A' B^T (B B^T)^{-1} for B = (1 ⊕ A), then check if M is orthogonal
and whether the reconstruction error is small.
"""
from __future__ import annotations

import argparse
import random
from typing import List

import numpy as np
from yungdiagram import YoungDiagram

from compute_matrix import A_matrix
from helper import find_yds_with_fixed_addable_cells


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-size", type=int, default=20)
    parser.add_argument("--fixed-index", type=int, default=0, help="Index of fixed A in addable=2 list.")
    parser.add_argument("--count", type=int, default=10, help="Number of random A' to test (addable=3).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument(
        "--unitary",
        action="store_true",
        help="Check M for unitarity (complex) instead of orthogonality.",
    )
    return parser.parse_args()


def _block_diag_one(A: np.ndarray) -> np.ndarray:
    B = np.zeros((3, 3), dtype=float)
    B[0, 0] = 1.0
    B[1:, 1:] = A
    return B


def main() -> None:
    args = parse_args()
    rng = random.Random(None if args.seed == 0 else args.seed)

    # Fixed A from addable=2
    diagrams_a = list(find_yds_with_fixed_addable_cells(2, args.max_size))
    if not diagrams_a:
        raise SystemExit("No diagrams found with 2 addable cells.")
    if args.fixed_index < 0 or args.fixed_index >= len(diagrams_a):
        raise SystemExit(f"--fixed-index out of range (0..{len(diagrams_a)-1}).")

    # dA = diagrams_a[args.fixed_index]
    dA = YoungDiagram([1, 1])
    A = np.array(A_matrix(dA), dtype=float)
    B = _block_diag_one(A)

    # Random A' from addable=3
    diagrams_ap = list(find_yds_with_fixed_addable_cells(3, args.max_size))
    if not diagrams_ap:
        raise SystemExit("No diagrams found with 3 addable cells.")

    if len(diagrams_ap) < args.count:
        sample = diagrams_ap
    else:
        sample = rng.sample(diagrams_ap, args.count)

    print("Fixed A diagram:", dA.partition)
    print("A:\n", A)
    print("B = 1 ⊕ A:\n", B)

    for i, dAp in enumerate(sample, start=1):
        Ap = np.array(A_matrix(dAp), dtype=complex)

        # Solve for M in A' = M B => M = A' B^T (B B^T)^{-1}
        if args.unitary:
            BBt = B @ B.conj().T
        else:
            BBt = B @ B.T
        if np.linalg.cond(BBt) > 1e12:
            print("=" * 72)
            print(f"[{i}] A' diagram: {getattr(dAp, 'partition', dAp)}")
            print("skipping: B B^T is ill-conditioned.")
            continue

        if args.unitary:
            M = Ap @ B.conj().T @ np.linalg.inv(BBt)
        else:
            M = Ap @ B.T @ np.linalg.inv(BBt)
        recon = M @ B
        err = np.linalg.norm(recon - Ap, ord="fro")
        if args.unitary:
            ortho_err = np.linalg.norm(M @ M.conj().T - np.eye(M.shape[0]), ord="fro")
        else:
            ortho_err = np.linalg.norm(M @ M.T - np.eye(M.shape[0]), ord="fro")

        print("=" * 72)
        print(f"[{i}] A' diagram: {getattr(dAp, 'partition', dAp)}")
        if args.unitary:
            print(f"unitarity error ||M M^† - I||: {ortho_err:.3e}")
        else:
            print(f"orthonormality error ||M M^T - I||: {ortho_err:.3e}")
        print(M)
        print(f"reconstruction error ||M B - A'||: {err:.3e}")


if __name__ == "__main__":
    main()
