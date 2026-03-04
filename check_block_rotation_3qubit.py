#!/usr/bin/env python3
"""Check if n-qubit A matrices block-diagonalize as Z ⊕ R(theta1) ⊕ ..."""
from __future__ import annotations

import argparse
import random

import numpy as np
from scipy.linalg import schur

from compute_matrix import A_matrix
from helper import find_yds_with_fixed_addable_cells


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-qubits", type=int, default=3)
    parser.add_argument("--max-size", type=int, default=30)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eig-tol", type=float, default=1e-6)
    parser.add_argument("--orth-tol", type=float, default=1e-6)
    parser.add_argument(
        "--use-schur",
        action="store_true",
        default=True,
        help="Use real Schur decomposition for robust block structure.",
    )
    parser.add_argument(
        "--no-schur",
        action="store_false",
        dest="use_schur",
        help="Disable Schur decomposition and use eigenstructure-based method.",
    )
    return parser.parse_args()


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


def _gs_add(basis: list[np.ndarray], v: np.ndarray, tol: float = 1e-8) -> np.ndarray | None:
    u = v.astype(float)
    for b in basis:
        u = u - np.dot(b, u) * b
    if np.linalg.norm(u) < tol:
        return None
    return _normalize(u)


def _build_basis(A: np.ndarray, eig_tol: float) -> np.ndarray | None:
    # Eigen-decomposition
    w, V = np.linalg.eig(A)

    # Pick eigenvectors near +1 and -1
    idx_pos = np.argmin(np.abs(w - 1.0))
    idx_neg = np.argmin(np.abs(w + 1.0))
    if abs(w[idx_pos] - 1.0) > eig_tol or abs(w[idx_neg] + 1.0) > eig_tol:
        return None

    basis: list[np.ndarray] = []
    v_pos = _normalize(V[:, idx_pos].real)
    basis.append(v_pos)
    v_neg = V[:, idx_neg].real
    v_neg = _gs_add(basis, v_neg)
    if v_neg is None:
        return None
    basis.append(v_neg)

    # Build 2D invariant subspaces from complex conjugate pairs
    for idx, lam in enumerate(w):
        if np.imag(lam) <= eig_tol:
            continue
        v = V[:, idx]
        u1 = _gs_add(basis, v.real)
        if u1 is None:
            continue
        basis.append(u1)
        u2 = v.imag
        u2 = _gs_add(basis, u2)
        if u2 is None:
            basis.pop()
            continue
        basis.append(u2)
        if len(basis) >= 8:
            break

    if len(basis) < A.shape[0]:
        return None

    Q = np.column_stack(basis[: A.shape[0]])
    # Orthonormalize for numerical stability
    Q, _ = np.linalg.qr(Q)
    return Q


def _analyze_schur(A: np.ndarray, tol: float):
    # Real Schur: A = Q T Q^T, T quasi-upper-triangular with 1x1 and 2x2 blocks.
    T, Q = schur(A, output="real")
    blocks = []
    i = 0
    while i < T.shape[0]:
        if i + 1 < T.shape[0] and abs(T[i + 1, i]) > tol:
            blk = T[i : i + 2, i : i + 2]
            blocks.append(blk)
            i += 2
        else:
            blocks.append(T[i : i + 1, i : i + 1])
            i += 1

    ones = [blk for blk in blocks if blk.shape == (1, 1) and abs(blk[0, 0] - 1.0) <= tol]
    negs = [blk for blk in blocks if blk.shape == (1, 1) and abs(blk[0, 0] + 1.0) <= tol]

    thetas = []
    rot_errs = []
    for blk in blocks:
        if blk.shape != (2, 2):
            continue
        a, b = blk[0, 0], blk[0, 1]
        c, d = blk[1, 0], blk[1, 1]
        # Try both rotation conventions and pick smaller error.
        theta1 = np.arctan2(b, a)
        R1 = np.array([[np.cos(theta1), np.sin(theta1)], [-np.sin(theta1), np.cos(theta1)]])
        err1 = np.linalg.norm(blk - R1, ord="fro")

        theta2 = np.arctan2(c, a)
        R2 = np.array([[np.cos(theta2), -np.sin(theta2)], [np.sin(theta2), np.cos(theta2)]])
        err2 = np.linalg.norm(blk - R2, ord="fro")

        if err1 <= err2:
            thetas.append(theta1)
            rot_errs.append(err1)
        else:
            thetas.append(theta2)
            rot_errs.append(err2)

    ortho_err = np.linalg.norm(Q.T @ Q - np.eye(A.shape[0]), ord="fro")
    return Q, T, ones, negs, thetas, rot_errs, ortho_err


def main() -> None:
    args = parse_args()
    rng = random.Random(None if args.seed == 0 else args.seed)

    if args.num_qubits < 1:
        raise SystemExit("--num-qubits must be >= 1.")
    addable = 1 << args.num_qubits
    diagrams = list(find_yds_with_fixed_addable_cells(addable, args.max_size))
    if not diagrams:
        raise SystemExit(f"No diagrams found with {addable} addable cells.")
    if len(diagrams) < args.count:
        print(f"Warning: only {len(diagrams)} diagrams available; using all of them.")
        sample = diagrams
    else:
        sample = rng.sample(diagrams, args.count)

    for i, d in enumerate(sample, start=1):
        A = np.array(A_matrix(d), dtype=float)
        print("=" * 80)
        print(f"[{i}] diagram: {getattr(d, 'partition', d)}")

        if args.use_schur:
            Qs, T, ones, negs, thetas, rot_errs, ortho_err = _analyze_schur(A, args.eig_tol)
            print(f"orthonormality error ||Q^T Q - I||: {ortho_err:.3e}")
            print(f"1x1 blocks near +1: {len(ones)}  near -1: {len(negs)}")
            for k, (th, err) in enumerate(zip(thetas, rot_errs), start=1):
                print(f"R{ k } theta: {th:.6g}, block error: {err:.3e}")
            continue

        Q = _build_basis(A, args.eig_tol)
        if Q is None:
            print("skipping: could not build basis from eigenstructure.")
            continue

        dim = A.shape[0]
        ortho_err = np.linalg.norm(Q.T @ Q - np.eye(dim), ord="fro")
        B = Q.T @ A @ Q

        # Z block
        Z = np.diag([1.0, -1.0])
        Z_block = B[0:2, 0:2]
        z_err = np.linalg.norm(Z_block - Z, ord="fro")

        thetas = []
        rot_errs = []
        num_rot = (dim - 2) // 2
        for k in range(num_rot):
            blk = B[2 + 2 * k : 2 + 2 * k + 2, 2 + 2 * k : 2 + 2 * k + 2]
            theta = np.arctan2(blk[1, 0], blk[0, 0])
            R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            err = np.linalg.norm(blk - R, ord="fro")
            thetas.append(theta)
            rot_errs.append(err)

        print(f"orthonormality error ||Q^T Q - I||: {ortho_err:.3e}")
        print(f"Z block error: {z_err:.3e}")
        for k, (th, err) in enumerate(zip(thetas, rot_errs), start=1):
            print(f"R{ k } theta: {th:.6g}, block error: {err:.3e}")


if __name__ == "__main__":
    main()
