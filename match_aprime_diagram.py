#!/usr/bin/env python3
"""Find (k-1)-addable Young diagrams whose A matrix matches A' from a k-addable diagram.

Procedure:
1) Pick a random k-addable diagram (up to max size) and build its A matrix.
2) Rotate rows to align the closest column to a computational basis vector.
3) Remove that row/column to obtain A'.
4) Search all (k-1)-addable diagrams (same max size) for A matrices equal to A'.
"""
from __future__ import annotations

import argparse
import math
import random

import numpy as np
import matplotlib.pyplot as plt

from compute_matrix import A_matrix
from helper import find_yds_with_fixed_addable_cells


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--addable",
        type=int,
        default=4,
        help="Number of addable cells for the starting diagram.",
    )
    parser.add_argument("--max-size", type=int, default=30, help="Max diagram size to search.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (0 for random).")
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Optional index into the starting diagram list.",
    )
    parser.add_argument("--tol", type=float, default=1e-8, help="Tolerance for zeroing and matching.")
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
    """Left-multiply by a Givens rotation acting on rows i,j."""
    G = np.eye(M.shape[0])
    G[i, i] = c
    G[i, j] = s
    G[j, i] = -s
    G[j, j] = c
    return G @ M


def _is_basis_column(col: np.ndarray, idx: int, sign: int, tol: float) -> bool:
    if abs(col[idx] - sign) > tol:
        return False
    for k in range(col.shape[0]):
        if k == idx:
            continue
        if abs(col[k]) > tol:
            return False
    return True


def _closest_column_to_basis(A: np.ndarray, tol: float) -> tuple[int, int, float, int]:
    # Prefer exact (-e_i) columns (within tol), then (+e_i).
    for j in range(A.shape[1]):
        col = A[:, j]
        for i in range(A.shape[0]):
            if _is_basis_column(col, i, -1, tol):
                return i, j, 0.0, -1
    for j in range(A.shape[1]):
        col = A[:, j]
        for i in range(A.shape[0]):
            if _is_basis_column(col, i, 1, tol):
                return i, j, 0.0, 1

    best_dist = float("inf")
    best_row = 0
    best_col = 0
    best_sign = 1
    for j in range(A.shape[1]):
        col = A[:, j]
        col_norm_sq = float(np.dot(col, col))
        for i in range(A.shape[0]):
            dist_sq_pos = col_norm_sq + 1.0 - 2.0 * float(col[i])
            if dist_sq_pos < best_dist:
                best_dist = dist_sq_pos
                best_row = i
                best_col = j
                best_sign = 1
            dist_sq_neg = col_norm_sq + 1.0 + 2.0 * float(col[i])
            if dist_sq_neg < best_dist:
                best_dist = dist_sq_neg
                best_row = i
                best_col = j
                best_sign = -1
    return best_row, best_col, math.sqrt(best_dist), best_sign


def _zero_column_to_basis(
    A: np.ndarray, col_idx: int, row_idx: int, tol: float
) -> tuple[np.ndarray, list[dict[str, float]]]:
    M = A.copy()
    angles: list[dict[str, float]] = []
    for i in range(M.shape[0] - 1, -1, -1):
        if i == row_idx:
            continue
        a = M[row_idx, col_idx]
        b = M[i, col_idx]
        if abs(b) <= tol:
            continue
        c, s, theta = _givens(a, b)
        M = _apply_givens(M, row_idx, i, c, s)
        angles.append({"row": i, "theta": theta, "c": c, "s": s})
    return M, angles


def _format_partition(d) -> str:
    return str(getattr(d, "partition", d))


def _annotate_matrix(ax: plt.Axes, mat: np.ndarray) -> None:
    if mat.size == 0:
        return
    for (i, j), val in np.ndenumerate(mat):
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black")


def _show_matrices(
    A: np.ndarray,
    M: np.ndarray,
    A_prime: np.ndarray,
    closest_A: np.ndarray,
    closest_label: str,
    closest_op_norm: float,
    target_addable: int,
    source_label: str,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(13, 4), constrained_layout=True)
    mats = [
        (f"A\n{source_label}", A),
        ("M = G A", M),
        ("A'", A_prime),
        (
            f"closest {target_addable}-addable\n{closest_label}\n"
            f"op_norm={closest_op_norm:.2e}",
            closest_A,
        ),
    ]
    for ax, (title, mat) in zip(axes, mats, strict=True):
        ax.matshow(mat, cmap="viridis")
        _annotate_matrix(ax, mat)
        ax.set_title(title)
        ax.set_xticks(range(mat.shape[1]) if mat.size else [])
        ax.set_yticks(range(mat.shape[0]) if mat.size else [])
    plt.show()


def main() -> None:
    args = parse_args()
    if args.addable < 2:
        raise SystemExit("--addable must be >= 2 to form A' with size >= 1.")
    rng = random.Random(None if args.seed == 0 else args.seed)

    diagrams_k = list(find_yds_with_fixed_addable_cells(args.addable, args.max_size))
    if not diagrams_k:
        raise SystemExit(f"No diagrams found with {args.addable} addable cells.")

    if args.index is not None:
        if args.index < 0 or args.index >= len(diagrams_k):
            raise SystemExit(f"--index out of range (0..{len(diagrams_k)-1}).")
        d4 = diagrams_k[args.index]
    else:
        d4 = rng.choice(diagrams_k)

    A = np.array(A_matrix(d4), dtype=float)
    target_row, target_col, target_dist, target_sign = _closest_column_to_basis(
        A, args.tol
    )
    M, angles = _zero_column_to_basis(A, target_col, target_row, args.tol)

    if any(
        abs(M[i, target_col]) > args.tol for i in range(M.shape[0]) if i != target_row
    ):
        zeroed = False
    else:
        zeroed = True

    A_prime = np.delete(np.delete(M, target_row, axis=0), target_col, axis=1)
    orth_ok = np.allclose(
        A_prime.T @ A_prime, np.eye(A_prime.shape[0]), atol=args.tol
    )

    target_addable = args.addable - 1
    diagrams_km1 = list(
        find_yds_with_fixed_addable_cells(target_addable, args.max_size)
    )
    if not diagrams_km1:
        raise SystemExit(f"No diagrams found with {target_addable} addable cells.")

    matches: list[tuple[str, float]] = []
    closest = (None, float("inf"), None)
    for d3 in diagrams_km1:
        A3 = np.array(A_matrix(d3), dtype=float)
        if A3.shape != A_prime.shape:
            continue
        op_norm = float(np.linalg.norm(A3 - A_prime, 2))
        if op_norm <= args.tol:
            matches.append((_format_partition(d3), op_norm))
        if op_norm < closest[1]:
            closest = (_format_partition(d3), op_norm, A3)

    print(f"{args.addable}-addable diagram:", _format_partition(d4))
    print(
        "closest column -> basis: "
        f"col={target_col} row={target_row} sign={target_sign:+d} "
        f"dist={target_dist:.6g}"
    )
    print("\nGivens rotations (rows target_row and i):")
    if angles:
        for entry in angles:
            print(
                f"  i={entry['row']}: theta={entry['theta']:.6g}, "
                f"c={entry['c']:.6g}, s={entry['s']:.6g}"
            )
    else:
        print("  (none)")
    print("\nZeroing success:", zeroed)
    print("A' orthogonal (within tol):", orth_ok)
    if closest[2] is not None:
        _show_matrices(
            A,
            M,
            A_prime,
            closest[2],
            closest[0],
            closest[1],
            target_addable,
            _format_partition(d4),
        )

    print(f"\n{target_addable}-addable matches (<= tol):", len(matches))
    if matches:
        for part, op_norm in matches:
            print(f"  diagram={part} op_norm={op_norm:.3e}")
        if len(matches) > 1:
            print("multiple diagrams match this A' (within tolerance)")


if __name__ == "__main__":
    main()
