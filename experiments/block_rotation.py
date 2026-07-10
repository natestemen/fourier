#!/usr/bin/env python3
"""Block-rotation normal form of A-matrices: A ≅ diag(1, −1) ⊕ R(θ₁) ⊕ ….

Question: does every 2^q-addable A-matrix conjugate, by a real orthogonal
change of basis, into a direct sum of diag(1, −1) and 2×2 rotation blocks?
Supports report.md Finding 1 (the eigenvalue-structure paragraph): the k = 4
case has spectrum {+1, −1, e^{iθ}, e^{−iθ}} and block form diag(1, −1) ⊕ R(θ),
and this experiment probes the same structure for larger k.

Expected result: every sampled diagram block-diagonalizes with exactly one
+1 and one −1 eigenvalue and rotation-block errors at numerical precision.

Migrated from check_block_rotation_3qubit.py (Schur-based general check)
plus the θ histogram of plot_theta_hist_block_rotation.py (--hist flag).
Behaviour changes: the 4×4 case uses `fourier.weyl.block_rotation_form`
instead of the hand-rolled eigenbasis construction (same normal form and θ
convention), the old --no-schur eigenstructure path is dropped, and the
histogram is saved to data/plots/block_rotation_theta_hist.png without
plt.show().
"""

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import schur

from fourier import a_matrix, block_rotation_form, diagrams_with_addable_cells

SURFACE = "#fcfcfb"
SERIES_BLUE = "#2a78d6"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-qubits", type=int, default=3)
    parser.add_argument("--max-size", type=int, default=30)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0, help="0 means unseeded sampling.")
    parser.add_argument("--eig-tol", type=float, default=1e-6)
    parser.add_argument(
        "--hist",
        action="store_true",
        help="Also histogram θ over all 4-addable diagrams up to --max-size.",
    )
    parser.add_argument("--bins", type=int, default=60)
    return parser.parse_args()


def analyze_schur(A: np.ndarray, tol: float):
    """Split the real Schur form A = Q·T·Qᵀ into 1×1 and 2×2 blocks.

    Returns (ones, negs, thetas, rot_errs, ortho_err): counts of 1×1 blocks
    near ±1, the rotation angle of each 2×2 block, each block's Frobenius
    distance from an exact rotation, and ‖QᵀQ − I‖."""
    T, Q = schur(A, output="real")
    blocks = []
    i = 0
    while i < T.shape[0]:
        if i + 1 < T.shape[0] and abs(T[i + 1, i]) > tol:
            blocks.append(T[i : i + 2, i : i + 2])
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
        # Try both rotation conventions and keep the better fit.
        theta1 = np.arctan2(blk[0, 1], blk[0, 0])
        R1 = np.array([[np.cos(theta1), np.sin(theta1)], [-np.sin(theta1), np.cos(theta1)]])
        err1 = np.linalg.norm(blk - R1, ord="fro")

        theta2 = np.arctan2(blk[1, 0], blk[0, 0])
        R2 = np.array([[np.cos(theta2), -np.sin(theta2)], [np.sin(theta2), np.cos(theta2)]])
        err2 = np.linalg.norm(blk - R2, ord="fro")

        if err1 <= err2:
            thetas.append(theta1)
            rot_errs.append(err1)
        else:
            thetas.append(theta2)
            rot_errs.append(err2)

    ortho_err = np.linalg.norm(Q.T @ Q - np.eye(A.shape[0]), ord="fro")
    return ones, negs, thetas, rot_errs, ortho_err


def check_sample(num_qubits: int, max_size: int, count: int, seed: int, eig_tol: float) -> None:
    """Print the block structure of `count` random 2^num_qubits-addable
    A-matrices with diagram size ≤ max_size."""
    addable = 1 << num_qubits
    diagrams = list(diagrams_with_addable_cells(addable, max_size))
    if not diagrams:
        raise SystemExit(f"No diagrams found with {addable} addable cells.")

    rng = random.Random(None if seed == 0 else seed)
    if len(diagrams) < count:
        print(f"Warning: only {len(diagrams)} diagrams available; using all of them.")
        sample = diagrams
    else:
        sample = rng.sample(diagrams, count)

    for i, diagram in enumerate(sample, start=1):
        A = a_matrix(diagram)
        print("=" * 80)
        print(f"[{i}] diagram: {diagram.partition}")

        if addable == 4:
            Q, theta, err = block_rotation_form(A)
            ortho_err = np.linalg.norm(Q.T @ Q - np.eye(4), ord="fro")
            print(f"orthonormality error ||Q^T Q - I||: {ortho_err:.3e}")
            print(f"R1 theta: {theta:.6g}, block form error: {err:.3e}")
            continue

        ones, negs, thetas, rot_errs, ortho_err = analyze_schur(A, eig_tol)
        print(f"orthonormality error ||Q^T Q - I||: {ortho_err:.3e}")
        print(f"1x1 blocks near +1: {len(ones)}  near -1: {len(negs)}")
        for k, (th, err) in enumerate(zip(thetas, rot_errs), start=1):
            print(f"R{k} theta: {th:.6g}, block error: {err:.3e}")


def theta_histogram(max_size: int, eig_tol: float, bins: int, destination: Path) -> None:
    """Histogram of the rotation angle θ of diag(1, −1) ⊕ R(θ) over every
    4-addable diagram of size ≤ max_size."""
    thetas = []
    for diagram in diagrams_with_addable_cells(4, max_size):
        _, theta, err = block_rotation_form(a_matrix(diagram))
        if err > eig_tol:
            continue
        thetas.append(theta)
    if not thetas:
        raise SystemExit("No valid thetas found.")

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.hist(thetas, bins=bins, color=SERIES_BLUE)
    ax.set_title("θ of the block-rotation form of 4-addable A-matrices")
    ax.set_xlabel("θ")
    ax.set_ylabel("count")
    ax.grid(True, color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)

    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved θ histogram over {len(thetas)} diagrams to {destination}.")


def main() -> None:
    args = parse_args()
    if args.num_qubits < 1:
        raise SystemExit("--num-qubits must be >= 1.")

    check_sample(args.num_qubits, args.max_size, args.count, args.seed, args.eig_tol)

    if args.hist:
        print("=" * 80)
        theta_histogram(
            args.max_size,
            args.eig_tol,
            args.bins,
            Path("data/plots/block_rotation_theta_hist.png"),
        )


if __name__ == "__main__":
    main()
